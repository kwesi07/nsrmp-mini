import numpy as np
import pytest
import torch

from nsrmp.batch import collate
from nsrmp.data import make_dataset, apply_action, ACTIONS
from nsrmp.learned_lr import RolePolicy
from nsrmp.model import apply_move, build, NSModel
from nsrmp.train import loss_fn


class FixedSimulator(torch.nn.Module):
    """Stand-in Action Simulator returning a given box for every example."""

    def __init__(self, new_box):
        super().__init__()
        self.new_box = torch.tensor(new_box, dtype=torch.float32)

    def forward(self, action_emb, subj_box, ref_box):
        return self.new_box.expand(subj_box.shape[0], 4)


def np_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_move_with_a_one_hot_subject_replaces_only_that_box_and_zero_loss_when_gold():
    ex = make_dataset(1, seed=0)[0]
    b = collate([ex])
    s, r = ex.subjects[0], ex.refs[0]
    gold_box = apply_action(ex.program[0][1], ex.objs[s].box, ex.objs[r].box)
    subj_w = torch.zeros(1, b.boxes.shape[1]); subj_w[0, s] = 1
    ref_w = torch.zeros_like(subj_w); ref_w[0, r] = 1
    pred = apply_move(FixedSimulator(gold_box), b.boxes, subj_w, ref_w, torch.zeros(1, 1))
    assert torch.allclose(pred, b.final, atol=1e-6)
    assert loss_fn(pred, b.final, b.mask).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_l1_plus_one_minus_iou_summed_over_objects():
    ex = make_dataset(1, seed=1)[0]
    b = collate([ex])
    s, r = ex.subjects[0], ex.refs[0]
    wrong = (0.40, 0.40, 0.50, 0.52)
    subj_w = torch.zeros(1, b.boxes.shape[1]); subj_w[0, s] = 1
    ref_w = torch.zeros_like(subj_w); ref_w[0, r] = 1
    pred = apply_move(FixedSimulator(wrong), b.boxes, subj_w, ref_w, torch.zeros(1, 1))
    gold = ex.final_boxes[s]
    expected = np.abs(np.array(wrong) - gold).sum() + (1 - np_iou(wrong, gold))
    assert loss_fn(pred, b.final, b.mask, beta=1.0).item() == pytest.approx(expected, rel=1e-4)


def test_loss_ignores_padded_objects():
    exs = make_dataset(4, seed=2, n_objects=(3, 5))
    b = collate(exs)
    pred = b.final.clone()
    pad = b.mask == 0
    pred[pad] = 5.0  # garbage in the padding must not matter
    assert loss_fn(pred, b.final, b.mask).item() == pytest.approx(0.0, abs=1e-6)


def test_ns_model_forward_returns_final_boxes_and_one_subject_distribution_per_step():
    b = collate(make_dataset(8, seed=3, n_steps=(1, 3)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        boxes, subj_w = NSModel()(b)
    assert boxes.shape == b.boxes.shape and subj_w.shape == (8, b.subjects.shape[1], b.boxes.shape[1])
    for k in range(8):
        n_steps = b.n_steps[k].item()
        assert torch.allclose(subj_w[k, :n_steps].sum(-1), torch.ones(n_steps), atol=1e-4)
        assert subj_w[k, n_steps:].abs().sum().item() == 0.0


def test_builder_registers_only_a_learned_parser():
    learned = build(parser_source="learned", backend="symbolic")
    rule = build(parser_source="rule", backend="symbolic")
    assert isinstance(learned.parser, RolePolicy)
    assert {key for key in learned.state_dict() if key.startswith("parser.")} == {
        f"parser.{key}" for key in learned.parser.state_dict()}
    assert rule.parser is None
    assert not any(key.startswith("parser.") for key in rule.state_dict())
