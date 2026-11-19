import pytest
import torch

from nsrmp.baseline import NeuralBaseline, ParsedNeuralBaseline
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.metrics import evaluate
from nsrmp.model import ActionSimulator, NSModel, apply_move


def _scene():
    torch.manual_seed(0)
    boxes = torch.rand(2, 4, 4)
    subj_w = torch.tensor([[0.1, 0.6, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]])  # 2nd row: inactive step
    ref_w = torch.tensor([[0.5, 0.2, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return boxes, subj_w, ref_w, torch.randn(2, 16)


def test_hard_move_changes_only_the_argmax_subject_using_argmax_subject_and_reference_boxes():
    boxes, subj_w, ref_w, emb = _scene()
    sim = ActionSimulator()
    out = apply_move(sim, boxes, subj_w, ref_w, emb, hard=True)
    expected = sim(emb[:1], boxes[:1, 1], boxes[:1, 0])[0]  # subject = object 1, reference = object 0
    assert torch.equal(out[0, [0, 2, 3]], boxes[0, [0, 2, 3]])
    assert torch.allclose(out[0, 1], expected, atol=1e-6)


def test_hard_move_of_an_inactive_step_moves_nothing():
    boxes, subj_w, ref_w, emb = _scene()
    out = apply_move(ActionSimulator(), boxes, subj_w, ref_w, emb, hard=True)
    assert torch.equal(out[1], boxes[1])


def test_soft_move_is_the_default_and_blends_every_object():
    boxes, subj_w, ref_w, emb = _scene()
    sim = ActionSimulator()
    default = apply_move(sim, boxes, subj_w, ref_w, emb)
    assert torch.equal(default, apply_move(sim, boxes, subj_w, ref_w, emb, hard=False))
    assert not torch.equal(default[0, 0], boxes[0, 0])  # soft: object 0 has weight 0.1 and moves a little


@pytest.mark.parametrize("factory", [NSModel, NeuralBaseline, ParsedNeuralBaseline])
def test_models_in_hard_mode_leave_every_object_but_the_subject_untouched(factory):
    b = collate(make_dataset(16, seed=3, n_steps=(1, 1)))
    torch.manual_seed(0)
    model = factory().eval()
    model.hard = True
    with torch.no_grad():
        pred, subj_w = model(b)
    subject = subj_w[:, 0].argmax(-1)
    others = torch.ones_like(b.mask, dtype=torch.bool)
    others[torch.arange(len(b)), subject] = False
    assert torch.equal(pred[others], b.boxes[others])


@pytest.mark.parametrize("factory", [NSModel, NeuralBaseline, ParsedNeuralBaseline])
def test_evaluate_defaults_to_soft_and_restores_the_models_mode(factory):
    b = collate(make_dataset(16, seed=4))
    torch.manual_seed(0)
    model = factory()
    soft = evaluate(model, b)
    assert soft == evaluate(model, b, hard=False)
    hard = evaluate(model, b, hard=True)
    assert model.hard is False
    assert hard != soft
    assert evaluate(model, b) == soft
