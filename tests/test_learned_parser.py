from types import SimpleNamespace

import pytest
import torch

import nsrmp.batch
import nsrmp.parser as rule_parser
from nsrmp._learned_parser import ROLES, lex_step, role_to_program
from nsrmp.batch import collate
from nsrmp.data import REL_PHRASE, RELATIONS, make_dataset
from nsrmp.learned_lr import RolePolicy
from nsrmp.combined_train import fit_programs, policy_objective
from nsrmp.model import build
from nsrmp.parser import parse


def _builder_fixture(subj_rel, ref_rel, action):
    """Explicit text and occurrence-aligned roles for one Move (no parse_roles)."""
    subj = "the red cube"
    roles = ["subj_head", "subj_head"]
    if subj_rel:
        subj += f" which is {REL_PHRASE[subj_rel]} the blue object"
        roles += ["subj_rel", "subj_anchor", "subj_anchor"]
    ref = "the red cube"
    ref_roles = ["ref_head", "ref_head"]
    if ref_rel:
        ref += f" which is {REL_PHRASE[ref_rel]} the dice"
        ref_roles += ["ref_rel", "ref_anchor"]
    return f"put {subj} {REL_PHRASE[action]} {ref}", roles + ["action"] + ref_roles


@pytest.mark.parametrize("subj_rel", [None] + RELATIONS)
@pytest.mark.parametrize("ref_rel", [None] + RELATIONS)
@pytest.mark.parametrize("action", RELATIONS)
def test_builder_rebuilds_the_parse_oracle(subj_rel, ref_rel, action):
    text, roles = _builder_fixture(subj_rel, ref_rel, action)
    step = lex_step(text)
    assert role_to_program(step.keywords, roles) == parse(text)[0]


def test_policy_masks_invalid_roles_before_sampling(monkeypatch):
    example = make_dataset(1, seed=4, p_complex=0.0)[0]
    example.text = "put the cube to the right of the dice"
    inputs = collate([example], mode="learned").inputs
    parser = RolePolicy()
    logits = torch.zeros(1, 1, 3, len(ROLES))
    invalid = [i for i, role in enumerate(ROLES) if role not in ("subj_head", "ref_head")]
    logits[0, 0, 0, invalid] = 100
    logits.requires_grad_()
    monkeypatch.setattr(parser, "forward", lambda inputs: logits)

    sample = parser.sample(inputs)
    assert sample.role_traces[0][0][0] in ("subj_head", "ref_head")
    sample.log_prob.sum().backward()
    assert torch.count_nonzero(logits.grad[0, 0, 0, invalid]).item() == 0


def test_toy_expected_reward_improves():
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([logits], lr=0.2)
    rewards = torch.tensor([[0.0], [1.0]])
    first_grad = None
    for step in range(12):
        logp = logits.log_softmax(0)[:, None]
        loss, _ = policy_objective(logp, rewards)
        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            first_grad = logits.grad.clone()
        optimizer.step()
    probs = logits.softmax(0)
    assert probs[1] > 0.5 and probs[1] > probs[0]
    assert first_grad[1] < 0 and first_grad[0] > 0


def _poison_example(example):
    class Poisoned(SimpleNamespace):
        def __getattr__(self, name):
            raise AssertionError(f"gold field accessed: {name}")

    return Poisoned(text=example.text, feats=example.feats,
                    objs=[Poisoned(box=obj.box) for obj in example.objs],
                    final_boxes=example.final_boxes, is_complex=example.is_complex)


def test_train_parser_never_reads_gold_or_rule_parser(monkeypatch):
    examples = [_poison_example(e)
                for e in make_dataset(3, seed=4, n_steps=(1, 2), p_complex=0.0)]

    def explode(*args, **kwargs):
        raise AssertionError("rule parser called")

    monkeypatch.setattr(rule_parser, "parse", explode)
    monkeypatch.setattr(rule_parser, "parse_roles", explode)
    monkeypatch.setattr(nsrmp.batch, "parse", explode)
    monkeypatch.setattr(nsrmp.batch, "parse_roles", explode)
    data = collate(examples, surface_context=True, mode="learned")
    model = build(parser_source="learned", backend="symbolic")
    before = [p.detach().clone() for p in model.parameters()]
    config = dict(epochs=1, curriculum=True, learning_rate=0.003, batch_size=3, num_samples=4,
                  move_mode="soft", loss="original", beta=1.0, entropy_coef=0.01, gradient_clip=5.0)
    fit_programs(model, data, config, phases=[("warm", 1)], seed=0)
    assert any(not torch.equal(a, b)
               for a, b in zip(before, model.parameters()))


def test_every_draw_contains_all_valid_steps():
    examples = (make_dataset(1, seed=4, n_steps=(1, 1)) +
                make_dataset(1, seed=5, n_steps=(2, 2)))
    inputs = collate(examples, mode="learned").inputs
    draws = RolePolicy().sample_many(inputs, count=2, greedy=False)
    assert [tuple(map(len, draw.programs)) for draw in draws] == [(1, 2), (1, 2)]
