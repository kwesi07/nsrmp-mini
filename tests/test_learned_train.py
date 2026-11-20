import pytest
import torch

from nsrmp import combined_train
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.learned_lr import ParserSample
from nsrmp.model import build
from run_experiments import _phase_schedule


@pytest.mark.parametrize("config, expected", [
    (dict(schedule="A", warm_epochs=1, mixed_epochs=2), [("warm", 1), ("mixed", 2)]),
    (dict(schedule="B", warm_epochs=1, sim_epochs=2, relation_epochs=3),
     [("warm", 1), ("sim", 2), ("relation", 3)]),
])
def test_phase_schedule_preserves_order_and_epochs(config, expected):
    assert _phase_schedule(config) == expected


def test_sim_to_relation_switches_trainable_parameters():
    model = build(parser_source="learned", backend="symbolic")
    for phase, prefixes in (("sim", ("sim.",)), ("relation", ("concepts.rel_mlp.", "parser."))):
        combined_train._set_phase(model, phase)
        assert {name for name, param in model.named_parameters() if param.requires_grad} == {
            name for name, _ in model.named_parameters() if name.startswith(prefixes)}


@pytest.mark.parametrize("previous_ema, expected", [(None, -2.0), (-3.0, 1.0)])
def test_single_draw_uses_the_supplied_baseline(previous_ema, expected):
    reward = torch.tensor([[-2.0]])
    _, advantage = combined_train.policy_objective(torch.zeros_like(reward), reward, previous_ema)
    torch.testing.assert_close(advantage, torch.tensor([[expected]]))


def test_multiple_draws_use_a_baseline_per_instruction():
    reward = torch.tensor([[-1., -6.], [-3., -2.]])
    _, advantage = combined_train.policy_objective(torch.zeros_like(reward), reward)
    torch.testing.assert_close(advantage, torch.tensor([[2., -4.], [-2., 4.]]))


def test_candidate_rewards_use_the_original_scene_row(monkeypatch):
    examples = (make_dataset(1, seed=4, n_objects=(3, 3), n_steps=(1, 1)) +
                make_dataset(1, seed=5, n_objects=(5, 5), n_steps=(2, 2)))
    data = collate(examples, mode="learned")
    torch.manual_seed(6)
    model = build(parser_source="learned", backend="symbolic")
    programs = [example.program for example in examples]
    draws = [programs, programs[::-1]]
    expected = torch.empty(2, 2)
    with torch.no_grad():
        for k, draw in enumerate(draws):
            for b, program in enumerate(draw):
                row = data.select([b])
                pred = model.execute(row.inputs, [program], move_mode="soft").final_boxes
                expected[k, b] = -combined_train.scene_loss(
                    pred, row.final_boxes, row.inputs.object_mask, objective="original")
    assert expected.unique().numel() == 4
    monkeypatch.setattr(combined_train, "shuffle_epoch", lambda blocks, generator: torch.arange(2))
    monkeypatch.setattr(model.parser, "sample_many", lambda inputs, *, count, greedy: [
        ParserSample(draw, [], torch.zeros(2), torch.zeros(2)) for draw in draws])
    objective = combined_train.policy_objective
    seen = []

    def record(log_prob, reward, previous_ema):
        seen.append(reward)
        return objective(log_prob, reward, previous_ema)

    monkeypatch.setattr(combined_train, "policy_objective", record)
    config = dict(epochs=1, curriculum=False, learning_rate=0.003, batch_size=2, num_samples=2,
                  move_mode="soft", loss="original", beta=1.0, entropy_coef=0.01, gradient_clip=5.0)
    combined_train.fit_programs(model, data, config, phases=[("mixed", 1)], seed=0)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected)
