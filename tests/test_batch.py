import numpy as np
import pytest

from nsrmp.batch import ROLE_GROUPS, VOCAB, collate
from nsrmp.data import make_dataset
from nsrmp.parser import parse_roles


def test_collate_pads_scenes_and_marks_exactly_the_subject_as_moved():
    exs = make_dataset(6, seed=0, n_objects=(3, 5))
    b = collate(exs)
    n_max = max(len(e.objs) for e in exs)
    assert b.feats.shape == (6, n_max, 32) and b.boxes.shape == (6, n_max, 4)
    for k, e in enumerate(exs):
        n = len(e.objs)
        assert b.mask[k].tolist() == [1.0] * n + [0.0] * (n_max - n)
        assert np.allclose(b.boxes[k, :n].numpy(), [o.box for o in e.objs])
        assert np.allclose(b.final[k, :n].numpy(), e.final_boxes)
        assert b.moved[k].nonzero().flatten().tolist() == [e.subjects[0]]
        assert b.subjects[k].tolist() == [e.subjects[0]]
        assert b.programs[k] == e.program


def test_collate_tokenises_each_step_separately_with_padding():
    exs = make_dataset(2, seed=1, n_steps=(2, 2), p_complex=0.0)
    b = collate(exs)
    assert b.tokens.shape[:2] == (2, 2) and b.n_steps.tolist() == [2, 2]
    for k, e in enumerate(exs):
        for s, clause in enumerate(e.text.split(", then ")):
            words = [VOCAB[t] for t in b.tokens[k, s, :b.lengths[k, s]].tolist()]
            assert words == clause.split()


def test_select_returns_the_requested_examples_in_order():
    b = collate(make_dataset(5, seed=2))
    sub = b.select([3, 1])
    assert sub.feats.equal(b.feats[[3, 1]]) and sub.programs == [b.programs[3], b.programs[1]]


def test_collate_stores_the_parsed_role_word_groups_of_every_step():
    exs = make_dataset(6, seed=3, n_steps=(1, 2), p_complex=1.0)
    b = collate(exs)
    assert b.role_tokens.shape[:3] == (6, 2, len(ROLE_GROUPS))
    for k, e in enumerate(exs):
        for s, clause in enumerate(e.text.split(", then ")):
            roles = parse_roles(clause)
            expected = list(roles["subject"]) + list(roles["reference"]) + [roles["action"]]
            for g, words in enumerate(expected):
                ids = [t for t in b.role_tokens[k, s, g].tolist() if t != 0]
                assert [VOCAB[t] for t in ids] == words
        assert b.role_tokens[k, e.n_steps:].abs().sum() == 0


@pytest.mark.parametrize("surface_context, positions", [(True, (2, 3, 5)), (False, (0, 1, 2))])
def test_learned_collate_preserves_keyword_positions(surface_context, positions):
    example = make_dataset(1, seed=4)[0]
    example.text = "put the cube to the right of the dice"
    inputs = collate([example], mode="learned", surface_context=surface_context).inputs
    assert tuple(inputs.keyword_positions[inputs.keyword_mask].tolist()) == positions
    assert inputs.step_keywords(0, 0) == ("cube", "right", "dice")
