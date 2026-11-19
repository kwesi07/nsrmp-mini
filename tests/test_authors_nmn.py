import torch

from nsrmp.baseline import AuthorsNMNBaseline
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.train import train


def test_authors_nmn_returns_boxes_and_a_distribution_over_real_objects_for_each_step():
    b = collate(make_dataset(8, seed=0, n_steps=(1, 3)))
    boxes, subj_w = AuthorsNMNBaseline()(b)
    assert boxes.shape == b.boxes.shape
    assert subj_w.shape == (8, b.subjects.shape[1], b.mask.shape[1])
    for k in range(8):
        n_steps = b.n_steps[k].item()
        active = subj_w[k, :n_steps]
        assert torch.allclose(active.sum(-1), torch.ones(n_steps), atol=1e-4)
        assert active[:, b.mask[k] == 0].abs().sum().item() == 0.0
        assert subj_w[k, n_steps:].abs().sum().item() == 0.0


def test_authors_nmn_reference_attention_is_a_distribution_over_real_objects_too():
    b = collate(make_dataset(8, seed=5))
    model = AuthorsNMNBaseline().eval()
    sentence = model.encode(b.tokens[:, 0], b.lengths[:, 0])
    ref = model.reference_head(b.feats, sentence, b.mask)
    assert torch.allclose(ref.sum(-1), torch.ones(8), atol=1e-4)
    assert ref[b.mask == 0].abs().sum().item() == 0.0


def test_authors_nmn_scores_objects_without_looking_at_boxes():
    b = collate(make_dataset(8, seed=1, n_steps=(1, 1)))
    model = AuthorsNMNBaseline().eval()
    shifted = b.select(range(8))
    shifted.boxes = torch.rand_like(b.boxes)
    assert torch.equal(model(b)[1], model(shifted)[1])
    assert not torch.allclose(model(b)[0], model(shifted)[0])


def test_authors_nmn_reads_the_words_not_the_parsed_program():
    b = collate(make_dataset(8, seed=2))
    model = AuthorsNMNBaseline().eval()
    scrambled = b.select(range(8))
    scrambled.programs = b.programs[::-1]
    assert torch.allclose(model(b)[0], model(scrambled)[0])
    other_words = b.select(range(8))
    other_words.tokens, other_words.lengths = b.tokens.flip(0), b.lengths.flip(0)
    assert not torch.allclose(model(b)[0], model(other_words)[0])


def test_authors_nmn_trains_for_a_few_steps_and_lowers_the_loss():
    data = collate(make_dataset(256, seed=3))
    model = AuthorsNMNBaseline()
    history = train(model, data, epochs=6, seed=0)
    assert all(torch.isfinite(torch.tensor(history)))
    assert history[-1] < history[0]


def test_authors_nmn_in_hard_mode_moves_only_the_argmax_subject():
    b = collate(make_dataset(8, seed=4, n_steps=(1, 1)))
    model = AuthorsNMNBaseline().eval()
    model.hard = True
    with torch.no_grad():
        pred, subj_w = model(b)
    others = torch.ones_like(b.mask, dtype=torch.bool)
    others[torch.arange(8), subj_w[:, 0].argmax(-1)] = False
    assert torch.equal(pred[others], b.boxes[others])
