"""Neuro-symbolic model (paper Sec. III): concepts + Action Simulator + program execution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsrmp.data import ANY, CONCEPTS, FEATURE_DIM
from nsrmp.executor import Executor

GAMMA, TAU = 0.2, 0.1  # Filter threshold and temperature (NS-CL style)
N_REAL_CONCEPTS = len(CONCEPTS) - 1  # every concept except the ANY wildcard


class Concepts(nn.Module):
    """Learned concept embeddings (colours, shapes) and relation classifiers."""

    def __init__(self, feat_dim=FEATURE_DIM, dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, dim)              # W: object feature -> concept space
        self.emb = nn.Embedding(N_REAL_CONCEPTS, dim)     # e_c
        # pair input: both boxes and their (scaled-up) difference, so tiny offsets are easy to see
        self.rel_mlp = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 4))

    def filter_scores(self, feats, concept_ids):
        """feats (B, N, F), concept_ids (B,) -> sigma((cos(W f_i, e_c) - gamma) / tau), shape (B, N)."""
        v = F.normalize(self.proj(feats), dim=-1)
        e = F.normalize(self.emb(concept_ids.clamp(max=N_REAL_CONCEPTS - 1)), dim=-1)
        score = torch.sigmoid((torch.einsum("bnd,bd->bn", v, e) - GAMMA) / TAU)
        is_any = (concept_ids == CONCEPTS.index(ANY))[:, None]
        return torch.where(is_any, torch.ones_like(score), score)

    def relation_probs(self, boxes, rel_ids):
        """boxes (B, N, 4), rel_ids (B,) -> probs[b, i, j] = P(object i is rel of object j)."""
        N = boxes.shape[1]
        bi, bj = boxes[:, :, None, :].expand(-1, -1, N, -1), boxes[:, None, :, :].expand(-1, N, -1, -1)
        logits = self.rel_mlp(torch.cat([bi, bj, 10 * (bi - bj)], dim=-1))  # (B, N, N, 4)
        idx = rel_ids[:, None, None, None].expand(-1, N, N, 1)
        return torch.sigmoid(logits.gather(-1, idx)[..., 0])


class ActionSimulator(nn.Module):
    """Paper Sec. III-C: (action, subject box, reference box) -> new subject box.

    The MLP predicts an offset added to the subject box, which makes "stay the same size" the
    easy default to learn.
    """

    def __init__(self, n_actions=4, emb_dim=16, hidden=128):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim + 8, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, action_emb, subj_box, ref_box):
        return subj_box + self.mlp(torch.cat([action_emb, subj_box, ref_box], dim=-1))


def _one_hot_argmax(w):
    """(B, N) weights -> one-hot of the argmax; rows with no weight at all (inactive step) stay zero."""
    return F.one_hot(w.argmax(-1), w.shape[-1]).float() * (w.sum(-1, keepdim=True) > 0).float()


def apply_move(sim, boxes, subj_w, ref_w, action_emb, hard=False):
    """Move the (soft) subject next to the (soft) reference; other objects keep their boxes.

    boxes (B, N, 4); subj_w, ref_w (B, N); action_emb (B, E).
    box_hat_i = (1 - w_i) * box_i + w_i * AS(action, subject box, reference box)

    With `hard=True` (evaluation, as in the authors' inference code) both weights are first
    replaced by their one-hot argmax: the simulator sees the argmax subject and reference boxes and
    only the argmax subject's box is overwritten.
    """
    if hard:
        subj_w, ref_w = _one_hot_argmax(subj_w), _one_hot_argmax(ref_w)
    subj_box = torch.einsum("bn,bnk->bk", subj_w, boxes)
    ref_box = torch.einsum("bn,bnk->bk", ref_w, boxes)
    new_box = sim(action_emb, subj_box, ref_box)
    return (1 - subj_w[..., None]) * boxes + subj_w[..., None] * new_box[:, None, :]


class NSModel(nn.Module):
    """Parser-given programs executed softly; trained end to end from final boxes only."""

    def __init__(self):
        super().__init__()
        self.concepts = Concepts()
        self.sim = ActionSimulator()
        self.executor = Executor(self.concepts)
        self.hard = False  # evaluate(..., hard=True) switches this on for argmax evaluation

    def forward(self, batch):
        """Returns predicted final boxes (B, N, 4) and subject weights per step (B, S, N)."""
        boxes, subj_ws = batch.boxes, []
        for k in range(batch.subjects.shape[1]):
            moves = [prog[k] if len(prog) > k else None for prog in batch.programs]
            subj, ref, action = self.executor.run_step(moves, batch.feats, boxes, batch.mask)
            boxes = apply_move(self.sim, boxes, subj, ref, self.sim.action_emb(action), self.hard)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, dim=1)
