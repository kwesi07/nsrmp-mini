"""Purely neural baselines (paper Sec. IV-B "NMN+" comparison).

`NeuralBaseline` is the end-to-end GRU baseline; `ParsedNeuralBaseline` (below) is the parse-matched one.

It sees the same object features/boxes and uses the same Action Simulator and loss as the
neuro-symbolic model, but reads the instruction with a GRU and never builds a program.
Each object is scored on its own (feature, box, sentence), so relations between objects have to
be squeezed into the sentence vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsrmp.batch import VOCAB
from nsrmp.data import ACTIONS, FEATURE_DIM
from nsrmp.model import ActionSimulator, apply_move


def _scorer(hidden):
    return nn.Sequential(nn.Linear(FEATURE_DIM + 4 + hidden, 64), nn.ReLU(), nn.Linear(64, 1))


class NeuralBaseline(nn.Module):
    def __init__(self, word_dim=32, hidden=64):
        super().__init__()
        self.words = nn.Embedding(len(VOCAB), word_dim, padding_idx=0)
        self.gru = nn.GRU(word_dim, hidden, batch_first=True)
        self.subject_head, self.reference_head = _scorer(hidden), _scorer(hidden)
        self.action_head = nn.Linear(hidden, len(ACTIONS))
        self.sim = ActionSimulator()
        self.hard = False  # set by evaluate(..., hard=True)

    def encode(self, tokens, lengths):
        """tokens (M, L), lengths (M,) -> sentence vector (M, hidden) = GRU state at the last word."""
        out, _ = self.gru(self.words(tokens))
        last = (lengths - 1).clamp(min=0)
        return out[torch.arange(len(tokens)), last]

    def _attend(self, head, feats, boxes, sent, mask):
        sent = sent[:, None, :].expand(-1, feats.shape[1], -1)
        logits = head(torch.cat([feats, boxes, sent], dim=-1))[..., 0]
        return F.softmax(logits.masked_fill(mask == 0, float("-inf")), dim=-1)

    def forward(self, batch):
        B, S, L = batch.tokens.shape
        sentences = self.encode(batch.tokens.view(B * S, L), batch.lengths.view(B * S)).view(B, S, -1)
        boxes, subj_ws = batch.boxes, []
        for k in range(S):
            active = (batch.n_steps > k).float()[:, None]
            subj = self._attend(self.subject_head, batch.feats, boxes, sentences[:, k], batch.mask) * active
            ref = self._attend(self.reference_head, batch.feats, boxes, sentences[:, k], batch.mask) * active
            action_probs = F.softmax(self.action_head(sentences[:, k]), dim=-1)
            boxes = apply_move(self.sim, boxes, subj, ref, action_probs @ self.sim.action_emb.weight, self.hard)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, dim=1)


class Find(nn.Module):
    """Neural attention over objects for one noun phrase, given its parsed word groups.

        word score        s(i, v) = <A f_i, B v>                    (bias-free dot product)
        anchor attention  a_j = softmax_j s(j, anchor_vec)
        context box       c   = sum_j a_j box_j
        attention         w_i = softmax_i( s(i, head_vec) + has_rel * MLP([box_i; c; 10 (box_i - c); rel_vec]) )

    Why this shape (README, "Making the parse-matched baseline learn"): with no words and no
    relation every object gets the same score, so attention can only become peaked by matching
    words; a concat-MLP scorer instead collapsed onto an arbitrary object. One module is shared
    by subject, reference and anchor phrases (as in NMN, and as NSModel shares Filter), so
    grounding learned for one role transfers to the others. The scaled box difference is the same
    input NSModel's relation MLP gets.
    """

    def __init__(self, word_dim, hidden):
        super().__init__()
        self.obj, self.word = nn.Linear(FEATURE_DIM, hidden, bias=False), nn.Linear(word_dim, hidden, bias=False)
        self.relation = nn.Sequential(nn.Linear(12 + word_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def score(self, feats, words):
        return torch.einsum("bnh,bh->bn", self.obj(feats), self.word(words))

    def forward(self, head_vec, rel_vec, anchor_vec, has_rel, feats, boxes, mask):
        N = feats.shape[1]
        anchor = _masked_softmax(self.score(feats, anchor_vec), mask)
        context = torch.einsum("bn,bnk->bk", anchor, boxes)[:, None].expand(-1, N, -1)
        pair = torch.cat([boxes, context, 10 * (boxes - context), rel_vec[:, None].expand(-1, N, -1)], dim=-1)
        logits = self.score(feats, head_vec) + has_rel[:, None] * self.relation(pair)[..., 0]
        return _masked_softmax(logits, mask)


def _masked_softmax(logits, mask):
    return F.softmax(logits.masked_fill(mask == 0, float("-inf")), dim=-1)


class ParsedNeuralBaseline(nn.Module):
    """Parse-matched neural baseline (closer to the paper's NMN+, which shares the language
    encoder and splitter with the main model).

    It gets the same parse as NSModel, as word groups per step (`ROLE_GROUPS` in batch.py): the
    subject's concept words / relation word / anchor concept words, the same for the reference,
    and the action word. Each group is a bag (mean) of word embeddings from one table. There is no
    Filter/Relate/Unique execution and there are no concept embeddings: each noun phrase is found
    by neural attention (`Find`), relations by attention-pooled context.
    """

    def __init__(self, word_dim=32, hidden=64):
        super().__init__()
        self.words = nn.Embedding(len(VOCAB), word_dim, padding_idx=0)
        self.find = Find(word_dim, hidden)
        self.action_head = nn.Linear(word_dim, len(ACTIONS))
        self.sim = ActionSimulator()
        self.hard = False  # set by evaluate(..., hard=True)

    def bag(self, ids):
        """ids (..., L) -> mean word embedding over non-padding words (zero for an empty group)."""
        present = (ids > 0).float()[..., None]
        return (self.words(ids) * present).sum(-2) / present.sum(-2).clamp(min=1)

    def forward(self, batch):
        groups = self.bag(batch.role_tokens)                    # (B, S, 7, D)
        has_rel = (batch.role_tokens > 0).any(-1).float()       # (B, S, 7)
        boxes, subj_ws = batch.boxes, []
        for k in range(batch.role_tokens.shape[1]):
            active = (batch.n_steps > k).float()[:, None]
            g, r = groups[:, k], has_rel[:, k]
            subj = self.find(g[:, 0], g[:, 1], g[:, 2], r[:, 1], batch.feats, boxes, batch.mask) * active
            ref = self.find(g[:, 3], g[:, 4], g[:, 5], r[:, 4], batch.feats, boxes, batch.mask) * active
            action_probs = F.softmax(self.action_head(g[:, 6]), dim=-1)
            boxes = apply_move(self.sim, boxes, subj, ref, action_probs @ self.sim.action_emb.weight, self.hard)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, dim=1)
