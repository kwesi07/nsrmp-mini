"""Turn Examples into padded tensors for rule-based or learned parsing."""
from dataclasses import dataclass

import numpy as np
import torch

from nsrmp.data import COLOURS, FEATURE_DIM, SHAPES
from nsrmp.learned_lr import decode_tokens, encode_step
from nsrmp.parser import parse, parse_roles, split_steps

VOCAB = (["<pad>", "put", "the", "which", "is", "to", "left", "right", "of", "in", "front",
          "behind", "object"] + COLOURS + SHAPES)
_WORD_ID = {w: i for i, w in enumerate(VOCAB)}
# Word groups the parser gives the parse-matched baseline, one row each in Batch.role_tokens.
ROLE_GROUPS = ("subj_head", "subj_rel", "subj_anchor", "ref_head", "ref_rel", "ref_anchor", "action")


@dataclass
class Batch:
    feats: torch.Tensor     # (B, N, 32)
    boxes: torch.Tensor     # (B, N, 4) initial boxes
    final: torch.Tensor     # (B, N, 4) gold final boxes
    mask: torch.Tensor      # (B, N) 1 for real objects, 0 for padding
    moved: torch.Tensor     # (B, N) bool, gold final box differs from the initial one
    subjects: torch.Tensor  # (B, S) gold subject per step, -1 for missing steps
    n_steps: torch.Tensor   # (B,)
    is_complex: torch.Tensor  # (B,) bool
    tokens: torch.Tensor    # (B, S, L) word ids per step (for the baseline)
    lengths: torch.Tensor   # (B, S)
    role_tokens: torch.Tensor  # (B, S, len(ROLE_GROUPS), Lr) word ids per parsed role, 0 = padding
    programs: list          # parsed program (tuple of Moves) per example

    def __len__(self):
        return len(self.programs)

    def select(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        fields = {name: (value[idx] if torch.is_tensor(value) else [value[i] for i in idx.tolist()])
                  for name, value in self.__dict__.items()}
        return Batch(**fields)


@dataclass
class ParserInputs:
    feats: torch.Tensor
    boxes: torch.Tensor
    object_mask: torch.Tensor
    context_ids: torch.Tensor
    context_lengths: torch.Tensor
    keyword_positions: torch.Tensor
    keyword_mask: torch.Tensor
    step_mask: torch.Tensor
    n_steps: torch.Tensor

    def step_keywords(self, b, s):
        mask = self.keyword_mask[b, s]
        positions = self.keyword_positions[b, s, mask]
        ids = self.context_ids[b, s, positions].tolist()
        keywords = decode_tokens(ids)
        if self.context_lengths[b, s] <= 0 or not 0 < len(keywords) <= 11:
            raise ValueError(f"expected 1 to 11 keywords in example {b}, "
                             f"step {s}: {keywords!r}")
        return keywords

    def __len__(self):
        return self.feats.shape[0]

    def select(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return ParserInputs(**{name: value[idx] for name, value in self.__dict__.items()})


# Keep parser inputs separate from final-scene targets and curriculum metadata.
@dataclass
class LearnedBatch:
    inputs: ParserInputs
    final_boxes: torch.Tensor
    is_complex: torch.Tensor

    def __len__(self):
        return len(self.inputs)

    def select(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return LearnedBatch(self.inputs.select(idx), self.final_boxes[idx], self.is_complex[idx])


def _role_groups(clause):
    r = parse_roles(clause)
    return [*r["subject"], *r["reference"], r["action"]]


def _scene_tensors(examples):
    B, N = len(examples), max(len(e.objs) for e in examples)
    feats = np.zeros((B, N, FEATURE_DIM))
    boxes, final = np.zeros((B, N, 4)), np.zeros((B, N, 4))
    mask = np.zeros((B, N), dtype=bool)
    for k, e in enumerate(examples):
        n = len(e.objs)
        feats[k, :n], mask[k, :n] = e.feats, True
        boxes[k, :n] = [o.box for o in e.objs]
        final[k, :n] = e.final_boxes
    return (torch.tensor(feats, dtype=torch.float32),
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(final, dtype=torch.float32),
            torch.tensor(mask))


def _collate_rule(examples):
    feats, boxes, final, object_mask = _scene_tensors(examples)
    B = len(examples)
    S = max(e.n_steps for e in examples)
    clauses = [split_steps(e.text) for e in examples]
    L = max(len(c.replace(",", " ").split()) for cs in clauses for c in cs)
    subjects = -np.ones((B, S), dtype=np.int64)
    tokens, lengths = np.zeros((B, S, L), dtype=np.int64), np.zeros((B, S), dtype=np.int64)
    roles = [[_role_groups(c) for c in cs] for cs in clauses]
    Lr = max(len(g) for rs in roles for r in rs for g in r)
    role_tokens = np.zeros((B, S, len(ROLE_GROUPS), Lr), dtype=np.int64)
    for k, e in enumerate(examples):
        subjects[k, :e.n_steps] = e.subjects
        for s, clause in enumerate(clauses[k]):
            words = clause.replace(",", " ").split()
            tokens[k, s, :len(words)] = [_WORD_ID[word] for word in words]
            lengths[k, s] = len(words)
            for g, group in enumerate(roles[k][s]):
                role_tokens[k, s, g, :len(group)] = [_WORD_ID[word] for word in group]
    return Batch(
        feats=feats, boxes=boxes, final=final, mask=object_mask.float(),
        moved=((boxes - final).abs().sum(-1) > 1e-6) & object_mask,
        subjects=torch.tensor(subjects),
        n_steps=torch.tensor([e.n_steps for e in examples]),
        is_complex=torch.tensor([e.is_complex for e in examples]),
        tokens=torch.tensor(tokens), lengths=torch.tensor(lengths),
        role_tokens=torch.tensor(role_tokens),
        programs=[parse(e.text) for e in examples])


# Build keyword-aligned inputs without reading gold roles or parsed programs.
# Targets remain outside ParserInputs in the returned LearnedBatch.
def _collate_learned(examples, surface_context):
    lexer_mode = "surface_context" if surface_context else "keywords_only"
    clauses = [[encode_step(c, lexer_mode) for c in split_steps(e.text)] for e in examples]
    feats, boxes, final, object_mask = _scene_tensors(examples)
    B = len(examples)
    S = max(len(cs) for cs in clauses)
    T = max(len(ids) for cs in clauses for ids, _ in cs)
    W = max(len(positions) for cs in clauses for _, positions in cs)
    context_ids = np.zeros((B, S, T), dtype=np.int64)
    context_lengths = np.zeros((B, S), dtype=np.int64)
    keyword_positions = np.zeros((B, S, W), dtype=np.int64)
    keyword_mask, step_mask = np.zeros((B, S, W), dtype=bool), np.zeros((B, S), dtype=bool)
    for k, steps in enumerate(clauses):
        for s, (ids, positions) in enumerate(steps):
            context_ids[k, s, :len(ids)] = ids
            context_lengths[k, s] = len(ids)
            keyword_positions[k, s, :len(positions)] = positions
            keyword_mask[k, s, :len(positions)] = True
            step_mask[k, s] = True
    inputs = ParserInputs(
        feats=feats, boxes=boxes, object_mask=object_mask,
        context_ids=torch.tensor(context_ids),
        context_lengths=torch.tensor(context_lengths),
        keyword_positions=torch.tensor(keyword_positions),
        keyword_mask=torch.tensor(keyword_mask),
        step_mask=torch.tensor(step_mask),
        n_steps=torch.tensor([len(cs) for cs in clauses], dtype=torch.long))
    return LearnedBatch(inputs, final, torch.tensor([e.is_complex for e in examples]))


# Preserve the legacy Batch by default; learned mode returns LearnedBatch.
# Only the learned path uses surface_context to retain non-keyword tokens.
def collate(examples, surface_context=False, *, mode="rule"):
    if not examples:
        raise ValueError("cannot collate an empty batch")
    if mode == "rule":
        if surface_context:
            raise ValueError("surface_context requires learned mode")
        return _collate_rule(examples)
    if mode == "learned":
        return _collate_learned(examples, surface_context)
    raise ValueError(f"unknown collate mode: {mode!r}")
