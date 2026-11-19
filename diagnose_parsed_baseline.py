"""Diagnostics behind the README section "Making the parse-matched baseline learn" (about 6 minutes).

Every attention variant we tried for the parse-matched baseline, plus the measurements that explain why
the spec's concat-MLP attention collapses. Diagnostic only: oracle rows use gold labels, which the
reported results never do. Writes results/diagnose_parsed_baseline.{json,txt}.

    python diagnose_parsed_baseline.py
"""
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nsrmp.baseline as baseline_module
import nsrmp.model as model_module
from nsrmp.baseline import Find, NeuralBaseline, ParsedNeuralBaseline, _masked_softmax
from nsrmp.batch import VOCAB, collate
from nsrmp.data import ACTIONS, COLOURS, FEATURE_DIM, RELATIONS, REL_PHRASE, make_dataset
from nsrmp.metrics import evaluate
from nsrmp.model import ActionSimulator, NSModel
from nsrmp.parser import split_steps
from nsrmp.train import loss_fn, train


@dataclass
class Settings:
    n_train: int = 4000
    n_test: int = 1000
    epochs_short: int = 30   # single-seed collapse checks
    epochs_long: int = 60    # the 3-seed runs, same as run_experiments.py
    seeds: tuple = (0, 1, 2)


# ---------------------------------------------------------------------------------------------
# Word groups. "concepts" is what the final baseline gets (batch.role_tokens). "full" is our first
# version: every word of the phrase, including "the" and "object".
# ---------------------------------------------------------------------------------------------
def _full_np(tokens, pos):
    start, pos = pos, pos + 1               # "the"
    pos += tokens[pos] in COLOURS          # optional colour
    pos += 1                               # shape or "object"
    head = tokens[start:pos]
    if tokens[pos:pos + 2] != ["which", "is"]:
        return (head, [], []), pos
    rel, pos = _match(tokens, pos + 2)
    (anchor, _, _), pos = _full_np(tokens, pos)
    return (head, [rel], anchor), pos


def _match(tokens, pos):
    for rel, phrase in REL_PHRASE.items():
        words = phrase.split()
        if tokens[pos:pos + len(words)] == words:
            return rel, pos + len(words)
    raise ValueError(tokens)


def _full_groups(clause):
    tokens = clause.split()
    subject, pos = _full_np(tokens, 1)
    action, pos = _match(tokens, pos)
    reference, _ = _full_np(tokens, pos)
    return [*subject, *reference, [action]]


def full_word_role_tokens(examples, batch):
    groups = [[_full_groups(c) for c in split_steps(e.text)] for e in examples]
    L = max(len(g) for steps in groups for step in steps for g in step)
    out = torch.zeros(batch.role_tokens.shape[:3] + (L,), dtype=torch.long)
    for k, steps in enumerate(groups):
        for s, step in enumerate(steps):
            for g, words in enumerate(step):
                out[k, s, g, :len(words)] = torch.tensor([VOCAB.index(w) for w in words], dtype=torch.long)
    return out


# ---------------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------------
def _mlp(n_in, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class _Product(nn.Module):
    """NMN Find-style fusion: w . ReLU(A x * B v)."""

    def __init__(self, obj_dim, word_dim, hidden=64):
        super().__init__()
        self.obj_dim = obj_dim
        self.a, self.b, self.out = nn.Linear(obj_dim, hidden), nn.Linear(word_dim, hidden), nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(torch.relu(self.a(x[..., :self.obj_dim]) * self.b(x[..., self.obj_dim:])))


class ConcatParsed(nn.Module):
    """Our first parse-matched baseline: per-role MLP([f_i; box_i; c; 10(box_i - c); head; rel])."""

    def __init__(self, fusion="concat", shared=False, boxes_only_via_relation=False, oracle=None, word_dim=32):
        super().__init__()
        self.words = nn.Embedding(len(VOCAB), word_dim, padding_idx=0)
        role_in, anchor_in = FEATURE_DIM + 12 + 2 * word_dim, FEATURE_DIM + 4 + word_dim
        if fusion == "concat":
            self.subject_head, self.reference_head, self.anchor_head = _mlp(role_in), _mlp(role_in), _mlp(anchor_in)
        else:
            self.subject_head = _Product(FEATURE_DIM + 12, 2 * word_dim)
            self.reference_head = _Product(FEATURE_DIM + 12, 2 * word_dim)
            self.anchor_head = _Product(FEATURE_DIM + 4, word_dim)
        if shared:
            self.reference_head = self.subject_head
        self.action_head = nn.Linear(word_dim, len(ACTIONS))
        self.sim = ActionSimulator()
        self.boxes_only_via_relation, self.oracle = boxes_only_via_relation, oracle
        self.last_subj = self.last_ref = None

    def bag(self, ids):
        present = (ids > 0).float()[..., None]
        return (self.words(ids) * present).sum(-2) / present.sum(-2).clamp(min=1)

    def find(self, head, groups, ids, feats, boxes, mask):
        N = feats.shape[1]
        head_vec, rel_vec, anchor_vec = groups.unbind(1)
        anchor_in = torch.cat([feats, boxes, anchor_vec[:, None].expand(-1, N, -1)], -1)
        anchor = _masked_softmax(self.anchor_head(anchor_in)[..., 0], mask)
        has_rel = (ids[:, 1] > 0).any(-1).float()[:, None, None]
        context = torch.einsum("bn,bnk->bk", anchor, boxes)[:, None].expand(-1, N, -1)
        if self.boxes_only_via_relation:
            spatial = torch.cat([boxes, context, 10 * (boxes - context)], -1) * has_rel
        else:
            spatial = torch.cat([boxes, context * has_rel, 10 * (boxes - context * has_rel)], -1)
        words = torch.cat([head_vec, rel_vec], -1)[:, None].expand(-1, N, -1)
        return _masked_softmax(head(torch.cat([feats, spatial, words], -1))[..., 0], mask)

    def forward(self, batch):
        groups = self.bag(batch.role_tokens)
        boxes, subj_ws, rel_ids = batch.boxes, [], torch.tensor([VOCAB.index(r) for r in RELATIONS])
        for k in range(batch.role_tokens.shape[1]):
            active = (batch.n_steps > k).float()[:, None]
            ids, g = batch.role_tokens[:, k], groups[:, k]
            subj = self.find(self.subject_head, g[:, 0:3], ids[:, 0:3], batch.feats, boxes, batch.mask) * active
            if self.oracle == "subject":
                subj = F.one_hot(batch.subjects[:, k].clamp(min=0), boxes.shape[1]).float() * active
            ref = self.find(self.reference_head, g[:, 3:6], ids[:, 3:6], batch.feats, boxes, batch.mask) * active
            if self.oracle == "action":
                action = self.sim.action_emb((ids[:, 6, :1] == rel_ids).float().argmax(-1))
            else:
                action = F.softmax(self.action_head(g[:, 6]), -1) @ self.sim.action_emb.weight
            if k == 0:
                self.last_subj, self.last_ref = subj, ref
            boxes = model_module.apply_move(self.sim, boxes, subj, ref, action)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, 1)


class _FindOwnAnchor(Find):
    """Find whose anchor attention has its own word-object projection."""

    def __init__(self, word_dim, hidden):
        super().__init__(word_dim, hidden)
        self.anchor_obj = nn.Linear(FEATURE_DIM, hidden, bias=False)
        self.anchor_word = nn.Linear(word_dim, hidden, bias=False)

    def forward(self, head_vec, rel_vec, anchor_vec, has_rel, feats, boxes, mask):
        N = feats.shape[1]
        anchor_logits = torch.einsum("bnh,bh->bn", self.anchor_obj(feats), self.anchor_word(anchor_vec))
        anchor = _masked_softmax(anchor_logits, mask)
        context = torch.einsum("bn,bnk->bk", anchor, boxes)[:, None].expand(-1, N, -1)
        pair = torch.cat([boxes, context, 10 * (boxes - context), rel_vec[:, None].expand(-1, N, -1)], dim=-1)
        logits = self.score(feats, head_vec) + has_rel[:, None] * self.relation(pair)[..., 0]
        return _masked_softmax(logits, mask)


class DotParsed(ParsedNeuralBaseline):
    """The final dot-product baseline, with Find per role ("per_role") or shared but with separate
    anchor weights ("shared_own_anchor")."""

    def __init__(self, mode):
        super().__init__()
        if mode == "per_role":
            self.subject_find, self.reference_find = _FindOwnAnchor(32, 64), _FindOwnAnchor(32, 64)
        else:
            self.subject_find = self.reference_find = _FindOwnAnchor(32, 64)
        del self.find

    def forward(self, batch):
        groups, has_rel = self.bag(batch.role_tokens), (batch.role_tokens > 0).any(-1).float()
        boxes, subj_ws = batch.boxes, []
        for k in range(batch.role_tokens.shape[1]):
            active = (batch.n_steps > k).float()[:, None]
            g, r = groups[:, k], has_rel[:, k]
            subj = self.subject_find(g[:, 0], g[:, 1], g[:, 2], r[:, 1], batch.feats, boxes, batch.mask) * active
            ref = self.reference_find(g[:, 3], g[:, 4], g[:, 5], r[:, 4], batch.feats, boxes, batch.mask) * active
            action = F.softmax(self.action_head(g[:, 6]), -1) @ self.sim.action_emb.weight
            boxes = model_module.apply_move(self.sim, boxes, subj, ref, action)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, 1)


@contextmanager
def subject_excluded_from_reference():
    """Patch apply_move for every model: reference weights renormalised over objects other than the subject."""
    original = model_module.apply_move

    def patched(sim, boxes, subj_w, ref_w, action_emb, hard=False):
        ref = ref_w * (1 - subj_w)
        return original(sim, boxes, subj_w, ref / (ref.sum(-1, keepdim=True) + 1e-6), action_emb, hard)

    model_module.apply_move = baseline_module.apply_move = patched
    try:
        yield
    finally:
        model_module.apply_move = baseline_module.apply_move = original


# (row name, model factory, word groups, epochs, all seeds?, subject excluded from reference?)
ROWS = [
    ("Concat-MLP attention per role (spec design)", lambda: ConcatParsed(), "full", "short", False, False),
    ("... one head shared by subject and reference", lambda: ConcatParsed(shared=True), "full", "short", False, False),
    ("... NMN-style product fusion", lambda: ConcatParsed(fusion="product"), "full", "short", False, False),
    ("... product fusion, shared head", lambda: ConcatParsed(fusion="product", shared=True), "full", "short", False, False),
    ("... boxes fed only through the relation term", lambda: ConcatParsed(boxes_only_via_relation=True),
     "full", "short", False, False),
    ("... subject excluded from the reference", lambda: ConcatParsed(), "full", "short", False, True),
    ("GRU baseline, subject excluded from the reference", NeuralBaseline, "full", "short", False, True),
    ("NSModel, subject excluded from the reference", NSModel, "full", "short", False, True),
    ("... gold action given (diagnostic)", lambda: ConcatParsed(oracle="action"), "full", "short", False, False),
    ("... gold subject given (diagnostic)", lambda: ConcatParsed(oracle="subject"), "full", "short", False, False),
    ("Dot-product word score, 'the' kept", lambda: DotParsed("per_role"), "full", "short", False, False),
    ("Dot-product, separate Find per role, concept words only", lambda: DotParsed("per_role"),
     "concepts", "long", True, False),
    ("Dot-product, shared Find with separate anchor weights", lambda: DotParsed("shared_own_anchor"),
     "concepts", "long", True, False),
    ("Dot-product, one shared Find (final ParsedNeuralBaseline)", ParsedNeuralBaseline, "concepts", "long", True, False),
]


def _data(cfg, seed, words, cache):
    if (seed, words) not in cache:
        sets = []
        for count, data_seed in ((cfg.n_train, seed), (cfg.n_test, 1000 + seed)):
            examples = make_dataset(count, seed=data_seed)
            batch = collate(examples)
            if words == "full":
                batch.role_tokens = full_word_role_tokens(examples, batch)
            sets.append(batch)
        cache[(seed, words)] = sets
    return cache[(seed, words)]


def _split_metrics(model, test):
    out = {}
    for name, keep in (("simple", ~test.is_complex), ("complex", test.is_complex)):
        r = evaluate(model, test.select(keep.nonzero().flatten()))
        out[f"iou_m_{name}"], out[f"subject_acc_{name}"] = r["iou_m"], r["subject_acc"]
    return out


def _mechanism(cfg, cache):
    """Train the spec-design baseline (seed 0) while recording, on the test set, the entropy of the
    subject attention and the weight on the gold subject; then measure how often subject and
    reference pick the same object, and how often the initial gradient favours the gold subject."""
    train_set, test = _data(cfg, 0, "full", cache)
    torch.manual_seed(0)
    model = ConcatParsed()
    entropy, gold = [], []

    def record(epoch, n):
        model.eval()
        with torch.no_grad():
            w = model(test)[1][:, 0]
        entropy.append((-(w.clamp(min=1e-9).log() * w).sum(-1)).mean().item())
        gold.append(w.gather(1, test.subjects[:, :1]).mean().item())

    train(model, train_set, cfg.epochs_short, seed=0, on_epoch=record)
    model.eval()
    with torch.no_grad():
        model(test)
    same = (model.last_subj.argmax(-1) == model.last_ref.argmax(-1)).float().mean().item()

    torch.manual_seed(0)
    fresh = ConcatParsed()
    simple = test.select((~test.is_complex).nonzero().flatten())
    pred, _ = fresh(simple)
    fresh.last_subj.retain_grad()
    loss_fn(pred, simple.final, simple.mask).backward()
    g = -fresh.last_subj.grad
    gold_g = g.gather(1, simple.subjects[:, :1])[:, 0]
    others = ((g * simple.mask).sum(-1) - gold_g) / (simple.mask.sum(-1) - 1)
    return {"entropy_by_epoch": entropy, "gold_weight_by_epoch": gold, "same_object_fraction": same,
            "gold_gradient_favoured_fraction": (gold_g > others).float().mean().item()}


def run_all(cfg=Settings()):
    cache, rows = {}, []
    for name, factory, words, length, all_seeds, exclude in ROWS:
        per_seed = []
        for seed in (cfg.seeds if all_seeds else cfg.seeds[:1]):
            train_set, test = _data(cfg, seed, words, cache)
            epochs = cfg.epochs_long if length == "long" else cfg.epochs_short
            torch.manual_seed(seed)
            if exclude:
                with subject_excluded_from_reference():
                    model = factory()
                    train(model, train_set, epochs, seed=seed)
                    metrics = _split_metrics(model, test)
            else:
                model = factory()
                train(model, train_set, epochs, seed=seed)
                metrics = _split_metrics(model, test)
            per_seed.append({"seed": seed, **metrics})
        rows.append({"row": name, "epochs": epochs, "per_seed": per_seed})
        print(_format_row(rows[-1]), flush=True)
    return {"rows": rows, "mechanism": _mechanism(cfg, cache)}


def _format_row(row):
    cells = [f"seed {r['seed']}: IOU-M {r['iou_m_simple']:.3f}/{r['iou_m_complex']:.3f}, "
             f"subject acc. {r['subject_acc_simple']:.3f}/{r['subject_acc_complex']:.3f}" for r in row["per_seed"]]
    return f"{row['row']} ({row['epochs']} epochs; simple/complex) | " + "; ".join(cells)


def main():
    results = run_all()
    m = results["mechanism"]
    lines = [_format_row(r) for r in results["rows"]] + [
        "subject-attention entropy on the test set by epoch: " + ", ".join(f"{x:.2f}" for x in m["entropy_by_epoch"]),
        "weight on the gold subject by epoch: " + ", ".join(f"{x:.2f}" for x in m["gold_weight_by_epoch"]),
        f"subject and reference attention pick the same object: {m['same_object_fraction']:.2f}",
        f"initial -dL/dw favours the gold subject over the others: {m['gold_gradient_favoured_fraction']:.2f}"]
    Path("results").mkdir(exist_ok=True)
    Path("results/diagnose_parsed_baseline.txt").write_text("\n".join(lines) + "\n")
    Path("results/diagnose_parsed_baseline.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
