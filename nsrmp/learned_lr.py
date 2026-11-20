from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsrmp.data import keyword_kind
from . import _learned_parser


ROLES = _learned_parser.ROLES


def encode_step(clause, lexer_mode):
    step = _learned_parser.lex_step(clause, lexer_mode)
    return tuple(_learned_parser._WORD_ID[word] for word in step.context_tokens), step.keyword_positions


def decode_tokens(ids):
    return tuple(_learned_parser.VOCAB[word_id] for word_id in ids)


@lru_cache(maxsize=None)
def _role_mask(pattern, prefix):
    prototypes = {"colour": "red", "noun": "cube", "relation": "left"}
    grammar = _learned_parser.Grammar(tuple(prototypes[kind] for kind in pattern))
    for i in range(len(prefix)):
        grammar.valid_roles(prefix[:i])
    allowed = grammar.valid_roles(prefix)
    if not any(allowed):
        raise ValueError(f"expected a completable role assignment for {pattern!r} after {prefix!r}")
    return allowed


@dataclass
class ParserSample:
    programs: list
    role_traces: list
    log_prob: torch.Tensor
    entropy: torch.Tensor


class RolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(_learned_parser.VOCAB), 32, padding_idx=0)
        self.gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, len(ROLES))

    # Issue #1 (Policy): encode text and predict seven roles for each keyword.
    # surface_context retains context; keywords_only follows the issue's input.
    def forward(self, inputs):
        """inputs: ParserInputs (context IDs/lengths, keyword positions/mask, step mask)
        -> role_logits: Tensor[B, S, W, 7], ordered as ROLES
        """
        B, S, T = inputs.context_ids.shape
        W = inputs.keyword_positions.shape[-1]
        active = inputs.step_mask & (inputs.context_lengths > 0)
        logits = self.head.weight.new_zeros(B, S, W, len(ROLES))
        if not active.any():
            return logits
        embedded = self.emb(inputs.context_ids[active])
        packed = pack_padded_sequence(embedded, inputs.context_lengths[active].cpu(),
                                      batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=T)
        mask = inputs.keyword_mask[active]
        positions = inputs.keyword_positions[active].masked_fill(~mask, 0)
        keywords = hidden.gather(1, positions[..., None].expand(-1, -1, 128))
        logits[active] = self.head(keywords).masked_fill(~mask[..., None], 0)
        return logits

    def sample(self, inputs, greedy=False):
        return self.sample_many(inputs, count=1, greedy=greedy)[0]

    def sample_many(self, inputs, *, count, greedy):
        if count < 1:
            raise ValueError("count must be positive")
        step_pairs = inputs.step_mask.nonzero()
        words = [inputs.step_keywords(b, s) for b, s in step_pairs.tolist()]
        patterns = [tuple(map(keyword_kind, row)) for row in words] * count
        logits = self(inputs)[inputs.step_mask].repeat(count, 1, 1)
        traces = [() for _ in patterns]
        log_prob = logits.new_zeros(len(patterns))
        entropy = torch.zeros_like(log_prob)
        for pos in range(max(map(len, words), default=0)):
            active = [i for i, pattern in enumerate(patterns) if pos < len(pattern)]
            ids = step_pairs.new_tensor(active)
            allowed = torch.tensor([_role_mask(patterns[i], traces[i]) for i in active],
                                   device=logits.device)
            scores = logits[ids, pos].masked_fill(~allowed, -torch.inf)
            logs = scores.log_softmax(-1)
            probs = logs.exp()
            role = scores.argmax(-1, keepdim=True) if greedy else torch.multinomial(probs, 1)
            log_prob = log_prob.index_add(0, ids, logs.gather(1, role).flatten())
            local_entropy = -(probs * logs.masked_fill(~allowed, 0)).sum(-1)
            entropy = entropy.index_add(0, ids, local_entropy)
            for i, chosen in zip(active, role.flatten().tolist()):
                traces[i] += (ROLES[chosen],)

        size = len(words)
        rows = step_pairs[:, 0]
        lengths = inputs.n_steps.tolist()
        draws = []
        for draw in range(count):
            sl = slice(draw * size, (draw + 1) * size)
            moves = [_learned_parser.role_to_program(row, trace) for row, trace in zip(words, traces[sl])]
            programs, role_traces = [], []
            start = 0
            for length in lengths:
                stop = start + length
                programs.append(tuple(moves[start:stop]))
                role_traces.append(tuple(traces[sl.start + start:sl.start + stop]))
                start = stop
            draws.append(ParserSample(programs, role_traces,
                                      logits.new_zeros(len(inputs)).index_add(0, rows, log_prob[sl]),
                                      logits.new_zeros(len(inputs)).index_add(0, rows, entropy[sl])))
        return draws
