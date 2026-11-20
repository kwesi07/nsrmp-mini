from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass
class GroundedStep:
    actions: tuple
    subject: torch.Tensor
    reference: torch.Tensor


@dataclass
class ExecutionResult:
    final_boxes: torch.Tensor
    grounded_program: list[GroundedStep]


def _hard_weights(weights):
    selected = F.one_hot(weights.argmax(-1), num_classes=weights.shape[-1]).to(weights)
    active = weights.ne(0).any(-1, keepdim=True)
    return (selected + (weights - weights.detach())) * active


def _move(sim, boxes, subject, reference, action, move_mode):
    if move_mode == "hard":
        subject, reference = _hard_weights(subject), _hard_weights(reference)
    elif move_mode != "soft":
        raise ValueError(f"unknown move mode {move_mode!r}")
    subject_box = torch.einsum("bn,bnk->bk", subject, boxes)
    reference_box = torch.einsum("bn,bnk->bk", reference, boxes)
    target = sim(action, subject_box, reference_box)
    boxes = (1 - subject[..., None]) * boxes + subject[..., None] * target[:, None, :]
    return boxes, subject, reference


def apply_move(sim, boxes, subj_w, ref_w, action_emb, *, move_mode="soft"):
    return _move(sim, boxes, subj_w, ref_w, action_emb, move_mode)[0]


def execute_steps(scene, program, sim, ground_step, *, move_mode):
    boxes = scene.boxes
    steps = []
    for index in range(max(map(len, program))):
        moves = [p[index] if index < len(p) else None for p in program]
        subject, reference, action = ground_step(moves, scene.feats, boxes, scene.object_mask)
        boxes, subject, reference = _move(sim, boxes, subject, reference, action, move_mode)
        actions = tuple(move[1] if move is not None else None for move in moves)
        steps.append(GroundedStep(actions, subject, reference))
    return ExecutionResult(boxes, steps)
