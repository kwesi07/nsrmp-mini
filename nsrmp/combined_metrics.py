"""Offline scene and exact-program evaluation for the learned parser."""
from dataclasses import dataclass

import torch

from nsrmp.learned_lr import RolePolicy
from nsrmp.metrics import iou


@dataclass
class EvaluationAnnotations:
    programs: list
    subjects: torch.Tensor
    moved: torch.Tensor


def build_annotations(examples, learned_batch):
    """Align gold Move tuples and subject indices with an evaluation batch."""
    inputs = learned_batch.inputs
    if len(examples) != len(learned_batch):
        raise ValueError("expected one annotation per example: mismatched examples count")
    if inputs.object_mask.shape != inputs.boxes.shape[:2]:
        raise ValueError("expected object mask aligned with boxes")

    subjects = torch.full_like(inputs.step_mask, -1, dtype=torch.long)
    programs = []
    for b, example in enumerate(examples):
        steps = inputs.step_mask[b]
        objects = inputs.object_mask[b]
        program, indices = example.program, example.subjects
        if len(program) != int(steps.sum()) or len(program) != len(indices):
            raise ValueError(f"expected gold program and subjects for all valid steps: example {b}")
        if len(example.objs) != int(objects.sum()):
            raise ValueError(f"expected object mask to match the number of objects: example {b}")
        for idx in indices:
            if not 0 <= idx < len(objects) or not objects[idx]:
                raise ValueError(f"expected a valid subject index: example {b}, index {idx}")
        subjects[b, steps] = subjects.new_tensor(indices)
        programs.append(program)
    moved = ((inputs.boxes - learned_batch.final_boxes).abs().sum(-1) > 1e-6)
    return EvaluationAnnotations(programs, subjects, moved & inputs.object_mask)


_DENOMINATORS = {"iou": "objects", "iou_m": "moved", "subject_acc": "examples",
                 "placement_acc": "moved", "program_acc": "examples", "step_program_acc": "steps"}


def _scene_totals(programs, result, batch, annotations, rows, gold):
    inputs = batch.inputs
    steps = inputs.step_mask
    pred = result.final_boxes
    weights = torch.stack([step.subject for step in result.grounded_program], dim=1)
    width = weights.shape[1]
    traced_steps = steps[:, :width]
    objects = inputs.object_mask
    weights = weights.masked_fill(~objects[:, None, :], -torch.inf)
    valid_weights = traced_steps[..., None] & objects[:, None, :]
    if not (torch.isfinite(pred[objects]).all() and torch.isfinite(weights[valid_weights]).all()):
        raise ValueError("non-finite scene prediction")
    moved = annotations.moved[rows] & objects
    target = batch.final_boxes
    overlap = iou(pred, target)
    center = (pred[..., :2] + pred[..., 2:]) / 2
    inside = ((center >= target[..., :2]) & (center <= target[..., 2:])).all(-1)
    correct = (weights.argmax(-1) == annotations.subjects[rows, :width]) & traced_steps
    step_matches = [sum(a == b for a, b in zip(p, g)) for p, g in zip(programs, gold)]
    return {
        "examples": torch.ones_like(inputs.n_steps),
        "objects": objects.sum(-1),
        "moved": moved.sum(-1),
        "steps": steps.sum(-1),
        "iou": overlap.masked_fill(~objects, 0).double().sum(-1),
        "iou_m": overlap.masked_fill(~moved, 0).double().sum(-1),
        "subject_acc": correct.sum(-1).double() / steps.sum(-1),
        "placement_acc": (inside & moved).sum(-1),
        "program_acc": steps.new_tensor([p == g for p, g in zip(programs, gold)]),
        "step_program_acc": inputs.n_steps.new_tensor(step_matches),
    }


def _diagnose(parser, inputs, gold, chunk, sampled_count, uniform_count, seed):
    sampled_matches = uniform_matches = 0
    if len(inputs) and (sampled_count or uniform_count):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            uniform = RolePolicy().eval() if uniform_count else None
            if uniform is not None:
                uniform.head.weight.zero_()
                uniform.head.bias.zero_()
            for start in range(0, len(inputs), chunk):
                stop = min(start + chunk, len(inputs))
                current = inputs.select(range(start, stop))
                expected = gold[start:stop]
                for policy, count in ((parser, sampled_count), (uniform, uniform_count)):
                    if not count:
                        continue
                    draws = policy.sample_many(current, count=count, greedy=False)
                    matches = sum(p == g for draw in draws for p, g in zip(draw.programs, expected))
                    if policy is parser:
                        sampled_matches += matches
                    else:
                        uniform_matches += matches
    result = {"seed": seed}
    for name, count, matches in (("sampled", sampled_count, sampled_matches),
                                 ("mask_uniform", uniform_count, uniform_matches)):
        draws = len(inputs) * count
        result[name] = {"count": count, "matches": matches, "draws": draws,
                        "program_acc": matches / draws if draws else None}
    return result


@torch.no_grad()
def evaluate_learned(model, learned_batch, annotations, *, chunk=500,
                     sampled_count=0, uniform_count=0, eval_seed=0):
    """Evaluate greedy programs and scenes; empty denominators are reported as None."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if sampled_count < 0 or uniform_count < 0:
        raise ValueError("diagnostic sample counts must be non-negative")
    if len(annotations.programs) != len(learned_batch):
        raise ValueError("expected one annotation per example: mismatched examples count")
    was_training = model.training
    model.eval()
    groups = ("overall", "simple", "complex")
    totals = {group: dict.fromkeys((*_DENOMINATORS, "examples", "objects", "moved", "steps"), 0)
              for group in groups}
    try:
        for start in range(0, len(learned_batch), chunk):
            rows = range(start, min(start + chunk, len(learned_batch)))
            batch = learned_batch.select(rows)
            gold = annotations.programs[rows.start:rows.stop]
            programs = model.parser.sample(batch.inputs, greedy=True).programs
            result = model.execute(batch.inputs, programs, move_mode="soft")
            values = _scene_totals(programs, result, batch, annotations, rows, gold)
            for group, keep in zip(groups, (torch.ones_like(batch.is_complex),
                                            ~batch.is_complex, batch.is_complex)):
                for key, value in values.items():
                    totals[group][key] += value[keep].sum().item()
        diagnostics = _diagnose(model.parser, learned_batch.inputs, annotations.programs, chunk,
                                sampled_count, uniform_count, eval_seed)
    finally:
        model.train(was_training)

    result = {group: {
        **{key: total[key] / total[n] if total[n] else None for key, n in _DENOMINATORS.items()},
        "counts": {key: total[key] for key in ("examples", "objects", "moved", "steps")},
    } for group, total in totals.items()}
    return {**result, "diagnostics": diagnostics}
