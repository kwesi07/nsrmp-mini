import os
from pathlib import Path

import torch

from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.metrics import iou
from nsrmp.model import build
from nsrmp.parser import parse


def save_checkpoint(path, model, *, group, seed, phase, spec, move_mode, config, history):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(dict(group=group, seed=seed, phase=phase, spec=spec, move_mode=move_mode,
                    model=model.state_dict(), config=config, history=history), temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(checkpoint, examples=None, *, steps=None, parser_source=None, move_mode=None):
    state = torch.load(checkpoint, map_location="cpu")
    if examples is None:
        config = state["config"]
        torch.set_num_threads(config["threads"])
        offset = config[{1: "single_test_seed", 2: "multi_test_seed"}[steps]]
        examples = make_dataset(config["test_examples"], offset + state["seed"], n_steps=(steps, steps))
    spec = state["spec"]
    model = build(parser_source=spec["parser_source"], backend=spec["backend"]).eval()
    model.load_state_dict(state["model"])
    source = spec["parser_source"] if parser_source is None else parser_source
    mode = state["move_mode"] if move_mode is None else move_mode
    if source not in ("learned", "rule") or (source == "learned" and model.parser is None):
        raise ValueError(f"checkpoint cannot evaluate parser {source!r}")
    moved_count = 0
    overlap_sum = 0.0
    size = state["config"]["batch_size"]
    for start in range(0, len(examples), size):
        items = examples[start:start + size]
        batch = collate(items, surface_context=True, mode="learned")
        scene = batch.inputs
        if source == "rule":
            programs = [parse(e.text) for e in items]
        else:
            programs = model.parser.sample(scene, greedy=True).programs
        result = model.execute(scene, programs, move_mode=mode)
        moved = (scene.boxes - batch.final_boxes).abs().sum(-1).gt(1e-6) & scene.object_mask
        overlap_sum += iou(result.final_boxes, batch.final_boxes)[moved].double().sum().item()
        moved_count += int(moved.sum())
    return {"iou_m": overlap_sum / moved_count if moved_count else None,
            "moved_objects": moved_count, "examples": len(examples)}
