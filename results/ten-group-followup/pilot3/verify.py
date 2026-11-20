import json
import math
from pathlib import Path

import torch

from nsrmp.batch import collate
from nsrmp.model import build
from nsrmp.data import make_dataset, training_examples
from experiment_eval import evaluate
from nsrmp.combined_train import epoch_indices, shuffle_epoch
from nsrmp.parser import parse

torch.set_num_threads(1)
root = Path(__file__).parent / "curriculum"
cfg = json.loads((root / "config.json").read_text())
summary = json.loads((root / "summary.json").read_text())
assert len(summary["runs"]) == 10
assert not (root / "shared").exists()
mixed = make_dataset(35, seed=1999, n_steps=(1, 2))
scene = collate(mixed, mode="learned", surface_context=True).inputs
programs = [parse(e.text) for e in mixed]
assert programs == [e.program for e in mixed]
initial = {}
for run in summary["runs"]:
    g, s = run["group"], run["seed"]
    directory = root / f"group-{g}/seed-{s}"
    warm = torch.load(directory / "warm.pt")
    final = torch.load(directory / run["checkpoint"])
    spec = final["spec"]
    assert (final["group"], final["seed"]) == (g, 0)
    learned = g in (1, 2, 5, 6, 9, 10)
    assert (spec["parser_source"] == "learned") == learned
    assert (spec["backend"] == "neural") == (g >= 9)
    assert (spec["move_mode"] == "hard") == (g in (6, 8, 10))
    assert (spec["loss"] == "original") == (g in (1, 3))
    phases = ("warm", "mixed") if g <= 4 else ("warm", "sim", "relation")
    assert tuple(final["history"]) == phases
    for phase in phases:
        h = final["history"][phase]
        size = 200 if phase in ("warm", "sim") else 400
        epochs = cfg[f"{phase}_epochs"]
        assert h["examples_per_epoch"] == size
        assert len(h["scene_loss"]) == epochs and all(map(math.isfinite, h["scene_loss"]))
        assert h["updates"] == math.ceil(size / cfg["batch_size"]) * epochs
    torch.manual_seed(s)
    model = build(parser_source=spec["parser_source"], backend=spec["backend"])
    initial[g] = model.state_dict()
    assert (model.parser is not None) == learned
    assert not learned or any(not torch.equal(v, warm["model"][k]) for k,v in initial[g].items()
                              if k.startswith("parser."))
    if g >= 5:
        sim = torch.load(directory / "sim.pt")
        sim_changes = {k for k,v in sim["model"].items() if not torch.equal(v, warm["model"][k])}
        assert sim_changes and all(k.startswith("sim.") for k in sim_changes)
        relation = "find.relation." if g >= 9 else "concepts.rel_mlp."
        changed = {k for k,v in final["model"].items() if not torch.equal(v, sim["model"][k])}
        assert any(k.startswith(relation) for k in changed)
        assert any(k.startswith("parser.") for k in changed) == learned
        assert all(k.startswith((relation, "parser.")) for k in changed)
    with torch.inference_mode():
        backend = build(parser_source=spec["parser_source"], backend=spec["backend"])
        backend.load_state_dict(final["model"])
        result = backend.execute(scene, programs, move_mode="hard")
        selected = torch.zeros_like(scene.object_mask)
        for i,step in enumerate(result.grounded_program):
            for w in (step.subject,step.reference):
                assert torch.all((w == 0) | (w == 1))
                assert torch.equal(w.sum(1), scene.step_mask[:,i].to(w))
                assert not w[~scene.object_mask].any()
            selected |= step.subject.bool()
        assert torch.equal(result.final_boxes[~selected], scene.boxes[~selected])
    for length,offset in [(1,cfg["single_test_seed"]),(2,cfg["multi_test_seed"])]:
        examples = make_dataset(150, offset+s, n_steps=(length,length))
        assert evaluate(directory / run["checkpoint"], examples) == run[f"{length}-step"]
for a,b in ((1,2),(3,4),(5,6),(7,8),(9,10),(5,7),(5,9)):
    for k in initial[a].keys() & initial[b].keys():
        assert torch.equal(initial[a][k], initial[b][k])
print("PASS: 10 independent fits; correct groups, epochs, updates, common initial parameters, stage freezes, hard writes and 20 exact held-out IoU-M replays")

examples = training_examples(cfg, 0)
base = make_dataset(400, 0)
assert len(examples) == 800 and all(e.is_complex for e in examples[400:])
assert all(a.text == b.text and (a.final_boxes == b.final_boxes).all() for a,b in zip(base, examples))
batch = collate(examples, mode="learned", surface_context=True)
shuffle = torch.Generator().manual_seed(0)
simple = shuffle_epoch(epoch_indices(batch, cfg, "warm"), shuffle)
mixed = shuffle_epoch(epoch_indices(batch, cfg, "mixed"), shuffle)
assert len(simple) == 200 and (simple < 400).all() and not batch.is_complex[simple].any()
assert len(mixed) == 400 and batch.is_complex[mixed].sum().item() == 200 and (mixed >= 400).any()
assert len(mixed.unique()) == 400
print("PASS: base scenes preserved; 400 extra complex scenes; same update budget and 50:50 mixed balance")
