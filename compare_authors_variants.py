"""Checks against the authors' code: hard (argmax) evaluation and a port of their NMN+ scoring.

Trains ours, the parse-matched baseline, the GRU baseline and `AuthorsNMNBaseline` on exactly the
phase-1 setting of run_experiments.py (same data seeds, epochs, curriculum), then evaluates each on
the phase-1 test set twice: soft (the reported evaluation) and hard (argmax subject and reference,
as in the authors' inference code). Writes results/authors_variants.{json,md}.

    python compare_authors_variants.py      # about 10 minutes on an M2 Pro CPU
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from nsrmp.baseline import AuthorsNMNBaseline, NeuralBaseline, ParsedNeuralBaseline
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.model import NSModel
from nsrmp.train import train
from run_experiments import LABELS, METRICS, Config, _evaluate_splits, fmt

MODELS = (("ours", NSModel), ("parsed_baseline", ParsedNeuralBaseline),
          ("baseline", NeuralBaseline), ("authors_nmn", AuthorsNMNBaseline))
ROW_LABELS = {**{k: LABELS[k] for k in ("ours", "parsed_baseline", "baseline")},
              "authors_nmn": "NMN+ (authors' scoring)"}
SPLITS = ("overall", "simple", "complex")


def run(cfg, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs, start = {}, time.time()
    for seed in cfg.seeds:
        train_data = collate(make_dataset(cfg.n_train, seed=seed, n_objects=(3, 5)))
        test = collate(make_dataset(cfg.n_test, seed=1000 + seed, n_objects=(3, 5)))
        for name, factory in MODELS:
            torch.manual_seed(seed)
            model = factory()
            t0 = time.time()
            history = train(model, train_data, cfg.epochs, seed=seed, curriculum=True)
            record = {"seed": seed, "final_train_loss": history[-1], "train_seconds": time.time() - t0,
                      "soft": _evaluate_splits(model, test), "hard": _evaluate_splits(model, test, hard=True)}
            runs.setdefault(name, []).append(record)
            print(f"seed {seed} {name}: IOU-M soft {record['soft']['overall']['iou_m']:.3f} "
                  f"hard {record['hard']['overall']['iou_m']:.3f} ({record['train_seconds']:.0f}s)", flush=True)
    results = {"config": asdict(cfg), "runs": runs, "total_seconds": time.time() - start}
    (outdir / "authors_variants.json").write_text(json.dumps(results, indent=1))
    (outdir / "authors_variants.md").write_text(_markdown(results))
    return results


def _table(results, mode):
    lines = ["| Model | Split | IOU | IOU-M | Subject acc. | Placement acc. |", "|---|---|---|---|---|---|"]
    for name, runs in results["runs"].items():
        for split in SPLITS:
            cells = " | ".join(fmt([r[mode][split][m] for r in runs]) for m in METRICS)
            lines.append(f"| {ROW_LABELS[name]} | {split} | {cells} |")
    return "\n".join(lines)


def _markdown(results):
    return ("### Soft evaluation (current): every object blended by its subject weight "
            "(mean ± sample std over seeds)\n\n" + _table(results, "soft")
            + "\n\n### Hard evaluation (authors' inference): argmax subject and reference, only the subject moves "
            "(mean ± sample std over seeds)\n\n" + _table(results, "hard")
            + f"\n\nTotal runtime: {results['total_seconds']:.0f} s.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run(Config(), args.out)


if __name__ == "__main__":
    main()
