"""Run the experiments and write results/ (JSON, markdown tables, PNG plots).

Paper link: Sec. IV, Table II (simple vs complex instructions, ours vs NMN+) and Fig. 4
(generalisation to more objects / more steps). Everything is reported as mean +- sample std
(ddof=1) over seeds.

    python run_experiments.py --phase 1      # Table II + Fig. 4a analogues
    python run_experiments.py --phase 2      # Fig. 4b analogue (multi-step)
    python run_experiments.py --phase all
    python run_experiments.py --phase noise  # feature-noise sweep (not part of "all")
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsrmp.baseline import NeuralBaseline, ParsedNeuralBaseline  # noqa: E402
from nsrmp.batch import collate  # noqa: E402
from nsrmp.data import make_dataset  # noqa: E402
from nsrmp.metrics import evaluate  # noqa: E402
from nsrmp.model import NSModel  # noqa: E402
from nsrmp.train import train  # noqa: E402

METRICS = ("iou", "iou_m", "subject_acc", "placement_acc")
LABELS = {"ours": "Ours (neuro-symbolic)",
          "parsed_baseline": "Parse-matched neural baseline (NMN+-like)",
          "baseline": "End-to-end GRU baseline",
          "ours_no_curriculum": "Ours, no curriculum"}


@dataclass
class Config:
    n_train: int = 4000
    n_test: int = 1000
    n_gen: int = 500      # test examples per object count / step count
    epochs: int = 60
    seeds: tuple = (0, 1, 2)


def mean_std(values):
    """Mean and sample standard deviation (ddof=1: only 3 seeds); std is 0 for a single value."""
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt(values):
    m, s = mean_std(values)
    return f"{m:.3f} ± {s:.3f}" if len(values) > 1 else f"{m:.3f} ± n/a"


def _subset(batch, keep):
    return batch.select(keep.nonzero().flatten())


def _evaluate_splits(model, test, hard=False):
    return {"overall": evaluate(model, test, hard=hard),
            "simple": evaluate(model, _subset(test, ~test.is_complex), hard=hard),
            "complex": evaluate(model, _subset(test, test.is_complex), hard=hard)}


def _models(curriculum_ablation):
    """(name, factory, use_curriculum)."""
    entries = [("ours", NSModel, True), ("parsed_baseline", ParsedNeuralBaseline, True),
               ("baseline", NeuralBaseline, True)]
    if curriculum_ablation:
        entries.append(("ours_no_curriculum", NSModel, False))
    return entries


def _run_all(cfg, outdir, train_kwargs, test_kwargs, gen_axis, gen_kwargs, name, ablation):
    """Shared driver: train every model on every seed, evaluate on the test set and on the
    generalisation sets (one per value of `gen_axis`)."""
    runs = {}
    start = time.time()
    for seed in cfg.seeds:
        train_data = collate(make_dataset(cfg.n_train, seed=seed, **train_kwargs))
        test = collate(make_dataset(cfg.n_test, seed=1000 + seed, **test_kwargs))
        gen = {v: collate(make_dataset(cfg.n_gen, seed=2000 + 100 * seed + v, **gen_kwargs(v)))
               for v in gen_axis}
        for model_name, factory, curriculum in _models(ablation):
            torch.manual_seed(seed)
            model = factory()
            t0 = time.time()
            history = train(model, train_data, cfg.epochs, seed=seed, curriculum=curriculum)
            record = {"seed": seed, "final_train_loss": history[-1], "train_seconds": time.time() - t0,
                      "test": _evaluate_splits(model, test),
                      "generalisation": {str(v): evaluate(model, g) for v, g in gen.items()}}
            runs.setdefault(model_name, []).append(record)
            print(f"[{name}] seed {seed} {model_name}: IOU-M {record['test']['overall']['iou_m']:.3f} "
                  f"({record['train_seconds']:.0f}s)", flush=True)
    results = {"config": asdict(cfg), "runs": runs, "total_seconds": time.time() - start}
    (outdir / f"{name}.json").write_text(json.dumps(results, indent=1))
    return results


def _plot(results, outdir, filename, xlabel, train_range):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, runs in results["runs"].items():
        xs = sorted(int(v) for v in runs[0]["generalisation"])
        series = np.array([[r["generalisation"][str(x)]["iou_m"] for x in xs] for r in runs])
        mean, std = series.mean(0), (series.std(0, ddof=1) if len(series) > 1 else np.zeros(len(xs)))
        ax.plot(xs, mean, marker="o", label=LABELS[model_name])
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.axvspan(*train_range, color="gray", alpha=0.1, label="training range")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IOU-M (moved objects)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close(fig)


def _table(results, rows, header, cell):
    lines = ["| Model | " + " | ".join(header) + " |", "|---|" + "---|" * len(header)]
    for model_name, runs in results["runs"].items():
        lines.append(f"| {LABELS[model_name]} | "
                     + " | ".join(cell(runs, r) for r in rows) + " |")
    return "\n".join(lines)


def _table_ii_markdown(results):
    """Rows: models. Columns follow the paper's Table II (IOU, IOU-M overall/simple/complex)
    plus subject and placement accuracy."""
    columns = [("IOU", "overall", "iou"), ("IOU-M", "overall", "iou_m"),
               ("IOU-M simple", "simple", "iou_m"), ("IOU-M complex", "complex", "iou_m"),
               ("Subject acc. simple", "simple", "subject_acc"),
               ("Subject acc. complex", "complex", "subject_acc"),
               ("Placement acc.", "overall", "placement_acc")]
    return _table(results, columns, [c[0] for c in columns],
                  lambda runs, c: fmt([r["test"][c[1]][c[2]] for r in runs]))


def _generalisation_markdown(results, axis_name):
    values = sorted(int(v) for v in next(iter(results["runs"].values()))[0]["generalisation"])
    return _table(results, values, [f"{axis_name}={v}" for v in values],
                  lambda runs, v: fmt([r["generalisation"][str(v)]["iou_m"] for r in runs]))


def run_phase1(cfg, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = _run_all(cfg, outdir, dict(n_objects=(3, 5)), dict(n_objects=(3, 5)), range(3, 11),
                       lambda n: dict(n_objects=(n, n)), "phase1", ablation=True)
    _plot(results, outdir, "fig4a_objects.png", "number of objects in the scene", (3, 5))
    (outdir / "results_phase1.md").write_text(
        "### Table II analogue (mean ± sample std over seeds)\n\n" + _table_ii_markdown(results)
        + "\n\n### IOU-M vs number of objects (Fig. 4a analogue)\n\n"
        + _generalisation_markdown(results, "objects") + "\n")
    return results


def run_phase2(cfg, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scenes = dict(n_objects=(3, 5))
    results = _run_all(cfg, outdir, dict(scenes, n_steps=(1, 2)), dict(scenes, n_steps=(1, 2)), range(1, 6),
                       lambda k: dict(scenes, n_steps=(k, k)), "phase2", ablation=False)
    _plot(results, outdir, "fig4b_steps.png", "number of steps in the instruction", (1, 2))
    columns = [("IOU", "iou"), ("IOU-M", "iou_m"), ("Subject acc.", "subject_acc"),
               ("Placement acc.", "placement_acc")]
    overall = _table(results, columns, [c[0] for c in columns],
                     lambda runs, c: fmt([r["test"]["overall"][c[1]] for r in runs]))
    (outdir / "results_phase2.md").write_text(
        "### Multi-step test set, 1-2 steps (mean ± sample std over seeds)\n\n" + overall
        + "\n\n### IOU-M vs number of steps (Fig. 4b analogue)\n\n"
        + _generalisation_markdown(results, "steps") + "\n")
    return results


NOISE_LEVELS = (0.1, 0.3, 0.5)


def run_noise(cfg, outdir, levels=NOISE_LEVELS):
    """Feature-noise sweep on phase-1 data (train and test at the same sigma): NSModel vs the
    parse-matched baseline. A cheap stand-in for the paper's gold vs detected box gap."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs, start = {}, time.time()
    for sigma in levels:
        for seed in cfg.seeds:
            train_data = collate(make_dataset(cfg.n_train, seed=seed, feature_noise=sigma))
            test = collate(make_dataset(cfg.n_test, seed=1000 + seed, feature_noise=sigma))
            for model_name, factory in (("ours", NSModel), ("parsed_baseline", ParsedNeuralBaseline)):
                torch.manual_seed(seed)
                model = factory()
                train(model, train_data, cfg.epochs, seed=seed)
                record = {"seed": seed, "test": _evaluate_splits(model, test)}
                runs.setdefault(model_name, {}).setdefault(str(sigma), []).append(record)
                print(f"[noise] sigma {sigma} seed {seed} {model_name}: "
                      f"IOU-M {record['test']['overall']['iou_m']:.3f}", flush=True)
    results = {"config": asdict(cfg), "runs": runs, "total_seconds": time.time() - start}
    (outdir / "noise.json").write_text(json.dumps(results, indent=1))
    columns = [(str(sigma), metric) for metric in ("iou_m", "subject_acc") for sigma in levels]
    header = [f"{'IOU-M' if m == 'iou_m' else 'Subject acc.'} σ={sig}" for sig, m in columns]
    table = _table(results, columns, header,
                   lambda by_sigma, c: fmt([r["test"]["overall"][c[1]] for r in by_sigma[c[0]]]))
    (outdir / "results_noise.md").write_text(
        "### Feature-noise sweep, phase-1 test set (mean ± sample std over seeds)\n\n" + table + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all", "noise"], default="all")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--n-train", type=int, default=Config.n_train)
    parser.add_argument("--seeds", type=int, default=len(Config.seeds))
    args = parser.parse_args()
    cfg = Config(n_train=args.n_train, epochs=args.epochs, seeds=tuple(range(args.seeds)))
    if args.phase in ("1", "all"):
        run_phase1(cfg, args.out)
    if args.phase in ("2", "all"):
        run_phase2(cfg, args.out)
    if args.phase == "noise":
        run_noise(cfg, args.out)


if __name__ == "__main__":
    main()
