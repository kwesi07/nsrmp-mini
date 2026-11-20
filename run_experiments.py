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
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from experiment_eval import evaluate as evaluate_checkpoint  # noqa: E402
from experiment_eval import save_checkpoint  # noqa: E402
from nsrmp.baseline import NeuralBaseline, ParsedNeuralBaseline  # noqa: E402
from nsrmp.batch import collate  # noqa: E402
from nsrmp.combined_metrics import build_annotations, evaluate_learned  # noqa: E402
from nsrmp.combined_train import fit_programs  # noqa: E402
from nsrmp.data import FEATURE_NOISE, make_dataset, training_examples  # noqa: E402
from nsrmp.metrics import evaluate  # noqa: E402
from nsrmp.model import NSModel, build  # noqa: E402
from nsrmp.parser import parse  # noqa: E402
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


def _evaluate_splits(model, test):
    return {"overall": evaluate(model, test),
            "simple": evaluate(model, _subset(test, ~test.is_complex)),
            "complex": evaluate(model, _subset(test, test.is_complex))}


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


def _plot_mini(results, outdir, filename, xlabel, train_range):
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
    _plot_mini(results, outdir, "fig4a_objects.png", "number of objects in the scene", (3, 5))
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
    _plot_mini(results, outdir, "fig4b_steps.png", "number of steps in the instruction", (1, 2))
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


def _phase_schedule(config):
    if "schedule" in config:
        phases = {"A": ("warm", "mixed"), "B": ("warm", "sim", "relation")}[config["schedule"]]
        return [(phase, config[f"{phase}_epochs"]) for phase in phases]
    epochs = config["epochs"]
    warm = max(1, epochs // 3) if config["curriculum"] else 0
    return [(phase, count) for phase, count in (("warm", warm), ("mixed", epochs - warm))
            if count > 0]


_MODES = ("surface_context", "keywords_only")
_TRAIN_OPTIONS = {"curriculum": True, "lr": 3e-3, "batch_size": 64, "beta": 1.0,
                  "entropy_coef": 0.01, "ema_decay": 0.9}
_METRICS = ("iou", "iou_m", "subject_acc", "placement_acc", "program_acc", "step_program_acc")


def _environment():
    root = Path(__file__).resolve().parent

    def read(*command):
        try:
            return subprocess.check_output(command, cwd=root, stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    if platform.system() == "Darwin":
        memory = read("sysctl", "-n", "hw.memsize")
        memory = int(memory) if memory else None
        cpu = read("sysctl", "-n", "machdep.cpu.brand_string")
    else:
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            memory = None
        cpu = platform.processor()
    status = read("git", "status", "--porcelain", "--untracked-files=all")
    return {"git_commit": read("git", "rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(), "cpu": cpu or platform.machine(),
            "memory_bytes": memory, "logical_cpus": os.cpu_count(), "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def _splits(config, phase, seed):
    normal = {"n_objects": (3, 5)}
    if phase == "2":
        normal["n_steps"] = (1, 2)
    splits = {
        "train": (config["n_train"], seed, normal),
        "validation": (config["n_test"], 5000 + seed, normal),
        "test": (config["n_test"], 1000 + seed, normal),
    }
    for value in (range(3, 11) if phase == "1" else range(1, 6)):
        params = ({"n_objects": (value, value)} if phase == "1"
                  else {"n_objects": (3, 5), "n_steps": (value, value)})
        splits[str(value)] = (config["n_gen"], 2000 + 100 * seed + value, params)
    return splits


def _describe_splits(splits):
    return {name: {"count": size, "seed": seed, **{"n_steps": (1, 1), **settings},
                   "p_complex": 0.5, "feature_noise": FEATURE_NOISE}
            for name, (size, seed, settings) in splits.items()}


def _evaluate(model, examples, config, context, seed):
    data = collate(examples, mode="learned", surface_context=context)
    result = evaluate_learned(model, data, build_annotations(examples, data),
                              sampled_count=config["sampled_count"],
                              uniform_count=config["uniform_count"], eval_seed=seed)
    json.dumps(result, allow_nan=False)
    return result


def _failure(record, error):
    record.update(status="failed", error=f"{type(error).__name__}: {error}",
                  traceback=traceback.format_exc())
    print(f"[learned phase {record['phase']}] {record['mode']} seed {record['seed']}: "
          f"{record['error']}", flush=True)


def _run_seed(config, phase, seed, runs):
    splits = _splits(config, phase, seed)
    records = []
    for mode, rows in runs.items():
        record = {"phase": phase, "mode": mode, "seed": seed, "status": "pending",
                  "model_seed": seed, "training_seed": seed, "splits": _describe_splits(splits),
                  "seconds": {"data_generation": None, "train": None, "eval": None,
                              "end_to_end": None}}
        rows.append(record)
        records.append(record)

    start = time.perf_counter()
    try:
        examples = {name: make_dataset(count, seed=split_seed, **params)
                    for name, (count, split_seed, params) in splits.items()}
    except Exception as error:
        elapsed = time.perf_counter() - start
        for record in records:
            record["seconds"].update(data_generation=elapsed, end_to_end=elapsed)
            _failure(record, error)
        return

    data_seconds = time.perf_counter() - start
    for record in records:
        seconds = record["seconds"]
        seconds["data_generation"] = data_seconds
        started = time.perf_counter()
        print(f"[learned phase {phase}] {record['mode']} seed {seed}: training", flush=True)
        try:
            context = record["mode"] == "surface_context"
            training = collate(examples["train"], mode="learned", surface_context=context)
            torch.default_generator.manual_seed(seed)
            model = build(parser_source="learned", backend="symbolic")
            options = config["training"]
            settings = {**options, "learning_rate": options["lr"], "epochs": config["epochs"],
                        "num_samples": config["num_samples"], "move_mode": "soft",
                        "loss": "original", "gradient_clip": 5.0}
            history = []

            def collect_history(phase, mode, rows, stats):
                history.extend(rows)

            tick = time.perf_counter()
            try:
                fit_programs(model, training, settings, phases=_phase_schedule(settings),
                             seed=seed, on_phase=collect_history)
            finally:
                seconds["train"] = time.perf_counter() - tick
            record["history"] = [{k: v if math.isfinite(v) else None for k, v in row.items()}
                                 for row in history]
            record["train_program_executions"] = sum(row["n_samples"] for row in history)
            if record["history"] != history:
                raise ValueError("non-finite training history (stored as null)")

            tick = time.perf_counter()
            try:
                record["generalisation"] = {}
                for name, (_, split_seed, _) in splits.items():
                    if name == "train":
                        continue
                    eval_config = config if name == "test" else {**config, "uniform_count": 0}
                    summary = _evaluate(model, examples[name], eval_config, context, split_seed)
                    if name in ("validation", "test"):
                        record[name] = summary
                    else:
                        record["generalisation"][name] = summary
            finally:
                seconds["eval"] = time.perf_counter() - tick
            record["status"] = "ok"
            print(f"[learned phase {phase}] {record['mode']} seed {seed}: "
                  f"IoU-M {record['test']['overall']['iou_m']:.3f}, "
                  f"program accuracy {record['test']['overall']['program_acc']:.3f} "
                  f"({seconds['train']:.1f}s training)", flush=True)
        except Exception as error:
            _failure(record, error)
        finally:
            seconds["end_to_end"] = data_seconds + time.perf_counter() - started


def _stat(values):
    total = len(values)
    used = [value for value in values if value is not None]
    if not used:
        return f"N/A (0/{total})"
    std = f"{statistics.stdev(used):.3f}" if len(used) > 1 else "N/A"
    return f"{statistics.mean(used):.3f} \u00b1 {std} ({len(used)}/{total})"


def _markdown(result):
    lines = ["# Our results - mini 2D learned parser", "",
             "Greedy test metrics from the last epoch. Cells show mean ± sample std "
             "and contributing / configured seeds.", "",
             "| Mode | Split | " + " | ".join(_METRICS) + " |",
             "| --- | --- |" + " --- |" * len(_METRICS)]
    for mode, records in result["runs"].items():
        for group in ("overall", "simple", "complex"):
            cells = [_stat([r["test"][group][metric] if r["status"] == "ok" else None
                            for r in records]) for metric in _METRICS]
            lines.append(f"| {mode} | {group} | " + " | ".join(cells) + " |")
    lines += ["", "| Mode | Seed | Status | Train seconds | Eval seconds | Program executions |",
              "| --- | --- | --- | --- | --- | --- |"]
    for mode, records in result["runs"].items():
        for r in records:
            seconds = r["seconds"]
            times = [f"{seconds[key]:.3f}" if seconds[key] is not None else "N/A"
                     for key in ("train", "eval")]
            lines.append(f"| {mode} | {r['seed']} | {r['status']} | "
                         f"{' | '.join(times)} | {r.get('train_program_executions', 'N/A')} |")
    lines += ["", "Training execution counts include all K draws per visited instruction. "
              "See JSON for runtime boundaries, validation, generalisation, diagnostics and failures."]
    return "\n".join(lines) + "\n"


_LEGACY_LABELS = {
    "ours": "Rule-parser NS",
    "parsed_baseline": "Parse-matched neural",
    "baseline": "Raw-sentence GRU",
    "ours_no_curriculum": "Rule-parser NS (no curriculum)",
}


def _load_legacy(path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or not isinstance(result.get("config"), dict)
                or not isinstance(result.get("runs"), dict)):
            raise ValueError("expected config and runs objects")
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid legacy comparison JSON at {path}: {error}") from error
    return result


def _comparison_markdown(result, legacy, legacy_path):
    def scores(records, seeds, group, metric):
        indexed = {}
        for record in records:
            seed = record["seed"]
            if seed in indexed:
                raise ValueError(f"duplicate comparison seed {seed}")
            if seed not in seeds:
                raise ValueError(f"unexpected comparison seed {seed}")
            indexed[seed] = record

        values = []
        for seed in seeds:
            record = indexed.get(seed)
            if record is None or record.get("status", "ok") != "ok":
                values.append(None)
                continue
            try:
                value = record["test"][group][metric]
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing comparison metric {group}.{metric}") from error
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"invalid comparison metric {group}.{metric}: {value!r}")
            values.append(value)
        return _stat(values)

    config = result["config"]
    legacy_seeds = legacy["config"].get("seeds") if legacy is not None else config["seeds"]
    if not isinstance(legacy_seeds, list):
        raise ValueError("invalid legacy comparison config.seeds")
    missing = f"N/A (0/{len(legacy_seeds)})"
    lines = ["## Comparison with existing models", "",
             "Mini 2D task, final-epoch greedy test results: mean ± sample std "
             "(contributing/configured seeds). Program accuracy is whole-instruction "
             "AST exact-match for learned parsers.", "",
             "| Model | IoU-M overall | IoU-M simple | IoU-M complex | Program accuracy | Source |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, label in _LEGACY_LABELS.items():
        if name == "ours_no_curriculum" and (legacy is None or name not in legacy["runs"]):
            continue
        records = legacy["runs"].get(name) if legacy is not None else None
        if records is not None and not isinstance(records, list):
            raise ValueError(f"invalid legacy comparison records for {name}")
        values = ([scores(records, legacy_seeds, group, "iou_m")
                   for group in ("overall", "simple", "complex")]
                  if records is not None else [missing] * 3)
        if records is None:
            program = "N/A (no source)"
        else:
            program = ("By construction" if name in ("ours", "ours_no_curriculum")
                       else "N/A (not measured)")
        source = f"Legacy {legacy_path.name}" if legacy is not None else "Legacy missing"
        lines.append(f"| {label} | {' | '.join(values)} | {program} | {source} |")

    for mode, records in result["runs"].items():
        values = [scores(records, config["seeds"], group, "iou_m")
                  for group in ("overall", "simple", "complex")]
        program = scores(records, config["seeds"], "overall", "program_acc")
        lines.append(f"| Learned Parser ({mode}) | {' | '.join(values)} | {program} "
                     f"| Learned learned_phase{result['phase']}.json |")

    keys = ("n_train", "n_test", "n_gen", "epochs", "seeds")

    def describe(source):
        return ", ".join(f"{key}={source.get(key)}" for key in keys)

    lines += ["", f"Learned source: `learned_phase{result['phase']}.json`; "
              f"config: {describe(config)}, num_samples={config['num_samples']}."]
    if legacy is None:
        lines.append(f"Legacy source missing: `{legacy_path}`; legacy rows are unavailable.")
    else:
        old_config = legacy["config"]
        lines.append(f"Legacy source (read-only): `{legacy_path}`; config: {describe(old_config)}.")
        mismatches = [f"{key}: legacy={old_config.get(key)}, learned={config.get(key)}"
                      for key in keys if old_config.get(key) != config.get(key)]
        if mismatches:
            lines.append("**NOT DIRECTLY COMPARABLE:** " + "; ".join(mismatches) + ".")
        else:
            lines.append("Recorded dataset sizes, epochs and seeds match; this does not prove "
                         "identical data-generator revisions or compute-matched training.")
        lines.append("Legacy JSON has no recorded Git commit; values are existing results, not a rerun.")

    lines.append("Learned training samples K programs per instruction; legacy models do not "
                 "use the same sampling protocol or compute budget.")
    return "\n".join(lines) + "\n"


def _plot_learned_parser(result, path):
    fig = Figure(figsize=(10, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    loss, reward, generalisation = fig.subplots(1, 3)
    for mode, records in result["runs"].items():
        for r in records:
            history = r.get("history", [])
            label = f"{mode} seed {r['seed']}"
            for ax, metric in ((loss, "task_loss"), (reward, "reward")):
                if history:
                    ax.plot([e["epoch"] + 1 for e in history], [e[metric] for e in history],
                            marker=".", label=label)
            if r["status"] == "ok":
                by_axis = r["generalisation"]
                generalisation.plot([int(k) for k in by_axis],
                                    [v["overall"]["iou_m"] for v in by_axis.values()],
                                    marker=".", label=label)
    loss.set(xlabel="Epoch", ylabel="Scene loss")
    reward.set(xlabel="Epoch", ylabel="Reward")
    axis = "Objects" if result["phase"] == "1" else "Steps"
    generalisation.set(xlabel=axis, ylabel="IoU-M")
    if loss.lines:
        loss.legend(fontsize="xx-small")
    fig.savefig(path)


def run_learned_parser(cfg, outdir, *, phase="1", modes=("surface_context",), num_samples=4,
                       sampled_count=0, uniform_count=None):
    """Run one phase (or both), writing a JSON record and reports for each phase."""
    if phase not in ("1", "2", "all"):
        raise ValueError(f"unsupported learned phase: {phase!r}")
    uniform_count = num_samples if uniform_count is None else uniform_count
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes).difference(_MODES):
        raise ValueError(f"expected distinct learned modes from {_MODES}: {modes!r}")
    config = {key: getattr(cfg, key) for key in ("n_train", "n_test", "n_gen", "epochs")}
    seeds = list(cfg.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("expected non-empty distinct non-negative seeds")
    if min(*config.values(), num_samples) < 1 or min(sampled_count, uniform_count) < 0:
        raise ValueError("expected positive experiment sizes and non-negative diagnostic counts")
    config.update(seeds=seeds, modes=list(modes), num_samples=num_samples,
                  sampled_count=sampled_count, uniform_count=uniform_count,
                  training=_TRAIN_OPTIONS, checkpoint="last_epoch",
                  validation_count=config["n_test"], eval_seed="split_seed")
    outdir = Path(outdir)
    if outdir.resolve() == Path(__file__).resolve().parent / "results":
        raise ValueError("use a learned output directory separate from legacy results")
    outdir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    phases = ("1", "2") if phase == "all" else (phase,)
    results = {}
    failures = []
    for current in phases:
        start = time.perf_counter()
        runs = {mode: [] for mode in modes}
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                _run_seed(config, current, seed, runs)
        result = {"config": config, "phase": current, "runs": runs, "environment": environment,
                  "total_seconds": time.perf_counter() - start,
                  "timing_scope": "Includes data generation, training and evaluation; excludes "
                                  "environment setup and report rendering. Paired modes share generation."}
        stem = outdir / f"learned_phase{current}"
        raw = json.dumps(result, indent=2, allow_nan=False)
        stem.with_suffix(".json").write_text(raw + "\n", encoding="utf-8")
        result = json.loads(raw)
        legacy_path = outdir.parent / f"phase{current}.json"
        report = _markdown(result) + "\n" + _comparison_markdown(
            result, _load_legacy(legacy_path), legacy_path)
        stem.with_suffix(".md").write_text(report, encoding="utf-8")
        _plot_learned_parser(result, stem.with_suffix(".png"))
        results[current] = result
        failures += [f"phase {current}, {mode}, seed {r['seed']}" for mode, rows in runs.items()
                     for r in rows if r["status"] != "ok"]
    if failures:
        raise RuntimeError(f"learned runs failed ({'; '.join(failures)}); see records in {outdir}")
    return results if phase == "all" else results[phase]


def _run_curriculum_seed(group, seed, config):
    spec = config["groups"][str(group)]
    directory = Path(config["output"]) / f"group-{group}" / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    examples = training_examples(config, seed)
    batch = collate(examples, surface_context=True, mode="learned")
    model = build(parser_source=spec["parser_source"], backend=spec["backend"])
    rules = [parse(e.text) for e in examples] if model.parser is None else None
    settings = {key: value for key, value in config.items() if key != "groups"}
    settings.update(spec)
    history = {}

    def save_phase(phase, mode, rows, stats):
        history[phase] = stats
        path = directory / f"{phase}.pt"
        save_checkpoint(path, model, group=group, seed=seed, phase=phase, spec=spec,
                        move_mode=mode, config=settings, history=history)
        print(f"group {group} seed {seed} {phase} loss {stats['scene_loss'][-1]:.5f}", flush=True)

    fit_programs(model, batch, settings, phases=_phase_schedule(settings),
                 seed=seed, rules=rules, on_phase=save_phase)
    checkpoint = directory / f"{next(reversed(history))}.pt"
    metrics = {f"{steps}-step": evaluate_checkpoint(checkpoint, steps=steps)
               for steps in (1, 2)}
    destination = checkpoint.parent / "metrics.json"
    result = {"group": group, "seed": seed, "checkpoint": checkpoint.name, **metrics}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def _matrix_summary(results):
    groups = {}
    for result in results:
        groups.setdefault(result["group"], {})[result["seed"]] = result

    def summarize(values):
        return {"mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}

    steps = ("1-step", "2-step")
    means = {g: {k: summarize([r[k]["iou_m"] for r in runs.values()]) for k in steps}
             for g, runs in groups.items()}
    contrasts = {}
    for a, b in ((1, 2), (3, 4), (1, 3), (2, 4), (2, 5), (5, 7), (6, 8),
                 (5, 9), (6, 10), (5, 6), (7, 8), (9, 10)):
        if a in groups and b in groups:
            diffs = {k: {s: groups[b][s][k]["iou_m"] - groups[a][s][k]["iou_m"] for s in sorted(groups[a])}
                     for k in steps}
            contrasts[f"{b} minus {a}"] = {k: {**summarize(list(d.values())), "by_seed": d}
                                          for k, d in diffs.items()}
    return {"groups": means, "paired_differences": contrasts, "runs": results}


def _run_curriculum(args):
    config = json.loads(args.experiment_config.read_text())
    config["output"] = str(Path(args.out) / "curriculum")
    jobs = [(group, seed) for seed in config["seeds"] for group in args.experiment]
    directory = Path(config["output"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    with ProcessPoolExecutor(max_workers=min(len(jobs), config["workers"])) as pool:
        futures = [pool.submit(_run_curriculum_seed, group, seed, config) for group, seed in jobs]
        results = [future.result() for future in futures]
    (directory / "summary.json").write_text(json.dumps(_matrix_summary(results), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all", "noise"], default="all")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--n-train", type=int, default=Config.n_train)
    parser.add_argument("--seeds", type=int, default=len(Config.seeds))
    parser.add_argument("--learned-parser", choices=("surface_context", "keywords_only", "both"))
    entry = parser.add_mutually_exclusive_group()
    entry.add_argument("--experiment", nargs="+", choices=range(1, 11), type=int)
    entry.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-config", type=Path, default=Path(__file__).with_name("experiments.json"))
    parser.add_argument("--parser-source", choices=("learned", "rule"))
    parser.add_argument("--move-mode", choices=("soft", "hard"))
    parser.add_argument("--steps", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.checkpoint:
        metrics = evaluate_checkpoint(args.checkpoint, steps=args.steps, parser_source=args.parser_source,
                                      move_mode=args.move_mode)
        print(json.dumps(metrics, indent=2, allow_nan=False))
        return
    if args.experiment:
        _run_curriculum(args)
        return
    cfg = Config(n_train=args.n_train, epochs=args.epochs, seeds=tuple(range(args.seeds)))
    if args.learned_parser is not None:
        if args.phase == "noise":
            parser.error("--learned-parser does not support --phase noise")
        modes = (("surface_context", "keywords_only") if args.learned_parser == "both"
                 else (args.learned_parser,))
        run_learned_parser(cfg, Path(args.out) / "learned-parser", phase=args.phase, modes=modes)
        return
    if args.phase in ("1", "all"):
        run_phase1(cfg, args.out)
    if args.phase in ("2", "all"):
        run_phase2(cfg, args.out)
    if args.phase == "noise":
        run_noise(cfg, args.out)


if __name__ == "__main__":
    main()
