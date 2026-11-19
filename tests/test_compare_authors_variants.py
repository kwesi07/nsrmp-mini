import json

from compare_authors_variants import run
from run_experiments import Config

TINY = Config(n_train=64, n_test=40, epochs=3, seeds=(0, 1))


def test_comparison_trains_four_models_and_reports_soft_and_hard_metrics_per_split(tmp_path):
    results = run(TINY, tmp_path)
    saved = json.loads((tmp_path / "authors_variants.json").read_text())
    assert set(saved["runs"]) == {"ours", "parsed_baseline", "baseline", "authors_nmn"}
    assert saved["total_seconds"] > 0 and saved["config"]["epochs"] == 3
    for runs in saved["runs"].values():
        assert [r["seed"] for r in runs] == [0, 1]
        for mode in ("soft", "hard"):
            assert set(runs[0][mode]) == {"overall", "simple", "complex"}
            assert set(runs[0][mode]["overall"]) == {"iou", "iou_m", "subject_acc", "placement_acc"}
    assert results["runs"]["ours"][0]["soft"] == saved["runs"]["ours"][0]["soft"]
    md = (tmp_path / "authors_variants.md").read_text()
    assert "Soft" in md and "Hard" in md and "authors_nmn" not in md
    assert "NMN+ (authors' scoring)" in md and "± " in md


def test_soft_numbers_equal_the_phase1_evaluation_of_the_same_trained_models(tmp_path):
    from run_experiments import run_phase1
    ref = run_phase1(Config(n_train=64, n_test=40, n_gen=10, epochs=3, seeds=(0,)), tmp_path / "p1")
    mine = run(Config(n_train=64, n_test=40, epochs=3, seeds=(0,)), tmp_path / "av")
    for name in ("ours", "parsed_baseline", "baseline"):
        assert mine["runs"][name][0]["soft"] == ref["runs"][name][0]["test"]
