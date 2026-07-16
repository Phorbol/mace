from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import (
    summarize_ablation_table,
    summarize_case,
    summarize_pairwise_comparisons,
)


def _write_case(tmp_path: Path, case: str, log_text: str, nvdmon_text: str = ""):
    root = tmp_path / "matrix"
    case_dir = root / case
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / f"{case}.log").write_text(log_text)
    if nvdmon_text:
        (case_dir / f"nvdmon_job-1_{case}.log").write_text(nvdmon_text)
    return root, case_dir


def test_summarize_case_reports_update_runtime_and_manifest_metadata(tmp_path):
    manifest = {
        "target_steps": 20000,
        "max_num_updates": 20000,
        "train_size": 5000,
        "batch_size": 8,
        "scheduler": "WSD",
        "lr_scheduler_interval": "step",
        "lr_wsd_warmup_ratio": 0.03,
        "lr_wsd_stop_lr_ratio": 0.001,
        "lr_wsd_decay_phase_ratio": 0.1,
        "lr_wsd_decay_type": "inverse_linear",
        "lr": 0.001,
        "weight_decay": 0.001,
        "hybrid_muon_mode": "2d",
        "hybrid_muon_routing": "mace",
        "hybrid_muon_tace_module_lr_scale": 0.25,
        "hybrid_muon_lr_factor": 0.1,
        "hybrid_muon_weight_decay": 0.0,
        "hybrid_muon_adam_variant": "adamw",
        "hybrid_muon_lr_scale_mode": "original",
        "stage_two_start_update": 15000,
    }
    log_text = """
2026-07-12 00:00:00.000 INFO Epoch 0: head: Default, loss=1.0, MAE_E_per_atom=20.00 meV, MAE_F=140.00 meV / A
2026-07-12 00:00:10.000 INFO Epoch 1: head: Default, loss=0.8, MAE_E_per_atom=18.00 meV, MAE_F=120.00 meV / A
2026-07-12 00:00:30.000 INFO Epoch 2: update=1250, head: Default, loss=0.7, MAE_E_per_atom=19.00 meV, MAE_F=110.00 meV / A
"""
    root, case_dir = _write_case(tmp_path, "cueq_hybrid_muon", log_text)

    row = summarize_case(root, case_dir, manifest)

    assert row["effective_updates"] == 20000
    assert row["steps_per_epoch"] == 625
    assert row["scheduler"] == "WSD"
    assert row["lr_scheduler_interval"] == "step"
    assert row["lr"] == 0.001
    assert row["weight_decay"] == 0.001
    assert row["hybrid_muon_adam_variant"] == "adamw"
    assert row["hybrid_muon_tace_module_lr_scale"] == 0.25
    assert row["final_update"] == 1250
    assert row["final_mae_e_mev_atom"] == pytest.approx(19.0)
    assert row["final_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["best_mae_e_mev_atom"] == pytest.approx(18.0)
    assert row["best_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["mean_seconds_per_epoch"] == pytest.approx(15.0)
    assert row["seconds_per_update"] == pytest.approx(15.0 / 625)
    assert row["updates_per_second"] == pytest.approx(625 / 15.0)


def test_summarize_case_overrides_hybrid_muon_routing_from_case_name(tmp_path):
    manifest = {
        "hybrid_muon_routing": "mace",
        "hybrid_muon_mode": "2d",
    }
    root, case_dir = _write_case(
        tmp_path,
        "hybrid_muon_tace",
        "2026-07-12 10:20:00.000 INFO: Epoch 0: head: Default, loss=0.12, "
        "MAE_E_per_atom=  150.00 meV, MAE_F=   40.00 meV / A\n",
    )

    row = summarize_case(root, case_dir, manifest)

    assert row["hybrid_muon"] is True
    assert row["hybrid_muon_routing"] == "tace"


def test_summarize_case_overrides_hybrid_muon_module_routing_from_case_name(tmp_path):
    manifest = {
        "hybrid_muon_routing": "mace",
        "hybrid_muon_mode": "2d",
    }
    root, case_dir = _write_case(
        tmp_path,
        "cueq_hybrid_muon_module",
        "2026-07-12 10:20:00.000 INFO: Epoch 0: head: Default, loss=0.12, "
        "MAE_E_per_atom=  150.00 meV, MAE_F=   40.00 meV / A\n",
    )

    row = summarize_case(root, case_dir, manifest)

    assert row["cueq"] is True
    assert row["hybrid_muon_routing"] == "module"


def test_summarize_case_reports_early_update_speed_from_initial(tmp_path):
    manifest = {
        "target_steps": 20000,
        "max_num_updates": 20000,
        "train_size": 5000,
        "batch_size": 8,
    }
    log_text = """
2026-07-12 00:00:00.000 INFO: Initial: update=0, head: Default, loss=1.0, MAE_E_per_atom=200.00 meV, MAE_F=50.00 meV / A
2026-07-12 00:06:40.000 INFO: Epoch 6: update=4000, head: Default, loss=0.8, MAE_E_per_atom=180.00 meV, MAE_F=45.00 meV / A
"""
    root, case_dir = _write_case(tmp_path, "hybrid_muon", log_text)

    row = summarize_case(root, case_dir, manifest)

    assert row["final_update"] == 4000
    assert row["mean_seconds_per_epoch"] is None
    assert row["seconds_per_update"] is None
    assert row["early_seconds_per_update"] == pytest.approx(0.1)
    assert row["early_updates_per_second"] == pytest.approx(10.0)


def test_summarize_case_marks_partial_run_from_train_metrics(tmp_path):
    manifest = {
        "target_steps": 20000,
        "max_num_updates": 20000,
        "train_size": 5000,
        "batch_size": 8,
    }
    log_text = """
2026-07-12 00:00:00.000 INFO: Initial: update=0, head: Default, loss=1.0, MAE_E_per_atom=200.00 meV, MAE_F=50.00 meV / A
2026-07-12 00:06:40.000 INFO: Epoch 16: update=10000, head: Default, loss=0.8, MAE_E_per_atom=180.00 meV, MAE_F=45.00 meV / A
"""
    root, case_dir = _write_case(tmp_path, "cueq_hybrid_muon", log_text)
    results_dir = case_dir / "results"
    results_dir.mkdir()
    metrics_path = results_dir / "demo_train.txt"
    metrics_path.write_text(
        "\n".join("{\"loss\": 1.0, \"mode\": \"opt\"}" for _ in range(12345)) + "\n"
    )

    row = summarize_case(root, case_dir, manifest)

    assert row["last_eval_update"] == 10000
    assert row["current_train_updates"] == 12345
    assert row["is_complete"] is False
    assert row["run_status"] == "partial"

def test_summarize_case_reports_train_metrics_timing(tmp_path):
    manifest = {
        "target_steps": 20000,
        "max_num_updates": 20000,
        "train_size": 5000,
        "batch_size": 8,
    }
    log_text = (
        "2026-07-12 00:00:00.000 INFO: Initial: update=0, head: Default, loss=1.0, MAE_E_per_atom=200.00 meV, MAE_F=50.00 meV / A\n"
        "2026-07-12 00:06:40.000 INFO: Epoch 16: update=10000, head: Default, loss=0.8, MAE_E_per_atom=180.00 meV, MAE_F=45.00 meV / A\n"
    )
    root, case_dir = _write_case(tmp_path, "cueq_hybrid_muon", log_text)
    results_dir = case_dir / "results"
    results_dir.mkdir()
    metrics_path = results_dir / "demo_train.txt"
    metrics_path.write_text(
        "{\"time\": 0.05, \"train_optimizer_step_seconds\": 0.002, \"mode\": \"opt\"}\n"
        "{\"time\": 0.07, \"train_optimizer_step_seconds\": 0.004, \"mode\": \"opt\"}\n"
    )

    row = summarize_case(root, case_dir, manifest)

    assert row["train_metrics_updates"] == 2
    assert row["train_metrics_seconds_per_update"] == pytest.approx(0.06)
    assert row["train_metrics_updates_per_second"] == pytest.approx(1.0 / 0.06)
    assert row["train_metrics_optimizer_step_seconds"] == pytest.approx(0.003)

def test_summarize_case_reports_next_eval_and_completion_eta(tmp_path):
    manifest = {
        "target_steps": 200000,
        "max_num_updates": 200000,
        "train_size": 5000,
        "batch_size": 8,
        "eval_interval_updates": 20000,
    }
    log_text = (
        "2026-07-12 00:00:00.000 INFO: Epoch 95: update=60000, head: Default, loss=0.8, MAE_E_per_atom=100.00 meV, MAE_F=33.00 meV / A\n"
    )
    root, case_dir = _write_case(tmp_path, "cueq_adamw", log_text)
    results_dir = case_dir / "results"
    results_dir.mkdir()
    metrics_path = results_dir / "demo_train.txt"
    metrics_path.write_text("\n".join("{\"time\": 0.05, \"mode\": \"opt\"}" for _ in range(85000)) + "\n")

    row = summarize_case(root, case_dir, manifest)

    assert row["next_eval_update"] == 100000
    assert row["updates_to_next_eval"] == 15000
    assert row["seconds_to_next_eval_estimate"] == pytest.approx(750.0)
    assert row["updates_to_target"] == 115000
    assert row["seconds_to_target_estimate"] == pytest.approx(5750.0)

def test_eval_history_intervals_report_per_case_improvement():
    from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import summarize_eval_history_intervals

    rows = [
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "eval_history": [
                {"update": 20000, "mae_e_mev_atom": 126.0, "mae_f_mev_a": 41.0},
                {"update": 40000, "mae_e_mev_atom": 89.0, "mae_f_mev_a": 37.0},
                {"update": 60000, "mae_e_mev_atom": 67.0, "mae_f_mev_a": 35.5},
            ],
        },
    ]

    intervals = summarize_eval_history_intervals(rows)

    assert [row["to_update"] for row in intervals] == [40000, 60000]
    assert intervals[0]["mae_e_improvement_mev_atom"] == pytest.approx(37.0)
    assert intervals[0]["mae_f_improvement_mev_a"] == pytest.approx(4.0)
    assert intervals[1]["mae_e_delta_mev_atom"] == pytest.approx(-22.0)
    assert intervals[1]["mae_f_delta_mev_a"] == pytest.approx(-1.5)

def test_eval_history_comparison_reports_same_update_deltas():
    from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import summarize_eval_history_comparisons

    rows = [
        {
            "root": "run",
            "case": "cueq_adamw",
            "eval_history": [
                {"update": 20000, "mae_e_mev_atom": 160.0, "mae_f_mev_a": 39.0},
                {"update": 40000, "mae_e_mev_atom": 130.0, "mae_f_mev_a": 35.0},
            ],
        },
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "eval_history": [
                {"update": 20000, "mae_e_mev_atom": 126.0, "mae_f_mev_a": 41.0},
                {"update": 40000, "mae_e_mev_atom": 89.0, "mae_f_mev_a": 37.0},
            ],
        },
    ]

    comparisons = summarize_eval_history_comparisons(rows)

    assert [row["update"] for row in comparisons] == [20000, 40000]
    assert comparisons[1]["baseline_case"] == "cueq_adamw"
    assert comparisons[1]["candidate_case"] == "cueq_hybrid_muon"
    assert comparisons[1]["mae_e_delta_mev_atom"] == pytest.approx(-41.0)
    assert comparisons[1]["mae_f_delta_mev_a"] == pytest.approx(2.0)

def test_eval_history_comparison_uses_cross_run_baseline():
    from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import summarize_eval_history_comparisons

    rows = [
        {
            "root": "adamw-root",
            "case": "cueq_adamw",
            "eval_history": [
                {"update": 40000, "mae_e_mev_atom": 130.0, "mae_f_mev_a": 35.0},
            ],
        },
        {
            "root": "muon-root",
            "case": "cueq_hybrid_muon",
            "eval_history": [
                {"update": 40000, "mae_e_mev_atom": 89.0, "mae_f_mev_a": 37.0},
            ],
        },
    ]

    [comparison] = summarize_eval_history_comparisons(rows)

    assert comparison["baseline_root"] == "adamw-root"
    assert comparison["candidate_root"] == "muon-root"
    assert comparison["update"] == 40000
    assert comparison["mae_e_delta_mev_atom"] == pytest.approx(-41.0)
    assert comparison["mae_f_delta_mev_a"] == pytest.approx(2.0)

def test_pairwise_comparison_reports_tace_routing_ablation():
    rows = [
        {
            "root": "run",
            "case": "adamw",
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 130.0,
            "final_update": 12000,
            "seconds_per_update": 0.060,
        },
        {
            "root": "run",
            "case": "hybrid_muon_tace",
            "hybrid_muon_routing": "tace",
            "final_mae_e_mev_atom": 18.0,
            "final_mae_f_mev_a": 110.0,
            "seconds_per_update": 0.066,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["baseline_case"] == "adamw"
    assert comparison["candidate_case"] == "hybrid_muon_tace"
    assert comparison["hybrid_muon_routing"] == "tace"
    assert comparison["final_mae_f_delta_mev_a"] == -20.0
    assert comparison["seconds_per_update_ratio"] == pytest.approx(1.1)


def test_pairwise_comparison_reports_cueq_adamw_speed_ablation():
    rows = [
        {
            "root": "run",
            "case": "adamw",
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 130.0,
            "final_update": 20000,
            "seconds_per_update": 0.080,
        },
        {
            "root": "run",
            "case": "cueq_adamw",
            "cueq": True,
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 130.0,
            "final_update": 12000,
            "seconds_per_update": 0.060,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["baseline_case"] == "adamw"
    assert comparison["candidate_case"] == "cueq_adamw"
    assert comparison["baseline_final_update"] == 20000
    assert comparison["candidate_final_update"] == 12000
    assert comparison["same_final_update"] is False
    assert comparison["speedup_vs_baseline"] == pytest.approx(4.0 / 3.0)
    assert comparison["final_mae_f_delta_mev_a"] == 0.0


def test_pairwise_comparison_prefers_train_metrics_speed_for_live_runs():
    rows = [
        {
            "root": "run",
            "case": "cueq_adamw",
            "seconds_per_update": 0.10,
            "train_metrics_seconds_per_update": 0.05,
        },
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "seconds_per_update": 0.10,
            "train_metrics_seconds_per_update": 0.06,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["speedup_vs_baseline"] == pytest.approx(0.05 / 0.06)

def test_pairwise_comparison_uses_train_metrics_speed_when_epoch_timing_missing():
    rows = [
        {
            "root": "run",
            "case": "cueq_adamw",
            "train_metrics_seconds_per_update": 0.05,
            "train_metrics_updates_per_second": 20.0,
            "train_metrics_optimizer_step_seconds": 0.001,
        },
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "train_metrics_seconds_per_update": 0.06,
            "train_metrics_updates_per_second": 16.6666667,
            "train_metrics_optimizer_step_seconds": 0.003,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["train_metrics_seconds_per_update_ratio"] == pytest.approx(1.2)
    assert comparison["train_metrics_optimizer_step_delta_seconds"] == pytest.approx(0.002)
    assert comparison["speedup_vs_baseline"] == pytest.approx(0.05 / 0.06)

def test_pairwise_comparison_reports_final_best_and_update_speed():
    rows = [
        {
            "root": "run",
            "case": "cueq_adamw",
            "effective_updates": 20000,
            "stage_two_start_update": 15000,
            "scheduler": "WSD",
            "lr_scheduler_interval": "step",
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 130.0,
            "best_mae_e_mev_atom": 18.0,
            "best_mae_f_mev_a": 120.0,
            "seconds_per_update": 0.060,
            "updates_per_second": 16.6666667,
            "max_fb_memory_mb": 10000,
        },
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "effective_updates": 20000,
            "stage_two_start_update": 15000,
            "scheduler": "WSD",
            "lr_scheduler_interval": "step",
            "hybrid_muon_mode": "2d",
            "hybrid_muon_routing": "mace",
            "hybrid_muon_tace_module_lr_scale": 0.25,
            "hybrid_muon_lr_factor": 0.1,
            "hybrid_muon_stage_two_lr_factor": 0.5,
            "final_mae_e_mev_atom": 17.0,
            "final_mae_f_mev_a": 105.0,
            "best_mae_e_mev_atom": 16.0,
            "best_mae_f_mev_a": 101.0,
            "seconds_per_update": 0.064,
            "updates_per_second": 15.625,
            "max_fb_memory_mb": 10150,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["baseline_case"] == "cueq_adamw"
    assert comparison["candidate_case"] == "cueq_hybrid_muon"
    assert comparison["hybrid_muon_stage_two_lr_factor"] == 0.5
    assert comparison["final_mae_e_delta_mev_atom"] == -3.0
    assert comparison["final_mae_f_delta_mev_a"] == -25.0
    assert comparison["best_mae_e_delta_mev_atom"] == -2.0
    assert comparison["best_mae_f_delta_mev_a"] == -19.0
    assert comparison["seconds_per_update_ratio"] == pytest.approx(0.064 / 0.060)
    assert comparison["updates_per_second_ratio"] == pytest.approx(15.625 / 16.6666667)


def test_ablation_table_compares_cross_run_hybrid_muon_to_cueq_adamw():
    rows = [
        {
            "root": "baseline",
            "case": "cueq_adamw",
            "final_update": 20000,
            "final_mae_e_mev_atom": 13.0,
            "final_mae_f_mev_a": 106.6,
            "best_mae_e_mev_atom": 13.0,
            "best_mae_f_mev_a": 40.6,
            "seconds_per_update": 0.053,
            "updates_per_second": 18.8,
            "max_fb_memory_mb": 10160,
        },
        {
            "root": "stage2_lr0",
            "case": "cueq_hybrid_muon",
            "final_update": 20000,
            "hybrid_muon_routing": "mace",
            "hybrid_muon_tace_module_lr_scale": 0.25,
            "hybrid_muon_lr_factor": 0.1,
            "hybrid_muon_stage_two_lr_factor": 0.0,
            "hybrid_muon_stage_two_route": "adamw",
            "hybrid_muon_lr_scale_mode": "original",
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 101.0,
            "best_mae_e_mev_atom": 20.0,
            "best_mae_f_mev_a": 46.0,
            "seconds_per_update": 0.056,
            "updates_per_second": 17.9,
            "max_fb_memory_mb": 10158,
        },
    ]

    table = summarize_ablation_table(rows)

    assert [row["label"] for row in table] == [
        "cueq_adamw",
        "cueq_hybrid_muon/routing=mace/tace_module_lr_scale=0.25/lr_factor=0.1/stage2_lr_factor=0.0/stage2_route=adamw/scale=original",
    ]
    baseline, candidate = table
    assert baseline["is_baseline"] is True
    assert candidate["is_baseline"] is False
    assert candidate["baseline_case"] == "cueq_adamw"
    assert candidate["hybrid_muon_tace_module_lr_scale"] == 0.25
    assert candidate["same_final_update"] is True
    assert candidate["final_mae_e_delta_mev_atom"] == pytest.approx(7.0)
    assert candidate["final_mae_f_delta_mev_a"] == pytest.approx(-5.6)
    assert candidate["speedup_vs_baseline"] == pytest.approx(0.053 / 0.056)
    assert candidate["max_fb_memory_delta_mb"] == -2

