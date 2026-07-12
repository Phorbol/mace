from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import (
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
    assert row["final_update"] == 1250
    assert row["final_mae_e_mev_atom"] == pytest.approx(19.0)
    assert row["final_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["best_mae_e_mev_atom"] == pytest.approx(18.0)
    assert row["best_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["mean_seconds_per_epoch"] == pytest.approx(15.0)
    assert row["seconds_per_update"] == pytest.approx(15.0 / 625)
    assert row["updates_per_second"] == pytest.approx(625 / 15.0)


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
            "hybrid_muon_lr_factor": 0.1,
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
    assert comparison["final_mae_e_delta_mev_atom"] == -3.0
    assert comparison["final_mae_f_delta_mev_a"] == -25.0
    assert comparison["best_mae_e_delta_mev_atom"] == -2.0
    assert comparison["best_mae_f_delta_mev_a"] == -19.0
    assert comparison["seconds_per_update_ratio"] == pytest.approx(0.064 / 0.060)
    assert comparison["updates_per_second_ratio"] == pytest.approx(15.625 / 16.6666667)
