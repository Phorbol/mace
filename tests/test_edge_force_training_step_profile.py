from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "profile_edge_force_training_steps.py"
    )
    spec = importlib.util.spec_from_file_location("edge_force_training_steps", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_csv_choices_accepts_known_values():
    profile = load_profile()

    assert profile.parse_csv_choices("adam,hybrid_muon", {"adam", "hybrid_muon"}) == [
        "adam",
        "hybrid_muon",
    ]


def test_parse_csv_choices_rejects_unknown_value():
    profile = load_profile()

    with pytest.raises(ValueError, match="unknown value"):
        profile.parse_csv_choices("adam,sgd", {"adam", "hybrid_muon"})


def test_build_parser_accepts_edge_compile_and_cueq_minus_linear_flags():
    profile = load_profile()

    args = profile.build_parser().parse_args([
        "--modes",
        "position_eager,edge_compile",
        "--optimizers",
        "adam,hybrid_muon",
        "--no-cueq-optimize-all",
        "--cueq-optimize-channelwise",
        "--cueq-optimize-fctp",
        "--cueq-optimize-symmetric",
        "--edge-compile-mode",
        "reduce-overhead",
        "--no-edge-compile-dynamic",
    ])

    assert args.modes == "position_eager,edge_compile"
    assert args.optimizers == "adam,hybrid_muon"
    assert args.cueq_optimize_all is False
    assert args.cueq_optimize_channelwise is True
    assert args.cueq_optimize_fctp is True
    assert args.cueq_optimize_symmetric is True
    assert args.edge_compile_mode == "reduce-overhead"
    assert args.edge_compile_dynamic is False


def load_parse_edge_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "parse_edge_force_step_profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "parse_edge_force_step_profile", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_edge_force_step_profile_summarizes_gates_and_speedups(tmp_path):
    parser = load_parse_edge_profile()
    payload = {
        "results": [
            {
                "optimizer": "adam",
                "mode": "position_eager",
                "setup_ms": 0.0,
                "gate_result": None,
                "timings": {"total_ms": {"mean_ms": 20.0, "median_ms": 19.0}},
            },
            {
                "optimizer": "adam",
                "mode": "edge_compile",
                "setup_ms": 1000.0,
                "gate_result": {"accepted": True, "fallback_reason": None},
                "timings": {"total_ms": {"mean_ms": 10.0, "median_ms": 9.5}},
            },
            {
                "optimizer": "hybrid_muon",
                "mode": "position_eager",
                "setup_ms": 0.0,
                "gate_result": None,
                "timings": {"total_ms": {"mean_ms": 25.0, "median_ms": 24.0}},
            },
        ]
    }
    path = tmp_path / "profile.json"
    path.write_text(__import__("json").dumps(payload))

    summary = parser.summarize_file(path)

    adam_compile = summary["rows"]["adam/edge_compile"]
    assert adam_compile["gate_accepted"] is True
    assert adam_compile["setup_ms"] == 1000.0
    assert adam_compile["mean_total_ms"] == 10.0
    assert adam_compile["speedup_vs_position_eager"] == 2.0
    assert summary["all_required_gates_accepted"] is True


def load_epoch_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "profile_edge_force_training_epochs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "edge_force_training_epochs", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_epoch_profile_splits_indices_into_limited_batches():
    profile = load_epoch_profile()

    batches = profile.split_index_batches([0, 1, 2, 3, 4], batch_size=2, max_batches=2)

    assert batches == [(0, 1), (2, 3)]


def test_epoch_profile_cache_key_is_batch_identity_not_shape():
    profile = load_epoch_profile()

    left = profile.compile_cache_key((0, 1, 2), num_atoms=18, num_edges=256)
    right = profile.compile_cache_key((3, 4, 5), num_atoms=18, num_edges=256)

    assert left != right


def test_epoch_profile_summary_reports_setup_and_steady_step_times():
    profile = load_epoch_profile()
    steps = [
        {"total_ms": 20.0, "setup_ms": 100.0, "cache_hit": False},
        {"total_ms": 10.0, "setup_ms": 0.0, "cache_hit": True},
    ]

    summary = profile.summarize_steps(steps)

    assert summary["steps"] == 2
    assert summary["compile_setups"] == 1
    assert summary["cache_hits"] == 1
    assert summary["mean_total_ms_excluding_setup"] == 15.0
    assert summary["mean_total_ms_including_setup"] == 65.0
