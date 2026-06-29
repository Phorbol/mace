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
