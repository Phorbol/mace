from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_probe():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "probe_force_backward_compile_ops.py"
    )
    spec = importlib.util.spec_from_file_location("force_backward_compile_ops", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_cases_accepts_defaults_and_rejects_unknown():
    probe = load_probe()

    assert {case.name for case in probe.select_cases("all")} == {
        "mm_4632x1280_1280x64",
        "bmm_1x286x128_128x128",
        "bmm_1x858x128_128x128",
    }

    try:
        probe.select_cases("missing")
    except ValueError as exc:
        assert "unknown synthetic force-backward case" in str(exc)
    else:
        raise AssertionError("unknown case should fail")


def test_force_loss_snapshot_contains_second_order_gradients():
    probe = load_probe()
    case = probe.SyntheticCase(
        name="tiny_mm",
        op="mm",
        input_shape=(4, 3),
        weight_shape=(3, 2),
    )

    snapshot = probe.force_loss_snapshot(case, device=torch.device("cpu"), dtype=torch.float32, seed=7)

    assert snapshot["loss"] > 0.0
    assert snapshot["input_grad_norm"] > 0.0
    assert snapshot["weight_grad_norm"] > 0.0


def test_compare_snapshots_reports_matching_loss_and_gradients():
    probe = load_probe()
    case = probe.SyntheticCase(
        name="tiny_bmm",
        op="bmm",
        input_shape=(1, 5, 3),
        weight_shape=(1, 3, 4),
    )

    left = probe.force_loss_snapshot(case, device=torch.device("cpu"), dtype=torch.float32, seed=11)
    right = probe.force_loss_snapshot(case, device=torch.device("cpu"), dtype=torch.float32, seed=11)
    comparison = probe.compare_snapshots(left, right, atol=1.0e-7, rtol=1.0e-6)

    assert comparison["ok"] is True
    assert comparison["loss_abs_diff"] == 0.0
    assert comparison["input_grad_max_abs_diff"] == 0.0
    assert comparison["weight_grad_max_abs_diff"] == 0.0



def test_disable_donated_buffer_parser_flag():
    probe = load_probe()

    args = probe.build_parser().parse_args(["--disable-donated-buffer"])

    assert args.disable_donated_buffer is True
