from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


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

def load_phase_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "profile_training_step_phases.py"
    )
    spec = importlib.util.spec_from_file_location("training_step_phases", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TinyOptimizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)


def _assert_profile_build_optimizer_passes_tace_routing(profile, monkeypatch):
    model = TinyOptimizerModel()
    captured = {}

    def fake_build_hybrid_muon_param_groups(named_parameters, **kwargs):
        captured["names"] = [name for name, _param in named_parameters]
        captured.update(kwargs)
        return [{"params": []}], []

    class FakeHybridMuon:
        def __init__(self, groups, *, lr, weight_decay):
            self.groups = groups
            self.lr = lr
            self.weight_decay = weight_decay

    monkeypatch.setattr(
        profile, "build_hybrid_muon_param_groups", fake_build_hybrid_muon_param_groups
    )
    monkeypatch.setattr(profile, "HybridMuon", FakeHybridMuon)

    spec = profile._build_optimizer(
        "hybrid_muon",
        model,
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_lr_factor=0.1,
        muon_mode="slice",
        muon_routing="tace",
    )

    assert spec.name == "hybrid_muon"
    assert captured["names"] == ["linear.weight"]
    assert captured["muon_mode"] == "slice"
    assert captured["routing"] == "tace"
    assert captured["module_map"]["linear"] is model.linear


def test_edge_step_profile_build_optimizer_passes_tace_routing(monkeypatch):
    _assert_profile_build_optimizer_passes_tace_routing(load_profile(), monkeypatch)


def test_phase_profile_build_optimizer_passes_tace_routing(monkeypatch):
    _assert_profile_build_optimizer_passes_tace_routing(load_phase_profile(), monkeypatch)


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

def test_summarize_fx_graph_reports_ops_targets_and_outputs():
    profile = load_profile()

    def fn(x, y):
        z = x + y
        return z, z.relu()

    graph_module = torch.fx.symbolic_trace(fn)

    stats = profile.summarize_fx_graph(graph_module)

    assert stats["node_count"] == len(list(graph_module.graph.nodes))
    assert stats["op_counts"]["placeholder"] == 2
    assert stats["op_counts"]["output"] == 1
    assert stats["target_counts"]["call_method:relu"] == 1
    assert stats["output_tensor_count"] == 2

def test_build_parser_accepts_edge_compile_and_cueq_minus_linear_flags():
    profile = load_profile()

    args = profile.build_parser().parse_args([
        "--modes",
        "position_eager,edge_compile,edge_compile_compiled_autograd,edge_compile_grads,edge_compile_grads_sequence",
        "--optimizers",
        "adam,hybrid_muon",
        "--no-cueq-optimize-all",
        "--cueq-optimize-channelwise",
        "--cueq-optimize-fctp",
        "--cueq-optimize-symmetric",
        "--edge-compile-mode",
        "reduce-overhead",
        "--no-edge-compile-dynamic",
        "--hybrid-muon-mode",
        "slice",
        "--hybrid-muon-routing",
        "tace",
        "--edge-compile-grad-filter",
        "readouts,conv_tp_weights",
        "--edge-compile-grad-filter-sequence",
        "readouts:conv_tp_weights",
    ])

    assert args.modes == (
        "position_eager,edge_compile,edge_compile_compiled_autograd,edge_compile_grads,edge_compile_grads_sequence"
    )
    assert args.optimizers == "adam,hybrid_muon"
    assert args.cueq_optimize_all is False
    assert args.cueq_optimize_channelwise is True
    assert args.cueq_optimize_fctp is True
    assert args.cueq_optimize_symmetric is True
    assert args.edge_compile_mode == "reduce-overhead"
    assert args.edge_compile_dynamic is False
    assert args.hybrid_muon_mode == "slice"
    assert args.hybrid_muon_routing == "tace"
    assert args.edge_compile_grad_filter == "readouts,conv_tp_weights"
    assert args.edge_compile_grad_filter_sequence == "readouts:conv_tp_weights"


def test_parse_edge_compile_grad_filter_sequence_rejects_empty_groups():
    profile = load_profile()

    assert profile.parse_edge_compile_grad_filter_sequence("readouts:conv_tp_weights") == [
        "readouts",
        "conv_tp_weights",
    ]

    with pytest.raises(ValueError, match="empty"):
        profile.parse_edge_compile_grad_filter_sequence("readouts::conv_tp_weights")


def test_compiled_autograd_context_uses_private_enable(monkeypatch):
    profile = load_profile()
    calls = []

    class FakeContext:
        def __enter__(self):
            calls.append(("enter",))

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type))

    class FakeCompiledAutograd:
        def _enable(self, compiler_fn, *, dynamic=True):
            calls.append(("enable", callable(compiler_fn), dynamic))
            return FakeContext()

    monkeypatch.setattr(
        profile,
        "_load_compiled_autograd_module",
        lambda: FakeCompiledAutograd(),
    )

    with profile._compiled_autograd_context(
        enabled=True,
        compile_mode="reduce-overhead",
        compile_dynamic=False,
    ):
        calls.append(("body",))

    assert calls == [
        ("enable", True, False),
        ("enter",),
        ("body",),
        ("exit", None),
    ]


def test_select_named_parameters_by_filter_matches_substrings():
    profile = load_profile()
    params = (
        ("node_embedding.linear.weight", torch.nn.Parameter(torch.ones(1))),
        ("interactions.0.conv_tp_weights.layer0.weight", torch.nn.Parameter(torch.ones(1))),
        ("readouts.1.linear_2.weight", torch.nn.Parameter(torch.ones(1))),
    )

    assert [name for name, _ in profile._select_named_parameters_by_filter(params, "")] == [
        "node_embedding.linear.weight",
        "interactions.0.conv_tp_weights.layer0.weight",
        "readouts.1.linear_2.weight",
    ]
    assert [
        name
        for name, _ in profile._select_named_parameters_by_filter(
            params, "readouts,conv_tp_weights"
        )
    ] == [
        "interactions.0.conv_tp_weights.layer0.weight",
        "readouts.1.linear_2.weight",
    ]


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
                "optimizer": "adam",
                "mode": "edge_compile_grads",
                "setup_ms": 1500.0,
                "gate_result": {
                    "accepted": True,
                    "fallback_reason": None,
                    "compiled_grad_count": 3,
                    "compiled_grad_filter": "readouts",
                    "graph_stats": {"node_count": 100, "output_tensor_count": 6},
                },
                "timings": {"total_ms": {"mean_ms": 8.0, "median_ms": 7.5}},
            },
            {
                "optimizer": "adam",
                "mode": "edge_compile_grads_sequence",
                "setup_ms": 2500.0,
                "gate_result": {
                    "accepted": True,
                    "fallback_reason": None,
                    "compiled_grad_count": 11,
                    "compiled_grad_filter_sequence": ["readouts", "conv_tp_weights"],
                    "graph_stats": {"node_count": 4602, "output_tensor_count": 17},
                },
                "timings": {"total_ms": {"mean_ms": 14.0, "median_ms": 13.5}},
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
    adam_compile_grads = summary["rows"]["adam/edge_compile_grads"]
    assert adam_compile_grads["setup_ms"] == 1500.0
    assert adam_compile_grads["mean_total_ms"] == 8.0
    assert adam_compile_grads["speedup_vs_position_eager"] == 2.5
    assert adam_compile_grads["compiled_grad_count"] == 3
    assert adam_compile_grads["compiled_grad_filter"] == "readouts"
    assert adam_compile_grads["graph_node_count"] == 100
    assert adam_compile_grads["graph_output_tensor_count"] == 6
    adam_sequence = summary["rows"]["adam/edge_compile_grads_sequence"]
    assert adam_sequence["setup_ms"] == 2500.0
    assert adam_sequence["compiled_grad_count"] == 11
    assert adam_sequence["compiled_grad_filter_sequence"] == [
        "readouts",
        "conv_tp_weights",
    ]
    assert adam_sequence["graph_node_count"] == 4602
    assert adam_sequence["graph_output_tensor_count"] == 17
    assert summary["mean_edge_compile_grads_total_ms"] == 8.0
    assert summary["mean_edge_compile_grads_sequence_total_ms"] == 14.0
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

def test_epoch_profile_runs_multi_case_parent_in_isolated_workers():
    profile = load_epoch_profile()

    assert profile.should_run_isolated(["adam"], ["position_eager"], case_worker=False) is False
    assert profile.should_run_isolated(["adam", "hybrid_muon"], ["edge_compile"], case_worker=False) is True
    assert profile.should_run_isolated(["adam"], ["position_eager", "edge_compile"], case_worker=False) is True
    assert profile.should_run_isolated(["adam", "hybrid_muon"], ["edge_compile"], case_worker=True) is False

def test_epoch_profile_edge_compile_input_names_exclude_labels_and_unused_geometry():
    profile = load_epoch_profile()
    data_keys = {
        "positions",
        "edge_index",
        "node_attrs",
        "batch",
        "ptr",
        "head",
        "shifts",
        "energy",
        "forces",
        "stress",
        "virials",
    }

    names = profile.edge_compile_input_names(data_keys)

    assert names == ("positions", "edge_index", "node_attrs", "batch", "ptr", "head")

def test_epoch_profile_shape_cache_key_ignores_batch_identity():
    profile = load_epoch_profile()

    left = profile.compile_shape_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={"positions": (286, 3), "node_attrs": (286, 4)},
    )
    right = profile.compile_shape_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={"positions": (286, 3), "node_attrs": (286, 4)},
    )

    assert left == right

def test_epoch_profile_gates_shape_cache_hits_by_default():
    profile = load_epoch_profile()

    assert profile.should_gate_cache_hit(cache_hit=True, scope="shape", enabled=True) is True
    assert profile.should_gate_cache_hit(cache_hit=False, scope="shape", enabled=True) is False
    assert profile.should_gate_cache_hit(cache_hit=True, scope="batch", enabled=True) is False
    assert profile.should_gate_cache_hit(cache_hit=True, scope="shape", enabled=False) is False

def test_epoch_profile_parser_accepts_skip_training_step():
    profile = load_epoch_profile()

    args = profile.build_parser().parse_args(["--skip-training-step"])

    assert args.skip_training_step is True


def test_epoch_profile_resets_compile_state_only_when_requested(monkeypatch):
    profile = load_epoch_profile()
    calls = []
    monkeypatch.setattr(profile, "_reset_compile_state", lambda: calls.append("reset"))

    default_args = profile.build_parser().parse_args([])
    profile.reset_compile_state_if_requested(default_args)
    assert calls == []

    enabled_args = profile.build_parser().parse_args(["--edge-reset-compile-state"])
    profile.reset_compile_state_if_requested(enabled_args)
    assert calls == ["reset"]


def test_epoch_profile_clears_compile_cache_only_when_requested():
    profile = load_epoch_profile()

    default_args = profile.build_parser().parse_args([])
    cache = {("shape", 1): object()}
    profile.clear_compile_cache_on_miss_if_requested(cache, default_args)
    assert cache

    enabled_args = profile.build_parser().parse_args(["--edge-clear-cache-on-miss"])
    profile.clear_compile_cache_on_miss_if_requested(cache, enabled_args)
    assert cache == {}


def test_epoch_profile_builds_cache_hit_gate_result_from_comparison():
    profile = load_epoch_profile()
    comparison = {"ok": False, "failed_checks": ["forces"]}

    result = profile.build_cache_hit_gate_result(
        comparison=comparison,
        cache_key=("shape", 2, 4),
    )

    assert result["enabled"] is True
    assert result["accepted"] is False
    assert result["fallback_reason"] == "cache_hit_equivalence_failed"
    assert result["comparison"] == comparison
    assert result["cache_hit"] is True
    assert result["cache_key"] == ["shape", 2, 4]
