from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


def load_op_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "profile_force_ops.py"
    )
    spec = importlib.util.spec_from_file_location("force_op_profile", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeEvent:
    key: str
    self_cuda_time_total: float = 0.0
    cuda_time_total: float = 0.0
    self_cpu_time_total: float = 0.0
    cpu_time_total: float = 0.0
    count: int = 1


def test_summarize_events_sorts_by_cuda_time_and_converts_to_ms():
    profile = load_op_profile()
    events = [
        FakeEvent("slow_cpu", self_cpu_time_total=9000.0, cpu_time_total=10000.0, count=3),
        FakeEvent("kernel_b", self_cuda_time_total=2000.0, cuda_time_total=7000.0, count=2),
        FakeEvent("kernel_a", self_cuda_time_total=3000.0, cuda_time_total=8000.0, count=4),
    ]

    summary = profile.summarize_events(events, sort_by="cuda_time_total", top_k=2)

    assert [row["name"] for row in summary] == ["kernel_a", "kernel_b"]
    assert summary[0] == {
        "name": "kernel_a",
        "count": 4,
        "self_cuda_time_ms": 3.0,
        "cuda_time_ms": 8.0,
        "self_cpu_time_ms": 0.0,
        "cpu_time_ms": 0.0,
    }


def test_summarize_events_can_sort_by_cpu_time_when_cuda_is_absent():
    profile = load_op_profile()
    events = [
        FakeEvent("small", cpu_time_total=1000.0, count=1),
        FakeEvent("large", cpu_time_total=4000.0, count=1),
    ]

    summary = profile.summarize_events(events, sort_by="cpu_time_total", top_k=1)

    assert summary[0]["name"] == "large"
    assert summary[0]["cpu_time_ms"] == 4.0


@dataclass
class FakeDeviceEvent:
    key: str
    self_device_time_total: float = 0.0
    device_time_total: float = 0.0
    self_cpu_time_total: float = 0.0
    cpu_time_total: float = 0.0
    count: int = 1


def test_summarize_events_supports_pytorch_device_time_fields():
    profile = load_op_profile()
    events = [
        FakeDeviceEvent("device_small", device_time_total=1000.0, count=1),
        FakeDeviceEvent("device_large", self_device_time_total=2500.0, device_time_total=5000.0, count=2),
    ]

    summary = profile.summarize_events(events, sort_by="device_time_total", top_k=1)

    assert summary[0]["name"] == "device_large"
    assert summary[0]["cuda_time_ms"] == 5.0
    assert summary[0]["self_cuda_time_ms"] == 2.5


def test_summarize_events_excludes_outer_mace_record_function_by_default():
    profile = load_op_profile()
    events = [
        FakeDeviceEvent("mace_force_loss_backward", device_time_total=10000.0, count=1),
        FakeDeviceEvent("aten::bmm", device_time_total=3000.0, count=2),
    ]

    summary = profile.summarize_events(events, sort_by="device_time_total", top_k=2)

    assert [row["name"] for row in summary] == ["aten::bmm"]


@dataclass
class FakeShapeEvent:
    key: str
    input_shapes: tuple = ()
    self_device_time_total: float = 0.0
    device_time_total: float = 0.0
    self_cpu_time_total: float = 0.0
    cpu_time_total: float = 0.0
    count: int = 1


def test_summarize_events_includes_input_shapes_when_recorded():
    profile = load_op_profile()
    events = [
        FakeShapeEvent(
            "aten::bmm",
            input_shapes=([32, 16, 64], [32, 64, 32]),
            device_time_total=3000.0,
            count=2,
        ),
    ]

    summary = profile.summarize_events(events, sort_by="device_time_total", top_k=1)

    assert summary[0]["input_shapes"] == [[32, 16, 64], [32, 64, 32]]



def test_group_by_input_shape_parser_enables_record_shapes():
    profile = load_op_profile()

    args = profile.build_parser().parse_args(["--group-by-input-shape"])

    assert args.group_by_input_shape is True
    assert args.record_shapes is True
