import pytest
import torch

from mace.tools.precision import (
    TrainingPrecisionConfig,
    get_autocast_context,
    get_training_precision_context,
)


def test_precision_none_returns_disabled_config():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))

    assert config.enabled is False
    assert config.dtype is None
    assert config.float32_matmul_precision is None


def test_precision_tf32_sets_high_matmul_precision_for_cuda():
    config = TrainingPrecisionConfig.from_name(
        "none", torch.device("cuda"), tf32=True
    )

    assert config.enabled is False
    assert config.dtype is None
    assert config.float32_matmul_precision == "high"


def test_precision_rejects_tf32_on_cpu():
    with pytest.raises(ValueError, match="TF32 training matmul precision requires a CUDA device"):
        TrainingPrecisionConfig.from_name("none", torch.device("cpu"), tf32=True)


def test_precision_rejects_bf16_on_cpu():
    with pytest.raises(ValueError, match="bf16 AMP requires a CUDA device"):
        TrainingPrecisionConfig.from_name("bf16", torch.device("cpu"))


def test_precision_bf16_accepts_supported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    config = TrainingPrecisionConfig.from_name("bf16", torch.device("cuda"))

    assert config.enabled is True
    assert config.dtype is torch.bfloat16


def test_precision_rejects_bf16_on_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(ValueError, match="bf16 AMP is not supported"):
        TrainingPrecisionConfig.from_name("bf16", torch.device("cuda"))


def test_precision_fp16_accepts_cuda_without_bf16_check(monkeypatch):
    def fail_if_called():
        raise AssertionError("fp16 should not query bf16 support")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fail_if_called)

    config = TrainingPrecisionConfig.from_name("fp16", torch.device("cuda"))

    assert config.enabled is True
    assert config.dtype is torch.float16


def test_autocast_context_none_is_noop():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))
    x = torch.ones(2, dtype=torch.float32)

    with get_autocast_context(config):
        y = x + 1

    assert y.dtype == torch.float32


def test_precision_config_constructs_for_default_training_path():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))

    assert config.enabled is False


def test_autocast_context_uses_cuda_dtype(monkeypatch):
    calls = []

    class FakeAutocast:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    def fake_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return FakeAutocast()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    config = TrainingPrecisionConfig(enabled=True, dtype=torch.bfloat16)

    with get_autocast_context(config):
        calls.append("body")

    assert calls == [("cuda", torch.bfloat16), "enter", "body", "exit"]


def test_training_precision_context_sets_and_restores_tf32_matmul_precision(monkeypatch):
    calls = []
    current = {"precision": "highest"}

    def fake_get_precision():
        calls.append(("get", current["precision"]))
        return current["precision"]

    def fake_set_precision(value):
        calls.append(("set", value))
        current["precision"] = value

    monkeypatch.setattr(torch, "get_float32_matmul_precision", fake_get_precision)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", fake_set_precision)
    config = TrainingPrecisionConfig(
        enabled=False, dtype=None, float32_matmul_precision="high"
    )

    with get_training_precision_context(config):
        calls.append(("body", current["precision"]))

    assert calls == [("get", "highest"), ("set", "high"), ("body", "high"), ("set", "highest")]
    assert current["precision"] == "highest"
