import pytest
import torch

from mace.tools.precision import TrainingPrecisionConfig, get_autocast_context


def test_precision_none_returns_disabled_config():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))

    assert config.enabled is False
    assert config.dtype is None


def test_precision_rejects_bf16_on_cpu():
    with pytest.raises(ValueError, match="bf16 AMP requires a CUDA device"):
        TrainingPrecisionConfig.from_name("bf16", torch.device("cpu"))


def test_autocast_context_none_is_noop():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))
    x = torch.ones(2, dtype=torch.float32)

    with get_autocast_context(config):
        y = x + 1

    assert y.dtype == torch.float32
