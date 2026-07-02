from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingPrecisionConfig:
    enabled: bool
    dtype: torch.dtype | None
    float32_matmul_precision: str | None = None

    @classmethod
    def from_name(
        cls, name: str, device: torch.device, *, tf32: bool = False
    ) -> "TrainingPrecisionConfig":
        normalized = str(name).lower()
        matmul_precision = None
        if tf32:
            if device.type != "cuda":
                raise ValueError("TF32 training matmul precision requires a CUDA device")
            matmul_precision = "high"
        if normalized in {"none", "false", "off", "fp32"}:
            return cls(enabled=False, dtype=None, float32_matmul_precision=matmul_precision)
        if normalized == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 AMP requires a CUDA device")
            if not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 AMP is not supported by this CUDA device/build")
            return cls(enabled=True, dtype=torch.bfloat16, float32_matmul_precision=matmul_precision)
        if normalized == "fp16":
            if device.type != "cuda":
                raise ValueError("fp16 AMP requires a CUDA device")
            return cls(enabled=True, dtype=torch.float16, float32_matmul_precision=matmul_precision)
        raise ValueError(f"Unknown training AMP dtype: {name!r}")


def get_autocast_context(config: TrainingPrecisionConfig):
    if not config.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=config.dtype)


@contextmanager
def get_float32_matmul_precision_context(config: TrainingPrecisionConfig):
    precision = config.float32_matmul_precision
    if precision is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@contextmanager
def get_training_precision_context(config: TrainingPrecisionConfig):
    with get_float32_matmul_precision_context(config):
        with get_autocast_context(config):
            yield
