from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingPrecisionConfig:
    enabled: bool
    dtype: torch.dtype | None

    @classmethod
    def from_name(cls, name: str, device: torch.device) -> "TrainingPrecisionConfig":
        normalized = str(name).lower()
        if normalized in {"none", "false", "off", "fp32"}:
            return cls(enabled=False, dtype=None)
        if normalized == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 AMP requires a CUDA device")
            if not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 AMP is not supported by this CUDA device/build")
            return cls(enabled=True, dtype=torch.bfloat16)
        if normalized == "fp16":
            if device.type != "cuda":
                raise ValueError("fp16 AMP requires a CUDA device")
            return cls(enabled=True, dtype=torch.float16)
        raise ValueError(f"Unknown training AMP dtype: {name!r}")


def get_autocast_context(config: TrainingPrecisionConfig):
    if not config.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=config.dtype)
