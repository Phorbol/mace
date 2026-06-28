from __future__ import annotations

import logging

import torch

from mace.tools import compile as mace_compile


def prepare_model_for_training_compile(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str,
    fullgraph: bool,
    allow_fallback: bool,
) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        message = "torch.compile is unavailable in this PyTorch build"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message)

    try:
        mace_compile.configure_autograd_for_compile(allow_autograd=True)
        import torch._dynamo.config as dynamo_config

        dynamo_config.optimize_ddp = False
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logging.info(
            "Enabled training torch.compile: mode=%s fullgraph=%s",
            mode,
            fullgraph,
        )
        return compiled
    except Exception as exc:
        message = f"training torch.compile setup failed: {exc}"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message) from exc
