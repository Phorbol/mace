from __future__ import annotations

import logging

import torch

from mace.tools import compile as mace_compile


class RuntimeFallbackCompiledModule(torch.nn.Module):
    def __init__(
        self,
        *,
        eager_model: torch.nn.Module,
        compiled_model: torch.nn.Module,
        allow_fallback: bool,
    ) -> None:
        super().__init__()
        self.eager_model = eager_model
        self.__dict__["compiled_model"] = compiled_model
        self.allow_fallback = allow_fallback
        self.disabled = False

    def forward(self, *args, **kwargs):
        if self.disabled:
            return self.eager_model(*args, **kwargs)
        try:
            return self.compiled_model(*args, **kwargs)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            logging.warning(
                "training torch.compile failed at runtime; disabling compiled "
                "training model and retrying eager: %s",
                exc,
            )
            self.disabled = True
            return self.eager_model(*args, **kwargs)


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
        if allow_fallback:
            dynamo_config.suppress_errors = True
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logging.info(
            "Enabled training torch.compile: mode=%s fullgraph=%s",
            mode,
            fullgraph,
        )
        return RuntimeFallbackCompiledModule(
            eager_model=model,
            compiled_model=compiled,
            allow_fallback=allow_fallback,
        )
    except Exception as exc:
        message = f"training torch.compile setup failed: {exc}"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message) from exc
