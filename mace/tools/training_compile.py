from __future__ import annotations

import logging

import torch

from mace.modules.utils import get_outputs, prepare_graph
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

    def disable_compile_fallback(self, exc: Exception) -> bool:
        if not self.allow_fallback:
            return False
        logging.warning(
            "training torch.compile failed during backward; disabling compiled "
            "training model and retrying eager: %s",
            exc,
        )
        self.disabled = True
        return True

    def forward(self, *args, **kwargs):
        if self.disabled:
            return self.eager_model(*args, **kwargs)
        try:
            return self.compiled_model(*args, **kwargs)
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self.eager_model(*args, **kwargs)


class EnergyOnlyForceCompiledModule(RuntimeFallbackCompiledModule):
    def forward(self, data, *args, **kwargs):
        compute_force = kwargs.get("compute_force", True)
        unsupported_force_outputs = any(
            kwargs.get(name, False)
            for name in (
                "compute_virials",
                "compute_stress",
                "compute_displacement",
                "compute_hessian",
                "compute_edge_forces",
                "compute_atomic_stresses",
                "lammps_mliap",
            )
        )
        if self.disabled or not compute_force or unsupported_force_outputs:
            return self.eager_model(data, *args, **kwargs)

        try:
            if "positions" in data:
                data["positions"].requires_grad_(True)
            energy_kwargs = dict(kwargs)
            energy_kwargs.update(
                {
                    "compute_force": False,
                    "compute_virials": False,
                    "compute_stress": False,
                    "compute_displacement": False,
                    "compute_hessian": False,
                    "compute_edge_forces": False,
                    "compute_atomic_stresses": False,
                }
            )
            output = self.compiled_model(data, *args, **energy_kwargs)
            ctx = prepare_graph(data)
            forces, _, _, _, _ = get_outputs(
                energy=output["energy"],
                positions=ctx.positions,
                displacement=ctx.displacement,
                vectors=ctx.vectors,
                cell=ctx.cell,
                training=kwargs.get("training", False),
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            output = dict(output)
            output.update(
                {
                    "forces": forces,
                    "virials": None,
                    "stress": None,
                    "hessian": None,
                    "edge_forces": None,
                }
            )
            return output
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self.eager_model(data, *args, **kwargs)


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
        return EnergyOnlyForceCompiledModule(
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
