from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from ase.io import read
from e3nn import o3

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import torch_geometric
from mace.tools.training_compile import prepare_model_for_training_compile


@dataclass(frozen=True)
class ProbeMode:
    name: str
    compiled: bool
    compute_force: bool
    loss_kind: str


PROBE_MODES: tuple[ProbeMode, ...] = (
    ProbeMode(
        name="eager_force_loss",
        compiled=False,
        compute_force=True,
        loss_kind="energy_forces",
    ),
    ProbeMode(
        name="compile_force_loss",
        compiled=True,
        compute_force=True,
        loss_kind="energy_forces",
    ),
    ProbeMode(
        name="compile_energy_only",
        compiled=True,
        compute_force=False,
        loss_kind="energy",
    ),
)


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(part) if part else None for part in chunk.split(":")]
            if len(parts) > 3:
                raise ValueError(f"invalid index range: {chunk}")
            start = 0 if parts[0] is None else parts[0]
            stop = parts[1]
            step = 1 if len(parts) < 3 or parts[2] is None else parts[2]
            if stop is None:
                raise ValueError(f"range stop is required: {chunk}")
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(chunk))
    if not indices:
        raise ValueError("at least one structure index is required")
    return indices


def get_probe_modes(names: Iterable[str] | None = None) -> list[ProbeMode]:
    modes = list(PROBE_MODES)
    if names is None:
        return modes
    by_name = {mode.name: mode for mode in modes}
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"unknown probe mode {name!r}; choices: {sorted(by_name)}")
        selected.append(by_name[name])
    return selected


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cueq_config(enabled: bool, device: torch.device):
    if not enabled:
        return None
    return CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=(device.type == "cuda"),
    )


def _load_batch(xyz: Path, indices: list[int], cutoff: float, device: torch.device):
    atoms_list = [read(xyz, index=index) for index in indices]
    atomic_numbers = sorted({int(z) for atoms in atoms_list for z in atoms.numbers})
    z_table = tools.AtomicNumberTable(atomic_numbers)
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    dataset = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
        for config in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to(device)
    return batch, z_table


def _create_model(
    *,
    z_table: tools.AtomicNumberTable,
    cutoff: float,
    device: torch.device,
    hidden_channels: int,
    max_ell: int,
    num_interactions: int,
    correlation: int,
    enable_cueq: bool,
) -> torch.nn.Module:
    model_config = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": max_ell,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": num_interactions,
        "num_elements": len(z_table.zs),
        "hidden_irreps": o3.Irreps(
            f"{hidden_channels}x0e + {hidden_channels}x1o"
        ),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": np.zeros(len(z_table.zs), dtype=float),
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": correlation,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": _cueq_config(enable_cueq, device),
    }
    return modules.ScaleShiftMACE(**model_config).to(device)


def _batch_dict(batch) -> dict:
    batch_dict = batch.to_dict()
    if "positions" in batch_dict:
        batch_dict["positions"] = batch_dict["positions"].detach().clone()
    return batch_dict


def _loss(output: dict, batch, loss_kind: str) -> torch.Tensor:
    energy = output["energy"]
    target_energy = getattr(batch, "energy", None)
    if target_energy is None:
        loss = energy.square().mean()
    else:
        loss = (energy - target_energy).square().mean()
    if loss_kind == "energy":
        return loss
    if output.get("forces") is None:
        raise RuntimeError("force-loss probe requested forces but model returned None")
    target_forces = getattr(batch, "forces", None)
    if target_forces is None:
        force_loss = output["forces"].square().mean()
    else:
        force_loss = (output["forces"] - target_forces).square().mean()
    return loss + force_loss


def _run_backward_once(model: torch.nn.Module, batch, mode: ProbeMode) -> float:
    model.zero_grad(set_to_none=True)
    output = model(
        _batch_dict(batch),
        training=True,
        compute_force=mode.compute_force,
        compute_virials=False,
        compute_stress=False,
    )
    loss = _loss(output, batch, mode.loss_kind)
    loss.backward()
    return float(loss.detach().cpu())


def _run_with_training_fallback(model: torch.nn.Module, batch, mode: ProbeMode) -> float:
    try:
        return _run_backward_once(model, batch, mode)
    except RuntimeError as exc:
        disable_compile_fallback = getattr(model, "disable_compile_fallback", None)
        if disable_compile_fallback is None or not disable_compile_fallback(exc):
            raise
        return _run_backward_once(model, batch, mode)


def run_mode(
    *,
    base_model: torch.nn.Module,
    batch,
    mode: ProbeMode,
    compile_mode: str,
    compile_fullgraph: bool,
    allow_fallback: bool,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict:
    model = copy.deepcopy(base_model)
    if mode.compiled:
        model = prepare_model_for_training_compile(
            model,
            enabled=True,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            allow_fallback=allow_fallback,
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    result = {"mode": asdict(mode), "status": "ok"}
    try:
        for _ in range(warmup):
            _run_with_training_fallback(model, batch, mode)
            _sync(device)
        seconds = []
        losses = []
        for _ in range(repeats):
            start = time.perf_counter()
            losses.append(_run_with_training_fallback(model, batch, mode))
            _sync(device)
            seconds.append(time.perf_counter() - start)
    except Exception as exc:  # pylint: disable=broad-except
        result.update({"status": "error", "error": repr(exc)})
        return result

    result.update(
        {
            "loss_last": losses[-1] if losses else None,
            "seconds_per_step": seconds,
            "mean_seconds_per_step": mean(seconds) if seconds else None,
            "compile_disabled": bool(getattr(model, "disabled", False)),
        }
    )
    if device.type == "cuda":
        result["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result


def run_probe(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch_geometric.seed_everything(args.seed)
    if args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"unsupported dtype: {args.dtype}")

    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(Path(args.xyz), indices, args.cutoff, device)
    base_model = _create_model(
        z_table=z_table,
        cutoff=args.cutoff,
        device=device,
        hidden_channels=args.hidden_channels,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        enable_cueq=args.enable_cueq,
    )
    modes = get_probe_modes(args.modes)
    results = [
        run_mode(
            base_model=base_model,
            batch=batch,
            mode=mode,
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            allow_fallback=args.allow_fallback,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        for mode in modes
    ]
    return {
        "xyz": str(args.xyz),
        "indices": indices,
        "num_graphs": len(indices),
        "num_atoms": int(batch.num_nodes),
        "atomic_numbers": z_table.zs,
        "device": str(device),
        "dtype": args.dtype,
        "enable_cueq": args.enable_cueq,
        "torch_version": torch.__version__,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xyz",
        default="/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz",
    )
    parser.add_argument("--indices", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--max-ell", type=int, default=1)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true", default=True)
    parser.add_argument("--no-allow-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.name for mode in PROBE_MODES],
        default=None,
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
