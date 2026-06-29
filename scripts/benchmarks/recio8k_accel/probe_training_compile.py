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


EQUIVALENCE_CANDIDATES = ("eager_copy", "compile_readouts")


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


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.detach() - right.detach()).abs().max().cpu())


def _within_tolerance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if left.shape != right.shape:
        return False
    return bool(torch.allclose(left.detach(), right.detach(), atol=atol, rtol=rtol))


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._orig_mod", "").replace("_orig_mod.", "")


def _force_loss_snapshot(model: torch.nn.Module, batch) -> dict:
    model.zero_grad(set_to_none=True)
    output = model(
        _batch_dict(batch),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    loss = _loss(output, batch, "energy_forces")
    loss.backward()
    grads = {
        _canonical_parameter_name(name): (
            None if param.grad is None else param.grad.detach().clone()
        )
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    return {
        "energy": output["energy"].detach().clone(),
        "forces": output["forces"].detach().clone(),
        "loss": loss.detach().clone(),
        "grads": grads,
    }


def compare_force_loss_equivalence(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    batch,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> dict:
    reference = _force_loss_snapshot(reference_model, batch)
    candidate = _force_loss_snapshot(candidate_model, batch)

    failed_checks: list[str] = []
    if not _within_tolerance(reference["energy"], candidate["energy"], atol=atol, rtol=rtol):
        failed_checks.append("energy")
    if not _within_tolerance(reference["forces"], candidate["forces"], atol=atol, rtol=rtol):
        failed_checks.append("forces")
    if not _within_tolerance(reference["loss"], candidate["loss"], atol=atol, rtol=rtol):
        failed_checks.append("loss")

    grad_diffs: dict[str, float] = {}
    reference_grad_names = set(reference["grads"])
    candidate_grad_names = set(candidate["grads"])
    for missing_name in sorted(reference_grad_names ^ candidate_grad_names):
        grad_diffs[missing_name] = float("inf")
        failed_checks.append(f"grad:{missing_name}")
    for name in sorted(reference_grad_names & candidate_grad_names):
        ref_grad = reference["grads"][name]
        cand_grad = candidate["grads"][name]
        if ref_grad is None and cand_grad is None:
            grad_diffs[name] = 0.0
            continue
        if ref_grad is None or cand_grad is None:
            grad_diffs[name] = float("inf")
            failed_checks.append(f"grad:{name}")
            continue
        grad_diffs[name] = _max_abs_diff(ref_grad, cand_grad)
        if not _within_tolerance(ref_grad, cand_grad, atol=atol, rtol=rtol):
            failed_checks.append(f"grad:{name}")

    return {
        "ok": not failed_checks,
        "atol": atol,
        "rtol": rtol,
        "energy_max_abs_diff": _max_abs_diff(reference["energy"], candidate["energy"]),
        "forces_max_abs_diff": _max_abs_diff(reference["forces"], candidate["forces"]),
        "loss_abs_diff": _max_abs_diff(reference["loss"], candidate["loss"]),
        "param_grad_max_abs_diff": grad_diffs,
        "failed_checks": failed_checks,
    }


def build_equivalence_candidate_model(
    base_model: torch.nn.Module,
    *,
    candidate: str,
    compile_mode: str,
    compile_fullgraph: bool,
) -> torch.nn.Module:
    candidate_model = copy.deepcopy(base_model)
    if candidate == "eager_copy":
        return candidate_model
    if candidate == "compile_readouts":
        if not hasattr(candidate_model, "readouts"):
            raise ValueError("compile_readouts candidate requires model.readouts")
        for index, readout in enumerate(candidate_model.readouts):
            candidate_model.readouts[index] = torch.compile(
                readout,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
            )
        return candidate_model
    raise ValueError(
        f"unknown equivalence candidate {candidate!r}; "
        f"choices: {list(EQUIVALENCE_CANDIDATES)}"
    )


def run_force_loss_equivalence_gate(
    base_model: torch.nn.Module,
    batch,
    *,
    candidate: str,
    compile_mode: str,
    compile_fullgraph: bool,
    atol: float,
    rtol: float,
) -> dict:
    try:
        candidate_model = build_equivalence_candidate_model(
            base_model,
            candidate=candidate,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
        )
        result = compare_force_loss_equivalence(
            base_model,
            candidate_model,
            batch,
            atol=atol,
            rtol=rtol,
        )
        result["candidate"] = candidate
        result["status"] = "ok"
        return result
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "candidate": candidate,
            "status": "error",
            "ok": False,
            "error": repr(exc),
            "failed_checks": ["exception"],
        }


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
    payload = {
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
    if getattr(args, "equivalence_gate", False):
        payload["force_loss_equivalence"] = run_force_loss_equivalence_gate(
            base_model,
            batch,
            candidate=getattr(args, "equivalence_candidate", "eager_copy"),
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            atol=args.equivalence_atol,
            rtol=args.equivalence_rtol,
        )
    return payload


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
    parser.add_argument("--equivalence-gate", action="store_true")
    parser.add_argument(
        "--equivalence-candidate",
        choices=EQUIVALENCE_CANDIDATES,
        default="eager_copy",
    )
    parser.add_argument("--equivalence-atol", type=float, default=1.0e-6)
    parser.add_argument("--equivalence-rtol", type=float, default=1.0e-5)
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
