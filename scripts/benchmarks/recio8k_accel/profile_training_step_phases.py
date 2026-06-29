from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

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
from mace.tools.hybrid_muon import (
    HybridMuon,
    build_hybrid_muon_param_groups,
    summarize_hybrid_muon_routes,
)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    optimizer: torch.optim.Optimizer
    route_summary: list[dict] | None = None


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
    avg_num_neighbors: float,
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
        "avg_num_neighbors": avg_num_neighbors,
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


def _build_optimizer(
    optimizer_name: str,
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    muon_lr_factor: float,
) -> OptimizerSpec:
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
        )
        return OptimizerSpec(name=optimizer_name, optimizer=optimizer)
    if optimizer_name == "hybrid_muon":
        groups, summary = build_hybrid_muon_param_groups(
            model.named_parameters(),
            lr=lr,
            weight_decay=weight_decay,
            muon_weight_decay=0.0,
            muon_lr_factor=muon_lr_factor,
            beta=0.9,
            adam_betas=(0.9, 0.999),
            eps=1.0e-8,
            amsgrad=False,
        )
        optimizer = HybridMuon(groups, lr=lr, weight_decay=weight_decay)
        return OptimizerSpec(name=optimizer_name, optimizer=optimizer, route_summary=summary)
    raise ValueError(f"unknown optimizer {optimizer_name!r}")


def _empty_timing() -> dict[str, list[float]]:
    return {
        "zero_grad_ms": [],
        "forward_force_loss_ms": [],
        "backward_clip_ms": [],
        "optimizer_step_ms": [],
        "total_ms": [],
    }


def _record_phase(device: torch.device, timings: dict[str, list[float]], key: str, start: float) -> float:
    _sync(device)
    now = time.perf_counter()
    timings[key].append((now - start) * 1.0e3)
    return now


def profile_optimizer(
    *,
    optimizer_name: str,
    batch,
    z_table: tools.AtomicNumberTable,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    model = _create_model(
        z_table=z_table,
        cutoff=args.cutoff,
        device=device,
        hidden_channels=args.hidden_channels,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        avg_num_neighbors=args.avg_num_neighbors,
        enable_cueq=args.enable_cueq,
    )
    loss_fn = modules.WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight, forces_weight=args.forces_weight
    )
    optimizer_spec = _build_optimizer(
        optimizer_name,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr_factor=args.hybrid_muon_lr_factor,
    )
    optimizer = optimizer_spec.optimizer
    timings = _empty_timing()
    losses: list[float] = []

    for step in range(args.warmup + args.repeats):
        collect = step >= args.warmup
        _sync(device)
        total_start = phase_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if collect:
            phase_start = _record_phase(device, timings, "zero_grad_ms", phase_start)
        else:
            _sync(device)
            phase_start = time.perf_counter()

        output = model(
            _batch_dict(batch),
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        loss = loss_fn(pred=output, ref=batch)
        if collect:
            phase_start = _record_phase(
                device, timings, "forward_force_loss_ms", phase_start
            )
        else:
            _sync(device)
            phase_start = time.perf_counter()

        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if collect:
            phase_start = _record_phase(device, timings, "backward_clip_ms", phase_start)
        else:
            _sync(device)
            phase_start = time.perf_counter()

        optimizer.step()
        if collect:
            _record_phase(device, timings, "optimizer_step_ms", phase_start)
            _sync(device)
            timings["total_ms"].append((time.perf_counter() - total_start) * 1.0e3)
            losses.append(float(loss.detach().cpu()))

    summary = {
        key: {
            "mean_ms": mean(values),
            "median_ms": median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for key, values in timings.items()
    }
    total_mean = summary["total_ms"]["mean_ms"]
    total_median = summary["total_ms"]["median_ms"]
    for key, stats in summary.items():
        stats["mean_fraction_pct"] = 100.0 * stats["mean_ms"] / total_mean
        stats["median_fraction_pct"] = 100.0 * stats["median_ms"] / total_median
    payload = {
        "optimizer": optimizer_name,
        "timings": summary,
        "loss_mean": mean(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "route_summary": optimizer_spec.route_summary,
    }
    if optimizer_spec.route_summary is not None:
        payload["route_text"] = summarize_hybrid_muon_routes(optimizer_spec.route_summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"))
    parser.add_argument("--indices", default="0:32")
    parser.add_argument("--output", type=Path, default=Path("training_step_phase_profile.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizers", default="adam,hybrid_muon")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--max-ell", type=int, default=3)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--avg-num-neighbors", type=float, default=25.55)
    parser.add_argument("--enable-cueq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=5.0e-7)
    parser.add_argument("--hybrid-muon-lr-factor", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=40.0)
    parser.add_argument("--forces-weight", type=float, default=1000.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(args.xyz, indices, args.cutoff, device)
    optimizers = [name.strip() for name in args.optimizers.split(",") if name.strip()]
    results = [
        profile_optimizer(
            optimizer_name=name,
            batch=batch,
            z_table=z_table,
            args=args,
            device=device,
        )
        for name in optimizers
    ]
    payload = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "xyz": str(args.xyz),
        "indices": indices,
        "num_structures": len(indices),
        "num_atoms": int(batch.positions.shape[0]),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "enable_cueq": args.enable_cueq,
        "max_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
