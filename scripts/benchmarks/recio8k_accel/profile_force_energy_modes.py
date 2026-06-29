from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch

from mace import modules, tools
from scripts.benchmarks.recio8k_accel.profile_training_step_phases import (
    _batch_dict,
    _create_model,
    _load_batch,
    _sync,
    parse_indices,
)


def _energy_loss(output: dict, batch, energy_weight: float) -> torch.Tensor:
    energy_error = output["energy"] - batch.energy
    return energy_weight * energy_error.square().mean()


def _force_loss(output: dict, batch, energy_weight: float, forces_weight: float) -> torch.Tensor:
    loss_fn = modules.WeightedEnergyForcesLoss(
        energy_weight=energy_weight, forces_weight=forces_weight
    )
    return loss_fn(pred=output, ref=batch)


def _summarize(values: list[float]) -> dict:
    return {
        "mean_ms": mean(values),
        "median_ms": median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_mode(model, batch, mode: str, args: argparse.Namespace, device: torch.device) -> dict:
    timings: list[float] = []
    losses: list[float] = []
    for step in range(args.warmup + args.repeats):
        collect = step >= args.warmup
        model.zero_grad(set_to_none=True)
        _sync(device)
        start = time.perf_counter()

        if mode == "energy_forward":
            output = model(
                _batch_dict(batch),
                training=True,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
            )
            loss = _energy_loss(output, batch, args.energy_weight)
        elif mode == "energy_loss_backward":
            output = model(
                _batch_dict(batch),
                training=True,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
            )
            loss = _energy_loss(output, batch, args.energy_weight)
            loss.backward()
        elif mode == "force_forward":
            output = model(
                _batch_dict(batch),
                training=True,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            loss = _force_loss(output, batch, args.energy_weight, args.forces_weight)
        elif mode == "force_loss_backward":
            output = model(
                _batch_dict(batch),
                training=True,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            loss = _force_loss(output, batch, args.energy_weight, args.forces_weight)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1.0e3
        if collect:
            timings.append(elapsed_ms)
            losses.append(float(loss.detach().cpu()))
        del output, loss

    return {
        "mode": mode,
        "timing": _summarize(timings),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": mean(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"))
    parser.add_argument("--indices", default="0:32")
    parser.add_argument("--output", type=Path, default=Path("force_energy_mode_profile.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default="energy_forward,energy_loss_backward,force_forward,force_loss_backward")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--max-ell", type=int, default=3)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--avg-num-neighbors", type=float, default=25.55)
    parser.add_argument("--enable-cueq", action=argparse.BooleanOptionalAction, default=True)
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
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [_time_mode(model, batch, mode, args, device) for mode in modes]

    by_mode = {result["mode"]: result["timing"]["median_ms"] for result in results}
    derived = {}
    if "energy_forward" in by_mode and "force_forward" in by_mode:
        derived["force_construction_increment_median_ms"] = (
            by_mode["force_forward"] - by_mode["energy_forward"]
        )
    if "energy_loss_backward" in by_mode and "force_loss_backward" in by_mode:
        derived["force_loss_backward_increment_median_ms"] = (
            by_mode["force_loss_backward"] - by_mode["energy_loss_backward"]
        )

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
        "derived": derived,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
