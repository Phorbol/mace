import argparse
import csv
import json
import os
import statistics
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
import torch.nn.functional as F
from ase import io
from e3nn import o3

from mace import modules
from mace.calculators.mace import MACECalculator
from mace.modules.wrapper_ops import CuEquivarianceConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark MACECalculator eager/compile/auto-dispatch latency and memory."
    )
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--dispatch-min-atoms", type=int, default=200)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--tf32", action="store_true", default=False)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def setup_cueq_disabled():
    return CuEquivarianceConfig(enabled=False)


def cuda_enabled(device: str):
    return device == "cuda" and torch.cuda.is_available()


def sync_if_cuda(device: str):
    if cuda_enabled(device):
        torch.cuda.synchronize()


def configure_cuda_precision(device: str, tf32: bool):
    if not cuda_enabled(device):
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def build_model(zs, cutoff, device):
    atomic_energies = np.zeros((len(zs),), dtype=float)
    cfg = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": len(zs),
        "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": zs,
        "correlation": 3,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": setup_cueq_disabled(),
    }
    return modules.ScaleShiftMACE(**cfg).to(device)


def summarize(times_ms):
    return {
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def bench_calc(calc, atoms, warmup, iters, device):
    times = []
    use_cuda = cuda_enabled(device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        start_event = end_event = None
        start_t = 0.0
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_t = time.perf_counter()
        calc.calculate(atoms=atoms)
        _ = calc.results.get("energy", None)
        _ = calc.results.get("forces", None)
        if use_cuda:
            end_event.record()
            sync_if_cuda(device)
        if i >= warmup:
            if use_cuda:
                times.append(start_event.elapsed_time(end_event))
            else:
                times.append((time.perf_counter() - start_t) * 1000.0)
    stats = summarize(times)
    stats["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if use_cuda else 0.0
    )
    return stats


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    configure_cuda_precision(args.device, args.tf32)

    rows = []
    for structure in args.structures:
        atoms0 = io.read(structure)
        zs = sorted({int(z) for z in atoms0.numbers})
        base_model = build_model(zs=zs, cutoff=5.0, device=args.device)
        variants = {
            "eager": MACECalculator(
                models=[deepcopy(base_model)],
                model_type="MACE",
                device=args.device,
                compile_mode=None,
                auto_compile_dispatch=False,
                tf32=args.tf32,
                amp=args.amp,
            ),
            "compile": MACECalculator(
                models=[deepcopy(base_model)],
                model_type="MACE",
                device=args.device,
                compile_mode=args.compile_mode,
                auto_compile_dispatch=False,
                compile_dispatch_min_atoms=args.dispatch_min_atoms,
                tf32=args.tf32,
                amp=args.amp,
            ),
            "auto": MACECalculator(
                models=[deepcopy(base_model)],
                model_type="MACE",
                device=args.device,
                compile_mode=args.compile_mode,
                auto_compile_dispatch=True,
                compile_dispatch_min_atoms=args.dispatch_min_atoms,
                tf32=args.tf32,
                amp=args.amp,
            ),
        }
        for size in args.sizes:
            atoms = atoms0.repeat((size, size, size))
            for mode, calc in variants.items():
                stats = bench_calc(
                    calc=calc,
                    atoms=atoms,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                )
                row = {
                    "structure_name": Path(structure).name,
                    "size": int(size),
                    "num_atoms": int(len(atoms)),
                    "mode": mode,
                    "dispatch_mode": calc.last_dispatch_mode,
                    "mean_ms": stats["mean_ms"],
                    "p50_ms": stats["p50_ms"],
                    "peak_mem_mb": stats["peak_mem_mb"],
                }
                rows.append(row)
                print(
                    f"[done] {row['structure_name']} size={size} mode={mode} "
                    f"dispatch={row['dispatch_mode']} mean={row['mean_ms']:.2f}ms "
                    f"peak={row['peak_mem_mb']:.1f}MB"
                )

    rows.sort(key=lambda r: (r["structure_name"], r["size"], r["mode"]))
    grouped = {}
    for r in rows:
        grouped[(r["structure_name"], r["size"])] = grouped.get(
            (r["structure_name"], r["size"]), {}
        )
        grouped[(r["structure_name"], r["size"])][r["mode"]] = r

    comparisons = []
    for key, g in grouped.items():
        if "eager" not in g or "compile" not in g or "auto" not in g:
            continue
        eager = g["eager"]
        comp = g["compile"]
        auto = g["auto"]
        comparisons.append(
            {
                "structure_name": key[0],
                "size": key[1],
                "num_atoms": auto["num_atoms"],
                "auto_dispatch_mode": auto["dispatch_mode"],
                "auto_vs_eager_speedup_x": eager["mean_ms"] / auto["mean_ms"],
                "auto_vs_compile_speedup_x": comp["mean_ms"] / auto["mean_ms"],
                "auto_vs_eager_mem_delta_mb": auto["peak_mem_mb"] - eager["peak_mem_mb"],
                "auto_vs_compile_mem_delta_mb": auto["peak_mem_mb"] - comp["peak_mem_mb"],
                "eager_mean_ms": eager["mean_ms"],
                "compile_mean_ms": comp["mean_ms"],
                "auto_mean_ms": auto["mean_ms"],
                "eager_peak_mem_mb": eager["peak_mem_mb"],
                "compile_peak_mem_mb": comp["peak_mem_mb"],
                "auto_peak_mem_mb": auto["peak_mem_mb"],
            }
        )

    out = {
        "device": (
            torch.cuda.get_device_name(0)
            if cuda_enabled(args.device)
            else args.device
        ),
        "torch": torch.__version__,
        "dispatch_min_atoms": args.dispatch_min_atoms,
        "compile_mode": args.compile_mode,
        "tf32": args.tf32,
        "amp": args.amp,
        "rows": rows,
        "comparisons": comparisons,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "structure_name",
                "size",
                "num_atoms",
                "mode",
                "dispatch_mode",
                "mean_ms",
                "p50_ms",
                "peak_mem_mb",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"JSON written: {out_json}")
    print(f"CSV written: {out_csv}")


if __name__ == "__main__":
    main()
