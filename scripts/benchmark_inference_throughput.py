import argparse
import json
import os
import random
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
        description="Benchmark end-to-end batched inference throughput."
    )
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--dispatch-min-atoms", type=int, default=200)
    p.add_argument("--warmup-jobs", type=int, default=4)
    p.add_argument("--tf32", action="store_true", default=False)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def setup_cueq_disabled():
    return CuEquivarianceConfig(enabled=False)


def cuda_enabled(device: str):
    return device == "cuda" and torch.cuda.is_available()


def sync_if_cuda(device: str):
    if cuda_enabled(device):
        torch.cuda.synchronize()


def build_model(all_zs, device):
    atomic_energies = np.zeros((len(all_zs),), dtype=float)
    cfg = {
        "r_max": 5.0,
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
        "num_elements": len(all_zs),
        "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": all_zs,
        "correlation": 3,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": setup_cueq_disabled(),
    }
    return modules.ScaleShiftMACE(**cfg).to(device)


def make_jobs(structures, sizes, repeats, seed):
    jobs = []
    for _ in range(repeats):
        for structure in structures:
            atoms0 = io.read(structure)
            for size in sizes:
                atoms = atoms0.repeat((size, size, size))
                jobs.append(
                    {
                        "structure_name": Path(structure).name,
                        "size": int(size),
                        "num_atoms": int(len(atoms)),
                        "atoms": atoms,
                    }
                )
    rng = random.Random(seed)
    rng.shuffle(jobs)
    return jobs


def run_variant(name, calc, jobs, warmup_jobs, device):
    for j in jobs[:warmup_jobs]:
        calc.calculate(atoms=j["atoms"])
        _ = calc.results.get("energy", None)
        _ = calc.results.get("forces", None)
    sync_if_cuda(device)

    if cuda_enabled(device):
        torch.cuda.reset_peak_memory_stats()
    route_stats = {"eager": 0, "compile": 0}
    t0 = time.perf_counter()
    for j in jobs:
        calc.calculate(atoms=j["atoms"])
        _ = calc.results.get("energy", None)
        _ = calc.results.get("forces", None)
        route_stats[calc.last_dispatch_mode] = (
            route_stats.get(calc.last_dispatch_mode, 0) + 1
        )
    sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        if cuda_enabled(device)
        else 0.0
    )
    return {
        "variant": name,
        "jobs": len(jobs),
        "elapsed_s": elapsed,
        "throughput_jobs_per_s": len(jobs) / elapsed,
        "mean_ms_per_job": elapsed * 1000.0 / len(jobs),
        "peak_mem_mb": peak_mem_mb,
        "route_stats": route_stats,
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_zs = sorted(
        {
            int(z)
            for structure in args.structures
            for z in io.read(structure).numbers
        }
    )
    jobs = make_jobs(args.structures, args.sizes, args.repeats, args.seed)

    base_model = build_model(all_zs=all_zs, device=args.device)
    eager_calc = MACECalculator(
        models=[deepcopy(base_model)],
        model_type="MACE",
        device=args.device,
        compile_mode=None,
        auto_compile_dispatch=False,
        tf32=args.tf32,
        amp=args.amp,
    )
    compile_calc = MACECalculator(
        models=[deepcopy(base_model)],
        model_type="MACE",
        device=args.device,
        compile_mode=args.compile_mode,
        auto_compile_dispatch=False,
        compile_dispatch_min_atoms=args.dispatch_min_atoms,
        tf32=args.tf32,
        amp=args.amp,
    )
    auto_calc = MACECalculator(
        models=[deepcopy(base_model)],
        model_type="MACE",
        device=args.device,
        compile_mode=args.compile_mode,
        auto_compile_dispatch=True,
        compile_dispatch_min_atoms=args.dispatch_min_atoms,
        tf32=args.tf32,
        amp=args.amp,
    )

    rows = []
    for name, calc in (
        ("eager", eager_calc),
        ("compile", compile_calc),
        ("auto", auto_calc),
    ):
        row = run_variant(
            name=name,
            calc=calc,
            jobs=jobs,
            warmup_jobs=args.warmup_jobs,
            device=args.device,
        )
        rows.append(row)
        print(
            f"[done] {name} throughput={row['throughput_jobs_per_s']:.3f} jobs/s "
            f"mean={row['mean_ms_per_job']:.2f} ms peak={row['peak_mem_mb']:.1f} MB "
            f"routes={row['route_stats']}"
        )

    table = {r["variant"]: r for r in rows}
    summary = {
        "device": (
            torch.cuda.get_device_name(0)
            if cuda_enabled(args.device)
            else args.device
        ),
        "jobs": len(jobs),
        "dispatch_min_atoms": args.dispatch_min_atoms,
        "tf32": args.tf32,
        "amp": args.amp,
        "rows": rows,
        "throughput_speedup_auto_vs_eager_x": table["auto"]["throughput_jobs_per_s"]
        / table["eager"]["throughput_jobs_per_s"],
        "throughput_speedup_auto_vs_compile_x": table["auto"]["throughput_jobs_per_s"]
        / table["compile"]["throughput_jobs_per_s"],
        "peak_mem_auto_minus_eager_mb": table["auto"]["peak_mem_mb"]
        - table["eager"]["peak_mem_mb"],
        "peak_mem_auto_minus_compile_mb": table["auto"]["peak_mem_mb"]
        - table["compile"]["peak_mem_mb"],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"JSON written: {out_json}")


if __name__ == "__main__":
    main()
