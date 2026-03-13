import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from ase import io

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from e3nn import o3

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", required=True)
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["eager", "compile"], default="eager")
    p.add_argument("--task", choices=["train", "infer"], default="train")
    p.add_argument("--compute-force", action="store_true", default=False)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def setup_cueq_disabled():
    return CuEquivarianceConfig(enabled=False)


def build_model_factory(zs: List[int], cutoff: float):
    atomic_energies = np.zeros((len(zs),), dtype=float)

    def factory(device: str = "cuda", enable_cueq: bool = False):
        _ = enable_cueq
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

    return factory


def make_batch(structure_file: str, size: int, cutoff: float, device: str):
    atoms = io.read(structure_file)
    atoms = atoms.repeat((size, size, size))
    zs = sorted({int(z) for z in atoms.numbers})
    table = tools.AtomicNumberTable(zs)
    conf = data.config_from_atoms(atoms)
    dataset = [data.AtomicData.from_config(conf, z_table=table, cutoff=cutoff)]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device).to_dict()
    return atoms, zs, batch


def clone_batch(batch: Dict):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_train(model, batch, warmup: int, iters: int):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    times_ms = []
    for i in range(warmup + iters):
        b = clone_batch(batch)
        b["positions"].requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push("train_step")
        start.record()
        torch.cuda.nvtx.range_push("forward")
        out = model(b, training=True)
        torch.cuda.nvtx.range_pop()
        loss = out["energy"].sum()
        torch.cuda.nvtx.range_push("backward")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer_step")
        opt.step()
        torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if i >= warmup:
            times_ms.append(start.elapsed_time(end))
    return times_ms


def run_infer(model, batch, warmup: int, iters: int, compute_force: bool):
    times_ms = []
    model.eval()
    for i in range(warmup + iters):
        b = clone_batch(batch)
        if compute_force:
            b["positions"].requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push("infer_step")
        start.record()
        _ = model(b, training=False, compute_force=compute_force)
        end.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if i >= warmup:
            times_ms.append(start.elapsed_time(end))
    return times_ms


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch_geometric.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    cutoff = 5.0
    atoms, zs, batch = make_batch(args.structure, args.size, cutoff, args.device)
    model_factory = build_model_factory(zs, cutoff)

    if args.mode == "eager":
        model = model_factory(device=args.device, enable_cueq=False)
    else:
        torch.compiler.reset()
        model = torch.compile(
            mace_compile.prepare(model_factory)(args.device, enable_cueq=False),
            mode=args.compile_mode,
        )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if args.task == "train":
        times_ms = run_train(model, batch, warmup=args.warmup, iters=args.iters)
    else:
        times_ms = run_infer(
            model,
            batch,
            warmup=args.warmup,
            iters=args.iters,
            compute_force=args.compute_force,
        )
    elapsed_s = time.time() - t0

    out = {
        "structure": args.structure,
        "structure_name": Path(args.structure).name,
        "size": args.size,
        "mode": args.mode,
        "task": args.task,
        "compute_force": args.compute_force,
        "num_atoms": int(len(atoms)),
        "num_edges": int(batch["edge_index"].shape[1]),
        "iters": args.iters,
        "warmup": args.warmup,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "steps_per_s": float(1000.0 / np.mean(times_ms)),
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "elapsed_s": elapsed_s,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
