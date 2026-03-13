import argparse
import csv
import json
import os
from contextlib import nullcontext
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
    p.add_argument("--mode", choices=["eager", "compile"], default="eager")
    p.add_argument("--task", choices=["train", "infer"], default="train")
    p.add_argument("--compute-force", action="store_true", default=False)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-mode", default="default")
    p.add_argument(
        "--amp",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="Autocast dtype used in forward path",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        default=False,
        help="Enable TensorFloat32 matmul/cuDNN on CUDA",
    )
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def setup_cueq_disabled():
    return CuEquivarianceConfig(enabled=False)


def resolve_amp_dtype(amp: str):
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return None


def configure_cuda_precision(device: str, tf32: bool):
    if device != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def autocast_ctx(device: str, amp_dtype):
    if device == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def cuda_enabled(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_available()


def sync_if_cuda(device: str):
    if cuda_enabled(device):
        torch.cuda.synchronize()


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


def classify_op(name: str, mem_bytes: float, self_dev_ms: float) -> str:
    lname = name.lower()
    if "scatter" in lname or "index" in lname or "copy" in lname:
        return "io_bound_likely"
    if "matmul" in lname or "mm" in lname or "gemm" in lname:
        return "compute_bound_likely"
    if mem_bytes > 64 * 1024 * 1024 and self_dev_ms < 2.0:
        return "io_bound_likely"
    if self_dev_ms > 5.0 and mem_bytes < 8 * 1024 * 1024:
        return "compute_bound_likely"
    return "mixed"


def run_once(model, batch, task: str, compute_force: bool, amp_dtype, device: str):
    if task == "train":
        with autocast_ctx(device, amp_dtype):
            out = model(batch, training=True)
            loss = out["energy"].sum()
        loss.backward()
    else:
        with autocast_ctx(device, amp_dtype):
            _ = model(batch, training=False, compute_force=compute_force)


def run_profile_pass(model, batch, task, compute_force, amp_dtype, device):
    b = clone_batch(batch)
    if task == "train" or compute_force:
        b["positions"].requires_grad_(True)
    with torch.profiler.profile(
        activities=(
            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if cuda_enabled(device)
            else [torch.profiler.ProfilerActivity.CPU]
        ),
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        run_once(model, b, task, compute_force, amp_dtype, device)
        sync_if_cuda(device)
    return prof


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    amp_dtype = resolve_amp_dtype(args.amp)
    configure_cuda_precision(args.device, args.tf32)
    torch_geometric.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    cutoff = 5.0
    atoms, zs, batch = make_batch(args.structure, args.size, cutoff, args.device)
    model_factory = build_model_factory(zs, cutoff)
    if args.mode == "eager":
        model = model_factory(device=args.device, enable_cueq=False)
    else:
        model = torch.compile(
            mace_compile.prepare(model_factory)(args.device, enable_cueq=False),
            mode=args.compile_mode,
        )

    profile_compute_force = args.compute_force
    note = ""
    prof = None
    for attempt in range(2):
        try:
            # Warmup
            for _ in range(2):
                b = clone_batch(batch)
                if args.task == "train" or profile_compute_force:
                    b["positions"].requires_grad_(True)
                run_once(
                    model,
                    b,
                    args.task,
                    profile_compute_force,
                    amp_dtype,
                    args.device,
                )
                if args.task == "train":
                    model.zero_grad(set_to_none=True)
            sync_if_cuda(args.device)

            prof = run_profile_pass(
                model=model,
                batch=batch,
                task=args.task,
                compute_force=profile_compute_force,
                amp_dtype=amp_dtype,
                device=args.device,
            )
            break
        except Exception as exc:  # pylint: disable=broad-except
            if (
                attempt == 0
                and args.mode == "compile"
                and args.task == "infer"
                and profile_compute_force
            ):
                profile_compute_force = False
                note = (
                    "compile+infer force diagnostic failed, fallback to energy-only: "
                    f"{type(exc).__name__}"
                )
                continue
            raise
    assert prof is not None

    rows = []
    events = sorted(
        prof.key_averages(),
        key=lambda e: getattr(e, "self_device_time_total", 0.0),
        reverse=True,
    )
    for e in events[: args.top_k]:
        self_dev_ms = float(e.self_device_time_total) / 1000.0
        dev_total_ms = float(e.device_time_total) / 1000.0
        mem_bytes = float(abs(getattr(e, "self_device_memory_usage", 0.0)))
        rows.append(
            {
                "op": e.key,
                "calls": int(e.count),
                "self_device_ms": self_dev_ms,
                "device_total_ms": dev_total_ms,
                "self_cpu_ms": float(e.self_cpu_time_total) / 1000.0,
                "self_device_mem_mb": mem_bytes / 1024.0 / 1024.0,
                "intensity_hint": classify_op(e.key, mem_bytes, self_dev_ms),
            }
        )

    result = {
        "structure": args.structure,
        "structure_name": Path(args.structure).name,
        "size": args.size,
        "mode": args.mode,
        "task": args.task,
        "compute_force": profile_compute_force,
        "note": note,
        "num_atoms": int(len(atoms)),
        "num_edges": int(batch["edge_index"].shape[1]),
        "device": (
            torch.cuda.get_device_name(0)
            if args.device == "cuda" and torch.cuda.is_available()
            else args.device
        ),
        "torch": torch.__version__,
        "tf32": args.tf32,
        "amp": args.amp,
        "top_ops": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "op",
                "calls",
                "self_device_ms",
                "device_total_ms",
                "self_cpu_ms",
                "self_device_mem_mb",
                "intensity_hint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
