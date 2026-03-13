import argparse
import csv
import json
import os
import statistics
import time
from contextlib import nullcontext
from copy import deepcopy
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
    p.add_argument(
        "--structures",
        nargs="+",
        required=True,
        help="Absolute paths to structure files readable by ASE",
    )
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup-train", type=int, default=8)
    p.add_argument("--iters-train", type=int, default=30)
    p.add_argument("--warmup-infer", type=int, default=8)
    p.add_argument("--iters-infer", type=int, default=50)
    p.add_argument("--seed", type=int, default=1702)
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
    p.add_argument(
        "--compile-bucket-atoms",
        type=int,
        default=0,
        help="Atom-count bucket size for compile model cache. 0 disables bucketing.",
    )
    p.add_argument(
        "--compile-bucket-edges",
        type=int,
        default=0,
        help="Edge-count bucket size for compile model cache. 0 disables bucketing.",
    )
    p.add_argument("--compute-force", action="store_true", default=True)
    p.add_argument("--no-compute-force", dest="compute_force", action="store_false")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def clone_batch(batch: Dict):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def summarize_ms(times_ms: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "steps_per_s": 1000.0 / statistics.mean(times_ms),
    }


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


def bucket_ceil(value: int, bucket: int) -> int:
    if bucket <= 0:
        return value
    return ((value + bucket - 1) // bucket) * bucket


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


def bench_training(model, base_batch, warmup: int, iters: int, amp_dtype, device: str):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda" and amp_dtype == torch.float16),
    )
    times = []
    use_cuda = cuda_enabled(device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        batch = clone_batch(base_batch)
        batch["positions"].requires_grad_(True)
        start_event = end_event = None
        start_t = 0.0
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp_dtype):
            out = model(batch, training=True)
            loss = out["energy"].sum()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if use_cuda:
            end_event.record()
            sync_if_cuda(device)
        if i >= warmup:
            if use_cuda:
                times.append(start_event.elapsed_time(end_event))
            else:
                times.append((time.perf_counter() - start_t) * 1000.0)
    ret = summarize_ms(times)
    ret["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if use_cuda else 0.0
    )
    return ret


def bench_inference(model, base_batch, warmup: int, iters: int, amp_dtype, device: str):
    return bench_inference_with_mode(
        model=model,
        base_batch=base_batch,
        warmup=warmup,
        iters=iters,
        compute_force=True,
        amp_dtype=amp_dtype,
        device=device,
    )


def bench_inference_with_mode(
    model,
    base_batch,
    warmup: int,
    iters: int,
    compute_force: bool,
    amp_dtype,
    device: str,
):
    times = []
    model.eval()
    use_cuda = cuda_enabled(device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        batch = clone_batch(base_batch)
        batch["positions"].requires_grad_(True)
        start_event = end_event = None
        start_t = 0.0
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_t = time.perf_counter()
        with autocast_ctx(device, amp_dtype):
            _ = model(batch, training=False, compute_force=compute_force)
        if use_cuda:
            end_event.record()
            sync_if_cuda(device)
        if i >= warmup:
            if use_cuda:
                times.append(start_event.elapsed_time(end_event))
            else:
                times.append((time.perf_counter() - start_t) * 1000.0)
    ret = summarize_ms(times)
    ret["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if use_cuda else 0.0
    )
    return ret


def run_case(
    structure_file: str,
    size: int,
    mode: str,
    device: str,
    compile_mode: str,
    warmup_train: int,
    iters_train: int,
    warmup_infer: int,
    iters_infer: int,
    compute_force: bool,
    amp_dtype,
    compile_cache: Dict = None,
    compile_bucket_atoms: int = 0,
    compile_bucket_edges: int = 0,
):
    cutoff = 5.0
    atoms, zs, batch = make_batch(structure_file, size, cutoff, device)
    model_factory = build_model_factory(zs, cutoff)
    compile_bucket = None

    if mode == "eager":
        train_model = model_factory(device=device, enable_cueq=False)
        infer_model = model_factory(device=device, enable_cueq=False)
    else:
        num_atoms = int(len(atoms))
        num_edges = int(batch["edge_index"].shape[1])
        compile_bucket = {
            "atoms": (
                bucket_ceil(num_atoms, compile_bucket_atoms)
                if compile_bucket_atoms > 0
                else None
            ),
            "edges": (
                bucket_ceil(num_edges, compile_bucket_edges)
                if compile_bucket_edges > 0
                else None
            ),
        }
        cache_key = (
            tuple(zs),
            compile_bucket["atoms"],
            compile_bucket["edges"],
            device,
            compile_mode,
        )
        if compile_cache is not None and cache_key in compile_cache:
            entry = compile_cache[cache_key]
            train_model = entry["train_model"]
            infer_model = entry["infer_model"]
            train_model.load_state_dict(entry["train_init"], strict=True)
            infer_model.load_state_dict(entry["infer_init"], strict=True)
        else:
            torch.compiler.reset()
            train_model = torch.compile(
                mace_compile.prepare(model_factory)(device, enable_cueq=False),
                mode=compile_mode,
            )
            torch.compiler.reset()
            infer_model = torch.compile(
                mace_compile.prepare(model_factory)(device, enable_cueq=False),
                mode=compile_mode,
            )
            if compile_cache is not None:
                compile_cache[cache_key] = {
                    "train_model": train_model,
                    "infer_model": infer_model,
                    "train_init": deepcopy(train_model.state_dict()),
                    "infer_init": deepcopy(infer_model.state_dict()),
                }

    train_stats = bench_training(
        train_model,
        batch,
        warmup_train,
        iters_train,
        amp_dtype=amp_dtype,
        device=device,
    )
    infer_compute_force = compute_force
    infer_note = ""
    try:
        infer_stats = bench_inference_with_mode(
            infer_model,
            batch,
            warmup_infer,
            iters_infer,
            compute_force=infer_compute_force,
            amp_dtype=amp_dtype,
            device=device,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if mode == "compile" and compute_force:
            infer_compute_force = False
            infer_note = f"compile+force failed, fallback to energy-only: {type(exc).__name__}"
            infer_stats = bench_inference_with_mode(
                infer_model,
                batch,
                warmup_infer,
                iters_infer,
                compute_force=infer_compute_force,
                amp_dtype=amp_dtype,
                device=device,
            )
        else:
            raise

    return {
        "structure": structure_file,
        "structure_name": Path(structure_file).name,
        "size": size,
        "mode": mode,
        "num_atoms": int(len(atoms)),
        "num_edges": int(batch["edge_index"].shape[1]),
        "train": train_stats,
        "infer": infer_stats,
        "infer_compute_force": infer_compute_force,
        "amp": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
        "compile_bucket": compile_bucket,
        "note": infer_note,
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    amp_dtype = resolve_amp_dtype(args.amp)
    configure_cuda_precision(args.device, args.tf32)
    torch_geometric.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    use_compile_cache = args.compile_bucket_atoms > 0 or args.compile_bucket_edges > 0
    compile_cache: Dict = {} if use_compile_cache else None
    all_rows = []
    t0 = time.time()
    for structure in args.structures:
        for size in args.sizes:
            for mode in ("eager", "compile"):
                row = run_case(
                    structure_file=structure,
                    size=size,
                    mode=mode,
                    device=args.device,
                    compile_mode=args.compile_mode,
                    warmup_train=args.warmup_train,
                    iters_train=args.iters_train,
                    warmup_infer=args.warmup_infer,
                    iters_infer=args.iters_infer,
                    compute_force=args.compute_force,
                    amp_dtype=amp_dtype,
                    compile_cache=compile_cache if mode == "compile" else None,
                    compile_bucket_atoms=args.compile_bucket_atoms,
                    compile_bucket_edges=args.compile_bucket_edges,
                )
                all_rows.append(row)
                print(
                    f"[done] {Path(structure).name} size={size} mode={mode} "
                    f"train={row['train']['mean_ms']:.2f}ms infer={row['infer']['mean_ms']:.2f}ms"
                )

    out = {
        "device": (
            torch.cuda.get_device_name(0)
            if args.device == "cuda" and torch.cuda.is_available()
            else args.device
        ),
        "torch": torch.__version__,
        "cueq_enabled": False,
        "compile_mode": args.compile_mode,
        "tf32": args.tf32,
        "amp": args.amp,
        "compile_bucket_atoms": args.compile_bucket_atoms,
        "compile_bucket_edges": args.compile_bucket_edges,
        "compile_cache_entries": len(compile_cache) if compile_cache is not None else 0,
        "elapsed_s": time.time() - t0,
        "results": all_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "structure_name",
                "size",
                "mode",
                "num_atoms",
                "num_edges",
                "amp",
                "compile_bucket_atoms",
                "compile_bucket_edges",
                "train_mean_ms",
                "train_steps_per_s",
                "train_peak_mem_mb",
                "infer_mean_ms",
                "infer_steps_per_s",
                "infer_peak_mem_mb",
            ]
        )
        for r in all_rows:
            writer.writerow(
                [
                    r["structure_name"],
                    r["size"],
                    r["mode"],
                    r["num_atoms"],
                    r["num_edges"],
                    r["amp"],
                    r["compile_bucket"]["atoms"] if r["compile_bucket"] else "",
                    r["compile_bucket"]["edges"] if r["compile_bucket"] else "",
                    r["train"]["mean_ms"],
                    r["train"]["steps_per_s"],
                    r["train"]["peak_mem_mb"],
                    r["infer"]["mean_ms"],
                    r["infer"]["steps_per_s"],
                    r["infer"]["peak_mem_mb"],
                ]
            )

    print(f"JSON written: {out_json}")
    print(f"CSV written: {out_csv}")


if __name__ == "__main__":
    main()
