from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

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
from mace.tools.force_compile import compile_fx_graph_module, trace_force_closure
from mace.tools.hybrid_muon import (
    HybridMuon,
    build_hybrid_muon_param_groups,
    summarize_hybrid_muon_routes,
)
from mace.tools.training_compile import edge_force_compile_result_from_trace

from scripts.benchmarks.recio8k_accel.profile_edge_force_training_steps import (  # noqa: E402
    MODE_CHOICES,
    OPTIMIZER_CHOICES,
    _edge_force_loss_outputs,
    _edge_snapshot,
    _edge_vector_inputs,
    _force_loss_fn,
    _position_snapshot,
    _sync,
    compare_snapshots,
    parse_csv_choices,
    parse_indices,
)
from scripts.benchmarks.recio8k_accel.probe_training_compile import _batch_dict  # noqa: E402


@dataclass
class CachedStep:
    executable: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    gate_result: dict | None
    setup_ms: float
    cache_key: tuple[int, ...]


def split_index_batches(
    indices: list[int], *, batch_size: int, max_batches: int | None = None
) -> list[tuple[int, ...]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = [
        tuple(indices[i : i + batch_size])
        for i in range(0, len(indices), batch_size)
    ]
    if max_batches is not None:
        batches = batches[:max_batches]
    if not batches:
        raise ValueError("at least one batch is required")
    return batches


def compile_cache_key(
    batch_indices: tuple[int, ...], *, num_atoms: int, num_edges: int
) -> tuple[int, ...]:
    # Current edge-compile closures capture non-vector batch tensors and labels.
    # Shape-only reuse across different batches would be incorrect until those
    # tensors are promoted to explicit FX graph inputs.
    return tuple(batch_indices) + (int(num_atoms), int(num_edges))


def summarize_steps(steps: list[dict]) -> dict:
    if not steps:
        raise ValueError("cannot summarize an empty step list")
    total = [float(step["total_ms"]) for step in steps]
    setup = [float(step.get("setup_ms", 0.0)) for step in steps]
    including_setup = [t + s for t, s in zip(total, setup, strict=True)]
    cache_hits = sum(1 for step in steps if step.get("cache_hit"))
    compile_setups = sum(1 for value in setup if value > 0.0)
    return {
        "steps": len(steps),
        "cache_hits": cache_hits,
        "compile_setups": compile_setups,
        "setup_ms_total": sum(setup),
        "mean_total_ms_excluding_setup": mean(total),
        "median_total_ms_excluding_setup": median(total),
        "mean_total_ms_including_setup": mean(including_setup),
        "median_total_ms_including_setup": median(including_setup),
        "loss_first": steps[0].get("loss"),
        "loss_last": steps[-1].get("loss"),
    }


def _cueq_config(args: argparse.Namespace, device: torch.device):
    if not args.enable_cueq:
        return None
    conv_fusion = args.cueq_conv_fusion
    if conv_fusion is None:
        conv_fusion = device.type == "cuda"
    return CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=args.cueq_optimize_all,
        optimize_linear=args.cueq_optimize_linear,
        optimize_channelwise=args.cueq_optimize_channelwise,
        optimize_symmetric=args.cueq_optimize_symmetric,
        optimize_fctp=args.cueq_optimize_fctp,
        conv_fusion=conv_fusion,
    )


def _load_batches(
    xyz: Path,
    index_batches: list[tuple[int, ...]],
    *,
    cutoff: float,
    device: torch.device,
):
    all_indices = [index for batch in index_batches for index in batch]
    atoms_by_index = {index: read(xyz, index=index) for index in all_indices}
    atomic_numbers = sorted(
        {int(z) for atoms in atoms_by_index.values() for z in atoms.numbers}
    )
    z_table = tools.AtomicNumberTable(atomic_numbers)
    batches = []
    for batch_indices in index_batches:
        configs = [data.config_from_atoms(atoms_by_index[index]) for index in batch_indices]
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
        batches.append(next(iter(loader)).to(device))
    return batches, z_table


def _create_model(args: argparse.Namespace, z_table, device: torch.device):
    model_config = {
        "r_max": args.cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": args.max_ell,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": args.num_interactions,
        "num_elements": len(z_table.zs),
        "hidden_irreps": o3.Irreps(
            f"{args.hidden_channels}x0e + {args.hidden_channels}x1o"
        ),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": np.zeros(len(z_table.zs), dtype=float),
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": args.correlation,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": _cueq_config(args, device),
    }
    return modules.ScaleShiftMACE(**model_config).to(device)


def _build_optimizer(args: argparse.Namespace, optimizer_name: str, model):
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        ), None
    if optimizer_name == "hybrid_muon":
        groups, summary = build_hybrid_muon_param_groups(
            model.named_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            muon_weight_decay=0.0,
            muon_lr_factor=args.hybrid_muon_lr_factor,
            beta=0.9,
            adam_betas=(0.9, 0.999),
            eps=1.0e-8,
            amsgrad=False,
        )
        return HybridMuon(groups, lr=args.lr, weight_decay=args.weight_decay), summary
    raise ValueError(f"unknown optimizer {optimizer_name!r}")


def _make_edge_compile_step(
    *,
    model,
    batch,
    batch_indices: tuple[int, ...],
    loss_fn,
    args: argparse.Namespace,
) -> CachedStep:
    setup_start = time.perf_counter()
    data_dict, positions, edge_index, vectors = _edge_vector_inputs(batch)

    def closure(vectors_arg: torch.Tensor):
        return _edge_force_loss_outputs(
            model, batch, data_dict, positions, edge_index, vectors_arg, loss_fn
        )

    trace_result = trace_force_closure(
        closure,
        (vectors,),
        tracing_mode=args.edge_tracing_mode,
        strip_detach=args.edge_strip_detach,
    )
    executable, compile_kwargs = compile_fx_graph_module(
        trace_result.graph_module,
        compile_graph=args.edge_compile_graph,
        compile_mode=args.edge_compile_mode,
        compile_dynamic=args.edge_compile_dynamic,
    )
    reference = _position_snapshot(model, batch, loss_fn)
    candidate = _edge_snapshot(model, batch, loss_fn, executable=executable)
    comparison = compare_snapshots(reference, candidate, atol=args.atol, rtol=args.rtol)
    gate_result = edge_force_compile_result_from_trace(
        trace_result=trace_result,
        comparison=comparison,
        compile_kwargs=compile_kwargs,
    ).__dict__
    if not gate_result["accepted"] and not args.allow_gate_failure:
        raise RuntimeError(f"edge compile gate failed for batch {batch_indices}: {comparison}")
    return CachedStep(
        executable=executable,
        gate_result=gate_result,
        setup_ms=(time.perf_counter() - setup_start) * 1.0e3,
        cache_key=compile_cache_key(
            batch_indices,
            num_atoms=batch.positions.shape[0],
            num_edges=edge_index.shape[1],
        ),
    )


def _loss_for_mode(
    *,
    mode_name: str,
    model,
    batch,
    batch_indices: tuple[int, ...],
    loss_fn,
    args: argparse.Namespace,
    cache: dict[tuple[int, ...], CachedStep],
) -> tuple[torch.Tensor, dict]:
    if mode_name == "position_eager":
        output = model(
            _batch_dict(batch),
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        return loss_fn(pred=output, ref=batch), {"setup_ms": 0.0, "cache_hit": False}

    if mode_name == "edge_eager":
        data_dict, positions, edge_index, vectors = _edge_vector_inputs(batch)
        vectors = vectors.detach().clone().requires_grad_(True)
        _, _, loss = _edge_force_loss_outputs(
            model, batch, data_dict, positions, edge_index, vectors, loss_fn
        )
        return loss, {"setup_ms": 0.0, "cache_hit": False}

    if mode_name == "edge_compile":
        key = compile_cache_key(
            batch_indices,
            num_atoms=batch.positions.shape[0],
            num_edges=batch.edge_index.shape[1],
        )
        cached = cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            cached = _make_edge_compile_step(
                model=model,
                batch=batch,
                batch_indices=batch_indices,
                loss_fn=loss_fn,
                args=args,
            )
            cache[key] = cached
        _, _, _, vectors = _edge_vector_inputs(batch)
        vectors = vectors.detach().clone().requires_grad_(True)
        _, _, loss = cached.executable(vectors)
        return loss, {
            "setup_ms": 0.0 if cache_hit else cached.setup_ms,
            "cache_hit": cache_hit,
            "gate_result": cached.gate_result,
            "cache_key": list(cached.cache_key),
        }

    raise ValueError(f"unknown mode {mode_name!r}")


def run_case(
    *,
    optimizer_name: str,
    mode_name: str,
    batches,
    index_batches: list[tuple[int, ...]],
    z_table,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    model = _create_model(args, z_table, device)
    optimizer, route_summary = _build_optimizer(args, optimizer_name, model)
    loss_fn = _force_loss_fn(args)
    cache: dict[tuple[int, ...], CachedStep] = {}
    steps: list[dict] = []

    for epoch in range(args.epochs):
        for batch_id, (batch, batch_indices) in enumerate(zip(batches, index_batches, strict=True)):
            _sync(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, metadata = _loss_for_mode(
                mode_name=mode_name,
                model=model,
                batch=batch,
                batch_indices=batch_indices,
                loss_fn=loss_fn,
                args=args,
                cache=cache,
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            _sync(device)
            elapsed_ms = (time.perf_counter() - start) * 1.0e3
            setup_ms = float(metadata.get("setup_ms", 0.0))
            step = {
                "epoch": epoch,
                "batch_id": batch_id,
                "batch_indices": list(batch_indices),
                "num_atoms": int(batch.positions.shape[0]),
                "num_edges": int(batch.edge_index.shape[1]),
                "total_ms": max(0.0, elapsed_ms - setup_ms),
                "elapsed_ms_including_setup": elapsed_ms,
                "loss": float(loss.detach().cpu()),
                **metadata,
            }
            steps.append(step)
            print(
                f"[edge-force-epoch] optimizer={optimizer_name} mode={mode_name} "
                f"epoch={epoch} batch={batch_id} total_ms={step['total_ms']:.3f} "
                f"setup_ms={step.get('setup_ms', 0.0):.3f}",
                file=sys.stderr,
                flush=True,
            )

    payload = {
        "optimizer": optimizer_name,
        "mode": mode_name,
        "summary": summarize_steps(steps),
        "steps": steps,
        "compile_cache_scope": "batch_identity_captured_tensors",
        "route_summary": route_summary,
    }
    if route_summary is not None:
        payload["route_text"] = summarize_hybrid_muon_routes(route_summary)
    return payload


def _append_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true")
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=Path("train.xyz"))
    parser.add_argument("--indices", default="0:64")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("edge_force_training_epochs.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizers", default="adam,hybrid_muon")
    parser.add_argument("--modes", default="position_eager,edge_compile")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--edge-tracing-mode", default="real")
    parser.add_argument("--edge-compile-mode", default="default")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hybrid-muon-lr-factor", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=1000.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    _append_bool_flag(parser, "enable-cueq", True)
    parser.add_argument("--cueq-conv-fusion", action="store_true", default=None)
    _append_bool_flag(parser, "cueq-optimize-all", False)
    _append_bool_flag(parser, "cueq-optimize-linear", False)
    _append_bool_flag(parser, "cueq-optimize-channelwise", True)
    _append_bool_flag(parser, "cueq-optimize-symmetric", True)
    _append_bool_flag(parser, "cueq-optimize-fctp", True)
    _append_bool_flag(parser, "edge-strip-detach", True)
    _append_bool_flag(parser, "edge-compile-graph", True)
    _append_bool_flag(parser, "edge-compile-dynamic", True)
    _append_bool_flag(parser, "allow-gate-failure", False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    optimizers = parse_csv_choices(args.optimizers, OPTIMIZER_CHOICES)
    modes = parse_csv_choices(args.modes, MODE_CHOICES)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    index_batches = split_index_batches(
        parse_indices(args.indices),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    batches, z_table = _load_batches(
        args.xyz,
        index_batches,
        cutoff=args.cutoff,
        device=device,
    )
    results = [
        run_case(
            optimizer_name=optimizer_name,
            mode_name=mode_name,
            batches=batches,
            index_batches=index_batches,
            z_table=z_table,
            args=args,
            device=device,
        )
        for optimizer_name in optimizers
        for mode_name in modes
    ]
    payload = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "indices": [list(batch) for batch in index_batches],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": {
            "hidden_channels": args.hidden_channels,
            "max_ell": args.max_ell,
            "num_interactions": args.num_interactions,
            "correlation": args.correlation,
        },
        "cueq": {
            "enabled": args.enable_cueq,
            "conv_fusion": args.cueq_conv_fusion,
            "optimize_all": args.cueq_optimize_all,
            "optimize_linear": args.cueq_optimize_linear,
            "optimize_channelwise": args.cueq_optimize_channelwise,
            "optimize_symmetric": args.cueq_optimize_symmetric,
            "optimize_fctp": args.cueq_optimize_fctp,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
