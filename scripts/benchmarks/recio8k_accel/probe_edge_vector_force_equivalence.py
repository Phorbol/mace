from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import modules
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scatter import scatter_sum

from scripts.benchmarks.recio8k_accel.probe_training_compile import (  # noqa: E402
    _batch_dict,
    _load_batch,
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


def create_probe_model(
    *,
    z_table,
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


def edge_gradient_to_atomic_forces(
    edge_grad: torch.Tensor, *, edge_index: torch.Tensor, num_atoms: int
) -> torch.Tensor:
    """Scatter dE/d(edge_vec) into atomic forces for v = r_receiver - r_sender."""
    sender = edge_index[0]
    receiver = edge_index[1]
    sender_force = scatter_sum(edge_grad, sender, dim=0, dim_size=num_atoms)
    receiver_force = scatter_sum(edge_grad, receiver, dim=0, dim_size=num_atoms)
    return sender_force - receiver_force


def _is_detach_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target == torch.ops.aten.detach.default


def count_fx_detach_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if _is_detach_node(node))


def strip_fx_saved_tensor_detach(gm: torch.fx.GraphModule) -> None:
    """Remove make_fx saved-tensor detach chains while preserving other nodes."""
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach_node(node):
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach_node(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach_node(user) for user in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)
    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


def rebuild_fx_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    new_graph = torch.fx.Graph()
    value_map: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        value_map[node] = new_graph.node_copy(node, lambda old: value_map[old])
    new_graph.lint()
    return torch.fx.GraphModule(gm, new_graph)


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._orig_mod", "").replace("_orig_mod.", "")


def _force_loss(output: dict, batch) -> torch.Tensor:
    energy = output["energy"]
    target_energy = getattr(batch, "energy", None)
    if target_energy is None:
        loss = energy.square().mean()
    else:
        loss = (energy - target_energy).square().mean()
    target_forces = getattr(batch, "forces", None)
    if target_forces is None:
        return loss + output["forces"].square().mean()
    return loss + (output["forces"] - target_forces).square().mean()


def _named_parameter_grads(model: torch.nn.Module) -> dict[str, Optional[torch.Tensor]]:
    return {
        _canonical_parameter_name(name): (
            None if param.grad is None else param.grad.detach().clone()
        )
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _mace_energy_from_vectors(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    *,
    vectors: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, modules.ScaleShiftMACE):
        raise TypeError("edge-vector probe currently supports ScaleShiftMACE only")

    num_graphs = int(data["ptr"].numel() - 1)
    num_atoms_arange = torch.arange(data["positions"].shape[0], device=vectors.device)
    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    ).to(torch.int64)

    node_e0 = model.atomic_energies_fn(data["node_attrs"])[
        num_atoms_arange.to(torch.int64), node_heads
    ]
    e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs).to(
        vectors.dtype
    )

    node_feats = model.node_embedding(data["node_attrs"])
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )

    if hasattr(model, "pair_repulsion"):
        pair_node_energy = model.pair_repulsion_fn(
            lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
        )
        pair_energy = scatter_sum(
            src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        )
    else:
        pair_node_energy = torch.zeros_like(node_e0)
        pair_energy = torch.zeros_like(e0)

    if hasattr(model, "joint_embedding"):
        embedding_features: dict[str, torch.Tensor] = {}
        for name, _ in model.embedding_specs.items():
            embedding_features[name] = data[name]
        node_feats = node_feats + model.joint_embedding(data["batch"], embedding_features)
        if hasattr(model, "embedding_readout"):
            embedding_node_energy = model.embedding_readout(node_feats, node_heads).squeeze(-1)
            embedding_energy = scatter_sum(
                src=embedding_node_energy,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            e0 = e0 + embedding_energy

    energies = [e0, pair_energy]
    node_energies_list = [node_e0, pair_node_energy]
    node_feats_concat: list[torch.Tensor] = []
    for i, (interaction, product) in enumerate(zip(model.interactions, model.products)):
        node_feats, sc = interaction(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=(i == 0),
        )
        node_feats = product(
            node_feats=node_feats,
            sc=sc,
            node_attrs=data["node_attrs"],
        )
        node_feats_concat.append(node_feats)

    for i, readout in enumerate(model.readouts):
        feat_idx = -1 if len(model.readouts) == 1 else i
        node_es = readout(node_feats_concat[feat_idx], node_heads)[
            num_atoms_arange.to(torch.int64), node_heads
        ]
        energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
        energies.append(energy)
        node_energies_list.append(node_es)

    contributions = torch.stack(energies, dim=-1)
    total_energy = torch.sum(contributions, dim=-1)
    node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
    return {
        "energy": total_energy,
        "node_energy": node_energy,
        "contributions": contributions,
    }


def _position_snapshot(model: torch.nn.Module, batch) -> dict:
    model.zero_grad(set_to_none=True)
    output = model(
        _batch_dict(batch),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    loss = _force_loss(output, batch)
    loss.backward()
    return {
        "energy": output["energy"].detach().clone(),
        "forces": output["forces"].detach().clone(),
        "loss": loss.detach().clone(),
        "grads": _named_parameter_grads(model),
    }


def _edge_vector_snapshot(model: torch.nn.Module, batch) -> dict:
    model.zero_grad(set_to_none=True)
    data = _batch_dict(batch)
    positions = data["positions"].detach()
    edge_index = data["edge_index"]
    vectors, _ = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=data["shifts"].detach(),
    )
    vectors = vectors.detach().clone().requires_grad_(True)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    output = _mace_energy_from_vectors(model, data, vectors=vectors, lengths=lengths)
    edge_grad = torch.autograd.grad(
        outputs=[output["energy"]],
        inputs=[vectors],
        grad_outputs=[torch.ones_like(output["energy"])],
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    output = dict(output)
    output["forces"] = edge_gradient_to_atomic_forces(
        edge_grad,
        edge_index=edge_index,
        num_atoms=positions.shape[0],
    )
    loss = _force_loss(output, batch)
    loss.backward()
    return {
        "energy": output["energy"].detach().clone(),
        "forces": output["forces"].detach().clone(),
        "loss": loss.detach().clone(),
        "grads": _named_parameter_grads(model),
    }


def _edge_vector_inputs(batch) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _batch_dict(batch)
    positions = data["positions"].detach()
    edge_index = data["edge_index"]
    vectors, _ = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=data["shifts"].detach(),
    )
    vectors = vectors.detach().clone().requires_grad_(True)
    return data, positions, edge_index, vectors


def _edge_force_loss_outputs(
    model: torch.nn.Module,
    batch,
    data: dict[str, torch.Tensor],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    output = _mace_energy_from_vectors(model, data, vectors=vectors, lengths=lengths)
    edge_grad = torch.autograd.grad(
        outputs=[output["energy"]],
        inputs=[vectors],
        grad_outputs=[torch.ones_like(output["energy"])],
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    forces = edge_gradient_to_atomic_forces(
        edge_grad, edge_index=edge_index, num_atoms=positions.shape[0]
    )
    output = dict(output)
    output["forces"] = forces
    loss = _force_loss(output, batch)
    return output["energy"], forces, loss


def _make_fx_edge_vector_snapshot(
    model: torch.nn.Module,
    batch,
    *,
    tracing_mode: str,
    strip_detach: bool,
) -> dict:
    from torch.fx.experimental.proxy_tensor import make_fx

    model.zero_grad(set_to_none=True)
    data, positions, edge_index, vectors = _edge_vector_inputs(batch)

    def fn(vectors_arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _edge_force_loss_outputs(
            model, batch, data, positions, edge_index, vectors_arg
        )

    traced = make_fx(fn, tracing_mode=tracing_mode)(vectors)
    detach_nodes_before = count_fx_detach_nodes(traced)
    if strip_detach:
        strip_fx_saved_tensor_detach(traced)
        traced = rebuild_fx_graph_module(traced)
    detach_nodes_after = count_fx_detach_nodes(traced)
    energy, forces, loss = traced(vectors)
    loss.backward()
    return {
        "status": "ok",
        "tracing_mode": tracing_mode,
        "strip_detach": strip_detach,
        "node_count": len(list(traced.graph.nodes)),
        "detach_nodes_before": detach_nodes_before,
        "detach_nodes_after": detach_nodes_after,
        "snapshot": {
            "energy": energy.detach().clone(),
            "forces": forces.detach().clone(),
            "loss": loss.detach().clone(),
            "grads": _named_parameter_grads(model),
        },
    }


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


def compare_snapshots(left: dict, right: dict, *, atol: float, rtol: float) -> dict:
    failed_checks: list[str] = []
    for key in ("energy", "forces", "loss"):
        if not _within_tolerance(left[key], right[key], atol=atol, rtol=rtol):
            failed_checks.append(key)

    grad_diffs: dict[str, float] = {}
    left_names = set(left["grads"])
    right_names = set(right["grads"])
    for missing_name in sorted(left_names ^ right_names):
        grad_diffs[missing_name] = float("inf")
        failed_checks.append(f"grad:{missing_name}")
    for name in sorted(left_names & right_names):
        left_grad = left["grads"][name]
        right_grad = right["grads"][name]
        if left_grad is None and right_grad is None:
            grad_diffs[name] = 0.0
            continue
        if left_grad is None or right_grad is None:
            grad_diffs[name] = float("inf")
            failed_checks.append(f"grad:{name}")
            continue
        grad_diffs[name] = _max_abs_diff(left_grad, right_grad)
        if not _within_tolerance(left_grad, right_grad, atol=atol, rtol=rtol):
            failed_checks.append(f"grad:{name}")

    return {
        "ok": not failed_checks,
        "atol": atol,
        "rtol": rtol,
        "energy_max_abs_diff": _max_abs_diff(left["energy"], right["energy"]),
        "forces_max_abs_diff": _max_abs_diff(left["forces"], right["forces"]),
        "loss_abs_diff": _max_abs_diff(left["loss"], right["loss"]),
        "param_grad_max_abs_diff": grad_diffs,
        "failed_checks": failed_checks,
    }


def _summarize_snapshot(snapshot: dict) -> dict:
    grad_norms = {
        name: None if grad is None else float(grad.norm().cpu())
        for name, grad in snapshot["grads"].items()
    }
    return {
        "energy": snapshot["energy"].detach().cpu().tolist(),
        "forces_norm": float(snapshot["forces"].norm().cpu()),
        "loss": float(snapshot["loss"].cpu()),
        "param_grad_norm": grad_norms,
    }


def run_probe(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(Path(args.xyz), indices, cutoff=args.cutoff, device=device)
    model = create_probe_model(
        z_table=z_table,
        cutoff=args.cutoff,
        device=device,
        hidden_channels=args.hidden_channels,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        enable_cueq=args.enable_cueq,
    )
    position = _position_snapshot(model, batch)
    edge = _edge_vector_snapshot(model, batch)
    comparison = compare_snapshots(position, edge, atol=args.atol, rtol=args.rtol)
    make_fx_result = None
    if args.make_fx:
        try:
            make_fx_result = _make_fx_edge_vector_snapshot(
                model,
                batch,
                tracing_mode=args.make_fx_tracing_mode,
                strip_detach=args.strip_make_fx_detach,
            )
            make_fx_result["comparison"] = compare_snapshots(
                position,
                make_fx_result["snapshot"],
                atol=args.atol,
                rtol=args.rtol,
            )
            make_fx_result["snapshot"] = _summarize_snapshot(make_fx_result["snapshot"])
        except Exception as exc:  # pylint: disable=broad-except
            make_fx_result = {
                "status": "error",
                "tracing_mode": args.make_fx_tracing_mode,
                "strip_detach": args.strip_make_fx_detach,
                "error": repr(exc),
            }
    payload = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "enable_cueq": args.enable_cueq,
        "xyz": args.xyz,
        "indices": indices,
        "num_atoms": int(batch.num_nodes),
        "atomic_numbers": z_table.zs,
        "model": {
            "hidden_channels": args.hidden_channels,
            "max_ell": args.max_ell,
            "num_interactions": args.num_interactions,
            "correlation": args.correlation,
        },
        "position_snapshot": _summarize_snapshot(position),
        "edge_vector_snapshot": _summarize_snapshot(edge),
        "comparison": comparison,
    }
    if make_fx_result is not None:
        payload["make_fx_edge_vector"] = make_fx_result
    if device.type == "cuda":
        payload["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xyz",
        default="/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz",
    )
    parser.add_argument("--indices", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--max-ell", type=int, default=1)
    parser.add_argument("--num-interactions", type=int, default=1)
    parser.add_argument("--correlation", type=int, default=1)
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--make-fx", action="store_true")
    parser.add_argument(
        "--make-fx-tracing-mode",
        choices=["real", "fake", "symbolic"],
        default="real",
    )
    parser.add_argument("--strip-make-fx-detach", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
