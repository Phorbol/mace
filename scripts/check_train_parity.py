import argparse
import csv
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
import torch.nn.functional as F
from ase import io
from e3nn import o3

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import torch_geometric


def parse_args():
    p = argparse.ArgumentParser(
        description="Strict training parity check between baseline and optimized configs."
    )
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--candidate-mode", choices=["eager", "compile"], default="compile")
    p.add_argument("--candidate-compile-mode", default="default")
    p.add_argument("--candidate-tf32", action="store_true", default=False)
    p.add_argument("--candidate-amp", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--loss-rel-tol", type=float, default=5e-2)
    p.add_argument("--param-rel-tol", type=float, default=5e-2)
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


def autocast_ctx(device: str, amp_dtype):
    if device == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def configure_cuda_precision(device: str, tf32: bool):
    if device != "cuda":
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


def make_batch(structure_file, size, cutoff, device):
    atoms = io.read(structure_file).repeat((size, size, size))
    zs = sorted({int(z) for z in atoms.numbers})
    table = tools.AtomicNumberTable(zs)
    conf = data.config_from_atoms(atoms)
    dataset = [data.AtomicData.from_config(conf, z_table=table, cutoff=cutoff)]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device).to_dict()
    return atoms, zs, batch


def clone_batch(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def l2_rel_diff(a: torch.Tensor, b: torch.Tensor):
    num = torch.norm((a - b).float())
    den = torch.norm(a.float()) + 1e-12
    return (num / den).item()


def model_param_rel_diff(model_a, model_b):
    diffs = []
    for (name_a, pa), (name_b, pb) in zip(
        model_a.state_dict().items(), model_b.state_dict().items()
    ):
        if name_a != name_b:
            raise RuntimeError("State dict keys mismatch")
        if not torch.is_floating_point(pa):
            continue
        diffs.append(l2_rel_diff(pa, pb))
    return float(max(diffs) if diffs else 0.0)


def run_steps(
    model,
    batch,
    steps,
    lr,
    device,
    amp_dtype=None,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device == "cuda" and amp_dtype == torch.float16)
    )
    losses = []
    for _ in range(steps):
        b = clone_batch(batch)
        b["positions"].requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp_dtype):
            out = model(b, training=True, compute_force=True)
            loss = out["energy"].pow(2).mean() + 0.01 * out["forces"].pow(2).mean()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_geometric.seed_everything(args.seed)

    rows = []
    cutoff = 5.0
    for structure in args.structures:
        for size in args.sizes:
            atoms, zs, batch = make_batch(structure, size, cutoff, args.device)
            base_model = build_model(zs=zs, cutoff=cutoff, device=args.device)
            candidate_model = build_model(zs=zs, cutoff=cutoff, device=args.device)
            candidate_model.load_state_dict(base_model.state_dict(), strict=True)

            configure_cuda_precision(args.device, tf32=False)
            base_losses = run_steps(
                model=base_model,
                batch=batch,
                steps=args.steps,
                lr=args.lr,
                device=args.device,
                amp_dtype=None,
            )

            configure_cuda_precision(args.device, tf32=args.candidate_tf32)
            if args.candidate_mode == "compile":
                candidate_model = torch.compile(
                    candidate_model,
                    mode=args.candidate_compile_mode,
                )
            cand_losses = run_steps(
                model=candidate_model,
                batch=batch,
                steps=args.steps,
                lr=args.lr,
                device=args.device,
                amp_dtype=resolve_amp_dtype(args.candidate_amp),
            )

            if hasattr(candidate_model, "_orig_mod"):
                candidate_plain = candidate_model._orig_mod  # pylint: disable=protected-access
            else:
                candidate_plain = candidate_model
            loss_rel = abs(base_losses[-1] - cand_losses[-1]) / (abs(base_losses[-1]) + 1e-12)
            param_rel = model_param_rel_diff(base_model, candidate_plain)
            row = {
                "structure_name": Path(structure).name,
                "size": int(size),
                "num_atoms": int(len(atoms)),
                "steps": args.steps,
                "baseline_final_loss": base_losses[-1],
                "candidate_final_loss": cand_losses[-1],
                "final_loss_rel_diff": loss_rel,
                "max_loss_abs_diff": max(
                    abs(a - b) for a, b in zip(base_losses, cand_losses)
                ),
                "param_rel_diff_max": param_rel,
                "pass": bool(
                    loss_rel <= args.loss_rel_tol and param_rel <= args.param_rel_tol
                ),
            }
            rows.append(row)
            print(
                f"[done] {row['structure_name']} size={size} "
                f"loss_rel={row['final_loss_rel_diff']:.3e} "
                f"param_rel={row['param_rel_diff_max']:.3e} pass={row['pass']}"
            )

    result = {
        "candidate_mode": args.candidate_mode,
        "candidate_compile_mode": args.candidate_compile_mode,
        "candidate_tf32": args.candidate_tf32,
        "candidate_amp": args.candidate_amp,
        "loss_rel_tol": args.loss_rel_tol,
        "param_rel_tol": args.param_rel_tol,
        "rows": rows,
        "all_pass": all(r["pass"] for r in rows),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "structure_name",
                "size",
                "num_atoms",
                "steps",
                "baseline_final_loss",
                "candidate_final_loss",
                "final_loss_rel_diff",
                "max_loss_abs_diff",
                "param_rel_diff_max",
                "pass",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"JSON written: {out_json}")
    print(f"CSV written: {out_csv}")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
