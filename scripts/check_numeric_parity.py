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
from mace.tools import torch_geometric


def parse_args():
    p = argparse.ArgumentParser(description="Check numerical parity for AMP/TF32 inference.")
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1702)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--tf32", action="store_true", default=False)
    p.add_argument("--compute-force", dest="compute_force", action="store_true", default=True)
    p.add_argument("--no-compute-force", dest="compute_force", action="store_false")
    p.add_argument("--energy-abs-tol", type=float, default=2e-1)
    p.add_argument("--energy-rel-tol", type=float, default=3e-2)
    p.add_argument("--force-abs-tol", type=float, default=5e-3)
    p.add_argument("--force-rel-tol", type=float, default=5e-2)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def setup_cueq_disabled():
    return CuEquivarianceConfig(enabled=False)


def build_model_factory(zs: List[int], cutoff: float):
    atomic_energies = np.zeros((len(zs),), dtype=float)

    def factory(device: str = "cuda"):
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


def rel_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    return (a - b).abs() / denom


def rel_rms_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).float()
    ref = a.float()
    return torch.sqrt(torch.mean(diff * diff)).item() / (
        torch.sqrt(torch.mean(ref * ref)).item() + 1e-12
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch_geometric.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    amp_dtype = resolve_amp_dtype(args.amp)
    all_rows = []
    cutoff = 5.0
    for structure in args.structures:
        for size in args.sizes:
            atoms, zs, batch = make_batch(structure, size, cutoff, args.device)
            model = build_model_factory(zs, cutoff)(args.device)
            model.eval()

            batch_ref = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch_ref["positions"].requires_grad_(True)
            with torch.no_grad():
                out_ref = model(batch_ref, training=False, compute_force=False)
            e_ref = out_ref["energy"].detach()

            batch_opt = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch_opt["positions"].requires_grad_(True)
            with autocast_ctx(args.device, amp_dtype):
                out_opt = model(batch_opt, training=False, compute_force=False)
            e_opt = out_opt["energy"].detach().to(e_ref.dtype)

            energy_abs = (e_ref - e_opt).abs().max().item()
            energy_rel = rel_error(e_ref, e_opt).max().item()
            energy_rel_rms = rel_rms_error(e_ref, e_opt)

            force_abs = 0.0
            force_rel = 0.0
            force_rel_rms = 0.0
            if args.compute_force:
                batch_ref_f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch_ref_f["positions"].requires_grad_(True)
                out_ref_f = model(batch_ref_f, training=False, compute_force=True)
                f_ref = out_ref_f["forces"].detach()

                batch_opt_f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch_opt_f["positions"].requires_grad_(True)
                with autocast_ctx(args.device, amp_dtype):
                    out_opt_f = model(batch_opt_f, training=False, compute_force=True)
                f_opt = out_opt_f["forces"].detach().to(f_ref.dtype)

                force_abs = (f_ref - f_opt).abs().max().item()
                force_rel = rel_error(f_ref, f_opt).max().item()
                force_rel_rms = rel_rms_error(f_ref, f_opt)

            energy_pass = (energy_abs <= args.energy_abs_tol) or (energy_rel_rms <= args.energy_rel_tol)
            force_pass = (force_abs <= args.force_abs_tol) or (force_rel_rms <= args.force_rel_tol)
            row = {
                "structure_name": Path(structure).name,
                "size": size,
                "num_atoms": int(len(atoms)),
                "num_edges": int(batch["edge_index"].shape[1]),
                "amp": args.amp,
                "tf32": args.tf32,
                "energy_abs_max": energy_abs,
                "energy_rel_max": energy_rel,
                "energy_rel_rms": energy_rel_rms,
                "force_abs_max": force_abs,
                "force_rel_max": force_rel,
                "force_rel_rms": force_rel_rms,
                "pass": bool(energy_pass and force_pass),
            }
            all_rows.append(row)
            print(
                f"[done] {row['structure_name']} size={size} "
                f"Eabs={energy_abs:.3e} Erms={energy_rel_rms:.3e} "
                f"Fabs={force_abs:.3e} Frms={force_rel_rms:.3e} pass={row['pass']}"
            )

    result = {
        "amp": args.amp,
        "tf32": args.tf32,
        "energy_abs_tol": args.energy_abs_tol,
        "energy_rel_tol": args.energy_rel_tol,
        "force_abs_tol": args.force_abs_tol,
        "force_rel_tol": args.force_rel_tol,
        "rows": all_rows,
        "all_pass": all(r["pass"] for r in all_rows),
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
                "num_edges",
                "amp",
                "tf32",
                "energy_abs_max",
                "energy_rel_max",
                "energy_rel_rms",
                "force_abs_max",
                "force_rel_max",
                "force_rel_rms",
                "pass",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    print(f"JSON written: {out_json}")
    print(f"CSV written: {out_csv}")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
