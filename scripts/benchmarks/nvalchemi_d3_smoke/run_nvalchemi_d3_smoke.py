#!/usr/bin/env python
"""Smoke test the optional nvalchemi D3 backend on CPU and CUDA."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import molecule
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace.calculators.nvalchemi_d3 import NvalchemiDFTD3Calculator


def build_atoms(xyz: str | None, index: int):
    if xyz:
        return read(xyz, index=index)
    atoms = molecule("H2O")
    atoms.center(vacuum=6.0)
    return atoms


def evaluate(device: str, repeat: int, xyz: str | None, index: int):
    atoms = build_atoms(xyz=xyz, index=index)
    atoms.calc = NvalchemiDFTD3Calculator(device=device, auto_download=False)

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        if atoms.calc is not None:
            atoms.calc.results.clear()
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "energy": energy,
        "forces": forces,
        "seconds_per_eval": elapsed / max(repeat, 1),
    }


def assert_finite(label: str, result: dict):
    energy = result["energy"]
    forces = result["forces"]
    if not math.isfinite(energy):
        raise RuntimeError(f"{label} energy is not finite: {energy}")
    if not np.isfinite(forces).all():
        raise RuntimeError(f"{label} forces contain non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--xyz", default=None, help="Optional ASE-readable structure file")
    parser.add_argument("--index", type=int, default=0, help="Structure index when --xyz is set")
    args = parser.parse_args()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this smoke test on a GPU node")

    atoms = build_atoms(xyz=args.xyz, index=args.index)
    print(
        "structure="
        f"natoms={len(atoms)} formula={atoms.get_chemical_formula()} "
        f"pbc={atoms.pbc.tolist()} cell_lengths={atoms.cell.lengths().tolist()}"
    )

    cpu = evaluate("cpu", args.repeat, args.xyz, args.index)
    cuda = evaluate("cuda", args.repeat, args.xyz, args.index)
    assert_finite("cpu", cpu)
    assert_finite("cuda", cuda)

    energy_abs_diff = abs(cpu["energy"] - cuda["energy"])
    force_max_abs_diff = float(np.max(np.abs(cpu["forces"] - cuda["forces"])))
    force_ref = max(float(np.max(np.abs(cpu["forces"]))), 1.0)

    energy_ok = energy_abs_diff <= args.atol + args.rtol * max(abs(cpu["energy"]), 1.0)
    force_ok = force_max_abs_diff <= args.atol + args.rtol * force_ref
    if not energy_ok or not force_ok:
        raise RuntimeError(
            "CPU/CUDA D3 mismatch: "
            f"energy_abs_diff={energy_abs_diff:.6e}, force_max_abs_diff={force_max_abs_diff:.6e}"
        )

    print(f"cpu_energy_eV={cpu['energy']:.12f}")
    print(f"cuda_energy_eV={cuda['energy']:.12f}")
    print(f"energy_abs_diff={energy_abs_diff:.6e}")
    print(f"force_max_abs_diff={force_max_abs_diff:.6e}")
    print(f"cpu_seconds_per_eval={cpu['seconds_per_eval']:.6e}")
    print(f"cuda_seconds_per_eval={cuda['seconds_per_eval']:.6e}")


if __name__ == "__main__":
    main()
