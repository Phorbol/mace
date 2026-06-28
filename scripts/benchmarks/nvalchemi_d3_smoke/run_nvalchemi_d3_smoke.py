#!/usr/bin/env python
"""Smoke test the optional nvalchemi D3 backend on CPU and CUDA."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from ase.build import molecule

from mace.calculators.nvalchemi_d3 import NvalchemiDFTD3Calculator


def build_atoms():
    atoms = molecule("H2O")
    atoms.center(vacuum=6.0)
    return atoms


def evaluate(device: str, repeat: int):
    atoms = build_atoms()
    atoms.calc = NvalchemiDFTD3Calculator(device=device, auto_download=False)

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
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
    args = parser.parse_args()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this smoke test on a GPU node")

    cpu = evaluate("cpu", args.repeat)
    cuda = evaluate("cuda", args.repeat)
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
