#!/usr/bin/env python
"""Benchmark optional nvalchemi D3 on one or more ASE structures."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch
from ase import units
from ase.atoms import Atoms
from ase.build import molecule
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace.calculators.nvalchemi_d3 import NvalchemiDFTD3Calculator


def parse_structure_indices(
    value: str | None,
    *,
    total: int,
    max_structures: int | None = None,
) -> list[int]:
    if total <= 0:
        raise ValueError("total must be positive")

    if value is None or value.strip() == "":
        indices = list(range(total))
    elif ":" in value:
        parts = value.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid slice index expression: {value!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else total
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        if step == 0:
            raise ValueError("slice step must not be zero")
        indices = list(range(start, stop, step))
    else:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]

    if max_structures is not None:
        if max_structures <= 0:
            raise ValueError("max_structures must be positive")
        indices = indices[:max_structures]

    if not indices:
        raise ValueError("selected indices must include at least one structure")

    out_of_range = [idx for idx in indices if idx < 0 or idx >= total]
    if out_of_range:
        raise ValueError(
            f"structure indices out of range for total={total}: {out_of_range}"
        )
    return indices


def parse_supercell(value: str | None) -> tuple[int, int, int]:
    if value is None or value.strip() == "":
        return (1, 1, 1)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("supercell must have three comma-separated integers")
    repeat = tuple(int(part) for part in parts)
    if any(item <= 0 for item in repeat):
        raise ValueError("supercell repeat values must be positive")
    return repeat


def apply_supercell(atoms: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    if repeat == (1, 1, 1):
        return atoms.copy()
    return atoms.repeat(repeat)


def _mean_key(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return mean(values)


def _max_key(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return max(values)


def summarize_records(
    records: list[dict[str, Any]],
    *,
    torch_dftd_available: bool,
) -> dict[str, Any]:
    cpu_mean = _mean_key(records, "nvalchemi_cpu_seconds_per_eval")
    cuda_mean = _mean_key(records, "nvalchemi_cuda_seconds_per_eval")
    torch_mean = _mean_key(records, "torch_dftd_seconds_per_eval")

    summary: dict[str, Any] = {
        "num_structures": len(records),
        "total_atoms": sum(int(record.get("natoms", 0)) for record in records),
        "max_energy_abs_diff_eV": _max_key(records, "energy_abs_diff_eV"),
        "max_force_abs_diff_eV_A": _max_key(records, "force_max_abs_diff_eV_A"),
        "mean_nvalchemi_cpu_seconds_per_eval": cpu_mean,
        "mean_nvalchemi_cuda_seconds_per_eval": cuda_mean,
        "mean_nvalchemi_cuda_speedup_vs_cpu": (
            cpu_mean / cuda_mean if cpu_mean is not None and cuda_mean else None
        ),
        "torch_dftd_comparison": "available" if torch_dftd_available else "unavailable",
    }
    if torch_dftd_available:
        summary.update(
            {
                "mean_torch_dftd_seconds_per_eval": torch_mean,
                "mean_nvalchemi_cuda_speedup_vs_torch_dftd": (
                    torch_mean / cuda_mean if torch_mean is not None and cuda_mean else None
                ),
                "max_torch_dftd_energy_abs_diff_eV": _max_key(
                    records, "torch_dftd_energy_abs_diff_eV"
                ),
                "max_torch_dftd_force_abs_diff_eV_A": _max_key(
                    records, "torch_dftd_force_max_abs_diff_eV_A"
                ),
            }
        )
    return summary


def build_json_payload(
    *,
    records: list[dict[str, Any]],
    torch_dftd_available: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "summary": summarize_records(
            records, torch_dftd_available=torch_dftd_available
        ),
        "records": records,
    }


def _assert_finite(label: str, energy: float, forces: np.ndarray) -> None:
    if not math.isfinite(energy):
        raise RuntimeError(f"{label} energy is not finite: {energy}")
    if not np.isfinite(forces).all():
        raise RuntimeError(f"{label} forces contain non-finite values")


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def evaluate_atoms(
    atoms: Atoms,
    *,
    calculator_factory: Callable[[], Any],
    repeat: int,
    device: str,
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    working = atoms.copy()
    working.calc = calculator_factory()

    energy = float(working.get_potential_energy())
    forces = np.asarray(working.get_forces(), dtype=np.float64)
    _assert_finite(device, energy, forces)

    _sync_if_cuda(device)
    start = time.perf_counter()
    for _ in range(repeat):
        if working.calc is not None:
            working.calc.results.clear()
        energy = float(working.get_potential_energy())
        forces = np.asarray(working.get_forces(), dtype=np.float64)
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    return {
        "energy_eV": energy,
        "forces_eV_A": forces,
        "seconds_per_eval": elapsed / repeat,
    }


def load_structures(xyz: str | None) -> list[Atoms]:
    if xyz is None:
        atoms = molecule("H2O")
        atoms.center(vacuum=6.0)
        return [atoms]
    loaded = read(xyz, index=":")
    if isinstance(loaded, Atoms):
        return [loaded]
    return list(loaded)


def torch_dftd_factory(
    *, device: str, xc: str, damping: str, cutoff: float
) -> Callable[[], Any] | None:
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError:
        return None

    def factory():
        return TorchDFTD3Calculator(
            device=device,
            damping=damping,
            xc=xc,
            cutoff=cutoff,
        )

    return factory


def benchmark_structure(
    atoms: Atoms,
    *,
    index: int,
    repeat: int,
    xc: str,
    damping: str,
    cutoff: float,
    compare_torch_dftd: bool,
) -> tuple[dict[str, Any], bool]:
    cpu = evaluate_atoms(
        atoms,
        calculator_factory=lambda: NvalchemiDFTD3Calculator(
            device="cpu", xc=xc, damping=damping, cutoff=cutoff, auto_download=False
        ),
        repeat=repeat,
        device="cpu",
    )
    cuda = evaluate_atoms(
        atoms,
        calculator_factory=lambda: NvalchemiDFTD3Calculator(
            device="cuda", xc=xc, damping=damping, cutoff=cutoff, auto_download=False
        ),
        repeat=repeat,
        device="cuda",
    )

    energy_abs_diff = abs(cpu["energy_eV"] - cuda["energy_eV"])
    force_max_abs_diff = float(
        np.max(np.abs(cpu["forces_eV_A"] - cuda["forces_eV_A"]))
    )
    record: dict[str, Any] = {
        "index": index,
        "natoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "pbc": atoms.pbc.tolist(),
        "energy_abs_diff_eV": energy_abs_diff,
        "force_max_abs_diff_eV_A": force_max_abs_diff,
        "nvalchemi_cpu_energy_eV": cpu["energy_eV"],
        "nvalchemi_cuda_energy_eV": cuda["energy_eV"],
        "nvalchemi_cpu_seconds_per_eval": cpu["seconds_per_eval"],
        "nvalchemi_cuda_seconds_per_eval": cuda["seconds_per_eval"],
    }

    torch_available = False
    if compare_torch_dftd:
        factory = torch_dftd_factory(
            device="cuda", xc=xc, damping=damping, cutoff=cutoff
        )
        if factory is not None:
            torch_available = True
            torch_result = evaluate_atoms(
                atoms,
                calculator_factory=factory,
                repeat=repeat,
                device="cuda",
            )
            record.update(
                {
                    "torch_dftd_energy_eV": torch_result["energy_eV"],
                    "torch_dftd_seconds_per_eval": torch_result["seconds_per_eval"],
                    "torch_dftd_energy_abs_diff_eV": abs(
                        cuda["energy_eV"] - torch_result["energy_eV"]
                    ),
                    "torch_dftd_force_max_abs_diff_eV_A": float(
                        np.max(
                            np.abs(
                                cuda["forces_eV_A"] - torch_result["forces_eV_A"]
                            )
                        )
                    ),
                }
            )
    return record, torch_available


def enforce_tolerances(
    records: list[dict[str, Any]], *, rtol: float, atol: float
) -> None:
    for record in records:
        energy_ref = max(abs(float(record["nvalchemi_cpu_energy_eV"])), 1.0)
        if record["energy_abs_diff_eV"] > atol + rtol * energy_ref:
            raise RuntimeError(
                "CPU/CUDA D3 energy mismatch for "
                f"index={record['index']}: {record['energy_abs_diff_eV']:.6e}"
            )
        force_ref = 1.0
        if record["force_max_abs_diff_eV_A"] > atol + rtol * force_ref:
            raise RuntimeError(
                "CPU/CUDA D3 force mismatch for "
                f"index={record['index']}: {record['force_max_abs_diff_eV_A']:.6e}"
            )


def print_text_report(payload: dict[str, Any]) -> None:
    print("metadata=" + json.dumps(payload["metadata"], sort_keys=True))
    print("summary=" + json.dumps(payload["summary"], sort_keys=True))
    for record in payload["records"]:
        print("record=" + json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default=None, help="Optional ASE-readable structure file")
    parser.add_argument("--indices", default=None, help="CSV or slice indices, e.g. 0,4,8 or 0:20:2")
    parser.add_argument("--max-structures", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--supercell", default=None, help="Repeat each selected structure, e.g. 2,2,2")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--damping", default="bj")
    parser.add_argument("--cutoff", type=float, default=40.0 * units.Bohr)
    parser.add_argument("--compare-torch-dftd", action="store_true")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this benchmark on a GPU node")

    structures = load_structures(args.xyz)
    indices = parse_structure_indices(
        args.indices, total=len(structures), max_structures=args.max_structures
    )
    supercell = parse_supercell(args.supercell)

    records: list[dict[str, Any]] = []
    torch_dftd_available = False
    for idx in indices:
        record, torch_available_for_record = benchmark_structure(
            apply_supercell(structures[idx], supercell),
            index=idx,
            repeat=args.repeat,
            xc=args.xc,
            damping=args.damping,
            cutoff=args.cutoff,
            compare_torch_dftd=args.compare_torch_dftd,
        )
        torch_dftd_available = torch_dftd_available or torch_available_for_record
        records.append(record)

    enforce_tolerances(records, rtol=args.rtol, atol=args.atol)
    payload = build_json_payload(
        records=records,
        torch_dftd_available=torch_dftd_available,
        metadata={
            "source": args.xyz or "ase.build.molecule:H2O",
            "indices": indices,
            "repeat": args.repeat,
            "supercell": list(supercell),
            "xc": args.xc,
            "damping": args.damping,
            "cutoff_bohr": args.cutoff / units.Bohr,
            "compare_torch_dftd_requested": args.compare_torch_dftd,
        },
    )
    print_text_report(payload)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
