#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


@dataclass(frozen=True)
class AbacusLogRecord:
    experiment_root: Path
    case_id: str
    role: str
    eval_id: str
    running_log: Path
    kind: str = "eval"


@dataclass
class LabelledStructure:
    record: AbacusLogRecord
    image_index: int
    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None


@dataclass
class LoadedRecord:
    images: list[LabelledStructure]
    skip_reason: str | None = None


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.abs_err = 0.0
        self.sq_err = 0.0

    def add_errors(self, errors: np.ndarray | Sequence[float]) -> None:
        arr = np.asarray(errors, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        self.n += int(arr.size)
        self.abs_err += float(np.abs(arr).sum())
        self.sq_err += float(np.square(arr).sum())

    def as_dict(self) -> dict[str, float | int | None]:
        if self.n == 0:
            return {"n": 0, "mae": None, "rmse": None}
        return {
            "n": self.n,
            "mae": self.abs_err / self.n,
            "rmse": math.sqrt(self.sq_err / self.n),
        }


def _path_role(path: Path, case_index: int, stop_name: str) -> str:
    parts = path.parts
    stop_index = parts.index(stop_name, case_index + 1)
    role_parts = parts[case_index + 1 : stop_index]
    return "/".join(role_parts) if role_parts else "unknown"


def _record_from_eval_log(experiment_root: Path, running_log: Path) -> AbacusLogRecord:
    parts = running_log.parts
    case_index = parts.index("cases") + 1
    eval_index = parts.index("abacus_evals") + 1
    return AbacusLogRecord(
        experiment_root=experiment_root,
        case_id=parts[case_index],
        role=_path_role(running_log, case_index, "abacus_evals"),
        eval_id=parts[eval_index],
        running_log=running_log,
        kind="eval",
    )


def _record_from_socket_log(experiment_root: Path, running_log: Path) -> AbacusLogRecord:
    parts = running_log.parts
    case_index = parts.index("cases") + 1
    socket_index = parts.index("abacus_socket_workdir")
    role_parts = parts[case_index + 1 : socket_index]
    return AbacusLogRecord(
        experiment_root=experiment_root,
        case_id=parts[case_index],
        role="/".join(role_parts) if role_parts else "socket",
        eval_id="socket",
        running_log=running_log,
        kind="socket",
    )


def discover_abacus_eval_logs(experiment_roots: Iterable[Path | str]) -> list[AbacusLogRecord]:
    records: list[AbacusLogRecord] = []
    seen: set[Path] = set()
    for root_like in experiment_roots:
        root = Path(root_like).resolve()
        for log in sorted(root.glob("**/abacus_evals/eval_*/OUT.ABACUS/running_*.log")):
            resolved = log.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(_record_from_eval_log(root, resolved))
    return sorted(records, key=lambda rec: str(rec.running_log))


def discover_abacus_socket_logs(experiment_roots: Iterable[Path | str]) -> list[AbacusLogRecord]:
    records: list[AbacusLogRecord] = []
    seen: set[Path] = set()
    for root_like in experiment_roots:
        root = Path(root_like).resolve()
        for log in sorted(root.glob("**/abacus_socket_workdir/OUT.ABACUS/running_socket.log")):
            resolved = log.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(_record_from_socket_log(root, resolved))
    return sorted(records, key=lambda rec: str(rec.running_log))


def _labelled_structure_from_atoms(record: AbacusLogRecord, image_index: int, atoms: Atoms) -> LabelledStructure:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress: np.ndarray | None
    try:
        stress = np.asarray(atoms.get_stress(), dtype=np.float64)
    except Exception:
        stress = None
    clean = atoms.copy()
    clean.calc = None
    # ABACUS calculations here are periodic; abacuslite does not always set ASE pbc.
    clean.pbc = True
    clean.info.update(
        {
            "case_id": record.case_id,
            "role": record.role,
            "eval_id": record.eval_id,
            "abacus_running_log": str(record.running_log),
            "abacus_kind": record.kind,
        }
    )
    return LabelledStructure(record=record, image_index=image_index, atoms=clean, energy=energy, forces=forces, stress=stress)


def _atoms_from_abacus_frame(frame: dict, energy: dict, forces: np.ndarray, stress: np.ndarray | None) -> Atoms:
    atoms = Atoms(
        symbols=np.asarray(frame["elem"]).tolist(),
        positions=np.asarray(frame["coords"], dtype=float),
        cell=np.asarray(frame["cell"], dtype=float),
        pbc=True,
    )
    calc_kwargs = {
        "energy": float(energy["E_KohnSham"]),
        "free_energy": float(energy.get("E_KohnSham", energy["E_KohnSham"])),
        "forces": np.asarray(forces, dtype=np.float64),
    }
    if stress is not None:
        calc_kwargs["stress"] = np.asarray(stress, dtype=np.float64)
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


def _final_energies_from_abacus_lines(io_module, lines: list[str]) -> list[dict]:
    energies = io_module.read_energies_from_running_log(lines)[1]
    try:
        headers = io_module.read_iter_header_from_running_log(lines)
        if headers:
            return io_module.find_final_info_with_iter_header(energies, headers)
    except Exception:
        pass
    return energies


def _assemble_record_from_abacus_helpers(record: AbacusLogRecord, io_module) -> LoadedRecord:
    lines = record.running_log.read_text(errors="replace").splitlines()
    try:
        frames = io_module.read_traj_from_running_log(lines)
        forces = io_module.read_forces_from_running_log(lines)
        stresses = io_module.read_stress_from_running_log(lines)
        energies = _final_energies_from_abacus_lines(io_module, lines)
    except Exception as exc:
        return LoadedRecord(images=[], skip_reason=f"abacuslite helper read failed: {type(exc).__name__}: {exc}")

    if len(stresses) == 0:
        stresses = [None] * len(frames)
    if len(stresses) != len(frames):
        return LoadedRecord(
            images=[],
            skip_reason=f"{record.kind} output has {len(frames)} structure frames and {len(stresses)} stress frames",
        )
    if not (len(frames) == len(forces) == len(energies)):
        return LoadedRecord(
            images=[],
            skip_reason=(
                f"{record.kind} output has {len(frames)} structure frames, "
                f"{len(forces)} force frames, and {len(energies)} energy frames; "
                "refusing to align labels without per-frame structures"
            ),
        )

    images = []
    for index, (frame, energy, force, stress) in enumerate(zip(frames, energies, forces, stresses)):
        atoms = _atoms_from_abacus_frame(frame, energy, force, stress)
        images.append(_labelled_structure_from_atoms(record, index, atoms))
    return LoadedRecord(images=images)


def _read_socket_record(record: AbacusLogRecord, latestio_module=None) -> LoadedRecord:
    if latestio_module is None:
        from abacuslite.io import latestio as latestio_module
    return _assemble_record_from_abacus_helpers(record, latestio_module)


def read_abacus_record(record: AbacusLogRecord, latestio_module=None, legacyio_module=None) -> LoadedRecord:
    if record.kind == "socket":
        return _read_socket_record(record, latestio_module=latestio_module)
    if legacyio_module is None:
        from abacuslite.io import legacyio as legacyio_module
    try:
        atoms = legacyio_module.read_abacus_out(record.running_log, index=-1)
    except Exception as exc:
        fallback = _assemble_record_from_abacus_helpers(record, legacyio_module)
        if fallback.skip_reason:
            fallback.skip_reason = (
                f"abacuslite legacy read failed: {type(exc).__name__}: {exc}; "
                f"{fallback.skip_reason}"
            )
        return fallback
    if isinstance(atoms, list):
        image_list = atoms
    else:
        image_list = [atoms]
    images = [_labelled_structure_from_atoms(record, i, image) for i, image in enumerate(image_list)]
    return LoadedRecord(images=images)


def read_all_abacus_records(records: Iterable[AbacusLogRecord]) -> tuple[list[LabelledStructure], list[dict[str, str]]]:
    structures: list[LabelledStructure] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        loaded = read_abacus_record(record)
        if loaded.skip_reason:
            skipped.append(
                {
                    "case_id": record.case_id,
                    "role": record.role,
                    "eval_id": record.eval_id,
                    "kind": record.kind,
                    "running_log": str(record.running_log),
                    "reason": loaded.skip_reason,
                }
            )
        structures.extend(loaded.images)
    return structures, skipped


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise ValueError("model spec must be name=backend:/path/to/model")
    name, rest = spec.split("=", 1)
    backend, path = rest.split(":", 1)
    if backend not in {"mace", "deepmd", "dp"}:
        raise ValueError(f"unknown model backend {backend!r}")
    return name, backend, path


def make_calculator(backend: str, model_path: str, args: argparse.Namespace):
    if backend == "mace":
        from mace.calculators import MACECalculator

        kwargs = {
            "model_paths": model_path,
            "device": args.device,
            "default_dtype": args.default_dtype,
            "enable_cueq": args.enable_cueq,
        }
        if args.mace_head:
            kwargs["head"] = args.mace_head
        return MACECalculator(**kwargs)
    from deepmd.calculator import DP

    kwargs = {"model": model_path, "nlist_backend": args.deepmd_nlist_backend}
    if args.deepmd_head:
        kwargs["head"] = args.deepmd_head
    return DP(**kwargs)


def evaluate_calculator(model_name: str, calculator, structures: Sequence[LabelledStructure]) -> list[dict]:
    rows: list[dict] = []
    for structure in structures:
        atoms = structure.atoms.copy()
        atoms.calc = calculator
        pred_energy = float(atoms.get_potential_energy())
        pred_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        rows.append(
            {
                "model": model_name,
                "case_id": structure.record.case_id,
                "role": structure.record.role,
                "eval_id": structure.record.eval_id,
                "kind": structure.record.kind,
                "running_log": str(structure.record.running_log),
                "image_index": structure.image_index,
                "natoms": len(structure.atoms),
                "reference_energy": structure.energy,
                "predicted_energy": pred_energy,
                "reference_forces": structure.forces,
                "predicted_forces": pred_forces,
            }
        )
    return rows


def summarize_prediction_rows(rows: Sequence[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    summary = {"models": {}}
    for model, model_rows in sorted(by_model.items()):
        energy_total = RunningStats()
        energy_per_atom = RunningStats()
        force_components = RunningStats()
        role_counts: dict[str, int] = {}
        case_errors: dict[str, list[float]] = {}
        for row in model_rows:
            natoms = int(row["natoms"])
            e_error = float(row["predicted_energy"]) - float(row["reference_energy"])
            epa_error = e_error / natoms
            energy_total.add_errors([e_error])
            energy_per_atom.add_errors([epa_error])
            force_components.add_errors(np.asarray(row["predicted_forces"], dtype=np.float64) - np.asarray(row["reference_forces"], dtype=np.float64))
            role_counts[str(row.get("role", "unknown"))] = role_counts.get(str(row.get("role", "unknown")), 0) + 1
            case_errors.setdefault(str(row["case_id"]), []).append(epa_error)

        bias_corrected = RunningStats()
        for errors in case_errors.values():
            arr = np.asarray(errors, dtype=np.float64)
            bias_corrected.add_errors(arr - float(arr.mean()))

        summary["models"][model] = {
            "structures": len(model_rows),
            "roles": dict(sorted(role_counts.items())),
            "energy_total": energy_total.as_dict(),
            "energy_per_atom": energy_per_atom.as_dict(),
            "energy_per_atom_case_bias_corrected": bias_corrected.as_dict(),
            "force_components": force_components.as_dict(),
        }
    return summary


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_prediction_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "model",
        "case_id",
        "role",
        "eval_id",
        "kind",
        "running_log",
        "image_index",
        "natoms",
        "reference_energy",
        "predicted_energy",
        "energy_error",
        "energy_error_per_atom",
        "force_mae",
        "force_rmse",
        "force_max_abs_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            force_err = np.asarray(row["predicted_forces"], dtype=np.float64) - np.asarray(row["reference_forces"], dtype=np.float64)
            e_error = float(row["predicted_energy"]) - float(row["reference_energy"])
            writer.writerow(
                {
                    "model": row["model"],
                    "case_id": row["case_id"],
                    "role": row["role"],
                    "eval_id": row["eval_id"],
                    "kind": row["kind"],
                    "running_log": row["running_log"],
                    "image_index": row["image_index"],
                    "natoms": row["natoms"],
                    "reference_energy": row["reference_energy"],
                    "predicted_energy": row["predicted_energy"],
                    "energy_error": e_error,
                    "energy_error_per_atom": e_error / int(row["natoms"]),
                    "force_mae": float(np.abs(force_err).mean()),
                    "force_rmse": float(np.sqrt(np.square(force_err).mean())),
                    "force_max_abs_error": float(np.abs(force_err).max()),
                }
            )


def write_label_extxyz(path: Path, structures: Sequence[LabelledStructure]) -> None:
    images = []
    for structure in structures:
        atoms = structure.atoms.copy()
        kwargs = {"energy": structure.energy, "forces": structure.forces}
        if structure.stress is not None:
            kwargs["stress"] = structure.stress
        atoms.calc = SinglePointCalculator(atoms, **kwargs)
        images.append(atoms)
    if images:
        write(path, images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MACE/DPA4 models on raw ABACUS eval outputs read with abacuslite.")
    parser.add_argument("--experiment-root", action="append", required=True, help="ABACUS experiment root containing cases/*/abacus_evals or socket workdirs")
    parser.add_argument("--include-socket", action="store_true", help="Discover socket outputs; multi-force single-structure socket logs are skipped")
    parser.add_argument("--model", action="append", default=[], help="Model spec name=backend:/path, backend in mace/deepmd/dp")
    parser.add_argument("--output-dir", default="runs/abacus_raw_extrapolation_eval")
    parser.add_argument("--limit-structures", type=int, default=None)
    parser.add_argument("--write-label-extxyz", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float32")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--mace-head", default=None)
    parser.add_argument("--deepmd-head", default=None)
    parser.add_argument("--deepmd-nlist-backend", default="ase")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(root) for root in args.experiment_root]

    records = discover_abacus_eval_logs(roots)
    if args.include_socket:
        records.extend(discover_abacus_socket_logs(roots))
        records = sorted(records, key=lambda rec: str(rec.running_log))

    structures, skipped = read_all_abacus_records(records)
    if args.limit_structures is not None:
        structures = structures[: args.limit_structures]
    if args.write_label_extxyz:
        write_label_extxyz(output_dir / "abacus_raw_labels.extxyz", structures)

    all_rows: list[dict] = []
    for spec in args.model:
        name, backend, model_path = parse_model_spec(spec)
        calc = make_calculator(backend, model_path, args)
        all_rows.extend(evaluate_calculator(name, calc, structures))

    summary = summarize_prediction_rows(all_rows)
    summary["source"] = {
        "experiment_roots": [str(root) for root in roots],
        "discovered_logs": len(records),
        "loaded_structures": len(structures),
        "skipped_logs": len(skipped),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_safe) + "\n")
    (output_dir / "skipped_logs.json").write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n")
    if all_rows:
        write_prediction_csv(output_dir / "prediction_errors.csv", all_rows)


if __name__ == "__main__":
    main()
