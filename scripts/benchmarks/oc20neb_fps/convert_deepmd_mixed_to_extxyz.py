#!/usr/bin/env python3
"""Convert the OC20NEB FPS DeepMD mixed benchmark split to MACE extxyz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


DEFAULT_MIXED_ROOT = Path(
    "/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/reference_repos/"
    "deepmd-kit/benchmarks/dpa4_oc20neb_fps/"
    "oc20neb_fps5k_train_random50_valid/deepmd_mixed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepMD mixed OC20NEB FPS data into train.extxyz and "
            "valid.extxyz for MACE."
        )
    )
    parser.add_argument("--mixed-root", type=Path, default=DEFAULT_MIXED_ROOT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/oc20neb_fps_extxyz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=None,
        help="Optional smoke-test cap applied independently to train and valid.",
    )
    return parser.parse_args()


def read_systems(mixed_root: Path, split: str) -> list[Path]:
    list_path = mixed_root / f"{split}_systems.txt"
    if list_path.exists():
        return [Path(line.strip()) for line in list_path.read_text().splitlines() if line.strip()]
    split_root = mixed_root / split
    return sorted(path for path in split_root.iterdir() if path.is_dir())


def read_type_map(system: Path) -> list[str]:
    return [line.strip() for line in (system / "type_map.raw").read_text().splitlines() if line.strip()]


def load_real_atom_types(set_dir: Path, frame_count: int, natoms: int) -> np.ndarray:
    real_types_path = set_dir / "real_atom_types.npy"
    if not real_types_path.exists():
        raise FileNotFoundError(
            f"{real_types_path} is required; type.raw is a placeholder in this benchmark"
        )
    real_types = np.load(real_types_path)
    if real_types.shape != (frame_count, natoms):
        raise ValueError(
            f"{real_types_path} has shape {real_types.shape}, expected {(frame_count, natoms)}"
        )
    return real_types.astype(np.int64, copy=False)


def iter_system_atoms(system: Path):
    type_map = read_type_map(system)
    for set_dir in sorted(system.glob("set.*")):
        coord = np.load(set_dir / "coord.npy")
        box = np.load(set_dir / "box.npy")
        energy = np.load(set_dir / "energy.npy")
        force = np.load(set_dir / "force.npy")
        frame_count = int(coord.shape[0])
        if coord.ndim != 2 or coord.shape[1] % 3 != 0:
            raise ValueError(f"{set_dir / 'coord.npy'} has invalid shape {coord.shape}")
        natoms = coord.shape[1] // 3
        real_types = load_real_atom_types(set_dir, frame_count, natoms)
        positions = coord.reshape(frame_count, natoms, 3)
        forces = force.reshape(frame_count, natoms, 3)
        cells = box.reshape(frame_count, 3, 3)
        for frame_index in range(frame_count):
            symbols = [type_map[int(type_id)] for type_id in real_types[frame_index]]
            atoms = Atoms(
                symbols=symbols,
                positions=positions[frame_index],
                cell=cells[frame_index],
                pbc=True,
            )
            atoms.info["energy"] = float(energy[frame_index])
            atoms.arrays["forces"] = np.asarray(forces[frame_index], dtype=np.float32)
            yield atoms


def write_split(
    *,
    mixed_root: Path,
    outdir: Path,
    split: str,
    max_frames: int | None,
) -> dict[str, int]:
    output = outdir / f"{split}.extxyz"
    systems = read_systems(mixed_root, split)
    written = 0
    natom_counts: dict[int, int] = {}
    with output.open("w") as handle:
        for system in systems:
            for atoms in iter_system_atoms(system):
                if max_frames is not None and written >= max_frames:
                    return {
                        "frames": written,
                        "systems": len(systems),
                        "min_natoms": min(natom_counts) if natom_counts else 0,
                        "max_natoms": max(natom_counts) if natom_counts else 0,
                    }
                write(handle, atoms, format="extxyz")
                natom_counts[len(atoms)] = natom_counts.get(len(atoms), 0) + 1
                written += 1
    return {
        "frames": written,
        "systems": len(systems),
        "min_natoms": min(natom_counts) if natom_counts else 0,
        "max_natoms": max(natom_counts) if natom_counts else 0,
    }


def convert_mixed_dataset(
    mixed_root: Path,
    outdir: Path,
    *,
    overwrite: bool = False,
    max_frames_per_split: int | None = None,
) -> dict[str, dict[str, int]]:
    mixed_root = mixed_root.resolve()
    if not mixed_root.exists():
        raise FileNotFoundError(mixed_root)
    if outdir.exists() and not overwrite:
        existing = [outdir / "train.extxyz", outdir / "valid.extxyz"]
        if any(path.exists() for path in existing):
            raise FileExistsError(f"{outdir} already contains converted data; pass overwrite=True")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        split: write_split(
            mixed_root=mixed_root,
            outdir=outdir,
            split=split,
            max_frames=max_frames_per_split,
        )
        for split in ("train", "valid")
    }
    summary["source"] = {"mixed_root": str(mixed_root)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = convert_mixed_dataset(
        args.mixed_root,
        args.outdir,
        overwrite=args.overwrite,
        max_frames_per_split=args.max_frames_per_split,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
