#!/usr/bin/env python3
"""Prepare a reproducible OC20NEB fullcase-200 FPS extxyz split for MACE."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread, write


DEFAULT_EXPERIMENT_ROOT = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/bin/RSQN-sai-experimental-data"
)
DEFAULT_MANIFEST = (
    DEFAULT_EXPERIMENT_ROOT
    / "experiments/sella_interp_first10_selected200/selected200_manifest.jsonl"
)
DEFAULT_DATA_ROOT = DEFAULT_EXPERIMENT_ROOT / "data/dft_trajs_for_release"


@dataclass(frozen=True)
class CandidateFrame:
    source_key: str
    case_id: str
    frame_index: int
    atoms: Atoms


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_traj_path(row: dict, manifest_path: Path, data_root: Path) -> Path:
    raw_path = Path(row["path"])
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(manifest_path.parent / raw_path)
        if raw_path.parts[:2] == ("data", "dft_trajs_for_release"):
            candidates.append(data_root / Path(*raw_path.parts[2:]))
        candidates.append(data_root / raw_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve trajectory for {row.get('case_id')}: {raw_path}")


def _exportable_atoms(atoms: Atoms, *, source_key: str, case_id: str, frame_index: int) -> Atoms:
    if atoms.calc is None or not atoms.calc.results:
        raise ValueError(f"{source_key} has no calculator results")
    results = atoms.calc.results
    if "energy" not in results or "forces" not in results:
        raise ValueError(f"{source_key} missing energy/forces labels: {sorted(results)}")
    out = atoms.copy()
    out.calc = None
    out.info["energy"] = float(np.asarray(results["energy"]))
    out.arrays["forces"] = np.asarray(results["forces"], dtype=np.float32)
    out.info["source_key"] = source_key
    out.info["case_id"] = case_id
    out.info["source_frame"] = int(frame_index)
    if "stress" in results:
        out.info["stress"] = np.asarray(results["stress"], dtype=np.float32).reshape(-1)
    return out


def collect_labeled_frames(manifest_path: Path, data_root: Path) -> list[CandidateFrame]:
    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    candidates: list[CandidateFrame] = []
    for row in read_jsonl(manifest_path):
        case_id = str(row["case_id"])
        traj_path = resolve_traj_path(row, manifest_path, data_root)
        for frame_index, atoms in enumerate(iread(str(traj_path), ":")):
            if atoms.calc is None or not atoms.calc.results:
                continue
            results = atoms.calc.results
            if "energy" not in results or "forces" not in results:
                continue
            source_key = f"{case_id}:{frame_index}"
            candidates.append(
                CandidateFrame(
                    source_key=source_key,
                    case_id=case_id,
                    frame_index=frame_index,
                    atoms=_exportable_atoms(
                        atoms,
                        source_key=source_key,
                        case_id=case_id,
                        frame_index=frame_index,
                    ),
                )
            )
    return candidates


def load_feature_table(features_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(features_npz, allow_pickle=False)
    if "features" not in payload or "source_keys" not in payload:
        raise KeyError(f"{features_npz} must contain 'features' and 'source_keys' arrays")
    features = np.asarray(payload["features"], dtype=np.float32)
    source_keys = np.asarray(payload["source_keys"]).astype(str)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    if len(features) != len(source_keys):
        raise ValueError(
            f"features/source_keys length mismatch: {len(features)} != {len(source_keys)}"
        )
    return features, source_keys


def align_features(candidates: list[CandidateFrame], features_npz: Path) -> np.ndarray:
    features, source_keys = load_feature_table(features_npz)
    by_key = {key: idx for idx, key in enumerate(source_keys)}
    missing = [candidate.source_key for candidate in candidates if candidate.source_key not in by_key]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"features missing {len(missing)} candidate source key(s): {preview}")
    return np.stack([features[by_key[candidate.source_key]] for candidate in candidates], axis=0)


def farthest_point_indices(features: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    if count > len(features):
        raise ValueError(f"Requested {count} FPS points from only {len(features)} candidates")
    feats = np.asarray(features, dtype=np.float32)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = 0
    min_dist2 = np.sum((feats - feats[0]) ** 2, axis=1)
    for out_idx in range(1, count):
        next_idx = int(np.argmax(min_dist2))
        selected[out_idx] = next_idx
        dist2 = np.sum((feats - feats[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


def random_valid_indices(
    *,
    candidate_count: int,
    train_indices: np.ndarray,
    valid_size: int,
    seed: int,
) -> np.ndarray:
    if valid_size <= 0:
        return np.zeros(0, dtype=np.int64)
    train_set = set(int(idx) for idx in train_indices)
    pool = np.asarray([idx for idx in range(candidate_count) if idx not in train_set], dtype=np.int64)
    if valid_size > len(pool):
        raise ValueError(f"Requested {valid_size} valid frames from only {len(pool)} remaining candidates")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=valid_size, replace=False))


def write_atoms(path: Path, atoms: list[Atoms]) -> None:
    with path.open("w") as handle:
        for item in atoms:
            write(handle, item, format="extxyz")


def prepare_fullcase200_fps_extxyz(
    *,
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    train_size: int = 5000,
    valid_size: int = 10000,
    seed: int = 20260711,
    features_npz: Path,
    overwrite: bool = False,
) -> dict:
    output_dir = output_dir.resolve()
    train_path = output_dir / "train.extxyz"
    valid_path = output_dir / "valid.extxyz"
    if not overwrite and (train_path.exists() or valid_path.exists()):
        raise FileExistsError(f"{output_dir} already contains train/valid extxyz; pass overwrite=True")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_labeled_frames(manifest_path, data_root)
    if train_size + valid_size > len(candidates):
        raise ValueError(
            f"Requested {train_size}+{valid_size} frames from only {len(candidates)} labeled candidates"
        )
    features = align_features(candidates, features_npz)
    train_indices = farthest_point_indices(features, train_size)
    valid_indices = random_valid_indices(
        candidate_count=len(candidates),
        train_indices=train_indices,
        valid_size=valid_size,
        seed=seed,
    )

    train_atoms = [candidates[int(idx)].atoms for idx in train_indices]
    valid_atoms = [candidates[int(idx)].atoms for idx in valid_indices]
    write_atoms(train_path, train_atoms)
    write_atoms(valid_path, valid_atoms)

    summary = {
        "source": {
            "manifest_path": str(Path(manifest_path).resolve()),
            "data_root": str(Path(data_root).resolve()),
            "features_npz": str(Path(features_npz).resolve()),
            "candidate_frames": len(candidates),
            "seed": int(seed),
        },
        "train": {
            "path": str(train_path),
            "frames": len(train_atoms),
            "source_keys": [candidates[int(idx)].source_key for idx in train_indices],
        },
        "valid": {
            "path": str(valid_path),
            "frames": len(valid_atoms),
            "source_keys": [candidates[int(idx)].source_key for idx in valid_indices],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/oc20neb_fullcase200_fps_extxyz"))
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_fullcase200_fps_extxyz(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_size=args.train_size,
        valid_size=args.valid_size,
        seed=args.seed,
        features_npz=args.features_npz,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
