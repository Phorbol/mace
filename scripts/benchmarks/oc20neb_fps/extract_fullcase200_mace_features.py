#!/usr/bin/env python3
"""Export mean-pooled MACE node features for OC20NEB fullcase-200 FPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric, torch_tools, utils

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_fullcase200_fps_extxyz import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    CandidateFrame,
    iter_labeled_frames,
)

DEFAULT_MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--head", default="oc20_usemppbe")
    parser.add_argument("--output", type=Path, default=Path("runs/oc20neb_fullcase200_fps_extxyz/mace_mh1_node_feats_features.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--descriptor-key", default="node_feats")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(model_path: Path, *, device_name: str, default_dtype: str, enable_cueq: bool):
    torch_tools.set_default_dtype(default_dtype)
    device = torch_tools.init_device(device_name) if device_name == "cuda" else torch.device("cpu")
    model = torch.load(f=model_path, map_location=device)
    if enable_cueq:
        model = run_e3nn_to_cueq(model, device=device)
    dtype = torch.float32 if default_dtype == "float32" else torch.float64
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    return model, z_table, heads, device


def atoms_with_head(atoms: Atoms, head: str) -> Atoms:
    copied = atoms.copy()
    copied.calc = None
    copied.info["head"] = head
    return copied


def make_loader(atoms_list: list[Atoms], model, z_table, heads, batch_size: int):
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    dataset = [
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max),
            heads=heads,
        )
        for config in configs
    ]
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def tensor_to_numpy(value) -> np.ndarray | None:
    if value is None or not torch.is_tensor(value):
        return None
    return torch_tools.to_numpy(value)


def choose_descriptor(output: dict, key: str, total_atoms: int) -> np.ndarray:
    value = tensor_to_numpy(output.get(key))
    if value is not None and value.ndim == 2 and value.shape[0] == total_atoms:
        return value
    available = []
    for out_key, out_value in output.items():
        if torch.is_tensor(out_value):
            available.append(f"{out_key}:{tuple(out_value.shape)}")
        else:
            available.append(f"{out_key}:{type(out_value).__name__}")
    raise KeyError(
        f"Descriptor key {key!r} is not an atom-level tensor with {total_atoms} rows. "
        f"Available outputs: {', '.join(available)}"
    )


def mean_pool_node_features(node_features: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    ptr = np.asarray(ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 2:
        raise ValueError("ptr must be a one-dimensional cumulative atom pointer")
    node_features = np.asarray(node_features, dtype=np.float32)
    if int(ptr[-1]) != int(node_features.shape[0]):
        raise ValueError(
            f"ptr[-1] ({int(ptr[-1])}) does not match node feature rows ({node_features.shape[0]})"
        )
    return np.stack(
        [node_features[int(start):int(stop)].mean(axis=0) for start, stop in zip(ptr[:-1], ptr[1:], strict=True)],
        axis=0,
    ).astype(np.float32)


def forward_features(
    atoms_list: list[Atoms],
    *,
    model,
    z_table,
    heads,
    device,
    batch_size: int,
    descriptor_key: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    loader = make_loader(atoms_list, model, z_table, heads, batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(
                batch.to_dict(),
                compute_force=False,
                compute_stress=False,
                compute_virials=False,
            )
            ptr = tensor_to_numpy(batch.ptr).astype(np.int64)
            node_desc = choose_descriptor(output, descriptor_key, int(ptr[-1]))
            features.append(mean_pool_node_features(node_desc, ptr))
    return np.concatenate(features, axis=0).astype(np.float32)


def _forward_feature_chunk(
    candidates: list[CandidateFrame],
    *,
    head: str,
    model,
    z_table,
    heads,
    device,
    batch_size: int,
    descriptor_key: str,
) -> np.ndarray:
    atoms_list = [atoms_with_head(candidate.atoms, head) for candidate in candidates]
    return forward_features(
        atoms_list,
        model=model,
        z_table=z_table,
        heads=heads,
        device=device,
        batch_size=batch_size,
        descriptor_key=descriptor_key,
    )


def export_features(
    *,
    manifest: Path,
    data_root: Path,
    model_path: Path,
    head: str,
    output: Path,
    batch_size: int,
    chunk_size: int,
    device_name: str,
    default_dtype: str,
    descriptor_key: str,
    enable_cueq: bool,
    limit_frames: int | None,
    overwrite: bool,
) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    model, z_table, heads, device = load_model(
        model_path,
        device_name=device_name,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
    )
    feature_chunks: list[np.ndarray] = []
    source_keys: list[str] = []
    case_ids: list[str] = []
    frame_indices: list[int] = []
    chunk: list[CandidateFrame] = []
    seen = 0

    def flush_chunk() -> None:
        nonlocal chunk, seen
        if not chunk:
            return
        features = _forward_feature_chunk(
            chunk,
            head=head,
            model=model,
            z_table=z_table,
            heads=heads,
            device=device,
            batch_size=batch_size,
            descriptor_key=descriptor_key,
        )
        feature_chunks.append(features)
        source_keys.extend(candidate.source_key for candidate in chunk)
        case_ids.extend(candidate.case_id for candidate in chunk)
        frame_indices.extend(int(candidate.frame_index) for candidate in chunk)
        seen += len(chunk)
        print(
            f"[extract_fullcase200_mace_features] processed {seen} frames",
            file=sys.stderr,
            flush=True,
        )
        chunk = []

    for candidate in iter_labeled_frames(manifest, data_root):
        if limit_frames is not None and seen + len(chunk) >= int(limit_frames):
            break
        chunk.append(candidate)
        if len(chunk) >= chunk_size:
            flush_chunk()
    flush_chunk()

    if not feature_chunks:
        raise ValueError("no labeled frames found for feature extraction")
    features = np.concatenate(feature_chunks, axis=0).astype(np.float32)
    source_keys_array = np.asarray(source_keys)
    case_ids_array = np.asarray(case_ids)
    frame_indices_array = np.asarray(frame_indices, dtype=np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        source_keys=source_keys_array,
        case_ids=case_ids_array,
        frame_indices=frame_indices_array,
    )
    summary = {
        "output": str(output.resolve()),
        "manifest": str(manifest.resolve()),
        "data_root": str(data_root.resolve()),
        "model": str(model_path.resolve()),
        "head": head,
        "descriptor_key": descriptor_key,
        "frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "chunk_size": int(chunk_size),
        "device": str(device),
        "default_dtype": default_dtype,
        "enable_cueq": bool(enable_cueq),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = export_features(
        manifest=args.manifest,
        data_root=args.data_root,
        model_path=args.model,
        head=args.head,
        output=args.output,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device_name=args.device,
        default_dtype=args.default_dtype,
        descriptor_key=args.descriptor_key,
        enable_cueq=bool(args.enable_cueq),
        limit_frames=args.limit_frames,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
