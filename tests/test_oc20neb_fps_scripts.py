from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io import write


SCRIPT_ROOT = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "oc20neb_fps"
)


def load_script(name: str):
    module_path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_deepmd_system(system: Path, type_map: list[str]) -> None:
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n0\n0\n")
    (system / "type_map.raw").write_text("\n".join(type_map) + "\n")
    np.save(set_dir / "coord.npy", np.asarray([[0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0]], dtype=np.float32))
    np.save(set_dir / "box.npy", np.asarray([[8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]], dtype=np.float32))
    np.save(set_dir / "energy.npy", np.asarray([-7.5], dtype=np.float32))
    np.save(set_dir / "force.npy", np.asarray([[0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.0, 0.4, -0.4]], dtype=np.float32))
    np.save(set_dir / "real_atom_types.npy", np.asarray([[5, 7, 28]], dtype=np.int64))


def test_converter_uses_real_atom_types_not_placeholder_type_raw(tmp_path):
    converter = load_script("convert_deepmd_mixed_to_extxyz.py")
    mixed = tmp_path / "deepmd_mixed"
    train_system = mixed / "train" / "natoms_003"
    valid_system = mixed / "valid" / "natoms_003"
    type_map = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
    write_deepmd_system(train_system, type_map)
    write_deepmd_system(valid_system, type_map)
    (mixed / "train_systems.txt").write_text(str(train_system) + "\n")
    (mixed / "valid_systems.txt").write_text(str(valid_system) + "\n")

    outdir = tmp_path / "extxyz"
    converter.convert_mixed_dataset(mixed, outdir)

    train_atoms = read(outdir / "train.extxyz", index=0)
    valid_atoms = read(outdir / "valid.extxyz", index=0)

    assert train_atoms.get_chemical_symbols() == ["C", "O", "Cu"]
    assert valid_atoms.get_chemical_symbols() == ["C", "O", "Cu"]
    assert train_atoms.get_potential_energy() == -7.5
    np.testing.assert_allclose(
        train_atoms.get_forces(),
        np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.4, -0.4]], dtype=np.float32),
    )
    assert train_atoms.pbc.all()


def test_sai_wrapper_uses_valid_file_and_sai_safe_resource_flags():
    wrapper = SCRIPT_ROOT / "run_mace_oc20neb_fps_sai.sh"
    text = wrapper.read_text()

    assert "--valid_file=\"${VALID_FILE}\"" in text
    assert "--num_channels=\"${NUM_CHANNELS:-64}\"" in text
    assert "--max_L=\"${MAX_L:-1}\"" in text
    assert "--scheduler=\"${SCHEDULER:-WSD}\"" in text
    assert "--enable_cueq=\"${ENABLE_CUEQ:-True}\"" in text
    assert "--edge_force_compile_graph" in text
    assert "#SBATCH --cpus-per-task" not in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --ntasks-per-node" not in text


def test_prepare_fullcase200_fps_writes_manifest_selected_extxyz(tmp_path):
    preparer = load_script("prepare_fullcase200_fps_extxyz.py")
    data_root = tmp_path / "data" / "dft_trajs_for_release"
    traj_dir = data_root / "dissociations"
    traj_dir.mkdir(parents=True)
    traj_path = traj_dir / "case_a.traj"

    images = []
    for frame_index in range(5):
        atoms = Atoms(
            symbols=["H", "O"],
            positions=[[0.0, 0.0, 0.0], [1.0 + 0.1 * frame_index, 0.0, 0.0]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(frame_index),
            forces=np.asarray([[frame_index, 0.0, 0.0], [0.0, -frame_index, 0.0]]),
        )
        images.append(atoms)
    write(traj_path, images)

    manifest = tmp_path / "selected200_manifest.jsonl"
    manifest.write_text(
        '{"case_id": "case_a", "path": "data/dft_trajs_for_release/dissociations/case_a.traj"}\n'
    )
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ],
        dtype=np.float32,
    )
    feature_path = tmp_path / "features.npz"
    np.savez(
        feature_path,
        features=features,
        source_keys=np.asarray([f"case_a:{i}" for i in range(5)]),
    )

    summary = preparer.prepare_fullcase200_fps_extxyz(
        manifest_path=manifest,
        data_root=data_root,
        output_dir=tmp_path / "out",
        train_size=2,
        valid_size=2,
        seed=7,
        features_npz=feature_path,
        overwrite=False,
    )

    assert summary["train"]["frames"] == 2
    assert summary["valid"]["frames"] == 2
    assert summary["source"]["candidate_frames"] == 5

    train_atoms = read(tmp_path / "out" / "train.extxyz", index=":")
    valid_atoms = read(tmp_path / "out" / "valid.extxyz", index=":")
    train_keys = {atoms.info["source_key"] for atoms in train_atoms}
    valid_keys = {atoms.info["source_key"] for atoms in valid_atoms}

    assert train_keys == {"case_a:0", "case_a:4"}
    assert train_keys.isdisjoint(valid_keys)
    assert all(
        atoms.get_potential_energy() == atoms.info["source_frame"]
        for atoms in train_atoms + valid_atoms
    )
    assert all(atoms.get_forces().shape == (2, 3) for atoms in train_atoms + valid_atoms)


def test_fullcase200_ef_20k_demo_sbatch_targets_current_env_and_compile():
    sbatch = SCRIPT_ROOT / "fullcase200-ef-20k-demo.sbatch"
    text = sbatch.read_text()

    assert "MACE_OC20NEB_TARGET_STEPS:-20000" in text
    assert "conda-envs/mace-dpa4-cu126/bin/python" in text
    assert "runs/oc20neb_fullcase200_fps_extxyz" in text
    assert '--train_file="${TRAIN_FILE}"' in text
    assert '--valid_file="${VALID_FILE}"' in text
    assert "--loss=weighted" in text
    assert "--energy_key=energy" in text
    assert "--forces_key=forces" in text
    assert "--edge_force_compile_force_gradient_mode=positions" in text
    assert "--no-edge_force_compile_allow_fallback" in text
    assert "parse_metrics.py" in text
