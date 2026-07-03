from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from ase.io import read


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
