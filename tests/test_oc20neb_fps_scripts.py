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
        fps_feature_dim=1,
        overwrite=False,
    )

    assert summary["source"]["original_feature_dim"] == 2
    assert summary["source"]["fps_feature_dim"] == 1
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


def test_extract_fullcase200_mace_features_mean_pools_by_graph():
    extractor = load_script("extract_fullcase200_mace_features.py")
    node_features = np.asarray(
        [
            [1.0, 3.0],
            [3.0, 5.0],
            [10.0, 20.0],
        ],
        dtype=np.float32,
    )
    ptr = np.asarray([0, 2, 3], dtype=np.int64)

    pooled = extractor.mean_pool_node_features(node_features, ptr)

    np.testing.assert_allclose(
        pooled,
        np.asarray([[2.0, 4.0], [10.0, 20.0]], dtype=np.float32),
    )


def test_extract_fullcase200_mace_features_streams_candidates_by_chunk(tmp_path, monkeypatch):
    extractor = load_script("extract_fullcase200_mace_features.py")
    candidates = [
        type(
            "Candidate",
            (),
            {
                "source_key": f"case_a:{idx}",
                "case_id": "case_a",
                "frame_index": idx,
                "atoms": Atoms("H", positions=[[float(idx), 0.0, 0.0]]),
            },
        )()
        for idx in range(5)
    ]
    observed_chunk_sizes = []

    monkeypatch.setattr(
        extractor,
        "iter_labeled_frames",
        lambda _manifest, _data_root: iter(candidates),
    )
    monkeypatch.setattr(
        extractor,
        "load_model",
        lambda *_args, **_kwargs: (object(), object(), None, "cpu"),
    )

    def fake_forward_features(atoms_list, **_kwargs):
        observed_chunk_sizes.append(len(atoms_list))
        start = sum(observed_chunk_sizes[:-1])
        return np.asarray([[float(start + offset), 1.0] for offset in range(len(atoms_list))], dtype=np.float32)

    monkeypatch.setattr(extractor, "forward_features", fake_forward_features)

    summary = extractor.export_features(
        manifest=tmp_path / "manifest.jsonl",
        data_root=tmp_path / "data",
        model_path=tmp_path / "model.pt",
        head="oc20_usemppbe",
        output=tmp_path / "features.npz",
        batch_size=4,
        chunk_size=2,
        device_name="cpu",
        default_dtype="float32",
        descriptor_key="node_feats",
        enable_cueq=False,
        limit_frames=None,
        overwrite=False,
    )

    payload = np.load(tmp_path / "features.npz", allow_pickle=False)
    assert observed_chunk_sizes == [2, 2, 1]
    assert summary["frames"] == 5
    np.testing.assert_allclose(
        payload["features"],
        np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]], dtype=np.float32),
    )
    assert payload["source_keys"].astype(str).tolist() == [f"case_a:{idx}" for idx in range(5)]


def test_project_features_for_fps_is_seeded_and_dimension_reducing():
    preparer = load_script("prepare_fullcase200_fps_extxyz.py")
    features = np.arange(24, dtype=np.float32).reshape(4, 6)

    first = preparer.project_features_for_fps(features, target_dim=3, seed=11)
    second = preparer.project_features_for_fps(features, target_dim=3, seed=11)
    unchanged = preparer.project_features_for_fps(features[:, :2], target_dim=3, seed=11)

    assert first.shape == (4, 3)
    np.testing.assert_allclose(first, second)
    assert unchanged.shape == (4, 2)
    np.testing.assert_allclose(unchanged, features[:, :2])


def test_fullcase200_ef20k_demo_defaults_focus_noncompile_cueq_muon_matrix():
    sbatch = SCRIPT_ROOT / "fullcase200-ef-20k-demo.sbatch"
    text = sbatch.read_text()

    assert "MACE_OC20NEB_CASES:-eager,cueq,hybrid_muon,cueq_hybrid_muon" in text
    assert "MACE_OC20NEB_CASES:-eager,compile,cueq,cueq_compile" not in text
    assert "MACE_OC20NEB_COMPILE_PARITY_GRADIENTS:-False" in text
    assert "--no-edge_force_compile_parity_check_gradients" in text
    assert "compile_parity_gradients" in text
    assert "run_selected_case compile" in text
    assert "run_selected_case cueq_compile" in text
    assert "run_selected_case hybrid_muon_compile" in text
    assert "run_selected_case cueq_hybrid_muon_compile" in text


def test_prepare_fullcase200_fps_sbatch_extracts_mace_features_before_fps():
    sbatch = SCRIPT_ROOT / "prepare-fullcase200-fps-extxyz.sbatch"
    text = sbatch.read_text()

    assert "#SBATCH --partition=16V100" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "MACE_OC20NEB_PREP_TRAIN_SIZE:-5000" in text
    assert "MACE_OC20NEB_PREP_VALID_SIZE:-10000" in text
    assert "/home/gengjianrui/.cache/mace/mace-mh-1.model" in text
    assert "extract_fullcase200_mace_features.py" in text
    assert "prepare_fullcase200_fps_extxyz.py" in text
    assert "MACE_OC20NEB_PREP_CHUNK_SIZE:-256" in text
    assert "MACE_OC20NEB_PREP_FPS_FEATURE_DIM:-128" in text
    assert "--chunk-size" in text
    assert "--fps-feature-dim" in text
    assert "Skipping feature extraction" in text
    assert "--features-npz" in text


def test_fullcase200_ef_20k_demo_sbatch_targets_current_env_and_compile():
    sbatch = SCRIPT_ROOT / "fullcase200-ef-20k-demo.sbatch"
    text = sbatch.read_text()

    assert "#SBATCH --partition=16V100" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "MACE_OC20NEB_TARGET_STEPS:-20000" in text
    assert "conda-envs/mace-dpa4-cu126/bin/python" in text
    assert "runs/oc20neb_fullcase200_fps_extxyz" in text
    assert '--train_file="${TRAIN_FILE}"' in text
    assert '--valid_file="${VALID_FILE}"' in text
    assert "--loss=weighted" in text
    assert "MACE_OC20NEB_STAGE_TWO:-True" in text
    assert "MACE_OC20NEB_STAGE_TWO_FRACTION:-0.75" in text
    assert "--stage_two" in text
    assert '--start_stage_two_update="${START_STAGE_TWO_UPDATE}"' in text
    assert "MACE_OC20NEB_STAGE1_ENERGY_WEIGHT:-1.0" in text
    assert "MACE_OC20NEB_STAGE1_FORCES_WEIGHT:-100.0" in text
    assert "MACE_OC20NEB_STAGE2_ENERGY_WEIGHT:-100.0" in text
    assert "MACE_OC20NEB_STAGE2_FORCES_WEIGHT:-1.0" in text
    assert "--stage_two_energy_weight=" in text
    assert "--stage_two_forces_weight=" in text
    assert "stage_two_start_update" in text
    assert "max_num_updates" in text
    assert "eval_interval_updates" in text
    assert "valid_batch_size" in text
    assert "slurm_gpus_per_node" in text
    assert "--energy_key=energy" in text
    assert "--forces_key=forces" in text
    assert '--max_num_updates="${TARGET_STEPS}"' in text
    assert '--eval_interval_updates="${EVAL_INTERVAL_UPDATES}"' in text
    assert '--scheduler="${MACE_OC20NEB_SCHEDULER:-WSD}"' in text
    assert '--lr_scheduler_interval="${MACE_OC20NEB_LR_SCHEDULER_INTERVAL:-step}"' in text
    assert "--edge_force_compile_force_gradient_mode=positions" in text
    assert "--edge_force_compile_cache_policy=bucket" in text
    assert "--edge_force_compile_bucket_atoms=" in text
    assert "--edge_force_compile_bucket_edges=" in text
    assert "--edge_force_compile_bucket_margin=" in text
    assert "MACE_OC20NEB_COMPILE_BUCKET_ATOMS:-384,512,640,800" in text
    assert "MACE_OC20NEB_COMPILE_BUCKET_EDGES:-8192,12288,16384,24576,32768,40960" in text
    assert "MACE_OC20NEB_COMPILE_BUCKET_MARGIN:-0" in text
    assert "MACE_OC20NEB_COMPILE_MAX_CACHE_ENTRIES:-2" in text
    assert "--edge_force_compile_max_cache_entries=" in text
    assert "--edge_force_compile_cache_policy=dynamic" not in text
    assert "--no-edge_force_compile_allow_fallback" in text
    assert "MACE_OC20NEB_CASES:-eager,cueq,hybrid_muon,cueq_hybrid_muon" in text
    assert "run_selected_case hybrid_muon" in text
    assert "run_selected_case hybrid_muon_compile" in text
    assert "run_selected_case cueq" in text
    assert "run_selected_case cueq_compile" in text
    assert "run_selected_case cueq_hybrid_muon" in text
    assert "run_selected_case cueq_hybrid_muon_compile" in text
    assert "--enable_cueq=True" in text
    assert "--optimizer=hybrid_muon" in text
    assert '--hybrid_muon_mode="${MACE_OC20NEB_HYBRID_MUON_MODE:-2d}"' in text
    assert '--hybrid_muon_routing="${MACE_OC20NEB_HYBRID_MUON_ROUTING:-mace}"' in text
    assert '--hybrid_muon_lr_factor="${MACE_OC20NEB_HYBRID_MUON_LR_FACTOR:-0.1}"' in text
    assert "parse_metrics.py" in text
    assert "summarize_fullcase200_ef20k_matrix.py" in text
    assert '--error_table="${MACE_OC20NEB_ERROR_TABLE:-PerAtomMAE}"' in text


def test_abacus_raw_eval_discovery_includes_vib_and_ignores_sella_traj(tmp_path):
    evaluator = load_script("evaluate_abacus_raw_extrapolation.py")
    exp = tmp_path / "abacus_exp"
    sella_log = exp / "cases" / "000_case_a" / "sella" / "abacus_evals" / "eval_000001" / "OUT.ABACUS" / "running_scf.log"
    vib_log = exp / "cases" / "000_case_a" / "sella" / "vib_tag2_resume" / "abacus_evals" / "eval_000002" / "OUT.ABACUS" / "running_scf.log"
    sella_log.parent.mkdir(parents=True)
    vib_log.parent.mkdir(parents=True)
    sella_log.write_text("raw scf")
    vib_log.write_text("raw vib scf")
    (exp / "cases" / "000_case_a" / "sella" / "sella_ts.traj").write_text("not a label source")

    records = evaluator.discover_abacus_eval_logs([exp])

    assert [(record.case_id, record.role, record.eval_id) for record in records] == [
        ("000_case_a", "sella", "eval_000001"),
        ("000_case_a", "sella/vib_tag2_resume", "eval_000002"),
    ]
    assert all("sella_ts.traj" not in str(record.running_log) for record in records)


def test_abacus_eval_falls_back_to_abacuslite_helpers_when_band_tables_are_missing(tmp_path):
    evaluator = load_script("evaluate_abacus_raw_extrapolation.py")
    running_log = tmp_path / "cases" / "case_a" / "sella" / "abacus_evals" / "eval_000001" / "OUT.ABACUS" / "running_scf.log"
    running_log.parent.mkdir(parents=True)
    running_log.write_text("raw scf without kpoint table")
    record = evaluator.AbacusLogRecord(
        experiment_root=tmp_path,
        case_id="case_a",
        role="sella",
        eval_id="eval_000001",
        running_log=running_log,
        kind="eval",
    )
    frame = {
        "elem": np.asarray(["H", "O"]),
        "coords": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": np.eye(3) * 8.0,
    }

    class FakeLegacy:
        @staticmethod
        def read_abacus_out(*_args, **_kwargs):
            raise AssertionError("No k-point found")

        @staticmethod
        def read_traj_from_running_log(_lines):
            return [frame]

        @staticmethod
        def read_forces_from_running_log(_lines):
            return [np.asarray([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])]

        @staticmethod
        def read_stress_from_running_log(_lines):
            return []

        @staticmethod
        def read_energies_from_running_log(_lines):
            return [], [{"E_KohnSham": -7.5, "E_Fermi": 0.0}]

        @staticmethod
        def read_iter_header_from_running_log(_lines):
            return [(1, 1)]

        @staticmethod
        def find_final_info_with_iter_header(energies, _headers):
            return energies

    loaded = evaluator.read_abacus_record(record, legacyio_module=FakeLegacy)

    assert loaded.skip_reason is None
    assert len(loaded.images) == 1
    assert loaded.images[0].energy == -7.5
    np.testing.assert_allclose(
        loaded.images[0].forces,
        [[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]],
    )


def test_abacus_socket_multiforce_single_structure_is_skipped_without_misalignment(tmp_path):
    evaluator = load_script("evaluate_abacus_raw_extrapolation.py")
    running_log = tmp_path / "OUT.ABACUS" / "running_socket.log"
    running_log.parent.mkdir()
    running_log.write_text("socket raw")
    record = evaluator.AbacusLogRecord(
        experiment_root=tmp_path,
        case_id="case_socket",
        role="sella_socket_wrapped",
        eval_id="socket",
        running_log=running_log,
        kind="socket",
    )

    frame = {
        "elem": np.asarray(["H", "O"]),
        "coords": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": np.eye(3) * 8.0,
    }

    class FakeLatest:
        @staticmethod
        def read_traj_from_running_log(_lines):
            return [frame]

        @staticmethod
        def read_forces_from_running_log(_lines):
            return [np.zeros((2, 3)), np.ones((2, 3))]

        @staticmethod
        def read_stress_from_running_log(_lines):
            return []

        @staticmethod
        def read_energies_from_running_log(_lines):
            return [], [
                {"E_KohnSham": -1.0, "E_Fermi": 0.0},
                {"E_KohnSham": -2.0, "E_Fermi": 0.0},
            ]

        @staticmethod
        def read_iter_header_from_running_log(_lines):
            return [(1, 1), (2, 1)]

        @staticmethod
        def find_final_info_with_iter_header(energies, _headers):
            return energies

    loaded = evaluator.read_abacus_record(record, latestio_module=FakeLatest)

    assert loaded.images == []
    assert loaded.skip_reason is not None
    assert "1 structure frame" in loaded.skip_reason
    assert "2 force frames" in loaded.skip_reason


def test_abacus_extrapolation_metrics_include_force_and_bias_corrected_energy():
    evaluator = load_script("evaluate_abacus_raw_extrapolation.py")
    rows = [
        {
            "model": "model_a",
            "case_id": "case_1",
            "role": "sella",
            "natoms": 2,
            "reference_energy": 10.0,
            "predicted_energy": 12.0,
            "reference_forces": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "predicted_forces": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        },
        {
            "model": "model_a",
            "case_id": "case_1",
            "role": "sella",
            "natoms": 2,
            "reference_energy": 20.0,
            "predicted_energy": 24.0,
            "reference_forces": np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "predicted_forces": np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        },
    ]

    summary = evaluator.summarize_prediction_rows(rows)

    model_summary = summary["models"]["model_a"]
    assert model_summary["structures"] == 2
    assert model_summary["force_components"]["mae"] == 1.0 / 3.0
    assert model_summary["force_components"]["rmse"] == np.sqrt(1.0 / 3.0)
    assert model_summary["energy_per_atom"]["mae"] == 1.5
    assert model_summary["energy_per_atom_case_bias_corrected"]["mae"] == 0.5



def test_abacus_raw_extrapolation_sbatch_uses_raw_reader_models_and_dpa4_preflight():
    sbatch = SCRIPT_ROOT / "abacus-raw-extrapolation-eval.sbatch"
    text = sbatch.read_text()

    assert "#SBATCH --partition=16V100" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "evaluate_abacus_raw_extrapolation.py" in text
    assert "conda-envs/mace-dpa4-cu126/bin/python" in text
    assert "abacus_rpbe_last10_seed_pure_sella_selected20" in text
    assert '--model "${name}=mace:${model}"' in text
    assert "run_mace_eval mace_mh1" in text
    assert "oc20_usemppbe" in text
    assert "run_mace_eval mace_omat50k" in text
    assert "oc20neb" in text
    assert "run_mace_eval mace_fps5k" in text
    assert "dpa4_preflight" in text
    assert "Unknown model type: dpa4" in text
    assert "--write-label-extxyz" in text


def test_summarize_fullcase200_ef20k_matrix_reports_case_config_and_metrics(tmp_path):
    summarizer = load_script("summarize_fullcase200_ef20k_matrix.py")
    root = tmp_path / "matrix_compile_a65aa53_20260712"
    case_dir = root / "compile"
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (root / "manifest.json").write_text(
        '{\n  "batch_size": 8,\n  "target_steps": 20000,\n  "train_size": 5000,\n  "max_num_epochs": 32,\n  "max_num_updates": 20000,\n  "stage_two_start_update": 15000,\n  "stage1_energy_weight": 1.0,\n  "stage1_forces_weight": 100.0,\n  "stage2_energy_weight": 100.0,\n  "stage2_forces_weight": 1.0,\n  "compile_setup_gate": "strict",\n  "compile_max_cache_entries": 2,\n  "compile_parity_gradients": "False",\n  "compile_parity_check_strict": "False"\n}\n'
    )
    (log_dir / "train.log").write_text(
        "2026-07-12 10:20:00.000 INFO: Epoch 0: head: Default, loss=0.12, "
        "MAE_E_per_atom=  150.00 meV, MAE_F=   40.00 meV / A\n"
        "2026-07-12 10:30:00.000 INFO: Epoch 6: head: Default, loss=0.10, "
        "MAE_E_per_atom=  120.00 meV, MAE_F=   35.00 meV / A\n"
    )
    (case_dir / "nvdmon_job-1_compile.log").write_text(
        "# gpu pwr gtemp mtemp sm mem enc dec mclk pclk pviol tviol fb bar1 ccpm\n"
        "0 0 250 45 40 50 20 0 0 877 1380 0 0 0 0 1234\n"
    )

    rows = summarizer.summarize_roots([root])

    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "compile"
    assert row["compile"] is True
    assert row["cueq"] is False
    assert row["hybrid_muon"] is False
    assert row["effective_updates"] == 20000
    assert row["stage_two_start_update"] == 15000
    assert row["final_epoch"] == 6
    assert row["mae_e_mev_atom"] == 120.0
    assert row["mae_f_mev_a"] == 35.0
    assert row["mean_seconds_per_epoch"] == 100.0
    assert row["max_fb_memory_mb"] == 1234
    assert row["compile_setup_gate"] == "strict"
    assert row["compile_max_cache_entries"] == 2
