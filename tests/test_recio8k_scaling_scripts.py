from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


RUN_WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmarks"
    / "recio8k_accel"
    / "run_edge_force_cache_policy_sai.sh"
)


def load_script(name: str):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / name
    )
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_case_keeps_reference_settings_and_enables_stage_two(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")

    case = generator.build_batch_case(
        batch_size=16,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
    )

    assert case.name == "bs0016"
    assert case.max_num_epochs == 43
    assert case.start_swa == 32
    assert case.eval_interval == 14
    assert case.params["optimizer"] == "adam"
    assert case.params["scheduler"] == "ReduceLROnPlateau"
    assert case.params["lr_wsd_warmup_ratio"] == 0.03
    assert case.params["lr_wsd_decay_type"] == "inverse_linear"
    assert case.params["enable_cueq"] is True
    assert case.params["cueq_optimize_all"] is True
    assert case.params["cueq_optimize_linear"] is True
    assert case.params["cueq_optimize_fctp"] is True
    assert case.params["cueq_optimize_channelwise"] is True
    assert case.params["cueq_optimize_symmetric"] is True
    assert case.params["edge_force_compile"] is True
    assert case.params["num_channels"] == 64
    assert case.params["max_L"] == 1
    assert case.params["num_interactions"] == 2
    assert case.params["correlation"] == 3
    assert case.params["r_max"] == 5.0
    assert case.params["swa"] is True
    assert case.params["energy_weight"] == 1.0
    assert case.params["forces_weight"] == 100.0
    assert case.params["swa_energy_weight"] == 1000.0
    assert case.params["swa_forces_weight"] == 100.0


def test_write_case_outputs_manifest_and_sbatch_without_submitting(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")
    case = generator.build_batch_case(
        batch_size=32,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
    )

    generator.write_case(case)

    manifest = json.loads((tmp_path / "bs0032" / "manifest.json").read_text())
    sbatch = (tmp_path / "bs0032" / "run.sbatch").read_text()
    submit = (tmp_path / "submit_all.sh").read_text()

    assert manifest["batch_size"] == 32
    assert manifest["max_num_epochs"] == 85
    assert manifest["start_swa"] == 63
    assert "--swa" in sbatch
    assert "--start_swa=63" in sbatch
    assert "--batch_size=32" in sbatch
    assert "--edge_force_compile" in sbatch
    assert "--enable_cueq=True" in sbatch
    assert "--cueq_optimize_all" in sbatch
    assert "--cueq_optimize_fctp" in sbatch
    assert "sbatch" in submit
    assert "Submitted" not in submit


def test_model_sweep_uses_one_factor_changes_around_reference(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")

    cases = generator.build_model_cases(
        batch_size=32,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
    )

    names = {case.name for case in cases}
    assert "baseline" in names
    assert "channels0128" in names
    assert "maxL2" in names
    assert "rmax7" in names
    assert "corr2" in names
    assert len(cases) == 10

    baseline = next(case for case in cases if case.name == "baseline")
    for case in cases:
        changed = [
            key
            for key in ("num_channels", "max_L", "num_interactions", "r_max", "correlation")
            if case.params[key] != baseline.params[key]
        ]
        assert len(changed) <= 1


def test_ablation_sweep_crosses_optimizer_cueq_and_compile(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")

    cases = generator.build_ablation_cases(
        batch_size=16,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
    )

    assert len(cases) == 8
    combos = {
        (
            case.params["optimizer"],
            case.params["enable_cueq"],
            case.params["edge_force_compile"],
        )
        for case in cases
    }
    assert combos == {
        (optimizer, cueq, compile_enabled)
        for optimizer in ("adam", "hybrid_muon")
        for cueq in (False, True)
        for compile_enabled in (False, True)
    }

    adam_cueq_compile_case = next(
        case for case in cases
        if case.params["optimizer"] == "adam"
        and case.params["enable_cueq"]
        and case.params["edge_force_compile"]
    )
    adam_compile_case = next(
        case for case in cases
        if case.params["optimizer"] == "adam"
        and not case.params["enable_cueq"]
        and case.params["edge_force_compile"]
    )
    assert adam_cueq_compile_case.conda_env == "mace_develop"
    assert adam_compile_case.conda_env == "mace_develop"

    muon_case = next(
        case for case in cases
        if case.params["optimizer"] == "hybrid_muon"
        and case.params["enable_cueq"]
        and case.params["edge_force_compile"]
    )
    assert muon_case.name == "muon_cueq_compile"
    assert muon_case.params["hybrid_muon_mode"] == "2d"
    assert muon_case.params["hybrid_muon_routing"] == "mace"
    assert muon_case.params["hybrid_muon_lr_factor"] == 0.1
    assert muon_case.params["train_tf32"] is True
    assert muon_case.params["train_amp_dtype"] == "none"


def test_ablation_case_sbatch_uses_mace_develop_and_hybrid_muon_flags(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")
    case = next(
        case for case in generator.build_ablation_cases(
            batch_size=16,
            output_root=tmp_path,
            partition="4V100PX",
            qos="rush-1o2gpu",
            target_steps=20_000,
        )
        if case.name == "muon_cueq_compile"
    )

    generator.write_case(case)

    sbatch = (tmp_path / "muon_cueq_compile" / "run.sbatch").read_text()
    manifest = json.loads((tmp_path / "muon_cueq_compile" / "manifest.json").read_text())

    assert 'conda activate "${MACE_CONDA_ENV:-mace_develop}"' in sbatch
    assert "#SBATCH --ntasks-per-node" not in sbatch
    assert "mps_mapping" not in sbatch
    assert "--hybrid_muon_mode=2d" in sbatch
    assert "--hybrid_muon_routing=mace" in sbatch
    assert "--train_tf32" in sbatch
    assert "--train_amp_dtype=none" in sbatch
    assert "--scheduler=ReduceLROnPlateau" in sbatch
    assert "--lr_wsd_decay_type=inverse_linear" in sbatch
    assert "--edge_force_compile" in sbatch
    assert "--enable_cueq=True" in sbatch
    assert "--cueq_optimize_all" in sbatch
    assert "--cueq_optimize_fctp" in sbatch
    assert "--amsgrad=True" not in sbatch
    assert "  --no-edge_force_compile \\" not in sbatch
    assert manifest["sweep"] == "ablation"


def test_edge_force_cache_policy_wrapper_defaults_to_safe_fx_dynamic_compile():
    wrapper = RUN_WRAPPER.read_text()

    assert 'EDGE_FORCE_GRAPH:-False' in wrapper
    assert 'EDGE_FORCE_REQUIRE_INDUCTOR_ACK:-False' in wrapper
    assert '--edge_force_compile_cache_policy="${EDGE_FORCE_CACHE_POLICY:-dynamic}"' in wrapper
    assert '--no-edge_force_compile_graph' in wrapper
    assert '--train_amp_dtype="${TRAIN_AMP_DTYPE:-none}"' in wrapper
    assert '--scheduler="${SCHEDULER:-ReduceLROnPlateau}"' in wrapper
    assert '--lr_wsd_decay_type="${LR_WSD_DECAY_TYPE:-inverse_linear}"' in wrapper


def test_edge_force_epoch_profile_sbatch_uses_sai_safe_defaults():
    sbatch = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "edge-force-training-epoch-profile.sbatch"
    ).read_text()

    assert "#SBATCH --ntasks-per-node" not in sbatch
    assert "#SBATCH --cpus-per-task" not in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert 'conda activate "${MACE_EDGE_EPOCH_CONDA_ENV:-mace_develop}"' in sbatch
    assert "MACE_EDGE_EPOCH_RESET_COMPILE_STATE" in sbatch
    assert "--edge-reset-compile-state" in sbatch
    assert "MACE_EDGE_EPOCH_CLEAR_CACHE_ON_MISS" in sbatch
    assert "--edge-clear-cache-on-miss" in sbatch
    assert "MACE_EDGE_EPOCH_SKIP_TRAINING_STEP" in sbatch
    assert "--skip-training-step" in sbatch
    assert "MACE_EDGE_EPOCH_DIAGNOSE_GATE_STATE" in sbatch
    assert "--edge-diagnose-gate-state" in sbatch


def test_generator_writes_wsd_scheduler_cases(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")
    case = generator.build_batch_case(
        batch_size=16,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
        scheduler="WSD",
    )

    generator.write_case(case)

    sbatch = (tmp_path / "bs0016" / "run.sbatch").read_text()
    manifest = json.loads((tmp_path / "bs0016" / "manifest.json").read_text())

    assert manifest["params"]["scheduler"] == "WSD"
    assert "--scheduler=WSD" in sbatch
    assert "--lr_wsd_warmup_ratio=0.03" in sbatch
    assert "--lr_wsd_stop_lr_ratio=0.001" in sbatch
    assert "--lr_wsd_decay_type=inverse_linear" in sbatch


def test_generator_writes_wsd_single_stage_per_step_case(tmp_path):
    generator = load_script("generate_recio8k_scaling_cases.py")
    case = generator.build_batch_case(
        batch_size=16,
        output_root=tmp_path,
        partition="4V100PX",
        qos="rush-1o2gpu",
        target_steps=20_000,
        scheduler="WSD",
        stage_two=False,
        lr_scheduler_interval="step",
    )

    generator.write_case(case)

    sbatch = (tmp_path / "bs0016" / "run.sbatch").read_text()
    manifest = json.loads((tmp_path / "bs0016" / "manifest.json").read_text())

    assert manifest["params"]["scheduler"] == "WSD"
    assert manifest["params"]["swa"] is False
    assert manifest["params"]["lr_scheduler_interval"] == "step"
    assert "--scheduler=WSD" in sbatch
    assert "--lr_scheduler_interval=step" in sbatch
    assert "--swa" not in sbatch
    assert "--start_swa" not in sbatch


def test_summary_reports_completed_metrics_memory_and_hot_step_time(tmp_path):
    summary = load_script("summarize_recio8k_scaling.py")
    case_dir = tmp_path / "bs0016"
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "results").mkdir()
    (case_dir / "manifest.json").write_text(
        json.dumps({"name": "bs0016", "batch_size": 16, "sweep": "batch"})
    )
    (case_dir / "logs" / "run.log").write_text(
        "2026-07-01 15:18:58.480 INFO: Edge-force compile epoch 0 summary: "
        "steps=475, compiled=475, cache_hits=474, new_compiles=1, fallbacks=0, "
        "runtime_recompiles=0, compile_setup_seconds=67.733, opt_step_seconds=77.201\n"
        "2026-07-01 15:19:09.332 INFO: Edge-force compile epoch 1 summary: "
        "steps=475, compiled=475, cache_hits=475, new_compiles=0, fallbacks=0, "
        "runtime_recompiles=0, compile_setup_seconds=0.000, opt_step_seconds=9.419\n"
        "2026-07-01 15:23:57.679 INFO: Epoch 28: head: Default, loss=0.65390795, "
        "MAE_E_per_atom=   80.91 meV, MAE_F=  294.05 meV / A\n"
        "2026-07-01 15:26:16.378 INFO: Training complete\n"
        "| valid_Default |          80.9      |        294.0    |        11.04     |\n"
        "| Default_Default_Default |          88.6      |        275.8    |        10.60     |\n"
        "2026-07-01 15:26:46.715 INFO: Done\n"
    )
    (case_dir / "nvdmon_job-123.log").write_text(
        "# header\n"
        " 04:12:44       0    130     37     37     66     17      0      0      -      -    877   1530      0      0   4198      5      0    248     34      0      0      0\n"
        " 04:12:45       0    100     37     37     65     18      0      0      -      -    877   1530      0      0   4201      5      0    268     39      0      0      0\n"
    )

    result = summary.summarize_case(case_dir)

    assert result["status"] == "completed"
    assert result["max_fb_memory_mb"] == 4201
    assert result["first_compile_setup_seconds"] == pytest.approx(67.733)
    assert result["mean_hot_opt_step_seconds"] == pytest.approx(9.419)
    assert result["last_valid_mae_e_mev_atom"] == pytest.approx(80.91)
    assert result["final_valid_mae_f_mev_a"] == pytest.approx(294.0)
    assert result["final_test_mae_e_mev_atom"] == pytest.approx(88.6)


def test_summary_marks_oom_from_error_text(tmp_path):
    summary = load_script("summarize_recio8k_scaling.py")
    case_dir = tmp_path / "bs1024"
    case_dir.mkdir()
    (case_dir / "manifest.json").write_text(
        json.dumps({"name": "bs1024", "batch_size": 1024, "sweep": "batch"})
    )
    (case_dir / "slurm-1.out").write_text("torch.cuda.OutOfMemoryError: CUDA out of memory\n")

    result = summary.summarize_case(case_dir)

    assert result["status"] == "oom"
