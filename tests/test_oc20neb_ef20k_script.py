from pathlib import Path

SCRIPT = Path("scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch")


def test_oc20neb_20k_defaults_to_non_compile_adamw_muon_matrix():
    text = SCRIPT.read_text()

    assert (
        "CASES=${MACE_OC20NEB_CASES:-adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon}"
        in text
    )
    assert "BASE_OPTIMIZER=${MACE_OC20NEB_BASE_OPTIMIZER:-adamw}" in text
    assert '--optimizer="${BASE_OPTIMIZER}"' in text
    assert '"base_optimizer": "${BASE_OPTIMIZER}"' in text
    assert '"git_commit": "${SOURCE_GIT_COMMIT}"' in text
    assert "ENFORCE_STATIC_SOURCE=${MACE_OC20NEB_ENFORCE_STATIC_SOURCE:-True}" in text
    assert "assert_static_source" in text
    assert "Repository HEAD changed during job" in text
    assert "run_selected_case adamw" in text
    assert "run_selected_case hybrid_muon" in text
    assert "run_selected_case cueq_adamw" in text
    assert "run_selected_case cueq_hybrid_muon" in text

    default_line = next(
        line for line in text.splitlines() if line.startswith("CASES=${MACE_OC20NEB_CASES:-")
    )
    assert "compile" not in default_line


def test_oc20neb_20k_uses_smooth_loss_prefactor_schedule_by_default():
    text = SCRIPT.read_text()

    assert "LOSS_PREFACTOR_SCHEDULE=${MACE_OC20NEB_LOSS_PREFACTOR_SCHEDULE:-step_linear}" in text
    assert "LOSS_PREFACTOR_START_UPDATE=${MACE_OC20NEB_LOSS_PREFACTOR_START_UPDATE:-${START_STAGE_TWO_UPDATE}}" in text
    assert "LOSS_PREFACTOR_END_UPDATE=${MACE_OC20NEB_LOSS_PREFACTOR_END_UPDATE:-${TARGET_STEPS}}" in text
    assert '"loss_prefactor_schedule": "${LOSS_PREFACTOR_SCHEDULE}"' in text
    assert '"loss_prefactor_start_update": ${LOSS_PREFACTOR_START_UPDATE}' in text
    assert '"loss_prefactor_end_update": ${LOSS_PREFACTOR_END_UPDATE}' in text
    assert '--loss_prefactor_schedule="${LOSS_PREFACTOR_SCHEDULE}"' in text
    assert '--loss_prefactor_start_update="${LOSS_PREFACTOR_START_UPDATE}"' in text
    assert '--loss_prefactor_end_update="${LOSS_PREFACTOR_END_UPDATE}"' in text
