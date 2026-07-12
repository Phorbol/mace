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
    assert "run_selected_case adamw" in text
    assert "run_selected_case hybrid_muon" in text
    assert "run_selected_case cueq_adamw" in text
    assert "run_selected_case cueq_hybrid_muon" in text

    default_line = next(
        line for line in text.splitlines() if line.startswith("CASES=${MACE_OC20NEB_CASES:-")
    )
    assert "compile" not in default_line
