from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_FILE = Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz")
DEFAULT_OUTPUT_ROOT = Path("runs/recio8k_20k_cueq061_adam_fxonly_scaling")
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TRAIN_SIZE = 7600
DEFAULT_TARGET_STEPS = 20_000
REFERENCE_MODEL = {
    "num_channels": 64,
    "max_L": 1,
    "num_interactions": 2,
    "correlation": 3,
    "r_max": 5.0,
}


@dataclass(frozen=True)
class ScalingCase:
    name: str
    sweep: str
    output_root: Path
    partition: str
    qos: str
    params: dict[str, Any]
    max_num_epochs: int
    start_swa: int
    eval_interval: int
    conda_env: str = "mace_env"

    @property
    def path(self) -> Path:
        return self.output_root / self.name


def _epochs_for_target_steps(
    *, batch_size: int, train_size: int, target_steps: int
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return max(2, math.ceil(target_steps * batch_size / train_size))


def _start_swa(max_num_epochs: int) -> int:
    if max_num_epochs <= 2:
        return 1
    return min(max_num_epochs - 1, max(1, int(max_num_epochs * 0.75)))


def _eval_interval(max_num_epochs: int) -> int:
    return max(1, max_num_epochs // 3)


def _base_params(
    *,
    name: str,
    batch_size: int,
    train_file: Path,
    target_steps: int,
    train_size: int,
    model_overrides: dict[str, Any] | None = None,
    scheduler: str = "ReduceLROnPlateau",
    stage_two: bool = True,
    lr_scheduler_interval: str = "auto",
) -> tuple[dict[str, Any], int, int, int]:
    max_num_epochs = _epochs_for_target_steps(
        batch_size=batch_size, train_size=train_size, target_steps=target_steps
    )
    start_swa = _start_swa(max_num_epochs)
    eval_interval = _eval_interval(max_num_epochs)

    model_params = dict(REFERENCE_MODEL)
    if model_overrides:
        model_params.update(model_overrides)

    params: dict[str, Any] = {
        "name": name,
        "train_file": str(train_file),
        "valid_fraction": 0.05,
        "test_file": str(train_file),
        "E0s": "average",
        "energy_key": "energy",
        "forces_key": "forces",
        "model": "ScaleShiftMACE",
        **model_params,
        "batch_size": batch_size,
        "valid_batch_size": min(32, max(1, batch_size)),
        "max_num_epochs": max_num_epochs,
        "patience": 999,
        "eval_interval": eval_interval,
        "error_table": "PerAtomMAE",
        "default_dtype": "float32",
        "train_amp_dtype": "none",
        "train_tf32": True,
        "device": "cuda",
        "seed": 456,
        "shuffle": False,
        "enable_cueq": True,
        "cueq_optimize_all": True,
        "cueq_optimize_linear": True,
        "cueq_optimize_channelwise": True,
        "cueq_optimize_symmetric": True,
        "cueq_optimize_fctp": True,
        "cueq_conv_fusion": True,
        "optimizer": "adam",
        "scheduler": scheduler,
        "lr_scheduler_interval": lr_scheduler_interval,
        "lr_factor": 0.8,
        "scheduler_patience": 50,
        "lr_scheduler_gamma": 0.9993,
        "lr_wsd_warmup_steps": 0,
        "lr_wsd_warmup_ratio": 0.03,
        "lr_wsd_warmup_start_factor": 0.1,
        "lr_wsd_stop_lr_ratio": 1.0e-3,
        "lr_wsd_decay_phase_ratio": 0.1,
        "lr_wsd_decay_type": "inverse_linear",
        "hybrid_muon_mode": "2d",
        "hybrid_muon_routing": "mace",
        "hybrid_muon_lr_factor": 0.1,
        "hybrid_muon_weight_decay": 0.0,
        "hybrid_muon_magma_lite": False,
        "lr": 0.01,
        "weight_decay": 5e-7,
        "amsgrad": True,
        "loss": "weighted",
        "energy_weight": 1.0,
        "forces_weight": 100.0,
        "swa": True,
        "start_swa": start_swa,
        "swa_energy_weight": 1000.0,
        "swa_forces_weight": 100.0,
        "swa_lr": 0.001,
        "clip_grad": 10.0,
        "restart_latest": False,
        "distributed": False,
        "edge_force_compile": True,
        "edge_force_compile_mode": "default",
        "edge_force_compile_tracing_mode": "symbolic",
        "edge_force_compile_graph": False,
        "edge_force_compile_dynamic": True,
        "edge_force_compile_shape_padding": True,
        "edge_force_compile_max_fusion_size": 8,
        "edge_force_compile_spherical_harmonics": "polynomial",
        "edge_force_compile_setup_gate": "strict",
        "edge_force_compile_cache_hit_gate": False,
        "edge_force_compile_atol": 0.02,
        "edge_force_compile_rtol": 0.0002,
        "edge_force_compile_cache_policy": "dynamic",
        "edge_force_compile_min_repeats": 2,
        "edge_force_compile_bucket_atoms": "512,768",
        "edge_force_compile_bucket_edges": "8192,16384",
        "edge_force_compile_bucket_margin": 0.0,
        "edge_force_compile_parity_check_interval": 0,
        "edge_force_compile_parity_check_gradients": True,
        "edge_force_compile_parity_check_strict": True,
        "edge_force_compile_allow_fallback": False,
    }
    if not stage_two:
        params["swa"] = False
        for key in ("start_swa", "swa_energy_weight", "swa_forces_weight", "swa_lr"):
            params.pop(key, None)
    return params, max_num_epochs, start_swa, eval_interval


def build_batch_case(
    *,
    batch_size: int,
    output_root: Path,
    partition: str,
    qos: str,
    target_steps: int = DEFAULT_TARGET_STEPS,
    train_size: int = DEFAULT_TRAIN_SIZE,
    train_file: Path = DEFAULT_TRAIN_FILE,
    scheduler: str = "ReduceLROnPlateau",
    stage_two: bool = True,
    lr_scheduler_interval: str = "auto",
) -> ScalingCase:
    name = f"bs{batch_size:04d}"
    params, max_num_epochs, start_swa, eval_interval = _base_params(
        name=f"recio8k_scaling_{name}",
        batch_size=batch_size,
        train_file=train_file,
        target_steps=target_steps,
        train_size=train_size,
        scheduler=scheduler,
        stage_two=stage_two,
        lr_scheduler_interval=lr_scheduler_interval,
    )
    return ScalingCase(
        name=name,
        sweep="batch",
        output_root=Path(output_root),
        partition=partition,
        qos=qos,
        params=params,
        max_num_epochs=max_num_epochs,
        start_swa=start_swa,
        eval_interval=eval_interval,
    )


def build_model_cases(
    *,
    batch_size: int,
    output_root: Path,
    partition: str,
    qos: str,
    target_steps: int = DEFAULT_TARGET_STEPS,
    train_size: int = DEFAULT_TRAIN_SIZE,
    train_file: Path = DEFAULT_TRAIN_FILE,
    scheduler: str = "ReduceLROnPlateau",
    stage_two: bool = True,
    lr_scheduler_interval: str = "auto",
) -> list[ScalingCase]:
    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {}),
        ("channels0128", {"num_channels": 128}),
        ("channels0256", {"num_channels": 256}),
        ("maxL0", {"max_L": 0}),
        ("maxL2", {"max_L": 2}),
        ("interactions1", {"num_interactions": 1}),
        ("interactions3", {"num_interactions": 3}),
        ("rmax6", {"r_max": 6.0}),
        ("rmax7", {"r_max": 7.0}),
        ("corr2", {"correlation": 2}),
    ]
    cases = []
    for case_name, overrides in variants:
        params, max_num_epochs, start_swa, eval_interval = _base_params(
            name=f"recio8k_model_{case_name}_bs{batch_size:04d}",
            batch_size=batch_size,
            train_file=train_file,
            target_steps=target_steps,
            train_size=train_size,
            model_overrides=overrides,
            scheduler=scheduler,
            stage_two=stage_two,
            lr_scheduler_interval=lr_scheduler_interval,
        )
        cases.append(
            ScalingCase(
                name=case_name,
                sweep="model",
                output_root=Path(output_root),
                partition=partition,
                qos=qos,
                params=params,
                max_num_epochs=max_num_epochs,
                start_swa=start_swa,
                eval_interval=eval_interval,
            )
        )
    return cases


def build_ablation_cases(
    *,
    batch_size: int,
    output_root: Path,
    partition: str,
    qos: str,
    target_steps: int = DEFAULT_TARGET_STEPS,
    train_size: int = DEFAULT_TRAIN_SIZE,
    train_file: Path = DEFAULT_TRAIN_FILE,
    scheduler: str = "ReduceLROnPlateau",
    stage_two: bool = True,
    lr_scheduler_interval: str = "auto",
) -> list[ScalingCase]:
    cases = []
    for optimizer in ("adam", "hybrid_muon"):
        for enable_cueq in (False, True):
            for edge_force_compile in (False, True):
                name = "_".join(
                    [
                        "muon" if optimizer == "hybrid_muon" else "adam",
                        "cueq" if enable_cueq else "nocueq",
                        "compile" if edge_force_compile else "eager",
                    ]
                )
                params, max_num_epochs, start_swa, eval_interval = _base_params(
                    name=f"recio8k_ablation_{name}_bs{batch_size:04d}",
                    batch_size=batch_size,
                    train_file=train_file,
                    target_steps=target_steps,
                    train_size=train_size,
                    scheduler=scheduler,
                    stage_two=stage_two,
                    lr_scheduler_interval=lr_scheduler_interval,
                )
                params["optimizer"] = optimizer
                params["enable_cueq"] = enable_cueq
                params["edge_force_compile"] = edge_force_compile
                conda_env = (
                    "mace_develop"
                    if optimizer == "hybrid_muon" or enable_cueq or edge_force_compile
                    else "mace_env"
                )
                if optimizer == "hybrid_muon":
                    params.update(
                        {
                            "hybrid_muon_mode": "2d",
                            "hybrid_muon_routing": "mace",
                            "hybrid_muon_lr_factor": 0.1,
                            "train_tf32": True,
                            "train_amp_dtype": "none",
                        }
                    )
                cases.append(
                    ScalingCase(
                        name=name,
                        sweep="ablation",
                        output_root=Path(output_root),
                        partition=partition,
                        qos=qos,
                        params=params,
                        max_num_epochs=max_num_epochs,
                        start_swa=start_swa,
                        eval_interval=eval_interval,
                        conda_env=conda_env,
                    )
                )
    return cases


def _bool_arg(name: str, enabled: bool) -> list[str]:
    return [f"--{name}" if enabled else f"--no-{name}"]


def _format_cli_args(case: ScalingCase) -> list[str]:
    args: list[str] = []
    bool_optional_keys = {
        "train_tf32",
        "cueq_optimize_all",
        "cueq_optimize_linear",
        "cueq_optimize_channelwise",
        "cueq_optimize_symmetric",
        "cueq_optimize_fctp",
        "cueq_conv_fusion",
        "edge_force_compile_graph",
        "edge_force_compile_dynamic",
        "edge_force_compile_shape_padding",
        "edge_force_compile_cache_hit_gate",
        "edge_force_compile_parity_check_gradients",
        "edge_force_compile_parity_check_strict",
        "edge_force_compile_allow_fallback",
    }
    store_true_keys = {
        "amsgrad",
        "distributed",
        "edge_force_compile",
        "hybrid_muon_magma_lite",
        "restart_latest",
        "swa",
    }
    value_bool_keys = {
        "enable_cueq",
        "shuffle",
    }
    for key, value in case.params.items():
        if key in bool_optional_keys:
            args.extend(_bool_arg(key, bool(value)))
            continue
        if key in store_true_keys:
            if bool(value):
                args.append(f"--{key}")
            continue
        if key in value_bool_keys:
            args.append(f"--{key}={bool(value)}")
            continue
        args.append(f"--{key}={value}")

    args.extend(
        [
            f"--work_dir={case.path}",
            f"--log_dir={case.path / 'logs'}",
            f"--model_dir={case.path / 'models'}",
            f"--checkpoints_dir={case.path / 'checkpoints'}",
            f"--results_dir={case.path / 'results'}",
        ]
    )
    return args


def _render_sbatch(case: ScalingCase) -> str:
    cli_args = " \\\n  ".join(shlex.quote(arg) for arg in _format_cli_args(case))
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=recio8k-{case.name}
#SBATCH --partition={case.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos={case.qos}
#SBATCH --output={case.path}/%x-%j.out
#SBATCH --error={case.path}/%x-%j.err

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
nvidia-smi dmon -s pucvmte -o T > {case.path}/nvdmon_job-${{SLURM_JOB_ID}}.log &
source /opt/envs/anaconda3.env
conda activate "${{MACE_CONDA_ENV:-mace_develop}}"

cd {shlex.quote(str(REPO_ROOT))}
export PYTHONPATH={shlex.quote(str(REPO_ROOT))}:${{PYTHONPATH:-}}

mkdir -p {shlex.quote(str(case.path / 'logs'))} \\
  {shlex.quote(str(case.path / 'models'))} \\
  {shlex.quote(str(case.path / 'checkpoints'))} \\
  {shlex.quote(str(case.path / 'results'))}

python -m mace.cli.run_train \\
  {cli_args}
"""


def write_case(case: ScalingCase) -> None:
    case.path.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "models", "checkpoints", "results"):
        (case.path / subdir).mkdir(exist_ok=True)

    manifest = {
        "name": case.name,
        "sweep": (
            "batch"
            if case.name.startswith("bs")
            else "ablation"
            if case.name.startswith(("adam_", "muon_"))
            else "model"
        ),
        "batch_size": case.params["batch_size"],
        "max_num_epochs": case.max_num_epochs,
        "start_swa": case.start_swa,
        "eval_interval": case.eval_interval,
        "partition": case.partition,
        "qos": case.qos,
        "params": case.params,
    }
    (case.path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (case.path / "run.sbatch").write_text(_render_sbatch(case))

    submit = case.output_root / "submit_all.sh"
    if not submit.exists():
        submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        submit.chmod(0o755)
    with submit.open("a") as handle:
        handle.write(f"sbatch {shlex.quote(str(case.path / 'run.sbatch'))}\n")


def write_cases(cases: list[ScalingCase]) -> None:
    if not cases:
        return
    output_root = cases[0].output_root
    output_root.mkdir(parents=True, exist_ok=True)
    submit = output_root / "submit_all.sh"
    if submit.exists():
        submit.unlink()
    for case in cases:
        write_case(case)


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.replace(",", " ").split()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RECIO8k training speed, memory, and accuracy benchmark cases."
    )
    parser.add_argument("--mode", choices=("batch", "model", "ablation"), default="batch")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--partition", default="4V100PX")
    parser.add_argument("--qos", default="rush-1o2gpu")
    parser.add_argument(
        "--scheduler",
        choices=("ReduceLROnPlateau", "WSD", "ExponentialLR"),
        default="ReduceLROnPlateau",
        help="Scheduler to write into generated training cases.",
    )
    parser.add_argument(
        "--lr-scheduler-interval",
        choices=("auto", "epoch", "step"),
        default="auto",
        help="Scheduler step interval to write into generated training cases.",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Do not write MACE Stage Two/SWA flags; useful for WSD/Muon ablations.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="Comma or space separated list for --mode=batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Best batch size to use for --mode=model.",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        cases = [
            build_batch_case(
                batch_size=batch_size,
                output_root=args.output_root,
                partition=args.partition,
                qos=args.qos,
                target_steps=args.target_steps,
                train_size=args.train_size,
                train_file=args.train_file,
                scheduler=args.scheduler,
                stage_two=not args.single_stage,
                lr_scheduler_interval=args.lr_scheduler_interval,
            )
            for batch_size in args.batch_sizes
        ]
    elif args.mode == "model":
        cases = build_model_cases(
            batch_size=args.batch_size,
            output_root=args.output_root,
            partition=args.partition,
            qos=args.qos,
            target_steps=args.target_steps,
            train_size=args.train_size,
            train_file=args.train_file,
            scheduler=args.scheduler,
            stage_two=not args.single_stage,
            lr_scheduler_interval=args.lr_scheduler_interval,
        )
    else:
        cases = build_ablation_cases(
            batch_size=args.batch_size,
            output_root=args.output_root,
            partition=args.partition,
            qos=args.qos,
            target_steps=args.target_steps,
            train_size=args.train_size,
            train_file=args.train_file,
            scheduler=args.scheduler,
            stage_two=not args.single_stage,
            lr_scheduler_interval=args.lr_scheduler_interval,
        )

    write_cases(cases)
    print(f"Wrote {len(cases)} cases under {args.output_root}")
    print(f"Review then submit with: bash {args.output_root / 'submit_all.sh'}")


if __name__ == "__main__":
    main()
