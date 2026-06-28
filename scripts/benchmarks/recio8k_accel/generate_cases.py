from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


CASES = {
    "baseline_fp32_adam_cueq": {},
    "fp32_hybrid_muon_cueq": {
        "optimizer": "hybrid_muon",
        "hybrid_muon_lr_factor": 0.1,
    },
    "compile_fp32_adam_cueq": {
        "train_compile": True,
        "train_compile_allow_fallback": True,
    },
    "bf16_adam_cueq": {"train_amp_dtype": "bf16"},
    "bf16_hybrid_muon_cueq": {
        "optimizer": "hybrid_muon",
        "hybrid_muon_lr_factor": 0.1,
        "train_amp_dtype": "bf16",
    },
}


def _write_sbatch(template: Path, destination: Path, partition: str, qos: str) -> None:
    script = template.read_text()
    script = script.replace("__PARTITION__", partition).replace("__QOS__", qos)
    destination.write_text(script)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--partition", default="4V100")
    parser.add_argument("--qos", default="improper-gpu")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    template = Path(__file__).with_name("mace-recio8k-template.sbatch")
    base_config = yaml.safe_load((source / "config.yaml").read_text())

    for name, overrides in CASES.items():
        case_dir = output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "train.xyz", case_dir / "train.xyz")
        shutil.copy2(source / "les_config.yaml", case_dir / "les_config.yaml")
        _write_sbatch(
            template,
            case_dir / "mace-recio8k.sbatch",
            partition=args.partition,
            qos=args.qos,
        )

        config = dict(base_config)
        config.update(overrides)
        config["max_num_epochs"] = args.epochs
        config["start_swa"] = max(1, int(args.epochs * 0.75))
        config["restart_latest"] = False
        config["distributed"] = False
        config["name"] = f"RECIO-8k-{name}"
        (case_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
