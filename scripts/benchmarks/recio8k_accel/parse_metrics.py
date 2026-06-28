from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+):.*MAE_E_per_atom=\s*(?P<mae_e>[0-9.]+) meV, "
    r"MAE_F=\s*(?P<mae_f>[0-9.]+)"
)


def parse_log(path: Path) -> dict:
    epochs = []
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "mae_e_mev_atom": float(match.group("mae_e")),
                    "mae_f_mev_a": float(match.group("mae_f")),
                }
            )
    return {"log": str(path), "epochs": epochs, "last": epochs[-1] if epochs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    summaries = {}
    for root_name in args.roots:
        root = Path(root_name)
        for log in root.glob("*/logs/*.log"):
            summaries[f"{root.name}/{log.parents[1].name}"] = parse_log(log)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
