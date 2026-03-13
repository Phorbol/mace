import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple


def parse_args():
    p = argparse.ArgumentParser(
        description="Fail if candidate latency regresses versus baseline beyond threshold."
    )
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Allowed regression threshold in percent",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["eager", "compile"],
        default=["eager", "compile"],
        help="Only gate selected profiling modes",
    )
    p.add_argument(
        "--min-atoms",
        type=int,
        default=0,
        help="Skip rows with fewer atoms than this threshold",
    )
    return p.parse_args()


def load_rows(path: str) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["structure_name"], r["size"], r["mode"])
            rows[key] = {
                "num_atoms": int(r.get("num_atoms", 0)),
                "train_mean_ms": float(r["train_mean_ms"]),
                "infer_mean_ms": float(r["infer_mean_ms"]),
            }
    return rows


def pct_change(base: float, cand: float) -> float:
    if abs(base) < 1e-12:
        return 0.0 if abs(cand) < 1e-12 else float("inf")
    return (cand - base) / base * 100.0


def main():
    args = parse_args()
    baseline = load_rows(args.baseline_csv)
    candidate = load_rows(args.candidate_csv)
    selected_modes = set(args.modes)

    failures = []
    checked = 0
    for key, b in baseline.items():
        mode = key[2]
        if mode not in selected_modes:
            continue
        if args.min_atoms > 0 and b.get("num_atoms", 0) < args.min_atoms:
            continue
        if key not in candidate:
            failures.append((key, "missing_in_candidate", b["train_mean_ms"], float("nan"), float("inf")))
            continue
        checked += 1
        c = candidate[key]
        train_reg = pct_change(b["train_mean_ms"], c["train_mean_ms"])
        infer_reg = pct_change(b["infer_mean_ms"], c["infer_mean_ms"])
        if train_reg > args.threshold_pct:
            failures.append((key, "train_mean_ms", b["train_mean_ms"], c["train_mean_ms"], train_reg))
        if infer_reg > args.threshold_pct:
            failures.append((key, "infer_mean_ms", b["infer_mean_ms"], c["infer_mean_ms"], infer_reg))

    print(f"[perf_gate] checked_rows={checked} threshold={args.threshold_pct:.2f}%")
    if failures:
        for key, metric, base, cand, reg in failures:
            sname, size, mode = key
            if metric == "missing_in_candidate":
                print(f"[FAIL] {sname} size={size} mode={mode} missing in candidate CSV")
                continue
            print(
                f"[FAIL] {sname} size={size} mode={mode} {metric}: "
                f"baseline={base:.4f} candidate={cand:.4f} regression={reg:.2f}%"
            )
        sys.exit(1)

    print("[perf_gate] PASS")


if __name__ == "__main__":
    main()
