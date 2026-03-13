import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser(
        description="Run baseline/optimized profiling matrix and gate regressions."
    )
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--warmup-train", type=int, default=4)
    p.add_argument("--iters-train", type=int, default=16)
    p.add_argument("--warmup-infer", type=int, default=4)
    p.add_argument("--iters-infer", type=int, default=24)
    p.add_argument("--optimized-amp", choices=["none", "bf16", "fp16"], default="bf16")
    tf32_group = p.add_mutually_exclusive_group()
    tf32_group.add_argument("--optimized-tf32", dest="optimized_tf32", action="store_true")
    tf32_group.add_argument("--no-optimized-tf32", dest="optimized_tf32", action="store_false")
    p.set_defaults(optimized_tf32=True)
    p.add_argument("--compile-bucket-atoms", type=int, default=512)
    p.add_argument("--compile-bucket-edges", type=int, default=0)
    p.add_argument("--threshold-pct", type=float, default=5.0)
    p.add_argument(
        "--gate-modes",
        nargs="+",
        choices=["eager", "compile"],
        default=["compile"],
        help="Modes included in perf gate",
    )
    p.add_argument(
        "--gate-min-atoms",
        type=int,
        default=0,
        help="Only gate rows with num_atoms >= this threshold",
    )
    p.add_argument("--out-dir", default="scripts/profile_outputs/infra_pipeline")
    p.add_argument("--run-id", default=None)
    p.add_argument("--skip-baseline", action="store_true", default=False)
    p.add_argument("--baseline-csv", default=None)
    parity_group = p.add_mutually_exclusive_group()
    parity_group.add_argument(
        "--run-parity-check", dest="run_parity_check", action="store_true"
    )
    parity_group.add_argument(
        "--no-run-parity-check", dest="run_parity_check", action="store_false"
    )
    p.set_defaults(run_parity_check=True)
    p.add_argument(
        "--parity-scope",
        choices=["representative", "full"],
        default="representative",
    )
    p.add_argument("--parity-energy-abs-tol", type=float, default=2e-1)
    p.add_argument("--parity-energy-rel-tol", type=float, default=3e-2)
    p.add_argument("--parity-force-abs-tol", type=float, default=5e-3)
    p.add_argument("--parity-force-rel-tol", type=float, default=5e-2)
    diag_group = p.add_mutually_exclusive_group()
    diag_group.add_argument(
        "--run-ops-diagnostics", dest="run_ops_diagnostics", action="store_true"
    )
    diag_group.add_argument(
        "--no-run-ops-diagnostics", dest="run_ops_diagnostics", action="store_false"
    )
    p.set_defaults(run_ops_diagnostics=True)
    p.add_argument(
        "--diag-scope", choices=["representative", "full"], default="representative"
    )
    p.add_argument("--diag-top-k", type=int, default=30)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def run_cmd(cmd, allow_fail: bool = False) -> bool:
    print("[cmd]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        if allow_fail:
            return False
        raise


def load_csv(path: Path) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["structure_name"], row["size"], row["mode"])
            out[key] = {
                "num_atoms": int(row["num_atoms"]),
                "num_edges": int(row["num_edges"]),
                "train_mean_ms": float(row["train_mean_ms"]),
                "infer_mean_ms": float(row["infer_mean_ms"]),
            }
    return out


def select_cases(structures: List[str], sizes: List[int], scope: str):
    if scope == "full":
        return [(s, size) for s in structures for size in sizes]
    return [(structures[0], max(sizes))]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def op_mix(top_ops):
    totals = {"io_bound_likely": 0.0, "compute_bound_likely": 0.0, "mixed": 0.0}
    for op in top_ops:
        key = op.get("intensity_hint", "mixed")
        if key not in totals:
            key = "mixed"
        totals[key] += float(op.get("self_device_ms", 0.0))
    total_ms = sum(totals.values())
    if total_ms <= 1e-12:
        return {
            "total_top_ops_self_device_ms": 0.0,
            "io_share": 0.0,
            "compute_share": 0.0,
            "mixed_share": 0.0,
        }
    return {
        "total_top_ops_self_device_ms": total_ms,
        "io_share": totals["io_bound_likely"] / total_ms,
        "compute_share": totals["compute_bound_likely"] / total_ms,
        "mixed_share": totals["mixed"] / total_ms,
    }


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_json = out_dir / f"{run_id}_baseline.json"
    baseline_csv = out_dir / f"{run_id}_baseline.csv"
    optimized_json = out_dir / f"{run_id}_optimized.json"
    optimized_csv = out_dir / f"{run_id}_optimized.csv"
    summary_json = out_dir / f"{run_id}_summary.json"
    summary_md = out_dir / f"{run_id}_summary.md"
    parity_json = out_dir / f"{run_id}_parity.json"
    parity_csv = out_dir / f"{run_id}_parity.csv"

    if args.baseline_csv is not None:
        baseline_csv_path = Path(args.baseline_csv)
    else:
        baseline_csv_path = baseline_csv

    if not args.skip_baseline and args.baseline_csv is None:
        base_cmd = [
            args.python,
            "scripts/profile_no_cueq_real_structures.py",
            "--structures",
            *args.structures,
            "--sizes",
            *[str(s) for s in args.sizes],
            "--device",
            args.device,
            "--compile-mode",
            args.compile_mode,
            "--warmup-train",
            str(args.warmup_train),
            "--iters-train",
            str(args.iters_train),
            "--warmup-infer",
            str(args.warmup_infer),
            "--iters-infer",
            str(args.iters_infer),
            "--out-json",
            str(baseline_json),
            "--out-csv",
            str(baseline_csv),
        ]
        run_cmd(base_cmd)

    opt_cmd = [
        args.python,
        "scripts/profile_no_cueq_real_structures.py",
        "--structures",
        *args.structures,
        "--sizes",
        *[str(s) for s in args.sizes],
        "--device",
        args.device,
        "--compile-mode",
        args.compile_mode,
        "--warmup-train",
        str(args.warmup_train),
        "--iters-train",
        str(args.iters_train),
        "--warmup-infer",
        str(args.warmup_infer),
        "--iters-infer",
        str(args.iters_infer),
        "--amp",
        args.optimized_amp,
        "--compile-bucket-atoms",
        str(args.compile_bucket_atoms),
        "--compile-bucket-edges",
        str(args.compile_bucket_edges),
        "--out-json",
        str(optimized_json),
        "--out-csv",
        str(optimized_csv),
    ]
    if args.optimized_tf32:
        opt_cmd.append("--tf32")
    run_cmd(opt_cmd)

    gate_cmd = [
        args.python,
        "scripts/perf_gate.py",
        "--baseline-csv",
        str(baseline_csv_path),
        "--candidate-csv",
        str(optimized_csv),
        "--threshold-pct",
        str(args.threshold_pct),
        "--modes",
        *args.gate_modes,
        "--min-atoms",
        str(args.gate_min_atoms),
    ]
    gate_pass = True
    gate_pass = run_cmd(gate_cmd, allow_fail=True)

    parity_result = None
    if args.run_parity_check:
        parity_cases = select_cases(args.structures, args.sizes, args.parity_scope)
        parity_structures = [c[0] for c in parity_cases]
        parity_sizes = sorted({c[1] for c in parity_cases})
        parity_cmd = [
            args.python,
            "scripts/check_numeric_parity.py",
            "--structures",
            *parity_structures,
            "--sizes",
            *[str(s) for s in parity_sizes],
            "--device",
            args.device,
            "--amp",
            args.optimized_amp,
            "--energy-abs-tol",
            str(args.parity_energy_abs_tol),
            "--energy-rel-tol",
            str(args.parity_energy_rel_tol),
            "--force-abs-tol",
            str(args.parity_force_abs_tol),
            "--force-rel-tol",
            str(args.parity_force_rel_tol),
            "--out-json",
            str(parity_json),
            "--out-csv",
            str(parity_csv),
        ]
        if args.optimized_tf32:
            parity_cmd.append("--tf32")
        parity_ok = run_cmd(parity_cmd, allow_fail=True)
        if parity_json.exists():
            parity_result = load_json(parity_json)
            parity_result["cmd_ok"] = parity_ok
        else:
            parity_result = {"all_pass": False, "cmd_ok": False, "rows": []}

    diagnostics = []
    if args.run_ops_diagnostics:
        diag_cases = select_cases(args.structures, args.sizes, args.diag_scope)
        for structure, size in diag_cases:
            sname = Path(structure).stem
            for mode in ("eager", "compile"):
                for task in ("train", "infer"):
                    diag_json = out_dir / f"{run_id}_diag_{sname}_s{size}_{mode}_{task}.json"
                    diag_csv = out_dir / f"{run_id}_diag_{sname}_s{size}_{mode}_{task}.csv"
                    diag_cmd = [
                        args.python,
                        "scripts/profile_torch_ops_diagnostics.py",
                        "--structure",
                        structure,
                        "--size",
                        str(size),
                        "--mode",
                        mode,
                        "--task",
                        task,
                        "--device",
                        args.device,
                        "--compile-mode",
                        args.compile_mode,
                        "--amp",
                        args.optimized_amp,
                        "--top-k",
                        str(args.diag_top_k),
                        "--out-json",
                        str(diag_json),
                        "--out-csv",
                        str(diag_csv),
                    ]
                    if task == "infer":
                        diag_cmd.append("--compute-force")
                    if args.optimized_tf32:
                        diag_cmd.append("--tf32")
                    diag_ok = run_cmd(diag_cmd, allow_fail=True)
                    if diag_json.exists():
                        diag_data = load_json(diag_json)
                        mix = op_mix(diag_data.get("top_ops", []))
                        diagnostics.append(
                            {
                                "structure_name": Path(structure).name,
                                "size": size,
                                "mode": mode,
                                "task": task,
                                "cmd_ok": diag_ok,
                                "json": str(diag_json),
                                "csv": str(diag_csv),
                                "top_ops": diag_data.get("top_ops", [])[:5],
                                **mix,
                            }
                        )
                    else:
                        diagnostics.append(
                            {
                                "structure_name": Path(structure).name,
                                "size": size,
                                "mode": mode,
                                "task": task,
                                "cmd_ok": False,
                                "json": str(diag_json),
                                "csv": str(diag_csv),
                                "top_ops": [],
                                "total_top_ops_self_device_ms": 0.0,
                                "io_share": 0.0,
                                "compute_share": 0.0,
                                "mixed_share": 0.0,
                            }
                        )

    base = load_csv(baseline_csv_path)
    opt = load_csv(optimized_csv)
    rows = []
    train_speedups = []
    infer_speedups = []
    for key, b in base.items():
        if key not in opt:
            continue
        o = opt[key]
        train_speedup = b["train_mean_ms"] / o["train_mean_ms"]
        infer_speedup = b["infer_mean_ms"] / o["infer_mean_ms"]
        rows.append(
            {
                "structure_name": key[0],
                "size": int(key[1]),
                "mode": key[2],
                "num_atoms": b["num_atoms"],
                "num_edges": b["num_edges"],
                "baseline_train_ms": b["train_mean_ms"],
                "optimized_train_ms": o["train_mean_ms"],
                "train_speedup_x": train_speedup,
                "baseline_infer_ms": b["infer_mean_ms"],
                "optimized_infer_ms": o["infer_mean_ms"],
                "infer_speedup_x": infer_speedup,
            }
        )
        train_speedups.append(train_speedup)
        infer_speedups.append(infer_speedup)

    rows.sort(key=lambda r: (r["structure_name"], r["size"], r["mode"]))
    summary = {
        "run_id": run_id,
        "baseline_csv": str(baseline_csv_path),
        "optimized_csv": str(optimized_csv),
        "gate_pass": gate_pass,
        "threshold_pct": args.threshold_pct,
        "gate_modes": args.gate_modes,
        "gate_min_atoms": args.gate_min_atoms,
        "optimized_amp": args.optimized_amp,
        "optimized_tf32": args.optimized_tf32,
        "compile_bucket_atoms": args.compile_bucket_atoms,
        "compile_bucket_edges": args.compile_bucket_edges,
        "median_train_speedup_x": median(train_speedups) if train_speedups else None,
        "median_infer_speedup_x": median(infer_speedups) if infer_speedups else None,
        "parity_result": parity_result,
        "diagnostics": diagnostics,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Infra Speed Summary ({run_id})",
        "",
        f"- Gate: {'PASS' if gate_pass else 'FAIL'}",
        f"- Threshold: {args.threshold_pct:.2f}%",
        f"- Baseline CSV: `{baseline_csv_path}`",
        f"- Optimized CSV: `{optimized_csv}`",
        (
            f"- Numeric Parity: {'PASS' if parity_result and parity_result.get('all_pass') else 'SKIP/FAIL'}"
            if args.run_parity_check
            else "- Numeric Parity: SKIPPED"
        ),
        f"- Diagnostics Cases: {len(diagnostics)}",
        "",
        "| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['structure_name']} | {r['size']} | {r['mode']} | "
            f"{r['baseline_train_ms']:.2f} -> {r['optimized_train_ms']:.2f} | {r['train_speedup_x']:.2f}x | "
            f"{r['baseline_infer_ms']:.2f} -> {r['optimized_infer_ms']:.2f} | {r['infer_speedup_x']:.2f}x |"
        )
    if diagnostics:
        md_lines.extend(
            [
                "",
                "## Operator Diagnostics (Top-k Mix)",
                "",
                "| structure | size | mode | task | io share | compute share | mixed share |",
                "|---|---:|---|---|---:|---:|---:|",
            ]
        )
        for d in diagnostics:
            md_lines.append(
                f"| {d['structure_name']} | {d['size']} | {d['mode']} | {d['task']} | "
                f"{d['io_share']:.2%} | {d['compute_share']:.2%} | {d['mixed_share']:.2%} |"
            )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("[done] summary_json:", summary_json)
    print("[done] summary_md:", summary_md)
    if not gate_pass:
        sys.exit(1)
    if args.run_parity_check and (not parity_result or not parity_result.get("all_pass")):
        sys.exit(1)


if __name__ == "__main__":
    main()
