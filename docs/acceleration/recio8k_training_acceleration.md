# RECIO/8k Training Acceleration Benchmark

Generate isolated short-run cases:

```bash
python scripts/benchmarks/recio8k_accel/generate_cases.py \
  --output /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke \
  --epochs 20
```

Submit one case from a clean login-node shell:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke/fp32_hybrid_muon_cueq
sbatch mace-recio8k.sbatch
```

Parse completed logs:

```bash
python scripts/benchmarks/recio8k_accel/parse_metrics.py \
  /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke
```

The default template uses the existing RECIO `4V100` partition. V100 does not provide native bf16 tensor cores, so `bf16_*` cases should be submitted only on a bf16-capable GPU partition. On `4V100`, use `baseline_fp32_adam_cueq`, `fp32_hybrid_muon_cueq`, and `compile_fp32_adam_cueq` as the first smoke comparison.

The historical full baseline final stage-two validation is about `22.1 meV/atom` energy and `260.2 meV/A` force. Smoke runs are for finite-loss, speed, memory, and trend checks; long runs are required before claiming no accuracy or generalization regression.
