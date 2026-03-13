# Infra Speed Summary (industrial_smoke_v3)

- Gate: PASS
- Threshold: 1000.00%
- Baseline CSV: `scripts/profile_outputs/infra_pipeline/industrial_smoke_v3_baseline.csv`
- Optimized CSV: `scripts/profile_outputs/infra_pipeline/industrial_smoke_v3_optimized.csv`
- Numeric Parity: PASS
- Diagnostics Cases: 4

| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |
|---|---:|---|---:|---:|---:|---:|
| TiO2.cif | 2 | compile | 102.07 -> 154.41 | 0.66x | 14.79 -> 72.45 | 0.20x |
| TiO2.cif | 2 | eager | 223.88 -> 1213.12 | 0.18x | 55.64 -> 246.97 | 0.23x |

## Operator Diagnostics (Top-k Mix)

| structure | size | mode | task | io share | compute share | mixed share |
|---|---:|---|---|---:|---:|---:|
| TiO2.cif | 2 | eager | train | 52.15% | 20.05% | 27.80% |
| TiO2.cif | 2 | eager | infer | 58.18% | 17.62% | 24.20% |
| TiO2.cif | 2 | compile | train | 25.61% | 74.39% | 0.00% |
| TiO2.cif | 2 | compile | infer | 55.79% | 44.21% | 0.00% |
