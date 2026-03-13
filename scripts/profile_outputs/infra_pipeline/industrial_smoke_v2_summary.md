# Infra Speed Summary (industrial_smoke_v2)

- Gate: FAIL
- Threshold: 30.00%
- Baseline CSV: `scripts/profile_outputs/infra_pipeline/industrial_smoke_v2_baseline.csv`
- Optimized CSV: `scripts/profile_outputs/infra_pipeline/industrial_smoke_v2_optimized.csv`
- Numeric Parity: PASS
- Diagnostics Cases: 4

| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |
|---|---:|---|---:|---:|---:|---:|
| TiO2.cif | 2 | compile | 101.18 -> 54.84 | 1.85x | 36.74 -> 52.08 | 0.71x |
| TiO2.cif | 2 | eager | 363.03 -> 1076.73 | 0.34x | 62.49 -> 468.57 | 0.13x |

## Operator Diagnostics (Top-k Mix)

| structure | size | mode | task | io share | compute share | mixed share |
|---|---:|---|---|---:|---:|---:|
| TiO2.cif | 2 | eager | train | 38.35% | 21.71% | 39.94% |
| TiO2.cif | 2 | eager | infer | 47.09% | 19.57% | 33.34% |
| TiO2.cif | 2 | compile | train | 13.74% | 70.03% | 16.22% |
| TiO2.cif | 2 | compile | infer | 0.00% | 0.00% | 0.00% |
