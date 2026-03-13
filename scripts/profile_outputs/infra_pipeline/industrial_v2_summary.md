# Infra Speed Summary (industrial_v2)

- Gate: FAIL
- Threshold: 12.00%
- Baseline CSV: `scripts/profile_outputs/infra_pipeline/industrial_v2_baseline.csv`
- Optimized CSV: `scripts/profile_outputs/infra_pipeline/industrial_v2_optimized.csv`
- Numeric Parity: PASS
- Diagnostics Cases: 4

| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |
|---|---:|---|---:|---:|---:|---:|
| AlN_mp-661_computed.cif | 2 | compile | 43.86 -> 29.02 | 1.51x | 20.83 -> 26.94 | 0.77x |
| AlN_mp-661_computed.cif | 2 | eager | 49.15 -> 40.37 | 1.22x | 29.42 -> 26.46 | 1.11x |
| AlN_mp-661_computed.cif | 3 | compile | 106.63 -> 77.05 | 1.38x | 16.55 -> 24.71 | 0.67x |
| AlN_mp-661_computed.cif | 3 | eager | 106.61 -> 85.03 | 1.25x | 61.43 -> 49.29 | 1.25x |
| AlN_mp-661_computed.cif | 4 | compile | 208.94 -> 157.59 | 1.33x | 28.19 -> 29.97 | 0.94x |
| AlN_mp-661_computed.cif | 4 | eager | 220.13 -> 177.82 | 1.24x | 120.95 -> 104.17 | 1.16x |
| TiO2.cif | 2 | compile | 98.87 -> 64.80 | 1.53x | 34.05 -> 19.85 | 1.72x |
| TiO2.cif | 2 | eager | 162.88 -> 156.89 | 1.04x | 37.57 -> 33.64 | 1.12x |
| TiO2.cif | 3 | compile | 262.00 -> 205.39 | 1.28x | 35.56 -> 36.16 | 0.98x |
| TiO2.cif | 3 | eager | 268.86 -> 219.39 | 1.23x | 138.61 -> 121.46 | 1.14x |
| TiO2.cif | 4 | compile | 464.11 -> 371.74 | 1.25x | 77.04 -> 64.89 | 1.19x |
| TiO2.cif | 4 | eager | 553.41 -> 441.41 | 1.25x | 244.12 -> 223.28 | 1.09x |

## Operator Diagnostics (Top-k Mix)

| structure | size | mode | task | io share | compute share | mixed share |
|---|---:|---|---|---:|---:|---:|
| TiO2.cif | 4 | eager | train | 23.22% | 56.87% | 19.92% |
| TiO2.cif | 4 | eager | infer | 33.07% | 51.17% | 15.76% |
| TiO2.cif | 4 | compile | train | 14.29% | 85.71% | 0.00% |
| TiO2.cif | 4 | compile | infer | 22.04% | 62.29% | 15.68% |
