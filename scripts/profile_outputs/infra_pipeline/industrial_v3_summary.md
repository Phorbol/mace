# Infra Speed Summary (industrial_v3)

- Gate: PASS
- Threshold: 12.00%
- Baseline CSV: `scripts/profile_outputs/infra_pipeline/industrial_v3_baseline.csv`
- Optimized CSV: `scripts/profile_outputs/infra_pipeline/industrial_v3_optimized.csv`
- Numeric Parity: PASS
- Diagnostics Cases: 4

| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |
|---|---:|---|---:|---:|---:|---:|
| AlN_mp-661_computed.cif | 2 | compile | 43.10 -> 29.68 | 1.45x | 17.20 -> 10.86 | 1.58x |
| AlN_mp-661_computed.cif | 2 | eager | 47.88 -> 42.42 | 1.13x | 28.23 -> 25.57 | 1.10x |
| AlN_mp-661_computed.cif | 3 | compile | 103.17 -> 73.01 | 1.41x | 18.30 -> 12.03 | 1.52x |
| AlN_mp-661_computed.cif | 3 | eager | 106.90 -> 91.47 | 1.17x | 63.70 -> 55.22 | 1.15x |
| AlN_mp-661_computed.cif | 4 | compile | 217.84 -> 151.99 | 1.43x | 27.30 -> 24.87 | 1.10x |
| AlN_mp-661_computed.cif | 4 | eager | 220.10 -> 177.43 | 1.24x | 117.19 -> 104.21 | 1.12x |
| TiO2.cif | 2 | compile | 98.43 -> 65.33 | 1.51x | 16.35 -> 18.35 | 0.89x |
| TiO2.cif | 2 | eager | 163.21 -> 167.08 | 0.98x | 37.61 -> 33.34 | 1.13x |
| TiO2.cif | 3 | compile | 257.12 -> 183.77 | 1.40x | 34.27 -> 29.64 | 1.16x |
| TiO2.cif | 3 | eager | 276.14 -> 226.63 | 1.22x | 140.17 -> 120.23 | 1.17x |
| TiO2.cif | 4 | compile | 460.64 -> 374.37 | 1.23x | 73.90 -> 69.72 | 1.06x |
| TiO2.cif | 4 | eager | 538.92 -> 441.39 | 1.22x | 244.88 -> 227.89 | 1.07x |

## Operator Diagnostics (Top-k Mix)

| structure | size | mode | task | io share | compute share | mixed share |
|---|---:|---|---|---:|---:|---:|
| TiO2.cif | 4 | eager | train | 23.28% | 56.84% | 19.89% |
| TiO2.cif | 4 | eager | infer | 32.81% | 51.46% | 15.73% |
| TiO2.cif | 4 | compile | train | 14.25% | 85.75% | 0.00% |
| TiO2.cif | 4 | compile | infer | 21.99% | 62.16% | 15.85% |
