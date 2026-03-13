# Infra Speed Summary (industrial_v1)

- Gate: PASS
- Threshold: 8.00%
- Baseline CSV: `scripts/profile_outputs/infra_pipeline/industrial_v1_baseline.csv`
- Optimized CSV: `scripts/profile_outputs/infra_pipeline/industrial_v1_optimized.csv`

| structure | size | mode | train ms (base->opt) | train speedup | infer ms (base->opt) | infer speedup |
|---|---:|---|---:|---:|---:|---:|
| AlN_mp-661_computed.cif | 2 | compile | 42.37 -> 22.23 | 1.91x | 11.62 -> 4.72 | 2.46x |
| AlN_mp-661_computed.cif | 2 | eager | 41.96 -> 39.39 | 1.07x | 23.72 -> 19.86 | 1.19x |
| AlN_mp-661_computed.cif | 3 | compile | 103.90 -> 59.15 | 1.76x | 13.01 -> 12.66 | 1.03x |
| AlN_mp-661_computed.cif | 3 | eager | 99.60 -> 71.98 | 1.38x | 57.66 -> 44.29 | 1.30x |
| AlN_mp-661_computed.cif | 4 | compile | 194.86 -> 122.07 | 1.60x | 27.34 -> 27.84 | 0.98x |
| AlN_mp-661_computed.cif | 4 | eager | 199.36 -> 142.11 | 1.40x | 103.64 -> 90.08 | 1.15x |
| TiO2.cif | 2 | compile | 94.42 -> 52.63 | 1.79x | 11.65 -> 12.46 | 0.93x |
| TiO2.cif | 2 | eager | 73.46 -> 70.54 | 1.04x | 31.89 -> 25.16 | 1.27x |
| TiO2.cif | 3 | compile | 232.17 -> 146.24 | 1.59x | 34.75 -> 33.92 | 1.02x |
| TiO2.cif | 3 | eager | 238.28 -> 172.40 | 1.38x | 116.76 -> 108.21 | 1.08x |
| TiO2.cif | 4 | compile | 416.95 -> 255.93 | 1.63x | 69.29 -> 63.67 | 1.09x |
| TiO2.cif | 4 | eager | 445.08 -> 333.59 | 1.33x | 232.69 -> 215.57 | 1.08x |
