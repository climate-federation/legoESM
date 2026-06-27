# DINO run configs

Self-contained YAML configs for the DINO (Kamm/Deshayes/Madec 2025) idealized
single-basin runs on both grid backends. Each reproduces a delivered run:

```bash
JAX_ENABLE_X64=1 python scripts/run/run_dino.py --config scripts/experiment/dino/<name>.yaml
```

The YAML keys are the `run_dino.py` long flags with `-` → `_` (e.g. `grid`,
`days`, `snapshot_every_days`, `n_lon`, `mpas_resolution_km`,
`mpas_eq_visc_boost`, `dt`, `output_dir`, `no_forcing`, `physics_off`). They are
loaded as **defaults**, so any explicit CLI flag still overrides the file; an
unknown key is a hard error (typo guard).

| config | grid | window | purpose |
|---|---|---|---|
| `matched_latlon_90d.yaml` | lat-lon Mercator | 90 d / 5 d | cross-grid consistency pair |
| `matched_mpas_90d.yaml`   | MPAS Voronoi     | 90 d / 5 d | cross-grid consistency pair |
| `prev_latlon_full_year.yaml` | lat-lon Mercator | 365 d / 10 d | original full-year deliverable |
| `prev_mpas_90d.yaml`      | MPAS Voronoi     | 90 d / 10 d | original clean 90-day deliverable |

Cross-grid comparison (after running a matched pair):

```bash
JAX_ENABLE_X64=1 python scripts/plot/plot_dino_cross_grid.py \
    results/dino_latlon results/dino_mpas
# -> dino_cross_grid.png (maps + corr) and dino_cross_grid_corr_vs_time.png
```

## MPAS stability notes (see DINOConfig + docs)

The MPAS regional-Voronoi DINO has three discretization-specific instabilities;
two are fixed and on by default, the third bounds the run to ~90 days:

1. **mesh-gen** west-seam float-fold (negative `lon_min`) — fixed in
   `grids/voronoi.py`.
2. **equatorial f→0 jet** (days 0–30) — fixed by `mpas_eq_visc_boost` (default
   8.0; sweep with `--mpas-eq-visc-boost {3,5,8}` reproduces the calibration:
   3 still NaNs day 50, 5 peaks 4.5 m/s, 8 peaks ~2.1 m/s).
3. **distorted southern-channel cells** (~day 130) — viscosity-insensitive
   mesh-quality limit; not a config knob (the regional generator skips Lloyd).

Diagnostic sweeps used one-off `--mpas-eq-visc-boost` / `--dt` overrides on top
of these configs and are not themselves committed (they were throwaway probes).
