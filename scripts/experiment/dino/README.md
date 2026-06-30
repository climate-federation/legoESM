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
| `acc_kpp_latlon_365d.yaml` | lat-lon, KPP     | 365 d / 5 d | barotropic-Psi + ACC-transport spin-up |
| `paper_tke_latlon.yaml`   | lat-lon, **TKE** | 365 d / 5 d | NEMO TKE closure, runnable via the 5e-4 momentum-viscosity floor |

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

## Fidelity to Kamm et al. 2025 (GMD 18, 8091–8107)

**Match exactly** (verified against the paper + the upstream NEMO namelist):
domain / channel / depth, 36 z* levels, wind-stress knots, T*/S* restoring
targets + coefficients (A_Θ=40, A_S=3.858e-3), Jerlov-I shortwave penetration
(ζ=0.35/23 m, 0.58/0.42), background vertical visc/diff (1.2e-4 / 1.2e-5),
convective adjustment (100 m²/s), quadratic bottom drag, energy/enstrophy
vector-invariant momentum advection, FCT/TVD tracer advection.

**Remaining gaps** (not bit-identical to the paper):
- **Vertical mixing**: the paper uses NEMO TKE. `--vmix tke` selects our TKE
  closure configured to the paper. At the paper's A_v_bg=1.2e-4 it NaNs ~day 39
  (a TKE-specific surface-momentum instability at the SW channel-wall corner —
  NOT a CFL; the `constant` scheme is stable there). It is now **runnable** with
  the default `tke_momentum_visc_bg=5e-4` floor (4× the paper momentum
  viscosity, a documented compromise; the tracer K_v_bg stays at the paper
  value). Add `tke_momentum_visc_bg: 1.2e-4` to reproduce the unstable paper
  attempt. The default closure is still **KPP** (`vmix: kpp`). On MPAS the
  vertical mixing is implicit constant-coefficient, so the scheme is inert.
- **EOS**: default = Wright (1997); `--eos nemo_seos` selects the paper's
  simplified S-EOS (Roquet et al. 2015) with the DINO coefficients (the oracle
  EOS for the thermocline comparison).
- **Restoring**: ours = annual-mean; paper = seasonal (360-day, 1-month lag).
- **Spin-up**: the paper's R1 numbers are at **3000-year equilibrium**; our runs
  are ≤1 year, so equilibrium diagnostics (ACC 206 Sv, MOC, deep stratification)
  are reached only in trajectory, not in magnitude.

## Barotropic streamfunction + ACC transport

```bash
JAX_ENABLE_X64=1 python scripts/plot/plot_dino_acc.py results/dino_latlon  # lat-lon only
# -> dino_acc.png: ACC channel-transport spin-up series vs the paper's 206 Sv
#    + the barotropic streamfunction map (gyres + channel; cf. paper Fig. 4)
```

Reuses the canonical partial-cell `ocean.diagnostics_streamfunction.barotropic_streamfunction`
+ `ocean.diagnostics_climate.acc_transport`. A KPP 365-day run spins the ACC up
0 → ~48 Sv (≈¼ of the 206 Sv equilibrium — the wind-driven transport establishes
over the first year; the rest is the multi-millennial baroclinic adjustment).
