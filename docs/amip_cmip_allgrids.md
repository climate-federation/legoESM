# AMIP CMIP on all grid types — wiring + 30-year campaign runbook

Status: 2026-06-10. Campaign: verify AMIP physics wiring on every grid,
close the forcing gaps on the standalone (spectral/MPAS) paths, wire
aerosol→microphysics, then run 30-year 2.5° AMIP CMIP on GPU for all
four grids.

## What was wired (2026-06-10)

| Gap | Fix | Where |
|-----|-----|-------|
| Spectral path = dry gray radiation only | Full `make_physics(model_type="spectral_pe")` pipeline (RRTMGP + SBM + Louis + microphysics + clouds), moist tracers initialized + attached to the spectral state | `model_driver._run_spectral`, state init |
| MPAS/spectral ignored external ozone/aerosol/GHG | Per-step traced `forcing` dict keys `o3_vmr` / `aerosol_od` / `ghg_vmr` consumed by the radiation factories | `_run_mpas`, `_run_spectral`, `radiation/integration.py` |
| Insolation frozen at init day on MPAS/spectral (no seasonal/diurnal cycle) | Traced `forcing["day_of_year"]` / `forcing["seconds_of_day"]` override the JIT-static `set_time` closure | radiation factories + both run loops |
| MPAS radiated clear-sky regardless of `--clouds` | `cloud_scheme` + `include_clouds` threaded into the MPAS/spectral `RadiationConfig` | `_run_mpas`, `_run_spectral` |
| `dynamic_albedo` raised NotImplementedError | Briegleb (1992) zenith-dependent ocean albedo in `compute_radiation_core` (diurnal: instantaneous cos SZA; else daytime-effective μ) | `physics_pipeline.py`, `--dynamic-albedo` |
| Aerosol only hit radiation (no CCN link) | Andreae (2009) AOT=0.0027·CCN^0.640 inversion → specified per-column N_c for Morrison (`nc_from_aerosol`); same N_c feeds radiation r_eff (Twomey) and KK2000 autoconversion (lifetime) | `microphysics/aerosol_activation.py`, `_warm_rain.effective_Nc`, `physics_pipeline.py`, `--aerosol-ccn` |

Known remaining gaps (honest report):
- Solar FILE (TSI + 14-band) still inert on spectral/MPAS paths
  (constant S_0); deck prints this. Cube/lat-lon consume it.
- `--aerosol-ccn` / `--dynamic-albedo` are coupled-pipeline only
  (cubed_sphere / latlon); run_amip hard-errors on MPAS/spectral for
  aerosol-ccn, deck gates both.
- Spectral path refuses prognostic-carry schemes (tke/mynn25/edmf,
  bechtold): phys_state is not threaded through the spectral step.
- Prognostic CCN activation (Abdul-Razzak & Ghan) not implemented —
  needs aerosol size/composition the AOD forcing does not carry.

## Production defaults (deck)

`run_amip_cmip6_deck.py`: radiation rrtmg, clouds sundqvist,
microphysics **morrison** (M2005/MG, number-aware r_eff), convection
sbm, turbulence louis, `--aerosol-ccn` + `--dynamic-albedo` default-on
(pipeline grids). Rationale: scheme-validation survey (2026-06-10) —
Morrison is SAM-oracle validated (`docs/md_files/CRM_faithful_SAM.md`)
+ RCEMIP-viable; SBM/Louis are the only AMIP-multi-day-proven
convection/PBL. Mass-flux schemes (ZM/Tiedtke/Bechtold/KF) carry a
documented CAPE bias and zero AMIP validation — candidates, not
defaults.

## 2.5° resolution map

| Grid | `--grid-type/--discretization` | `--resolution` | approx |
|------|-------------------------------|----------------|--------|
| Cubed-sphere | cubed_sphere / finite_volume | 36 (C36) | 2.5° |
| Lat-lon | latlon / finite_volume | 72 | 2.5° |
| Gaussian | gaussian / spectral | 47 (T47) | ~2.5° |
| Voronoi | voronoi / mpas | 5 (level 5, 10242 cells) | ~2.2° |

## 30-year run template (local GPU, RTX 5090)

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  .venv/bin/python scripts/run/run_amip_cmip6_deck.py \
  --forcing-dir forcing_amip --auto-generate \
  --start-year 1979 --end-year 2009 \
  --grid-type cubed_sphere --discretization finite_volume \
  --resolution 36 --nlev 30 --days 10950 \
  --checkpoint-days 365 --diag-days 30 \
  --radiation rrtmg --output results/amip30y_cube \
  --extra --production-profile
```
Repeat with the other three grid rows. Voronoi needs
`--extra --turbulence none --dt <stable>` until the MPAS hidden-CFL
issue (AMIP.md Known #2) is resolved at level 5.

## MPAS hidden-CFL fix (2026-06-10)

The MPAS `--dt 60` workaround is RESOLVED.  Root cause: run_amip's
global `--time-integrator` default (`ssp_rk3`, tuned for cube/lat-lon)
was silently forwarded to the MPAS dycore, whose biharmonic
hyperdiffusion eigenvalues at dt=600 fall outside ssp_rk3's stability
region.  Fix: `--time-integrator auto` (new CLI default) →
`component_factory` resolves to each dycore's own default
(`ssp_rk54_scan` on MPAS).  dt=600 now stable (2-day MPAS COMPLETED).
An explicit integrator name (incl. `ssp_rk3`) is still forwarded
verbatim for sensitivity runs.

## Stability-ladder dt (`--dt-auto`)

`--dt-auto` (run_amip + deck) picks the empirically-validated stable
timestep per (grid, resolution) from `rce_dt.auto_dt_rce`: C36→150 s,
latlon72→75 s, T47→150 s, voronoi→300 s.  Use for long production runs
so a too-large dt cannot blow up mid-chain.

## Validation gates (all PASSED 2026-06-10/11)

1. `smoke_test_amip_all_grids.py` — 8/8 (gray + Morrison defaults).
2. 1-day rrtmg GPU runs green on all four grids (new wiring).
3. 30-day Morrison C36 GPU — COMPLETED, mass 1.9e-5, no blowup.
4. Codex adversarial review (3 rounds) — all real findings fixed
   (N_c silent no-op, solar/ghg fp32 cast, integrator auto-sentinel,
   aerosol-CCN fail-fast).
5. MPI changed-path subset (latlon-AMIP + MPAS-AMIP + halo, 12 tests)
   green via `.venv-mpi`.  NOTE: the FULL `tests/distributed/` suite
   hangs on a pre-existing non-AMIP collective deadlock (unrelated to
   this work) — run AMIP MPI tests with `--timeout` isolated.

## Forcing provenance — REAL science requires ESGF

`forcing_amip/` is **synthetic** (`generate_amip_forcing.py`):
synthetic-noise SST, real-historical GHG.  Fine for pipeline proof,
NOT scientific AMIP.  Real PCMDI AMIP II / input4MIPs SST/SIC needs an
ESGF account (`config/data_catalog.yaml: amip_sst_sic`, no auto-URL).

**To run the real 30-yr campaign:**
1. Download AMIP II bcs SST/SIC (1979-2009) from ESGF input4MIPs into
   `forcing_amip/sst_sic_amip_1979-2009.nc` (+ matching GHG/ozone/
   solar/aerosol/volcanic, or `--auto-generate` the non-SST channels).
2. `bash scripts/run/run_amip30y_allgrids_local.sh` (checkpointed,
   resumable, `--dt-auto`, 3-h radiation, production physics).

## Pipeline proof (running 2026-06-11)

`scripts/run/_amip_pipeline_proof.sh` — 30-day production-stack run on
all 4 grids (synthetic forcing, `--dt-auto`, checkpoint at day 15),
proving the full validated pipeline end-to-end before committing GPU
months to the real science run.
