# Scripts

Canonical run scripts for legoESM validation, benchmarking, and production runs.

## Atmosphere

| Script | Purpose |
|--------|---------|
| `run_atmosphere_test_matrix.py` | Master test suite: 64+ cases across SW/hydrostatic/NH × cubed-sphere/lat-lon/icosahedral/spectral. Use `--only sw`, `--only hydro`, `--only nh` to filter. |
| `run_amip.py` | Production AMIP CLI wrapper around `ModelDriver`. Supports real forcing (COBE, HadISST), RRTMG radiation, checkpointing. |
| `run_baroclinic_wave_benchmark.py` | Publication-quality baroclinic wave figures (replicates CliMA/Yatunin et al. 2026 Figure 3). |
| `validate_cubed_sphere_fv3_atmos.py` | Targeted FV3 C-D grid edge-discontinuity validation. |

## Ocean

| Script | Purpose |
|--------|---------|
| `run_ocean_test_matrix.py` | Master test suite: 9 cases × 4 grids (cubed-sphere, lat-lon, MPAS, spectral). Use `--grid` and `--only` to filter. |
| `run_ocean_spectral_tests.py` | Gaussian-grid spectral ocean tests (rest state, gravity wave, baroclinic front). |

## Convergence & Scaling

| Script | Purpose |
|--------|---------|
| `run_w2_mpas_convergence.py` | Williamson TC2 convergence study on MPAS Voronoi meshes. |
| `run_levante_gpu_scaling.py` | GPU weak/strong scaling benchmarks (baroclinic wave). |
| `run_levante_gpu_scaling.sh` | SLURM batch driver for multi-node GPU scaling on Levante (DKRZ). |

## ML

| Script | Purpose |
|--------|---------|
| `run_sfno_campaign.py` | SFNO subseasonal-to-seasonal inference campaign runner. |
| `sfno_slab.py` | Thin CLI wrapper for SFNO slab-ocean workflow. |

## Cross-grid comparison wrappers

Each wrapper invokes the per-domain runner once per grid type
(cubed-sphere / lat-lon / MPAS-Voronoi / Gaussian-spectral) with
consistent flags, then calls the matrix runner's
``--cross-grid-plots-only`` (atmosphere) or ``--replot`` (ocean) path
to produce shared-colorbar / shared-projection comparison plots in a
single output tree.  See `docs/CROSS_GRID_COMPARISON_REPORT.md` for
the bug-fix history and convention details.

| Script | Purpose |
|--------|---------|
| `run_rce_cross_grid.sh OUTPUT [DAYS]` | Moist RCE on all 4 atmosphere grids, then cross-grid timeseries plots.  Default DAYS=30. |
| `run_omip_cross_grid.sh OUTPUT [DAYS] [PHYSICS]` | OMIP on all 4 ocean grids, then cross-grid timeseries plots via the iter-49-relaxed collector.  Default DAYS=30, PHYSICS=full. |
| `run_amip_cross_grid.sh OUTPUT [DAYS] [GHG_FILE] [OZONE_FILE] [AEROSOL_FILE]` | Real AMIP via `run_amip.py` on all 4 atmosphere grids with optional CMIP6 input4MIPs forcing files; output is bridged to the matrix-runner format by `_amip_to_matrix_format.py`.  Default DAYS=30; constant GHG / standard ozone / off aerosol when forcing files aren't passed. |

The matrix runners themselves accept matrix-wide CLI overrides for
steady-state CMIP6 forcing in `run_atmosphere_test_matrix.py`:

| Flag | Effect |
|------|--------|
| `--co2-ppmv 280`, `--ch4-ppbv`, `--n2o-ppbv` | Override RRTMGP GHG concentrations (iter-31). |
| `--cloud-scheme {none, sundqvist}` | RRTMGP cloud-radiation coupling (iter-34/36). |
| `--ozone-source {standard, analytical, none}`, `--ozone-peak-hpa`, `--ozone-max-vmr` | Ozone profile selection (iter-39/40). |
