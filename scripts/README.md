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
