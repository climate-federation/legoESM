# Scripts

Canonical run scripts for legoESM validation, benchmarking, and production runs.
See `docs/validation/TESTING.md` for the overarching tiered test & experiment strategy.

## `experiment/` — the experiment harness

Versioned templates + provenance ergonomics layered on `legoesm run` /
`legoesm reproduce` (see `docs/validation/TESTING.md` §4 and `config/templates/`):

| Script | Purpose |
|--------|---------|
| `experiment/wizard.py` (or `legoesm wizard`) | Interactive wizard: pick component(s), region/model type, grid, integrator, duration, precision, device, physics-vs-ML — option menus derived live from the model registries (so it versions with the model) — then emit a runnable bundle (`config.yaml`/`run.sh`/`wizard.yaml`, version-stamped) and optionally run it. Needs `pip install 'legoesm[wizard]'`. Pure logic in `wizard_core.py`. |
| `experiment/init_experiment.py <category/name> --name N --output-dir D [-o k=v]` | Resolve a `config/templates/` template + overrides + machine profile → runnable dir (`config.yaml`, `run.sh`, `run.yaml`). Strict-validates before writing. |
| `experiment/validate_templates.py [--write-status]` | Validate every template through `Config…validate_strict`; regenerate `project_status.md`. |
| `experiment/lego_detect_machine.py` | Resolve the `config/machines/` profile for the current host. |
| `experiment/fetch_data.py check\|fetch <template>` | Check/stage a template's external datasets (`config/data_catalog.yaml`). |

## `tmp/` — throwaway staging (NOT for production)

Debug/diagnostic/iteration scripts and archived artifacts live under
`scripts/tmp/` (e.g. `tmp/dycore_iter_archive/`, the non-curated ralph-loop
dycore tests). Nothing here is wired into CI; the directory is slated for
eventual deletion. Do not add production scripts here.

## Layout

Scripts are organized into bucket subdirs (the `scripts/` root holds only
`__init__.py`). The tables below name scripts by basename — find each under its
bucket:

| Bucket | Contents |
|--------|----------|
| `run/` | Production experiment drivers (`run_amip`, `run_omip`, `run_rce`, `run_held_suarez*`, `run_dino`, `run_coupled`, `run_aimip*`, the cross-grid `.sh` wrappers, …). |
| `matrix/` | The complexity-tiered test-matrix runners (`run_{atmosphere,ocean,sea_ice,scm}_test_matrix`, `summarize_matrix_results`, `validate_matrix_report`, `check_conservation_all`). See `docs/validation/TESTING.md`. |
| `experiment/` | The template/provenance harness (see above). |
| `bench/` | Benchmarking, scaling, profiling (`bench_*`, `profile_*`, `*scaling*`, `analyze_*`). |
| `plot/` | Plotting / figure regeneration (`plot_*`, `replot_*`, `regen_*`). |
| `validate/` | Validators + intercomparison (`validate_*`, `verify_*`, `eval_*`, `audit_*`, `compare_*`). |
| `data/` | Forcing/data prep + environment setup (`download_*`, `prepare_*`, `generate_*`, `build_*`, `make_ryf`, `setup_*`). |
| `cluster/` | SLURM job wrappers (e.g. `cluster/omip_nemo/`). |
| `tmp/` | Throwaway (above) — slated for deletion. |

Campaign suites and script packages live as subdirs of the bucket matching
their dominant function (enforced by `tests/test_scripts_layout.py`):
`run/{global_overturning,mpas_realistic_geometry,ocean_long_runs,s2s}/`,
`matrix/{scm,ocean_test_matrix}/`,
`validate/{ocean_fidelity,realistic_geometry,pgf_validation}/`,
`bench/jit_profile/`.

## Data-path environment variables

Scripts that read large external datasets take a CLI flag first and fall back
to an environment variable (machine-portable; no personal paths):

| Variable | Used by | Meaning |
|----------|---------|---------|
| `LEGOESM_ETOPO_PATH` | `run/global_overturning/run_{comparison_latlon,comparison_mpas,tripole_20yr,global_overturning_mpas_etopo}.py` (`--etopo`) | ETOPO bathymetry NetCDF (default `data/bathymetry/etopo_1deg.nc`). |
| `LEGOESM_GSAM_ROOT` | `run/run_lba_plane.py`, `run/run_gate_plane.py` (`--case-dir`) | gSAM checkout root holding `CASES/`. |
| `LEGOESM_CMIP7_RAW` | `data/inspect_cmip7_forcing.py` | Raw CMIP7 forcing staging root (Levante default). |

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

## Data assimilation

| Script | Purpose |
|--------|---------|
| `data/generate_mpas_nmc.py` | Generate same-verification-time full-state MPAS NMC forecast pairs from ERA5 initial conditions. |
| `data/fit_mpas_gen_be.py` | Fit and balance-adjust MPAS GEN_BE parameters from NMC samples. |
| `run/mpas_4dvar_single/run_assimilation.py` | Run the configured single-point T500 3DVar and strong-constraint 4DVar experiment. |
| `run/mpas_4dvar_single/run_tlm.py` | Propagate the 4DVar initial increment with the tangent-linear MPAS model. |
| `plot/plot_mpas_single_point_t500.py` | Regenerate the three-row T500 response figure. |

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
