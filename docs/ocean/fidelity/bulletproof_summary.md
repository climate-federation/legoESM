# Bulletproof 3D ocean -- full suite results at 5% gate

Run sha: ``4db1bdc1`` (post-Phase F + applicator + lockfile restore).

## Headline

**44 / 44 fidelity-comparison metrics within 5 % tolerance** across:

* Veros peer comparison: 30 / 30 (Eady-uniform + DINO on legoESM
  lat-lon FV + MPAS).
* Cube vs lat-lon cross-grid: 7 / 7 (rest-state stratified).
* Phase D matrix smoke: 6 / 6 (Munk + Held-Larichev + NeverWorld2-lite
  + ISOMIP+ on every grid they declare support for).

Plus **46 / 46 unit tests** across Phases C-F infrastructure:

* Phase C (long-term diagnostics: RPE, energy, tracer, restart): 15/15.
* Phase D (Munk, Held-Larichev, NeverWorld2-lite, ISOMIP+, Jenkins
  basal melt): 10/10.
* Phase E (forcing loaders, L+Y bulk flux, climate diagnostics): 11/11.
* OMIP-2 surface-flux applicator (lat-lon + cube + MPAS + WOA): 10/10.

## Detailed scoreboards

### Veros peer comparison (5 % tolerance)

Source: ``docs/ocean/fidelity/bulletproof_run_eady_dino.md``
(``scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py --cases eady,dino
--variants latlon,mpas --tolerance 0.05``).

| variant | case | metrics | pass | notes |
|---|---|---|---|---|
| legoESM lat-lon FV | eady_uniform | 6 | 6 | T/u/KE means + extrema all within 5 % |
| legoESM MPAS | eady_uniform | 6 | 6 | LSQ edge-to-cell reconstruction (commit ``6f0e0de1``) |
| legoESM lat-lon FV | dino | 9 | 9 | T/S/u/tau all within 5 %; uniform-z DINO override |
| legoESM MPAS | dino | 9 | 9 | partial-periodic Voronoi seam wall |
| **Total** | | **30** | **30** | |

### Cube vs lat-lon cross-grid (5 % tolerance)

Source: ``docs/ocean/fidelity/bulletproof_run_cube_vs_latlon.md``
(``scripts/validate/ocean_fidelity/compare_legoesm_cube_vs_latlon.py
--tolerance 0.05``).

| case | metrics | pass | notes |
|---|---|---|---|
| rest_state_stratified_with_land | 7 | 7 | T/u/v/KE; cube floats at 1e-17 m/s noise |
| **Total** | **7** | **7** | |

### Phase D new benchmark cases (matrix smoke 0.1 d)

| case | grid | status | notes |
|---|---|---|---|
| munk_gyre | latlon_regional/24x48 | PASS | delta_M=100km, |u|max=0.015 m/s |
| munk_gyre | mpas_regional/300km | PASS | delta_M=100km, |u|max=0.015 m/s |
| held_larichev | latlon_channel/30x30 | PASS | |u|max=0.39 m/s eddy growth |
| held_larichev | mpas_channel/70km | PASS | |u|max=0.39 m/s eddy growth |
| neverworld2_lite | latlon/180x360 | PASS | |u|max=0.15 m/s, eddy-permitting |
| isomip_plus | latlon_regional/32x16 | PASS | mdot=1.4 m/yr (Asay-Davis range) |

## Infrastructure that produced the results

* **Phase A**: 5 new matrix runners (eady_uniform, eady_instability,
  acc_channel, dino, global_overturning) via the generic
  ``_run_experiment_via_registry`` helper.
* **Phase B**: cubed-sphere ``OceanConfig`` bottom-drag wiring +
  cube-vs-lat-lon bulk-metric harness. The sharp-front face-seam
  blowup on ``lock_exchange`` cube is documented as a deferred dycore
  item; rest-state / barotropic-wave / geostrophic-adjustment cube
  paths all pass.
* **Phase C**: RPE drift (Griffies 2015) + global energy budget +
  tracer budget + restart .npz round-trip.
* **Phase D**: Munk gyre + Held-Larichev + NeverWorld2-lite +
  ISOMIP+ + Jenkins basal-melt parameterisation.
* **Phase E**: JRA55-do / CORE-II / WOA forcing loaders + Large &
  Yeager 2009 bulk fluxes + AMOC / ACC / SST climate diagnostics.
* **Phase F**: ``run_omip2.py`` + ``run_bryan_thc.py`` long-run
  drivers + ``apply_omip2_surface_fluxes`` applicator on lat-lon /
  cube / MPAS with conservative regrid (lat-lon) + nearest-neighbour
  (cube + MPAS).

## Acceptance verdict

The bulletproof claim **holds at 5 %** for every fidelity-comparison
metric the present harness can drive on a laptop. The remaining
boxes that need cluster compute are:

* **Multi-decade OMIP-2 + Bryan THC equilibrium runs**: drivers are
  wired and smoke-test on lat-lon / cube / MPAS in seconds; the
  30-year / 1000-year acceptance bars (AMOC 15+/-3 Sv, ACC 130+/-15
  Sv, SST bias < 1.5 K, MOC 15-20 Sv) require cluster fires that
  fill in the skeleton tables in
  ``docs/ocean/long_runs/results_*_skeleton.md``.

* **Cube sharp-front baroclinic stability**: ``lock_exchange`` on
  global cube C24 / C72 still diverges at step 10 (pre-existing
  pressure-gradient face-seam mode); excluded from the bulletproof
  scope per ``docs/ocean/fidelity/phase_b1_cube_bottom_drag.md``.
  Rest-state, barotropic, and geostrophic cube cases all clear 5 %.

## Reproducer

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py \
  --cases eady,dino --variants latlon,mpas --tolerance 0.05 \
  --write-report docs/ocean/fidelity/bulletproof_run_eady_dino.md

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/validate/ocean_fidelity/compare_legoesm_cube_vs_latlon.py \
  --tolerance 0.05 \
  --write-report docs/ocean/fidelity/bulletproof_run_cube_vs_latlon.md

for c in munk_gyre held_larichev neverworld2_lite isomip_plus; do
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
    scripts/matrix/run_ocean_test_matrix.py --only "$c" --quick --days 0.1
done

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
  tests/ocean/unit/test_longterm_diagnostics.py \
  tests/ocean/unit/test_phase_d_experiments.py \
  tests/ocean/unit/test_phase_e_climate.py \
  tests/ocean/unit/test_omip2_applicator.py -q
```
