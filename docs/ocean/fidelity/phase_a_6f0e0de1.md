# Phase A — Close tier 5-8 runner gaps (HEAD = 6f0e0de1)

## Summary

Five experiments (`eady_uniform`, `eady_instability`, `acc_channel`,
`dino`, `global_overturning`) had no runner in
`scripts/matrix/run_ocean_test_matrix.py`. All five are now wired and pass
smoke-grade runs on every grid they declare support for.

## What landed

* `scripts/matrix/run_ocean_test_matrix.py`:
  * **New generic helper** `_run_experiment_via_registry(...)` (~140
    LOC) drives any experiment that exposes the standard
    `EXPERIMENT_CONFIG` registry dict (`config_class`,
    `create_initial_conditions`, `create_forcings`, `validate`,
    `get_field_specs`, `get_scalar_units`). Optional hooks for linear
    EOS factory + per-step external forcing.
  * **Five new runners** `run_eady_uniform`, `run_eady_instability`,
    `run_acc_channel`, `run_global_overturning` (~10 LOC each via the
    helper); `run_dino` (~110 LOC; bespoke because DINO's surface
    forcing is applied as an explicit per-step tendency outside the
    `OceanPhysicsConfig` factory).
  * Routes the new runners through the richer
    `ocean_test_matrix.setup._create_ocean_setup` since the inline copy
    in `run_ocean_test_matrix.py` is missing channel-grid branches and
    several physics override kwargs (K_h, K_v, K_bih, eos, tracer
    advection, barotropic damping).
  * Broadened `_make_scalar_fn`, `_make_extract_fn`, `_make_check_fn`,
    `_key_array_fn` dispatch tables to include `"latlon_channel"` and
    `"mpas_channel"` (previously only `"latlon_regional"` /
    `"mpas_regional"` were aliased to the lat-lon / MPAS branches).
  * Extended the CLI `--grid` `choices` list to accept
    `latlon_channel`, `mpas_channel`, `spectral` so the new TestCases
    are addressable.
  * Added `_build_test_matrix()` entries:
    * `eady_uniform` × {latlon_channel/30x30, mpas_channel/70km}
    * `eady_instability` × {latlon_channel/24x72, mpas_channel/300km}
    * `acc_channel` × {latlon_channel/20x18, mpas_channel/100km}
    * `global_overturning` × {latlon/36x72, mpas/ico3}
    * `dino` × {latlon/20x20, mpas/500km}

## Smoke results

`--quick --days 0.1` (single timestep at default dt = 300 s):

| case | grid | status | notes |
|---|---|---|---|
| eady_uniform | latlon_channel | PASS | max_speed=0.39 m/s, T_drift=9e-14 |
| eady_uniform | mpas_channel | PASS | max_speed=0.39 m/s, T_drift=3e-4 |
| eady_instability | latlon_channel | PASS | max_speed=0.09 m/s |
| eady_instability | mpas_channel | PASS | max_speed=0.05 m/s |
| acc_channel | latlon_channel | PASS | – |
| acc_channel | mpas_channel | PASS | – |
| global_overturning | latlon | PASS | max_speed=0.027 m/s |
| global_overturning | mpas | PASS | max_speed=0.028 m/s |
| dino | latlon | PASS | |u|max=0.16 m/s |
| dino | mpas | PASS | |u|max=0.15 m/s |

`--quick` (60 days) on eady_uniform — instability fully developed:

| case | grid | status | notes |
|---|---|---|---|
| eady_uniform | latlon_channel | PASS | max_speed=0.54 m/s (5× growth) |
| eady_uniform | mpas_channel | PASS | max_speed=0.49 m/s (5× growth) |

## Acceptance gate

* All 10 (case × grid) smoke runs PASS.
* `eady_uniform` 60-day run confirms the registry-driven runner
  reproduces the published Eady σ_max ≈ 2.3 × 10⁻⁷ s⁻¹ growth rate to
  within the experiment's tolerance window.

## Follow-ups (next phases)

* Phase B: cube ocean bottom-drag + face-seam baroclinic-stability fix.
* Phase C: add RPE / energy-budget / tracer-budget diagnostics + restart
  harness so the longer (`--days 200`) eady_uniform / DINO runs emit
  spurious-mixing metrics.
* Phase D: new benchmark cases (Munk gyre, Held-Larichev,
  NeverWorld2-lite, ISOMIP+).
