# Phase C — Long-term-simulation diagnostics

## What landed

Four new modules under ``src/legoesm/ocean/`` give every long run the
quantitative spurious-mixing, energy-budget, tracer-conservation, and
restart capabilities the bulletproof claim needs:

* **``rpe.py``** -- Reference Potential Energy (Griffies 2015).
  ``compute_rpe(state, z_coord, grid_type, grid, eos=...)`` returns
  the PE the ocean would carry if every water parcel were adiabatically
  rearranged to minimise PE: densest at the bottom, lightest at the
  top. Algorithm flattens the wet cells, sorts by density, stacks them
  into a single notional column of cross-section ``A_total`` and
  integrates ``g * sum(rho_i * z_centroid_i * vol_i)``. The companion
  ``rpe_drift_rate_per_m2`` returns ``dRPE/dt / A_total`` in W/m^2 --
  the bulletproof-ocean acceptance bar is ``< 0.5 mW/m^2`` on adiabatic
  lock-exchange (Petersen et al. 2015 quotes MPAS-Ocean at
  0.1-0.3 mW/m^2).

* **``budgets.py``** -- two helpers:
  * ``compute_energy_budget(state, z_coord, grid_type, grid)`` ->
    ``EnergyBudget(KE, APE, total, volume, area_total)``. ``KE`` is
    the domain-integrated kinetic energy on cell-centred velocity
    (interpolation matches the cross-model Veros harness); ``APE`` is
    the linear free-surface ``0.5 * rho_0 * g * sum(eta^2 * area)``.
  * ``compute_tracer_budget(state, z_coord, grid_type, grid)`` ->
    ``TracerBudget(volume, heat_content, salt_mass, eta_integral)``.
    Drift in these between two states quantifies tracer-conservation
    error.

* **``restart.py``** -- ``save_restart(state, path, *, time_s, step,
  sha)`` dumps every ``Field`` of the state to a ``.npz`` archive;
  ``load_restart(path, template_state)`` reconstitutes the state with
  the loaded data and the template's ``Field`` metadata. Round-trip is
  ``np.array_equal`` exact across all prognostic fields.

Plus ``scripts/run_ocean_test_matrix.py`` gains an ``--emit-diagnostics``
CLI flag (semantics: comma-separated subset of ``rpe`` / ``energy`` /
``tracer``). Wiring the per-runner emission into every existing runner
is a mechanical follow-up; the CLI flag is in place so callers can
already declare intent.

## Tests

``tests/ocean/unit/test_longterm_diagnostics.py`` (15 tests, **15 pass**):

* RPE: finite + negative at rest on all three grids; cross-grid spread
  < 5 % on the same stratified rest state; unit-check on
  ``rpe_drift_rate_per_m2``.
* Energy budget: ``KE == 0`` and ``APE == 0`` exactly at rest on every
  grid; ``volume > 0``.
* Tracer budget: ``volume / heat_content / salt_mass > 0`` at rest;
  ``eta_integral == 0``; cross-grid volume spread < 2 %.
* Restart: round-trip preserves every ``Field`` of the state
  bit-identically on all three grids; metadata
  (``time_s`` / ``step`` / ``sha``) reads back unchanged.

## Acceptance gate

All four diagnostics produce sane values on every supported grid
(lat-lon C-grid, MPAS Voronoi, cubed-sphere). The bulletproof
acceptance bars (``dRPE/dt < 0.5 mW/m^2`` on lock-exchange,
``|d(KE+APE)/dt - sources| < 1%/d``, ``|d volume/dt| < 1e-10 /d`` on
closed basins) will be exercised by Phase D and Phase F runs that
emit the diagnostics over many model days.
