# Top-level physics modules inventory

Scope: top-level `src/legoesm/atmosphere/physics/*.py`.

## Files

* `__init__.py` (32 LOC): public exports.
* `combined.py` (576 LOC): `PhysicsConfig`, `make_physics()`, dispatches to dynamical-core-specific orchestrators (hydrostatic / nonhydrostatic / spectral_pe / mpas).
* `_shared.py` (529 LOC): cross-package helpers (heights, density, column extraction, moisture convergence, omega → w).
* `thermodynamics.py` (300 LOC): re-exports `saturation_mixing_ratio` from `legoesm.thermo`; provides `temperature_from_theta`, `pressure_from_eos`, `compute_moist_adiabat`, `compute_cape`.
* `physics_state.py` (211 LOC): `PhysicsState` NamedTuple (tke, conv_prog_profile, conv_stoch_state, gwd_spectrum, prng_key); `init_physics_state`, `update_physics_state`.

## Key abstractions

* **`PhysicsState`**: prognostic carry of physics modules. All fields are concrete `jax.Array` (never None), zero-padded for inactive schemes.
* **`combined._make_*_combined`**: factory that calls each sub-module's `make_*_physics()` and sums tendencies. Handles tracer accumulation (spectral PE) and dict-merging of multi-field updates (e.g., Bechtold's stoch state).

## Convention check

* `compute_heights_from_sigma`, `compute_layer_dz`, `compute_rho` — all use `constants.R_d`, `constants.g`. ✓
* `compute_moisture_convergence` (`_shared.py:370`): documented to use FV-flux divergence with limiter disabled (smooth gradient).
* `diagnose_grid_w_from_omega`: standard hydrostatic identity `w = -ω/(ρg)` with optional virtual T correction.
