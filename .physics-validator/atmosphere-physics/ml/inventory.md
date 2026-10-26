# ML physics inventory

Scope: top-level ML modules: `learned_column.py`, `ml_parameterization.py`, `neural_physics.py`. NN plumbing — minimal physics review.

## Files

* `neural_physics.py` (457 LOC): `NeuralPhysics` Equinox module — column MLP. `make_neural_step_unified` (`make_hybrid_step_unified` removed 2026-10-02, unused).
* `learned_column.py` (~180 LOC): `build_column_physics`, `make_column_physics_fn` for spectral PE.
* `ml_parameterization.py` (435 LOC): joint training of physics + dycore.

## Status

These are mostly NN architecture / data plumbing. Physics-validator scope is limited:

* Verify dimensions match `(ncol, n_input)` → `(ncol, n_output)`.
* Verify `_pack_column_features` includes correct units / normalization.
* No raw physics formulas to review — they delegate to the learned model.

## Convention

* Inputs: T, u, v, q_v, p_s — column features.
* Output: temperature/wind tendencies, optionally moisture/cloud water tendencies.
* `residual_scale = 0.01` ensures untrained network produces near-zero tendencies (avoids initial training instability).

## Issues out of scope

Detailed NN architecture review (skip connections, normalization, activation choice) is outside the physics-validator protocol and should be a separate audit.
