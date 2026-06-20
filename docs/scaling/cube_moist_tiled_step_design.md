# Cube-MOIST tiled np>6 step — design (the moist analogue of the dry tiled step)

**Status (2026-06-19):** KICKOFF. The DRY tiled 3D-PE SSP-RK3 step
(`make_tiled_fv3_hydrostatic_step_stage_2d`) is built + np24/54 bit-identity
parity-gated (`test_tiled_fv3_hydrostatic_step.py`, kt=2 np24 PASS confirmed job
8532478). Task #31 made cube-MOIST scale multi-device on the REPLICATED cs-spmd
path (`make_sharded_step` + `model.step_with_physics`), but that path is capped at
**np≤6** (one face per device; no sub-face decomposition). To reach the cube's
theoretical limit for MOIST (np=6·kt², kt≥2 → 24, 54, …) the moist column physics
+ tracer transport must ride the SAME sub-face tiling the dry step already uses.
Future-HW (np>6 anti-scales on Ginsburg CPU); every increment gated by BIT-IDENTITY
(tiled reassembly == global serial-moist), the proven U3 methodology.

## What the dry tiled step already gives us (reuse, do NOT re-derive)

- `make_tiled_fv3_hydrostatic_step_stage_2d(mesh, cdgrid, coord, n, kt)` →
  `step(u_d, v_d, T, p_s, phis) -> (u_d, v_d, T, p_s)` — the dry SSP-RK3 step on a
  `(6, kt, kt)` mesh, no gather.
- `make_tiled_fv3_hydrostatic_thermo_stage_2d` — the per-tile dT/dt **scalar**
  advection (horizontal AL/PPM flux + vertical). T is a passive scalar on the
  C-grid winds; **a tracer q advects by the IDENTICAL stencil** (same u_c/v_c,
  same vertical mass-flux `_tracer_vert_fn`). So the moist tendency is the thermo
  kernel applied to each of q_v/q_c/q_r.
- The tiled vertical transport path already handles "T/tracers/momentum" vertical
  advection (tiled_production_cdgrid.py:505).
- The sub-face halo (`make_tiled_pad_body` scalar / `make_tiled_pad_vector_body`
  vector, incl. the diagonal-corner ppermute) is bit-exact to serial
  `pad_halo`/`pad_halo_vector` — the tracer scalar halo reuses it unchanged.
- Serial reference: `fv3_hydrostatic_tendencies` already advects tracers via the
  shared `advective_tracer_tendency` (primitive_eq_cdgrid.py:898) and adds
  `physics_tendency_cc.tracer_tendencies` (Kessler). `make_kessler_forcing_cube`
  is the column-local (no-halo) physics.

## Increments (each: bit-identity np24 vs serial-moist, 24/54 faked CPU devices)

1. **Tiled single-tracer tendency.** Generalize the thermo tile body to advect a
   passed scalar `q` (NOT just T): `make_tiled_fv3_hydrostatic_tracer_stage_2d` or
   a `tracers=` arg on the thermo stage. Parity: tiled dq/dt == serial
   `advective_tracer_tendency` for one tracer at kt=2. (The scalar halo + vertical
   fn already exist — this is threading, not new numerics.)
2. **Tracer pack (q_v/q_c/q_r).** Extend to the dict/packed tracers the FV3 state
   carries (`FV3HydrostaticState.tracers`); one tiled scalar-advection per tracer
   (or batched on a trailing tracer axis like the serial `_q_packed`). Parity vs
   serial packed tracer tendency.
3. **Kessler per-tile in the step.** Apply `make_kessler_forcing_cube(dt)` column
   physics INSIDE the tiled step (post-dynamics, pre/post-RK stage as in the serial
   path) — column-local, no halo, so it runs SPMD-local on each tile shard exactly
   like the replicated path's `step_with_physics`. Parity: tiled moist step ==
   serial moist step.
4. **Full moist tiled step.** `make_tiled_fv3_hydrostatic_moist_step_stage_2d` →
   `step(u_d, v_d, T, p_s, phis, tracers) -> (..., tracers)` + post-step tracer
   floor. np24/54 bit-identity gate `test_tiled_fv3_hydrostatic_moist_step.py`.

## Non-goals / caveats

- NOT Ginsburg-benchable (np>6 anti-scales on CPU; the value is the CAPABILITY +
  the parity receipt, not a speed number — do NOT sell a microbench as a win).
- Numerics ONLY from the shared `*_core` ops + the existing tiled kernels; the moist
  path adds NO new dynamics numerics (Kessler is the only physics, already shared).
- Base cut matches the dry step (div_damp=0, hyperdiff=0, non-duogrid); optional
  terms are later increments, not base-case-blocking.
