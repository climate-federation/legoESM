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
   **DONE (commits 155ff92a5 increment-1 + 30fbb3cd3 increment-2; gates 8532495 /
   8532505 PASS).** Single-tracer + packed q_v/q_c/q_r tiled advection in
   `tiled_production_cdgrid.py`; np24 bit-identity vs serial
   `advective_tracer_tendency`; thermo refactored onto the shared helper
   (bit-identical, no regression).
3. **Kessler per-tile (combined moist tracer tendency).** Add the warm-rain tracer
   tendencies to the increment-2 advection.  ARCHITECTURE FINDING (scout
   2026-06-20): kessler is pointwise (no halo, `mesh` unused), BUT the public
   state-based closures (`make_kessler_forcing_cube`/`_gridspace`) flatten with
   `reshape(-1, nlev)` over the horizontal axes — that **breaks under tile
   sharding** (a reshape across a sharded axis). So the COLUMN core must run
   per-tile INSIDE the shard_map (each tile reshapes only its OWN local columns).
   That core is `_kessler_column_tendencies` (atmosphere, `_`-private); core
   cannot import atmosphere. CLEAN FIX (dependency injection, keeps core
   physics-agnostic): (a) PROMOTE `_kessler_column_tendencies` ->
   `kessler_column_tendencies` (public, + update its 3 internal callers); (b) core
   `make_tiled_fv3_moist_tracer_tendency_stage_2d(mesh, cdgrid, n, kt, nlev, *,
   column_tracer_physics_fn)` = increment-2 advection + an INJECTED per-tile
   `column_tracer_physics_fn(T_t, p_s_t, q_v_t, q_c_t, q_r_t) -> (dq_v, dq_c,
   dq_r)` (built by the caller/test from `kessler_column_tendencies` with
   sigma_coord/dt/cfg closed over).  Serial reference (parity): the dynamics add
   is `_dtracers[k] += physics_tendency_cc.tracer_tendencies[k]`
   (primitive_eq_cdgrid.py:1195) so total = advection + kessler tracer rates.
   sigma-only (kessler raises on hybrid).  Gate: tiled combined dq_pack == serial
   (advective_tracer_tendency + kessler) at np24 bit-identity.
   **DONE (commit 7297b2237, gate 8532506 PASS).** `kessler_column_tendencies`
   promoted public; `make_tiled_fv3_moist_tracer_tendency_stage_2d` (advection +
   injected per-tile column physics); np24 bit-identity vs serial advection +
   kessler.
4. **Full moist tiled step. DONE (code 9ac698e42 codex-CLEAN; test fixes
   9425e84bd; gate moist_step[2] PASSED np24).** `make_tiled_fv3_hydrostatic
   _moist_step_stage_2d` → `step(u_d, v_d, T, p_s, phis, q_pack) -> (..., q_pack)`
   + post-step tracer floor. Parity vs the serial base-cut moist RK3: u_d/v_d/T/p_s
   rel<1e-10, q_pack rel 3.6e-10 (Kessler nonlinear + 3-stage RK3 FMA reorder,
   bit-identity class). LESSON — all 3 initial gate failures were TEST-REFERENCE
   bugs, NOT tiled code: (i) the reference passed `dt_actual` → the iter-189
   corner-div adaptive cap (a shape error) the tiled base cut omits; (ii) D-grid
   CORNER fields u_d/v_d need a block-wise `_rel_corner` compare (the tiled output
   is kt blocks of nl+1 with the staggered shared edge duplicated, NOT n+1);
   (iii) q_pack rides nonlinear Kessler → looser (1e-8) FMA tol than the linear
   dynamics. The tiled code (the shared tracer-aware `_build_hydro_tile_tendency
   _fns` + `_ssp_rk3_tile_step`) was correct from the first build.

   **np24/54 bit-identity gate vs the serial moist step (the original plan):**
   ARCHITECTURE FINDING (scout 2026-06-20): NOT "compose increment 3 into the dry
   RK3" — the tracer VERTICAL advection is coupled to the dynamics' vertical
   velocity (`sigma_dot`/`mass_flux`), which is computed INSIDE the shared
   `_build_hydro_tile_tendency_fns::_tile_tendency` (the `_vadv_drive`/`_vadv`
   pair, also used for T's `vert_adv_T`).  And the serial moist step recomputes
   physics PER-RK-STAGE (`_step_fv3.tendency_fn` calls `physics_fn` each stage,
   feeding `physics_tendency_cc` into `fv3_hydrostatic_tendencies`; dT_dt already
   includes the kessler dT at primitive_eq_cdgrid.py:1188).  So increment 4 =
   EXTEND `_build_hydro_tile_tendency_fns` to be tracer-aware: when a `q_pack`
   (+ injected `column_tracer_physics_fn`) is given, additionally compute
   `dq_pack = -(u . grad q)` (the shared `_tiled_scalar_horiz_advect`, packed)
   + `_vadv(q, _vadv_drive)` (vertical, the SAME driver as T) + kessler tracer
   rates, and add the kessler dT to dT_dt; **dry path (`tracers=None`) returns
   the 4-tuple bit-identically (gated dry step/tendency UNCHANGED)**.  Then a
   `make_tiled_fv3_hydrostatic_moist_step_stage_2d` threads the 5-tuple
   (u_d,v_d,T,p_s,q_pack) through the SAME SSP-RK3 combine (generic pytree axpy
   so the dry step's explicit RK3 is not duplicated) + a post-step
   `jnp.maximum(q,0)` floor (serial floor primitive_eq_cdgrid.py:1758-1765).
   Reference: the serial cube `model.step_with_physics(state, dt, kessler)` base
   cut.  sigma-only (kessler).  DEEP + must preserve the dry gate — build as its
   own iteration.

## Non-goals / caveats

- NOT Ginsburg-benchable (np>6 anti-scales on CPU; the value is the CAPABILITY +
  the parity receipt, not a speed number — do NOT sell a microbench as a win).
- Numerics ONLY from the shared `*_core` ops + the existing tiled kernels; the moist
  path adds NO new dynamics numerics (Kessler is the only physics, already shared).
- Base cut matches the dry step (div_damp=0, hyperdiff=0, non-duogrid); optional
  terms are later increments, not base-case-blocking.
