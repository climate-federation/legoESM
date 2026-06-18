# Tiled cube hydrostatic STEP — design (next track after the tendency capstone)

**Status (2026-06-17):** the full `fv3_hydrostatic_tendencies` TENDENCY is now
tiled on np=6·kt² (`make_tiled_fv3_hydrostatic_tendencies_stage_2d`, commit
4ac0cb3ee, gate 8512958 5/5). This doc scopes the remaining track: wrapping that
tiled tendency in the full SSP-RK3 **time step** (the production `_step_fv3`).
**Future-HW only** — np>6 anti-scales on Ginsburg's Gloo/TCP fabric; gate by
BIT-IDENTITY vs the global `_step_fv3` base cut, never a wall-clock number.

> **UPDATE (2026-06-18) — BUILT.** This track is complete:
> `make_tiled_fv3_hydrostatic_step_stage_2d` (no-gather 3-stage SSP-RK3, commit
> 308202ff3), `make_tiled_zero_mean_tendency_stage_2d` + the delta-first
> `make_tiled_fix_ps_mass_stage_2d` (post-RK3 conservation psums, 798e174f5 +
> b66f82bb7) are all in `parallel/tiled_production_cdgrid.py`, bit-identity- /
> conservation-gated (np24/np54), cavecrew-clean and codex-reviewed (3 findings
> fixed: mesh/kt + nl≥2 guards 0f50c4afe, f32 delta-first fixer b66f82bb7).
> **Still UNWIRED into production** (`ModelDriver` / `make_sharded_step` do not
> call the tiled step — the driver warns) — by design: np>6 anti-scales on Gloo,
> so wiring is deferred to NVLink/IB/TPU hardware. The capability exists and is
> verified; the production hookup is the remaining future-HW task.

## The layout problem (the crux)

`ssp_rk3_step(state, tendency_fn, dt)` (timestepping/ssp_rk3.py) is generic: it
calls `tendency_fn(state) -> tend` (SAME pytree layout as `state`) then does
pointwise `axpy`/`linear_combination`. The state is FACE-REPLICATED
(`P("face",None,None,None)`); the tiled capstone OUTPUTS are TILE-SHARDED, and
the D-grid winds are CORNER-STAGGERED with a duplicated shared tile face
(gathered `(6, kt·(nl+1), kt·(nl+1), nlev) != (6, n+1, n+1, nlev)`). So the
tendency layout ≠ the state layout — the RK3 pointwise combine cannot align them
directly.

## Two approaches

1. **Gather-RK3 (capability demo, LOW value):** `tendency_fn` runs the tiled
   capstone then DEDUPs the corner-staggered tiles back to face-replicated
   `(6,n+1,n+1,nlev)` (lower-owns-shared, like the gate's corner compare) before
   returning. `ssp_rk3_step` then runs unchanged on face-replicated arrays. Proves
   the step is bit-identical with a tiled tendency, BUT the per-stage gather is
   O(global) comm → negates the tiling benefit. NOT a scaling win — skip unless a
   pure capability checkpoint is wanted.

2. **One-shard_map-3-stage (the REAL no-gather win):** put the entire RK3 (3
   stages + the 2 SSP combines) inside ONE `shard_map`. Body per device tile:
   `tend = <capstone body>(tile_state); k1 = state + dt·tend; ...` ×3. State stays
   TILE-LOCAL across stages (no gather); halos exchanged 3× (once per stage's
   tendency). ~3× the capstone body. **Requires:** the corner-staggered
   duplicated shared-face tendency to be BIT-IDENTICAL between adjacent tiles so
   the duplicated state stays self-consistent across stages (verify first — the
   momentum gate's lower-owns-shared check is necessary but the cross-stage
   accumulation must also stay consistent). Gate: bit-identity of the stepped
   state vs the global `_step_fv3` base cut.

## Increment plan

1. **Tiled global-reduction primitive (`psum`-in-shard_map):** the first tiled
   GLOBAL reduction (all tendency stages so far are halo-only). Validate via the
   `zero_mean_tendency` variant (the capstone's documented `_apply_zero_mean_
   per_stage=True` follow-up) — an area-weighted `psum` over (face, tile_i,
   tile_j) of dp_s/dt before omega. Needed downstream by the mass-fixer.
2. **Shared-face tendency self-consistency check** (gate): adjacent tiles' shared
   corner-face `du_d_dt`/`dv_d_dt` are bit-identical — the precondition for the
   no-gather state to stay consistent across RK3 stages.
3. **One-shard_map 3-stage SSP-RK3** wrapping the capstone body (base cut: no
   post-step). Gate vs global `_step_fv3` base cut, sigma+hybrid, kt2/kt3.
4. **Post-step ops** (each needs a tiled global reduction / the psum primitive):
   `fix_ps_mass` (global mass), `damp_v` (post-step vorticity damp), the
   `sponge_implicit` multiplicative damp (pointwise, easy). Add incrementally,
   gated.

## Notes
- Reuse the capstone's halos verbatim (scalar `{B,zeta,1/T,ln_ps,hf,T}` + vector
  `vert_adv_uv` lift); the 3-stage body re-exchanges them per stage.
- The cc reduction-bearing tendencies (dT, dp_s) match to ~5e-8 under XLA fusion
  (diag 8512922/8512950 — reordering, not a bug); the stepped state will inherit
  similar O(1e-8) reordering on the cc fields — gate the step at a diag-justified
  tol, keep corner winds at 1e-10.
- All bit-identity, no wall-clock; np>6 is future-HW (TPU pod / NVLink), not
  Ginsburg-benchable.
