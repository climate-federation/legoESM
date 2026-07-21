# Cube >6-device sub-face tiling — PORT TO THE PRODUCTION DYCORE (task (a))

**Goal.** The U3/U4 sub-face tiling (d2a2c, PPM transport, B-grid KE, d_sw1 ut/vt)
proved approach-C on the EXPERIMENTAL FB chain (`_d_sw_native`). Production cube
uses a DIFFERENT op set — `fv3_sw_tendencies` (operators_cdgrid.py, SW) and
`fv3_hydrostatic_tendencies` (primitive_eq_cdgrid.py, 3D AMIP). To let the
PRODUCTION cube scale past 6 devices (np = 6·kt·kt) on future fast-interconnect
HW (TPU pods / NVLink), tile the production ops with the SAME approach-C.

**Not Ginsburg-benchable** (Ginsburg = ≤2 GPU/node, no IB/NVLink; np>6 = CPU
shard_map which anti-scales on Gloo-TCP, or cross-node ppermute 70-245× < HBM).
So every increment is gated by **bit-identity** (tiled reassembly == global), the
proven U3 methodology — correctness, not a wall-clock win.

## Approach-C recap (reuse the U3/U4 machinery)
1. cheap GLOBAL face-replicated pre-pad (cross-face halo) — already
   `packed_pad_halo_4d` / `pad_halo_*` (face-replicated stage inputs).
2. per-tile `lax.dynamic_slice` of the tile window from the padded global field.
3. device-uniform local body (the op's stencil) — NO in-stage ppermute.
4. reassemble (lower tile owns the shared staggered face) == the global op.
5. shard_map stage on a `(6, kt, kt)` mesh (`tiled_d2a2c.make_tiled_*` pattern).

## Production `fv3_sw_tendencies` ops (operators_cdgrid.py:1490) — tile order
Each is a local stencil after a 1-cell (or h) halo → tileable. Increment order
= simplest-first, each its own kernel + bit-identity test + shard_map stage:

- **P-i  `dgrid_vorticity`** (corner winds → cell-centre ζ): local curl stencil.
  Smallest, no metric subtlety beyond rarea. FIRST increment (establish the
  production-tiling test harness mirroring tests/parallel/test_tiled_*).
- **P-ii `interp_corner_to_center` / `interp_center_to_corner`** (4-cell avg):
  identical box-stencil to U4a dsw1 ut/vt — reuse the dsw1_ut_vt_tile_2d shape
  logic. The Bernoulli-gradient corner winds (sec e) + dB cc (sec g) use these.
- **P-iii `arakawa_lamb_gradient`** (B, ln_ps): the corner AL gradient — already
  `padded=`-aware (the PE coalescing reuses it). Tile the corner stencil.
- **P-iv `cgrid_divergence` / `cgrid_mass_flux_divergence`** (mass tendency +
  div damp): C-grid flux divergence — local; tile the flux + divergence.
- **P-v  `fv3_d2cc` / `fv3_cc2c`** (D→cc→C velocity): vector interp (needs the
  vector halo rotation at cube edges — like the d2a2c vector case). HARDEST
  (vector); do after the scalar ops.
- **P-vi  Bernoulli `KE = 0.5*(u_cc²+v_cc²)` + `B = KE+g(h+h_s)`**: pointwise
  (trivial; rides whatever cc winds the tiles hold).

Then assemble the full `fv3_sw_tendencies` shard_map stage → bit-identity vs
global at C18/C24 (kt=2,3) → the SW production np24 capability.

## 3D follow-on (`fv3_hydrostatic_tendencies`)
Same ops + the vertical (nlev) trailing axis (all kernels already 4D-native via
the `(F,n,n,nlev)` shape). The PE stage-pack coalescing (ln_ps/hf/div_v,
shipped) already tiles the HALO; the per-op stencils are the remaining tiling.

## Gate strategy (per increment)
- Host-body bit-identity (kt=3 → interior+edge+corner tiles), `rtol=0,atol=1e-12`
  (portable; exact is non-portable per the U4a FMA finding).
- `(6,2,2)` shard_map np24 (skip < 24 host devices), same reassembly.
- Duplicate-shared-face parity (validates lower-owns trim).
- codex adversarial review each increment; smoke = the bit-identity gate.

## PRE-EXISTING tiling (Jun-11, via parallel.mesh staggered_tile_block + *_local cores) — DO NOT RE-DUP
A prior session already tiled several cube ops (host-body + shard_map) using
`staggered_tile_block`/`tiled_face_block` (parallel.mesh) + leading-axis-agnostic
`*_local` cores, with tests `test_tiled_{cgrid_divergence,cgrid_gradient,
d2a2c_ua_va,cdgrid_field_classification,staggered_layout}`:
- **`cgrid_divergence`** (`cgrid_divergence_local`) — DONE, and it IS the
  fv3_sw_tendencies mass/div-damp divergence → reuse it (do NOT re-tile; P-iv
  cgrid_divergence was attempted then reverted as a dup 2026-06-14).
- `cgrid_gradient_2d` (`cgrid_gradient_2d_local`) — DONE, but a DIFFERENT op
  (NOT arakawa_lamb_gradient; not on the fv3_sw_tendencies path).
- d2a2c / staggered-layout / field-classification — infra + experimental-d2a2c.
The two slicing styles coexist: Jun-11 `staggered_tile_block` is STATIC-index
(host-body); my `tiled_production_cdgrid` uses `dynamic_slice_in_dim` (works in
shard_map too). Both call the shared `*_local`/`*_core` numerics (no dup of
numerics).

## Status (corrected after the Jun-11 reconciliation)
- **DONE (my P-series, tiled_production_cdgrid.py, fv3_sw_tendencies-relevant):**
  P-i `dgrid_vorticity` (f0246a46), P-ii box interps (79243312), P-iii
  `arakawa_lamb_gradient` (1ab7edae). All host-body 3D+4D + np24 shard_map,
  bit-identity-gated, codex-clean.
- **DONE (Jun-11):** `cgrid_divergence` — reuse `cgrid_divergence_local`.
- **P-v `fv3_d2cc` + `fv3_cc2c` SHIPPED (406ddc6a)** — velocity transforms
  (fv3_d2cc local 2-pt avg; fv3_cc2c VECTOR via the pad_halo_vector pre-pad +
  the extracted fv3_cc2c_core; 3D-only, no 4D caller). host-body + np24, codex-clean.
- **MOMENTUM tendency (du_d_dt/dv_d_dt) now FULLY tileable** — every op it needs
  is done: P-v (cc/C winds) -> Bernoulli (pointwise) -> P-iii (A-L grad) + P-ii
  (corner winds + interp_corner_to_center) -> P-i (dgrid_vorticity) ->
  cgrid_divergence (Jun-11, div damp).
- **ASSEMBLY: tiled `fv3_sw_tendencies` MOMENTUM stage SHIPPED** —
  `make_tiled_fv3_sw_momentum_stage_2d` (tiled_production_cdgrid.py). The
  genuine np24 momentum unlock: chains fv3_d2cc -> Bernoulli (inline pointwise)
  -> [B SCALAR in-stage halo] -> A-L grad -> [u_cc/v_cc VECTOR in-stage halo]
  -> corner winds -> dgrid_vorticity -> interp_corner_to_center -> du_cc/dv_cc
  -> [VECTOR in-stage halo] -> D-grid project. The INTERMEDIATE halos (B; cc
  winds; cc tendencies — no global pre-pad) are exchanged IN-STAGE via the
  EXISTING `make_tiled_pad_body` (scalar) / `make_tiled_pad_vector_body`
  (vector) — the foundational sub-face halo (incl. the mode-1 diagonal corner
  ppermute) was already built (U3). Base case only: div_damp=0, hyperdiff=0,
  boundary_fix=False, fortran_*=False, non-duogrid. Bit-identity vs global
  `fv3_sw_tendencies` (du_d_dt, dv_d_dt) at kt=2 (np24) + kt=3 (np54),
  rel<1e-10; codex-clean (7/7 vectors). Gate: `test_tiled_fv3_sw_momentum.py`.
- **MASS-PPM `cgrid_mass_flux_divergence` (dh_dt) SHIPPED** —
  `make_tiled_cgrid_mass_divergence_stage_2d` (tiled_production_cdgrid.py). The
  design-doc HARDEST op, tiled via the U3 DEEP-GLOBAL-PRE-PAD pattern (NOT an
  in-stage halo): `h` is a STAGE INPUT, pre-padded one ring deeper than the
  production halo=2 (`h_deep` = `_pad_halo_auto_h2` + 1 edge ring, n+6) and the
  deep window sliced per tile -> the per-tile PPM reconstruction is LOCAL. The
  cc winds u_c/v_c are staggered stage inputs (NO halo — divergence reads only a
  cell's own bounding faces). Depth insight: PPM reconstruction of the
  tile-boundary cell (local -1) reads cells [-3..1], so a tile needs `halo_in=3`
  (one ring deeper than the global's halo=2); the deep pad's outer ring is
  edge-extended, so a FACE-edge tile reproduces the global's internal
  `mode='edge'` ghost. Achieved by `_cgrid_ppm_fluxes_core(h_pad, u_c, v_c, dy,
  dx, n_local, *, halo_in)` extracted from `_cgrid_ppm_fluxes_2d_no_sync` — ONE
  core, `halo_in=2` drives the global op (the inline 2D path + the 4D no_sync
  wrapper both now call it, dedup), `halo_in=3` drives the tile; the face
  indices `q_R[halo_in-1:...]`/`q_L[halo_in:...]` collapse to the exact historical
  `[1:n+2]`/`[2:n+3]` at halo_in=2. Base case: non-duogrid (no
  `synchronize_cgrid_fluxes`), `apply_fortran_xppm_boundary=False` (the
  `n_interior` face-edge override stays a later increment). Bit-identity vs
  global at kt=2 (np24) + kt=3 (np54), rel<1e-10; codex-clean (8/8 vectors).
  Refactor proven bit-identical: clean-HEAD vs refactor isolated SW integration
  runs give IDENTICAL results to 16 digits (test_fv_cubesphere 7/7,
  test_shallow_water 11/11; test_boundary_fix mass-conservation is a PRE-EXISTING
  2.51e-6-vs-1e-6 miss, identical on clean HEAD). Gate:
  `test_tiled_mass_divergence.py`.
- **FULL `fv3_sw_tendencies` np24 STAGE SHIPPED (CAPSTONE)** —
  `make_tiled_fv3_sw_tendencies_stage_2d` (tiled_production_cdgrid.py) ->
  `stage(h,u_d,v_d,h_s) -> (dh_dt, du_d_dt, dv_d_dt)`. Composes the momentum
  assembly + mass divergence, with the cc-wind VECTOR halo computed ONCE and
  SHARED: one `vector_body(u_cc,v_cc)` feeds BOTH `fv3_cc2c_core` (-> u_c/v_c
  for the mass PPM) AND the momentum corner winds — so the full stage carries
  the SAME halo budget as momentum alone (1 scalar B + 2 vector + the global
  deep-h pre-pad). h for Bernoulli is taken from the deep-window interior
  (`hw[:, 3:-3, 3:-3]`). Bit-identity of ALL THREE tendencies vs global at kt=2
  (np24) + kt=3 (np54), rel<1e-10; codex-clean (8/8 composition vectors). Gate:
  `test_tiled_fv3_sw_full.py` (job 8482569: TILED_FULL_GATE_OK). The production
  cube SW dycore tendency now runs on np=6*kt^2 devices.
- **MULTI-NODE VALIDATION** — `scripts/validate/validate_tiled_fv3_sw_multinode.py`
  + `scripts/cluster/scaling_ginsburg/tiled_fv3_sw_multinode.sbatch`: runs the
  full stage under REAL multi-controller `jax.distributed` across 2 nodes (np24,
  in-stage ppermutes -> cross-NODE collective-permute); each process
  self-validates its local tile vs serial (rel<1e-9). Correctness, not a bench.
- **PRODUCTION ASSEMBLY WIRED (2026-07-09)** — the blocked persistent step
  (`make_tiled_fv3_hydrostatic_step_blocked_2d`, input layout == output
  layout, in-stage telescoping mass fixer, optional moist column physics)
  + `make_tiled_cc_loop` (adapter enter/step/exit_) + the
  `run_cpu_mpi_scaling --cs-spmd` 6·kt² dispatch and
  `bench_cube_tiled_step_scaling --closed-loop` lane.  Bitwise-identical to
  the gated step stages (dry + moist gates in
  `tests/parallel/test_tiled_blocked_loop.py`).  See
  `cube_tiled_step_design.md` (2026-07-09 update) for the full contract.
- **REMAINING (later increments, NOT base-case-blocking):**
  - Optional momentum terms (div damp / hyperdiff / boundary smoothing /
    Fortran corner specials) — each rides the shipped per-op kernels + one more
    in-stage halo; their own increments.
  - The Fortran xppm boundary overrides (`n_interior` keyed to GLOBAL face
    index) + duogrid `synchronize_cgrid_fluxes` for the mass tile.
  - Kessler column bridge for the moist blocked loop in the drivers +
    `ModelDriver` (full-model coupling) hookup.
- LESSON: a prior session built tiling infra; ALWAYS grep `tests/parallel/
  test_tiled_*` + `parallel.mesh` before tiling a cube op.
