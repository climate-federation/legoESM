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
- **REMAINING (U3-scale HARD + future-HW, the genuine deferred-deep work):**
  - **mass `cgrid_mass_flux_divergence` (PPM)** — HARDEST op. BASE case
    (apply_fortran_xppm_boundary=False, non-duogrid) reduces to: halo=2 in-stage
    scalar exchange of h (`make_tiled_pad_body(halo=2)`) + the cc-wind VECTOR
    halo + `fv3_cc2c_core` (u_c/v_c are intermediates too) -> `_ppm_reconstruct_1d`
    (xppm boundary OFF -> `n_interior` unused) -> upwind face -> flux ->
    divergence; NO `synchronize_cgrid_fluxes` (non-duogrid). The face-extent
    boundary logic (`n_interior=n`) only bites with the Fortran xppm overrides
    (a later increment). NEXT unit.
  - Optional momentum terms (div damp / hyperdiff / boundary smoothing /
    Fortran corner specials) — each rides the shipped per-op kernels + one more
    in-stage halo; their own increments.
  - With dh_dt tiled, the FULL `fv3_sw_tendencies` np24 stage = momentum
    assembly + mass assembly sharing the cc-wind vector halo.
- LESSON: a prior session built tiling infra; ALWAYS grep `tests/parallel/
  test_tiled_*` + `parallel.mesh` before tiling a cube op.
