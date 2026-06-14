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

## Status
- Design pinned (this doc). Increment P-i (`dgrid_vorticity`) next.
