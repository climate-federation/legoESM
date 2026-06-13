# Cube tiled np>6 stage — PPM transport tiling design (2026-06-13)

STEP 2 of the full-tendency tiling grind (task #3; d2a2c DONE). The next
operator after d2a2c in `fv3_csw_tendencies` is the B-grid KE transport
(`_bgrid_ke_transport`, fv3_sw_core.py:2545), which is two 1-D PPM hord=9
sweeps (`_ppm_transport_1d`, :2278).

## The halo requirement (carefully derived — NOT a uniform h=4 exchange)

Production `_bgrid_ke_transport` provides the cross-face halo at depth
**h_dg=2** (`_pad_halo_dgrid_for_ppm(..., halo=2)`, :2576) and calls
`_ppm_transport_1d(external_halo=2)`. PPM then pads INTERNALLY to its
stencil halo **h3=4** with `mode='edge'` for the gap (:2300-2307). So
even globally the 2 outermost cells at a FACE boundary are edge-replicated.

PPM flux-at-interface stencil reach (sw_core.F90 jord=9, verified from the
code): the flux at interface j needs the monotone slope `dm[j±1]` (→
`vp[j-2..j+2]`) and the pmp/lac limiter `dq[j-2..j+1]`, i.e. ~3-4 cells
each side. h3=4 is the FV3 margin.

Per-tile consequence (tile owns face cells `[t·n_loc:(t+1)·n_loc]` on the
sweep axis):
* **Interior tile cuts** (neighbour tile is SAME FACE): the global field
  is contiguous-real there, so the tile needs the neighbour's **4 real
  cells** (depth-4) — a PLAIN copy, NO rotation, NO interp. depth-2 +
  edge-pad would put edge-replicated values where the global has real
  cells ⇒ flux mismatch at the tile-boundary interfaces. **depth-4
  interior halo is REQUIRED.**
* **Face-edge cuts** (neighbour is another FACE): only depth-2 cross-face
  data exists in production (rotated/interp'd) + PPM edge-pads 2 more —
  exactly the EXISTING d2a2c depth-2 cross-face machinery. Reuse it.
* **Corners**: a 1-D sweep reads only along its axis ⇒ **NO corner cells**.
  The hard 4×4 cube-vertex cascade is NOT needed for transport.

⇒ STEP 1 is a depth-4 SAME-FACE interior-cut strip exchange (plain
ppermute copy between same-face neighbour tiles) + the existing depth-2
cross-face path at face edges + PPM's own edge-pad. This is far more
tractable than extending `_build_tiled_pad` to a full h=4 (corner-cascade)
exchange.

## Corrections from codex design review (2026-06-13) — apply BEFORE coding

1. **Depth is 3-sufficient, 4-safe.** The active flux at an interior tile
   boundary reaches neighbour cell `N+2` (negative/upwind branch) and `-3`
   (positive branch at a left cut), so **depth-3 real same-face halo is
   sufficient**; depth-2 is INSUFFICIENT. Use depth-4 (safe, matches the
   `h3=4` storage convention; the 4th cell is non-load-bearing).
2. **rdelta (metric) ALSO needs an interior halo — MISSED in v1.**
   `_ppm_transport_1d` edge-pads `rd=rdelta` (`fv3_sw_core.py:2519`) and
   uses the UPWIND value for the CFL in BOTH flux branches
   (`rd_pad[:nn+1]` / `rd_pad[1:nn+2]`, :2520-2521, used :2530-2534). At an
   interior tile cut, edge-padding `rd` is WRONG for the branch whose
   upwind cell is in the neighbour tile ⇒ **interior cuts need a depth-1
   real `rdelta` halo along the sweep axis** (or a tile-specialised CFL
   that supplies the global upwind `rdelta` at the boundary interface). At
   true face edges, global edge-pads `rd` — MATCH that (no cross-face rd).
3. **Face-edge field halo must be BIT-IDENTICAL to `_pad_halo_dgrid_for_ppm`**
   (the depth-2 DGRID duogrid cross-face halo, `fv3_sw_core.py:40`), NOT a
   generic "d2a2c-ish" halo — the D-grid wind cross-face halo has its own
   rotation/interp; reproduce that exact function tile-locally, then PPM
   edge-pads to 4.
4. **Corners-free is `_bgrid_ke_transport`-ONLY (scope).** The two PPM
   calls are independent 1-D sweeps (axis=2 ytp_v, axis=1 xtp_u) — no
   diagonal read. But the LATER tendency transports do NOT generalise:
   `transport_step` (mass) and `fv_tp_2d` (vorticity, Lin-Rood 2-D with
   cross-sweep halo exchanges + del-n damping, `fv_tp_2d.py:922-1067`) ARE
   2-D and WILL need corners — a separate (harder) sub-build later.

Corrected STEP-1 halo spec: interior same-face FIELD depth-4 (≥3) real +
RDELTA depth-1 real; face-edge FIELD = `_pad_halo_dgrid_for_ppm` depth-2 +
PPM edge-pad, rdelta edge-pad (match global); NO corners (this op only).

## Plan (gate + codex EACH piece — user directive "be very careful")

1. **depth-4 interior-cut strip exchange** (`make_tiled_ppm_halo` or a
   `_build_tiled_pad` halo=4 EDGE-ONLY path): per sweep axis, each tile
   ppermutes its 4 boundary cells to the same-face neighbour tile; at
   face-edge cuts fall back to depth-2 cross-face (existing) + edge-pad.
   Parity vs the GLOBAL `_pad_halo_dgrid_for_ppm(halo=2)` →
   `_ppm_transport_1d` intermediate, sliced to the tile, on INTERIOR
   tiles (kt=3) and FACE-EDGE tiles. f64 atol/rtol 1e-12.
2. **tile-local transport** (`_ppm_transport_1d_tile` or feed the depth-4
   tile into the existing `_ppm_transport_1d` with the right
   external_halo): the per-tile transported field == global transport
   sliced to the tile, kt=3 all tile positions.
3. Wire into the tiled stage alongside d2a2c.
4. Bernoulli/gradient + vorticity/PV operators (next ops).
5. Full RK tendency shard_map stage + np24 parity (vs serial) + bench
   (target ~15-30ms vs GSPMD-auto 2253ms C96).

No measurable payoff until the full stage assembles — deliberate
multi-session grind. Each piece: extract → host+in-shard_map parity gate
→ codex adversarial review → commit.
