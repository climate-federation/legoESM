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

## APPROACH C + boundary-fix-OFF (2026-06-13 — the implementation approach)

Two findings collapse the implementation to the proven d2a2c approach-C
pattern (compute halo GLOBALLY in the GSPMD view, strided-slice per tile),
NOT a new ppermute exchange:

1. **`_pad_halo_dgrid_for_ppm` is a FULL-FACE op** (built on `ext_vector_dgrid`
   — D→A avg + duogrid cross-face rotation, `fv3_sw_core.py:40-84`), NOT a
   strip exchange. Reproducing it tile-locally is wrong-headed. Instead:
   compute it GLOBALLY (cheap, once per face), pre-pad the field to the PPM
   storage halo h3=4 exactly as `_ppm_transport_1d` does internally
   (`_pad_halo_dgrid_for_ppm(halo=2)` then `jnp.pad(mode='edge', gap=2)`),
   then each tile takes a strided window `[t·n_loc : t·n_loc + n_loc + 2·h3]`
   (= `tiled_padded_block`, the d2a2c approach-C helper). Interior cuts get
   real depth-4 from the contiguous global interior; face edges get the
   ext_vector depth-2 + edge-pad — bit-faithful to global by construction.
   Same for the courant `c` (interface-located, tile-sliceable) and `rdelta`
   (global edge-pad depth-1 then slice — satisfies the rd-halo requirement).
2. **Boundary-fix is OFF for THIS transport.** `_bgrid_ke_transport` calls
   `_ppm_transport_1d` WITHOUT `apply_d_sw3_boundary_fix` (default False;
   "iter-967 NEGATIVE: d_sw3 boundary fix conflicts with the iter-945
   halo"). So the SOUTH/NORTH boundary-fix branch (`fv3_sw_core.py:2449-2516`
   — the one-sided edge-PPM specials + pert_ppm) is DEAD here ⇒ the
   reconstruction is UNIFORM PPM, NO edge specials. Unlike d2a2c (which had
   intricate per-side edge specials), tiled transport is uniform — the only
   tile-position-dependence is the halo slice, which approach-C handles.

IMPLEMENTATION UNITS (gate + codex each):
* U1: refactor `_ppm_transport_1d` → extract `_ppm_flux_core(vp_h3, c, rd_pad)`
  taking a PRE-PADDED field (h3=4) + courant + pre-padded rd, with the public
  `_ppm_transport_1d` calling it (BIT-IDENTICAL global path — sbatch parity
  gate). This removes the internal edge-pad so a tile can supply real halo.
* U2: `transport_tile` = global pre-pad (U1 inputs) → `tiled_padded_block`
  slice per tile → `_ppm_flux_core` → tile interface fluxes. Parity: tiled
  == global `_bgrid_ke_transport` PPM output, all tile positions, kt=3.
* U3: wire into the tiled stage next to d2a2c; np-parity.

This is deliberate multi-hour work (a delicate dycore refactor kept
bit-identical); do U1→U2→U3 as separate gated+codex commits.

## U3 design (shard_map stage) — mirror make_tiled_d2a2c_stage (2026-06-13)

U1 (rd_prepadded, 65fa25fe) + U2 (synthetic approach-C parity, 6779dcbf) +
U2b (production cross-face/staggered i-sweep parity, 2150ce17) PROVE the
per-tile PPM compute. U3 wires it into a `shard_map(face, tile_i, tile_j)`
stage, mirroring `make_tiled_d2a2c_stage` (HISTORICAL: packages/core/legoesm/
parallel/tiled_d2a2c.py was deleted as test-only in #1863; see git history) —
the established APPROACH-C-under-shard_map pattern at the time:
* inputs FACE-REPLICATED `P("face", None, None)`: the GLOBAL h3=4 pre-padded
  transport field (computed outside shard_map in the GSPMD view, e.g.
  `_pad_halo_dgrid_for_ppm` then `jnp.pad` to h3), the courant, and the
  depth-1 pre-padded rd;
* inside `_stage`: `ti/tj = axis_index`; each device `lax.dynamic_slice`s
  its tile's sweep window (i-window `[ti*nl : ti*nl+nl+2*h3]`, j-block) from
  the replicated padded face — NO halo ppermute (the padded face already
  carries the h3 halo, the d2a2c-stage insight), then
  `_ppm_transport_1d(external_halo=4, rd_prepadded=True)` per tile;
* output tile-sharded `P("face","tile_i","tile_j")`; reassembly dedups the
  shared boundary interface (lower tile owns it), == global PPM sweep.

CARE-POINT (regression risk, do carefully + gated): the STAGGERED j-axis
slice (u_d j=n+1). Start with an i-ONLY mesh `(6, kt, 1)` (full j, no
staggered-j tiling → np=6·kt, kt=4 gives np24) to prove the shard_map
i-sweep first; then the full `(6, kt, kt)` with the staggered-j block slice
(match `tiled_face_block`'s nl-vs-nl+1 convention) = U3b. Then ytp_v
(axis=2 j-sweep, symmetric). Parity: in-shard_map vs global
`_ppm_transport_1d` (pattern: tests/parallel/test_tiled_d2a2c_ua_va.py,
XLA_FLAGS=--xla_force_host_platform_device_count). THEN real Courant
(ub/vb from uc/vc) + B-grid corner sync = full `_bgrid_ke_transport` tiled.

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

## U3d — real-Courant composition of _bgrid_ke_transport (2026-06-13)

Status after U3b (full 2-D `(6,kt,kt)` sweep tiling, both i/j) + U3c
(`bgrid_corner_courant_local`, corner Courant tiled): the remaining
transport pieces are the **real-Courant composition** + the **BGRID_NE
corner sync**. The CRUX (verified this session) is the STAGGERED cross-axis
mismatch — do NOT assume the U3b synthetic matched shapes:

- ytp_v (axis=2) call: `_ppm_transport_1d(v_d_jhalo, vb, rdy, axis=2,
  external_halo=h_dg=2)`. Shapes: `v_d_jhalo (6, n, n+1+2h)` [field, cell on
  axis=1, j-halo'd corner on axis=2], `vb (6, n+1, n+1)` [CORNER Courant],
  `rdy (6, n+1, n)`. Inside, swapaxes(1,2) makes the sweep axis=1; the
  field's CROSS axis is `n` (i-cells) but `vb`'s cross axis is `n+1`
  (i-corners) — **off by the staggering**. So a tile's `vb` cross slice
  (n+1 corners → nl+1) is ONE LONGER than the field's cross slice (n cells →
  nl). The U3b `transport_jsweep_tile_2d` cross slice (`[a_i:a_i+nl]`) is
  CORRECT for the field but the courant needs its own `[a_i:a_i+nl+1]` (or
  the relevant n-of-(n+1) sub-slice that `_ppm_transport_1d` actually
  indexes — TRACE the courant cross-indexing in `_ppm_transport_1d` before
  slicing; it likely uses only `n` of the `n+1` corner courant rows).
- xtp_u (axis=1) is the symmetric stagger (ub corner cross-axis n+1 vs u_d
  cell n).

PLAN (each: extract → host + in-shard_map parity → codex → commit):
1. TRACE `_ppm_transport_1d`'s courant cross-axis usage (which `n` of the
   `n+1` corner-courant rows it reads) → derive the exact courant tile slice.
   Add a `transport_sweep_tile_2d`/`_jsweep` variant (or a `courant_cross`
   arg) that slices the courant cross-axis by `nl+1` (corner) while the field
   cross by `nl` (cell). Parity: real cdgrid + real vb/ub (from U3c on the
   real cross-face-halo'd uc/vc) → tiled transported_y/x == global
   `_bgrid_ke_transport` (BEFORE Step-5), on INTERIOR (kt=3) + face-edge.
2. CROSS-FACE HALO (approach-C, GLOBAL pre-pad → slice; NOT tiled): the
   stage takes the globally-prepadded `uc_pad/vc_pad`
   (`_pad_halo_uc_vc_new_via_old_delta`) + `u_d_ihalo/v_d_jhalo`
   (`_pad_halo_dgrid_for_ppm`) as FACE-REPLICATED inputs; the tile slices
   them. The corner metrics (`cosa_corner/rsin2_corner`, `(6,n+1,n+1)`) slice
   via the mesh.py STAGGERED `tiled_face_block` (nl+1 ownership).
3. BGRID_NE corner sync (Step 5, the c2l z-matrix corner rotation) — the
   ONLY genuinely cross-TILE-coupled piece (a corner vector avg through the
   geographic frame); needs a tiled corner exchange (mirror the d2a2c
   corner rounds). Highest-risk piece; gate hardest.
4. Assemble `bgrid_ke_transport_tiled_stage` (corner Courant → sweeps →
   corner sync) under one `shard_map(face,tile_i,tile_j)`; np24 parity vs
   serial.
5. Bernoulli/gradient + vorticity/PV → full RK tendency stage → np24 bench.

GATE NOTE (banked): the bit-identity guards for any `_bgrid_ke_transport`
refactor are conftest-`slow`-marked (run with `-m "slow or not slow"`) AND
hit the LLVM section-memory OOM at 32G → use **64G + `XLA_FLAGS=
--xla_cpu_multi_thread_eigen=false` + `TF_NUM_INTEROP_THREADS=1`**.

DELIBERATE multi-session work — NOT loop micro-turns. No measurable SYPD
payoff until the full RK stage assembles, AND the np24 payoff is NOT
Ginsburg-benchable (no 24 real devices; CPU-virtual oversubscribes) — it is
a multi-GPU-node (A100/H100) capability validated for CORRECTNESS here.

## U3d shapes — RESOLVED by runtime probe (job 8480355, C18 duogrid, n=18)

`scripts/tmp/_probe_bgrid_shapes.py`. n+1=19, h_dg=2 → n+2h=22.
- `u_d (6,n,n+1)=(6,18,19)`, `v_d (6,n+1,n)=(6,19,18)`
- `uc (6,n+1,n)` → `uc_pad (6,n+1,n+2)`; `vc (6,n,n+1)` → `vc_pad (6,n+2,n+1)`
- `vb=ub=(6,n+1,n+1)=(6,19,19)` (corner)
- `u_d_ihalo (6,n+2h,n+1)=(6,22,19)`; `v_d_jhalo (6,n+1,n+2h)=(6,19,22)`
- `dy_edge_x=(6,n+1,n)`; `dx_edge_y=(6,n,n+1)`

ytp_v `_ppm_transport_1d(v_d_jhalo, vb, rdy=1/dy_edge_x, axis=2, ext=h_dg=2)`:
**ALL share CROSS axis=1 = n+1 (corners)** — `v_d_jhalo[:,n+1,·]`,
`vb[:,n+1,·]`, `rdy[:,n+1,·]`. Sweep axis=2: field n+2h, vb n+1 (interfaces),
rdy n (cells). So the U3b `transport_jsweep_tile_2d` composes with the REAL
arrays IF the **cross slice = nl+1 (corner ownership, shared corner), not
nl** — NO field-vs-courant cross mismatch (my earlier worry was wrong; the
field's cross is n+1, not n). xtp_u is the symmetric transpose (cross
axis=2 = n+1).

⇒ U3d impl (next): a corner-cross sweep tile = U3b j/i-sweep with cross
slice nl+1 + the U2b global-pre-pad-to-h3 sweep handling, driven by the REAL
vb/ub (from U3c on real cross-face-halo'd uc/vc) + real v_d_jhalo/u_d_ihalo.
Reassemble: cross (corner) lower-tile-owns-shared nl+1→n+1; sweep
lower-tile-owns-shared nl+1→n+1. Parity vs global transported_y/x (pre
Step-5) on kt=3 interior + face-edge. Then Step-5 BGRID_NE corner sync.

## U3f — tiled BGRID_NE corner SCALAR sync: KEY SIMPLIFICATION (2026-06-13)

After U3e (pointwise local↔geo factored), the only remaining corner-sync
piece is the cross-tile `synchronize_corner_scalar` (halo.py:2678). Reading
it (verify-first) gives a MAJOR simplification for the tiled stage:

**synchronize_corner_scalar modifies ONLY the FACE-BOUNDARY corners**
(Pass 1: face EDGES i=0/n, j=0/n via CONNECTIVITY pairwise avg; Pass 2: the
8 cube VERTICES, 3-face mean). The face-INTERIOR corners are UNTOUCHED.

⇒ The tiling introduces interior cuts, but the sync does NOT average at
interior corners — so the tiled stage needs **NO interior-cut corner sync**.
Each interior corner is a single global value; approach-C (face-replicated /
global-pre-synced input, per-tile slice) hands every sharing tile the SAME
value automatically. This KILLS the hardest-feared part (the same-face
interior-cut all-reduce / d2a2c diagonal+sliver rounds are NOT needed for the
scalar sync).

**Remaining tiled work = ONLY the cross-FACE part at FACE-BOUNDARY tiles:**
- Pass 1 cross-face edge avg: only tiles with ti∈{0,kt-1} or tj∈{0,kt-1}
  hold a face edge; exchange+avg that edge with the CONNECTIVITY neighbour
  face's edge tile. O(n) per face edge.
- Pass 2 vertex 3-face mean: only the 4 corner tiles (ti,tj ∈ {0,kt-1}²) per
  face hold a cube vertex; 3-face mean across the corner tiles meeting there.
- _tiled_corner_modes mode 0 (true cube vertex) already enumerates the
  vertex-tile set; modes 2/3 (slivers) and 1 (diagonal) are NOT needed
  (those are interior/halo-fill, irrelevant to the boundary-only scalar sync).

**Cleanest approach-C implementation:** run synchronize_corner_scalar
GLOBALLY on the (face-replicated) corner field BEFORE the per-tile slice —
it is a cheap O(n) edge/vertex op, NOT per-cell. The full-field is already
face-replicated in the stage inputs (P("face",None,None)), so the global
sync needs NO collective beyond what the face-replication already provides;
the tiled stage then slices the synced corners. I.e. the corner SCALAR sync
can live in the GLOBAL pre-step (like the cross-face halo pre-pad), NOT
inside the per-tile shard body — NO new tiled cross-face exchange required.
Validate: synced-global-then-sliced == per-tile-of-(global sync). This is
the same approach-C move as the d2a2c cross-face halo (global view → slice).

⇒ U3f is therefore SMALL: thread the (already-global) corner scalar sync +
the U3e pointwise geo conversion into the corner-sync wrapper so the stage
consumes pre-synced corner Courant. Then Step-6 KE (pointwise) + assemble.
