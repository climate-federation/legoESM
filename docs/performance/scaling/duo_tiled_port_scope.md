# DUO → sub-face tiling port — SCOPE (draft for codex+GLM review, 2026-08-31)

## Goal
Run the CERTIFIED DUO FV3 dycore (bit-exact Zenodo-Fortran twin; hydrostatic +
NH; `fv3_dynamics.fv_dynamics_step`) on `np = 6·kt²` devices with neighbor-only
(O(halo)) communication — removing both measured DUO-lane limits:
the 6-device layout cap and the ~5.0× all-gather topology waste
(`bench_fv3_duo_ring_replay.py`, d=6, resolution-flat).

## Why a port, not reuse of the existing tiled lane's numerics
`tiled_production_cdgrid.py` tiles a DIFFERENT dycore (`operators_cdgrid`,
Eulerian sigma/hybrid vertical). Requirement is the faithful-FV3 lane
(Lagrangian vertical remap is FV3's signature; gap register + design review
2026-08-30 both conclude no experiment can waive it). What we REUSE from the
tiled lane is its PARALLEL MACHINERY and its METHODOLOGY (bit-identity gates,
`_check_tiled_mesh`, strip exchange, corner routing patterns), never its
numerics.

## Why the DUO lane cannot simply be resharded
The DUO halo lives in FLAT TABLES with global six-face indexing
(`fv3_duo_halos._FlatLayout`); under a sharded i/j axis those gathers lower to
full-materialization collectives (measured: the x[t] all-reduce storm; the
census/ring design assumes whole faces per device). The port therefore replaces
the EXCHANGE layer at tile granularity and re-hosts the certified KERNELS
(audit 2026-08-30, codex+GLM: all kernels are local stencils reach ≤ 3 +
per-column vertical solves; PORT FEASIBLE).

## Deck scope (frozen for the port; anything else is out)
Pinned oracle deck: dry (`zvir=0`), `consv_te=0`, `n_sponge=-1`, `tau=-1`,
`k_split=1`, static `n_split`, `km ∈ {5,10}`, `kord_* = ±9`, both hydrostatic
and NH arms. fp64 only (coarse-fp32 exists on the six-face lane; NOT ported in
v1). No HS forcing under multi-process (already refused). AD/`jax.grad` is NOT
a gate for this lane (parity lane, same as today) — noted, not built.

## Design pillars (each is a review question)
P1 **Tile ownership with persistent tile-sharded state** (the tiled lane's
   BLOCKED layout as implemented, tiled_production_cdgrid.py:2251): state is
   block-concatenated global arrays — e.g. `(6, kt*(nl+1), kt*(nl+1), km)` —
   partitioned `P("face","tile_i","tile_j", None)`; input layout == output
   layout; no per-step global gather. kt=1 degenerates to one-tile-per-face
   (the port's bridge config to the certified lane).
P2 **Exchange = ordered multi-source gathers per target, certified
   semantics.** NOT symmetric pairwise swaps (GLM): a stencil row whose
   sources straddle an internal tile boundary of the adjacent face is fed by
   2+ sender tiles — the split emits ORDERED multi-source gathers per target
   row. Neighbor topology is 8-WAY (diagonal-tile strips for within-face
   tile-corner blocks), per-stagger parameterized (A/B/C/D offsets shift
   ownership). The split is INTEGER-ONLY (pure index remap, zero FP
   arithmetic; table WEIGHTS are BIT-COPIED from the certified blobs and
   hashed into the build — never re-derived from formulas) with a
   build-time COMPLETENESS PROOF: every table row lands in exactly one
   owner, every source resolves to exactly one tile, else abort.
   Divisibility `n % (kt*nl) == 0` asserted at build (ragged tiles are a
   v-next decision made explicitly, not silently). Cube corners: the 3-way
   transform is NOT a composition of pairwise edge rotations; corner tiles
   get their diagonal block from the third face via a dedicated schedule
   whose merge order is FIXED (face-id order), at full depth ng.
P3 **Sub-step choreography preserved.** Halos fire INSIDE the n_split acoustic
   loop on partially-updated fields, exactly at the exchange sites the DUO
   step already has. The dispatch seam is the COMPOSED exchange APIs — the
   `ext_scalar_sixface` / `ext_vector_*_sixface` dispatchers
   (fv3_duo_halos.py:1957+) AND the separate `*_allk` dispatchers (:2012+) the
   acoustic path calls. The port swaps EVERY one of these dispatchers' ring
   arm for a tile arm behind the same name; an enumerated dispatcher checklist
   (grepped, not recalled) is part of M3's gate so no full-step exchange site
   is left on the ring path. Equality is SET + WIDTH + TIMING per firing
   (GLM): the tile arm must not fuse/batch firings across names nor apply
   early, honors per-call halo width, and every ghost read is served by the
   LAST firing of that name — pads are per-firing scratch, never
   built-once. One-off instrument on the twin asserts no
   ghost-read-before-wait, making this provable not assumed. NH halo widths
   (w, delz/pt) RESERVED in the layout now so M5 needs no relayout.
   k_split=1 ⇒ comm scales with n_split — cost model, not correctness.
P4 **Global ops become explicit collectives** (audit list, closed):
   tracer-CFL cube-wide Courant max (`fv3_tracer2d.py:322`) → tile-max +
   one `pmax` when `nq>0`; energy fixer (consv_te) stays gated OFF and
   REFUSED under tiling in v1 (raise, not silently skip).
P5 **Min-tile guard**: PER-STAGGER build-time assert against the WIDEST
   stencil incl. corner-Lagrange reach and NH (not one global constant);
   v1 floor `nl ≥ 8`. Guard raises at build.

## Gates (the tiled lane's proven methodology, adapted)
G0 **kt=1 bridge**: the tiled step at kt=1 == the certified six-face step
   BIT-IDENTICALLY (same kernels, same order, exchange degenerates to the
   certified cross-face maps; `ppermute` is bit-preserving and the frozen
   deck has NO horizontal reductions, so exactness is achievable). This is
   the load-bearing gate. TRIAGE RULE (codex): a G0 failure must first be
   attributed — changed XLA fusion/codegen from the new shard_map boundary
   vs a real exchange-semantic defect (changed evaluation order in weighted
   k2e / Lagrange / projection / signed-vector transforms). Only the former
   may relax G0 to the few-ulp envelope, with the lowering diff shown.
   PRECONDITIONS, in writing (GLM): bit-copied table weights (hashed),
   ordered UNROLLED per-target sums (no dot/tree-reduce, no
   scatter-accumulate), FMA contraction pinned off or emission parity
   proven, fixed corner merge order. COVERAGE CAVEAT: kt=1 has no interior
   tile edges — G0 certifies face-edge + corner semantics ONLY; interior
   strips first bind at G1/kt=2. SCOPE: G0 = certified kernels + tile-arm
   exchange ONLY (no kernel retiling in the graph). G0 is RE-RUN after M4
   as a regression gate.
G1 **Exchange unit gate**, two tiers. (a) Primitives (per stagger A/B/C/D +
   k2e + corners): tiled == flat-table output on random fields, kt ∈ {1,2,3},
   POISON-sentinel test (NaN/face-id, not zeros — catches width/packing/
   orientation/permutation), census width assert. (b) COMPOSED pipelines:
   the complete `ext_vector_c_sixface` and `ext_vector_d_sixface` flows
   (c2l, geographic-lattice exchange, projection, strip write-back, final
   Lagrange fills — fv3_duo_halos.py:1910/:1934) tiled-vs-flat at kt ∈
   {1,2,3}; primitives alone can pass while composed-stage tile ownership or
   routing is wrong (codex). (c) DYNAMIC variant (GLM): poison-sentinel
   INSIDE the acoustic loop, per substep per dispatcher name — catches
   stale-pad and mistimed-firing classes static table equality cannot.
G2 **Phase gates**: each 3-D phase (c_sw, C-pressure, d_sw, D-pressure/tail,
   remap, NH) tiled == loop reference, kt ∈ {1,2,3} on CPU. Tolerances
   PER-FIELD (rtol 1e-13 hydro baseline; NH fields get their own bounds —
   sqrt/div amplification, GLM); G3 bounded by step count (step-1 tight,
   step-N growth-bounded).
G3 **Step gate**: full `fv_dynamics_step` kt=2,3 vs kt=1 — bit-target where
   reduction order is unchanged; else the pre-registered few-ulp envelope
   (4e-15 rel, the GSPMD sharding floor already accepted on this lane).
   DCMIP16 10-step realism check (finite, physical, matches six-face run).
G4 **Scaling receipt**: C96/C192 km=10 on 6 vs 12/24 GPUs (3 nodes × 2 A100
   exists; >6 GPU needs a larger allocation — receipts gated on HW access,
   the tiled lane's own "future-HW" posture). Non-vacuity: output sharding
   asserted post-timing; excess-ratio re-measured with the port (expect ~1).

## Increments (each: build → sbatch-validate → codex+GLM → land)
M1 Build-time table split + tile exchange for A-grid scalar (1 stagger),
   G1 for it. The riskiest object first — if the split is wrong, it dies
   here at minimum cost.                                   [core risk]
M1b WALKING SKELETON (pulled forward, GLM): ONE field end-to-end through
   the real acoustic-substep harness (M6 shell + M4 pad/vmap + one exchange
   name) at kt=2 — de-risks M4 (the schedule-killer: per-tile branch
   heterogeneity, loop-carried state, pad management) while M1/M2 proceed.
M2 Remaining staggers (B/C/D vector, k2e rings, corner modes), full G1.
M3 kt=1 bridge of the full step (G0) — exchange swap behind the dispatch
   names, no kernel edits.
M4 Tile the per-face kernel loops (the phases already vmap over faces;
   re-express as vmap over (face,tile) on pre-padded tile windows), G2+G3
   hydrostatic.  ** DESIGN v2 — see "M4 design, corrected" below. The v1
   claim ("certified kernels run verbatim on padded tile windows, only the
   window/metrics/bounds change") was reviewed UNSOUND by codex, with GLM
   independently naming the same crux; v2 is built on measurements, not on
   that claim. **
M5 NH arm (update_dz_d halos inside acoustic loop + per-column Riemann), G2/G3 NH.
M6 Driver/factory wiring (`component_factory` layout policy extends from
   {2,3,6} to 6·kt²), restart reshard reuse, G4 receipts.
M7 Tracer arm (nq>0) with the pmax collective; refused until built.

## Cost model / honest risks
- Compile time: the n_split-unrolled step already compiles ~3 h at C192+
  (jobs 9529170/9529655). Tiling multiplies nothing (SPMD one program) but
  does not fix it either; substep-in-scan is a SEPARATE lever, out of scope.
- The certified kernels are reused VERBATIM on padded tile windows — the port
  touches exchange + orchestration only. Any kernel edit is scope creep and a
  review flag.
- Ginsburg fabric: np>6 cross-node ppermute historically anti-scales here
  (tiled lane's own note) — G4 receipts may need Derecho/NVLink; feasibility
  gates (G0-G3) are all CPU/small-GPU and land regardless.
- 8 corner tiles × orientation cases are the bug farm; G1's poison test is
  the instrument that catches them.

## Out of scope (named, per no-extrapolation)
fp32/mixed on the tiled lane; consv_te fixer under tiling; substep-in-scan;
moist/zvir≠0; k_split>1; AD gates; MPI lane.

## M4 design, corrected (v2) — built on measurements, not on the v1 claim

WHY v2 EXISTS. The v1 claim was "the compute surface is local stencils, so
the certified kernels run VERBATIM on padded tile windows; only the window,
the sliced metrics and the bounds object change". codex ruled it UNSOUND and
GLM independently named the same crux. Three findings, each now MEASURED
rather than argued:

(a) REACH IS PARAMETER-DEPENDENT (codex CRITICAL). `del6_vt_flux` initialises
    through `is-1-nord` and runs an ordered `nord` recurrence;
    `d_sw5_duo` chains `divg_d -> uc/vc -> divg_d` with `nord`-controlled
    widening; `fv_tp_2d` runs sequential PPM passes, not independent point
    stencils. "Fixed-reach stencil" is false for exactly the operators that
    matter.
(b) `DuoTileComm.depth == 7` CERTIFIES THE EXCHANGE TABLES, NOT THE KERNEL
    CHAINS (codex CRITICAL). It is the all-family census of table sources; it
    analyses no kernel-produced intermediate. Its sufficiency for the `nord`
    chain is accidental, not proven.
(c) METRICS ARE NOT UNIFORMLY HALO-VALID (codex MAJOR): some gridstruct
    fields carry the Fortran `big_number` sentinel outside narrow regions, so
    a blind rectangular tile slice can hand a kernel a sentinel the
    whole-face lane never reads.

MEASUREMENT 1 — PAD DEPTH (`tiled_m4_reach_census.py --mode paddepth`, job
9610203). Poison the outermost `d` halo layers after EVERY firing (that is
exactly a tile pad of depth `ng-d` refreshed at the certified points) and
compare the final compute window to the unpoisoned control. Result, hydro
km=5, C24, 1 step: poisoning ONE layer (pad depth 2) turns the ENTIRE
interior non-finite -- 17280/17280 cells of delp and pt, 18000/18000 of u
and v (w clean: hydrostatic). Same at pad depth 1.
  => REQUIREMENT: each tile carries the FULL `ng` = 3-layer pad on all four
  sides (corners included), refreshed at EVERY certified firing point.
  Shallower is wrong, and this is a per-DECK number: a deck that changes
  `nord`/`hord` must re-run this census (that is finding (a), answered
  empirically instead of by reading widths).

MEASUREMENT 2 — METRIC SLICING RULES (`--mode metrics`, job 9610177; exact
sentinel match, identical at n=24 and n=48, so the invalid set is a FIXED
RING, not a function of face size):
  * 30 metrics fully valid            -> rectangular tile slice is safe;
  *  9 sentinel in the outer ring only -> safe for an INTERIOR tile; an
      EDGE tile must take the face-edge treatment (`cosa`, `sina`, `rsina`,
      `cosa_u/v`, `sina_u/v`, `rsin_u/v`, `divg_u/v`, `del6_u/v`);
  *  4 cube-wide scalars (`da_max`, `da_min`, `da_max_c`, `da_min_c`)
      -> replicate, never slice;
  *  4 one-dimensional edge factors (`edge_e/n/s/w`, length n+1)
      -> GLOBAL-INDEX-AWARE slice; the one genuinely new rule.
  NONE is invalid inside the compute window. (An earlier run of this probe
  reported `area`/`area_c` as interior-invalid; that was the PROBE -- a
  magnitude test flags Earth-radius cell areas ~1e11 m^2. Fixed to exact
  sentinel equality.)

WHAT v2 THEREFORE COMMITS TO, and how it differs from v1:
1. Per-tile PADDED windows of `nl + 2*ng` (v1 assumed the M3 blocked layout,
   which partitions the padded face and gives interior tiles NO halo).
2. The exchange schedule gains INTERNAL-SEAM refresh at every existing
   firing point -- new work the whole-face lane never needed, because a
   face got read-after-write consistency inside itself for free (GLM's
   crux). Firing SET, ORDER and TIMING stay exactly as certified (scope P3):
   seams are refreshed AT the existing firings, never at new ones.
3. Metrics are tiled BY CLASS per the table above, not by one rule.
4. Kernels are still reused verbatim -- that part of v1 survives -- but the
   claim now rests on the measured pad depth, not on an unproven
   "fixed-reach" premise.

OPEN, NOT ASSUMED:
  * NH pad depth: the hydro number is measured; the NH arm of the same job
    must confirm it before M5 relies on it.
  * TRACER REACH (GLM): `fv_tp_2d` goes semi-Lagrangian when Courant > 1, so
    its reach is `ceil(C) + scheme order` and can exceed `ng`. The frozen
    deck is nq=1 and M7 is unbuilt, but the tracer arm needs its OWN reach
    census before it is tiled -- it is NOT covered by MEASUREMENT 1.
  * FACE ORIENTATION (cross-lane report, 2026-09-02): faces 4/5 are
    i-mirrored vs the reference and face 6 is unverified. The tile split is
    UNAFFECTED -- it decodes sources from our own certified tables, and the
    single place that consults face adjacency (`neighbor_tiles`) is a
    REFUSAL guard, whose adjacency set is invariant under an i-mirror. The
    tile-vs-ring and tile-vs-certified gates compare our lane against
    itself, so orientation cancels on both sides. The oracle-parity gate,
    which does map faces, is where that report bites; it is outside this
    scope doc.

GATES FOR M4 (both reviewers' own proposals, adopted):
  G4a FALSIFIER FIRST (GLM): single-device emulation -- pack the certified
      post-exchange state into (face,tile) padded windows, run the
      vmap(face,tile) path with class-tiled metrics and per-tile bounds,
      unpack, and compare BITWISE against the whole-face step. No
      distributed run. If this fails, everything downstream was sand.
  G4b SEAM DETECTOR (codex): the named most-likely wrong-but-finite failure
      is a STALE `divg_d` at an internal tile seam during the second
      `d_sw5` damping pass. Poison intermediate pads one layer at a time
      and report the first differing seam row after each recurrence pass.
  G4c The M3 gates re-run unchanged (miss-set parity, exchange purity,
      converter round-trip, planted-collision fuzz) -- correctness must
      track scale at kt in {1,2,3}, not only at kt=1.
  G4d SCALING RECEIPT: per-device state must FALL with device count (the
      M3 census, job 9610112, measures it as 1.00x today -- the
      1/kt^2 efficiency ceiling M4 exists to remove), and the model
      exchange must approach the BARE collective cost (jobs 9610238-42:
      the bridge runs 4.9x/10.4x the bare permute at kt=2/kt=3 while the
      bare collective is flat ~0.5 ms and node-spread-insensitive).

## M4 design v3 — the measured numbers, and what v2 got wrong

v2 was reviewed UNSOUND by codex (log codex_m4_v2_9610257.log) and refuted on
its central argument by GLM. Both were right; v3 records the corrections and
the numbers that replace the arguments.

WHAT v2 GOT WRONG
1. SUFFICIENCY ARGUMENT (GLM, fatal). v2 argued: the whole-face lane carries
   exactly ng=3 halo layers and is certified bit-exact, therefore no
   inter-firing kernel chain can reach further than 3. FALSE -- that bound
   covers only CROSS-FACE reads. Inside a face every intra-face read is
   served by full-face data, so the face lane never exposes its intra-face
   reach. A tile seam IS an intra-face boundary, so a tile can need MORE
   than a face ever did. MEASURED: it does (see below).
2. "NO NEW FIRINGS" (codex, fatal). v2 committed to refreshing internal
   seams only at the EXISTING firing points. codex found the counter-example:
   `uc`, `vc` and `divg_d` are refreshed once after the pressure gradient
   (fv3_dsw_phase_3d.py:576) and NO firing occurs before the later
   `d_sw5_duo` chain (fv3_dsw_tail_3d.py:305) -- so that internal seam is
   stale through the composed damping chain. v3 therefore REQUIRES at least
   one NEW firing, before the second damping pass. Scope P3's contract is
   amended from "no new firings" to "no REMOVED or REORDERED firings; new
   internal-seam refreshes permitted only where a measurement shows a stale
   seam, and each one recorded here with that measurement."
3. METRIC LIST (codex, correction). v2's "9 ring-only" parenthesis actually
   named 13 fields: `divg_u/v` and `del6_u/v` are FULLY VALID per the log
   (m4_metrics_rerun_9610177.log:28). The ring-only set is the 9 trig/edge
   factors: cosa, sina, rsina, cosa_u/v, sina_u/v, rsin_u/v.
4. METRIC COVERAGE (codex, gap). The census enumerates FLOAT arrays only, so
   it never examined `cell_ok`/`node_ok` and the tile-dependent corner and
   bounded-domain FLAGS that kernels consume (fv3_native_gridstruct.py:934).
   Those are static per-face booleans whose meaning is positional -- a
   sub-face tile is NOT a face, so a tile that does not touch a cube corner
   must not inherit that face's corner flags. OPEN: their tiling rules are
   unwritten, and no code lands until they are.

THE MEASURED PAD DEPTH (the number v2 argued for and v3 measures)
Instrument: perturb ONE interior line by 1e-8 (NOT NaN -- a NaN footprint
mixes structural dependence with value-triggered blow-ups, which is exactly
why the first attempt reported 12/24/6 cells at three face sizes), run ONE
substep with the exchanges OFF so only kernel chains propagate, and record
how far outputs move. Jobs 9610299 (hydro) and 9610372 (NH):

  face   hydro (delp/pt/u/v/w)      NH (delp/pt/u/v/w)
  C24    3 / 3 / 5 / 5 / clean      3 / 3 / 5 / 5 / 3
  C48    3 / 4 / 5 / 5 / clean      3 / 3 / 5 / 5 / 3
  C96    4 / 4 / 5 / 5 / clean      --

  => COMPOSED INTERIOR REACH = 5 CELLS, SIZE-INDEPENDENT, BOTH DECKS.
  A tile pad of 5 is required; 3 (the face lane's own halo) is NOT enough,
  which is finding 1 above, quantified. The number is a SAFE UPPER BOUND for
  any single inter-firing segment (the measurement composes the whole
  substep with no refresh at all). Instrument checks that passed: `w` is
  CLEAN on the hydrostatic deck (never prognosed) and ACTIVE at reach 3 on
  NH (prognosed) -- the probe tracks real dependence, not smeared poison.
  NOT USABLE: the exchanges-ON control (job 9610372) reports 27 at C48
  because it perturbs the same line on ALL SIX faces, so the exchange
  carries it between faces and back; it is a confound, not a reach.

  STATIC COROLLARY (asked for by the coordinator, run as a grep): there is
  NO face-wide horizontal reduction anywhere in the acoustic substep chain
  -- the only min/max are host-side sanity checks that raise on a tracer,
  the only scans are k-recurrences, and the shallow-water core, pressure
  gradient and transport core contain no axis reductions at all. The one
  cube-wide reduction in the model (tracer Courant max) is outside the
  substep. So the local reading is supported by inspection as well as by
  the size-independence.

v3 COMMITMENTS
  * per-tile padded windows of nl + 2*5 (not 2*ng);
  * internal-seam refresh at every existing firing, PLUS at least one new
    firing before the second damping pass (finding 2), each new firing
    justified by a measurement recorded here;
  * metrics tiled by class, with the corrected membership (finding 3) and
    the non-float flags as a BLOCKING open item (finding 4);
  * kernels still reused verbatim.

STILL OPEN, BLOCKING CODE
  * non-float gridstruct flags: tiling rules unwritten (finding 4);
  * tracer arm: `fv_tp_2d` goes semi-Lagrangian above Courant 1, so its
    reach is not bounded by this census -- it needs its own before M7;
  * per-deck re-census on ANY reach-affecting parameter, not just nord/hord;
  * provenance: every number above comes from an UNCOMMITTED tree, which
    codex flagged; the receipts are reproducible only once this lands.

## M4 design v4 — WINDOWS (2026-09-04): verbatim kernels, pad 8, one refresh per substep

v3's open items (pad certification, edge/corner flag tiling, new firings)
are closed by measurement and by a layout choice, not by kernel edits.

MEASURED FIRST
* Static taint bound (`tiled_m4_static_reach.py`, job 9631177, km=10 so
  levels ≠ faces): intra-face reach of ONE acoustic substep with no halo
  refresh = **8 cells** at C48 and C96, hydrostatic AND NH, isotropic
  (point seed → 8×8 box, job 9631170), zero taint on the other five faces
  at every equation.  The perturbation ladder's 5..7 sits under it.  The
  earlier 15/27/48 were probe confounds (per-face outputs measured along
  the seam line; a full-line seed carried into every face by the
  entry/barrier halo tables, which `exchange=False` does not remove).
* Gate (`tiled_m4_window_gate.py`, jobs 9631664/9631684): windows kt=2
  pad=8 at C48 km=10 hydro n_split=3 == the flat batched arm **BITWISE on
  every state/press field**, under default lowering and under the FMA-off
  pin.  The flat batched and loop arms differ from EACH OTHER at 1e-14
  (lowering), so the batched arm is the window arm's reference and the
  loop arm the attribution arm.

THE LAYOUT (`packages/core/legoesm/grids/fv3_duo_windows.py`)
* Window = `W = nl + 2*pad` slab of the face's padded array; the kernel is
  told it is a face of size `W - 2ng` with the normal `ng` halo.  Interior
  seams: the window boundary lies `pad` cells inside the neighbour tile.
  Real face edges: the window is placed FLUSH with the padded array (its
  outer `ng` cells are the true cross-face ring) — edge/corner windows are
  shifted inward, never centred.  kt=1 is the flat layout byte for byte.
* Corner flags stay True everywhere.  The kernels' cube-edge/corner
  treatments fire at every window boundary; at a seam (or a fake corner
  where a real edge meets a seam) they corrupt only cells within the pad
  zone, which the tile never publishes.  This is what makes the v3
  "non-float flag tiling rules" item MOOT: no flag is tiled.
* Ownership: each tile publishes its own `nl` cells; a shared node belongs
  to the east/north tile; the rings belong to the edge tiles.  Every flat
  cell has exactly one owner (`tests/grids/test_fv3_duo_windows.py`).
* Refresh: seam pads rebuilt from the owners ONCE per substep, at entry
  (`DuoWindowComm.refresh`, hooked in `acoustic_substep_3d`), for state,
  the NH carry and the flux capacitors.  No new firing inside the substep
  (v3's finding 2 was about pad-3 seams; with pad ≥ reach it does not
  arise).
* Cross-face firings: the ~23 existing firings run the certified flat impl
  on the owners' scatter and hand each window exactly the impl's
  WRITE-SET (censused on three random trials at trace time).  The two
  edge-blend barriers route through the same bundle (`_barrier_comm`).
* Cost: halo area `(nl+2pad)^2/nl^2` = 2.8× at nl=24, 1.8× at 48, 1.4× at
  96 — the price of zero kernel edits; per-firing communication is the
  ring only.

THE PAD (measured 2026-09-04, jobs 9631742-45, 9631824-26, 9631935)
* Bitwise gate at TRUE interior tiles (kt=3 at C48, kt=4 at C96; kt=2 is
  not a test — its edge windows sit flush with the face and carry 13
  cells on the seam side):
    hydro  pad 6 DIFFERS (C96 kt=4) | pad 8 BITWISE (C48 kt=3, C96 kt=4)
    NH     pad 8, 9 DIFFER (C48 kt=3, C96 kt=4; diffs on the seam-adjacent
           owned row of u/v/gz/pkc within one substep) | pad 10 BITWISE
  Negative controls (pad 5 at kt=2: 1e-6 rel on winds) prove the gate can
  fail.  Hydro at n_split=6 pad 9 bitwise.
* Static certification on the WINDOW program (`tiled_m4_static_reach.py
  --window KT:PAD[:BAND]`): seed = the outer ``band`` (= ng) cells on all
  four sides of EVERY window (non-owned cells only) for the state and the
  NH carry, refresh disabled, all firings on, taint through one substep,
  read on an interior tile's owned cells: pad 10 → overshoot 1, pad 11 →
  0 (hydro and NH, C48 kt=3 and C96 kt=4).  Widening the band by one
  raises the requirement by exactly one (band 4 → 12), i.e. the band
  models the corruption depth one-for-one.  **Certified pad = 11**, the
  empirical minimum being 10 (NH) / 7-8 (hydro).
* The pad is an explicit layout argument, never a default.  Halo area
  at pad 11: (nl+22)²/nl² = 3.7× (nl 24), 2.1× (48), 1.5× (96).

STATUS: single-device "windows as views" form (M4a) LANDED — the
correctness gate and the layout the SPMD arm shards.  codex round 2:
items 1-5 CLOSED; item 6 (seed completeness) answered by the full seed
model above.  GLM: agrees with pad 11; requires the sharded arm to be
re-certified (reach across pack/unpack).  M4b = the same layout sharded
one window per device with the M1/M2 split machinery re-targeted at
window origins plus a seam-refresh collective.

## M4b-A (2026-09-04): the window layout SHARDED — one window per device

`packages/core/legoesm/grids/fv3_duo_window_spmd.py`.  Window stacks placed
`P(('face','tile_i','tile_j'))` on the M1-M3 tile mesh (window order ==
device order); kernels = the whole-face kernels vmapped over the window
axis (GSPMD, zero comm).  Each firing = ONE shard_map: cut this device's
BLOCK out of its window (padded partition, `kt | m_a`, flush edge windows
→ dynamic offset), run the certified M1-M3 exchange BODY
(`fv3_duo_spmd.tiled_split_body` / `tiled_vector_body`, factored out of the
M3 runtime unchanged), write the block back, refresh every seam pad with
two ppermute rounds (2·pad rows each way, node axes skip the shared row,
corners via the second round).  Barriers: flat fallback (O(state)),
OPEN.  Requires `2·pad ≤ nl`.

RECEIPTS (24/54 virtual CPU devices, FMA off)
* per-firing unit tests: refresh fills every pad from NaN (5 extents),
  scalar A/B, vector D/C bitwise vs the certified impl; planted sabotage
  (identity pad exchange, identity body) FAILS them (job 9632167, 6 passed).
* full acoustic loop, n_split=3: window-SPMD == the flat step
  GSPMD-sharded `P('face')` on 6 devices BITWISE on every state/press
  field, block coherence 0 — hydro C48 kt=2 (9632146), NH C48 kt=2
  (9632147), hydro C96 kt=3 on 54 devices (9632148).  Both partitioned
  programs differ from the UNSHARDED flat step by the same 1e-13 rel on
  the same cells (the partitioned-kernel lowering class, identical
  counts, job 9632138); the unsharded flat step == the single-device
  window arm bitwise (M4a).  GLM's same-shape "no-comm control" (flat
  gather exchange, results pinned to the window sharding) is NOT
  codegen-invariant (fusion around the exchange sites) and differs at the
  same ulp level; retired as a discriminator.
* codex (9632151): items 1-4 CLEAN (offsets/clamps/corners, normalisation,
  block placement, M3 refactor byte-identical), 2 MINOR fixed (owner
  docstring; sabotage non-vacuity).  GLM: exchange-exactness established
  for the tested geometries; open = other kt/n_split/FMA-on, and the
  barriers.

NEXT (M4b-B): tiled barriers (a "blend" op kind in the split runtime on
padded-embedded compute arrays), write-set-restricted pad refresh, then
multi-process ranks on the CPU+mpi venue.

## M4b-B (2026-09-04): the barriers tiled — every exchange is now O(halo)

A duo barrier blend ``out[dst] = 0.5*(in[dst] + s*in[src])`` (s = ±1) is
the two-source stencil row ``0.5*in[dst] + (0.5*s)*in[src]`` EXACTLY for
normal doubles (halving is exact, rounding commutes with a power-of-two
scaling; the cancellation corner gives +0 both ways; FMA off) —
`fv3_duo_spmd.blend_as_stencil`, pinned by
`test_barrier_blend_as_stencil_is_bitwise` on the real tables and by the
24-device parity tests.  Barrier 1 (`avg_c`, C-flux pair), barrier 2
(`avg_b`, B pair) and the allflux barrier (`avg_c` batched over the
selected slots) become splits on PADDED twins of their compute layouts
(`build_tiled_barrier_split`); the window arm embeds the compute operands
(`_normalize`, explicit axes) and fires them through the same
block/body/pad path.  `DuoWindowSpmdComm.barrier_mode == "tiled"`; the
flat fallback is gone.
Precondition (codex MAJOR, accepted as a documented contract): the
identity excludes subnormal-producing halving and overflow; barrier
operands are O(1e-1..1e5) winds/fluxes, and a traced program cannot
refuse on values.
RECEIPTS: unit 11/11 incl. planted sabotage (jobs 9632468/9632498);
full-step window-SPMD == GSPMD-face-sharded flat BITWISE, block
coherence 0: hydro+NH C48 kt=2 (24 devices, 9632470/9632471), hydro+NH
C96 kt=3 (54 devices, 9632472/9632473).  codex round: 2 CLEAN, 1 MAJOR
(domain contract, documented), 1 MAJOR fixed (slot count could alias a
horizontal extent: barriers now pass explicit axes), 1 MINOR fixed
(sabotage arm).  GLM: identity holds for the stated operands; argument,
not sampling, closes the cancellation corner (recorded in the docstring).

## M6 (2026-09-05): the model steps on windows

`FV3DuoDynamicsModel(grid, cfg, step_windows=(kt, pad), step_spmd_mesh=mesh)`
builds a fresh stepper context, attaches the window comm (SPMD on the
(6,kt,kt) mesh, or the single-device window bundle without a mesh), forces
the batched arm and defaults the output sharding to the window sharding.
The IC is built on faces (p_var, NH carry from the flat hs6) and
`to_windows()`-ed; `to_flat()` scatters each window's owned cells on the
host.  The full outer step (acoustic loop, tracer transport with its
sub-cycle count from a GSPMD all-reduced Courant max, vertical remap) runs
with the leading batch axis `batch_size(ctx)` (fv3_dynamics.py,
fv3_tracer2d.py incl. the batched tracer arm); `alloc_flux_capacitors`
takes `nb`.  A km+1 inside the horizontal extent range is refused at
construction (the axis classifier would be ambiguous — toy grids only).
RECEIPTS (`tiled_m6_model_gate.py`, every leaf of the bundle — state,
press, tracers, omga, NH carry — window arm vs the same model on faces
GSPMD-sharded P('face'), FMA off, 2 outer steps): hydro C48 kt=2 12/12
leaves BITWISE (9634735); NH C48 kt=2 22/22 BITWISE (9634875); hydro C96
kt=3 on 54 devices BITWISE (9634364).  No window output comes back
replicated.  codex (9634369): 3 MAJORs fixed (step_out_shardings recorded
after the window default; km/horizontal ambiguity refused at construction;
gate vacuity — every leaf scored, steps>=1, replication detected via
`sharding.is_fully_replicated`), 2 CLEAN (the nb sweep; the Courant
all-reduce replicates nsplt).  GLM: the first real-rank measurement is the
per-rank nsplt read and its latency (barrier cost of the host guard).

## M6 scaling finding (2026-09-05): the C-grid pressure phase ran its LOOP arm inside the batched window step

The first real-rank ladder (job 9648613) scaled 1.42x for 9x ranks at C96.
A perfetto profile put ~60 % of the 54-rank step in four ~117 ms all-reduces
per step, and an HLO census (`scripts/cluster/fv3_native/tiled_m6_hlo_census.sbatch`)
traced 276 whole-window `collective-permute`s (`{i->0}` pairs, `f64[1,49,49,10]`)
and 3 tuple all-reduces per step to `fv3_cgrid_phase_3d.py:618/624/625` —
the per-face loop arm (`csw_outs["delpc"][t]`, `["ptc"][t]`, `["uc"][t]`,
`["vc"][t]`) and its `stack_faces`.  Cause: `cgrid_pressure_phase_3d` and
`cgrid_nh_pressure_phase_3d` accept `batched` but `acoustic_substep_3d` never
passed it, so every batched step ran the loop arm; on a window-sharded state
GSPMD gathers each window to one device per iteration and all-reduces the
restacked tuple.  Fix: pass `batched=batched` at both call sites (every other
`batched:`-accepting phase already received it — grep of `batched=batched`
across `fv3_acoustic_3d/fv3_dynamics/fv3_tracer2d/fv3_dsw_tail_3d`).

After the fix the census shows one all-reduce per step (`f64[10]`, the Courant
max) and no whole-window permutes.  Gates re-run bitwise: in-process M6 gate
hydro + NH C48 kt=2 pad=11 vs the GSPMD face-sharded flat step (jobs
9648696/9648697); real ranks vs 6-rank flat references regenerated with the
fixed code (`ladder_ref_c{48,96,192}_v2.npz`).

| row (CPU, XLA mpi collectives, --exclusive, p50 of 20 steps, cross-rank max) | 6 ranks flat | window ranks | s/step | speedup |
|---|---|---|---|---|
| C48 km=10 kt=2 pad=11  | 0.376 s | 24 (3 nodes)  | 0.184 s | 2.04x on 4x  |
| C96 km=10 kt=3 pad=11  | 1.124 s | 54 (7 nodes)  | 0.258 s | 4.36x on 9x (was 1.42x) |
| C192 km=10 kt=6 pad=11 | 3.752 s | 216 (27 nodes) | pending job 9648703 | |

Remaining gap candidates (unmeasured): pad refresh of 2·pad rows per firing,
~1306 halo-sized collectives per step, the per-step host readback of the
Courant max; a diagnostic arm (`--diag-no-pad-refresh`) put the refresh at
~5 % BEFORE the fix — re-measure after it.

Lowering note after the fix (job 9649596 vs 9649094): the substep gate's
window arm is still BITWISE against the GSPMD face-sharded flat step (the
adopted reference), but no longer against the UNSHARDED single-device flat
batched step (max rel 1e-11..1e-6 in u/v, diffuse) — before the fix both
sides ran the per-face loop for the C-pressure phase, whose per-face ops
lower identically sharded or not; the vmapped C-pressure arm fuses
differently unsharded vs sharded.  The certified single-device production
trace (batched=False) is untouched by the fix.  Memory: per-rank RSS is
3.9 GB at C48/24 ranks (2.7 GB of it appears at the first step), 5.2 GB at
C96/54, 24.6 GB at C192/216 (OOM at 8 ranks/node, job 9648703); compiled
constants are only 55/162 MiB at C48/C96, so the growth is not literals —
the gate now prints per-stage RSS to name it.

## 2026-09-05/06 ladder incidents: fabric, initial state, thread pool

* **Cluster fabric, not code.** Three 54-rank rows ran 150x slow (36-46 s/step,
  compile 3000 s).  The bare neighbour-ppermute bench on the same nodes
  measured 38-51 ms (gloo AND mpi; 0.11 ms on a healthy set), and later
  allocations on unrelated nodes measured 48-73 ms — time-varying and
  cluster-wide, coincident with the 99 %-full Lustre (same InfiniBand
  fabric) stalling directory listings.  The MPI launcher now runs a bare
  ppermute PREFLIGHT per allocation and refuses the timing row above
  1 ms (`PREFLIGHT_MAX_US`), printing the slowest host.
* **CPU binding.** task/cgroup binds nothing here (every rank saw all 32
  cores, so XLA:CPU's Eigen pool was 32 threads per rank); the launcher
  pins each rank to its `--cpus-per-task` cores (`bind_rank.sh`).  A/B in one
  allocation on a healthy fabric: bound 0.256 s/step, unbound 0.258 — neutral
  for the step, kept as the correct configuration (BIND=0 to revert).
* **Initial state on many ranks.** `to_windows` built the windows with eager
  per-window device slices (nb x leaves compiled programs) and
  `jax.device_put(numpy, multi-host sharding)`, whose dispatch path asserts the
  host array equal across ALL processes by gathering the whole array per leaf:
  216 ranks C192 = 27 GB and 5 h per rank (job 9654155) vs 3 GB / 6 min in one
  process (tiled_m6_ic_memory_probe.py).  Now: host numpy gather +
  `make_array_from_callback` (addressable shards only).  Pending: a multi-rank
  re-measurement once the fabric is healthy.
* **kt=6 (C192, 216 ranks) tracer mismatch.** Step 1 bitwise on all 12
  leaves; step 2 bitwise on 11, the tracer `q` differs on 94 % of cells (rel
  4.3).  In-process C192 kt=6 gate (216 virtual devices) queued to separate an
  exchange defect from the multi-process run on a degraded fabric.
* Persistent compile cache (`~/.cache/legoesm/jit_cache` on Lustre) is now
  disabled in the MPI launcher (`LEGOESM_JIT_CACHE_DIR=""`).
* Profile on a healthy fabric (54 ranks C96): compute ~0.12 s/step (= flat/9,
  ideal), ppermute thunks 2702/step at ~24 us + gaps ~0.12 s -> count-bound.
  M8 (fewer messages) claim sent to codex + GLM before code.

**RETRACTION (2026-09-06): every C192 number above ran an UNSTABLE deck.**
The ladder reused the C48 timestep (dt = 900 s, n_split = 3) at C192; the
6-rank flat reference reports nsplt = 0 at step 2 (floor(1 + cmax) < 1 only
for a NaN Courant), so the state is NaN from step 2 on.  The gate compared
NaN to NaN as "bitwise", so the C192 kt=6 "tracer mismatch" was the window
arm's finite-but-huge Courant (nsplt 7) against the reference's NaN, and the
C192 timings (6-rank 3.752 s/step, 216-rank) are timings of NaN arithmetic.
The gate now REFUSES any non-finite value in either arm and any nsplt < 1 on
either side.  The C192 (and larger) rows need a dt scaled with resolution --
a deck decision, asked, not taken.

## M8 go/no-go (2026-09-06, job 9664237): ROUNDS, not ops

One allocation (g[257-263]), healthy fabric (bare 93-150 us), C96 kt=3
pad=11, 54 ranks, p50 of 20 steps:

| arm | s/step | gate |
|---|---|---|
| baseline | 0.256 | bitwise |
| M8-A packed pad refresh (1305 -> 926 ops) | 0.261 | bitwise |
| diagnostic: no pad refresh | 0.189 | differs (expected) |

Pre-registered count-bound prediction for the diagnostic arm was 0.167;
packing (-29 % ops, same rounds, same bytes) saved nothing.  Corrected
reading: the overhead is proportional to the number of SEQUENTIAL exchange
rounds (~3 per firing x ~199 firings; removing the two refresh rounds saved
67 ms = 199 x 2 x ~0.17 ms), not to op count or bytes.  M8-A stays off.
Round-cutting levers (claim sent to codex + GLM): R1 one-round refresh with
diagonal corner sends (3 -> 2 rounds), R2 batch per-level firings across k
(fewer firings), R3 fold the refresh into the body's exchange round (1
round; soundness at face seams to be reviewed).

## M8-B/C results (2026-09-06), C96 kt=3, 54 ranks, healthy fabric, all bitwise

| change | s/step |
|---|---|
| baseline (per-level tracer scans, full 2·pad refresh with padded copies) | 0.256 |
| + tracer sub-cycle batched over levels (runtime firings 244 -> ~100) | 0.253 |
| + D-grid barriers batched over levels, in-place slab placement | 0.237 |
| + per-firing refresh restricted to face-edge bands (depth ng+1, flag) | 0.226 |
| diagnostic floor: no per-firing refresh at all (incorrect) | 0.189 |

Neither collective count (packing: 0.261) nor firing count (tracer
batching: -3 ms) nor bytes alone (bands: -11 ms for -85 % bytes) explains
the refresh's cost; the padded-copy placement did (-19 ms).  GLM and
codex both read the residual as XLA:CPU per-thunk dispatch of the many
small slice/update/ppermute ops per array per round; the discriminating
measurement (op-count sweep at fixed bytes, or the profiler's thunk gaps)
and the structural fix (stack the firing's arrays so the refresh is one
chain of ops per firing, and pack the two bands per direction: 8 -> 4
ppermutes per array) are next.  Reviews: codex PASS on the semantics of
all four changes (independent enumeration of the placement over 2464
geometries; ng+1 tight because the barriers write the edge node), HOLD
items fixed in fbc519e36; GLM: corners delivered by round composition,
ng+1 minimal (depth 3 differs).

Addendum (2026-09-06): packing both face-edge bands of every array into one
message per direction: 0.226 -> **0.211 s/step** (bitwise; e1b89da24).
Packing the substep-entry FULL refresh as well aborts: MPI_ERR_TRUNCATE on
54 ranks and, in one process, XLA:CPU's collective-permute rendezvous
"id can't be larger than the number of participating threads" -- two
concurrent collective-permutes sharing a rendezvous key.  Open item; the
gate refuses that combination.  Cumulative this session at C96/54 ranks:
0.98 (loop-arm C-pressure) -> 0.256 -> 0.211 s/step, 6-rank flat 1.124
(5.3x on 9x ranks); compute floor 0.12.

## 2026-09-07 — NOISE FLOOR, and a correction to the M8 ladder

The same configuration, timed three times inside ONE allocation (job
9675236, C96 kt=3, 54 ranks): 0.218 / 0.220 / 0.219 s/step -- repeatable
to about 1 ms.  But the SAME code measured 0.211 on one node set and
0.222 on another, so the between-allocation spread is ~10 ms.  **Every
rung of the M8 ladder above was measured in a different allocation and
each was 11-19 ms, i.e. at that spread**: the cumulative 0.256 -> ~0.219
is real, the individual rungs are NOT attributable.  The band lever is
the only flag-selectable one, so it is now measured as an interleaved
full/band/band/full A/B inside one allocation
(`tiled_m6_band_ab.sbatch`); the level-batching and placement changes
would need their own runtime switch to be attributed the same way, and
they are kept for their other merits (fewer ops, no whole-window copies,
bitwise) rather than for a claimed time.

Also this session: the flux barrier's level batching had landed on the
LOOP arm only, so the window step still fired it per level (59 -> 32
firings, 748 -> 451 collectives when fixed, dd1662f1f); and the
single-device window bundle called the per-level barrier impls with the
whole level stack, aliasing levels inside their flattening -- found by
codex, fixed with a vmap and a K>1 parity test in dcfa867da (the model
gates could not see it: they run the SPMD arm).

### Band lever, attributed (job 9676161, one allocation, interleaved)

| arm | s/step |
|---|---|
| full refresh | 0.236 |
| band refresh | 0.217 |
| band refresh (repeat) | 0.229 |
| full refresh (repeat) | 0.239 |

Both band arms sit below both full arms; full clusters at 0.236-0.239
(and 0.237 in the earlier in-place row), band at 0.217-0.229 (and
0.218-0.220 in the three-repeat job).  So the band-restricted per-firing
refresh is worth about **14 ms/step (~6 %)**, CONFIRMED in-allocation and
bitwise in every arm.  Note the within-allocation spread here was 12 ms
between the two band arms, larger than the 1 ms of job 9675236 -- treat
~10 ms as the resolution of a single row either way.

## 2026-09-09 — the packed-full-refresh abort is GONE; codex's barrier not merged

codex (authoring in a worktree, GLM + Claude reviewing) diagnosed the
abort as two collective-permutes sharing a rendezvous with no ordering
dependency and fixed it with a `jax.lax.optimization_barrier` between the
two directions.  Both reviewers called the diagnosis unproven, and the
controls settled it:

* channel ids: ALL 322 collective-permutes of the step carry
  `channel_id=1` on this build, so the shared-channel premise is right
  (and GLM's "JAX gives each ppermute its own channel" is wrong here);
* NON-VACUITY: with codex's change reverted, the packed gate passes
  BITWISE -- the abort does not reproduce;
* the true failing configuration was band refresh AND packing together;
  reproduced directly (two steps, both arms) it does not abort either,
  in one process or on 54 real ranks.

The failing executable no longer exists: the flux barrier now fires once
per step instead of once per level, which changed the refresh program.
So the barrier is a fix for a defect we can no longer exhibit -- NOT
merged (an unmeasured finding must not buy a knob).  The gate's refusal
of the flag pair is lifted, and the pair is bitwise on 54 ranks at
0.210 s/step (band alone 0.217-0.229 in the same-allocation A/B), i.e.
packing the entry refresh is worth nothing beyond the band lever.

### Where the residual sits (job 9692615, one allocation, interleaved)

| arm | s/step |
|---|---|
| shipped configuration | 0.220 / 0.218 |
| substep-entry refresh elided (WRONG, decomposition only) | 0.202 / 0.193 |

So the three substep-entry full refreshes cost ~21 ms/step, and the
remaining ~48 ms of the 69 ms between the no-per-firing-refresh floor
(0.189) and compute (0.120) is the PER-FIRING boundary: 32 shard_map
entries/exits per step, each with a sharding constraint, the normalize
embed/restore for compute-extent arrays and the block cut/write-back.
GLM's proposal for it is one shard_map per SUBSTEP with the firings
chained inside (pad once, restore once, repaint the pad bands between
bodies); its traps are the frozen tables' reach beyond the band, any body
that writes pad cells a later firing reads, and two-hop corner pads.  Not
attempted yet: it is a large refactor of the certified path for ~22 % of
the step, and the entry refresh (21 ms) is the cheaper target.

### Fused entry refresh: bitwise, and worth nothing (job 9705121)

Interleaved same-allocation A/B, C96 kt=3, 54 ranks: fused 0.219 / 0.220,
unfused 0.220 / 0.221 s/step.  Merging the three substep-entry firings
into one (32 -> 29 firings, collectives unchanged) is inside the 2 ms
noise.  Kept anyway -- fewer boundaries, one code path, bitwise -- but
**not** claimed as a speedup.

THIRD null result from reducing the NUMBER of things: packing collectives
(M8-A), batching firings over levels, and now fusing the entry firings
all bought ~0.  The only levers that ever paid were the ones that removed
WORK: the in-place pad placement (whole-window copies) and the band
refresh (85 % of the refresh's bytes, ~14 ms).  The 21 ms the entry
refresh costs is therefore its BYTES, not its three boundaries -- the
next attempt on it should shrink what a full refresh copies (e.g. the
same face-edge band argument, which needs the certified pad >= reach
invariant re-derived for a whole substep rather than one firing), not
reorganise the calls.
