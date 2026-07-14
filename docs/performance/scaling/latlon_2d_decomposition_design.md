# Lat-lon 2-D pencil decomposition — design (next deliberate scaling lever)

Status: DESIGN (not implemented). The current MPI lat-lon path is a 1-D
LATITUDE-band decomposition (`packages/core/legoesm/parallel/latlon_mpi.py`):
every rank owns all longitudes, partitions only latitude rows. It strong-
scales poorly once `n_lat_local` is small (≤ a few rows/rank) and cannot
weak-scale both horizontal dimensions. This pins the 2-D extension so a
fresh session implements it cleanly. Motivation: codex P2 (2026-06-13) +
the literature review (Veros/POP both 2-D-decompose) — the strong-scaling
limit at high ranks for atm+ocean lat-lon/tripole.

## What exists (1-D band — reuse, don't rebuild)

- `LatLonBandLayout` (latlon_mpi.py:84): rank/n_ranks, n_lat/lon_global,
  n_lat_local, lat_start/end, south_rank/north_rank, fold.
- N/S halo: `exchange_halo_latlon` (:391), `pad_with_pole_bc_lat_mpi`
  (:826) + multi (:964), AD-safe via `_sendrecv_vjp`. Longitude wrap is
  LOCAL (`jnp.roll`) because each rank owns all lon.
- scatter/gather: `scatter_state_latlon[_cgrid_ocean]`, `gather_*` — lat
  slice only.
- step: `make_latlon_mpi_step` (:1790); grid slice `slice_*_to_band`.
- global reductions: `global_sum_mpi` (allreduce over ALL ranks) — already
  2-D-correct (rank-set-agnostic).

## The 2-D delta (pencils: proc_lat × proc_lon rank grid)

### 1. Layout — `LatLon2DLayout` (new, or extend the band)
Add `lon_start/lon_end`, `n_lon_local`, `west_rank/east_rank`, and the
2-D rank grid `(proc_lat, proc_lon)` with row/col indices. west/east are
ALWAYS defined (longitude is globally periodic — the seam wraps
`proc_col == proc_lon-1` → `proc_col == 0`), unlike N/S which is None at
poles. Keep `LatLonBandLayout` as the `proc_lon == 1` special case for
back-compat (the whole existing path is the 1-row-of-the-grid case).

### 2. E/W halo — NEW primitive `exchange_halo_lon` + `pad_lon_periodic_mpi`
Longitude is PERIODIC, so unlike N/S there are no pole/wall ends — every
rank sends west, receives east and vice-versa, and the wrap is just the
ring topology (no special seam rank). mpi4jax sendrecv routed through
`get_sendrecv_vjp` (AD-safe, same contract as the N/S halo). At
`proc_lon == 1` this degenerates to the existing local `jnp.roll`
(guard on it so the 1-D path is byte-identical). **First increment** —
build + unit-test this in isolation (serial periodic-roll oracle) BEFORE
any layout/step wiring; lowest risk, foundational.

### 3. N/S halo — reuse `exchange_halo_latlon`, lon-subset rows
The halo rows are now `(n_lon_local + 2*halo)` wide, not full lon. The
existing N/S code is lon-width-agnostic (it pads axis 0); just ensure the
lon halo is applied so corner cells are correct (apply N/S then E/W, or
E/W then N/S consistently — corners need both; FV3-style face→edge order
or an explicit corner exchange; for a 5-point stencil corners are not
read so order is free, for 9-point do E/W after N/S so the N/S ghost rows
already carry the lon halo).

### 3-COUPLING FINDING (2026-06-13, from an isolation attempt): interior pad is NOT separable
Tried to ship an "interior-only" 2-D pad (N/S interior sendrecv + E/W
ring, pole rows guarded out) — it DEADLOCKED (job 8476623 timed out)
and the reason is structural, not a test bug: **the N/S exchange is
COLLECTIVE over the entire lat line.**  An interior rank's sendrecv to
its lat neighbours has no match if those neighbours are pole rows that
took the guard and posted nothing.  And an artificial all-interior lat
RING can't substitute (it would deadlock ``exchange_halo_latlon``,
which — like the pre-fix lon halo — is only safe on a pole-terminated
LINE, not a ring).  ⇒ the interior pad and the pole-fold CANNOT be
separated: increment 3 must implement the WHOLE lat line at once
(interior sendrecv for middle rows + pole-fold for end rows, the latter
via the §4 lat-pencil transpose).  So 3 and 3b/4 are ONE unit; there is
no shippable interior-only intermediate.  (Reverted the partial; incr 1
exchange_halo_lon + incr 2 LatLon2DLayout remain the validated building
blocks.)

### 3b. N/S POLE-FOLD under lon-split — same all-lon problem as the filter
INCREMENT-3 FINDING (2026-06-13): the 2-D pad is NOT a clean "N/S then
E/W".  Interior lat-cuts are simple sendrecv (exchange_halo_latlon's
interior path) and compose fine with the E/W ring.  But the
POLE/FOLD boundary rows are the catch: the atmospheric pole-fold mirrors
across the pole with a 180° LONGITUDE shift (n_lon//2), and the ocean
tripole north-fold is a global-longitude PERMUTATION (perm_T/perm_v) —
BOTH need the FULL longitude axis, which a lon-split rank does not own.
So a pole-row rank cannot pole-fold its lon-SUBSET locally (it would
shift/permute within the subset = wrong).  Same all-lon dependency as
the polar filter (§4).  Implementation path: (i) 2-D pad correct for
INTERIOR lat rows now (build a band-like sub-layout with the 2-D
lat-neighbour ranks → exchange_halo_latlon interior sendrecv + E/W
exchange_halo_lon; corners via N/S-then-E/W ordering), with a LOUD
guard on pole/fold rows; (ii) pole/fold rows via the SAME lat-pencil
transpose §4 uses (gather full-lon for the boundary row, fold, scatter).
Do the interior pad + guard as increment 3a (bounded, testable), the
transpose boundary as 3b/4 (shared with the filter).

### 3c. WALL-pole case SHIPPED — the separable half (2026-06-13)
The §3 "interior and pole-fold are inseparable" finding is specific to
the FOLD/TRIPOLE pole BC (which needs all-lon → transpose). The other
pole BC — a constant WALL — IS separable: a wall ghost row is purely
LOCAL (no lon dependency), so it composes cleanly with the interior
sendrecv and the E/W ring. This is the REGULAR lat-lon BC and the
ocean's closed poles. Shipped as `pad_halo_latlon_2d(..., pole_bc=
"wall")` (`packages/core/legoesm/parallel/latlon_mpi.py`):
  * pole-touching rows fill `south_value`/`north_value` locally;
  * interior lat rows do same-neighbour sendrecv (rank-as-tag — the
    LINE-safe pattern the 1-D band uses; NOT the ring, so no phase-tag
    deadlock); EVERY non-pole rank posts its interior sendrecv (the line
    terminates at the pole rows' local fill — fixes the reverted 3a
    guard-and-skip deadlock);
  * E/W via `exchange_halo_lon` on the lat-padded block → lon ghosts +
    the 4 diagonal corners (N/S-then-E/W order; wall corner = wall
    constant since the wall row is constant in lon).
Fully AD-safe (sendrecv-VJP on both axes; wall is a no-grad constant),
deadlock-free, no transpose. `pole_bc in {"fold","tripole"}` raises
`NotImplementedError` (those still need §3b/§4's transpose — next
increment), so a fold deck can't silently get a wall. Gated np=6 (3×2)
parity vs serial lat-wall+lon-periodic ref + single-proc AD grad +
raise test (`tests/parallel/test_latlon_2d_pad_wall.py`,
`L2D_PAD_WALL_GATE_OK`). This unlocks the 2-D decomposition for the
ocean and regular lat-lon atm NOW; the fold/tripole/polar-filter rows
remain the transpose sub-project below.

### 3d. Lat-pencil transpose is now AD-SAFE (2026-06-13)
`lon_gather_full` (the §3b/§4 transpose primitive) was forward-only
(`mpi4jax.allgather` has no native VJP). Now wrapped in a `jax.custom_vjp`
(`packages/core/legoesm/parallel/latlon_mpi.py`): the gather is a LINEAR
map (each rank's lon block → the full-lon array, REPLICATED across the
row ring), so its exact adjoint is `x̄_s = Σ_r ȳ_r[:, block_s]` =
`allreduce(SUM)` over the row ring (the only AD-safe collective) then
slice the rank's block — no reduce-scatter primitive needed.
`lon_scatter_full` stays a plain slice (native pad-zeros adjoint;
correct in the gather∘fold∘scatter chain because the gather VJP's
allreduce does the ring summation). Verified by the dot-product adjoint
identity `<Gx,y>==<x,Gᵀy>` (global) at np=2/3/6 + grad-finiteness through
a block-mixing 180°-roll loss (`tests/distributed/
test_latlon_transpose_ad_mpi.py`, `L2D_TRANSPOSE_AD_GATE_OK`). ⇒ the
pole-fold and polar-filter built on the transpose (§3b/§4) can now be
reverse-mode differentiable (the prior blocker for AD through the pole
rows under lon-split is removed).

### 4. Polar filter — THE hard part (transpose or local filter)
`grids/polar_filter.py` does `jnp.fft.rfft(field, axis=lon)` per lat row
— needs ALL longitudes. With lon split this breaks. Options, ranked:
  (a) **Lat-pencil transpose for the filter**: all-to-all transpose the
      filtered rows so each rank holds full-lon for a lat subset, FFT,
      transpose back. Correct + general but adds 2 all-to-alls/filter
      call (latency). This is the POP/Veros-class answer.
  (b) **Local real-space polar filter**: replace the spectral filter with
      a local FIR/diffusive zonal filter (finite stencil, E/W-halo-able)
      — no transpose, but a DIFFERENT filter (needs re-validation of the
      polar stability it provides). 
  (c) **Disable polar filter in 2-D** (`use_polar_filter=False`) + rely
      on a smaller dt or hyperdiffusion — only if the deck tolerates it.
  Recommend (a) for fidelity; gate behind the layout so 1-D keeps the
  exact spectral filter. The filter is atm-specific; ocean (no polar
  filter, uses the implicit barotropic) skips this entirely.

### 5. scatter/gather — 2-D block
`scatter_state_latlon_2d`: lat-band × lon-band slice. gather: 2-D
process_allgather or staged (lon-gather then lat-gather). I/O-only.

### 6. Ocean wet-cell load balance (separate concern)
Equal lat×lon blocks give unequal WET-cell counts (land-heavy blocks
idle). Balance by wet cells/edges (codex P2). Orthogonal to correctness;
a partitioner refinement after the 2-D path works.

## Increment plan (each its own gated commit)

1. `exchange_halo_lon` + `pad_lon_periodic_mpi` + serial-roll parity test
   (np>1 ring, AD-safe gradient check). **No live wiring.**
2. `LatLon2DLayout` + `make_latlon_2d_layout` + 2-D scatter/gather +
   roundtrip test (scatter∘gather == id).
3. Wire N/S+E/W into a 2-D step variant; serial-vs-2D parity (dry,
   atm + ocean) at proc=(2,2) on host devices.
   3c DONE: `pad_halo_latlon_2d(pole_bc="wall")` — the full 2-D pad for
   regular lat-lon / closed-pole ocean (no transpose). fold/tripole
   raise pending §4. Gated `L2D_PAD_WALL_GATE_OK` (np=6 3×2).
4. Polar filter transpose (atm only) + filtered-step parity.
5. Ocean wet-cell partitioner + load-balance metric.
6. Bench: 2-D vs 1-D strong scaling at high ranks (the payoff — 1-D
   stalls when n_lat_local→small; 2-D keeps both dims large).

## Risks / gotchas

- Corner cells need BOTH halos — pick and document the N/S↔E/W order.
- Tripole fold (north): the fold permutation is a GLOBAL-longitude
  operation (perm over all lon) — under lon-split it needs the same
  transpose as the polar filter, or restrict the fold rank-row to
  proc_lon==1 for the north row. Tripole 2-D is a sub-project.
- mpi4jax deadlock: keep the E/W and N/S exchanges on a UNIFORM schedule
  across ranks (all ranks issue the same sendrecv sequence) — token
  serialization means a divergent order deadlocks.
- Keep `proc_lon==1` byte-identical to today (the existing 1-D path is a
  shipping production path — must not regress).

## VERDICT (2026-06-13): fabric-blocked at ≤32 ranks — DEFER step integration

A halo micro-bench on the shipped primitives
(`scripts/bench/bench_latlon_2d_halo.py`, jobs 8477020/8477039,
LL256×512 nlev=60 halo=2, 4 nodes ≤8/node, slowest-rank time per
`pad_halo_latlon_2d`) measured EVERY `(proc_lat, proc_lon)`
factorization. Result — the 2-D split LOSES at every reachable rank
count:

| np | 1D-band (pc=1, N/S only) | best balanced 2-D | lon-only (pr=1, E/W only) |
|----|--------------------------|-------------------|---------------------------|
| 16 | 34.2 ms                  | 4×4: 59.3 ms (1.73×)  | 32.3 ms (fastest) |
| 32 | 43.1 ms                  | 8×4: 68.0 ms (1.58×)  | 40.2 ms (fastest) |

Single-DIRECTION decompositions (pure lat `pc=1` or pure lon `pr=1`)
beat any two-direction (2-D) split by 1.4–1.8×. **Why** (the recurring
latency-bound-fabric law): cross-node Gloo-TCP (no IB/NVLink) makes halo
cost ≈ (#collective calls) × latency, ~INDEPENDENT of message size
(np16→32 halves the 1-D tile, 16→8 rows/rank, yet time only rises
34→43 ms). The 2-D decomposition CUTS message SIZE but ADDS a second
halo direction (E/W ring on top of N/S) ⇒ more collective calls ⇒ net
LOSS. Worse, the 1-D band does not EXHAUST until `np > n_lat` (256 here)
— far beyond Ginsburg's reachable ≤32 ranks — so the 2-D decomposition's
raison d'être (decompose past `n_lat` ranks) never engages at our scale.

⇒ **DEFER the 2-D STEP integration** (the full operator-backend wiring of
items 3/4/5 above). It is NOT a near-term Ginsburg lever; it is a
BANDWIDTH-fabric (NVLink/IB) / `np ≫ n_lat` capability — the same class
as multinode-GPU SPMD. The 2-D FOUNDATION shipped this campaign
(`exchange_halo_lon`, `LatLon2DLayout`, scatter/gather,
`pad_halo_latlon_2d` wall, AD-safe `lon_gather_full`) STANDS as a
validated, AD-safe capability for that future regime — no further
near-term investment. Revisit when (a) an IB/NVLink fabric is available,
or (b) a grid/rank-count regime with `np > n_lat` is the target.

## M3a increment-1 (2026-07): the SPMD (shard_map) 2-D leg SHIPPED for the atmosphere

The deferral verdict above concerns the **MPI (mpi4jax, Gloo-TCP) leg** —
latency-bound fabric, one collective per direction per pad, `np ≤ 32 ≪
n_lat`.  The **single-controller SPMD leg** (`shard_map` + `ppermute` over
a `("lat", "lon")` device mesh — NVLink/PCIe intra-node, `jax.distributed`
multi-node GPU) is the bandwidth-fabric regime the deferral pointed to, and
increment-1 wires it natively for the atmosphere step
(`perf/m3a-latlon-2d-tiling`):

* **Periodic longitude = cyclic ring ppermute** (`latlon_lon_ring_perms`,
  `lon_ring_ghosts_spmd`); `p_lon == 1` is a STATIC local-wrap branch —
  bit-identical to the 1-D band path (gated).
* **True 180° pole fold under a lon split** — the piece the MPI leg lacks
  (its `proc_lon > 1` benchmark substitutes a wall pole): the pole tiles
  `all_gather` their `(halo, w)` edge rows over the `"lon"` ring, apply the
  SERIAL `_pole_fold` on the reassembled circle, and dynamic-slice their own
  padded window back out (`make_latlon_2d_pad_body`) — bit-identical to the
  serial fold by construction, gated per tile against the serial pad window.
  Follow-up optimisation: a 180°-partner ppermute pair (even `p_lon`)
  instead of the all_gather (which every tile executes uniformly).
* **Staggered ownership**: `v_lower = v[:n_lat]` (as 1-D) + `u_left =
  u[:, :n_lon]` — the u seam column is reconstructed from the east
  neighbour on the wrap (`reconstruct_uface_left`), relying on the
  step-preserved periodic-closure identity `u[:, n_lon] == u[:, 0]`.
* **Corners**: lat-then-lon two-pass exchange composition; the C-grid
  operator chain has no explicit-diagonal stencil (vertex circulations
  combine lat-padded u with lon-padded v), so no corner messages exist.
* **Topology choice** (`choose_latlon_2d_topology`): minimize the modeled
  per-tile received VOLUME of one full pad,
  `(w if p_lat>1) + (nl + n_lon if p_lon>1)` over feasible factorizations —
  the `n_lon` term is the two pole-fold `all_gather`s every `p_lon > 1` pad
  executes on EVERY tile (both `jnp.where` fold operands evaluate; codex
  M3a finding 4: a perimeter-only score mis-ranked lon splits by ignoring
  them).  Consequence: the band `(N, 1)` wins whenever FEASIBLE under the
  current fold implementation; the 2-D tiling is selected exactly in the
  beyond-band regime (`n_lat % N != 0` or `n_lat/N < 2`) it exists for.
  The 1.25× `band_preference` hysteresis is retained for when the
  partner-ppermute fold lands and removes the gather term.  The 1-D band
  lane stays the default production path.

REMAINDER (increment-2+): the ocean 2-D SPMD step; a stateful
`PhysicsState` carry under 2-D tiles (the `(ncol,)` lat-major flatten is
band-contiguous only — refused loudly); polar-filter FFT under a lon split
(lon-gather FFT — refused loudly); tripolar fold × lon split (refused, as
on the MPI leg); GPU A/B of the fold all_gather vs partner-ppermute; wiring
`run_atm_latlon_spmd` / the production driver to the 2-D factories once a
`np > n_lat` or bandwidth-fabric target exists.
