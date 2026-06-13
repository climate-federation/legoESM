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
