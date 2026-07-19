# lat-lon 2D pencil MPI step — executable build plan (codex-designed)

Goal: fix the atm lat-lon 1D-band scaling laggard (weak E≈0.05 at high rank)
with a 2D `(proc_lat × proc_lon)` pencil decomposition (smaller halo perimeter,
SOTA MOM6/E3SM). **Wall-pole only** — see caveat. This is a DEDICATED,
correctness-critical, multi-rank-validated build (NOT a parked-loop fragment:
the dispatch + wall-helper patches must land together and be MPI-tested, or
interior process rows silently treat latitude cuts as poles).

## Done (committed, tested)
- `scatter_state_latlon_2d` (latlon_mpi.py) — slice global C-grid state to a
  lat×lon block; v keeps the shared boundary row. proc_lon=1 == band.
- `slice_latlon_grid_to_block_2d` (latlon_mpi.py) — slice every metric to the
  block; total_area = global allreduce. proc_lon=1 == band slice.
- **Lat-axis 2-D dispatch + step (this increment).**
  - 4 fold-family dispatchers (`pad_halo_latlon[_vector][_3d]`): at
    **proc_lon==1** reuse the validated band POLE-FOLD (lon full per rank ⇒
    local 180-deg fold via the equivalent `LatLonBandLayout` —
    bit-identical to serial, correct for EVERY scalar caller incl. the
    A-grid operators that read the pole ghost directly); at **proc_lon>1**
    fall back to the wall-pole benchmark pad (`pole_bc="wall"`, only
    reachable by an explicit lon split — the step refuses proc_lon>1).
    Codex caught the naive "always wall-zero" version as a latent
    A-grid-corruption footgun.
  - `zero_polar_lat_ends` + `pad_with_pole_bc_lat[_multi]` (wall family) +
    `get_band_mpi_cut_layout`/`lat_ends_are_poles`
    (`operators_latlon_cgrid.py`) all widened to recognise `LatLon2DLayout`
    (same `south_rank`/`north_rank` pole-touch test) — without the last two,
    a proc_lat>1 interior rank would clamp its band edge as a physical pole
    (curl_vertex sin clamp, codex finding). Additive; band/local/serial
    paths byte-for-byte unchanged.
  - `_pad_lat_wall_2d` (shared lat-axis sendrecv+wall core) + thin
    `pad_with_pole_bc_lat_2d` wrapper; `pad_halo_latlon_2d` refactored to call
    the shared core (no N/S duplication).
  - `make_latlon_2d_mpi_step` — clone of the band step; **proc_lon>1 WIRED
    for regular / wall-pole grids** (lon ops through the dispatched lon halo;
    the polar filter through the lat-pencil transpose). Still refuses a
    tripolar grid and an uneven lon split under `use_polar_filter`.
  - Tests: `tests/parallel/test_latlon_2d_dispatch_serial.py` (serial CI:
    routing at 1×1 + the two guards) and
    `tests/distributed/test_latlon_2d_dispatch_mpi.py` (np={2,3,6}:
    `pad_with_pole_bc_lat_2d` lat-only parity + AD-VJP + dispatch fires under
    real MPI). Validate via sbatch (login node cannot run MPI/JIT).

## SCOPE CORRECTION (the plan under-counted the lon axis)
The original plan scoped only the LAT-axis (4 dispatchers + 3 wall helpers).
But in the band decomposition longitude is NOT split — every rank holds the
full lon circle — so the C-grid operators take their lon halo with LOCAL
`jnp.roll(…, axis=1)` / wrap-column appends. A genuine 2-D LONGITUDE split
(proc_lon>1) makes those wrap the rank-local lon block instead of the lon
neighbour → SILENTLY wrong longitude gradients that still ~conserve after the
mass fixer (risk #1, now on the lon axis). Audited raw lon ops needing
conversion to the dispatched `exchange_halo_lon` (bit-identical to the local
wrap at proc_lon==1, so the band/serial path stays exact):
`operators_latlon_cgrid.py` — `gradient_x_cgrid` (line ~442 `jnp.roll(f,1,axis=1)`),
`interp_cell_to_uface` (~187), `curl_vertex_cgrid` (~729/843/847/885/915),
`m_sw` (~1197) — ~8–11 wall-pole-relevant sites (fold-only rolls excluded;
the wall-pole benchmark has fold inactive). This is the **next increment**;
until it lands `make_latlon_2d_mpi_step` raises for proc_lon>1.

## Build status (per increment)
1. **Dispatch** (`grids/halo_latlon.py`) — ✅ DONE. 4 fold-family dispatchers
   route `LatLon2DLayout -> pad_halo_latlon_2d(pole_bc="wall")`; band/local
   branches untouched. (No `is_vector_*` arg — at a wall pole v=0 is the wall
   constant, no fold sign flip.)
2. **Wall helpers** — ✅ DONE. `zero_polar_lat_ends` widened to
   `(LatLonBandLayout, LatLon2DLayout)` (identical proc_row-0/last pole-touch
   test); `pad_with_pole_bc_lat` routes 2-D to the lat-only
   `pad_with_pole_bc_lat_2d` (fold seam / vector-u raise); `_multi` routes
   per-field (fused path stays band-only). **Riskiest-bug guard** covered by
   the dispatch tests.
3. **Step** `make_latlon_2d_mpi_step` — ✅ DONE for proc_lon==1 (band-
   equivalent), **raises for proc_lon>1** pending the lon-op conversion. Clone
   of the band step (set_halo_backend, global_total_area allreduce,
   pole_v_bc).
4. **MPI gate** — ⏳ PARTIAL. Cheap primitive/dispatch/AD gates DONE
   (`test_latlon_2d_dispatch_{serial,mpi}.py`). The full dycore-STEP
   conservation smoke (2×1 proc_lon==1, mass drift `<1e-12`) is deferred to an
   sbatch run with a long walltime — the C-grid `_step_cgrid` JIT compile is
   ~25 min on Ginsburg (same blocker the band `test_latlon_mpi_step.py` skips
   document). The 2-D step reuses the band-validated `_step_cgrid` with
   unit-tested lat-only 2-D pads, so residual step-level risk is low.
5. **lon-op conversion** (the SCOPE-CORRECTION work) — ⏳ NEXT. Route the
   ~8–11 C-grid operator lon ops through `exchange_halo_lon`; then drop the
   proc_lon>1 guard and add the 2×2 dycore conservation gate.
6. Wire `run_cpu_mpi_scaling --latlon-2d` + measure 2-D vs 1-D band — ⏳ after 5.

## Caveat (codex, non-negotiable)
Wall-pole 2D is valid ONLY as a labeled midlatitude throughput benchmark. It is
NOT the atmosphere's 180° pole fold (`pad_halo_latlon_2d(pole_bc="fold")`
deliberately raises). Do not advertise it as atmospheric-pole-correct. The
production fold-under-lon-split needs a lat-pencil transpose (separate project).

## Risks
- #1 silent fallback to local halo on a LatLon2DLayout (interior cuts as poles)
  -> wrong tendencies that still ~conserve after the mass fixer. The 2×2 MPI
  test + a 2D-vs-serial-wall-pole gather check catch it.
- #2 v shared lat-face rows diverging between neighbour proc rows after an RK
  stage (only one side zeroed/updated).
