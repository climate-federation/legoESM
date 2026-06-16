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

## To build (in one validated unit)
1. **Dispatch** (`grids/halo_latlon.py`): in `pad_halo_latlon` (scalar, ~152)
   and `pad_halo_latlon_vector` MPI branches, add
   `elif isinstance(topology, LatLon2DLayout): return pad_halo_latlon_2d(data,
   topology, halo=halo, pole_bc="wall", is_vector_*=...)`. Keep the
   `LatLonBandLayout -> pad_halo_latlon_mpi` branch untouched.
2. **Wall helpers** (`halo_latlon.py` `zero_polar_lat_ends` ~400,
   `pad_with_pole_bc_lat` ~519, `pad_with_pole_bc_lat_multi` ~607): make
   2D-aware so ONLY `proc_row 0/last` zero/pad physical poles; interior rows
   use the neighbour halo. **This is the riskiest-bug guard** (silent
   wrong tendencies / wall-zeroed interior cuts).
3. **Step** `make_latlon_2d_mpi_step` (latlon_mpi.py, next to band ~2402):
   clone the band step — `set_halo_backend("mpi", layout2d)`, global_total_area
   allreduce, `pole_v_bc=(south_rank is None, north_rank is None)`, delegate to
   `_step_cgrid`. Model built on `slice_latlon_grid_to_block_2d`; state from
   `scatter_state_latlon_2d`; do NOT pre-pad (operators pad per RK stage).
4. **MPI conservation gate** `tests/distributed/test_latlon_2d_mpi_step.py`:
   2×2 process grid; global dry mass before/after `< 1e-12` rel drift; compare
   2D-sharded gathered vs a WALL-POLE serial reference only (NOT the
   atmosphere pole-fold serial path).
5. Wire `run_cpu_mpi_scaling --latlon-2d` (proc_lat×proc_lon from rank count);
   measure latlon 2D vs 1D-band weak/strong.

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
