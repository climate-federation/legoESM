# FV3 duo as a column model inside the MPAS lane (route A)

User decisions 2026-09-26: run the CAM6-equivalent AMIP suite
(`config/amip/amip_production.yaml`) on the FV3 duo dycore at coarse
resolution; the duo becomes the dynamics operator of an EXISTING driver
loop through a state adapter; no physics is rewritten; ponytail; GLM +
codex review at every rung. Route A chosen after the v1 claim reviews
(codex 22 findings, GLM 4) showed the CAM6 suite is orchestrated ONLY by
the MPAS lane (`ModelDriver._run_mpas`): prognostic CLUBB moments carry,
CAM6 clouds, physics every N steps with the macro/micro sub-cycle,
convective momentum. The physics FUNCTION is shared across lanes
(`make_physics` -> `_make_hydrostatic_combined`); the lane is what the
duo must fit.

## What the MPAS lane actually needs (surveyed 2026-09-26, file:line in
the survey; `self.grid` IS the mesh)

* Lane-level mesh attributes: `latCell`, `lonCell`, `areaCell`,
  `nCells`, `grid_lat`/`grid_lon` (rad). All cell-centre scalars.
  CMOR/monthly output builds k-nearest IDW weights from latCell/lonCell
  only (`driver/diagnostics.py:506-517`). Land data, SST/SIC regrid and
  radiation take lat/lon only.
* State = `MPASHydrostaticState` = HydrostaticState with `u` on EDGES,
  `v=None`, `T`/`p_s`/`phis`/`tracers` on cells `(nCells[, nlev])`.
  Because winds are edge-normal, physics integration modules
  (turbulence, GWD, convection, land columns, KE diag, ERA5 IC, qv
  smoothing) call `reconstruct_cell_velocity` / `cell_vector_to_edge_normal`
  on edge topology. That is the ONLY MPAS-specific coupling in physics.
* Step contract (`primitive_eq_mpas.py:960-1156`): `model.step(state, dt,
  physics_fn=, forcing=, phys_state=)`; the step runs dynamics, calls
  `physics_fn(state_new, mesh, sigma_coord, phys_state=, forcing=)`,
  adds `dt*du_dt` (edges), `dt*dT_dt`, `dt*dp_s_dt` (zeros from physics),
  `dt*tracer_tendencies`, then the uniform additive dry-mass p_s fixer,
  and stashes `_phys_state` / `_sfc_diag`.
* Conventions: q_v = specific humidity per moist mass (same as FV3
  sphum); physics never changes p_s; orography enters as `state.phis`.
* The lane reads `state.T.data.shape[0]` as nCells in ~27 places.

## Adapter v3 (after the v2 reviews: codex 6 findings, GLM 5)

1. `make_physics(model_type="columns")`: the MPAS combined path (which
   already runs on `(ncol, nlev)` columns and forwards cadence / macmic
   / f_land / land_beta / ledger, `combined.py:290-311`) with DIRECT cell
   winds: `state.u`, `state.v` geographic, `(ncol, nlev)`. Every
   edge-bound site in the integration modules gets the column branch
   and returns BOTH cell components (today the MPAS branch returns
   edge-normal `du_dt` and `dv_dt=None`): turbulence `:820,:1064`,
   convection `:446,:934` (cell path exists `:433-438,939-951`), GWD
   `:746,:802-807`. Frontogenesis GWD source needs spatial gradients
   and stays refused on columns (not in the CAM6 deck). No kernel
   changes.
2. `FV3DuoColumnState` as v2, with the closed lane's ORDER-4 c2l view.
3. `step(state, dt, physics_fn=, forcing=, phys_state=)`: dynamics;
   column view; physics; apply with the FV3 twins in fv_update_phys
   order: tracers + layer mass + T through the moist block
   (`fv_update_phys_moist_duo_jax`: `delp *= ps_dt`, tracers renormalised,
   T on cvm) so p_s follows evaporation minus precipitation (GLM: a
   fixed delp with additive tracers random-walks the dry mass; codex:
   same), winds through the dry block (geographic A-grid u_dt/v_dt ->
   D) AFTER the D-wind halo exchange and the one-ring tendency exchange
   the twin's contract requires (`fv3_native_physics_coupling.py:750-799`);
   pressures rebuilt (`p_var_hydrostatic`). NO additive p_s fixer on the
   duo: FV3 conserves dry mass by construction; the fixer increment is
   asserted zero as an audit.
4. `DuoColumnMesh`: latCell/lonCell/areaCell/nCells/grid_lat/grid_lon/
   `grid_shape_2d`; no edge topology.
5. Lane dispatch: `run()` -> `_run_mpas` under the new explicit switch
   `fv3_duo_column_lane=True`; duo branches at EVERY edge-bound site of
   the lane (inventory, `model_driver.py`): qv smoothing `:10355,10372,
   10387,11957-11958` (REFUSED unless both coefficients are 0: it is an
   MPAS-lane numerics knob -- ASK, the deck sets del4 3.6421e14); land
   reconstruction `:10999`; KE diag `:12238`; sponge on `u` only
   `:11938-11941`; max-wind on `u` only `:12080,12122`; CMOR helper
   reconstruction `:7992` and Voronoi divergence for `wap` `:8023`
   (duo: omega from its own step); distributed edge exchange `:8303`;
   Voronoi MPI builders `:11426-11440` (column lane single-process
   first; windows are rung 7); ERA5 conversion `:2748` (item 6).
6. IC/vertical as v2 (ERA5 -> A-grid columns; D winds by the lift at
   t=0 after the same halo/one-ring exchanges; CAM L32 ak/bk table;
   km stamped in the checkpoint).
7. Checkpoint/restart: the MPAS lane's writer (`:6121-6269`) gains the
   duo bundle as an explicit schema (it enumerates `u/T/p_s/phis`
   today, no `v`, no private leaves) and the FV3 dispatch at `:6772`
   learns the column lane. `held_physics` is reseeded on restart on the
   MPAS lane too (`physics_state.py:275-278`), so restart equality is
   the MPAS lane's guarantee, not bitwise.
8. Cadence is a TIME invariant: physics every 1800 s, radiation hourly;
   `physics_update_steps` / `rad_update_steps` are set from the duo's dt
   (GLM: a 16-step lump at dt 1800 s would be an 8 h physics step).
9. DECIDED (user 2026-09-26): (a) qv del2/del4 smoothing OFF on the duo
   (refused unless both coefficients are 0); (b) orography gate =
   smoothed ERA5 terrain (CAM-style filter to the C24 scale) + 10-day
   dry run checked for hydrostatic heights vs ERA5 and no grid-scale
   noise over the Andes/Himalaya, plus the rest-state balance test, no
   Fortran dump; (c) C24 smoke, C48 scorecard, CAM L32; dt proposed
   with the deck.

## M2 landed (2026-09-26): the column model, rungs 1-3 at C12 km=5

`atmosphere/dynamics/gcm/fv3_duo_column.py` (`FV3DuoColumnModel`,
`FV3DuoColumnState`, `DuoColumnMesh`) over two shared six-face
functions in `core/fv3_native_physics_coupling.py`
(`column_view_sixface_jax`, `apply_column_increments_sixface_jax`) that
the Held-Suarez twin and the Kessler bridge now also run on (receipt:
both bitwise vs HEAD, eager and jit, C12 km=5).  Decisions fixed by the
M2 dual review (codex 4xP1 + 2xP2, GLM 5):

* Heating convention is CONFIG-time, not data-driven: the moist deck
  (`FV3DuoConfig.moist=True`) always applies `dT` through
  `fv_update_phys`'s moist block (`c_pd -> cp_air` rescale, then
  `cp_air/cvm`), water tendency or not; the dry deck applies `dT` as
  given (idealized Held-Suarez only) and REFUSES water tendencies.
* Tracer names must start with `(q_v, q_c, q_r)` (the warm-rain slots
  the nwat block reads) and be unique.  The CAM6 deck carries ICE
  (`q_i`...): rung 5 needs FV3's nwat=6 block (fv_update_phys.F90
  sums every water species); a tendency on any other slot is refused
  until then.  Tracked, not hidden.
* `dp_s_dt` is not consumed (MPAS physics returns zeros); p_s follows
  the layer mass the water tendencies move.  Louis on the moist deck:
  water mass change == dt*sum(delp*dq) (the surface evaporation), p_s
  rises by exactly that, dry mass delp*(1-q) invariant to 1e-12.
* `_sfc_diag` merges slot-wise (held-radiation steps keep the last
  sw/lw), as the MPAS model.
* Cross-program identities (column lane vs bridge, both jitted) hold to
  1e-12 of peak, the twin gate's tolerance, not bitwise (XLA fusion
  differs between programs; the HS identity within one program IS
  bitwise).
* Findings on the way: the gridstruct cell areas sum to the sphere to
  3.5e-6 relative at C12 (not exact spherical excess);
  `test_held_suarez_jax_differentiable_wrt_state` fails on untouched
  HEAD (FD gap/floor 18.4 vs 10) -- pre-existing, not M2's.

## Certification ladder v3

1. Zero-tendency physics through the column lane == closed duo lane
   bitwise, C24 km=5 (plumbing identity).
2. Held-Suarez: the JAX twin as physics_fn vs the closed lane's NumPy
   twin at 1e-12 of peak (the existing twin gate's tolerance; codex:
   the closed lane runs the NumPy authority, `:8795-8797`).
3. Kessler: column lane (moist block inside step) vs the closed lane's
   Kessler bridge bitwise (same block), then water / dry-mass / energy
   budgets over 10 days.
4. CAM L32: rest-state hydrostatic gate; DCMIP16 km=32 10 days finite;
   orography gate per ask (b).
5. CAM6 deck at C24 L32, 10-day smoke, restart once, equality per the
   MPAS lane's own restart guarantee.
6. AMIP scorecard vs the MPAS CAM6 run, same windows, cadence matched
   in TIME.
7. Window SPMD through the column lane vs the flat batched reference,
   the existing window gate's definition.

## Recorded on the way (out of scope)

The fused compiled cube lane (`compiled_segments.py`) never consumes
`phys_out.du_dt/dv_dt` (codex CONFIRMED statically 2026-09-26):
turbulence and GWD momentum tendencies are dropped on that lane. Needs
a planted-tendency control and its own issue.
