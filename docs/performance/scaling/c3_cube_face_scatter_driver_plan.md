# C3 — Wire cube face-scatter into the production driver (CHANGE PLAN)

Status: **NOT implemented** (reverted 2026-07-18 at user request — code being modified
elsewhere). This note is the pick-up-later spec. All file:line refs are vs
`origin/main` @ `7103e7e12` (post-#1199).

## Goal
Opt-in `cs_mpi_scatter`: SCATTER the cubed-sphere state to per-rank owned faces
`(n_local, n, n, ...)` and run the dycore on the sliced model via the face-only
MPI halo — replacing (behind a flag) the default REPLICATE convention (full
`(6, n, n)` on every rank, physics de-sliced by owned-face index). Memory ≈
`1/n_ranks` the replicate footprint.

The primitive already exists + is bit-validated: `cube_face_scatter.py`
(`make_rank_local_cube_model`, `slice_cubed_sphere_grid/cdgrid`) +
`layout.scatter_pytree`/`gather`; `tests/distributed/test_cube_face_scatter_mpi.py`
proves scattered `model.step == serial` to 1e-10 incl AD. C3 is the DRIVER wiring.

## KEY INSIGHT — the identity-index trick (makes Stages 2+3 nearly free)
Under scatter set `self._owned_face_ids = jnp.arange(n_local)`. Then every
owned-face index op in the hot path becomes the IDENTITY:
- segment `carry.T[_ofi]`, `T_new.at[_ofi].set(...)`, held-rad/double-moment
  extract+writeback (compiled_segments.py 1807-2027, 2217-2298),
- diagnostics `state.u.data[_ofi]`, `arr[_ofi]` (model_driver.py 2471-2500),
- CFL `state.u.data[_ofi]` (8968), `_owned_p_s_and_lat` (2105).

So the ONLY real change to the segment/diagnostics is the `owned_mask` LENGTH:
hardcoded `jnp.zeros(6)` → `n_local`. Conservation is convention-invariant —
owned-partial (`mask * area * field`) + allreduce gives the SAME global reduction
whether the carry is full-6-masked or scattered-n_local-all-ones.

## Stages (dependency: 0 → 1 → {2,3} → 4)

### Stage 0 — opt-in flag
- `DeviceConfig` (packages/core/legoesm/parallel/mesh.py:44 NamedTuple): add
  `cs_mpi_scatter: bool = False` (trailing, default keeps all constructors valid).
- `ExperimentConfig` (packages/coupler/legoesm/driver/config.py:370, near
  `distributed_mode`:1076): add `cs_mpi_scatter: bool = False`. Parallel-milestone
  -gated ⇒ CLI-flag EXEMPT (like `distributed_mode='spmd'`).
- Driver: after `rc = bootstrap(...)` / `self._device_config = rc.device_config`
  (model_driver.py:2880) stamp `self._device_config = self._device_config._replace(
  cs_mpi_scatter=True)` when `self.config.cs_mpi_scatter`.
- `__init__` (~294): `self._cs_mpi_scatter_active = False`.

### Stage 1 — `_setup_parallel` scatter branch
In the cube `DistributedLayout` block (model_driver.py:3033-3178), when
`self._device_config.cs_mpi_scatter`:
1. **zero-mean auto-disable (BLOCKER — see below):** if `cfg.dycore.fix_mass` and
   `model.config.zero_mean_ps_tendency`, `self.model.config =
   model.config._replace(zero_mean_ps_tendency=False)` BEFORE make_rank_local.
2. `make_rank_local_cube_model(self.model, face_ids)` — slice dycore grid+cdgrid.
3. `self.grid = slice_cubed_sphere_grid(self.grid, face_ids)` — the DRIVER grid
   too (its area/metrics feed the between-segment mass/moisture fixer at 8401;
   also slices `subgrid_topo_stddev`). `_grid_lat/_grid_lon` are cached FULL at
   init (777) so the physics `scatter(self._grid_lat)` at 3082 still works.
4. `self.state = scatter_pytree(self.state, layout)`;
   `self.tracers = scatter_pytree(self.tracers, layout)` (Field is a registered
   pytree; NamedTuple state auto-pytree; scatter maps `(6,...)` leaves only).
5. `self._owned_face_ids = jnp.arange(n_local, dtype=int32)`;
   `self._cs_mpi_scatter_active = True`.
6. SKIP the manual `subgrid_topo_stddev` scatter (3090) under scatter — step 3
   already sliced it (double-scatter → OOB). Physics scatters (lat/lon, f_land,
   multilayer land, SST wrapper) are SHARED/unchanged (owned columns).

### Stage 2+3 — owned_mask length (the only real hot-path change)
- `build_segment_fn` (compiled_segments.py ~1415): add `scattered: bool = False`.
- segment owned_mask (1609): `_mask_len = owned_face_ids.shape[0] if scattered
  else 6`; `jnp.zeros(_mask_len).at[owned_face_ids].set(1.0)`.
- driver owned_mask (8386): same, gated on `self._cs_mpi_scatter_active`.
- both cube-MPI `build_segment_fn` call sites (8539, 9085): pass
  `scattered=self._cs_mpi_scatter_active`.
- Diagnostics/CFL/owned_p_s: NO change (identity under `arange`).

### Stage 4 — DEFER (checkpoint reshard is a separate follow-up PR)
- `save_checkpoint` (3823): if `self._cs_mpi_scatter_active` raise
  NotImplementedError — the distributed checkpoint writers + restart loader
  assume full `(6,n,n)`; owned-face save/load + cross-rank-count reshard
  (distributed_checkpoint.py, restart.py) is the follow-up.

## BLOCKER — zero_mean_ps_tendency under cube-MPI (must handle in Stage 1)
Default cdgrid model: `zero_mean_ps_tendency=True` (primitive_eq_cdgrid.py:126).
- SCATTER: `make_rank_local_cube_model` **refuses** it — the per-stage zero-mean
  divides by the rank-LOCAL owned-face `jnp.sum(area)`, not allreduced ⇒ wrong
  under decomposition. Fix = the Stage-1 auto-disable above (safe: the segment
  already runs the inner dynamics with zero_mean=False when `fix_mass` is on —
  compiled_segments.py:1639-1657; tiled lane does the same at model_driver.py:2829).
- REPLICATE cube-MPI **ALSO blows up** (day 0, non-finite winds; dynamics-only
  too) at C8/np2 — **PRE-EXISTING** (same local-area zero-mean bug). Single-
  process COMPLETES. ⇒ replicate-MPI is a BAD parity reference.

## Test (was mid-restructure when reverted)
`tests/distributed/test_cube_face_scatter_driver_mpi.py`, mpirun np{2,3,6}.
Reference = **single-process SERIAL** (`distributed=False`, completes), NOT
replicate-MPI. Both run zero_mean=False effective ⇒ scattered-gathered ==
serial to ~1e-10. Config: C8, NLEV4, DT100, gray+tke, analytical, fix_mass.
Assert `_cs_mpi_scatter_active`, `state.T.shape[0]==n_local==6//nproc`, field
parity, non-vacuity.

## Verified-safe ripples (checked)
- held-rad init derives shape from the actual (scattered) state
  (model_driver.py:7749 `shape_2d = self.state.p_s.data.shape`) ⇒ n_local auto.
- `self.q_v` reads scattered `self.tracers` (property 355).

## UNVERIFIED ripples (check when re-implementing)
- Secondary carry-build/ctx paths using init-time `shape_3d=(6,N,N,NLEV)`
  (model_driver.py:1208) vs state-derived — audit all `held_dT_rad`/ctx builders.
- `restart.py` / `distributed_checkpoint.py` full-`(6,n,n)` assumptions (Stage 4).
- `create_output_shardings` shape[0]==6 classifier (compiled_segments.py:2454) —
  SPMD path, likely irrelevant under mpi scatter but confirm.
- The mpirun np{2,3,6} test is the empirical detector — run it, fix shape/value
  mismatches until green, then codex adversarial review.

## Files to touch (re-apply order)
mesh.py · config.py · model_driver.py (init marker, device stamp, _setup_parallel
scatter, zero-mean disable, owned_mask, 2 segment call sites, checkpoint guard) ·
compiled_segments.py (scattered param, owned_mask) · new test file. Then codex
adversarial review to clean, PR, land. Stage 4 (checkpoint reshard) = follow-up PR.
