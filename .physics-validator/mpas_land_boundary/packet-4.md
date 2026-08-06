You are an independent adversarial reviewer for LegoESM (JAX-native differentiable ESM).
ROUND-4 mini closing review of the MPAS land surface boundary + mass-weighted mean-T change.
Rounds 1-3 found F1-F5, F-B1..F-B7, F-C1, F-C2; this round verifies three small fixes.
FINDINGS ONLY — do NOT modify any file. Read the actual source (repo cwd, working tree);
cite file:line; label CONFIRMED/PLAUSIBLE + severity. mpi4py is NOT installed here, so
tests/unit/test_mpas_global_diag.py is SKIPPED — reason about MPI CI behavior.

Context: the 60-day A/B is CONFIRMED single-process (1 GPU, no MPI); F-C2 was fixed anyway.

## Round-4 deltas to verify
- F-C1 (test nlev mismatch): the three 2-level tests in tests/unit/test_mpas_global_diag.py
  now call `_make_stub(nlev=2)` (lines ~87, ~116, ~124) so `create_sigma_coordinate(2)` gives
  `dp` shape `(6,2)` matching `T=(6,2)`. The first test keeps `nlev=3` with `T=(6,3)`.
- F-C2 (rank-local guard -> global): `_run_mpas` (model_driver.py ~5854-5872) now computes
  `_has_land` locally, then `if self._voronoi_layout is not None:` does
  `_has_land = bool(MPI.COMM_WORLD.allreduce(_has_land, op=MPI.LOR))`, and raises only on the
  GLOBAL all-zero. Single-process (`_voronoi_layout is None`) keeps the local verdict.
- F-B1 (docstring): `_mpas_global_diag` docstring (model_driver.py ~5318) now says
  "PRESSURE-WEIGHTED (sum T*dp / sum dp; equal cell weight — areaCell weighting is a deferred
  refinement on the quasi-uniform SCVT)".

## Specific asks (rank CONFIRMED/PLAUSIBLE)
1. Is the F-C2 allreduce deadlock-free? Do ALL MPI ranks reach the `allreduce` call
   unconditionally when the knobs are on (is the trigger `_land_beta!=1 or _land_lapse_K_m>0`
   config-static and identical on every rank; is `self._voronoi_layout` set on every rank in a
   distributed MPAS run)? Any rank-divergent path that would call the collective on only some
   ranks and hang? Is `MPI` imported safely (only inside the MPI branch, no single-process
   dependency)?
2. Is the F-C2 verdict correct (LOR over per-rank `jnp.any(local_f_land>0)` == global has-land;
   raise iff globally all-ocean)? Does the subsequent logging block (`jnp.mean(_f_land_cells)`)
   have any None/empty hazard?
3. F-C1: do all four tests now have consistent nlev between the stub's
   `create_sigma_coordinate(nlev)` and their `T` array's level count? Any residual
   broadcast mismatch, and is `create_sigma_coordinate(2)` a valid 2-level coordinate?
4. F-B1: does the docstring now match the implementation (equal cell weight, not area-weighted)?
5. Any NEW issue introduced by these three deltas. Anything you would still BLOCK the
   single-process 60-day A/B on.

## Relevant current source (verbatim)

model_driver.py `_run_mpas` guard block (~5844-5884):
```python
        _land_beta = float(getattr(cfg, "mpas_land_beta", 1.0))
        _land_lapse_K_m = (
            float(getattr(cfg, "mpas_land_lapse_K_per_km", 0.0)) * 1.0e-3)
        _f_land_cells = None
        if self._f_land is not None:
            _f_land_cells = jnp.asarray(self._f_land).reshape(-1)
        if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
            # Under MPI self._f_land is the RANK-LOCAL (owned+halo) field; an
            # ocean-only rank must not falsely abort a run whose GLOBAL mask
            # has land (codex F-C2).  The logical-OR allreduce is collective
            # and every rank computes the same verdict, so the raise (or
            # not) is deadlock-free.
            _has_land = (_f_land_cells is not None
                         and bool(jnp.any(_f_land_cells > 0.0)))
            if self._voronoi_layout is not None:
                from mpi4py import MPI as _MPI
                _has_land = bool(
                    _MPI.COMM_WORLD.allreduce(_has_land, op=_MPI.LOR))
            if not _has_land:
                raise ValueError(...)
        if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
            logger.info("  MPAS land boundary: lapse=%.2f K/km, beta=%.2f "
                        "(f_land mean=%.3f)", _land_lapse_K_m*1e3, _land_beta,
                        float(jnp.mean(_f_land_cells)))
        physics_fn = make_physics(phys_cfg, model_type="mpas", dt=DT,
                                  column_mesh=_column_mesh,
                                  f_land=(_f_land_cells if _land_beta != 1.0 else None),
                                  land_beta=_land_beta)
```

test_mpas_global_diag.py (nlev usage): `_make_stub(n_cells=6, n_owned=4, n_edges=5,
n_owned_edges=3, nlev=3)` sets `drv.sigma = create_sigma_coordinate(nlev)`.
`test_owned_masked_means_and_extrema_single_rank`: `_make_stub()` + `T=(6,3)`.
`test_finite_flag_owned_only`: `_make_stub(nlev=2)` + `T=(6,2)` (+ halo-NaN on T and on p_s).
`test_cwv_none_returns_nan`: `_make_stub(nlev=2)` + `T=(6,2)`.
`test_n_cells_cache_populated_once`: `_make_stub(n_owned=4, nlev=2)` + `T=(6,2)`.
