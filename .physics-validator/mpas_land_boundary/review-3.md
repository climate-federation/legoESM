- **CONFIRMED — High:** `test_mpas_global_diag.py` will fail once `mpi4py` enables it. `_make_stub()` defaults to `nlev=3` ([test_mpas_global_diag.py:36](...)), giving `dp.shape == (6, 3)` ([vertical.py:136](...)); three tests instead pass `T.shape == (6, 2)` ([test_mpas_global_diag.py:87](...), [test_mpas_global_diag.py:115](...), [test_mpas_global_diag.py:123](...)). The diagnostic multiplies them at [model_driver.py:5343](...), causing a broadcast-shape error before the new halo assertions execute. If CI runs under `mpiexec -n >1`, the cache assertion at [test_mpas_global_diag.py:129](...) also fails because the allreduced count is `4 * n_ranks` ([model_driver.py:5369](...)).

- **CONFIRMED — High:** The F-B2 runtime guard is rank-local, not global. MPI setup replaces `self.grid` with the rank-local owned+halo mesh ([model_driver.py:837](...)); topography/mask loading then builds `self._f_land` on that local grid ([model_driver.py:884](...), [model_driver.py:906](...)). The `jnp.any` guard at [model_driver.py:5849](...) therefore rejects a valid nondegenerate global mask whenever one rank’s local/halo region is all ocean.  
  - flat + no mask + knobs: static rejection is correct ([config.py:1745](...)).
  - flat + nonempty mask + knobs: static skip is correct, but distributed runtime may falsely reject.
  - flat + globally all-ocean mask + knobs: serial runtime rejection is correct; MPI needs a global owned-cell OR reduction to distinguish this from the valid case above.  
  This blocks a multi-rank land A/B unless every rank is known to include land.

- **CONFIRMED — Medium:** The MPI docstring calls the result “MASS-WEIGHTED” ([model_driver.py:5318](...)), but the implementation/body explicitly uses equal cell weight ([model_driver.py:5332](...), [model_driver.py:6569](...)). True horizontal mass weighting requires `areaCell * dp`; `areaCell` is available on the MPAS grid ([voronoi.py:111](...)). This is a documentation/scientific-label inconsistency, not a vertical-coordinate regression. Block only if the launch’s acceptance claim is an exact mass-weighted global mean.

No F-B3 finding: `normalize_grid_type` is module-scoped before the call and maps `voronoi`, `icosahedral`, `ico`, and `mpas_voronoi` to `mpas` ([config.py:72](...), [config.py:80](...)).

No F-B6 halo-NaN finding: both numerator and denominator mask after forming their values ([model_driver.py:5341](...)), so halo `p_s=NaN` cannot enter either reduction. `T_finite` intentionally tests temperature only ([model_driver.py:5329](...)).
