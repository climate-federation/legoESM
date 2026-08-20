# Final confirmation: T 2-D guard (round-5 fix)
Round 5 found: a malformed 1-D required T was not rejected (zonal-only add_2d committed a scalar T_low before add_3d's reshape failed). FIXED with a dimensionality guard at the very top of feed_cmip_accumulators_native, before T_low/_ncol/_nlev are derived. _nlev is now always int and the 3-D sibling shape checks always run.

## The guard + shape-check block
```python
        T_np = np.asarray(T)
        # T is the REQUIRED field and sets (nCells, nlev) for every shape check
        # below.  Reject a malformed 1-D (or empty-level) T FIRST — otherwise a
        # zonal-only feed would commit a scalar ``T_low`` in add_2d before
        # add_3d raises on the reshape (a partial, non-atomic commit).
        if T_np.ndim != 2 or T_np.shape[1] < 1:
            raise ValueError(
                "feed_cmip_accumulators_native: T must be 2-D (nCells, nlev), "
                f"got shape {T_np.shape}")
        p_s_np = np.asarray(p_s)
        # Level -1 is the lowest (near-surface) model level (sigma_full is
        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
        T_low = T_np[..., -1]
        # tas is the 2 m MOST temperature when the caller supplies it (sst/sic
        # available), else the lowest-model-level fallback (applied AFTER the
        # shape check below) so the field is never dropped.
        tas_field = None if tas is None else np.asarray(tas)
        q_v_np = None if q_v is None else np.asarray(q_v)
        u_east_np = None if u_east is None else np.asarray(u_east)
        v_north_np = None if v_north is None else np.asarray(v_north)
        precip_np = None if precip is None else np.asarray(precip)
        phis_np = None if phis is None else np.asarray(phis)

        # Shape contract — validated UP FRONT so BOTH the spatial regrid AND the
        # zonal binning are transactional.  A malformed optional input raises
        # HERE (PHASE 1), before any accumulator is mutated: the ZONAL-only
        # config has no regrid phase to catch a wrong-length precip / cell wind
        # / q_v, so without this a bad sibling field would commit ``T_low`` then
        # raise inside ``np.add.at``.  ``ncol``/``nlev`` come from ``T`` (the
        # one required field).
        _ncol, _nlev = int(T_np.shape[0]), int(T_np.shape[1])
        _shape_checks: list[tuple[str, np.ndarray | None, tuple]] = [
            ("p_s", p_s_np, (_ncol,)),
            ("precip", precip_np, (_ncol,)),
            ("phis", phis_np, (_ncol,)),
            ("tas", tas_field, (_ncol,)),
            ("q_v", q_v_np, (_ncol, _nlev)),
            ("u_east", u_east_np, (_ncol, _nlev)),
            ("v_north", v_north_np, (_ncol, _nlev)),
        ]
        for _nm, _arr, _shp in _shape_checks:
            if _arr is not None and _arr.shape != _shp:
                raise ValueError(
                    f"feed_cmip_accumulators_native: {_nm} shape {_arr.shape} "
                    f"!= expected {_shp}")

        # tas is the 2 m MOST temperature when supplied, else the lowest level.
        if tas_field is None:
            tas_field = T_low

        # Sea-level pressure (hypsometric) — shared by the spatial ``psl`` and
        # the zonal ``psl`` band; compute once when phis is available (shape
        # validated above, so the broadcast is safe).
        psl = None
        if phis_np is not None:
```
## Test (round 5)
```python
def test_feed_atomic_rejects_1d_T(mesh):
    """A malformed 1-D required `T` is rejected in PHASE 1 (zonal-only mode has
    no regrid to catch it) so add_2d never commits a scalar before add_3d's
    reshape fails."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=True, cmip_output=False)
    f = _synthetic_cell_fields(mesh, sigma_full)
    T_1d = f["T"][:, -1]                              # (nCells,) — wrong rank
    with pytest.raises(ValueError, match="T must be 2-D"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=T_1d, p_s=f["p_s"], lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == 0
    assert not dc.monthly_accum._data
```

25 feed tests pass. Confirm: is a 1-D (or empty-nlev) T now rejected in PHASE 1 before any commit, and is the feed transactional in ALL modes with no remaining commit-time failure path? If the only residual is the documented inherited CMOR-writer non-atomicity, state the review is CLEAN with no substantive blocking bugs.
