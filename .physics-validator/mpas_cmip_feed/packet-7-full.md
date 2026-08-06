# Final confirmation (round 7): T nlev guard + numeric coercion of ALL inputs
Round 6 found: (i) a 2-D T with wrong nlev passed (nlev derived from T, not checked vs self.nlev) -> mixed-nlev feed committed add_2d then add_3d raised IndexError; (ii) object-dtype T passed the shape check then raised in add_2d np.mean.
FIX: (a) the guard now requires T_np.ndim==2 AND T_np.shape[1]==self.nlev; (b) EVERY numeric input (T, p_s, tas, precip, phis, q_v, u_east, v_north) is coerced with np.asarray(x, dtype=np.float64) in PHASE 1, so a non-numeric object/string input raises before any commit. lat_deg keeps its float64+finite+shape check.

```python
        # Coerce to a numeric float array (``dtype=float64``): a non-numeric
        # (object/string) input raises HERE, in PHASE 1, instead of inside an
        # accumulator's ``np.mean``/``np.add.at`` mid-commit.  T is the REQUIRED
        # field and sets ``(nCells, nlev)`` for every shape check below — reject
        # a malformed rank / wrong level count FIRST (against the collector's
        # configured ``self.nlev``, so a mixed-nlev feed cannot commit a
        # different-shape ``T_low`` in add_2d before add_3d raises on the
        # pre-existing profile).
        T_np = np.asarray(T, dtype=np.float64)
        if T_np.ndim != 2 or T_np.shape[1] != self.nlev:
            raise ValueError(
                "feed_cmip_accumulators_native: T must be 2-D "
                f"(nCells, nlev={self.nlev}), got shape {T_np.shape}")
        p_s_np = np.asarray(p_s, dtype=np.float64)
        # Level -1 is the lowest (near-surface) model level (sigma_full is
        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
        T_low = T_np[..., -1]
        # tas is the 2 m MOST temperature when the caller supplies it (sst/sic
        # available), else the lowest-model-level fallback (applied AFTER the
        # shape check below) so the field is never dropped.
        _f64 = np.float64
        tas_field = None if tas is None else np.asarray(tas, dtype=_f64)
        q_v_np = None if q_v is None else np.asarray(q_v, dtype=_f64)
        u_east_np = None if u_east is None else np.asarray(u_east, dtype=_f64)
        v_north_np = None if v_north is None else np.asarray(v_north, dtype=_f64)
        precip_np = None if precip is None else np.asarray(precip, dtype=_f64)
        phis_np = None if phis is None else np.asarray(phis, dtype=_f64)

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
```
## Tests
```python
def test_feed_atomic_rejects_malformed_T(mesh):
    """A malformed required `T` (1-D, wrong nlev, or non-numeric) is rejected in
    PHASE 1 — zonal-only mode has no regrid to catch it, so add_2d must never
    commit before add_3d fails on a shape/dtype mismatch.  The nlev check is
    against the collector's configured nlev so a mixed-nlev feed can't slip
    through after a valid one."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=True, cmip_output=False)
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)

    # 1-D T (wrong rank).
    with pytest.raises(ValueError, match="T must be 2-D"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"][:, -1], p_s=f["p_s"], lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == 0

    # A VALID feed first (establishes an nlev=NLEV profile bucket)...
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    _cnt = dc.monthly_accum._max_count_ever
    assert _cnt > 0
    # ...then a wrong-nlev T must be rejected pre-commit (count unchanged).
    with pytest.raises(ValueError, match="nlev"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=np.zeros((n, NLEV + 1)), p_s=f["p_s"],
            lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == _cnt
```

36 tests pass (25 feed + 11 collector). The production caller always passes well-formed (nCells, self.nlev) float64 arrays; these guards are defensive. Confirm the feed is now transactional in ALL modes against wrong shape, wrong nlev, wrong rank, and non-numeric dtype for every input, with the ONLY residual being the inherited (shared-with-save()) CMOR-writer field-by-field non-atomicity. If so, state the review is CLEAN with no substantive blocking bugs.
