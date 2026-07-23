### ROUND-5 fix: up-front shape validation of ALL inputs (transactional in every mode)
```python
        T_np = np.asarray(T)
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
        _ncol = int(T_np.shape[0])
        _nlev = int(T_np.shape[1]) if T_np.ndim == 2 else None
        _shape_checks: list[tuple[str, np.ndarray | None, tuple]] = [
            ("p_s", p_s_np, (_ncol,)),
            ("precip", precip_np, (_ncol,)),
            ("phis", phis_np, (_ncol,)),
            ("tas", tas_field, (_ncol,)),
        ]
        if _nlev is not None:
            _shape_checks += [
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
            T_low_safe = np.maximum(T_low, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis_np / (_c.R_d * T_low_safe))
```
### zonal lat validation (unchanged logic, reuses up-front _ncol)
```python
        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit
            # (the sibling zonal fields were shape-checked up front).  Force a
            # numeric float array so a shape-correct-but-non-numeric (object/
            # string) lat_deg raises here, not inside ``np.digitize`` mid-
            # commit; also require finite values.
            lat_np = np.asarray(lat_deg, dtype=np.float64)
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            if not np.all(np.isfinite(lat_np)):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg has non-finite "
                    "values")
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
            if psl is not None:
                z2d['psl'] = np.asarray(psl)
            z3d['T'] = T_np
            if u_east_np is not None:
                z3d['u'] = u_east_np
            if q_v_np is not None:
```
