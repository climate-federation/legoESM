# Adversarial review packet 2 — CMOR flux feed, AFTER round-1 fixes

You are an independent adversarial physics + software reviewer. Round 1 you
confirmed 4 defects (C1, NEW-1 HS wrapper, C2 MPI-voronoi, NEW-2 units) and
refuted C3/C4/C5. I have applied fixes for all 4. **Re-review the FIXES**: verify
each is correct AND complete, and find any NEW bug the fix introduced (sign,
dtype, tuple-arity, dv_dt/tracer preservation, JAX purity, contract drift).
Cite file:line. If clean, say "no substantive findings" and justify per fix.

Context: baseline test set now 103 passed / 1 skipped (was 100/1; +3 new guard
tests). All changes UNCOMMITTED on mpas-stability-campaign @ 055c851cc.

## Fix 1 — C1 rsdt uses toa_insolation (radiation/integration.py:1191-1210)
```python
        _swn = rad_out.sw_flux_down[:, -1] - rad_out.sw_flux_up[:, -1]
        _lwn = rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]
        # rsdt = PRESCRIBED toa_insolation (#620) ... matches physics_pipeline.py:2225-2228
        return _pack_hydrostatic_tendencies(
            dT_dt, state, shape_3d, shape_2d,
            sw_net_sfc=_swn, lw_net_sfc=_lwn,
            sw_up_toa=rad_out.sw_flux_up[:, 0],
            lw_up_toa=rad_out.lw_flux_up[:, 0],
            sw_down_toa=(rad_out.toa_insolation
                         if rad_out.toa_insolation is not None
                         else rad_out.sw_flux_down[:, 0]))
```
`rad_out` here comes from `_call_radiation_backend` (integration.py:1159) which
sets `toa_insolation=insolation` (integration.py:739), so it is populated on this
lean rrtmgp branch. rsut/rlut keep `[:, 0]` (unchanged) — matches compiled path.

## Fix 2 — NEW-1 HS wrapper via _replace (model_driver.py:6205-6224)
```python
                    hs_tend = held_suarez_forcing_mpas(state, mesh, sigma_coord)
                    # HS adds only the 4 dynamics tendencies; _replace preserves
                    # every diagnostic field (sw/lw net, precip, TOA trio, turb
                    # fluxes) and is field-count agnostic.
                    summed = rrtmgp_tend._replace(
                        du_dt=rrtmgp_tend.du_dt.replace(
                            data=rrtmgp_tend.du_dt.data + hs_tend.du_dt.data),
                        dT_dt=rrtmgp_tend.dT_dt.replace(
                            data=rrtmgp_tend.dT_dt.data + hs_tend.dT_dt.data),
                        dp_s_dt=rrtmgp_tend.dp_s_dt.replace(
                            data=rrtmgp_tend.dp_s_dt.data + hs_tend.dp_s_dt.data),
                        dphis_dt=rrtmgp_tend.dphis_dt.replace(
                            data=rrtmgp_tend.dphis_dt.data + hs_tend.dphis_dt.data),
                    )
                    return summed, phys_state_out
```
Rationale for _replace over the explicit constructor: HS-MPAS produces only du_dt
(edge-normal) + dT/dp_s/dphis; rrtmgp_tend.dv_dt is None on MPAS (state.v is None,
integration.py:483) and tracer_tendencies is carried unchanged — so _replace of
the 4 dynamics fields is behavior-identical for those and additionally preserves
the 5 new diagnostics that the old enumerate-constructor dropped.

## Fix 3 — C2 MPI-voronoi producer now emits 8-slot (voronoi_mpi.py:1054-1072,1143-1157)
```python
            _sw_sfc = getattr(_pt, "sw_net_sfc", None)
            _lw_sfc = getattr(_pt, "lw_net_sfc", None)
            _pr_sfc = getattr(_pt, "precip", None)
            _extras = tuple(getattr(_pt, _k, None) for _k in (
                "lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc"))
            if (_sw_sfc is not None or _lw_sfc is not None
                    or _pr_sfc is not None
                    or any(_e is not None for _e in _extras)):
                sfc_diag = (_sw_sfc, _lw_sfc, _pr_sfc) + _extras
    # ... merge in _step_carry now padded (mirrors serial primitive_eq_mpas):
                _prev = getattr(model, "_sfc_diag", None) or ()
                _n = max(len(_sfc), len(_prev))
                _prev = _prev + (None,) * (_n - len(_prev))
                _sfc = _sfc + (None,) * (_n - len(_sfc))
                model._sfc_diag = tuple(
                    new if new is not None else old
                    for new, old in zip(_sfc, _prev))
```
The default `sfc_diag = (None, None, None)` (physics_fn=None case) is left 3-slot;
the padded merge lifts it to the persisted length. Slot order identical to serial.

## Fix 4 — NEW-2 turbulence units (turbulence/integration.py:668-681)
```python
            # units="W/m^2": p_s.replace would otherwise INHERIT p_s's "Pa".
            shflx_sfc=None if _shf is None else state.p_s.replace(
                data=_shf.reshape(p_s.shape), name="shflx_sfc_turb",
                units="W/m^2"),
            lhflx_sfc=None if _lhf is None else state.p_s.replace(
                data=_lhf.reshape(p_s.shape), name="lhflx_sfc_turb",
                units="W/m^2"),
```

## New guard tests (tests/unit/test_mpas_cmor_flux_feed.py, all pass)
- `test_producer_slot_order_matches_consumer`: locks (sw,lw,precip,lw_up_toa,
  sw_up_toa,sw_down_toa,shflx,lhflx) → consumer slots 3-7 rlut/rsut/rsdt/hfss/hfls.
- `test_replace_preserves_cmor_diagnostics`: HS `_replace` of dT_dt keeps all 8
  diagnostics.
- `test_padded_merge_never_truncates`: held-step 3-slot vs 8-slot prev retains
  radiation slots.

## Rebuttals (round-1 false positives, codex-agreed)
- C3 zonal omission: intentional; zonal scope = T_low/precip/psl on BOTH lean and
  compiled paths; new fields are Amon spatial products.
- C4 evspsbl=hfls/L_v: consistent with the bulk scheme's L_v-only lhflx AND the
  compiled path (physics_pipeline.py:1466); aerobulk caveat documented, out of
  scope for the rrtmgp+bulk campaign lane.
- C5 restart staleness: radiation runs on local step 0 before feed; absent-not-
  stale; same non-persistence as pre-existing sw/lw/precip slots.

## Ask
Re-review Fixes 1-4. Specifically probe: (a) does `_replace` in Fix 2 change any
behavior vs the old constructor (dv_dt None-vs-field, tracer_tendencies)? (b) is
the Fix-3 padded merge correct when the FIRST publish is 3-slot then 8-slot, and
vice-versa? (c) any remaining producer/consumer that still assumes 3 slots? (d)
Fix-1 toa_insolation availability + sign. Report CONFIRMED / FALSE-POSITIVE /
"no substantive findings".
