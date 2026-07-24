# Adversarial review packet 1 — CMOR radiation/turbulent-flux feed, MPAS lean lane

You are an independent adversarial physics + software reviewer. Find every bug,
sign error, unit inconsistency, broken-gradient pattern, conservation violation,
tuple-contract mismatch, and consistency divergence in this implementation.
Cite file:line. If you believe a candidate concern is NOT a bug, say so and
explain why. Be concrete and skeptical.

## Motivation
The first ClimateEval scorecard ran with `rlut/rsut/rsdt/hfss/hfls/evspsbl`
absent — the lean MPAS loop (serial `MPASPrimitiveEquationModel.step()` +
`ModelDriver._feed_mpas_cmip_accumulators` → `feed_cmip_accumulators_native`)
never exported them, so the radiation-budget and surface-flux ClimateEval
suites were skipped. `cmor_output.py` attr tables already had these vars; only
the feed chain was missing. This change wires the feed. All changes are
UNCOMMITTED working tree on `mpas-stability-campaign` @ 055c851cc.

## CMOR sign conventions (target)
- `rlut` = TOA outgoing LW, positive UP.
- `rsut` = TOA outgoing SW, positive UP.
- `rsdt` = TOA incident SW, positive DOWN.
- `hfss` = surface sensible heat flux, positive UP.
- `hfls` = surface latent heat flux, positive UP.
- `evspsbl` = evaporation, positive UP (kg/m^2/s).

## The chain (7 files + 1 test), full diffs vs 055c851cc

### 1. packages/core/legoesm/core/state.py — HydrostaticTendencies +5 optional fields
```python
    precip: Field | None = None
    # TOA radiative fluxes (CMOR sign conventions: *_up positive UPWARD /
    # outgoing, sw_down_toa positive DOWNWARD / incoming) [W/m^2] ...
    sw_up_toa: Field | None = None
    lw_up_toa: Field | None = None
    sw_down_toa: Field | None = None
    # Surface turbulent fluxes [W/m^2, positive UPWARD out of the surface —
    # the CMOR hfss/hfls convention ...]. ... (evspsbl is derived downstream
    # as lhflx / L_v).
    shflx_sfc: Field | None = None
    lhflx_sfc: Field | None = None
```

### 2. radiation/integration.py — _pack_hydrostatic_tendencies +3 TOA kwargs; rrtmgp branch
```python
def _pack_hydrostatic_tendencies(dT_dt, state, shape_3d, shape_2d,
                                 sw_net_sfc=None, lw_net_sfc=None,
                                 sw_up_toa=None, lw_up_toa=None,
                                 sw_down_toa=None):
    ...
    def _toa_field(arr, name):
        if arr is None:
            return None
        return Field(data=arr.reshape(shape_2d).astype(_ps_dtype),
                     name=name, dims=dims_2d, units="W/m^2")
    return HydrostaticTendencies(
        ...
        sw_net_sfc=sw_field, lw_net_sfc=lw_field,
        sw_up_toa=_toa_field(sw_up_toa, "sw_up_toa_rad"),
        lw_up_toa=_toa_field(lw_up_toa, "lw_up_toa_rad"),
        sw_down_toa=_toa_field(sw_down_toa, "sw_down_toa_rad"),
    )

# in _make_hydrostatic_radiation, rrtmgp branch (surface = last half-level):
        _swn = rad_out.sw_flux_down[:, -1] - rad_out.sw_flux_up[:, -1]
        _lwn = rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]
        # TOA is the FIRST half-level:
        return _pack_hydrostatic_tendencies(
            dT_dt, state, shape_3d, shape_2d,
            sw_net_sfc=_swn, lw_net_sfc=_lwn,
            sw_up_toa=rad_out.sw_flux_up[:, 0],
            lw_up_toa=rad_out.lw_flux_up[:, 0],
            sw_down_toa=rad_out.sw_flux_down[:, 0])
```

### 3. turbulence/integration.py — _make_mpas_turbulence attaches shflx/lhflx
```python
        _shf = getattr(turb_out, "shflx", None)
        _lhf = getattr(turb_out, "lhflx", None)
        tendencies = HydrostaticTendencies(
            ...,
            shflx_sfc=None if _shf is None else state.p_s.replace(
                data=_shf.reshape(p_s.shape), name="shflx_sfc_turb"),
            lhflx_sfc=None if _lhf is None else state.p_s.replace(
                data=_lhf.reshape(p_s.shape), name="lhflx_sfc_turb"),
        )
```

### 4. physics/combined.py — _accumulate collects sfc_diag_extras; re-attach helper
```python
        _DIAG_FIELDS = ("sw_up_toa", "lw_up_toa", "sw_down_toa",
                        "shflx_sfc", "lhflx_sfc")
        sfc_diag_extras = {k: getattr(first, k, None) for k in _DIAG_FIELDS}
        # ... per additional module:
            for _k in _DIAG_FIELDS:
                if sfc_diag_extras[_k] is None:
                    sfc_diag_extras[_k] = getattr(t, _k, None)
        return (du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
                combined_tracer_tends, phys_updates, first, precip_accum,
                sfc_diag_extras)

    def _attach_sfc_diag_extras(combined, extras):
        _set = {k: v for k, v in extras.items() if v is not None}
        return combined._replace(**_set) if _set else combined
    # held-radiation branch AND full-rad branch BOTH call:
    #   combined = _attach_sfc_precip(combined, first, precip_accum)
    #   combined = _attach_sfc_diag_extras(combined, sfc_diag_extras)
    # both _accumulate call sites unpacked with the extra return value.
```

### 5. dynamics/gcm/primitive_eq_mpas.py — sfc_diag tuple 3→8; pad-merge; publish
```python
            _prev = getattr(self, "_sfc_diag", None) or ()
            _n = max(len(sfc_diag), len(_prev))
            _prev = _prev + (None,) * (_n - len(_prev))
            sfc_diag = sfc_diag + (None,) * (_n - len(sfc_diag))
            self._sfc_diag = tuple(
                new if new is not None else old
                for new, old in zip(sfc_diag, _prev))
        ...
            _extras = tuple(getattr(_pt, _k, None) for _k in (
                "lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc"))
            if (_sw_sfc is not None or _lw_sfc is not None
                    or _pr_sfc is not None
                    or any(_e is not None for _e in _extras)):
                sfc_diag = (_sw_sfc, _lw_sfc, _pr_sfc) + _extras
            # tuple contract: (sw_net, lw_net, precip, lw_up_toa, sw_up_toa,
            #                  sw_down_toa, shflx, lhflx)
```

### 6. coupler/driver/model_driver.py — _feed_mpas_cmip_accumulators reads slots 3-7
```python
            def _sfc_slot(i):
                if (_sfc_diag is not None and len(_sfc_diag) > i
                        and _sfc_diag[i] is not None):
                    return _sfc_diag[i].data
                return None
            rlut = _sfc_slot(3)   # lw_up_toa
            rsut = _sfc_slot(4)   # sw_up_toa
            rsdt = _sfc_slot(5)   # sw_down_toa
            hfss = _sfc_slot(6)   # shflx
            hfls = _sfc_slot(7)   # lhflx
            ... feed_cmip_accumulators_native(..., rlut=rlut, rsut=rsut,
                rsdt=rsdt, hfss=hfss, hfls=hfls)
```

### 7. coupler/driver/diagnostics.py — feed_cmip_accumulators_native +5 kwargs
```python
    def feed_cmip_accumulators_native(self, ..., rlut=None, rsut=None,
                                      rsdt=None, hfss=None, hfls=None):
        ...
        rlut_np = None if rlut is None else np.asarray(rlut, dtype=_f64)
        # ... same for rsut/rsdt/hfss/hfls
        # PHASE 1 shape checks, each (ncol,):
        #   ("rlut", rlut_np, (_ncol,)), ... ("hfls", hfls_np, (_ncol,)),
        if have_spatial:
            evspsbl_np = None if hfls_np is None else hfls_np / _c.L_v
            for _name, _src in (('tas',...),('ps',...),('pr',...),('psl',...),
                ('rlut', rlut_np),('rsut', rsut_np),('rsdt', rsdt_np),
                ('hfss', hfss_np),('hfls', hfls_np),('evspsbl', evspsbl_np)):
                if _src is None: continue
                r = self._regrid_to_latlon_2d(_src)
                if r is not None: fields_2d[_name] = r
        # zonal path (z2d) NOT updated with the 5 fields.
```

### 8. tests/unit/test_mpas_cmor_flux_feed.py — 5 tests
uniform-field IDW-exact regrid for all 5 + evspsbl derivation; wrong-shape
transactional raise; backward-compat absence; packer None-default; packer
values+units.

## Reference code (existing, for cross-checking)

### COMPILED-path TOA assembly — physics_pipeline.py:2214-2228
```python
        sw_down_sfc = ad.unflatten_2d(rad_out.sw_flux_down[:, -1])
        sw_net_sfc = sw_down_sfc * (1.0 - albedo)
        lw_net_sfc = ad.unflatten_2d(rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1])
        sw_up_toa = ad.unflatten_2d(rad_out.sw_flux_up[:, 0])
        lw_up_toa = ad.unflatten_2d(rad_out.lw_flux_up[:, 0])
        # rsdt = prescribed TOA incident SW the solver was given (#620), not the
        # quadratically clamped top-halo SW flux (rad_out.sw_flux_down[:, 0],
        # ~15% low).  Halo fallback keeps a value for any path ... toa_insolation=None.
        sw_down_toa = ad.unflatten_2d(
            rad_out.toa_insolation if rad_out.toa_insolation is not None
            else rad_out.sw_flux_down[:, 0])
```

### RadiationOutput.toa_insolation doc — radiation/output.py:37-52
> Prescribed TOA incident SW the solver was given (ncol,) [W/m^2] (#620). ...
> Used for the CMOR `rsdt` diagnostic so it reflects the true TOA insolation
> rather than the quadratically clamped top-halo SW flux. `None` for paths that
> do not set it.
Fluxes are at interface (half) levels, shape (ncol, nlev+1). TOA = index 0,
surface = index -1 (confirmed by idealized/radiative_convective_column.py:128
using [:,0] for TOA net, and integration.py using [:,-1] for surface).

### Top-halo extrapolation BUG B — two_stream.py:172-214 (_replace_top_flux)
> The quadratic Lagrange stencil ... makes the raw extrapolation OVERSHOOT —
> producing super-physical TOA shortwave (the `[:, 0]` face is read straight into
> `rsdt`/`rsut`) or driving the upwelling longwave below zero (negative OLR).
> This was BUG B: a 30-day rrtmgp coupled run inflated `rsdt` ~2x and turned
> `rlut` negative at day ~15-20 (cells to 1121 W/m2 / -52 W/m2). Fix: RANGE-LIMIT
> the extrapolated top-halo face to the local range of the three interior faces.

### Turbulence surface-flux sign — surface_layer.py:47-48
> shflx = rho c_pd Ch |V| (T_sfc - T) and lhflx = rho L_v Ch |V| (q_sfc - q_v)
> are POSITIVE UPWARD (out of the surface).
Compiled path derives evap = lhflx / L_v (physics_pipeline.py:1466) — matches.

### SECOND _sfc_diag producer — MPI-voronoi path — voronoi_mpi.py:1036,1053-1058,1135-1138
```python
        sfc_diag = (None, None, None)                       # line 1036
        ...
            _sw_sfc = getattr(_pt, "sw_net_sfc", None)
            _lw_sfc = getattr(_pt, "lw_net_sfc", None)
            _pr_sfc = getattr(_pt, "precip", None)
            if (_sw_sfc is not None or _lw_sfc is not None or _pr_sfc is not None):
                sfc_diag = (_sw_sfc, _lw_sfc, _pr_sfc)      # 3-tuple, NO extras
        return ..., sfc_diag
    # in _step_carry:
        _prev = getattr(model, "_sfc_diag", None) or (None, None, None)
        model._sfc_diag = tuple(new if new is not None else old
                               for new, old in zip(_sfc, _prev))   # unpadded zip
```
This distributed/coupled MPAS producer was NOT updated: it emits a 3-slot
sfc_diag (no TOA/turbulent extras), and merges with a hardcoded 3-tuple prev via
an unpadded `zip`.

## My static-analysis candidate concerns (adjudicate + find more)

**C1 (rsdt source divergence — likely CONFIRMED).** The lean rrtmgp branch
(integration.py:1202) sets `sw_down_toa = rad_out.sw_flux_down[:, 0]`. The
compiled path (physics_pipeline.py:2225) uses `rad_out.toa_insolation` with the
halo flux only as a fallback, and #620 + two_stream.py document `sw_flux_down[:, 0]`
as "~15% low" (historically ~2x high before the range-limit). So rsdt on the
lean lane diverges from the compiled path and reintroduces the #620 artifact.
`sw_up_toa`/`lw_up_toa` (rsut/rlut) DO use `[:, 0]`, matching the compiled path
lines 2219-2220 — those look correct. Is C1 a real bug?

**C2 (MPI-voronoi producer not updated — coverage gap).** The serial producer
(primitive_eq_mpas.py) emits 8 slots; the MPI-voronoi producer
(voronoi_mpi.py:1058) emits 3. So the CMOR feed is silently absent on the
distributed/coupled MPAS path (consumer reads slots 3-7 → None). The
contract-growth padding was only added to the serial producer. Bug or acceptable
scoping (campaign is single-GPU serial)?

**C3 (zonal omission).** feed_cmip_accumulators_native adds the 5 fields to the
SPATIAL fields_2d only; the zonal z2d dict (diagnostics.py:1550-1559) is not
updated. Deliberate (ClimateEval reads Amon spatial) or a gap?

**C4 (evspsbl approximation).** evspsbl = hfls / L_v with no L_s sublimation
split over ice. Documented; matches the bulk scheme's L_v-only lhflx. Acceptable?

**C5 (restart staleness).** `_sfc_diag` is a Python instance attribute, not
persisted; after restart the first feeds have None TOA until the first radiation
solve. Same as sw_net/lw_net. Acceptable?

**C6 — check independently:** tuple slot ORDER consistency producer↔consumer
(off-by-one silently swaps rlut/rsut/rsdt); dtype; byte-identical default when
radiation/turbulence off; Field-vs-raw-array handling at each boundary; any
OTHER consumer of `_accumulate` or `_sfc_diag`; differentiability of the new
`.reshape`/`.astype`/`getattr` paths; shape-check transactionality.

Report each finding as CONFIRMED / FALSE-POSITIVE / AMBIGUOUS with file:line and
a one-line fix.
