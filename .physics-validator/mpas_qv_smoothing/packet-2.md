# Adversarial review packet — MPAS q_v smoothing (round 2, post-fix)

You are an independent adversarial reviewer. In round 1 you raised two BLOCKERS
on this change; both are CONFIRMED and now FIXED. Re-review the fix. Find any
REMAINING bug, any NEW bug the fix introduced, and challenge the new design
choice. Cite line numbers. If clean, say so and justify per item.

## Round-1 outcome (your findings, all confirmed)

- **B2 (CRITICAL, confirmed numerically):** MPAS defaults to
  `vertical_coord="hybrid"`; `make_hybrid_levels(40, p_top=200, stretching=2.0)`
  has surface-layer `dA=-0.027<0`, so `dp = dA*p_ref + dB*p_s` crosses zero at
  **p_s = 662.7 hPa** and is negative below (Tibet 600 hPa: min dp = -256 Pa).
  The mass-weighted smoother divided by this dp -> Inf/NaN / negative q_v over
  high terrain; the dycore's next-step `maximum(q_v,0)` then fabricates water.
- **B1 (confirmed):** `dp_edge/dp_cell` is unbounded (148x sea/Tibet); a
  setup-time geometry-only CFL guard cannot bound the mass-weighted operator.
- The "same convention as diagnostics" justification was false (smoother used
  hybrid dp; the hard-sat drain uses pure-sigma pressure).

## The fix (root cause, mirrors the FV lane)

The FV lane's own q_v smoother `_apply_qv_smoothing`
(compiled_segments.py:1368) is a **PLAIN scalar hyperdiffusion + `max(q,0)`
floor**; water conservation is handled separately by `fix_moisture`, NOT by the
smoothing being mass-conservative. The MPAS smoother now mirrors this: **PLAIN
SCVT del2 (no dp) + positivity floor.** This is coordinate-agnostic (no dp
division -> no NaN), makes the geometry CFL guard EXACT (plain-del2
monotonicity factor is geometry-only), guarantees positivity (convex combo
under the guard; the floor is a strict no-op insurance), and conserves the
per-level `sum_c A_c q_c` integral exactly. It drops the (unachievable, and
non-FV) "exact column water MASS" guarantee.

### New driver apply block (model_driver.py, eager step loop)

```python
if _qv_smooth_nu > 0.0:
    _trc_sm = self.state.tracers
    _qv_sm = _trc_sm["q_v"].data
    _lap_sm = scalar_del2_cell_3d(_qv_sm, self.grid)          # PLAIN, no dp
    _qv_new_sm = jnp.maximum(
        _qv_sm + DT * _qv_smooth_nu * _lap_sm.astype(_qv_sm.dtype), 0.0)
    _new_trc_sm = dict(_trc_sm)
    _new_trc_sm["q_v"] = _trc_sm["q_v"].replace(data=_qv_new_sm)
    self.state = self.state._replace(tracers=_new_trc_sm)
```

### New driver setup guard (model_driver.py)

```python
_qv_smooth_nu = float(getattr(cfg, "mpas_qv_smooth_del2_m2s", 0.0))
if _qv_smooth_nu > 0.0:
    from legoesm.core.operators_voronoi import (
        scalar_del2_cell_3d, scalar_del2_cell_cfl_factor,
    )
    if self._voronoi_layout is not None:
        raise ValueError("... distributed Voronoi (MPI) not wired ...")
    if self.state.tracers is None or "q_v" not in self.state.tracers:
        raise ValueError("... needs a q_v tracer (dry configuration) ...")
    # q + nu*dt*lap is a convex combination iff nu*dt*g_max <= 1 with
    # g_c = (1/A_c) sum_e dvEdge_e/dcEdge_e.  Enforce <= 0.5 (2x safety;
    # keeps the q>=0 floor a strict no-op so sum_c A_c q_c is conserved).
    # EXACT for the plain form actually applied — no dp-ratio guess.
    _g_max = scalar_del2_cell_cfl_factor(self.grid)
    _cfl = _qv_smooth_nu * DT * _g_max
    if _cfl > 0.5:
        raise ValueError(f"... nu*dt*g_max = {_cfl:.3f} > 0.5 ...")
    logger.info("  MPAS q_v del2 smoothing ON: nu=%.3g m^2/s ...", _qv_smooth_nu, _cfl)
```

### New shared helper (operators_voronoi.py)

```python
def scalar_del2_cell_cfl_factor(mesh):
    """g_max for the plain-del2 explicit stability bound: q + nu*dt*lap is a
    convex combination iff nu*dt*g_max <= 1, g_c = (1/A_c) sum_e dvEdge/dcEdge.
    Geometry-only; shared by the driver guard and the test (no drift)."""
    import numpy as np
    eoc = np.asarray(mesh.edgesOnCell); mask = eoc >= 0
    safe = np.maximum(eoc, 0)
    dv = np.asarray(mesh.dvEdge)[safe]; dc = np.asarray(mesh.dcEdge)[safe]
    g_cell = np.sum(np.where(mask, dv / np.maximum(dc, 1e-30), 0.0),
                    axis=0) / np.asarray(mesh.areaCell)
    return float(np.max(g_cell))
```

The operator `scalar_del2_cell_3d(q, mesh, dp_cell_3d=None)` is UNCHANGED (still
supports the mass-weighted branch as a general capability for strictly-positive
dp; its docstring now documents the dp>0 requirement and that the MPAS smoother
uses the plain form). config field, validate_strict, and CLI are unchanged from
round 1 except comment/help text now say "plain SCVT del2 + q>=0 floor (mirrors
the FV lane)".

## New/updated tests (18 pass, `JAX_ENABLE_X64=1`)

- Plain-form conservation asserted PER LEVEL (`sum_c A_c lap_c = 0` each k).
- `test_plain_step_is_monotone_and_positive`: driver path (plain), nu at
  `0.5/(dt*g_max)`, convex-combination bounds + q>=0.
- `test_plain_step_conserves_area_integral`: `max(q + nu*dt*plain_lap, 0)`
  conserves per-level `sum_c A_c q_c` to rel 1e-13.
- `test_mass_weighted_conserves_water_mass_positive_dp`: operator's dp branch
  still exact for POSITIVE dp (general-capability test).
- `test_hybrid_low_ps_smoother_finite_and_positive` (REGRESSION for B2): builds
  the real `make_hybrid_levels(8,...)`, p_s spanning 350..1013 hPa so
  `min(dp) < 0` (asserted, non-vacuous); the driver PLAIN path stays finite and
  >= 0; the mass-weighted operator on that hybrid dp is non-finite and/or drives
  q_v negative (asserted).
- `test_cfl_factor_positive_and_matches_manual`: the shared helper equals a
  manual recompute (rel 1e-13).
- checkerboard variance-decay and autodiff now on the PLAIN form.
- validate_strict tests unchanged (accept MPAS, refuse cd-grid, bounds).

## Explicit ask

1. Are B1/B2 fully resolved by the plain form? Any residual path where the
   smoother can still NaN or produce negative q_v (e.g. non-finite input q,
   `DT*nu*lap` overflow in fp32, the `.astype` cast, tracer dtype)?
2. Is dropping exact water-mass conservation for `sum_c A_c q_c` (mixing-ratio
   integral) + relying on the separate moisture fixer acceptable and consistent
   with the FV lane? Any physics objection (systematic moisture bias from plain
   del2 across a p_s gradient) that matters for a gentle opt-in speckle smoother?
3. Is the guard `<= 0.5` correct/sufficient for the PLAIN form (exact bound is
   `<= 1`)? Is the shared helper's masking/edge handling correct?
4. Is the regression test non-vacuous and are the other tolerances non-vacuous
   (would a wrong operator pass)?
5. Any NEW bug the fix introduced (scope, import, dtype, shape, the retained
   mass-weighted branch now being "test-only" for prod)?
6. The field is still inserted mid-`ExperimentConfig` NamedTuple (positional ABI
   note from round 1). Keep or move to the end?
