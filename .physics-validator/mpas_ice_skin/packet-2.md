# Adversarial review packet 2 — prognostic sea-ice skin (round-1 fixes)

You are an independent adversarial physics + software reviewer. This is a
RE-REVIEW. In round 1 you found 7 confirmed bugs + limitations (packet-1.md,
review-1.md, both in this directory). I accepted ALL 7 and fixed them. Verify
each fix is correct AND complete, and find any NEW bug the fixes introduced.
Cite file:line. Be skeptical.

## Root-cause consolidation
Four of your seven findings (1 diurnal snapshot, 2 restart drop, 5 SIC timing,
7 conditional stability) shared ONE root cause: the skin was advanced ONCE per
forcing day with `dt_s=86400` off a single instantaneous `_sfc_diag` snapshot.
Fix: **advance the skin every MODEL step (`dt_s=DT`)** with the fresh per-step
fluxes. That one change resolves all four:
- 1: each step uses its own flux -> diurnal cycle resolved (at the radiation-
  update cadence for SW; turbulent fluxes refresh every step).
- 2: the checkpointed `ice_T_skin` is always advanced through the last step, so
  no pending daily advance exists to drop or double-count on restart.
- 5: the open->ice transition starts from the freezing point (the `where(sic>0)`
  snap held it there while open) and cools with the current step's flux.
- 7: `r*lambda = DT*lambda/C << 1` (DT~240 s, C=1.93e6), so the lagged feedback
  amplification `|1 - r*lambda|/(1+r*g)` ~ 1 - r*lambda < 1 for ANY realistic
  lambda — stable and monotone.

## Fixes (file:line, current working tree)

### Finding 4 — melt cap temperature (`surface_utils.py`)
New param `T_melt_surface_K: float = constants.T_freeze` (0 C surface melt);
the basal conduction target and open-water snap keep `T_freeze_K =
constants.T_freeze_ocean` (seawater freezing). Mirrors `ice/sea_ice.py:1066,1083`
(`T_base = T_freeze_ocean`, clip to `T_melt_surface = constants.T_freeze`).
```python
    T_new = jnp.clip(T_new, _ICE_SKIN_FLOOR_K, T_melt_surface_K)   # was T_freeze_K
    return jnp.where(sic > 0.0, T_new, T_freeze_K)                 # snap unchanged
```
Docstring now: distinct base vs surface BCs; conditional-stability of the
coupled system; partial-SIC single-tile closure limitation; global-h_ice
limitation; differentiability caveat (interior smooth `r/(1+r*g)`, cap/mask
dead-gradient is physically correct).

### Findings 1/2/5/7 — per-step advance (`model_driver.py`)
Daily boundary now only SAMPLES the day's SIC (no advance):
```python
                        if _ice_skin_on:
                            _, _sic_now = self.get_sst_sic(_force_day_canonical)
                            _ice_sic_cur = jnp.asarray(_sic_now).reshape(-1)
                        _forcing_daily["T_sfc"] = _compute_T_sfc(_force_day_canonical)
```
`_ice_sic_cur = None` initialised before the loop; `_last_force_day = None`
(line 6369) so the step-0 boundary sets `_ice_sic_cur` BEFORE the first advance.
Per-step advance, right after the eager `self.model.step(...)` (serial else
branch; the feature refuses the MPI/voronoi lane):
```python
                _phys_state = self.model._phys_state
                if _ice_skin_on and _ice_sic_cur is not None:
                    _sd = getattr(self.model, "_sfc_diag", None)
                    _swn = (_sd[0].data if (_sd is not None and len(_sd) > 0
                                            and _sd[0] is not None) else None)
                    _lwn = (_sd[1].data if (_sd is not None and len(_sd) > 1
                                            and _sd[1] is not None) else None)
                    if _swn is not None and _lwn is not None:
                        _f_net = (jnp.asarray(_swn).reshape(-1)
                                  + jnp.asarray(_lwn).reshape(-1))
                        if len(_sd) > 6 and _sd[6] is not None:
                            _f_net = _f_net - jnp.asarray(_sd[6].data).reshape(-1)
                        if len(_sd) > 7 and _sd[7] is not None:
                            _f_net = _f_net - jnp.asarray(_sd[7].data).reshape(-1)
                        self._ice_T_skin = prognostic_ice_skin_temperature(
                            self._ice_T_skin, _f_net, _ice_sic_cur,
                            dt_s=DT, h_ice_m=_h_ice)
```
`prognostic_ice_skin_temperature` is imported at setup under the same
`if _ice_skin_on:` (line 6198), so the name is bound whenever the advance runs.

### Finding 3 — CMOR tas uses the skin (`model_driver.py`, ~5560)
```python
                    _tas_ice = getattr(self.config, "T_ice", None)
                    if (getattr(self.config, "mpas_ice_skin_prognostic", False)
                            and getattr(self, "_ice_T_skin", None) is not None):
                        _tas_ice = self._ice_T_skin
                    tas = diag._tas_2m(state, q_v, _sst, _sic, _tas_ice,
                                       u_low=u_east[..., -1], v_low=v_north[..., -1])
```
`_tas_2m` feeds `T_ice` into `blend_surface_temperature` (broadcasts on an array).

### Finding 6 — checkpoint save laundering (`model_driver.py`, ~4260)
```python
            if (getattr(self.config, "mpas_ice_skin_prognostic", False)
                    and getattr(self, "_ice_T_skin", None) is not None):
                _save["ice_T_skin"] = np.asarray(self._ice_T_skin)
```
Load side unchanged (clears `self._ice_T_skin=None`, stages into `_carry_aux`,
stale-drops). A feature-on->off reused driver now cannot leak a stale skin.

## Verification (this reviewer ran these; CPU x64, srun on the interactive alloc)
- Full suite green: **255 passed** — `test_mpas_ice_skin.py` (now 20 tests, incl.
  `test_melt_cap_at_surface_melting_point`, `test_equilibrates_in_the_melt_band_
  above_seawater_freezing`, `test_restart_split_invariance`), `test_run_amip_cli.py`,
  `test_mpas_land_boundary.py`, `test_mpas_qv_smoothing.py`, `test_mpas_cmor_flux_feed.py`.
- Probe (`scripts/tmp/_probe_ice_skin.py`) still green: FD Jacobian
  max|autodiff-FD|=8.2e-11, diag=r/(1+rg); coupled feedback monotone for
  lambda=5/10/20 (amp 0.10-0.74); lambda=47/60 caps at the new 273.15 (was
  271.35) confirming the melt-cap fix; jit/vmap bit-identical.

## Open questions for you
1. Per-step advance: any correctness/perf/retrace concern with an eager helper
   call every model step (small elementwise on nCells; outside the compiled
   step)? The anchor T_sfc is still re-blended DAILY (skin lags the anchor by
   <=1 day) — acceptable lagged coupling?
2. Restart bit-identity: the first step of each link re-solves radiation while
   the straight run may have HELD it there, so the skin inherits the model's
   pre-existing ~1e-8 first-step restart non-determinism. Is that acceptable
   (i.e. not a NEW break introduced by the skin), or do you want the skin's
   restart tolerance pinned separately?
3. Did any fix introduce a NEW sign error, shape/dtype mismatch, dead gradient,
   or unguarded None? Specifically the `_tas_ice` array path and the per-step
   `_sfc_diag` slot access.
4. Anything from round 1 you consider only PARTIALLY fixed.

If you find nothing substantive, say so and justify why each round-1 finding is
now resolved.
