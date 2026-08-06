# Adversarial review packet 3 — prognostic sea-ice skin (round-2 fixes)

Independent adversarial re-review. Round 2 (review-2.md) confirmed #1,#3,#4,#5
fixed and raised 3 remaining items; all 3 are now fixed by ONE change plus a
save-path fallback. Verify correctness + completeness, and find any NEW bug.
Cite file:line. Be skeptical.

## Root cause of round-2 findings 1 & 2 (same fix)
Both stem from: the skin advanced per-step, but the physics still consumed the
DAILY-HELD `T_sfc` anchor. So the feedback stayed daily-lagged (finding 2) and a
mid-day restart re-blended the anchor from the already-advanced restored skin
while the straight run did not until midnight (finding 1).

**Fix: re-blend `T_sfc` EVERY model step** from the cached daily SST/SIC against
the current skin. `_compute_T_sfc` was split so the loop can re-anchor without
re-sampling SST:
```python
            def _blend_T_sfc(_sst, _sic):
                _ice_component = self._ice_T_skin if _ice_skin_on else _T_ice
                _ts = blend_surface_temperature(
                    jnp.asarray(_sst), jnp.asarray(_sic), _ice_component).reshape(-1)
                if _lapse_z is not None:
                    _ts = land_lapse_adjusted_surface_temperature(
                        _ts, _f_land_cells.astype(_ts.dtype),
                        _lapse_z.astype(_ts.dtype), _land_lapse_K_m)
                return _ts
            def _compute_T_sfc(day):
                _sst, _sic = self.get_sst_sic(day)
                return _blend_T_sfc(jnp.asarray(_sst), jnp.asarray(_sic))
```
Daily boundary caches SST+SIC and blends from the cache:
```python
                        if _ice_skin_on:
                            _sst_now, _sic_now = self.get_sst_sic(_force_day_canonical)
                            _ice_sst_cur = jnp.asarray(_sst_now).reshape(-1)
                            _ice_sic_cur = jnp.asarray(_sic_now).reshape(-1)
                            _forcing_daily["T_sfc"] = _blend_T_sfc(_ice_sst_cur, _ice_sic_cur)
                        else:
                            _forcing_daily["T_sfc"] = _compute_T_sfc(_force_day_canonical)
```
Per-step, right AFTER the skin advance, re-anchor for the NEXT step (the loop
rebuilds `_forcing = dict(_forcing_daily)` every step, so this propagates):
```python
                        self._ice_T_skin = prognostic_ice_skin_temperature(
                            self._ice_T_skin, _f_net, _ice_sic_cur, dt_s=DT, h_ice_m=_h_ice)
                    # re-anchor next step's T_sfc against the advanced skin
                    _forcing_daily["T_sfc"] = _blend_T_sfc(_ice_sst_cur, _ice_sic_cur)
```
Why this fixes both:
- **Finding 2 (stability):** physics now consumes the freshly advanced skin each
  step -> the feedback IS per-step -> `r*lambda = DT*lambda/C << 1` (DT~240 s,
  C=1.93e6) genuinely holds. DT is the CFL-bounded dycore step, always small.
- **Finding 1 (mid-day restart):** at a mid-day resume the daily boundary fires
  (`_last_force_day=None`), samples SST/SIC (deterministic per canonical day =
  the straight run's cached values) and blends against the RESTORED skin
  (= the straight run's skin at that step, checkpointed). So `T_sfc` matches the
  straight run's `_blend(cached, skin_after_prev_step)`. No anchor branch; the
  only residual is the model's pre-existing radiation-first-step recompute, which
  is state-wide, not skin-specific.

Non-feature path unchanged: `_compute_T_sfc` for `_ice_skin_on=False` is
byte-identical to before (same blend+lapse); the per-step re-anchor is gated on
`_ice_skin_on`.

## Round-2 finding 3 (load->save-without-run strips the skin)
Save falls back to the staged value when the live field is unadopted:
```python
            if getattr(self.config, "mpas_ice_skin_prognostic", False):
                _skin = getattr(self, "_ice_T_skin", None)
                if _skin is None and isinstance(self._carry_aux, dict):
                    _skin = self._carry_aux.get("ice_T_skin")   # loaded, not yet adopted
                if _skin is not None:
                    _save["ice_T_skin"] = np.asarray(_skin)
```
Still config-gated (feature-on->off reuse cannot leak, codex-1 #6).

## Helper docstring (stability, restated)
Now states the per-step amplification `|1-r*lambda|/(1+r*g)` holds ONLY if the
caller refreshes the surface anchor at the SAME cadence as the advance (the MPAS
driver re-blends T_sfc every step); a held anchor reverts to the coarse-cadence
thresholds and the melt cap only bounds, not cures, a cap/floor oscillation.

## New test (codex-2 asked for a feature-on restart test)
`test_end_to_end_mpas_ice_skin_wiring_and_checkpoint` (subprocess, ~40 s):
real `run_amip.py` MPAS lane, feature on, analytical, 2 days, `--checkpoint-days
1`; asserts exit 0 / no NaN / "ice skin ON"; the day-2 checkpoint carries a
finite `ice_T_skin` (642,) in `[floor, T_freeze]`; then RESTARTS from the day-1
checkpoint (exercising load->stage->adopt) and reproduces the straight run's
skin. Analytical SST/SIC is ice-free (sic=0, verified: SST min 273.09 > 271.35),
so the skin correctly snaps to `T_freeze_ocean` everywhere — this asserts the
open-water path + checkpoint/restart wiring, not the cooling magnitude (covered
by the helper equilibrium tests). Full mid-day-split bit-diff is deferred:
`--checkpoint-days` is integer (no mid-day checkpoint via CLI); consistency is
guaranteed by construction + `test_restart_split_invariance`.

## Verification (srun, CPU x64)
- **256 passed**: the 5 baseline files incl. `test_mpas_ice_skin.py` (21 tests:
  20 unit + the subprocess integration test).
- Manual feature-on run: 36.6 s, "ice skin ON" logged, `ice_T_skin` persisted at
  the correct open-water value, restart round-trips.

## Ask
1. Is the per-step re-anchor correct and complete for findings 1 & 2? Any
   remaining path where the physics consumes a stale anchor, or a straight-vs-
   restart branch beyond the pre-existing radiation recompute?
2. Finding 3: any remaining strip path?
3. Did the `_blend_T_sfc` split change the non-feature (byte-identical) path or
   the shape guard `_ts0 = _compute_T_sfc(START_DAY)`?
4. Any NEW sign/shape/dtype/None/gradient bug, or perf regression from the
   per-step re-blend (eager blend+lapse on nCells each step)?
5. Anything only partially fixed.

If nothing substantive remains, say so and justify each round-2 item resolved.
