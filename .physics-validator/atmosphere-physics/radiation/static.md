# Radiation — static analysis

## A. Confirmed correctness (gray.py)

1. **LW two-stream layer recursion**: `F_below = t_k * F_above + (1 - t_k) * B_k` — standard Schwarzschild solution for hemispheric mean. ✓
2. **Optical depth**: `dtau_dry_k = tau_at_half[k+1] - tau_at_half[k]`, then floored at 0. `dtau_moist = tau_moist_coeff * q_v * dp / g` — correct dimensional analysis (`[m²/kg] × [kg/kg] × [Pa] / [m/s²] = dimensionless`). ✓ The "prior code divided by p_s" comment notes a previous bug already fixed.
3. **Surface emission**: `F_up_sfc = ε_sfc σ T_sfc^4 + (1-ε_sfc) F_down_sfc` — correct combined surface emission + reflection. ✓
4. **Heating rate**: `dT/dt = (g/c_pd) * d(F_net_up)/dp`. Sign: when more F_net_up exits the bottom than the top, the layer gains energy ⇒ positive heating. ✓ Standard form.
5. **SW Beer-Lambert** `tau_sw(σ) = tau_sw_0 * σ^exponent`; `F_down = S * exp(-tau_sw)`. ✓ Frierson/Isca convention: reflected upward SW escapes directly to TOA without further atmospheric absorption.
6. **Solar declination / zenith**: standard formulas, clipped to `[-1, 1]`. ✓

## B. Issues (RANKED)

### B1. **`solar.py` defaults `S_0=1360.0` but `constants.S_0=1361.0`** [LOW BUT REAL]
`solar.py:89, 179`:
```python
def daily_mean_insolation(lat, day_of_year, S_0=1360.0, ...):
def perpetual_equinox_insolation(lat, S_0=1360.0):
```
But `constants.S_0 = 1361.0` (Kopp & Lean 2011 modern value).
Practically: `radiation/integration.py` and `RadiationConfig` correctly use 1361.0, so the defaults are not normally invoked. However:
- Direct callers of these helpers (e.g., diagnostics, plotting) will silently get the wrong value.
- Violates CLAUDE.md "no hardcoded physical constants in function signatures".

**Fix candidate**: change defaults to `constants.S_0` (forbidden style — should remove default entirely, requiring explicit pass).

### B2. **`obliquity=23.45` hardcoded** [LOW]
Default arg in `solar_declination`, `cos_zenith_angle`, `daily_mean_insolation`, `daylight_fraction`. Not in `constants.py`. Should be `constants.obliquity_deg = 23.45`. Also Spencer's formula uses `day_of_year - 80` ⇒ vernal equinox at day 80; canonical Julian dates are slightly different.

### B3. **`solar_declination` Spencer-style approximation neglects eccentricity** [INFO]
`solar.py:21-41`: simple sinusoidal model. Real Earth declination has higher harmonics; this is a ±0.5° error year-round. For climate-mean diagnostics this is fine; for tightly-coupled real-time forecasting it's known limitation. Documented as "simplified Spencer formula approximation".

### B4. **Gray LW emissivity formula uses `1 - exp(-D*dtau)`** [VERIFIED]
`gray.py:140-141`: `transmittance = exp(-D * dtau)`, `emissivity = 1 - transmittance`. The diffusivity factor `D = config.lw_diff_factor` (typically 1.66) accounts for hemispheric vs cosine integration. ✓ Standard.

### B5. **Gray LW boundary `F_down(TOA) = 0`** [VERIFIED]
Set via `F_down_toa = jnp.zeros(ncol, ...)`. ✓ correct.

### B6. **Gray SW: `sw_up` uniform** [VERIFIED] 
`gray.py:254`: `sw_up = jnp.broadcast_to(F_up_sfc[:, None], p_half.shape)`. Frierson/Isca convention. The implication is that surface-reflected SW does not contribute to in-atmospheric heating (no upward-stream absorption). Documented.

### B7. **`_compute_heating_rate` uses `dp = p_half[1:] - p_half[:-1]`** [VERIFIED]
Then `dF / dp` with positive `dp` (TOA-first ordering). Sign matches the layer indexing where flux divergence into a layer (more leaving the bottom than the top) gives positive heating. ✓

### B8. **RRTMGP subdir not audited in this pass** [DEFERRED]
6500 LOC across optics/rte/lookup pathways. The subpackage has its own constants module that re-exports from `legoesm.constants`. Should be a dedicated audit cycle.

## C. Top issues

* **B1** (S_0 default 1360 vs constants.S_0 = 1361) — concrete defect, would cause silent wrong forcing if defaults are ever invoked in diagnostics/plotting. Fix recommended.
* **B2** (obliquity not in constants.py) — minor style issue.
* **B8** — RRTMGP requires its own audit cycle.
