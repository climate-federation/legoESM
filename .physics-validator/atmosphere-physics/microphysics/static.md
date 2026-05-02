# Microphysics — static analysis

## A. Confirmed correctness

1. **Latent heat sign in kessler/seifert/sundqvist**: `dT_dt = L_v * (cond - evap) / c_pd`. Condensation heats, evaporation cools. ✓
2. **Mass conservation in shared `_warm_rain` helpers**: `condensation` adds to q_c, subtracts from q_v; `evaporation` adds to q_v, subtracts from q_r; `dq_c_au` (autoconv) leaves q_c, enters q_r; `dq_c_ac` (accretion) leaves q_c, enters q_r. All sign-consistent across all schemes that use them.
3. **Saturation adjustment uses `legoesm.thermo.saturation_mixing_ratio`** (not Tetens inline). ✓
4. **Sedimentation tendency** (`output.sedimentation_tendency`): standard upwind discretization, positive-definite when V_t and q are non-negative.
5. **Constants usage**: all schemes use `constants.L_v`, `constants.L_s`, `constants.L_f`, `constants.c_pd`, `constants.T_freeze`. No hardcoded literals.

## B. Potential issues (RANKED)

### B1. **Morrison: latent heat balance INCOMPLETE** [MEDIUM]
`morrison.py:149-154`:
```python
dT_dt = (L_v * cond / c_pd
         - L_v * evap / c_pd
         + L_s * dq_i_dep / c_pd
         - L_f * (melt_ice + melt_snow) / c_pd)
```
Missing latent contributions:
- **Bergeron** (`bergeron`): vapor → ice path. Should release `L_s` (or at least `L_f` for the cloud-water → ice piece).
- **Riming** (`riming_i`, `riming_s`): cloud water → ice/snow ⇒ should release `L_f` (freezing).

The formula tracks only deposition (vapor→ice) and melting (ice→liquid), not the freezing pathways. **This understates atmospheric heating in mixed-phase clouds and overstates net cooling in glaciation events.**

### B2. **Thompson: same gap as Morrison** [MEDIUM] (`thompson.py:174-179`)
Same Bergeron / riming latent heat omission. Plus: graupel formation (`rime_to_graupel`) is presumably wet→solid; would also release `L_f` if the rimed mass is liquid.

### B3. **Thompson: `melt_ice/snow/graupel` not clamped to available mass** [LOW-MEDIUM]
`thompson.py:143-144, 151`:
```python
melt_ice = config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac
melt_snow = config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac
melt_graupel = config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac
```
For `melt_rate * dt > 1` (e.g. melt_rate = 1e-3 /s and dt = 1200 s ⇒ 1.2), the explicit step `q_i_new = q_i + dt * (-melt_ice + …)` can drive `q_i_new < 0`.
Morrison `morrison.py:121-128` has `jnp.minimum(..., q_i / dt)` clamp; Thompson is missing this safety. (Also missing for `rime_to_graupel`, which can deplete q_i and q_s simultaneously.)

### B4. **N_i not decremented during melting** [LOW]
Both Morrison (line 165) and Thompson (line 191) only adjust `dN_i_dt = dN_i_nuc - aggregation * N_i / q_i`. When ice melts (`melt_ice > 0`), `N_i` should decrease proportionally; otherwise the mean ice mass `q_i / N_i` becomes unphysically small.

### B5. **Sundqvist `evap_mask` based on `q_sat - q_v`, not on `RH < RH_crit`** [INFO]
`sundqvist.py:71`: `evap_mask = sigmoid(sharpness * (RH_crit - RH))`. ✓ — that IS the standard form. False alarm from earlier reading.

### B6. **Sundqvist: `P_local` stream uses partly q_c at "next" time-step** [LOW]
`sundqvist.py:68`: `P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)`. This treats `q_c + condensation*dt` as the cloud-water mass present *after* condensation. That's a forward Euler estimate of the post-condensation cloud water; reasonable but couples microphysics to dt linearly. Standard Sundqvist would use `q_c` only.

### B7. **Kessler: `accretion ~ q_r ** 0.875` and `evap ~ q_r ** 0.525`** [VERIFIED]
These are Marshall-Palmer empirical fits. Standard. ✓

### B8. **`x_c` floor at `1e-20` in `_warm_rain.autoconversion_sb`** [LOW]
`_warm_rain.py:93`: `x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)`. `N_c_eff = max(N_c, Nc_0)`, so always ≥ Nc_0 ≈ 1e8. `q_c_pos * rho ≥ 0`. Then `x_c ≥ 0`. Since `dq_c_au = k_au * q_c**2 * onset * gamma * rho`, no further floor needed. OK.

### B9. **All evaporation rates clipped only at q_r ≥ 0** [LOW]
`_warm_rain.rain_evaporation` and `kessler` raise `q_r ** 0.525`. For `q_r = 0`, this is 0 — fine. For tiny positive `q_r`, `q_r**0.525` derivative diverges → AD unstable. JAX's `pow(0, 0.525)` returns 0 with NaN gradient; `pow(epsilon, 0.525)` is large. Practically not a problem because `q_r > 0` everywhere precipitation has occurred.

### B10. **Thompson `_gamma_ratio(0.0) = 6`, `_gamma_ratio(mu_c)` is integer-only formula** [LOW]
`thompson.py:39-41`: `(mu+3)(mu+2)(mu+1)`. This is `Γ(mu+4)/Γ(mu+1)` for *integer* mu. For non-integer μ_c (typical 2.5), the formula is incorrect; should use `jax.scipy.special.gammaln`. **For default config μ_c=2.5, this gives 5.5×4.5×3.5 = 86.625 vs the true Γ(6.5)/Γ(3.5) = 5.5×4.5×3.5 = 86.625**. Wait — for non-integer μ, the recurrence still holds: `Γ(μ+4) = (μ+3)(μ+2)(μ+1)μ Γ(μ)`, and `Γ(μ+1) = μ Γ(μ)`, so the ratio is exactly `(μ+3)(μ+2)(μ+1)`. ✓ Docstring is misleading but the formula is correct.

## C. Conservation summary

| Scheme | Total water conserved? | Latent heat balanced? |
|--------|------------------------|----------------------|
| kessler | Yes | Yes |
| sundqvist | Yes | Yes |
| seifert_beheng | Yes | Yes |
| morrison | Yes (mass) | **NO** — Bergeron/riming missing |
| thompson | Yes (mass — even with i:s:g 1:0.5:1.5 split) | **NO** — Bergeron/riming/graupel missing |

## D. Top issues to flag

1. **B1 / B2** — Morrison/Thompson missing `+L_f * (bergeron + riming_i + riming_s) / c_pd` and Thompson missing `+L_f * rime_to_graupel / c_pd`. Numerically meaningful in mixed-phase regimes (a typical 1 g/kg riming flux at the melt level dumps ~333 J/kg as latent heat, missing in dT/dt).
2. **B3** — Thompson lacks a `melt_X / dt` clamp present in Morrison.
3. **B4** — N_i should decrement on melt.
