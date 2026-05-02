# Microphysics — codex findings rebuttals & verification

## A. Confirmed substantive bugs (need source change)

### M1 [confirmed, CRITICAL]: `sundqvist.py:150-152` column-water non-conservation
Verified. `dq_r_dt = autoconversion - evaporation` does not include sedimentation/precipitation flux out of the column. `precipitation` is computed (the column-bottom rain flux) but never deducted from any `q_r`. Either:
- `q_r` is meant as "rain produced this step, instantly precipitated" (then `dq_r_dt` should be zero or omitted). 
- OR sedimentation flux is missing.

### M2 [confirmed, MAJOR]: Morrison `morrison.py:149-154` missing `+L_f * (bergeron + riming_i + riming_s) / c_pd`
Verified. Liquid → ice phase change releases L_f.

### M3 [confirmed, MAJOR]: Thompson `thompson.py:174-179` same as M2 plus possibly missing for `rime_to_graupel`
Verified Bergeron + riming. Codex notes `rime_to_graupel` is "frozen-species-to-frozen-species" so no L_f needed there — agreed (mass transfer between i/s/g is solid-to-solid).

### M4 [confirmed, CRITICAL]: Thompson melt overdraft (`thompson.py:141-151, 184-187`)
Verified. `melt_rate * dt = 6` for default `5e-3 /s × 1200 s`. Morrison clamps via `jnp.minimum(..., q_X / dt)` (line 121-128); Thompson does not.

### M5 [confirmed, MAJOR]: `_warm_rain.py:39-42` saturation adjustment can return negative "condensation"
Verified. For subsaturated air (`q_v < q_sat`), `excess < 0`, `cond_frac` near 0 (small), but `condensation = cond_frac * excess / dt` can be slightly negative. Adding to `q_c` produces negative cloud water if there's no donor limiter. Probably doesn't fire in practice (cond_frac < 0.01 in most subsaturated columns), but technically incorrect.

### M6 [confirmed, MAJOR]: `_warm_rain.py:93-96` Seifert-Beheng `x_c` units inconsistency
Verified. The docstring states `N_c [1/kg]`, but the formula `x_c = q_c * rho / N_c` only gives kg/drop if `N_c [1/m³]`. Default `Nc_0 = 1e8` matches `1/m³` (typical 100/cm³). **The documentation is wrong, but the formula effectively works** if `N_c` is reinterpreted as `1/m³`. Either the docstring should be fixed, or all callers should switch to `1/m³` units.

### M7 [confirmed, MAJOR]: `q ** 0.525` and `q ** 0.875` infinite gradients at q=0
Verified. `kessler.py:92, 96`, etc. use `clip(q_r, 0) ** 0.525`. At q_r = 0, `d/dq_r (q_r^0.525) = 0.525 * q_r^(-0.475) → ∞`. JAX evaluates this to NaN gradient. Practical effect: training runs with q_r very close to zero may NaN.
**Fix candidate**: use `(q + epsilon) ** 0.525` with epsilon small but non-zero, or smooth_pow.

### M8 [confirmed, MAJOR]: Number moments not adjusted for several mass-removal processes
Verified. `N_i` not decremented during melting (M1 from my static.md). Codex extends this: `N_c` ignores accretion/riming/Bergeron losses; `N_r` ignores evaporation/sedimentation losses. **Real bug** for two-moment consistency.

### M9 [confirmed, CRITICAL]: Mixed-phase donor over-extraction
Verified. Default `bergeron_rate = 1e-3 /s × dt = 1200 s = 1.2 ⇒ explicit step removes 1.2 × q_c`, leaving negative q_c. Same for riming and rime_to_graupel.
Morrison/Thompson should clamp donor depletion (analogous to Morrison's melt clamp).

### M10 [confirmed, MAJOR]: Ice nucleation creates N_i without q_i seed
`morrison.py:88-103`, `thompson.py:109-124`. `dN_i_nuc > 0` even when `q_i = 0`. Then `dq_i_dep ∝ q_i` ⇒ no growth. Result: massless ice crystals stay massless forever.

### M11 [confirmed, MAJOR]: Sundqvist autoconversion not donor-limited (`sundqvist.py:68`)
For `auto_rate * dt > 1`, P_auto * dt exceeds available q_c. Need `min(P_auto, (q_c + cond*dt) / dt)` clamp.

## B. Hypotheses that were not bugs

- H8 (`_gamma_ratio`): rebutted ✓ — the recurrence Γ(μ+1)=μΓ(μ) makes the formula exact for any real μ.
- H9 (no Newton iteration): the design is fine for AD; the bug is in unbounded condensation magnitude.
- H10 (xfailed conservation tests): no microphysics test marked `xfail` was found; the gap is in test coverage, not a marker.

## Summary

11 confirmed substantive findings (3 critical, 8 major). All require source-level fixes — out of scope for this audit cycle.

Top priority follow-ups:
1. **M1**: Sundqvist column water leak — most impactful for AMIP runs.
2. **M9 / M4 / M11**: donor-limit clamps (multiple schemes).
3. **M2 / M3**: missing `+L_f * (bergeron + riming) / c_pd` in mixed-phase heating budget.
4. **M7**: `q ** fractional_power` AD-instability at q=0.
5. **M6**: SB units documentation vs formula mismatch.
