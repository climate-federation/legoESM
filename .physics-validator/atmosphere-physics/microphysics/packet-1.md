# Adversarial review: legoESM microphysics package

You are an independent adversarial physics reviewer for an Earth-system model.

Your task: read the source files below and find every concrete bug, sign error, unit inconsistency, broken-gradient pattern, conservation violation, and AD-safety issue. Cite line numbers. Severity: critical | major | minor.

The codebase under review is a JAX implementation of atmospheric microphysics that must be:
- Physically correct (units, signs, mass+latent-heat conservation, monotonicity).
- Differentiable end-to-end (no hard `if` on traced values, AD-safe limiters).

Static analysis already produced these candidate concerns for you to verify or rebut:

1. **Morrison (`morrison.py:149-154`) — Bergeron + riming missing latent heat contributions.** The current `dT_dt` only includes condensation, evaporation, deposition, and melting. **Bergeron (cloud water → ice) and riming (cloud water → ice/snow) should each release `L_f` (latent heat of fusion).** For default `bergeron_rate = 1e-3 /s` and `q_c = 1e-4 kg/kg`, the missing heating is `L_f * 1e-7 / c_pd ~ 3.3e-5 K/s ~ 2.8 K/day` in mixed-phase clouds. Correct?

2. **Thompson (`thompson.py:174-179`) — same Bergeron + riming + graupel formation latent heat omitted.**

3. **Thompson (`thompson.py:143-144, 151`) — `melt_X = melt_rate * q_X * melt_frac` not clamped to `q_X / dt`.** Morrison (`morrison.py:121-128`) has this clamp. For melt_rate=5e-3/s and dt=1200s, `melt_rate * dt = 6` ⇒ explicit step yields negative q_i_new for small q_i.

4. **Morrison/Thompson — N_i not decremented during melting** (`morrison.py:165`, `thompson.py:191`). When q_i melts, N_i stays the same; mean particle mass `q_i / N_i` becomes unphysically small.

5. **Sundqvist (`sundqvist.py:68`) — `P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)` uses post-condensation cloud water.** This is forward-Euler coupled to dt linearly. Standard Sundqvist uses `q_c` only. May or may not be a bug.

6. **`_warm_rain.autoconversion_sb:93` — `x_c = q_c_pos * rho / max(N_c_eff, 1.0)`.** When q_c=0, x_c=0; `dq_c_au = k_au * q_c**2 * onset(x_c-x_star)`. For q_c approaching zero, onset(0-x_star) ≈ 0, dq_c_au ≈ 0. AD: gradient w.r.t. q_c is well-defined.

7. **`_warm_rain.rain_evaporation` and `kessler` use `q_r ** 0.525`.** AD-unstable at q_r = 0 (derivative diverges). Practically, q_r > 0 once precipitation has occurred. Probably fine in practice.

8. **Thompson `_gamma_ratio(mu) = (mu+3)(mu+2)(mu+1)`** — Γ(μ+4)/Γ(μ+1). Comment says "for integer-like μ" but the recurrence Γ(μ+1) = μΓ(μ) makes this exact for ANY real μ ≥ 0. Non-bug.

9. **All schemes lack a saturation adjustment iteration**: a single sigmoid-based saturation step does not enforce thermodynamic equilibrium. The `cond_frac = sigmoid(s * excess) * excess / dt` produces a smooth condensation but doesn't iterate to (T_new, q_new) on the saturation curve. Acceptable for AD-friendly column physics but not as accurate as Newton-iterated saturation.

10. **Conservation guard tests**: I'm told some conservation tests are `xfail` in the suite. Verify by checking conservation column-by-column in the source.

Files to read:
- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py` (176 LOC).
- `src/legoesm/atmosphere/physics/microphysics/kessler.py` (136 LOC).
- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py` (170 LOC).
- `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py` (130 LOC).
- `src/legoesm/atmosphere/physics/microphysics/morrison.py` (189 LOC).
- `src/legoesm/atmosphere/physics/microphysics/thompson.py` (213 LOC).
- `src/legoesm/atmosphere/physics/microphysics/config.py`.
- `src/legoesm/atmosphere/physics/microphysics/output.py`.

Constants from `legoesm.constants`: L_v = 2.501e6 J/kg, L_s = 2.838e6 J/kg, L_f = L_s - L_v = 0.337e6 J/kg, c_pd = 1004.64 J/kg/K, T_freeze = 273.15 K, R_v = 461.5 J/kg/K.

For each finding, output:
- File path : line range.
- One-sentence summary.
- Brief technical explanation (≤3 sentences).
- Severity.
- If addressing one of the 10 hypotheses above, explicitly say so.

If no findings beyond those flagged, end with `NO ADDITIONAL SUBSTANTIVE FINDINGS`.
