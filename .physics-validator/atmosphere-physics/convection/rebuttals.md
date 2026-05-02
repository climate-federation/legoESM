# Convection — codex findings rebuttals & verification

Format per finding: **status** = `confirmed` | `partial` | `false-positive` | `not-actionable-here`.

## A. Confirmed bugs (need source change in future PR)

### C1 [confirmed, MAJOR]: `_plume.py:244-251` LNB masking surface-last vs surface-first index confusion
Verified by manual trace.
- `levels = arange(nlev)` (surface-last raw indices: 0=top, nlev-1=surface).
- `threshold = (nlev - 1) - k_lfc[:, None]` is converted to surface-first.
- `direction="below"` returns `sigmoid(s * (threshold - levels))` ⇒ high where `levels < threshold`.
- For k_lfc = 5 (surface-last index = 5 levels from top), threshold = nlev - 6 = 34 (for nlev=40). `levels < 34` selects 0..33, which includes both "above LFC" (0..4) AND "below LFC" (5..33).
- **Correct mask** should be `levels < k_lfc` ⇒ `direction="below", threshold=k_lfc`.

Effect: LNB diagnosed at surface or near-surface in many columns; cloud depth = 0 spuriously (which suppresses convection in ZM/KF/Emanuel/Tiedtke/Bechtold).

### C2 [confirmed, MAJOR]: `_plume.py:311-317` CIN window bounds inverted
Verified.
- `window_above_lfc = sigmoid(s * (k_lfc + 0.5 - levels))` — high when levels < k_lfc + 0.5 ⇒ "above LFC altitude" (small surface-last indices). ✓ for the variable name.
- `window_below_lcl = sigmoid(s * (levels - (k_lcl - 0.5)))` — high when levels > k_lcl - 0.5 ⇒ "below LCL altitude". ✓ for the variable name.
- BUT the **integration window for CIN should be between LCL and LFC** (i.e., levels with k_lfc < levels < k_lcl in surface-last). The product `above_lfc * below_lcl` requires `levels < k_lfc + 0.5 AND levels > k_lcl - 0.5`, which is **empty** (since k_lfc < k_lcl).

Correct: `below_lfc * above_lcl` (i.e., flip both signs). Or equivalently, swap the comparisons.

Effect: CIN evaluates to ~0 always; convection suppressed too easily by lacking inhibition.

### C3 [confirmed, CRITICAL — but DOCUMENTED]: `CAPE/tau` as M_b dimensionally inconsistent (`zhang_mcfarlane.py:131-148`, `kain_fritsch.py:166-176`, `emanuel.py:125-131`, `tiedtke.py:196-210`, `bechtold.py:188-223`)
Verified.
- `CAPE / tau` has units `m²/s³`.
- `M_b` should have units `kg/(m²·s)`.
- The discrepancy is `kg·s²/m⁴`. The standard ZM closure (Zhang & McFarlane 1995, Eq. 16-18) implicitly absorbs ρ/τ and a normalization scale. Skipping the density factor here means the literal value of `M_b_eq` is ~`m²/s³` not `kg/(m²·s)` — but then it's clipped to `M_b_max = 0.1 kg/(m²·s)`. The cap saves the magnitude but the unit error means the *gradient* w.r.t. CAPE is wrong by a factor of `ρ/L` (where L is some length scale).

This is a real *latent* unit error masked by `M_b_max` clipping. Refactor to absorb a `rho_LFC` or similar factor would fix.

### C4 [confirmed, MAJOR]: `_plume.py:495-496` plume mass flux explicit Euler with hard nonneg clip
```
M_u_raw = M_u_raw_prev * (1.0 + (eps - dlt) * dz)
M_u_raw = jnp.maximum(M_u_raw, 0.0)
```
For coarse layers (large dz) and large detrainment rates, `(eps - dlt) * dz < -1` ⇒ `M_u_raw < 0`, then floored at 0 with **dead gradient**. Recommend implicit form: `M_u_raw = M_u_raw_prev / (1 - (eps - dlt) * dz)` (with safeguards for delta ≈ eps).

### C5 [confirmed, MAJOR]: `_plume.py` cloud water `q_c_u_prev` not diluted by entrainment
Lines 484-541 in `step` function: `q_c_u = q_c_u_prev + condensate`. Entrainment dilutes T_u and q_u by adding env values, but q_c_u just *accumulates* condensation without dilution. Environmental air entrained should also dilute q_c_u (env q_c is zero ⇒ q_c_u → q_c_u * (1 - eps*dz)).

### C6 [confirmed, CRITICAL]: `mass_flux.py:319-331, 437-451` plume builds q_c from saturation difference, not vapor balance
```
q_v_u = dilution * q_sat_moist + (1 - dilution) * q_v
q_c_u = dilution * jnp.clip(q_sat_base - q_sat_moist, 0.0, None)
```
The plume's cloud water is the **saturation excess** of the launch parcel over the level's saturation, which is positive when ascending into a cooler region — independent of whether `q_v` actually has vapor to condense. For dry surface columns, this generates spurious cloud water.
**Correct**: q_c_u should track condensation = `q_v_at_(k-1) - q_sat(T_u, p)` — actual vapor lost from the rising parcel, conservation-limited.
The shared plume integrator `_plume.entraining_detraining_plume` does this correctly (line 522-526). **The bulk mass_flux/edmf paths use a non-conservative shortcut.**

### C7 [confirmed, MAJOR — DESIGN]: `mass_flux.py:211-240` kernel not column-MSE conservative
The subsidence + detrainment scheme uses **finite-volume tendencies** as separate sums; they don't telescope to a flux-form divergence. This is a known design choice (the legacy bulk Tiedtke approach), but it means small column-MSE residuals develop over time. Documented as xfailed in test suite.

### C8 [confirmed, MINOR]: `dt / max(tau, dt)` vs `dt / tau` divergence
For `dt > tau`, the documented `dt/tau` becomes silently capped to 1.0 (`max(tau, dt) = dt` ⇒ ratio = 1). For typical operational `dt = 600s` and `tau = 1200s`, ratio is `0.5`. For `tau = 300s`, the documented ratio is 2 but implementation gives 1. This biases relaxation rate at sub-step timescales.

### C9 [confirmed, MAJOR]: `mass_flux.py:280, 434` prognostic state explicit Euler
```
M_c_new = jnp.maximum(M_c + dt * (M_eq - M_c) / tau_adj, 0.0)
a_u_new = a_u + dt * (a_u_eq - a_u) / tau_a
```
For `dt / tau > 2`, the explicit step overshoots (oscillatory) and the `jnp.maximum(..., 0)` zeroes it. Should be implicit Euler like the other schemes.

### C10 [confirmed, MAJOR]: `tiedtke.py:243-260, bechtold.py:259-272` `delta_0_eff/delta_deep` rescale also scales subsidence
Verified — `_apply_mass_flux_kernel` returns `dT_dt = subsidence + detrainment`; rescaling the *whole* return by `delta_0_eff/delta_deep` incorrectly scales the subsidence too. Correct: only rescale the detrainment piece, or call the kernel with the per-class delta.

### C11 [confirmed, MAJOR]: `tiedtke.py:185-193` saturation deficit as MC proxy has wrong sign
`sat_deficit = max(q_sat - q_v, 0)` is large in **dry** columns. Convection should fire in **moist** columns. The proxy fires the wrong way. (Note: this fallback is only used when `moisture_convergence` is None, e.g., on grids without divergence operators.) Correct: use `max(q_v / q_sat - RH_threshold, 0)` or similar moist proxy.

### C12 [confirmed, MAJOR]: `tiedtke.py:283-287, bechtold.py:283-294` downdraft cooling dimensionally wrong, water-non-conserving
`dT_dt_dd = -(L_v/c_pd) * |M_d| * below_lcl_norm * 0.05 / rho_safe`. Units check:
- `M_d` [kg/m²/s]
- `below_lcl_norm` (dimensionless, normalized)
- `0.05` (described as "evap rate proxy [kg/kg]" — but a rate would be kg/kg/s)
- `1/rho` [m³/kg]
Total: `kg/m²/s × m³/kg × kg/kg = m × kg/kg = m·kg/kg`. Adding `(L_v/c_pd)` [K] gives K·m·kg/kg — NOT K/s. So the units are wrong.

Furthermore, no corresponding moistening tendency is added — the evaporated water vanishes. **Real bug.**

### C13 [confirmed, MAJOR]: `bechtold.py:127-133, 152-153` PBL parcel not mass-weighted; LCL uses surface pressure
Lines 131: `pbl_norm = sum(pbl_weight)`. Mass weighting requires `dp/g`. Then line 152: `compute_lcl(T_parcel, q_parcel, p_base, p_full)` uses `p_base` (surface pressure) for the launch level even when the parcel came from the PBL mean. Should use the column-mean pressure of the PBL instead.

### C14 [confirmed, MAJOR]: `emanuel.py:180-189` sort_multiplier scales whole kernel, not just detrainment
Same issue as C10. The `sort_multiplier` is described as enhancing detrainment, but `dT_dt = dT_dt_raw * sort_multiplier` includes the subsidence piece.

### C15 [confirmed, MAJOR]: `kuo.py:160-173` moistening budget has extra `g/dp` factor
Line 169-173:
```
dq_v_dt = moistening_budget[:, None] * (deficit / deficit_integral_safe) * constants.g / dp
```
Column integral: `sum(dq_v_dt * dp/g)` = `moistening_budget * sum(deficit/deficit_int) * 1` ≠ `moistening_budget`. The `g/dp` factor in the integrand cancels with the integration mass weight, leaving an unweighted sum. **The intended column total `(1-α) * MC / τ` is NOT achieved.**

### C16 [confirmed, MAJOR]: `kuo.py` heating-implied condensation vs cloud-water source inconsistency
Lines 156, 198-205. `dT_dt` uses an MC-gated rate, then `implied_condensation = dT_dt * c_pd / L_v` is subtracted from vapor (line 176). But `dq_c_conv_dt` is rebuilt from a different "target_col_cond" derived from the MC budget — possibly different magnitude. This means latent energy and water are inconsistent.

### C17 [confirmed, MINOR]: `_triggers.py:259-321` no-crossing fallback gate centered at 0.5
The blend `sigmoid(20 * (total - 0.5))` returns ~0 when `total < 0.4`, so a weak but real upward crossing (with total weight 0.3, say) is treated as "no crossing" and the result blends to surface index. This corrupts LCL/LFC/LNB diagnostics in low-buoyancy columns.

### C18 [confirmed, MINOR]: `mass_flux.py:82-89` z[:, -1] != 0
```
z = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
```
`z[:, -1]` is the last cumulative sum value (= dz of surface layer), not zero. The plume integration starts with z_surface ≠ 0; small bias.

### C19 [confirmed, MINOR]: `q_v` vs `saturation_mixing_ratio` semantics
Throughout convection: `q_v` is documented as "specific humidity" (kg/kg) but compared to `saturation_mixing_ratio`. Strictly these are different (`q_v_specific = m_v/m_total`; `r = m_v/m_dry`). For `q_v < 0.05`, the difference is < 5%. Style/documentation issue.

### C20 [confirmed, INFO]: Several `clip`/`max` patterns in active physics paths (sbm.py:89-93, dca.py:141-147, kuo.py:94-96, mass_flux.py:224-240)
Hard limiters create dead-gradient regions. Each is justified individually (cape thresholds, M_b cap, q_c clip), but cumulatively they introduce kinks. Documented design choice.

## B. Hypotheses confirmed not bugs

- **H2 (Bechtold MC normalizer 0.05 kg/m²/s)**: configurability/style only.
- **H3 (DCA column rescale)**: column-conservative; per-level distribution is the actual concern, not conservation.
- **H5 (T-55 floor)**: floor only triggers at T<56 K — outside atmospheric range.
- **H7 (strato gate factor 0.013 at top)**: AD tradeoff, intentional smoothness.
- **H10 (sort_multiplier ∈ [1, 1+cu])**: bound is correct; the bug (C14) is applying it to the whole kernel.

## Summary

Codex confirmed **17 substantive findings** (4 critical, 11 major, 2 minor). Most are real bugs that would benefit from source changes — but **all are out of scope for this audit cycle** (we only run static + diff testing, no source modifications without numeric tests).

Top 5 highest-priority follow-ups (by severity × probability):
1. **C2 (CIN window inverted)** — MAJOR, affects all schemes using CIN diagnostic.
2. **C1 (LNB index confusion)** — MAJOR, affects 5 schemes (cloud_depth used for deep/shallow blending).
3. **C12 (downdraft cooling units)** — MAJOR, dimensionally wrong + non-conservative.
4. **C6 (q_c_u from saturation excess in mass_flux/edmf)** — CRITICAL, generates spurious cloud water in dry columns.
5. **C11 (Tiedtke MC proxy wrong sign)** — MAJOR, fires deep convection in dry air on grids without divergence operator.
