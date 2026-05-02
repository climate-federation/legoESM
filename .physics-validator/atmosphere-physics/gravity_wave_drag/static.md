# Gravity wave drag — static analysis

## A. Confirmed correctness

1. **Sign of drag tendency**: `du_dt = accel * cos_a` where `accel = -drag/(ρ dz)` and `cos_a` is along low-level wind. Drag opposes wind ⇒ `accel < 0`, `du_dt < 0` for u > 0. ✓
2. **Frictional heating**: `dT_dt = -(u du_dt + v dv_dt) / c_pd`. For decelerating flow (u du_dt < 0), heating is positive. ✓
3. **Lindzen saturation cap**: `tau_new = jnp.minimum(tau_new, tau_carry)` enforces monotone decrease of stress. ✓
4. **McFarlane smooth-min** uses log-sum-exp (`logsumexp`) for differentiable min — correct AD treatment.

## B. Issues (RANKED)

### B1. **Hines `sigma_grown` recurrence may be incorrect** [POTENTIAL BUG — NEEDS VERIFICATION]
`hines.py:90`:
```python
rho_ratio = jnp.sqrt(rho_sfc / rho)  # cumulative surface-to-k ratio
def scan_fn(carry, k_rev):
    sigma_gw = carry          # carries previous level's sigma_new
    k = nlev - 1 - k_rev
    sigma_grown = sigma_gw * rho_ratio[:, k]   # <-- multiplies by CUMULATIVE ratio
    ...
```
The carry holds `sigma_new_at_(k+1)` (the previous level's amplitude after dissipation). Multiplying by `rho_ratio[k] = sqrt(rho_sfc/rho_k)` does NOT give the correct WKB amplitude growth between adjacent levels — it gives `sigma_new_(k+1) * sqrt(rho_sfc/rho_k)`, which compounds the surface ratio across levels. Correct WKB growth between levels is `sigma_(k+1) * sqrt(rho_(k+1)/rho_k)`.

**Effect**: in the absence of saturation dissipation, `sigma_grown` overshoots correct cumulative WKB growth; with saturation kicking in (which it almost always does in the stratosphere), the bug is masked because `sigma_new = sigma_sat[k]` clips the runaway. So this is likely a *latent* bug that only matters in tropospheric-only or weakly-saturated regimes.

**Recommendation**: change to `sigma_grown = sigma_init * rho_ratio[:, k]` (carrying surface amplitude unchanged) and apply dissipation as a stress sink, OR change the recurrence to use the inter-level ratio `sqrt(rho[:, k+1] / rho[:, k])`. Adding a regression test against an analytical WKB profile would settle this.

### B2. **Hines `sigma_sat = N / (m_star * rho_ratio)`** [QUESTIONABLE]
`hines.py:81`. The standard Hines saturation amplitude is `sigma_sat ∝ N/m_star` (a constant per column at given N). Dividing by `rho_ratio = sqrt(rho_sfc/rho)` (≥1) gives a sigma_sat that *decreases* with altitude (since rho decreases). That's the opposite of physical expectation: gravity wave amplitude should saturate at a **larger** value at altitude (since rho is smaller and the wind speed is larger).

This may be a direct cause of unphysical drag profiles. **Worth a deeper look** — could be a copy-paste typo where `* rho_ratio` was intended.

### B3. **All GWD schemes lack gradient tests** [TEST GAP]
Confirmed: `tests/unit/test_physics_gwd.py` does NOT use `jax.grad` (verified by grep). Given that GWD is differentiable by design (smooth saturation, sigmoid breaking), AD tests should be straightforward. Recommend adding a baseline `test_diff_gwd.py` to assert gradient flow w.r.t. u, v, T_sfc.

### B4. **Hardcoded sharpness 50.0 in `mcfarlane.py:100`** [TUNABLE-IN-BODY]
`alpha = 50.0` for the softmin. Should be `McFarlaneConfig.softmin_sharpness`.

### B5. **Hardcoded 20.0 in `mcfarlane.py:75`** [TUNABLE-IN-BODY]
`U_activated = sigmoid(20.0 * (U_ll - config.min_wind)) * U_ll`. The sharpness 20 is hardcoded.

### B6. **Lindzen `drag_all = drag_stack.T[:, ::-1]` ordering** [VERIFIED]
Reading carefully: scan iterates `k_rev = 0, ..., nlev-1`, with `k = nlev-1-k_rev`. `drag_stack[i, :]` corresponds to `k = nlev-1-i`. After `.T` and `[:, ::-1]`, `drag_all[:, j]` = `drag_stack[nlev-1-j, :]` = level `j` (top-first). ✓

### B7. **Lindzen `tau_0` clip at `0.0`** [INFO]
`lindzen.py:80`: `tau_0 = jnp.clip(tau_0, 0.0, None)`. Bottom clip preserves positivity. Stress was positive-definite by construction `rho * N * k * h^2 * U` (all non-negative). So the clip is redundant. ✓

### B8. **McFarlane `tau_0 = ... * directional_spread` clipped to `[0, 10.0]`** [HARDCODE]
`mcfarlane.py:86`: hard upper bound `10.0` (units: kg/m²/s × something — actually Pa = N/m²). Should be `McFarlaneConfig.tau_max`.

## C. Top issues

* **B1, B2 (Hines)**: potential numerical bugs in wave amplitude growth recursion and saturation amplitude formula. Recommend running a saturated-wave test against analytic WKB profile to confirm/refute.
* **B3 (no gradient tests)**: easy fix.
* **B4–B5, B8 (hardcoded tunables)**: CLAUDE.md violations, easy fix.
