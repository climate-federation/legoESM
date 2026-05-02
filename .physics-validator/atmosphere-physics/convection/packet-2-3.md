# Codex Adversarial Review — Microphysics (Cycle 2 — Iteration 3)

You are an independent adversarial physics-parameterization reviewer for legoESM. In iteration 2 you flagged the bug "subsaturated clear air can create negative cloud water" and I have applied a fix. Please verify the fix is correct.

## The bug (you flagged it in iteration 2)

`saturation_adjustment(T, q_v, p_full, dt)` returns `condensation = sigmoid(s · excess) · excess / dt` where `excess = q_v - q_sat`. For `q_v < q_sat` (subsaturated), `condensation < 0`, but Morrison/Thompson/Kessler all add this directly to `dq_c_dt` without checking q_c availability. In a clear-air column (q_c = 0), explicit Euler then drives q_c to a negative value of order ~3e-4 kg/kg per timestep at 95% RH, dt=1200 s.

## The fix

### Part 1: `_warm_rain.saturation_adjustment` now donor-clamps the evaporation branch

```python
def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    if q_c is not None:
        # Evaporation rate (negative ``condensation``) bounded by available q_c:
        # |condensation| × dt ≤ q_c, i.e. condensation ≥ -q_c / dt.
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(
            condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
        )
    return condensation, q_sat
```

### Part 2: Morrison/Thompson include `cond_evap_sink = max(-condensation, 0)` in the donor clamp

```python
cond_evap_sink = jnp.maximum(-condensation, 0.0)
qc_sink_total = (
    dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
)
qc_avail = jnp.clip(q_c, 0.0)
qc_scale = jnp.minimum(
    1.0,
    qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
)
# Scale all sinks (including condensation when negative):
dq_c_au = dq_c_au * qc_scale
# ... etc
condensation = jnp.where(
    condensation < 0.0, condensation * qc_scale, condensation,
)
```

### Part 3: Kessler (which inlines saturation logic) gets the same clamp inline

```python
condensation = cond_frac * excess / dt  # signed
q_c_avail = jnp.clip(q_c, 0.0, None)
condensation = jnp.maximum(
    condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
)
# Kessler doesn't have a multi-process donor clamp because:
#   accretion = k_ac * q_c * q_r^0.875  (proportional to q_c)
#   autoconv = max(q_c_updated - threshold, 0) * rate  (gated)
# So q_c=0 → accretion=0, autoconv=0; only condensation needs clamping.
```

## Mass conservation check

For Morrison/Thompson, after my fix:
- `dq_v_dt = -condensation + evaporation - dq_i_dep` (condensation already scaled)
- `dq_c_dt = condensation - (sinks scaled by qc_scale)` (condensation already scaled)
- Total `(dq_v + dq_c)` from saturation branch: `-condensation_scaled + condensation_scaled = 0` ✓

For Kessler:
- `dq_v_dt = -condensation + evaporation` (condensation now donor-clamped)
- `dq_c_dt = condensation - autoconv - accretion`
- Saturation contribution: `-condensation + condensation = 0` ✓

## Test verification

```
test_subsaturated_clear_air_does_not_create_negative_qc[kessler]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[seifert_beheng]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[morrison]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[thompson]: PASS
```

All 257 physics-related tests still pass.

## Your task

1. Verify the fix is mathematically and dimensionally correct.
2. Verify mass conservation holds in both clear-air (q_c=0) and cloudy (q_c>0) columns.
3. Confirm the fix handles all four code paths (Kessler, SB, Morrison, Thompson).
4. Check for any new bugs introduced by these changes.
5. Specifically: in Morrison/Thompson, the `condensation` variable is referenced *both* in the donor clamp (where it's scaled when negative) AND in `dq_v_dt = -condensation + evaporation - dq_i_dep`. The scaling is in-place via `condensation = jnp.where(...)`, so the subsequent reference picks up the scaled value. Verify this is correct.

## Outstanding deferred findings (from iteration 2)

These were not fixed and may need deferral notes:

1. **Hines drag dimensional inconsistency** (P1, deferred): `drag = rho * (sigma_grown - sigma_new)` has units `kg/(m²·s)` not Pa. Requires re-tuning so deferred.
2. **N_i nucleation creates number without ice mass** (P2): `dN_i_nuc > 0` but `dq_i_dep ∝ q_i`, so q_i can't grow from 0.
3. **Thompson `graupel_frac` uses pre-clamp `total_riming`** (P3): mass-conservative but threshold decision is inconsistent with what the limiter returns.

Provide:
- Confirmation that the subsaturated-clear-air fix is correct (or NEW concerns).
- Status assessment for deferred items.
- Any NEW bugs you see in the post-fix code.

If you have no substantive findings, state so explicitly.
