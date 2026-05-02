# Codex Adversarial Review — Convection & Microphysics (Cycle 2 — Iteration 2)

You are an independent adversarial physics-parameterization reviewer for legoESM, a JAX-native differentiable Earth System Model. Your job is to verify that fixes applied in response to your previous review (cycle 2 iteration 1) are correct and to find any new issues.

## Context: previous review (cycle 2 iteration 1) findings

You returned 6 findings:
1. **P1 (CONFIRMED)**: plume entrainment uses unstable explicit Euler `T_u_ent = T_u_prev + eps*dz*(T_e - T_u_prev)`. With Bechtold defaults `epsilon_shallow=3e-3` × layer thickness 500-1000 m, `eps*dz > 1` causes `q_u_ent` to overshoot past `q_e` (go negative).
2. **P1 (DEFERRED)**: Hines drag dimensional inconsistency `rho * (sigma_grown - sigma_new)` is `kg/(m²·s)` not Pa. Deferred because fixing requires re-tuning.
3. **P2**: plume termination is local (no buoyancy-death memory in carry). Documented as deferred design issue.
4. **P2**: Tiedtke MC proxy has zero dry-side gradient via `jnp.maximum`. Documented as P2.
5. **P2**: warm_rain `x_c = q_c * rho / N_c` and `dN_r_au = dq_c_au * rho / ...` use per-volume convention while N is documented per-kg. Documented as P3 (likely doc bug).
6. **P1 (CONFIRMED)**: Thompson rime_to_graupel uses fixed 1.0 / 0.5 / 1.5 split on total_riming, violating mass conservation when only one species is rimed (e.g. q_i=0, q_s>0).

## Fixes applied to your findings

### Fix for #1: Plume entrainment exponential relaxation

`src/legoesm/atmosphere/physics/convection/_plume.py:534-548`:
```python
# T_u_ent and q_u_ent: exponential relaxation toward env, exact for
# linear ODE dX/dz = -eps*(X - X_e)
decay_eps = jnp.exp(-eps * dz)
T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
q_u_ent = q_e + (q_u_prev - q_e) * decay_eps
```

Also changed `q_c_u_ent = q_c_u_prev * decay_eps` (was `max(q_c_u_prev * (1 - eps*dz), 0)`).

### Fix for #6: Thompson rime_to_graupel donor split

`src/legoesm/atmosphere/physics/microphysics/thompson.py`:
```python
# Compute conversion per donor:
rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

# Apply qc_scale to each donor (same scale, since both come from q_c):
rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

# Apply per-donor in tendency assembly:
dq_i_dt = (... + riming_i - aggregation - melt_ice - rime_to_graupel_from_i + sed_i)
dq_s_dt = (aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s)
dq_g_dt = rime_to_graupel - melt_graupel + sed_g  # all of it goes to graupel
```

Mass conservation: `dq_i loss + dq_s loss + dq_g gain = -rime_to_graupel_from_i - rime_to_graupel_from_s + (from_i + from_s) = 0` ✓.

### Tests added

- `tests/unit/test_convection_plume.py::test_plume_T_q_entrainment_does_not_overshoot_environment` — passes on fixed code, fails on buggy code.
- `tests/unit/test_convection_plume.py::test_plume_grad_through_epsilon_at_strong_entrainment` — verifies non-zero grad at eps·dz > 1.
- `tests/unit/test_physics_microphysics.py::test_thompson_rime_to_graupel_donor_split` — pathological column q_i=0, q_s>0 — verifies q_i stays non-negative.

## Your task (iteration 2)

1. Verify the two fixes above are mathematically correct and complete.
2. Look for **NEW issues** introduced by these fixes (regressions).
3. Re-read the code below and try to find any **OTHER bugs** the cycle-1 review missed.
4. Specifically: check the `q_u_ent = q_e + (q_u_prev - q_e) * decay_eps` formulation — does it correctly preserve total water (q_u_ent + q_c_u_ent) when entrainment dilutes both? Or is there a hidden mass leak?

## Source: convection/_plume.py (post-fix)

```python
def step(carry, layer_inputs):
    T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
    T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs

    dz = jnp.maximum(z_e - z_prev, 1.0)

    # Exact integration of dM/dz = (eps - dlt) M
    M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)

    # Exponential relaxation entrainment (NEW FIX)
    decay_eps = jnp.exp(-eps * dz)
    T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
    q_u_ent = q_e + (q_u_prev - q_e) * decay_eps

    rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
    Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
    dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
    T_u = T_u_ent + dT_dz * dz

    q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
    condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
    q_u = (q_u_ent - condensate).astype(_dtype)

    # Exponential cloud-water dilution (NEW FIX)
    q_c_u_ent = q_c_u_prev * decay_eps
    q_c_u = (q_c_u_ent + condensate).astype(_dtype)

    T_u = T_u.astype(_dtype)
    B_u = T_u - T_e

    plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
    M_u_reported = M_u_raw * plume_alive * abv

    new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
    outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
    return new_carry, outputs
```

## Source: microphysics/thompson.py (post-fix, key sections)

```python
# === GRAUPEL (Thompson extension) — donor split ===
graupel_frac = jax.nn.sigmoid(
    config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
)
rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
melt_graupel = jnp.minimum(
    config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
    jnp.clip(q_g, 0.0) / dt_safe,
)

# === DONOR CLAMP for q_c sinks ===
qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
qc_avail = jnp.clip(q_c, 0.0)
qc_scale = jnp.minimum(
    1.0,
    qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
)
dq_c_au = dq_c_au * qc_scale
dq_c_ac = dq_c_ac * qc_scale
bergeron = bergeron * qc_scale
riming_i = riming_i * qc_scale
riming_s = riming_s * qc_scale
total_riming = riming_i + riming_s
rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
dN_r_au = dN_r_au * qc_scale

# === COMBINE TENDENCIES ===
dq_v_dt = -condensation + evaporation - dq_i_dep
dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
dq_i_dt = (
    dq_i_dep + bergeron + riming_i - aggregation - melt_ice
    - rime_to_graupel_from_i + sed_i
)
dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
dq_g_dt = rime_to_graupel - melt_graupel + sed_g
```

## Specific verification questions

1. **Plume mass conservation under exponential entrainment**: with `decay_eps = exp(-eps*dz)`, the env air entrained into the plume is `q_e * (1 - decay_eps) + q_c_e * (1 - decay_eps)`. Since `q_c_e = 0` (no env cloud water), the plume's total water `q_u + q_c_u` after entrainment is `q_e * (1 - decay_eps) + q_u_prev * decay_eps + q_c_u_prev * decay_eps`. Is this equal to `q_u_prev + q_c_u_prev` (closed plume) plus an entrained source? YES if we add the env water in proportion to entrainment — checking that this is indeed what happens physically.

2. **q_u_ent monotonicity**: with exponential relaxation, is `q_u_ent` always between `min(q_e, q_u_prev)` and `max(q_e, q_u_prev)`? (Should be, since `decay_eps ∈ (0, 1]`.)

3. **Thompson conservation**: in the new tendency assembly, verify ∑(dq_v + dq_c + dq_r + dq_i + dq_s + dq_g) = sed terms only (no internal phase changes accumulate).

4. **Did anything else break?**: re-read `_plume.py` and `thompson.py` from a fresh perspective. Look for new bugs introduced by these specific changes. Pay attention to AD: does `decay_eps = exp(-eps*dz)` have well-defined gradients w.r.t. `eps` and `dz` everywhere?

5. Any **OTHER** issues you see in the cycle-1 source that you didn't flag the first time? (E.g. in microphysics — how about the `dN_i_dt = dN_i_nuc - aggregation * N_i / q_i` formula at q_i → 0?)

Provide a concise list of any new findings; if your previous findings are all addressed, say so explicitly.
