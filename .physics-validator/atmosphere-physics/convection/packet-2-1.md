# Codex Adversarial Review — Convection & Microphysics (Cycle 2)

You are an independent adversarial physics-parameterization reviewer for legoESM, a JAX-native differentiable Earth System Model. Your job is to find bugs, sign errors, unit inconsistencies, broken-gradient patterns, conservation violations, and test-case discrepancies in the code below. Cite specific line numbers. If you believe there are no bugs, say so and explain why each candidate concern is not one.

## Context: prior audit cycle (already addressed)

A first audit cycle (`.physics-validator/atmosphere-physics/REPORT.md`) flagged 7 P0 critical bugs in convection and microphysics. The fixes are in place and have been verified by unit tests. The Top 10 findings from cycle 1 were:

1. ✅ FIXED: convection LNB index confusion (`_plume.compute_lfc_lnb`)
2. ✅ FIXED: convection CIN window inversion (`_plume.compute_cin`)
3. ✅ FIXED: CAPE/τ→M_b dimensional fix in 5 schemes
4. ✅ FIXED: mass_flux/edmf q_c_u from saturation excess (dry-column bug)
5. ✅ FIXED: Sundqvist column-water conservation
6. ✅ FIXED: Morrison/Thompson Bergeron+riming donor over-extraction
7. ✅ FIXED: Hines WKB amplitude inter-level vs cumulative ratio
8. ✅ FIXED: Morrison/Thompson missing L_f for Bergeron + riming
9. ✅ FIXED: delta_0_eff/delta_deep rescale also scaling subsidence
10. ✅ FIXED: Tiedtke MC proxy wrong sign

Additional fixes since cycle 1:
- Plume q_c_u entrainment dilution: `q_c_u_ent = max(q_c_u_prev * (1 - eps*dz), 0)` before adding condensate
- Plume mass flux exact integration: `M_u_raw = M_u_raw_prev * exp((eps - dlt) * dz)` (was explicit Euler + clip that killed gradients)
- Tiedtke/Bechtold downdraft cooling: dimensionally correct, locally + column-water conserving
- ZM/Tiedtke/Bechtold implicit-Euler ratio: `dt/max(tau, 1e-30)` (was `dt/max(tau, dt)`)
- Bechtold PBL parcel: mass-weighted with `dp_full + p_pbl` for LCL launch
- safe_pow guard for fractional powers in microphysics (kessler, _warm_rain, morrison, thompson, seifert_beheng)

## Your task

Review the post-fix source code below for any:
1. **NEW bugs** introduced by the fixes (regressions)
2. **Residual issues** the prior audit missed
3. **Conservation violations** in any path that was modified
4. **Dimensional / unit errors** anywhere in the modified code
5. **Sign convention errors** (especially in the new downdraft logic)
6. **Differentiability issues** — broken gradients, NaN/Inf at edge cases (q=0, T=0, etc.)
7. **Numerical robustness** — silent clips that mask physics bugs, spurious zeros at boundaries

Focus areas (highest risk = most invasive recent changes):
- `convection/_plume.py` — entire plume integrator rewritten
- `convection/tiedtke.py` — downdraft + MC proxy + closure all changed
- `convection/bechtold.py` — PBL parcel + downdraft + MC enhancement changed
- `microphysics/_warm_rain.py` — `safe_pow` introduced, used by all microphysics

Do NOT re-flag the 10 prior findings as bugs (they're fixed). Look for things the prior audit missed.

## Source 1: convection/_plume.py

```python
"""Column physics primitives shared across convection schemes.
... (full file content below)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_vapor_pressure,
)
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
    moist_adiabat_lapse_rate,
)

from legoesm.atmosphere.physics.convection._triggers import (
    smooth_level_indicator,
    smooth_lowest_crossing_index,
    smooth_step,
)


class LCL(NamedTuple):
    p_lcl: jax.Array
    T_lcl: jax.Array
    k_lcl_smooth: jax.Array


def compute_lcl(
    T_parcel: jax.Array,
    q_parcel: jax.Array,
    p_parcel: jax.Array,
    p_full: jax.Array,
    *,
    crossing_sharpness: float = 0.001,
) -> LCL:
    e_sat = saturation_vapor_pressure(T_parcel)
    q_sat = saturation_mixing_ratio(T_parcel, p_parcel)
    RH = jnp.clip(q_parcel / jnp.maximum(q_sat, 1e-12), 1e-4, 1.0)
    T_minus_55 = jnp.maximum(T_parcel - 55.0, 1.0)
    T_lcl = 1.0 / (1.0 / T_minus_55 - jnp.log(RH) / 2840.0) + 55.0
    p_lcl = p_parcel * (T_lcl / T_parcel) ** (constants.c_pd / constants.R_d)
    k_lcl_smooth = smooth_lowest_crossing_index(
        -p_full, -p_lcl[:, None], crossing_sharpness,
    )
    return LCL(p_lcl=p_lcl, T_lcl=T_lcl, k_lcl_smooth=k_lcl_smooth)


def compute_lfc_lnb(
    T_env: jax.Array,
    T_parcel_ma: jax.Array,
    *,
    sharpness: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    buoyancy = T_parcel_ma - T_env
    nlev = buoyancy.shape[-1]
    k_lfc = smooth_lowest_crossing_index(buoyancy, 0.0, sharpness)

    # In surface-last indexing the cloud base is at k_lfc; "above LFC altitude"
    # means levels with surface-last index *smaller* than k_lfc.
    above_lfc_weight = smooth_level_indicator(
        jnp.broadcast_to(jnp.arange(nlev, dtype=buoyancy.dtype), buoyancy.shape),
        threshold=k_lfc[:, None],
        sharpness=sharpness,
        direction="below",
    )
    LARGE = jnp.asarray(1.0e6, dtype=buoyancy.dtype)
    guarded_neg_buoyancy = -buoyancy - LARGE * (1.0 - above_lfc_weight)
    k_lnb = smooth_lowest_crossing_index(guarded_neg_buoyancy, 0.0, sharpness)
    return k_lfc, k_lnb


def compute_cin(
    T_env: jax.Array,
    T_parcel_ma: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    k_lcl_smooth: jax.Array,
    k_lfc_smooth: jax.Array,
    *,
    indicator_sharpness: float = 1.0,
) -> jax.Array:
    nlev = T_env.shape[-1]
    levels = jnp.arange(nlev, dtype=T_env.dtype)
    levels = jnp.broadcast_to(levels, T_env.shape)
    window_below_lfc = jax.nn.sigmoid(
        indicator_sharpness * (levels - (k_lfc_smooth[:, None] + 0.5))
    )
    window_above_lcl = jax.nn.sigmoid(
        indicator_sharpness * (k_lcl_smooth[:, None] - 0.5 - levels)
    )
    window = window_below_lfc * window_above_lcl
    dp = p_half[:, 1:] - p_half[:, :-1]
    inhibiting_buoyancy = jnp.maximum(0.0, T_env - T_parcel_ma)
    return constants.R_d * jnp.sum(window * inhibiting_buoyancy * dp / p_full, axis=-1)


def entraining_detraining_plume(
    T_env, q_v_env, p_full, p_half, z_full,
    T_parcel_base, q_parcel_base, k_base_smooth,
    epsilon_profile, delta_profile, M_b,
    *, buoyancy_sharpness: float = 0.5,
):
    ncol, nlev = T_env.shape
    _dtype = T_env.dtype

    T_env_rev = T_env[:, ::-1].astype(_dtype)
    q_v_env_rev = q_v_env[:, ::-1].astype(_dtype)
    p_full_rev = p_full[:, ::-1].astype(_dtype)
    z_full_rev = z_full[:, ::-1].astype(_dtype)
    eps_rev = epsilon_profile[:, ::-1].astype(_dtype)
    del_rev = delta_profile[:, ::-1].astype(_dtype)

    k_rev = jnp.arange(nlev, dtype=T_env.dtype)
    k_rev = jnp.broadcast_to(k_rev, T_env.shape)
    k_base_rev = (nlev - 1.0) - k_base_smooth
    above_base_weight = jax.nn.sigmoid(buoyancy_sharpness * (k_rev - k_base_rev[:, None]))

    init_carry = (
        T_parcel_base.astype(_dtype),
        q_parcel_base.astype(_dtype),
        jnp.zeros_like(T_parcel_base, dtype=_dtype),
        M_b.astype(_dtype),
        z_full_rev[:, 0],
    )
    inputs = (
        jnp.moveaxis(T_env_rev, 1, 0),
        jnp.moveaxis(q_v_env_rev, 1, 0),
        jnp.moveaxis(p_full_rev, 1, 0),
        jnp.moveaxis(z_full_rev, 1, 0),
        jnp.moveaxis(eps_rev, 1, 0),
        jnp.moveaxis(del_rev, 1, 0),
        jnp.moveaxis(above_base_weight, 1, 0),
    )

    g = constants.g
    c_pd = constants.c_pd
    L_v = constants.L_v

    def step(carry, layer_inputs):
        T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
        T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs

        dz = jnp.maximum(z_e - z_prev, 1.0)

        # EXACT integration of dM/dz = (eps - dlt) M
        M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)

        T_u_ent = T_u_prev + eps * dz * (T_e - T_u_prev)
        q_u_ent = q_u_prev + eps * dz * (q_e - q_u_prev)

        rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
        T_u = T_u_ent + dT_dz * dz

        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
        q_u = (q_u_ent - condensate).astype(_dtype)

        # ENTRAINMENT DILUTION of q_c_u (NEW FIX)
        q_c_u_ent = jnp.maximum(q_c_u_prev * (1.0 - eps * dz), 0.0)
        q_c_u = (q_c_u_ent + condensate).astype(_dtype)

        T_u = T_u.astype(_dtype)
        B_u = T_u - T_e

        plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
        M_u_reported = M_u_raw * plume_alive * abv

        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
        outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
        return new_carry, outputs

    _, scan_out = jax.lax.scan(step, init_carry, inputs)
    T_u_rev, q_u_rev, q_c_u_rev, M_u_rev, B_u_rev = scan_out
    T_u = jnp.moveaxis(T_u_rev, 0, 1)[:, ::-1]
    q_u = jnp.moveaxis(q_u_rev, 0, 1)[:, ::-1]
    q_c_u = jnp.moveaxis(q_c_u_rev, 0, 1)[:, ::-1]
    M_u = jnp.moveaxis(M_u_rev, 0, 1)[:, ::-1]
    B_u = jnp.moveaxis(B_u_rev, 0, 1)[:, ::-1]
    return Plume(M_u=M_u, T_u=T_u, q_u=q_u, q_c_u=q_c_u, B_u=B_u)
```

## Source 2: convection/tiedtke.py (key sections)

```python
def tiedtke_convection(T, q_v, p_full, p_half, u, v, conv_prog_profile, dt, config, moisture_convergence=None):
    ncol, nlev = T.shape
    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
    T_base, q_base, p_base = T[:, -1], q_v[:, -1], p_full[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)

    T_parcel = T_base + config.parcel_dT
    q_parcel = q_base + config.parcel_dq
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)

    levels = jnp.arange(nlev, dtype=T.dtype)
    weight_lcl = jax.nn.softmax(-2.0 * (levels[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1)
    weight_lnb = jax.nn.softmax(-2.0 * (levels[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1)
    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)

    deep_weight = smooth_step(cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness)
    shallow_weight = smooth_step(config.cloud_depth_shallow_max - cloud_depth, config.depth_split_sharpness)
    midlevel_weight = jnp.clip(1.0 - deep_weight - shallow_weight, 0.0, 1.0)

    eps_per_class = (deep_weight[:, None] * config.epsilon_deep +
                     shallow_weight[:, None] * config.epsilon_shallow +
                     midlevel_weight[:, None] * config.epsilon_midlevel)
    dlt_per_class = (deep_weight[:, None] * config.delta_deep +
                     shallow_weight[:, None] * config.delta_shallow +
                     midlevel_weight[:, None] * config.delta_midlevel)
    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)

    q_sat_env = saturation_mixing_ratio(T, p_full)
    dp = p_half[:, 1:] - p_half[:, :-1]
    if moisture_convergence is not None:
        column_MC = jnp.sum(moisture_convergence * dp, axis=-1) / constants.g
        column_MC_proxy = jnp.maximum(column_MC, 0.0)
    else:
        # SATURATION EXCESS (NEW FIX) — positive in moist columns
        sat_excess = jnp.maximum(q_v - config.mc_proxy_RH_crit * q_sat_env, 0.0)
        column_MC_proxy = jnp.sum(sat_excess * dp, axis=-1) / (constants.g * config.tau_MC_proxy)
    mc_gate = smooth_positive_part(
        column_MC_proxy - config.moisture_convergence_threshold,
        config.moisture_convergence_sharpness,
    )

    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
    M_b_deep = mc_gate * cape_weight
    M_b_shallow = (cape_weight * rho_BL *
                   smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
                   / (constants.g * config.tau_shallow_M_b))
    M_b_midlevel = M_b_shallow * config.midlevel_M_b_fraction
    M_b = (deep_weight * M_b_deep + shallow_weight * M_b_shallow + midlevel_weight * M_b_midlevel)
    M_b = jnp.clip(M_b, 0.0, config.M_b_max)

    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z, T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b,
    )

    # IMPLICIT EULER FIX
    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, 1e-30)
    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
    M_u_for_kernel = M_u_new

    delta_0_eff = (deep_weight * config.delta_deep +
                   shallow_weight * config.delta_shallow +
                   midlevel_weight * config.delta_midlevel)
    # Pass per-column delta_0_eff to kernel (FIX: only detrainment scales with delta)
    dT_dt, dq_v_dt, _ = _apply_mass_flux_kernel(
        T, q_v, p_full, plume.T_u, plume.q_u, plume.q_c_u, M_u_for_kernel,
        z, rho, delta_0_eff[:, None], M_u_max=config.M_b_max,
    )
    rho_safe = jnp.clip(rho, 0.01, None)
    p_gate_qc = stratosphere_mass_flux_gate(p_full)
    dq_c_conv_dt = (delta_0_eff[:, None] * M_u_for_kernel * p_gate_qc * plume.q_c_u / rho_safe)

    if config.enable_downdraft:
        levels = jnp.arange(nlev, dtype=T.dtype)
        below_lcl = jax.nn.sigmoid(2.0 * (levels[None, :] - k_lcl_smooth[:, None]))
        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
        below_mass = jnp.sum(below_lcl * dp, axis=-1) + 1e-6
        rh_below = jnp.sum(below_lcl * rh_layer * dp, axis=-1) / below_mass
        downdraft_trigger = jax.nn.sigmoid(10.0 * (config.downdraft_RH_min - rh_below))
        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger

        # COLUMN-WATER CONSERVING DOWNDRAFT EVAP (NEW FIX)
        below_lcl_mass = jnp.sum(below_lcl * dp, axis=-1, keepdims=True).clip(1e-6, None)
        rain_source_total = jnp.sum(jnp.maximum(dq_c_conv_dt, 0.0) * dp, axis=-1) / constants.g
        evap_total = jnp.minimum(jnp.abs(M_d_base) * config.downdraft_evap_efficiency, rain_source_total)
        evap_rate = (evap_total[:, None] * below_lcl * constants.g / below_lcl_mass)
        dT_dt_dd = -(constants.L_v / constants.c_pd) * evap_rate
        dT_dt = dT_dt + dT_dt_dd
        dq_v_dt = dq_v_dt + evap_rate
        rain_source_safe = jnp.clip(rain_source_total[:, None], 1e-30, None)
        rain_scale = 1.0 - evap_total[:, None] / rain_source_safe
        dq_c_conv_dt = jnp.where(dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt)

    if config.enable_cmt:
        if config.enable_downdraft:
            M_d = -config.downdraft_alpha * M_u_for_kernel * 0.3
        else:
            M_d = None
        du_dt_conv, dv_dt_conv = cmt_gregory_1997(u, v, M_u_for_kernel, M_d, p_full, p_half, rho,
                                                   c_u=config.cmt_c_u, c_d=config.cmt_c_d)
    else:
        du_dt_conv = dv_dt_conv = None

    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
    out = ConvectionOutput(
        dT_dt=dT_dt, dq_v_dt=dq_v_dt, dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
        cape=cape, convective_mask=convective_mask,
        du_dt_conv=du_dt_conv, dv_dt_conv=dv_dt_conv,
    )
    return out, M_u_new
```

## Source 3: microphysics/_warm_rain.py (full)

```python
"""Shared warm-rain microphysics helpers."""
import jax
import jax.numpy as jnp
from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio


def safe_pow(x, p):
    """Differentiable x ** p with grad=0 wherever x <= 0."""
    positive = x > 0.0
    safe_x = jnp.where(positive, x, 1.0)
    return jnp.where(positive, safe_x ** p, 0.0)


def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    return condensation, q_sat


def effective_Nc(N_c, Nc_0):
    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))


def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
    q_c_pos = jnp.clip(q_c, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
    dN_r_au = dq_c_au * rho / (x_star * 20.0)
    return dq_c_au, dN_r_au, x_c


def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm


def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
    D_r_arg = (
        jnp.clip(q_r, 0.0) * rho
        / jnp.clip(N_r, 1.0)
        / (jnp.pi / 6.0 * constants.rho_water)
    )
    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
    dN_r_br = -dN_r_sc * breakup_frac
    return dN_r_sc, dN_r_br


def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)
```

## Source 4: microphysics/morrison.py (key tendency assembly)

```python
def morrison_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config):
    # ... (warm rain, ice phase, donor clamps, sedimentation as before)

    # Donor clamp on q_c sinks
    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = jnp.minimum(
        1.0,
        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
    )
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
    dN_r_au = dN_r_au * qc_scale  # NEW: number-tendency scales with mass

    # Marshall-Palmer fall speeds with safe_pow
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)

    # Latent heating with L_f for Bergeron + riming, melt minus L_f
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd
    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        + L_s * dq_i_dep / c_pd
        + L_f * (bergeron + riming_i + riming_s) / c_pd
        - L_f * (melt_ice + melt_snow) / c_pd
    )

    # Combine tendencies
    dq_v_dt = -condensation + evaporation - dq_i_dep
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
    dq_s_dt = aggregation + riming_s - melt_snow + sed_s
```

## Specific questions for the reviewer

1. **Plume q_c_u dilution**: `q_c_u_ent = max(q_c_u_prev * (1 - eps*dz), 0)` — for `eps*dz > 1` (corner case: deep entrainment in thick layers), this clip kills the gradient. Should it use `q_c_u_prev * exp(-eps*dz)` instead (always non-negative, fully differentiable)?

2. **Plume mass flux exact integration**: `M_u_raw = M_u_raw_prev * exp((eps - dlt) * dz)` — correct for constant `(eps - dlt)` over a layer. But the carry `M_u_raw_prev` does NOT include the buoyancy taper (`plume_alive`) or sub-cloud taper (`abv`); those are reporting filters only. Is this the right design choice? An alternative would be to apply the tapers in the carry so the plume "remembers" being killed by buoyancy.

3. **Tiedtke downdraft `where` redundancy**: `dq_c_conv_dt = jnp.where(dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt)` — given that `dq_c_conv_dt = delta_0_eff * M_u_for_kernel * p_gate * q_c_u / rho_safe` is non-negative by construction, the `where` is redundant. Confirm this is harmless (no AD issues).

4. **Tiedtke MC proxy**: when `q_v < RH_crit * q_sat`, `sat_excess = 0` and `column_MC_proxy = 0`. The threshold-crossing gradient at `q_v = RH_crit * q_sat` flows through the smooth `mc_gate` sigmoid only — is the smoothness adequate?

5. **Bechtold mass-weighted PBL parcel**: `pbl_norm.clip(1e-6, None)` floors against zero. For a column where `pbl_weight × dp_full = 0` everywhere (column with negative z?), what happens? Edge case worth investigating.

6. **Morrison number-tendency scaling**: `dN_r_au = dN_r_au * qc_scale` is documented as "Number tendency for autoconverted droplets must scale identically." But the original `dN_r_au = dq_c_au * rho / (x_star * 20.0)` already scales with `dq_c_au`. Multiplying by `qc_scale` again — does this double-scale the number tendency relative to mass?

7. **safe_pow at exponent p < 1**: the `where(positive, safe_x ** p, 0.0)` correctly returns 0 at x=0 in the forward pass. For the backward pass, the gradient is `p * safe_x^(p-1)` masked by `positive`, which is non-finite when `p < 1` and `safe_x = 1` → bounded. But what about `safe_x = 0` going into the second `where`? Verify this branch produces a finite gradient (the placeholder `1.0` should make it `p * 1.0^(p-1) = p`, which is finite).

8. **Morrison/Thompson `dq_v_dt = -condensation + evaporation - dq_i_dep`**: missing terms? Bergeron, riming convert q_c → ice (no q_v change); melting converts ice → q_r (no q_v change); aggregation is internal to the ice phase. Confirm this expression is complete.

9. **Microphysics `dN_i_dt = dN_i_nuc - aggregation * N_i / q_i`**: the aggregation tendency in number space uses `aggregation * N_i / q_i` (loss of number proportional to mass loss × current number). At `q_i → 0` the divisor is clipped to `1e-15`, but the numerator `aggregation = agg_coeff * q_i * f_ice` also goes to zero, so the ratio stays finite. Confirm.

10. **Hines drag `clip(drag, 0, Fmax)` near saturation transition**: the clip prevents "anti-drag" leak. But it's a hard clip — gradient is zero in the negative region. Is this acceptable for AD-based GWD parameter tuning?

Please review all sources and answer these questions, then provide a concise summary of any new bugs found.
