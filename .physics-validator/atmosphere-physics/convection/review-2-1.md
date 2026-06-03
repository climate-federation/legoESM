Reading prompt from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019de5bd-f6ae-7821-b252-36c14701abf3
--------
user
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

codex
I’ll read the live files rather than relying only on the snippets, then I’ll tie any findings to exact line numbers and separate true bugs from candidate concerns.
exec
/bin/zsh -lc 'git status --short' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-JCvDCUCY' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-4KCk7brc' (errno=Operation not permitted)
 M src/legoesm/atmosphere/physics/convection/_plume.py
 M src/legoesm/atmosphere/physics/convection/bechtold.py
 M src/legoesm/atmosphere/physics/convection/config.py
 M src/legoesm/atmosphere/physics/convection/emanuel.py
 M src/legoesm/atmosphere/physics/convection/kain_fritsch.py
 M src/legoesm/atmosphere/physics/convection/kuo.py
 M src/legoesm/atmosphere/physics/convection/mass_flux.py
 M src/legoesm/atmosphere/physics/convection/tiedtke.py
 M src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/config.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py
 M src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
 M src/legoesm/atmosphere/physics/microphysics/kessler.py
 M src/legoesm/atmosphere/physics/microphysics/morrison.py
 M src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
 M src/legoesm/atmosphere/physics/microphysics/sundqvist.py
 M src/legoesm/atmosphere/physics/microphysics/thompson.py
 M src/legoesm/atmosphere/physics/ml_parameterization.py
 M src/legoesm/atmosphere/physics/radiation/solar.py
 M src/legoesm/atmosphere/physics/thermodynamics.py
 M src/legoesm/atmosphere/physics/turbulence/config.py
 M src/legoesm/atmosphere/physics/turbulence/edmf.py
 M src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py
 M src/legoesm/ml/physics/data.py
 M src/legoesm/thermo.py
 M tests/unit/test_bechtold.py
 M tests/unit/test_convection_plume.py
 M tests/unit/test_diff_atmosphere_physics.py
 M tests/unit/test_ml_physics_parameterization.py
 M tests/unit/test_physics_convection.py
 M tests/unit/test_physics_gwd.py
 M tests/unit/test_physics_microphysics.py
 M tests/unit/test_tiedtke.py
 M tests/unit/test_zhang_mcfarlane.py
?? .claude/agents/physics-validator.md
?? .physics-validator/

exec
/bin/zsh -lc "rg -n \"q_c_u_ent|M_u_raw|dq_c_conv_dt|column_MC_proxy|sat_excess|dN_r_au|safe_pow|dN_i_dt|aggregation \\*|clip\\(drag|drag\" legoesm/atmosphere/physics -S" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 2 in 0ms:
rg: legoesm/atmosphere/physics: IO error for operation on legoesm/atmosphere/physics: No such file or directory (os error 2)

exec
/bin/zsh -lc 'rg --files legoesm/atmosphere/physics | rg "(convection/_plume.py|convection/tiedtke.py|convection/bechtold.py|microphysics/_warm_rain.py|microphysics/morrison.py|microphysics/thompson.py|hines|gravity|gwd)"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
rg: legoesm/atmosphere/physics: IO error for operation on legoesm/atmosphere/physics: No such file or directory (os error 2)

codex
The repo uses `src/legoesm/...`; I found the modified files there. I’m narrowing on the high-risk paths and will include the live line numbers from those files.
exec
/bin/zsh -lc "rg -n \"q_c_u_ent|M_u_raw|dq_c_conv_dt|column_MC_proxy|sat_excess|dN_r_au|safe_pow|dN_i_dt|aggregation \\*|clip\\(drag|jnp.clip\\(drag|drag\" src/legoesm/atmosphere/physics -S" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/ml_parameterization.py:264:        dN_i_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:360:        dN_i_dt=zeros,
src/legoesm/atmosphere/physics/__init__.py:19:from legoesm.atmosphere.physics.gravity_wave_drag import (
src/legoesm/atmosphere/physics/physics_state.py:66:        Gravity wave drag wave action spectrum for the prognostic
src/legoesm/atmosphere/physics/physics_state.py:151:    gwd_cfg = physics_config.gravity_wave_drag
src/legoesm/atmosphere/physics/convection/output.py:6:convective source for cloud water (``dq_c_conv_dt``). Downstream the
src/legoesm/atmosphere/physics/convection/output.py:40:    dq_c_conv_dt : jax.Array
src/legoesm/atmosphere/physics/convection/output.py:66:    dq_c_conv_dt: jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:1:"""Gravity wave drag output container.
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:17:    """Output from a gravity wave drag scheme (backend-agnostic).
src/legoesm/atmosphere/physics/neural_physics.py:49:    "dN_i_dt",
src/legoesm/atmosphere/physics/convection/tiedtke.py:183:        column_MC_proxy = jnp.maximum(column_MC, 0.0)
src/legoesm/atmosphere/physics/convection/tiedtke.py:195:        sat_excess = jnp.maximum(
src/legoesm/atmosphere/physics/convection/tiedtke.py:198:        column_MC_proxy = (
src/legoesm/atmosphere/physics/convection/tiedtke.py:199:            jnp.sum(sat_excess * dp, axis=-1)
src/legoesm/atmosphere/physics/convection/tiedtke.py:204:        column_MC_proxy - config.moisture_convergence_threshold,
src/legoesm/atmosphere/physics/convection/tiedtke.py:282:    dq_c_conv_dt = (
src/legoesm/atmosphere/physics/convection/tiedtke.py:311:        # ``dq_c_conv_dt`` (the cloud-water source that microphysics
src/legoesm/atmosphere/physics/convection/tiedtke.py:331:            jnp.maximum(dq_c_conv_dt, 0.0) * dp, axis=-1,
src/legoesm/atmosphere/physics/convection/tiedtke.py:334:        # convective rain so dq_c_conv_dt stays non-negative after the
src/legoesm/atmosphere/physics/convection/tiedtke.py:354:        dq_c_conv_dt = jnp.where(
src/legoesm/atmosphere/physics/convection/tiedtke.py:355:            dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/tiedtke.py:379:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/combined.py:5:microphysics, and **gravity wave drag** tendencies. Each sub-module
src/legoesm/atmosphere/physics/combined.py:9:Gravity wave drag is included as a first-class component on equal
src/legoesm/atmosphere/physics/combined.py:32:...     gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"),
src/legoesm/atmosphere/physics/combined.py:54:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
src/legoesm/atmosphere/physics/combined.py:68:from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
src/legoesm/atmosphere/physics/combined.py:93:    gravity_wave_drag : GravityWaveDragConfig
src/legoesm/atmosphere/physics/combined.py:94:        Gravity wave drag configuration (schemes: "rayleigh", "lindzen",
src/legoesm/atmosphere/physics/combined.py:101:    gravity_wave_drag: GravityWaveDragConfig = GravityWaveDragConfig()
src/legoesm/atmosphere/physics/combined.py:164:    if config.gravity_wave_drag.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:165:        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, model_type, dt), True, "gwd_spectrum"))
src/legoesm/atmosphere/physics/combined.py:305:    if config.gravity_wave_drag.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:306:        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "nonhydrostatic", dt), True, "gwd_spectrum"))
src/legoesm/atmosphere/physics/combined.py:418:    if config.gravity_wave_drag.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:419:        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt), True, "gwd_spectrum"))
src/legoesm/atmosphere/physics/microphysics/kessler.py:23:from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow
src/legoesm/atmosphere/physics/microphysics/kessler.py:93:    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
src/legoesm/atmosphere/physics/microphysics/kessler.py:94:    accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)
src/legoesm/atmosphere/physics/microphysics/kessler.py:98:    evaporation = config.evaporation_coeff * subsaturation * safe_pow(q_r, 0.525)
src/legoesm/atmosphere/physics/microphysics/kessler.py:135:        dN_i_dt=z,
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:1:"""Hines (1997) Doppler-spread gravity wave drag parameterization.
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:20:from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:84:    # and biased the drag deposition lower in the column (audit GWD-B2).
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:123:        # the floor a small "anti-drag" leak can appear in the transition
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:124:        # region (``sigma_new`` slightly larger than ``sigma_grown`` ⇒ drag
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127:        drag = (sigma_grown - sigma_new) * rho[:, k]
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:128:        drag = jnp.clip(drag, 0.0, config.Fmax)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:130:        return sigma_new, drag
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:135:    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:136:    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:139:    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
src/legoesm/atmosphere/physics/microphysics/output.py:41:    dN_i_dt: jax.Array        # ice number tendency
src/legoesm/atmosphere/physics/microphysics/output.py:76:        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
src/legoesm/atmosphere/physics/convection/sbm.py:146:    # dq_c_conv_dt = 0 everywhere, mirroring the legacy
src/legoesm/atmosphere/physics/convection/sbm.py:154:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/sbm.py:161:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/microphysics/thompson.py:30:    safe_pow,
src/legoesm/atmosphere/physics/microphysics/thompson.py:91:    dq_c_au, dN_r_au, x_c = autoconversion_sb(
src/legoesm/atmosphere/physics/microphysics/thompson.py:123:        * safe_pow(N_i, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:179:    dN_r_au = dN_r_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:183:    # [0.25, 0.5]); guard the AD path with safe_pow.
src/legoesm/atmosphere/physics/microphysics/thompson.py:186:    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
src/legoesm/atmosphere/physics/microphysics/thompson.py:188:    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
src/legoesm/atmosphere/physics/microphysics/thompson.py:190:    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
src/legoesm/atmosphere/physics/microphysics/thompson.py:192:    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
src/legoesm/atmosphere/physics/microphysics/thompson.py:225:    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
src/legoesm/atmosphere/physics/microphysics/thompson.py:226:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/thompson.py:245:        dN_i_dt=dN_i_dt,
src/legoesm/atmosphere/physics/microphysics/integration.py:412:            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
src/legoesm/atmosphere/physics/microphysics/integration.py:473:        "N_i": "dN_i_dt",
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:1:"""Rayleigh friction gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:3:Simplest GWD parameterization: applies linear drag proportional to wind
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:6:Follows the same pattern as held_suarez.py for sigma-based drag profiles.
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:15:from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:16:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:64:    # Boundary layer drag: ramps from 0 at sigma_b to k_max at surface
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:75:    # Combined drag coefficient
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:76:    k_drag = k_bl + k_sponge
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:79:    du_dt = -k_drag * u
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:80:    dv_dt = -k_drag * v
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:35:    safe_pow,
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:73:    dq_c_au, dN_r_au, x_c = autoconversion_sb(
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:89:    # has fractional exponent (b_v_r=0.5); guard the AD path with safe_pow.
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:91:    V_t_r = config.a_v_r * safe_pow(
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:105:    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:124:        dN_i_dt=z,
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:195:        dN_i_dt=z,
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py:121:        dN_i_dt=z,
src/legoesm/atmosphere/physics/convection/bechtold.py:292:    dq_c_conv_dt = (
src/legoesm/atmosphere/physics/convection/bechtold.py:314:        # from ``dq_c_conv_dt`` so the column water budget closes
src/legoesm/atmosphere/physics/convection/bechtold.py:320:            jnp.maximum(dq_c_conv_dt, 0.0) * dp_full, axis=-1,
src/legoesm/atmosphere/physics/convection/bechtold.py:334:        dq_c_conv_dt = jnp.where(
src/legoesm/atmosphere/physics/convection/bechtold.py:335:            dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/bechtold.py:358:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:4:heat fluxes using neutral drag and transfer coefficients.
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:1:"""Prognostic spectral gravity wave drag parameterization.
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:145:        drag_deposit = (F_carry - F_new) / jnp.clip(dp_flat[:, k], 1.0, None)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:146:        return F_new, drag_deposit
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:148:    _, drag_stack = jax.lax.scan(scan_fn, F_init, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:149:    # drag_stack: (nlev, ncol*n_az*n_wn)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:150:    drag_4d = drag_stack.T.reshape(ncol, n_az, n_wn, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:151:    drag_4d = drag_4d[:, :, :, ::-1]  # reverse to top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:155:    # drag is stress gradient -> acceleration = -drag_deposit (already divided by dp)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:156:    # Convert from dp-based to dz-based: multiply by dp/(rho*dz) -> just -drag
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:157:    du_dt_spec = -drag_4d * cos_az[None, :, None, None]  # (ncol, n_az, n_wn, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:158:    dv_dt_spec = -drag_4d * sin_az[None, :, None, None]
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:1:"""Smoothed Lindzen (1981) orographic gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:19:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:99:        drag = tau_carry - tau_new
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:100:        return tau_new, drag
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:102:    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:103:    # drag_stack: (nlev, ncol) — reverse to get (ncol, nlev) top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:104:    drag_all = drag_stack.T  # (ncol, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:105:    drag_all = drag_all[:, ::-1]  # back to top-first ordering
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:107:    # Convert stress deposit to tendency: drag / (rho * dz)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:110:    accel = -drag_all / (jnp.clip(rho * dz, 1e-10, None))
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:16:def safe_pow(x, p):
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:123:    dN_r_au : array
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:132:    dN_r_au = dq_c_au * rho / (x_star * 20.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:133:    return dq_c_au, dN_r_au, x_c
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:184:    # Cube-root has unbounded derivative at zero — guard with safe_pow.
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:190:    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:216:    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:217:    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:1:"""ML gravity wave drag emulator using Equinox MLP.
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:17:from legoesm.atmosphere.physics.gravity_wave_drag.config import MLEmulatorConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:201:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:224:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:1:"""Model integration bridge for gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:40:from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:41:from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:42:from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:43:from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:44:from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:47:from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:119:            "Gravity wave drag on MPAS Voronoi mesh is not yet "
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:123:            "MPAS with gravity_wave_drag='none'."
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:1:"""Gravity wave drag parameterization for legoESM.
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:4:1. **Rayleigh**: simple Rayleigh friction drag
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:18:>>> from legoesm.atmosphere.physics.gravity_wave_drag import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:26:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:35:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:36:from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:37:from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:38:from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:40:from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:43:from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:47:from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:1:"""Configuration for gravity wave drag schemes.
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:4:1. Rayleigh: simple Rayleigh friction drag
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:17:  wave drag on the general circulation of the lower stratosphere and
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:31:    """Configuration for Rayleigh friction drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:36:        Maximum drag coefficient [1/s] (default 1/(1*86400)).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:42:        Upper sponge drag coefficient [1/s] (default 1/(0.5*86400)).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:216:    """Top-level gravity wave drag configuration.
src/legoesm/atmosphere/physics/convection/emanuel.py:191:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/emanuel.py:197:    # below.  ``dq_c_conv_dt`` from the kernel is already enhanced by
src/legoesm/atmosphere/physics/convection/emanuel.py:203:    dq_c_conv_dt_raw = dq_c_conv_dt / jnp.maximum(sort_multiplier, 1e-30)
src/legoesm/atmosphere/physics/convection/emanuel.py:220:        column_condensate = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
src/legoesm/atmosphere/physics/convection/emanuel.py:233:        # cloud water is *produced* (i.e. dq_c_conv_dt_raw), not where
src/legoesm/atmosphere/physics/convection/emanuel.py:235:        # ``evap_rate`` from ``dq_c_conv_dt`` *at the BL*, then clipped
src/legoesm/atmosphere/physics/convection/emanuel.py:237:        # ``dq_c_conv_dt_raw``) and effectively created vapor from
src/legoesm/atmosphere/physics/convection/emanuel.py:247:        # ``dq_c_conv_dt ≥ 0`` and column water conserved.
src/legoesm/atmosphere/physics/convection/emanuel.py:248:        col_dq_c = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
src/legoesm/atmosphere/physics/convection/emanuel.py:249:        weight = dq_c_conv_dt_raw / jnp.maximum(col_dq_c[:, None], 1e-12)
src/legoesm/atmosphere/physics/convection/emanuel.py:258:        dq_c_conv_dt = dq_c_conv_dt - subtract
src/legoesm/atmosphere/physics/convection/emanuel.py:263:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:1:"""Smoothed McFarlane (1987) orographic gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:10:  wave drag on the general circulation of the lower stratosphere and
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:20:from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:89:    # ``10⁴`` (numerically) → clip truncated to 10 → drag ~1e-21 m/s²
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:134:        drag = tau_carry - tau_k
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:135:        return tau_k, drag
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:137:    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:138:    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:143:    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
src/legoesm/atmosphere/physics/convection/dca.py:248:    # rescaling so that ∫ dq_c_conv_dt dp/g equals the column-net
src/legoesm/atmosphere/physics/convection/dca.py:263:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/dca.py:273:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/microphysics/morrison.py:30:    safe_pow,
src/legoesm/atmosphere/physics/microphysics/morrison.py:76:    dq_c_au, dN_r_au, x_c = autoconversion_sb(
src/legoesm/atmosphere/physics/microphysics/morrison.py:102:        * safe_pow(N_i, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:152:    dN_r_au = dN_r_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:157:    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
src/legoesm/atmosphere/physics/microphysics/morrison.py:160:    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
src/legoesm/atmosphere/physics/microphysics/morrison.py:162:    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
src/legoesm/atmosphere/physics/microphysics/morrison.py:164:    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
src/legoesm/atmosphere/physics/microphysics/morrison.py:200:    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
src/legoesm/atmosphere/physics/microphysics/morrison.py:201:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/morrison.py:222:        dN_i_dt=dN_i_dt,
src/legoesm/atmosphere/physics/turbulence/config.py:44:        Neutral drag coefficient (default 1.5e-3).
src/legoesm/atmosphere/physics/convection/_plume.py:511:        T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
src/legoesm/atmosphere/physics/convection/_plume.py:533:        M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)
src/legoesm/atmosphere/physics/convection/_plume.py:575:        # Explicit Euler: ``q_c_u_ent = q_c_u_prev * (1 - eps·dz)``
src/legoesm/atmosphere/physics/convection/_plume.py:578:        q_c_u_ent = jnp.maximum(q_c_u_prev * (1.0 - eps * dz), 0.0)
src/legoesm/atmosphere/physics/convection/_plume.py:579:        q_c_u = (q_c_u_ent + condensate).astype(_dtype)
src/legoesm/atmosphere/physics/convection/_plume.py:590:        M_u_reported = M_u_raw * plume_alive * abv
src/legoesm/atmosphere/physics/convection/_plume.py:592:        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
src/legoesm/atmosphere/physics/convection/kuo.py:12:5. Emit per-level cloud-water source (``dq_c_conv_dt``) from the
src/legoesm/atmosphere/physics/convection/kuo.py:124:    # ``dq_c_conv_dt = 0`` whenever MC = 0 — no spurious heating,
src/legoesm/atmosphere/physics/convection/kuo.py:139:    # ``dq_c_conv_dt`` (see step 7). Fraction ``(1 - alpha_heat)``
src/legoesm/atmosphere/physics/convection/kuo.py:152:    # directly via ``dq_c_conv_dt`` so it can process the convective
src/legoesm/atmosphere/physics/convection/kuo.py:211:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/kuo.py:221:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:181:    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
src/legoesm/atmosphere/physics/convection/mass_flux.py:197:    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
src/legoesm/atmosphere/physics/convection/mass_flux.py:205:    ``dq_c_conv_dt`` through its full chain (autoconversion,
src/legoesm/atmosphere/physics/convection/mass_flux.py:240:    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
src/legoesm/atmosphere/physics/convection/mass_flux.py:241:    return dT_dt, dq_v_dt, dq_c_conv_dt
src/legoesm/atmosphere/physics/convection/mass_flux.py:340:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/mass_flux.py:357:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:461:    # water contributes ``-q_c`` (the loaded condensate is mass drag,
src/legoesm/atmosphere/physics/convection/mass_flux.py:480:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/mass_flux.py:497:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:221:    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:230:    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:235:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/config.py:338:    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
src/legoesm/atmosphere/physics/convection/integration.py:491:        # Convective detrained condensate (``dq_c_conv_dt``) feeds the
src/legoesm/atmosphere/physics/convection/integration.py:499:            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:506:                    data=dq_c_conv_dt, name="dq_c_conv_dt",
src/legoesm/atmosphere/physics/convection/integration.py:851:            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:852:            dtracers = dtracers.at[..., 1].set(dq_c_conv_dt)
src/legoesm/atmosphere/physics/convection/integration.py:1190:                dq_c_dt_grid = conv_out.dq_c_conv_dt.reshape(n_lat, n_lon, nlev)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/tiedtke.py | sed -n '1,360p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/_warm_rain.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_plume.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Shared warm-rain microphysics helpers.
     2	
     3	Functions here are used by multiple microphysics backends (Seifert-Beheng,
     4	Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import jax
    10	import jax.numpy as jnp
    11	
    12	from legoesm import constants
    13	from legoesm.thermo import saturation_mixing_ratio
    14	
    15	
    16	def safe_pow(x, p):
    17	    """Differentiable ``x ** p`` with grad=0 wherever ``x <= 0``.
    18	
    19	    Microphysics has many Marshall-Palmer-style fractional powers of
    20	    hydrometeor mixing ratios (``q_r``, ``q_i``, ``q_s``, ``q_g``,
    21	    ``N_i``, …) with exponents in (0, 1) — typically 0.5 for fall
    22	    speeds, 1/3 for diameters, 0.525/0.875 for ventilation/accretion.
    23	    Their analytic derivative ``p * x**(p-1)`` is unbounded at ``x=0``
    24	    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
    25	    returns ``inf`` (at zero) or ``nan`` (at negatives) under
    26	    ``jax.grad``, breaking AD on cold-start (no-precip) initial
    27	    conditions.
    28	
    29	    The double-where pattern below routes the AD graph through a
    30	    placeholder of 1.0 in the inactive branch so the gradient never
    31	    sees ``0**(p-1)``.
    32	
    33	    Parameters
    34	    ----------
    35	    x : array
    36	        Argument of the power.  May be zero or negative.
    37	    p : float or array
    38	        Exponent.  Intended for ``0 < p < 1`` where the bug applies;
    39	        also safe for ``p >= 1``.
    40	
    41	    Returns
    42	    -------
    43	    array
    44	        ``x ** p`` for ``x > 0``, else 0; gradient is finite
    45	        everywhere.
    46	    """
    47	    positive = x > 0.0
    48	    safe_x = jnp.where(positive, x, 1.0)
    49	    return jnp.where(positive, safe_x ** p, 0.0)
    50	
    51	
    52	def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    53	    """Compute smooth saturation adjustment (condensation tendency).
    54	
    55	    Parameters
    56	    ----------
    57	    T : array (ncol, nlev)
    58	        Temperature [K].
    59	    q_v : array (ncol, nlev)
    60	        Water vapor mixing ratio [kg/kg].
    61	    p_full : array (ncol, nlev)
    62	        Pressure [Pa].
    63	    dt : float
    64	        Time step [s].
    65	    sharpness : float
    66	        Sigmoid sharpness for smooth condensation switch.
    67	
    68	    Returns
    69	    -------
    70	    condensation : array (ncol, nlev)
    71	        Condensation tendency [kg/kg/s].
    72	    q_sat : array (ncol, nlev)
    73	        Saturation mixing ratio [kg/kg].
    74	    """
    75	    q_sat = saturation_mixing_ratio(T, p_full)
    76	    excess = q_v - q_sat
    77	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    78	    condensation = cond_frac * excess / dt
    79	    return condensation, q_sat
    80	
    81	
    82	def effective_Nc(N_c, Nc_0):
    83	    """Use config default cloud droplet number where N_c is zero.
    84	
    85	    Parameters
    86	    ----------
    87	    N_c : array
    88	        Cloud droplet number concentration [1/kg].
    89	    Nc_0 : float
    90	        Default cloud droplet number.
    91	
    92	    Returns
    93	    -------
    94	    array : Effective N_c.
    95	    """
    96	    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))
    97	
    98	
    99	def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
   100	    """Seifert-Beheng mass-dependent autoconversion.
   101	
   102	    Parameters
   103	    ----------
   104	    q_c : array
   105	        Cloud water mixing ratio [kg/kg].
   106	    N_c_eff : array
   107	        Effective cloud droplet number [1/kg].
   108	    rho : array
   109	        Air density [kg/m3].
   110	    k_au : float
   111	        Autoconversion rate constant.
   112	    x_star : float
   113	        Mean droplet mass threshold [kg].
   114	    sharpness : float
   115	        Sigmoid sharpness.
   116	    gamma_norm : float
   117	        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).
   118	
   119	    Returns
   120	    -------
   121	    dq_c_au : array
   122	        Cloud water autoconversion rate [kg/kg/s].
   123	    dN_r_au : array
   124	        Rain number formation rate [1/kg/s].
   125	    x_c : array
   126	        Mean cloud droplet mass [kg].
   127	    """
   128	    q_c_pos = jnp.clip(q_c, 0.0)
   129	    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
   130	    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
   131	    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
   132	    dN_r_au = dq_c_au * rho / (x_star * 20.0)
   133	    return dq_c_au, dN_r_au, x_c
   134	
   135	
   136	def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
   137	    """Rain collecting cloud water (accretion).
   138	
   139	    Parameters
   140	    ----------
   141	    q_c, q_r : array
   142	        Cloud water and rain mixing ratios [kg/kg].
   143	    rho : array
   144	        Air density [kg/m3].
   145	    k_ac : float
   146	        Accretion rate constant.
   147	    gamma_norm : float
   148	        Gamma distribution correction.
   149	
   150	    Returns
   151	    -------
   152	    array : Accretion rate [kg/kg/s].
   153	    """
   154	    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm
   155	
   156	
   157	def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
   158	    """Self-collection and breakup of rain drops.
   159	
   160	    Parameters
   161	    ----------
   162	    N_r : array
   163	        Rain drop number concentration [1/kg].
   164	    q_r : array
   165	        Rain mixing ratio [kg/kg].
   166	    rho : array
   167	        Air density [kg/m3].
   168	    k_sc : float
   169	        Self-collection rate constant.
   170	    breakup_sharpness : float
   171	        Sigmoid sharpness for breakup onset.
   172	    D_eq : float
   173	        Equilibrium drop diameter [m].
   174	
   175	    Returns
   176	    -------
   177	    dN_r_sc : array
   178	        Self-collection tendency [1/kg/s].
   179	    dN_r_br : array
   180	        Breakup tendency [1/kg/s].
   181	    """
   182	    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
   183	    # Mean drop diameter D ~ (q_r * rho / N_r / (pi/6 * rho_water))^(1/3).
   184	    # Cube-root has unbounded derivative at zero — guard with safe_pow.
   185	    D_r_arg = (
   186	        jnp.clip(q_r, 0.0) * rho
   187	        / jnp.clip(N_r, 1.0)
   188	        / (jnp.pi / 6.0 * constants.rho_water)
   189	    )
   190	    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
   191	    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
   192	    dN_r_br = -dN_r_sc * breakup_frac
   193	    return dN_r_sc, dN_r_br
   194	
   195	
   196	def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
   197	    """Compute rain evaporation in subsaturated air.
   198	
   199	    Parameters
   200	    ----------
   201	    q_v : array
   202	        Water vapor mixing ratio [kg/kg].
   203	    q_r : array
   204	        Rain mixing ratio [kg/kg].
   205	    q_sat : array
   206	        Saturation mixing ratio [kg/kg].
   207	    evap_coeff : float
   208	        Evaporation rate coefficient.
   209	
   210	    Returns
   211	    -------
   212	    array : Evaporation rate [kg/kg/s].
   213	    """
   214	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
   215	    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
   216	    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
   217	    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)

 succeeded in 0ms:
     1	"""Column physics primitives shared across convection schemes.
     2	
     3	Smooth, fully-differentiable helpers consumed by Zhang-McFarlane (PR 1),
     4	Kain-Fritsch (PR 2), Emanuel (PR 3), Tiedtke (PR 4), and Bechtold/IFS
     5	(PR 5).  The goal of this module is to provide a small set of
     6	well-tested column workhorses so that no convection scheme has to
     7	re-implement an LCL, LFC, plume integrator, or CMT closure.
     8	
     9	This module pairs with :mod:`._triggers` (sigmoid math) — together they
    10	are the only place these primitives live.
    11	
    12	What's in this PR (PR 0):
    13	
    14	* :func:`compute_lcl`           — Bolton (1980) lifting condensation level.
    15	* :func:`compute_lfc_lnb`       — smooth fractional levels of free
    16	                                  convection and neutral buoyancy.
    17	* :func:`compute_cin`           — CIN [J/kg] from the buoyancy profile.
    18	* :func:`entraining_detraining_plume` — vmappable updraft integrator
    19	                                  (``jax.lax.scan`` over levels).
    20	* :func:`cmt_gregory_1997`      — Gregory et al. 1997 convective
    21	                                  momentum transport closure.
    22	
    23	Deferred to later scheme PRs (each lands alongside its first consumer):
    24	
    25	* ``diagnose_grid_w_from_omega``     — Kain-Fritsch trigger (PR 2).
    26	* ``buoyancy_sort_emanuel``           — Emanuel mixing ensemble (PR 3).
    27	* ``downdraft_thermo``                — Tiedtke / Bechtold downdrafts
    28	                                         (PRs 4–5).
    29	
    30	This deferral was a pragmatic scope reduction within PR 0; each helper
    31	is small (~50–200 LOC) and is documented in
    32	``add-more-complex-convection-fluttering-quasar.md`` under its scheme.
    33	
    34	Conventions
    35	-----------
    36	* Column shape ``(ncol, nlev)``.  **Surface at the last index**
    37	  ``[:, -1]``; the model top is at ``[:, 0]``.  Pressure ``p_full``
    38	  *increases* with the array index (TOA ≪ surface).
    39	* Half levels ``p_half`` have shape ``(ncol, nlev+1)`` with ``p_half[:, 0]``
    40	  at TOA and ``p_half[:, -1]`` at the surface.
    41	* All temperatures in Kelvin, pressures in Pa, mass flux in kg/m²/s.
    42	* Reused upstream — never re-implement: ``legoesm.constants`` for
    43	  physical constants; ``legoesm.thermo.saturation_mixing_ratio`` and
    44	  ``saturation_vapor_pressure`` for thermodynamics; the ``moist_adiabat``
    45	  / ``moist_adiabat_lapse_rate`` / ``compute_cape`` family from
    46	  ``legoesm.atmosphere.physics.thermodynamics``; column geometry
    47	  helpers (``compute_heights_from_sigma``, ``compute_layer_dz``,
    48	  ``compute_rho``) from ``legoesm.atmosphere.physics._shared``.
    49	
    50	References
    51	----------
    52	* Bolton, D. (1980). The computation of equivalent potential
    53	  temperature.  *Mon. Wea. Rev.*, 108(7), 1046–1053.
    54	* Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
    55	  momentum transport by convection. II: Tests in single-column and
    56	  general circulation models.  *Quart. J. Roy. Meteor. Soc.*, 123,
    57	  1153–1183.
    58	* Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
    59	  simulations to the parameterization of cumulus convection in the
    60	  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
    61	  33(3), 407–446.
    62	"""
    63	
    64	from __future__ import annotations
    65	
    66	from typing import NamedTuple
    67	
    68	import jax
    69	import jax.numpy as jnp
    70	
    71	from legoesm import constants
    72	from legoesm.thermo import (
    73	    saturation_mixing_ratio,
    74	    saturation_vapor_pressure,
    75	)
    76	from legoesm.atmosphere.physics.thermodynamics import (
    77	    compute_cape,
    78	    compute_moist_adiabat,
    79	    moist_adiabat_lapse_rate,
    80	)
    81	
    82	from legoesm.atmosphere.physics.convection._triggers import (
    83	    smooth_level_indicator,
    84	    smooth_lowest_crossing_index,
    85	    smooth_step,
    86	)
    87	
    88	__all__ = (
    89	    "LCL",
    90	    "Plume",
    91	    "compute_lcl",
    92	    "compute_lfc_lnb",
    93	    "compute_cin",
    94	    "entraining_detraining_plume",
    95	    "cmt_gregory_1997",
    96	)
    97	
    98	
    99	# ---------------------------------------------------------------------------
   100	# Lifting condensation level (Bolton 1980)
   101	# ---------------------------------------------------------------------------
   102	
   103	class LCL(NamedTuple):
   104	    """LCL diagnostics for one column.
   105	
   106	    Fields
   107	    ------
   108	    p_lcl : jax.Array, shape (ncol,)
   109	        LCL pressure [Pa].
   110	    T_lcl : jax.Array, shape (ncol,)
   111	        LCL temperature [K].
   112	    k_lcl_smooth : jax.Array, shape (ncol,)
   113	        Smooth fractional level index (surface-last convention) at
   114	        which ``p_full`` crosses ``p_lcl``.  Used by downstream
   115	        helpers that need to localize the LCL on the model grid.
   116	    """
   117	    p_lcl: jax.Array
   118	    T_lcl: jax.Array
   119	    k_lcl_smooth: jax.Array
   120	
   121	
   122	def compute_lcl(
   123	    T_parcel: jax.Array,
   124	    q_parcel: jax.Array,
   125	    p_parcel: jax.Array,
   126	    p_full: jax.Array,
   127	    *,
   128	    crossing_sharpness: float = 0.001,
   129	) -> LCL:
   130	    """Lifting condensation level via Bolton (1980) Eq. 22.
   131	
   132	    Bolton's empirical formula::
   133	
   134	        T_LCL = 1 / [ 1/(T - 55) - ln(RH)/2840 ] + 55
   135	
   136	    where ``T`` is parcel temperature [K], ``RH`` is relative humidity
   137	    in ``(0, 1]``, and the result is the LCL temperature [K].  The
   138	    LCL pressure follows from Poisson's equation along a dry adiabat
   139	    from parcel level::
   140	
   141	        p_LCL = p * (T_LCL / T)^(c_pd / R_d)
   142	
   143	    Parameters
   144	    ----------
   145	    T_parcel : jax.Array, shape (ncol,)
   146	        Parcel temperature at the launch level [K].
   147	    q_parcel : jax.Array, shape (ncol,)
   148	        Parcel water-vapor specific humidity [kg/kg].
   149	    p_parcel : jax.Array, shape (ncol,)
   150	        Parcel launch pressure [Pa] (typically the lowest model
   151	        full-level pressure).
   152	    p_full : jax.Array, shape (ncol, nlev)
   153	        Full-level pressure [Pa] for the column, surface-last.  Used
   154	        only to compute ``k_lcl_smooth``.
   155	    crossing_sharpness : float
   156	        Sharpness for the soft fractional level diagnosis.  Units of
   157	        [1/Pa]; ``0.001`` sharpens to ~95%/5% transition over a
   158	        ~1000 Pa pressure range, which is finer than typical model
   159	        layer thickness in the boundary layer.
   160	
   161	    Returns
   162	    -------
   163	    LCL
   164	        ``p_lcl, T_lcl, k_lcl_smooth``.
   165	    """
   166	    # Vapor pressure and relative humidity at parcel level.
   167	    e_sat = saturation_vapor_pressure(T_parcel)
   168	    q_sat = saturation_mixing_ratio(T_parcel, p_parcel)
   169	    # RH = q / q_sat, clipped to (0, 1] so log is well-defined and
   170	    # supersaturated parcels produce LCL at parcel level.
   171	    RH = jnp.clip(q_parcel / jnp.maximum(q_sat, 1e-12), 1e-4, 1.0)
   172	
   173	    # Bolton (1980) Eq. 22.  ``T - 55`` floored to avoid singularity
   174	    # at very cold parcels (defensively — convective parcels are rarely
   175	    # below 200 K, but the formula is sensitive in pathological cases).
   176	    T_minus_55 = jnp.maximum(T_parcel - 55.0, 1.0)
   177	    T_lcl = 1.0 / (1.0 / T_minus_55 - jnp.log(RH) / 2840.0) + 55.0
   178	
   179	    # Poisson: dry-adiabatic descent from parcel to LCL.
   180	    p_lcl = p_parcel * (T_lcl / T_parcel) ** (constants.c_pd / constants.R_d)
   181	
   182	    # Soft fractional level index where p_full = p_lcl.  Pressure
   183	    # *decreases* with altitude (surface-first), so we apply
   184	    # smooth_lowest_crossing_index to negated pressure to convert the
   185	    # downward-with-altitude crossing into an upward one.
   186	    k_lcl_smooth = smooth_lowest_crossing_index(
   187	        -p_full, -p_lcl[:, None], crossing_sharpness,
   188	    )
   189	
   190	    return LCL(p_lcl=p_lcl, T_lcl=T_lcl, k_lcl_smooth=k_lcl_smooth)
   191	
   192	
   193	# ---------------------------------------------------------------------------
   194	# LFC and LNB
   195	# ---------------------------------------------------------------------------
   196	
   197	def compute_lfc_lnb(
   198	    T_env: jax.Array,
   199	    T_parcel_ma: jax.Array,
   200	    *,
   201	    sharpness: float = 1.0,
   202	) -> tuple[jax.Array, jax.Array]:
   203	    """Smooth fractional levels of free convection and neutral buoyancy.
   204	
   205	    The buoyancy proxy is ``T_parcel_ma - T_env`` (positive where the
   206	    parcel is warmer than the environment); the LFC is the lowest
   207	    level where this turns from negative to positive going upward,
   208	    and the LNB is the lowest level above the LFC where the proxy
   209	    turns back from positive to negative.
   210	
   211	    Both are returned as smooth fractional indices in surface-last
   212	    convention (``ncol, nlev`` indexing).  For columns with no clear
   213	    crossing the no-crossing fallback in
   214	    :func:`._triggers.smooth_lowest_crossing_index` returns a value
   215	    near the surface (LFC) or near the top (LNB).
   216	
   217	    Parameters
   218	    ----------
   219	    T_env : jax.Array, shape (ncol, nlev)
   220	        Environmental temperature [K].
   221	    T_parcel_ma : jax.Array, shape (ncol, nlev)
   222	        Moist-adiabatic parcel temperature [K] from
   223	        :func:`legoesm.atmosphere.physics.thermodynamics.compute_moist_adiabat`.
   224	    sharpness : float
   225	        Sigmoid sharpness in [1/K] on the buoyancy threshold.
   226	
   227	    Returns
   228	    -------
   229	    k_lfc_smooth, k_lnb_smooth : jax.Array, shape (ncol,)
   230	        Smooth fractional level indices.
   231	    """
   232	    buoyancy = T_parcel_ma - T_env
   233	    nlev = buoyancy.shape[-1]
   234	
   235	    # LFC: lowest UPWARD crossing of buoyancy = 0.
   236	    k_lfc = smooth_lowest_crossing_index(buoyancy, 0.0, sharpness)
   237	
   238	    # LNB: lowest UPWARD crossing of negative-buoyancy.  Equivalently,
   239	    # the lowest level (above the LFC) where the parcel turns from
   240	    # positive to negative buoyancy.  Implementation: apply the same
   241	    # primitive to ``-buoyancy`` AFTER the LFC is reached; the soft
   242	    # gating-by-LFC weight ensures we don't pick up sub-cloud layers
   243	    # where the parcel was negatively buoyant.
   244	    #
   245	    # Convention: ``profile`` and ``k_lfc`` are both surface-last indices
   246	    # (surface at index ``nlev-1``, top at index 0).  "Above LFC altitude"
   247	    # means *smaller* surface-last index than ``k_lfc``.  ``direction="below"``
   248	    # in ``smooth_level_indicator`` gives ``sigmoid(threshold - profile)``
   249	    # which is ~1 where ``profile < threshold`` — so the threshold IS
   250	    # ``k_lfc`` itself.  The earlier formula ``(nlev - 1) - k_lfc`` flipped
   251	    # ``k_lfc`` into surface-first space and then compared against the
   252	    # surface-last ``profile``, marking the wrong levels as "above LFC"
   253	    # and collapsing the LNB onto the LFC for many columns.
   254	    above_lfc_weight = smooth_level_indicator(
   255	        jnp.broadcast_to(jnp.arange(nlev, dtype=buoyancy.dtype), buoyancy.shape),
   256	        threshold=k_lfc[:, None],
   257	        sharpness=sharpness,
   258	        direction="below",
   259	    )
   260	    # Drive ``-buoyancy`` strongly negative below the LFC so that the

 succeeded in 0ms:
     1	"""Tiedtke (1989) bulk mass-flux convection.
     2	
     3	The IFS heritage scheme — three-class soft assignment of cloud type
     4	(deep / mid-level / shallow) blended on cloud depth, downdraft with
     5	RH-dependent trigger, and convective momentum transport via Gregory
     6	et al. 1997.  This is the **first scheme that exercises the
     7	``conv_prog_profile = M_u(k)`` profile carry**: the diagnosed
     8	per-level updraft mass flux relaxes via implicit Euler toward the
     9	new diagnosis on every step, smoothing fast oscillations.
    10	
    11	Smooth-everywhere implementation:
    12	
    13	* Three-class blend on cloud depth — sigmoid weights, not hard
    14	  threshold.
    15	* Downdraft RH trigger — sigmoid on (downdraft_RH_min - column_RH).
    16	* CAPE gate — sigmoid via :func:`._triggers.cape_trigger`.
    17	* Moisture-convergence proxy — saturation-deficit
    18	  ``MC_proxy = max(q_sat - q_v, 0) / tau_MC_proxy`` (a placeholder
    19	  until the PR-0 ``compute_moisture_convergence`` diagnostic ships).
    20	* Plume integrator's mass-flux profile is gated by buoyancy sigmoid.
    21	
    22	References
    23	----------
    24	* Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
    25	  parameterization in large-scale models.  *Mon. Wea. Rev.*, 117,
    26	  1779–1800.
    27	* Gregory, D., et al. (1997). Parametrization of momentum transport
    28	  by convection. II.  *Quart. J. Roy. Meteor. Soc.*, 123, 1153–1183.
    29	"""
    30	
    31	from __future__ import annotations
    32	
    33	import jax
    34	import jax.numpy as jnp
    35	
    36	from legoesm import constants
    37	from legoesm.thermo import saturation_mixing_ratio
    38	from legoesm.atmosphere.physics.thermodynamics import (
    39	    compute_cape,
    40	    compute_moist_adiabat,
    41	)
    42	from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
    43	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    44	from legoesm.atmosphere.physics.convection.mass_flux import (
    45	    _apply_mass_flux_kernel,
    46	    stratosphere_mass_flux_gate,
    47	    _compute_column_geometry,
    48	)
    49	from legoesm.atmosphere.physics.convection._triggers import (
    50	    cape_trigger,
    51	    smooth_positive_part,
    52	    smooth_step,
    53	)
    54	from legoesm.atmosphere.physics.convection._plume import (
    55	    cmt_gregory_1997,
    56	    compute_lcl,
    57	    compute_lfc_lnb,
    58	    entraining_detraining_plume,
    59	)
    60	
    61	
    62	__all__ = ("tiedtke_convection",)
    63	
    64	
    65	def tiedtke_convection(
    66	    T: jax.Array,
    67	    q_v: jax.Array,
    68	    p_full: jax.Array,
    69	    p_half: jax.Array,
    70	    u: jax.Array,
    71	    v: jax.Array,
    72	    conv_prog_profile: jax.Array,
    73	    dt: float,
    74	    config: TiedtkeConfig = TiedtkeConfig(),
    75	    moisture_convergence: jax.Array | None = None,
    76	) -> tuple[ConvectionOutput, jax.Array]:
    77	    """Tiedtke (1989) convection (smooth, differentiable).
    78	
    79	    Parameters
    80	    ----------
    81	    T : jax.Array, shape (ncol, nlev)
    82	        Environmental temperature [K].
    83	    q_v : jax.Array, shape (ncol, nlev)
    84	        Water-vapor specific humidity [kg/kg].
    85	    p_full, p_half : jax.Array
    86	        Full / half-level pressures [Pa].
    87	    u, v : jax.Array, shape (ncol, nlev)
    88	        Environmental wind components [m/s] for the CMT closure.
    89	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    90	        Updraft mass-flux profile from the previous time step.
    91	        Tiedtke is the first scheme that uses this as a real per-level
    92	        carry — relaxation is implicit Euler toward
    93	        ``M_u_diagnosed`` over ``tau_M_u_relax`` seconds.
    94	    dt : float
    95	        Time step [s].
    96	    config : TiedtkeConfig
    97	        Scheme tunables.
    98	
    99	    Returns
   100	    -------
   101	    out : ConvectionOutput
   102	        Tendencies on T, q_v, q_c plus CAPE diagnostic.  CMT
   103	        ``du_dt_conv`` / ``dv_dt_conv`` populated when
   104	        ``config.enable_cmt`` is ``True``.
   105	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   106	        Implicit-Euler-relaxed ``M_u(k)`` profile.
   107	    """
   108	    ncol, nlev = T.shape
   109	
   110	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   111	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   112	    T_base = T[:, -1]
   113	    q_base = q_v[:, -1]
   114	    p_base = p_full[:, -1]
   115	    T_moist = compute_moist_adiabat(T_base, p_full)
   116	    cape = compute_cape(T, T_moist, p_full, p_half)
   117	    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
   118	
   119	    # -- LCL, LFC, LNB diagnostics -----------------------------------------
   120	    T_parcel = T_base + config.parcel_dT
   121	    q_parcel = q_base + config.parcel_dq
   122	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   123	    k_lcl_smooth = lcl.k_lcl_smooth
   124	    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
   125	
   126	    # Cloud depth: smooth interpolation of z at fractional indices.
   127	    levels = jnp.arange(nlev, dtype=T.dtype)
   128	    weight_lcl = jax.nn.softmax(
   129	        -2.0 * (levels[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
   130	    )
   131	    weight_lnb = jax.nn.softmax(
   132	        -2.0 * (levels[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
   133	    )
   134	    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
   135	    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
   136	    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)
   137	
   138	    # -- Three-class soft assignment ---------------------------------------
   139	    # deep_weight rises with cloud depth; shallow_weight falls with it.
   140	    # mid-level fills the gap.
   141	    deep_weight = smooth_step(
   142	        cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness,
   143	    )
   144	    shallow_weight = smooth_step(
   145	        config.cloud_depth_shallow_max - cloud_depth,
   146	        config.depth_split_sharpness,
   147	    )
   148	    midlevel_weight = jnp.clip(
   149	        1.0 - deep_weight - shallow_weight, 0.0, 1.0
   150	    )
   151	
   152	    # -- Per-class entrainment / detrainment profiles ----------------------
   153	    eps_per_class = (
   154	        deep_weight[:, None] * config.epsilon_deep
   155	        + shallow_weight[:, None] * config.epsilon_shallow
   156	        + midlevel_weight[:, None] * config.epsilon_midlevel
   157	    )
   158	    dlt_per_class = (
   159	        deep_weight[:, None] * config.delta_deep
   160	        + shallow_weight[:, None] * config.delta_shallow
   161	        + midlevel_weight[:, None] * config.delta_midlevel
   162	    )
   163	    # Broadcast to (ncol, nlev) — entrainment is constant over the
   164	    # column for a given class blend.
   165	    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
   166	    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)
   167	
   168	    # -- Closure: deep uses moisture convergence; shallow and midlevel
   169	    # use a CAPE-relaxation closure.  Combined per-column closure is a
   170	    # class-weighted blend.  When the bridge supplies a real
   171	    # ``moisture_convergence`` array (from
   172	    # ``_shared.compute_moisture_convergence``, available for cubed-
   173	    # sphere and lat-lon dycores), use it directly; otherwise fall
   174	    # back to a saturation-deficit proxy that is qualitatively similar
   175	    # (positive in moist columns, vanishing in dry ones).
   176	    q_sat_env = saturation_mixing_ratio(T, p_full)
   177	    dp = p_half[:, 1:] - p_half[:, :-1]
   178	    if moisture_convergence is not None:
   179	        # Real MC — column-integrate per (kg/m^2/s).
   180	        column_MC = (
   181	            jnp.sum(moisture_convergence * dp, axis=-1) / constants.g
   182	        )
   183	        column_MC_proxy = jnp.maximum(column_MC, 0.0)
   184	    else:
   185	        # Saturation-EXCESS proxy: vapor in excess of RH_crit * q_sat,
   186	        # column-integrated and divided by a relaxation timescale.
   187	        # Positive in moist columns (q_v > RH_crit * q_sat), vanishing
   188	        # in dry ones — this is the qualitative signature of moisture
   189	        # convergence over a long timescale.
   190	        #
   191	        # The earlier formulation used the saturation DEFICIT
   192	        # ``max(q_sat - q_v, 0)`` which has the opposite sign: large
   193	        # in dry columns, vanishing in moist columns — convection
   194	        # would be suppressed exactly where it should fire.
   195	        sat_excess = jnp.maximum(
   196	            q_v - config.mc_proxy_RH_crit * q_sat_env, 0.0,
   197	        )
   198	        column_MC_proxy = (
   199	            jnp.sum(sat_excess * dp, axis=-1)
   200	            / (constants.g * config.tau_MC_proxy)
   201	        )
   202	    # Smooth gate on MC threshold for deep.
   203	    mc_gate = smooth_positive_part(
   204	        column_MC_proxy - config.moisture_convergence_threshold,
   205	        config.moisture_convergence_sharpness,
   206	    )
   207	
   208	    # Cloud-base mass flux (per class, then blended).
   209	    # ``M_b_deep`` is driven by column moisture convergence (kg/m^2/s units
   210	    # — already dimensionally correct) and stays as-is.
   211	    # ``M_b_shallow`` and ``M_b_midlevel`` use the dimensionally-correct
   212	    # CAPE-relaxation closure (Kain 2004 §3 form):
   213	    #     M_b = rho_BL * (CAPE - threshold)+ / (g * tau)   [kg/m^2/s]
   214	    # The earlier formula omitted ``rho_BL`` and ``g``; magnitude was
   215	    # masked operationally only by ``M_b_max``.
   216	    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
   217	    M_b_deep = mc_gate * cape_weight
   218	    M_b_shallow = (
   219	        cape_weight
   220	        * rho_BL
   221	        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
   222	        / (constants.g * config.tau_shallow_M_b)
   223	    )
   224	    M_b_midlevel = M_b_shallow * config.midlevel_M_b_fraction
   225	    M_b = (
   226	        deep_weight * M_b_deep
   227	        + shallow_weight * M_b_shallow
   228	        + midlevel_weight * M_b_midlevel
   229	    )
   230	    # See ZhangMcFarlaneConfig.M_b_max.
   231	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   232	
   233	    # -- Plume integration -------------------------------------------------
   234	    plume = entraining_detraining_plume(
   235	        T, q_v, p_full, p_half, z,
   236	        T_parcel, q_parcel, k_lcl_smooth,
   237	        eps_profile, dlt_profile, M_b,
   238	    )
   239	
   240	    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
   241	    # ``dt / tau`` (with ``tau`` floored against zero), NOT
   242	    # ``dt / max(tau, dt)`` — see ZM for the audit context.
   243	    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, 1e-30)
   244	    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
   245	    # Cap M_u_new at config.M_b_max so every downstream use (kernel
   246	    # tendencies, dq_c_conv_raw, downdraft trigger, CMT, carry update)
   247	    # sees the same bounded value.
   248	    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
   249	
   250	    # Use the relaxed M_u for the actual environmental tendencies — this
   251	    # smooths the time evolution of the convective forcing.
   252	    M_u_for_kernel = M_u_new
   253	
   254	    # -- Environmental tendencies ----------------------------------------
   255	    # Build an effective per-class delta_0 for the kernel and the
   256	    # cloud-water source.
   257	    delta_0_eff = (
   258	        deep_weight * config.delta_deep
   259	        + shallow_weight * config.delta_shallow
   260	        + midlevel_weight * config.delta_midlevel
   261	    )
   262	    # Pass the per-column blended delta_0 directly to the kernel.
   263	    # ``_apply_mass_flux_kernel`` uses ``delta_0`` ONLY in the
   264	    # detrainment terms (``delta_0 * M * (T_u - T) / rho``,
   265	    # ``delta_0 * M * (q_v_u - q_v) / rho``); the compensating-subsidence
   266	    # contributions are independent of ``delta_0``.  The earlier
   267	    # implementation called the kernel with ``config.delta_deep`` and
   268	    # then multiplied the FULL kernel output by ``delta_0_eff /
   269	    # delta_deep``, which incorrectly rescaled subsidence too — in a
   270	    # shallow-only column with ``delta_shallow > delta_deep`` this
   271	    # over-amplifies the subsidence drying / warming by the same factor
   272	    # the detrainment is enhanced.
   273	    dT_dt, dq_v_dt, _ = _apply_mass_flux_kernel(
   274	        T, q_v, p_full,
   275	        plume.T_u, plume.q_u, plume.q_c_u, M_u_for_kernel,
   276	        z, rho, delta_0_eff[:, None], M_u_max=config.M_b_max,
   277	    )
   278	    rho_safe = jnp.clip(rho, 0.01, None)
   279	    # Reuse the same stratospheric gate the kernel applies so this
   280	    # custom q_c path does not detrain condensate above the tropopause.
   281	    p_gate_qc = stratosphere_mass_flux_gate(p_full)
   282	    dq_c_conv_dt = (
   283	        delta_0_eff[:, None] * M_u_for_kernel * p_gate_qc * plume.q_c_u / rho_safe
   284	    )
   285	
   286	    # -- Optional downdraft (RH-dependent trigger) -------------------------
   287	    if config.enable_downdraft:
   288	        # Column-mean RH below LCL.
   289	        levels = jnp.arange(nlev, dtype=T.dtype)
   290	        below_lcl = jax.nn.sigmoid(
   291	            2.0 * (levels[None, :] - k_lcl_smooth[:, None])
   292	        )
   293	        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
   294	        below_mass = jnp.sum(below_lcl * dp, axis=-1) + 1e-6
   295	        rh_below = (
   296	            jnp.sum(below_lcl * rh_layer * dp, axis=-1) / below_mass
   297	        )
   298	        downdraft_trigger = jax.nn.sigmoid(
   299	            10.0 * (config.downdraft_RH_min - rh_below)
   300	        )
   301	        # Downdraft mass flux = -alpha * M_b at cloud base [kg/(m²·s)].
   302	        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
   303	        # Subcloud rain-evaporation cooling — dimensionally consistent,
   304	        # locally AND column-water conserving.
   305	        #
   306	        # Physical model: a fraction ``downdraft_evap_efficiency`` of the
   307	        # downdraft mass flux re-evaporates as rain falls through the
   308	        # subcloud layer.  Mass conservation requires that re-evaporated
   309	        # water be drawn from the same convective rain source that would
   310	        # otherwise reach the surface — implemented by reducing
   311	        # ``dq_c_conv_dt`` (the cloud-water source that microphysics
   312	        # converts to surface precip) by the same column-integrated rate
   313	        # that appears as a vapor source.  Capping ``evap_total`` at the
   314	        # available rain rate guarantees we never extract more rain than
   315	        # was generated this step.
   316	        #
   317	        # Earlier formulations were broken in two ways: (1) a literal
   318	        # ``0.05`` divided by ``rho_safe`` only — units came out as
   319	        # K·m/s not K/s; (2) cooling was added to dT_dt with no matching
   320	        # dq_v source, then in the next iteration the dq_v source was
   321	        # added but with no matching dq_c_conv sink — the column water
   322	        # budget gained mass every step (audit GWD/convection: "downdraft
   323	        # `0.05` cooling — dimensionally wrong AND non-water-conserving";
   324	        # Codex stop-time review: "downdraft fix still creates column
   325	        # water").
   326	        below_lcl_mass = jnp.sum(below_lcl * dp, axis=-1, keepdims=True).clip(1e-6, None)
   327	        # Column-integrated convective rain source available this step
   328	        # [kg/(m²·s)] (positive part — cloud water is generated where
   329	        # M_u detrains, never destroyed by this term).
   330	        rain_source_total = jnp.sum(
   331	            jnp.maximum(dq_c_conv_dt, 0.0) * dp, axis=-1,
   332	        ) / constants.g
   333	        # Total downdraft evap mass flux [kg/(m²·s)], capped at available
   334	        # convective rain so dq_c_conv_dt stays non-negative after the
   335	        # correction below.
   336	        evap_total = jnp.minimum(
   337	            jnp.abs(M_d_base) * config.downdraft_evap_efficiency,
   338	            rain_source_total,
   339	        )
   340	        # Per-layer evap rate [kg/(kg·s)], mass-weighted over below-LCL.
   341	        evap_rate = (
   342	            evap_total[:, None] * below_lcl * constants.g / below_lcl_mass
   343	        )
   344	        dT_dt_dd = -(constants.L_v / constants.c_pd) * evap_rate
   345	        dT_dt = dT_dt + dT_dt_dd
   346	        # Local water source: rain → vapor in subcloud layer.
   347	        dq_v_dt = dq_v_dt + evap_rate
   348	        # Column conservation: subtract the same column-integrated rate
   349	        # from the convective cloud-water source (proportional scaling
   350	        # over levels where it is positive).  Net column ∫(dq_v + dq_c)
   351	        # contribution from this term is then zero.
   352	        rain_source_safe = jnp.clip(rain_source_total[:, None], 1e-30, None)
   353	        rain_scale = 1.0 - evap_total[:, None] / rain_source_safe
   354	        dq_c_conv_dt = jnp.where(
   355	            dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
   356	        )
   357	
   358	    # -- CMT --------------------------------------------------------------
   359	    if config.enable_cmt:
   360	        if config.enable_downdraft:

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/tiedtke.py | sed -n '360,430p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_plume.py | sed -n '260,660p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   360	        if config.enable_downdraft:
   361	            M_d = -config.downdraft_alpha * M_u_for_kernel * 0.3
   362	        else:
   363	            M_d = None
   364	        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
   365	            u, v, M_u_for_kernel, M_d,
   366	            p_full, p_half, rho,
   367	            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
   368	        )
   369	    else:
   370	        du_dt_conv = None
   371	        dv_dt_conv = None
   372	
   373	    # -- Convective mask ---------------------------------------------------
   374	    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
   375	
   376	    out = ConvectionOutput(
   377	        dT_dt=dT_dt,
   378	        dq_v_dt=dq_v_dt,
   379	        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
   380	        cape=cape,
   381	        convective_mask=convective_mask,
   382	        du_dt_conv=du_dt_conv,
   383	        dv_dt_conv=dv_dt_conv,
   384	    )
   385	    return out, M_u_new

 succeeded in 0ms:
   260	    # Drive ``-buoyancy`` strongly negative below the LFC so that the
   261	    # smooth-crossing primitive does not register near-zero spurious
   262	    # crossings produced by multiplicative masking ``buoyancy * weight``
   263	    # (where ``-masked_buoyancy ≈ 0`` for an entire stretch of levels
   264	    # near the surface).  ``LARGE = 1e6 K`` is far below any physical
   265	    # buoyancy magnitude (which is bounded by the moist-adiabat /
   266	    # environmental temperature difference, ~tens of K), so a smooth
   267	    # blend is safe.  Above LFC altitude the offset vanishes and the
   268	    # crossing detector sees the genuine ``-buoyancy`` profile.
   269	    LARGE = jnp.asarray(1.0e6, dtype=buoyancy.dtype)
   270	    guarded_neg_buoyancy = -buoyancy - LARGE * (1.0 - above_lfc_weight)
   271	    k_lnb = smooth_lowest_crossing_index(guarded_neg_buoyancy, 0.0, sharpness)
   272	    return k_lfc, k_lnb
   273	
   274	
   275	# ---------------------------------------------------------------------------
   276	# CIN
   277	# ---------------------------------------------------------------------------
   278	
   279	def compute_cin(
   280	    T_env: jax.Array,
   281	    T_parcel_ma: jax.Array,
   282	    p_full: jax.Array,
   283	    p_half: jax.Array,
   284	    k_lcl_smooth: jax.Array,
   285	    k_lfc_smooth: jax.Array,
   286	    *,
   287	    indicator_sharpness: float = 1.0,
   288	) -> jax.Array:
   289	    """Convective Inhibition (CIN) [J/kg].
   290	
   291	    Integrates the negative buoyancy between the LCL and the LFC::
   292	
   293	        CIN = R_d * ∫_{LCL}^{LFC} max(0, T_env - T_parcel) * dp/p
   294	
   295	    The bounds of integration are encoded as a smooth window in
   296	    surface-last index space: ``window[k] = above_LCL(k) * below_LFC(k)``,
   297	    each factor a sigmoid on the level index relative to the
   298	    fractional indices.  Smooth bounds preserve gradients.
   299	
   300	    Convention check: for a positively-CAPE column where the parcel
   301	    is warmer than the environment between the LCL and the LFC,
   302	    ``T_env - T_parcel`` is negative and the ``max(., 0)`` clamp gives
   303	    zero.  CIN therefore counts only the genuinely-inhibiting layers.
   304	
   305	    Parameters
   306	    ----------
   307	    T_env, T_parcel_ma : jax.Array, shape (ncol, nlev)
   308	        Environmental and moist-adiabatic parcel temperatures [K].
   309	    p_full, p_half : jax.Array
   310	        Full-level (``ncol, nlev``) and half-level (``ncol, nlev+1``)
   311	        pressures [Pa].
   312	    k_lcl_smooth, k_lfc_smooth : jax.Array, shape (ncol,)
   313	        Smooth fractional level indices from :func:`compute_lcl` and
   314	        :func:`compute_lfc_lnb` (surface-last convention).
   315	    indicator_sharpness : float
   316	        Sigmoid sharpness on the level-window bounds, in [1/level].
   317	
   318	    Returns
   319	    -------
   320	    jax.Array, shape (ncol,)
   321	        CIN [J/kg].  Non-negative.
   322	    """
   323	    nlev = T_env.shape[-1]
   324	    levels = jnp.arange(nlev, dtype=T_env.dtype)
   325	    levels = jnp.broadcast_to(levels, T_env.shape)
   326	
   327	    # The integration window covers altitudes BETWEEN the LCL and the
   328	    # LFC.  In surface-last index space (surface at the LARGEST index)
   329	    # the LCL has a larger index than the LFC, so the CIN layer is at
   330	    # indices ``k_lfc < k < k_lcl``.  The smooth window is the product
   331	    # of two sigmoids: ``below LFC altitude`` (index larger than
   332	    # ``k_lfc``) AND ``above LCL altitude`` (index smaller than
   333	    # ``k_lcl``).  Earlier this product was the WRONG intersection
   334	    # (``above_LFC AND below_LCL``) which is empty for the natural
   335	    # ordering ``k_lnb < k_lfc < k_lcl`` — CIN was suppressed by ~93%
   336	    # in straightforward test columns.
   337	    window_below_lfc = jax.nn.sigmoid(
   338	        indicator_sharpness * (levels - (k_lfc_smooth[:, None] + 0.5))
   339	    )  # 1 at indices below LFC altitude (larger index), 0 above.
   340	    window_above_lcl = jax.nn.sigmoid(
   341	        indicator_sharpness * (k_lcl_smooth[:, None] - 0.5 - levels)
   342	    )  # 1 at indices above LCL altitude (smaller index), 0 below.
   343	    window = window_below_lfc * window_above_lcl
   344	
   345	    dp = p_half[:, 1:] - p_half[:, :-1]
   346	    inhibiting_buoyancy = jnp.maximum(0.0, T_env - T_parcel_ma)
   347	
   348	    return constants.R_d * jnp.sum(window * inhibiting_buoyancy * dp / p_full, axis=-1)
   349	
   350	
   351	# ---------------------------------------------------------------------------
   352	# Entraining-detraining updraft plume
   353	# ---------------------------------------------------------------------------
   354	
   355	class Plume(NamedTuple):
   356	    """Output of :func:`entraining_detraining_plume`.
   357	
   358	    All arrays are surface-last with shape ``(ncol, nlev)``.
   359	
   360	    Fields
   361	    ------
   362	    M_u : jax.Array
   363	        Updraft mass flux [kg/m²/s].  Non-negative; zero below the
   364	        cloud base and at/above the level of neutral buoyancy.
   365	    T_u : jax.Array
   366	        Updraft temperature [K].
   367	    q_u : jax.Array
   368	        Updraft water-vapor specific humidity [kg/kg].
   369	    q_c_u : jax.Array
   370	        Updraft cloud-water mixing ratio [kg/kg].  Non-negative.
   371	    B_u : jax.Array
   372	        Updraft buoyancy ``T_u - T_env`` [K].  Negative aloft is the
   373	        signal for the plume to terminate.
   374	    """
   375	    M_u: jax.Array
   376	    T_u: jax.Array
   377	    q_u: jax.Array
   378	    q_c_u: jax.Array
   379	    B_u: jax.Array
   380	
   381	
   382	def entraining_detraining_plume(
   383	    T_env: jax.Array,
   384	    q_v_env: jax.Array,
   385	    p_full: jax.Array,
   386	    p_half: jax.Array,
   387	    z_full: jax.Array,
   388	    T_parcel_base: jax.Array,
   389	    q_parcel_base: jax.Array,
   390	    k_base_smooth: jax.Array,
   391	    epsilon_profile: jax.Array,
   392	    delta_profile: jax.Array,
   393	    M_b: jax.Array,
   394	    *,
   395	    buoyancy_sharpness: float = 0.5,
   396	) -> Plume:
   397	    """Bulk entraining-detraining updraft from cloud base to LNB.
   398	
   399	    Integrates the standard plume budget upward from the cloud base
   400	    using :func:`jax.lax.scan` over levels (surface-first ordering
   401	    inside the scan).  At each layer the plume entrains environmental
   402	    air at fractional rate ``epsilon`` and detrains plume air at
   403	    rate ``delta`` (both with units [1/m]):
   404	
   405	    .. math::
   406	
   407	        \\frac{1}{M_u} \\frac{\\partial M_u}{\\partial z} = \\epsilon - \\delta
   408	
   409	        \\frac{\\partial T_u}{\\partial z} = -\\frac{g}{c_{pd}} - \\epsilon (T_u - T_{env})
   410	
   411	        \\frac{\\partial q_u}{\\partial z} = -\\epsilon (q_u - q_{v,env}) - C
   412	
   413	    where ``C`` is the condensation rate (the surplus of the parcel's
   414	    water vapor over its saturation value at the current level).
   415	    Cloud water accumulates at rate ``C - precip_rate`` (precipitation
   416	    is delegated to the calling scheme).
   417	
   418	    The plume's effective extent is encoded smoothly: the mass-flux
   419	    profile is multiplied by a sigmoid on the buoyancy ``T_u - T_env``
   420	    so that the plume tapers off (rather than being abruptly
   421	    truncated) once the parcel becomes negatively buoyant aloft.
   422	
   423	    Parameters
   424	    ----------
   425	    T_env, q_v_env : jax.Array, shape (ncol, nlev)
   426	        Environmental temperature [K] and vapor specific humidity
   427	        [kg/kg].
   428	    p_full, p_half : jax.Array
   429	        Pressures [Pa] at full and half levels.
   430	    z_full : jax.Array, shape (ncol, nlev)
   431	        Geopotential heights [m] at full levels.
   432	    T_parcel_base, q_parcel_base : jax.Array, shape (ncol,)
   433	        Parcel temperature [K] and specific humidity [kg/kg] at the
   434	        cloud-base launch level.
   435	    k_base_smooth : jax.Array, shape (ncol,)
   436	        Smooth fractional level index of the cloud base
   437	        (surface-last).  Levels below the cloud base contribute
   438	        zero plume mass flux.
   439	    epsilon_profile, delta_profile : jax.Array, shape (ncol, nlev)
   440	        Per-level fractional entrainment / detrainment rates [1/m].
   441	    M_b : jax.Array, shape (ncol,)
   442	        Cloud-base mass flux [kg/m²/s].
   443	    buoyancy_sharpness : float
   444	        Sharpness on the buoyancy-based plume-tapering sigmoid in
   445	        [1/K].  Default ``0.5`` per Kelvin of buoyancy means the
   446	        plume is at half mass flux when ``T_u - T_env`` reaches the
   447	        modest negative value of about ``-1.4 K``.
   448	
   449	    Returns
   450	    -------
   451	    Plume
   452	        Per-level mass flux, plume temperature, plume vapor, plume
   453	        cloud water, and plume buoyancy.
   454	    """
   455	    ncol, nlev = T_env.shape
   456	
   457	    # Pin everything to the input precision so saturation_mixing_ratio
   458	    # (which promotes f32 → f64 via its Clausius-Clapeyron literal
   459	    # constants) doesn't break ``jax.lax.scan``'s "carry-in dtype must
   460	    # equal carry-out dtype" invariant.
   461	    _dtype = T_env.dtype
   462	
   463	    # Reverse to surface-first for the scan.
   464	    T_env_rev = T_env[:, ::-1].astype(_dtype)
   465	    q_v_env_rev = q_v_env[:, ::-1].astype(_dtype)
   466	    p_full_rev = p_full[:, ::-1].astype(_dtype)
   467	    z_full_rev = z_full[:, ::-1].astype(_dtype)
   468	    eps_rev = epsilon_profile[:, ::-1].astype(_dtype)
   469	    del_rev = delta_profile[:, ::-1].astype(_dtype)
   470	
   471	    # Per-level above-base weight.  In surface-last indexing the cloud
   472	    # base is at ``k_base_smooth``; levels with surface-last index
   473	    # smaller than ``k_base_smooth`` are above the base.  In
   474	    # surface-first reversed indexing the relationship inverts:
   475	    # surface-first index ``k_rev`` corresponds to surface-last index
   476	    # ``nlev - 1 - k_rev``, and "above cloud base" means
   477	    # ``k_rev > (nlev - 1 - k_base_smooth)``.
   478	    k_rev = jnp.arange(nlev, dtype=T_env.dtype)
   479	    k_rev = jnp.broadcast_to(k_rev, T_env.shape)
   480	    k_base_rev = (nlev - 1.0) - k_base_smooth
   481	    above_base_weight = jax.nn.sigmoid(buoyancy_sharpness * (k_rev - k_base_rev[:, None]))
   482	
   483	    # Initial plume state at the surface-first index 0 (which is the
   484	    # actual surface).  We launch with the parcel values; the
   485	    # ``above_base_weight`` mask will suppress mass flux below cloud
   486	    # base.  All carry components are pinned to ``_dtype``.
   487	    init_carry = (
   488	        T_parcel_base.astype(_dtype),                       # T_u_prev
   489	        q_parcel_base.astype(_dtype),                       # q_u_prev
   490	        jnp.zeros_like(T_parcel_base, dtype=_dtype),        # q_c_u_prev
   491	        M_b.astype(_dtype),                                 # M_u_prev
   492	        z_full_rev[:, 0],                                   # z_prev (already cast)
   493	    )
   494	
   495	    # Per-level inputs to the scan.  Transpose to (nlev, ncol).
   496	    inputs = (
   497	        jnp.moveaxis(T_env_rev, 1, 0),
   498	        jnp.moveaxis(q_v_env_rev, 1, 0),
   499	        jnp.moveaxis(p_full_rev, 1, 0),
   500	        jnp.moveaxis(z_full_rev, 1, 0),
   501	        jnp.moveaxis(eps_rev, 1, 0),
   502	        jnp.moveaxis(del_rev, 1, 0),
   503	        jnp.moveaxis(above_base_weight, 1, 0),
   504	    )
   505	
   506	    g = constants.g
   507	    c_pd = constants.c_pd
   508	    L_v = constants.L_v
   509	
   510	    def step(carry, layer_inputs):
   511	        T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
   512	        T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs
   513	
   514	        dz = jnp.maximum(z_e - z_prev, 1.0)  # ascending; floor to avoid div-by-zero
   515	
   516	        # Raw plume mass flux: dM/dz = (epsilon - delta) * M.  Use the
   517	        # exact integration ``M(z+dz) = M(z) * exp((eps - dlt) * dz)``
   518	        # for this linear ODE — always positive, AD-safe everywhere,
   519	        # and exact when ``(eps - dlt)`` is constant over the layer.
   520	        # An earlier explicit-Euler form ``M * (1 + (eps - dlt) * dz)``
   521	        # could go negative for strong detrainment + thick layers
   522	        # (e.g. ``dlt = 5e-3 /m``, ``dz = 2000 m`` ⇒ multiplier =
   523	        # ``-7``); the subsequent ``jnp.maximum(..., 0)`` clipped the
   524	        # mass flux to 0 AND *zeroed the gradient* w.r.t. ``dlt`` /
   525	        # ``eps``, breaking AD-based sensitivity studies through the
   526	        # plume integrator (audit Codex finding: "plume mass flux uses
   527	        # explicit Euler plus a hard nonnegative clip ... after
   528	        # which jnp.maximum kills both mass flux and gradients").
   529	        # We intentionally do NOT bake the buoyancy / sub-cloud masks
   530	        # into the carry — those are reporting filters, not dynamics.
   531	        # Folding them into the carry would compound across levels and
   532	        # destroy the cloud-base-to-LNB profile that consumers expect.
   533	        M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)
   534	
   535	        # Entrainment of environmental T, q.
   536	        T_u_ent = T_u_prev + eps * dz * (T_e - T_u_prev)
   537	        q_u_ent = q_u_prev + eps * dz * (q_e - q_u_prev)
   538	
   539	        # Use the analytic moist-adiabatic lapse rate from
   540	        # ``moist_adiabat_lapse_rate`` (Iribarne–Godson) at the
   541	        # entrained parcel state.  The function evaluates dT/dp on
   542	        # the moist adiabat assuming the parcel is saturated; for
   543	        # *unsaturated* parcels this is approximate but matches the
   544	        # behavior of the existing ``compute_moist_adiabat`` helper
   545	        # that we're benchmarked against.  Latent heating is already
   546	        # baked into the lapse rate, so the post-hoc condensation
   547	        # step below does NOT add an additional ``L_v/c_pd *
   548	        # condensate`` correction — that would double-count.
   549	        rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
   550	        # ``moist_adiabat_lapse_rate`` returns dT/dp [K/Pa]; convert
   551	        # to dT/dz [K/m] via dp/dz = -rho*g.
   552	        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
   553	        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
   554	
   555	        T_u = T_u_ent + dT_dz * dz
   556	
   557	        # Condense any super-saturation into cloud water.  This is the
   558	        # diagnostic that resolves the q-budget; the temperature
   559	        # already incorporates the latent heat from condensation via
   560	        # the moist lapse rate.
   561	        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
   562	        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
   563	        q_u = (q_u_ent - condensate).astype(_dtype)
   564	        # Dilute plume cloud water by entrainment.  The continuity
   565	        # equation for an intensive quantity in an entraining-
   566	        # detraining plume is ``dq_c/dz = -eps · q_c + cond/M`` —
   567	        # environmental air carries q_c=0 so entrainment uniformly
   568	        # decreases ``q_c_u`` while detrainment is intensively
   569	        # neutral (it removes mass but not the per-kg amount).  An
   570	        # earlier formulation ``q_c_u = q_c_u_prev + condensate``
   571	        # carried ``q_c_u_prev`` forward unchanged and the plume's
   572	        # total water grew unphysically aloft (audit Codex finding:
   573	        # "plume cloud water is accumulated but not diluted by
   574	        # entrainment").
   575	        # Explicit Euler: ``q_c_u_ent = q_c_u_prev * (1 - eps·dz)``
   576	        # — clip to 0 to guard against ``eps·dz > 1`` corner cases
   577	        # (large eps × thick layer); preserves AD smoothness.
   578	        q_c_u_ent = jnp.maximum(q_c_u_prev * (1.0 - eps * dz), 0.0)
   579	        q_c_u = (q_c_u_ent + condensate).astype(_dtype)
   580	
   581	        T_u = T_u.astype(_dtype)
   582	
   583	        # Buoyancy at this level.
   584	        B_u = T_u - T_e
   585	
   586	        # Reporting filters: smoothly suppress the mass flux below
   587	        # cloud base (``abv``) and where the plume has lost buoyancy
   588	        # (``plume_alive``).  These do NOT enter the carry.
   589	        plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
   590	        M_u_reported = M_u_raw * plume_alive * abv
   591	
   592	        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
   593	        outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
   594	        return new_carry, outputs
   595	
   596	    _, scan_out = jax.lax.scan(step, init_carry, inputs)
   597	
   598	    T_u_rev, q_u_rev, q_c_u_rev, M_u_rev, B_u_rev = scan_out  # (nlev, ncol)
   599	
   600	    # Move axis back and reverse to surface-last.
   601	    T_u = jnp.moveaxis(T_u_rev, 0, 1)[:, ::-1]
   602	    q_u = jnp.moveaxis(q_u_rev, 0, 1)[:, ::-1]
   603	    q_c_u = jnp.moveaxis(q_c_u_rev, 0, 1)[:, ::-1]
   604	    M_u = jnp.moveaxis(M_u_rev, 0, 1)[:, ::-1]
   605	    B_u = jnp.moveaxis(B_u_rev, 0, 1)[:, ::-1]
   606	
   607	    return Plume(M_u=M_u, T_u=T_u, q_u=q_u, q_c_u=q_c_u, B_u=B_u)
   608	
   609	
   610	# ---------------------------------------------------------------------------
   611	# Convective momentum transport (Gregory et al. 1997)
   612	# ---------------------------------------------------------------------------
   613	
   614	def cmt_gregory_1997(
   615	    u_env: jax.Array,
   616	    v_env: jax.Array,
   617	    M_u: jax.Array,
   618	    M_d: jax.Array | None,
   619	    p_full: jax.Array,
   620	    p_half: jax.Array,
   621	    rho: jax.Array,
   622	    *,
   623	    c_u: float = 0.55,
   624	    c_d: float = 0.55,
   625	) -> tuple[jax.Array, jax.Array]:
   626	    """Gregory et al. 1997 convective momentum transport closure.
   627	
   628	    The eddy-flux closure carries plume momentum aloft minus a
   629	    pressure-gradient correction that limits the upward transport in
   630	    sheared environments.  In bulk-plume form ::
   631	
   632	        F_u = M_u * (u_u - u_env) - c_u * M_u * (du/dz)_layer * dz_layer
   633	
   634	    and the resulting tendency is ``du/dt = -(1/rho) * dF/dz``.  The
   635	    same form applies to the downdraft with sign convention
   636	    ``M_d < 0`` and parameter ``c_d``.
   637	
   638	    The implementation here is the **simplified bulk closure**: we
   639	    approximate ``u_u`` by the environmental wind at the cloud base
   640	    plus a fraction of the layer-by-layer environmental shear,
   641	    yielding a numerically stable form that does not require a
   642	    separate plume-momentum integrator.  This is the formulation
   643	    used in CESM/CAM with the ZM scheme and follows Gregory et al.
   644	    1997 Eq. 14.
   645	
   646	    Parameters
   647	    ----------
   648	    u_env, v_env : jax.Array, shape (ncol, nlev)
   649	        Environmental zonal / meridional wind [m/s].
   650	    M_u : jax.Array, shape (ncol, nlev)
   651	        Updraft mass flux [kg/m²/s].  Non-negative.
   652	    M_d : jax.Array or None, shape (ncol, nlev)
   653	        Downdraft mass flux [kg/m²/s] (negative by convention).
   654	        ``None`` skips the downdraft contribution.
   655	    p_full, p_half : jax.Array
   656	        Pressures [Pa] at full / half levels.  Used to compute layer
   657	        thickness ``dp = p_half[1:] - p_half[:-1]``.
   658	    rho : jax.Array, shape (ncol, nlev)
   659	        Air density [kg/m³].
   660	    c_u, c_d : float

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/mass_flux.py | sed -n '150,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/bechtold.py | sed -n '1,420p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   150	    leaving the upper troposphere unaffected.  ``p_gate_sharpness``
   151	    must be << ``p_min_convection`` for the sigmoid to saturate
   152	    within the integration range; sharpness ≥ p_min only attenuates.
   153	
   154	    Differentiable everywhere; ``p_gate_sharpness`` sets the width of
   155	    the transition (Pa).
   156	    """
   157	    return jax.nn.sigmoid(
   158	        (p_full - p_min_convection) / jnp.maximum(p_gate_sharpness, 1.0)
   159	    )
   160	
   161	
   162	def _apply_mass_flux_kernel(
   163	    T: jax.Array,
   164	    q_v: jax.Array,
   165	    p_full: jax.Array,
   166	    T_u: jax.Array,
   167	    q_v_u: jax.Array,
   168	    q_c_u: jax.Array,
   169	    M_profile: jax.Array,
   170	    z: jax.Array,
   171	    rho: jax.Array,
   172	    delta_0: float,
   173	    M_u_max: float = 0.05,
   174	    p_min_convection: float = 10_000.0,
   175	    p_gate_sharpness: float = 1_500.0,
   176	) -> Tuple[jax.Array, jax.Array, jax.Array]:
   177	    """Mass-flux core kernel: compensating subsidence + detrainment.
   178	
   179	    Given the vertical mass-flux profile ``M_profile`` and the
   180	    entraining updraft thermodynamics ``(T_u, q_v_u, q_c_u)``,
   181	    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
   182	    ``(ncol, nlev)``.
   183	
   184	    The decomposition follows Tiedtke (1989) / Siebesma et al. (2007):
   185	      (a) Compensating subsidence: ``(M/rho) * (dT/dz + g/c_p)`` for T
   186	          and ``(M/rho) * dq/dz`` for moisture (q is conserved so no
   187	          adiabatic correction).
   188	      (b) Detrainment mixing: ``+delta_0 * M * (X_u - X) / rho``.
   189	
   190	    The mass flux ``M_profile`` is gated by a smooth sigmoid in
   191	    pressure so that levels above ``p_min_convection`` (default 100
   192	    hPa, the canonical tropical tropopause) receive no convective
   193	    tendency.  See ``stratosphere_mass_flux_gate`` for details.
   194	
   195	    The convective source for cloud water is the per-level detrainment
   196	    of the plume's cloud water:
   197	    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
   198	    by construction.  Splitting the plume into separate vapor (``q_v_u``)
   199	    and cloud (``q_c_u``) pieces — instead of a single ``q_u`` that
   200	    conflates total water with vapor — is what makes the column MSE
   201	    budget close.  The earlier formulation passed ``q_u = q_v_u + q_c_u``
   202	    as if it were vapor and computed condensate as ``max(q_u - q_sat,
   203	    0)``, which is essentially zero for an entraining-diluted plume —
   204	    the leaf then leaked latent energy.  Microphysics processes
   205	    ``dq_c_conv_dt`` through its full chain (autoconversion,
   206	    sedimentation, evaporation) and produces the resulting surface
   207	    precipitation; convection no longer assumes the condensate falls
   208	    instantly.
   209	    Unit check: (1/m) * (kg/m²/s) * (kg/kg) / (kg/m³) = 1/s × kg/kg.
   210	    """
   211	    dT_dz, dq_dz = _compute_centered_gradients(T, q_v, z)
   212	    rho_safe = jnp.clip(rho, 0.01, None)
   213	
   214	    # Per-level mass-flux cap.  The plume integrator can yield ``M_u``
   215	    # that grows with height when ``epsilon > delta`` (entraining
   216	    # plumes) or that responds non-linearly to a high-CAPE column.
   217	    # Per-layer convective heating ``≈ delta_0 · M_u · (T_u−T)/ρ``
   218	    # scales linearly with ``M_u``, so an uncapped ``M_u`` produces
   219	    # column heating well in excess of what surface fluxes can supply
   220	    # and destabilises the integration.  Clipping at the cap (default
   221	    # ``0.05 kg/m²/s``, the literature peak tropical updraft mass flux)
   222	    # bounds per-layer tendencies without distorting the moist adiabat
   223	    # or the q_v / q_c split.
   224	    M_profile = jnp.clip(M_profile, 0.0, M_u_max)
   225	
   226	    # Stratospheric pressure gate — see ``stratosphere_mass_flux_gate``.
   227	    M_profile = M_profile * stratosphere_mass_flux_gate(
   228	        p_full, p_min_convection, p_gate_sharpness,
   229	    )
   230	
   231	    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
   232	    dq_subsidence = (M_profile / rho_safe) * dq_dz
   233	
   234	    dT_detrain = delta_0 * M_profile * (T_u - T) / rho_safe
   235	    dq_detrain = delta_0 * M_profile * (q_v_u - q_v) / rho_safe
   236	
   237	    dT_dt = dT_subsidence + dT_detrain
   238	    dq_v_dt = dq_subsidence + dq_detrain
   239	
   240	    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
   241	    return dT_dt, dq_v_dt, dq_c_conv_dt
   242	
   243	
   244	# =============================================================================
   245	# Arakawa-Wu prognostic mass-flux (M_c × fixed sinusoidal profile)
   246	# =============================================================================
   247	
   248	
   249	class MassFluxClosureDiagnostics(NamedTuple):
   250	    """Intermediate closure state reused by physical and ML mass-flux paths."""
   251	
   252	    dz: jax.Array
   253	    rho: jax.Array
   254	    z: jax.Array
   255	    T_moist: jax.Array
   256	    cape: jax.Array
   257	    M_eq: jax.Array
   258	    M_c_new: jax.Array
   259	    convective_mask: jax.Array
   260	

 succeeded in 0ms:
     1	"""Bechtold / IFS convection (Bechtold et al. 2008, 2014).
     2	
     3	Builds on the Tiedtke 1989 skeleton (see :mod:`.tiedtke`) and adds:
     4	
     5	1. **PBL-CAPE / departure-CAPE closure** (Bechtold 2008).  ``M_b`` is
     6	   diagnosed from a mass-weighted parcel within the boundary layer
     7	   rather than the surface parcel.  This sharpens the diurnal cycle
     8	   of deep convection over land.
     9	2. **AR1 stochastic perturbation** (Bechtold 2014).  ``M_b *= (1 +
    10	   amplitude * ε)`` where ``ε`` is an AR1 process with prescribed
    11	   decorrelation timescale.  The AR1 noise state is carried in
    12	   :attr:`legoesm.atmosphere.physics.physics_state.PhysicsState.conv_stoch_state`.
    13	   Stochasticity is OFF by default; when enabled, the leaf takes a
    14	   ``prng_key`` argument.
    15	
    16	The smooth-everywhere / differentiability properties are inherited
    17	from Tiedtke; the AR1 stochastic factor is treated as a fixed
    18	multiplier per call so ``jax.grad`` flows through the deterministic
    19	``M_b``.
    20	
    21	References
    22	----------
    23	* Bechtold, P., Köhler, M., Jung, T., Doblas-Reyes, F., Leutbecher,
    24	  M., Rodwell, M. J., Vitart, F., & Balsamo, G. (2008). Advances in
    25	  simulating atmospheric variability with the ECMWF model.  *Quart.
    26	  J. Roy. Meteor. Soc.*, 134, 1337–1351.
    27	* Bechtold, P., Semane, N., Lopez, P., Chaboureau, J.-P., Beljaars,
    28	  A., & Bormann, N. (2014). Representing equilibrium and
    29	  nonequilibrium convection in large-scale models.  *J. Atmos. Sci.*,
    30	  71, 734–753.
    31	"""
    32	
    33	from __future__ import annotations
    34	
    35	import jax
    36	import jax.numpy as jnp
    37	
    38	from legoesm import constants
    39	from legoesm.thermo import saturation_mixing_ratio
    40	from legoesm.atmosphere.physics.thermodynamics import (
    41	    compute_cape,
    42	    compute_moist_adiabat,
    43	)
    44	from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    45	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    46	from legoesm.atmosphere.physics.convection.mass_flux import (
    47	    _apply_mass_flux_kernel,
    48	    stratosphere_mass_flux_gate,
    49	    _compute_column_geometry,
    50	)
    51	from legoesm.atmosphere.physics.convection._triggers import (
    52	    cape_trigger,
    53	    smooth_level_indicator,
    54	    smooth_positive_part,
    55	    smooth_step,
    56	)
    57	from legoesm.atmosphere.physics.convection._plume import (
    58	    cmt_gregory_1997,
    59	    compute_lcl,
    60	    compute_lfc_lnb,
    61	    entraining_detraining_plume,
    62	)
    63	
    64	
    65	__all__ = ("bechtold_convection",)
    66	
    67	
    68	def bechtold_convection(
    69	    T: jax.Array,
    70	    q_v: jax.Array,
    71	    p_full: jax.Array,
    72	    p_half: jax.Array,
    73	    u: jax.Array,
    74	    v: jax.Array,
    75	    conv_prog_profile: jax.Array,
    76	    conv_stoch_state: jax.Array,
    77	    prng_key: jax.Array | None,
    78	    dt: float,
    79	    config: BechtoldConfig = BechtoldConfig(),
    80	    moisture_convergence: jax.Array | None = None,
    81	) -> tuple[ConvectionOutput, jax.Array, jax.Array]:
    82	    """Bechtold/IFS convection (smooth, differentiable).
    83	
    84	    Parameters
    85	    ----------
    86	    T, q_v : jax.Array, shape (ncol, nlev)
    87	        Environmental temperature and water-vapor specific humidity.
    88	    p_full, p_half : jax.Array
    89	        Full / half-level pressures.
    90	    u, v : jax.Array, shape (ncol, nlev)
    91	        Environmental winds (for CMT).
    92	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    93	        Updraft mass-flux profile from the previous step.
    94	    conv_stoch_state : jax.Array, shape (ncol,)
    95	        AR1 noise state from the previous step.
    96	    prng_key : jax.Array or None
    97	        PRNG key for the stochastic perturbation.  When
    98	        ``config.enable_stochastic`` is ``False`` this argument is
    99	        ignored.  When stochastic is on but ``prng_key`` is ``None``
   100	        the leaf falls back to a deterministic (zero-noise)
   101	        realization.  The convection bridge derives a per-step
   102	        sub-key from ``PhysicsState.prng_key`` (split + ``fold_in``
   103	        with module id ``0xBEC4``) when stochasticity is enabled.
   104	    dt : float
   105	        Time step [s].
   106	    config : BechtoldConfig
   107	
   108	    Returns
   109	    -------
   110	    out : ConvectionOutput
   111	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   112	        Updated M_u profile (implicit-Euler relaxed).
   113	    conv_stoch_state_new : jax.Array, shape (ncol,)
   114	        Updated AR1 noise state.
   115	    """
   116	    ncol, nlev = T.shape
   117	
   118	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   119	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   120	    T_base = T[:, -1]
   121	    q_base = q_v[:, -1]
   122	    p_base = p_full[:, -1]
   123	
   124	    # -- PBL parcel: mass-weighted average over the boundary-layer
   125	    # depth.  Smooth weighting via ``smooth_level_indicator`` so the
   126	    # PBL-depth threshold is differentiable.  The mass weight is
   127	    # ``pbl_weight * dp`` (dp/g per layer is mass per unit area) — an
   128	    # earlier form averaged with ``pbl_weight`` alone, which is only
   129	    # correct for uniform-thickness layers and gave a height-weighted,
   130	    # not mass-weighted, mean (audit Codex finding: "Bechtold PBL
   131	    # parcel is not actually mass weighted").
   132	    dp_full = p_half[:, 1:] - p_half[:, :-1]
   133	    pbl_weight = smooth_level_indicator(
   134	        z, threshold=config.cape_pbl_depth, sharpness=2.0e-3,
   135	        direction="below",
   136	    )                                                       # (ncol, nlev)
   137	    pbl_mass_weight = pbl_weight * dp_full
   138	    pbl_norm = jnp.sum(pbl_mass_weight, axis=-1, keepdims=True).clip(1e-6, None)
   139	    T_pbl = jnp.sum(pbl_mass_weight * T, axis=-1) / pbl_norm.squeeze(-1)
   140	    q_pbl = jnp.sum(pbl_mass_weight * q_v, axis=-1) / pbl_norm.squeeze(-1)
   141	    # Mass-weighted PBL pressure for the LCL launch level when the
   142	    # parcel comes from the PBL mean (otherwise use surface pressure).
   143	    p_pbl = jnp.sum(pbl_mass_weight * p_full, axis=-1) / pbl_norm.squeeze(-1)
   144	    if config.use_pbl_cape:
   145	        T_parcel_source = T_pbl
   146	        q_parcel_source = q_pbl
   147	        p_parcel_source = p_pbl
   148	    else:
   149	        T_parcel_source = T_base
   150	        q_parcel_source = q_base
   151	        p_parcel_source = p_base
   152	
   153	    T_parcel = T_parcel_source + config.parcel_dT
   154	    q_parcel = q_parcel_source + config.parcel_dq
   155	
   156	    T_moist = compute_moist_adiabat(T_parcel, p_full)
   157	    cape_pbl = compute_cape(T, T_moist, p_full, p_half)
   158	
   159	    cape_weight = cape_trigger(
   160	        cape_pbl, config.cape_threshold, config.cape_sharpness,
   161	    )
   162	
   163	    # -- LCL, LFC/LNB ------------------------------------------------------
   164	    lcl = compute_lcl(T_parcel, q_parcel, p_parcel_source, p_full)
   165	    k_lcl_smooth = lcl.k_lcl_smooth
   166	    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
   167	
   168	    # Cloud depth.
   169	    levels_arr = jnp.arange(nlev, dtype=T.dtype)
   170	    weight_lcl = jax.nn.softmax(
   171	        -2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
   172	    )
   173	    weight_lnb = jax.nn.softmax(
   174	        -2.0 * (levels_arr[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
   175	    )
   176	    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
   177	    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
   178	    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)
   179	
   180	    # -- Three-class blend -------------------------------------------------
   181	    deep_weight = smooth_step(
   182	        cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness,
   183	    )
   184	    shallow_weight = smooth_step(
   185	        config.cloud_depth_shallow_max - cloud_depth,
   186	        config.depth_split_sharpness,
   187	    )
   188	    midlevel_weight = jnp.clip(
   189	        1.0 - deep_weight - shallow_weight, 0.0, 1.0
   190	    )
   191	
   192	    # -- PBL-CAPE closure for cloud-base mass flux -------------------------
   193	    # Bechtold 2008 / 2014 use a hybrid closure: PBL-CAPE drives the
   194	    # baseline mass flux, optionally enhanced where the column is
   195	    # moisture-convergent.  We add the (column-integrated) MC term as
   196	    # a multiplicative enhancement (1 + MC_normalized) so the closure
   197	    # gracefully reduces to pure PBL-CAPE when MC is unavailable
   198	    # (zero-filled by the bridge for spectral PE and other dycores
   199	    # without an MC diagnostic).
   200	    # Dimensionally-correct PBL-CAPE closure (Kain 2004 §3 form):
   201	    #     M_b = rho_BL * (CAPE_pbl - threshold)+ / (g * tau_bl)   [kg/m^2/s]
   202	    # The earlier formula omitted ``rho_BL`` and ``g``; magnitude was
   203	    # masked operationally only by ``M_b_max``.
   204	    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
   205	    M_b_pbl_cape = (
   206	        cape_weight
   207	        * rho_BL
   208	        * smooth_positive_part(cape_pbl - config.cape_threshold, config.cape_sharpness)
   209	        / (constants.g * config.tau_bl)
   210	    )
   211	    if moisture_convergence is not None:
   212	        column_MC = jnp.sum(
   213	            jnp.maximum(moisture_convergence, 0.0) * dp_full, axis=-1,
   214	        ) / constants.g
   215	        # Normalize the MC term so it acts as an O(1) multiplier.
   216	        # ``mc_normalize_scale`` (default 0.05 kg/m²/s) is a typical
   217	        # strong-convergence value over tropical convective regions
   218	        # (Bechtold 2008 Fig. 2).  Lifted from a literal per CLAUDE.md
   219	        # 'no hardcoded tunables in physics body' (audit B9).
   220	        mc_enhancement = column_MC / config.mc_normalize_scale
   221	        M_b_deterministic = M_b_pbl_cape * (1.0 + mc_enhancement)
   222	    else:
   223	        M_b_deterministic = M_b_pbl_cape
   224	
   225	    # -- AR1 stochastic perturbation ---------------------------------------
   226	    if config.enable_stochastic and prng_key is not None:
   227	        alpha_AR1 = jnp.exp(-dt / config.stochastic_decorrelation)
   228	        innovation = jax.random.normal(prng_key, shape=(ncol,), dtype=T.dtype)
   229	        conv_stoch_state_new = (
   230	            alpha_AR1 * conv_stoch_state
   231	            + jnp.sqrt(jnp.maximum(1.0 - alpha_AR1 ** 2, 0.0)) * innovation
   232	        )
   233	        stoch_factor = 1.0 + config.stochastic_amplitude * conv_stoch_state_new
   234	    else:
   235	        # Either stochastic disabled or no PRNG provided — preserve
   236	        # input AR1 state and use deterministic factor 1.
   237	        conv_stoch_state_new = conv_stoch_state
   238	        stoch_factor = jnp.ones_like(M_b_deterministic)
   239	
   240	    M_b = M_b_deterministic * jnp.maximum(stoch_factor, 0.0)
   241	    # See ZhangMcFarlaneConfig.M_b_max.
   242	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   243	
   244	    # -- Per-class entrainment / detrainment profiles ----------------------
   245	    eps_per_class = (
   246	        deep_weight[:, None] * config.epsilon_deep
   247	        + shallow_weight[:, None] * config.epsilon_shallow
   248	        + midlevel_weight[:, None] * config.epsilon_midlevel
   249	    )
   250	    dlt_per_class = (
   251	        deep_weight[:, None] * config.delta_deep
   252	        + shallow_weight[:, None] * config.delta_shallow
   253	        + midlevel_weight[:, None] * config.delta_midlevel
   254	    )
   255	    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
   256	    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)
   257	
   258	    plume = entraining_detraining_plume(
   259	        T, q_v, p_full, p_half, z,
   260	        T_parcel, q_parcel, k_lcl_smooth,
   261	        eps_profile, dlt_profile, M_b,
   262	    )
   263	
   264	    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
   265	    # ``dt / tau`` (floor tau against zero), NOT ``dt / max(tau, dt)``;
   266	    # the latter under-stepped the relaxation when ``dt > tau`` (audit
   267	    # Codex finding).
   268	    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, 1e-30)
   269	    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
   270	    # Cap M_u_new at config.M_b_max so every downstream use (kernel
   271	    # tendencies, dq_c_conv_raw, downdraft trigger, CMT, carry update)
   272	    # sees the same bounded value.
   273	    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
   274	
   275	    # -- Environmental tendencies (using relaxed M_u) ---------------------
   276	    delta_0_eff = (
   277	        deep_weight * config.delta_deep
   278	        + shallow_weight * config.delta_shallow
   279	        + midlevel_weight * config.delta_midlevel
   280	    )
   281	    # Pass per-column blended delta_0 directly to the kernel — see
   282	    # tiedtke.py for the rationale.  Multiplying the kernel's full
   283	    # output by ``delta_0_eff / delta_deep`` would also rescale the
   284	    # delta-independent subsidence terms.
   285	    dT_dt, dq_v_dt, _ = _apply_mass_flux_kernel(
   286	        T, q_v, p_full,
   287	        plume.T_u, plume.q_u, plume.q_c_u, M_u_new,
   288	        z, rho, delta_0_eff[:, None], M_u_max=config.M_b_max,
   289	    )
   290	    rho_safe = jnp.clip(rho, 0.01, None)
   291	    p_gate_qc = stratosphere_mass_flux_gate(p_full)
   292	    dq_c_conv_dt = (
   293	        delta_0_eff[:, None] * M_u_new * p_gate_qc * plume.q_c_u / rho_safe
   294	    )
   295	
   296	    # -- Optional downdraft (RH-dependent) ---------------------------------
   297	    if config.enable_downdraft:
   298	        below_lcl = jax.nn.sigmoid(
   299	            2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None])
   300	        )
   301	        q_sat_env = saturation_mixing_ratio(T, p_full)
   302	        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
   303	        below_mass = jnp.sum(below_lcl * dp_full, axis=-1) + 1e-6
   304	        rh_below = (
   305	            jnp.sum(below_lcl * rh_layer * dp_full, axis=-1) / below_mass
   306	        )
   307	        downdraft_trigger = jax.nn.sigmoid(
   308	            10.0 * (config.downdraft_RH_min - rh_below)
   309	        )
   310	        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
   311	        # Subcloud rain-evaporation cooling — see tiedtke.py for the
   312	        # full derivation.  ``E_layer`` [kg/(kg·s)] mass-weighted over
   313	        # below-LCL layers; the rain mass that re-evaporates is drawn
   314	        # from ``dq_c_conv_dt`` so the column water budget closes
   315	        # (Codex stop-time review: "downdraft fix still creates column
   316	        # water" — earlier form added vapor without removing the
   317	        # corresponding cloud-water source).
   318	        below_lcl_mass = jnp.sum(below_lcl * dp_full, axis=-1, keepdims=True).clip(1e-6, None)
   319	        rain_source_total = jnp.sum(
   320	            jnp.maximum(dq_c_conv_dt, 0.0) * dp_full, axis=-1,
   321	        ) / constants.g
   322	        evap_total = jnp.minimum(
   323	            jnp.abs(M_d_base) * config.downdraft_evap_efficiency,
   324	            rain_source_total,
   325	        )
   326	        evap_rate = (
   327	            evap_total[:, None] * below_lcl * constants.g / below_lcl_mass
   328	        )
   329	        dT_dt_dd = -(constants.L_v / constants.c_pd) * evap_rate
   330	        dT_dt = dT_dt + dT_dt_dd
   331	        dq_v_dt = dq_v_dt + evap_rate
   332	        rain_source_safe = jnp.clip(rain_source_total[:, None], 1e-30, None)
   333	        rain_scale = 1.0 - evap_total[:, None] / rain_source_safe
   334	        dq_c_conv_dt = jnp.where(
   335	            dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
   336	        )
   337	
   338	    # -- CMT --------------------------------------------------------------
   339	    if config.enable_cmt:
   340	        if config.enable_downdraft:
   341	            M_d = -config.downdraft_alpha * M_u_new * 0.3
   342	        else:
   343	            M_d = None
   344	        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
   345	            u, v, M_u_new, M_d,
   346	            p_full, p_half, rho,
   347	            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
   348	        )
   349	    else:
   350	        du_dt_conv = None
   351	        dv_dt_conv = None
   352	
   353	    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
   354	
   355	    out = ConvectionOutput(
   356	        dT_dt=dT_dt,
   357	        dq_v_dt=dq_v_dt,
   358	        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
   359	        cape=cape_pbl,
   360	        convective_mask=convective_mask,
   361	        du_dt_conv=du_dt_conv,
   362	        dv_dt_conv=dv_dt_conv,
   363	    )
   364	    return out, M_u_new, conv_stoch_state_new

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/morrison.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/thompson.py | sed -n '1,290p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Morrison double-moment ice+liquid microphysics.
     2	
     3	Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
     4	(Cooper 1986), depositional growth, Bergeron process, riming, snow
     5	aggregation, and melting. Tracks cloud water, rain, ice, and snow.
     6	
     7	All operations use smooth (differentiable) approximations.
     8	
     9	References
    10	----------
    11	- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
    12	  double-moment microphysics parameterization. Part I: Description.
    13	  J. Atmos. Sci., 62, 1665-1677.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	    safe_pow,
    31	)
    32	from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    33	from legoesm.atmosphere.physics.microphysics.output import (
    34	    HydrometeorState,
    35	    MicrophysicsOutput,
    36	    sedimentation_tendency,
    37	)
    38	
    39	
    40	def morrison_microphysics(
    41	    T: jax.Array,
    42	    q_v: jax.Array,
    43	    hydrometeors: HydrometeorState,
    44	    p_full: jax.Array,
    45	    p_half: jax.Array,
    46	    rho: jax.Array,
    47	    dz: jax.Array,
    48	    dt: float,
    49	    config: MorrisonConfig = MorrisonConfig(),
    50	) -> MicrophysicsOutput:
    51	    """Compute Morrison double-moment microphysics tendencies.
    52	
    53	    Parameters
    54	    ----------
    55	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    56	        Same interface as all microphysics backends.
    57	
    58	    Returns
    59	    -------
    60	    MicrophysicsOutput
    61	    """
    62	    ncol, nlev = T.shape
    63	    q_c = hydrometeors.q_c
    64	    q_r = hydrometeors.q_r
    65	    q_i = hydrometeors.q_i
    66	    q_s = hydrometeors.q_s
    67	    N_c = hydrometeors.N_c
    68	    N_r = hydrometeors.N_r
    69	    N_i = hydrometeors.N_i
    70	    sharpness = config.saturation_sharpness
    71	
    72	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    73	
    74	    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    75	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    76	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    77	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    78	    )
    79	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    80	    dN_r_sc, dN_r_br = self_collection_breakup(
    81	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    82	    )
    83	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    84	
    85	    # === ICE PHASE ===
    86	    T_freeze = constants.T_freeze
    87	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
    88	
    89	    # 1. Ice nucleation (Cooper 1986, smoothed)
    90	    N_i_target = config.N_i0 * jnp.exp(
    91	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    92	    ) / jnp.clip(rho, 0.1)
    93	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
    94	
    95	    # 2. Depositional growth
    96	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    97	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    98	    dq_i_dep = (
    99	        config.dep_coeff
   100	        * jnp.maximum(S_i, 0.0)
   101	        * jnp.clip(q_i, 0.0)
   102	        * safe_pow(N_i, 1.0 / 3.0)
   103	        * f_ice
   104	    )
   105	
   106	    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
   107	    berg_window = (
   108	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   109	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   110	    )
   111	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   112	
   113	    # 4. Riming: ice/snow collect cloud water
   114	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   115	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   116	
   117	    # 5. Snow aggregation: ice -> snow
   118	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   119	
   120	    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
   121	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   122	    melt_ice = jnp.minimum(
   123	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   124	        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
   125	    )
   126	    melt_snow = jnp.minimum(
   127	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   128	        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
   129	    )
   130	
   131	    # === DONOR CLAMP for q_c sinks ===
   132	    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
   133	    # riming) by a common factor so the total loss per timestep does not
   134	    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
   135	    # in a mixed-phase column drive q_c negative on a single explicit step
   136	    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
   137	    # conserved because each process's matching source term in dq_r/dq_i/dq_s
   138	    # gets the same scale factor (the rates appear once as sinks in dq_c and
   139	    # once as sources elsewhere, so a uniform rescale preserves the budget).
   140	    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
   141	    qc_avail = jnp.clip(q_c, 0.0)
   142	    qc_scale = jnp.minimum(
   143	        1.0,
   144	        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
   145	    )
   146	    dq_c_au = dq_c_au * qc_scale
   147	    dq_c_ac = dq_c_ac * qc_scale
   148	    bergeron = bergeron * qc_scale
   149	    riming_i = riming_i * qc_scale
   150	    riming_s = riming_s * qc_scale
   151	    # Number tendency for autoconverted droplets must scale identically.
   152	    dN_r_au = dN_r_au * qc_scale
   153	
   154	    # === SEDIMENTATION ===
   155	    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
   156	    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
   157	    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
   158	    rho_sfc = rho[:, -1:]
   159	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   160	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   161	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   162	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   163	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   164	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   165	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   166	
   167	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   168	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   169	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   170	
   171	    # === LATENT HEATING ===
   172	    L_v = constants.L_v
   173	    L_s = constants.L_s
   174	    L_f = constants.L_f
   175	    c_pd = constants.c_pd
   176	
   177	    dT_dt = (
   178	        L_v * condensation / c_pd
   179	        - L_v * evaporation / c_pd
   180	        + L_s * dq_i_dep / c_pd
   181	        # Cloud water → ice/snow freezing releases latent heat of fusion
   182	        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
   183	        # riming is supercooled-droplet capture by ice/snow.  Both are
   184	        # phase changes that release L_f; the moist-enthalpy invariant
   185	        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
   186	        # column conservation.  Magnitude estimate: ~2 K/day at default
   187	        # rates in mixed-phase clouds.
   188	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   189	        - L_f * (melt_ice + melt_snow) / c_pd
   190	    )
   191	
   192	    # === COMBINE TENDENCIES ===
   193	    dq_v_dt = -condensation + evaporation - dq_i_dep
   194	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   195	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
   196	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
   197	    dq_s_dt = aggregation + riming_s - melt_snow + sed_s
   198	
   199	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   200	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   201	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   202	
   203	    # Precipitation (rain + ice + snow at surface)
   204	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   205	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   206	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   207	    precipitation = precip_r + precip_i + precip_s
   208	
   209	    # Pin dtype to the input precision so we never silently promote
   210	    # the unused-species placeholders to f64 under x64 mode.
   211	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   212	    return MicrophysicsOutput(
   213	        dT_dt=dT_dt,
   214	        dq_v_dt=dq_v_dt,
   215	        dq_c_dt=dq_c_dt,
   216	        dq_r_dt=dq_r_dt,
   217	        dq_i_dt=dq_i_dt,
   218	        dq_s_dt=dq_s_dt,
   219	        dq_g_dt=z,
   220	        dN_c_dt=dN_c_dt,
   221	        dN_r_dt=dN_r_dt,
   222	        dN_i_dt=dN_i_dt,
   223	        precipitation=precipitation,
   224	    )

 succeeded in 0ms:
     1	"""Thompson hybrid-moment microphysics.
     2	
     3	Extends Morrison with graupel formation from intense riming and
     4	gamma distribution shape corrections for autoconversion/accretion.
     5	
     6	All operations use smooth (differentiable) approximations.
     7	
     8	References
     9	----------
    10	- Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008).
    11	  Explicit forecasts of winter precipitation using an improved bulk
    12	  microphysics scheme. Part II: Implementation of a new snow
    13	  parameterization. Mon. Wea. Rev., 136, 5095-5115.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	    safe_pow,
    31	)
    32	from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
    33	from legoesm.atmosphere.physics.microphysics.output import (
    34	    HydrometeorState,
    35	    MicrophysicsOutput,
    36	    sedimentation_tendency,
    37	)
    38	
    39	
    40	def _gamma_ratio(mu):
    41	    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    42	    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)
    43	
    44	
    45	def thompson_microphysics(
    46	    T: jax.Array,
    47	    q_v: jax.Array,
    48	    hydrometeors: HydrometeorState,
    49	    p_full: jax.Array,
    50	    p_half: jax.Array,
    51	    rho: jax.Array,
    52	    dz: jax.Array,
    53	    dt: float,
    54	    config: ThompsonConfig = ThompsonConfig(),
    55	) -> MicrophysicsOutput:
    56	    """Compute Thompson hybrid-moment microphysics tendencies.
    57	
    58	    Parameters
    59	    ----------
    60	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    61	        Same interface as all microphysics backends.
    62	
    63	    Returns
    64	    -------
    65	    MicrophysicsOutput
    66	    """
    67	    ncol, nlev = T.shape
    68	    q_c = hydrometeors.q_c
    69	    q_r = hydrometeors.q_r
    70	    q_i = hydrometeors.q_i
    71	    q_s = hydrometeors.q_s
    72	    q_g = hydrometeors.q_g
    73	    N_c = hydrometeors.N_c
    74	    N_r = hydrometeors.N_r
    75	    N_i = hydrometeors.N_i
    76	    sharpness = config.saturation_sharpness
    77	
    78	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    79	
    80	    # === WARM RAIN ===
    81	    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s]
    82	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    83	
    84	    # Gamma distribution corrections
    85	    gamma_c = _gamma_ratio(config.mu_c)
    86	    gamma_r = _gamma_ratio(config.mu_r)
    87	    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=24)
    88	    gamma_r_norm = gamma_r / _gamma_ratio(0.0)
    89	
    90	    # Autoconversion (gamma-corrected)
    91	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    92	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness, gamma_norm=gamma_c_norm,
    93	    )
    94	
    95	    # Accretion (gamma-corrected)
    96	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)
    97	
    98	    # Self-collection / breakup
    99	    dN_r_sc, dN_r_br = self_collection_breakup(
   100	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
   101	    )
   102	
   103	    # Rain evaporation
   104	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
   105	
   106	    # === ICE PHASE (Morrison processes) ===
   107	    T_freeze = constants.T_freeze
   108	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
   109	
   110	    # Ice nucleation
   111	    N_i_target = config.N_i0 * jnp.exp(
   112	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
   113	    ) / jnp.clip(rho, 0.1)
   114	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
   115	
   116	    # Depositional growth
   117	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
   118	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
   119	    dq_i_dep = (
   120	        config.dep_coeff
   121	        * jnp.maximum(S_i, 0.0)
   122	        * jnp.clip(q_i, 0.0)
   123	        * safe_pow(N_i, 1.0 / 3.0)
   124	        * f_ice
   125	    )
   126	
   127	    # Bergeron
   128	    berg_window = (
   129	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   130	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   131	    )
   132	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   133	
   134	    # Riming
   135	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   136	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   137	    total_riming = riming_i + riming_s
   138	
   139	    # Aggregation
   140	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   141	
   142	    # Melting (clamp to available mass so an explicit Euler step cannot
   143	    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
   144	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   145	    dt_safe = jnp.maximum(dt, 1e-10)
   146	    melt_ice = jnp.minimum(
   147	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   148	        jnp.clip(q_i, 0.0) / dt_safe,
   149	    )
   150	    melt_snow = jnp.minimum(
   151	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   152	        jnp.clip(q_s, 0.0) / dt_safe,
   153	    )
   154	
   155	    # === GRAUPEL (Thompson extension) ===
   156	    graupel_frac = jax.nn.sigmoid(
   157	        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
   158	    )
   159	    rime_to_graupel = config.rime_to_graupel_rate * total_riming * graupel_frac
   160	    melt_graupel = jnp.minimum(
   161	        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
   162	        jnp.clip(q_g, 0.0) / dt_safe,
   163	    )
   164	
   165	    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
   166	    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
   167	    qc_avail = jnp.clip(q_c, 0.0)
   168	    qc_scale = jnp.minimum(
   169	        1.0,
   170	        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
   171	    )
   172	    dq_c_au = dq_c_au * qc_scale
   173	    dq_c_ac = dq_c_ac * qc_scale
   174	    bergeron = bergeron * qc_scale
   175	    riming_i = riming_i * qc_scale
   176	    riming_s = riming_s * qc_scale
   177	    total_riming = riming_i + riming_s
   178	    rime_to_graupel = rime_to_graupel * qc_scale
   179	    dN_r_au = dN_r_au * qc_scale
   180	
   181	    # === SEDIMENTATION ===
   182	    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
   183	    # [0.25, 0.5]); guard the AD path with safe_pow.
   184	    rho_sfc = rho[:, -1:]
   185	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   186	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   187	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   188	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   189	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   190	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   191	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   192	    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
   193	    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)
   194	
   195	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   196	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   197	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   198	    sed_g = sedimentation_tendency(q_g, rho, V_t_g, dz)
   199	
   200	    # === LATENT HEATING ===
   201	    L_v = constants.L_v
   202	    L_s = constants.L_s
   203	    L_f = constants.L_f
   204	    c_pd = constants.c_pd
   205	    dT_dt = (
   206	        L_v * condensation / c_pd
   207	        - L_v * evaporation / c_pd
   208	        + L_s * dq_i_dep / c_pd
   209	        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
   210	        # See morrison.py for the moist-enthalpy rationale; Thompson
   211	        # mirrors Morrison's ice-phase latent heating.
   212	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   213	        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
   214	    )
   215	
   216	    # === COMBINE TENDENCIES ===
   217	    dq_v_dt = -condensation + evaporation - dq_i_dep
   218	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   219	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
   220	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice - rime_to_graupel + sed_i
   221	    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel * 0.5 + sed_s
   222	    dq_g_dt = rime_to_graupel * 1.5 - melt_graupel + sed_g
   223	
   224	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   225	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   226	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   227	
   228	    # Precipitation
   229	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   230	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   231	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   232	    precip_g = jnp.clip(q_g[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_g[:, -1], 0.0)
   233	    precipitation = precip_r + precip_i + precip_s + precip_g
   234	
   235	    return MicrophysicsOutput(
   236	        dT_dt=dT_dt,
   237	        dq_v_dt=dq_v_dt,
   238	        dq_c_dt=dq_c_dt,
   239	        dq_r_dt=dq_r_dt,
   240	        dq_i_dt=dq_i_dt,
   241	        dq_s_dt=dq_s_dt,
   242	        dq_g_dt=dq_g_dt,
   243	        dN_c_dt=dN_c_dt,
   244	        dN_r_dt=dN_r_dt,
   245	        dN_i_dt=dN_i_dt,
   246	        precipitation=precipitation,
   247	    )

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/config.py | sed -n '250,430p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Hines (1997) Doppler-spread gravity wave drag parameterization.
     2	
     3	Non-orographic GWD scheme based on Doppler shifting and spectral
     4	saturation of gravity waves. Uses bottom-up propagation with smooth
     5	sigmoid activation for full differentiability.
     6	
     7	References
     8	----------
     9	- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
    10	  momentum deposition in the middle atmosphere. 1. Basic formulation.
    11	  J. Atmos. Solar-Terr. Phys., 59, 371-386.
    12	"""
    13	
    14	from __future__ import annotations
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm import constants
    20	from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
    21	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    22	
    23	
    24	def hines_gwd(
    25	    u: jax.Array,
    26	    v: jax.Array,
    27	    T: jax.Array,
    28	    p_full: jax.Array,
    29	    p_half: jax.Array,
    30	    z_full: jax.Array,
    31	    z_half: jax.Array,
    32	    rho: jax.Array,
    33	    lat: jax.Array,
    34	    dt: float,
    35	    config: HinesConfig,
    36	) -> GWDOutput:
    37	    """Compute Hines Doppler-spread GWD tendencies.
    38	
    39	    Parameters
    40	    ----------
    41	    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
    42	        Standard GWD backend signature. All column arrays (ncol, nlev).
    43	
    44	    Returns
    45	    -------
    46	    GWDOutput
    47	    """
    48	    ncol, nlev = u.shape
    49	
    50	    # Brunt-Väisälä frequency
    51	    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    52	    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    53	    dz_full = jnp.clip(dz_full, 1.0, None)
    54	    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    55	    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    56	    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    57	    N2_half = jnp.clip(N2_half, 1e-8, None)
    58	    N_half = jnp.sqrt(N2_half)
    59	
    60	    N_full = jnp.concatenate([
    61	        N_half[:, :1],
    62	        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
    63	        N_half[:, -1:],
    64	    ], axis=1)
    65	
    66	    # Wind magnitude at each level
    67	    U_mag = jnp.sqrt(u ** 2 + v ** 2 + 1e-10)
    68	
    69	    # Layer thickness
    70	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    71	    dz = jnp.clip(dz, 1.0, None)
    72	
    73	    # Saturation amplitude per level:
    74	    # As gravity waves propagate upward, their amplitude grows with
    75	    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
    76	    # Saturation occurs when the wave-induced velocity perturbation
    77	    # reaches the critical Doppler-broadening amplitude
    78	    # ``sigma_critical = N / m_*`` (Hines 1997 eq. 9), which is constant
    79	    # per column at fixed N and m_*.  An earlier formulation divided this
    80	    # by ``rho_ratio = sqrt(rho_sfc/rho) ≥ 1`` ⇒ ``sigma_sat`` *decreased*
    81	    # with altitude, the opposite of physical expectation: amplitudes
    82	    # grow with altitude (1/sqrt(rho)) so the cap should remain at least
    83	    # constant.  The /rho_ratio factor caused premature saturation aloft
    84	    # and biased the drag deposition lower in the column (audit GWD-B2).
    85	    sigma_sat = N_full / jnp.clip(config.m_star, 1e-6, None)
    86	
    87	    # Per-level WKB growth factor for the bottom-up scan.  Going from
    88	    # level (k+1) to level k (one step upward), the amplitude grows by
    89	    # ``sqrt(rho[k+1] / rho[k])`` (energy conservation rho * sigma^2).
    90	    # The carry already contains the integrated WKB amplitude from the
    91	    # surface to level k+1, so we multiply by the *inter-level* ratio,
    92	    # not the cumulative ``sqrt(rho_sfc/rho_k)``.  Multiplying by the
    93	    # cumulative factor at every step compounds the growth and
    94	    # over-amplifies the wave by a product of cumulative ratios — a
    95	    # bug masked in operational use only because the sigma_sat cap
    96	    # truncates the runaway.
    97	    rho_ratio_step = jnp.ones_like(rho)
    98	    rho_ratio_step = rho_ratio_step.at[:, :-1].set(
    99	        jnp.sqrt(jnp.clip(
   100	            rho[:, 1:] / jnp.clip(rho[:, :-1], 0.01, None), 1.0, None,
   101	        ))
   102	    )
   103	
   104	    # Bottom-up scan: propagate sigma_gw upward from surface.
   105	    # ``rho_ratio_step[:, k]`` carries amplitude from level k+1 to level k;
   106	    # at the surface (k = nlev-1) the step factor is 1 (initial condition).
   107	    def scan_fn(carry, k_rev):
   108	        sigma_gw = carry
   109	        k = nlev - 1 - k_rev
   110	
   111	        # Amplitude growth from density decrease (single-layer step)
   112	        sigma_grown = sigma_gw * rho_ratio_step[:, k]
   113	
   114	        # Dissipation where grown amplitude exceeds saturation
   115	        f_diss = jax.nn.sigmoid(
   116	            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
   117	        )
   118	        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss
   119	
   120	        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient.
   121	        # Clamp the lower bound to zero: the smooth ``f_diss`` sigmoid does
   122	        # not vanish exactly when ``sigma_grown < sigma_sat``, so without
   123	        # the floor a small "anti-drag" leak can appear in the transition
   124	        # region (``sigma_new`` slightly larger than ``sigma_grown`` ⇒ drag
   125	        # negative ⇒ accel positive ⇒ wave accelerates the resolved flow).
   126	        # GWD on the mean flow is always a momentum sink, never a source.
   127	        drag = (sigma_grown - sigma_new) * rho[:, k]
   128	        drag = jnp.clip(drag, 0.0, config.Fmax)
   129	
   130	        return sigma_new, drag
   131	
   132	    # Pin the carry dtype so the scan body stays at the input precision
   133	    # (defaulting allows x64 to silently promote the launch wind to f64).
   134	    sigma_gw_init = jnp.full((ncol,), config.total_rms_wind, dtype=u.dtype)
   135	    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
   136	    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
   137	
   138	    # Convert to acceleration
   139	    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
   140	
   141	    cos_a = u / jnp.clip(U_mag, config.U_mag_floor, None)
   142	    sin_a = v / jnp.clip(U_mag, config.U_mag_floor, None)
   143	    du_dt = accel * cos_a
   144	    dv_dt = accel * sin_a
   145	
   146	    # Frictional heating
   147	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
   148	
   149	    # Column dissipation (positive-definite: KE lost by the mean flow)
   150	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
   151	
   152	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

 succeeded in 0ms:
   250	    cape_consumption_time: float = 1800.0
   251	    parcel_perturb_T: float = 0.5
   252	    parcel_perturb_q: float = 1.0e-3
   253	    epsilon_0: float = 2.0e-3
   254	    delta_0: float = 2.0e-3
   255	    cloud_depth_min: float = 4000.0
   256	    cloud_depth_sharpness: float = 1.0e-3
   257	    enable_shallow: bool = True
   258	    cape_threshold: float = 0.0
   259	    cape_sharpness: float = 0.1
   260	    M_b_max: float = 0.05
   261	
   262	
   263	class EmanuelConfig(NamedTuple):
   264	    """Configuration for the Emanuel (1991) buoyancy-sorting scheme.
   265	
   266	    Single-plume mass-flux scheme with a buoyancy-sorted ensemble of
   267	    mixed parcels: at every cloud level the parcel may mix with
   268	    environmental air in a discrete spectrum of mixing fractions; the
   269	    fractions with positive buoyancy continue to ascend while the
   270	    fractions with negative buoyancy descend.  The smooth-everywhere
   271	    formulation replaces the hard ascend/descend switch with a
   272	    sigmoid weighting on buoyancy.
   273	
   274	    Distinct from Zhang-McFarlane (single bulk plume) and Kain-Fritsch
   275	    (single plume with deep/shallow blend) by the per-level
   276	    distribution of detrainment that the buoyancy-sorted ensemble
   277	    produces.
   278	
   279	    Fields
   280	    ------
   281	    n_mixing_fractions : int
   282	        Number of discrete mixing fractions in the buoyancy-sort
   283	        ensemble.  Default 8 — Emanuel 1991 uses 50; the smaller
   284	        value here is a cost / accuracy compromise.
   285	    cu_coefficient : float
   286	        Entrainment scale factor (Emanuel's α).  Default 0.7.
   287	    precip_efficiency_water : float
   288	        Precipitation efficiency above LCL (default 1.0).
   289	    precip_efficiency_lcl : float
   290	        Precipitation efficiency below LCL (default 0.0).
   291	    precip_threshold_qc : float
   292	        Cloud-water threshold above which precipitation falls
   293	        [kg/kg] (default 1e-3).
   294	    cape_threshold : float
   295	        CAPE gate [J/kg] (default 70.0).
   296	    cape_sharpness : float
   297	        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
   298	    parcel_perturb_T : float
   299	        Sub-cloud parcel temperature perturbation [K] (default 0.5).
   300	    parcel_perturb_q : float
   301	        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
   302	    sub_cloud_relaxation : float
   303	        Sub-cloud layer mixing timescale [s] (default 100.0).
   304	    enable_unsaturated_downdraft : bool
   305	        Whether to include the unsaturated downdraft branch (rain
   306	        evaporation cooling) (default ``True``).
   307	    downdraft_efficiency : float
   308	        Fraction of precipitation that re-evaporates below cloud base
   309	        in the downdraft (default 0.2).
   310	    smooth_trigger_sharpness : float
   311	        Sigmoid sharpness on the buoyancy-sort weighting [1/K]
   312	        (default 0.5).
   313	    epsilon_0 : float
   314	        Bulk-plume entrainment rate [1/m] (default 1.5e-3).
   315	    delta_0 : float
   316	        Bulk-plume detrainment rate [1/m] (default 1.5e-3).
   317	    M_b_max : float
   318	        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
   319	        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
   320	    """
   321	    n_mixing_fractions: int = 8
   322	    cu_coefficient: float = 0.7
   323	    precip_efficiency_water: float = 1.0
   324	    precip_efficiency_lcl: float = 0.0
   325	    precip_threshold_qc: float = 1.0e-3
   326	    cape_threshold: float = 70.0
   327	    cape_sharpness: float = 0.1
   328	    parcel_perturb_T: float = 0.5
   329	    parcel_perturb_q: float = 1.0e-3
   330	    # Emanuel 1991 §3 uses a sub-cloud-layer mixing timescale of
   331	    # several thousand seconds.  The earlier default of 100 s gave
   332	    # M_b ~72× larger than published values and produced 28 MW/m²
   333	    # of column heating from a CAPE-positive sounding.
   334	    sub_cloud_relaxation: float = 7200.0
   335	    # Default-OFF.  Emanuel 1991's downdraft re-evaporates a fraction
   336	    # of *precipitation* (rain) back to vapor in the BL.  In a model
   337	    # without an explicit q_r tracer the implementation can only draw
   338	    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
   339	    # produces a column-net moistening on CAPE-positive soundings —
   340	    # the wrong sign of ``Q_v`` that the validation script flags.
   341	    # Production runs with a full microphysics chain that owns q_r
   342	    # should override this to ``True``.
   343	    enable_unsaturated_downdraft: bool = False
   344	    downdraft_efficiency: float = 0.2
   345	    smooth_trigger_sharpness: float = 0.5
   346	    epsilon_0: float = 1.5e-3
   347	    delta_0: float = 1.5e-3
   348	    M_b_max: float = 0.05
   349	
   350	
   351	class TiedtkeConfig(NamedTuple):
   352	    """Configuration for the Tiedtke (1989) bulk mass-flux scheme.
   353	
   354	    Three-class scheme with deep, mid-level, and shallow branches
   355	    blended on cloud depth.  Downdraft is included with an RH-based
   356	    trigger; convective momentum transport via Gregory et al. 1997.
   357	    First scheme that actually exercises the new ``(ncol, nlev)``
   358	    profile carry for ``M_u(k)``.
   359	
   360	    The full Tiedtke 1989 closure uses column moisture convergence
   361	    for the deep branch.  Until the PR-0 ``compute_moisture_convergence``
   362	    diagnostic ships, we use a saturation-deficit proxy
   363	    ``MC_proxy = (q_sat - q_v) / tau_relax`` that has the same
   364	    qualitative behavior (positive in moist columns, zero in dry
   365	    columns).
   366	
   367	    Fields
   368	    ------
   369	    epsilon_deep, delta_deep : float
   370	        Entrainment / detrainment rates for deep branch [1/m].
   371	    epsilon_shallow, delta_shallow : float
   372	        Same for shallow branch.
   373	    epsilon_midlevel, delta_midlevel : float
   374	        Same for mid-level branch.
   375	    enable_downdraft : bool
   376	        Whether to include the downdraft branch (default ``True``).
   377	    downdraft_alpha : float
   378	        Downdraft / updraft mass flux ratio at LFS (default 0.3).
   379	    downdraft_RH_min : float
   380	        Below this column-mean RH the downdraft fires (default 0.2).
   381	    moisture_convergence_threshold : float
   382	        Saturation-deficit proxy threshold [kg/kg/s].  Below this the
   383	        deep branch is suppressed (default 1e-8).
   384	    moisture_convergence_sharpness : float
   385	        Sigmoid sharpness on the MC threshold [s/(kg/kg)] (default 1e8).
   386	    cape_threshold : float
   387	        Secondary CAPE gate [J/kg] (default 70.0).
   388	    cape_sharpness : float
   389	        Sigmoid sharpness on CAPE gate (default 0.02).
   390	    cloud_depth_deep : float
   391	        Depth threshold separating mid-level from deep branches [m]
   392	        (default 3000.0).
   393	    cloud_depth_shallow_max : float
   394	        Depth threshold separating shallow from mid-level branches
   395	        [m] (default 1500.0).
   396	    depth_split_sharpness : float
   397	        Sigmoid sharpness on the depth thresholds [1/m] (default 1e-3).
   398	    enable_cmt : bool
   399	        Whether to compute CMT (default ``True``).
   400	    cmt_c_u, cmt_c_d : float
   401	        Gregory et al. 1997 closure coefficients (default 0.7).
   402	    smooth_trigger_sharpness : float
   403	        Sigmoid sharpness on the buoyancy / RH soft triggers (default
   404	        0.02).
   405	    precip_efficiency : float
   406	        Fraction of detrained condensate that precipitates (default 0.5).
   407	    tau_M_u_relax : float
   408	        Implicit-Euler relaxation timescale [s] for the profile carry
   409	        ``M_u`` toward its diagnosed equilibrium (default 1800.0).
   410	    parcel_dT, parcel_dq : float
   411	        Sub-cloud parcel perturbations (defaults 0.5 K, 1e-3 kg/kg).
   412	    """
   413	    epsilon_deep: float = 1.0e-4
   414	    delta_deep: float = 1.0e-4
   415	    epsilon_shallow: float = 3.0e-4
   416	    delta_shallow: float = 3.0e-4
   417	    epsilon_midlevel: float = 1.0e-4
   418	    delta_midlevel: float = 2.0e-4
   419	    enable_downdraft: bool = True
   420	    downdraft_alpha: float = 0.3
   421	    downdraft_RH_min: float = 0.2
   422	    # Fraction of the downdraft mass flux that re-evaporates as rain
   423	    # falling through the subcloud layer (default 0.05 — matches a
   424	    # historical hardcoded literal that was previously dimensionally
   425	    # wrong; the current implementation distributes the resulting
   426	    # evaporation rate over below-LCL layers by mass weight, with a
   427	    # matching dq_v source so the column water budget closes).
   428	    downdraft_evap_efficiency: float = 0.05
   429	    moisture_convergence_threshold: float = 1.0e-8
   430	    moisture_convergence_sharpness: float = 1.0e8

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/config.py | sed -n '430,620p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   430	    moisture_convergence_sharpness: float = 1.0e8
   431	    cape_threshold: float = 70.0
   432	    cape_sharpness: float = 0.1
   433	    cloud_depth_deep: float = 3000.0
   434	    cloud_depth_shallow_max: float = 1500.0
   435	    depth_split_sharpness: float = 1.0e-3
   436	    enable_cmt: bool = True
   437	    cmt_c_u: float = 0.7
   438	    cmt_c_d: float = 0.7
   439	    smooth_trigger_sharpness: float = 0.02
   440	    precip_efficiency: float = 0.5
   441	    tau_M_u_relax: float = 1800.0
   442	    parcel_dT: float = 0.5
   443	    parcel_dq: float = 1.0e-3
   444	    tau_MC_proxy: float = 3600.0   # for saturation-excess MC proxy
   445	    # Critical column-mean RH above which the MC proxy starts firing.
   446	    # The proxy approximates moisture convergence as the column-integrated
   447	    # vapor in excess of ``RH_crit * q_sat``: positive in moist columns,
   448	    # vanishing in dry ones.
   449	    mc_proxy_RH_crit: float = 0.6
   450	    tau_shallow_M_b: float = 3600.0  # Shallow-cloud-base mass-flux timescale [s]
   451	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   452	    midlevel_M_b_fraction: float = 0.5  # M_b_midlevel = M_b_shallow * this
   453	
   454	
   455	class BechtoldConfig(NamedTuple):
   456	    """Configuration for the Bechtold/IFS convection scheme.
   457	
   458	    Builds on :class:`TiedtkeConfig` (three-class blend, downdraft,
   459	    CMT) with two distinguishing features:
   460	
   461	    * **PBL-CAPE / departure-CAPE closure** (Bechtold 2008):
   462	      ``M_b ∝ (CAPE_pbl - CAPE_eq)+ / tau_bl`` where ``CAPE_pbl`` is
   463	      diagnosed from a mass-weighted parcel within the boundary layer
   464	      rather than the surface parcel.
   465	    * **AR1 stochastic perturbation** (Bechtold 2014): ``M_b *= (1 +
   466	      amplitude * ε)`` where ``ε`` is an AR1-process realization with
   467	      decorrelation timescale ``stochastic_decorrelation``.  When
   468	      ``enable_stochastic`` is ``False`` the multiplier is 1.
   469	
   470	    Stochasticity defaults to OFF for reproducibility.  When enabled,
   471	    the leaf consumes a ``prng_key`` argument; the convection bridge
   472	    splits ``PhysicsState.prng_key`` into a Bechtold sub-key (folded
   473	    with module id ``0xBEC4``) and an advanced master key, returning
   474	    the latter as part of the multi-field carry update so subsequent
   475	    steps see independent random streams.
   476	
   477	    Inherits sensible defaults from Tiedtke 1989 with the entrainment
   478	    revision from Bechtold et al. 2008.
   479	
   480	    Fields
   481	    ------
   482	    epsilon_deep, delta_deep : float
   483	        Deep-branch entrainment / detrainment.  Default 1.75e-3 /
   484	        5e-4 (Bechtold et al. 2008 calibration — deeper entrainment
   485	        than Tiedtke 1989).
   486	    epsilon_shallow, delta_shallow : float
   487	        Shallow-branch (default 3e-3, 3e-3).
   488	    epsilon_midlevel, delta_midlevel : float
   489	        Mid-level branch (default 1e-4, 2e-4).
   490	    cape_pbl_depth : float
   491	        PBL depth [m] for the parcel-source mass weighting (default
   492	        500.0).
   493	    tau_bl : float
   494	        PBL closure timescale [s] (default 3600.0).
   495	    enable_stochastic : bool
   496	        Whether to apply the AR1 stochastic perturbation (default
   497	        ``False`` — reproducibility).
   498	    stochastic_amplitude : float
   499	        Multiplicative perturbation amplitude (default 0.5).
   500	    stochastic_decorrelation : float
   501	        AR1 decorrelation timescale [s] (default 7200.0).
   502	    enable_downdraft, downdraft_alpha, downdraft_RH_min : as Tiedtke.
   503	    enable_cmt, cmt_c_u, cmt_c_d : as Tiedtke.
   504	    cape_threshold, cape_sharpness, smooth_trigger_sharpness,
   505	    precip_efficiency, parcel_dT, parcel_dq, tau_M_u_relax,
   506	    cloud_depth_deep, cloud_depth_shallow_max, depth_split_sharpness :
   507	        as Tiedtke.
   508	    """
   509	    # Tiedtke-inherited / revised.
   510	    # Bechtold 2008 §2 calibrates ``delta_deep ≈ epsilon_deep`` for a
   511	    # near-neutral plume; the earlier default ``delta_deep = 5e-4`` (with
   512	    # ``epsilon_deep = 1.75e-3``) gives ``dM/dz ≈ +1.25e-3 M`` so the
   513	    # mass flux *grows* exponentially with height and peaks at the
   514	    # model top, not the cloud base — physically wrong.  Setting
   515	    # ``delta_deep = epsilon_deep`` matches the published calibration.
   516	    epsilon_deep: float = 1.75e-3
   517	    delta_deep: float = 1.75e-3
   518	    epsilon_shallow: float = 3.0e-3
   519	    delta_shallow: float = 3.0e-3
   520	    epsilon_midlevel: float = 1.0e-4
   521	    delta_midlevel: float = 2.0e-4
   522	    enable_downdraft: bool = True
   523	    downdraft_alpha: float = 0.3
   524	    downdraft_RH_min: float = 0.2
   525	    # See TiedtkeConfig.downdraft_evap_efficiency for definition.
   526	    downdraft_evap_efficiency: float = 0.05
   527	    enable_cmt: bool = True
   528	    cmt_c_u: float = 0.7
   529	    cmt_c_d: float = 0.7
   530	    # The earlier ``cape_threshold = 0.0`` with ``cape_sharpness = 0.005``
   531	    # left the closure essentially always-on (``softplus(0)/0.005 ≈ 138
   532	    # J/kg`` of phantom CAPE even when CAPE = 0).  Match ZM/Tiedtke at
   533	    # 70 J/kg, 0.02 1/(J/kg) so the trigger is meaningful.
   534	    cape_threshold: float = 70.0
   535	    cape_sharpness: float = 0.1
   536	    smooth_trigger_sharpness: float = 0.02
   537	    precip_efficiency: float = 0.55
   538	    parcel_dT: float = 0.5
   539	    parcel_dq: float = 1.0e-3
   540	    tau_M_u_relax: float = 1800.0
   541	    cloud_depth_deep: float = 3000.0
   542	    cloud_depth_shallow_max: float = 1500.0
   543	    depth_split_sharpness: float = 1.0e-3
   544	    # Bechtold-specific
   545	    use_pbl_cape: bool = True
   546	    cape_pbl_depth: float = 500.0
   547	    tau_bl: float = 3600.0
   548	    enable_stochastic: bool = False
   549	    stochastic_amplitude: float = 0.5
   550	    stochastic_decorrelation: float = 7200.0
   551	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   552	    # Strong-convergence normaliser used to make the moisture-convergence
   553	    # enhancement an O(1) multiplier of M_b (Bechtold 2008 Fig. 2 — typical
   554	    # tropical strong-convergence is ≈ 0.05 kg/m²/s).
   555	    mc_normalize_scale: float = 0.05
   556	
   557	
   558	class EDMFConfig(NamedTuple):
   559	    """Configuration for simplified EDMF convection (mass-flux part only).
   560	
   561	    Fields
   562	    ------
   563	    epsilon_0 : float
   564	        Entrainment rate [1/m].
   565	    delta_0 : float
   566	        Detrainment rate [1/m].
   567	    a_u_init : float
   568	        Initial updraft area fraction.
   569	    tau_a : float
   570	        Relaxation timescale for a_u [s].
   571	    w_u_min : float
   572	        Minimum updraft velocity [m/s].
   573	    cape_activation_scale : float
   574	        Sigmoid scale for CAPE trigger [J/kg].
   575	    cape_threshold : float
   576	        CAPE threshold [J/kg].
   577	    """
   578	    epsilon_0: float = 2e-3
   579	    delta_0: float = 2e-3
   580	    a_u_init: float = 0.1
   581	    tau_a: float = 1800.0
   582	    w_u_min: float = 0.1
   583	    # Trigger gating — see MassFluxConfig for rationale (sharper
   584	    # ``cape_activation_scale`` and a 70 J/kg threshold close the
   585	    # CAPE=0 leak from the earlier 50 % activation).
   586	    cape_activation_scale: float = 10.0
   587	    cape_threshold: float = 70.0
   588	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   589	
   590	
   591	class ConvectionConfig(NamedTuple):
   592	    """Top-level convection configuration.
   593	
   594	    Selects the active scheme and holds sub-configurations.
   595	
   596	    Fields
   597	    ------
   598	    scheme : str
   599	        Active convection scheme: ``"sbm"``, ``"dca"``, ``"kuo"``,
   600	        ``"mass_flux"``, ``"edmf"``, ``"zhang_mcfarlane"``, or
   601	        ``"none"``.  Future PRs (KF, Emanuel, Tiedtke, Bechtold) add
   602	        their literal here.
   603	    sbm, dca, kuo, mass_flux, edmf, zhang_mcfarlane :
   604	        Per-scheme configuration NamedTuples.
   605	    update_interval_steps : int
   606	        Recompute convection every N time steps (1 = every step).
   607	    """
   608	    scheme: str = "sbm"
   609	    sbm: SBMConfig = SBMConfig()
   610	    dca: DCAConfig = DCAConfig()
   611	    kuo: KuoConfig = KuoConfig()
   612	    mass_flux: MassFluxConfig = MassFluxConfig()
   613	    edmf: EDMFConfig = EDMFConfig()
   614	    zhang_mcfarlane: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig()
   615	    kain_fritsch: KainFritschConfig = KainFritschConfig()
   616	    emanuel: EmanuelConfig = EmanuelConfig()
   617	    tiedtke: TiedtkeConfig = TiedtkeConfig()
   618	    bechtold: BechtoldConfig = BechtoldConfig()
   619	    update_interval_steps: int = 1

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/output.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "rg -n \"def smooth_lowest_crossing_index|def smooth_level_indicator|def smooth_positive_part|def smooth_step\" src/legoesm/atmosphere/physics/convection/_triggers.py && nl -ba src/legoesm/atmosphere/physics/convection/_triggers.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Microphysics output containers.
     2	
     3	HydrometeorState holds the prognostic hydrometeor fields passed to backends.
     4	MicrophysicsOutput is the common interface returned by all backends.
     5	
     6	All backends accept and return the same containers so that integration
     7	code can be backend-agnostic.
     8	"""
     9	
    10	from __future__ import annotations
    11	
    12	from typing import NamedTuple
    13	
    14	import jax
    15	import jax.numpy as jnp
    16	
    17	
    18	class HydrometeorState(NamedTuple):
    19	    """Hydrometeor state for backends. All fields shape (ncol, nlev)."""
    20	    q_c: jax.Array    # cloud water [kg/kg]
    21	    q_r: jax.Array    # rain water [kg/kg]
    22	    q_i: jax.Array    # cloud ice [kg/kg]
    23	    q_s: jax.Array    # snow [kg/kg]
    24	    q_g: jax.Array    # graupel [kg/kg]
    25	    N_c: jax.Array    # cloud droplet number [1/kg]
    26	    N_r: jax.Array    # rain drop number [1/kg]
    27	    N_i: jax.Array    # ice crystal number [1/kg]
    28	
    29	
    30	class MicrophysicsOutput(NamedTuple):
    31	    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,)."""
    32	    dT_dt: jax.Array          # latent heating [K/s]
    33	    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    34	    dq_c_dt: jax.Array        # cloud water tendency
    35	    dq_r_dt: jax.Array        # rain tendency
    36	    dq_i_dt: jax.Array        # ice tendency
    37	    dq_s_dt: jax.Array        # snow tendency
    38	    dq_g_dt: jax.Array        # graupel tendency
    39	    dN_c_dt: jax.Array        # cloud number tendency [1/kg/s]
    40	    dN_r_dt: jax.Array        # rain number tendency
    41	    dN_i_dt: jax.Array        # ice number tendency
    42	    precipitation: jax.Array  # surface precip [kg/m^2/s]
    43	
    44	
    45	def make_zero_hydrometeors(
    46	    ncol: int, nlev: int, dtype=None,
    47	) -> HydrometeorState:
    48	    """Create a zero-initialized HydrometeorState.
    49	
    50	    ``dtype`` defaults to the JAX default float (``float64`` under x64,
    51	    ``float32`` otherwise).  Callers integrating with the column physics
    52	    pipeline should pass the upstream state dtype explicitly so this
    53	    fallback never silently promotes a float32 column path to float64.
    54	    """
    55	    z = jnp.zeros((ncol, nlev), dtype=dtype)
    56	    return HydrometeorState(
    57	        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
    58	        N_c=z, N_r=z, N_i=z,
    59	    )
    60	
    61	
    62	def make_zero_output(
    63	    ncol: int, nlev: int, dtype=None,
    64	) -> MicrophysicsOutput:
    65	    """Create a zero-initialized MicrophysicsOutput.
    66	
    67	    ``dtype`` is forwarded to ``jnp.zeros`` for the same reason as
    68	    ``make_zero_hydrometeors``: defaulting allows x64 mode to silently
    69	    promote the precip path.
    70	    """
    71	    z2 = jnp.zeros((ncol, nlev), dtype=dtype)
    72	    z1 = jnp.zeros((ncol,), dtype=dtype)
    73	    return MicrophysicsOutput(
    74	        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
    75	        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
    76	        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
    77	        precipitation=z1,
    78	    )
    79	
    80	
    81	def sedimentation_tendency(
    82	    q: jax.Array,
    83	    rho: jax.Array,
    84	    V_t: jax.Array,
    85	    dz: jax.Array,
    86	) -> jax.Array:
    87	    """Compute sedimentation tendency from vertical flux divergence.
    88	
    89	    Parameters
    90	    ----------
    91	    q : jax.Array
    92	        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    93	    rho : jax.Array
    94	        Air density [kg/m^3], shape (ncol, nlev).
    95	    V_t : jax.Array
    96	        Terminal velocity [m/s], shape (ncol, nlev).
    97	    dz : jax.Array
    98	        Layer thickness [m], shape (ncol, nlev).
    99	
   100	    Returns
   101	    -------
   102	    jax.Array
   103	        Sedimentation tendency [kg/kg/s], shape (ncol, nlev).
   104	    """
   105	    q_pos = jnp.clip(q, 0.0, None)
   106	    flux = V_t * q_pos * rho  # (ncol, nlev)
   107	
   108	    # Flux from above: zero at top, flux[k-1] enters level k.  Use
   109	    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
   110	    # zero buffer + concatenate.
   111	    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
   112	    dz_safe = jnp.clip(dz, 1.0, None)
   113	    return (flux_in - flux) / (rho * dz_safe)

 succeeded in 0ms:
58:def smooth_step(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
125:def smooth_positive_part(
160:def smooth_level_indicator(
201:def smooth_lowest_crossing_index(
     1	"""Smooth (differentiable) trigger and level-membership primitives.
     2	
     3	Shared helpers used by every convection scheme that needs a smooth
     4	replacement for a hard ``if``/``where`` branch in the reference paper.
     5	The legoESM model is trained via :func:`jax.grad` and
     6	``eqx.filter_value_and_grad`` — every trigger threshold therefore must
     7	have a non-zero gradient on both sides of the threshold so that
     8	:class:`legoesm.training.physics_params.TrainablePhysicsParams` and the
     9	neural-physics blend can flow gradients through the convection step.
    10	
    11	The conceptual split with :mod:`._plume`:
    12	
    13	* This module is **sigmoid math**: scalar-to-scalar smooth
    14	  approximations of step / max / positive-part / lowest-crossing-index
    15	  primitives, plus a thin convenience wrapper for the CAPE > threshold
    16	  trigger reused in four (now nine) convection backends.
    17	* :mod:`._plume` is **column physics math**: LCL / LFC / LNB / CIN
    18	  diagnostics, the entraining-detraining plume integrator,
    19	  unsaturated-downdraft thermodynamics, the Emanuel buoyancy-sorting
    20	  step, and the Gregory et al. 1997 convective momentum-transport
    21	  closure.
    22	
    23	The level-membership and lowest-crossing helpers consume profiles in
    24	the canonical legoESM convention: shape ``(ncol, nlev)`` with the
    25	**surface at the last index** ``[:, -1]`` and the model top at
    26	``[:, 0]``.  Indices returned by :func:`smooth_lowest_crossing_index`
    27	are in the same surface-last convention (``nlev - 1`` is the surface,
    28	``0`` is the top).
    29	
    30	References
    31	----------
    32	- Pattern for the soft-crossing × log-sum-exp softmin transcribed from
    33	  ``legoesm.atmosphere.physics.turbulence.pbl_height.diagnose_pbl_height_interp``.
    34	"""
    35	
    36	from __future__ import annotations
    37	
    38	import jax
    39	import jax.numpy as jnp
    40	
    41	
    42	__all__ = (
    43	    "smooth_step",
    44	    "smooth_heaviside",
    45	    "smooth_max",
    46	    "smooth_min",
    47	    "smooth_positive_part",
    48	    "smooth_level_indicator",
    49	    "smooth_lowest_crossing_index",
    50	    "cape_trigger",
    51	)
    52	
    53	
    54	# ---------------------------------------------------------------------------
    55	# Scalar / elementwise smooth primitives
    56	# ---------------------------------------------------------------------------
    57	
    58	def smooth_step(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
    59	    """Differentiable approximation of the unit step function.
    60	
    61	    ``smooth_step(x, s) = sigmoid(s * x) ∈ (0, 1)``.
    62	
    63	    The result tends to ``Heaviside(x)`` as ``sharpness → ∞`` while
    64	    remaining strictly differentiable for any finite ``sharpness``.
    65	    Gradient at ``x = 0`` is ``sharpness / 4``.
    66	
    67	    Parameters
    68	    ----------
    69	    x : jax.Array
    70	        Argument (typically a difference ``value - threshold``).
    71	    sharpness : float
    72	        Inverse-width of the transition.  Larger values approach a
    73	        sharp step; smaller values broaden the transition.
    74	
    75	    Returns
    76	    -------
    77	    jax.Array
    78	        Same shape and dtype family as ``x``, values in ``(0, 1)``.
    79	    """
    80	    return jax.nn.sigmoid(sharpness * x)
    81	
    82	
    83	def smooth_heaviside(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
    84	    """Alias for :func:`smooth_step` for callers that prefer the
    85	    Heaviside name at trigger sites."""
    86	    return smooth_step(x, sharpness)
    87	
    88	
    89	def smooth_max(
    90	    a: jax.Array,
    91	    b: jax.Array,
    92	    sharpness: float = 1.0,
    93	) -> jax.Array:
    94	    """Differentiable upper bound on ``max(a, b)``.
    95	
    96	    Implements the log-sum-exp soft-max ::
    97	
    98	        smooth_max(a, b, s) = (1/s) * log(exp(s*a) + exp(s*b)).
    99	
   100	    Properties:
   101	
   102	    * ``smooth_max(a, b, s) >= max(a, b)`` for all finite ``s > 0``;
   103	      the inequality tightens to equality as ``s → ∞``.
   104	    * Symmetric in ``a`` and ``b``.
   105	    * Gradients are well-defined everywhere, including at ``a == b``.
   106	
   107	    Implemented with :func:`jax.numpy.logaddexp` for numerical
   108	    stability across magnitudes (avoids ``exp`` overflow).
   109	    """
   110	    return jnp.logaddexp(sharpness * a, sharpness * b) / sharpness
   111	
   112	
   113	def smooth_min(
   114	    a: jax.Array,
   115	    b: jax.Array,
   116	    sharpness: float = 1.0,
   117	) -> jax.Array:
   118	    """Differentiable lower bound on ``min(a, b)``.
   119	
   120	    ``smooth_min(a, b, s) = -smooth_max(-a, -b, s)``.
   121	    """
   122	    return -smooth_max(-a, -b, sharpness)
   123	
   124	
   125	def smooth_positive_part(
   126	    x: jax.Array,
   127	    sharpness: float = 1.0,
   128	) -> jax.Array:
   129	    """Differentiable approximation of ``max(x, 0)``.
   130	
   131	    Returns ``softplus(sharpness * x) / sharpness``, equivalent to
   132	    :func:`smooth_max` against zero.  The gradient at ``x = 0`` is
   133	    ``0.5``, transitioning smoothly to ``1`` for ``x ≫ 0`` and ``0``
   134	    for ``x ≪ 0``.
   135	
   136	    Used wherever the reference paper writes ``(... )_+`` — most
   137	    notably the CAPE-relaxation cloud-base mass-flux closure
   138	    ``M_b ∝ (CAPE - CAPE_threshold)_+ / tau``.
   139	
   140	    Parameters
   141	    ----------
   142	    x : jax.Array
   143	        Argument.  Any shape.
   144	    sharpness : float
   145	        Larger values approach the kink at ``x = 0`` more sharply.
   146	
   147	    Returns
   148	    -------
   149	    jax.Array
   150	        Strictly positive everywhere; tends to ``max(x, 0)`` as
   151	        ``sharpness → ∞``.
   152	    """
   153	    return jax.nn.softplus(sharpness * x) / sharpness
   154	
   155	
   156	# ---------------------------------------------------------------------------
   157	# Per-level membership and crossing diagnostics
   158	# ---------------------------------------------------------------------------
   159	
   160	def smooth_level_indicator(
   161	    profile: jax.Array,
   162	    threshold: jax.Array | float,
   163	    sharpness: float,
   164	    direction: str = "above",
   165	) -> jax.Array:
   166	    """Per-level smooth membership in ``[0, 1]``.
   167	
   168	    Returns a per-level weight that smoothly transitions from 0 to 1
   169	    where the profile crosses the threshold.  Useful for
   170	    mass-weighted averages over a column subset (e.g. PBL-mean
   171	    parcel for the Bechtold/IFS departure-CAPE closure).
   172	
   173	    Parameters
   174	    ----------
   175	    profile : jax.Array, shape ``(..., nlev)``
   176	        Profile to test against the threshold.  Surface at last index.
   177	    threshold : jax.Array or float
   178	        Threshold value.  Broadcastable against ``profile``.
   179	    sharpness : float
   180	        Sigmoid sharpness on the threshold transition.
   181	    direction : str
   182	        ``"above"`` returns ``sigmoid(sharpness * (profile - threshold))``
   183	        (1 where profile is above threshold, 0 below).
   184	        ``"below"`` is the complement: 1 where profile is below
   185	        threshold (used to weight a PBL-depth average).
   186	
   187	    Returns
   188	    -------
   189	    jax.Array
   190	        Same shape as ``profile``; values in ``(0, 1)``.
   191	    """
   192	    if direction == "above":
   193	        return jax.nn.sigmoid(sharpness * (profile - threshold))
   194	    if direction == "below":
   195	        return jax.nn.sigmoid(sharpness * (threshold - profile))
   196	    raise ValueError(
   197	        f"direction must be 'above' or 'below', got {direction!r}"
   198	    )
   199	
   200	
   201	def smooth_lowest_crossing_index(
   202	    profile: jax.Array,
   203	    threshold: jax.Array | float,
   204	    sharpness: float,
   205	) -> jax.Array:
   206	    """Differentiable fractional index of the lowest upward crossing.
   207	
   208	    Scans each column from the surface upward (last index → first
   209	    index) and returns the fractional level index of the lowest
   210	    altitude where ``profile`` crosses ``threshold`` from below to
   211	    above.  The fractional component is a linear interpolation
   212	    between adjacent levels; the choice of which crossing to return
   213	    when several exist is made by a log-sum-exp softmin so that
   214	    columns with multiple candidate crossings still produce a smooth
   215	    (differentiable) answer.
   216	
   217	    Used for LFC / LCL / LNB localization where the reference paper
   218	    would normally take the first satisfying integer index.
   219	
   220	    Parameters
   221	    ----------
   222	    profile : jax.Array, shape ``(ncol, nlev)``
   223	        Per-level profile.  Surface at last index ``[:, -1]``.
   224	    threshold : jax.Array or float
   225	        Threshold value.  Scalar, ``(ncol,)``, or
   226	        ``(ncol, nlev-1)``-broadcastable.
   227	    sharpness : float
   228	        Sigmoid sharpness on the soft crossing indicator AND the
   229	        scale parameter for the softmin (larger = closer to a hard
   230	        argmin).
   231	
   232	    Returns
   233	    -------
   234	    jax.Array, shape ``(ncol,)``
   235	        Fractional index in surface-last coordinates: a return value
   236	        of ``ncol_index = nlev - 1`` means the surface, and ``0``
   237	        means the model top.  When no crossing is detected the
   238	        returned value falls back to ``nlev - 1`` (surface) — callers
   239	        that need a "no-crossing" sentinel should use
   240	        :func:`smooth_level_indicator` to compute a column-wide

exec
/bin/zsh -lc "rg -n \"safe_pow\\(|jnp\\.clip\\(["'^'"\\n]*\\) \\*|\\* safe_pow|\\* jnp\\.clip|\\*\\* 0\\.|\\*\\* \\(\" src/legoesm/atmosphere/physics/microphysics src/legoesm/atmosphere/physics/convection src/legoesm/atmosphere/physics/gravity_wave_drag -S" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:51:    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:56:    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
src/legoesm/atmosphere/physics/microphysics/thompson.py:122:        * jnp.clip(q_i, 0.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:123:        * safe_pow(N_i, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:132:    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
src/legoesm/atmosphere/physics/microphysics/thompson.py:135:    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/thompson.py:136:    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/thompson.py:140:    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/thompson.py:147:        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
src/legoesm/atmosphere/physics/microphysics/thompson.py:151:        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
src/legoesm/atmosphere/physics/microphysics/thompson.py:161:        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
src/legoesm/atmosphere/physics/microphysics/thompson.py:186:    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
src/legoesm/atmosphere/physics/microphysics/thompson.py:188:    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
src/legoesm/atmosphere/physics/microphysics/thompson.py:190:    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
src/legoesm/atmosphere/physics/microphysics/thompson.py:192:    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
src/legoesm/atmosphere/physics/microphysics/thompson.py:226:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/thompson.py:229:    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:230:    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:231:    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:232:    precip_g = jnp.clip(q_g[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_g[:, -1], 0.0)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:65:    k_bl = config.k_max * jnp.clip(
src/legoesm/atmosphere/physics/convection/bechtold.py:5:1. **PBL-CAPE / departure-CAPE closure** (Bechtold 2008).  ``M_b`` is
src/legoesm/atmosphere/physics/convection/bechtold.py:9:2. **AR1 stochastic perturbation** (Bechtold 2014).  ``M_b *= (1 +
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:16:def safe_pow(x, p):
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:24:    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:154:    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:182:    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:186:        jnp.clip(q_r, 0.0) * rho
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:190:    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:217:    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:61:    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:66:    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:126:        / (jnp.clip(N_4d, 1e-6, None) * wavelength[None, None, :, None])
src/legoesm/atmosphere/physics/convection/_plume.py:180:    p_lcl = p_parcel * (T_lcl / T_parcel) ** (constants.c_pd / constants.R_d)
src/legoesm/atmosphere/physics/convection/mass_flux.py:6:1. **Prognostic Mass-Flux** (``mass_flux_convection``): Arakawa-Wu type
src/legoesm/atmosphere/physics/convection/mass_flux.py:11:2. **Simplified EDMF** (``edmf_convection``): single-updraft mass-flux
src/legoesm/atmosphere/physics/convection/mass_flux.py:84:    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
src/legoesm/atmosphere/physics/convection/mass_flux.py:86:    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
src/legoesm/atmosphere/physics/convection/mass_flux.py:240:    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:50:    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:55:    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
src/legoesm/atmosphere/physics/microphysics/kessler.py:94:    accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)
src/legoesm/atmosphere/physics/microphysics/kessler.py:98:    evaporation = config.evaporation_coeff * subsaturation * safe_pow(q_r, 0.525)
src/legoesm/atmosphere/physics/convection/config.py:461:    * **PBL-CAPE / departure-CAPE closure** (Bechtold 2008):
src/legoesm/atmosphere/physics/convection/config.py:465:    * **AR1 stochastic perturbation** (Bechtold 2014): ``M_b *= (1 +
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:51:    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:56:    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:120:        / (jnp.clip(N_full, 1e-6, None) * envelope)
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:91:    V_t_r = config.a_v_r * safe_pow(
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:92:        jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1), config.b_v_r,
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:109:    precipitation = q_r_bot * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:101:        * jnp.clip(q_i, 0.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:102:        * safe_pow(N_i, 1.0 / 3.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:111:    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
src/legoesm/atmosphere/physics/microphysics/morrison.py:114:    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/morrison.py:115:    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/morrison.py:118:    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
src/legoesm/atmosphere/physics/microphysics/morrison.py:123:        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
src/legoesm/atmosphere/physics/microphysics/morrison.py:127:        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
src/legoesm/atmosphere/physics/microphysics/morrison.py:160:    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
src/legoesm/atmosphere/physics/microphysics/morrison.py:162:    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
src/legoesm/atmosphere/physics/microphysics/morrison.py:164:    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
src/legoesm/atmosphere/physics/microphysics/morrison.py:201:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/morrison.py:204:    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:205:    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:206:    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_triggers.py | sed -n '240,360p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   240	        :func:`smooth_level_indicator` to compute a column-wide
   241	        confidence first.
   242	    """
   243	    *_, nlev = profile.shape
   244	    if nlev < 2:
   245	        raise ValueError(
   246	            "smooth_lowest_crossing_index requires nlev >= 2; got "
   247	            f"profile.shape={profile.shape}"
   248	        )
   249	
   250	    # Reverse so that surface is first; matches the convention used in
   251	    # turbulence/pbl_height.py.
   252	    profile_rev = profile[..., ::-1]      # (ncol, nlev), surface-first
   253	
   254	    profile_below = profile_rev[..., :-1]  # (ncol, nlev-1)
   255	    profile_above = profile_rev[..., 1:]
   256	
   257	    # Soft upward-crossing indicator: simultaneously below threshold
   258	    # at the lower level and above at the upper level.
   259	    cross_weight = (
   260	        jax.nn.sigmoid(sharpness * (threshold - profile_below))
   261	        * jax.nn.sigmoid(sharpness * (profile_above - threshold))
   262	    )
   263	
   264	    # Linear interpolation fraction within each pair (in [0, 1]).
   265	    dprofile = profile_above - profile_below
   266	    # Numerical floor avoids 0/0 when adjacent levels are equal.
   267	    safe_d = jnp.where(jnp.abs(dprofile) > 1e-30, dprofile, 1e-30)
   268	    frac = jnp.clip((threshold - profile_below) / safe_d, 0.0, 1.0)
   269	
   270	    # Surface-first fractional indices of each candidate crossing.
   271	    # Pair k connects level k (below) and level k+1 (above), so the
   272	    # interpolated index is k + frac.
   273	    k_pairs = jnp.arange(nlev - 1, dtype=profile.dtype)
   274	    idx_surface_first = k_pairs + frac    # (ncol, nlev-1)
   275	
   276	    # First-crossing probability at each pair, computed as the
   277	    # sequential survival product
   278	    #
   279	    #     first_cross[k] = cross_weight[k] * Π_{j<k}(1 - cross_weight[j])
   280	    #
   281	    # This is the probability (under the soft-crossing model) that
   282	    # pair ``k`` is the *first* upward crossing — it naturally
   283	    # suppresses higher-altitude crossings even when they are
   284	    # individually strong.  Unlike a global log-sum-exp softmin,
   285	    # this formulation is robust to multiple actual crossings and to
   286	    # vanishingly-small but nonzero cross_weight at non-crossing
   287	    # levels (those contribute ``cross_weight ≈ 0`` and drop out
   288	    # multiplicatively).
   289	    not_yet_crossed = jnp.concatenate(
   290	        [
   291	            jnp.ones(profile.shape[:-1] + (1,), dtype=profile.dtype),
   292	            jnp.cumprod(1.0 - cross_weight, axis=-1)[..., :-1],
   293	        ],
   294	        axis=-1,
   295	    )
   296	    first_cross_weight = cross_weight * not_yet_crossed
   297	
   298	    # Weighted average of fractional indices.  ``maximum`` with a
   299	    # tiny floor protects against ``0/0`` when ``total_first_cross``
   300	    # is exactly zero; the no-crossing branch below replaces this
   301	    # value with the surface fallback when the gate fires.
   302	    total_first_cross = jnp.sum(first_cross_weight, axis=-1)
   303	    safe_total = jnp.maximum(total_first_cross, 1e-12)
   304	    idx_min_naive = (
   305	        jnp.sum(first_cross_weight * idx_surface_first, axis=-1)
   306	        / safe_total
   307	    )
   308	
   309	    # No-crossing fallback: blend toward the surface index (which is
   310	    # ``0`` in surface-first coordinates) when ``total_first_cross``
   311	    # is small.  Uses a sharpness independent of the user-supplied
   312	    # ``sharpness`` so the gate is consistent across calls — gate
   313	    # sharpness ``20`` around midpoint ``0.5`` gives ``≈ 0%`` trust at
   314	    # ``total = 0`` and ``≈ 100%`` trust at ``total = 1``.  At total
   315	    # values typical of a single firm crossing (~0.8–1.0) the gate
   316	    # fully selects the weighted-average answer.
   317	    no_cross_blend = jax.nn.sigmoid(20.0 * (total_first_cross - 0.5))
   318	    fallback = jnp.full_like(idx_min_naive, 0.0)  # surface-first 0 = surface-last nlev-1
   319	    idx_min_surface_first = (
   320	        no_cross_blend * idx_min_naive
   321	        + (1.0 - no_cross_blend) * fallback
   322	    )
   323	
   324	    # Convert to surface-last (canonical legoESM) coordinates:
   325	    #   surface-first index 0       -> surface-last index nlev - 1
   326	    #   surface-first index nlev-1  -> surface-last index 0
   327	    return float(nlev - 1) - idx_min_surface_first
   328	
   329	
   330	# ---------------------------------------------------------------------------
   331	# Convenience wrapper used at convection-trigger call sites
   332	# ---------------------------------------------------------------------------
   333	
   334	def cape_trigger(
   335	    cape: jax.Array,
   336	    threshold: jax.Array | float,
   337	    sharpness: float,
   338	) -> jax.Array:
   339	    """Smooth ``CAPE > threshold`` indicator for convection triggers.
   340	
   341	    Equivalent to ``smooth_step(cape - threshold, sharpness)``.  The
   342	    same soft-trigger formula is duplicated inline in four legacy
   343	    schemes (``sbm``, ``dca``, ``kuo``, ``mass_flux``) and in every
   344	    new scheme added by PRs 1 – 5; this convenience wrapper centralizes
   345	    the semantics.
   346	
   347	    Parameters
   348	    ----------
   349	    cape : jax.Array
   350	        CAPE diagnostic [J/kg], any shape.
   351	    threshold : jax.Array or float
   352	        CAPE threshold below which convection is suppressed.
   353	    sharpness : float
   354	        Sigmoid sharpness in units of [1/(J/kg)].  Smaller values
   355	        produce a wider transition; larger values approach a hard
   356	        ``CAPE > threshold`` step.
   357	
   358	    Returns
   359	    -------
   360	    jax.Array

codex
I’ve found one likely plume-regression class already: the mass flux integration was fixed to exponential form, but the T/q entrainment and q_c dilution still use explicit-Euler multipliers. I’m checking whether that is only a corner case or reachable with the current Bechtold defaults.
exec
/bin/zsh -lc "PYTHONDONTWRITEBYTECODE=1 python -c \"import jax, jax.numpy as jnp; from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow; xs=jnp.array([-1.0,0.0,1e-12,1.0]); f=lambda x: jnp.sum(safe_pow(x,0.525)); print('safe_pow', safe_pow(xs,0.525)); print('grad', jax.grad(f)(xs))\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 python -c "eps=1.75e-3; dzs=[100,333,500,1000,2000]; print([(dz, eps*dz, 1-eps*dz) for dz in dzs])"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

 exited 127 in 0ms:
zsh:1: command not found: python

exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 python -c "import jax.numpy as jnp; sharp=0.5; print([(d, float(jax_val)) for d, jax_val in [(0, jnp.asarray(1/(1+jnp.exp(-sharp*0)))), (1, jnp.asarray(1/(1+jnp.exp(-sharp*1)))), (-1, jnp.asarray(1/(1+jnp.exp(-sharp*(-1))))), (-2, jnp.asarray(1/(1+jnp.exp(-sharp*(-2)))))]])"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

exec
/bin/zsh -lc 'python3 -c "eps=1.75e-3; dzs=[100,333,500,1000,2000]; print([(dz, eps*dz, 1-eps*dz) for dz in dzs])"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'python3 -c "import math; sharp=0.5; print([(d, 1/(1+math.exp(-sharp*d))) for d in [0,1,-1,-2,-4,-6]])"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
[(0, 0.5), (1, 0.6224593312018546), (-1, 0.3775406687981454), (-2, 0.2689414213699951), (-4, 0.11920292202211755), (-6, 0.04742587317756678)]

 succeeded in 0ms:
[(100, 0.17500000000000002, 0.825), (333, 0.58275, 0.41725), (500, 0.875, 0.125), (1000, 1.75, -0.75), (2000, 3.5, -2.5)]

exec
/bin/zsh -lc "PYTHONDONTWRITEBYTECODE=1 python3 -c \"import jax, jax.numpy as jnp; from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow; xs=jnp.array([-1.0,0.0,1e-12,1.0]); f=lambda x: jnp.sum(safe_pow(x,0.525)); print('safe_pow', safe_pow(xs,0.525)); print('grad', jax.grad(f)(xs))\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 512ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import jax, jax.numpy as jnp; from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow; xs=jnp.array([-1.0,0.0,1e-12,1.0]); f=lambda x: jnp.sum(safe_pow(x,0.525)); print('safe_pow', safe_pow(xs,0.525)); print('grad', jax.grad(f)(xs))
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'legoesm'

exec
/bin/zsh -lc "PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -c \"import jax, jax.numpy as jnp; from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow; xs=jnp.array([-1.0,0.0,1e-12,1.0]); f=lambda x: jnp.sum(safe_pow(x,0.525)); print('safe_pow', safe_pow(xs,0.525)); print('grad', jax.grad(f)(xs))\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 434ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import jax, jax.numpy as jnp; from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow; xs=jnp.array([-1.0,0.0,1e-12,1.0]); f=lambda x: jnp.sum(safe_pow(x,0.525)); print('safe_pow', safe_pow(xs,0.525)); print('grad', jax.grad(f)(xs))
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/__init__.py", line 3, in <module>
    from legoesm.atmosphere.physics.radiation import (
    ...<2 lines>...
    )
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/__init__.py", line 40, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py", line 14, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py", line 26, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp_common
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp_common.py", line 17, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py", line 18, in <module>
    import dataclasses_json  # Used for JSON serialization.
    ^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'dataclasses_json'

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/integration.py | sed -n '360,540p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   360	        # Reshape to columns
   361	        T_col = T.reshape(ncol, nlev)
   362	        p_full_col = p.reshape(ncol, nlev)
   363	        rho_col = rho_total.reshape(ncol, nlev)
   364	
   365	        # Map tracers -> HydrometeorState
   366	        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
   367	        def _get_tracer(idx):
   368	            if n_tracers > idx:
   369	                return tracers[..., idx].reshape(ncol, nlev)
   370	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   371	
   372	        q_v_col = _get_tracer(0)
   373	        hydrometeors = HydrometeorState(
   374	            q_c=_get_tracer(1),
   375	            q_r=_get_tracer(2),
   376	            q_i=_get_tracer(3),
   377	            q_s=_get_tracer(4),
   378	            q_g=_get_tracer(5),
   379	            N_c=_get_tracer(6),
   380	            N_r=_get_tracer(7),
   381	            N_i=_get_tracer(8),
   382	        )
   383	
   384	        if is_ml:
   385	            if _ml_model_cache[0] is None:
   386	                key = jax.random.PRNGKey(scheme_config.seed)
   387	                _ml_model_cache[0] = MicrophysicsEmulator(
   388	                    scheme_config.n_input, scheme_config.n_hidden,
   389	                    scheme_config.n_layers, scheme_config.n_output, key=key,
   390	                )
   391	            micro_out = micro_fn(
   392	                T_col, q_v_col, hydrometeors,
   393	                p_full_col, p_half, rho_col, dz, dt,
   394	                scheme_config, _ml_model_cache[0],
   395	            )
   396	        else:
   397	            micro_out = micro_fn(
   398	                T_col, q_v_col, hydrometeors,
   399	                p_full_col, p_half, rho_col, dz, dt, scheme_config,
   400	            )
   401	
   402	        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
   403	        dT_dt = micro_out.dT_dt.reshape(shape_3d)
   404	        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)
   405	
   406	        # Map output fields -> dtracers_dt
   407	        dtracers = jnp.zeros_like(tracers)
   408	        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
   409	        tend_fields = [
   410	            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
   411	            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
   412	            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
   413	        ]
   414	        for idx, field in enumerate(tend_fields):
   415	            if n_tracers > idx:
   416	                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))
   417	
   418	        return NonHydrostaticTendencies(
   419	            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
   420	            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
   421	            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
   422	            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
   423	            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
   424	            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   425	            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
   426	        )
   427	
   428	    def reset_state():
   429	        _ml_model_cache[0] = None
   430	
   431	    physics_fn.reset_state = reset_state
   432	    return physics_fn
   433	
   434	
   435	# ===========================================================================
   436	# Spectral PE
   437	# ===========================================================================
   438	
   439	def _make_spectral_pe_microphysics(
   440	    microphysics_config: MicrophysicsConfig,
   441	    dt: float,
   442	) -> Callable:
   443	    """Create microphysics physics_fn for SpectralPEModel.
   444	
   445	    Signature: (state, grid, sigma_coord, grid_fields=None) -> SpectralHydrostaticState
   446	
   447	    The bridge pulls ``q_v`` and the full hydrometeor state out of
   448	    ``state.tracers`` (when present), runs the column microphysics
   449	    backend, and returns a ``SpectralHydrostaticState`` whose ``T_hat``
   450	    carries the spectral latent-heating tendency *and* whose ``tracers``
   451	    dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
   452	    etc.  The dycore RHS (``spectral_pe_tendencies``) adds these tracer
   453	    tendencies to its own advective tendencies during the SSP-RK stages.
   454	    """
   455	    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
   456	    is_ml = scheme_name == "ml_emulator"
   457	    _ml_model_cache = [None]
   458	
   459	    # Tracer key → MicrophysicsOutput attribute name.  Mirrors the
   460	    # ``HydrometeorState`` field layout in ``microphysics/output.py``
   461	    # plus ``q_v``.  The dycore RHS only flows tendencies for keys that
   462	    # exist on the input ``state.tracers``; missing keys are silently
   463	    # dropped (no carry to write into).
   464	    _TRACER_TEND_MAP = {
   465	        "q_v": "dq_v_dt",
   466	        "q_c": "dq_c_dt",
   467	        "q_r": "dq_r_dt",
   468	        "q_i": "dq_i_dt",
   469	        "q_s": "dq_s_dt",
   470	        "q_g": "dq_g_dt",
   471	        "N_c": "dN_c_dt",
   472	        "N_r": "dN_r_dt",
   473	        "N_i": "dN_i_dt",
   474	    }
   475	
   476	    def physics_fn(state, grid, sigma_coord, grid_fields=None):
   477	        from legoesm.atmosphere.dynamics.spectral_pe import (
   478	            SpectralHydrostaticState,
   479	            spectral_pe_to_grid,
   480	        )
   481	        from legoesm.atmosphere.physics._shared import zero_like_tracers
   482	        from legoesm.grids.gaussian import sh_analysis_3d
   483	
   484	        # Transform spectral state to grid space
   485	        fields = grid_fields
   486	        if fields is None:
   487	            fields = spectral_pe_to_grid(state, grid, sigma_coord)
   488	        T = fields['T']
   489	        p_s = fields['p_s']
   490	
   491	        nlev = sigma_coord.n_levels
   492	        n_lat, n_lon = p_s.shape
   493	
   494	        zero_3d = jnp.zeros_like(state.vor_hat.data)
   495	        zero_2d = jnp.zeros_like(state.lnps_hat.data)
   496	        # Pin the column-physics dtype to the gridded state precision so
   497	        # we do not silently flow x64 zeros into the column path.
   498	        _state_dtype = T.dtype
   499	
   500	        if micro_fn is None:
   501	            # Mirror the input tracer pytree shape with zeros so the
   502	            # orchestrator's accumulator and the dycore RHS see a
   503	            # consistent tendency structure even when microphysics is
   504	            # disabled.
   505	            return SpectralHydrostaticState(
   506	                vor_hat=state.vor_hat.replace(data=zero_3d),
   507	                div_hat=state.div_hat.replace(data=zero_3d),
   508	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
   509	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
   510	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
   511	                tracers=zero_like_tracers(state.tracers),
   512	            )
   513	
   514	        # Pressure at full and half levels
   515	        sigma_full = sigma_coord.sigma_full
   516	        sigma_half = sigma_coord.sigma_half
   517	        p_full = p_s[..., None] * sigma_full
   518	        p_half = p_s[..., None] * sigma_half
   519	
   520	        # Reshape to columns
   521	        ncol = n_lat * n_lon
   522	        T_col = T.reshape(ncol, nlev)
   523	        p_full_col = p_full.reshape(ncol, nlev)
   524	        p_half_col = p_half.reshape(ncol, nlev + 1)
   525	
   526	        # Pull tracer fields out of ``state.tracers`` and reshape to the
   527	        # column-physics ``(ncol, nlev)`` layout.  Backend microphysics
   528	        # schemes assume non-negative mixing ratios, so clip on the way
   529	        # in (matches the hydrostatic bridge's ``_get_tracer``).
   530	        def _get_tracer(name):
   531	            if state.tracers is not None and name in state.tracers:
   532	                raw = state.tracers[name]
   533	                data = raw.data if hasattr(raw, "data") else raw
   534	                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
   535	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   536	
   537	        q_v_col = _get_tracer("q_v")
   538	
   539	        rho = _compute_rho(T_col, p_full_col)
   540	        dz = _compute_heights_from_sigma(T_col, p_half_col)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/config.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for atmospheric microphysics schemes.
     2	
     3	Provides configuration NamedTuples for:
     4	1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
     5	2. Sundqvist — large-scale diagnostic condensation
     6	3. Seifert-Beheng — two-moment warm rain
     7	4. Morrison — double-moment ice+liquid
     8	5. Thompson — hybrid moment with graupel
     9	6. ML Emulator — Equinox MLP surrogate
    10	7. Top-level MicrophysicsConfig that selects the active scheme.
    11	
    12	References
    13	----------
    14	- Kessler (1969): On the Distribution and Continuity of Water Substance.
    15	- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
    16	- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
    17	- Morrison et al. (2005): A new double-moment microphysics scheme.
    18	- Thompson et al. (2008): Explicit forecasts of winter precipitation.
    19	"""
    20	
    21	from __future__ import annotations
    22	
    23	from typing import NamedTuple
    24	
    25	
    26	class KesslerConfig(NamedTuple):
    27	    """Configuration for Kessler warm-rain microphysics."""
    28	    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    29	    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    30	    accretion_coeff: float = 2.2                # Collection coefficient
    31	    evaporation_coeff: float = 1.0              # Evaporation coefficient
    32	    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    33	    saturation_sharpness: float = 100.0         # Smooth switch sharpness
    34	
    35	
    36	class SundqvistConfig(NamedTuple):
    37	    """Configuration for Sundqvist large-scale condensation."""
    38	    RH_crit: float = 0.8              # Critical relative humidity
    39	    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    40	    auto_rate: float = 1e-3           # Autoconversion rate [1/s]
    41	    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient
    42	
    43	
    44	class SeifertBehengConfig(NamedTuple):
    45	    """Configuration for Seifert-Beheng two-moment warm rain."""
    46	    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    47	    x_star: float = 2.6e-10          # Separation mass [kg]
    48	    Nc_0: float = 1e8                # Initial cloud droplet number [1/kg]
    49	    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    50	    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    51	    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    52	    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup
    53	    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    54	    b_v_r: float = 0.5               # Rain fall speed exponent b
    55	    evap_coeff: float = 1.0          # Evaporation coefficient
    56	    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation
    57	
    58	
    59	class MorrisonConfig(NamedTuple):
    60	    """Configuration for Morrison double-moment (ice+liquid)."""
    61	    # Warm rain (same as SB)
    62	    k_au: float = 6e2
    63	    x_star: float = 2.6e-10
    64	    Nc_0: float = 1e8
    65	    k_ac: float = 5.25
    66	    k_sc: float = 1e-3
    67	    D_eq: float = 1.1e-3
    68	    breakup_sharpness: float = 1e4
    69	    a_v_r: float = 130.0
    70	    b_v_r: float = 0.5
    71	    evap_coeff: float = 1.0
    72	    saturation_sharpness: float = 100.0
    73	    # Ice nucleation (Cooper 1986)
    74	    N_i0: float = 5e3               # Base ice crystal number [1/m^3]
    75	    cooper_a: float = 0.304          # Cooper exponent
    76	    cooper_T_act: float = 265.0      # Activation temperature [K]
    77	    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    78	    # Depositional growth
    79	    dep_coeff: float = 1e-3          # Deposition growth coefficient
    80	    # Bergeron
    81	    bergeron_rate: float = 1e-3      # Bergeron conversion rate [1/s]
    82	    T_center: float = 258.0          # Bergeron T window center [K]
    83	    T_width: float = 10.0            # Bergeron T window width [K]
    84	    # Riming
    85	    rime_coeff: float = 1.0          # Riming collection efficiency
    86	    # Aggregation
    87	    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    88	    # Melting
    89	    melt_rate: float = 5e-3          # Melting rate [1/s]
    90	    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    91	    # Ice sedimentation
    92	    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    93	    b_v_i: float = 0.25              # Ice fall speed exponent
    94	    # Snow sedimentation
    95	    a_v_s: float = 30.0              # Snow fall speed coefficient
    96	    b_v_s: float = 0.3               # Snow fall speed exponent
    97	
    98	
    99	class ThompsonConfig(NamedTuple):
   100	    """Configuration for Thompson hybrid-moment microphysics."""
   101	    # All Morrison params
   102	    k_au: float = 6e2
   103	    x_star: float = 2.6e-10
   104	    Nc_0: float = 1e8
   105	    k_ac: float = 5.25
   106	    k_sc: float = 1e-3
   107	    D_eq: float = 1.1e-3
   108	    breakup_sharpness: float = 1e4
   109	    a_v_r: float = 130.0
   110	    b_v_r: float = 0.5
   111	    evap_coeff: float = 1.0
   112	    saturation_sharpness: float = 100.0
   113	    N_i0: float = 5e3
   114	    cooper_a: float = 0.304
   115	    cooper_T_act: float = 265.0
   116	    ice_sigmoid_sharpness: float = 5.0
   117	    dep_coeff: float = 1e-3
   118	    bergeron_rate: float = 1e-3
   119	    T_center: float = 258.0
   120	    T_width: float = 10.0
   121	    rime_coeff: float = 1.0
   122	    agg_coeff: float = 1e-3
   123	    melt_rate: float = 5e-3
   124	    melt_sharpness: float = 2.0
   125	    a_v_i: float = 50.0
   126	    b_v_i: float = 0.25
   127	    a_v_s: float = 30.0
   128	    b_v_s: float = 0.3
   129	    # Graupel
   130	    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
   131	    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
   132	    graupel_sharpness: float = 1e4           # Sigmoid sharpness
   133	    a_v_g: float = 80.0                      # Graupel fall speed coefficient
   134	    b_v_g: float = 0.4                       # Graupel fall speed exponent
   135	    # Gamma distribution shape
   136	    mu_c: float = 3.0                        # Cloud droplet shape parameter
   137	    mu_r: float = 1.0                        # Rain drop shape parameter
   138	
   139	
   140	class MLEmulatorConfig(NamedTuple):
   141	    """Configuration for ML microphysics emulator (Equinox MLP)."""
   142	    n_input: int = 9
   143	    n_hidden: int = 128
   144	    n_layers: int = 3
   145	    n_output: int = 7
   146	    seed: int = 0
   147	    use_residual: bool = True
   148	    norm_T: float = 300.0       # Temperature scale [K] for input normalization
   149	    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
   150	    norm_rho: float = 1.2       # Air density scale [kg/m^3]
   151	    norm_dz: float = 1000.0     # Layer thickness scale [m]
   152	    norm_dt: float = 3600.0     # Time-step scale [s]
   153	
   154	
   155	class MicrophysicsConfig(NamedTuple):
   156	    """Top-level microphysics configuration.
   157	
   158	    Selects the active scheme and holds sub-configurations.
   159	
   160	    Fields
   161	    ------
   162	    scheme : str
   163	        Active scheme: "kessler", "sundqvist", "seifert_beheng",
   164	        "morrison", "thompson", "ml_emulator", or "none".
   165	    kessler : KesslerConfig
   166	    sundqvist : SundqvistConfig
   167	    seifert_beheng : SeifertBehengConfig
   168	    morrison : MorrisonConfig
   169	    thompson : ThompsonConfig
   170	    ml_emulator : MLEmulatorConfig
   171	    """
   172	    scheme: str = "none"
   173	    kessler: KesslerConfig = KesslerConfig()
   174	    sundqvist: SundqvistConfig = SundqvistConfig()
   175	    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
   176	    morrison: MorrisonConfig = MorrisonConfig()
   177	    thompson: ThompsonConfig = ThompsonConfig()
   178	    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/kessler.py | sed -n '70,120p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    70	    q_c = hydrometeors.q_c
    71	    q_r = hydrometeors.q_r
    72	    sharpness = config.saturation_sharpness
    73	
    74	    # Saturation mixing ratio
    75	    q_sat = saturation_mixing_ratio(T, p_full)
    76	
    77	    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s]
    78	    excess = q_v - q_sat
    79	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    80	    condensation = cond_frac * excess / dt  # [kg/kg/s]
    81	
    82	    dq_v_sat = -condensation
    83	    dq_c_sat = condensation
    84	
    85	    # 2. Autoconversion: cloud -> rain (threshold excess)
    86	    # dq_c_sat is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    87	    q_c_updated = q_c + dq_c_sat * dt
    88	    autoconv = config.autoconversion_rate * jnp.maximum(
    89	        q_c_updated - config.autoconversion_threshold, 0.0
    90	    )
    91	
    92	    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
    93	    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
    94	    accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)
    95	
    96	    # 4. Evaporation of rain (q_r^0.525).
    97	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    98	    evaporation = config.evaporation_coeff * subsaturation * safe_pow(q_r, 0.525)
    99	
   100	    # 5. Rain sedimentation
   101	    rho_sfc = rho[:, -1:]
   102	    V_t = config.rain_fall_speed * jnp.sqrt(
   103	        rho_sfc / jnp.clip(rho, 0.1)
   104	    )
   105	    sed_tend = sedimentation_tendency(q_r, rho, V_t, dz)
   106	
   107	    # 6. Latent heating
   108	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
   109	
   110	    # Combine tracer tendencies
   111	    dq_v_dt = dq_v_sat + evaporation
   112	    dq_c_dt = dq_c_sat - autoconv - accretion
   113	    dq_r_dt = autoconv + accretion - evaporation + sed_tend
   114	
   115	    # Precipitation: surface flux
   116	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   117	    V_t_bot = V_t[:, -1]
   118	    precipitation = q_r_bot * rho[:, -1] * V_t_bot
   119	
   120	    # Pin dtype to the input precision so we never silently promote

exec
/bin/zsh -lc 'rg -n "entrain|Euler|q_u_ent|subcloud|cloud base|safe_pow|saturation|PBL|downdraft|number|dN_r" .physics-validator/atmosphere-physics/REPORT.md' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
33:4. **CRITICAL — convection mass_flux/edmf q_c_u from saturation excess** `convection/mass_flux.py:319-331,437-451`
34:   Plume cloud water built as `dilution * (q_sat_base - q_sat_moist)` independent of actual q_v ⇒ produces cloud water in dry columns. Shared `_plume.entraining_detraining_plume` does this correctly; the bulk paths use the non-conservative shortcut.
51:10. **MAJOR — Tiedtke saturation-deficit MC proxy has wrong sign** `tiedtke.py:185-193`
56:- **Convection** (`convection/review-1.md`): codex executed JAX evaluations to verify LNB / CIN / crossing-index behavior. Confirmed all 4 critical findings; identified additional bugs (CIN window inversion, plume q_c_u accumulation without entrainment dilution).
60:Codex / static analysis agreement: ~80%. Codex correctly upgraded several "style" findings to numerical bugs (e.g., Bechtold downdraft `0.05` cooling — not just hardcoded but dimensionally wrong AND non-water-conserving).
87:| P0-7 | Hines WKB amplitude compounded (cumulative ρ-ratio per step instead of inter-level); also smooth limiter produced negative drag | `gravity_wave_drag/hines.py` | `tests/unit/test_physics_gwd.py::test_hines_no_saturation_zero_drag` | inter-level ratio in scan; drag clipped to [0, Fmax] |
100:| P1-2 | `delta_0_eff/delta_deep` rescale also rescaled subsidence in Tiedtke / Bechtold / Emanuel (subsidence is ``M/ρ × dT/dz`` — independent of delta_0) | `convection/{tiedtke,bechtold,emanuel}.py` | covered by existing scheme test suites + `tests/unit/test_emanuel.py::test_emanuel_downdraft_toggle_changes_subcloud_dT` | pass per-column / per-level `delta_0_eff` directly to the kernel; only the detrainment terms in the kernel use it |
108:| Tiedtke MC proxy | Saturation-deficit formula was *inverted*: large in dry columns (suppressed convection where it should fire), small in moist columns. | `convection/tiedtke.py`, `convection/config.py` (new `mc_proxy_RH_crit=0.6` field) | `tests/unit/test_tiedtke.py::test_tiedtke_mc_proxy_is_larger_in_moist_columns` | Replaced `max(q_sat - q_v, 0)` with saturation-EXCESS `max(q_v - RH_crit * q_sat, 0)` — now positive in moist columns, vanishing in dry. |
109:| McFarlane stress dimensional | Both `tau_0 = G_0 * U * h^2 * N * ρ` (units `kg²/(m²·s⁴)`) AND `tau_sat = eff * ρ * U³ / (N * envelope)` (units `kg/s²`) were missing the `k_wave` factor; neither was a proper stress.  Operational drag was ~1e-21 m/s².  (Codex stop-time review caught the `tau_sat` half after the initial `tau_0` fix.) | `gravity_wave_drag/mcfarlane.py`, `gravity_wave_drag/config.py` (G_0 docstring) | `tests/unit/test_physics_gwd.py::test_gwd_grad_through_*[mcfarlane]` (xfail removed; tests now active), `test_mcfarlane_stress_units_match_lindzen` | Inserted missing `k_wave` factor in BOTH stresses: `tau_0 = G_0 * ρ * N * k * h² * U` and `tau_sat = eff * ρ * U³ * k / (N * envelope)` (both proper Pa, matching Lindzen).  Updated docstring to mark `G_0` as dimensionless prefactor.  Test column updated to provide a critical level so the saturation path actually exercises (orographic schemes need shear or critical level). |
112:| P2 specific-humidity vs mixing-ratio doc drift | Codebase variable `q_v` is documented inconsistently: `saturation_mixing_ratio` returns mixing ratio (`r = ε e/(p-e)`) while many docstrings call `q_v` "specific humidity" (`q ≈ r / (1+r)`, ~1% drift). | `src/legoesm/thermo.py` (module docstring expanded with conventions section) | n/a (documentation only) | Added a "Conventions — water-vapor mass variables" section to the `thermo.py` module docstring noting that the codebase uses `r ≈ q` interchangeably and pointing physics that needs the distinction (e.g., saturated tropical columns, q_c bookkeeping) to convert explicitly. |
117:- **Units:** ❌ multiple unit errors (CAPE/τ→M_b in 5 convection schemes; Hines drag; McFarlane stress; Tiedtke/Bechtold downdraft cooling).
119:- **Differentiability:** ⚠ design is AD-friendly but `q ** 0.525` NaN at q=0; explicit Euler in mass_flux/edmf prognostics.

exec
/bin/zsh -lc 'rg -n "q_c_u|entrain|plume|M_u_raw|exp|negative|subcloud|base" tests/unit/test_convection_plume.py tests/unit/test_tiedtke.py tests/unit/test_bechtold.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/unit/test_bechtold.py:13:  expected stationary variance ``1`` for a sufficiently long run;
tests/unit/test_bechtold.py:65:    q = q_sfc * jnp.exp(-z_full / 3000.0)
tests/unit/test_bechtold.py:98:    changes the diagnosed cloud-base mass flux."""
tests/unit/test_bechtold.py:347:        "PhysicsState should expose conv_stoch_state field"
tests/unit/test_bechtold.py:578:# MSE conservation regression guard (currently expected to fail)
tests/unit/test_tiedtke.py:61:    q = q_sfc * jnp.exp(-z_full / 3000.0)
tests/unit/test_tiedtke.py:209:def test_tiedtke_downdraft_toggle_changes_subcloud_T():
tests/unit/test_tiedtke.py:246:         the subcloud layer must equal cloud-water source removed
tests/unit/test_convection_plume.py:1:"""Unit tests for ``legoesm.atmosphere.physics.convection._plume``.
tests/unit/test_convection_plume.py:8:* :func:`entraining_detraining_plume`     — vmappable updraft integrator.
tests/unit/test_convection_plume.py:16:* analytical limits of the plume integrator (no entrainment ⇒ moist
tests/unit/test_convection_plume.py:17:  adiabat; strong entrainment ⇒ environmental relaxation; sub-cloud
tests/unit/test_convection_plume.py:36:from legoesm.atmosphere.physics.convection import _plume as P
tests/unit/test_convection_plume.py:54:    """Surface-last column with a stable troposphere and an exponential
tests/unit/test_convection_plume.py:81:    q_v_env = q_sfc * jnp.exp(-z_full / q_scale_height_m)
tests/unit/test_convection_plume.py:173:    """LNB must localize where buoyancy transitions positive→negative
tests/unit/test_convection_plume.py:183:    LNB ≈ 1.5 (where buoyancy goes positive → negative).
tests/unit/test_convection_plume.py:197:        f"k_lfc = {float(k_lfc[0])}; expected ~6.5"
tests/unit/test_convection_plume.py:201:        f"k_lnb = {float(k_lnb[0])}; expected < 4.0 (well above LFC). "
tests/unit/test_convection_plume.py:209:    T_ma_base = compute_moist_adiabat(T_env[:, -1], p_full)
tests/unit/test_convection_plume.py:225:def test_compute_cin_non_negative_and_finite():
tests/unit/test_convection_plume.py:226:    """CIN is non-negative by construction (positive-part of negative
tests/unit/test_convection_plume.py:238:def test_compute_cin_integrates_negative_buoyancy_between_lcl_and_lfc():
tests/unit/test_convection_plume.py:239:    """CIN must integrate the negative buoyancy in the LCL→LFC layer.
tests/unit/test_convection_plume.py:292:# entraining_detraining_plume
tests/unit/test_convection_plume.py:295:def test_plume_outputs_finite_and_correct_shape():
tests/unit/test_convection_plume.py:296:    """Smoke: every output array is finite with the expected
tests/unit/test_convection_plume.py:300:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:301:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:302:    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
tests/unit/test_convection_plume.py:307:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:309:        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
tests/unit/test_convection_plume.py:311:    for arr in (plume.M_u, plume.T_u, plume.q_u, plume.q_c_u, plume.B_u):
tests/unit/test_convection_plume.py:316:def test_plume_no_entrainment_limit_matches_moist_adiabat():
tests/unit/test_convection_plume.py:318:    the in-cloud plume temperature follows the moist adiabat (within
tests/unit/test_convection_plume.py:324:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:325:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:327:    # Cloud base at surface.  With unsaturated air the plume condenses
tests/unit/test_convection_plume.py:329:    # T_base.
tests/unit/test_convection_plume.py:330:    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
tests/unit/test_convection_plume.py:331:    T_ma = compute_moist_adiabat(T_base, p_full)
tests/unit/test_convection_plume.py:337:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:339:        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
tests/unit/test_convection_plume.py:342:    # adiabat.  The first-order Euler ascent in the plume integrator
tests/unit/test_convection_plume.py:345:    plume_T = plume.T_u[0, sampled]
tests/unit/test_convection_plume.py:347:    diff = jnp.abs(plume_T - ma_T)
tests/unit/test_convection_plume.py:349:        f"No-entrainment plume should track moist adiabat within ~5 K, "
tests/unit/test_convection_plume.py:354:def test_plume_sub_cloud_mass_flux_is_suppressed():
tests/unit/test_convection_plume.py:355:    """Levels strictly below the cloud base have ``M_u ≈ 0`` because
tests/unit/test_convection_plume.py:356:    the ``above_base_weight`` mask suppresses them."""
tests/unit/test_convection_plume.py:359:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:360:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:361:    # Force a cloud-base index well above the surface by passing a
tests/unit/test_convection_plume.py:362:    # synthetic ``k_base_smooth`` (mid-column) so we have several
tests/unit/test_convection_plume.py:364:    k_base_synthetic = jnp.full((ncol,), float(nlev) - 4.0)
tests/unit/test_convection_plume.py:370:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:372:        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
tests/unit/test_convection_plume.py:375:    # Surface-last: levels with index > k_base are below the cloud
tests/unit/test_convection_plume.py:376:    # base (lower altitude).
tests/unit/test_convection_plume.py:377:    sub_cloud_mass = plume.M_u[:, -1]   # surface-most level
tests/unit/test_convection_plume.py:378:    base_mass = plume.M_u[:, int(float(nlev) - 4.0)]
tests/unit/test_convection_plume.py:379:    # Sub-cloud value should be much smaller than the cloud-base
tests/unit/test_convection_plume.py:381:    assert float(sub_cloud_mass[0]) < 0.5 * float(base_mass[0])
tests/unit/test_convection_plume.py:384:def test_plume_q_c_u_diluted_by_entrainment():
tests/unit/test_convection_plume.py:385:    """Audit Codex review (severity major): plume cloud water was
tests/unit/test_convection_plume.py:386:    accumulated but not diluted by environmental entrainment.  The
tests/unit/test_convection_plume.py:387:    proper continuity for an intensive quantity in an entraining-
tests/unit/test_convection_plume.py:388:    detraining plume is ``dq_c/dz = -eps · q_c + cond/M``: env air
tests/unit/test_convection_plume.py:389:    carries q_c=0, so entrainment uniformly decreases ``q_c_u``.
tests/unit/test_convection_plume.py:395:    entrainment, the carried-aloft q_c_u must DECREASE between the
tests/unit/test_convection_plume.py:396:    last condensation level and the column top (entrainment
tests/unit/test_convection_plume.py:397:    dilution dominates).  In the buggy form q_c_u was monotone
tests/unit/test_convection_plume.py:404:    # exercising the clip path, not the stable explicit-Euler dilution
tests/unit/test_convection_plume.py:408:    # so ``q_c_u`` decay is governed by the physics, not the clip.
tests/unit/test_convection_plume.py:413:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:414:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:415:    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
tests/unit/test_convection_plume.py:429:        "the test would not exercise stable explicit-Euler dilution."
tests/unit/test_convection_plume.py:432:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:434:        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
tests/unit/test_convection_plume.py:436:    q_c_u = plume.q_c_u[0]                       # surface-last (ncol, nlev)
tests/unit/test_convection_plume.py:438:    # The plume's q_c_u peaks somewhere above LCL where condensation
tests/unit/test_convection_plume.py:440:    # forcing on q_c_u is the entrainment dilution ``-eps·q_c_u``,
tests/unit/test_convection_plume.py:442:    q_c_max = float(jnp.max(q_c_u))
tests/unit/test_convection_plume.py:443:    q_c_top = float(q_c_u[0])                    # column top (surface-last)
tests/unit/test_convection_plume.py:446:        f"Test fixture broken — plume produced no cloud water "
tests/unit/test_convection_plume.py:447:        f"(max q_c_u = {q_c_max:.3e}); cannot test dilution."
tests/unit/test_convection_plume.py:459:        f"Plume q_c_u at column top ({q_c_top:.3e}) is not significantly "
tests/unit/test_convection_plume.py:461:        "expected < 0.85 from stable explicit-Euler dilution.  Audit "
tests/unit/test_convection_plume.py:462:        "Codex finding 'plume cloud water is accumulated but not "
tests/unit/test_convection_plume.py:463:        "diluted by entrainment' has regressed."
tests/unit/test_convection_plume.py:467:def test_plume_M_u_grad_through_strong_detrainment_is_finite_and_nonzero():
tests/unit/test_convection_plume.py:468:    """Audit Codex review (severity major): the explicit-Euler plume
tests/unit/test_convection_plume.py:469:    mass flux ``M * (1 + (eps - dlt) * dz)`` can produce a *negative*
tests/unit/test_convection_plume.py:474:    — making AD-based sensitivity studies silently mis-state the
tests/unit/test_convection_plume.py:478:    The exact integration ``M * exp((eps - dlt) * dz)`` is always
tests/unit/test_convection_plume.py:480:    very negative ``(eps - dlt) * dz``.
tests/unit/test_convection_plume.py:496:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:497:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:498:    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
tests/unit/test_convection_plume.py:504:        plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:506:            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
tests/unit/test_convection_plume.py:508:        return jnp.sum(plume.M_u)
tests/unit/test_convection_plume.py:523:        "discrimination threshold 0.5.  The buggy explicit-Euler form "
tests/unit/test_convection_plume.py:526:        "clip).  The fixed exponential form gives |grad| ≈ 4.8 — "
tests/unit/test_convection_plume.py:528:        "indicates the exp integration has been reverted to the buggy "
tests/unit/test_convection_plume.py:529:        "explicit-Euler form (audit Codex finding 'plume mass flux "
tests/unit/test_convection_plume.py:530:        "uses explicit Euler plus a hard nonnegative clip')."
tests/unit/test_convection_plume.py:534:def test_plume_grad_through_epsilon():
tests/unit/test_convection_plume.py:535:    """``d (sum M_u) / d epsilon_0`` is finite — the plume integrator
tests/unit/test_convection_plume.py:539:    T_base = T_env[:, -1]
tests/unit/test_convection_plume.py:540:    q_base = q_v_env[:, -1]
tests/unit/test_convection_plume.py:541:    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
tests/unit/test_convection_plume.py:547:        plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:549:            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
tests/unit/test_convection_plume.py:551:        return jnp.sum(plume.M_u)

exec
/bin/zsh -lc "nl -ba tests/unit/test_convection_plume.py | sed -n '354,460p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   354	def test_plume_sub_cloud_mass_flux_is_suppressed():
   355	    """Levels strictly below the cloud base have ``M_u ≈ 0`` because
   356	    the ``above_base_weight`` mask suppresses them."""
   357	    ncol, nlev = 1, 16
   358	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   359	    T_base = T_env[:, -1]
   360	    q_base = q_v_env[:, -1]
   361	    # Force a cloud-base index well above the surface by passing a
   362	    # synthetic ``k_base_smooth`` (mid-column) so we have several
   363	    # sub-cloud levels to inspect.
   364	    k_base_synthetic = jnp.full((ncol,), float(nlev) - 4.0)
   365	
   366	    eps = jnp.full((ncol, nlev), 5.0e-4)
   367	    dlt = jnp.full((ncol, nlev), 5.0e-4)
   368	    M_b = jnp.full((ncol,), 0.05)
   369	
   370	    plume = P.entraining_detraining_plume(
   371	        T_env, q_v_env, p_full, p_half, z_full,
   372	        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
   373	        buoyancy_sharpness=2.0,
   374	    )
   375	    # Surface-last: levels with index > k_base are below the cloud
   376	    # base (lower altitude).
   377	    sub_cloud_mass = plume.M_u[:, -1]   # surface-most level
   378	    base_mass = plume.M_u[:, int(float(nlev) - 4.0)]
   379	    # Sub-cloud value should be much smaller than the cloud-base
   380	    # value.  Tolerance reflects the smoothness of the sigmoid mask.
   381	    assert float(sub_cloud_mass[0]) < 0.5 * float(base_mass[0])
   382	
   383	
   384	def test_plume_q_c_u_diluted_by_entrainment():
   385	    """Audit Codex review (severity major): plume cloud water was
   386	    accumulated but not diluted by environmental entrainment.  The
   387	    proper continuity for an intensive quantity in an entraining-
   388	    detraining plume is ``dq_c/dz = -eps · q_c + cond/M``: env air
   389	    carries q_c=0, so entrainment uniformly decreases ``q_c_u``.
   390	
   391	    Test setup: a parcel that condenses at the LCL and ascends to a
   392	    level where the moist-adiabatic temperature exceeds the
   393	    saturation profile only slightly — so condensation occurs in a
   394	    narrow lower layer and is essentially zero aloft.  With strong
   395	    entrainment, the carried-aloft q_c_u must DECREASE between the
   396	    last condensation level and the column top (entrainment
   397	    dilution dominates).  In the buggy form q_c_u was monotone
   398	    non-decreasing aloft (carried unchanged when condensation = 0).
   399	    """
   400	    # High-resolution column + small ``eps`` so that ``eps·dz <= 0.25``
   401	    # at *every* layer.  Prior version used ``eps = 1e-3`` which gave
   402	    # ``eps·dz > 1`` in the top 5 levels — the dilution clip
   403	    # ``max(0, 1 - eps·dz)`` triggered there and the test was
   404	    # exercising the clip path, not the stable explicit-Euler dilution
   405	    # (Codex stop-time review: "regression test exercises the clip
   406	    # path, not stable dilution").  ``eps = 1e-4 /m`` keeps the
   407	    # dilution multiplier strictly in (0.75, 1) for the entire column
   408	    # so ``q_c_u`` decay is governed by the physics, not the clip.
   409	    ncol, nlev = 1, 60
   410	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
   411	        ncol, nlev, T_sfc=295.0, q_sfc=8.0e-3, lapse_rate_K_per_km=7.0,
   412	    )
   413	    T_base = T_env[:, -1]
   414	    q_base = q_v_env[:, -1]
   415	    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
   416	
   417	    eps_scalar = 1.0e-4
   418	    eps = jnp.full((ncol, nlev), eps_scalar)
   419	    dlt = jnp.full((ncol, nlev), 1.0e-5)        # weaker detrainment
   420	    M_b = jnp.full((ncol,), 0.05)
   421	
   422	    # Sanity: assert eps·dz < 1 at every layer so the test really
   423	    # exercises stable dilution rather than the corner-case clip.
   424	    dz_layers = z_full[0, :-1] - z_full[0, 1:]
   425	    max_eps_dz = float(jnp.max(eps_scalar * dz_layers))
   426	    assert max_eps_dz < 1.0, (
   427	        f"Test fixture broken — max eps·dz = {max_eps_dz:.3f} ≥ 1; "
   428	        "the dilution clip ``max(0, 1-eps·dz)`` would dominate and "
   429	        "the test would not exercise stable explicit-Euler dilution."
   430	    )
   431	
   432	    plume = P.entraining_detraining_plume(
   433	        T_env, q_v_env, p_full, p_half, z_full,
   434	        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
   435	    )
   436	    q_c_u = plume.q_c_u[0]                       # surface-last (ncol, nlev)
   437	
   438	    # The plume's q_c_u peaks somewhere above LCL where condensation
   439	    # is active; aloft, where condensation tapers off, the only
   440	    # forcing on q_c_u is the entrainment dilution ``-eps·q_c_u``,
   441	    # so it must decay.  Compare the column top to the peak.
   442	    q_c_max = float(jnp.max(q_c_u))
   443	    q_c_top = float(q_c_u[0])                    # column top (surface-last)
   444	
   445	    assert q_c_max > 1e-8, (
   446	        f"Test fixture broken — plume produced no cloud water "
   447	        f"(max q_c_u = {q_c_max:.3e}); cannot test dilution."
   448	    )
   449	    # Quantitative bound: with ``eps·dz ≈ 0.02`` per layer (typical
   450	    # mid-tropospheric value here) the multiplicative dilution
   451	    # ``(1 - eps·dz)`` over ~30 layers between peak and column top
   452	    # gives a ratio of order ``0.98^30 ≈ 0.55``.  We use a slightly
   453	    # looser threshold to leave headroom for the layer-thickness
   454	    # variation (top layers are thicker, dilution is faster).  The
   455	    # buggy form (no dilution) gives ratio ≈ 1.0 — orders of
   456	    # magnitude away from the threshold.
   457	    ratio = q_c_top / q_c_max
   458	    assert ratio < 0.85, (
   459	        f"Plume q_c_u at column top ({q_c_top:.3e}) is not significantly "
   460	        f"smaller than peak ({q_c_max:.3e}); ratio = {ratio:.3f}, "

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_plume.py | sed -n '614,780p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   614	def cmt_gregory_1997(
   615	    u_env: jax.Array,
   616	    v_env: jax.Array,
   617	    M_u: jax.Array,
   618	    M_d: jax.Array | None,
   619	    p_full: jax.Array,
   620	    p_half: jax.Array,
   621	    rho: jax.Array,
   622	    *,
   623	    c_u: float = 0.55,
   624	    c_d: float = 0.55,
   625	) -> tuple[jax.Array, jax.Array]:
   626	    """Gregory et al. 1997 convective momentum transport closure.
   627	
   628	    The eddy-flux closure carries plume momentum aloft minus a
   629	    pressure-gradient correction that limits the upward transport in
   630	    sheared environments.  In bulk-plume form ::
   631	
   632	        F_u = M_u * (u_u - u_env) - c_u * M_u * (du/dz)_layer * dz_layer
   633	
   634	    and the resulting tendency is ``du/dt = -(1/rho) * dF/dz``.  The
   635	    same form applies to the downdraft with sign convention
   636	    ``M_d < 0`` and parameter ``c_d``.
   637	
   638	    The implementation here is the **simplified bulk closure**: we
   639	    approximate ``u_u`` by the environmental wind at the cloud base
   640	    plus a fraction of the layer-by-layer environmental shear,
   641	    yielding a numerically stable form that does not require a
   642	    separate plume-momentum integrator.  This is the formulation
   643	    used in CESM/CAM with the ZM scheme and follows Gregory et al.
   644	    1997 Eq. 14.
   645	
   646	    Parameters
   647	    ----------
   648	    u_env, v_env : jax.Array, shape (ncol, nlev)
   649	        Environmental zonal / meridional wind [m/s].
   650	    M_u : jax.Array, shape (ncol, nlev)
   651	        Updraft mass flux [kg/m²/s].  Non-negative.
   652	    M_d : jax.Array or None, shape (ncol, nlev)
   653	        Downdraft mass flux [kg/m²/s] (negative by convention).
   654	        ``None`` skips the downdraft contribution.
   655	    p_full, p_half : jax.Array
   656	        Pressures [Pa] at full / half levels.  Used to compute layer
   657	        thickness ``dp = p_half[1:] - p_half[:-1]``.
   658	    rho : jax.Array, shape (ncol, nlev)
   659	        Air density [kg/m³].
   660	    c_u, c_d : float
   661	        Pressure-gradient correction coefficients in Gregory et al.
   662	        1997 Eq. 12.  ``0.55`` is the canonical value (range
   663	        ``0.3–0.7`` in the literature).
   664	
   665	    Returns
   666	    -------
   667	    du_dt, dv_dt : jax.Array, shape (ncol, nlev)
   668	        Convective momentum tendencies [m/s²].
   669	    """
   670	    nlev = u_env.shape[-1]
   671	
   672	    # Layer pressure thickness; with surface-last convention dp > 0.
   673	    dp = p_half[:, 1:] - p_half[:, :-1]
   674	
   675	    # Stratospheric mass-flux gate — same factor the kernel applies to
   676	    # T/q_v tendencies.  Without this, the CMT path detrains
   677	    # convective momentum into the model top (where the plume should
   678	    # already be dead), producing wind-driven dycore blowups (e.g.
   679	    # KF at day 10 in 1-year lat-lon FV RCE).  Imported lazily to
   680	    # avoid a circular import (`mass_flux` imports from `_plume`).
   681	    from legoesm.atmosphere.physics.convection.mass_flux import (
   682	        stratosphere_mass_flux_gate,
   683	    )
   684	    p_gate = stratosphere_mass_flux_gate(p_full)
   685	    M_u = M_u * p_gate
   686	    if M_d is not None:
   687	        M_d = M_d * p_gate
   688	
   689	    # Environmental shear (forward difference per layer).  Edge layers
   690	    # use one-sided differences via padded edges to keep shape
   691	    # ``(ncol, nlev)``.
   692	    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
   693	    dv_layer = jnp.diff(v_env, axis=-1, prepend=v_env[:, :1])
   694	
   695	    # Eddy momentum flux from Gregory et al. 1997 §3.  In the bulk
   696	    # plume approximation:
   697	    #     M_u (u_u - u_env) ≈ -c_u * M_u * du/dz_layer
   698	    # i.e. only the pressure-gradient correction term contributes —
   699	    # the leading ``M_u * u_env`` mass-transport term cancels when
   700	    # the mass-flux divergence is also accounted for in the
   701	    # large-scale momentum equation.  As a consequence a uniform
   702	    # wind column produces zero CMT regardless of mass flux.
   703	    flux_u_up = -c_u * M_u * du_layer
   704	    flux_v_up = -c_u * M_u * dv_layer
   705	
   706	    if M_d is not None:
   707	        flux_u_down = -c_d * M_d * du_layer
   708	        flux_v_down = -c_d * M_d * dv_layer
   709	    else:
   710	        flux_u_down = jnp.zeros_like(flux_u_up)
   711	        flux_v_down = jnp.zeros_like(flux_v_up)
   712	
   713	    flux_u = flux_u_up + flux_u_down
   714	    flux_v = flux_v_up + flux_v_down
   715	
   716	    # Vertical divergence of the flux: ``du/dt = -(1/rho) * dF/dz``.
   717	    # Using hydrostatic ``dz = -dp/(rho*g)`` gives
   718	    # ``du/dt = -(g) * dF/dp`` after the rho cancels.
   719	    dflux_u = jnp.diff(flux_u, axis=-1, append=flux_u[:, -1:])
   720	    dflux_v = jnp.diff(flux_v, axis=-1, append=flux_v[:, -1:])
   721	
   722	    du_dt = -constants.g * dflux_u / dp
   723	    dv_dt = -constants.g * dflux_v / dp
   724	
   725	    return du_dt, dv_dt

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/mass_flux.py | sed -n '55,110p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    55	    compute_cape,
    56	)
    57	from legoesm.atmosphere.physics.convection.config import (
    58	    EDMFConfig,
    59	    MassFluxConfig,
    60	)
    61	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    62	
    63	
    64	# =============================================================================
    65	# Shared helpers — used by both the Arakawa-Wu and simplified EDMF paths.
    66	# =============================================================================
    67	
    68	
    69	def _compute_column_geometry(
    70	    T: jax.Array,
    71	    p_full: jax.Array,
    72	    p_half: jax.Array,
    73	) -> Tuple[jax.Array, jax.Array, jax.Array]:
    74	    """Compute layer thickness, density, and surface-relative height.
    75	
    76	    Returns ``(dz, rho, z)``, all of shape ``(ncol, nlev)``. ``dz`` is
    77	    the layer thickness from hydrostatic balance using the mid-layer
    78	    pressure; ``rho`` is the dry-air density at full levels; ``z`` is
    79	    the cumulative height above the surface (note: levels are ordered
    80	    top-down, so ``z[:, -1]`` is the surface).
    81	    """
    82	    dp = p_half[:, 1:] - p_half[:, :-1]
    83	    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    84	    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    85	    dz = jnp.abs(dz)
    86	    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
    87	    # Cumulative height from the surface (level nlev-1) upward.
    88	    z = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    89	    return dz, rho, z
    90	
    91	
    92	def _compute_cape_diagnostics(
    93	    T: jax.Array,
    94	    p_full: jax.Array,
    95	    p_half: jax.Array,
    96	    cape_threshold: float,
    97	    cape_activation_scale: float,
    98	) -> Tuple[jax.Array, jax.Array, jax.Array]:
    99	    """Compute moist-adiabat profile, CAPE, and a smooth convective mask.
   100	
   101	    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` is shape
   102	    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are shape
   103	    ``(ncol,)``. The mask is a sigmoid of ``(cape -
   104	    cape_threshold) / cape_activation_scale`` and is reused as the
   105	    smooth activation factor for both schemes.
   106	    """
   107	    T_base = T[:, -1]
   108	    T_moist = compute_moist_adiabat(T_base, p_full)
   109	    cape = compute_cape(T, T_moist, p_full, p_half)
   110	    convective_mask = jax.nn.sigmoid(

exec
/bin/zsh -lc "rg -n \"def moist_adiabat_lapse_rate|def compute_moist_adiabat|def saturation_mixing_ratio\" src/legoesm/atmosphere/physics/thermodynamics.py src/legoesm/thermo.py && nl -ba src/legoesm/atmosphere/physics/thermodynamics.py | sed -n '1,170p' && nl -ba src/legoesm/thermo.py | sed -n '1,160p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/thermo.py:52:def saturation_mixing_ratio(
src/legoesm/thermo.py:85:def saturation_mixing_ratio_ice(
src/legoesm/atmosphere/physics/thermodynamics.py:160:def moist_adiabat_lapse_rate(
src/legoesm/atmosphere/physics/thermodynamics.py:195:def compute_moist_adiabat(
     1	"""Shared thermodynamic utilities for atmospheric physics.
     2	
     3	Provides fundamental thermodynamic functions used across physics
     4	parameterizations (Kessler microphysics, convection, etc.):
     5	
     6	1. Saturation mixing ratio (Tetens formula)
     7	2. Temperature-potential temperature conversions
     8	3. Pressure from equation of state
     9	4. Moist adiabatic lapse rate and profiles
    10	5. CAPE computation
    11	
    12	All operations are pure JAX and compatible with jit, grad, vmap, scan.
    13	"""
    14	
    15	from __future__ import annotations
    16	
    17	import jax
    18	import jax.numpy as jnp
    19	
    20	from legoesm import constants
    21	
    22	# Numerical guardrails used across atmosphere dynamics/physics bridges.
    23	_THETA_MIN = 50.0       # [K]
    24	_RHO_MIN = 1.0e-9       # [kg m^-3]
    25	_P_MIN = 1.0            # [Pa]
    26	_P_MAX = 2.0e7          # [Pa]
    27	
    28	
    29	# ==============================================================================
    30	# Basic thermodynamic relations (extracted from kessler.py)
    31	# ==============================================================================
    32	
    33	# Canonical implementations now live in legoesm.thermo so that non-atmosphere
    34	# packages (land, ice, ocean, coupler) can import them without pulling in the
    35	# full atmosphere.physics package.  Re-exported here for backward compatibility.
    36	from legoesm.thermo import saturation_mixing_ratio as saturation_mixing_ratio  # noqa: F401
    37	from legoesm.thermo import saturation_mixing_ratio_ice as saturation_mixing_ratio_ice  # noqa: F401
    38	
    39	
    40	def temperature_from_theta(
    41	    theta: jax.Array,
    42	    p: jax.Array,
    43	) -> jax.Array:
    44	    """Recover temperature from potential temperature and pressure.
    45	
    46	    T = theta * (p / p_0)^kappa
    47	
    48	    Parameters
    49	    ----------
    50	    theta : jax.Array
    51	        Potential temperature [K].
    52	    p : jax.Array
    53	        Pressure [Pa].
    54	
    55	    Returns
    56	    -------
    57	    jax.Array
    58	        Temperature [K].
    59	    """
    60	    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    61	    p_pos = jnp.clip(p, _P_MIN, _P_MAX)
    62	    return theta_pos * (p_pos / constants.p_ref) ** constants.kappa
    63	
    64	
    65	def sanitize_theta_rho(
    66	    theta: jax.Array,
    67	    rho: jax.Array,
    68	) -> tuple[jax.Array, jax.Array]:
    69	    """Clip thermodynamic state to physically positive ranges.
    70	
    71	    Returns
    72	    -------
    73	    theta_pos, rho_pos : jax.Array
    74	        Potential temperature [K] and density [kg/m^3] clipped to
    75	        positive finite floors for robust EOS/exner evaluations.
    76	    """
    77	    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    78	    rho_pos = jnp.clip(rho, _RHO_MIN, None)
    79	    return theta_pos, rho_pos
    80	
    81	
    82	def pressure_from_eos(
    83	    rho: jax.Array,
    84	    theta: jax.Array,
    85	) -> jax.Array:
    86	    """Compute pressure from density and potential temperature.
    87	
    88	    p = p_0 * (R_d * rho * theta / p_0)^(c_p / c_v)
    89	
    90	    Parameters
    91	    ----------
    92	    rho : jax.Array
    93	        Density [kg/m^3].
    94	    theta : jax.Array
    95	        Potential temperature [K].
    96	
    97	    Returns
    98	    -------
    99	    jax.Array
   100	        Pressure [Pa].
   101	    """
   102	    R_d = constants.R_d
   103	    c_p = constants.c_pd
   104	    c_v = constants.c_vd
   105	    p_0 = constants.p_ref
   106	
   107	    theta_pos, rho_pos = sanitize_theta_rho(theta, rho)
   108	    base = jnp.clip(R_d * rho_pos * theta_pos / p_0, 1.0e-20, 1.0e20)
   109	    p = p_0 * base ** (c_p / c_v)
   110	    return jnp.clip(p, _P_MIN, _P_MAX)
   111	
   112	
   113	def reconstruct_half_level_pressure_hydrostatic(
   114	    p_full: jax.Array,
   115	    rho_full: jax.Array,
   116	    z_half: jax.Array,
   117	) -> jax.Array:
   118	    """Reconstruct interface pressure from full-level state via hydrostatic balance.
   119	
   120	    This is intended for non-hydrostatic column physics bridges where
   121	    full-level pressure comes from the local EOS but interface pressure is
   122	    needed by parameterizations. Using the evolving column state avoids
   123	    relying on a fixed reference half-level pressure profile.
   124	
   125	    Parameters
   126	    ----------
   127	    p_full : jax.Array
   128	        Full-level pressure [Pa], shape (..., nlev).
   129	    rho_full : jax.Array
   130	        Full-level density [kg/m^3], shape (..., nlev).
   131	    z_half : jax.Array
   132	        Interface height [m], shape (..., nlev+1), top-to-bottom ordering.
   133	
   134	    Returns
   135	    -------
   136	    jax.Array
   137	        Reconstructed half-level pressure [Pa], shape (..., nlev+1).
   138	    """
   139	    # Layer thicknesses are positive with top-to-bottom level indexing.
   140	    dz = jnp.abs(z_half[..., :-1] - z_half[..., 1:])
   141	    rho_pos = jnp.clip(rho_full, 1e-9, None)
   142	
   143	    # Hydrostatic increment across each full layer.
   144	    dp = constants.g * rho_pos * dz
   145	
   146	    # Top interface: centered estimate from top full level.
   147	    p_top = p_full[..., 0] - 0.5 * dp[..., 0]
   148	    p_top = jnp.clip(p_top, 1.0, None)
   149	
   150	    # Downward integration to all interfaces.
   151	    p_interfaces_inner = p_top[..., None] + jnp.cumsum(dp, axis=-1)
   152	    p_half = jnp.concatenate([p_top[..., None], p_interfaces_inner], axis=-1)
   153	    return jnp.clip(p_half, 1.0, None)
   154	
   155	
   156	# ==============================================================================
   157	# Moist thermodynamic functions (for convection)
   158	# ==============================================================================
   159	
   160	def moist_adiabat_lapse_rate(
   161	    T: jax.Array,
   162	    p: jax.Array,
   163	) -> jax.Array:
   164	    """Compute the moist adiabatic lapse rate dT/dp.
   165	
   166	    Gamma_m = (R_d * T / (c_pd * p)) *
   167	              (1 + L_v * q_sat / (R_d * T)) /
   168	              (1 + L_v^2 * q_sat / (c_pd * R_v * T^2))
   169	
   170	    Parameters
     1	"""Lightweight saturation thermodynamics for legoESM.
     2	
     3	This module provides `saturation_mixing_ratio` and
     4	`saturation_mixing_ratio_ice` with *no* dependency on
     5	``atmosphere.physics`` so that ``land/``, ``ice/``, ``ocean/``, and
     6	``coupler/`` modules can import them without pulling in the full
     7	atmosphere physics package.
     8	
     9	All operations are pure JAX and compatible with jit, grad, vmap, scan.
    10	
    11	Conventions — water-vapor mass variables
    12	----------------------------------------
    13	This module returns the **mixing ratio** ``r_sat = ε e_sat / (p - e_sat)``
    14	(mass of water vapor per unit mass of *dry* air).  Throughout the
    15	``atmosphere/physics`` source tree the prognostic field is named
    16	``q_v`` and many docstrings call it "specific humidity".  In the
    17	typical atmospheric regime where ``e_sat ≪ p``, mixing ratio and
    18	specific humidity differ by ``q ≈ r / (1 + r)`` — about 1% for
    19	``r = 0.01``.  The codebase uses these interchangeably; physics that
    20	needs the distinction (vertical-flux conservation in saturated tropical
    21	columns, q_c bookkeeping) should read this caveat carefully and
    22	convert explicitly when the 1% drift matters.
    23	"""
    24	
    25	from __future__ import annotations
    26	
    27	import jax
    28	import jax.numpy as jnp
    29	
    30	from legoesm import constants
    31	
    32	
    33	def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
    34	    """Compute saturation vapor pressure using Tetens formula.
    35	
    36	    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))
    37	
    38	    Parameters
    39	    ----------
    40	    T : jax.Array
    41	        Temperature [K].
    42	
    43	    Returns
    44	    -------
    45	    jax.Array
    46	        Saturation vapor pressure [Pa].
    47	    """
    48	    T_c = T - constants.T_freeze
    49	    return 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    50	
    51	
    52	def saturation_mixing_ratio(
    53	    T: jax.Array,
    54	    p: jax.Array,
    55	) -> jax.Array:
    56	    """Compute saturation mixing ratio using Tetens formula.
    57	
    58	    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))   where T_c = T - 273.15
    59	    q_sat = epsilon * e_sat / (p - e_sat)
    60	
    61	    Parameters
    62	    ----------
    63	    T : jax.Array
    64	        Temperature [K].
    65	    p : jax.Array
    66	        Pressure [Pa].
    67	
    68	    Returns
    69	    -------
    70	    jax.Array
    71	        Saturation mixing ratio [kg/kg].
    72	    """
    73	    e_sat = saturation_vapor_pressure(T)
    74	    # Smooth floor on denominator: preserves gradients near e_sat ≈ p
    75	    # instead of a hard clip that creates a zero-gradient plateau.
    76	    # softplus(x - 1) + 1 ≈ x for x >> 1, ≈ 1 for x << 1, smooth at x = 1.
    77	    denom = jax.nn.softplus(p - e_sat - 1.0) + 1.0
    78	    q_sat = constants.epsilon * e_sat / denom
    79	    # Smooth cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    80	    # while allowing gradients to flow (unlike hard jnp.minimum).
    81	    # Uses LogSumExp smooth-min: 1 - softplus(β(1 - x))/β with β = 20.
    82	    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat)) / 20.0
    83	
    84	
    85	def saturation_mixing_ratio_ice(
    86	    T: jax.Array,
    87	    p: jax.Array,
    88	) -> jax.Array:
    89	    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).
    90	
    91	    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    92	    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)
    93	
    94	    Parameters
    95	    ----------
    96	    T : jax.Array
    97	        Temperature [K].
    98	    p : jax.Array
    99	        Pressure [Pa].
   100	
   101	    Returns
   102	    -------
   103	    jax.Array
   104	        Ice saturation mixing ratio [kg/kg].
   105	    """
   106	    e_sat_i = 611.2 * jnp.exp(
   107	        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
   108	    )
   109	    denom = jax.nn.softplus(p - e_sat_i - 1.0) + 1.0
   110	    q_sat_i = constants.epsilon * e_sat_i / denom
   111	    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat_i)) / 20.0
   112	
   113	
   114	def saturation_specific_humidity(
   115	    T: jax.Array,
   116	    p: jax.Array,
   117	) -> jax.Array:
   118	    """Compute saturation specific humidity from saturation mixing ratio.
   119	
   120	    q = w_sat / (1 + w_sat)
   121	
   122	    where w_sat = epsilon * e_sat / (p - e_sat) is the saturation mixing
   123	    ratio.  Use this function when working with specific humidity fields
   124	    (q = m_v / (m_v + m_d)) rather than mixing ratio (w = m_v / m_d).
   125	
   126	    Parameters
   127	    ----------
   128	    T : jax.Array
   129	        Temperature [K].
   130	    p : jax.Array
   131	        Pressure [Pa].
   132	
   133	    Returns
   134	    -------
   135	    jax.Array
   136	        Saturation specific humidity [kg/kg].
   137	    """
   138	    w_sat = saturation_mixing_ratio(T, p)
   139	    return w_sat / (1.0 + w_sat)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/thermodynamics.py | sed -n '160,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   160	def moist_adiabat_lapse_rate(
   161	    T: jax.Array,
   162	    p: jax.Array,
   163	) -> jax.Array:
   164	    """Compute the moist adiabatic lapse rate dT/dp.
   165	
   166	    Gamma_m = (R_d * T / (c_pd * p)) *
   167	              (1 + L_v * q_sat / (R_d * T)) /
   168	              (1 + L_v^2 * q_sat / (c_pd * R_v * T^2))
   169	
   170	    Parameters
   171	    ----------
   172	    T : jax.Array
   173	        Temperature [K].
   174	    p : jax.Array
   175	        Pressure [Pa].
   176	
   177	    Returns
   178	    -------
   179	    jax.Array
   180	        Moist adiabatic lapse rate dT/dp [K/Pa].
   181	    """
   182	    R_d = constants.R_d
   183	    c_pd = constants.c_pd
   184	    L_v = constants.L_v
   185	    R_v = constants.R_v
   186	
   187	    q_sat = saturation_mixing_ratio(T, p)
   188	
   189	    numerator = 1.0 + L_v * q_sat / (R_d * T)
   190	    denominator = 1.0 + L_v ** 2 * q_sat / (c_pd * R_v * T ** 2)
   191	
   192	    return (R_d * T / (c_pd * p)) * numerator / denominator
   193	
   194	
   195	def compute_moist_adiabat(
   196	    T_base: jax.Array,
   197	    p_levels: jax.Array,
   198	) -> jax.Array:
   199	    """Compute moist adiabatic temperature profile from surface upward.
   200	
   201	    Integrates dT/dp = Gamma_m(T, p) upward from the lowest pressure
   202	    level using trapezoidal predictor-corrector via jax.lax.scan.
   203	
   204	    Parameters
   205	    ----------
   206	    T_base : jax.Array
   207	        Temperature at the lowest level (surface) [K], shape (ncol,).
   208	    p_levels : jax.Array
   209	        Pressure at full levels [Pa], shape (ncol, nlev).
   210	        Ordered top-to-bottom (p increasing with index).
   211	
   212	    Returns
   213	    -------
   214	    jax.Array
   215	        Moist adiabatic temperature profile [K], shape (ncol, nlev).
   216	
   217	    Notes
   218	    -----
   219	    **Saturated-everywhere assumption.**  ``moist_adiabat_lapse_rate`` calls
   220	    ``saturation_mixing_ratio(T, p)`` at every level, i.e. the parcel is
   221	    treated as saturated all the way down to the surface.  For an
   222	    unsaturated launch parcel (LCL above the base) the *true* parcel curve
   223	    follows a dry adiabat from the base to the LCL and only switches to the
   224	    moist adiabat above LCL.  The current implementation uses the moist
   225	    rate the entire way down, which UNDERESTIMATES the parcel-environment
   226	    contrast in subsaturated boundary layers and therefore systematically
   227	    biases CAPE diagnosed from this profile *low* for unsaturated parcels.
   228	    See ``compute_lcl`` for explicit LCL diagnosis.  Fixing this requires
   229	    threading ``q_v_base`` through the 8 convection schemes that consume
   230	    this function and is intentionally deferred (audit B4 / 2026-05-01).
   231	    """
   232	    ncol, nlev = p_levels.shape
   233	
   234	    # Reverse to scan from surface (bottom) upward (top)
   235	    p_rev = p_levels[:, ::-1]  # (ncol, nlev), surface first
   236	
   237	    # Promote to common dtype so scan carry types are consistent.
   238	    # Physical constants in moist_adiabat_lapse_rate are Python float64;
   239	    # computation internally uses the promoted dtype, but the carry must
   240	    # have matching input/output dtypes for jax.lax.scan.

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/config.py | sed -n '120,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   120	    envelope_scale: float = 1.0
   121	    directional_spread: float = 1.0
   122	    min_wind_sharpness: float = 20.0
   123	    softmin_sharpness: float = 50.0
   124	    tau_max: float = 10.0
   125	
   126	
   127	class HinesConfig(NamedTuple):
   128	    """Configuration for Hines (1997) Doppler-spread parameterization.
   129	
   130	    Fields
   131	    ------
   132	    rms_gw_speed : float
   133	        RMS gravity wave speed [m/s] (default 1.0).
   134	    m_star : float
   135	        Characteristic vertical wavenumber [1/m] (default 2*pi/2e3).
   136	    total_rms_wind : float
   137	        Total RMS gravity wave wind [m/s] (default 2.0).
   138	    cutoff_wn : float
   139	        Maximum vertical wavenumber [1/m] (default 2*pi/500).
   140	    Fmax : float
   141	        Saturation momentum flux cap [Pa] (default 0.1).
   142	    doppler_sharpness : float
   143	        Sigmoid sharpness for Doppler saturation (default 50.0).
   144	    """
   145	    rms_gw_speed: float = 1.0
   146	    m_star: float = 2.0 * math.pi / 2e3
   147	    total_rms_wind: float = 2.0
   148	    cutoff_wn: float = 2.0 * math.pi / 500.0
   149	    Fmax: float = 0.1
   150	    doppler_sharpness: float = 50.0
   151	    U_mag_floor: float = 0.1  # Wind-magnitude floor for projection [m/s]
   152	
   153	
   154	class PrognosticSpectralConfig(NamedTuple):
   155	    """Configuration for prognostic spectral GWD.
   156	
   157	    Fields
   158	    ------
   159	    n_azimuths : int
   160	        Number of azimuthal directions (default 4).
   161	    n_wavenumbers : int
   162	        Number of spectral bins (default 20).
   163	    k_min : float
   164	        Minimum horizontal wavenumber [1/m] (default 2*pi/100e3).
   165	    k_max : float
   166	        Maximum horizontal wavenumber [1/m] (default 2*pi/1e3).
   167	    launch_flux : float
   168	        Source momentum flux [Pa] (default 1e-3).
   169	    breaking_threshold : float
   170	        Froude threshold for wave breaking (default 1.0).
   171	    breaking_sharpness : float
   172	        Sigmoid sharpness for breaking transition (default 10.0).
   173	    tau_decay : float
   174	        Relaxation timescale for prognostic spectrum [s] (default 86400).
   175	    """
   176	    n_azimuths: int = 4
   177	    n_wavenumbers: int = 20
   178	    k_min: float = 2.0 * math.pi / 100e3
   179	    k_max: float = 2.0 * math.pi / 1e3
   180	    launch_flux: float = 1e-3
   181	    breaking_threshold: float = 1.0
   182	    breaking_sharpness: float = 10.0
   183	    tau_decay: float = 86400.0
   184	
   185	
   186	class MLEmulatorConfig(NamedTuple):
   187	    """Configuration for ML-based GWD emulator.
   188	
   189	    Fields
   190	    ------
   191	    n_input : int
   192	        Number of input features per level (default 7).
   193	    n_hidden : int
   194	        Hidden layer width (default 128).
   195	    n_layers : int
   196	        Number of MLP layers (default 3).
   197	    n_output : int
   198	        Number of output tendencies per level (default 3).
   199	    seed : int
   200	        Random seed for initialization (default 0).
   201	    use_residual : bool
   202	        Scale outputs for residual learning (default True).
   203	    """
   204	    n_input: int = 7
   205	    n_hidden: int = 128
   206	    n_layers: int = 3
   207	    n_output: int = 3
   208	    seed: int = 0
   209	    use_residual: bool = True
   210	    norm_u: float = 30.0     # Wind scale [m/s] for u, v normalization
   211	    norm_T: float = 300.0    # Temperature scale [K]
   212	    norm_z: float = 30000.0  # Height scale [m]
   213	
   214	
   215	class GravityWaveDragConfig(NamedTuple):
   216	    """Top-level gravity wave drag configuration.
   217	
   218	    Selects the active scheme and holds sub-configurations.
   219	
   220	    Fields

exec
/bin/zsh -lc 'rg -n "hines|Fmax|drag|sigma|saturation" tests/unit/test_physics_gwd.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
3:Tests drag opposing wind, energy dissipation, Rayleigh sponge structure,
13:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
14:from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
19:    from legoesm.grids.vertical import create_sigma_coordinate
23:    sigma = create_sigma_coordinate(nlev)
24:    state = held_suarez_init(grid, sigma)
36:    return state, grid, sigma
40:    state, grid, sigma = _make_state(**kwargs)
43:    tend, prog = gwd_fn(state, grid, sigma)
47:ALL_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral"]
48:# Diagnostic schemes (not multi-directional) where drag should oppose wind
49:DIAGNOSTIC_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines"]
57:def test_drag_opposes_wind(scheme):
58:    """du_dt * u <= 0 wherever drag is active (drag decelerates)."""
69:            f"{scheme}: only {frac:.0%} of active points have drag opposing wind"
74:# 6d  Zero wind -> zero drag
78:def test_zero_wind_zero_drag(scheme):
80:    state, grid, sigma = _make_state(wind_speed=0.0)
87:    tend, _ = gwd_fn(state, grid, sigma)
126:    """Rayleigh drag should be active at model top and/or BL."""
129:    top_drag = float(jnp.max(jnp.abs(du_dt[..., :3])))
130:    bot_drag = float(jnp.max(jnp.abs(du_dt[..., -5:])))
131:    assert top_drag > 1e-8 or bot_drag > 1e-8, "Rayleigh: no drag anywhere"
138:def test_hines_no_saturation_zero_drag():
139:    """In the no-saturation limit (very tiny m_star, sigma_sat huge), Hines
140:    drag must vanish.
145:    saturation amplitude ``sigma_sat = N / (m_star * rho_ratio)`` ~ 1e6
149:    A bug that compounds the rho ratio at every scan step (sigma_grown
153:    re-activates saturation even when m_star is tiny.  Pre-fix this
158:    from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
159:    from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
184:    out = hines_gwd(u, v, T, p_full, p_half, z_full_b, z_half_b, rho, lat,
187:    # Post-fix expectation: drag <= numerical noise.  Pre-fix this is
190:        f"Hines: max |du/dt| = {max_du:.3e} m/s^2 in the no-saturation "
191:        "limit; expected ~0 (sigma_sat is set ~1e6 m/s by tiny m_star). "
192:        "A non-zero drag indicates the WKB amplitude is over-amplified "
204:# every scheme's drag profile through theta -> N^2 -> Hines/Lindzen
205:# saturation, through stress closure, or through the wind shear ->
210:def test_mcfarlane_drag_scales_linearly_with_k_wave():
211:    """McFarlane drag must scale ~linearly with ``k_wave`` — post-fix
214:    constant) and the absolute drag scales proportionally.
218:    * Post-fix (both have k):     drag ∝ k    (this test asserts)
219:    * k removed from tau_0 only:  drag DECREASES (or saturates) with k
220:    * k removed from tau_sat only: drag is non-monotonic / step-like
221:    * k removed from both (full pre-fix bug): drag k-independent
230:    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
231:    from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
255:    def drag_at_k(k_factor):
263:    drag_lo, eps_lo = drag_at_k(1.0)
264:    drag_hi, eps_hi = drag_at_k(10.0)
266:    # Pre-fix sanity: drag must actually exist (post-fix gives ~5e-3).
267:    assert drag_lo > 1e-6, (
268:        f"McFarlane drag at k_factor=1.0 = {drag_lo:.3e} — too small to "
272:    # Linear scaling: drag(k=10×default) / drag(k=default) ≈ 10.
274:    # in the saturation cap.  Pre-fix bug patterns:
276:    #   * k removed from tau_0 only:  ratio < 1 (drag falls or stays)
279:    ratio_du = drag_hi / drag_lo
293:def test_mcfarlane_saturation_path_fires_in_uniform_wind():
294:    """McFarlane saturation path must fire in a UNIFORM-wind column.
299:    drag deposition there regardless of the launch-stress units.
301:    A uniform-wind column has no critical level, so saturation fires
308:      clipped to 10) never exceeds tau_sat anywhere → drag = 0
315:      well above it near the top.  The wave breaks aloft and drag
321:    and reasonable numerical drift in the saturation scan.
324:    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
325:    from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
361:        "column — saturation never fires.  This is the signature of "
364:        "at 10), wave never breaks, drag at float-precision floor."
368:        "large; check that the launch-stress clip and saturation scan "
374:        "with active drag — must be positive."
382:    # ~22500 → clipped to 10) but the saturation stress is also
384:    # and drag collapses to ~0.
411:    the saturation path.
414:    from legoesm.grids.vertical import create_sigma_coordinate
418:    sigma = create_sigma_coordinate(nlev)
419:    state = held_suarez_init(grid, sigma)
434:    return state, grid, sigma
437:def _gwd_config_with_active_drag(scheme):
439:    actually generates non-trivial drag.
442:    ``tau_0 ~ rho * N * k * h^2 * U`` and the saturation stress
445:    wave never breaks → drag ≈ 0 → autodiff gradient rounds to zero.
448:    the model top, so the saturation scan deposits real drag and the
451:    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
467:@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane", "hines"])
472:    Rayleigh is a sponge layer that depends only on ``u, v, sigma`` —
480:    state, grid, sigma = _make_sheared_state()
486:        config = _gwd_config_with_active_drag(scheme)
488:        tend, _ = gwd_fn(state_p, grid, sigma)
500:@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane", "hines"])
509:    state, grid, sigma = _make_sheared_state()
515:        config = _gwd_config_with_active_drag(scheme)
517:        tend, _ = gwd_fn(state_p, grid, sigma)
537:    state, grid, sigma = _make_state(wind_speed=10.0)
538:    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
548:        tend, _ = gwd_fn(state, grid, sigma)
556:        "non-trivial drag profile.  A zero gradient indicates the spectral "

exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_gwd.py | sed -n '437,470p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   437	def _gwd_config_with_active_drag(scheme):
   438	    """Per-scheme config tuned so the held_suarez-init test column
   439	    actually generates non-trivial drag.
   440	
   441	    For the orographic schemes (Lindzen, McFarlane), the launch stress
   442	    ``tau_0 ~ rho * N * k * h^2 * U`` and the saturation stress
   443	    ``tau_sat ~ rho * U^3 / N`` differ by ``k * h^2 * N^2 / U^2``.  With
   444	    default ``h_topo=500 m`` and ``N=0.01`` the ratio is ~1e-4 — the
   445	    wave never breaks → drag ≈ 0 → autodiff gradient rounds to zero.
   446	
   447	    Bumping ``h_topo`` to 10 km pushes ``tau_0`` above ``tau_sat`` near
   448	    the model top, so the saturation scan deposits real drag and the
   449	    differentiability tests exercise a non-trivial code path.
   450	    """
   451	    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
   452	        LindzenConfig, McFarlaneConfig,
   453	    )
   454	    if scheme == "lindzen":
   455	        return GravityWaveDragConfig(
   456	            scheme="lindzen",
   457	            lindzen=LindzenConfig(h_topo=10_000.0),
   458	        )
   459	    if scheme == "mcfarlane":
   460	        return GravityWaveDragConfig(
   461	            scheme="mcfarlane",
   462	            mcfarlane=McFarlaneConfig(h_topo=10_000.0),
   463	        )
   464	    return GravityWaveDragConfig(scheme=scheme)
   465	
   466	
   467	@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane", "hines"])
   468	def test_gwd_grad_through_T_finite_and_nonzero(scheme):
   469	    """``d sum(dT_dt^2) / d (T-perturb)`` is finite and (for non-Rayleigh)
   470	    non-zero.

exec
/bin/zsh -lc 'rg -n "N_c|N_r|N_i|number" tests src/legoesm | head -n 120' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/cli.py:41:                             help="Test case number (2 or 5)")
src/legoesm/grids/gaussian.py:86:    ls: jax.Array           # Total wavenumber n for each SH index, (n_sh,)
src/legoesm/grids/gaussian.py:87:    ms: jax.Array           # Zonal wavenumber m for each SH index, (n_sh,)
src/legoesm/grids/gaussian.py:352:        Maximum total wavenumber.
src/legoesm/grids/gaussian.py:520:    zonal wavenumber *m* without materializing the full ``(n_lat, n_sh)``
src/legoesm/grids/gaussian.py:534:    # segment_sum groups along the spectral axis by wavenumber m.
src/legoesm/grids/cubed_sphere_cdgrid.py:135:    # Used for Courant number: crx = dt*ut*rdxa(upwind_cell) (sw_core.F90:850).
src/legoesm/atmosphere/physics/ml_parameterization.py:262:        dN_c_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:263:        dN_r_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:264:        dN_i_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:358:        dN_c_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:359:        dN_r_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:360:        dN_i_dt=zeros,
src/legoesm/grids/cubed_sphere.py:38:    All arrays have shape (6, n, n) where 6 = number of faces,
src/legoesm/grids/cubed_sphere.py:389:    Face numbering:
src/legoesm/grids/cubed_sphere.py:604:    - Omega scales as X (to keep Rossby number constant)
src/legoesm/diagnostics/monthly_means.py:85:            Year number (for multi-year runs).
src/legoesm/grids/halo.py:42:# Face numbering (from cubed_sphere.py _face_to_cartesian):
src/legoesm/grids/halo.py:312:    # right number of singleton axes so weights broadcast.
src/legoesm/grids/halo.py:1581:    n : int — number of cells per face edge
tests/distributed/test_halo_mpi.py:273:        # Each rank contributes its rank number.
src/legoesm/training/neural_gcm_spectral.py:494:        # Exponential spectral filter on highest wavenumbers
src/legoesm/grids/voronoi.py:1193:        # Exact integer number of cells around the periodic extent.
src/legoesm/training/sfno_dycore_coupling.py:34:    "dN_c_dt",
src/legoesm/training/sfno_dycore_coupling.py:35:    "dN_r_dt",
src/legoesm/training/sfno_dycore_coupling.py:36:    "dN_i_dt",
src/legoesm/atmosphere/physics/neural_physics.py:47:    "dN_c_dt",
src/legoesm/atmosphere/physics/neural_physics.py:48:    "dN_r_dt",
src/legoesm/atmosphere/physics/neural_physics.py:49:    "dN_i_dt",
tests/williamson_diagnostic.py:9:Test Case 6: Rossby-Haurwitz Wave number 4 (nonlinear dynamics)
tests/williamson_diagnostic.py:440:# Test Case 6: Rossby-Haurwitz Wave (Wave number 4)
tests/williamson_diagnostic.py:446:    Rossby-Haurwitz wave number 4.
tests/williamson_diagnostic.py:506:    print(f"WILLIAMSON TEST CASE 6: Rossby-Haurwitz Wave (number 4)")
src/legoesm/da/minimizer.py:105:        L-BFGS memory (number of correction pairs).
src/legoesm/ml/physics/data.py:193:            N_c=zeros_3d,
src/legoesm/ml/physics/data.py:194:            N_r=zeros_3d,
src/legoesm/ml/physics/data.py:195:            N_i=zeros_3d,
tests/atmosphere/shallow_water/test_cases/williamson_mpas.py:149:    R_val = 4       # wave number
src/legoesm/driver/model_driver.py:1827:                from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx_cubed_sphere
src/legoesm/driver/model_driver.py:1833:                    _seg_max_cfl = float(cfl_number_from_state(
src/legoesm/driver/model_driver.py:1839:                    _seg_max_cfl = float(cfl_number_from_state(
src/legoesm/driver/model_driver.py:2115:            # Apply ice/number tracer tendencies when full registry is active
src/legoesm/driver/model_driver.py:2120:                self.tracers["N_c"] = jnp.maximum(
src/legoesm/driver/model_driver.py:2121:                    self.tracers["N_c"] + DT * phys_out.dN_c_dt, 0.0
src/legoesm/driver/model_driver.py:2123:                self.tracers["N_r"] = jnp.maximum(
src/legoesm/driver/model_driver.py:2124:                    self.tracers["N_r"] + DT * phys_out.dN_r_dt, 0.0
src/legoesm/driver/model_driver.py:2126:                self.tracers["N_i"] = jnp.maximum(
src/legoesm/driver/model_driver.py:2127:                    self.tracers["N_i"] + DT * phys_out.dN_i_dt, 0.0
src/legoesm/da/gen_be.py:495:        # grid.lap = -n(n+1)/a^2, so this decays with total wavenumber (smoothing).
src/legoesm/da/gen_be.py:781:        states this is memory-proportional to the number of scalar operations in
tests/atmosphere/hydrostatic/validation/test_stability_fix.py:20:# Small hyperdiffusion for high wavenumbers
src/legoesm/da/background_error.py:191:        Decorrelation wavenumber.
src/legoesm/da/background_error.py:207:        ls = grid.ls  # Total wavenumber for each spectral index
src/legoesm/driver/physics_pipeline.py:41:    dN_c_dt: jax.Array
src/legoesm/driver/physics_pipeline.py:42:    dN_r_dt: jax.Array
src/legoesm/driver/physics_pipeline.py:43:    dN_i_dt: jax.Array
src/legoesm/driver/physics_pipeline.py:163:                            N_c=None, N_r=None, N_i=None):
src/legoesm/driver/physics_pipeline.py:300:        dN_c_dt = jnp.zeros(shape_3d, dtype=_sd)
src/legoesm/driver/physics_pipeline.py:301:        dN_r_dt = jnp.zeros(shape_3d, dtype=_sd)
src/legoesm/driver/physics_pipeline.py:302:        dN_i_dt = jnp.zeros(shape_3d, dtype=_sd)
src/legoesm/driver/physics_pipeline.py:314:            dN_c_dt = ad.unflatten_3d(micro_out_ml.dN_c_dt)
src/legoesm/driver/physics_pipeline.py:315:            dN_r_dt = ad.unflatten_3d(micro_out_ml.dN_r_dt)
src/legoesm/driver/physics_pipeline.py:316:            dN_i_dt = ad.unflatten_3d(micro_out_ml.dN_i_dt)
src/legoesm/driver/physics_pipeline.py:330:                N_c=ad.flatten_3d(N_c) if N_c is not None else _z,
src/legoesm/driver/physics_pipeline.py:331:                N_r=ad.flatten_3d(N_r) if N_r is not None else _z,
src/legoesm/driver/physics_pipeline.py:332:                N_i=ad.flatten_3d(N_i) if N_i is not None else _z,
src/legoesm/driver/physics_pipeline.py:370:            dN_c_dt = ad.unflatten_3d(micro_out.dN_c_dt)
src/legoesm/driver/physics_pipeline.py:371:            dN_r_dt = ad.unflatten_3d(micro_out.dN_r_dt)
src/legoesm/driver/physics_pipeline.py:372:            dN_i_dt = ad.unflatten_3d(micro_out.dN_i_dt)
src/legoesm/driver/physics_pipeline.py:464:            dN_c_dt=dN_c_dt,
src/legoesm/driver/physics_pipeline.py:465:            dN_r_dt=dN_r_dt,
src/legoesm/driver/physics_pipeline.py:466:            dN_i_dt=dN_i_dt,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:218:                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:222:                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:368:                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:372:                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:499:                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:503:                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
src/legoesm/grids/regridding.py:39:        Total number of source grid points.
tests/ocean/unit/test_barotropic_noise_invariant.py:280:    # ~ sqrt(N_cells) · tol, the cumulative drift should remain << 1e-6.
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:87:    # ``[0, 10] Pa`` — which combined with the missing wavenumber gave
src/legoesm/grids/latlon.py:31:    n_lat: int                  # number of latitude points
src/legoesm/grids/latlon.py:32:    n_lon: int                  # number of longitude points
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:408:    advects a passive zonal-wavenumber tracer.  The tracer must stay
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:426:        # zonal-wavenumber-1 perturbation in the lowest temperature
src/legoesm/grids/polar_filter.py:4:explicit time-stepping unstable unless high-wavenumber modes are damped.
src/legoesm/grids/polar_filter.py:7:keeping only wavenumbers that satisfy the CFL condition for the fastest
src/legoesm/grids/polar_filter.py:10:The maximum stable wavenumber at each latitude is:
src/legoesm/grids/polar_filter.py:56:    # CFL-based max wavenumber at each latitude
src/legoesm/grids/polar_filter.py:67:    wavenumbers = jnp.arange(n_freq)
src/legoesm/grids/polar_filter.py:68:    mask = (wavenumbers[None, :] <= max_k[:, None]).astype(jnp.float32)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:3:Multi-azimuthal, multi-wavenumber spectral GWD with a prognostic
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:47:        Input wave spectrum, shape (ncol, n_azimuths, n_wavenumbers).
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:58:    n_wn = config.n_wavenumbers
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:76:    # Wavenumber grid (log-spaced)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:92:    # Phase speed per wavenumber: c = N / k
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:104:    # Bottom-up scan per (azimuth, wavenumber)
tests/validation/test_ec_eigenvalues2.py:124:N_correct = R_d * S
tests/validation/test_ec_eigenvalues2.py:125:analyze_system("Correct PGF + current", N_correct, M_current)
tests/validation/test_ec_eigenvalues2.py:130:analyze_system("Correct PGF + EC flux", N_correct, M_ec)
tests/validation/test_ec_eigenvalues2.py:139:analyze_system("Correct PGF + theory", N_correct, M_theory_correct)
tests/validation/test_ec_eigenvalues2.py:197:        ("Correct PGF + current", N_correct, M_current),
tests/validation/test_ec_eigenvalues2.py:198:        ("Correct PGF + EC flux", N_correct, M_ec),
tests/ocean/unit/test_ocean_diagnostics.py:186:        """A single Fourier mode should produce a peak at the right wavenumber."""
tests/ocean/unit/test_ocean_diagnostics.py:190:        # Mode with wavenumber k = 4 * dk in x-direction
tests/ocean/unit/test_ocean_diagnostics.py:273:        """Sinusoidal vorticity has a spectral peak at the right wavenumber."""
tests/validation/test_ensemble_correctness.py:54:N_STEPS = 5    # number of integration steps
tests/validation/test_ensemble_correctness.py:55:N_ENSEMBLE = 2 # number of ensemble members
tests/validation/test_ensemble_correctness.py:65:    "dN_c_dt",
tests/validation/test_ensemble_correctness.py:66:    "dN_r_dt",
tests/validation/test_ensemble_correctness.py:67:    "dN_i_dt",
tests/validation/test_ensemble_correctness.py:246:    from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx_cubed_sphere
src/legoesm/driver/component_factory.py:251:        # Compute hyperdiffusion from truncation: 0.5-hour e-folding at max wavenumber
src/legoesm/driver/component_factory.py:329:        dt_max_advective = cfl_max_dt(dx_pole, c_grav, cfl_number=0.8, ndim=1)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:7:    - Flowaround regime (inverse Froude number > 1)
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:58:        Horizontal wavenumber [1/m] (default 2*pi/100e3).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:59:    N_ref : float
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:62:        Critical Froude number threshold (default 1.0).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:64:        Sigmoid sharpness for Froude number transition (default 20.0).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:68:    N_ref: float = 0.01

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py | sed -n '60,125p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    60	    """
    61	    ncol, nlev = T.shape
    62	    q_c = hydrometeors.q_c
    63	    q_r = hydrometeors.q_r
    64	    N_c = hydrometeors.N_c
    65	    N_r = hydrometeors.N_r
    66	    sharpness = config.saturation_sharpness
    67	
    68	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    69	
    70	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    71	
    72	    # 1. Autoconversion (mass-dependent)
    73	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    74	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    75	    )
    76	
    77	    # 2. Accretion
    78	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    79	
    80	    # 3-4. Self-collection and breakup
    81	    dN_r_sc, dN_r_br = self_collection_breakup(
    82	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    83	    )
    84	
    85	    # 5. Rain evaporation
    86	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    87	
    88	    # 6. Sedimentation — Marshall-Palmer fall speed (q_r * rho/rho_sfc)^b_v_r
    89	    # has fractional exponent (b_v_r=0.5); guard the AD path with safe_pow.
    90	    rho_sfc = rho[:, -1:]
    91	    V_t_r = config.a_v_r * safe_pow(
    92	        jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1), config.b_v_r,
    93	    )
    94	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    95	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
    96	
    97	    # 7. Latent heating
    98	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
    99	
   100	    # Combine tendencies
   101	    dq_v_dt = -condensation + evaporation
   102	    dq_c_dt = condensation - dq_c_au - dq_c_ac
   103	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + sed_r
   104	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   105	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   106	
   107	    # Precipitation
   108	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   109	    precipitation = q_r_bot * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   110	
   111	    # Pin dtype to the input precision so we never silently promote
   112	    # the unused-species placeholders to f64 under x64 mode.
   113	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   114	    return MicrophysicsOutput(
   115	        dT_dt=dT_dt,
   116	        dq_v_dt=dq_v_dt,
   117	        dq_c_dt=dq_c_dt,
   118	        dq_r_dt=dq_r_dt,
   119	        dq_i_dt=z,
   120	        dq_s_dt=z,
   121	        dq_g_dt=z,
   122	        dN_c_dt=dN_c_dt,
   123	        dN_r_dt=dN_r_dt,
   124	        dN_i_dt=z,
   125	        precipitation=precipitation,

codex
**Findings**

- **P1: plume entrainment still uses unstable explicit Euler.** [convection/_plume.py:536](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:536>), [537](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:537>), [578](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:578>)  
  `T_u_ent`, `q_u_ent`, and `q_c_u_ent` use `1 - eps*dz` style updates. With Bechtold defaults `epsilon_deep=1.75e-3` and `epsilon_shallow=3e-3` [config.py:516](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/config.py:516>), layers thicker than about 570 m or 333 m hit `eps*dz > 1`. That can overshoot `T/q`, make `q_u_ent` negative, and zero the `q_c_u` gradient. Use exponential relaxation/dilution: `X_e + (X_prev - X_e) * exp(-eps*dz)` and `q_c_u_prev * exp(-eps*dz)`.

- **P1: Hines drag has a dimensional inconsistency.** [hines.py:127](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127>), [139](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:139>)  
  The comments state wave flux scales like `rho * sigma^2`, and `Fmax` is documented as Pa [config.py:140](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:140>). But the code uses `rho * (sigma_grown - sigma_new)`, units `kg m^-2 s^-1`, not Pa. Then `drag/(rho*dz)` has units `1/s`, not `m/s^2`. The deposited stress should be based on a squared-amplitude difference, e.g. proportional to `rho * (sigma_grown**2 - sigma_new**2)`.

- **P2: plume termination is local, so a dead plume can revive aloft.** [convection/_plume.py:529](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:529>), [590](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:590>)  
  Keeping `plume_alive` out of the carry means negative buoyancy only suppresses the current reported level. If buoyancy becomes positive again above an inversion, `M_u_raw` reappears. That contradicts the documented “zero ... at/above LNB” behavior. A cumulative survival factor or terminal-detrainment sink is needed. I would not put the sub-cloud `abv` mask directly into the carry, because that would attenuate `M_b` before cloud base.

- **P2: Tiedtke MC proxy is not smooth at the RH-crit onset.** [tiedtke.py:195](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/tiedtke.py:195>), [203](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/tiedtke.py:203>)  
  `jnp.maximum(q_v - RHcrit*q_sat, 0)` gives exactly zero dry-side gradient. The later smooth `mc_gate` cannot restore that. If AD tuning through marginally dry columns matters, replace this with `smooth_positive_part` using a moisture-scale sharpness.

- **P2: warm-rain number tendency units are internally inconsistent.** [microphysics/_warm_rain.py:129](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:129>), [132](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:132>)  
  The docs say `N_c`, `N_r`, `N_i` are per kg, but `x_c = q_c*rho/N_c` and `dN_r_au = dq_c_au*rho/(...)` are per-volume style formulas. Under the documented tracer units, `rho` should not be in those two expressions.

- **P2: Thompson graupel conversion can draw from the wrong donor.** [thompson.py:159](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:159>), [220](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:220>)  
  `rime_to_graupel` is based on total riming, but the tendency always subtracts a full `rime_to_graupel` from `q_i` and half from `q_s`. If `q_i=0` and snow riming is active, this still drives `q_i` negative. Split by `riming_i/riming_s` donors or clamp against available `q_i/q_s`.

**Answers**

1. Yes. `q_c_u_prev * exp(-eps*dz)` is the correct differentiable dilution, and the same exponential relaxation should be applied to `T_u_ent` and `q_u_ent`.
2. Sub-cloud taper should not simply enter the carry, but buoyancy death should have memory. Current local-only `plume_alive` permits plume revival.
3. The `where` on `dq_c_conv_dt` is harmless and redundant; no unsafe AD branch is present.
4. Not smooth enough for dry-side gradients; the hard `maximum` is the limiting kink.
5. For valid columns, `pbl_weight` is sigmoid-positive and `dp>0`, so `pbl_norm` should not vanish. If geometry is invalid, the floor can produce a zero-pressure parcel; use a surface-parcel fallback defensively.
6. `dN_r_au *= qc_scale` is not double-scaling; it scales the pre-clamp number tendency once. Separate issue: the base formula’s `rho` factor appears unit-inconsistent.
7. `safe_pow` is AD-safe at `x<=0`; the inactive branch routes through `1.0`, so no `0**(p-1)` appears.
8. `dq_v_dt = -condensation + evaporation - dq_i_dep` is complete for Bergeron/riming/melting/aggregation as written.
9. The value is finite, but the gradient near `q_i=0` can be extremely large when `N_i>0` because of the `1e-15` denominator floor.
10. The hard clip is acceptable as a sign guard, but it creates zero-gradient regions. The larger issue is the Hines stress unit bug above.

Static review only; I did not run the full unit suite.
2026-05-01T22:59:46.371345Z ERROR codex_core::session: failed to record rollout items: thread 019de5bd-f6ae-7821-b252-36c14701abf3 not found
tokens used
147,433
**Findings**

- **P1: plume entrainment still uses unstable explicit Euler.** [convection/_plume.py:536](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:536>), [537](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:537>), [578](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:578>)  
  `T_u_ent`, `q_u_ent`, and `q_c_u_ent` use `1 - eps*dz` style updates. With Bechtold defaults `epsilon_deep=1.75e-3` and `epsilon_shallow=3e-3` [config.py:516](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/config.py:516>), layers thicker than about 570 m or 333 m hit `eps*dz > 1`. That can overshoot `T/q`, make `q_u_ent` negative, and zero the `q_c_u` gradient. Use exponential relaxation/dilution: `X_e + (X_prev - X_e) * exp(-eps*dz)` and `q_c_u_prev * exp(-eps*dz)`.

- **P1: Hines drag has a dimensional inconsistency.** [hines.py:127](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127>), [139](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:139>)  
  The comments state wave flux scales like `rho * sigma^2`, and `Fmax` is documented as Pa [config.py:140](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:140>). But the code uses `rho * (sigma_grown - sigma_new)`, units `kg m^-2 s^-1`, not Pa. Then `drag/(rho*dz)` has units `1/s`, not `m/s^2`. The deposited stress should be based on a squared-amplitude difference, e.g. proportional to `rho * (sigma_grown**2 - sigma_new**2)`.

- **P2: plume termination is local, so a dead plume can revive aloft.** [convection/_plume.py:529](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:529>), [590](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:590>)  
  Keeping `plume_alive` out of the carry means negative buoyancy only suppresses the current reported level. If buoyancy becomes positive again above an inversion, `M_u_raw` reappears. That contradicts the documented “zero ... at/above LNB” behavior. A cumulative survival factor or terminal-detrainment sink is needed. I would not put the sub-cloud `abv` mask directly into the carry, because that would attenuate `M_b` before cloud base.

- **P2: Tiedtke MC proxy is not smooth at the RH-crit onset.** [tiedtke.py:195](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/tiedtke.py:195>), [203](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/tiedtke.py:203>)  
  `jnp.maximum(q_v - RHcrit*q_sat, 0)` gives exactly zero dry-side gradient. The later smooth `mc_gate` cannot restore that. If AD tuning through marginally dry columns matters, replace this with `smooth_positive_part` using a moisture-scale sharpness.

- **P2: warm-rain number tendency units are internally inconsistent.** [microphysics/_warm_rain.py:129](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:129>), [132](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:132>)  
  The docs say `N_c`, `N_r`, `N_i` are per kg, but `x_c = q_c*rho/N_c` and `dN_r_au = dq_c_au*rho/(...)` are per-volume style formulas. Under the documented tracer units, `rho` should not be in those two expressions.

- **P2: Thompson graupel conversion can draw from the wrong donor.** [thompson.py:159](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:159>), [220](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:220>)  
  `rime_to_graupel` is based on total riming, but the tendency always subtracts a full `rime_to_graupel` from `q_i` and half from `q_s`. If `q_i=0` and snow riming is active, this still drives `q_i` negative. Split by `riming_i/riming_s` donors or clamp against available `q_i/q_s`.

**Answers**

1. Yes. `q_c_u_prev * exp(-eps*dz)` is the correct differentiable dilution, and the same exponential relaxation should be applied to `T_u_ent` and `q_u_ent`.
2. Sub-cloud taper should not simply enter the carry, but buoyancy death should have memory. Current local-only `plume_alive` permits plume revival.
3. The `where` on `dq_c_conv_dt` is harmless and redundant; no unsafe AD branch is present.
4. Not smooth enough for dry-side gradients; the hard `maximum` is the limiting kink.
5. For valid columns, `pbl_weight` is sigmoid-positive and `dp>0`, so `pbl_norm` should not vanish. If geometry is invalid, the floor can produce a zero-pressure parcel; use a surface-parcel fallback defensively.
6. `dN_r_au *= qc_scale` is not double-scaling; it scales the pre-clamp number tendency once. Separate issue: the base formula’s `rho` factor appears unit-inconsistent.
7. `safe_pow` is AD-safe at `x<=0`; the inactive branch routes through `1.0`, so no `0**(p-1)` appears.
8. `dq_v_dt = -condensation + evaporation - dq_i_dep` is complete for Bergeron/riming/melting/aggregation as written.
9. The value is finite, but the gradient near `q_i=0` can be extremely large when `N_i>0` because of the `1e-15` denominator floor.
10. The hard clip is acceptable as a sign guard, but it creates zero-gradient regions. The larger issue is the Hines stress unit bug above.

Static review only; I did not run the full unit suite.
