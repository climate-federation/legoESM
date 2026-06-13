"""Seed baseline for the inline-physics-coefficient ratchet.

``COEFF_BUDGET`` is the per-file allowance of sanctioned inline empirical-float
(and large-int) sites measured from the tree when the gate
(``tests/test_no_inline_physics_coeffs``) was introduced. Format:
``{repo_rel_path: {normalized_source_line: count}}``. EXACT / shrink-only: a
migrated file must drop (or reduce) its entry in the same commit; a file absent
here gets a zero budget (new files born clean); ``set(COEFF_BUDGET) <=
_SEED_FILES`` is asserted so a new file can never be budgeted. Re-seed with
``scripts/tmp/_seed_coeff_baseline.py``.
"""

from __future__ import annotations

COEFF_BUDGET: dict[str, dict[str, int]] = {
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/_thompson_snow.py': {
        'dv = 8.794e-5 * safe_pow(T, 1.81) / jnp.clip(p_full, 1.0)': 2,
        'ka = 2.3971e-2 + 7.078e-5 * (T - constants.T_freeze)': 2,
        'mu_air = 1.458e-6 * safe_pow(T, 1.5) / (T + 110.4)': 3,
        'return jnp.clip(V_s, 0.0, 5.0)': 1,
        'rhof = jnp.sqrt(_RHO_NOT / jnp.clip(rho, 0.1))': 2,
        't1 = 0.86': 1,
        't2 = 0.28 * sc3 * jnp.sqrt(_AV_S)': 1,
        'tc = jnp.clip(T - constants.T_freeze, -55.0, -0.1)': 2,
        'vsc2 = jnp.sqrt(jnp.clip(rho, 0.01) / jnp.clip(mu_air, 1.0e-8))': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py': {
        '(1.2 * umr - 0.95 * umg) ** 2 + 0.08 * umg * umr, 0.5)': 3,
        '(qs_pos >= 1.0e-4) & (qc_pos >= 5.0e-4) & (psacws > 0.0), pgsacw, 0.0)': 2,
        '* (nr_pos / jnp.clip(rho, 0.1)) * x / (lamr3 * lamr3))': 1,
        '* config.rho_snow ** ((-2.0 - bs) / 3.0) / (4.0 * 720.0))': 1,
        '/ (jnp.clip(rho, 0.1) * safe_pow(lams, 2.0 * bs + 2.0)))': 1,
        '/ jnp.clip(rho, 0.1))': 1,
        '2.0 * jnp.pi * n0g_m * jnp.clip(rho, 0.1) * dv': 1,
        'agn = config.fall_a_g * safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.54)': 6,
        'arn = config.fall_a_r * safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.54)': 2,
        'asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.54)': 10,
        'bracket = (5.0 / (safe_pow(lamr, 3.0) * lamg)': 1,
        'cons15 = (1108.0 * config.snow_aggregation_eii': 1,
        'cons34 = 2.5 + config.fall_b_r / 2.0': 1,
        'cons35 = 2.5 + config.fall_b_s / 2.0': 2,
        'cons36 = 2.5 + config.fall_b_g / 2.0': 2,
        'dN_r_au = dq_c_au * rho / (x_star * 20.0)': 1,
        'def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):': 1,
        'def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):': 1,
        'dum = safe_pow(config.rho_su / rho_eff, 0.54)': 1,
        'mnuccr = (20.0 * jnp.pi ** 2 * constants.rho_water * bimm': 1,
        'n0g_m = config.n0_graupel / jnp.clip(rho, 0.1)': 1,
        'rate = evap_coeff * subsaturation * safe_pow(q_r_pos, 0.525)': 1,
        'rho_eff = jnp.clip(rho, 0.1)': 2,
        'umg = jnp.minimum(umg, 20.0 * dum)': 1,
        'umr = jnp.minimum(umr, 9.1 * dum)': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/column.py': {
        '* jnp.exp(-1.5 * jnp.log(config.cloud_geom_std) ** 2)': 1,
        'r, jnp.asarray(1.1), jnp.asarray(9.0e4), jnp.asarray(283.0))': 3,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/diffusional_growth.py': {
        'f = jnp.where(reynolds < 2.5, 1.0 + 0.108 * x * x, 0.78 + 0.308 * x)': 4,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/remap.py': {
        'rrs = jnp.concatenate([masses, masses[-1:] * 1024.0])': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/supersaturation.py': {
        'small = jnp.abs(x) < 1.0e-4': 1,
        'small, dt * dt * (0.5 - x / 6.0 + x * x / 24.0),': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py': {
        'dt: float = 300.0,': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/kessler.py': {
        'accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)': 1,
        'rho_sfc / jnp.clip(rho, 0.1)': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py': {
        'dT_dt = dT_dt * 0.01 # scale raw output': 1,
        'precipitation = jax.nn.softplus(precip_raw[:, -1]) * 1e-3': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py': {
        ') / jnp.clip(rho, 0.1)': 2,
        '* jax.nn.sigmoid(config.nuc_T_sharpness * (265.15 - T))': 1,
        'V_n_g = jnp.minimum(V_n_g, 20.0 * dum_g)': 1,
        'V_n_r = jnp.minimum(V_n_r, 9.1 * dum)': 1,
        'V_n_s = jnp.minimum(V_n_s, 1.2 * dum)': 1,
        'V_t_g = jnp.minimum(V_t_g, 20.0 * dum_g)': 1,
        'V_t_i = jnp.clip(V_t_i, 0.0, 5.0)': 1,
        'V_t_r = jnp.clip(V_t_r, 0.0, 20.0)': 1,
        'V_t_r = jnp.minimum(V_t_r, 9.1 * dum)': 1,
        'V_t_s = jnp.clip(V_t_s, 0.0, 5.0)': 2,
        'V_t_s = jnp.minimum(V_t_s, 1.2 * dum)': 1,
        'dN_s_dt = (dN_i_autoconv + freeze_N_to_snow / jnp.clip(rho, 0.1)': 1,
        'dum = safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.54)': 2,
        'dum_g = safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.54)': 2,
        'dum_i = safe_pow(config.rho_su / jnp.clip(rho, 0.1), 0.35)': 2,
        'dum_i_cap = 1.2 * dum_i': 1,
        'dv_vap = 8.794e-5 * safe_pow(T, 1.81) / jnp.clip(p_full, 1.0)': 2,
        'ferrier = (jnp.clip(q_i, 0.0) / 1080.0) * jnp.exp(-d_rat) \\': 1,
        'freeze_N_to_graupel = (freeze_N_r / jnp.clip(rho, 0.1)': 1,
        'gate_ice = jax.nn.sigmoid(config.nuc_rh_sharpness * (rh_ice - 1.08))': 1,
        'jax.nn.sigmoid(config.nuc_rh_sharpness * (rh_liq - 0.999))': 1,
        'jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), 80.0)': 1,
        'lami_max=1.0 / 10.0e-6,': 1,
        'n_hom_target = config.hom_ice_nuc_N / jnp.clip(rho, 0.1) # per-mass': 1,
        'rho_eff = jnp.maximum(rho, 0.1)': 1,
        'rho_ratio = rho / jnp.clip(rho_sfc, 0.1)': 1,
        'rho_snow=250.0,': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/p3.py': {
        ') / jnp.clip(rho, 0.1)': 1,
        '* jnp.exp(jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), 80.0)),': 1,
        'V_t_r = jnp.clip(V_t_r, 0.0, 20.0)': 1,
        'rho_ratio = rho / jnp.clip(rho_sfc, 0.1)': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/condensation.py': {
        'lambda_v = 2.0 * D / jnp.sqrt(8.0 * T * Rv / jnp.pi)': 2,
        'out = out - 1.5 * gamma * R_inv5': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/kernels.py': {
        '(_VISC_MU0 + _VISC_SLOPE * Tc - _VISC_CURV * Tc * Tc) * 1.0e-5,': 1,
        '(_VISC_MU0 + _VISC_SLOPE * Tc) * 1.0e-5,': 1,
        '- jnp.exp(1.5 * jnp.log(d1 * d1 + lam1 * lam1))) / (3.0 * d1 * lam1) - d1': 1,
        '- jnp.exp(1.5 * jnp.log(d2 * d2 + lam2 * lam2))) / (3.0 * d2 * lam2) - d2': 1,
        'M_air_kg = constants.M_air * 1.0e-3': 1,
        'c1 = 1.2570 + 0.40 * jnp.exp(-0.550 * d1 / lam_air)': 3,
        'c1 = gxdrow / (18.0 * visc)': 1,
        'c2 = 1.2570 + 0.40 * jnp.exp(-0.550 * d2 / lam_air)': 3,
        'c_cloud = 4.5e8 * (r_l * r_l) * (1.0 - 3.0e-6 / jnp.maximum(3.01e-6, r_l))': 3,
        'c_rate = jnp.where(r_l <= 5.0e-5, c_cloud, 1.0)': 1,
        'denom = sumdia / (sumdia + 2.0 * sumg) + (8.0 * sumd) / (sumdia * sumc)': 1,
        'iqq = jnp.clip(jnp.searchsorted(rat, ratio, side="left"), 1, 20)': 1,
        'lam1 = (8.0 / jnp.pi) * D1 / cb1': 1,
        'lam2 = (8.0 / jnp.pi) * D2 / cb2': 1,
        'lam_air = (2.0 * visc) / (p * jnp.sqrt(8.0 * M_air_kg / (jnp.pi * constants.R_d * T)))': 1,
        'vcoef = 8.0 * kB * T / jnp.pi': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/seifert_beheng.py': {
        'V_t_r = jnp.clip(V_t_r, 0.0, 20.0)': 1,
        'jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1), config.b_v_r,': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/thompson.py': {
        ') / jnp.clip(rho, 0.1)': 1,
        'V_t_g = jnp.clip(V_t_g, 0.0, 30.0)': 1,
        'V_t_i = jnp.clip(V_t_i, 0.0, 5.0)': 1,
        'V_t_r = jnp.clip(V_t_r, 0.0, 20.0)': 1,
        'V_t_s = jnp.clip(V_t_s, 0.0, 5.0)': 1,
        'dv_vap = 8.794e-5 * safe_pow(T, 1.81) / jnp.clip(p_full, 1.0)': 2,
        'jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), 80.0)': 1,
        'rho_ratio = rho / jnp.clip(rho_sfc, 0.1)': 1,
    },
}

_SEED_FILES: frozenset[str] = frozenset({
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/_thompson_snow.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/column.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/diffusional_growth.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/remap.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/supersaturation.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/kessler.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/p3.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/condensation.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/kernels.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/seifert_beheng.py',
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/thompson.py',
})
