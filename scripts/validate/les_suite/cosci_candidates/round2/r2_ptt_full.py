"""Round-2 candidate r2_ptt_full: r2_ptt_r plus the radiatively driven top-down branch.

EXPERIMENTAL co-scientist closure (KIND = "both"). FULL VARIANT of the R1-PTT-r
hypothesis: everything of ``r2_ptt_r.py`` (imported, not copied) plus the Lock et al.
(2000) top-down launch from cloud top that the reduced variant left out. It READS the
LES radiative temperature tendency ``inputs.rad_T_tendency`` Q_rad [K/s]
(``READS_RAD = True``; the rollout refuses a case without that channel). The tuned
benchmark schemes do not receive this input.

Top-down branch (surface-first; cell i, mass m_i [kg m-2], heights z_i, thickness dz_i)
-------------------------------------------------------------------------------------
    sigma_i  cloud weight of the reduced variant (logistic in (q_t - q_sat)/q_sat).
    dF_rad = -c_pd sum_k sigma_k m_k Q_rad,k [W m-2]   net radiative flux jump across
             the cloudy layers (PI formula, w_k = sigma_k, VOTED D1).
    z_t    = sum sigma m Qm z / (sum sigma m Qm + tiny),  Qm = Q_s softplus(-Q_rad/Q_s)
             (cooling-weighted cloud-top height, VOTED D2; Q_s VOTED D7).
    h_c    = sqrt((sum sigma dz)^2 + h_min^2) (cloud depth, parent's h floor, VOTED D4).
    L^t_i  = exp(-r_i/(0.1 h_c)) * 0.1 h_c / sum_k dz_k exp(-r_k/(0.1 h_c)),
             r_i = sqrt((z_t - z_i)^2 + (0.01 h_c)^2) (two-sided, |.| regularised;
             VOTED D3).
    rho_t, theta_v,t = launch-weighted means sum dz L^t (.) / (0.1 h_c) (VOTED D5).
    X = g h_c dF_rad / (rho_t c_pd theta_v,t),  w_t^3 = v_f^3 P_d(X/v_f^3) (zero when
             the deck is not net cooled), w_t = a - v_f with a = (w_t^3 + v_f^3)^(1/3),
             evaluated as w_t^3/(a^2 + a v_f + v_f^2) (smooth at w_t = 0).
    Downward transfer i -> j < i [kg m-2 s-1]:
             M^t_ij = a_N (w_t/h_c) m_i L^t_i p^t_ij,
             p^t_ij = m_j exp(-d_ij/h_c) exp(-BAR_ij/V_t^2) / sum_{j'<i} m_j' exp(-d_ij'/h_c),
             BAR_ij = sum_{j<=k<i} dz_k b_s softplus(b_ik/b_s), b_ik the buoyancy of cell
             i's parcel brought reversibly to level k (positive buoyancy resists the
             descent), V_t^2 = w_t^2 + v_f^2 (VOTED D6).
    Return: compensating ascent, one interface at a time, upwind (value of the cell
             BELOW the interface): G^T_K = -sum_{j<K<=i} M^t_ij X_i + U^t_K X_{K-1},
             U^t_K = sum_{j<K<=i} M^t_ij. Net mass flux through every interface is zero,
             a uniform X gives G^T = 0, and the operator is Metzler (q_t >= 0 kept).
    The same operator is added to the exchange of theta_l (enthalpy form), q_t, u, v,
    inside the reduced variant's expm step (one hook, ``b_extra``).
Energy: KE removed by the momentum fluxes is returned as heat by the harness (as the
reduced variant). Radiation itself is applied by the SCM forcing, never here; Q_rad only
sets the branch's strength and launch height.
Control: Q_rad = 0 (or a net-heated deck) gives w_t = 0 exactly, so M^t = 0 and the
fluxes equal r2_ptt_r's bit for bit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm import constants

_spec = importlib.util.spec_from_file_location(
    "_r2_ptt_full_base", Path(__file__).resolve().parent / "r2_ptt_r.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

KIND = "both"
READS_RAD = True
PARAMS = dict(_base.PARAMS)
init_aux = _base.init_aux

_Q_SCALE = 1.0e-6      # [K/s] Q_s, smooth positive part of the radiative cooling (D7)
_R_REG_FRAC = 0.01     # [1] regularisation of |z_t - z| as a fraction of h_c (D3)
_TINY = 1.0e-30        # safety floor of the z_t weight sum (clear column)


def top_down(params, inputs):
    """M^t (n, n; strictly lower: source i -> target j < i) and diagnostics."""
    g, cpd = constants.g, constants.c_pd
    zf, zh, m = inputs.z_full, inputs.z_half, inputs.mass
    n = zf.shape[0]
    dz = zh[1:] - zh[:-1]
    sig = _base._cloud_weight(inputs)
    q_rad = inputs.rad_T_tendency
    d_f = -cpd * jnp.sum(sig * m * q_rad)
    qm = _Q_SCALE * jax.nn.softplus(-q_rad / _Q_SCALE)
    wz = sig * m * qm
    z_t = jnp.sum(wz * zf) / (jnp.sum(wz) + _TINY)
    h_c = jnp.sqrt(jnp.sum(sig * dz) ** 2 + _base._PBL.h_min ** 2)
    depth = _base._LAUNCH_FRAC * h_c
    r = jnp.sqrt((z_t - zf) ** 2 + (_R_REG_FRAC * h_c) ** 2)
    e_l = jnp.exp(-r / depth)
    launch = e_l * depth / jnp.sum(dz * e_l)
    wgt = dz * launch / depth
    rho_t = jnp.sum(wgt * inputs.rho_full)
    theta = inputs.temp / inputs.exner
    thv_t = jnp.sum(wgt * theta * (1.0 + _base._EPS_V * inputs.q_v - inputs.q_c))
    vf = _base._V_FLOOR
    x = g * h_c * d_f / (rho_t * cpd * thv_t)
    wt3 = vf ** 3 * _base._p_delta(x / vf ** 3)
    a3 = (wt3 + vf ** 3) ** (1.0 / 3.0)
    w_t = wt3 / (a3 ** 2 + a3 * vf + vf ** 2)
    vt2 = w_t ** 2 + vf ** 2

    b_par = _base._parcels(inputs)                          # [i, k]
    low = jnp.tril(jnp.ones((n, n), zf.dtype), -1)          # k < i (and j < i)
    pos = low * dz[None, :] * _base._B_SCALE * jax.nn.softplus(b_par / _base._B_SCALE)
    bar = jnp.flip(jnp.cumsum(jnp.flip(pos, 1), 1), 1)      # sum_{k=j}^{i-1}
    d = jnp.abs(zf[None, :] - zf[:, None])
    w_d = jnp.exp(-d / h_c)
    norm = jnp.sum(low * m[None, :] * w_d, axis=1)
    norm = jnp.where(norm > 0.0, norm, 1.0)                 # bottom cell launches nothing
    big_t = (params["a_N"] * (w_t / h_c) * low * (m * launch / norm)[:, None]
             * m[None, :] * w_d * jnp.exp(-bar / vt2))
    return dict(big_t=big_t, d_f=d_f, z_t=z_t, h_c=h_c, w_t=w_t, launch=launch)


def top_down_matrix(big_t):
    """B_t (n+1, n): (B_t X)_K = -sum_{j<K<=i} M^t_ij X_i + U^t_K X_{K-1} (+up),
    the bottom-up operator of the mirrored column with the flux sign reversed."""
    mirrored = _base._nonlocal_matrix(jnp.flip(big_t, (0, 1)))
    return -jnp.flip(mirrored, (0, 1))


def fluxes(params, inputs, aux):
    return _base.fluxes(params, inputs, aux,
                        b_extra=top_down_matrix(top_down(params, inputs)["big_t"]))


def implied_surface_fluxes(params, inputs, aux):
    return _base.implied_surface_fluxes(
        params, inputs, aux, b_extra=top_down_matrix(top_down(params, inputs)["big_t"]))
