"""Tests for the #1515 column-replay probe + the detonation-column regression.

Two parts:

1. Instrument tests for ``scripts/validate/replay_column_microphysics.py``
   (every new script gets a direct test): config resolution through the
   driver's own applier, a healthy synthetic column producing bounded
   physical tendencies, NaN counting, and the unknown-scheme refusal.

2. The issue-#1515 regression: the REAL pre-detonation profile of MPAS cell
   1432 (fine_r1 arm, checkpoint day 967 — 14.4 h before the day-967.64
   blowup; provenance:
   /scratch/b/b309178/amip_blowup970/arms/fine_r1/checkpoint_day_0967.npz,
   run tree ff3bbdc1d).  The column carries the chronic orphan-N_r
   pathology (N_r up to 1.41e19 /m^3 at q_r ~ 0 — 10+ orders above the PSD
   ceiling) that fed the stochastic single-column moist detonation.  After
   one Morrison step with the caller's state-update semantics
   ``N_r_new = max(N_r + dt * dN_r_dt, 0)`` (model_driver.
   _apply_double_moment_tendencies / the MPAS tracer update), the post-step
   rain number must respect the SAM-style PSD consistency ceiling
   ``N_r_hi = lamr_max^3 * rho * q_r_new / (pi * rho_w)`` and orphan number
   (q_r_new < 1e-14) must be CLEARED.  This is red on reverting the
   #1476 N_r consistency ceiling (verified against the pre-ceiling tree
   ff3bbdc1d: post-step N_r stays at 1.4e19, 10 orders over the bound).

Run with JAX_ENABLE_X64=1 (numerics test).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from scripts.validate.replay_column_microphysics import (
    build_morrison_config,
    replay_cells,
)

# --- Real state: MPAS cell 1432, fine_r1 checkpoint day 967 (issue #1515) --
# Extracted verbatim (fp64) from the checkpoint named in the module
# docstring.  This is STATE DATA (a saved trajectory sample), not physical
# constants.
P_S_1432 = 100059.11632504975
SIGMA_HALF = np.array([
    0.010, 0.043, 0.076, 0.109, 0.142, 0.175, 0.208, 0.241, 0.274, 0.307,
    0.340, 0.373, 0.406, 0.439, 0.472, 0.505, 0.538, 0.571, 0.604, 0.637,
    0.670, 0.703, 0.736, 0.769, 0.802, 0.835, 0.868, 0.901, 0.934, 0.967,
    1.000])
T_1432 = np.array([
    1.995579e+02, 2.035569e+02, 2.055561e+02, 2.076130e+02, 2.121713e+02,
    2.185667e+02, 2.267024e+02, 2.345421e+02, 2.413890e+02, 2.469045e+02,
    2.522670e+02, 2.566083e+02, 2.605156e+02, 2.640499e+02, 2.672312e+02,
    2.701049e+02, 2.726953e+02, 2.750033e+02, 2.771953e+02, 2.792174e+02,
    2.826465e+02, 2.846799e+02, 2.864302e+02, 2.880608e+02, 2.896187e+02,
    2.911651e+02, 2.927935e+02, 2.945185e+02, 2.962095e+02, 2.984694e+02])
Q_V_1432 = np.array([
    3.598153e-06, 9.630069e-06, 1.613053e-05, 2.280403e-05, 3.737521e-05,
    7.242206e-05, 1.700437e-04, 3.770558e-04, 7.439652e-04, 1.380435e-03,
    2.034882e-03, 2.701109e-03, 3.436412e-03, 4.228189e-03, 5.050913e-03,
    5.887861e-03, 6.716863e-03, 7.507839e-03, 8.341411e-03, 9.181160e-03,
    1.097239e-02, 1.200743e-02, 1.289539e-02, 1.374367e-02, 1.458579e-02,
    1.547624e-02, 1.650970e-02, 1.775631e-02, 1.909793e-02, 2.109630e-02])
Q_C_1432 = np.array([
    1.124534e-17, 1.144645e-16, 0.0, 0.0, 5.919628e-12, 5.703923e-09,
    1.866016e-09, 0.0, 3.014529e-09, 1.025537e-04, 2.038800e-04,
    1.907387e-04, 1.676715e-04, 1.613978e-04, 1.900800e-04, 1.964952e-04,
    2.203430e-04, 1.196071e-04, 4.729950e-05, 3.308923e-05, 1.620639e-05,
    1.504545e-05, 1.445355e-05, 1.297916e-05, 1.118986e-05, 9.437512e-06,
    6.807159e-06, 8.640540e-06, 3.049354e-05, 5.143506e-04])
Q_R_1432 = np.array([
    3.252012e-15, 0.0, 0.0, 0.0, 1.075402e-09, 1.100910e-09, 1.195475e-07,
    2.953522e-07, 5.185156e-07, 8.359933e-07, 1.205870e-06, 3.750447e-07,
    4.188188e-07, 5.239662e-07, 4.141770e-07, 4.377339e-07, 6.664402e-07,
    2.525351e-04, 1.010968e-03, 2.133822e-03, 2.407517e-03, 2.656822e-03,
    2.880548e-03, 3.070171e-03, 3.223926e-03, 3.343730e-03, 3.428945e-03,
    3.511607e-03, 3.523315e-03, 3.493340e-03])
Q_I_1432 = np.array([
    1.324875e-19, 0.0, 0.0, 0.0, 3.903548e-06, 2.590508e-05, 5.153314e-05,
    4.135324e-05, 4.753281e-05, 2.456386e-05, 6.379931e-06, 1.652614e-06,
    4.504555e-07, 1.265368e-07, 1.426936e-08, 1.445629e-09, 2.625000e-10,
    9.511103e-11, 3.473049e-11, 1.291293e-11, 4.865955e-12, 1.851997e-12,
    7.138162e-13, 2.786443e-13, 1.101458e-13, 4.408015e-14, 1.786130e-14,
    1.011520e-14, 2.921627e-15, 4.523857e-17])
Q_S_1432 = np.array([
    6.949229e-11, 7.921844e-10, 0.0, 0.0, 7.036982e-08, 1.735114e-06,
    1.317561e-05, 4.196605e-05, 9.019885e-05, 1.303929e-04, 1.425537e-04,
    1.449514e-04, 1.442697e-04, 1.426821e-04, 1.356833e-04, 1.293907e-04,
    1.221792e-04, 5.571568e-05, 2.267565e-05, 8.065733e-06, 2.404525e-06,
    5.607121e-07, 9.457738e-08, 1.031478e-08, 6.206184e-10, 1.559363e-11,
    0.0, 0.0, 9.893811e-17, 2.954009e-17])
Q_G_1432 = np.array([
    2.191997e-15, 0.0, 0.0, 2.214075e-13, 1.815136e-10, 0.0, 0.0,
    8.330223e-08, 1.315960e-06, 1.194443e-05, 4.035444e-05, 7.073372e-05,
    1.025238e-04, 1.343000e-04, 1.671584e-04, 1.980757e-04, 2.329803e-04,
    3.533397e-04, 1.164140e-03, 1.552526e-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    2.454947e-15, 2.535737e-15, 2.415894e-15, 2.385201e-15])
N_R_1432 = np.array([
    3.826731e+18, 1.406542e+19, 3.016300e+18, 2.994041e+15, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.286807e+14, 2.822501e+14, 3.599103e+14, 3.597175e+14,
    3.108749e+14, 2.389153e+14, 1.523016e+14, 3.639116e+13, 3.249067e+12,
    1.340487e+11, 3.452025e+10, 7.988924e+09, 2.526356e+09, 0.0, 0.0, 0.0,
    1.441101e+09, 1.777633e+09, 1.680991e+08, 0.0])
N_I_1432 = np.array([
    0.0, 0.0, 0.0, 0.0, 1.887333e+06, 1.946271e+06, 1.445679e+06,
    3.800414e+05, 1.859163e+05, 3.129423e+04, 5.600826e+03, 1.378900e+03,
    3.944440e+02, 1.280885e+02, 6.083044e+00, 8.201295e-02, 0.0,
    2.803221e-04, 1.023616e-04, 3.805846e-05, 1.434150e-05, 5.458416e-06,
    2.103840e-06, 8.212523e-07, 3.246342e-07, 1.299180e-07, 5.264285e-08,
    2.981267e-08, 0.0, 0.0])

DT = 75.0

# The fine_r1 arm's resolved_config subset the probe consumes (from its
# run_manifest.json).
RESOLVED_1432 = {
    "microphysics": "morrison",
    "morrison_flavor": "mg",
    "morrison_agg_coeff": 0.001,
    "morrison_bergeron_rate": 0.001,
    "morrison_dep_coeff": 0.001,
    "morrison_fall_a_i": 700.0,
    "morrison_hom_ice_nuc_N": 1000000.0,
    "morrison_ice_snow_d_auto": 0.00025,
    "morrison_k_au": 600.0,
    "morrison_rime_coeff": 1.0,
    "hard_saturation_adjustment": True,
    "hard_sat_adjust_threshold": None,
    "hard_sat_max_heating_K": None,
    "nc_from_aerosol": False,
    "subgrid_autoconversion": False,
    "dycore": {"dt": DT},
}


def _state_1432():
    """The probe's npz-like mapping for the single-cell state."""
    nlev = T_1432.shape[0]
    zeros = np.zeros((1, nlev))
    return {
        "meta_vgrid": np.stack([np.zeros_like(SIGMA_HALF), SIGMA_HALF]),
        "p_s": np.array([P_S_1432]),
        "T": T_1432[None, :],
        "trc_q_v": Q_V_1432[None, :],
        "trc_q_c": Q_C_1432[None, :],
        "trc_q_r": Q_R_1432[None, :],
        "trc_q_i": Q_I_1432[None, :],
        "trc_q_s": Q_S_1432[None, :],
        "trc_q_g": Q_G_1432[None, :],
        "trc_N_c": zeros,
        "trc_N_r": N_R_1432[None, :],
        "trc_N_i": N_I_1432[None, :],
    }


def test_build_morrison_config_resolves_run_flags():
    cfg = build_morrison_config(RESOLVED_1432)
    assert cfg.morrison_flavor == "mg"
    assert cfg.hard_saturation_adjustment is True
    assert cfg.k_au == pytest.approx(600.0)
    assert cfg.fall_a_i == pytest.approx(700.0)


def test_build_morrison_config_refuses_other_scheme():
    with pytest.raises(SystemExit):
        build_morrison_config({"microphysics": "thompson"})


def test_replay_refuses_hybrid_vgrid():
    state = _state_1432()
    state["meta_vgrid"] = np.stack(
        [np.full_like(SIGMA_HALF, 10.0), SIGMA_HALF])
    cfg = build_morrison_config(RESOLVED_1432)
    with pytest.raises(SystemExit):
        replay_cells(state, [0], cfg, DT)


def test_replay_healthy_column_bounded_and_finite():
    """Instrument self-check: a benign warm column gives physical, finite
    tendencies (|dT| per step well under 1 K, no drain)."""
    nlev = 30
    sigma_full = 0.5 * (SIGMA_HALF[:-1] + SIGMA_HALF[1:])
    T = 300.0 - 70.0 * (1.0 - sigma_full)          # ~230-298 K
    q_v = 0.016 * sigma_full ** 2                   # subsaturated
    state = _state_1432()
    state["T"] = T[None, :]
    state["trc_q_v"] = q_v[None, :]
    for k in ("q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        state[f"trc_{k}"] = np.zeros((1, nlev))
    cfg = build_morrison_config(RESOLVED_1432)
    res = replay_cells(state, [0], cfg, DT)
    assert res["nonfinite"] == []
    assert res["max_abs_dT_dt_x_dt"] < 1.0
    assert res["drain_max_dq_g_kg"] == 0.0


def test_replay_counts_nonfinite_inputs():
    state = _state_1432()
    state["T"] = state["T"].copy()
    state["T"][0, 5] = np.nan
    cfg = build_morrison_config(RESOLVED_1432)
    res = replay_cells(state, [0], cfg, DT)
    assert any(e["field"] == "T" for e in res["nonfinite"])


def test_issue1515_poisoned_nr_repaired_in_one_step():
    """#1515 regression: the real pre-detonation cell-1432 column.

    With the caller's state-update semantics, one Morrison step must leave
    the rain number PSD-consistent: N_r_new <= lamr_max^3*rho*q_r_new/(pi*
    rho_w) wherever rain exists, and CLEARED where q_r_new < 1e-14 (orphan
    number).  The input column is 10+ orders of magnitude over that bound
    (max N_r = 1.41e19 /m^3 at q_r ~ 0) — the immortal-orphan fuel of the
    stochastic single-column detonation.  Red without the #1476 N_r
    consistency ceiling (verified on tree ff3bbdc1d: max post-step
    N_r/ceiling ~ 1e10).
    """
    from legoesm.atmosphere.physics._shared import compute_layer_dz, compute_rho
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
        resolve_morrison_flavor,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    cfg = resolve_morrison_flavor(build_morrison_config(RESOLVED_1432))
    sigma_full = 0.5 * (SIGMA_HALF[:-1] + SIGMA_HALF[1:])
    p_full = jnp.asarray(sigma_full * P_S_1432)[None, :]
    p_half = jnp.asarray(SIGMA_HALF * P_S_1432)[None, :]
    T = jnp.asarray(T_1432)[None, :]
    q_v = jnp.asarray(Q_V_1432)[None, :]
    rho = compute_rho(T, p_full, q_v)
    dz = compute_layer_dz(T, p_half, q_v)
    hyd = HydrometeorState(
        q_c=jnp.asarray(Q_C_1432)[None, :],
        q_r=jnp.asarray(Q_R_1432)[None, :],
        q_i=jnp.asarray(Q_I_1432)[None, :],
        q_s=jnp.asarray(Q_S_1432)[None, :],
        q_g=jnp.asarray(Q_G_1432)[None, :],
        N_c=jnp.zeros_like(q_v),
        N_r=jnp.asarray(N_R_1432)[None, :],
        N_i=jnp.asarray(N_I_1432)[None, :],
    )
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, DT, cfg)

    for f in out._fields:
        v = getattr(out, f)
        if v is not None:
            assert bool(jnp.all(jnp.isfinite(v))), f"non-finite {f}"

    # Caller semantics (model_driver._apply_double_moment_tendencies and the
    # MPAS tracer update): x_new = max(x + dt * dx_dt, 0).
    q_r_new = np.maximum(Q_R_1432 + DT * np.asarray(out.dq_r_dt)[0], 0.0)
    n_r_new = np.maximum(N_R_1432 + DT * np.asarray(out.dN_r_dt)[0], 0.0)
    rho_np = np.asarray(rho)[0]
    ceiling = (cfg.lamr_max ** 3 * rho_np * q_r_new
               / (math.pi * constants.rho_water))

    # The input really is pathological — otherwise this test is vacuous.
    ceiling_in = (cfg.lamr_max ** 3 * rho_np * np.maximum(Q_R_1432, 0.0)
                  / (math.pi * constants.rho_water))
    assert np.max(N_R_1432 - ceiling_in) > 1.0e18

    orphan = q_r_new < 1.0e-14
    np.testing.assert_array_equal(n_r_new[orphan], 0.0)
    ok = n_r_new[~orphan] <= ceiling[~orphan] * (1.0 + 1.0e-9)
    assert bool(np.all(ok)), (
        f"post-step N_r over PSD ceiling by up to "
        f"{np.max(n_r_new[~orphan] / np.maximum(ceiling[~orphan], 1e-300)):.3g}x"
    )
