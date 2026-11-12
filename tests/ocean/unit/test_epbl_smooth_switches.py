"""RULE 5 smooth switches of the ePBL port (issue #2180).

MOM6 switches on traced state that the MOM6 single-column fixtures
exercises are smooth by default (``EPBLConfig.smooth_*_sharpness*``;
``math.inf`` = MOM6's exact switch):

- buoyancy flux [s^3/m^2]: ``max(0, B)**2`` in the REICHL_H18 mstar and
  ``min(0, B)`` in the MSTAR_CONV_ADJ reduction (``_find_mstar``);
- ratio [1]: the sigma_max clip of the equation-discovery shape
  (``_eqdisc_shape``);
- column height [1/m]: ``min(ColHt_chg, 0)`` in ``_find_pe_chg_orig``;
- energy [m^2/J]: the top-layer ``conv_PErel = max(TKE_forced, 0)`` and the
  ``mech_TKE`` spend/max at the same kink (``_top_layer_energy``).

The exact limit of each is the MOM6 oracle test
(``test_epbl_mom6_oracle.py`` pins every sharpness to ``math.inf``).  Here:

1. the smooth-default gap to MOM6, per fixture case, stored for the run with
   every switch smooth and for each unit class smooth alone (the others
   exact), to 1e-6 relative with a 1e-12 degC floor.  Reverting any one of
   the smoothed sites (one site, the others still smooth) changes at least
   one stored gap of the all-smooth run, except the mech_TKE
   spend (C6b/c), which is value-blind in the fixture and is caught by the
   top-layer gradient test (per-site measurements in the PR for #2180);
2. ``jax.test_util.check_grads`` at the default, on each kernel exactly at its
   kink (order 2, forward and reverse), and on the full column (order 1).
   The inputs are scaled by the default width so that the finite-difference
   step resolves the smoothed kink.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing.config import EPBLConfig
from legoesm.ocean.physics.vertical_mixing.epbl import (
    _EQD_ML_C,
    _eqdisc_shape,
    _eqdisc_v0,
    _eqdisc_v0h,
    _find_mstar,
    _heat_cool,
    _interior_spend_scale,
    _linear_shape,
    _mixing_length,
    _rh18_surface_scale,
    _tke_gate,
    _find_pe_chg_orig,
    _top_layer_energy,
    epbl_vertical_mixing,
    surface_tke_forcing,
    validate_epbl_config,
)
from tests.ocean.unit.test_epbl import (
    C_SW, G, OMEGA, RHO0, _b_flux, _column, _ref_cfg)
from tests.ocean.unit.test_epbl_mom6_oracle import (
    EXACT_SWITCHES, FIXTURE, FIXTURE_VARIANTS, MEASURED_GAP,
    MEASURED_GAP_VARIANTS, VARIANTS, _Q, _fp64, _legoesm_T_8h)

_DEFAULT = {f: EPBLConfig._field_defaults[f] for f in EXACT_SWITCHES}
_CLASSES = {
    "bflux": "smooth_bflux_sharpness_s3m2",
    "ratio": "smooth_ratio_sharpness",
    "colht": "smooth_colht_sharpness_per_m",
    "energy": "smooth_energy_sharpness_m2j",
    "velocity": "smooth_velocity_sharpness_s_m",
    "slope": "smooth_slope_sharpness",
}
RUNS = {"all": dict(_DEFAULT)}
RUNS.update({k: {**EXACT_SWITCHES, f: _DEFAULT[f]} for k, f in _CLASSES.items()})

# Measured max |T_legoESM - T_MOM6| at 8 h [degC] at the smooth defaults, fp64
# policy, per (run, variant, case, dt); variant "" is the base fixture, the
# others are VARIANTS of test_epbl_mom6_oracle.py (SLURM jobs 28844616-19,
# 28852879, 28857845-46; JAX_ENABLE_X64=1).  The exact values are MEASURED_GAP and
# MEASURED_GAP_VARIANTS in test_epbl_mom6_oracle.py.
SMOOTH_GAP = {
    ('all', '', 'q-200', 300): 4.698218347698457e-06,
    ('all', '', 'q-200', 1200): 0.0002472713762102785,
    ('all', '', 'q0', 300): 0.003989479135373841,
    ('all', '', 'q0', 1200): 0.003486197875453456,
    ('all', '', 'q200', 300): 0.0001336320234841537,
    ('all', '', 'q200', 1200): 0.0011014508953159918,
    ('all', 'lin', 'q-200', 300): 5.619513039789581e-06,
    ('all', 'lin', 'q-200', 1200): 0.00025366102212842634,
    ('all', 'lin', 'q0', 300): 0.001579137084242177,
    ('all', 'lin', 'q0', 1200): 0.00589485711904203,
    ('all', 'lin', 'q200', 300): 0.0003226347932319129,
    ('all', 'lin', 'q200', 1200): 0.0017144126286687822,
    ('all', 'om4', 'q-200', 300): 1.139794648707948e-05,
    ('all', 'om4', 'q-200', 1200): 0.00025614154640152265,
    ('all', 'om4', 'q0', 300): 0.00041343213484523744,
    ('all', 'om4', 'q0', 1200): 0.00038436011876896714,
    ('all', 'om4', 'q200', 300): 0.00026507809769071855,
    ('all', 'om4', 'q200', 1200): 0.001309759014844758,
    ('all', 'v0', 'q-200', 300): 1.3980830768645092e-05,
    ('all', 'v0', 'q-200', 1200): 0.000324998717317726,
    ('all', 'v0', 'q0', 300): 0.003980090479906551,
    ('all', 'v0', 'q0', 1200): 0.003518964540933922,
    ('all', 'v0', 'q200', 300): 0.0001724458240701665,
    ('all', 'v0', 'q200', 1200): 0.0013581699086842036,
    ('bflux', '', 'q-200', 300): 2.5869512185749954e-05,
    ('bflux', '', 'q-200', 1200): 0.0002513095684157918,
    ('bflux', '', 'q0', 300): 0.004030932365150619,
    ('bflux', '', 'q0', 1200): 0.0034871684223567456,
    ('bflux', '', 'q200', 300): 0.00013344868046871738,
    ('bflux', '', 'q200', 1200): 0.0011014506582398553,
    ('bflux', 'lin', 'q-200', 300): 2.6455125762936404e-05,
    ('bflux', 'lin', 'q-200', 1200): 0.00025758760738270325,
    ('bflux', 'lin', 'q0', 300): 0.002763478138557929,
    ('bflux', 'lin', 'q0', 1200): 0.005289735922303862,
    ('bflux', 'lin', 'q200', 300): 0.00032260035419895416,
    ('bflux', 'lin', 'q200', 1200): 0.0017144116551328636,
    ('bflux', 'om4', 'q-200', 300): 8.64962640179101e-06,
    ('bflux', 'om4', 'q-200', 1200): 0.0002603887351035894,
    ('bflux', 'om4', 'q0', 300): 0.000378039419953069,
    ('bflux', 'om4', 'q0', 1200): 0.0003775882737180325,
    ('bflux', 'om4', 'q200', 300): 0.0002650435506161841,
    ('bflux', 'om4', 'q200', 1200): 0.0013097581624137433,
    ('bflux', 'v0', 'q-200', 300): 1.6560177403590615e-05,
    ('bflux', 'v0', 'q-200', 1200): 0.0003285928491933987,
    ('bflux', 'v0', 'q0', 300): 0.004013810913445326,
    ('bflux', 'v0', 'q0', 1200): 0.0035199660060349913,
    ('bflux', 'v0', 'q200', 300): 0.00017245949268840377,
    ('bflux', 'v0', 'q200', 1200): 0.0013581699095901456,
    ('colht', '', 'q-200', 300): 4.6967527609353965e-06,
    ('colht', '', 'q-200', 1200): 0.0002472713754357869,
    ('colht', '', 'q0', 300): 2.1695459292203623e-05,
    ('colht', '', 'q0', 1200): 6.8694184314210816e-06,
    ('colht', '', 'q200', 300): 0.00013363197895088774,
    ('colht', '', 'q200', 1200): 0.0011014509289672958,
    ('colht', 'lin', 'q-200', 300): 5.619512510435243e-06,
    ('colht', 'lin', 'q-200', 1200): 0.00025366102236290544,
    ('colht', 'lin', 'q0', 300): 1.755838814432309e-05,
    ('colht', 'lin', 'q0', 1200): 3.590922421636833e-06,
    ('colht', 'lin', 'q200', 300): 0.00032263479368666026,
    ('colht', 'lin', 'q200', 1200): 0.0017144126285195682,
    ('colht', 'om4', 'q-200', 300): 1.13979462845748e-05,
    ('colht', 'om4', 'q-200', 1200): 0.00025614154613862183,
    ('colht', 'om4', 'q0', 300): 3.551239190002775e-05,
    ('colht', 'om4', 'q0', 1200): 6.788208665398088e-06,
    ('colht', 'om4', 'q200', 300): 0.0002650780964295052,
    ('colht', 'om4', 'q200', 1200): 0.0013097590153101635,
    ('colht', 'v0', 'q-200', 300): 1.3980894834730861e-05,
    ('colht', 'v0', 'q-200', 1200): 0.0003249987168949531,
    ('colht', 'v0', 'q0', 300): 2.1699225474236528e-05,
    ('colht', 'v0', 'q0', 1200): 6.872178520467287e-06,
    ('colht', 'v0', 'q200', 300): 0.00017244582948094944,
    ('colht', 'v0', 'q200', 1200): 0.001358169909348561,
    ('energy', '', 'q-200', 300): 2.586951155336692e-05,
    ('energy', '', 'q-200', 1200): 0.00025130956863961273,
    ('energy', '', 'q0', 300): 8.575283416689672e-08,
    ('energy', '', 'q0', 1200): 1.9286016339492562e-08,
    ('energy', '', 'q200', 300): 0.00013344868033726698,
    ('energy', '', 'q200', 1200): 0.0011014506582256445,
    ('energy', 'lin', 'q-200', 300): 2.6455126178603905e-05,
    ('energy', 'lin', 'q-200', 1200): 0.0002575876070771699,
    ('energy', 'lin', 'q0', 300): 0.001403001496907308,
    ('energy', 'lin', 'q0', 1200): 0.000549488277261645,
    ('energy', 'lin', 'q200', 300): 0.00032260035393960607,
    ('energy', 'lin', 'q200', 1200): 0.00171441165515418,
    ('energy', 'om4', 'q-200', 300): 8.649626522583276e-06,
    ('energy', 'om4', 'q-200', 1200): 0.0002603887342935707,
    ('energy', 'om4', 'q0', 300): 1.178125970113797e-07,
    ('energy', 'om4', 'q0', 1200): 1.6549254411302172e-08,
    ('energy', 'om4', 'q200', 300): 0.0002650435506090787,
    ('energy', 'om4', 'q200', 1200): 0.0013097581627938837,
    ('energy', 'v0', 'q-200', 300): 1.6560177261482067e-05,
    ('energy', 'v0', 'q-200', 1200): 0.0003285928495095902,
    ('energy', 'v0', 'q0', 300): 8.573519849619515e-08,
    ('energy', 'v0', 'q0', 1200): 1.9284208008230053e-08,
    ('energy', 'v0', 'q200', 300): 0.0001724594925889278,
    ('energy', 'v0', 'q200', 1200): 0.0013581699095297495,
    ('ratio', '', 'q-200', 300): 2.586804790993824e-05,
    ('ratio', '', 'q-200', 1200): 0.00025130956916186165,
    ('ratio', '', 'q0', 300): 1.1479883710308059e-10,
    ('ratio', '', 'q0', 1200): 7.958078640513122e-13,
    ('ratio', '', 'q200', 300): 0.00013344872069964708,
    ('ratio', '', 'q200', 1200): 0.001101450624108935,
    ('ratio', 'lin', 'q-200', 300): 2.6455125812674396e-05,
    ('ratio', 'lin', 'q-200', 1200): 0.0002575876078338979,
    ('ratio', 'lin', 'q0', 300): 4.902744876744691e-13,
    ('ratio', 'lin', 'q0', 1200): 2.3447910280083306e-13,
    ('ratio', 'lin', 'q200', 300): 0.0003226003528347121,
    ('ratio', 'lin', 'q200', 1200): 0.0017144116548450938,
    ('ratio', 'om4', 'q-200', 300): 8.649626241918895e-06,
    ('ratio', 'om4', 'q-200', 1200): 0.0002603887346630529,
    ('ratio', 'om4', 'q0', 300): 7.384528544207569e-10,
    ('ratio', 'om4', 'q0', 1200): 4.618527782440651e-14,
    ('ratio', 'om4', 'q200', 300): 0.0002650435517743688,
    ('ratio', 'om4', 'q200', 1200): 0.0013097581617671494,
    ('ratio', 'v0', 'q-200', 300): 1.6560113223818007e-05,
    ('ratio', 'v0', 'q-200', 1200): 0.000328592849015763,
    ('ratio', 'v0', 'q0', 300): 1.1595346904869075e-10,
    ('ratio', 'v0', 'q0', 1200): 4.014566457044566e-13,
    ('ratio', 'v0', 'q200', 300): 0.00017245948730604255,
    ('ratio', 'v0', 'q200', 1200): 0.0013581699091602673,
    ('slope', '', 'q-200', 300): 2.5869512786158566e-05,
    ('slope', '', 'q-200', 1200): 0.0002513095687390887,
    ('slope', '', 'q0', 300): 1.2505552149377763e-12,
    ('slope', '', 'q0', 1200): 1.6342482922482304e-13,
    ('slope', '', 'q200', 300): 0.00013344867987186149,
    ('slope', '', 'q200', 1200): 0.0011014506591884299,
    ('slope', 'lin', 'q-200', 300): 2.6455125620827857e-05,
    ('slope', 'lin', 'q-200', 1200): 0.00025758760811456227,
    ('slope', 'lin', 'q0', 300): 5.897504706808832e-13,
    ('slope', 'lin', 'q0', 1200): 2.2737367544323206e-13,
    ('slope', 'lin', 'q200', 300): 0.00032260035401421305,
    ('slope', 'lin', 'q200', 1200): 0.001714411655054704,
    ('slope', 'om4', 'q-200', 300): 8.649626202839045e-06,
    ('slope', 'om4', 'q-200', 1200): 0.000260388734798056,
    ('slope', 'om4', 'q0', 300): 1.1972645097557688e-12,
    ('slope', 'om4', 'q0', 1200): 4.618527782440651e-14,
    ('slope', 'om4', 'q200', 300): 0.0002650435502538073,
    ('slope', 'om4', 'q200', 1200): 0.0013097581623213728,
    ('slope', 'v0', 'q-200', 300): 1.656017894546835e-05,
    ('slope', 'v0', 'q-200', 1200): 0.0003285928493710344,
    ('slope', 'v0', 'q0', 300): 4.902744876744691e-13,
    ('slope', 'v0', 'q0', 1200): 2.2382096176443156e-13,
    ('slope', 'v0', 'q200', 300): 0.00017245949263156035,
    ('slope', 'v0', 'q200', 1200): 0.0013581699093769828,
    ('velocity', '', 'q-200', 300): 2.5869512619181023e-05,
    ('velocity', '', 'q-200', 1200): 0.00025130956890606626,
    ('velocity', '', 'q0', 300): 1.0196288258157438e-12,
    ('velocity', '', 'q0', 1200): 2.5934809855243657e-13,
    ('velocity', '', 'q200', 300): 0.00013344868057174608,
    ('velocity', '', 'q200', 1200): 0.0011014506586555228,
    ('velocity', 'lin', 'q-200', 300): 2.6455125510693733e-05,
    ('velocity', 'lin', 'q-200', 1200): 0.00025758760738270325,
    ('velocity', 'lin', 'q0', 300): 5.471179065352771e-13,
    ('velocity', 'lin', 'q0', 1200): 2.4158453015843406e-13,
    ('velocity', 'lin', 'q200', 300): 0.0003226003538401301,
    ('velocity', 'lin', 'q200', 1200): 0.001714411655004966,
    ('velocity', 'om4', 'q-200', 300): 8.649626259682464e-06,
    ('velocity', 'om4', 'q-200', 1200): 0.00026038873468081647,
    ('velocity', 'om4', 'q0', 300): 1.2043699371133698e-12,
    ('velocity', 'om4', 'q0', 1200): 4.618527782440651e-14,
    ('velocity', 'om4', 'q200', 300): 0.0002650435507476345,
    ('velocity', 'om4', 'q200', 1200): 0.0013097581622147914,
    ('velocity', 'v0', 'q-200', 300): 1.656017735029991e-05,
    ('velocity', 'v0', 'q-200', 1200): 0.00032859285014197326,
    ('velocity', 'v0', 'q0', 300): 4.902744876744691e-13,
    ('velocity', 'v0', 'q0', 1200): 1.4566126083082054e-13,
    ('velocity', 'v0', 'q200', 300): 0.00017245949256050608,
    ('velocity', 'v0', 'q200', 1200): 0.0013581699091567145,
}

_fp64 = _fp64  # the oracle file's fp64-policy autouse fixture


def test_every_sharpness_field_has_a_run():
    assert set(_CLASSES.values()) == set(EXACT_SWITCHES)


def test_every_run_and_case_has_a_stored_gap():
    cases = {("", c, d) for (c, d) in MEASURED_GAP}
    cases |= set(MEASURED_GAP_VARIANTS)
    assert set(SMOOTH_GAP) == {(r, *k) for r in RUNS for k in cases}


@pytest.mark.parametrize("run,variant,case,dt", sorted(SMOOTH_GAP))
def test_smooth_default_gap_to_mom6_is_the_recorded_one(run, variant, case,
                                                        dt):
    key = f"{variant}_{case}_dt{dt}" if variant else f"{case}_dt{dt}"
    with np.load(FIXTURE_VARIANTS if variant else FIXTURE) as fx:
        T_mom6, T0, S0 = (fx[f"{key}_T_8h"], fx[f"{key}_T_init"],
                          fx[f"{key}_S_init"])
    T_lego = _legoesm_T_8h(_Q[case], float(dt), T0, S0, smooth=RUNS[run],
                           variant=VARIANTS.get(variant))
    gap = float(np.max(np.abs(T_lego - T_mom6)))
    np.testing.assert_allclose(gap, SMOOTH_GAP[(run, variant, case, dt)],
                               rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# Gradients at the default, through each kink
# ---------------------------------------------------------------------------


def _check(f, x0=0.0, order=2):
    """check_grads with jax's default tolerances, which are ABSOLUTE as well
    as relative: every ``f`` below is scaled to O(1) in value and slope so a
    kink cannot hide under the absolute tolerance."""
    check_grads(f, (jnp.asarray(x0),), order=order, modes=("fwd", "rev"))


def test_mstar_gradient_through_zero_buoyancy_flux():
    """C1r max(0,B)**2 and C2 min(0,B), both kinked at B = 0."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_bflux_sharpness_s3m2

    def f(x):
        return _find_mstar(cfg, x * w, jnp.asarray(0.014), jnp.asarray(22.0),
                           jnp.asarray(9.37e-5))

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_eqdisc_shape_gradient_through_the_sigma_max_floor():
    """C3 sigma_max = 0.1 exactly: absf chosen in closed form (B = 0)."""
    cfg = _ref_cfg()
    c = _EQD_ML_C
    u_star, hbl = 0.014, 25.0
    lo = cfg.eqd_sigma_max_lower_cap
    f_eh = lo * c[2] / (1.0 - lo * c[1])
    f_fn = 1.0 / (c[3] + c[4]) + c[6]                 # lh = 0 at B = 0
    absf0 = (f_eh / f_fn) * u_star / hbl
    z_int = jnp.arange(51, dtype=jnp.float64)
    # d(sigma_max)/d(ln absf) ~ sigma_max at the floor, so a relative absf
    # step of width/sigma_max moves sigma_max by about one width.
    rel = (1.0 / cfg.smooth_ratio_sharpness) / lo

    def f(x):
        return _eqdisc_shape(cfg, z_int, absf0 * (1.0 + rel * x),
                             jnp.asarray(0.0), jnp.asarray(u_star),
                             jnp.asarray(hbl))

    _check(f)
    assert float(jnp.max(jnp.abs(jax.jacfwd(f)(0.0)))) > 0.0


_PE_ARGS = dict(h_k=1.3, b_den_1=2.1, dte_term=-0.8, dse_term=0.15,
                dt_km1_t2=0.42, ds_km1_t2=-0.07,
                dt_to_dpe_k=3.1e3, ds_to_dpe_k=-2.4e3,
                dt_to_dpe_a=2.7e3, ds_to_dpe_a=-1.9e3, pres_z=1.1e4,
                dt_to_dcolht_k=1.7e-4, ds_to_dcolht_k=-8.0e-5,
                dt_to_dcolht_a=1.3e-4, ds_to_dcolht_a=-6.0e-5)


def test_pe_change_gradient_through_zero_column_height_change():
    """C5 min(ColHt_chg, 0): ColHt_chg = 0 exactly at Kddt_h = 0."""
    s = EPBLConfig._field_defaults["smooth_colht_sharpness_per_m"]
    a = _PE_ARGS
    # d ColHt_chg / d Kddt_h at 0 (MOM6's dColHt_dKd_0); scale Kddt_h so that
    # x = 1 moves ColHt_chg by one default width.
    slope = abs((a["dt_to_dcolht_a"] * a["dt_km1_t2"]
                 + a["ds_to_dcolht_a"] * a["ds_km1_t2"]) / a["b_den_1"]
                + (a["dt_to_dcolht_k"] * a["dte_term"]
                   + a["ds_to_dcolht_k"] * a["dse_term"])
                / (a["h_k"] * a["b_den_1"]))

    # The PE change of the T/S redistribution itself is zeroed, so pe_chg is
    # the column-height term -pres_z * min(ColHt_chg, 0) alone, here in units
    # of pres_z * width.
    no_pe = {**a, "dt_to_dpe_k": 0.0, "ds_to_dpe_k": 0.0,
             "dt_to_dpe_a": 0.0, "ds_to_dpe_a": 0.0}

    def f(x):
        return _find_pe_chg_orig(x / (s * slope), **no_pe,
                                 colht_sharpness=s,
                                 slope_sharpness=math.inf)[0] * (s / a["pres_z"])

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def _column_kv(cfg, q_net, b_scale=1.0, tau=0.2):
    h, T, S, dSV_dT, dSV_dS = _column()
    absf = abs(2 * OMEGA * math.sin(math.radians(40.0)))
    top = surface_tke_forcing(jnp.asarray(q_net), None, h[0], dSV_dT[0],
                              dSV_dS[0], g=G, rho_0=RHO0, c_sw=C_SW,
                              dt=cfg.dt_s)
    tke = jnp.zeros_like(T).at[0].set(top)
    out = epbl_vertical_mixing(
        T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
        dSV_dT[None, :], dSV_dS[None, :], h, cfg,
        tau_x=jnp.reshape(tau, (1,)), tau_y=jnp.zeros(1),
        B_flux=jnp.reshape(
            b_scale * _b_flux(q_net, float(T[0]), float(S[0])), (1,)),
        tke_forcing=tke[None, :], absf=jnp.asarray([absf]),
        g=G, rho_0=RHO0)
    return jnp.sum(out.K_v)


def test_top_layer_energy_gradient_through_zero_forcing():
    """C6: conv_PErel = max(TKE_forced, 0) and the mech_TKE spend at 0."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_energy_sharpness_m2j

    # mech_TKE = 10 widths keeps its own max(0, .) kink 10 widths away; the
    # outputs are in units of the width.
    def f(x):
        return jnp.stack(_top_layer_energy(cfg, x * w, jnp.asarray(10.0 * w))) / w

    _check(f)
    assert float(jnp.min(jnp.abs(jax.jacfwd(f)(0.0)))) > 0.0
    # C6c's own kink, mech_TKE + TKE_forced = 0, sits at x = -10.
    _check(f, x0=-10.0)
    assert float(jnp.abs(jax.jacfwd(f)(-10.0))[1]) > 0.0


@pytest.mark.parametrize("tkef0", [-2.0, -1.0, -0.5, 0.0, 0.5])
def test_top_layer_energy_exact_limit_is_mom6(tkef0):
    """At sharpness = inf both outputs are MOM6's expressions exactly,
    including the mech_TKE cap at mech_TKE + TKE_forced = 0 (tkef0 = -1)."""
    cfg = _ref_cfg(smooth_energy_sharpness_m2j=math.inf)
    mech = 1.0
    conv, m = _top_layer_energy(cfg, jnp.asarray(tkef0), jnp.asarray(mech))
    assert float(conv) == (tkef0 if tkef0 > 0.0 else 0.0)
    assert float(m) == (max(0.0, mech + tkef0) if tkef0 <= 0.0 else mech)


def test_full_column_gradient_at_the_cooling_case():
    """End to end, order 1, at the Q = -200 W/m^2 fixture state.

    Not a per-switch RULE 5 test (the perturbations stay on one side of every
    smoothed kink): it checks that the smoothed column, with all switches
    composed, still has finite AD derivatives that match finite differences.
    """
    cfg = _ref_cfg()
    eps = 1e-4

    def f_b(x):
        return _column_kv(cfg, -200.0, b_scale=1.0 + eps * x)

    def f_tau(y):
        return _column_kv(cfg, -200.0, tau=0.2 * (1.0 + eps * y))

    for f in (f_b, f_tau):
        _check(f, order=1)
        assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


@pytest.mark.parametrize("field", sorted(EXACT_SWITCHES))
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_non_positive_or_nan_sharpness_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        validate_epbl_config(EPBLConfig(**{field: bad}))


# ---------------------------------------------------------------------------
# Gradients at the default through the sites smoothed for the branch-variant
# cases (OM4 mstar, eqdisc v0 / v0h, linear shape, PE-kernel slopes)
# ---------------------------------------------------------------------------

_U, _BLD, _F = 0.014, 22.0, 9.37e-5


def _om4(**over):
    return _ref_cfg(mstar_scheme="om4", **over)


def test_om4_mstar_gradient_through_zero_buoyancy_flux():
    """S1 max(0, B) in the OM4 mstar_s, kinked at B = 0 (B * smoothstep:
    exactly 0 below, cubic onset, so sqrt(.) is C1).  The boundary layer
    (300 m) is deeper than the Ekman depth, so mstar_n ~ 0 and mstar_s wins
    the S3 max."""
    cfg = _om4()
    w = 1.0 / cfg.smooth_bflux_sharpness_s3m2

    def f(x):
        return _find_mstar(cfg, x * w, jnp.asarray(_U), jnp.asarray(300.0),
                           jnp.asarray(_F))

    # At B = 0 the composite goes as B**1.5 on one side: C1 (checked below
    # through the derivative's continuity) but beyond a finite difference.
    for x0 in (-0.5, 0.5, 1.5):
        _check(f, x0=x0, order=1)
    assert float(jnp.abs(jax.grad(f)(0.5))) > 0.0
    g0 = float(jax.grad(f)(0.0))
    for x in (-1e-4, 1e-4):
        # measured 5.4e-3; a hard switch gives ~1 (bound voted 3e-2)
        assert abs(float(jax.grad(f)(x)) - g0) < 3e-2 * abs(float(jax.grad(f)(0.5)))
    # A neutral column gets no shear-limited mstar, as in MOM6: S1 alone
    # (convective reduction off) gives the same mstar as its exact form,
    # at f = 0 too, where a softplus offset would be unbounded.
    for f0 in (_F, 0.0):
        args = (jnp.asarray(0.0), jnp.asarray(_U), jnp.asarray(300.0),
                jnp.asarray(f0))
        smooth = _find_mstar(_om4(mstar_convect_coef=0.0), *args)
        exact = _find_mstar(_om4(mstar_convect_coef=0.0,
                                 smooth_bflux_sharpness_s3m2=math.inf), *args)
        assert float(smooth) == float(exact)


def test_om4_mstar_gradient_through_the_ekman_ratio_kink():
    """S2 log(max(u*/(f h), 1)) at u* = f h (and S3 max(mstar_s, mstar_n),
    both near zero in a stable column, B = -100 buoyancy widths)."""
    cfg = _om4()
    w = 1.0 / cfg.smooth_ratio_sharpness

    def f(x):
        bld = _U / (_F * (1.0 + x * w))
        return _find_mstar(cfg, jnp.asarray(-1.0e-8), jnp.asarray(_U), bld,
                           jnp.asarray(_F)) / (cfg.c_ek * w)

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def _om4_mstar_n():
    cfg = _om4()
    return cfg, cfg.c_ek * math.log(_U / (_F * _BLD))


def test_om4_mstar_gradient_where_shear_and_rotation_limits_cross():
    """S3 max(mstar_s, mstar_n) at mstar_s = mstar_n."""
    cfg, mstar_n = _om4_mstar_n()
    w = 1.0 / cfg.smooth_ratio_sharpness
    b0 = (mstar_n / cfg.mstar_coef) ** 2 * _U**2 * _F
    rel = 2.0 * w / mstar_n          # x = 1 moves mstar_s by one width

    def f(x):
        return _find_mstar(cfg, b0 * (1.0 + rel * x), jnp.asarray(_U),
                           jnp.asarray(_BLD), jnp.asarray(_F)) / w

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_om4_mstar_gradient_at_the_obukhov_cap():
    """S3 min(1.25, mstar_N) at mstar_N = 1.25 (a very shallow boundary
    layer; stable column, so mstar_S ~ 0)."""
    cfg = _om4()
    w = 1.0 / cfg.smooth_ratio_sharpness
    bld0 = _U / (_F * math.exp(1.25 / cfg.c_ek))   # c_ek log(u*/(f h)) = 1.25

    def f(x):
        bld = bld0 / jnp.exp(x * w / cfg.c_ek)
        return _find_mstar(cfg, jnp.asarray(-1.0e-8), jnp.asarray(_U), bld,
                           jnp.asarray(_F)) / w

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_om4_mstar_gradient_at_the_optional_cap():
    """S3 min(MSTAR_CAP, mstar) with the cap set to the uncapped value (no
    MOM6 fixture case enables the cap)."""
    _, mstar_n = _om4_mstar_n()
    cfg = _om4(mstar_cap=mstar_n)
    w = 1.0 / cfg.smooth_ratio_sharpness

    def f(x):
        bld = _BLD / (1.0 + x * w / cfg.c_ek)   # x = 1: mstar_n + one width
        return _find_mstar(cfg, jnp.asarray(0.0), jnp.asarray(_U), bld,
                           jnp.asarray(_F)) / w

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


@pytest.mark.parametrize("which", ["v0", "v0h"])
def test_eqdisc_velocity_gradient_through_the_heat_cool_choice(which):
    """S5 the bflux >= 0 branch choice.  MOM6's branches take sqrt / cube
    roots of |B| (infinite slope at B = 0, not a switch), so the check is
    taken two widths either side of zero, inside the blend."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_bflux_sharpness_s3m2

    def f(x):
        b = x * w
        v = (_eqdisc_v0(cfg, jnp.asarray(_F), b, jnp.asarray(_U))
             if which == "v0" else
             _eqdisc_v0h(cfg, b, jnp.asarray(_U), jnp.asarray(_BLD)))
        return v / _U

    for x0 in (-2.0, 2.0):
        _check(f, x0=x0, order=1)
    blend = jax.grad(f)(2.0) - jax.grad(f)(-2.0)
    assert bool(jnp.isfinite(blend)) and float(jnp.abs(blend)) > 0.0


@pytest.mark.parametrize("which", ["v0", "v0h"])
def test_eqdisc_velocity_gradient_at_the_buoyancy_flux_cap(which):
    """S4 the bflux clip at its upper cap."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_bflux_sharpness_s3m2
    cap = cfg.eqd_bflux_upper_cap

    def f(x):
        b = cap + x * w
        v = (_eqdisc_v0(cfg, jnp.asarray(_F), b, jnp.asarray(_U))
             if which == "v0" else
             _eqdisc_v0h(cfg, b, jnp.asarray(_U), jnp.asarray(_BLD)))
        return v / (_U * w / cap)

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


@pytest.mark.parametrize("which", ["v0", "v0h"])
def test_eqdisc_velocity_gradient_at_the_velocity_cap(which):
    """S6 the v0 / v0h clip, with the upper cap set to the uncapped value."""
    base = _ref_cfg()
    b = 1.0e-8

    def v(cfg, u):
        return (_eqdisc_v0(cfg, jnp.asarray(_F), jnp.asarray(b), u)
                if which == "v0" else
                _eqdisc_v0h(cfg, jnp.asarray(b), u, jnp.asarray(_BLD)))

    v_free = float(v(base, jnp.asarray(_U)))
    cfg = base._replace(eqd_v0_upper_cap=v_free)
    w = 1.0 / cfg.smooth_velocity_sharpness_s_m
    # dv/du* ~ v/u* (v ~ u* at fixed B), so x = 1 moves v about one width.
    rel = w / v_free

    def f(x):
        return v(cfg, _U * (1.0 + rel * x)) / w

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_eqdisc_shape_gradient_at_the_eh_and_lh_caps():
    """S7 min(Eh, cap) and clip(Lh, -cap, cap), each cap set to the value."""
    base = _ref_cfg()
    hbl, b = 25.0, -1.0e-8
    eh = hbl * _F / _U
    lh = -b * hbl / _U**3
    w = 1.0 / base.smooth_ratio_sharpness
    z_int = jnp.arange(51, dtype=jnp.float64)
    for cfg, arg in ((base._replace(eqd_eh_upper_cap=eh), "f"),
                     (base._replace(eqd_lh_cap=lh), "b")):
        def f(x, cfg=cfg, arg=arg):
            absf = _F * (1.0 + x * w / eh) if arg == "f" else _F
            bf = b * (1.0 + x * w / lh) if arg == "b" else b
            return _eqdisc_shape(cfg, z_int, jnp.asarray(absf),
                                 jnp.asarray(bf), jnp.asarray(_U),
                                 jnp.asarray(hbl))

        _check(f)
        assert float(jnp.max(jnp.abs(jax.jacfwd(f)(0.0)))) > 0.0


def test_linear_shape_gradient_where_the_guess_crosses_an_interface():
    """S8 max(0, (MLD - z) / MLD) at MLD = z of interface 10."""
    cfg = _ref_cfg(eqdisc_shape=False, eqdisc_v0h=False)
    w = 1.0 / cfg.smooth_ratio_sharpness
    z_int = jnp.arange(51, dtype=jnp.float64)
    z0 = 10.0

    def f(x):
        return _linear_shape(cfg, z_int, z0 * (1.0 + x * w)) / w

    _check(f)
    assert float(jnp.max(jnp.abs(jax.jacfwd(f)(0.0)))) > 0.0


def _pe_root(out_index, a=_PE_ARGS, kddt=0.37):
    """dte_term at which the column-height quantity behind output
    ``out_index`` of _find_pe_chg_orig crosses zero, and its slope.

    With the T/S part of the PE change zeroed, the output is
    -pres_z * min(q, 0) with q affine in dte_term; two exact evaluations on
    the q < 0 side give the line."""
    no_pe = {**a, "dt_to_dpe_k": 0.0, "ds_to_dpe_k": 0.0,
             "dt_to_dpe_a": 0.0, "ds_to_dpe_a": 0.0}

    def q(d):
        args = {**no_pe, "dte_term": d}
        return -float(_find_pe_chg_orig(
            kddt, **args, colht_sharpness=math.inf,
            slope_sharpness=math.inf)[out_index]) / a["pres_z"]

    for d1 in (-1.0, 1.0, -10.0, 10.0):
        d2 = d1 * 2.0
        if q(d1) < 0.0 and q(d2) < 0.0:
            slope = (q(d2) - q(d1)) / (d2 - d1)
            return no_pe, kddt, d1 - q(d1) / slope, slope
    raise AssertionError("no q < 0 side found")


@pytest.mark.parametrize("out_index,field", [
    (1, "smooth_slope_sharpness"),         # S9a dColHt_dKd
    (2, "smooth_colht_sharpness_per_m"),   # S9b dColHt_max
    (3, "smooth_slope_sharpness"),         # S9c dColHt_dKd_0
])
def test_pe_kernel_gradient_through_each_column_height_limit(out_index,
                                                             field):
    """S9 the three min(., 0) on the column-height slope and limit."""
    s = EPBLConfig._field_defaults[field]
    no_pe, kddt, d0, slope = _pe_root(out_index)
    pres = _PE_ARGS["pres_z"]

    def f(x):
        args = {**no_pe, "dte_term": d0 + x / (s * slope)}
        return _find_pe_chg_orig(
            kddt, **args,
            colht_sharpness=EPBLConfig._field_defaults[
                "smooth_colht_sharpness_per_m"],
            slope_sharpness=EPBLConfig._field_defaults[
                "smooth_slope_sharpness"])[out_index] * (s / pres)

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def _column_kv_interior(cfg, tke_k, level=2):
    """Column K_v with an interior TKE forcing ``tke_k`` at ``level`` (inside
    the ~4 m mixed layer of the cooling column, where TKE is left to spend)."""
    h, T, S, dSV_dT, dSV_dS = _column()
    absf = abs(2 * OMEGA * math.sin(math.radians(40.0)))
    top = surface_tke_forcing(jnp.asarray(-200.0), None, h[0], dSV_dT[0],
                              dSV_dS[0], g=G, rho_0=RHO0, c_sw=C_SW,
                              dt=cfg.dt_s)
    tke = jnp.zeros_like(T).at[0].set(top).at[level].set(tke_k)
    out = epbl_vertical_mixing(
        T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
        dSV_dT[None, :], dSV_dS[None, :], h, cfg,
        tau_x=jnp.full((1,), 0.2), tau_y=jnp.zeros(1),
        B_flux=jnp.reshape(_b_flux(-200.0, float(T[0]), float(S[0])), (1,)),
        tke_forcing=tke[None, :], absf=jnp.asarray([absf]), g=G, rho_0=RHO0)
    return jnp.sum(out.K_v)


def test_interior_tke_forcing_gradient_through_zero():
    """S10 interior conv_PErel += max(0, TKE_forced) (C1 positive part: zero
    slope at and below 0, exact above one width) and the TKE spend, both
    kinked at TKE_forced = 0 (full column, order 1)."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_energy_sharpness_m2j

    def f(x):
        return _column_kv_interior(cfg, x * w)

    for x0 in (-1.0, 0.0, 0.5, 1.5):
        _check(f, x0=x0, order=1)
    g_pos, g_neg = float(jax.grad(f)(0.5)), float(jax.grad(f)(-0.5))
    assert abs(g_pos) > 0.0 and abs(g_neg) > 0.0
    # The derivative is continuous across 0 (MOM6's max / where jump there,
    # which a central difference at 0 cannot see).
    jump = float(jax.grad(f)(1e-3)) - float(jax.grad(f)(-1e-3))
    assert abs(jump) < 1e-2 * max(abs(g_pos), abs(g_neg))  # measured 1.0e-3


@pytest.mark.parametrize("variant", [
    {"mstar_scheme": "om4"},
    {"eqdisc_v0h": False, "eqdisc_v0": True},
    {"eqdisc_shape": False, "eqdisc_v0h": False},
])
def test_full_column_gradient_for_each_branch_variant(variant):
    """End to end, order 1, at Q = -200 W/m^2 for each variant case (the
    velocity / mixing-length caps S11 sit inside this path; their kinks are
    not hit here, as in the cooling-case check above)."""
    cfg = _ref_cfg(**variant)
    eps = 1e-4

    def f_tau(y):
        return _column_kv(cfg, -200.0, tau=0.2 * (1.0 + eps * y))

    _check(f_tau, order=1)
    assert float(jnp.abs(jax.grad(f_tau)(0.0))) > 0.0


def test_heat_cool_choice_gradient_with_distinct_constant_branches():
    """S5 alone: the branch choice blends 1 (heating) and 0 (cooling) across
    B = 0, so its derivative at B = 0 is the blend's, not a branch's."""
    cfg = _ref_cfg()
    w = 1.0 / cfg.smooth_bflux_sharpness_s3m2

    def f(x):
        return _heat_cool(cfg, x * w, jnp.asarray(1.0), jnp.asarray(0.0))

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_tke_gate_gradient_through_zero_tke():
    """S11c: MOM6's where(TKE > 0, x, 0) on vstar / mixlen / Kd."""
    s_e = EPBLConfig._field_defaults["smooth_energy_sharpness_m2j"]

    def f(x):
        return _tke_gate(jnp.asarray(1.0), x / s_e, s_e)

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_rh18_surface_scale_gradient_at_its_floor():
    """S11b: max(0.05, 1 - dz/MLD) at dz = 0.95 MLD."""
    s_r = EPBLConfig._field_defaults["smooth_ratio_sharpness"]
    mld = 20.0

    def f(x):
        return _rh18_surface_scale((0.95 - x / s_r) * mld, jnp.asarray(mld),
                                   s_r) * s_r

    _check(f)
    assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_mixing_length_gradient_at_the_shape_cap_and_the_floor():
    """S11a min(hb_hs, shape) at hb_hs = shape, and S11d the
    EPBL_MIN_MIX_LEN floor (default 0) where the length crosses it, at
    vstar = 0 (length ~ vstar / (Ekman coef * f) there)."""
    base = _ref_cfg()
    s_r = base.smooth_ratio_sharpness
    dz, mls, vs = 6.0, 0.4, 0.02

    def f_a(x):
        return _mixing_length(base, dz, mls + x / s_r, mls, vs, _F)[0] * s_r

    s_c = base.smooth_colht_sharpness_per_m

    def f_d(x):
        v = x * base.ekman_scale_coef * _F / s_c   # x = 1: one width of length
        return _mixing_length(base, dz, 0.9, mls, v, _F)[0] * s_c

    for f in (f_a, f_d):
        _check(f)
        assert float(jnp.abs(jax.grad(f)(0.0))) > 0.0


def test_interior_spend_gradient_at_zero_forcing_and_at_depletion():
    """S10b/c alone: the spend switches on at TKE_forced = 0 and the layer's
    TKE is floored at TKE_forced = -tot_TKE; scaled by the energy width."""
    s_e = EPBLConfig._field_defaults["smooth_energy_sharpness_m2j"]
    tot = 10.0 / s_e                       # ten widths of TKE

    def f(x):
        return _interior_spend_scale(jnp.asarray(tot), x / s_e, s_e) * 10.0

    for x0 in (0.0, -10.0):
        _check(f, x0=x0)
        assert float(jnp.abs(jax.grad(f)(x0))) > 0.0
