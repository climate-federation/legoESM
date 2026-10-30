"""RULE 5 smooth switches of the ePBL port (issue #2180).

Five MOM6 switches on traced state that the MOM6 single-column fixture
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
    _find_mstar,
    _find_pe_chg_orig,
    _top_layer_energy,
    epbl_vertical_mixing,
    surface_tke_forcing,
    validate_epbl_config,
)
from tests.ocean.unit.test_epbl import (
    C_SW, G, OMEGA, RHO0, _b_flux, _column, _ref_cfg)
from tests.ocean.unit.test_epbl_mom6_oracle import (
    EXACT_SWITCHES, FIXTURE, MEASURED_GAP, _Q, _fp64, _legoesm_T_8h)

_DEFAULT = {f: EPBLConfig._field_defaults[f] for f in EXACT_SWITCHES}
_CLASSES = {
    "bflux": "smooth_bflux_sharpness_s3m2",
    "ratio": "smooth_ratio_sharpness",
    "colht": "smooth_colht_sharpness_per_m",
    "energy": "smooth_energy_sharpness_m2j",
}
RUNS = {"all": dict(_DEFAULT)}
RUNS.update({k: {**EXACT_SWITCHES, f: _DEFAULT[f]} for k, f in _CLASSES.items()})

# Measured max |T_legoESM - T_MOM6| at 8 h [degC] at the smooth defaults, fp64
# policy, per run and case (SLURM job 28557232; JAX_ENABLE_X64=1).  The exact
# values are MEASURED_GAP in test_epbl_mom6_oracle.py.
SMOOTH_GAP = {
    ('all', 'q-200', 300): 4.698217974663521e-06,
    ('all', 'q-200', 1200): 0.00024727137541802335,
    ('all', 'q0', 300): 0.003989479135640295,
    ('all', 'q0', 1200): 0.0034861978754996414,
    ('all', 'q200', 300): 0.00013363202374705452,
    ('all', 'q200', 1200): 0.0011014508948399282,
    ('bflux', 'q-200', 300): 2.5869512281673224e-05,
    ('bflux', 'q-200', 1200): 0.0002513095684157918,
    ('bflux', 'q0', 300): 0.004030932365150619,
    ('bflux', 'q0', 1200): 0.0034871684223567456,
    ('bflux', 'q200', 300): 0.00013344868051134995,
    ('bflux', 'q200', 1200): 0.0011014506583038042,
    ('ratio', 'q-200', 300): 2.5868048552979417e-05,
    ('ratio', 'q-200', 1200): 0.00025130956909080737,
    ('ratio', 'q0', 300): 1.148698913766566e-10,
    ('ratio', 'q0', 1200): 6.608047442568932e-13,
    ('ratio', 'q200', 300): 0.00013344872124321228,
    ('ratio', 'q200', 1200): 0.0011014506245707878,
    ('colht', 'q-200', 300): 4.6967527609353965e-06,
    ('colht', 'q-200', 1200): 0.0002472713754357869,
    ('colht', 'q0', 300): 2.1695459292203623e-05,
    ('colht', 'q0', 1200): 6.8694184314210816e-06,
    ('colht', 'q200', 300): 0.00013363197895088774,
    ('colht', 'q200', 1200): 0.0011014509289672958,
    ('energy', 'q-200', 300): 2.586951249128333e-05,
    ('energy', 'q-200', 1200): 0.0002513095684157918,
    ('energy', 'q0', 300): 8.575278442890522e-08,
    ('energy', 'q0', 1200): 1.928611226276189e-08,
    ('energy', 'q200', 300): 0.0001334486804722701,
    ('energy', 'q200', 1200): 0.0011014506587869732,
}

_fp64 = _fp64  # the oracle file's fp64-policy autouse fixture


def test_every_sharpness_field_has_a_run():
    assert set(_CLASSES.values()) == set(EXACT_SWITCHES)


@pytest.mark.parametrize("run,case,dt", sorted(
    (r, c, d) for r in RUNS for (c, d) in MEASURED_GAP))
def test_smooth_default_gap_to_mom6_is_the_recorded_one(run, case, dt):
    key = f"{case}_dt{dt}"
    with np.load(FIXTURE) as fx:
        T_mom6, T0, S0 = (fx[f"{key}_T_8h"], fx[f"{key}_T_init"],
                          fx[f"{key}_S_init"])
    T_lego = _legoesm_T_8h(_Q[case], float(dt), T0, S0, smooth=RUNS[run])
    gap = float(np.max(np.abs(T_lego - T_mom6)))
    np.testing.assert_allclose(gap, SMOOTH_GAP[(run, case, dt)],
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
                                 colht_sharpness=s)[0] * (s / a["pres_z"])

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
