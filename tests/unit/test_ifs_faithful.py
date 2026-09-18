"""The faithful IFS chain behind BechtoldConfig.use_ifs_ascent.

The switch is OFF by default, so the first test is the one that matters for
every production run: turning the field on must be the ONLY thing that changes
behaviour, and leaving it alone must change nothing at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.config import BechtoldConfig
from legoesm.atmosphere.physics.convection import _ifs_faithful as F
from legoesm.atmosphere.physics.convection import _ifs_test_ascent as TA
from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat


def _deep_column(nlev=30, ps=101300.0):
    """The frozen deep sounding of the other IFS tests (same builder)."""
    sig_h = np.linspace(0.0, 1.0, nlev + 1)
    p_half = (sig_h * ps)[None, :].astype(np.float32).copy()
    p_half[0, 0] = max(p_half[0, 0], 1000.0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    pf = p_full[0].astype(np.float64)
    T0, q0 = 300.0, 0.017
    rcpl = constants.R_d / constants.c_pd
    Tad = np.asarray(compute_moist_adiabat(
        jnp.array([T0]), jnp.array(pf)[None, :], jnp.array([q0])))[0]
    T = np.where(pf > 95000, T0 * (pf / ps) ** rcpl, Tad - 1.0)
    T = np.where(pf < 15000, np.maximum(T, 200.0), T)
    qs = np.asarray(saturation_mixing_ratio(jnp.array(T), jnp.array(pf)))
    q = np.minimum(np.where(pf > 95000, q0, 0.8 * qs), q0)
    return (jnp.asarray(T[None, :], jnp.float32), jnp.asarray(q[None, :], jnp.float32),
            jnp.asarray(p_full), jnp.asarray(p_half))


def _kwargs(ncol, nlev):
    z = jnp.zeros((ncol, nlev))
    return dict(u=z, v=z, conv_prog_profile=z, conv_stoch_state=jnp.zeros(ncol),
                prng_key=None, dt=112.5, land_frac=jnp.zeros(ncol),
                shf_w_m2=jnp.full((ncol,), -20.0),
                lhf_w_m2=jnp.full((ncol,), -100.0),
                dT_dt_dyn=z, dq_dt_dyn=z)


def test_default_off_is_byte_identical():
    """The default must select the legacy scheme EXACTLY -- the switch is the
    only difference between the two configs, and an explicit False must be
    bit-identical to not setting it at all."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    a, mua, _ = bechtold_convection(T, q, p_full, p_half,
                                    config=BechtoldConfig(), **kw)
    b, mub, _ = bechtold_convection(T, q, p_full, p_half,
                                    config=BechtoldConfig(use_ifs_ascent=False), **kw)
    assert BechtoldConfig().use_ifs_ascent is False
    np.testing.assert_array_equal(np.asarray(a.dT_dt), np.asarray(b.dT_dt))
    np.testing.assert_array_equal(np.asarray(a.dq_v_dt), np.asarray(b.dq_v_dt))
    np.testing.assert_array_equal(np.asarray(mua), np.asarray(mub))


def test_faithful_chain_runs_and_is_finite():
    """Switching it on runs the whole ported chain (trigger -> first-guess ->
    ascent -> closure -> tendencies) and returns finite fields of the right
    shape.  It does NOT assert a non-zero tendency: see the declared gap in
    test_shallow_supply_gap_is_declared below."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    out, M_u, stoch = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True), **kw)
    for name, arr in (("dT_dt", out.dT_dt), ("dq_v_dt", out.dq_v_dt),
                      ("M_u", M_u)):
        assert np.asarray(arr).shape == T.shape, name
        assert bool(jnp.all(jnp.isfinite(arr))), name
    assert out.du_dt_conv is None and out.dv_dt_conv is None   # no CMT yet


def test_trigger_fires_on_this_column():
    """Guard against a silent no-op: the chain's FIRST stage must actually
    diagnose convection on this sounding, otherwise the finiteness test above
    would pass on a column that never convects."""
    T, q, p_full, p_half = _deep_column()
    geo_full, geo_half = F._hydrostatic_geopotential(T, q, p_half)
    ncol = T.shape[0]
    trig = TA.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((ncol,), -20.0), jnp.full((ncol,), -100.0),
        jnp.full((ncol,), 0.1), jnp.zeros(ncol), jnp.zeros_like(T),
        TA.IFSTestAscentConfig())
    assert float(np.asarray(trig.ldcum)[0]) > 0.5
    assert int(np.asarray(trig.ktype)[0]) in (1, 2)
    assert int(np.asarray(trig.k_cbot)[0]) > int(np.asarray(trig.k_ctop)[0])


def test_shallow_supply_gap_is_declared():
    """DECLARED GAP, pinned so it cannot be forgotten.

    cumastrn:570-581 sets the shallow cloud-base mass flux from ZDHPBL, the
    sub-cloud integral of the TOTAL physics tendencies (PTENT/PTENQ), and sets
    LDCUM=.FALSE. when that supply is not positive (:578).  This call site does
    not carry a total-physics tendency pair yet, so shallow columns are
    switched off and the chain returns zero for them.  When the pipeline
    plumbing lands, this test should FAIL and be replaced by a real
    shallow-convection assertion.
    """
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True), **kw)
    geo_full, geo_half = F._hydrostatic_geopotential(T, q, p_half)
    trig = TA.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((1,), -20.0), jnp.full((1,), -100.0), jnp.full((1,), 0.1),
        jnp.zeros(1), jnp.zeros_like(T), TA.IFSTestAscentConfig())
    if int(np.asarray(trig.ktype)[0]) == 2:
        assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0, (
            "shallow column with no total-physics tendency must be inert "
            "(cumastrn:578); if this fires, the PTENT/PTENQ plumbing landed "
            "and this test needs replacing")


def test_no_cloud_edge_dipole():
    """REGRESSION for the +-11000 K/day dipole (2026-09-17).

    Two stages were missing between the closure and the tendencies: CUFLXN's
    environmental subtraction (cuflxn.F90:250-251) and the absolute-flux
    reconstruction cumastrn does before CUDTDQN when RMFSOLTQ > 0
    (cumastrn.F90:1194-1216). Without them the environment is subtracted twice
    and the two lowest levels exchange ~1e4 K/day.  Convective heating on a
    single column cannot physically reach that; bound it well below, and check
    the bottom pair is not an equal-and-opposite pair.
    """
    T, q, p_full, p_half = _deep_column()
    ncol, nlev = T.shape
    kw = _kwargs(ncol, nlev)
    kw["dq_dt_dyn"] = jnp.full((ncol, nlev), 1e-8)      # a real sub-cloud supply
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True),
        dT_dt_rad=jnp.full((ncol, nlev), -1.5 / 86400.0), **kw)
    dT_day = np.asarray(out.dT_dt[0]) * 86400.0
    assert np.all(np.abs(dT_day) < 500.0), (
        "cloud-edge dipole is back", dT_day[np.abs(dT_day) > 500.0])
    # the failure mode was specifically equal-and-opposite at the bottom pair
    bottom_pair = abs(dT_day[-1] + dT_day[-2])
    bottom_size = abs(dT_day[-1]) + abs(dT_day[-2])
    assert not (bottom_size > 100.0 and bottom_pair < 0.05 * bottom_size), (
        dT_day[-2], dT_day[-1])
    # not vacuous: the chain must actually be convecting here
    assert float(np.max(np.abs(dT_day))) > 1.0
