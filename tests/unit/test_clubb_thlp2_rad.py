"""Unit tests for the radiative thlp2 source (calculate_thlp2_rad port).

Exercises the public entry ``clubb_turbulence_prognostic`` with a small
cloudy column: a near-surface saturated layer guarantees PDF cloud water
(``rcm > rc_tol``) so the ``thlp2_rad_coef * 2 * radht_zm / rcm_zm * thlprcp``
source is active where it should be, zero elsewhere, bitwise-inert for
``None``/static-zero configurations, sign-consistent with ``thlprcp``, and
differentiable end to end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    _RC_TOL,
    clubb_turbulence_prognostic,
    pack_clubb_moments,
    thlp2_rad_source,
    unpack_clubb_moments,
    init_clubb_moments,
    CLUBBConfig,
    CLUBBMomentState,
)
from legoesm import constants  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402


@pytest.fixture(autouse=True)
def _release_jax_compilation_cache():
    """Clear JAX's compiled-executable cache between tests (large programs)."""
    yield
    jax.clear_caches()


_NCOL, _NLEV = 2, 12
_THLP2_IDX = CLUBBMomentState._fields.index("thlp2")


def _column():
    """Small top-down column with the lowest 3 layers near saturation."""
    ncol, nlev = _NCOL, _NLEV
    p_half = np.linspace(3.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_full = np.tile(np.linspace(11000.0, 50.0, nlev), (ncol, 1))
    z_half = np.tile(np.linspace(12000.0, 0.0, nlev + 1), (ncol, 1))
    exner = (p_full / constants.p_ref) ** constants.kappa
    theta = 290.0 + 3e-3 * z_full
    T = theta * exner
    u = np.tile(5.0 + 0.1 * np.arange(nlev), (ncol, 1))
    v = 0.5 * np.ones((ncol, nlev))
    # Sub-saturated (30 % RH) free troposphere so the PDF holds NO cloud water
    # aloft, and a slightly supersaturated lowest 3 layers so it does hold
    # cloud water there (rcm > rc_tol) -- the source must act only there.
    q_sat = np.asarray(saturation_mixing_ratio(T, p_full))
    q_v = 0.3 * q_sat
    q_v[:, -3:] = 1.02 * q_sat[:, -3:]
    return dict(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.asarray(T),
        q_v=jnp.asarray(q_v),
        p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
        z_full=jnp.asarray(z_full), z_half=jnp.asarray(z_half),
        T_sfc=jnp.asarray(T[:, -1] + 1.5), q_sfc=jnp.asarray(q_v[:, -1]),
        rho=jnp.asarray(p_full / (constants.R_d * T)),
    )


def _rad(ncol=_NCOL, nlev=_NLEV, magnitude=2e-4):
    """Cooling in the (top-down) lowest 3 layers, i.e. the first ascending zt levels, zero elsewhere [K/s]."""
    rad = np.zeros((ncol, nlev))
    rad[:, -3:] = -magnitude
    return jnp.asarray(rad)


def _moments(config):
    """Fresh moments with the scalar variances lifted OFF their floors.

    At the tolerance-squared floors the variance solve clips any negative
    source straight back to the floor and the PDF's thl-rc covariance is
    tiny, so a floor-seeded column cannot show the source in either
    direction.  Physically plausible boundary-layer values are used instead.
    """
    m = init_clubb_moments(_NCOL, _NLEV, config)
    # Total-water variance only in the lowest 5 momentum levels (ascending grid:
    # index 0 is the surface): a broad rt distribution aloft would put a
    # saturated tail -- and hence PDF cloud water -- into the dry free
    # troposphere and defeat the clear-air test.
    rtp2 = m.rtp2.at[:, :5].set(1e-7)
    return m._replace(
        rtp2=rtp2, thlp2=jnp.full_like(m.thlp2, 0.25),
        rtpthlp=m.rtpthlp.at[:, :5].set(-1e-4), wp2=jnp.full_like(m.wp2, 0.2))


def _run(col, rad_dT_dt=None, dt=300.0, coef=None):
    config = CLUBBConfig()
    if coef is not None:
        config = config._replace(
            params=config.params._replace(thlp2_rad_coef=coef))
    moments = _moments(config)
    out, packed = clubb_turbulence_prognostic(
        col["u"], col["v"], col["T"], col["q_v"],
        pack_clubb_moments(moments),
        col["p_full"], col["p_half"], col["z_full"], col["z_half"],
        col["T_sfc"], col["q_sfc"], col["rho"], dt, config,
        rad_dT_dt=rad_dT_dt)
    thlp2 = unpack_clubb_moments(packed).thlp2
    return out, packed, np.asarray(thlp2)


def test_prognostic_entry_publishes_the_water_beside_the_heat():
    """The production prognostic entry pairs lhflx with evap_sfc = the
    moisture BC the column applied (surface_moisture_flux at the fixed
    T_sfc), same sign, and NOT lhflx / L_v (L(T_sfc) differs from L_v)."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        surface_moisture_flux,
    )
    col = _column()
    # A moist surface (the column helper's q_sfc equals the lowest-level q_v,
    # which would make the latent flux exactly zero and the pin vacuous).
    col["q_sfc"] = col["q_sfc"] * 1.5
    out, _, _ = _run(col)
    assert out.evap_sfc is not None
    want = surface_moisture_flux(CLUBBConfig().surface, out.lhflx, col["T_sfc"])
    np.testing.assert_allclose(np.asarray(out.evap_sfc), np.asarray(want),
                               rtol=1e-6)
    assert np.all(np.sign(np.asarray(out.evap_sfc)) == np.sign(np.asarray(out.lhflx)))
    assert np.all(np.abs(np.asarray(out.lhflx)) > 1.0)   # a real flux, not a 0 == 0 pin
    ratio = np.asarray(out.evap_sfc) / (np.asarray(out.lhflx) / constants.L_v)
    assert np.all(np.abs(ratio - 1.0) > 1e-4)


def test_radiative_source_raises_thlp2_where_cloudy():
    col = _column()
    out0, pack0, thl0 = _run(col, rad_dT_dt=None)
    _, _, thl_cool = _run(col, rad_dT_dt=_rad())
    _, _, thl_warm = _run(col, rad_dT_dt=-_rad())
    # The source is 2*coef*radht/rcm*thlprcp: its sign is the product of the
    # heating sign and the PDF's thl-rc covariance sign, so ONE of the two
    # heating signs must raise thlp2 in the cloudy levels.  The unpacked
    # moments are on the ASCENDING CLUBB grid: zm index 0 is the surface, so
    # the cloudy (top-down lowest 3) layers are the first few zm levels.
    d_cool = thl_cool[:, :6] - thl0[:, :6]
    d_warm = thl_warm[:, :6] - thl0[:, :6]
    assert max(d_cool.max(), d_warm.max()) > 1e-6
    # ... and the two signs move it in opposite directions somewhere.
    assert np.any(d_cool * d_warm < 0.0)
    # Far-above-cloud zm levels (the top of the ascending grid) untouched:
    # no rcm there and the diffusion does not reach.
    assert np.array_equal(thl_cool[:, -3:], thl0[:, -3:])


def test_none_and_static_zero_are_bit_identical():
    col = _column()
    rad = _rad()
    for dt in (300.0, 900.0):  # n_sub == 1 and n_sub == 3
        out_n, pack_n, _ = _run(col, rad_dT_dt=None, dt=dt)
        out_z, pack_z, _ = _run(col, rad_dT_dt=rad, dt=dt, coef=0.0)
        assert np.array_equal(np.asarray(pack_n), np.asarray(pack_z))
        for fld in ("du_dt", "dv_dt", "dT_dt", "dq_v_dt", "Km", "Kh",
                    "shflx", "lhflx", "ustar", "cloud_fraction"):
            a = np.asarray(getattr(out_n, fld))
            b = np.asarray(getattr(out_z, fld))
            assert np.array_equal(a, b), (dt, fld)


def test_sign_follows_thlprcp():
    col = _column()
    _, _, thl0 = _run(col, rad_dT_dt=None)
    _, _, thl_p = _run(col, rad_dT_dt=-_rad())       # warming in cloudy layers
    _, _, thl_m = _run(col, rad_dT_dt=_rad())        # cooling in cloudy layers
    d_p = thl_p - thl0
    d_m = thl_m - thl0
    prod = d_p * d_m
    assert np.all(prod <= 0.0)
    assert np.any(prod < 0.0)


def test_gradient_finite():
    col = _column()
    rad_np = np.asarray(_rad())
    packed0 = pack_clubb_moments(_moments(CLUBBConfig()))

    def _sum_thlp2(mult):
        rad = mult * jnp.asarray(rad_np)
        _, packed = clubb_turbulence_prognostic(
            col["u"], col["v"], col["T"], col["q_v"], packed0,
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], 300.0, CLUBBConfig(),
            rad_dT_dt=rad)
        return jnp.sum(unpack_clubb_moments(packed).thlp2)

    g = jax.grad(_sum_thlp2)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(g))


def test_source_formula_sign_gate_and_magnitude():
    """The forcing is exactly 2*coef*radht/rcm*thlprcp where cloudy, 0 where not."""
    radht = jnp.array([-2e-4, -2e-4, 3e-4, -2e-4])
    rcm = jnp.array([5e-3, 5e-3, 5e-3, 0.5 * _RC_TOL])      # last: clear
    thlprcp = jnp.array([-0.03, 0.03, -0.03, -0.03])
    src = np.asarray(thlp2_rad_source(radht, rcm, thlprcp, 1.0))
    expect = np.asarray(2.0 * radht / rcm * thlprcp)
    assert np.allclose(src[:3], expect[:3])
    assert src[0] > 0.0 and src[1] < 0.0 and src[2] < 0.0     # sign = radht * thlprcp
    assert src[3] == 0.0                                        # gated, not tiny
    assert np.allclose(np.asarray(thlp2_rad_source(radht, rcm, thlprcp, 0.5)), 0.5 * src)
    g = jax.grad(lambda m: jnp.sum(thlp2_rad_source(m * radht, rcm, thlprcp, 1.0)))(1.0)
    assert np.isfinite(g) and g != 0.0


def test_heating_in_clear_air_is_inert():
    """Non-zero heating where the PDF holds no cloud water changes nothing."""
    col = _column()
    _, pack0, _ = _run(col, rad_dT_dt=None)
    rad = np.zeros((_NCOL, _NLEV))
    rad[:, :4] = -5e-4                       # top-down uppermost 4 layers: dry
    _, pack1, _ = _run(col, rad_dT_dt=jnp.asarray(rad))
    assert np.array_equal(np.asarray(pack0), np.asarray(pack1))


def test_subcycle_forwards_the_source():
    """With dt = 3*clubb_dt (lax.scan branch) the heating still reaches the core."""
    col = _column()
    _, pack0, _ = _run(col, rad_dT_dt=None, dt=900.0)
    _, pack1, _ = _run(col, rad_dT_dt=_rad(), dt=900.0)
    assert not np.array_equal(np.asarray(pack0), np.asarray(pack1))


def test_gradient_nonzero_and_matches_finite_difference():
    col = _column()
    rad_np = np.asarray(_rad())
    packed0 = pack_clubb_moments(_moments(CLUBBConfig()))

    def _sum_thlp2(mult):
        _, packed = clubb_turbulence_prognostic(
            col["u"], col["v"], col["T"], col["q_v"], packed0,
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], 300.0, CLUBBConfig(),
            rad_dT_dt=mult * jnp.asarray(rad_np))
        return jnp.sum(unpack_clubb_moments(packed).thlp2)

    g = float(jax.grad(_sum_thlp2)(jnp.asarray(1.0)))
    eps = 1e-3
    fd = float((_sum_thlp2(1.0 + eps) - _sum_thlp2(1.0 - eps)) / (2 * eps))
    assert g != 0.0
    assert abs(g - fd) <= 0.05 * abs(fd) + 1e-12
