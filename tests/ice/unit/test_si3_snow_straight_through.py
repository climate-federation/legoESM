"""Snow-free test (h_s == 0): NEMO's exact hard forward value, straight-through surrogate gradient.

User ruling 2026-11-04 (option 3a).  The tier-4 ORCA1 record holds trace snow down to 1.2e-30 m in
355 ice cells, so no smooth forward ramp is negligible; instead the forward value stays NEMO's IEEE
test (class A) and, in smooth mode, the gradient is that of ``1 - smoothstep_cubic(h_s / w)`` with
``w = SI3SwitchConfig.snow_ramp_width_m = 2e-4 m = 0.01 x snow_cover_depth_nemo`` (vote Y1), via
``core.smooth.straight_through``.  Both lanes (``sea_ice._nemo_si3_*`` and ``si3_surface``) call
``si3_surface.si3_snow_free``; the lane's own forward pin lives in test_si3_lane_surface_smooth.py.
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIXDIR = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4"


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _all_fixture_snow():
    out = []
    for f in sorted(FIXDIR.glob("*.npz")):
        z = np.load(f)
        out += [z[k] for k in z.files if k.endswith(":h_s")]
    h = np.concatenate(out)
    assert h.size > 500 and np.any(h == 0.0) and np.any((h > 0.0) & (h < 1e-14))
    return jnp.asarray(h)


def test_straight_through_forward_is_hard_bitwise():
    from legoesm.core.smooth import straight_through
    hard = jnp.asarray([0.0, 1.0, 1.0, 0.0, -0.0])
    soft = jnp.asarray([0.3, 0.7, 1e-300, 0.999999, 0.5])
    for f in (straight_through, jax.jit(straight_through)):
        out = f(hard, soft)
        assert np.array_equal(np.asarray(out).view(np.int64), np.asarray(hard).view(np.int64))   # incl. -0.0
        out = f(hard, jnp.full_like(soft, jnp.nan))      # a NaN surrogate never reaches the forward value
        assert np.array_equal(np.asarray(out).view(np.int64), np.asarray(hard).view(np.int64))
    g = jax.grad(lambda s: jnp.sum(straight_through(hard, s * s)))(soft)
    np.testing.assert_array_equal(np.asarray(g), 2.0 * np.asarray(soft))
    # a hard term that depends on the input contributes nothing to the gradient
    g2 = jax.grad(lambda x: jnp.sum(straight_through(3.0 * x, x * x)))(soft)
    np.testing.assert_array_equal(np.asarray(g2), 2.0 * np.asarray(soft))


def _width():
    from legoesm.ice.config import SI3SwitchConfig
    return SI3SwitchConfig().snow_ramp_width_m


def test_default_surrogate_width():
    from legoesm import constants
    assert _width() == 2.0e-4
    np.testing.assert_allclose(_width(), 0.01 * constants.snow_cover_depth_nemo, rtol=1e-15)


def test_snow_free_forward_bitwise_hard_on_all_fixtures():
    from legoesm.ice.si3_surface import si3_snow_free
    tiny = np.nextafter(0.0, 1.0)                         # smallest subnormal
    h = jnp.concatenate([_all_fixture_snow(), jnp.asarray([tiny, -tiny, -0.0, np.nan])])
    for preserve in (True, False):
        for f in (si3_snow_free, jax.jit(si3_snow_free, static_argnums=(1, 2))):
            hard = f(h, 0.0, preserve)
            smooth = f(h, _width(), preserve)
            assert np.array_equal(np.asarray(hard).view(np.int64), np.asarray(smooth).view(np.int64))
    assert float(si3_snow_free(jnp.asarray(tiny), _width(), True)) == 0.0   # subnormal snow is snow


def _blk2(i, switches):
    from legoesm.ice import sea_ice as si
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    return si._nemo_si3_blk_ice_2(
        T_surface=i["t_su"], h_ice=i["h_i"], h_snow=i["h_s"], cloud_fraction=i["cloud_fra"],
        theta_air=i["theta_air_zt"], q_air=i["q_air_zt"], p_surface=i["slp"], lw_down=i["qlw"],
        precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rho_air=i["rhoa"], wind=i["wndm_ice"],
        Ch=i["Ch_ice"], Ce=i["Ce_ice"], ice_fraction_before=i["at_i_b"],
        category_fraction_before=i["a_i_b"], sst_celsius=i["sst_m"], qns_ocean=i["qns_oce"],
        qsr_ocean=i["qsr_oce"], emp_ocean_raw=i["emp"], land_mask=i["smask0"], precip_factor=1.0,
        evap_factor=1.0, snow_blow_exponent=0.66, ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
        switches=switches)


def test_ports_forward_unchanged_by_the_surrogate():
    """Smooth-mode port outputs are bitwise those with the surrogate switched off (forward untouched)."""
    from legoesm.ice import sea_ice as si
    from legoesm.ice.config import SI3SwitchConfig
    z = np.load(FIXDIR / "si3_land_mask.npz")
    i = {k.split(":", 1)[1]: jnp.asarray(z[k]) for k in z.files if k.startswith("blk2_in:")}
    on, off = SI3SwitchConfig(), SI3SwitchConfig()._replace(snow_ramp_width_m=0.0)
    for run in (_blk2, jax.jit(_blk2, static_argnums=1)):
        with_st, without = run(i, on), run(i, off)
        for k in with_st:
            assert np.array_equal(np.asarray(with_st[k]).view(np.int64),
                                  np.asarray(without[k]).view(np.int64)), (run, k)
    a = {k.split(":", 1)[1]: jnp.asarray(z[k]) for k in z.files if k.startswith("alb_in:")}

    def alb(a, sw):
        return si._nemo_si3_ice_albedo(a["t_su"], a["h_i"], a["h_s"], a["cloud_fra"],
                                       land_mask=a["smask0"], switches=sw)
    for run in (alb, jax.jit(alb, static_argnums=1)):
        assert np.array_equal(np.asarray(run(a, on)).view(np.int64), np.asarray(run(a, off)).view(np.int64))


def test_subnormal_width_is_refused():
    from legoesm.ice.si3_surface import si3_snow_free
    with pytest.raises(ValueError, match="subnormal"):
        si3_snow_free(jnp.asarray(0.0), 1e-308, True)
    w = float(np.finfo(np.float64).tiny)                 # smallest normal: finite gradients
    for h0 in (-1.0, 0.0, 0.5 * w, 1.0):
        assert np.isfinite(float(jax.grad(lambda h: si3_snow_free(h, w, True))(jnp.asarray(h0))))


@pytest.mark.parametrize("h0", [1e-30, 2e-5, 1e-4, 1.9e-4])
def test_snow_free_gradient_is_the_surrogate_derivative(h0):
    """jax.grad of the straight-through indicator == central FD of the surrogate 1 - smoothstep(h/w)."""
    from legoesm.core.smooth import smoothstep_cubic
    from legoesm.ice.si3_surface import si3_snow_free
    w = _width()
    g = float(jax.grad(lambda h: si3_snow_free(h, w, True))(jnp.asarray(h0)))
    surrogate = lambda h: 1.0 - smoothstep_cubic(h / w)  # noqa: E731
    step = 1e-9 * w
    fd = (float(surrogate(h0 + step)) - float(surrogate(max(h0 - step, 0.0)))) / (h0 + step - max(h0 - step, 0.0))
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9 / w)
    if h0 > 1e-10:
        assert g < 0.0


def test_port_gradient_in_snow_depth_is_finite_and_nonzero():
    """d(qtr_ice_top)/d(h_s) through the port: finite everywhere, nonzero inside the surrogate band."""
    from legoesm.ice.config import SI3SwitchConfig
    z = np.load(FIXDIR / "si3_land_mask.npz")
    sel = np.flatnonzero((z["blk2_group"] == "wet_ice") & (z["blk2_in:qsr"] > 10.0))[:3]
    assert sel.size == 3
    for c in sel:
        i = {k.split(":", 1)[1]: jnp.asarray(z[k][c]) for k in z.files if k.startswith("blk2_in:")}

        def qtr(h):
            return _blk2({**i, "h_s": h}, SI3SwitchConfig())["qtr_ice_top"]
        for h0 in (0.0, 1e-30, 1e-4):
            g = float(jax.grad(qtr)(jnp.asarray(h0)))
            assert np.isfinite(g), (c, h0)
        assert float(jax.grad(qtr)(jnp.asarray(1e-4))) < 0.0
