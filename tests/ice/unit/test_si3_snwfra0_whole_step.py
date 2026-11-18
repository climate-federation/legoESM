"""nn_snwfra = 0 albedo: whole-step straight-through over the snow-free test (vote A, 3-0).

Per-site straight-through fed the HARD snow fraction (1 - snow_free) into the bare-ice floor
max(1 - pond - fs, 0) and the pond cap min(pond, 1 - fs); with snow present both bind and the
gradient lost a cancelling term (d alb / d h_s 5294 vs 769 from the soft path at h_s = 1e-4 m;
2769 vs 5.7 with ponds).  Now forward = hard (bitwise), gradient = the albedo evaluated with the
soft indicator.  ORCA1 (nn_snwfra = 2) is untouched.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ice import si3_surface
from tests.ice.unit.test_si3_lane_surface_smooth import _alb, _col, _load, _orca1, _sec

jax.config.update("jax_enable_x64", True)

POND = dict(a_ip_eff=jnp.asarray(0.2), h_ip=jnp.asarray(0.1))
H0 = 1.0e-4                     # inside the 2e-4 m surrogate band


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _cfgs(fx):
    base = _orca1(fx)._replace(nn_snwfra=0)
    return {"no_pond": (base, {}), "pond": (base._replace(ld_pnd_alb=True), POND)}


def _off(cfg):
    return cfg._replace(switches=cfg.switches._replace(snow_ramp_width_m=0.0))


@pytest.mark.parametrize("case", ["no_pond", "pond"])
def test_forward_bitwise_hard_on_fixture(case):
    fx = _load()
    cfg, over = _cfgs(fx)[case]
    i = _sec(fx, "alb_in")
    fwd = lambda i, c: _alb(i, c, **over)  # noqa: E731
    for f in (fwd, jax.jit(fwd, static_argnums=1)):
        np.testing.assert_array_equal(np.asarray(f(i, cfg)).view(np.int64), np.asarray(f(i, _off(cfg))).view(np.int64))


def _soft_alb(col, cfg, over):
    """Reference: the albedo with the soft indicator 1 - C1(h_s / w) in the FORWARD pass.

    Built by swapping ``si3_snow_free`` itself, independent of the code under test.
    """
    from legoesm.core.smooth import smoothstep_cubic
    w = cfg.switches.snow_ramp_width_m
    real = si3_surface.si3_snow_free

    def soft(h):
        si3_surface.si3_snow_free = lambda x, _w, _p: 1.0 - smoothstep_cubic(x / w)
        try:
            return jnp.sum(_alb(col, cfg, h_s=h, **over))
        finally:
            si3_surface.si3_snow_free = real
    return soft


@pytest.mark.parametrize("case", ["no_pond", "pond"])
def test_gradient_is_the_whole_soft_path(case):
    fx = _load()
    cfg, over = _cfgs(fx)[case]
    col = _col(fx, "alb", fx["alb_in:h_s"] == 0.0)
    h = jnp.asarray(H0)
    st = lambda t: jnp.sum(_alb(col, cfg, h_s=t, **over))  # noqa: E731
    soft = _soft_alb(col, cfg, over)
    g, g_soft = jax.grad(st)(h), jax.grad(soft)(h)
    assert np.isfinite(float(g)) and float(g) != 0.0
    assert float(g) == float(g_soft)                       # the tangent is the soft path's
    e = 1e-9
    fd = (float(soft(h + e)) - float(soft(h - e))) / (2 * e)
    np.testing.assert_allclose(float(g), fd, rtol=1e-6)
    check_grads(soft, (h,), order=1, modes=("fwd", "rev"), eps=e)
    assert float(st(h)) == float(jnp.sum(_alb(col, _off(cfg), h_s=h, **over)))   # forward still hard


def test_orca1_snwfra2_does_not_take_the_whole_step():
    fx = _load()
    cfg = _orca1(fx)
    assert cfg.nn_snwfra == 2
    col = _col(fx, "alb", fx["alb_in:h_s"] == 0.0)
    h = jnp.asarray(H0)
    g = float(jax.grad(lambda t: jnp.sum(_alb(col, cfg, h_s=t)))(h))
    g_soft = float(jax.grad(_soft_alb(col, cfg, {}))(h))
    assert g != g_soft                                      # per-site straight-through, as #2428
