"""Smooth-by-default limits in the SI3 ORCA1 aEVP / landfast / deformation and Prather limiter ports.

User rule: every limit differentiable.  Coordinator ruling (2026-11-08): each existing hard NEMO limit is a C1 limit by
default (band ``scale / limit_sharpness_scale`` on one side of the threshold, bitwise NEMO outside it), and
``limit_sharpness_scale = math.inf`` runs NEMO's literal statements for the oracle tests.  Inputs are the committed
tier-4 ORCA1 fixtures (aEVP top band of D1, Prather windows of D2) with a few cells planted inside the bands.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import dynamics as dyn
from legoesm.ice import transport as tr

from tests.ice.fidelity.test_si3_aevp_north_fold_orca1_oracle import _FIX as _BAND_FIX
from tests.ice.fidelity.test_si3_aevp_north_fold_orca1_oracle import _inputs as _band_inputs
from tests.ice.fidelity.test_si3_prather_open_water_residuals_orca1 import (
    DT,
    ICYCLE,
    KT,
    NN_FSBC,
    TR,
    _window,
)

_SCALE = dyn.SI3_AEVP_DEFAULT_LIMIT_SHARPNESS_SCALE


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


def _band(n_subcycles=1):
    return _band_inputs(np.load(_BAND_FIX), "orca_f_pivot", n_subcycles)


def _ice_cells(forcing):
    at = np.asarray(forcing.concentration_t)
    ii, jj = np.nonzero(at[2:-2, 5:10] > 0.5)
    return [(int(i) + 2, int(j) + 5) for i, j in zip(ii, jj)]


def _plant(forcing, cells, rho_i):
    """Put cells inside the epsi10 presence band and the 1 kg/m2 mass-floor band (rho_i v = 1.004 kg/m2)."""
    at = np.asarray(forcing.concentration_t).copy()
    vi = np.asarray(forcing.ice_volume_t).copy()
    vs = np.asarray(forcing.snow_volume_t).copy()
    (a0, b0), (a1, b1) = cells[:2]
    at[a0, b0] = 1.0e-10 * (1.0 + 0.005)
    vi[a1, b1] = 1.004 / rho_i
    vs[a1, b1] = 0.0
    return forcing._replace(
        concentration_t=jnp.asarray(at), ice_volume_t=jnp.asarray(vi), snow_volume_t=jnp.asarray(vs)
    )


def test_step_gate_exact_outside_band():
    x = jnp.array([-1.0, 0.0, 0.25e-2, 1.0e-2, 2.0])
    g = dyn._si3_step_c1(x, 1.0, _SCALE)
    assert float(g[0]) == 0.0 and float(g[1]) == 0.0 and float(g[3]) == 1.0 and float(g[4]) == 1.0
    assert 0.0 < float(g[2]) < 1.0
    with pytest.raises(ValueError, match="limit_sharpness_scale"):
        dyn._si3_validate_limit_sharpness(0.0)


def test_aevp_default_is_smooth_and_bounded_in_planted_bands():
    """The default differs from the hard arm only in the planted band cells' stencils, by a bounded amount."""
    state, forcing, metrics, config = _band()
    planted = _plant(forcing, _ice_cells(forcing), config.rho_ice)
    hard = dyn.si3_cgrid_aevp_solver(state, planted, metrics, config, limit_sharpness_scale=math.inf)
    default = dyn.si3_cgrid_aevp_solver(state, planted, metrics, config)
    explicit = dyn.si3_cgrid_aevp_solver(state, planted, metrics, config, limit_sharpness_scale=_SCALE)
    changed = 0
    for h, d, e in zip(hard, default, explicit, strict=True):
        h, d, e = np.asarray(h), np.asarray(d), np.asarray(e)
        assert np.isfinite(d).all()
        assert np.array_equal(d, e)
        changed += int((h != d).sum())
    assert changed > 0, "planted band cells must move the smooth default off the hard arm"


@pytest.mark.parametrize("field", ["concentration_t", "ice_volume_t"])
def test_aevp_smooth_gradient_finite_and_matches_fd(field):
    """d(sum u, v)/d(input) at a planted band cell: finite, and equal to a central difference (ratio ~ 1)."""
    state, forcing, metrics, config = _band()
    cells = _ice_cells(forcing)
    planted = _plant(forcing, cells, config.rho_ice)
    cell = cells[0] if field == "concentration_t" else cells[1]
    base = np.asarray(getattr(planted, field))
    scale = float(base[cell]) if base[cell] != 0 else 1.0
    unit = jnp.zeros_like(getattr(planted, field)).at[cell].set(scale)

    def f(t):
        out = dyn.si3_cgrid_aevp_solver(
            state, planted._replace(**{field: getattr(planted, field) + t * unit}), metrics, config,
            differentiable=True,
        )
        return jnp.sum(out.u_ice_u) + jnp.sum(out.v_ice_v)

    g = float(jax.grad(f)(0.0))
    eps = 1.0e-7  # the planted bands are narrow: keep the central difference inside them
    fd = (float(f(eps)) - float(f(-eps))) / (2 * eps)
    f0 = abs(float(f(0.0)))
    assert math.isfinite(g)
    # central-difference roundoff ~ 1e-16 |f| / eps; allow 1e-8 |f| on top of 1e-4 relative
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-8 * f0, (g, fd, f0)


def test_deformation_and_landfast_default_smooth():
    state, forcing, metrics, config = _band()
    planted = _plant(forcing, _ice_cells(forcing), config.rho_ice)
    hard = dyn.si3_cgrid_deformation(state, planted, metrics, config, limit_sharpness_scale=math.inf)
    smooth = dyn.si3_cgrid_deformation(state, planted, metrics, config)
    assert any(np.any(np.asarray(h) != np.asarray(s)) for h, s in zip(hard, smooth))
    # landfast: excess band (v - v_crit in (0, 1 m / s)); with ice equal to its critical volume + 1 mm
    a = jnp.asarray(planted.concentration_t)
    crit = a * config.landfast_depth_fraction * planted.depth_t
    probe = planted._replace(ice_volume_t=crit + 1.0e-3)
    h = dyn.si3_cgrid_landfast_l16_basal_stress(probe, metrics, config, a, a, limit_sharpness_scale=math.inf)[2]
    s = dyn.si3_cgrid_landfast_l16_basal_stress(probe, metrics, config, a, a)[2]
    wet = np.asarray(a) > 0
    assert np.all(np.abs(np.asarray(s))[wet] <= np.abs(np.asarray(h))[wet] + 1e-30)
    assert np.any(np.asarray(s)[wet] != np.asarray(h)[wet])


def _limit_inputs(seed=0, *, epsi20_cells=True):
    rng = np.random.default_rng(seed)
    content = rng.uniform(0.1, 2.0, (6, 5, 3))
    if epsi20_cells:  # the epsi20 bands are far narrower than any finite-difference step: value tests only
        content[0, 0, 0] = 0.0
        content[1, 0, 0] = 3.0e-23  # inside the epsi20 content band
    moments = [rng.normal(0.0, 1.0, content.shape) for _ in range(5)]
    cap = 1.5 * content
    moments[0][2, 0, 0] = cap[2, 0, 0] * (1.0 - 0.002)  # sx just below the slope cap: inside its band
    wet = np.ones(content.shape[:2])
    return jnp.asarray(content), tuple(jnp.asarray(m) for m in moments), jnp.asarray(wet)


@pytest.mark.parametrize("axis", [0, 1])
def test_prather_limiter_smooth_exact_outside_bands(axis):
    """Moments far inside their caps: the smooth limiter is bitwise the hard one, except in the planted cells
    (a slope 0.2 % below its cap -- the one-sided band sits on the inactive side -- and the epsi20 content band), which move off the hard value but stay finite."""
    limit = (tr._si3_prather_limit_x, tr._si3_prather_limit_y)[axis]
    rng = np.random.default_rng(3)
    content = rng.uniform(0.1, 2.0, (6, 5, 3))
    moments = [0.01 * c for c in (rng.normal(0.0, 1.0, content.shape) * content for _ in range(5))]
    content[1, 0, 0] = 3.0e-23  # epsi20 content band
    moments[axis][2, 0, 0] = 1.5 * content[2, 0, 0] * (1.0 - 0.002)  # slope inside its cap band
    planted = np.zeros(content.shape, bool)
    planted[1, 0, 0] = planted[2, 0, 0] = True
    args = (jnp.asarray(content), tuple(jnp.asarray(m) for m in moments), jnp.ones(content.shape[:2]))
    hc, hm = limit(*args, math.inf)
    sc, sm = limit(*args, _SCALE)
    changed = np.asarray(hc) != np.asarray(sc)
    for h, m in zip(hm, sm, strict=True):
        assert np.isfinite(np.asarray(m)).all()
        changed |= np.asarray(h) != np.asarray(m)
    assert changed[2, 0, 0] and changed[1, 0, 0]
    assert not (changed & ~planted).any()


@pytest.mark.parametrize("limit", [tr._si3_prather_limit_x, tr._si3_prather_limit_y])
def test_prather_limiter_smooth_check_grads(limit):
    content, moments, wet = _limit_inputs(1, epsi20_cells=False)

    def f(c, m0, m2):
        mom = (m0, moments[1], m2, moments[3], moments[4])
        out_c, out_m = limit(c, mom, wet, _SCALE)
        return jnp.sum(out_c) + sum(jnp.sum(m * m) for m in out_m)

    check_grads(f, (content, moments[0], moments[2]), order=1, modes=["rev"], atol=1e-6, rtol=1e-6, eps=1e-7)


def _prather(f, lss):
    x = jnp.asarray(f["X"])
    area = jnp.asarray(f["e1e2t"])
    wet = jnp.asarray(f["tmask1"] > 0)
    contents, moments, _ = tr.advect_si3_prather_2d(
        tr.si3_prather_pack_intensives(x, area), tuple(jnp.asarray(m) for m in f["MOM"]),
        jnp.asarray(f["u_ice"]), jnp.asarray(f["v_ice"]), area, wet, DT,
        dx=jnp.asarray(f["e1v"]), dy=jnp.asarray(f["e2u"]), ice_step_index=int(KT), nn_fsbc=int(NN_FSBC),
        halo_width=2, ice_volume_index=TR.index("v_i"), concentration_index=TR.index("a_i"),
        subcycles=int(ICYCLE), limit_sharpness_scale=lss)
    return np.asarray(contents), [np.asarray(m) for m in moments]


@pytest.mark.parametrize("w", [0, 1, 2])
def test_prather_smooth_default_budget(w):
    """Closed budget: ice velocity zeroed on a 3-cell ring at the window edge, so no tracer crosses the scored
    interior boundary.  Every transported tracer total is then the same for the smooth default and the hard arm
    to <= 1e-13 relative; ice area only to 1e-8 (the hard Hbig concentration cap acts on the changed transport;
    full ORCA1 domain: 9.3e-10)."""
    f = dict(_window(w))
    for k in ("u_ice", "v_ice"):
        v = np.array(f[k])
        v[:5], v[-5:], v[:, :5], v[:, -5:] = 0.0, 0.0, 0.0, 0.0
        f[k] = v
    hc, _ = _prather(f, math.inf)
    sc, _ = _prather(f, _SCALE)
    assert np.isfinite(sc).all()
    th, ts = hc[2:-2, 2:-2].sum(axis=(0, 1)), sc[2:-2, 2:-2].sum(axis=(0, 1))
    entry = np.asarray(tr.si3_prather_pack_intensives(jnp.asarray(f["X"]), jnp.asarray(f["e1e2t"])))
    t0 = entry[2:-2, 2:-2].sum(axis=(0, 1))
    tol = np.full(th.shape, 1e-13)
    tol[TR.index("a_i")] = 1e-8
    scale = np.maximum(np.abs(t0), 1e-300)
    assert np.all(np.abs(ts - th) <= tol * scale), (ts - th) / scale
    keep = np.arange(th.size) != TR.index("a_i")
    assert np.all(np.abs(ts - t0)[keep] <= 1e-13 * scale[keep]), (ts - t0) / scale


def test_prather_default_is_smooth():
    f = _window(0)
    assert tr.advect_si3_prather_2d.__kwdefaults__["limit_sharpness_scale"] == tr.SI3_PRA_DEFAULT_LIMIT_SHARPNESS_SCALE
    sc_default, _ = _prather(f, tr.SI3_PRA_DEFAULT_LIMIT_SHARPNESS_SCALE)
    assert np.isfinite(sc_default).all()
    with pytest.raises(ValueError, match="limit_sharpness_scale"):
        _prather(f, 0.0)
