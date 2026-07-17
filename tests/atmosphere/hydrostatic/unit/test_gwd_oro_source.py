"""Direct unit tests for the shared E3SM gw_oro_src source-region helper
(``gravity_wave_drag.oro_source.depth_averaged_oro_source``; gw_oro.F90:
119-145) — every-new-py-gets-a-direct-test rule.

Hand-mirrors the E3SM selection loop and dp-weighted averages on tiny
columns where the arithmetic can be done by eye.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.gravity_wave_drag.oro_source import (
    depth_averaged_oro_source,
)

NLEV = 8


def _column(hdsp=0.0):
    """nlev=8 column, k=0 top.  zm chosen so penetration is easy to reason
    about; uniform dpm=100 hPa/nlev keeps the dp-weights trivial."""
    ncol = 1
    pint = jnp.linspace(100.0, 1.0e5, NLEV + 1)[None, :]
    dpm = pint[:, 1:] - pint[:, :-1]
    zm = jnp.asarray([[8000.0, 6000.0, 4200.0, 2800.0, 1800.0, 1100.0, 600.0, 200.0]])
    u = jnp.asarray([[10., 12., 14., 16., 18., 20., 22., 24.]])
    v = jnp.zeros((ncol, NLEV))
    rho = jnp.asarray([[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]])
    nm = jnp.full((ncol, NLEV), 1.2e-2)
    return u, v, rho, jnp.full((ncol,), hdsp), pint, dpm, zm, nm


def _hand_mirror(u, v, rho, hdsp, pint, dpm, zm, nm):
    """Literal Python transcription of the E3SM loop (1-based -> 0-based)."""
    nlev = u.shape[1]
    include = [False] * nlev
    include[nlev - 1] = True                     # surface always
    lo = nlev // 2 - 1
    for i in range(lo, nlev - 1):                # bottom-half midpoints
        gm = float(np.sqrt(zm[0, i] * zm[0, i + 1]))
        if float(hdsp[0]) > gm:
            include[i] = True
    top = min(i for i in range(nlev) if include[i])
    src_level = top
    dpsrc = float(pint[0, nlev] - pint[0, src_level])
    w = np.array([float(dpm[0, i]) if include[i] else 0.0 for i in range(nlev)])
    rsrc = float((w * np.asarray(rho[0])).sum() / dpsrc)
    usrc = float((w * np.asarray(u[0])).sum() / dpsrc)
    nsrc = float((w * np.asarray(nm[0])).sum() / dpsrc)
    return rsrc, usrc, nsrc, src_level


def test_no_penetration_surface_only():
    """hdsp below every geometric-mean height: only the surface midpoint is
    included; src_level = nlev-1; averages = dp-weighted surface values over
    the pressure interval (NOT the raw surface values — dpsrc uses the
    pint interval)."""
    args = _column(hdsp=100.0)   # gm at the deepest pair = sqrt(600*200)=346
    rsrc, usrc, vsrc, nsrc, src = depth_averaged_oro_source(*args)
    hr, hu, hn, hs = _hand_mirror(*args)
    assert int(src[0]) == NLEV - 1 == hs
    np.testing.assert_allclose(float(rsrc[0]), hr, rtol=1e-12)
    np.testing.assert_allclose(float(usrc[0]), hu, rtol=1e-12)
    np.testing.assert_allclose(float(nsrc[0]), hn, rtol=1e-12)


def test_partial_penetration_matches_hand_mirror():
    """hdsp between two geometric means: exactly the penetrated levels are
    averaged; src_level = interface above the topmost included midpoint."""
    args = _column(hdsp=900.0)   # gm pairs: sqrt(1100*600)=812<900, sqrt(1800*1100)=1407>900
    rsrc, usrc, vsrc, nsrc, src = depth_averaged_oro_source(*args)
    hr, hu, hn, hs = _hand_mirror(*args)
    assert int(src[0]) == hs
    assert hs < NLEV - 1, "fixture must actually penetrate"
    np.testing.assert_allclose(float(rsrc[0]), hr, rtol=1e-12)
    np.testing.assert_allclose(float(usrc[0]), hu, rtol=1e-12)
    np.testing.assert_allclose(float(nsrc[0]), hn, rtol=1e-12)


def test_penetration_capped_at_mid_column():
    """A monster hdsp cannot include midpoints above nlev//2 - 1 (the E3SM
    kk = pver/2 loop bound)."""
    args = _column(hdsp=1.0e9)
    *_, src = depth_averaged_oro_source(*args)
    assert int(src[0]) == NLEV // 2 - 1


def test_differentiable_through_averages():
    """grad wrt u flows through the dp-weighted average (selection mask is
    piecewise-constant; the averages themselves are linear in u)."""
    u, v, rho, hdsp, pint, dpm, zm, nm = _column(hdsp=900.0)

    def f(uu):
        _, usrc, _, _, _ = depth_averaged_oro_source(
            uu, v, rho, hdsp, pint, dpm, zm, nm)
        return jnp.sum(usrc)

    g = jax.grad(f)(u)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
