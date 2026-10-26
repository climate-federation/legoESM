"""Cube C-D FCT tracer advection is differentiated EXACTLY (review F6).

The former custom JVP linearised the unlimited PPM divergence (plus a duogrid
flux sync the forward never applies), so its tangent disagreed with finite
differences wherever the limiter bound.  Plain AD of the limited operator must
match a central difference on a fixture where the limiter demonstrably binds.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import (  # noqa: E402
    cgrid_mass_flux_divergence,
    cgrid_tracer_advection_fct,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid  # noqa: E402

N = 6


@pytest.fixture(scope="module")
def case():
    cdgrid = create_cubed_sphere_cdgrid(create_cubed_sphere(N, use_duogrid=True))
    rng = np.random.default_rng(6)
    lat = np.asarray(cdgrid.base.lat)
    lon = np.asarray(cdgrid.base.lon)
    # sharp blob + noise: local extrema where FCT always binds
    q = 1.0 + np.exp(-((lat - 0.3) ** 2 + (lon - 1.0) ** 2) / 0.05) \
        + 0.05 * rng.standard_normal(lat.shape)
    u = 10.0 * rng.standard_normal((6, N + 1, N))
    v = 10.0 * rng.standard_normal((6, N, N + 1))
    dq = rng.standard_normal(q.shape)
    w = rng.standard_normal(q.shape)
    return cdgrid, jnp.asarray(q), jnp.asarray(u), jnp.asarray(v), jnp.asarray(dq), jnp.asarray(w)


def test_limiter_binds_on_the_fixture(case):
    cdgrid, q, u, v, *_ = case
    fct = cgrid_tracer_advection_fct(q, u, v, cdgrid)
    unl = cgrid_mass_flux_divergence(q, u, v, cdgrid)
    assert float(jnp.abs(fct - unl).max()) > 1e-2 * float(jnp.abs(unl).max())


def test_ad_matches_central_difference_where_limiter_binds(case):
    cdgrid, q, u, v, dq, w = case

    def loss(q_):
        return jnp.sum(w * cgrid_tracer_advection_fct(q_, u, v, cdgrid))

    ad = float(jnp.vdot(jax.grad(loss)(q), dq))
    h = 1e-6
    fd = float((loss(q + h * dq) - loss(q - h * dq)) / (2 * h))
    assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1e-30), (ad, fd)


def test_ad_matches_central_difference_wrt_velocity(case):
    cdgrid, q, u, v, _, w = case
    du = jnp.asarray(np.random.default_rng(9).standard_normal(u.shape))

    def loss(u_):
        return jnp.sum(w * cgrid_tracer_advection_fct(q, u_, v, cdgrid))

    ad = float(jnp.vdot(jax.grad(loss)(u), du))
    h = 1e-6
    fd = float((loss(u + h * du) - loss(u - h * du)) / (2 * h))
    assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1e-30), (ad, fd)

