"""Tests for the tracer vertical-transport conservation probe.

The substantive claim these pin: on the hybrid lane the conservative
Simmons-Burridge operator conserves the mass-weighted column integral of a
tracer in flux form, and the advective upwind operator in use for tracers does
not.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load():
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "validate" / "diag_tracer_vert_transport_conservation.py"
    spec = importlib.util.spec_from_file_location("_tvt_probe", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_tvt_probe"] = m
    spec.loader.exec_module(m)
    return m


def test_masked_int_is_area_and_mass_weighted():
    m = _load()
    rate = np.array([[2.0, 4.0]])
    dp = np.full((1, 2), 50.0)
    w = np.array([1.0])
    mask = np.array([[True, False]])
    assert abs(m.masked_int(rate, dp, w, 10.0, mask) - 10.0) < 1e-12


def _coord_and_flux(jnp):
    from legoesm.grids.vertical import make_cam6_l32_levels
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.array([1.0e5])
    # Interface mass flux, zero at top and surface as the core guarantees.
    mf = jnp.concatenate([
        jnp.zeros((1, 1)),
        jnp.asarray(np.sin(np.linspace(0.3, 2.9, nlev - 1))[None, :]) * 0.5,
        jnp.zeros((1, 1))], axis=-1)
    return coord, p_s, mf


def test_conservative_operator_conserves_in_flux_form_and_advective_does_not():
    """The whole point of the lever, on a sharp vertical gradient.

    Non-vacuous: the advective assertion fails if the two operators are ever
    made identical, and the conservative assertion fails if the flux-form
    telescoping is broken.
    """
    import jax.numpy as jnp
    from legoesm.grids.vertical import (
        vertical_advection_hybrid, vertical_advection_hybrid_sb,
        dp_from_hybrid)

    coord, p_s, mf = _coord_and_flux(jnp)
    nlev = coord.A_full.shape[0]
    # A sharp vertical gradient, the case the docstring says the advective
    # form handles badly: a condensate-like layer with steep edges.
    q = jnp.asarray(
        np.exp(-0.5 * ((np.arange(nlev) - nlev * 0.6) / 1.5) ** 2)[None, :])
    dp = dp_from_hybrid(coord, p_s)
    dmdot = mf[..., 1:] - mf[..., :-1]

    def column_integral(adv):
        return float(jnp.sum(dp * (adv - q * dmdot / jnp.clip(dp, 1e-10, None))))

    adv = vertical_advection_hybrid(q, mf, p_s, coord)
    sb = vertical_advection_hybrid_sb(q, mf, p_s, coord)
    scale = float(jnp.sum(dp * jnp.abs(adv)))
    assert scale > 0.0, "the test state must actually be transported"

    assert abs(column_integral(sb)) <= 1.0e-12 * scale, (
        "the Simmons-Burridge flux form must telescope to zero")
    assert abs(column_integral(adv)) > 1.0e-6 * scale, (
        "the advective upwind form is not conservative; if this passes the "
        "two operators have become identical and the lever is inert")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_conservative_operator_is_not_positivity_preserving_on_sparse_condensate():
    """Why routing tracers through 'sb' is NOT a one-line fix (codex, 2026-09-23).

    SYNTHETIC case, not a production dycore step: an idealised single-layer
    profile with a uniform interface mass flux, integrated once by forward
    Euler.  It demonstrates that the operator lacks a positivity guarantee, not
    that a production step produces this magnitude.

    The Simmons-Burridge form uses centered interface values and carries no
    limiter, so a single cloudy layer between empty ones goes NEGATIVE in one
    physics step.  The upwind form in use is monotone and does not.  Any
    reroute of condensate onto the conservative operator therefore has to bring
    a positivity-preserving variant or a validated repair with it, because the
    existing repair borrows mass and changes the band inventory that this whole
    investigation is measuring.
    """
    import jax.numpy as jnp
    from legoesm.grids.vertical import (
        make_cam6_l32_levels, vertical_advection_hybrid,
        vertical_advection_hybrid_sb, vertical_advection_hybrid_van_leer)

    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.array([1.0e5])
    q = jnp.asarray(np.eye(1, nlev, 18) * 1.0e-3)   # one cloudy layer
    mf = jnp.concatenate([jnp.zeros((1, 1)), jnp.full((1, nlev - 1), 0.3),
                          jnp.zeros((1, 1))], axis=-1)
    dt = 1800.0

    q_up = q + dt * vertical_advection_hybrid(q, mf, p_s, coord)
    q_sb = q + dt * vertical_advection_hybrid_sb(q, mf, p_s, coord)

    assert float(jnp.min(q_up)) >= 0.0, (
        "the upwind form in use is monotone and must stay non-negative")
    assert float(jnp.min(q_sb)) < 0.0, (
        "if this passes, the conservative form has gained a limiter and the "
        "positivity caveat on rerouting tracers can be dropped")
    q_vl = q + dt * vertical_advection_hybrid_van_leer(q, mf, p_s, coord)
    assert float(jnp.min(q_vl)) >= 0.0, (
        "the limited conservative operator the MPAS tracer path now uses must "
        "stay non-negative on the case that defeats the centered one")
