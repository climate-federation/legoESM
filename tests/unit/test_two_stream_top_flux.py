"""BUG B: the RRTMGP two-stream top-of-atmosphere flux-halo extrapolation
(`_replace_top_flux`) must RANGE-LIMIT the quadratic Lagrange stencil so a
drifted-state near-TOA flux curvature cannot overshoot (super-physical TOA SW,
read straight into rsdt/rsut) or undershoot below zero (negative OLR, rlut<0).

Root-caused 2026-06-15 by replaying the dumped day-20 coupled state: the raw
unbounded `3*f[-2]-3*f[-3]+f[-4]` reproduced sw_down max 1121 / lw_up min -50;
disabling it gave sw_down 477 (= insolation) / lw_up +160; the range-limit gave
sw_down 449 / lw_up +176.  These tests pin the bound directly on the function.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import (
    _replace_top_flux,
)


def _raw_quad(f):
    return np.asarray(3 * f[:, :, -2] - 3 * f[:, :, -3] + f[:, :, -4])


def test_flat_profile_is_noop():
    """Flat near-TOA (the physical asymptotic regime): top halo = that value,
    interior untouched, no clamp activation."""
    f = jnp.full((4, 1, 8), 300.0)
    out = _replace_top_flux(f)
    assert np.allclose(np.asarray(out[:, :, -1]), 300.0)
    assert np.array_equal(np.asarray(out[:, :, :-1]), np.asarray(f[:, :, :-1]))


def test_low_curvature_within_interior_range():
    """A gently varying profile: the extrapolated face stays within the local
    interior range (the no-op / mild-clamp regime that never regresses)."""
    base = np.linspace(200.0, 210.0, 8)[None, None, :].repeat(4, 0)
    f = jnp.asarray(base)
    out = np.asarray(_replace_top_flux(f)[:, :, -1])
    lo = np.minimum.reduce([base[:, :, -2], base[:, :, -3], base[:, :, -4]])
    hi = np.maximum.reduce([base[:, :, -2], base[:, :, -3], base[:, :, -4]])
    assert np.all(out >= lo - 1e-9) and np.all(out <= hi + 1e-9)


def test_overshoot_is_capped_to_interior_max():
    """Strong upward near-TOA curvature (the BUG-B SW overshoot): the raw
    quadratic blows up (>2x), the range-limit caps it at the interior max."""
    # f[-4]=100, f[-3]=100, f[-2]=400  ->  raw quad = 3*400-3*100+100 = 1000
    f = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 400., 0.]]]))
    assert _raw_quad(f)[0, 0] > 900.0  # raw overshoots
    out = float(_replace_top_flux(f)[0, 0, -1])
    assert abs(out - 400.0) < 1e-9  # capped at hi = max(interior 3)
    assert out <= 400.0 + 1e-9


def test_undershoot_is_floored_nonnegative():
    """Strong downward near-TOA curvature (the BUG-B negative-OLR undershoot):
    the raw quadratic goes negative, the range-limit floors it at the interior
    min (>= 0 for a non-negative flux)."""
    # f[-2]=10, f[-3]=100, f[-4]=100  ->  raw quad = 3*10-3*100+100 = -170
    f = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 10., 0.]]]))
    assert _raw_quad(f)[0, 0] < 0.0  # raw undershoots below zero
    out = float(_replace_top_flux(f)[0, 0, -1])
    assert out >= 10.0 - 1e-9  # floored to lo = min(interior 3) = 10
    assert out >= 0.0           # non-negative flux preserved


def test_divergent_curvature_components_bounded():
    """The call site clips flux_up and flux_down INDEPENDENTLY (each to its own
    interior range) and recomputes flux_net = up - down afterwards.  Pin that
    each component stays bounded even when up overshoots while down undershoots
    (the divergent-curvature case) — so the recomputed net is consistent AND
    bounded, not the broken independent-clip of net."""
    up = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 400., 0.]]]))
    dn = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 10., 0.]]]))
    up_top = float(_replace_top_flux(up)[0, 0, -1])
    dn_top = float(_replace_top_flux(dn)[0, 0, -1])
    assert abs(up_top - 400.0) < 1e-9   # up capped at its interior max (raw quad 1000)
    assert abs(dn_top - 10.0) < 1e-9    # down floored at its interior min (raw quad -170)
    # The call site sets flux_net[-1] = up_top - dn_top -> consistent by construction.
    assert abs((up_top - dn_top) - 390.0) < 1e-9


def test_only_top_halo_modified():
    """Only index -1 (the top halo) changes; all interior faces are untouched."""
    rng = np.random.default_rng(1)
    f = jnp.asarray(rng.standard_normal((5, 1, 10)) * 50.0 + 200.0)
    out = _replace_top_flux(f)
    assert np.array_equal(np.asarray(out[:, :, :-1]), np.asarray(f[:, :, :-1]))
