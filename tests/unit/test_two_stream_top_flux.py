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

# compute_heating_rate uses the RTE-local constants module (G, CP_D); the
# closure test must use the SAME constants so the telescoping check is exact.
from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rte_const
from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import (
    _replace_top_flux,
    compute_heating_rate,
)


def _clip_and_recompute_net(flux_up, flux_down, flux_net):
    """Replicate the EXACT production clip-then-recompute sequence used in
    ``solve_lw``/``solve_sw`` (clip up/down independently at the TOA face,
    then RECOMPUTE flux_net = up - down at that face only).
    """
    flux_up = _replace_top_flux(flux_up)
    flux_down = _replace_top_flux(flux_down)
    flux_net = flux_net.at[:, :, -1].set(
        flux_up[:, :, -1] - flux_down[:, :, -1]
    )
    return flux_up, flux_down, flux_net


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


# ---------------------------------------------------------------------------
# Column energy closure (item 5): clipping flux_up/flux_down independently at
# the TOA face MUST keep flux_net = up - down at that face, so the top-LAYER
# heating computed from flux_net divergence sees the SAME (clipped) TOA flux
# as rsdt/rsut/rlut.  Clipping flux_net independently would silently break the
# column energy budget (the divergence at the top layer would use a flux_net
# that no longer equals up - down).
# ---------------------------------------------------------------------------


def test_net_identity_holds_at_clipped_toa_face():
    """After the production clip+recompute, flux_net[-1] == up[-1] - down[-1]
    EXACTLY, even on the divergent-curvature case where the two components
    clip in opposite directions.  This is the invariant a naive independent
    clip of flux_net would violate."""
    up = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 400., 0.]]]))
    dn = jnp.asarray(np.array([[[0., 0., 0., 0., 100., 100., 10., 0.]]]))
    # Raw (pre-clip) net carried the linear identity; seed with up-dn.
    net = up - dn
    up_c, dn_c, net_c = _clip_and_recompute_net(up, dn, net)
    # Identity restored at the clipped top face.
    np.testing.assert_allclose(
        np.asarray(net_c[:, :, -1]),
        np.asarray(up_c[:, :, -1] - dn_c[:, :, -1]),
        rtol=0, atol=1e-12,
    )
    # And the recomputed net is the BOUNDED value (390), not the raw-quad net
    # (raw up quad 1000, raw dn quad -170 -> raw net 1170) that the unclipped
    # extrapolation would have produced.
    assert abs(float(net_c[0, 0, -1]) - 390.0) < 1e-9


def test_independent_net_clip_would_break_closure():
    """Non-vacuous: demonstrate that clipping flux_net DIRECTLY (the bug the
    fix avoids) yields a top-face net that does NOT equal up-down, i.e. an
    inconsistent column-top energy boundary.

    Operating point (found by brute search over the stencil): up and down each
    individually overshoot their interior range and get CLAMPED, while the net
    profile's own raw quadratic lands INSIDE the net interior range (so an
    independent net clip leaves the raw quad untouched).  The two answers then
    diverge: clamped-up - clamped-down = 100, but the independently-clipped net
    = 0.  Only the production recompute (net = up_c - down_c) restores closure.
    """
    up = jnp.asarray(np.array([[[0., 0., 0., 0., 50., 150., 300., 0.]]]))
    dn = jnp.asarray(np.array([[[0., 0., 0., 0., 50., 50., 200., 0.]]]))
    net = up - dn  # interior net = [...,0,100,100,...]
    # The BROKEN approach: clip net independently.
    net_broken = float(_replace_top_flux(net)[0, 0, -1])
    up_c = float(_replace_top_flux(up)[0, 0, -1])
    dn_c = float(_replace_top_flux(dn)[0, 0, -1])
    # Confirm up/down were actually clamped (non-vacuous).
    assert abs(up_c - 300.0) < 1e-9   # clamped to interior max
    assert abs(dn_c - 200.0) < 1e-9   # clamped to interior max
    # Broken net top (0) != clipped up - clipped down (100): closure violated.
    assert abs(net_broken - (up_c - dn_c)) > 50.0
    # The PRODUCTION recompute restores the identity exactly.
    _, _, net_fixed = _clip_and_recompute_net(up, dn, net)
    np.testing.assert_allclose(
        float(net_fixed[0, 0, -1]), up_c - dn_c, rtol=0, atol=1e-12
    )


def test_column_heating_integral_closes_against_clipped_toa():
    """End-to-end column energy closure: the mass-weighted column integral of
    the radiative heating rate equals the net radiative flux convergence
    (flux_net[bottom] - flux_net[top]) using the CLIPPED top-face net.

    Telescoping identity for forward_difference (excluding the periodic-roll
    wraparound at the last face):

        sum_k H[k]*|dp[k]|*CP_D/G = -(flux_net[-1] - flux_net[k0])
                                  =  flux_net[k0] - flux_net[-1]

    where k0 is the lowest interior face the heating spans.  We verify the
    closure uses the clipped (physical) flux_net[-1], not the raw overshoot.
    """
    ncol, nface = 3, 9
    rng = np.random.default_rng(7)
    # Build physically plausible monotone-ish up/down face fluxes, then inject
    # a sharp near-TOA curvature into UP so its raw quad would overshoot.
    base = np.linspace(150.0, 240.0, nface)[None, None, :].repeat(ncol, 0)
    up = np.array(base)
    dn = np.array(base) * 0.4
    up[:, :, -2] = 350.0  # sharp jump just below TOA -> raw quad overshoots
    up = jnp.asarray(up)
    dn = jnp.asarray(dn)
    net = up - dn
    up_c, dn_c, net_c = _clip_and_recompute_net(up, dn, net)

    # Layer pressure thickness (faces -> nface, centers -> nface layers under
    # forward_difference; we integrate over the interior layers that telescope
    # cleanly, i.e. layers spanning faces 0..nface-2).
    pressure = jnp.asarray(
        np.linspace(1000.0e2, 1.0e2, nface)[None, None, :].repeat(ncol, 0)
    )
    # Provide an explicit positive dp per layer so the heating sign is set by
    # the flux divergence alone (matches solve_columns).
    dp = jnp.abs(
        jnp.asarray(
            np.diff(np.linspace(1000.0e2, 1.0e2, nface))[None, None, :]
            .repeat(ncol, 0)
        )
    )  # (ncol, 1, nface-1) layer thicknesses

    # Pad dp to nface so compute_heating_rate's forward_difference aligns; the
    # last (wraparound) layer is excluded from the closure sum below.
    dp_full = jnp.concatenate([dp, dp[:, :, -1:]], axis=2)
    H = compute_heating_rate(net_c, pressure, dp=dp_full)  # (ncol,1,nface) K/s

    # Column integral over interior layers k=0..nface-2 (telescopes to
    # net[0] - net[nface-1]); the periodic-roll layer k=nface-1 is dropped.
    lhs = jnp.sum(
        H[:, :, :-1] * dp_full[:, :, :-1] * rte_const.CP_D / rte_const.G,
        axis=2,
    )
    rhs = net_c[:, :, 0] - net_c[:, :, -1]
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-10, atol=1e-8)

    # And confirm the closure used the CLIPPED top net (bounded), not the raw
    # quadratic: raw net top would be 3*net[-2]-3*net[-3]+net[-4].
    raw_net_top = float(3 * net[0, 0, -2] - 3 * net[0, 0, -3] + net[0, 0, -4])
    assert abs(float(net_c[0, 0, -1]) - raw_net_top) > 1.0
