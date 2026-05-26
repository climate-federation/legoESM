"""Regression test for the iter-95 hydrostatic-BC fix.

Pins:
1. Legacy mode (``p_sfc=None``, top-down ``T_avg=250 K``) gives the
   wrong T at the lowest model level for Wing 2018 RCEMIP1 IC on a
   tall 33 km column — ``T_lowest`` is ~12 K above the prescribed
   Wing surface temperature of 296.8 K (≈ 309 K).
2. iter-95 ``p_sfc=101480.0`` (Wing 2018 Tab A1) switches to the
   bottom-up integration with a known surface BC. Resulting
   ``T_lowest`` matches the Wing 2018 spec to within 0.5 K
   (≈ 296.8 K).
3. ``exner_0`` is strictly monotonically decreasing with height
   in both modes (basic hydrostatic-BC sanity).
4. ``rho_0`` is strictly positive everywhere in both modes.
5. The ``p_sfc`` branch is JIT-compilable and ``jax.grad``-friendly
   (preserves legoESM end-to-end ``jax.grad`` compat).

The legacy behaviour is intentionally retained for backward-compat
with hundreds of existing tests that assert specific reference-state
values; this test pins both branches so neither can silently break.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    WING_GAMMA,
    WING_Q_SFC_DEFAULT,
    WING_T_SFC_DEFAULT,
    WING_Z_T,
    make_wing2018_theta_ref_fn,
)
from legoesm.grids.vertical import compute_reference_state


def _wing_theta_fn():
    return make_wing2018_theta_ref_fn(
        T_sfc=WING_T_SFC_DEFAULT,
        q_sfc=WING_Q_SFC_DEFAULT,
        z_t=WING_Z_T,
        Gamma=WING_GAMMA,
    )


def _build_z_topdown(n_levels=30, H=33_000.0):
    """Top-down z-coord matching create_height_coordinate uniform mode.

    Returns ``z_full`` of length ``n_levels`` with z[0] = z_top and
    z[-1] = dz/2 (lowest cell-centre).
    """
    dz = H / n_levels
    return jnp.linspace(H - dz / 2.0, dz / 2.0, n_levels)


def _t_from_pi_and_theta(theta_0, exner_0):
    return theta_0 * exner_0


def test_legacy_mode_gives_too_hot_lowest_level():
    """Pins the bug iter-95 fixed.

    Without p_sfc, T at the lowest model level (z ≈ 550 m for
    H=33 km / n_lev=30) is ~12 K above the Wing 2018 RCE300 IC
    of 296.8 K. This is the bug that broke CRM surface-flux
    coupling: air ~9 K above the prescribed SST=300 K drove the
    wrong sign of sensible heat flux, preventing convection
    initiation.
    """
    z = _build_z_topdown()
    rho_0, theta_0, exner_0 = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=None,
    )
    T = _t_from_pi_and_theta(theta_0, exner_0)
    T_lowest = float(T[-1])
    assert 305.0 < T_lowest < 315.0, (
        f"Legacy mode T_lowest expected in (305, 315) K (the bug); "
        f"got {T_lowest:.2f} K. Test pins the legacy behaviour — if "
        f"this fails, the top-down BC has changed."
    )


def test_p_sfc_mode_matches_wing_spec():
    """iter-95 fix: bottom-up BC matches Wing 2018 T_sfc to 0.5 K.

    With p_sfc=101480.0 Pa (Wing 2018 Tab A1), T at the lowest
    model level should match T_sfc - Gamma * z_lowest from the
    Wing 2018 troposphere lapse rate (296.8 K at z=550 m).
    """
    z = _build_z_topdown()
    rho_0, theta_0, exner_0 = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=101480.0,
    )
    T = _t_from_pi_and_theta(theta_0, exner_0)
    T_lowest = float(T[-1])
    z_lowest = float(z[-1])
    T_expected = WING_T_SFC_DEFAULT - WING_GAMMA * z_lowest
    assert abs(T_lowest - T_expected) < 0.5, (
        f"p_sfc mode T_lowest expected within 0.5 K of "
        f"T_sfc - Gamma*z = {T_expected:.2f} K at z={z_lowest:.1f} m; "
        f"got {T_lowest:.2f} K."
    )


def test_p_sfc_mode_vs_legacy_diff_at_lowest_level():
    """iter-95 fix is materially different at the lowest level.

    The whole reason for the iter-95 patch is that the two
    integration paths disagree by ~12 K at z ≈ 550 m. Pin that
    so neither branch can silently drift to match the other.
    """
    z = _build_z_topdown()
    _, theta_legacy, exner_legacy = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=None,
    )
    _, theta_psfc, exner_psfc = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=101480.0,
    )
    T_legacy = float((theta_legacy * exner_legacy)[-1])
    T_psfc = float((theta_psfc * exner_psfc)[-1])
    diff = T_legacy - T_psfc
    assert diff > 8.0, (
        f"Legacy mode should be > 8 K hotter than p_sfc mode at "
        f"the lowest level (the iter-95 bug); got diff={diff:.2f} K."
    )


def test_exner_monotonic_legacy():
    z = _build_z_topdown()
    _, _, exner_0 = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=None,
    )
    # z is top-down; exner should increase from top to bottom.
    deltas = jnp.diff(exner_0)
    assert bool(jnp.all(deltas > 0.0)), (
        "exner_0 not strictly monotonic (legacy mode)."
    )


def test_exner_monotonic_p_sfc():
    z = _build_z_topdown()
    _, _, exner_0 = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=101480.0,
    )
    deltas = jnp.diff(exner_0)
    assert bool(jnp.all(deltas > 0.0)), (
        "exner_0 not strictly monotonic (p_sfc mode)."
    )


def test_rho_positive_both_modes():
    z = _build_z_topdown()
    for p_sfc in (None, 101480.0):
        rho_0, _, _ = compute_reference_state(
            z, _wing_theta_fn(), p_sfc=p_sfc,
        )
        assert bool(jnp.all(rho_0 > 0.0)), (
            f"rho_0 not strictly positive (p_sfc={p_sfc})."
        )


def test_p_sfc_at_lowest_pressure_matches_input():
    """iter-95 BC pins the surface pressure.

    The lowest model level sits at z = dz/2, not at z=0. The
    Exner function at z=dz/2 should be slightly less than
    pi_sfc = (p_sfc/p_ref)^kappa, by exactly the hydrostatic
    extrapolation used in the function.
    """
    z = _build_z_topdown()
    p_sfc = 101480.0
    _, _, exner_0 = compute_reference_state(
        z, _wing_theta_fn(), p_sfc=p_sfc,
    )
    pi_sfc_expected = (p_sfc / constants.p_ref) ** (
        constants.R_d / constants.c_pd
    )
    pi_lowest = float(exner_0[-1])
    # pi_lowest at z=dz/2 is below pi_sfc by the hydrostatic
    # extrapolation step pi_sfc + (-g/(c_p*theta_sfc))*z[-1].
    assert pi_lowest < pi_sfc_expected, (
        f"pi at z=dz/2 should be < pi_sfc; got "
        f"pi_lowest={pi_lowest:.6f}, pi_sfc={pi_sfc_expected:.6f}."
    )
    # Verify analytically: expected drop is exactly
    # -g/(c_p*theta_sfc) * z_lowest.
    theta_sfc = float(_wing_theta_fn()(jnp.array([0.0]))[0])
    expected_drop = (
        constants.g / (constants.c_pd * theta_sfc) * float(z[-1])
    )
    actual_drop = pi_sfc_expected - pi_lowest
    assert abs(actual_drop - expected_drop) < 1e-6, (
        f"pi drop from sfc to z[-1] expected "
        f"{expected_drop:.6f}; got {actual_drop:.6f}."
    )


def test_p_sfc_branch_jit_compatible():
    z = _build_z_topdown()
    theta_fn = _wing_theta_fn()

    @jax.jit
    def f(p_sfc):
        # theta_fn closed over because compute_reference_state expects
        # a Python callable (not a traceable arg). p_sfc threads
        # through as a traceable float.
        rho, theta, exner = compute_reference_state(
            z, theta_fn, p_sfc=p_sfc,
        )
        return (theta * exner)[-1]  # T at lowest level

    t_lowest = float(f(101480.0))
    assert 295.0 < t_lowest < 298.0


def test_p_sfc_branch_grad_compatible():
    """Differentiability check (legoESM end-to-end jax.grad goal)."""
    z = _build_z_topdown()
    theta_fn = _wing_theta_fn()

    def f(p_sfc):
        _, theta, exner = compute_reference_state(
            z, theta_fn, p_sfc=p_sfc,
        )
        return (theta * exner)[-1]

    grad_f = jax.grad(f)
    g = float(grad_f(101480.0))
    # dT_lowest/dp_sfc should be positive and finite — higher
    # p_sfc → higher pi_lowest → higher T_lowest at fixed theta.
    assert np.isfinite(g), f"grad not finite; got {g}"
    assert g > 0.0, f"dT/dp_sfc expected positive; got {g}"
    # Order-of-magnitude check: dT_lowest/dp_sfc ≈ T_lowest *
    # kappa / p_sfc ≈ 297 * 0.286 / 101480 ≈ 8.4e-4 K/Pa.
    assert 1e-4 < g < 1e-2, (
        f"dT/dp_sfc expected ~8.4e-4 K/Pa; got {g:.3e}"
    )
