"""Radiative-equilibrium column spin-up tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.idealized.radiative_equilibrium import (
    relax_to_analytic_rce_temperature_target,
    relax_to_radiative_equilibrium_column,
)
from legoesm.grids.vertical import create_stretched_height_coordinate

jax.config.update("jax_enable_x64", True)


def test_relax_preserves_grid_dimensions():
    """Output HeightCoordinate has same n_levels, z_full, dz as input."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=100.0,
    )
    hc_relaxed = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=10, dt=360.0,
    )
    assert hc_relaxed.n_levels == hc.n_levels
    np.testing.assert_array_equal(hc_relaxed.z_full, hc.z_full)
    np.testing.assert_array_equal(hc_relaxed.dz, hc.dz)


def test_relax_produces_finite_reference_state():
    hc = create_stretched_height_coordinate(
        n_levels=30, H=30_000.0, dz_sfc=50.0,
    )
    hc_relaxed = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=50, dt=360.0,
    )
    assert bool(jnp.all(jnp.isfinite(hc_relaxed.theta_ref)))
    assert bool(jnp.all(jnp.isfinite(hc_relaxed.rho_ref)))
    assert bool(jnp.all(jnp.isfinite(hc_relaxed.exner_ref)))


def test_relax_moves_T_toward_equilibrium():
    """After many spin-up steps, the relaxed T column should be CLOSER
    to the analytic equilibrium target than the starting isothermal
    profile."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=200.0,
    )
    T_initial = np.asarray(hc.theta_ref * hc.exner_ref)
    # Equilibrium target: T_eq = T_sfc · exp(-z/(4H_scale))
    z = np.asarray(hc.z_full)
    T_eq = 300.0 * np.exp(-z / (4.0 * 7_500.0))
    T_eq = np.maximum(T_eq, 200.0)
    dist_before = np.max(np.abs(T_initial - T_eq))

    hc_relaxed = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=500, dt=360.0, tau=86_400.0,
    )
    T_after = np.asarray(hc_relaxed.theta_ref * hc_relaxed.exner_ref)
    dist_after = np.max(np.abs(T_after - T_eq))

    assert dist_after < dist_before, (
        f"Relaxation did not reduce distance to equilibrium: "
        f"before={dist_before:.3f}, after={dist_after:.3f}"
    )


def test_relax_tiny_dt_over_tau_is_near_no_op():
    """Codex iter-2: renamed from misleading 'zero_steps' name.
    n_steps=1, dt/tau → 0 effectively → exp(-dt/tau) → 1 → no change."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=100.0,
    )
    hc_relaxed = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=1, dt=1.0, tau=1.0e10,
    )
    # Very small dt/tau ratio → ~no change.
    np.testing.assert_allclose(
        np.asarray(hc_relaxed.theta_ref),
        np.asarray(hc.theta_ref),
        rtol=1.0e-6,
    )


def test_relax_rejects_zero_steps():
    hc = create_stretched_height_coordinate(
        n_levels=10, H=10_000.0, dz_sfc=100.0,
    )
    with pytest.raises(ValueError, match="n_steps"):
        relax_to_radiative_equilibrium_column(hc, n_steps=0)


def test_relax_rejects_nonpositive_tau():
    hc = create_stretched_height_coordinate(
        n_levels=10, H=10_000.0, dz_sfc=100.0,
    )
    with pytest.raises(ValueError, match="tau"):
        relax_to_radiative_equilibrium_column(hc, tau=0.0)


def test_relax_T_sfc_anchors_surface_temperature():
    """Higher T_sfc → higher relaxed T near the surface."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=100.0,
    )
    hc_300 = relax_to_radiative_equilibrium_column(
        hc, T_sfc=300.0, n_steps=1000, dt=360.0, tau=86_400.0,
    )
    hc_310 = relax_to_radiative_equilibrium_column(
        hc, T_sfc=310.0, n_steps=1000, dt=360.0, tau=86_400.0,
    )
    # T at lowest cell should be higher for the warmer T_sfc.
    T_300_sfc = float((hc_300.theta_ref * hc_300.exner_ref)[-1])
    T_310_sfc = float((hc_310.theta_ref * hc_310.exner_ref)[-1])
    assert T_310_sfc > T_300_sfc


def test_relax_rejects_nonpositive_dt():
    """Codex iter-1: validate dt > 0 (was missing)."""
    hc = create_stretched_height_coordinate(
        n_levels=10, H=10_000.0, dz_sfc=100.0,
    )
    with pytest.raises(ValueError, match="dt"):
        relax_to_analytic_rce_temperature_target(hc, dt=0.0)


def test_relax_stable_for_large_dt_over_tau():
    """Codex iter-1: exact exponential update is unconditionally
    stable. Drive dt/tau → 10 (would be explosively unstable under
    explicit Euler) and verify output is bounded + finite."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=100.0,
    )
    hc_relaxed = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=50, dt=1.0e5, tau=1.0e4,
    )
    T = np.asarray(hc_relaxed.theta_ref * hc_relaxed.exner_ref)
    assert np.all(np.isfinite(T))
    # Bounds loose because rebuilt T = θ_ref·exner_ref differs
    # slightly from the relaxed T_final (the hydrostatic
    # reconstruction shifts values by ~10 K). What we PIN here is
    # the absence of explosive oscillation — explicit Euler at
    # dt/tau=10 would produce |T| > 1e6 K.
    assert np.all((T >= 150.0) & (T <= 320.0))


def test_relax_custom_H_scale_changes_decay_rate():
    """Codex iter-1: H_scale is now a kwarg (was hardcoded).
    Larger H_scale → flatter T target → relaxed T at high z higher."""
    hc = create_stretched_height_coordinate(
        n_levels=20, H=20_000.0, dz_sfc=100.0,
    )
    hc_flat = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=1000, dt=360.0, H_scale=15_000.0,
    )
    hc_steep = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=1000, dt=360.0, H_scale=3_000.0,
    )
    T_flat_top = float((hc_flat.theta_ref * hc_flat.exner_ref)[0])
    T_steep_top = float((hc_steep.theta_ref * hc_steep.exner_ref)[0])
    assert T_flat_top > T_steep_top


def test_legacy_name_alias_works_and_emits_deprecation():
    """Codex iter-2: the legacy alias keeps working for one release
    cycle but emits DeprecationWarning so callers migrate."""
    import warnings
    hc = create_stretched_height_coordinate(
        n_levels=10, H=10_000.0, dz_sfc=100.0,
    )
    with warnings.catch_warnings(record=True) as w_record:
        warnings.simplefilter("always")
        hc_via_legacy = relax_to_radiative_equilibrium_column(
            hc, T_sfc=300.0, n_steps=10, dt=360.0,
        )
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "relax_to_analytic_rce_temperature_target" in str(w.message)
        for w in w_record
    ), "legacy alias must emit DeprecationWarning"
    hc_via_new = relax_to_analytic_rce_temperature_target(
        hc, T_sfc=300.0, n_steps=10, dt=360.0,
    )
    np.testing.assert_array_equal(
        np.asarray(hc_via_legacy.theta_ref),
        np.asarray(hc_via_new.theta_ref),
    )


def test_relax_jit_compilable():
    """Inner spin-up loop is fori_loop → JIT-traceable."""
    hc = create_stretched_height_coordinate(
        n_levels=10, H=10_000.0, dz_sfc=100.0,
    )

    @jax.jit
    def _relax_jit(hc):
        return relax_to_radiative_equilibrium_column(
            hc, T_sfc=300.0, n_steps=50, dt=100.0,
        )

    out = _relax_jit(hc)
    assert bool(jnp.all(jnp.isfinite(out.theta_ref)))
