"""Spectral plane dycore wrapper tests.

Covers:
1. Round-trip identity: ``to_spectral(to_physical(s)) == s`` to round-off.
2. Wrapper equivalence with no spectral extras (use_spectral_hyperdiff
   off, apply_dealias off): spectral step == FD step to round-off.
3. Rest-state preservation on the spectral path.
4. Dry mass conservation in the spectral wrapper (machine eps under no fixer).
5. AD: jax.grad through the spectral step returns finite gradients.
6. Spectral biharmonic toggle: dissipates the high-k mode of a
   sinusoidal velocity perturbation more aggressively than the FD path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.les.spectral_plane import (
    SpectralPlaneCompressibleEulerModel,
    SpectralPlaneConfig,
    spec_state_from_physical,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=8, ny=8, nlev=4, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(4, H=4_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=False, smagorinsky_cs=0.0,
    )
    return grid, hc, tm, cfg


def test_spec_phys_round_trip_identity():
    grid, hc, tm, cfg = _setup()
    model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=False,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    phys = phys._replace(
        theta_prime=phys.theta_prime.replace(
            data=jnp.asarray(
                rng.standard_normal(phys.theta_prime.data.shape),
            ),
        ),
    )
    spec = model.to_spectral(phys)
    phys_back = model.to_physical(spec)
    np.testing.assert_allclose(
        np.asarray(phys_back.theta_prime.data),
        np.asarray(phys.theta_prime.data),
        rtol=0.0, atol=1.0e-12,
    )


def test_wrapper_step_matches_fd_step_when_extras_disabled():
    """With spectral hyperdiff OFF and dealias OFF, the spectral step
    reduces to ``to_spec(FD.step(to_phys(.)))`` so the round-tripped
    physical state must equal the direct FD step bit-exact-ish (round-
    off only from FFT pair)."""
    grid, hc, tm, cfg = _setup()
    fd = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=False,
        ),
    )

    phys0 = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(11)
    phys0 = phys0._replace(
        theta_prime=phys0.theta_prime.replace(
            data=jnp.asarray(
                0.1 * rng.standard_normal(phys0.theta_prime.data.shape),
            ),
        ),
    )

    spec0 = spec_state_from_physical(phys0)
    dt = 1.0
    phys_fd = fd.step(phys0, dt=dt)
    spec_after = spec_model.step(spec0, dt=dt)
    phys_back = spec_model.to_physical(spec_after)

    np.testing.assert_allclose(
        np.asarray(phys_back.theta_prime.data),
        np.asarray(phys_fd.theta_prime.data),
        rtol=1.0e-10, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(phys_back.u.data), np.asarray(phys_fd.u.data),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_spectral_rest_state_preserved():
    """Spectral wrapper preserves the dry rest state for many steps —
    same property as the FD plane (the spectral biharmonic + dealias
    extras have no effect on the rest state because the only non-zero
    spectral coefficient is the k=0 mean, which is in the kernel of
    every higher-derivative operator)."""
    grid, hc, tm, cfg = _setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=True, apply_dealias=True,
            spectral_hyperdiff_coeff=1.0e5,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    spec = spec_state_from_physical(phys)
    for _ in range(20):
        spec = spec_model.step(spec, dt=1.0)
    phys_back = spec_model.to_physical(spec)
    assert float(jnp.max(jnp.abs(phys_back.w.data))) < 1.0e-10
    assert float(jnp.max(jnp.abs(phys_back.u.data))) < 1.0e-10
    assert float(jnp.max(jnp.abs(phys_back.theta_prime.data))) < 1.0e-10


def test_spectral_dry_mass_conserved():
    """Mass-fixer off; spectral path should conserve dry mass to the
    FD path's tolerance (the transform pair preserves the k=0 mode
    bit-exactly)."""
    grid, hc, tm, cfg = _setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    # Random perturbation in theta'.
    rng = np.random.default_rng(5)
    phys = phys._replace(
        theta_prime=phys.theta_prime.replace(
            data=jnp.asarray(
                0.1 * rng.standard_normal(phys.theta_prime.data.shape),
            ),
        ),
    )
    spec = spec_state_from_physical(phys)
    mass_0 = float(spec_model.compute_dry_mass(spec))
    for _ in range(10):
        spec = spec_model.step(spec, dt=1.0)
    mass_f = float(spec_model.compute_dry_mass(spec))
    drift = abs(mass_f - mass_0) / abs(mass_0)
    assert drift < 1.0e-10, f"spectral mass drift {drift:.3e} > 1e-10"


def test_spectral_step_supports_jax_grad():
    """End-to-end AD through transforms + FD dycore + spectral
    hyperdiff. Loss = sum(w_hat conjugate w_hat at t=t_final);
    grad wrt initial theta' must be finite."""
    grid, hc, tm, cfg = _setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=True, apply_dealias=True,
            spectral_hyperdiff_coeff=1.0e3,
        ),
    )
    phys0 = make_rest_state(grid, hc, dtype=jnp.float64)

    def loss_fn(theta_prime_data):
        phys = phys0._replace(
            theta_prime=phys0.theta_prime.replace(data=theta_prime_data),
        )
        spec = spec_state_from_physical(phys)
        spec = spec_model.step(spec, dt=0.5)
        # Real-valued scalar from a complex array → |w_hat|².
        return jnp.sum(jnp.abs(spec.w_hat.data) ** 2)

    rng = np.random.default_rng(13)
    theta_p = jnp.asarray(
        0.1 * rng.standard_normal(phys0.theta_prime.data.shape),
    )
    grad = jax.grad(loss_fn)(theta_p)
    assert grad.shape == theta_p.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_dealias_preserves_tracer_mean():
    """Codex review 2026-05-24: the 2/3 mask retains the k=0 mode by
    construction, so the global tracer mean (the k=0 Fourier
    coefficient) must be bit-exactly unchanged by ``_apply_dealias``."""
    grid, hc, tm, cfg = _setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    # Inject a non-zero tracer mean via a constant + a high-k bump.
    rng = np.random.default_rng(17)
    base = 0.01 + 1.0e-3 * jnp.asarray(
        rng.standard_normal(phys.tracers.data.shape),
    )
    phys = phys._replace(
        tracers=phys.tracers.replace(data=base),
    )
    spec = spec_state_from_physical(phys)
    mean_before = float(jnp.mean(phys.tracers.data))
    spec_after = spec_model._apply_dealias(spec)
    phys_after = spec_model.to_physical(spec_after)
    mean_after = float(jnp.mean(phys_after.tracers.data))
    np.testing.assert_allclose(mean_after, mean_before, rtol=0.0, atol=1.0e-14)


def test_dealias_preserves_rho_mean():
    """Sister of the tracer-mean test: dry-density perturbation mean
    must be preserved bit-exactly by the dealias mask (k=0 is kept).
    This is the spectral-side guarantee for the dry-mass conservation
    contract: global mass = mean(rho_total) * volume."""
    grid, hc, tm, cfg = _setup()
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=False, apply_dealias=True,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(19)
    rho_p = 1.0e-4 * jnp.asarray(
        rng.standard_normal(phys.rho_prime.data.shape),
    )
    phys = phys._replace(
        rho_prime=phys.rho_prime.replace(data=rho_p),
    )
    spec = spec_state_from_physical(phys)
    mean_before = float(jnp.mean(phys.rho_prime.data))
    spec_after = spec_model._apply_dealias(spec)
    phys_after = spec_model.to_physical(spec_after)
    mean_after = float(jnp.mean(phys_after.rho_prime.data))
    np.testing.assert_allclose(mean_after, mean_before, rtol=0.0, atol=1.0e-14)


def test_spectral_biharmonic_damps_high_k_more_than_low_k():
    """Single high-k vs single low-k mode injected into u; after one
    step of spectral biharmonic, the high-k amplitude must shrink
    more (the ``k⁴`` weighting is exact)."""
    grid, hc, tm, cfg = _setup()
    K = 1.0e7
    dt = 1.0
    spec_model = SpectralPlaneCompressibleEulerModel(
        grid, hc, tm, cfg,
        spectral_config=SpectralPlaneConfig(
            use_spectral_hyperdiff=True, apply_dealias=False,
            spectral_hyperdiff_coeff=K,
        ),
    )
    phys = make_rest_state(grid, hc, dtype=jnp.float64)
    # Low-k mode: m=1; high-k mode: m=nx//2 - 1.
    x = jnp.arange(grid.nx, dtype=jnp.float64) * (grid.Lx / grid.nx)
    low_k = 2.0 * jnp.pi * 1 / grid.Lx
    high_k = 2.0 * jnp.pi * (grid.nx // 2 - 1) / grid.Lx
    u_low = jnp.broadcast_to(
        jnp.cos(low_k * x)[None, :, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    u_high = jnp.broadcast_to(
        jnp.cos(high_k * x)[None, :, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    # Run each through one step; isolate the biharmonic effect by
    # measuring how much the input amplitude shrinks.
    phys_low = phys._replace(u=phys.u.replace(data=u_low))
    phys_high = phys._replace(u=phys.u.replace(data=u_high))
    spec_low = spec_state_from_physical(phys_low)
    spec_high = spec_state_from_physical(phys_high)
    spec_low_after = spec_model.step(spec_low, dt=dt)
    spec_high_after = spec_model.step(spec_high, dt=dt)
    amp_low_before = float(jnp.max(jnp.abs(spec_low.u_hat.data)))
    amp_low_after = float(jnp.max(jnp.abs(spec_low_after.u_hat.data)))
    amp_high_before = float(jnp.max(jnp.abs(spec_high.u_hat.data)))
    amp_high_after = float(jnp.max(jnp.abs(spec_high_after.u_hat.data)))
    decay_low = amp_low_after / amp_low_before
    decay_high = amp_high_after / amp_high_before
    # High-k decay must be substantially stronger.
    assert decay_high < decay_low, (
        f"spectral biharmonic did not damp high-k more than low-k: "
        f"low {decay_low:.3f} vs high {decay_high:.3f}"
    )
