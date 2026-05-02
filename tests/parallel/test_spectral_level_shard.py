"""Tests that the spectral PE step is numerically equivalent across
single-device and level-sharded multi-device execution.

Spectral state arrays have shape ``(n_sh, nlev)``; ``shard_pytree``
under the level mesh shards along axis 1 (``P(None, "level")``).  Each
level's spherical-harmonic transform is independent, so SSP-RK3 over
the sharded spectral state should reproduce the single-device result
to floating-point precision (cross-device reduction order is
identical because spectral transforms don't reduce across the level
axis).

The tests skip if fewer than 2 CPU devices are exposed; run with
``XLA_FLAGS=--xla_force_host_platform_device_count=2`` to exercise
the level-sharded path.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _need_multi_device(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


class TestSpectralLevelShardEquivalence:
    """Single-device vs level-sharded spectral PE produce equivalent
    state after several SSP-RK3 steps.

    Equivalence is ``np.testing.assert_allclose(rtol=1e-12, atol=1e-12)``
    rather than bit-exact because the SH transform's (lat, lon, lev)
    accumulation order can differ slightly under sharding (the level
    axis is independent, but XLA may reorder the FMA chain).  Real
    hardware drift is typically below 1e-10 in the spectral
    coefficients.
    """

    def _run(self, *, devices: int):
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
        )
        from legoesm.parallel.mesh import create_level_mesh, shard_pytree
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

        n_max, n_lev, dt, n_steps = 21, 8, 870.0, 5
        grid = create_gaussian_grid(n_max)
        sigma = create_sigma_coordinate(n_lev)
        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)
        cfg = SpectralPEConfig(
            hyperdiff_coeff=1e16, hyperdiff_order=4,
            spectral_filter_strength=0.01, spectral_filter_order=8,
            time_integrator='ssp_rk3',
        )
        m = SpectralPrimitiveEquationModel(grid, sigma, cfg)
        if devices > 1:
            dev_config = create_level_mesh(n_devices=devices)
            state = shard_pytree(state, dev_config)
        for _ in range(n_steps):
            state = m.step(state, dt)
        return state

    def test_2device_matches_1device(self):
        _need_multi_device(2)
        ref = self._run(devices=1)
        out = self._run(devices=2)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            r = getattr(ref, name).data
            o = getattr(out, name).data
            np.testing.assert_allclose(
                np.asarray(o), np.asarray(r),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{name} drifts under level sharding",
            )
