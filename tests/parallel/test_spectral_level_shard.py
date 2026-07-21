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

    def _run(self, *, devices: int, n_max: int = 21, n_steps: int = 5):
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
        )
        from legoesm.parallel.mesh import create_level_mesh, shard_pytree
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

        # dt scales inversely with resolution to stay CFL-stable
        dt = 870.0 if n_max <= 21 else 450.0
        n_lev = 8
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

    @pytest.mark.parametrize("devices", [2, 4])
    def test_Ndevice_matches_1device(self, devices):
        """Single-device vs N-device level-sharded spectral PE at T21.

        With nlev=8 the level axis can be split evenly across 2 or 4
        devices.  Iter-29 extends iter-20's 2-device test to also
        cover 4 devices.
        """
        _need_multi_device(devices)
        self._assert_equivalent(devices=devices, n_max=21)

    def test_T42_2device_matches(self):
        """Verify the level-shard equivalence at higher horizontal
        resolution (T42).  Same bit-equivalence envelope as T21:
        spectral transforms are linear in the SH coefficients so the
        accumulation order is independent of horizontal resolution.
        """
        _need_multi_device(2)
        self._assert_equivalent(devices=2, n_max=42)

    def test_long_run_2device_T21_matches(self):
        """20-step regression check that the iter-50/51 cumsum-as-sum
        refactors do not accumulate FP drift over many RK3 steps.

        Iter-50 dropped 2 of 3 cross-level collectives in the spectral
        PE σ-coordinate path (``_compute_sigma_dot_gaussian``); iter-51
        did the same in the hybrid path (``compute_mass_flux_hybrid``).
        Both refactors trade one ``jnp.sum`` for ``cumsum[..., -1]``,
        which has slightly different float-pt accumulation order.

        This test runs 20 SSP-RK3 steps at T21 and asserts the level-
        sharded result matches the single-device reference at the same
        rtol/atol used for the 5-step test (1e-12).  If the drift were
        accumulating (e.g., 1e-13/step) it would have grown to ~2e-12
        and tripped the bound.
        """
        _need_multi_device(2)
        ref = self._run(devices=1, n_max=21, n_steps=20)
        out = self._run(devices=2, n_max=21, n_steps=20)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            r = getattr(ref, name).data
            o = getattr(out, name).data
            np.testing.assert_allclose(
                np.asarray(o), np.asarray(r),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{name} drifts under 2-device level sharding "
                        f"after 20 RK3 steps at T21",
            )

    def _assert_equivalent(self, *, devices: int, n_max: int):
        ref = self._run(devices=1, n_max=n_max)
        out = self._run(devices=devices, n_max=n_max)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            r = getattr(ref, name).data
            o = getattr(out, name).data
            np.testing.assert_allclose(
                np.asarray(o), np.asarray(r),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{name} drifts under {devices}-device level "
                        f"sharding at T{n_max}",
            )
