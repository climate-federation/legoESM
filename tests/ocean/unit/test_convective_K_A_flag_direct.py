"""Direct unit tests for the shared convective coefficient kernel
``ocean/physics/convection/enhanced_diffusion.convective_K_A_flag``.

This is the single source of truth for the Oceananigans
``ConvectiveAdjustmentVerticalDiffusivity`` coefficients (tracer ``K_conv`` and
the independent momentum ``nu_conv``), shared bit-for-bit by the explicit kernel
and the implicit ``compute_vertical_K_profiles`` fallback.

The hard-step branch, dry-column AD safety, and conservation are pinned in
``test_enhanced_diffusion_momentum.py`` / ``test_ocean_physics_validation_group.py``.
This file pins the leaf branches those don't assert directly:

  * the SMOOTH (sigmoid) transition — K/A interpolate strictly between the
    background and convective values and the flag stays in [0, 1];
  * a fully STABLE column convects nowhere (flag == 0, K == K_bg);
  * dry interfaces (jacobian implies dz <= 0) are zeroed on BOTH branches;
  * output shape is the interior-interface shape (nlev - 1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    convective_K_A_flag,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _columns(n_levels=6):
    """Return (z_coord, J) plus a stable rho profile builder.

    rho increases with depth (stable); caller may invert the surface to
    create an unstable cap.
    """
    z = create_ocean_z_star(n_levels=n_levels, H_max=4000.0)
    J = jnp.ones((6, 3, 3), dtype=jnp.float64)
    return z, J


def _rho_from_T(T):
    # Linear EOS: denser where colder (alpha > 0).
    return constants.rho_ocean - 0.2 * (T - 4.0)


class TestSmoothTransition:
    def test_K_A_interpolate_between_backgrounds_and_flag_bounded(self):
        z, J = _columns()
        shape = (6, 3, 3, z.n_levels)
        T = jnp.broadcast_to(jnp.linspace(18.0, 2.0, z.n_levels), shape)
        T = T.at[..., 0].set(0.0)            # cold dense cap -> N²<0 at top
        rho = _rho_from_T(T)
        cfg = EnhancedDiffusionConfig(
            K_conv=1.0, K_bg=1e-5, nu_conv=0.3, nu_bg=2e-5,
            smooth_transition=True,
        )
        K, A, flag = convective_K_A_flag(rho, z.dz_ref, J, cfg)
        nlev = z.n_levels
        assert K.shape == (6, 3, 3, nlev - 1)
        # Sigmoid output: K/A strictly within [bg, conv]; flag within [0, 1].
        assert bool(jnp.all(K >= cfg.K_bg - 1e-12))
        assert bool(jnp.all(K <= cfg.K_conv + 1e-12))
        assert bool(jnp.all(A >= cfg.nu_bg - 1e-12))
        assert bool(jnp.all(A <= cfg.nu_conv + 1e-12))
        assert bool(jnp.all((flag >= 0.0) & (flag <= 1.0)))
        # The unstable cap drives the top interface toward the convective end.
        assert float(jnp.max(flag)) > 0.5
        assert bool(jnp.all(jnp.isfinite(K)))


class TestHardStepStableColumn:
    def test_stable_column_convects_nowhere(self):
        z, J = _columns()
        shape = (6, 3, 3, z.n_levels)
        # Strictly stable: warm at surface, cold below -> N² > 0 everywhere.
        T = jnp.broadcast_to(jnp.linspace(20.0, 2.0, z.n_levels), shape)
        rho = _rho_from_T(T)
        cfg = EnhancedDiffusionConfig(
            K_conv=1.0, K_bg=1e-5, nu_conv=0.3, nu_bg=2e-5,
            smooth_transition=False,
        )
        K, A, flag = convective_K_A_flag(rho, z.dz_ref, J, cfg)
        assert jnp.allclose(flag, 0.0)
        assert jnp.allclose(K, cfg.K_bg)
        assert jnp.allclose(A, cfg.nu_bg)


class TestDryInterfaceZeroing:
    @pytest.mark.parametrize("smooth", [False, True])
    def test_dry_column_zeroed_and_finite(self, smooth):
        z, J = _columns()
        shape = (6, 3, 3, z.n_levels)
        T = jnp.broadcast_to(jnp.linspace(18.0, 2.0, z.n_levels), shape)
        T = T.at[..., 0].set(0.0)
        rho = _rho_from_T(T)
        # Mark face 0 dry (jacobian = 0 -> dz_actual <= 0 there).
        J = J.at[0].set(0.0)
        cfg = EnhancedDiffusionConfig(
            K_conv=1.0, K_bg=1e-5, nu_conv=0.3, nu_bg=2e-5,
            smooth_transition=smooth,
        )
        K, A, flag = convective_K_A_flag(rho, z.dz_ref, J, cfg)
        assert bool(jnp.all(jnp.isfinite(K)))
        assert bool(jnp.all(jnp.isfinite(A)))
        # Dry face: no mixing, no flag.
        assert jnp.allclose(K[0], 0.0)
        assert jnp.allclose(A[0], 0.0)
        assert jnp.allclose(flag[0], 0.0)
