"""FV3_3D iter 796: dynamic_tropopause_fv3 (|PV| > 2 PVU crossing).

Composes iter-795 ``potential_vorticity_ertel_fv3``.

Tests
-----

1. ``test_dt_monotone_crossing``: |PV| crosses 2 PVU at level 3 → z[3].
2. ``test_dt_no_crossing``: all |PV| < threshold → top of column.
3. ``test_dt_all_above``: all |PV| > threshold → surface.
4. ``test_dt_sh_negative_pv``: SH PV<0; |PV|>2 still works.
5. ``test_dt_custom_thresh``: 4 PVU vs 2 PVU pick different level.
6. ``test_dt_composes_iter795``: full (η, ρ, ∂θ/∂z) → PV → z_dt.
7. ``test_dt_shapes_batched``: 3-D columns (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dynamic_tropopause_fv3,
    potential_vorticity_ertel_fv3,
)


def test_dt_monotone_crossing():
    """|PV| crosses 2 PVU at level 3."""
    pv = jnp.array([0.5e-6, 1.0e-6, 1.5e-6, 3.0e-6, 8.0e-6])  # PVU: 0.5,1,1.5,3,8
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert float(z_dt) == 10_000.0


def test_dt_no_crossing():
    """All |PV| < 2 PVU → top of column."""
    pv = jnp.array([0.5e-6, 1.0e-6, 1.5e-6, 1.8e-6, 1.9e-6])
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert float(z_dt) == 15_000.0


def test_dt_all_above():
    """All |PV| > 2 PVU → surface."""
    pv = jnp.array([3.0e-6, 5.0e-6, 8.0e-6, 12.0e-6, 20.0e-6])
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert float(z_dt) == 1000.0


def test_dt_sh_negative_pv():
    """SH negative PV: |PV|>2 PVU still triggered."""
    pv = jnp.array([-0.5e-6, -1.0e-6, -1.5e-6, -3.0e-6, -8.0e-6])
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert float(z_dt) == 10_000.0


def test_dt_custom_thresh():
    """4 PVU picks higher level than 2 PVU."""
    pv = jnp.array([0.5e-6, 1.0e-6, 1.5e-6, 3.0e-6, 8.0e-6])
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    z_dt_2 = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    z_dt_4 = dynamic_tropopause_fv3(pv, z, pv_thresh=4.0e-6)
    assert float(z_dt_2) == 10_000.0  # first |PV|>2 at idx 3
    assert float(z_dt_4) == 15_000.0  # first |PV|>4 at idx 4


def test_dt_composes_iter795():
    """Full chain: (η, ρ, ∂θ/∂z) → PV → z_dt."""
    # Profile with strong stratospheric gradient transition
    eta = jnp.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4])
    rho = jnp.array([1.0, 0.8, 0.5, 0.3, 0.1])
    dtheta_dz = jnp.array([3e-3, 4e-3, 6e-3, 1.5e-2, 3e-2])  # ↑ aloft
    z = jnp.array([1000.0, 4000.0, 7000.0, 10_000.0, 15_000.0])
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    # Verify PV ramp: PVU = 0.3, 0.5, 1.2, 5, 30
    pvu = pv * 1e6
    assert float(pvu[3]) > 2.0  # crosses near idx 3
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert float(z_dt) == 10_000.0


def test_dt_shapes_batched():
    """3-D batched: (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=796)
    n_x, n_y, km = 4, 5, 20
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(200.0, 800.0, size=(n_x, n_y, km))), axis=-1
    )
    pv = jnp.linspace(0.1e-6, 10.0e-6, km)
    pv = jnp.broadcast_to(pv, (n_x, n_y, km))
    z_dt = dynamic_tropopause_fv3(pv, z, pv_thresh=2.0e-6)
    assert z_dt.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(z_dt))
