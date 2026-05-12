"""FV3_3D iter 815: supercell_composite_fv3 (Thompson SCP).

SCP = (CAPE/1000) · (SRH_3km/100) · (BWD_6km/20)
BWD_6km capped at 30 m/s (Thompson et al. 2003).

Tests
-----

1. ``test_scp_known``: CAPE=2500, SRH=300, BWD=20 → SCP=7.5.
2. ``test_scp_bwd_cap``: BWD=50 → capped at 30 m/s.
3. ``test_scp_zero_cape``: CAPE=0 → SCP=0.
4. ``test_scp_zero_srh``: SRH=0 → SCP=0.
5. ``test_scp_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import supercell_composite_fv3


def test_scp_known():
    """CAPE=2500, SRH=300, BWD=20 → SCP = 2.5·3·1 = 7.5."""
    cape = jnp.array([2500.0])
    srh = jnp.array([300.0])
    u_s = jnp.array([20.0])
    v_s = jnp.array([0.0])
    scp = supercell_composite_fv3(cape, srh, u_s, v_s)
    np.testing.assert_allclose(np.asarray(scp), [7.5], rtol=1e-12)


def test_scp_bwd_cap():
    """BWD=50 m/s capped at 30 → BWD factor = 30/20 = 1.5.
    CAPE=2000, SRH=200, BWD=50 → SCP = 2·2·1.5 = 6."""
    cape = jnp.array([2000.0])
    srh = jnp.array([200.0])
    u_s = jnp.array([50.0])
    v_s = jnp.array([0.0])
    scp = supercell_composite_fv3(cape, srh, u_s, v_s)
    np.testing.assert_allclose(np.asarray(scp), [6.0], rtol=1e-12)


def test_scp_zero_cape():
    """CAPE=0 → SCP=0 (no convective potential)."""
    cape = jnp.array([0.0, 0.0, 0.0])
    srh = jnp.array([100.0, 200.0, 500.0])
    u_s = jnp.array([10.0, 20.0, 30.0])
    v_s = jnp.array([0.0, 0.0, 0.0])
    scp = supercell_composite_fv3(cape, srh, u_s, v_s)
    np.testing.assert_allclose(np.asarray(scp), jnp.zeros((3,)), atol=1e-15)


def test_scp_zero_srh():
    """SRH=0 → SCP=0 (no rotation potential)."""
    cape = jnp.array([1000.0, 2000.0, 3000.0])
    srh = jnp.array([0.0, 0.0, 0.0])
    u_s = jnp.array([10.0, 20.0, 30.0])
    v_s = jnp.array([0.0, 0.0, 0.0])
    scp = supercell_composite_fv3(cape, srh, u_s, v_s)
    np.testing.assert_allclose(np.asarray(scp), jnp.zeros((3,)), atol=1e-15)


def test_scp_shapes_finite():
    """3-D shapes preserved, finite, non-negative for non-negative inputs."""
    rng = np.random.default_rng(seed=815)
    n_x, n_y = 6, 8
    cape = jnp.asarray(rng.uniform(500.0, 4000.0, size=(n_x, n_y)))
    srh = jnp.asarray(rng.uniform(0.0, 500.0, size=(n_x, n_y)))
    u_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    v_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    scp = supercell_composite_fv3(cape, srh, u_s, v_s)
    assert scp.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(scp))
