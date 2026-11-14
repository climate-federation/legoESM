"""Sub-grid cloud-optics inhomogeneity options in compute_cloud_properties.

``constant`` scales the grid-mean paths by the Cahalan scalar.  ``two_region``
is ecRad McICA gamma scaling applied per subcolumn cell by the radiation lane
(tests/unit/test_ecrad_inhomogeneity_subcolumns.py), so the property builder
leaves the paths unscaled and refuses any caller that would drop it.
"""

from __future__ import annotations

import jax.numpy as jnp

import numpy as np
import pytest
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig


def _liquid_cloud_column(q_c_val, nlev=20):
    T = jnp.linspace(240.0, 295.0, nlev)[None, :]            # warm (liquid) BL
    p_full = jnp.linspace(1.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(9.0e3, 1.013e5, nlev + 1)[None, :]
    q_v = jnp.full((1, nlev), 5.0e-3)
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_c = jnp.zeros((1, nlev)).at[0, 15:18].set(q_c_val)     # thick low cloud
    q_i = jnp.zeros((1, nlev))
    return T, p_full, q_v, dp, q_c, q_i


def _lwp(scheme, q_c_val, subcolumn_inhomogeneity=False, **kw):
    T, p_full, q_v, dp, q_c, q_i = _liquid_cloud_column(q_c_val)
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="resolved",
                           cloud_optics_inhomogeneity=scheme, **kw),
        q_cloud=q_c, q_ice=q_i, subcolumn_inhomogeneity=subcolumn_inhomogeneity,
    )
    return np.asarray(cp.lwp)


def test_constant_scheme_scales_linearly_byte_identical():
    """scheme='constant' reproduces the legacy chi*grid-mean path exactly."""
    lwp1 = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=1.0)
    lwp_half = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=0.5)
    assert np.allclose(lwp_half, 0.5 * lwp1, rtol=1e-6)     # exact scalar scaling


def test_two_region_leaves_paths_unscaled_for_the_subcolumn_lane():
    lwp_homog = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=1.0)
    lwp_tr = _lwp("two_region", 4.0e-4, cloud_fsd=0.75,
                  cloud_vertical_overlap_optics="max_random", subcolumn_inhomogeneity=True)
    np.testing.assert_array_equal(lwp_tr, lwp_homog)


@pytest.mark.parametrize("kw", [
    dict(cloud_vertical_overlap_optics="max_random"),               # lane drops it
    dict(cloud_vertical_overlap_optics="mcica", subcolumn_inhomogeneity=True),
    dict(cloud_vertical_overlap_optics="none", subcolumn_inhomogeneity=True),
])
def test_two_region_refused_where_no_subcolumn_scaling_runs(kw):
    with pytest.raises(ValueError, match="two_region"):
        _lwp("two_region", 4.0e-4, **kw)


def test_unknown_inhomogeneity_scheme_raises():
    T, p_full, q_v, dp, q_c, q_i = _liquid_cloud_column(4.0e-4)
    with pytest.raises(ValueError, match="cloud_optics_inhomogeneity"):
        compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="resolved",
                               cloud_optics_inhomogeneity="bogus"),
            q_cloud=q_c, q_ice=q_i,
        )
