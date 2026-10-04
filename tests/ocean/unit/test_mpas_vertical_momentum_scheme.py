"""MPAS vertical momentum advection: NEMO dynzad option (``nemo_advective``)
versus the default upwind-perturbation scheme, and unknown selections raise."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(scope="module")
def pieces():
    mesh = create_voronoi_mesh(2)
    z = create_ocean_z_star(4, H_max=400.0)
    st = rest_state_mpas_ocean(mesh, z, H_max=400.0)
    u = np.asarray(st.u.data)
    lat_e = np.asarray(mesh.latEdge)[:, None]
    u = 0.1 * np.cos(lat_e) * np.cos(np.pi * np.arange(u.shape[1]) / u.shape[1])[None, :]
    st = st._replace(u=st.u.replace(data=jnp.asarray(u, st.u.data.dtype)))
    return mesh, z, st


def _run(pieces, scheme, n=4):
    mesh, z, st = pieces
    m = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=0.0, vertical_momentum_scheme=scheme))
    for _ in range(n):
        st = m.step(st, 600.0)
    return st


def test_default_is_unchanged_upwind():
    assert MPASOceanConfig().vertical_momentum_scheme == "upwind_perturbation"


def test_nemo_advective_runs_and_differs_from_upwind(pieces):
    up = _run(pieces, "upwind_perturbation")
    na = _run(pieces, "nemo_advective")
    assert bool(jnp.all(jnp.isfinite(na.u.data)))
    assert float(jnp.max(jnp.abs(na.u.data - up.u.data))) > 0.0


def test_unknown_scheme_raises(pieces):
    with pytest.raises(ValueError, match="vertical_momentum_scheme"):
        _run(pieces, "centred", n=1)
