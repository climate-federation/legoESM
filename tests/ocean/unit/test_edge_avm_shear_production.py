"""NEMO zdf_sh2 on Voronoi edges (``_shared.edge_avm_weighted_shear_production``)
and its MPAS TKE-bridge wiring (``tke_shear_avm_weighting="nemo_face"``).

* a rectangular C-grid written as an edge list reproduces the C-grid NEMO
  transcription (``avm_weighted_shear_production``) to round-off on every
  cell, including NEMO's coast doubling at the walls;
* horizontally uniform shear + uniform avm gives avm |du/dz|^2 on the sphere,
  also next to dry neighbours (the wet-weight coast renormalisation);
* a horizontal avm contrast changes the result (the content of the option);
* the MPAS bridge feeds it to the kernel: K changes vs tpoint under nemo_ri.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing._shared import (
    avm_weighted_shear_production,
    edge_avm_weighted_shear_production,
)


@pytest.fixture(autouse=True)
def _x64():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    orig, orig_policy = jax.config.jax_enable_x64, get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_policy)
    jax.config.update("jax_enable_x64", orig)


def _rect_as_edges(n_lat, n_lon, nlev, dx, dy, rng):
    """C-grid box, periodic E-W, walls N/S, and the same box as an edge list."""
    u_face = rng.standard_normal((n_lat, n_lon + 1, nlev))
    u_face[:, n_lon] = u_face[:, 0]                        # periodic duplicate
    v_face = rng.standard_normal((n_lat + 1, n_lon, nlev))
    v_face[0] = v_face[n_lat] = 0.0
    cell = lambda j, i: j * n_lon + (i % n_lon)
    ue, c1, c2, dc, dv = [], [], [], [], []
    uid, vid = {}, {}
    for j in range(n_lat):
        for i in range(n_lon):                             # west face of (j,i)
            uid[j, i] = len(ue); ue.append(u_face[j, i])
            c1.append(cell(j, i - 1)); c2.append(cell(j, i)); dc.append(dx); dv.append(dy)
    for j in range(n_lat + 1):
        for i in range(n_lon):                             # south face of (j,i)
            vid[j, i] = len(ue); ue.append(v_face[j, i])
            c1.append(cell(j - 1, i) if j > 0 else -1)
            c2.append(cell(j, i) if j < n_lat else -1)
            dc.append(dy); dv.append(dx)
    # edges in order around the cell (W, S, E, N), as MPAS edgesOnCell
    eoc = np.array([[uid[j, i], vid[j, i], uid[j, (i + 1) % n_lon], vid[j + 1, i]]
                    for j in range(n_lat) for i in range(n_lon)]).T
    edges = dict(cells_on_edge=jnp.asarray(np.array([c1, c2])),
                 edges_on_cell=jnp.asarray(eoc), dc_edge=jnp.asarray(dc),
                 dv_edge=jnp.asarray(dv),
                 area_cell=jnp.full((n_lat * n_lon,), dx * dy))
    return jnp.asarray(u_face), jnp.asarray(v_face), jnp.asarray(np.array(ue)), edges


def test_rectangle_equals_the_cgrid_nemo_transcription_incl_walls():
    n_lat, n_lon, nlev = 4, 5, 6
    rng = np.random.default_rng(1)
    u_face, v_face, u_edge, edges = _rect_as_edges(n_lat, n_lon, nlev, 7.0e4, 4.0e4, rng)
    K = jnp.asarray(rng.random((n_lat, n_lon, nlev - 1)) * 1e-2)
    dz = jnp.full((n_lat, n_lon, nlev - 1), 12.0)
    u_mask = jnp.ones(u_face.shape)
    v_mask = jnp.ones(v_face.shape).at[0].set(0.0).at[n_lat].set(0.0)
    ref = np.asarray(avm_weighted_shear_production(
        u_face, v_face, u_face, v_face, dz, u_mask, v_mask, K))
    got = np.asarray(edge_avm_weighted_shear_production(
        u_edge, dz.reshape(-1, nlev - 1), K.reshape(-1, nlev - 1),
        jnp.ones((n_lat * n_lon, nlev), bool), **edges)).reshape(ref.shape)
    # every cell, incl. the wall rows where NEMO doubles the wet v-face
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(subdivision_level=3)


def _uniform_flow(mesh, nlev):
    """Edge-normal components of a horizontally uniform flow (U(z), V(z)) in
    the local east/north frame, and |dU/dz|^2 per interface."""
    U = np.linspace(0.6, 0.0, nlev); V = np.linspace(-0.3, 0.1, nlev)
    ang = np.asarray(mesh.angleEdge)[:, None]
    u_edge = jnp.asarray(U[None] * np.cos(ang) + V[None] * np.sin(ang))
    return u_edge, (np.diff(U) ** 2 + np.diff(V) ** 2)


def _edges(mesh):
    return dict(cells_on_edge=mesh.cellsOnEdge, edges_on_cell=mesh.edgesOnCell,
                dc_edge=mesh.dcEdge, dv_edge=mesh.dvEdge, area_cell=mesh.areaCell)


def test_uniform_shear_uniform_avm_gives_avm_shear2_also_at_coasts(mesh):
    """Horizontally uniform shear, uniform avm: avm |du/dz|^2 away from the
    poles (where the east/north frame of the test field is singular), and
    still at coasts thanks to the NEMO pair doubling.  Tolerances are the
    measured sphere-curvature residuals of this 642-cell mesh (2026-10-12:
    interior 0.976-1.020, coast 0.948-1.042)."""
    assert mesh.cellsOnEdge.shape[0] == 2
    assert mesh.edgesOnCell.shape[1] == mesh.areaCell.shape[0]
    nlev, n = 5, mesh.areaCell.shape[0]
    u_edge, s2 = _uniform_flow(mesh, nlev)
    lat = np.degrees(np.asarray(mesh.latCell))
    active = np.ones((n, nlev), bool)
    active[lat > 30.0, 2:] = False                      # a "shelf" north of 30N
    p = np.asarray(edge_avm_weighted_shear_production(
        u_edge, jnp.ones((n, nlev - 1)), jnp.full((n, nlev - 1), 3e-3),
        jnp.asarray(active), **_edges(mesh)))
    ratio = p / (3e-3 * s2[None, :])
    wet = active[:, 1:]
    mid = wet & (np.abs(lat) < 70.0)[:, None]
    nbr = np.asarray(mesh.cellsOnCell).T.clip(0)
    coast = mid & ~np.all(active[nbr][..., 1:], axis=1)
    assert coast.sum() > 50
    assert np.all(np.abs(ratio[mid & ~coast] - 1.0) < 0.025)
    assert np.all(np.abs(ratio[coast] - 1.0) < 0.06)
    assert np.all(p[~wet] == 0.0)


def test_pentagon_with_a_dry_neighbour_uses_the_wet_weight_renormalisation(mesh):
    """Odd-edge cells have no opposite edge: p = (W_all/W_wet) sum_wet w zsh2."""
    eoc = np.asarray(mesh.edgesOnCell)
    n_e = (eoc >= 0).sum(0)
    c = int(np.where(n_e == 5)[0][0])
    nlev, n = 4, mesh.areaCell.shape[0]
    rng = np.random.default_rng(11)
    u = rng.standard_normal((mesh.dcEdge.shape[0], nlev))
    K = rng.random((n, nlev - 1)) * 1e-2
    dz = np.full((n, nlev - 1), 8.0)
    active = np.ones((n, nlev), bool)
    dry = int(np.asarray(mesh.cellsOnCell)[0, c])
    active[dry] = False
    p = np.asarray(edge_avm_weighted_shear_production(
        jnp.asarray(u), jnp.asarray(dz), jnp.asarray(K), jnp.asarray(active),
        **_edges(mesh)))[c]
    coe = np.asarray(mesh.cellsOnEdge)
    dc, dv = np.asarray(mesh.dcEdge), np.asarray(mesh.dvEdge)
    w_all = w_wet = num = 0.0
    for e in eoc[:5, c]:
        w = dc[e] * dv[e] / (4.0 * float(mesh.areaCell[c]))
        a, b = coe[:, e]
        wet = active[a, 1:] & active[b, 1:]
        zsh2 = (K[a] + K[b]) * np.diff(-u[e]) ** 2 / dz[c] ** 2 * wet
        w_all += w; w_wet = w_wet + w * wet; num = num + w * zsh2
    np.testing.assert_allclose(p, num * w_all / w_wet, rtol=1e-12, atol=0)

def test_horizontal_avm_contrast_changes_it(mesh):
    nlev, n = 5, mesh.areaCell.shape[0]
    rng = np.random.default_rng(3)
    u_edge = jnp.asarray(rng.standard_normal((mesh.dcEdge.shape[0], nlev)))
    dz = jnp.full((n, nlev - 1), 10.0)
    active = jnp.ones((n, nlev), bool)
    K = jnp.asarray(rng.random((n, nlev - 1)) * 1e-2)
    Kbar = jnp.full_like(K, float(jnp.mean(K)))
    a = edge_avm_weighted_shear_production(u_edge, dz, K, active, **_edges(mesh))
    b = edge_avm_weighted_shear_production(u_edge, dz, Kbar, active, **_edges(mesh))
    assert not np.allclose(np.asarray(a), np.asarray(b))
    assert bool(jnp.all(a >= 0.0)) and bool(jnp.all(jnp.isfinite(a)))


def test_mpas_bridge_feeds_it_to_the_kernel():
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
        make_tke_profiles_mpas,
    )
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import create_ocean_z_star
    from scripts.run.run_omip_core2 import orca1_zdftke_config
    m = create_voronoi_mesh(subdivision_level=2)
    z = create_ocean_z_star(n_levels=20, H_max=200.0)
    st = rest_state_mpas_ocean(m, z, T_water_init_C=20.0, T_deep=2.0,
                               S_uniform=35.0, H_max=200.0, land_lat_threshold=85.0)
    rng = np.random.default_rng(5)
    st = st._replace(u=st.u.replace(data=jnp.asarray(
        0.2 * rng.standard_normal(st.u.data.shape) * np.linspace(1, 0, 20)[None])))
    n = st.T.data.shape[0]
    f = OceanSurfaceForcing(tau_x=jnp.full((n,), 0.1), tau_y=jnp.zeros((n,)),
                            ice_concentration=jnp.zeros((n,)))

    def run(w):
        card = orca1_zdftke_config(prognostic=True)._replace(
            eice=0, prandtl_mode="nemo_ri", tke_shear_avm_weighting=w)
        return make_tke_profiles_mpas(VerticalMixingConfig(scheme="tke", tke=card))(
            st, m, z, f, dt_tke=150.0)
    A_t, K_t, _ = run("tpoint")
    A_f, K_f, _ = run("nemo_face")
    assert bool(jnp.all(jnp.isfinite(K_f))) and bool(jnp.all(K_f >= 0.0))
    assert float(jnp.max(jnp.abs(K_f - K_t))) > 1e-3 * float(jnp.max(K_t))
    with pytest.raises(ValueError, match="tke_shear_avm_weighting"):
        make_tke_profiles_mpas(VerticalMixingConfig(scheme="tke", tke=orca1_zdftke_config(
            prognostic=True)._replace(tke_shear_avm_weighting="bogus")))
