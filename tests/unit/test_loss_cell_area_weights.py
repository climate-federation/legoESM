"""Cell-area weighting of the WB loss on the cubed sphere (user decision Q2)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.losses import grid_area_weights, lat_weighted_mean


def _cube():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(6)


def test_cube_default_weights_are_the_cell_areas():
    g = _cube()
    np.testing.assert_array_equal(np.asarray(grid_area_weights(g)),
                                  np.asarray(g.area))


def test_area_weighted_mean_of_a_column_field_with_levels():
    g = _cube()
    area = np.asarray(g.area)
    f = np.random.default_rng(0).normal(size=area.shape + (3,))
    want = np.sum(f * area[..., None]) / (np.sum(area) * 3)
    got = float(lat_weighted_mean(jnp.asarray(f), jnp.asarray(area)))
    # the grid stores fp32 areas: their sum carries ~1e-7 relative error
    assert got == pytest.approx(want, rel=1e-6)
    # Not the plain mean: the cube's cells differ in area.
    assert abs(got - float(np.mean(f))) > 1e-6


def test_shape_mismatch_raises():
    area = jnp.ones((6, 4, 4))
    with pytest.raises(ValueError):
        lat_weighted_mean(jnp.ones((6, 5, 5)), area)


def test_structured_grids_keep_their_latitude_weights():
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8, 16)
    assert grid_area_weights(g) is g.weights


# --- MPAS (user decisions Q2 / Q7) -------------------------------------------

def _mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(2, lloyd_iterations=5)


def test_mpas_default_weights_are_the_cell_areas():
    m = _mesh()
    np.testing.assert_array_equal(np.asarray(grid_area_weights(m)),
                                  np.asarray(m.areaCell))


def test_mpas_area_weighted_mean_of_a_column_field_with_levels():
    m = _mesh()
    area = np.asarray(m.areaCell)
    f = np.random.default_rng(0).normal(size=(area.size, 3))
    want = np.sum(f * area[:, None]) / (np.sum(area) * 3)
    got = float(lat_weighted_mean(jnp.asarray(f), jnp.asarray(area)))
    assert got == pytest.approx(want, rel=1e-6)
    assert abs(got - float(np.mean(f))) > 1e-6     # Voronoi areas differ


def _mpas_carry(m, u, T):
    from legoesm.training.era5_to_state import MPASCarry
    nc = int(m.nCells)
    return MPASCarry(u=jnp.asarray(u), T=jnp.asarray(T),
                     p_s=jnp.full((nc,), 1.0e5), phis=jnp.zeros((nc,)),
                     q_v=jnp.full((nc, 2), 1e-3))


def test_mpas_loss_scores_reconstructed_cell_winds_on_both_sides():
    """The wind terms equal the area-weighted MSE of the cell (u, v) that
    reconstruct_cell_velocity rebuilds from the prediction's AND the target's
    edge winds; T terms unchanged by the view."""
    from legoesm.grids.voronoi import reconstruct_cell_velocity
    from legoesm.training.losses import LossConfig, combined_loss_terms

    m = _mesh()
    rng = np.random.default_rng(1)
    ne, nc = int(m.nEdges), int(m.nCells)
    T = 250.0 + rng.normal(size=(nc, 2))
    up, ut = rng.normal(size=(ne, 2)), rng.normal(size=(ne, 2))
    pred, tgt = _mpas_carry(m, up, T), _mpas_carry(m, ut, T)
    cfg = LossConfig(normalize_by_scale=False, w_T=1.0, w_u=1.0, w_v=1.0,
                     w_q=0.0, w_ps=0.0)
    _, terms = combined_loss_terms(pred, tgt, jnp.asarray([0.3, 0.8]),
                                   grid=m, config=cfg)
    from legoesm.training.losses import level_weights
    lev = np.asarray(level_weights(jnp.asarray([0.3, 0.8]), config=cfg))
    area = np.asarray(m.areaCell)
    (pu, pv), (tu, tv) = (reconstruct_cell_velocity(jnp.asarray(x), m)
                          for x in (up, ut))
    for name, d in (("mse_u", np.asarray(pu - tu)), ("mse_v", np.asarray(pv - tv))):
        want = np.sum(d ** 2 * lev * area[:, None]) / (np.sum(area) * 2)
        assert float(terms[name]) == pytest.approx(want, rel=1e-6), name
    assert float(terms["mse_T"]) == 0.0


@pytest.mark.parametrize("over", [{"w_flux_olr": 1.0}, {"spectral_weight": 0.5},
                                  {"w_bias_flux_rsut": 1.0}])
def test_mpas_loss_refuses_flux_and_spectral_terms(over):
    from legoesm.training.losses import LossConfig, combined_loss_terms
    m = _mesh()
    c = _mpas_carry(m, np.zeros((int(m.nEdges), 2)),
                    np.full((int(m.nCells), 2), 250.0))
    with pytest.raises(ValueError, match="state only"):
        combined_loss_terms(c, c, jnp.asarray([0.3, 0.8]), grid=m,
                            config=LossConfig(**over))
