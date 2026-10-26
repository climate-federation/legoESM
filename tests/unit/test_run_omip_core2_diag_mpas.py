"""``_diag`` locates the 3-D max|u| on MPAS edges (u is (nEdges, nlev))."""
from types import SimpleNamespace

import numpy as np

from scripts.run.run_omip_core2 import _diag


def _leaf(a):
    return SimpleNamespace(data=np.asarray(a))


def test_diag_locates_max_edge_on_mpas():
    n_cells, n_edges, nlev = 4, 6, 3
    u = np.zeros((n_edges, nlev)); u[4, 2] = -2.5
    state = SimpleNamespace(
        T=_leaf(np.full((n_cells, nlev), 10.0)),
        S=_leaf(np.full((n_cells, nlev), 35.0)),
        u=_leaf(u), v=None,
        land_mask=_leaf(np.ones(n_cells)))
    lat_e = np.array([0., 10., 20., 30., -45.5, 60.]); lon_e = np.arange(6) * 30.0
    d = _diag(state, lat2d=np.zeros(n_cells), lon2d=np.zeros(n_cells),
              edge_latlon=(lat_e, lon_e))
    assert d["max_abs_u"] == 2.5
    assert (d["umax_lat"], d["umax_lon"], d["umax_lev"]) == (-45.5, 120.0, 2)


def test_diag_without_edges_leaves_location_nan():
    u = np.zeros((6, 3)); u[1, 0] = 1.0
    state = SimpleNamespace(T=_leaf(np.zeros((4, 3))), S=_leaf(np.zeros((4, 3))),
                            u=_leaf(u), v=None, land_mask=_leaf(np.ones(4)))
    d = _diag(state, lat2d=np.zeros(4), lon2d=np.zeros(4))
    assert np.isnan(d["umax_lat"]) and d["umax_lev"] == -1
