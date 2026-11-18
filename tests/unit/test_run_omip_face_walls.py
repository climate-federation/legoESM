"""run_omip --face-wall-file: parsing and geometry attachment."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run.run_omip import attach_face_walls, parse_args


def test_flag_round_trip():
    assert parse_args([]).face_wall_file is None
    assert parse_args(["--face-wall-file", "w.npz"]).face_wall_file == "w.npz"


Z = dict(dz_ref=np.full(3, 100.0), H_min=10.0)     # every open depth far above H_min


def test_attach_validates_and_sets_geometry():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, cfg = create_latlon_grid(4, 6), LatLonCGridOceanConfig.from_flat()
    fu, fv = np.full((4, 7), 3), np.full((5, 6), 3)
    g = attach_face_walls(grid, {"face_nlev_u": fu, "face_nlev_v": fv}, cfg, **Z)
    assert np.array_equal(g.face_nlev_u, fu) and g.n_lat == 4
    bad = fu.copy(); bad[1, 0] = 1
    with pytest.raises(ValueError, match="periodic"):
        attach_face_walls(grid, {"face_nlev_u": bad, "face_nlev_v": fv}, cfg, **Z)
    with pytest.raises(ValueError, match="shapes"):
        attach_face_walls(grid, {"face_nlev_u": fu[:, :-1], "face_nlev_v": fv}, cfg, **Z)


@pytest.mark.parametrize("missing", ["face_nlev_u", "face_nlev_v"])
def test_attach_refuses_half_configured_walls(missing):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.state import LatLonCGridOceanConfig
    walls = {"face_nlev_u": np.full((4, 7), 3), "face_nlev_v": np.full((5, 6), 3)}
    del walls[missing]
    with pytest.raises(ValueError, match="face_nlev_u and face_nlev_v"):
        attach_face_walls(create_latlon_grid(4, 6), walls, LatLonCGridOceanConfig.from_flat(), **Z)


def test_attach_closes_faces_shallower_than_the_minimum_depth():
    """A walled face with less open depth than the run's minimum ocean depth is closed."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.state import LatLonCGridOceanConfig
    dz = np.array([1.0, 2.0, 4.0, 8.0])                 # open depths 1, 3, 7, 15 m
    fu, fv = np.full((4, 7), 4), np.full((5, 6), 4)
    fu[1, 2], fu[2, 3], fu[3, 4] = 1, 3, 2              # 1 m, 7 m, 3 m open
    fv[2, 1], fv[3, 5] = 1, 3
    g = attach_face_walls(create_latlon_grid(4, 6), {"face_nlev_u": fu, "face_nlev_v": fv},
                          LatLonCGridOceanConfig.from_flat(), dz_ref=dz, H_min=6.9)
    gu, gv = np.asarray(g.face_nlev_u), np.asarray(g.face_nlev_v)
    assert gu[1, 2] == 0 and gu[3, 4] == 0 and gv[2, 1] == 0      # below 6.9 m: closed
    assert gu[2, 3] == 3 and gv[3, 5] == 3                        # 7 m, just above: unchanged
    assert (gu == 4).sum() == fu.size - 3 and (gv == 4).sum() == fv.size - 2
    assert fu[1, 2] == 1                                          # the caller's array is not edited
