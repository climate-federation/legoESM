"""run_omip --face-wall-file: parsing and geometry attachment."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run.run_omip import attach_face_walls, parse_args


def test_flag_round_trip():
    assert parse_args([]).face_wall_file is None
    assert parse_args(["--face-wall-file", "w.npz"]).face_wall_file == "w.npz"


def test_attach_validates_and_sets_geometry():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, cfg = create_latlon_grid(4, 6), LatLonCGridOceanConfig.from_flat()
    fu, fv = np.full((4, 7), 3), np.full((5, 6), 3)
    g = attach_face_walls(grid, {"face_nlev_u": fu, "face_nlev_v": fv}, cfg)
    assert np.array_equal(g.face_nlev_u, fu) and g.n_lat == 4
    bad = fu.copy(); bad[1, 0] = 1
    with pytest.raises(ValueError, match="periodic"):
        attach_face_walls(grid, {"face_nlev_u": bad, "face_nlev_v": fv}, cfg)
    with pytest.raises(ValueError, match="shapes"):
        attach_face_walls(grid, {"face_nlev_u": fu[:, :-1], "face_nlev_v": fv}, cfg)
