"""Shared VEROS→legoESM layout bridges (ponytail dedup 2026-06-17): the 1deg
and flexible recipes now alias these instead of carrying byte-identical copies."""
from __future__ import annotations

import numpy as np

from legoesm.ocean.fidelity.veros_layout import (
    veros_xy_to_legoesm,
    veros_xyz_to_legoesm,
)


def test_xyz_shape_walls_and_zflip():
    nx, ny, nz = 5, 3, 4
    arr = np.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz)
    out = veros_xyz_to_legoesm(arr, fill=-9.0)
    assert out.shape == (ny + 2, nx, nz)            # (lat+walls, lon, z)
    assert np.all(out[0] == -9.0) and np.all(out[-1] == -9.0)  # wall rows
    # interior = transpose(1,0,2) with z reversed (k=0 deepest -> surface)
    np.testing.assert_array_equal(
        out[1:-1], np.transpose(arr, (1, 0, 2))[:, :, ::-1])


def test_xy_shape_and_transpose():
    nx, ny = 6, 4
    arr = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    out = veros_xy_to_legoesm(arr)
    assert out.shape == (ny + 2, nx)
    np.testing.assert_array_equal(out[1:-1], arr.T)
    assert np.all(out[0] == 0.0) and np.all(out[-1] == 0.0)


def test_recipes_alias_the_shared_impl():
    from legoesm.ocean.fidelity import (
        veros_global_1deg_recipe as r1,
        veros_global_flexible_recipe as rf,
    )
    assert r1.veros_xyz_to_legoesm_1deg is veros_xyz_to_legoesm
    assert r1.veros_xy_to_legoesm_1deg is veros_xy_to_legoesm
    assert rf.veros_xyz_to_legoesm_flex is veros_xyz_to_legoesm
    assert rf.veros_xy_to_legoesm_flex is veros_xy_to_legoesm
