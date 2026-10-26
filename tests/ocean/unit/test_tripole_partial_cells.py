"""The tripole lane's NEMO partial-cell bottom geometry.

Why this exists: the z* coordinate stretches the WHOLE reference column into
every water column, so feeding it NEMO's 75-level grid gave the shallowest
shelf column of eORCA1 (24.58 m, 72.2 N 73.6 E) a 4.2 mm surface layer, and one
step of ordinary surface heating drove that cell past 1000 C.  Partial cells
keep the reference thicknesses and cut only the bottom cell, which is what NEMO
does.  These tests pin the two pieces that fix it:

* ``read_mesh_vertical_1d`` -> the mesh's own e3t_1d / gdept_1d / gdepw_1d,
  validated rather than reconstructed (the T-point depths decide which level is
  the bottom one on a stretched grid).
* ``_snap_thin_partial_cells`` -> bathymetry rounded up to the interface above
  wherever the bottom partial cell would be too thin, land mask kept consistent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_RUN_OMIP = Path(__file__).resolve().parents[3] / "scripts" / "run" / "run_omip.py"


def _run_omip():
    spec = importlib.util.spec_from_file_location("_run_omip_pc", _RUN_OMIP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_mesh(tmp_path, e3t_1d, gdept_1d, gdepw_1d):
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(
        {
            "e3t_1d": ("z", np.asarray(e3t_1d, dtype=np.float64)),
            "gdept_1d": ("zt", np.asarray(gdept_1d, dtype=np.float64)),
            "gdepw_1d": ("zw", np.asarray(gdepw_1d, dtype=np.float64)),
        }
    )
    p = tmp_path / "mesh_zgr.nc"
    ds.to_netcdf(p)
    return p


def _nemo_like(n=6):
    """A stretched NEMO-like reference grid: thin at the top, thick below.

    The T-point depths are deliberately NOT the midpoints of the interfaces.
    Real NEMO builds both ladders from an analytic stretching function, so
    gdept sits above the midpoint on a stretched grid; a reader that
    reconstructed the T ladder from the thicknesses would agree with the
    midpoints and disagree with this, which is what the reader test checks.
    """
    e3t = np.array([1.0, 1.2, 2.0, 5.0, 20.0, 60.0])[:n]
    gdepw = np.concatenate([[0.0], np.cumsum(e3t)])
    mid = 0.5 * (gdepw[:-1] + gdepw[1:])
    gdept = gdepw[:-1] + 0.4 * e3t          # 40% into each layer, not 50%
    assert not np.allclose(gdept, mid)
    return e3t, gdept, gdepw


def test_read_mesh_vertical_1d_returns_mesh_arrays(tmp_path):
    from legoesm.ocean.init_tripole import read_mesh_vertical_1d

    e3t, gdept, gdepw = _nemo_like()
    p = _write_mesh(tmp_path, e3t, gdept, gdepw)
    got_e3t, got_gdept, got_gdepw = read_mesh_vertical_1d(str(p))
    assert np.allclose(got_e3t, e3t)
    # Read, not reconstructed: the T-point depths are NOT the midpoints of a
    # naive cumulative sum on a stretched grid, and a caller that rebuilt them
    # would place the bottom level differently.
    assert np.allclose(got_gdept, gdept)
    assert np.allclose(got_gdepw, gdepw)


def test_read_mesh_vertical_1d_rejects_broken_grids(tmp_path):
    from legoesm.ocean.init_tripole import read_mesh_vertical_1d

    e3t, gdept, gdepw = _nemo_like()
    bad = gdept.copy()
    bad[3] = bad[2]                      # not strictly increasing
    p = _write_mesh(tmp_path, e3t, bad, gdepw)
    with pytest.raises(ValueError, match="strictly increasing"):
        read_mesh_vertical_1d(str(p))

    neg = e3t.copy()
    neg[2] = -1.0
    (tmp_path / "sub").mkdir()
    p2 = _write_mesh(tmp_path / "sub", neg, gdept, gdepw)
    with pytest.raises(ValueError, match="non-positive thickness"):
        read_mesh_vertical_1d(str(p2))

    (tmp_path / "sub2").mkdir()
    p3 = _write_mesh(tmp_path / "sub2", e3t, gdept[:-1], gdepw)
    with pytest.raises(ValueError, match="inconsistent"):
        read_mesh_vertical_1d(str(p3))

    (tmp_path / "sub3").mkdir()
    p4 = _write_mesh(tmp_path / "sub3", e3t, gdept, gdepw[:3])
    with pytest.raises(ValueError, match="inconsistent"):
        read_mesh_vertical_1d(str(p4))


def test_snap_thin_partial_cells_rounds_up_and_relands():
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    ro = _run_omip()
    e3t, gdept, _ = _nemo_like()
    z = create_z_star_from_thicknesses(e3t, gdept,
                                       nemo_e3w_source="depth_difference")
    edges = np.concatenate([[0.0], np.cumsum(e3t)])   # 0, 1, 2.2, 4.2, 9.2, 29.2, 89.2

    # col 0: 0.2 m into a 1.2 m layer -> 17% -> too thin -> snaps up to 1.0
    # col 1: 0.9 m into the same layer -> 75% -> kept
    # col 2: 0.1 m into the FIRST layer -> snaps to 0 -> becomes land
    # col 3: exactly on an interface -> untouched
    H = np.array([1.2, 1.9, 0.1, 9.2])
    lm = np.ones(4)
    H_out, lm_out = ro._snap_thin_partial_cells(H, lm, z)
    H_out = np.asarray(H_out)
    lm_out = np.asarray(lm_out)

    assert H_out[0] == pytest.approx(edges[1])       # snapped up to 1.0
    assert H_out[1] == pytest.approx(1.9)            # thick enough, untouched
    assert H_out[2] == pytest.approx(0.0)            # snapped to the surface
    assert H_out[3] == pytest.approx(9.2)            # on an interface
    assert lm_out[2] == 0.0                          # and therefore land
    assert list(lm_out[[0, 1, 3]]) == [1.0, 1.0, 1.0]


def test_partial_cells_keep_reference_thickness_where_zstar_squeezes():
    """The defect this whole path exists for, in three lines of arithmetic."""
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.vertical import (
        compute_layer_thickness, create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    e3t, gdept, _ = _nemo_like()
    z = create_z_star_from_thicknesses(e3t, gdept,
                                       nemo_e3w_source="depth_difference")
    H_shallow = 3.0                                   # a shelf column
    H = jnp.asarray([[H_shallow]])
    eta = jnp.zeros((1, 1))

    # z*: the whole 89.2 m reference column is squeezed into 3 m, so the
    # surface layer collapses far below its reference 1.0 m.
    h_zstar = np.asarray(compute_layer_thickness(eta, H, z))[0, 0, 0]
    assert h_zstar < 0.1 * e3t[0]

    # Partial cells: the surface layer keeps its reference thickness and only
    # the bottom cell is cut.
    pc = create_partial_cell_coordinate(z, H, bottom_index_rule="nemo_tpoint")
    h_pc = np.asarray(compute_layer_thickness(eta, H, pc))[0, 0]
    assert h_pc[0] == pytest.approx(e3t[0], rel=1e-6)
    # NEMO's zgr_zps picks the bottom level from the T-point depths and clips
    # the bottom cell at that reference interface, so the column depth is
    # reproduced to within one reference layer, never stretched.
    wet_sum = float(h_pc[h_pc > 0.0].sum())
    assert wet_sum <= H_shallow + 1e-9
    assert H_shallow - wet_sum < e3t[2]
    # every wet layer above the bottom one keeps its reference thickness
    n_wet = int((h_pc > 0.0).sum())
    assert np.allclose(h_pc[: n_wet - 1], e3t[: n_wet - 1], rtol=1e-6)

def test_nemo_tpoint_rule_differs_from_the_interface_rule():
    """Deleting ``bottom_index_rule="nemo_tpoint"`` must not go unnoticed.

    The two rules pick a different bottom level on a stretched grid, so a
    driver that dropped the argument (falling back to the legacy interface
    rule) would place the seafloor one level away from where NEMO puts it.
    """
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses,
    )

    e3t, gdept, _ = _nemo_like()
    z = create_z_star_from_thicknesses(e3t, gdept,
                                       nemo_e3w_source="depth_difference")
    # depths spread across the stretched part of the column
    H = jnp.asarray([[2.5, 4.0, 8.0, 25.0]])
    nemo = create_partial_cell_coordinate(z, H, bottom_index_rule="nemo_tpoint")
    legacy = create_partial_cell_coordinate(z, H)          # interface rule
    assert not np.array_equal(np.asarray(nemo.bottom_level),
                              np.asarray(legacy.bottom_level))


def test_driver_exposes_the_partial_cell_switch():
    """The lane default is partial cells; the escape hatch must round-trip."""
    ro = _run_omip()
    assert ro.parse_args(["--grid", "tripole"]).tripole_partial_cells is True
    off = ro.parse_args(["--grid", "tripole", "--no-tripole-partial-cells"])
    assert off.tripole_partial_cells is False


def test_snap_clamps_columns_deeper_than_the_reference_column():
    """A column deeper than the deepest interface would index past the grid."""
    pytest.importorskip("jax.numpy")
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    ro = _run_omip()
    e3t, gdept, _ = _nemo_like()
    z = create_z_star_from_thicknesses(e3t, gdept,
                                       nemo_e3w_source="depth_difference")
    total = float(np.sum(e3t))
    H_out, _ = ro._snap_thin_partial_cells(
        np.array([total + 25.0, total, 9.2]), np.ones(3), z)
    assert float(np.asarray(H_out)[0]) == pytest.approx(total)
    assert float(np.asarray(H_out)[1]) == pytest.approx(total)


def test_reader_rejects_sheared_and_degenerate_ladders(tmp_path):
    from legoesm.ocean.init_tripole import read_mesh_vertical_1d

    e3t, gdept, gdepw = _nemo_like()
    (tmp_path / "a").mkdir()
    zero = e3t.copy(); zero[3] = 0.0
    with pytest.raises(ValueError, match="non-positive thickness"):
        read_mesh_vertical_1d(str(_write_mesh(tmp_path / "a", zero, gdept, gdepw)))

    # T depths from a DIFFERENT stretching than the thicknesses: same level
    # count, silently sheared coordinate before this check existed.
    (tmp_path / "b").mkdir()
    uniform = np.full_like(e3t, float(np.sum(e3t)) / e3t.size)
    sheared = np.cumsum(uniform) - 0.5 * uniform      # a uniform grid's T depths
    with pytest.raises(ValueError, match="different vertical"):
        read_mesh_vertical_1d(
            str(_write_mesh(tmp_path / "b", e3t, sheared, gdepw)))
