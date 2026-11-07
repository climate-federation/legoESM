"""Tests for the zdfiwm forcing-map loader (``ocean/iwm_forcing.py``).

Covers: file-layout validation, same-mesh passthrough, nearest-neighbour
regrid + global power-total preservation, decay-scale guarding and the
``hcri_inv`` inversion, and land-mask application.
"""

from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from legoesm.ocean.iwm_forcing import load_iwm_forcing, read_iwm_file  # noqa: E402


def _write_file(path, lon2d, lat2d, values: dict):
    with netCDF4.Dataset(path, "w") as ds:
        ny, nx = lat2d.shape
        ds.createDimension("time", 1)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        for name, arr in (("nav_lon", lon2d), ("nav_lat", lat2d)):
            v = ds.createVariable(name, "f8", ("y", "x"))
            v[:] = arr
        for name, arr in values.items():
            v = ds.createVariable(name, "f8", ("time", "y", "x"))
            v[0] = arr


def _source(ny=12, nx=24):
    lon1 = np.linspace(-180.0 + 7.5, 180.0 - 7.5, nx)
    lat1 = np.linspace(-75.0, 75.0, ny)
    lat2d, lon2d = np.meshgrid(lat1, lon1, indexing="ij")
    rng = np.random.default_rng(7)
    vals = {
        "power_bot": rng.uniform(0.0, 2e-3, (ny, nx)),
        "power_cri": rng.uniform(0.0, 1e-3, (ny, nx)),
        "power_nsq": rng.uniform(0.0, 3e-3, (ny, nx)),
        "power_sho": rng.uniform(0.0, 1e-3, (ny, nx)),
        "scale_bot": rng.uniform(50.0, 500.0, (ny, nx)),
        "scale_cri": rng.uniform(50.0, 300.0, (ny, nx)),
    }
    # a couple of land cells: zero power + zero scale (guarded downstream)
    for v in vals.values():
        v[0, 0] = 0.0
        v[5, 3] = 0.0
    return lon2d, lat2d, vals


def test_missing_variable_raises(tmp_path):
    lon2d, lat2d, vals = _source()
    vals.pop("power_sho")
    p = tmp_path / "bad.nc"
    _write_file(p, lon2d, lat2d, vals)
    with pytest.raises(ValueError, match="power_sho"):
        read_iwm_file(str(p))


def test_same_mesh_passthrough(tmp_path):
    lon2d, lat2d, vals = _source()
    p = tmp_path / "iwm.nc"
    _write_file(p, lon2d, lat2d, vals)
    f = load_iwm_forcing(str(p), lat2d, lon2d)
    np.testing.assert_allclose(np.asarray(f.ensq), vals["power_nsq"])
    np.testing.assert_allclose(np.asarray(f.hbot)[1, 1], vals["scale_bot"][1, 1])
    # hcri is stored INVERTED (zdfiwm_init: hcri = 1/scale_cri)
    np.testing.assert_allclose(
        np.asarray(f.hcri_inv)[2, 2], 1.0 / vals["scale_cri"][2, 2])
    # zero decay scales are guarded to the NEMO default 100 m
    assert float(np.asarray(f.hbot)[0, 0]) == pytest.approx(100.0)
    assert float(np.asarray(f.hcri_inv)[0, 0]) == pytest.approx(1.0 / 100.0)


def test_regrid_preserves_global_power_totals(tmp_path):
    lon2d, lat2d, vals = _source()
    p = tmp_path / "iwm.nc"
    _write_file(p, lon2d, lat2d, vals)
    lat_t = np.linspace(-80.0, 80.0, 17)
    lon_t = np.linspace(-180.0 + 5.0, 180.0 - 5.0, 36)
    f = load_iwm_forcing(str(p), lat_t, lon_t)
    lat2d_t, _ = np.meshgrid(lat_t, lon_t, indexing="ij")
    for name, field in (("power_bot", f.ebot), ("power_cri", f.ecri),
                        ("power_nsq", f.ensq), ("power_sho", f.esho)):
        src_total = float((vals[name] * np.cos(np.deg2rad(lat2d))).sum())
        tgt_total = float(
            (np.asarray(field) * np.cos(np.deg2rad(lat2d_t))).sum())
        np.testing.assert_allclose(tgt_total, src_total, rtol=1e-10,
                                   err_msg=name)
    assert np.all(np.asarray(f.hbot) > 0.0)
    assert np.all(np.asarray(f.hcri_inv) > 0.0)


def test_land_mask_zeroes_power_only(tmp_path):
    lon2d, lat2d, vals = _source()
    p = tmp_path / "iwm.nc"
    _write_file(p, lon2d, lat2d, vals)
    mask = np.ones(lat2d.shape)
    mask[3, :] = 0.0
    f = load_iwm_forcing(str(p), lat2d, lon2d, land_mask=mask)
    assert float(np.abs(np.asarray(f.ensq)[3]).max()) == 0.0
    assert np.all(np.asarray(f.hbot)[3] > 0.0)   # scales stay unmasked


def test_paired_cells_mpas(tmp_path):
    """paired_cells=True: 1-D lat/lon are (nCells,) centres, not axes.

    Guards the MPAS wiring: no outer-product meshgrid, and the power-total
    renorm pairs true target areas [m^2] with an ESTIMATED source area (both
    m^2) — cos-lat vs m^2 would rescale the TW totals by ~1/cell-area.
    """
    from legoesm import constants

    lon2d, lat2d, vals = _source()
    p = tmp_path / "iwm.nc"
    _write_file(p, lon2d, lat2d, vals)

    rng = np.random.default_rng(11)
    n_cells = 400
    cell_lat = rng.uniform(-78.0, 78.0, n_cells)
    cell_lon = rng.uniform(-180.0, 180.0, n_cells)
    cell_area = np.full(n_cells, 8.0e9)

    f = load_iwm_forcing(str(p), cell_lat, cell_lon,
                         paired_cells=True, target_area=cell_area)
    # paired: (nCells,), never the (nCells, nCells) meshgrid cross
    assert np.asarray(f.ensq).shape == (n_cells,)
    from legoesm.ocean.forcing.curvilinear_regrid import (
        estimate_curvilinear_cell_area,
    )
    src_area = estimate_curvilinear_cell_area(lat2d, lon2d)
    for name, field in (("power_bot", f.ebot), ("power_nsq", f.ensq)):
        src_total = float((vals[name] * src_area).sum())
        tgt_total = float((np.asarray(field) * cell_area).sum())
        np.testing.assert_allclose(tgt_total, src_total, rtol=1e-10,
                                   err_msg=name)
    assert np.all(np.asarray(f.hbot) > 0.0)


def test_source_area_without_target_area_rejected(tmp_path):
    lon2d, lat2d, vals = _source()
    p = tmp_path / "iwm.nc"
    _write_file(p, lon2d, lat2d, vals)
    with pytest.raises(ValueError, match="mixed weight units"):
        load_iwm_forcing(str(p), np.linspace(-80, 80, 17),
                         np.linspace(-175, 175, 36),
                         source_area=np.ones_like(lat2d))
