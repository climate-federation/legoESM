"""Direct tests for the Oceananigans NetCDF reader (oceananigans_io.py).

Builds a synthetic NetCDF mimicking Oceananigans ``NetCDFOutputWriter`` output
(staggered coords, ``(time, z, y, x)`` axis order, bottom-up ``zC``) and
round-trips it through :func:`read_oceananigans_netcdf`. The reader returns RAW
arrays — convention reconciliation is the state bridge's job — so the test
asserts the on-disk axis order and bottom-up z are preserved untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.fidelity.oceananigans_io import (
    OceananigansNetCDF,
    read_oceananigans_netcdf,
)


def _write_synthetic_oceananigans_nc(path):
    """Write a tiny Oceananigans-style NetCDF: u (Face,Center,Center),
    b (Center,Center,Center) tracer, and η (Center,Center) free surface."""
    xr = pytest.importorskip("xarray")
    nt, nz, ny, nx = 2, 3, 4, 5
    # Oceananigans: zC increases UPWARD, k=1 is the bottom (z negative -> 0).
    zC = np.linspace(-1000.0, -100.0, nz)      # bottom -> surface
    yC = np.linspace(15.5, 74.5, ny)
    xC = np.linspace(-29.5, 29.5, nx)
    xF = np.linspace(-30.0, 29.0, nx)          # west faces (Face in x)
    time = np.array([0.0, 3600.0])
    rng = np.random.default_rng(0)
    u = rng.standard_normal((nt, nz, ny, nx))  # (time, zC, yC, xF)
    b = rng.standard_normal((nt, nz, ny, nx))  # (time, zC, yC, xC)
    eta = rng.standard_normal((nt, ny, nx))    # (time, yC, xC)
    ds = xr.Dataset(
        data_vars={
            "u": (("time", "zC", "yC", "xF"), u),
            "b": (("time", "zC", "yC", "xC"), b),
            "eta": (("time", "yC", "xC"), eta),
        },
        coords={
            "time": time, "zC": zC, "yC": yC, "xC": xC, "xF": xF,
        },
        attrs={"Oceananigans": "synthetic-test"},
    )
    ds.to_netcdf(path)
    return dict(nt=nt, nz=nz, ny=ny, nx=nx, zC=zC, xF=xF, u=u, b=b, eta=eta,
                time=time)


def test_round_trip_all_variables(tmp_path):
    pytest.importorskip("xarray")
    nc = tmp_path / "synthetic.nc"
    truth = _write_synthetic_oceananigans_nc(nc)

    out = read_oceananigans_netcdf(nc)
    assert isinstance(out, OceananigansNetCDF)
    # All three data variables loaded, on-disk axis order PRESERVED (raw).
    assert set(out.variables) == {"u", "b", "eta"}
    np.testing.assert_allclose(out.variables["u"], truth["u"])
    np.testing.assert_allclose(out.variables["b"], truth["b"])
    np.testing.assert_allclose(out.variables["eta"], truth["eta"])
    assert out.variables["u"].shape == (truth["nt"], truth["nz"],
                                        truth["ny"], truth["nx"])
    # Coords present, including BOTH xC and xF (staggered).
    assert "xF" in out.coords and "xC" in out.coords
    np.testing.assert_allclose(out.coords["xF"], truth["xF"])
    # Bottom-up z preserved untouched (NOT reversed here — bridge's job).
    np.testing.assert_allclose(out.coords["zC"], truth["zC"])
    assert out.coords["zC"][0] < out.coords["zC"][-1]  # bottom -> surface
    # Times + sizes.
    np.testing.assert_allclose(out.times, truth["time"])
    assert out.sizes["time"] == truth["nt"]
    assert out.sizes["zC"] == truth["nz"]
    assert out.attrs.get("Oceananigans") == "synthetic-test"


def test_variable_whitelist(tmp_path):
    pytest.importorskip("xarray")
    nc = tmp_path / "synthetic.nc"
    _write_synthetic_oceananigans_nc(nc)
    out = read_oceananigans_netcdf(nc, variables=("u", "eta"))
    assert set(out.variables) == {"u", "eta"}


def test_missing_variable_raises(tmp_path):
    pytest.importorskip("xarray")
    nc = tmp_path / "synthetic.nc"
    _write_synthetic_oceananigans_nc(nc)
    with pytest.raises(KeyError, match="not in"):
        read_oceananigans_netcdf(nc, variables=("u", "does_not_exist"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_oceananigans_netcdf(tmp_path / "nope.nc")
