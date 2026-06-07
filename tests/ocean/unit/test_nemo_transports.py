"""Unit test for the NEMO AMOC reference reader's pure core (amoc_core).

``scripts/validate/nemo_transports.amoc_core`` computes the Atlantic MOC
strength from pure arrays (split out from the NetCDF I/O for testability).
Exercise it on a SYNTHETIC analytic overturning with a known streamfunction
peak, the Atlantic longitude mask, and the deep-AABW exclusion.  Pure NumPy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load the validator script by path (scripts/ is not an importable package).
_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "nemo_transports", _ROOT / "scripts" / "validate" / "nemo_transports.py")
_nt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nt)


def _synthetic(nz=10, ny=20, nx=8, j_target=None, atlantic=True, deep_cell=False):
    """Build (voe3, e1v, gphiv, glamv, depthv) with an analytic overturning at
    the row nearest 26.5 N: a(z) = +1 (upper half) / -1 (lower half), e1v=1e6,
    nx cells in band -> ψ = -8·cumsum(a) [Sv], peak |ψ-ψ_surf| = 8*nz/2*... ."""
    depthv = np.array([100., 300., 600., 1000., 1500., 2200., 2800.,
                       3500., 4200., 5000.])[:nz]
    lat_axis = np.linspace(-30.0, 60.0, ny)
    gphiv = np.broadcast_to(lat_axis[:, None], (ny, nx)).copy()
    lon = -35.0 if atlantic else 160.0                    # in / out of [-75,15]
    glamv = np.full((ny, nx), lon)
    e1v = np.full((ny, nx), 1.0e6)
    j = int(np.argmin(np.abs(lat_axis - 26.5))) if j_target is None else j_target
    a = np.where(np.arange(nz) < nz // 2, 1.0, -1.0)      # + upper, - lower
    voe3 = np.zeros((nz, ny, nx))
    voe3[:, j, :] = a[:, None]
    if deep_cell:
        # A strong abyssal overturning below 3000 m that must be EXCLUDED.
        voe3[7:, j, :] = 10.0
    return voe3, e1v, gphiv, glamv, depthv, j


def test_amoc_core_analytic_peak():
    voe3, e1v, gphiv, glamv, depthv, j = _synthetic()
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    # ψ = -cumsum(8e6·a)/1e6 = -8·cumsum(a); a=[+1×5,-1×5] -> cumsum peaks at 5,
    # ψ' = ψ-ψ[0]; peak |ψ'| in upper (depth<3000) = 8*(5-1) = 32 Sv at z=4.
    assert r["amoc_Sv"] == np.float64(32.0) or abs(r["amoc_Sv"] - 32.0) < 1e-9
    assert abs(r["row_lat_deg"] - 26.5) < 5.0
    assert r["j"] == j


def test_amoc_core_atlantic_mask_excludes_pacific():
    """A Pacific-longitude overturning (outside the Atlantic band) -> ~0 AMOC."""
    voe3, e1v, gphiv, glamv, depthv, _ = _synthetic(atlantic=False)
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    assert r["amoc_Sv"] < 1e-9


def test_amoc_core_excludes_deep_aabw_cell():
    """A huge abyssal (>3000 m) overturning is excluded; AMOC stays the upper
    32 Sv, not the deep cell."""
    voe3, e1v, gphiv, glamv, depthv, _ = _synthetic(deep_cell=True)
    r = _nt.amoc_core(voe3, e1v, gphiv, glamv, depthv, target_lat=26.5)
    assert abs(r["amoc_Sv"] - 32.0) < 1e-9
    assert r["depth_of_max_m"] < 3000.0


# ---------------------------------------------------------------------------
# ACC@Drake (acc_drake_core) — fixed-i meridian section, signed, regularity-checked
# ---------------------------------------------------------------------------
_LON_ACC = np.array([-90., -80., -70., -68., -66., -60., -55., -50.])  # -68 exact
_E2U = 1.0e5     # u-face meridional width [m] (100 km)
_U_COL = 200.0   # depth-integrated zonal transport per U-point [m^2/s]
_BAND = (-65.0, -45.0)   # default Drake band (matches diagnostics_climate)


def _synthetic_acc(nz=5, sign=1.0, extra_lon_col=None, lon_shear_deg_per_row=0.0):
    """(uoe3, e2u, gphiu, glamu, lat_axis) with eastward flow concentrated at the
    Drake meridian (lon = -68, exact in ``_LON_ACC``).  Depth-integrated u at
    that column = ``sign*_U_COL`` for every row; e2u = _E2U everywhere.
    ``lon_shear_deg_per_row`` tilts glamu with the j-index to emulate a
    CURVILINEAR grid where a constant i is no longer a meridian."""
    lat_axis = np.linspace(-80.0, -40.0, 20)
    nx = _LON_ACC.size
    ny = lat_axis.size
    gphiu = np.broadcast_to(lat_axis[:, None], (ny, nx)).copy()
    glamu = np.broadcast_to(_LON_ACC[None, :], (ny, nx)).copy()
    if lon_shear_deg_per_row:
        glamu = glamu + lon_shear_deg_per_row * (np.arange(ny)[:, None] - ny // 2)
    e2u = np.full((ny, nx), _E2U)
    i_drake = int(np.argmin(np.abs(_LON_ACC - (-68.0))))
    uoe3 = np.zeros((nz, ny, nx))
    uoe3[:, :, i_drake] = sign * _U_COL / nz                # Σ_z = sign*_U_COL
    if extra_lon_col is not None:                            # flow off the section
        uoe3[:, :, extra_lon_col] = 5.0 * _U_COL / nz
    return uoe3, e2u, gphiu, glamu, lat_axis


def _n_band_rows(lat_axis, lat_south=_BAND[0], lat_north=_BAND[1]):
    return int(((lat_axis >= lat_south) & (lat_axis <= lat_north)).sum())


def test_acc_core_analytic_transport():
    """Uniform eastward Drake-section flow -> SIGNED transport = n_band·U_col·e2u,
    on the fixed -68 column with zero longitude deviation (regular grid)."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc()
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0)
    expect = _n_band_rows(lat_axis) * _U_COL * _E2U / 1.0e6
    assert r["n_rows"] == _n_band_rows(lat_axis)
    assert abs(r["acc_Sv"] - expect) < 1e-9                  # signed, positive
    assert abs(r["section_lon_deg"] - (-68.0)) < 1e-9
    assert r["max_lon_dev_deg"] < 1e-9                       # regular -> no drift


def test_acc_core_picks_only_section_meridian():
    """Flow at a column OFF the fixed Drake meridian is NOT summed."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc(extra_lon_col=0)  # lon -90
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0)
    expect = _n_band_rows(lat_axis) * _U_COL * _E2U / 1.0e6
    assert abs(r["acc_Sv"] - expect) < 1e-9                  # extra column ignored


def test_acc_core_band_excludes_out_of_band_rows():
    """A narrower band cuts the transport proportionally (fewer section rows)."""
    uoe3, e2u, gphiu, glamu, lat_axis = _synthetic_acc()
    r = _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0,
                           lat_south=-60.0, lat_north=-58.0)
    n = _n_band_rows(lat_axis, -60.0, -58.0)
    assert r["n_rows"] == n
    assert abs(r["acc_Sv"] - n * _U_COL * _E2U / 1.0e6) < 1e-9
    assert n < _n_band_rows(lat_axis)                        # genuinely narrower


def test_acc_core_sign_preserved():
    """Sign is PRESERVED (not abs): eastward -> +, westward -> -, equal |.|.
    A reversed U convention would surface as a sign flip, not be hidden."""
    pos = _nt.acc_drake_core(*_synthetic_acc(sign=+1.0)[:4], drake_lon=-68.0)
    neg = _nt.acc_drake_core(*_synthetic_acc(sign=-1.0)[:4], drake_lon=-68.0)
    assert pos["acc_Sv"] > 0.0
    assert neg["acc_Sv"] < 0.0
    assert abs(pos["acc_Sv"] + neg["acc_Sv"]) < 1e-9        # equal magnitude


def test_acc_core_circular_drake_lon():
    """drake_lon given as +292 (== -68 mod 360) selects the same section."""
    args = _synthetic_acc()[:4]
    a = _nt.acc_drake_core(*args, drake_lon=-68.0)
    b = _nt.acc_drake_core(*args, drake_lon=292.0)
    assert a["i_col"] == b["i_col"]
    assert abs(a["acc_Sv"] - b["acc_Sv"]) < 1e-9


def test_acc_core_rejects_curvilinear_section():
    """On a CURVILINEAR grid (longitude shears with j so a constant i is not a
    meridian) the fixed-i section RAISES instead of silently mis-sampling."""
    import pytest
    uoe3, e2u, gphiu, glamu, _ = _synthetic_acc(lon_shear_deg_per_row=2.0)
    with pytest.raises(ValueError, match="not regular"):
        _nt.acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=-68.0,
                           max_lon_dev_deg=5.0)
