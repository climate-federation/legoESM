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
