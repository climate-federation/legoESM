"""Equatorial A_h shaping through the CORE-II driver: ``--A-h-eq-boost`` now
accepts REDUCTIONS (< 1, the NEMO ORCA1 eddy_viscosity_3D shape: 20000 m2/s
midlatitude -> 1000 at the equator) and ``--A-h-eq-sigma-deg`` sets the ramp
width.  Covers the argparse round-trip and the ``replace_flat`` threading into
the nested LateralViscosityConfig — a flag that parses but never reaches the
config is the silently-ignored-flag footgun.
"""

from __future__ import annotations

from scripts.run.run_omip_core2 import _build_arg_parser
from legoesm.ocean.state import LatLonCGridOceanConfig


def test_cli_round_trip():
    p = _build_arg_parser()
    a = p.parse_args([])
    assert a.A_h_eq_boost is None and a.A_h_eq_sigma_deg is None
    a = p.parse_args(["--A-h-eq-boost", "0.05", "--A-h-eq-sigma-deg", "7.0"])
    assert a.A_h_eq_boost == 0.05
    assert a.A_h_eq_sigma_deg == 7.0


def test_replace_flat_routes_reduction_into_lateral_viscosity():
    cfg = LatLonCGridOceanConfig()
    out = cfg.replace_flat(A_h=20000.0, A_h_eq_boost=0.05,
                           A_h_eq_sigma_deg=7.0)
    assert out.lateral_viscosity.A_h == 20000.0
    assert out.lateral_viscosity.A_h_eq_boost == 0.05
    assert out.lateral_viscosity.A_h_eq_sigma_deg == 7.0
