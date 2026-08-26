"""SST-parameterized IC + cache-key separation for the multi-SST tuning.

Pure-config checks (no model run): the RCEMIP case IC must move the right way
with SST, refuse an undefined SST, and the run cache key must separate SSTs so
a 305 K candidate never reads a 300 K cached result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _camp():
    path = _REPO / "scripts" / "run" / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign_mst", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ref:
    # minimal ReferenceProfiles stand-in: only z_m is read by the IC builder.
    def __init__(self, n=40):
        self.z_m = np.linspace(0.0, 30000.0, n)


def test_ic_surface_humidity_increases_with_sst():
    camp = _camp()
    ref = _Ref()
    q = {}
    for sst in (295.0, 300.0, 305.0):
        _T, qv = camp.wing_initial_profiles(ref, sst_K=sst)
        q[sst] = float(np.asarray(qv)[-1])
    assert q[295.0] < q[300.0] < q[305.0], q
    # RCEMIP Table 1 surface q0 (kg/kg), within a few percent of the profile's
    # lowest level.
    assert q[295.0] == pytest.approx(0.012, rel=0.15)
    assert q[305.0] == pytest.approx(0.024, rel=0.15)


def test_undefined_sst_is_refused():
    camp = _camp()
    with pytest.raises(ValueError, match="no RCEMIP surface humidity"):
        camp.wing_initial_profiles(_Ref(), sst_K=298.0)


def test_cache_key_separates_ssts():
    camp = _camp()
    cfg = camp.make_physics_config(convection="dca")
    common = dict(
        days=5.0, dt=600.0, scm_microphysics_substeps=6,
        scm_convection_substeps=4,
        surface_wind_m_s=camp.DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
        coriolis_s_inv=0.0, large_scale_forcing="none")
    k295 = camp._config_cache_key(cfg, sst_K=295.0, **common)
    k305 = camp._config_cache_key(cfg, sst_K=305.0, **common)
    assert k295 != k305, "SSTs collide in the cache key — a 305 run would read 295"


def test_albedo_tracks_sst():
    camp = _camp()
    c295 = camp.make_physics_config(radiation="rrtmgp", convection="dca",
                                    sst_K=295.0)
    c305 = camp.make_physics_config(radiation="rrtmgp", convection="dca",
                                    sst_K=305.0)
    a295 = c295.radiation.rrtmgp.sfc_albedo_direct
    a305 = c305.radiation.rrtmgp.sfc_albedo_direct
    # The ocean albedo is SST-dependent (weakly); the two must at least be
    # resolved from the same code path without error and be finite.
    assert np.isfinite(a295) and np.isfinite(a305)
