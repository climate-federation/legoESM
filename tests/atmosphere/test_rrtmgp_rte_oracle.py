"""Oracle match: production RRTMGP clear-sky fluxes vs the latest RTE+RRTMGP.

Fixture ``oracle_fixtures/rrtmgp_rte_clear.npz``: 20 RFMIP columns on a CAM-like
top with the overhead layer, sun angles including twilight, fluxes from RUNNING
unpatched rte-rrtmgp d4b3c51 (generator and provenance in its ``meta``).  The
two-stream LW solver is called one g-point at a time because
mo_rte_solver_kernels.F90:422 drops the g-point index of lev_source.

Cloud optics are out of scope here: test_rrtmgp_cloud_optics_oracle.py pins them.

KNOWN DIFFERENCE, LW only: RTE zeroes layer sources for tau <= 1e-8 and its
cancelling form loses up to ~1e-2 W/m2 per layer near conservative thin layers;
ours is the exact closed form (tests/unit/test_two_stream_exact_forms.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_rte_clear.npz"
# [W/m2].  float64 LW: oracle's own rounding and tau cutoff.  float32: eps x solar
# constant x ~10 g-point sums.
_TOL = {False: {"lw": 1e-5, "sw": 1e-8}, True: {"lw": 2e-3, "sw": 2e-3}}


def _fluxes(z, fp32):
    ghg = json.loads(str(z["ghg"]))
    cfg = RRTMGPConfig(S_0=float(z["s0"]), include_clouds=False, overhead_layer=True,
                       compute_fp32=fp32, gpoint_batch_size=0, gpoint_checkpoint=False,
                       use_optimal_angle=False, use_scan=True, co2_ppmv=ghg["co2"] * 1e6,
                       ch4_ppbv=ghg["ch4"] * 1e9, n2o_ppbv=ghg["n2o"] * 1e9)
    a = lambda k: jnp.asarray(z[k])  # noqa: E731
    return RRTMGP.from_legoesm_config(cfg).solve_columns(
        T=a("T"), p_full=a("p_full"), p_half=a("p_half"), sfc_temperature=a("tsfc"), q_v=a("q"),
        cos_zenith=a("mu0"), sfc_albedo=a("alb"), sfc_emissivity=a("emis"), o3_vmr=a("o3"),
        o3_top_vmr=a("o3_top"), ghg_vmr_override=ghg)


@pytest.fixture(scope="module")
def z():
    return np.load(_FIX)


@pytest.mark.parametrize("fp32", [False, True])
def test_clear_sky_fluxes_match_rte(z, fp32):
    o = _fluxes(z, fp32)
    for band, pairs in (("lw", ((o.lw_flux_up, "lw_up"), (o.lw_flux_down, "lw_dn"))),
                        ("sw", ((o.sw_flux_up, "sw_up"), (o.sw_flux_down, "sw_dn")))):
        for ours, key in pairs:
            err = float(np.max(np.abs(np.asarray(ours, np.float64) - z[key])))
            assert err <= _TOL[fp32][band], f"{key} fp32={fp32}: {err:.3e} W/m2"


def test_fixture_reaches_twilight(z):
    """The slant-path floor only shows in columns with mu0 < 0.01."""
    assert np.sum(z["mu0"] < 0.01) >= 5
