"""Oracle match: radiation droplet/ice effective radius on CAM6 MG2's in-cloud footing.

Production path (MPAS ``_run_column``): ``radiation.integration._make_hydrostatic_radiation``
physics_fn overrides the droplet number with the aerosol-activated (specified, IN-CLOUD)
number and calls ``_call_radiation_backend(..., n_cloud_in_cloud=True)`` ->
``compute_cloud_properties``.  CAM6 forms REL from that kind of number (MG2 ``nccons``)
with the in-cloud water ``qc / max(ast, mincld)``; before #1521 was fixed legoESM paired it
with the grid-mean q_c, so r_eff came out low by cf^(1/3).

Fixture ``oracle_fixtures/reff_incloud.npz``: outputs from RUNNING CAM cam_cesm2_1_rel_60
(a03b84b7c4e34f965b115686f22a043b85739e56) micro_mg2_0.F90 L877-879, 2739-2768,
2772-2808, 2811-2865 and micro_mg_cam.F90 L2296, 2534-2554 (the grid REL radiation reads)
verbatim with micro_mg_utils.F90 compiled unmodified (dcs = 500e-6).  Inputs: the 5 tier-1
CAM6-L32 columns with ast < mincld edge levels, cap-bound levels, a grid q_c below 1e-14
whose in-cloud value is above qsmall, two dead-droplet-number levels (oracle number =
CloudConfig.Nc_default), and a deep convective mass flux (radiative cover != ast).
Production CloudConfig (amip_production.yaml, resolved through run_amip + model_driver) is
stored in the fixture.
Generator: scripts/validate/oracle/reff_incloud/{build.sh,oracle_driver.F90,make_fixture.py}.

Liquid: asserted on EVERY level against ``rel_cam`` (CAM6 REL: in-cloud water capped at
5 g/kg, MG2 size_dist_param_liq, qsmall gate, 10 um fill).  ``effc_wrf`` (legoESM's former
M2005 PSD on the same inputs) is kept for the PSD gap only.
Ice: excluded where MG2's ice-slope bounds adjust the number (``ni_adj``) or its 5 g/kg cap
binds (absent in legoESM; the RRTMGP table clip bounds the size downstream).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

_FIX = Path(__file__).parent / "oracle_fixtures" / "reff_incloud.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_MG2_QC_CAP = 5.0e-3  # micro_mg2_0.F90:2762, micro_mg_cam.F90:2296 in-cloud cap


@pytest.fixture(scope="module")
def z():
    return np.load(_FIX)


def _kwargs(z):
    cfg = CloudConfig(**json.loads(str(z["meta"]))["cloud_config"])
    return dict(
        T=jnp.asarray(z["T"]), p_full=jnp.asarray(z["p_full"]), q_v=jnp.asarray(z["q_v"]),
        dp=jnp.asarray(np.diff(z["p_half"], axis=1)), config=cfg,
        q_cloud=jnp.asarray(z["q_c"]), q_ice=jnp.asarray(z["q_i"]),
        n_ice=jnp.asarray(z["n_i"]), n_cloud=jnp.asarray(z["n_ccn"]),
        cloud_fraction_override=jnp.asarray(z["alst"]), lat=jnp.asarray(z["lat"]),
        conv_mass_flux_up=jnp.asarray(z["conv_mass_flux_up"]),
        conv_icwmr=jnp.asarray(z["conv_icwmr"]), p_half=jnp.asarray(z["p_half"]))


def _max_rel(ours, ref, mask):
    assert mask.sum() > 0
    return float(np.max(np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])))


def test_liquid_reff_matches_cam6_rel(z):
    props = compute_cloud_properties(**_kwargs(z), n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    ours = np.asarray(props.r_eff_liq) * 1e6  # m -> um (oracle REL units)
    q_ic = z["q_c"] / z["lcldm"]
    dead = z["n_ccn"] <= 1.0
    live = ~dead & (z["q_c"] > 0.0)
    # the fixture exercises both halves of lcldm = max(ast, mincld), ast != cover,
    # the 5 g/kg cap, the in-cloud qsmall gate below legoESM's old 1e-14 grid gate,
    # the 10 um fill, and the dead-number floor between mincld and 1e-3 and below mincld
    assert np.any(live & (z["ast"] < 1.0e-4)) and np.any(live & (z["ast"] > 0.05))
    assert np.any(live & (np.asarray(props.cloud_fraction) > z["ast"] + 1e-6))
    assert np.any(live & (q_ic > _MG2_QC_CAP))
    assert np.any(live & (z["q_c"] <= 1.0e-14) & (q_ic >= 1.0e-18))
    assert np.any(z["q_c"] <= 0.0)
    a = np.asarray(props.cloud_fraction)[dead]
    assert np.any((a > 1.0e-4) & (a < 1.0e-3)) and np.any(a < 1.0e-4)
    rel = _max_rel(ours, z["rel_cam"], np.ones_like(dead))
    assert rel <= _TOL, f"r_eff_liq max rel {rel:.3e} > {_TOL}"
    # the no-liquid fill is CAM's 10 um, not the CloudConfig.r_eff_liq knob
    kw = _kwargs(z)
    kw["config"] = kw["config"]._replace(r_eff_liq=14.0e-6)
    ours14 = np.asarray(compute_cloud_properties(
        **kw, n_cloud_in_cloud=True, clubb_config=CLUBBConfig()).r_eff_liq) * 1e6
    np.testing.assert_array_equal(ours14, ours)


def test_dead_number_reff_ignores_deep_convective_cover(z):
    """Follow-up row 46: the dead-number REL divides the water by ast, as
    CAM does for every level (micro_mg_cam.F90:2539), not by a radiative
    cover that also holds deep cu.  The fixture's dead levels sit in a column
    without convection, so give that column the deepest column's convective
    mass flux: cover rises above ast there and CAM's REL (which never sees
    the convective cover) must still be matched."""
    kw = _kwargs(z)
    dead = np.asarray(z["n_ccn"]) <= 1.0
    cols = np.unique(np.nonzero(dead)[0])
    mf = np.asarray(z["conv_mass_flux_up"]).copy()
    icw = np.asarray(z["conv_icwmr"]).copy()
    # deep convection through the whole of those columns
    mf[cols] = np.max(mf)
    icw[cols] = np.max(icw)
    kw["conv_mass_flux_up"] = jnp.asarray(mf)
    kw["conv_icwmr"] = jnp.asarray(icw)
    props = compute_cloud_properties(**kw, n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    cover = np.asarray(props.cloud_fraction)
    assert np.any(dead & (cover > np.asarray(z["ast"]) + 1e-6))
    ours = np.asarray(props.r_eff_liq) * 1e6
    rel = _max_rel(ours, z["rel_cam"], dead)
    assert rel <= _TOL, f"dead-number r_eff_liq max rel {rel:.3e} > {_TOL}"

def test_ice_reff_matches_mg2_effi(z):
    props = compute_cloud_properties(**_kwargs(z), n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    ours = np.asarray(props.r_eff_ice) * 1e6
    m = (z["q_i"] > 1.0e-14) & (z["ni_adj"] == 0.0) & (z["q_i"] / z["lcldm"] <= _MG2_QC_CAP)
    rel = _max_rel(ours, z["effi_mg2"], m)
    assert rel <= _TOL, f"r_eff_ice max rel {rel:.3e} > {_TOL}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_reff_gradient_finite(z, dtype):
    kw = {k: (jnp.asarray(v, dtype=dtype) if isinstance(v, jnp.ndarray) else v)
          for k, v in _kwargs(z).items()}

    def loss(q_c, n_c, alst):
        p = compute_cloud_properties(**{**kw, "q_cloud": q_c, "n_cloud": n_c,
                                        "cloud_fraction_override": alst},
                                     n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
        return jnp.sum(p.r_eff_liq) + jnp.sum(p.lwp / p.r_eff_liq)

    grads = jax.grad(loss, argnums=(0, 1, 2))(kw["q_cloud"], kw["n_cloud"],
                                               kw["cloud_fraction_override"])
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(grads[2]) != 0.0)  # r_eff depends on the cloud fraction
