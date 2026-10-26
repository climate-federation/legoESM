"""Oracle match: RRTMGP cloud optics (LW absorption-only, SW v1.8.2 table, CAM ice size).

Production path: ``two_stream.solve_lw/solve_sw`` -> ``RRTMOptics.add_cloud_optical_properties``
-> ``_combine_gas_and_cloud_properties`` -> ``cloud_optics.compute_optical_properties``
(LW: ``absorption_only=True``; SW: then ``_apply_delta_scaling_for_cloud``).  The test calls
``add_cloud_optical_properties`` with zero gas optics so the cloud contribution is compared
alone.

Fixture ``oracle_fixtures/rrtmgp_cloud_optics.npz``: outputs come from RUNNING RTE+RRTMGP v1.8
(rte-rrtmgp 41c5fcd950fed09b8afe186dede266824eca7fd3) ``mo_cloud_optics_rrtmgp%cloud_optics``
on rrtmgp-data v1.8.2 (34967d68703e682305ec4da72ee22bbe1f11b198) rrtmgp-clouds-{lw,sw}.nc,
ice roughness medium: LW into ty_optical_props_1scl (absorption tau, no scattering), SW into
ty_optical_props_2str + delta_scale.  Ice table coordinate = CAM6 deffi = effi*500/917*2 (CAM
cam_cesm2_1_rel_60 micro_mg2_0.F90:2795).  Inputs: 5 CAM6-L32 columns through the production
compute_cloud_properties (amip_production.yaml CloudConfig) and McICA in-cloud paths.
Generator: scripts/validate/oracle/rrtmgp_cloud_optics/{build.sh,oracle_driver.F90,make_fixture.py}.

Excluded layers = legoESM's existing AD-safety regulariser ``cloud_optics._EPSILON`` (1e-6):
a phase path in (0, 1e-6) g/m2 is masked out, and ssa/g are filled 0 where tau or tau*ssa
<= 1e-6 (RRTMGP uses >0 and epsilon); tier-1 audit finding, unchanged here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_cloud_optics.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_EPS = cloud_optics._EPSILON
_DATA = (Path(cloud_optics.__file__).parent / "rrtmgp_data")


@pytest.fixture(scope="module")
def case():
    z = np.load(_FIX)
    opt = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=True)).optics_lib
    return z, opt


def _cloud(opt, z, is_lw):
    """Per-band cloud props through the production combine (first g-point of each band)."""
    gas = opt.gas_optics_lw if is_lw else opt.gas_optics_sw
    bnd = np.asarray(gas.g_point_to_bnd)
    zero = jnp.zeros_like(jnp.asarray(z["cpl"]))
    gas_props = {"optical_depth": zero, "ssa": zero, "asymmetry_factor": zero}
    res = {"optical_depth": [], "ssa": [], "asymmetry_factor": []}
    for b in range(int(bnd.max()) + 1):
        ig = int(np.argmax(bnd == b))
        p = opt.add_cloud_optical_properties(
            jnp.asarray(ig), gas_props, is_lw, jnp.asarray(z["r_liq"]), jnp.asarray(z["cpl"]),
            jnp.asarray(z["r_ice"]), jnp.asarray(z["cpi"]))
        for k in res:
            res[k].append(np.asarray(p[k]))
    return {k: np.stack(v, axis=-1) for k, v in res.items()}


def _path_ok(z):
    ok = lambda p: (p == 0.0) | (p * 1e3 >= _EPS)  # noqa: E731
    return (ok(z["cpl"]) & ok(z["cpi"]))[..., None]


def _max_rel(ours, ref, mask):
    mask = np.broadcast_to(mask, ref.shape)
    assert mask.sum() > 0
    zero = mask & (ref == 0.0)
    assert np.all(ours[zero] == 0.0)
    nz = mask & (ref != 0.0)
    return float(np.max(np.abs(ours[nz] - ref[nz]) / np.abs(ref[nz])))


def test_bundled_cloud_tables_are_rrtmgp_data_v182():
    sha = {t: hashlib.sha256((_DATA / f"cloudysky_{t}.nc").read_bytes()).hexdigest()
           for t in ("lw", "sw")}
    z = np.load(_FIX)
    assert sha["lw"] == str(z["sha256_lw"])
    assert sha["sw"] == str(z["sha256_sw"])
    assert sha["sw"] == "fced616f92c682809dc0be387841b1e71c40cfc73285362b64354464c855104d"
    assert "v1.8.2" in json.loads(str(z["meta"]))["rrtmgp_data"]


def test_lw_cloud_is_absorption_only_and_matches_rrtmgp_1scl(case):
    z, opt = case
    ours = _cloud(opt, z, is_lw=True)
    ref = z["lw_tau_abs"]
    assert ours["optical_depth"].shape == ref.shape and np.max(ref) > 1.0
    rel = _max_rel(ours["optical_depth"], ref, _path_ok(z))
    assert rel <= _TOL, f"LW absorption tau max rel {rel:.3e} > {_TOL}"
    assert np.all(ours["ssa"] == 0.0) and np.all(ours["asymmetry_factor"] == 0.0)


def test_sw_cloud_matches_rrtmgp_2str_delta_scaled(case):
    z, opt = case
    ours = _cloud(opt, z, is_lw=False)
    m = _path_ok(z)
    rel = _max_rel(ours["optical_depth"], z["sw_tau_ds"], m)
    assert rel <= _TOL, f"SW tau max rel {rel:.3e} > {_TOL}"
    # ssa/g: also outside the safe_divide fill (tau and tau*ssa > eps, before and after scaling)
    m2 = m & (z["sw_tau"] > _EPS) & (z["sw_tau"] * z["sw_ssa"] > _EPS) \
        & (z["sw_tau_ds"] > _EPS) & (z["sw_tau_ds"] * z["sw_ssa_ds"] > _EPS)
    for k, r in (("ssa", "sw_ssa_ds"), ("asymmetry_factor", "sw_g_ds")):
        rel = _max_rel(ours[k], z[r], m2)
        assert rel <= _TOL, f"SW {k} max rel {rel:.3e} > {_TOL}"


def test_cloud_optics_gradient_finite(case):
    z, opt = case

    def loss(cpl, cpi, rl, ri):
        tot = 0.0
        for is_lw in (True, False):
            zero = jnp.zeros_like(cpl)
            p = opt.add_cloud_optical_properties(
                jnp.asarray(0), {"optical_depth": zero, "ssa": zero, "asymmetry_factor": zero},
                is_lw, rl, cpl, ri, cpi)
            tot = tot + jnp.sum(p["optical_depth"]) + jnp.sum(p["ssa"] * p["optical_depth"])
        return tot

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*(jnp.asarray(z[k]) for k in
                                                    ("cpl", "cpi", "r_liq", "r_ice")))
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(grads[3]) != 0.0)  # ice size reaches the optics
