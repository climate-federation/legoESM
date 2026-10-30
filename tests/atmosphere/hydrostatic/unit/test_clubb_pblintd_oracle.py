"""CAM6 post-CLUBB PBL height (the ``pblh`` ZM reads) vs the RUNNING oracle.

CAM6 diagnoses the PBL height ZM reads next step at the end of
``clubb_tend_cam`` (clubb_intr.F90:2594-2617, cam_cesm2_1_rel_60,
a03b84b7c4e34f965b115686f22a043b85739e56): ``th = t*exner``,
``thv = th*(1+zvir*q)``, ``calc_ustar``/``calc_obklen`` from the coupler's
surface stress, sensible heat and water flux, then ``hb_diff::pblintd`` -- on
``state1`` AFTER the CLUBB update.  Ported as
``holtslag_boville.cam_pblintd_height`` and wired into the MPAS/column lane
for ``turbulence='clubb'`` with the injected surface flux.

Fixture ``oracle_fixtures/clubb_pblintd_cam.npz``: the CAM Fortran
(pbl_utils.F90 + hb_diff.F90 + the clubb_intr block, unmodified) RUN by
``scripts/validate/oracle/clubb_pblintd/{build.sh,make_fixture.py}`` on the
21 L32 columns of ``holtslag_boville.npz``.

Metric (the merged tier-1 HB oracle test's, reused unchanged): per field
``max |ours - oracle| / max |oracle|`` per column; bound 1e-10 in fp64 with
every existing sharpness field at its hard-switch limit (``HARD``).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence import holtslag_boville as HB
from legoesm.atmosphere.physics.turbulence.config import HoltslagBovilleConfig

from legoesm import constants
from tests.atmosphere.hydrostatic.unit.test_holtslag_boville_oracle import HARD, _err

jax.config.update("jax_enable_x64", True)

TOL = 1.0e-10
ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests" / "atmosphere" / "oracle_fixtures" / "clubb_pblintd_cam.npz"
GEN = ROOT / "scripts" / "validate" / "oracle" / "clubb_pblintd" / "make_fixture.py"


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE))


def _args(fx, dtype=jnp.float64):
    f = {k: jnp.asarray(fx[f"in_{k}"], dtype) for k in
         ("T", "q", "u", "v", "p_full", "zm", "zi", "taux", "tauy", "shflx",
          "qflx", "pref_mid")}
    return (f["T"], f["q"], f["u"], f["v"], f["p_full"], f["zm"], f["zi"],
            f["taux"], f["tauy"], f["shflx"], f["qflx"], f["pref_mid"])


def _pblh(fx, cfg_kw, dtype=jnp.float64):
    return HB.cam_pblintd_height(*_args(fx, dtype),
                                 HoltslagBovilleConfig()._replace(**cfg_kw),
                                 p_sfc=jnp.asarray(fx["in_p_half"], dtype)[:, -1])


def test_fixture_provenance(fx):
    meta = json.loads(str(fx["meta"]))
    assert meta["oracle"]["sha"] == "a03b84b7c4e34f965b115686f22a043b85739e56"
    c = meta["constants"]
    assert (c["g"], c["vk"], c["cpair"], c["rair"]) == (
        constants.g, constants.kappa_von_karman, constants.c_pd, constants.R_d)
    assert c["zvir"] == constants.R_v / constants.R_d - 1.0
    assert FIXTURE.stat().st_size < 1_000_000
    # the set exercises both pblintd passes and the kbfs = 0 edge
    assert (fx["ora_kbfs"] > 0).sum() >= 5 and (fx["ora_kbfs"] < 0).sum() >= 3
    assert (fx["ora_kbfs"] == 0).sum() >= 1


def test_pblh_matches_cam_oracle_at_hard_limit(fx):
    err = _err(_pblh(fx, HARD), fx["ora_pblh"][:, None], True)
    assert err <= TOL, err


def test_surface_scales_match_cam_oracle(fx):
    """calc_ustar / calc_obklen as the clubb_intr block calls them."""
    T, q, *_ = _args(fx)
    th, thv = HB._theta_thetav_se(T, q, jnp.asarray(fx["in_p_full"]),
                                  jnp.asarray(fx["in_p_half"])[:, -1])
    ustar, _, _, kbfs, obklen = HB._calc_ustar_obklen(
        T[:, -1], jnp.asarray(fx["in_p_full"])[:, -1],
        jnp.asarray(fx["in_taux"]), jnp.asarray(fx["in_tauy"]),
        jnp.asarray(fx["in_shflx"]), jnp.asarray(fx["in_qflx"]),
        th[:, -1], thv[:, -1], HoltslagBovilleConfig().ustar_min)
    for name, ours in (("ustar", ustar), ("kbfs", kbfs), ("obklen", obklen)):
        err = _err(ours, fx[f"ora_{name}"][:, None], True)
        assert err <= TOL, (name, err)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_gradient_is_finite_at_production_smoothing(fx, dtype):
    T, q, u, v, pf, zm, zi, tx, ty, sh, qf, pref = _args(fx, dtype)
    cfg = HoltslagBovilleConfig()

    def loss(T, u, sh):
        return jnp.sum(HB.cam_pblintd_height(
            T, q, u, v, pf, zm, zi, tx, ty, sh, qf, pref, cfg,
            p_sfc=jnp.asarray(fx["in_p_half"])[:, -1]) ** 2)
    grads = jax.grad(loss, argnums=(0, 1, 2))(T, u, sh)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.max(jnp.abs(grads[0]))) > 0.0


def test_generator_rebuilds_the_committed_inputs(fx, tmp_path):
    """Direct test of scripts/validate/oracle/clubb_pblintd/make_fixture.py."""
    spec = importlib.util.spec_from_file_location("clubb_pblintd_mf", GEN)
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    d = mf.build_inputs(np.load(mf.SRC))
    for k, v in d.items():
        if k == "exner":
            # derived via libm pow, whose last bit differs across machines
            np.testing.assert_allclose(v, fx["in_exner"], rtol=1e-15, atol=0,
                                       err_msg=k)
        else:
            np.testing.assert_array_equal(v, fx[f"in_{k}"], err_msg=k)
    p = tmp_path / "o.txt"
    p.write_text("1 2 3 4 5\n6 7 8 9 10\n")
    o = mf.parse_outputs(p, 2)
    np.testing.assert_array_equal(o["ora_pblh"], [4.0, 9.0])
    np.testing.assert_array_equal(o["ora_ustar"], [1.0, 6.0])


# ---------------------------------------------------------------------------
# Wiring: the MPAS/column lane feeds pblintd the POST-CLUBB state
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mpas_case():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    dtype = state.T.data.dtype
    q_v = jnp.full((ncell, nlev), 0.008, dtype=dtype)
    # sheared eastward flow, so CLUBB moves momentum and there is a stress
    u_edge = (jnp.cos(jnp.asarray(mesh.angleEdge))[:, None]
              * jnp.linspace(20.0, 2.0, nlev)[None, :]).astype(dtype)
    return mesh, sigma, state._replace(
        u=state.u.replace(data=u_edge), tracers={"q_v": q_v})


def _land_forcing(ncell, shflx, lhflx):
    return {"T_sfc": jnp.full((ncell,), 300.0),
            "shflx_land": jnp.full((ncell,), shflx),
            "lhflx_land": jnp.full((ncell,), lhflx),
            "evap_land": jnp.full((ncell,), lhflx / constants.L_v)}


def test_mpas_clubb_lane_publishes_cam_pblintd_on_the_post_clubb_state(
        mpas_case, monkeypatch):
    """Fails if the lane publishes CLUBB's own bulk-Ri height, or feeds
    pblintd the pre-CLUBB state / heights / a different surface flux."""
    from legoesm.atmosphere.physics._shared import compute_heights_from_sigma
    from legoesm.atmosphere.physics.turbulence import integration as TI
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    mesh, sigma, state = mpas_case
    ncell, nlev = state.T.data.shape
    cap = {}
    real = TI.cam_pblintd_height

    def spy(*a, **k):
        cap["args"], cap["kw"] = a, k
        cap["out"] = real(*a, **k)
        return cap["out"]
    monkeypatch.setattr(TI, "cam_pblintd_height", spy)
    dt = 600.0
    tc = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True))
    fn = TI._make_mpas_turbulence(tc, dt, f_land=jnp.ones((ncell,)))
    forcing = _land_forcing(ncell, 150.0, 80.0)
    tend, carry = fn(state, mesh, sigma, forcing=forcing)
    assert "args" in cap, "the clubb lane never ran the CAM pblintd"
    T, q, u, v, pf, zf, zh, tx, ty, sh, qf, pref, hbc = cap["args"]
    np.testing.assert_array_equal(np.asarray(carry["pblh"]), np.asarray(cap["out"]))
    p_full = sigma.pressure_at_full(state.p_s.data)
    p_half = sigma.pressure_at_half(state.p_s.data)
    # CAM-SE exner reference (follow-up row 20): the column's own surface
    np.testing.assert_array_equal(np.asarray(cap["kw"]["p_sfc"]),
                                  np.asarray(p_half)[:, -1])
    T_post = state.T.data + dt * tend.dT_dt.data
    q_post = state.tracers["q_v"] + dt * tend.tracer_tendencies["q_v"]
    np.testing.assert_allclose(np.asarray(T), np.asarray(T_post), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(q), np.asarray(q_post), rtol=1e-14)
    assert float(jnp.max(jnp.abs(T - state.T.data))) > 0.0, "CLUBB moved nothing"
    u0, v0 = reconstruct_cell_velocity(state.u.data, mesh)
    assert float(jnp.max(jnp.abs(u - u0))) > 0.0 or float(
        jnp.max(jnp.abs(v - v0))) > 0.0, "winds are not the post-CLUBB winds"
    zf_ref, zh_ref = compute_heights_from_sigma(T, p_half, q_v=q)
    np.testing.assert_array_equal(np.asarray(zf), np.asarray(zf_ref))
    np.testing.assert_array_equal(np.asarray(zh), np.asarray(zh_ref))
    np.testing.assert_array_equal(np.asarray(pf), np.asarray(p_full))
    # f_land = 1: the flux handed over is exactly the land's (state dtype)
    for ours, key in ((sh, "shflx_land"), (qf, "evap_land")):
        np.testing.assert_array_equal(
            np.asarray(ours), np.asarray(forcing[key]).astype(np.asarray(ours).dtype))
    assert np.all(np.asarray(tx) ** 2 + np.asarray(ty) ** 2 > 0.0)
    np.testing.assert_array_equal(
        np.asarray(pref), np.asarray(sigma.pressure_at_full(
            jnp.asarray(constants.p_ref, dtype=state.T.data.dtype))))
    assert hbc == tc.holtslag_boville
