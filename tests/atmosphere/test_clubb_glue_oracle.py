"""CLUBB <-> CAM host glue vs CAM's clubb_intr (oracle fixture).

Fixture ``oracle_fixtures/clubb_glue_cam.npz`` was produced by RUNNING the
verbatim clubb_intr.F90 / tropopause.F90 line ranges of ESCOMP/CAM
cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56), compiled with
gfortran (legoESM constants in the stubs).  Generator:
scripts/validate/oracle/clubb_glue/ (run.sh = build_inputs.py + build.sh +
gen_clubb_glue_fixture.py).

The CLUBB core is replaced on BOTH sides by the same prescribed increments
(dthl, drt, du = dv = 0) and absolute rcm / cloud_frac / rtp2, so the
comparison covers exactly the host glue: the theta-l / total-water map into
CLUBB (clubb_intr.F90:1546-1550), the surface heat-flux lower BC (:1781), the
map back to T (:2048-2051, 2159-2160), the water reset/return above clubbtop
and the column energy fixer (:2042-2150), and the ice-stratus cut above top_lev
(:2535), for 8 RFMIP-based columns and CLUBB tops of 15000 Pa (old production
deck) and 100 Pa (CAM6 namelist, production since 2026-10-20).  Both sides get
CAM's rho_ds = pdel/(g dz).

Metric: per field, max |ours - oracle| / max |oracle| over the case
(field-normalised, the tier-1 audit metric); bound 1e-10 in fp64 (user
decision 2026-10-19), at the hard-switch limit of the existing top taper
(``trop_cloud_taper_lnp_width`` -> 1e-6).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig  # noqa: E402

_FIX = Path(__file__).parent / "oracle_fixtures" / "clubb_glue_cam.npz"
_TOL = 1.0e-10
_HARD = 1.0e-6
_FIELDS = ("thlm", "rtm", "wpthlp_sfc", "wprtp_sfc", "du", "dv", "dT", "dqv",
           "dqc")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def run_glue(fx, case, width, monkeypatch, T=None, q_c=None):
    """Our glue with the oracle's prescribed 'core'; returns outputs + the
    host->CLUBB inputs the core received."""
    g = lambda k: jnp.asarray(fx[f"in/{k}"])  # noqa: E731
    c = lambda k: jnp.asarray(fx[f"{case}/in/{k}"])  # noqa: E731
    flip = C.flip_vertical
    T = g("T") if T is None else T
    q_c = g("qc") if q_c is None else q_c
    ncol, nlev = T.shape
    cap = {}

    def fake_core(state, forcing, **kw):
        cap.update(state=state, kw=kw)
        new = state._replace(um=state.um + flip(c("du")), vm=state.vm + flip(c("dv")),
                             thlm=state.thlm + flip(c("dthl")),
                             rtm=state.rtm + flip(c("drt")), rtp2=flip(g("rtp2")))
        z = jnp.zeros_like(state.thlm)
        return new, dict(cloud_frac=flip(jnp.minimum(c("cf_out"), 1.0)),
                         rcm=flip(c("rcm_out")), wpthvp=jnp.zeros_like(state.wp2),
                         Kh_zt=z, Kh_zm=jnp.zeros_like(state.wp2),
                         rcp2_zt=z)  # relvar input; not compared here

    monkeypatch.setattr(C, "advance_clubb_core", fake_core)
    dt = float(fx["in/dt"])
    cfg = C.CLUBBConfig(
        prognostic=True, clubb_dt=dt, trop_cloud_top_press=float(case),
        trop_cloud_taper_lnp_width=width,
        surface=SurfaceLayerConfig()._replace(prescribed_evap_kg_m2_s=g("E")))
    rho = g("rho_cam")
    mom = C.pack_clubb_moments(C.init_clubb_moments(ncol, nlev, cfg))
    ustar = jnp.sqrt(jnp.hypot(g("taux"), g("tauy")) / rho[:, -1])
    o, _ = C.clubb_turbulence_prognostic(
        g("u"), g("v"), T, g("qv"), mom, g("p_full"), g("p_half"),
        g("z_full"), g("z_half"), T[:, -1] + 1.0, g("qv")[:, -1], rho, dt, cfg,
        surface_flux=(g("taux"), g("tauy"), g("shf"), g("lhf"), ustar),
        q_c=q_c, lat=g("lat"))
    s = cap["state"]
    return dict(thlm=s.thlm, rtm=s.rtm, wpthlp_sfc=s.wpthlp[:, 0],
                wprtp_sfc=s.wprtp[:, 0], du=o.du_dt, dv=o.dv_dt, dT=o.dT_dt,
                dqv=o.dq_v_dt, dqc=o.dq_c_dt)


def errors(fx, case, ours):
    """Field -> field-normalised max error vs the oracle."""
    nclubb = fx[f"{case}/oracle/thlm"].shape[1]
    ref = {k: fx[f"{case}/oracle/{k}"] for k in ("thlm", "rtm", "du", "dv",
                                                  "dT", "dqv", "dqc")}
    ref["wpthlp_sfc"] = fx[f"{case}/oracle/sfc"][:, 0]
    ref["wprtp_sfc"] = fx[f"{case}/oracle/sfc"][:, 1]
    out = {}
    for k in _FIELDS:
        a = np.asarray(ours[k])
        if k in ("thlm", "rtm"):
            a = a[:, :nclubb]          # the levels CLUBB runs on, ascending
        b = ref[k]
        ok = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        out[k] = (float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
                  if ok else np.inf)
    return out


@pytest.mark.parametrize("case", ["15000", "100"])
def test_glue_matches_cam_clubb_intr(fix, case, monkeypatch):
    err = errors(fix, case, run_glue(fix, case, _HARD, monkeypatch))
    bad = {k: v for k, v in err.items() if not v <= _TOL}
    assert not bad, f"top {case} Pa: fields above {_TOL}: {bad}"


@pytest.mark.parametrize("path", ["bulk", "prescribed"])
def test_heat_bc_and_energy_fixer_target_share_the_cam_convention(fix, path,
                                                                 monkeypatch):
    """The paths CAM has no counterpart for (CLUBB's own bulk formula, an
    LES-style prescribed kinematic flux) use the same host convention
    shf = c_pd rho w'thl' (clubb_intr.F90:1781): the energy handed to the
    column fixer equals shflx + L_v*E (CAM :2135; no latent-enthalpy
    correction, oracle row 9)."""
    g = lambda k: jnp.asarray(fix[f"in/{k}"])  # noqa: E731
    T, qv = g("T"), g("qv")
    ncol, nlev = T.shape

    def fake_core(state, forcing, **kw):
        z = jnp.zeros_like(state.thlm)
        return state, dict(cloud_frac=z, rcm=z, wpthvp=jnp.zeros_like(state.wp2),
                           Kh_zt=z, Kh_zm=jnp.zeros_like(state.wp2),
                           rcp2_zt=z)  # relvar input; not compared here

    monkeypatch.setattr(C, "advance_clubb_core", fake_core)
    cfg = C.CLUBBConfig(prognostic=True, trop_cloud_top_press=0.0)
    sfc = {} if path == "bulk" else dict(
        sfc_wpthlp=jnp.full((ncol,), 0.05), sfc_wprtp=jnp.full((ncol,), 4e-5),
        sfc_upwp=jnp.full((ncol,), -0.05), sfc_vpwp=jnp.full((ncol,), 0.01))
    *_, d = C.clubb_step(
        g("u"), g("v"), T, qv, C.init_clubb_moments(ncol, nlev, cfg),
        g("p_full"), g("p_half"), g("z_full"), g("z_half"), T[:, -1] + 3.0,
        qv[:, -1] * 1.2, g("rho_cam"), 600.0, cfg, q_c=g("qc"), **sfc)
    from legoesm import constants
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        surface_moisture_flux,
    )
    evap = (surface_moisture_flux(cfg.surface, d["lhflx"], T[:, -1] + 3.0)
            if path == "bulk" else d["lhflx"] / constants.L_v)   # latent-ok: prescribed lhflx is wprtp*rho*L_v
    tot = np.asarray(d["shflx"] + constants.L_v * evap)   # latent-ok: host moist-enthalpy reference L_v
    assert np.max(np.abs(tot)) > 1.0
    np.testing.assert_allclose(np.asarray(d["sfc_energy_in"]), tot,
                               rtol=_TOL, atol=0.0)


def test_fixture_exercises_the_liquid_and_heat_terms(fix):
    """Non-vacuity of the inputs: the latent term of the theta-l map and the
    (1 - exner) share of the surface heat flux are far above the bound."""
    qc = fix["in/qc"]
    assert qc.max() > 1e-4
    exner_low = np.asarray(C.exner_function(jnp.asarray(fix["in/p_full"][:, -1])))
    assert np.max(np.abs((1.0 - exner_low) * fix["in/shf"])) > 0.1


def test_gradient_is_finite_on_the_fixture_columns(fix, monkeypatch):
    """jax.grad of a scalar of the glue output w.r.t. T and the host liquid,
    at the production taper width (0.15)."""
    T0, qc0 = jnp.asarray(fix["in/T"]), jnp.asarray(fix["in/qc"])

    def loss(T, qc):
        o = run_glue(fix, "15000", 0.15, monkeypatch, T=T, q_c=qc)
        return jnp.sum(o["dT"]) + 1e3 * jnp.sum(o["dqv"]) + jnp.sum(o["thlm"])

    gT, gq = jax.grad(loss, argnums=(0, 1))(T0, qc0)
    assert np.all(np.isfinite(np.asarray(gT))) and np.all(np.isfinite(np.asarray(gq)))
    assert float(jnp.max(jnp.abs(gq))) > 0.0


_GEN_DIR = (Path(__file__).resolve().parents[2]
            / "scripts" / "validate" / "oracle" / "clubb_glue")


def _module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _GEN_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_parses_the_oracle_record(tmp_path):
    """gen_clubb_glue_fixture.parse reads main.F90's per-column record in
    FIELDS order (one line per field)."""
    gen = _module("gen_clubb_glue_fixture")
    lines = ["TROPLEV  11  15", "TOP_LEV  14", "CLUBBTOP  1  14", "COL  1"]
    lines += [" ".join(f"{i + 0.5 * j:.17e}" for j in range(3))
              for i in range(len(gen.FIELDS))]
    p = tmp_path / "o.txt"
    p.write_text("\n".join(lines) + "\n")
    o = gen.parse(p)
    assert o["top_lev"] == 14 and o["clubbtop"] == {0: 14}
    assert list(o["troplev"]) == [11, 15]
    for i, f in enumerate(gen.FIELDS):
        np.testing.assert_array_equal(o["cols"][0][f], i + 0.5 * np.arange(3))


def test_production_clubb_top_is_cam6_and_an_ab_arm():
    """The production deck uses CAM6's trop_cloud_top_press (100 Pa,
    namelist_defaults_cam.xml:1295), and the A/B probe accepts it as an arm."""
    import yaml
    deck = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "amip"
                           / "amip_production.yaml").read_text())
    assert deck["clubb_trop_cloud_top_press"] == 100.0
    assert _module("top_press_scm_ab")._P_CUT == 15000.0


def _aist(fx, top, width):
    """Our cam6 ice-stratus fraction on the fixture's INPUT state (the state
    the oracle assembles cloud fraction on); ``top = 0`` = no cut."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        cam6_ice_stratus_fraction,
    )
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    g = lambda k: jnp.asarray(fx[f"in/{k}"])  # noqa: E731
    return cam6_ice_stratus_fraction(
        g("qv"), g("T"), g("p_full"), g("qi"), g("lat"),
        CloudConfig(scheme="cam6_clubb"), g("p_half")[:, :-1],
        trop_cloud_top_press=top, trop_cloud_taper_lnp_width=width)


def _aist_top_error(fx, case, width):
    """Field-normalised max error of our cut aist against CAM's top_lev cut
    (clubb_intr.F90:2535 ``aist(:,:top_lev-1) = 0``): the oracle's aist above
    top_lev, our UNCUT aist at and below it (the aist curves themselves are a
    separate finding, outside this cut)."""
    top_lev = int(fx[f"{case}/in/top_lev"])          # 0-based first CLUBB level
    orc = fx[f"{case}/oracle/aist"]
    uncut = np.asarray(_aist(fx, 0.0, _HARD))
    ref = np.where(np.arange(orc.shape[1]) < top_lev, orc, uncut)
    ours = np.asarray(_aist(fx, float(case), width))
    return float(np.max(np.abs(ours - ref)) / max(np.max(np.abs(orc)), 1e-300))


@pytest.mark.parametrize("case", ["15000", "100"])
def test_aist_is_cut_at_the_clubb_top_as_cam(fix, case):
    top_lev = int(fix[f"{case}/in/top_lev"])
    assert np.all(fix[f"{case}/oracle/aist"][:, :top_lev] == 0.0)
    err = _aist_top_error(fix, case, _HARD)
    assert err <= _TOL, f"top {case} Pa: aist cut error {err}"


def test_aist_cut_is_exercised_by_the_fixture(fix):
    """Non-vacuity: at 15000 Pa the uncut aist above top_lev is far from 0."""
    top_lev = int(fix["15000/in/top_lev"])
    assert np.max(np.asarray(_aist(fix, 0.0, _HARD))[:, :top_lev]) > 0.1


def test_aist_cut_gradient_is_finite_on_the_fixture_columns(fix):
    """jax.grad of the cut aist w.r.t. T and q_i at the production width."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        cam6_ice_stratus_fraction,
    )
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    g = lambda k: jnp.asarray(fix[f"in/{k}"])  # noqa: E731
    for dt in (jnp.float64, jnp.float32):
        def loss(T, qi):
            return jnp.sum(cam6_ice_stratus_fraction(
                g("qv").astype(dt), T, g("p_full").astype(dt), qi,
                g("lat").astype(dt), CloudConfig(scheme="cam6_clubb"),
                g("p_half")[:, :-1].astype(dt), trop_cloud_top_press=15000.0,
                trop_cloud_taper_lnp_width=0.15))
        gT, gq = jax.grad(loss, argnums=(0, 1))(g("T").astype(dt), g("qi").astype(dt))
        assert np.all(np.isfinite(np.asarray(gT))) and np.all(np.isfinite(np.asarray(gq)))
