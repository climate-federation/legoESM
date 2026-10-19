"""GABLS3 (Basu 2008): forcing tables, the 0.25 m MOST surface layer, the lid
inversion flux, and the driver end to end on a tiny grid (CPU, x64): column
budgets close to round-off, the recorded forcing integrals are what the LES
applied, production is refused while values are pending, restart is
bit-identical."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.forcing import gabls3_les as gb  # noqa: E402

_RUN = Path(__file__).resolve().parents[3] / "scripts" / "run"
_TINY = ["--smoke", "--hours", "0.003", "--nx", "8", "--ny", "8", "--nz", "16",
         "--record-frames", "3", "--print-every", "100000"]


def _load():
    if str(_RUN) not in sys.path:
        sys.path.insert(0, str(_RUN))
    spec = importlib.util.spec_from_file_location("run_gabls3_les",
                                                  _RUN / "run_gabls3_les.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_gabls3_les"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- forcing tables -----------------------------------------------------------
def test_series_mean_integrates_step_changes_exactly():
    # θ advection: −2.5e-5 until 01 UTC, then 7.5e-5 (spec Table 6)
    assert gb.series_mean(gb.THETA_ADV, 0.5 * 3600, 1.5 * 3600) == pytest.approx(
        0.5 * (-2.5e-5) + 0.5 * 7.5e-5, rel=1e-14)
    assert gb.series_mean(gb.THETA_ADV, 3600.0, 3600.0) == 7.5e-5   # right value
    assert gb.series_mean(gb.Q_ADV, 0.0, 3600.0) == 0.0
    assert gb.series_mean(gb.Q_ADV, 3 * 3600, 4 * 3600) == -8.0e-8
    # linear segment: the mean is the midpoint value
    th0, th1 = gb.SURFACE_THETA[0][1], gb.SURFACE_THETA[1][1]
    assert gb.series_mean(gb.SURFACE_THETA, 0.0, 3600.0) == pytest.approx(
        0.5 * (th0 + th1), rel=1e-14)


def test_geostrophic_and_advection_profiles_follow_the_spec():
    ug, vg = gb.geostrophic_mean(np.array([0.0, 1000.0, 2000.0]), 0.0, 0.0)
    # Table 4: −6.5 at 23 UTC → −5.0 at 03 UTC, so −6.125 at 00 UTC; (−2, 2) at 2 km
    np.testing.assert_allclose(ug, [-6.125, 0.5 * (-6.125 - 2.0), -2.0], rtol=1e-14)
    np.testing.assert_allclose(vg, [4.5, 3.25, 2.0], rtol=1e-14)
    adv = gb.advection_mean(np.array([0.0, 100.0, 200.0, 700.0]), 0.0, 0.0)
    np.testing.assert_allclose(adv["u"], [0.0, 2.5e-4, 5.0e-4, 5.0e-4], rtol=1e-14)
    np.testing.assert_allclose(adv["theta"], [0.0, -1.25e-5, -2.5e-5, -2.5e-5],
                               rtol=1e-14)
    assert gb.coriolis_parameter() == pytest.approx(
        2 * constants.Omega * math.sin(math.radians(51.9711)), rel=1e-14)
    sig_uv, sig_th = gb.perturbation_std(np.array([0.0, 100.0, 250.0]))
    np.testing.assert_allclose(sig_uv, [math.sqrt(0.2), 0.5 * math.sqrt(0.2), 0.0])
    np.testing.assert_allclose(sig_th, [math.sqrt(0.1), math.sqrt(0.1), 0.0])


# --- MOST from a reference height ----------------------------------------------
def test_most_reference_height_uses_the_integral_scalar_relation():
    kw = dict(spd_mean=jnp.asarray(5.0), z1=3.125, z0=0.15, theta_ref=290.0)
    # near-neutral: θ* = κ Δθ / ln(z1/z_ref), independent of z0
    us, ths, *_ = sl.most_surface_flux(th_air_mean=290.0 + 1e-6, t_sfc=290.0,
                                       z_ref=0.25, **kw)
    kap = constants.kappa_von_karman
    assert float(ths) == pytest.approx(kap * 1e-6 / math.log(3.125 / 0.25), rel=1e-4)
    # stable: moisture and heat share one denominator, so w'q'/w'θ' = Δq/Δθ
    out = sl.most_surface_flux(th_air_mean=292.0, t_sfc=290.0, z_ref=0.25,
                               dq=jnp.asarray(-1e-3), **kw)
    assert float(out[4]) / float(out[2]) == pytest.approx(-1e-3 / 2.0, rel=1e-12)
    # virtual L: moister air above (dq > 0) adds stability, weakening the flux
    dry = sl.most_surface_flux(th_air_mean=290.5, t_sfc=290.0, z_ref=0.25,
                               dq=jnp.asarray(2e-3), **kw)
    vir = sl.most_surface_flux(th_air_mean=290.5, t_sfc=290.0, z_ref=0.25,
                               dq=jnp.asarray(2e-3), virtual=True, **kw)
    assert abs(float(vir[2])) < abs(float(dry[2]))
    # the converged solution satisfies the spec's relations (Basu 2008, eq. for w'θ')
    from legoesm.core.bulk_flux import psi_h, psi_m
    for dth in (1.5, -0.8):                       # stable night, unstable morning
        us, ths, q0, _cd, qf = sl.most_surface_flux(
            th_air_mean=290.0 + dth, t_sfc=290.0, z_ref=0.25, dq=jnp.asarray(-5e-4),
            virtual=True, n_iter=60, **kw)
        qs = -float(qf) / float(us)
        inv_L = kap * constants.g * (float(ths) + (1 / constants.epsilon - 1) * 290.0
                                     * qs) / (float(us) ** 2 * 290.0)
        z1, zr = 3.125, 0.25
        den_h = (math.log(z1 / zr) - float(psi_h(z1 * inv_L))
                 + float(psi_h(zr * inv_L)))
        assert float(us) == pytest.approx(
            kap * 5.0 / (math.log(z1 / 0.15) - float(psi_m(z1 * inv_L))), rel=1e-9)
        assert float(ths) == pytest.approx(kap * dth / den_h, rel=1e-9)
        assert qs == pytest.approx(kap * -5e-4 / den_h, rel=1e-9)
        assert float(q0) == pytest.approx(-float(us) * float(ths), rel=1e-12)
        # the driver's solve (default iteration count) is converged to that solution
        *prod, resid = sl.most_surface_flux(
            th_air_mean=290.0 + dth, t_sfc=290.0, z_ref=0.25, dq=jnp.asarray(-5e-4),
            virtual=True, return_residual=True, **kw)
        assert float(resid) < 1e-6
        for got, ref in zip(prod, (us, ths, q0, _cd, qf)):
            assert float(got) == pytest.approx(float(ref), rel=1e-6)


# --- driver ---------------------------------------------------------------------
def _budget_lines(out):
    rows = {}
    for line in out.splitlines():
        if line.strip().startswith("budget "):
            parts = line.split()
            rows[parts[1].rstrip(":")] = {
                k: float(v) for k, v in (t.split("=") for t in parts[2:])}
    return rows


def test_production_is_refused_while_values_are_pending(tmp_path):
    from legoesm.atmosphere.les_suite.catalog import GABLS3_PENDING
    assert GABLS3_PENDING
    mod = _load()
    with pytest.raises(SystemExit, match="not approved"):
        mod.main(["--emit-suite-artifact", str(tmp_path / "a.npz")])


def test_gabls3_smoke_closes_budgets_and_records_the_applied_forcing(
        tmp_path, capsys, monkeypatch):
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    from legoesm.atmosphere.les_suite.bridge import forcing_profile_fn, load_artifact
    from legoesm.atmosphere.les_suite.scm_runner import build_cbl_scm_from_artifact
    mod = _load()
    # the smoke window sits before 01 UTC, where the spec's q advection is 0 and θ's
    # has no step: plant both inside it so the applied amounts are checked
    monkeypatch.setattr(gb, "THETA_ADV", ((0.0, -2.5e-5), (5.0, -2.5e-5),
                                          (5.0, 7.5e-5), (100.0, 7.5e-5)))
    monkeypatch.setattr(gb, "Q_ADV", ((0.0, -8.0e-8), (100.0, -8.0e-8)))
    art_path = tmp_path / "g3.npz"
    assert mod.main(_TINY + ["--emit-suite-artifact", str(art_path)]) == 0
    out = capsys.readouterr().out
    if not list_cases():
        register_default_catalog()
    case = get_case("gabls3")
    assert case.held_out and case.driver == "run_gabls3_les.py"
    art = load_artifact(art_path)
    assert art.is_moist and art.nt == 4 and art.les_setup["smoke"] is True
    assert art.les_setup["sponge"][0] == 600.0     # the approved base, via the catalog
    rows = _budget_lines(out)
    for lab in ("theta", "u", "v", "water"):
        r = rows[lab]
        assert abs(r["residual"]) <= 1e-13 * abs(r["column"]) + 1e-15, (lab, r)
    # the lid inversion and the 0.25 m surface layer both act
    assert rows["theta"]["lid_heat_flux"] > 0.0
    assert rows["theta"]["sfc_heat_flux"] != 0.0 and rows["water"]["sfc_qv_flux"] != 0.0
    # the recorded forcing integrals are exactly what the steps applied
    dt = float(art.les_setup["dt_s"])
    n = int(round(float(art.les_setup["hours"]) * 3600.0 / dt))
    z = np.asarray(art.heights_m)
    f = gb.coriolis_parameter()
    want = {k: np.zeros(z.size) for k in ("th_adv", "q_adv", "ug", "vg")}
    for i in range(n):
        fr = mod.forcing_at(z, f, i * dt, (i + 1) * dt)
        for k in want:
            want[k] += dt * fr[k]
    # the state change the advection operator made (measured column contents) is the
    # independently integrated prescribed tendency
    dz = float(z[1] - z[0])
    assert rows["theta"]["ls_theta"] == pytest.approx(dz * want["th_adv"].sum(),
                                                       rel=1e-9)
    assert rows["water"]["ls_water"] == pytest.approx(dz * want["q_adv"].sum(),
                                                       rel=1e-9)
    assert rows["water"]["ls_water"] < 0.0
    np.testing.assert_allclose(np.asarray(art.theta_adv_cum)[-1], want["th_adv"],
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(art.qv_adv_cum)[-1], want["q_adv"],
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(art.u_geo_cum)[-1], want["ug"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(art.v_geo_cum)[-1], want["vg"], rtol=1e-12)
    # momentum advection is in the effective geostrophic wind: v_g − u_adv/f
    ug0, vg0 = gb.geostrophic_mean(z, 0.0, dt)
    adv = gb.advection_mean(z, 0.0, dt)
    np.testing.assert_allclose(mod.forcing_at(z, f, 0.0, dt)["vg"],
                               vg0 - adv["u"] / f, rtol=1e-14)
    # the SCM receives interval means that integrate back to the recorded series
    t = np.asarray(art.times_s)
    fn = forcing_profile_fn(art, channel="u_geo")
    integral = sum(np.asarray(fn(0.5 * (t[k] + t[k + 1]))) * (t[k + 1] - t[k])
                   for k in range(art.nt - 1))
    np.testing.assert_allclose(integral, np.asarray(art.u_geo_cum)[-1], rtol=1e-12)
    # the top-cell SGS heat flux carries the lid inversion's downward flux
    assert float(np.asarray(art.wtheta_sgs)[-1, -1]) < 0.0
    # the SCM builds from the artifact with the recorded (no) microphysics
    from legoesm.atmosphere.physics import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.config import (
        MYNN25Config,
        SurfaceLayerConfig,
    )
    turb = TurbulenceConfig(scheme="mynn25", mynn25=MYNN25Config(
        surface=SurfaceLayerConfig(z0=gb.Z0_M, Cd_neutral=1.5e-3, Ch_neutral=0.0)))
    scm, grid = build_cbl_scm_from_artifact(art, turb, nlev=16, dt=dt)
    # the SCM's own forcing callbacks deliver exactly the recorded integrals (on its
    # levels) when stepped at the LES dt
    zs = np.asarray(grid.z_scm_m)
    for ch in ("theta_adv", "qv_adv", "u_geo", "v_geo"):
        fn_scm = getattr(scm.forcing, ch)
        assert fn_scm is not None, ch
        got = sum(np.asarray(fn_scm(k * dt)) for k in range(n)) * dt
        want_scm = np.interp(zs, z, np.asarray(getattr(art, f"{ch}_cum"))[-1])
        np.testing.assert_allclose(got, want_scm, rtol=1e-9, atol=1e-15, err_msg=ch)


def test_gabls3_resumed_run_is_bit_identical(tmp_path):
    from legoesm.atmosphere.les_suite.checkpoint import EXIT_CHECKPOINTED
    mod = _load()
    tiny = _TINY + ["--hours", "0.001"]           # 14 steps, one checkpoint each
    whole = tmp_path / "whole.npz"
    assert mod.main(tiny + ["--emit-suite-artifact", str(whole)]) == 0
    got, ckpt = tmp_path / "chained.npz", tmp_path / "run.ckpt"
    codes = []
    for _ in range(30):
        rc = mod.main(tiny + ["--emit-suite-artifact", str(got), "--checkpoint",
                               str(ckpt), "--wall-limit-s", "0"])
        codes.append(rc)
        if rc != EXIT_CHECKPOINTED:
            break
    assert codes[-1] == 0 and len(codes) >= 3 and not ckpt.exists()
    with np.load(whole, allow_pickle=True) as a, np.load(got, allow_pickle=True) as b:
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in "fc"), k
