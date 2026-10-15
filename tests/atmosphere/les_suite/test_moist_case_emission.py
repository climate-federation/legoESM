"""End-to-end moist reference emission on tiny grids (CPU, x64): each cloudy case
driver (BOMEX, DYCOMS RF01 / RF02, ASTEX, RICO) runs with Kessler microphysics and
``--emit-suite-artifact``; the artifact must carry the turbulence + moisture
statistics, the realised surface series, the microphysics record the SCM reads,
and the printed column budgets must close to round-off."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_RUN = Path(__file__).resolve().parents[3] / "scripts" / "run"
_TINY = ["--nx", "8", "--ny", "8", "--nz", "12", "--record-frames", "3",
         "--microphysics", "kessler", "--print-every", "100000"]


def _load(name):
    if str(_RUN) not in sys.path:
        sys.path.insert(0, str(_RUN))
    spec = importlib.util.spec_from_file_location(name, _RUN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _budget_lines(out):
    rows = {}
    for line in out.splitlines():
        if line.strip().startswith("budget "):
            parts = line.split()
            rows[parts[1].rstrip(":")] = {
                k: float(v) for k, v in (t.split("=") for t in parts[2:])}
    return rows


_CASES = [
    ("run_bomex_les", ["--hours", "0.004", "--dt", "1.0"], "bomex_cu"),
    ("run_dycoms_les", ["--case", "dycoms", "--hours", "0.002"], "dycoms_rf01_sc"),
    ("run_dycoms_les", ["--case", "rf02", "--hours", "0.002"], "dycoms_rf02_sc"),
    ("run_dycoms_les", ["--case", "astex", "--hours", "0.002"], "astex_sc"),
    ("run_rico_les", ["--hours", "0.004", "--dt", "1.0"], "rico_cu"),
]


@pytest.mark.parametrize("driver,extra,label", _CASES, ids=[c[2] for c in _CASES])
def test_moist_driver_emits_complete_artifact_and_closes_budgets(
        driver, extra, label, tmp_path, capsys, monkeypatch):
    from legoesm.atmosphere.les_suite.bridge import (
        MOIST_STAT_FIELDS,
        SURFACE_SERIES_FIELDS,
        TURBULENCE_STAT_FIELDS,
        load_artifact,
        surface_flux_fn,
    )
    from legoesm.atmosphere.les_suite.scm_runner import moist_scm_microphysics
    mod = _load(driver)
    art_path = tmp_path / f"{label}.npz"
    # the registry name goes in --case-label (the drivers' default labels are the
    # deck names); the scorecard resolves an artifact's regime by that name
    monkeypatch.setattr(sys, "argv", [driver] + _TINY + extra + [
        "--case-label", label, "--output", str(tmp_path / "rec"),
        "--emit-suite-artifact", str(art_path)])
    assert mod.main() == 0
    out = capsys.readouterr().out
    art = load_artifact(art_path)
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    assert get_case(art.case_name).driver == f"{driver}.py"
    # --record-frames 3 = initial + two in-loop frames, plus the terminal state
    assert art.is_moist and art.case_name == label and art.nt == 4
    t_s = np.asarray(art.times_s)
    hours = float(extra[extra.index("--hours") + 1])
    assert t_s[0] == 0.0 and t_s[-1] >= hours * 3600.0 and np.all(np.diff(t_s) > 0)
    for name in (TURBULENCE_STAT_FIELDS + SURFACE_SERIES_FIELDS + MOIST_STAT_FIELDS
                 + ("w_qv_s", "w_qv_s_cum", "precip_s", "precip_s_cum",
                    "wqt_sgs", "qc")):
        arr = np.asarray(getattr(art, name))
        assert arr.shape[0] == art.nt and np.all(np.isfinite(arr)), name
    # SGS fluxes are the closure's own, not zeros; the surface face is the applied flux
    assert np.max(np.abs(np.asarray(art.wqt_sgs))) > 0.0
    assert np.max(np.abs(np.asarray(art.wtheta_sgs))) > 0.0
    assert np.asarray(art.w_qv_s_cum)[0] == 0.0 and np.asarray(art.precip_s_cum)[-1] >= 0.0
    # the SCM is driven by the realised moisture-flux series: its interval means
    # integrate to exactly the recorded integral
    fn = surface_flux_fn(art, "w_qv_s")
    t, cum = np.asarray(art.times_s), np.asarray(art.w_qv_s_cum)
    integral = sum(float(fn(0.5 * (t[k] + t[k + 1]))) * (t[k + 1] - t[k])
                   for k in range(art.nt - 1))
    np.testing.assert_allclose(integral, cum[-1], rtol=1e-12)
    # the SCM arm runs the LES's own microphysics, coefficients included
    mp = moist_scm_microphysics(art)
    assert mp.scheme == "kessler"
    assert art.les_setup["microphysics"]["kessler"] == dict(mp.kessler._asdict())
    # subsidence sinks (positive-up convention) wherever it is applied
    assert float(np.min(np.asarray(art.subsidence_w))) < 0.0
    rows = _budget_lines(out)
    assert set(rows) == {"theta", "u", "v", "water", "micro_water",
                         "micro_enthalpy"}, out
    for lab, r in rows.items():
        assert abs(r["residual"]) <= 1e-13 * abs(r["column"]) + 1e-15, (lab, r)
    assert rows["water"]["sfc_qv_flux"] > 0.0          # evaporation entered the column
    np.testing.assert_allclose(rows["micro_water"]["precip"],
                               -float(np.asarray(art.precip_s_cum)[-1]), rtol=1e-6, atol=1e-30)
    if driver != "run_dycoms_les":
        # theta_adv / qv_adv are what the LES forcing operator applies: one forcing
        # step on a height-uniform state (no subsidence term) changes θ and q_v by
        # exactly dt·theta_adv and dt·qv_adv (tls is a T tendency the LES converts
        # with the reference Exner; the SCM reads theta_adv as dθ/dt)
        import jax.numpy as jnp
        bx = getattr(mod, "bx", mod)
        monkeypatch.setattr(sys, "argv", [driver] + _TINY + extra)
        g, st, ref, forc, _th = bx.build(mod.parse_args(), jnp.float64)
        st0 = st._replace(theta=jnp.full_like(st.theta, 300.0),
                          tracers=st.tracers.at[..., 0].set(0.01))
        dt = 3.0
        st1 = bx.make_forcing_fn(g, ref, forc, jnp.float64)(st0, dt)
        dth = np.asarray((st1.theta - st0.theta).mean((0, 1))) / dt
        dqv = np.asarray((st1.tracers[..., 0] - st0.tracers[..., 0]).mean((0, 1))) / dt
        np.testing.assert_allclose(art.theta_adv, dth, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(art.qv_adv, dqv, rtol=1e-8, atol=1e-15)
        assert float(np.max(np.abs(art.theta_adv))) > 0.0


def test_rico_records_interactive_surface_fluxes(tmp_path, monkeypatch, capsys):
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    mod = _load("run_rico_les")
    art_path = tmp_path / "rico.npz"
    monkeypatch.setattr(sys, "argv", ["run_rico_les"] + _TINY + [
        "--hours", "0.004", "--dt", "1.0", "--output", str(tmp_path / "rec"),
        "--emit-suite-artifact", str(art_path)])
    assert mod.main() == 0
    art = load_artifact(art_path)
    w = np.asarray(art.w_qv_s)
    assert np.all(w > 0.0) and np.ptp(w) > 0.0          # bulk flux evolves with the state
    assert art.les_setup["surface"].startswith("interactive bulk")


def test_dycoms_theta_adv_is_the_lw_cooling(tmp_path, monkeypatch, capsys):
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    mod = _load("run_dycoms_les")
    arts = []
    # the applied radiative forcing is accumulated every step it fires, so the
    # exchanged profile cannot depend on how often the driver prints (the old
    # print-cadence sample gave zero cooling when nothing was printed)
    for every in ("1", "100000"):
        art_path = tmp_path / f"dycoms_{every}.npz"
        monkeypatch.setattr(sys, "argv", ["run_dycoms_les"] + _TINY + [
            "--case", "dycoms", "--hours", "0.004", "--print-every", every,
            "--output", str(tmp_path / f"rec_{every}"),
            "--emit-suite-artifact", str(art_path)])
        assert mod.main() == 0
        arts.append(load_artifact(art_path))
    np.testing.assert_array_equal(arts[0].theta_adv, arts[1].theta_adv)
    art = arts[1]
    assert float(np.min(art.theta_adv)) < 0.0          # cloud-top radiative cooling
    assert float(np.max(np.abs(art.qv_adv))) == 0.0    # no prescribed q_v advection
    # the cloudy start is recorded: θ_l below the mean θ where cloud water exists
    assert float(np.max(np.asarray(art.qc)[0])) > 0.0


def test_rain_reaches_the_surface_and_the_microphysics_closes_with_it(
        tmp_path, monkeypatch, capsys):
    """A precipitating run (RF01 with Kessler autoconversion made fast and rain
    evaporation off so rain forms and falls out within minutes): the surface
    precipitation is nonzero, the microphysics' density-weighted water change
    equals minus that precipitation,
    and every column budget still closes to round-off."""
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig,
        MicrophysicsConfig,
    )
    mod = _load("run_dycoms_les")
    # evaporation off: the default coefficient evaporates sub-saturated rain within
    # one step, so with it no rain would reach the surface
    fast = KesslerConfig(autoconversion_threshold=1e-4, autoconversion_rate=5e-2,
                         evaporation_coeff=0.0)

    def _cfg(**kw):
        if kw.get("scheme") == "kessler":
            kw.setdefault("kessler", fast)
        return MicrophysicsConfig(**kw)

    monkeypatch.setattr(mod, "MicrophysicsConfig", _cfg)
    art_path = tmp_path / "rain.npz"
    monkeypatch.setattr(sys, "argv", ["run_dycoms_les"] + _TINY + [
        "--case", "dycoms", "--hours", "0.08", "--output", str(tmp_path / "rec"),
        "--emit-suite-artifact", str(art_path)])
    assert mod.main() == 0
    rows = _budget_lines(capsys.readouterr().out)
    art = load_artifact(art_path)
    assert art.les_setup["microphysics"]["kessler"]["autoconversion_rate"] == 5e-2
    p_cum = float(np.asarray(art.precip_s_cum)[-1])
    assert p_cum > 1e-6                                     # rain left the column
    assert float(np.max(np.asarray(art.qr))) > 0.0
    np.testing.assert_allclose(rows["micro_water"]["d(column)"], -p_cum, rtol=1e-6)
    # 576 steps, each adding ~7 column-sum differences that round at ~1 ulp of the
    # column: the accumulated round-off bound grows with the step count
    n_steps = 0.08 * 3600.0 / 0.5
    for lab, r in rows.items():
        assert abs(r["residual"]) <= 2e-15 * n_steps * abs(r["column"]) + 1e-15, (lab, r)


def test_emit_requires_record_frames(tmp_path, monkeypatch):
    mod = _load("run_bomex_les")
    monkeypatch.setattr(sys, "argv", ["run_bomex_les"] + [
        "--nx", "8", "--ny", "8", "--nz", "12", "--hours", "0.001",
        "--record-frames", "0", "--output", str(tmp_path),
        "--emit-suite-artifact", str(tmp_path / "x.npz")])
    with pytest.raises(SystemExit):
        mod.main()


def test_record_theta_l_derives_thetal_not_theta():
    # The spectral moist LES prognoses ACTUAL θ; recorded frames carry
    # θ_l = θ − (L_v/c_pd)·q_c/Π from θ + cloud q_c (slot 1), rain excluded.
    from legoesm import constants
    mod = _load("run_bomex_les")
    ny, nx, nz = 2, 2, 6
    theta = np.full((ny, nx, nz), 300.0)
    tr = np.zeros((ny, nx, nz, 4))
    tr[..., 1] = 1.0e-3
    tr[..., 2] = 5.0e-4
    exner = np.linspace(1.0, 0.9, nz)
    thl = mod._record_theta_l(SimpleNamespace(theta=theta, tracers=tr),
                              SimpleNamespace(exner_c=exner))
    expect = 300.0 - (constants.L_v / constants.c_pd) * 1.0e-3 / exner
    assert np.allclose(thl[0, 0], expect, rtol=1e-12)
    assert np.all(thl < theta)


def test_moist_cases_resolve_to_registry_regimes():
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    assert get_case("dycoms_rf01_sc").regime == "stratocumulus"
    assert get_case("dycoms_rf02_sc").regime == "stratocumulus"
    assert get_case("astex_sc").regime == "stratocumulus"
    assert get_case("rico_cu").regime == "shallow_cumulus"
    assert get_case("bomex_cu").regime == "shallow_cumulus"
