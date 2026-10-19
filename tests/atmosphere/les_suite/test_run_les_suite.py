"""CPU dispatch tests for the LES-suite emission driver (no GPU LES integration).

The end-to-end emit (running the spectral core) is GPU-gated and exercised
separately; here we lock the registry dispatch + not-wired-regime hardening.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))


def _driver():
    import run_les_suite  # noqa: PLC0415
    return run_les_suite


def test_unknown_case_raises():
    m = _driver()
    with pytest.raises(Exception):
        m.main(["--case", "does_not_exist"])


def test_not_wired_regime_raises_systemexit():
    # a stratocumulus/moist case is registered but its emission is not wired yet;
    # the driver must refuse (SystemExit), not emit a wrong-regime artifact.
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "dycoms_rf01_sc"])


def test_wired_regimes_contains_dry_convective_and_stable():
    m = _driver()
    assert "dry_convective" in m._WIRED_REGIMES
    assert "dry_stable" in m._WIRED_REGIMES  # SBL now wired


# --- stable (SBL) regime ------------------------------------------------------
def test_build_sbl_stratified_ic_and_geostrophic_wind():
    # _build_sbl builds the config + IC (no integration) → CPU-safe. GABLS1: θ increases
    # with height (STABLE, unlike the CBL's mixed layer), u≈U_g, and Q0<0 (cooling).
    import argparse

    import jax.numpy as jnp
    import numpy as np
    m = _driver()
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    case = get_case("sbl_gabls1")
    assert case.regime == "dry_stable"
    args = argparse.Namespace(theta0=265.0, z0=0.1, pr_sgs=1.0, nu_floor=0.01,
                              q0=None, dt=None)
    g, st, q0 = m._build_sbl(case, args, jnp.float32, sgs="vreman", u_geo_mag=8.0)
    assert g.cfg.nu_floor == 0.01
    thm = np.asarray(st.theta).mean((0, 1))
    nz = thm.shape[0]
    assert thm[-1] > thm[0] + 1.0     # stratified: θ increases upward (unlike a CBL)
    # the free atmosphere (upper half, above the perturbed z<0.5·Lz seed) is monotone
    assert bool(np.all(np.diff(thm[nz // 2:]) >= -1e-3))
    assert float(np.asarray(st.u).mean()) == pytest.approx(8.0, abs=0.2)  # ~geostrophic
    # published GABLS1 surface BC: prescribed cooling, flux from MOST (no fixed Q0)
    assert q0 is None
    assert case.surface_cooling_rate_K_per_h == 0.25
    assert m._surface_temperature(case, 265.0, 3600.0) == pytest.approx(264.75)
    with pytest.raises(SystemExit):   # a fixed flux would contradict the cooling BC
        m._build_sbl(case, argparse.Namespace(**{**vars(args), "q0": -0.01}),
                     jnp.float32, sgs="vreman", u_geo_mag=8.0)


def test_emit_step_factory_rejects_unknown_regime():
    import jax.numpy as jnp
    from legoesm.atmosphere.les_suite.catalog import catalog_case
    m = _driver()
    with pytest.raises(SystemExit):
        m._make_emit_step(None, catalog_case("bomex_cu"), (0.0, 0.0), 0.0, 0.06,
                          jnp.float32)


# --- SGS selection for the D7 σ_LES spread ------------------------------------
def test_sgs_les_config_maps_each_variant():
    m = _driver()
    assert m._sgs_les_config("lasd") == {
        "smagorinsky_dynamic": True, "sgs_model": "smagorinsky"}
    assert m._sgs_les_config("smagorinsky") == {
        "smagorinsky_dynamic": False, "sgs_model": "smagorinsky"}
    assert m._sgs_les_config("vreman") == {
        "smagorinsky_dynamic": False, "sgs_model": "vreman"}


def test_sgs_les_config_unknown_raises():
    m = _driver()
    with pytest.raises(SystemExit):
        m._sgs_les_config("amd")  # not wired into the emit reconstruction → hard error


def test_sgs_not_in_case_variants_raises():
    # --sgs must be one of the case's declared sgs_variants; a stray one is a hard
    # error, not a silent emission of an unclaimed closure.
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--sgs", "amd"])


# --- sheared CBL (U_g axis) ---------------------------------------------------
def test_coriolis_f_from_latitude():
    m = _driver()
    import numpy as np

    from legoesm import constants
    # f = 2Ω sin(φ); GABLS1 latitude ~73° gives ~1.39e-4
    assert m._coriolis_f(0.0) == 0.0
    assert m._coriolis_f(90.0) == pytest.approx(2.0 * constants.Omega)
    assert m._coriolis_f(73.0) == pytest.approx(
        2.0 * constants.Omega * np.sin(np.radians(73.0)))


def test_build_cbl_sheared_initialises_geostrophic_wind():
    import argparse

    import jax.numpy as jnp
    import numpy as np
    m = _driver()
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    case = get_case("cbl_nieuwstadt")
    args = argparse.Namespace(theta0=300.0, z0=0.1, pr_sgs=1.0, gamma=0.008,
                              zi0=800.0, q0=None)
    # free convection: u=0
    g0, st0, _ = m._build_cbl(case, args, jnp.float32, sgs="lasd", u_geo_mag=0.0)
    assert float(np.mean(np.asarray(st0.u))) == pytest.approx(0.0, abs=1e-6)
    # sheared: u initialised to U_g everywhere
    g8, st8, _ = m._build_cbl(case, args, jnp.float32, sgs="lasd", u_geo_mag=8.0)
    assert float(np.mean(np.asarray(st8.u))) == pytest.approx(8.0, abs=1e-5)
    assert float(np.mean(np.asarray(st8.v))) == pytest.approx(0.0, abs=1e-6)


def test_negative_ug_rejected():
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--Ug", "-1.0"])


def test_nonfinite_ug_rejected():
    # a NaN comparison is False, so `--Ug nan` would slip past `< 0` → must be caught
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--Ug", "nan"])


def test_bad_latitude_rejected_for_sheared():
    # these raise at input validation BEFORE the GPU emit (Ug valid, lat invalid)
    m = _driver()
    for lat in ("100", "-91", "nan"):
        with pytest.raises(SystemExit):
            m.main(["--case", "cbl_nieuwstadt", "--Ug", "8", "--lat", lat])


def test_build_cbl_selects_sgs_in_config():
    # _build_cbl only builds the config + IC (no integration) → CPU-safe. Each SGS
    # variant must land in the SpectralLESConfig the core will integrate.
    import argparse

    m = _driver()
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    case = get_case("cbl_nieuwstadt")
    args = argparse.Namespace(
        theta0=300.0, z0=0.1, pr_sgs=1.0, gamma=0.008, zi0=800.0, q0=None)
    import jax.numpy as jnp
    for sgs, dyn, model in (("lasd", True, "smagorinsky"),
                            ("smagorinsky", False, "smagorinsky"),
                            ("vreman", False, "vreman")):
        g, _st, _q0 = m._build_cbl(case, args, jnp.float32, sgs=sgs)
        assert bool(g.cfg.smagorinsky_dynamic) is dyn
        assert g.cfg.sgs_model == model


# --- frame scheduling (codex round-2 regression) ------------------------------
def test_frame_schedule_production_evenly_spaced():
    m = _driver()
    # 7200 steps (dt=1, T=7200), 12 frames → 11 post-IC snapshots, last = n_steps
    rec = m.frame_step_schedule(7200, 12)
    assert len(rec) == 11
    assert rec == sorted(set(rec))           # strictly increasing + distinct
    assert rec[-1] == 7200                    # final frame at T
    assert all(1 <= k <= 7200 for k in rec)


def test_frame_schedule_distinct_and_capped_when_dt_coarse():
    m = _driver()
    # only 3 steps but 6 frames requested → at most 3 distinct post-IC snapshots
    rec = m.frame_step_schedule(3, 6)
    assert rec == [1, 2, 3]                    # distinct, no duplicates/drops
    assert rec == sorted(set(rec))


# --- end-to-end emission on tiny registered cases (CPU) -----------------------
_TINY = dict(nx=8, ny=8, nz=12)


@pytest.fixture
def tiny_cases():
    from legoesm.atmosphere.les_suite.registry import (
        LES_CASE_REGISTRY,
        LESCase,
        LESGrid,
        register_case,
    )
    saved = dict(LES_CASE_REGISTRY)
    LES_CASE_REGISTRY.clear()
    register_case(LESCase(
        name="tiny_sbl", regime="dry_stable", core="spectral",
        grid=LESGrid(**_TINY, Lx_m=400.0, Ly_m=400.0, Lz_m=400.0, dt_s=0.5),
        duration_hours=0.01, surface_theta_flux_K_m_s=None, geostrophic_wind_m_s=8.0,
        sgs_variants=("vreman",), driver="run_spectral_sbl.py", reference="test",
        surface_cooling_rate_K_per_h=25.0))  # fast: cooling ≫ IC noise in 36 s
    register_case(LESCase(
        name="tiny_cbl", regime="dry_convective", core="spectral",
        grid=LESGrid(**_TINY, Lx_m=1600.0, Ly_m=1600.0, Lz_m=1600.0, dt_s=1.0),
        duration_hours=0.01, surface_theta_flux_K_m_s=0.06, geostrophic_wind_m_s=5.0,
        sgs_variants=("lasd",), driver="run_spectral_cbl.py", reference="test"))
    yield
    LES_CASE_REGISTRY.clear()
    LES_CASE_REGISTRY.update(saved)


def _budget_lines(out):
    rows = {}
    for line in out.splitlines():
        if line.strip().startswith("budget "):
            parts = line.split()
            vals = {k: float(v) for k, v in (t.split("=") for t in parts[2:])}
            rows[parts[1].rstrip(":")] = vals
    return rows


@pytest.mark.parametrize("case", ["tiny_sbl", "tiny_cbl"])
def test_emit_writes_turbulence_stats_and_closes_budgets(tiny_cases, tmp_path, capsys,
                                                         case):
    import numpy as np
    from legoesm.atmosphere.les_suite.bridge import (
        SURFACE_SERIES_FIELDS,
        TURBULENCE_STAT_FIELDS,
        load_artifact,
    )
    m = _driver()
    assert m.main(["--case", case, "--frames", "4", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    art = load_artifact(next(tmp_path.glob(f"{case}__*.npz")))
    for name in TURBULENCE_STAT_FIELDS + SURFACE_SERIES_FIELDS:
        arr = np.asarray(getattr(art, name))
        assert arr.shape[0] == art.nt and np.all(np.isfinite(arr)), name
    # the realised surface heat flux series is what the core applied
    if case == "tiny_sbl":
        assert np.all(np.asarray(art.w_theta_s)[1:] < 0.0)   # MOST cooling
        assert "cooling=25.0K/h(MOST)" in out
        assert float(np.max(np.abs(np.asarray(art.wtheta_floor)))) > 0.0
    else:
        np.testing.assert_allclose(np.asarray(art.w_theta_s), 0.06)
        np.testing.assert_allclose(np.asarray(art.wtheta_floor), 0.0)
    # surface value of the SGS momentum flux = the wall stress
    assert np.all(np.asarray(art.tau_x_s)[1:] < 0.0)
    assert np.asarray(art.w_theta_s_cum)[0] == 0.0
    setup = art.les_setup
    assert setup["nu_floor_m2_s"] == (0.01 if case == "tiny_sbl" else 0.0)
    assert setup["surface_bc"] == ("cooling_K_per_h" if case == "tiny_sbl"
                                   else "flux_K_m_s")
    # the SCM is driven by the realised series: interval means of the exact integral
    from legoesm.atmosphere.les_suite.bridge import surface_flux_fn
    fn = surface_flux_fn(art, "w_theta_s")
    t, cum = np.asarray(art.times_s), np.asarray(art.w_theta_s_cum)
    integral = sum(float(fn(0.5 * (t[k] + t[k + 1]))) * (t[k + 1] - t[k])
                   for k in range(art.nt - 1))
    np.testing.assert_allclose(integral, cum[-1], rtol=1e-12)
    assert "nu_floor share of total flux" in out
    rows = _budget_lines(out)
    for lab in ("theta", "u", "v"):
        r = rows[lab]
        # round-off of a column sum: relative to the column CONTENT (θ ~ 1e5 K m)
        assert abs(r["residual"]) <= 1e-13 * abs(r["column"]) + 1e-12, (lab, r)
        assert abs(r["residual"]) < 1e-6 * max(abs(r["d(column)"]), 1e-30), (lab, r)


def test_refine_doubles_grid_halves_dt_and_tags(tiny_cases, tmp_path, capsys):
    from legoesm.atmosphere.les_suite import get_case
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    m = _driver()
    fine = m._refined_case(get_case("tiny_sbl"), 2)
    assert (fine.grid.nx, fine.grid.ny, fine.grid.nz) == (16, 16, 24)
    assert fine.grid.dt_s == 0.25 and fine.grid.Lz_m == 400.0
    assert m.main(["--case", "tiny_cbl", "--refine", "2", "--hours", "0.002",
                   "--frames", "2", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "refine=2" in out and "dt=0.5000" in out    # base CFL dt 1.0 / 2
    art = load_artifact(next(tmp_path.glob("tiny_cbl__lasd__ug5__r2.npz")))
    assert art.nz == 24 and art.les_setup["refine"] == 2
    # an explicit --dt is used as given, not divided by the refinement
    assert m.main(["--case", "tiny_cbl", "--refine", "2", "--dt", "0.7", "--hours",
                   "0.002", "--frames", "2", "--output", str(tmp_path)]) == 0
    assert "dt=0.7000" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        m.main(["--case", "tiny_cbl", "--refine", "0"])


def test_frame_schedule_last_is_nsteps():
    m = _driver()
    for n_steps, frames in [(100, 5), (7, 4), (14400, 12), (5, 10)]:
        rec = m.frame_step_schedule(n_steps, frames)
        assert rec[-1] == n_steps
        assert rec == sorted(set(rec))


def test_emit_step_sponge_is_the_catalog_cases(monkeypatch):
    import jax.numpy as jnp
    from legoesm.atmosphere.les_suite.catalog import catalog_case
    m = _driver()
    got = []
    monkeypatch.setattr(m, "_make_step", lambda *a, sponge=None: got.append(sponge))
    m._make_emit_step(None, catalog_case("sbl_gabls1"), (8.0, 0.0), 1e-4, None,
                      jnp.float64)
    m._make_emit_step(None, catalog_case("cbl_nieuwstadt"), (0.0, 0.0), 0.0, 0.06,
                      jnp.float64)
    assert got == [(0.75 * 400.0, m._SBL_TAU_SPONGE_S), None]
