"""Published full-setup LES cases: catalog parameters, the unapproved-value gate,
the generic setup builder end to end (column budgets closed), the DEPHY reader for
a dry LES, and bit-identical checkpoint/restart.

CPU, tiny grids. The physical parameters asserted here are the papers' own
(Conzemius & Fedorovich 2006 preprint 5.6; Sullivan & Patton 2011; Sullivan et
al. 2016; Huang & Bou-Zeid 2013; Couvreux et al. 2020 via DEPHY; Ayotte 1996 via
DEPHY), so an edit that drifts a case off its publication fails here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.les_suite.catalog import PUBLISHED_CASES  # noqa: E402
from legoesm.atmosphere.les_suite.registry import (  # noqa: E402
    LES_CASE_REGISTRY,
    LESCase,
    LESGrid,
    LESSetup,
    RegistryError,
    register_case,
)

_BY_NAME = {c.name: c for c in PUBLISHED_CASES}
# The user's 2026-10-15 split: held out = Ayotte, Sullivan–Patton, GABLS3, GABLS4,
# neutral Ekman, RICO; everything else trains.
_HELD_OUT_PREFIXES = ("ayotte96_", "sullivan_patton11", "gabls3", "gabls4", "ekman",
                      "rico")


def _driver():
    import run_les_suite  # noqa: PLC0415
    return run_les_suite


def _f(lat):
    return 2.0 * constants.Omega * math.sin(math.radians(lat))


# --- catalog ----------------------------------------------------------------
def test_every_published_case_validates_and_runs_through_the_suite_driver():
    assert len(_BY_NAME) == len(PUBLISHED_CASES) == 26
    for c in PUBLISHED_CASES:
        c.validate()
        assert c.setup is not None and c.driver == "run_les_suite.py"
        # user 2026-10-15: every proposed value approved, incl. GABLS4's raised dt
        # the Pedersen Ekman values and the FC01 suite values and duration
        assert c.setup.approved, c.name
        assert not c.setup.pending, c.name


def test_held_out_split_is_the_users():
    for c in PUBLISHED_CASES:
        assert c.held_out == c.name.startswith(_HELD_OUT_PREFIXES), c.name


def test_conzemius_fedorovich_matrix_points():
    want = {  # name: (flux K m/s, lapse rate K/m, U_g at surface, U_g at 1600 m)
        "cf06_q10_g03_ns": (0.10, 0.003, 0.0, 0.0),
        "cf06_q30_g01_ns": (0.30, 0.001, 0.0, 0.0),
        "cf06_q03_g10_ns": (0.03, 0.010, 0.0, 0.0),
        "cf06_q10_g03_gc": (0.10, 0.003, 20.0, 20.0),
        "cf06_q10_g03_gs": (0.10, 0.003, 0.0, 20.0),
    }
    m = _driver()
    for name, (q, gam, ug0, ugtop) in want.items():
        c = _BY_NAME[name]
        s = c.setup
        assert c.surface_theta_flux_K_m_s == q
        assert (s.lat_deg, s.z0_m, s.theta_ref_K) == (40.0, 0.01, 300.0)
        # θ = 300 K + γ z from the surface (initial CBL depth 0)
        np.testing.assert_allclose(m._knots([0.0, 800.0, 1600.0], s.theta_knots, 1),
                                   [300.0, 300.0 + 800 * gam, 300.0 + 1600 * gam])
        np.testing.assert_allclose(m._knots([0.0, 1600.0], s.geostrophic_knots, 1),
                                   [ug0, ugtop])
        assert s.sponge_bottom_m == 0.8 * c.grid.Lz_m            # top 20 %
        assert (c.grid.Lx_m / c.grid.nx, c.grid.Lz_m / c.grid.nz) == (20.0, 20.0)
    assert any("NOT RUN in the paper" in p
               for p in _BY_NAME["cf06_q03_g10_ns"].setup.approved)


def test_sullivan_patton_run_d():
    c = _BY_NAME["sullivan_patton11_d"]
    s = c.setup
    assert (c.grid.nx, c.grid.ny, c.grid.nz) == (256, 256, 256)
    assert (c.grid.Lx_m, c.grid.Lz_m) == (5120.0, 2048.0)
    assert c.surface_theta_flux_K_m_s == 0.24
    np.testing.assert_allclose(_f(s.lat_deg), 1.0e-4, rtol=1e-12)
    m = _driver()
    np.testing.assert_allclose(m._knots([500.0, 974.0, 1024.0, 1074.0, 2074.0],
                                        s.theta_knots, 1),
                               [300.0, 300.0, 304.0, 308.0, 308.0 + 0.003 * 974.0])
    assert s.z0_m == 0.1


@pytest.mark.parametrize("name,rate,grid", [
    ("sullivan16_cr025", 0.25, (200, 200, 200, 400.0)),
    ("sullivan16_cr100", 1.0, (200, 200, 200, 400.0)),
    ("huang13_cr100", 1.0, (162, 162, 160, 800.0)),
    ("huang13_cr250", 2.5, (162, 162, 160, 800.0)),
])
def test_gabls1_family_cases(name, rate, grid):
    c = _BY_NAME[name]
    s = c.setup
    assert c.surface_cooling_rate_K_per_h == rate
    assert (c.grid.nx, c.grid.ny, c.grid.nz, c.grid.Lx_m) == grid
    assert c.grid.Lz_m == 400.0
    np.testing.assert_allclose(_f(s.lat_deg), 1.39e-4, rtol=1e-12)
    assert (s.z0_m, s.z0h_m, s.theta_ref_K) == (0.1, None, 265.0)
    # Huang 2013 too gets the suite's GABLS1 sponge (user 2026-10-15)
    assert (s.sponge_bottom_m, s.sponge_tau_s) == (300.0, 50.0)
    m = _driver()
    # sharp GABLS1 sounding as the papers state it: 265 K to 100 m, then +0.01 K/m
    np.testing.assert_allclose(m._knots([50.0, 100.0, 200.0], s.theta_knots, 1),
                               [265.0, 265.0, 266.0])
    np.testing.assert_allclose(m._knots([0.0], s.geostrophic_knots, 1), [8.0])


@pytest.mark.parametrize("run,gamma,hours", [("n01", 0.001, 24.0), ("n02", 0.003, 32.0),
                                              ("n03", 0.006, 24.0), ("n04", 0.010, 24.0)])
def test_pedersen14_neutral_ekman_cases(run, gamma, hours):
    """Pedersen, Gryning & Kelly (2014) Sect. 2.2 + Table 1 neutral cases."""
    c = _BY_NAME[f"ekman_pedersen14_{run}"]
    s = c.setup
    assert (c.regime, c.held_out, c.duration_hours) == ("dry_neutral", True, hours)
    assert (c.grid.nx, c.grid.ny, c.grid.nz) == (256, 256, 256)
    assert (c.grid.Lx_m, c.grid.Ly_m, c.grid.Lz_m) == (3000.0, 3000.0, 2000.0)
    assert c.surface_theta_flux_K_m_s == 0.0 and s.z0_m == 0.01
    np.testing.assert_allclose(_f(s.lat_deg), 1.0e-4, rtol=1e-12)
    m = _driver()
    np.testing.assert_allclose(m._knots([0.0, 1000.0], s.theta_knots, 1),
                               [290.0, 290.0 + gamma * 1000.0])
    np.testing.assert_allclose(m._knots([0.0, 2000.0], s.geostrophic_knots, 1), [10.0] * 2)
    np.testing.assert_allclose(m._knots([0.0, 2000.0], s.wind_init_knots, 1), [10.0] * 2)
    # the values the paper does not fix were proposed, then approved by the user
    assert c.grid.dt_s == 0.5 and (s.sponge_bottom_m, s.sponge_tau_s) == (1600.0, 50.0)
    topics = [p.removeprefix("APPROVED 2026-10-15 (was: ").split(":")[0]
              for p in s.approved]
    assert topics == ["dt", "top boundary", "θ_ref", "seed noise",
                      "SGS closure + numerics"], topics


def test_dephy_cases_point_at_registered_files():
    from legoesm.atmosphere.forcing.scm.dephy_scm import DEPHY_SCM_CASES
    for c in PUBLISHED_CASES:
        if c.setup.dephy_case is not None:
            assert c.setup.dephy_case in DEPHY_SCM_CASES, c.name
    g4 = _BY_NAME["gabls4_e2"].setup
    assert (g4.lat_deg, g4.z0_m, g4.z0h_m) == (-75.1, 0.001, 0.0001)
    assert _BY_NAME["gabls4_e2"].duration_hours == 24.0
    assert _BY_NAME["gabls4_e2"].grid.dt_s == 0.3          # user 2026-10-15


# --- registry rules -----------------------------------------------------------
def _setup(**kw):
    base = dict(lat_deg=45.0, z0_m=0.1, theta_ref_K=300.0, perturb_K=0.1,
                perturb_top_m=100.0, theta_knots=((0.0, 300.0), (800.0, 302.4)),
                geostrophic_knots=((0.0, 5.0, 0.0),))
    base.update(kw)
    return LESSetup(**base)


def _case(name="tiny_setup", **kw):
    base = dict(name=name, regime="dry_convective", core="spectral",
                grid=LESGrid(nx=8, ny=8, nz=12, Lx_m=1600.0, Ly_m=1600.0, Lz_m=800.0,
                             dt_s=1.0),
                duration_hours=0.01, surface_theta_flux_K_m_s=0.06,
                geostrophic_wind_m_s=None, sgs_variants=("lasd",),
                driver="run_les_suite.py", reference="test", setup=_setup())
    base.update(kw)
    return LESCase(**base)


def test_setup_needs_exactly_one_surface_forcing():
    _case().validate()
    with pytest.raises(RegistryError, match="exactly one surface forcing"):
        _case(surface_theta_flux_K_m_s=None).validate()
    with pytest.raises(RegistryError, match="exactly one surface forcing"):
        _case(setup=_setup(surface_theta_knots=((0.0, 300.0), (60.0, 301.0)))).validate()


@pytest.mark.parametrize("bad", [
    dict(surface_drag="nope"), dict(sponge_bottom_m=600.0),
    dict(theta_knots=((10.0, 300.0), (5.0, 301.0))), dict(z0h_m=0.0),
    dict(dephy_case="ayotte_05sc"),
])
def test_setup_rejects_bad_values(bad):
    with pytest.raises(RegistryError):
        _case(setup=_setup(**bad)).validate()


def test_dephy_setup_rejects_an_initial_wind_it_would_ignore():
    s = _setup(dephy_case="ayotte_05sc", theta_knots=(), geostrophic_knots=(),
               wind_init_knots=((0.0, 99.0, 0.0),))
    with pytest.raises(RegistryError, match="do not also give knots"):
        s.validate("x")
    import dataclasses
    dataclasses.replace(s, wind_init_knots=()).validate("x")


# --- driver: gate, clashes, end to end ----------------------------------------
@pytest.fixture
def tiny_registry():
    saved = dict(LES_CASE_REGISTRY)
    LES_CASE_REGISTRY.clear()
    # flux-forced, MOST drag, a sheared geostrophic PROFILE and a sponge
    register_case(_case(name="tiny_flux", setup=_setup(
        geostrophic_knots=((0.0, 2.0, 0.0), (800.0, 10.0, -1.0)),
        sponge_bottom_m=600.0, sponge_tau_s=50.0)))
    # a prescribed surface-θ SERIES with separate thermal roughness (GABLS4 path)
    register_case(_case(
        name="tiny_ts", regime="dry_stable", surface_theta_flux_K_m_s=None,
        grid=LESGrid(nx=8, ny=8, nz=12, Lx_m=400.0, Ly_m=400.0, Lz_m=400.0, dt_s=0.5),
        setup=_setup(lat_deg=-75.1, z0_m=0.01, z0h_m=0.001, theta_ref_K=265.0,
                     theta_knots=((0.0, 265.0), (100.0, 265.0), (400.0, 268.0)),
                     geostrophic_knots=((0.0, 4.0, 2.0),),
                     surface_theta_knots=((0.0, 265.0), (3600.0, 240.0)),
                     sponge_bottom_m=300.0, sponge_tau_s=50.0)))
    register_case(_case(name="tiny_pending",
                        setup=_setup(pending=("dt: PROPOSED 1 s",),
                                     approved=("seed: APPROVED 0.1 K",))))
    yield
    LES_CASE_REGISTRY.clear()
    LES_CASE_REGISTRY.update(saved)


def _budget_lines(out):
    rows = {}
    for line in out.splitlines():
        if line.strip().startswith("budget "):
            parts = line.split()
            rows[parts[1].rstrip(":")] = {k: float(v) for k, v in
                                          (t.split("=") for t in parts[2:])}
    return rows


def test_pending_case_refuses_production_and_allows_smoke(tiny_registry, tmp_path):
    m = _driver()
    with pytest.raises(SystemExit, match="not approved"):
        m.main(["--case", "tiny_pending", "--output", str(tmp_path)])
    assert m.main(["--case", "tiny_pending", "--smoke", "--hours", "0.01",
                   "--frames", "2",
                   "--output", str(tmp_path)]) == 0
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    art = load_artifact(tmp_path / "tiny_pending__lasd__smoke.npz")
    assert art.les_setup["smoke"] is True
    assert art.les_setup["setup"]["pending"] == ["dt: PROPOSED 1 s"]
    assert art.les_setup["setup"]["approved"] == ["seed: APPROVED 0.1 K"]


@pytest.mark.parametrize("flag", [["--z0", "0.2"], ["--theta0", "290"], ["--Ug", "3"],
                                  ["--q0", "0.1"], ["--lat", "30"], ["--gamma", "0.01"],
                                  ["--zi0", "500"], ["--dt", "0.25"], ["--pr-sgs", "0.5"],
                                  ["--nu-floor", "0.1"]])
def test_cli_cannot_override_a_setup(tiny_registry, tmp_path, flag):
    with pytest.raises(SystemExit, match="conflict with the case's LESSetup"):
        _driver().main(["--case", "tiny_flux", *flag, "--output", str(tmp_path)])


def test_published_cases_record_the_approved_driver_closure():
    """Every published case records the suite closure as an approved item, and
    the item quotes the values the driver actually uses."""
    from legoesm.atmosphere.les_suite import catalog
    m = _driver()
    text = catalog._PENDING_CLOSURE
    for v in (f"Prandtl {m._PR_SGS}", f"viscosity {m._NU_FLOOR_M2_S} m2/s",
              f"C_s {m._SBL_STATIC_CS}"):
        assert v in text, v
    for c in catalog.PUBLISHED_CASES:
        assert c.setup.approved[-1] == f"APPROVED 2026-10-15 (was: {text})", c.name


@pytest.mark.parametrize("tag,n1,n2", [("lin_n008", 0.008, 0.008),
                                       ("lin_n011", 0.011, 0.011),
                                       ("lin_n014", 0.014, 0.014),
                                       ("ws", 0.005, 0.018), ("sw", 0.018, 0.005)])
def test_fedorovich_conzemius_2001_cases(tag, n1, n2):
    """Fedorovich & Conzemius (2001): B_s = 0.0098 m2/s3, 5 x 5 x 4 km at
    100 x 100 x 20 m, top-20 % sponge, stop at z_i = 0.6 Lz; user fill-ins
    θ0 = 300 K, switch 1000 m, z0 = 0.01 m, linear from the surface."""
    c = _BY_NAME[f"fc01_{tag}"]
    s = c.setup
    assert (c.regime, c.held_out, c.geostrophic_wind_m_s) == ("dry_convective",
                                                                False, None)
    assert (c.grid.nx, c.grid.ny, c.grid.nz) == (50, 50, 200)
    assert (c.grid.Lx_m, c.grid.Ly_m, c.grid.Lz_m) == (5000.0, 5000.0, 4000.0)
    # duration: the time the CBL (C_eps = 0.17, paper Fig. 2) needs to reach
    # 0.6 Lz = 2400 m. Linear N: z_i = sqrt(2 (1+2C) B_s t) / N (paper eq. 10);
    # two-layer: the mixed layer must gain the buoyancy b0(z_i) over the depth,
    # i.e. ∫_0^zi (b0(z_i) - b0(z)) dz, evaluated here by quadrature.
    zi, zs, k = 2400.0, 1000.0, (1.0 + 2 * 0.17) * 0.0098
    z = np.linspace(0.0, zi, 240001)
    b0 = np.where(z <= zs, n1 ** 2 * z, n1 ** 2 * zs + n2 ** 2 * (z - zs))
    f = b0[-1] - b0
    t = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(z))) / k
    if n1 == n2:
        np.testing.assert_allclose(t, (zi * n1) ** 2 / (2 * k), rtol=1e-9)
    np.testing.assert_allclose(c.duration_hours * 3600.0, t, rtol=1e-8)
    assert c.duration_hours * 3600.0 < 1.0e5
    np.testing.assert_allclose(c.surface_theta_flux_K_m_s * constants.g / 300.0,
                               0.0098, rtol=1e-12)
    assert s.z0_m == 0.01 and s.theta_ref_K == 300.0 and s.sponge_bottom_m == 3200.0
    m = _driver()
    th = m._knots([0.0, 1000.0, 4000.0], s.theta_knots, 1)
    np.testing.assert_allclose((th[1] - th[0]) / 1000.0 * constants.g / 300.0, n1 ** 2,
                               rtol=1e-12)
    np.testing.assert_allclose((th[2] - th[1]) / 3000.0 * constants.g / 300.0, n2 ** 2,
                               rtol=1e-12)
    fills = [a for a in s.approved if "user fill-in" in a]
    assert len(fills) == 3 and any("1000 m" in a for a in fills)
    assert not s.pending                          # user approved 2026-10-15
    topics = [p.removeprefix("APPROVED 2026-10-15 (was: ").split(":")[0]
              for p in s.approved[3:]]
    assert s.lat_deg == 0.0                       # paper: Coriolis neglected
    assert topics == ["dt", "sponge", "seed noise", "surface drag", "duration",
                      "SGS closure + numerics"], topics


def test_stable_dry_grid_points_run_on_the_gabls1_grid():
    """User 2026-10-15: the stable dry-grid points on the GABLS1 grid; the
    convective points keep the 33 m grid."""
    from legoesm.atmosphere.les_suite import catalog
    for c in catalog.dry_shear_buoyancy_grid():
        g = c.grid
        if c.regime == "dry_stable":
            assert (g.nx, g.ny, g.nz, g.Lx_m, g.Lz_m, g.dt_s) == (64, 64, 96, 400.0,
                                                                  400.0, 0.1), c.name
        else:
            assert g == catalog._DRY_GRID, c.name


def test_setup_duration_is_the_published_one(tiny_registry, tmp_path):
    """--hours may only shorten a setup case under --smoke; --smoke needs --hours."""
    with pytest.raises(SystemExit, match="published duration"):
        _driver().main(["--case", "tiny_flux", "--hours", "0.005",
                        "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="--smoke needs --hours"):
        _driver().main(["--case", "tiny_flux", "--smoke", "--output", str(tmp_path)])


def test_dry_neutral_needs_a_setup(tiny_registry, tmp_path):
    register_case(LESCase(
        name="tiny_neutral_legacy", regime="dry_neutral", core="spectral",
        grid=LESGrid(nx=8, ny=8, nz=12, Lx_m=800.0, Ly_m=800.0, Lz_m=800.0, dt_s=1.0),
        duration_hours=0.01, surface_theta_flux_K_m_s=0.0, geostrophic_wind_m_s=5.0,
        sgs_variants=("lasd",), driver="run_les_suite.py", reference="test"))
    with pytest.raises(SystemExit, match="dry_neutral case needs a full LESSetup"):
        _driver().main(["--case", "tiny_neutral_legacy", "--output", str(tmp_path)])


@pytest.mark.parametrize("case", ["tiny_flux", "tiny_ts"])
def test_setup_case_emits_and_closes_column_budgets(tiny_registry, tmp_path, capsys,
                                                     case):
    """RULE 0 for the new forcing paths: a height-varying geostrophic wind, the
    prescribed-flux MO drag, a surface-θ series with its own thermal roughness and
    the sponge — the column change of θ, u, v equals the boundary + body sources
    the RK stages applied plus the sponge term, to round-off."""
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    m = _driver()
    assert m.main(["--case", case, "--frames", "4", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    art = load_artifact(tmp_path / f"{case}__lasd.npz")
    rows = _budget_lines(out)
    # the prescribed-surface-temperature solver reports its convergence residual
    resid = art.les_setup["most_residual_max"]
    bad = art.les_setup["most_residual_nonfinite_frames"]
    if case == "tiny_ts":
        assert resid is not None and np.isfinite(resid) and resid >= 0.0 and bad == 0
        assert "MOST-resid=" in out
    else:
        assert resid is None and bad is None and "MOST-resid=" not in out
    # the emitted record carries the roughness the SCM tuner reads back
    from legoesm.atmosphere.les_suite.bridge import surface_roughness
    s = LES_CASE_REGISTRY[case].setup
    assert surface_roughness(art) == (s.z0_m, s.z0_m if s.z0h_m is None else s.z0h_m)
    if case == "tiny_ts":
        assert surface_roughness(art) == (0.01, 0.001)
    for lab in ("theta", "u", "v"):
        r = rows[lab]
        assert abs(r["residual"]) <= 1e-13 * abs(r["column"]) + 1e-12, (lab, r)
        assert abs(r["d(column)"]) > 0.0, (lab, r)
    z = np.asarray(art.heights_m)
    s = LES_CASE_REGISTRY[case].setup
    np.testing.assert_allclose(np.asarray(art.u_geo), m._knots(z, s.geostrophic_knots, 1))
    np.testing.assert_allclose(np.asarray(art.v_geo), m._knots(z, s.geostrophic_knots, 2))
    np.testing.assert_allclose(art.f_c, 2.0 * constants.Omega
                               * math.sin(math.radians(s.lat_deg)), rtol=1e-12)
    if case == "tiny_flux":
        np.testing.assert_allclose(np.asarray(art.w_theta_s), 0.06)
        assert art.les_setup["sfc_flux_drag"] == "most"
        assert "Q0=0.06(drag=most)" in out
    else:
        assert art.f_c < 0.0                       # southern hemisphere
        assert art.les_setup["surface_bc"] == "surface_theta_series"
        assert art.les_setup["z0h_m"] == 0.001
        assert np.all(np.asarray(art.w_theta_s)[1:] < 0.0)   # cooled surface
        assert "theta_s=series[265.00..240.00K](MOST)" in out


def test_setup_run_outlasting_its_surface_series_is_refused(tiny_registry, tmp_path):
    with pytest.raises(SystemExit, match="outlasts its surface-temperature series"):
        _driver().main(["--case", "tiny_ts", "--smoke", "--hours", "2",
                        "--output", str(tmp_path)])


# --- checkpoint / restart --------------------------------------------------------
def test_restart_is_bit_identical(tiny_registry, tmp_path, capsys, monkeypatch):
    """A run stopped at its wall limit and resumed writes the SAME artifact, array
    for array, as an uninterrupted run; a checkpoint from a different run
    configuration is refused."""
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    m = _driver()
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    common = ["--case", "tiny_flux", "--frames", "4"]
    assert m.main(common + ["--output", str(a_dir)]) == 0
    ck = tmp_path / "ck.npz"
    rc = m.main(common + ["--output", str(b_dir), "--checkpoint", str(ck),
                          "--wall-limit-s", "0"])
    assert rc == m.EXIT_CHECKPOINTED and ck.exists()
    with pytest.raises(SystemExit, match="different run configuration"):
        m.main(["--case", "tiny_flux", "--frames", "5",
                "--output", str(b_dir), "--checkpoint", str(ck)])
    orig = m._resolve_setup

    def shifted(case, z_c):              # same options, different resolved forcing
        r = dict(orig(case, z_c))
        r["theta"] = np.asarray(r["theta"]) + 1e-6
        return r
    monkeypatch.setattr(m, "_resolve_setup", shifted)
    with pytest.raises(SystemExit, match="different run configuration"):
        m.main(common + ["--output", str(b_dir), "--checkpoint", str(ck)])
    monkeypatch.undo()
    assert m.main(common + ["--output", str(b_dir), "--checkpoint", str(ck)]) == 0
    assert "[RESUME]" in capsys.readouterr().out
    assert not ck.exists()                          # removed once the artifact exists
    a = np.load(a_dir / "tiny_flux__lasd.npz")
    b = np.load(b_dir / "tiny_flux__lasd.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    assert load_artifact(b_dir / "tiny_flux__lasd.npz").nt == 4


def test_setup_builds_in_float32_on_a_non_binary_refined_grid(tiny_registry, tmp_path):
    assert _driver().main(["--case", "tiny_flux", "--f32", "--refine", "3",
                           "--frames", "2", "--smoke", "--hours", "0.002",
                           "--output", str(tmp_path)]) == 0


def test_fingerprint_covers_resolved_config_and_inputs():
    """A core-config default or a resolved input that changes under the same CLI
    options changes the fingerprint (a stale checkpoint is then refused)."""
    import argparse
    m = _driver()
    args = argparse.Namespace(**{k: None for k in (
        "pr_sgs", "theta0", "gamma", "zi0", "q0", "Ug", "lat", "z0", "nu_floor",
        "refine", "f32", "smoke", "label")})
    cfg = m.sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0)

    def fp(c, x):
        return m._fingerprint(_case(), args, "lasd", 1.0, 10, [5, 10], 0.1, c, (x,))
    base = fp(cfg, np.zeros(3))
    assert fp(cfg, np.zeros(3)) == base
    assert fp(cfg._replace(c_s=cfg.c_s + 0.01), np.zeros(3)) != base
    assert fp(cfg, np.array([0.0, 0.0, 1e-12])) != base


def test_fingerprint_covers_the_source(tmp_path, monkeypatch):
    import argparse
    m = _driver()
    args = argparse.Namespace(**{k: None for k in (
        "pr_sgs", "theta0", "gamma", "zi0", "q0", "Ug", "lat", "z0", "nu_floor",
        "refine", "f32", "smoke", "label")})
    cfg = m.sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0)
    (tmp_path / "pkg").mkdir()
    mod = tmp_path / "pkg" / "physics.py"
    monkeypatch.setattr(m.legoesm, "__path__", [str(tmp_path)])

    def fp():
        return m._fingerprint(_case(), args, "lasd", 1.0, 10, [10], 0.1, cfg, ())
    mod.write_text("C = 1.0\n")
    base = fp()
    assert fp() == base
    mod.write_text("C = 1.1\n")
    assert fp() != base


def test_wall_limit_needs_a_checkpoint(tiny_registry, tmp_path):
    with pytest.raises(SystemExit, match="needs --checkpoint"):
        _driver().main(["--case", "tiny_flux", "--wall-limit-s", "10",
                        "--output", str(tmp_path)])


# --- DEPHY reader for a dry LES ----------------------------------------------------
def _cached(name):
    from legoesm.atmosphere.forcing.scm.dephy_scm import resolve_dephy_case_path
    return Path(resolve_dephy_case_path(name)).is_file()


@pytest.mark.skipif(not _cached("ayotte_05sc"), reason="ayotte_05sc not cached")
def test_dephy_les_forcing_ayotte_flux_case():
    from legoesm.atmosphere.forcing.scm.dephy_scm import load_dephy_les_forcing
    from legoesm.atmosphere.physics._shared import compute_rho, exner_function
    d = load_dephy_les_forcing("ayotte_05sc")
    assert (d.lat_deg, d.z0_m, d.z0h_m) == (45.0, pytest.approx(0.16), None)
    assert d.theta_s is None and d.duration_s == 7 * 3600.0
    np.testing.assert_allclose(d.ug_m_s, 15.0)
    # H / (ρ_s c_pd Π_s) with ρ_s from the file's surface T and p
    rho = float(compute_rho(np.float64(300.5), np.float64(1.0e5), np.float64(0.0)))
    pi_s = float(exner_function(np.float64(1.0e5)))
    np.testing.assert_allclose(d.w_theta_s, 56.27 / (rho * constants.c_pd * pi_s),
                               rtol=1e-6)


@pytest.mark.skipif(not _cached("gabls4_stage3"), reason="gabls4_stage3 not cached")
def test_dephy_les_forcing_gabls4_surface_theta_series():
    from legoesm.atmosphere.forcing.scm.dephy_scm import load_dephy_les_forcing
    d = load_dephy_les_forcing("gabls4_stage3")
    assert d.w_theta_s is None and d.theta_s.shape == d.time_s.shape == (37,)
    assert (d.lat_deg, d.z0h_m) == (pytest.approx(-75.1), pytest.approx(1e-4))
    np.testing.assert_allclose(d.theta_s[0], 273.0107, atol=1e-3)   # thetas_forc
    assert d.theta_K[0] == pytest.approx(271.3)


def test_dephy_les_forcing_refuses_what_a_dry_les_cannot_run(tmp_path, monkeypatch):
    """Advection / moisture in the file is a hard error, never silently dropped."""
    xr = pytest.importorskip("xarray")
    from legoesm.atmosphere.forcing.scm import dephy_scm
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(tmp_path))
    dest = Path(dephy_scm.resolve_dephy_case_path("ayotte_05sc"))
    dest.parent.mkdir(parents=True)
    ds = xr.Dataset({"qv": (("t0", "lev"), np.full((1, 3), 1e-3))},
                    attrs={"adv_theta": 1, "radiation": "off"})
    ds.to_netcdf(dest)
    with pytest.raises(ValueError, match="adv_theta.*moist"):
        dephy_scm.load_dephy_les_forcing("ayotte_05sc", verify=False)


_CLEAN_ATTRS = {"radiation": "off", "forc_geo": 1, "surface_forcing_wind": "z0",
                "surface_forcing_moisture": "none"}


@pytest.mark.parametrize("attrs, data, token", [
    ({"surface_forcing_moisture": "surface_flux"}, {"hfls": 5.0}, "surface_forcing_moisture"),
    ({"surface_forcing_moisture": "surface_flux"}, {}, "surface_forcing_moisture"),
    ({"surface_forcing_moisture": "beta"}, {"beta": 0.3}, "surface_forcing_moisture"),
    ({"surface_forcing_moisture": "surface_flux"}, {"hfls": np.nan}, "surface_forcing_moisture"),
    ({"surface_forcing_moisture": "beta"}, {"beta": np.nan}, "surface_forcing_moisture"),
    ({}, {"qv": np.nan}, "qv"),
    ({}, {"rl": 1e-4}, "rl"),
    ({}, {"ri": np.nan}, "ri"),
    ({"forc_tau": 1}, {}, "forc_tau"),            # a switch the loader does not know
    ({"forc_p": 1}, {}, "forc_p"),                # pressure-level forcing
    ({"forc_geo": "yes"}, {}, "forc_geo"),
    ({"adv_ua": "yes"}, {}, "adv_ua"),
    ({"surface_forcing_temp": "ts"}, {"hfss": 20.0}, "hfss"),
    ({"surface_forcing_temp": "surface_flux"}, {"ts_forc": 280.0}, "ts_forc"),
    ({"surface_forcing_moisture": "rh"}, {}, "surface_forcing_moisture"),
    ({"surface_forcing_wind": "ustar"}, {}, "surface_forcing_wind"),
    ({"forc_geo": 0}, {}, "forc_geo"),
    ({}, {}, None),                       # control: a clean file passes this gate
])
def test_dephy_les_forcing_refuses_each_unsupported_channel(tmp_path, monkeypatch,
                                                            attrs, data, token):
    """Each unsupported surface/geostrophic channel alone is refused; the same
    moisture channel at zero (Ayotte hfls, GABLS4 beta) passes the gate."""
    xr = pytest.importorskip("xarray")
    from legoesm.atmosphere.forcing.scm import dephy_scm
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(tmp_path))
    dest = Path(dephy_scm.resolve_dephy_case_path("ayotte_05sc"))
    dest.parent.mkdir(parents=True)

    def gate_error(values):
        xr.Dataset({k: (("time",), np.full(2, v)) for k, v in values.items()},
                   attrs={**_CLEAN_ATTRS, **attrs}).to_netcdf(dest)
        with pytest.raises(Exception) as exc:
            dephy_scm.load_dephy_les_forcing("ayotte_05sc", verify=False)
        return str(exc.value)

    msg = gate_error(data)
    if token is None:
        assert "cannot run" not in msg
    else:
        assert "cannot run" in msg and token in msg
    if data:
        assert "cannot run" not in gate_error({k: 0.0 for k in data})


@pytest.mark.parametrize("series, match", [
    ([[1.0, 2.0], [1.0, 2.0], [1.0, np.nan]], "missing"),   # NaN only at a later time
    ([[1.0, 2.0], [1.0, 2.5]], "varies in time"),
    ([], "missing"),
])
def test_dephy_constant_series_check_sees_every_time(series, match):
    from legoesm.atmosphere.forcing.scm.dephy_scm import _require_time_constant
    with pytest.raises(ValueError, match=match):
        _require_time_constant("x", "ug", np.asarray(series))
    _require_time_constant("x", "ug", np.asarray([[1.0, 2.0], [1.0, 2.0]]))


@pytest.mark.parametrize("coord", [[0.0, 7200.0, 3600.0], [0.0, 0.0, 10.0], [0.0, np.nan],
                                   []])
def test_dephy_coordinates_must_increase(coord):
    from legoesm.atmosphere.forcing.scm.dephy_scm import _require_increasing
    with pytest.raises(ValueError, match="strictly"):
        _require_increasing("x", "time", np.asarray(coord))
    _require_increasing("x", "time", np.asarray([0.0, 3600.0, 7200.0]))
