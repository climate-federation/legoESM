"""Prognostic sea-ice skin temperature (Semtner 1976 zero-layer + inertia).

Targets the +8.6 K polar tas warm bias of the first ClimateEval scorecard:
the prescribed-SST anchor pinned ice-covered cells at the constant T_ice
(271.35 K) year-round; the skin lets polar-night ice surfaces cool to the
conductive equilibrium ``T_s = T_f + F_net_down * h / k_i``.

Covers: the pure helper (equilibrium, sign, surface-melt cap vs basal
freezing, floor, open-water snap, conduction-implicit stability, per-step
restart-split invariance, autodiff), validate_strict corners, the CLI
round-trip, and an end-to-end subprocess run exercising the real _run_column
per-step advance + per-step T_sfc re-anchor + checkpoint save/load/restart.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.forcing.surface_utils import (
    _ICE_SKIN_FLOOR_K,
    prognostic_ice_skin_temperature,
)

T_F = constants.T_freeze_ocean
DAY_S = 86400.0
H = 2.0
G_COND = constants.k_ice_default / H  # ~1.02 W/m^2/K


def _step_n(T0, f_net, sic, n, dt=DAY_S, h=H):
    T = T0
    for _ in range(n):
        T = prognostic_ice_skin_temperature(T, f_net, sic, dt, h)
    return T


class TestHelper:
    def test_zero_flux_equilibrium_is_freezing_point(self):
        T = _step_n(jnp.full(4, T_F), jnp.zeros(4), jnp.ones(4), 50)
        np.testing.assert_allclose(np.asarray(T), T_F, atol=1e-9)

    def test_winter_loss_converges_to_zero_layer_equilibrium(self):
        """Steady net loss F: T_s -> T_f + F*h/k (the Semtner zero-layer
        balance). F = -25 W/m^2, h=2 m -> ~24.5 K below freezing — the
        observed central-Arctic winter skin range."""
        f = jnp.full(3, -25.0)
        T = _step_n(jnp.full(3, T_F), f, jnp.ones(3), 400)
        expect = T_F - 25.0 / G_COND
        np.testing.assert_allclose(np.asarray(T), expect, rtol=1e-6)
        assert 240.0 < float(T[0]) < 255.0  # physically sane winter skin

    def test_melt_cap_at_surface_melting_point(self):
        """Summer net gain caps the skin at the FRESH-ICE SURFACE melting
        point (0 C = constants.T_freeze), NOT the basal seawater freezing
        point T_f (271.35 K).  Distinct boundary conditions: conduction is
        toward the 271.35 K base, the surface melts at 0 C (codex-1 finding
        4; mirrors ice/sea_ice.py T_base vs T_melt_surface)."""
        T = _step_n(jnp.full(2, T_F - 5.0), jnp.full(2, 80.0),
                    jnp.ones(2), 200)
        np.testing.assert_allclose(np.asarray(T), constants.T_freeze,
                                   atol=1e-9)
        assert float(T[0]) > T_F  # cap is ABOVE seawater freezing

    def test_equilibrates_in_the_melt_band_above_seawater_freezing(self):
        """A weak steady gain settles BETWEEN the seawater-freezing base and
        the 0 C surface cap — the 1.8 K band the old cap-at-271.35 wrongly
        discarded.  F = +1 W/m^2 -> T_f + 1/g ~ +0.98 K (codex-1 finding 4)."""
        T = _step_n(jnp.full(1, T_F), jnp.full(1, 1.0), jnp.ones(1), 2000)
        eq = T_F + 1.0 / G_COND
        np.testing.assert_allclose(np.asarray(T), eq, rtol=1e-6)
        assert T_F < float(T[0]) < constants.T_freeze

    def test_restart_split_invariance(self):
        """The per-step driver advance makes the skin a PURE carry: advancing
        [0..k] then [k..N] equals advancing [0..N] for any split k.  This is
        the property that makes checkpointing only ice_T_skin sufficient for
        exact restart continuity — no pending daily advance to drop or
        double-count (codex-1 finding 2, per-step redesign)."""
        f = jnp.full(2, -18.0)
        sic = jnp.ones(2)
        dt = 240.0  # a model step, not a day
        for k in (1, 37, 200):
            straight = _step_n(jnp.full(2, T_F), f, sic, 300, dt=dt)
            link1 = _step_n(jnp.full(2, T_F), f, sic, k, dt=dt)
            restarted = _step_n(link1, f, sic, 300 - k, dt=dt)
            np.testing.assert_array_equal(np.asarray(restarted),
                                          np.asarray(straight))

    def test_floor_guard(self):
        T = _step_n(jnp.full(2, 200.0), jnp.full(2, -500.0),
                    jnp.ones(2), 100)
        assert float(jnp.min(T)) >= _ICE_SKIN_FLOOR_K

    def test_open_water_snaps_to_freezing_point(self):
        T = prognostic_ice_skin_temperature(
            jnp.full(3, 240.0), jnp.full(3, -50.0),
            jnp.asarray([0.0, 0.5, 1.0]), DAY_S, H)
        assert float(T[0]) == pytest.approx(T_F)     # open water
        assert float(T[1]) < T_F                     # partial ice evolves
        assert float(T[2]) < T_F

    def test_unconditionally_stable_large_dt(self):
        """Conduction-implicit: even a 30-day step lands between the start
        and the equilibrium (no overshoot/oscillation)."""
        f = jnp.full(1, -25.0)
        T1 = prognostic_ice_skin_temperature(
            jnp.full(1, T_F), f, jnp.ones(1), 30 * DAY_S, H)
        eq = T_F - 25.0 / G_COND
        assert eq - 1e-9 <= float(T1[0]) <= T_F

    def test_relaxation_timescale_weeks(self):
        """One daily step moves the analytic implicit fraction
        r*g/(1+r*g) with r = dt/C toward equilibrium (~4.4% at h=2 m —
        the slab's ~3-week PURE-CONDUCTION inertia; the coupled system
        equilibrates faster once the atmospheric fluxes respond to the
        cooling skin). Guards both the C and g_cond wiring."""
        f = jnp.full(1, -25.0)
        T1 = _step_n(jnp.full(1, T_F), f, jnp.ones(1), 1)
        eq = T_F - 25.0 / G_COND
        frac = (T_F - float(T1[0])) / (T_F - eq)
        r = DAY_S / (0.5 * constants.rho_ice * constants.c_pi * H)
        expect = r * G_COND / (1.0 + r * G_COND)
        assert frac == pytest.approx(expect, rel=1e-9)
        assert 0.02 < frac < 0.5  # inertial, not instant

    def test_thinner_ice_cools_less(self):
        """Thinner slab -> larger conductance -> warmer winter skin (more
        ocean heat reaches the surface)."""
        f = jnp.full(1, -25.0)
        T_thin = _step_n(jnp.full(1, T_F), f, jnp.ones(1), 400, h=0.5)
        T_thick = _step_n(jnp.full(1, T_F), f, jnp.ones(1), 400, h=3.0)
        assert float(T_thin[0]) > float(T_thick[0])

    def test_autodiff_flows(self):
        def loss(f):
            T = prognostic_ice_skin_temperature(
                jnp.full(3, 250.0), f, jnp.ones(3), DAY_S, H)
            return jnp.sum(T ** 2)

        g = jax.grad(loss)(jnp.full(3, -20.0))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0


def test_blend_flattens_2d_source_against_array_skin():
    """A (nCells,1)-shaped SST/SIC source must blend ELEMENTWISE against the
    (nCells,) prognostic skin, not broadcast to (nCells,nCells).  The driver's
    _blend_T_sfc flattens the source first; without it a 2-D forcing source
    (which the old scalar-T_ice blend tolerated) would blow up the anchor shape
    and be rejected (codex-4)."""
    from legoesm.forcing.surface_utils import blend_surface_temperature
    n = 5
    sst = jnp.full((n, 1), 290.0)
    sic = jnp.full((n, 1), 0.5)
    skin = jnp.linspace(240.0, 250.0, n)          # (n,) array ice component
    out = blend_surface_temperature(sst.reshape(-1), sic.reshape(-1), skin)
    assert out.shape == (n,)
    np.testing.assert_allclose(np.asarray(out),
                               0.5 * np.asarray(skin) + 0.5 * 290.0)
    # Documents WHY the flatten is required: the un-flattened blend is (n,n).
    assert blend_surface_temperature(sst, sic, skin).shape == (n, n)


# ---------------------------------------------------------------------------
# validate_strict corners (mirrors the sibling MPAS-knob tests)
# ---------------------------------------------------------------------------

def _mpas_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    kw.setdefault("radiation", "gray")
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        **kw,
    )


def _cdgrid_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


def test_validate_mpas_accepts_ice_skin():
    _mpas_cfg(ice_skin_prognostic=True,
              ice_thickness_m=1.5).validate_strict()


def test_validate_default_off():
    _mpas_cfg().validate_strict()


def test_validate_cdgrid_accepts_ice_skin():
    # The compiled / per-step / operator-split lanes carry the skin (#2106).
    _cdgrid_cfg(ice_skin_prognostic=True).validate_strict()


def test_validate_spectral_refuses_ice_skin():
    from legoesm.driver.config import DycoreConfig
    with pytest.raises(ValueError, match="discretization=.spectral. lane"):
        _cdgrid_cfg(ice_skin_prognostic=True)._replace(
            dycore=DycoreConfig(dt=600.0, discretization="spectral")
        ).validate_strict()


def test_validate_refuses_radiation_none():
    with pytest.raises(ValueError, match="radiation"):
        _mpas_cfg(ice_skin_prognostic=True,
                  radiation="none", turbulence="none").validate_strict()


def test_validate_refuses_inert_thickness():
    with pytest.raises(ValueError, match="ice_thickness_m"):
        _mpas_cfg(ice_skin_prognostic=False, ice_thickness_m=1.0).validate_strict()


@pytest.mark.parametrize("bad", [0.05, 11.0, float("nan")])
def test_validate_thickness_bounds(bad):
    with pytest.raises(ValueError, match="ice_thickness_m"):
        _mpas_cfg(ice_skin_prognostic=True,
                  ice_thickness_m=bad).validate_strict()


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.ice_skin_prognostic is None   # auto
    assert cfg_default.ice_thickness_m == 2.0
    off = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--no-ice-skin-prognostic"]), parser))
    assert off.ice_skin_prognostic is False

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-ice-skin-prognostic", "--mpas-ice-thickness-m", "1.5",
    ]), parser))
    assert cfg.ice_skin_prognostic is True
    assert cfg.ice_thickness_m == 1.5


# ---------------------------------------------------------------------------
# End-to-end wiring (subprocess): exercises the FULL per-step advance +
# per-step T_sfc re-anchor + checkpoint save/load path in the real _run_column
# loop — the parts the pure-helper tests above cannot reach.  ~40 s.
#
# Analytical SST/SIC is ice-FREE (sic=0 everywhere, SST > freezing), so this
# asserts the correct OPEN-WATER behaviour (skin snaps to T_freeze_ocean) and
# a clean checkpoint round-trip, NOT the cooling magnitude — the conductive
# cooling physics is covered by the helper equilibrium tests above.  A restart
# from the day-1 checkpoint exercises the load->stage->adopt path and must
# reproduce the straight run's skin (restart continuity of the pure carry).
# ---------------------------------------------------------------------------

def _run_amip_mpas_skin(out_dir, extra=(), days=2):
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(repo / "scripts" / "run" / "run_amip.py"),
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--resolution", "3", "--nlev", "8",
        "--days", str(days), "--checkpoint-days", "1",
        "--radiation", "gray", "--mpas-ice-skin-prognostic",
        "--output", str(out_dir), *extra,
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=600)
    combined = r.stdout + r.stderr
    assert r.returncode == 0, f"run failed ({r.returncode}):\n{combined[-2000:]}"
    assert "Traceback" not in combined, f"traceback:\n{combined[-2000:]}"
    assert "nan" not in r.stdout.lower(), f"NaN:\n{r.stdout[-2000:]}"
    return combined


def test_end_to_end_mpas_ice_skin_wiring_and_checkpoint(tmp_path):
    straight = tmp_path / "straight"
    log = _run_amip_mpas_skin(straight, days=2)
    assert "ice skin ON" in log, "per-step ice-skin setup did not run"

    ck2 = straight / "checkpoint_day_0002.npz"
    d = np.load(ck2, allow_pickle=True)
    assert "ice_T_skin" in d.files, "checkpoint did not persist ice_T_skin"
    skin = np.asarray(d["ice_T_skin"])
    assert skin.shape == (642,)                       # nCells at res 3
    assert np.all(np.isfinite(skin)), "non-finite skin in checkpoint"
    # Valid physical band [floor, surface melt]; ice-free analytical data ->
    # the open-water snap pins every cell at the seawater freezing point.
    assert np.all(skin >= _ICE_SKIN_FLOOR_K)
    assert np.all(skin <= constants.T_freeze + 1e-6)
    np.testing.assert_allclose(skin, constants.T_freeze_ocean, atol=1e-6)

    # Restart from the day-1 checkpoint: exercises load->stage->adopt and must
    # reproduce the straight run's day-2 skin (pure-carry restart continuity).
    resumed = tmp_path / "resumed"
    _run_amip_mpas_skin(
        resumed, days=2,
        extra=("--restart-from", str(straight / "checkpoint_day_0001.npz"),
               "--restart-start-day", "1"))
    dr = np.load(resumed / "checkpoint_day_0002.npz", allow_pickle=True)
    assert "ice_T_skin" in dr.files
    np.testing.assert_allclose(np.asarray(dr["ice_T_skin"]), skin, atol=1e-6)


def test_restart_refuses_nonfinite_checkpoint_skin(tmp_path):
    """A corrupt (NaN) checkpoint skin must be REFUSED at resume, not adopted:
    a NaN poisons even open-water (sic=0) anchors via 0*NaN in the blend
    (codex-3).  Verifies the finiteness guard in the _run_column seed overlay."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    base = tmp_path / "base"
    _run_amip_mpas_skin(base, days=1)
    good = base / "checkpoint_day_0001.npz"
    d = dict(np.load(good, allow_pickle=True))
    skin = np.asarray(d["ice_T_skin"]).copy()
    skin[0] = np.nan
    d["ice_T_skin"] = skin
    bad = tmp_path / "checkpoint_day_0001.npz"
    np.savez(bad, **d)

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run" / "run_amip.py"),
         "--dataset", "analytical", "--grid-type", "voronoi",
         "--discretization", "mpas", "--resolution", "3", "--nlev", "8",
         "--days", "2", "--radiation", "gray", "--mpas-ice-skin-prognostic",
         "--restart-from", str(bad), "--restart-start-day", "1",
         "--output", str(tmp_path / "corrupt_out")],
        env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0, "restart from a NaN skin should have failed"
    assert "non-finite" in (r.stdout + r.stderr), \
        f"expected a non-finite skin refusal:\n{(r.stdout + r.stderr)[-2000:]}"


# ---------------------------------------------------------------------------
# Melt-dependent sea-ice albedo (#2106): sea_ice_albedo_scheme
# ---------------------------------------------------------------------------

def test_melt_albedo_ramp_values_with_production_cold_end():
    """The lane passes albedo_cold_bare=albedo_ice (0.80); at the 2 m slab the
    thickness factor is 1, so cold skin -> 0.80, 0 C skin -> 0.50, and the
    smoothstep midpoint is the linear midpoint 0.65."""
    from legoesm.ice.shortwave import maykut_untersteiner_albedo
    T = jnp.array([constants.T_freeze - 5.0, constants.T_freeze - 1.0,
                   constants.T_freeze - 0.5, constants.T_freeze])
    a = maykut_untersteiner_albedo(T, 2.0, albedo_cold_bare=0.80, T_width=1.0)
    np.testing.assert_allclose(np.asarray(a), [0.80, 0.80, 0.65, 0.50],
                               atol=1e-12)


# KNOWN DIFFERENCE vs the old linear ramp, measured per case at T_width=1 K,
# h=2 m (r = 1 + (T - T_freeze)/T_width): smoothstep minus linear albedo.
_MU_R = np.array([0.1, 0.25, 0.5 - 0.5 / np.sqrt(3), 0.75,
                  0.5 + 0.5 / np.sqrt(3), 0.9])
_MU_GAPS = {
    0.80: [0.0215999999999964, 0.028124999999999956, 0.028867513459485128,
           -0.028124999999999956, -0.028867513459485128, -0.0215999999999964],
    0.70: [0.014399999999997526, 0.018750000000000044, 0.019245008972990085,
           -0.018750000000000044, -0.019245008972989974, -0.014399999999997526],
}


@pytest.mark.parametrize("cold", sorted(_MU_GAPS))
def test_melt_albedo_smoothstep_gap_vs_linear_per_case(cold):
    from legoesm.ice.shortwave import maykut_untersteiner_albedo
    T = jnp.asarray(constants.T_freeze + (_MU_R - 1.0))
    a = np.asarray(maykut_untersteiner_albedo(
        T, 2.0, albedo_cold_bare=cold, T_width=1.0))
    linear = cold - (cold - 0.5) * _MU_R
    np.testing.assert_allclose(a - linear, _MU_GAPS[cold],
                               rtol=1e-6, atol=1e-12)


def test_melt_albedo_smoothstep_grad_at_default():
    from jax.test_util import check_grads
    from legoesm.ice.shortwave import maykut_untersteiner_albedo
    T = jnp.asarray(constants.T_freeze + np.array([-0.95, -0.5, -0.05]))

    def f(t):
        return maykut_untersteiner_albedo(t, 2.0, albedo_cold_bare=0.80)

    check_grads(f, (T,), order=2, modes=("fwd", "rev"))
    g = np.asarray(jax.grad(lambda t: f(t).sum())(T))
    assert np.all(np.isfinite(g)) and np.all(g < 0.0)


def test_validate_melt_albedo_accepts_with_skin():
    _mpas_cfg(ice_skin_prognostic=True,
              sea_ice_albedo_scheme="maykut_untersteiner",
              sea_ice_melt_ramp_width_K=2.0).validate_strict()


def test_validate_melt_albedo_refuses_unknown_scheme():
    with pytest.raises(ValueError, match="sea_ice_albedo_scheme"):
        _mpas_cfg(ice_skin_prognostic=True,
                  sea_ice_albedo_scheme="ccsm3").validate_strict()


def test_validate_melt_albedo_needs_skin():
    with pytest.raises(ValueError, match="ice_skin_prognostic=True"):
        _mpas_cfg(ice_skin_prognostic=False,
                  sea_ice_albedo_scheme="maykut_untersteiner"
                  ).validate_strict()


def test_validate_melt_albedo_refuses_inert_width():
    # auto albedo with the skin off resolves to constant: the width is inert.
    with pytest.raises(ValueError, match="sea_ice_melt_ramp_width_K"):
        _mpas_cfg(ice_skin_prognostic=False,
                  sea_ice_melt_ramp_width_K=2.0).validate_strict()
    with pytest.raises(ValueError, match="sea_ice_melt_ramp_width_K"):
        _mpas_cfg(ice_skin_prognostic=True, sea_ice_albedo_scheme="constant",
                  sea_ice_melt_ramp_width_K=2.0).validate_strict()


def test_auto_resolves_per_ice_temperature():
    from legoesm.driver.config import resolve_sea_ice_albedo_scheme
    assert _mpas_cfg().sea_ice_albedo_scheme == "auto"
    assert resolve_sea_ice_albedo_scheme(_mpas_cfg()) == "constant"
    assert resolve_sea_ice_albedo_scheme(
        _mpas_cfg(ice_skin_prognostic=True)) == "maykut_untersteiner"
    assert resolve_sea_ice_albedo_scheme(_mpas_cfg(
        ice_skin_prognostic=True, sea_ice_albedo_scheme="constant")) \
        == "constant"
    # the default without a temperature must not raise
    _mpas_cfg().validate_strict()
    _cdgrid_cfg().validate_strict()


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_validate_cap_width_bounds(bad):
    with pytest.raises(ValueError, match="ice_skin_melt_cap_width_K"):
        _mpas_cfg(ice_skin_prognostic=True,
                  ice_skin_melt_cap_width_K=bad).validate_strict()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_validate_melt_albedo_width_bounds(bad):
    with pytest.raises(ValueError, match="sea_ice_melt_ramp_width_K"):
        _mpas_cfg(ice_skin_prognostic=True,
                  sea_ice_albedo_scheme="maykut_untersteiner",
                  sea_ice_melt_ramp_width_K=bad).validate_strict()


def test_validate_cdgrid_refuses_melt_albedo():
    with pytest.raises(ValueError, match="ice_skin_prognostic=True"):
        _cdgrid_cfg(ice_skin_prognostic=False,
                    sea_ice_albedo_scheme="maykut_untersteiner"
                    ).validate_strict()


def test_cli_round_trip_melt_albedo():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.sea_ice_albedo_scheme == "auto"
    assert cfg_default.ice_skin_melt_cap_width_K == 0.01
    assert cfg_default.sea_ice_melt_ramp_width_K == 1.0
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-ice-skin-prognostic",
        "--sea-ice-albedo-scheme", "maykut_untersteiner",
        "--sea-ice-melt-ramp-width-K", "2.5",
        "--ice-skin-melt-cap-width-K", "0.0",
    ]), parser))
    assert cfg.ice_skin_melt_cap_width_K == 0.0
    assert cfg.sea_ice_albedo_scheme == "maykut_untersteiner"
    assert cfg.sea_ice_melt_ramp_width_K == 2.5


_ICY_RUNNER = r'''
import sys, runpy
import jax.numpy as jnp
import numpy as np
from legoesm import constants
import legoesm.forcing.analytical as an
_orig = an.analytical_sst_sic
def _icy(lat_deg, day, T_ice=constants.T_freeze_ocean):
    sst, sic = _orig(lat_deg, day, T_ice=T_ice)
    polar = np.abs(np.asarray(lat_deg)) >= 60.0
    return (jnp.where(polar, T_ice, sst).astype(sst.dtype),
            jnp.where(polar, 1.0, sic).astype(sic.dtype))
an.analytical_sst_sic = _icy
import os
if os.environ.get("LEGOESM_TEST_COUNT_MU"):
    import legoesm.ice.shortwave as sw
    _mu = sw.maykut_untersteiner_albedo
    def _counted(T_sfc, h_ice, **kw):
        print("MU_CALL", float(np.max(np.asarray(T_sfc))), flush=True)
        return _mu(T_sfc, h_ice, **kw)
    sw.maykut_untersteiner_albedo = _counted
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


_MPAS_GRID = ("--grid-type", "voronoi", "--discretization", "mpas",
              "--resolution", "3")


def _run_icy(out_dir, extra=(), days=1, start_day=172.0, env_extra=None,
             grid=_MPAS_GRID, skin_flag=("--mpas-ice-skin-prognostic",)):
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.update(env_extra or {})
    cmd = [sys.executable, "-c", _ICY_RUNNER,
           str(repo / "scripts" / "run" / "run_amip.py"),
           "--dataset", "analytical",
           *grid, "--nlev", "8",
           "--days", str(days), "--checkpoint-days", "1",
           "--start-day", str(start_day),
           "--radiation", "gray", *skin_flag,
           "--output", str(out_dir), *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=900)
    combined = r.stdout + r.stderr
    (Path(out_dir) / "run.log").write_text(combined)
    assert r.returncode == 0, f"run failed ({r.returncode}):\n{combined[-2000:]}"
    assert "Traceback" not in combined, f"traceback:\n{combined[-2000:]}"
    return Path(out_dir)


def _last_ckpt(d):
    return np.load(sorted(d.glob("checkpoint_day_*.npz"))[-1],
                   allow_pickle=True)


_MELT = ("--sea-ice-albedo-scheme", "maykut_untersteiner",
         "--sea-ice-melt-ramp-width-K", "5.0")


def test_end_to_end_melt_albedo_reaches_radiation(tmp_path):
    """Polar ice (sic=1 poleward of 60 deg), June start: with a 5 K ramp the
    seed skin (271.35 K) already sits inside the ramp, so the melt scheme
    darkens the ice from step one.  The state must then differ from the
    constant-albedo run — fails if the scheme is wired in but inert."""
    # Both arms at the hard melt cap: with the smooth cap a sub-cap skin can
    # end ~1e-6 K cooler under darker ice through the atmospheric response.
    hard = ("--ice-skin-melt-cap-width-K", "0.0")
    base = _last_ckpt(_run_icy(tmp_path / "const", extra=(
        "--sea-ice-albedo-scheme", "constant") + hard))
    melt = _last_ckpt(_run_icy(tmp_path / "melt", extra=_MELT + hard))
    sb, sm = np.asarray(base["ice_T_skin"]), np.asarray(melt["ice_T_skin"])
    assert np.all(np.isfinite(sm))
    # Darker ice absorbs more sunlight: the skin can only be warmer or equal.
    assert np.all(sm >= sb - 1e-9)
    dT = np.abs(np.asarray(melt["T"]) - np.asarray(base["T"]))
    assert float(dT.max()) > 1e-4, "melt albedo never reached radiation"


def test_melt_albedo_refreshes_every_step_from_current_skin(tmp_path):
    """The melt albedo must follow the CURRENT skin every model step (same
    cadence as the T_sfc re-anchor), not a day-start snapshot: count the ramp
    evaluations in a 1-day run (144 steps) and check they see a warming skin.
    A once-daily albedo would evaluate it once."""
    out = _run_icy(tmp_path / "melt", extra=_MELT
                   + ("--mpas-ice-thickness-m", "10.0"),
                   env_extra={"LEGOESM_TEST_COUNT_MU": "1"})
    calls = [ln.split()[1:] for ln in (out / "run.log").read_text().splitlines()
             if ln.startswith("MU_CALL ")]
    assert len(calls) >= 144, f"only {len(calls)} albedo evaluations"
    skin_max = [float(c[0]) for c in calls]
    assert skin_max[-1] > skin_max[0] + 1e-3, "albedo never saw the skin warm"


def test_committed_deck_selects_melt_albedo():
    from pathlib import Path
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    deck = (Path(__file__).resolve().parents[2] / "config" / "amip"
            / "amip_production_seaice_melt_albedo.yaml")
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(deck, parser))  # run_amip.main's load
    args = parser.parse_args([])
    assert args.sea_ice_albedo_scheme == "maykut_untersteiner"
    assert args.sea_ice_melt_ramp_width_K == 1.0
    assert args.ice_skin_prognostic is True
    assert args.albedo_ice == 0.80


def test_production_deck_selects_melt_albedo_explicitly():
    from pathlib import Path
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    deck = (Path(__file__).resolve().parents[2] / "config" / "amip"
            / "amip_production.yaml")
    keys = load_yaml_config(deck, build_arg_parser())
    assert keys["sea_ice_albedo_scheme"] == "maykut_untersteiner"
    assert keys["sea_ice_melt_ramp_width_K"] == 1.0
    assert keys["ice_skin_melt_cap_width_K"] == 0.01


def test_auto_is_resolved_into_the_recorded_config(tmp_path):
    """Default "auto" with the skin on: the run records the RESOLVED scheme
    (experiment_config.json) and logs it — never the word "auto"."""
    import json
    out = _run_icy(tmp_path / "auto")
    cfg = json.loads((out / "experiment_config.json").read_text())
    cfg = cfg.get("config", cfg)
    assert cfg["sea_ice_albedo_scheme"] == "maykut_untersteiner"
    assert "Sea-ice albedo scheme: maykut_untersteiner" in (
        out / "run.log").read_text()
    assert "maykut_untersteiner" in (out / "run_manifest.json").read_text()


@pytest.mark.parametrize("grid,extra", [
    (("--grid-type", "latlon", "--discretization", "centered",
      "--resolution", "8"), ()),
    (("--grid-type", "cubed_sphere", "--discretization", "centered",
      "--resolution", "8"), ()),
    (("--grid-type", "latlon", "--discretization", "centered",
      "--resolution", "8"), ("--per-step-rollout",)),
], ids=["latlon_compiled", "cube_compiled", "latlon_per_step"])
def test_structured_lane_skin_evolves_and_auto_resolves(tmp_path, grid, extra):
    """Every structured lane carries the shared skin with NO skin flag: the
    default 'auto' turns the skin on and the albedo to the melt ramp, both
    recorded, and the checkpointed skin has moved off its 271.35 K seed under
    June sun on the sic=1 polar caps."""
    import json
    out = _run_icy(tmp_path / "run", extra=extra, grid=grid, skin_flag=())
    log = (out / "run.log").read_text()
    assert ("Sea-ice albedo scheme: maykut_untersteiner (ice skin ON: the "
            "lane carries the skin)") in log
    rec = json.loads((out / "experiment_config.json").read_text())
    rec = rec.get("config", rec)
    assert rec["ice_skin_prognostic"] is True
    ck = _last_ckpt(out)
    keys = [k for k in ck.files if k.endswith("ice_T_skin")]
    assert keys, f"no ice skin in checkpoint: {ck.files}"
    skin = np.asarray(ck[keys[0]])
    assert np.all(np.isfinite(skin))
    assert np.all(skin <= constants.T_freeze + 1e-9)
    assert float(np.max(np.abs(skin - constants.T_freeze_ocean))) > 1e-3


def _skin_step(skin_on, ice_T_skin):
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.unit.test_amip_snow_albedo import NLEV, N_CS, _sigma
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, turbulence="louis",
                           surface_bulk_scheme="coare3",
                           ice_skin_prognostic=skin_on)
    pipe = build_physics_pipeline(create_cubed_sphere(N_CS), _sigma(), cfg)
    pipe.rad_update_steps = 1
    ad = pipe.adapter
    s2 = ad.shape_2d
    s3 = (*s2, NLEV)
    held3, held2 = jnp.zeros(s3), jnp.zeros(s2)
    o3 = jnp.zeros((ad.ncol, NLEV))
    return pipe.build_step_unified()(
        jnp.bool_(True),
        jnp.full(s3, 250.0), jnp.full(s2, 1.0e5),
        jnp.full(s3, 0.001), jnp.zeros(s3), jnp.zeros(s3),
        jnp.zeros((ad.ncol,)),
        jnp.full(s3, 5.0), jnp.zeros(s3),
        jnp.full(s2, constants.T_freeze_ocean), jnp.ones(s2),
        jnp.full(s2, 1.2), jnp.full(s2, 1.0),
        172.0, 43200.0, 600.0,
        jnp.array([]), constants.S_0, o3, o3,
        held3, held2, held2, held2, held2, held2,
        **({} if ice_T_skin is None else {"ice_T_skin": ice_T_skin}))


def test_step_unified_refuses_missing_skin_when_prognostic():
    """A lane whose carry was built without the skin (e.g. an IC builder
    that never seeds it) must fail, not run the constant T_ice silently."""
    with pytest.raises(ValueError, match="no ice_T_skin"):
        _skin_step(True, None)


def test_step_unified_advances_the_carried_skin():
    from tests.unit.test_amip_snow_albedo import SHAPE_2D
    assert len(_skin_step(False, None)) == 4
    seed = jnp.full(SHAPE_2D, constants.T_freeze_ocean)
    ret = _skin_step(True, seed)
    assert len(ret) == 5
    new = np.asarray(ret[4])
    assert new.shape == seed.shape and np.all(np.isfinite(new))
    assert float(np.max(np.abs(new - constants.T_freeze_ocean))) > 1e-6


def test_advance_ice_skin_flux_assembly_and_signs():
    """F_net_down = sw + lw - sh - lh (radiation +down, turbulence +up);
    missing turbulent fluxes add nothing; the result keeps T_skin's shape."""
    from legoesm.forcing.surface_utils import advance_ice_skin
    T = jnp.full((2, 3), 260.0)
    sw, lw = jnp.full((2, 3), 100.0), jnp.full((2, 3), -40.0)
    sh, lh = jnp.full((2, 3), 15.0), jnp.full((2, 3), 5.0)
    sic = jnp.ones((2, 3))
    got = advance_ice_skin(T, sw, lw, sh, lh, sic, dt_s=600.0, h_ice_m=2.0)
    ref = prognostic_ice_skin_temperature(
        T.reshape(-1), jnp.full(6, 100.0 - 40.0 - 15.0 - 5.0), sic.reshape(-1),
        600.0, 2.0)
    assert got.shape == T.shape
    np.testing.assert_allclose(np.asarray(got).reshape(-1), np.asarray(ref),
                               rtol=0.0, atol=1e-12)
    no_turb = advance_ice_skin(T, sw, lw, None, None, sic, dt_s=600.0,
                               h_ice_m=2.0)
    # Upward turbulent loss cools: removing it leaves a warmer skin.
    assert np.all(np.asarray(no_turb) > np.asarray(got))



def test_auto_skin_turns_on_where_the_lane_carries_it():
    from legoesm.driver.config import resolve_ice_skin_prognostic
    assert resolve_ice_skin_prognostic(_cdgrid_cfg()) == (
        True, "the lane carries the skin")
    assert resolve_ice_skin_prognostic(_mpas_cfg())[0] is True


def _auto_off_cases():
    from legoesm.driver.config import DycoreConfig, OutputConfig
    spectral = _cdgrid_cfg()._replace(
        dycore=DycoreConfig(dt=600.0, discretization="spectral"))
    duo = _cdgrid_cfg()._replace(
        dycore=DycoreConfig(dt=600.0, discretization="fv3_duo"))
    no_rad = _cdgrid_cfg()._replace(radiation="none")
    zarr = _cdgrid_cfg()._replace(output=OutputConfig(
        diag_days=1, checkpoint_days=1, checkpoint_format="zarr"))
    ckpt = _cdgrid_cfg()._replace(output=OutputConfig(
        diag_days=1, checkpoint_days=1))
    return [(spectral, {}, "spectral"), (duo, {}, "fv3_duo"),
            (no_rad, {}, "radiation"), (_cdgrid_cfg(), {"coupled": True},
                                        "coupled"),
            (zarr, {}, "zarr"), (ckpt, {"latlon_mpi": True}, "lat-lon MPI")]


@pytest.mark.parametrize("i", range(6))
def test_auto_skin_stays_off_with_a_reason_and_validates(i):
    from legoesm.driver.config import resolve_ice_skin_prognostic
    cfg, kw, why = _auto_off_cases()[i]
    on, reason = resolve_ice_skin_prognostic(cfg, **kw)
    assert on is False and why in reason
    # the resolved run must validate: no raise on a lane without the skin
    cfg._replace(ice_skin_prognostic=on).validate_strict()


def test_explicit_skin_on_a_coupled_driver_raises():
    from legoesm.driver.config import resolve_ice_skin_prognostic
    with pytest.raises(ValueError, match="coupled"):
        resolve_ice_skin_prognostic(_cdgrid_cfg(ice_skin_prognostic=True),
                                    coupled=True)
    assert resolve_ice_skin_prognostic(
        _cdgrid_cfg(ice_skin_prognostic=False), coupled=True) == (
            False, "explicit")


_DUO_COLUMN_AUTO = r'''
import json, sys, numpy as np
import jax.numpy as jnp
from legoesm import constants
import legoesm.forcing.analytical as an
_orig = an.analytical_sst_sic
def _icy(lat_deg, day, T_ice=constants.T_freeze_ocean):
    sst, sic = _orig(lat_deg, day, T_ice=T_ice)
    polar = np.abs(np.asarray(lat_deg)) >= 60.0
    return (jnp.where(polar, T_ice, sst).astype(sst.dtype),
            jnp.where(polar, 1.0, sic).astype(sic.dtype))
an.analytical_sst_sic = _icy
from legoesm.driver.config import (ExperimentConfig, GridConfig,
                                   DycoreConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver
out = sys.argv[1]
dt = 600.0
cfg = ExperimentConfig(
    grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
    dycore=DycoreConfig(model_type="hydrostatic", discretization="fv3_duo",
                        column_lane=True, dt=dt),
    output=OutputConfig(output_dir=out, diag_days=0, checkpoint_days=0),
    days=dt / 86400.0, dataset="analytical", precision="fp64",
    distributed=False, n_devices=1, radiation="gray", turbulence="none",
    convection="none", microphysics="none", gravity_wave_drag="none",
    cloud_scheme="none")
assert cfg.ice_skin_prognostic is None
d = ModelDriver(cfg, output_dir=out)
d.setup()
status = d.run(compiled=True)
skin = getattr(d, "_ice_T_skin", None)
print("RESULT " + json.dumps({
    "status": status, "skin_on": d.config.ice_skin_prognostic,
    "albedo": d.config.sea_ice_albedo_scheme,
    "skin_finite": None if skin is None
    else bool(np.all(np.isfinite(np.asarray(skin)))),
    "skin_moved": None if skin is None else float(np.max(np.abs(
        np.asarray(skin) - constants.T_freeze_ocean)))}))
'''


def test_fv3_duo_column_lane_auto_turns_the_skin_on_and_steps(tmp_path):
    """Auto on the FV3-duo COLUMN lane (the MPAS column loop with duo
    dynamics) resolves the skin ON, passes validation, and steps."""
    import json
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "PMI", "PMIX", "OMPI_"))}
    env.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", _DUO_COLUMN_AUTO,
                        str(tmp_path)], env=env, capture_output=True,
                       text=True, timeout=1200)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    res = json.loads(r.stdout.split("RESULT ", 1)[1].splitlines()[0])
    assert res["status"] == "COMPLETED"
    assert res["skin_on"] is True
    assert res["albedo"] == "maykut_untersteiner"
    assert res["skin_finite"] is True
    assert res["skin_moved"] > 0.0, "the skin never stepped off its seed"



def test_coupled_marker_keeps_the_auto_skin_off():
    """The marker the coupled drivers set before setup() is what auto reads:
    with it, ModelDriver resolves the skin OFF (and the albedo constant);
    without it, the same config resolves ON."""
    from legoesm.driver.model_driver import ModelDriver
    for coupled, want in ((True, False), (False, True)):
        d = ModelDriver(_cdgrid_cfg())
        if coupled:
            d._requires_surface_flux_export = True
        d._resolve_ice_skin()
        assert d.config.ice_skin_prognostic is want
        assert d._input_config.ice_skin_prognostic is want
        assert d.config.sea_ice_albedo_scheme == (
            "maykut_untersteiner" if want else "constant")
