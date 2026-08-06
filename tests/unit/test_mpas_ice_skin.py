"""Prognostic sea-ice skin temperature (Semtner 1976 zero-layer + inertia).

Targets the +8.6 K polar tas warm bias of the first ClimateEval scorecard:
the prescribed-SST anchor pinned ice-covered cells at the constant T_ice
(271.35 K) year-round; the skin lets polar-night ice surfaces cool to the
conductive equilibrium ``T_s = T_f + F_net_down * h / k_i``.

Covers: the pure helper (equilibrium, sign, surface-melt cap vs basal
freezing, floor, open-water snap, conduction-implicit stability, per-step
restart-split invariance, autodiff), validate_strict corners, the CLI
round-trip, and an end-to-end subprocess run exercising the real _run_mpas
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
    _mpas_cfg(mpas_ice_skin_prognostic=True,
              mpas_ice_thickness_m=1.5).validate_strict()


def test_validate_default_off():
    _mpas_cfg().validate_strict()


def test_validate_cdgrid_refuses_ice_skin():
    with pytest.raises(ValueError, match="MPAS-lane"):
        _cdgrid_cfg(mpas_ice_skin_prognostic=True).validate_strict()


def test_validate_refuses_radiation_none():
    with pytest.raises(ValueError, match="radiation"):
        _mpas_cfg(mpas_ice_skin_prognostic=True,
                  radiation="none", turbulence="none").validate_strict()


def test_validate_refuses_inert_thickness():
    with pytest.raises(ValueError, match="mpas_ice_thickness_m"):
        _mpas_cfg(mpas_ice_thickness_m=1.0).validate_strict()


@pytest.mark.parametrize("bad", [0.05, 11.0, float("nan")])
def test_validate_thickness_bounds(bad):
    with pytest.raises(ValueError, match="mpas_ice_thickness_m"):
        _mpas_cfg(mpas_ice_skin_prognostic=True,
                  mpas_ice_thickness_m=bad).validate_strict()


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.mpas_ice_skin_prognostic is False
    assert cfg_default.mpas_ice_thickness_m == 2.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--grid-type", "voronoi", "--discretization", "mpas",
        "--mpas-ice-skin-prognostic", "--mpas-ice-thickness-m", "1.5",
    ]), parser))
    assert cfg.mpas_ice_skin_prognostic is True
    assert cfg.mpas_ice_thickness_m == 1.5


# ---------------------------------------------------------------------------
# End-to-end wiring (subprocess): exercises the FULL per-step advance +
# per-step T_sfc re-anchor + checkpoint save/load path in the real _run_mpas
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
    (codex-3).  Verifies the finiteness guard in the _run_mpas seed overlay."""
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
