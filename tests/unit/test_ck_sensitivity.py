"""Test the pure clause-6 feasibility metric in scripts/experiment/ck_sensitivity_vs_era5.

The model-run + real-ERA5 ``main`` is data-dependent (a model integration + a local ERA5
archive); this locks the ``bias_ck_sensitivity`` metric that turns two C_K runs' biases
into the go/no-go an operator reads.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_bias_ck_sensitivity_flags_insensitive_vs_sensitive():
    from scripts.experiment.ck_sensitivity_vs_era5 import bias_ck_sensitivity

    # Idealization-dominated (the REAL iter-412 result: 11.2 vs 11.2, Δ ≈ 0.001) — a
    # bias that barely moves with C_K ⇒ a correction loop can't lower it ⇒ NOT feasible.
    lo = np.full((4, 4), 11.2)
    hi = lo + 0.001
    s = bias_ck_sensitivity(lo, hi)
    assert not s["c_k_feasible"] and s["controllable_fraction"] < 0.01
    assert s["mean_bias"] == pytest.approx(11.2, abs=0.01)   # avg of the two runs
    assert s["mean_abs_delta"] == pytest.approx(0.001, rel=1e-4)

    # C_K-SENSITIVE: mean bias 4, C_K moves it by 2 (50%) ⇒ feasible.
    s2 = bias_ck_sensitivity(np.full((4, 4), 5.0), np.full((4, 4), 3.0))
    assert s2["controllable_fraction"] == pytest.approx(0.5) and s2["c_k_feasible"]

    # NaN cells (e.g. an SST-over-land env tag) are IGNORED, not propagated.
    lo3 = np.array([[5.0, np.nan], [5.0, 5.0]])
    hi3 = np.array([[3.0, np.nan], [3.0, 3.0]])
    assert np.isfinite(bias_ck_sensitivity(lo3, hi3)["controllable_fraction"])

    # Shape mismatch fails loud (a wrong pairing would silently mis-diff).
    with pytest.raises(ValueError, match="share shape"):
        bias_ck_sensitivity(np.zeros((2, 2)), np.zeros((3,)))


def test_bias_ck_sensitivity_ocean_only_mask_rescues_a_land_diluted_verdict():
    """The OCEAN-only mask must score the same columns the ocean-only campaign ranks — a
    land-model-driven bias the closure cannot move would otherwise dilute the fraction and
    FALSELY gate an ocean-only campaign NO-GO (the iter-466 silent-drop class, in the
    pre-flight)."""
    from scripts.experiment.ck_sensitivity_vs_era5 import bias_ck_sensitivity

    # 6 LAND columns: a big, C_K-INSENSITIVE bias (Δ=0) — the land model's own error.
    # 2 OCEAN columns: a C_K-SENSITIVE bias (mean 4, Δ=2 ⇒ 50% controllable).
    lo = np.array([50.0] * 6 + [5.0, 5.0])
    hi = np.array([50.0] * 6 + [3.0, 3.0])

    # Over ALL columns the land bias dilutes the fraction below the 5% floor ⇒ NO-GO.
    s_all = bias_ck_sensitivity(lo, hi)
    assert not s_all["c_k_feasible"] and s_all["controllable_fraction"] < 0.05

    # OCEAN-only (mask out the 6 land columns) recovers the true 50% sensitivity ⇒ GO.
    ocean = np.array([False] * 6 + [True, True])
    s_ocn = bias_ck_sensitivity(lo, hi, valid_mask=ocean)
    assert s_ocn["c_k_feasible"]
    assert s_ocn["controllable_fraction"] == pytest.approx(0.5)
    assert s_ocn["mean_bias"] == pytest.approx(4.0)

    # A mask that shares no shape with the scores fails loud (a row-major misalignment
    # would silently score the wrong columns).
    with pytest.raises(ValueError, match="must share"):
        bias_ck_sensitivity(lo, hi, valid_mask=np.array([True, False, True]))

    # A mask selecting nothing is an error, not a NaN no-op.
    with pytest.raises(ValueError, match="selects no columns"):
        bias_ck_sensitivity(lo, hi, valid_mask=np.zeros(8, dtype=bool))


def test_main_fails_fast_on_a_nonexistent_land_mask(tmp_path):
    """A typo'd --land-mask-path must fail BEFORE the (slow) model runs — a non-existent
    mask would otherwise surface only deep in the model build, after the setup is spent
    (mirrors the iter-454 config-writer fail-fast)."""
    from scripts.experiment.ck_sensitivity_vs_era5 import main

    missing = tmp_path / "no_such_mask.nc"
    with pytest.raises(SystemExit, match="does not exist"):
        main(["--local-era5-dir", str(tmp_path), "--local-era5-date", "20200101",
              "--ocean-only", "--land-mask-path", str(missing)])


def test_preflight_exit_code_gates_the_hpc_launch():
    """The pre-flight exit code gates an HPC launch: GO (0) when the C_K loop CAN lower the
    bias (a rising sensitivity trend OR a C_K-feasible config), NO-GO (1) when it cannot
    (a FLAT idealization-dominated trend / an insensitive config) — mirrors the campaign's
    ``_campaign_exit_code`` so ``ck_sensitivity … && sbatch`` is a real go/no-go gate."""
    from scripts.experiment.ck_sensitivity_vs_era5 import (
        _preflight_exit_code,
        bias_ck_sensitivity,
        ck_sensitivity_trend,
    )

    assert _preflight_exit_code(True) == 0 and _preflight_exit_code(False) == 1
    # --days-sweep: a FLAT trend (idealization-dominated) → NO-GO; a rising trend → GO.
    flat = ck_sensitivity_trend([1, 3, 6], [0.01, 0.01, 0.01])
    assert _preflight_exit_code(flat["monotonic_increasing"]) == 1
    rising = ck_sensitivity_trend([1, 3, 6], [0.003, 0.004, 0.005])
    assert _preflight_exit_code(rising["monotonic_increasing"]) == 0
    # single-config: a C_K-sensitive bias → GO; the idealization-dominated iter-412 case → NO-GO.
    feasible = bias_ck_sensitivity(np.full((4, 4), 5.0), np.full((4, 4), 3.0))
    assert _preflight_exit_code(feasible["c_k_feasible"]) == 0
    insensitive = bias_ck_sensitivity(np.full((4, 4), 11.2), np.full((4, 4), 11.201))
    assert _preflight_exit_code(insensitive["c_k_feasible"]) == 1


def test_per_level_ck_bias_sensitivity_localizes_to_the_boundary_layer():
    """The per-level breakdown (iter 416) localizes C_K control: a free-tropospheric bias
    that C_K does NOT move shows a ~0 controllable fraction, while a BL level C_K DOES
    move shows a large one — vindicating that the LES→C_K correction is effective WHERE it
    acts even when the full-column bias is free-trop (radiation) dominated."""
    from scripts.experiment.ck_sensitivity_vs_era5 import per_level_ck_bias_sensitivity

    ncol, nlev = 4, 3
    ref = np.full((ncol, nlev), 250.0)
    # Free-trop (levels 0,1): a big bias UNCHANGED by C_K (both runs +10 K).
    # BL (level 2, near-surface): C_K MOVES it (+2 K at low C_K → +0.5 K at high C_K).
    t_lo = ref.copy()
    t_lo[:, :2] += 10.0
    t_lo[:, 2] += 2.0
    t_hi = ref.copy()
    t_hi[:, :2] += 10.0
    t_hi[:, 2] += 0.5
    out = per_level_ck_bias_sensitivity(t_lo, t_hi, ref, np.ones(ncol))
    frac = out["controllable_fraction_per_level"]
    assert frac[0] < 1e-6 and frac[1] < 1e-6           # free-trop: C_K-INDEPENDENT
    assert frac[2] > 0.5                                # BL: C_K-CONTROLLABLE
    assert out["bias_per_level"][0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="share shape"):
        per_level_ck_bias_sensitivity(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 3)), np.ones(2))


def test_ck_sensitivity_trend_distinguishes_equilibration_from_idealization_limited():
    """The run-length trend (iter 418) classifies WHY the model is C_K-insensitive: a BL
    C_K-controllable fraction that CLIMBS with run length is equilibration-limited (a longer
    run is the lever), a FLAT one is idealization-limited (needs real radiation/SST)."""
    from scripts.experiment.ck_sensitivity_vs_era5 import ck_sensitivity_trend

    # The REAL iter-418 shape (1/3/6-day aquaplanet): climbing but shallow — equilibration
    # is a lever, but a linear extrapolation needs an impractically long run to the floor.
    t = ck_sensitivity_trend([1, 3, 6], [0.00304, 0.00384, 0.00474])
    assert t["monotonic_increasing"] and t["slope_per_day"] > 0.0
    assert t["days_to_feasible_floor_linear"] > 100   # shallow ⇒ impractically long alone

    # Flat ⇒ NOT equilibration-limited; no finite days-to-floor (run length is not the lever).
    t2 = ck_sensitivity_trend([1, 3, 6], [0.01, 0.01, 0.01])
    assert not t2["monotonic_increasing"]
    assert t2["days_to_feasible_floor_linear"] is None

    # Already at/above the feasibility floor ⇒ None (no extrapolation needed).
    t3 = ck_sensitivity_trend([1, 2], [0.06, 0.07])
    assert t3["days_to_feasible_floor_linear"] is None

    # Unsorted input is sorted internally (the monotonicity check is in run-length order).
    t4 = ck_sensitivity_trend([6, 1, 3], [0.00474, 0.00304, 0.00384])
    assert t4["monotonic_increasing"]
    assert t4["slope_per_day"] == pytest.approx(t["slope_per_day"])

    with pytest.raises(ValueError, match="share shape"):
        ck_sensitivity_trend([1, 2], [0.003])
    with pytest.raises(ValueError, match=">= 2"):
        ck_sensitivity_trend([1], [0.003])
    # Duplicate run lengths are noise, not a trend — reject (codex-review iter 418).
    with pytest.raises(ValueError, match="duplicate day"):
        ck_sensitivity_trend([1, 1, 6], [0.003, 0.004, 0.005])
    # NaN would silently poison the fitted slope — reject.
    with pytest.raises(ValueError, match="NaN"):
        ck_sensitivity_trend([1, 3, 6], [0.003, np.nan, 0.005])


def test_format_per_level_report_flags_the_most_controllable_level():
    """The operator-facing per-level report (iter 417) flags the most C_K-controllable
    level so an operator reading a C_K-INSENSITIVE full-column verdict still sees that the
    closure controls the boundary layer (surface-last sigma → 1) — the actionable nuance
    that the insensitivity is the model's free-trop error, not the correction approach."""
    from scripts.experiment.ck_sensitivity_vs_era5 import format_per_level_report

    # Free-trop (sigma 0.1, 0.5): big bias, ~0 C_K-controllable; BL (sigma 0.95): controllable.
    pl = {
        "bias_per_level": np.array([10.0, 9.0, 2.0]),
        "controllable_fraction_per_level": np.array([0.001, 0.002, 0.40]),
    }
    sigma_full = np.array([0.1, 0.5, 0.95])         # surface-last
    lines = format_per_level_report(pl, sigma_full)
    assert len(lines) == 3
    assert "most C_K-controllable" in lines[2]      # the BL level is flagged
    assert "most C_K-controllable" not in lines[0]  # ... and only it
    assert "most C_K-controllable" not in lines[1]
    assert "0.950" in lines[2]
    # The bias value lands in the T-bias field of line 0 (not merely a sigma substring).
    assert "10 K" in lines[0].split("T-bias")[1]

    # Length mismatch fails loud (a wrong sigma would mis-label every level).
    with pytest.raises(ValueError, match="share length"):
        format_per_level_report(pl, np.array([0.1, 0.5]))


def test_format_per_level_report_degenerate_all_nan_fraction_no_flag_no_crash():
    """A degenerate run (every level perfectly matches / fails to compare ⇒ all-NaN
    controllable fractions) must NOT crash ``np.nanargmax`` (codex-review iter 417) — it
    reports every level with NO most-controllable flag rather than raising."""
    from scripts.experiment.ck_sensitivity_vs_era5 import (
        _most_controllable_level,
        format_per_level_report,
    )

    assert _most_controllable_level(np.full(3, np.nan)) is None
    assert _most_controllable_level(np.array([])) is None
    assert _most_controllable_level(np.array([0.1, 0.4, 0.2])) == 1
    pl = {
        "bias_per_level": np.array([5.0, 5.0, 5.0]),
        "controllable_fraction_per_level": np.full(3, np.nan),
    }
    lines = format_per_level_report(pl, np.array([0.1, 0.5, 0.95]))
    assert len(lines) == 3
    assert all("most C_K-controllable" not in ln for ln in lines)


def test_build_ck_atm_config_matches_the_campaign_grid_type(tmp_path):
    """The C_K-sensitivity twin's config carries the SAME grid family as the campaign (iter
    493) — the inline config was hardcoded latlon, so a cubed-sphere/Gaussian campaign's
    go/no-go was measured on the WRONG grid. Pure (no model run), so the grid-type threading
    + the C_K override are unit-testable; each builds + validate_strict-passes."""
    from scripts.experiment.ck_sensitivity_vs_era5 import (
        _build_arg_parser,
        _build_ck_atm_config,
    )

    for gt in ("latlon", "cubed_sphere"):
        cfg = _build_ck_atm_config(0.7, resolution=4, nlev=5, days=1, radiation="gray",
                                   land_mask_path="", grid_type=gt)
        cfg.validate_strict()                              # builds a schema-valid config
        assert cfg.grid.grid_type == gt                   # the campaign's grid family
        assert cfg.turbulence == "clubb_lite"
        assert float(cfg.turbulence_override.clubb_lite.C_K) == 0.7   # the C_K under test
    # gaussian is REJECTED (iter 494): clubb_lite physics state is not threaded by the
    # spectral loop gaussian forces (issue #405) — a config that validates but fails the build.
    with pytest.raises(ValueError, match="unsupported for the clubb_lite"):
        _build_ck_atm_config(0.7, resolution=4, nlev=5, days=1, radiation="gray",
                             land_mask_path="", grid_type="gaussian")
    # the CLI exposes --grid-type and defaults to latlon (back-compat); gaussian is not a choice
    p = _build_arg_parser()
    base = ["--local-era5-dir", str(tmp_path), "--local-era5-date", "20200101"]
    assert p.parse_args(base).grid_type == "latlon"
    assert p.parse_args(base + ["--grid-type", "cubed_sphere"]).grid_type == "cubed_sphere"
    with pytest.raises(SystemExit):                        # argparse rejects gaussian
        p.parse_args(base + ["--grid-type", "gaussian"])


@pytest.mark.slow
def test_run_model_state_builds_and_runs_on_cubed_sphere():
    """REGRESSION (iter 493/494): the non-lat-lon ck-sensitivity EXECUTION. iter 494 found
    that the DRY-RUN does NOT build the model, so it missed that GAUSSIAN fails the build
    (clubb_lite carries prognostic physics state the spectral loop does not thread, issue
    #405) — gaussian was removed. CUBED-SPHERE (finite_volume; the per-step loop DOES thread
    the state, ocean_grid=None ⇒ the driver's own atm grid) is the remaining non-lat-lon
    option: this BUILDS + RUNS the tiny coupled model and asserts a finite (6,n,n,nlev) state
    + the (6,n,n) land fraction — a build break (the gaussian class) would fail here, not on a
    multi-day HPC run."""
    from scripts.experiment.ck_sensitivity_vs_era5 import _run_model_state

    m, grid, sigma, fland = _run_model_state(
        0.4, resolution=4, nlev=3, days=1, radiation="gray", grid_type="cubed_sphere")
    assert tuple(np.asarray(m.T).shape) == (6, 4, 4, 3)   # cubed-sphere column shape
    assert tuple(fland.shape) == (6, 4, 4)                 # static land fraction, grid-shaped
    assert bool(np.all(np.isfinite(np.asarray(m.T))))     # coupled run produced finite state
