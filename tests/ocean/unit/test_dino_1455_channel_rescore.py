"""Direct tests for the #1455 channel-rescore instrument.

Every guard below includes a violating input and asserts that the guard fires;
green tests therefore cannot be obtained from controls that only pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = ROOT / "scripts/validate/ocean_fidelity/dino_1226"
sys.path.insert(0, str(PROBE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "channel_rescore", PROBE_DIR / "channel_rescore.py")
C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(C)


def test_pearson_names_denominator_and_refuses_constant():
    x = np.arange(C.N_ROWS, dtype=np.float64)
    r, den = C.pearson(x, 2.0 * x + 1.0)
    assert r == pytest.approx(1.0)
    assert den > 0.0
    with pytest.raises(ValueError, match="UNMEASURABLE"):
        C.pearson(np.ones(C.N_ROWS), x)


def test_effective_rows_rejects_independent_row_plant():
    x = np.repeat(np.arange(7, dtype=np.float64), 5)
    n_eff, denominator = C.effective_rows(x, x)
    assert 3.0 <= n_eff < C.N_ROWS
    assert denominator > 1.0
    with pytest.raises(AssertionError):
        C._assert_close(n_eff, float(C.N_ROWS))


def test_fisher_interval_withholds_precision_at_three_effective_rows():
    assert C.fisher_interval(0.99, 3.0) == (-1.0, 1.0)
    lo, hi = C.fisher_interval(0.8, 12.0)
    assert -1.0 < lo < 0.8 < hi < 1.0


def test_row_to_band_closure_guard_fires_on_wrong_row_weight():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(C.R.A.NY, C.R.A.NX, C.R.A.NZ))
    rows = C.R.row_transports(u, C.R.A.umask)
    band = C.R.band_transport(u, C.R.A.umask, C.CHANNEL_ROWS)
    C._assert_close(float(np.sum(rows[C.CHANNEL_ROWS])), band, 1e-9)
    rows[C.CHANNEL_INDEX[0]] += 1e-3
    with pytest.raises(AssertionError, match="closure failed"):
        C._assert_close(float(np.sum(rows[C.CHANNEL_ROWS])), band, 1e-9)


def test_thickness_weight_control_rejects_layer_average():
    u = np.zeros((C.R.A.NY, C.R.A.NX, C.R.A.NZ), dtype=np.float64)
    u[:, :, 0] = 1.0
    official = C.R.row_transports(u, C.R.A.umask)[C.CHANNEL_ROWS]
    layer_average = np.mean(np.where(C.R.A.umask, u, 0.0), axis=(1, 2))[
        C.CHANNEL_ROWS]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(layer_average, official,
                                   rtol=1e-12, atol=1e-12)


def test_pattern_plant_crosses_persistence_bar():
    base = np.sin(np.linspace(0.0, 2.0 * np.pi, C.N_ROWS))
    stable = {(m, t): base + 0.01 * m + 0.001 * t
              for m in range(C.N_MEM) for t in C.HORIZONS}
    stable_r = np.median(C._correlations(stable, C._pair_specs("time")))
    assert stable_r >= C.PERSIST_HI

    rng = np.random.default_rng(22)
    shuffled = {(m, t): base[rng.permutation(C.N_ROWS)]
                for m in range(C.N_MEM) for t in C.HORIZONS}
    shuffled_r = np.median(C._correlations(shuffled, C._pair_specs("time")))
    assert shuffled_r <= C.PERSIST_LO


def test_roughness_and_cancellation_plants_fire():
    alternating = (-1.0) ** np.arange(C.N_ROWS)
    smooth = np.linspace(1.0, 2.0, C.N_ROWS)
    assert C.roughness(alternating)[0] >= C.ROUGH_GRID
    assert C.roughness(smooth)[0] <= C.ROUGH_SMOOTH
    assert C.cancellation_status(0.0, 3.0) == "CONFIRMED_SENSITIVE"
    assert C.cancellation_status(0.8, 0.5) == "CONFIRMED_ROBUST"
    assert C.cancellation_status(0.0, 3.0) != "CONFIRMED_ROBUST"


def test_row_specific_floor_plant_fires():
    gaps = np.asarray([1.0, 5.0])
    floors = np.asarray([0.1, 10.0])
    local = np.abs(gaps) > C.ESCALATION_BAR * floors
    globalized = np.abs(gaps) > C.ESCALATION_BAR * np.mean(floors)
    assert not np.array_equal(local, globalized)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(globalized, local)


def test_final_classifiers_have_indeterminate_middle():
    assert C.row_agreement_status(0.95, 0.05) == "CONFIRMED_GENUINE"
    assert C.row_agreement_status(0.40, 0.30) == "REFUTED_GENUINE"
    assert C.row_agreement_status(0.80, 0.15) == "UNRESOLVED"
    assert C.cancellation_status(0.25, 1.5) == "UNRESOLVED"


def test_clock_helper_compat_aliases_rename_without_wrapping():
    original_public = getattr(C.R.X.T, "restart_elapsed_seconds", None)
    private = C.R.X.T._restart_elapsed_seconds
    if original_public is not None:
        delattr(C.R.X.T, "restart_elapsed_seconds")
    try:
        label = C._install_clock_helper_compat()
        assert "compat alias" in label
        assert C.R.X.T.restart_elapsed_seconds is private
    finally:
        if original_public is None:
            delattr(C.R.X.T, "restart_elapsed_seconds")
        else:
            C.R.X.T.restart_elapsed_seconds = original_public


def _synthetic_ensemble(pattern_removed=False):
    """Four-member/four-horizon row ensembles with an analytic answer."""
    rng = np.random.default_rng(1455)
    y = np.linspace(-1.0, 1.0, C.R.A.NY)
    channel_y = np.linspace(-1.0, 1.0, C.N_ROWS)
    smooth = 0.012 * (0.3 + np.sin(np.pi * channel_y))
    hetero = np.linspace(0.15, 1.8, C.N_ROWS)
    independent = {
        day: rng.normal(0.0, 0.012, C.N_ROWS) for day in C.HORIZONS
    }
    lego, nemo = {}, {}
    member_coeff = np.asarray([-1.5, -0.5, 0.5, 1.5])
    for it, day in enumerate(C.HORIZONS):
        lego[day] = np.empty((C.N_MEM, C.R.A.NY), dtype=np.float64)
        nemo[day] = np.empty_like(lego[day])
        signal = independent[day] if pattern_removed else smooth * (1 + 0.04 * it)
        for m, coeff in enumerate(member_coeff):
            oracle = 34.0 + 0.2 * np.sin(3.0 * y) + coeff * 2e-5 * (1 + y)
            member_shape = coeff * 2e-4 * hetero * (
                1.0 + 0.15 * np.cos((m + 1) * np.pi * channel_y))
            gap = signal + member_shape
            nemo[day][m] = oracle
            lego[day][m] = oracle
            lego[day][m, C.CHANNEL_ROWS] += gap
    return lego, nemo


def _assert_engine_oracle(known, removed, known_data):
    assert known["final_verdict"] == (
        "PLAUSIBLE_PERSISTENT_UNRESOLVED["
        "E_band_sensitivity,B_member_stability]")
    assert removed["final_verdict"] == (
        "PLAUSIBLE_DECORRELATING_UNRESOLVED["
        "C_spatial_structure,B_member_stability]")
    assert removed["time_stability"]["status"] != "CONFIRMED_PERSISTENT"  # M12
    assert removed["spatial_structure"]["status"] != (                    # M14
        "CONFIRMED_PHYSICAL_STRUCTURE")

    expected_floor = np.sqrt(
        np.std(known_data[0][360][:, C.CHANNEL_ROWS],
               axis=0, ddof=1) ** 2
        + np.std(known_data[1][360][:, C.CHANNEL_ROWS],
                 axis=0, ddof=1) ** 2)
    np.testing.assert_allclose(known["row_floors_sv"]["360"], expected_floor)
    assert np.ptp(expected_floor) > 1e-4                              # M8
    assert known["time_stability"]["bootstrap_unique_median_draws"] > 5  # M11
    assert known["member_stability"]["self_comparison_count"] == 0       # M16
    assert known["member_stability"]["per_horizon_median_r"]["360"] < 1.0


def test_engine_known_pattern_and_pattern_removed_twin_kill_mutants():
    """End-to-end oracle for formerly surviving M8/M11/M12/M14/M16."""
    known_data = _synthetic_ensemble(False)
    known = C.evaluate(*known_data, n_boot=400)
    removed = C.evaluate(*_synthetic_ensemble(True), n_boot=400)
    _assert_engine_oracle(known, removed, known_data)


@pytest.mark.parametrize("mutation", ["M8", "M11", "M12", "M14", "M16"])
def test_formerly_surviving_engine_mutations_now_go_red(monkeypatch, mutation):
    """Inject each reviewed mutation and prove the end-to-end oracle rejects it."""
    original_floor = C.row_floors
    original_spatial = C.spatial_decider
    original_pairs = C._pair_specs
    original_compose = C.compose_verdict

    if mutation == "M8":
        def global_floor(lego, nemo):
            local = original_floor(lego, nemo)
            return {day: np.full(C.N_ROWS, np.mean(value))
                    for day, value in local.items()}
        monkeypatch.setattr(C, "row_floors", global_floor)
    elif mutation == "M11":
        monkeypatch.setattr(
            C, "moving_block_indices",
            lambda n, block, rng: np.arange(n))
    elif mutation == "M12":
        def forced_persistent(time, member, spatial, row_level, sensitivity):
            mutated_time = dict(time)
            mutated_time["status"] = "CONFIRMED_PERSISTENT"
            return original_compose(mutated_time, member, spatial, row_level,
                                    sensitivity)
        monkeypatch.setattr(C, "compose_verdict", forced_persistent)
    elif mutation == "M14":
        def forced_physical(*args, **kwargs):
            result = original_spatial(*args, **kwargs)
            result["status"] = "CONFIRMED_PHYSICAL_STRUCTURE"
            return result
        monkeypatch.setattr(C, "spatial_decider", forced_physical)
    elif mutation == "M16":
        def self_pairs(kind):
            if kind == "member":
                return [((m, day), (m, day)) for day in C.HORIZONS
                        for m in range(C.N_MEM)]
            return original_pairs(kind)
        monkeypatch.setattr(C, "_pair_specs", self_pairs)

    known_data = _synthetic_ensemble(False)
    known = C.evaluate(*known_data, n_boot=200)
    removed = C.evaluate(*_synthetic_ensemble(True), n_boot=200)
    with pytest.raises(AssertionError):
        _assert_engine_oracle(known, removed, known_data)


def test_signed_r_decorrelation_guard_fires_on_coherent_sign_reversal():
    base = np.sin(np.linspace(0.0, 2.0 * np.pi, C.N_ROWS))
    profiles = {}
    for m in range(C.N_MEM):
        for it, day in enumerate(C.HORIZONS):
            profiles[(m, day)] = (-base if it == 3 else base) + 0.001 * m
    scored = C.aggregate_stability(profiles, "time", n_boot=200)
    assert scored["max_abs_r"] >= C.PERSIST_HI
    assert scored["status"] != "CONFIRMED_DECORRELATING"
    assert "WITHHELD" in scored["decorrelation_guard"]


def test_upstream_sha_pin_rejects_mutated_reducer(monkeypatch):
    monkeypatch.setattr(C, "UPSTREAM_REGIONAL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="exact reducer reuse is not proven"):
        C._verify_upstream()
