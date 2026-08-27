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
    assert C.cancellation_status(0.0, 3.0, 0.0) == "CONFIRMED_SENSITIVE"
    assert C.cancellation_status(0.8, 0.5, 0.5) == "CONFIRMED_ROBUST"
    assert C.cancellation_status(0.0, 3.0, 0.0) != "CONFIRMED_ROBUST"


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
    assert C.cancellation_status(0.25, 1.5, 1.5) == "UNRESOLVED"
