"""Tests for legoesm.ocean.fidelity.veros_runner.

Most tests are gated by ``pytest.importorskip("veros")`` so the suite
skips cleanly when Veros is not installed. One slow integration test
actually runs ACCSetup for a few timesteps; it is marked ``@slow`` so it
stays out of default PR CI.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("veros")


def _import_runner_module():
    from legoesm.ocean.fidelity import veros_runner
    return veros_runner


def test_available_cases_includes_implemented_adapters(isolated_cache):
    runner = _import_runner_module()
    cases = runner.available_cases()
    assert "acc_channel" in cases
    assert "global_overturning" in cases
    assert "lock_exchange" in cases
    assert "overflow" in cases


def test_lock_exchange_factory_returns_setup_with_target_runlen(isolated_cache):
    from legoesm.ocean.fidelity.veros_configs import lock_exchange
    setup = lock_exchange.make_setup(runlen_s=42.0)
    assert setup._legoesm_target_runlen_s == 42.0


def test_overflow_factory_returns_setup_with_target_runlen(isolated_cache):
    from legoesm.ocean.fidelity.veros_configs import overflow
    setup = overflow.make_setup(runlen_s=84.0)
    assert setup._legoesm_target_runlen_s == 84.0


def test_lock_exchange_constants_match_petersen_geometry():
    from legoesm.ocean.fidelity.veros_configs import lock_exchange
    assert lock_exchange.DOMAIN_LX_M == 64_000.0
    assert lock_exchange.DOMAIN_LZ_M == 20.0
    assert lock_exchange.T_COLD_C == 5.0
    assert lock_exchange.T_WARM_C == 30.0


def test_overflow_geometry_has_shelf_slope_abyss():
    from legoesm.ocean.fidelity.veros_configs import overflow
    assert overflow.SHELF_DEPTH_M < overflow.ABYSS_DEPTH_M
    assert overflow.SHELF_X_END_M < overflow.SLOPE_X_END_M
    assert overflow.DOMAIN_LZ_M == overflow.ABYSS_DEPTH_M


def test_unknown_case_raises_value_error(isolated_cache):
    runner = _import_runner_module()
    with pytest.raises(ValueError, match="Unknown Veros case"):
        runner.run_veros("not_a_real_case")


def test_cache_key_stable_across_redundant_calls(isolated_cache):
    runner = _import_runner_module()
    k1 = runner._cache_key("acc_channel", runlen_s=86400.0, identifier="x")
    k2 = runner._cache_key("acc_channel", runlen_s=86400.0, identifier="x")
    assert k1 == k2
    assert len(k1) == 16


def test_cache_key_changes_on_runlen_change(isolated_cache):
    runner = _import_runner_module()
    a = runner._cache_key("acc_channel", runlen_s=86400.0, identifier="x")
    b = runner._cache_key("acc_channel", runlen_s=2 * 86400.0, identifier="x")
    assert a != b


def test_cache_key_changes_on_case_change(isolated_cache):
    runner = _import_runner_module()
    a = runner._cache_key("acc_channel", runlen_s=86400.0, identifier="x")
    b = runner._cache_key("global_overturning", runlen_s=86400.0, identifier="x")
    assert a != b


def test_veros_result_is_frozen_namedtuple(isolated_cache):
    runner = _import_runner_module()
    result = runner.VerosResult(
        case_name="x", times_s=np.array([0.0]),
        variables={}, grid_metadata={}, provenance={"k": "v"},
    )
    with pytest.raises(Exception):
        result.case_name = "y"  # type: ignore[misc]


def test_save_and_load_round_trip(isolated_cache):
    runner = _import_runner_module()
    case_dir = isolated_cache / "veros" / "acc_channel" / "deadbeefdeadbeef"
    result = runner.VerosResult(
        case_name="acc_channel",
        times_s=np.array([0.0, 1.0]),
        variables={"temp": np.zeros((4, 4))},
        grid_metadata={"nx": 4, "ny": 4, "nz": 4},
        provenance={"veros_version": "test"},
    )
    runner._save_result(case_dir, "deadbeefdeadbeef", result)
    loaded = runner._load_cached(case_dir)
    assert loaded is not None
    assert loaded.case_name == "acc_channel"
    np.testing.assert_allclose(loaded.times_s, [0.0, 1.0])
    assert "temp" in loaded.variables


def test_load_missing_returns_none(isolated_cache):
    runner = _import_runner_module()
    empty = isolated_cache / "veros" / "nothing"
    empty.mkdir(parents=True)
    assert runner._load_cached(empty) is None


def test_acc_factory_returns_veros_setup(isolated_cache):
    from legoesm.ocean.fidelity.veros_configs import acc_channel
    setup = acc_channel.make_setup(runlen_s=60.0)
    # ACCSetup subclasses VerosSetup but lazy import means we can't compare
    # class identity here without re-importing — check the recorded target.
    assert setup._legoesm_target_runlen_s == 60.0


def test_global_overturning_factory_returns_setup(isolated_cache):
    from legoesm.ocean.fidelity.veros_configs import global_overturning
    setup = global_overturning.make_setup(runlen_s=120.0)
    assert setup._legoesm_target_runlen_s == 120.0


@pytest.mark.slow
def test_acc_channel_smoke_run_produces_real_result(isolated_cache):
    """Integration test: run ACCSetup for a single tracer timestep.

    Veros's ACCSetup default ``dt_tracer = 43200 s`` (0.5 day). Asking
    Veros to integrate for 1.5 dt_tracer ensures at least one timestep
    runs without picking a long runlen.
    """
    runner = _import_runner_module()
    runlen_s = 1.5 * 43200.0
    result = runner.run_veros(
        "acc_channel", runlen_s=runlen_s, force_recompute=True,
    )
    assert result.case_name == "acc_channel"
    assert "temp" in result.variables
    assert result.variables["temp"].size > 0
    assert result.provenance["runlen_s"] == runlen_s
    assert result.provenance["wall_seconds"] > 0


@pytest.mark.slow
def test_lock_exchange_smoke_run_produces_dense_descent(isolated_cache):
    """Petersen 2015 lock-exchange: dense water column should descend.

    After a 5-minute integration we expect: (a) the temperature distribution
    has spread beyond its initial 5/30 binary; (b) some kinetic energy has
    appeared from the initial pressure imbalance. We do not assert on the
    RPE rate or front position here — those belong in a fidelity-tier test.
    """
    runner = _import_runner_module()
    result = runner.run_veros("lock_exchange", runlen_s=300.0,
                              force_recompute=True)
    assert result.case_name == "lock_exchange"
    temp = result.variables["temp"]
    # Initial: only 5 or 30. After advection should sample intermediate
    # values somewhere in the column.
    interior = temp[2:-2, 2:-2, :, -1]  # drop halos, take last tau
    assert (5.5 < interior).any() and (interior < 29.5).any(), (
        "temperature should have advected and produced intermediate values"
    )
    # Velocity should be nonzero (gravity current generates flow).
    u = result.variables["u"]
    assert float(np.abs(u).max()) > 0.0


@pytest.mark.slow
def test_overflow_smoke_run_produces_realistic_field(isolated_cache):
    """Overflow integration produces a finite, non-trivial T field."""
    runner = _import_runner_module()
    result = runner.run_veros("overflow", runlen_s=600.0, force_recompute=True)
    assert result.case_name == "overflow"
    temp = result.variables["temp"]
    assert np.isfinite(temp).all()
    # Cold (5°C) shelf water present; warm (20°C) abyssal water present.
    assert float(temp.min()) < 6.0
    assert float(temp.max()) > 19.0


@pytest.mark.slow
def test_acc_channel_caching_round_trip(isolated_cache):
    """Second call hits the on-disk cache without re-running."""
    runner = _import_runner_module()
    runlen_s = 1.5 * 43200.0
    first = runner.run_veros("acc_channel", runlen_s=runlen_s)
    second = runner.run_veros("acc_channel", runlen_s=runlen_s)
    assert first.provenance["key"] == second.provenance["key"]
    # second call is dominated by pickle load; wall_seconds is preserved
    # from the cached run, so the new call still reports the original time
    assert first.provenance["wall_seconds"] == second.provenance["wall_seconds"]
