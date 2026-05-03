"""Regression-pin the BCW benchmark's ``--scan-steps auto`` dispatch
table introduced in iter-219, plus the iter-216 RuntimeWarning that
fires for the empirically-bad icosahedral + scan>1 combination.
Mapping was determined by iter-212 honest measurements (precision-
fixed timing) on the RTX 5090 Laptop + 8-thread CPU baseline.  Future
retunes belong here too.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest


def _load_bcw_module():
    """Load ``scripts/run_baroclinic_wave_benchmark.py`` as a module
    via importlib so the AUTO_SCAN_STEPS table can be inspected without
    invoking ``main()``.  Importing the script as a top-level package
    isn't possible (it lives in ``scripts/`` which isn't a package).
    """
    here = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_bcw_benchmark",
        here / "scripts" / "run_baroclinic_wave_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auto_scan_steps_dispatch_table():
    """The iter-219 ``--scan-steps auto`` mapping must match the
    iter-212 honest measurements: spectral=24 (+29..45 %),
    cubed-sphere=24 (+20 %), icosahedral=1 (scan>1 was -9 % at I5).
    Drift here = a quiet throughput regression across the entire
    strong-sweep driver.
    """
    bcw = _load_bcw_module()
    assert bcw.AUTO_SCAN_STEPS == {
        "spectral":     24,
        "cubed-sphere": 24,
        "icosahedral":  1,
    }, (
        "AUTO_SCAN_STEPS dispatch drifted from the iter-212 honest "
        f"measurements: got {bcw.AUTO_SCAN_STEPS!r}.  Re-run the "
        f"full strong sweep before changing this table."
    )


def test_auto_scan_steps_covers_all_grid_choices():
    """Every grid in GRID_CHOICES must have an entry in the dispatch
    table — silent ``KeyError`` would degrade gracefully but lose the
    iter-212 recommendation.
    """
    bcw = _load_bcw_module()
    missing = set(bcw.GRID_CHOICES) - set(bcw.AUTO_SCAN_STEPS)
    assert not missing, (
        f"GRID_CHOICES contains {missing!r} but AUTO_SCAN_STEPS "
        f"does not — ``--scan-steps auto`` will fall back to 1 with "
        f"no warning."
    )


def test_resolve_scan_steps_auto_dispatches_per_grid():
    """``resolve_scan_steps(grid, "auto")`` must return the table entry
    for each known grid (iter-219 happy path).
    """
    bcw = _load_bcw_module()
    for grid, expected in bcw.AUTO_SCAN_STEPS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # surface unexpected warnings
            try:
                got = bcw.resolve_scan_steps(grid, "auto", verbose=False)
            except RuntimeWarning:
                # icosahedral with scan>1 should warn; for icosahedral the
                # auto value is 1 so no warning is expected — re-raise.
                pytest.fail(f"resolve_scan_steps('{grid}', 'auto') warned unexpectedly")
        assert got == expected, (
            f"resolve_scan_steps('{grid}', 'auto') returned {got}, "
            f"expected {expected} from AUTO_SCAN_STEPS"
        )


def test_resolve_scan_steps_icosahedral_explicit_warns():
    """iter-216: explicit ``--scan-steps > 1`` on icosahedral must
    emit a RuntimeWarning.  Reason: iter-212 measured -9 % throughput
    at I5 because the MPAS dycore is already well-fused, so users who
    forgot to pass ``auto`` should be told.
    """
    bcw = _load_bcw_module()
    with pytest.warns(RuntimeWarning, match=r"iter-212 measured -9 %"):
        resolved = bcw.resolve_scan_steps("icosahedral", 24, verbose=False)
    assert resolved == 24, (
        "resolve_scan_steps must still return the user-requested "
        "value — the warning is informational, not a hard reject"
    )


def test_resolve_scan_steps_icosahedral_one_is_silent():
    """The warning must fire only for ``scan_steps > 1``; plain ``1``
    or ``auto`` (which resolves to 1 for icosahedral) must be silent
    or no user would ever see a clean run.
    """
    bcw = _load_bcw_module()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails
        assert bcw.resolve_scan_steps("icosahedral", 1, verbose=False) == 1
        assert bcw.resolve_scan_steps("icosahedral", "auto", verbose=False) == 1


def test_resolve_scan_steps_passthrough_for_other_grids():
    """Spectral and cubed-sphere with scan>1 must resolve to the user
    value with no RuntimeWarning — only icosahedral has the -9 % regret.
    """
    bcw = _load_bcw_module()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bcw.resolve_scan_steps("spectral", 12, verbose=False) == 12
        assert bcw.resolve_scan_steps("cubed-sphere", 8, verbose=False) == 8


# ---------------------------------------------------------------------------
# iter-223: chunk_crosses_boundary regression — pins the iter-208 fix that
# replaced ``current_step % period == 0`` with the chunk-crossing predicate.
# Without this, --scan-steps > diag_interval silently dropped every hourly
# diag sample.

def test_chunk_crosses_boundary_modulo_regime():
    """When chunks advance one step at a time (scan_steps=1), the
    chunk-crossing predicate must be identical to the modulo check.
    """
    bcw = _load_bcw_module()
    period = 12
    for current in range(1, 50):
        prev = current - 1
        modulo_result = (current % period == 0)
        chunk_result = bcw.chunk_crosses_boundary(prev, current, period)
        assert chunk_result == modulo_result, (
            f"divergence at current={current}: modulo={modulo_result}, "
            f"chunk={chunk_result}"
        )


def test_chunk_crosses_boundary_scan_steps_skip_regime():
    """iter-208 bug: with chunk size > period, the modulo check would
    miss every boundary.  The chunk-crossing predicate must still fire
    once per crossed boundary even when the chunk leaps over it.
    """
    bcw = _load_bcw_module()
    period = 12
    # Chunks of 24: prev=0, current=24 — crosses boundary 12 (and 24).
    assert bcw.chunk_crosses_boundary(0, 24, period) is True
    # prev=24, current=48 — crosses boundary 36 (and 48).
    assert bcw.chunk_crosses_boundary(24, 48, period) is True
    # prev=12, current=36 — crosses boundary 24 (and 36).
    assert bcw.chunk_crosses_boundary(12, 36, period) is True


def test_chunk_crosses_boundary_no_crossing():
    """Predicate must be False when the chunk lies strictly inside one
    period (e.g. scan_steps < period and the chunk doesn't span an edge).
    """
    bcw = _load_bcw_module()
    # prev=0, current=11 — does not reach 12.
    assert bcw.chunk_crosses_boundary(0, 11, 12) is False
    # prev=13, current=23 — both inside [12, 24).
    assert bcw.chunk_crosses_boundary(13, 23, 12) is False


def test_chunk_crosses_boundary_zero_period_safe():
    """A degenerate period of 0 (or negative) must short-circuit to
    False instead of hitting ZeroDivisionError.  Defensive-programming
    pin so a future caller passing a bad ``diag_interval_steps`` value
    fails closed (no diag) rather than crashing the whole run.
    """
    bcw = _load_bcw_module()
    assert bcw.chunk_crosses_boundary(0, 100, 0) is False
    assert bcw.chunk_crosses_boundary(0, 100, -1) is False
