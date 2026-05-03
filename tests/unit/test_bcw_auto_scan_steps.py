"""Tests for the iter-221 auto ``--scan-steps`` resolver.

The BCW benchmark exposes ``--scan-steps`` as either an integer
(legacy) or the literal ``"auto"`` (default), in which case the
per-grid optimum from the iter-220 sweep table is selected.
This module verifies:

- Each known (grid, resolution) pair returns the iter-220 optimum.
- A resolution above the largest tabulated entry falls back to the
  largest entry ≤ n_grid.
- A resolution below the smallest tabulated entry falls back to the
  grid-default key (``None``).
- An unknown grid type falls back to K=1 instead of raising.
- A non-integer, non-"auto" value raises ``SystemExit`` with a
  helpful message.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BCW = _REPO_ROOT / "scripts" / "run_baroclinic_wave_benchmark.py"


def _load_bcw_module():
    """Import ``run_baroclinic_wave_benchmark.py`` as a module without
    triggering ``argparse`` parsing (``main()`` would call
    ``parser.parse_args()`` and grab the pytest CLI flags).  Loading
    via ``importlib.util.spec_from_file_location`` and exec'ing is
    enough to expose ``_resolve_scan_steps`` and
    ``_AUTO_SCAN_STEPS_TABLE``."""
    spec = importlib.util.spec_from_file_location("bcw_benchmark", _BCW)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bcw_benchmark"] = module
    # The module guards ``main()`` execution behind ``__name__ ==
    # '__main__'`` so importing as ``bcw_benchmark`` is safe.
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bcw():
    return _load_bcw_module()


class TestAutoScanStepsTable:
    """The iter-220 sweep optimums must round-trip."""

    @pytest.mark.parametrize(
        "grid,n_grid,expected",
        [
            ("spectral", 21, 24),
            ("spectral", 42, 48),
            ("cubed-sphere", 24, 12),
            ("cubed-sphere", 48, 24),
            ("cubed-sphere", 96, 24),
            ("icosahedral", 4, 24),
            ("icosahedral", 5, 48),
            ("icosahedral", 6, 48),
        ],
    )
    def test_known_optimums(self, bcw, grid, n_grid, expected):
        out = bcw._resolve_scan_steps("auto", grid, n_grid)
        assert out == expected, (
            f"auto pick for {grid}/{n_grid} should be {expected}, got {out}"
        )


class TestAutoScanStepsFallbacks:
    def test_resolution_above_largest_falls_back_to_largest_known(self, bcw):
        # cubed-sphere has C96 in the table; C192 should fall to C96's K.
        assert bcw._resolve_scan_steps("auto", "cubed-sphere", 192) == 24

    def test_resolution_below_smallest_uses_grid_default(self, bcw):
        # spectral table has T21 as smallest; T10 should fall to None default.
        assert bcw._resolve_scan_steps("auto", "spectral", 10) == 24

    def test_unknown_grid_falls_back_to_1(self, bcw):
        # An unknown grid name ⇒ default K=1 (no-op).
        assert bcw._resolve_scan_steps("auto", "voronoi-2d-foo", 32) == 1


class TestExplicitScanSteps:
    def test_integer_passthrough(self, bcw):
        assert bcw._resolve_scan_steps("12", "spectral", 21) == 12

    def test_zero_clamps_to_one(self, bcw):
        assert bcw._resolve_scan_steps("0", "spectral", 21) == 1

    def test_negative_clamps_to_one(self, bcw):
        assert bcw._resolve_scan_steps("-3", "spectral", 21) == 1

    def test_non_integer_raises(self, bcw):
        with pytest.raises(SystemExit):
            bcw._resolve_scan_steps("smol", "spectral", 21)

    def test_auto_case_insensitive(self, bcw):
        assert bcw._resolve_scan_steps("AUTO", "spectral", 21) == 24
        assert bcw._resolve_scan_steps("Auto", "spectral", 21) == 24
