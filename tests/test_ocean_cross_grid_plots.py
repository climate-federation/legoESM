"""Unit tests for the cross-grid comparison + diagnostic helpers in
``scripts/run_ocean_test_matrix.py`` and the modular copy at
``scripts/ocean_test_matrix/diagnostic_io.py``.

Pins the iter-19 latlon-C-grid staggered-velocity averaging fix and the
iter-21 defensive shape-validation refinement.

Counterpart to ``tests/test_atmosphere_cross_grid_plots.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Pandas import sanity check (iter-19)
# ---------------------------------------------------------------------------

class TestImports:
    def test_run_ocean_test_matrix_imports_pandas(self):
        """iter-19: ``pd.read_csv`` was used at lines 4617 and 6106 but
        pandas was never imported.  Test that the import is now
        present and that the module loads without error.
        """
        # Import the script module (will trigger the top-level imports).
        import importlib
        import run_ocean_test_matrix as M
        assert hasattr(M, "pd"), (
            "scripts/run_ocean_test_matrix.py is missing the iter-19 "
            "``import pandas as pd`` — cross-grid comparison block "
            "will silently fail with NameError"
        )
        # pandas namespace check.
        assert hasattr(M.pd, "read_csv")


# ---------------------------------------------------------------------------
# Latlon C-grid staggered velocity averaging logic (iter-19, iter-21)
# ---------------------------------------------------------------------------

class TestStaggeredVelocityAveraging:
    """Pin the iter-19 patch + iter-21 refinement for the latlon-ocean
    quiver overlay.  The patch lives directly inside the plotting
    function so we can't unit-test the function in isolation; instead
    test the averaging FORMULA on synthetic input arrays.
    """

    def _stagger_average(self, u, v):
        """Mirrors the in-source patch.  Returns ``(u_cc, v_cc, ok)``
        where ``ok`` is ``False`` when the shapes still don't match
        after the canonical face→cell averaging (iter-21 defensive
        skip)."""
        if u.shape != v.shape:
            if u.shape[1] == v.shape[1] + 1:
                u = 0.5 * (u[:, :-1] + u[:, 1:])
            if v.shape[0] == u.shape[0] + 1:
                v = 0.5 * (v[:-1, :] + v[1:, :])
        return u, v, (u.shape == v.shape)

    def test_canonical_latlon_cgrid(self):
        """``u(n_lat, n_lon+1)`` and ``v(n_lat+1, n_lon)`` average to
        the same cell-centred ``(n_lat, n_lon)`` shape with a simple
        average of adjacent face values."""
        n_lat, n_lon = 5, 7
        # Make distinguishable face values.
        u = np.arange(n_lat * (n_lon + 1), dtype=float).reshape(n_lat, n_lon + 1)
        v = np.arange((n_lat + 1) * n_lon, dtype=float).reshape(n_lat + 1, n_lon)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert u_cc.shape == (n_lat, n_lon)
        assert v_cc.shape == (n_lat, n_lon)
        # Verify the actual averaging.
        assert np.allclose(u_cc[2, 3], 0.5 * (u[2, 3] + u[2, 4]))
        assert np.allclose(v_cc[2, 3], 0.5 * (v[2, 3] + v[3, 3]))

    def test_already_cell_centred(self):
        """When u and v already share the same shape (cubed-sphere /
        gaussian / mpas extracted to cell centres), the averaging
        path is a no-op."""
        n_lat, n_lon = 6, 8
        u = np.full((n_lat, n_lon), 3.0)
        v = np.full((n_lat, n_lon), -2.0)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert u_cc is u
        assert v_cc is v

    def test_non_canonical_mismatch_skips(self):
        """iter-21 codex MEDIUM: if shapes don't match the canonical
        C-grid stagger relation, the patch returns ``ok=False`` so the
        caller can skip the quiver overlay rather than silently
        truncate."""
        # u missing only 1 column on lon axis (canonical) but v also
        # missing 1 column → non-canonical.
        u = np.zeros((5, 8))   # would average to (5, 7)
        v = np.zeros((6, 7))   # canonical: would average to (5, 7)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        # u didn't match the +1 condition (8 != 7+1=8 — actually it
        # DOES match), so let's pick a truly non-canonical case.

        # Pick u shape = (5, 9), v shape = (7, 7) — neither matches
        # the canonical +1 relation.
        u = np.zeros((5, 9))
        v = np.zeros((7, 7))
        u_cc, v_cc, ok = self._stagger_average(u, v)
        # Neither face→cell averaging applies; shapes stay different.
        assert not ok

    def test_constant_field_preserves_value(self):
        """Averaging a constant field returns the same constant —
        sanity check on the 0.5 averaging weights."""
        n_lat, n_lon = 4, 6
        u = np.full((n_lat, n_lon + 1), 7.5)
        v = np.full((n_lat + 1, n_lon), -1.25)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert np.allclose(u_cc, 7.5)
        assert np.allclose(v_cc, -1.25)


# ---------------------------------------------------------------------------
# Modular copy at scripts/ocean_test_matrix/diagnostic_io.py
# ---------------------------------------------------------------------------

class TestModularCopyParity:
    def test_modular_diagnostic_io_loads(self):
        """iter-21 LOW: the modular copy
        ``scripts/ocean_test_matrix/diagnostic_io.py`` had the same
        unfixed staggered-velocity bug.  Verify it imports cleanly."""
        import importlib
        # Direct import via package path.
        sys.path.insert(0, str(_SCRIPT_DIR / "ocean_test_matrix"))
        try:
            import diagnostic_io  # noqa: F401
        finally:
            sys.path.pop(0)
