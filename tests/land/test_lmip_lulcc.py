"""LULCC option wiring for the LMIP biophysics driver (run_lmip_biophys.resolve_lulcc).

The transient-cover engine keys off the surfdata's number of cover years, so a
run that declares a reconstruction but is handed a static (single-year) surfdata
would silently apply no land-use change.  ``resolve_lulcc`` turns that into a
hard error and reports the transient-cover status; these tests pin that contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "scripts" / "run" / "run_lmip_biophys.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_lmip_biophys", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clm5_static_is_static_cover():
    mod = _load_driver()
    assert mod.resolve_lulcc("clm5", 1, np.array([2000.0])) == "static cover (clm5)"


def test_reconstruction_transient_reports_years():
    mod = _load_driver()
    s = mod.resolve_lulcc("hyde", 10, np.arange(1920, 1930, dtype=float))
    assert "transient LULCC ON" in s and "hyde" in s and "1920-1929" in s


def test_reconstruction_with_static_surfdata_raises():
    """The core guard: declared reconstruction + single-year surfdata = hard error
    (a silent no-LULCC run is exactly what the codebase forbids)."""
    mod = _load_driver()
    with pytest.raises(SystemExit, match="single cover year"):
        mod.resolve_lulcc("hyde", 1, np.array([1920.0]))


def test_unknown_dataset_raises():
    mod = _load_driver()
    with pytest.raises(ValueError, match="unknown land_cover_dataset"):
        mod.resolve_lulcc("bogus", 5, np.arange(5, dtype=float))


def test_clm5_with_transient_surfdata_flags_provenance():
    """clm5 + a multi-year surfdata is allowed (transient cover still applies) but
    the status flags the provenance mismatch rather than hiding it."""
    mod = _load_driver()
    s = mod.resolve_lulcc("clm5", 3, np.array([1850.0, 1900.0, 1950.0]))
    assert "transient LULCC ON" in s and "dataset=clm5" in s
