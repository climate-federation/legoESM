"""Direct test of scripts/validate/land_snow_compaction_check.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "land_snow_compaction_check.py"


def _load():
    sys.path.insert(0, str(_PATH.parent))
    spec = importlib.util.spec_from_file_location("land_snow_compaction_check", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_layered_runs_a_short_trajectory():
    """The diagnostic's land run (freeze/thaw off, as the demo) runs."""
    m = _load()
    cfg, rec = m.run_layered(days=2.0 * m.demo.DT / 86400.0)
    assert not cfg.thermal.enable_freeze_thaw
    assert rec.shape[:2] == (2, 4) and np.all(np.isfinite(rec))

