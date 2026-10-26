"""Smoke test for scripts/validate/land_layered_snow_demo.py (the bulk vs layered
single-column demo quoted in the layered-snowpack PR)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "land_layered_snow_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("land_layered_snow_demo", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forcing_schedule_and_short_run():
    demo = _load()
    spd = int(86400 / demo.DT)
    assert float(demo.forcing_at(2 * spd).precip_snow[0]) > 0.0          # day-2 snowfall
    assert float(demo.forcing_at(25 * spd).precip_total[0]) > 0.0        # day-25 rain
    assert float(demo.forcing_at(25 * spd).precip_snow[0]) == 0.0
    rows, held = demo.run("layered", days=2.0 / spd)
    assert rows.shape == (2, 4) and np.all(np.isfinite(rows)) and held == 0
