#!/usr/bin/env python
"""Compatibility wrapper: delegates to scripts/legacy/run_atmosphere_25deg_ssp45_full.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "legacy" / "run_atmosphere_25deg_ssp45_full.py"
    runpy.run_path(str(target), run_name="__main__")
