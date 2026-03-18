#!/usr/bin/env python
"""Compatibility wrapper: delegates to scripts/legacy/run_dcmip_transport.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "legacy" / "run_dcmip_transport.py"
    runpy.run_path(str(target), run_name="__main__")
