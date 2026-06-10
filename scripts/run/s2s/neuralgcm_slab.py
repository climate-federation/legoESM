#!/usr/bin/env python
"""CLI entrypoint for NeuralGCM slab-ocean experiments."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legoesm.ml.s2s.neuralgcm_slab.cli import main


if __name__ == "__main__":
    main()
