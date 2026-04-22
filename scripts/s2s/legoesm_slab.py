#!/usr/bin/env python
"""Unified CLI for the initialized pure-physics legoESM slab workflow."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ml.s2s.legoesm_slab.cli import main


if __name__ == "__main__":
    main()
