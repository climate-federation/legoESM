#!/usr/bin/env python3
"""Render L8 from G9 plus the certified L1/L2/L4b/L6 deck changes."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


BBL_OFF = """&nambbl        !   bottom boundary layer scheme
!-----------------------------------------------------------------------
   ln_trabbl   = .false.
/"""

BBL_ON = """&nambbl        !   bottom boundary layer scheme
!-----------------------------------------------------------------------
   ln_trabbl   = .true.
   nn_bbl_ldf  = 1
   nn_bbl_adv  = 0
   rn_ahtbbl   = 1000.0
   rn_gambbl   = 10.0
/"""


def _l6_module():
    path = Path(__file__).parents[1] / "nemo_testcase_l6_orca2dmp/l6_deck.py"
    spec = importlib.util.spec_from_file_location("l8_l6_deck", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def render_text(g9_text: str) -> str:
    """Apply L2/L4b/L6, then replace exactly G9's OFF BBL block."""

    text = _l6_module().render_text(g9_text)
    if text.count(BBL_OFF) != 1 or BBL_ON in text:
        raise ValueError("G9/L6 input lacks exactly one certified OFF BBL block")
    return text.replace(BBL_OFF, BBL_ON)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    args.target.write_text(render_text(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
