#!/usr/bin/env python3
"""Render L6 as the certified L4b deck plus exactly ORCA2's damping block."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


L4B_DMP = """&namtra_dmp    !   tracer: T & S newtonian damping
!-----------------------------------------------------------------------
   ln_tradmp   = .false.
/"""

L6_DMP = """&namtra_dmp    !   tracer: T & S newtonian damping
!-----------------------------------------------------------------------
   ln_tradmp   =  .true.   !  add a damping term (using resto.nc coef.)
      nn_zdmp  =    0         !  vertical shape =0    damping throughout the water column
/"""


def _l4b_module():
    path = Path(__file__).parents[1] / "nemo_testcase_l4b_orca2iso_bbc/l4b_deck.py"
    spec = importlib.util.spec_from_file_location("l6_l4b_deck", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def render_text(g5_text: str) -> str:
    """Apply certified L4b, then replace one exact damping block."""

    l4b_text = _l4b_module().render_text(g5_text)
    if l4b_text.count(L4B_DMP) != 1 or L6_DMP in l4b_text:
        raise ValueError("L4b input lacks exactly one certified OFF damping block")
    return l4b_text.replace(L4B_DMP, L6_DMP)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    args.target.write_text(render_text(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
