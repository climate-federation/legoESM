#!/usr/bin/env python3
"""Render L4b as the certified L2 deck plus exactly L4's BBC block."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


G5_BBC = """&nambbc        !   bottom temperature boundary condition
!-----------------------------------------------------------------------
   ln_trabbc   = .false.
/"""

L4_BBC = """&nambbc        !   bottom temperature boundary condition
!-----------------------------------------------------------------------
   ln_trabbc   = .true.
   nn_geoflx   = 2
   cn_dir      = './'
   sn_qgh      = 'geothermal_heating.nc', -12., 'heatflow', .false., .true., 'yearly', '', '', ''
/"""


def _l2_module():
    path = Path(__file__).parents[1] / "nemo_testcase_l2_orca2iso/l2_deck.py"
    spec = importlib.util.spec_from_file_location("l4b_l2_deck", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def render_text(g5_text: str) -> str:
    """Apply the certified L2 renderer, then replace one exact BBC block."""

    l2_text = _l2_module().render_text(g5_text)
    if l2_text.count(G5_BBC) != 1 or L4_BBC in l2_text:
        raise ValueError("L2 input does not contain exactly the certified G5 BBC block")
    return l2_text.replace(G5_BBC, L4_BBC)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    args.target.write_text(render_text(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
