#!/usr/bin/env python3
"""Render L9 from L8 plus ORCA2's certified L3 EIV namelist group."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


EIV_OFF = """&namtra_eiv    !   eddy induced velocity param.
!-----------------------------------------------------------------------
   ln_ldfeiv   = .false.
/"""

EIV_ON = """&namtra_eiv    !   eddy induced velocity param.
!-----------------------------------------------------------------------
   ln_ldfeiv      = .true.
   nn_aei_ijk_t   = 21
   rn_Ue           = 0.03
   rn_Le           = 200.e+3
   ln_eke_equ      = .false.
/"""


def _l8_module():
    path = Path(__file__).parents[1] / "nemo_testcase_l8_orca2krhs/l8_deck.py"
    spec = importlib.util.spec_from_file_location("l9_l8_deck", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def render_text(g9_text: str) -> str:
    """Apply L8, then replace exactly its disabled EIV block."""

    text = _l8_module().render_text(g9_text)
    if text.count(EIV_OFF) != 1 or EIV_ON in text:
        raise ValueError("L8 input lacks exactly one certified OFF EIV block")
    return text.replace(EIV_OFF, EIV_ON)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    args.target.write_text(render_text(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
