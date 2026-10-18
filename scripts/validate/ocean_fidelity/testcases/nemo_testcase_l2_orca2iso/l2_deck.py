#!/usr/bin/env python3
"""Render lane-B L2's namelist: G5 plus ORCA2's resolved namtra_ldf group."""

from __future__ import annotations

import argparse
from pathlib import Path

G5_GROUP = """&namtra_ldf    !   lateral diffusion scheme for tracers
!-----------------------------------------------------------------------
   ln_traldf_OFF   = .true.   !  TSUNAMI
   ln_traldf_lap   = .false.
   ln_traldf_blp   = .false.
   ln_traldf_iso   = .false.
   ln_traldf_triad = .false.
   ln_traldf_msc   = .false.
/"""

L2_GROUP = """&namtra_ldf    !   lateral diffusion scheme for tracers
!-----------------------------------------------------------------------
   ln_traldf_OFF   = .false.
   ln_traldf_lap   = .true.
   ln_traldf_blp   = .false.
   ln_traldf_lev   = .false.
   ln_traldf_hor   = .false.
   ln_traldf_iso   = .true.
   ln_traldf_triad = .false.
   ln_traldf_msc   = .true.
   rn_slpmax       = 0.01
   ln_triad_iso    = .false.
   rn_sw_triad     = 1
   ln_botmix_triad = .false.
   nn_aht_ijk_t    = 20
   rn_Ud            = 0.018
   rn_Ld            = 200.e+3
/"""


def render_text(text: str) -> str:
    """Replace exactly one certified G5 group; refuse drift or ambiguity."""

    if text.count(G5_GROUP) != 1 or L2_GROUP in text:
        raise ValueError("input is not exactly the certified G5 namtra_ldf group")
    return text.replace(G5_GROUP, L2_GROUP)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    args.target.write_text(render_text(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
