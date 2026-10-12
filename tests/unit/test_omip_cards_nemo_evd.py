"""The three ORCA1-faithful production cards (tripole, MPAS, FESOM) run
NEMO's zdfevd with one identical flag set (user decision 2026-10-12), and that
set builds NEMO's configuration: rn_evd = 100, replace, bn2 trigger at
-1e-12, tracers only (nn_evdm = 0)."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    build_enhanced_diffusion_config,
    validate_fesom_stage,
)

CARDS = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "omip_nemo"
NAMES = ("orca1_faithful_card.sbatch", "_mpas_orca1faithtint_d30.sbatch",
         "_fesom_orca1faithtint_d30.sbatch")


def _evd_tokens(name):
    lines = [ln for ln in (CARDS / name).read_text().splitlines()
             if "--convection " in ln and not ln.lstrip().startswith("#")]
    assert len(lines) == 1, f"{name}: expected one EVD flag line, got {len(lines)}"
    return shlex.split(re.sub(r"`#.*?`", "", lines[0]).rstrip(" \\"))


def test_all_three_cards_carry_the_same_evd_flags():
    sets = {n: _evd_tokens(n) for n in NAMES}
    assert len({tuple(v) for v in sets.values()}) == 1, sets


@pytest.mark.parametrize("grid", ["tripole", "mpas", "fesom"])
def test_the_flags_build_nemos_zdfevd(grid):
    p = _build_arg_parser()
    lane = (["--fesom-mesh-dir", "/f"] if grid == "fesom" else ["--mesh", "/m.nc"])
    args = p.parse_args(["--grid", grid, "--output", "/o"] + lane
                        + _evd_tokens(NAMES[0]))
    cfg = build_enhanced_diffusion_config(args)
    assert (cfg.K_conv, cfg.K_bg, cfg.nu_conv) == (100.0, 0.0, 0.0)
    assert cfg.n2_mode == "nemo_bn2" and cfg.n2_eos_form == "teos10"
    assert cfg.smooth_transition is False and cfg.n2_threshold == -1e-12
    assert cfg.evd_composition == "nemo_replace"
    if grid == "fesom":
        validate_fesom_stage(args, p)   # allowlist accepts every EVD flag
