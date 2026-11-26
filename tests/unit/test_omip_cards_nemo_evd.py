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



@pytest.mark.parametrize("grid", ["cubed_sphere", "latlon_bathy"])
def test_grids_that_do_not_consume_evd_refuse_it(grid, monkeypatch):
    """cube / lat-lon builders never receive the convection config: main()
    must refuse --convection enhanced_diffusion even without trigger flags
    (it used to be accepted and silently dropped)."""
    import sys
    from scripts.run import run_omip_core2
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", grid,
                                      "--convection", "enhanced_diffusion"])
    with pytest.raises(SystemExit, match="enhanced_diffusion"):
        run_omip_core2.main()


def _sh2_tokens(name):
    lines = [ln for ln in (CARDS / name).read_text().splitlines()
             if "--tke-prandtl-mode" in ln and not ln.lstrip().startswith("#")]
    assert len(lines) == 1, f"{name}: expected one zdf_sh2/Prandtl line, got {len(lines)}"
    return shlex.split(re.sub(r"`#.*?`", "", lines[0]).rstrip(" \\"))


@pytest.mark.parametrize("name,grid", zip(NAMES, ("tripole", "mpas", "fesom")))
def test_every_card_runs_nemo_zdf_sh2_and_prandtl(name, grid):
    """User decision 2026-10-12: NEMO's neighbour-summed avm shear production
    and nn_pdl=1 Prandtl on all three grids; the tripole additionally needs its
    face-native RK3 geometry, the unstructured lanes supply their own."""
    from scripts.run.run_omip_core2 import build_tripole_vmix_config
    toks = _sh2_tokens(name)
    p = _build_arg_parser()
    lane = (["--fesom-mesh-dir", "/f"] if grid == "fesom" else ["--mesh", "/m.nc"])
    args = p.parse_args(["--grid", grid, "--output", "/o"] + lane + toks)
    tke = build_tripole_vmix_config(
        "tke", tke_shear_production=args.tke_shear_production,
        tke_shear_avm_weighting=args.tke_shear_avm_weighting,
        tke_prandtl_mode=args.tke_prandtl_mode).tke
    assert tke.tke_shear_avm_weighting == "nemo_face"
    assert tke.prandtl_mode == "nemo_ri"
    assert tke.tke_shear_production == (
        "nemo_face_native_nbb2" if grid == "tripole" else "squared_centered")
    if grid == "fesom":
        validate_fesom_stage(args, p)
