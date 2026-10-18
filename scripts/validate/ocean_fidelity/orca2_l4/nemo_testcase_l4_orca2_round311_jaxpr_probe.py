#!/usr/bin/env python3
"""Round 311: print one model step's jaxpr for a card, from a given tree.

argv: <repo> <case> <out.txt>.  Imports come from <repo> (the caller sets
PYTHONPATH to that tree).  Cases:
  orca2:<rung>    shipped ORCA2 ladder card (rung0, omt0..omt4, rung10)
  orca2full       build_orca2_zps_card on the ORCA2 input deck
  recipe:<CASE>   build_nemo_testcase_card(CASE) (GYRE, VORTEX, ...)
  orca2g:<CASE>   a grid-ladder rung card (ORCA2G1-flat .. ORCA2G8-ldf)
  dino            the DINO driver's first step
Callback addresses are masked; two outputs are compared with cmp.
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

import jax

REPO, CASE, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(REPO))
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
G1_DOMAIN = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
                 "round4/oracle_g1_r4/domaincfg/domain_cfg.nc")
R = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
ROOTS = {
    "omt0": R / "round203/acquisition/orca2_omt0_frames_10step_a_np2",
    "rung0": R / "round90/acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2",
    "omt1": R / "round211/acquisition/orca2_omt1_frames_8step_a_np2",
    "omt2": R / "round218/acquisition/orca2_omt2_frames_10step_a_np2",
    "omt3": R / "round220/acquisition/orca2_omt3_frames_10step_a_np2",
    "omt4": R / "round222/acquisition/orca2_omt4_frames_10step_a_np2",
    "rung10": R / "round5/acquisition/orca1ice_surface_entry_every_step_a_np2"}
G_DIRS = {"ORCA2G1-flat": "g1_orca2grid", "ORCA2G1b-vec": "g1b_orca2vec",
          "ORCA2G2-zps": "g2_orca2bathy", "ORCA2G3-ts": "g3_orca2ts",
          "ORCA2G4-sbc": "g4_orca2sbc", "ORCA2G5-ice": "g5_orca2ice",
          "ORCA2G6-vec": "g6_orca2vec", "ORCA2G7-drg": "g7_orca2drg",
          "ORCA2G8-ldf": "g8_orca2ldf"}

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.fidelity import nemo_testcase_recipe as rec  # noqa: E402


def _write(text, cfg=None):
    flags = ("" if cfg is None else
             f"stage1_order={cfg.rk3_stage1_ldf_source_order} "
             f"vector_order={cfg.rk3_vector_rhs_nemo_order} "
             f"momentum_advection={cfg.momentum_advection}\n")
    OUT.write_text(flags + re.sub(r" at 0x[0-9a-f]+", " at 0xADDR", text))
    print(f"{CASE}: {len(text.splitlines())} jaxpr lines -> {OUT}")


def _step_jaxpr(card):
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=getattr(r, "iwm_forcing", None))
    _write(str(jax.make_jaxpr(lambda s: model.step(s, card.dt_s))(r.initial_state)),
           r.model_config)


def _g_root(case):
    rung = REPO / "scripts/validate/ocean_fidelity/testcases" / f"nemo_testcase_{G_DIRS[case]}"
    root = Path(tempfile.mkdtemp(prefix="r311_g_"))
    for path in rung.iterdir():
        if path.name.startswith("namelist"):
            shutil.copy(path, root / path.name)
    for path in INPUTS.iterdir():
        if not (root / path.name).exists():
            (root / path.name).symlink_to(path)
    domain = G1_DOMAIN if case in ("ORCA2G1-flat", "ORCA2G1b-vec") else INPUTS / "ORCA_R2_zps_domcfg.nc"
    (root / "domain_cfg.nc").symlink_to(domain)
    return root


if CASE.startswith("orca2:"):
    rung = CASE.split(":", 1)[1]
    if rung == "rung10":
        from scripts.validate.ocean_fidelity.orca2_l4 import (
            nemo_testcase_l4_orca2_round1_ladder_gate as rung10)
        card = rung10.card_fields(ROOTS[rung])[1]
    else:
        from scripts.validate.ocean_fidelity.orca2_l4 import (
            nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior)
        card = prior._selected_card(rung, ROOTS[rung])
    _step_jaxpr(card)
elif CASE == "orca2full":
    _step_jaxpr(rec.build_orca2_zps_card(INPUTS))
elif CASE.startswith("recipe:"):
    _step_jaxpr(rec.build_nemo_testcase_card(CASE.split(":", 1)[1]))
elif CASE.startswith("orca2g:"):
    case = CASE.split(":", 1)[1]
    _step_jaxpr(rec.build_nemo_testcase_card(case, deck_root=_g_root(case)))
elif CASE == "dino":
    orig = LatLonCGridOceanModel.step

    def _capture(self, state, *args, **kwargs):
        _write(str(jax.make_jaxpr(lambda s: orig(self, s, *args, **kwargs))(state)),
               self.config)
        raise SystemExit(0)

    LatLonCGridOceanModel.step = _capture
    sys.argv = ["run_dino.py", "--config",
                str(REPO / "scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml"),
                "--grid", "latlon", "--recipe", "nemo_dino_kamm_mlf",
                "--n-lon", "50", "--nemo-faithful-grid", "--allow-multiyear",
                "--dt", "2700", "--days", "1", "--snapshot-every-days", "1",
                "--output-dir", str(OUT.parent / f"dino_run_{OUT.stem}")]
    sys.path.insert(0, str(REPO / "scripts/run"))
    import run_dino
    run_dino.main()
    raise SystemExit("DINO driver never called LatLonCGridOceanModel.step")
else:
    raise SystemExit(f"unknown case {CASE!r}")
