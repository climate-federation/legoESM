#!/usr/bin/env python3
"""One-step jaxpr purity probe for ORCA2 lane-B cards and flag-off controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import jax
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

CASES = {
    "g5": "ORCA2G5-ice",
    "v1": "ORCA2V1-tke",
    "v2": "ORCA2V2-evd",
    "v3": "ORCA2V3-ddm",
    "v4": "ORCA2V4-iwm",
    "v5": "ORCA2V5-havtb",
    "l1": "ORCA2L1-bbl",
    "l2": "ORCA2L2-iso",
}


def normalize(text: str) -> str:
    return re.sub(r" at 0x[0-9a-f]+", " at 0xADDR", text)


def expected_text(path: Path, *, has_header: bool) -> str:
    text = path.read_text()
    return "\n".join(text.splitlines()[2:]) if has_header else text


def ladder_jaxpr(case: str, deck: Path, record: Path) -> str:
    import g5_records as g5

    card = recipe.build_nemo_testcase_card(CASES[case], deck_root=deck)
    run = card.recipe
    freshwater, forcing = g5.nemo_supplied_forcing(card, record, 1)
    forcing = forcing._replace(ice_mean_thickness_hm_i=g5.read_v1_hm_i(record, 1))
    model = LatLonCGridOceanModel(
        run.grid, run.z_coord, run.model_config, iwm_forcing=run.iwm_forcing)
    return str(jax.make_jaxpr(lambda state: model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=forcing))(
            run.initial_state))


def full_orca2_jaxpr(deck: Path) -> str:
    card = recipe.build_orca2_zps_card(deck)
    run = card.recipe
    model = LatLonCGridOceanModel(
        run.grid, run.z_coord, run.model_config, iwm_forcing=run.iwm_forcing)
    return str(jax.make_jaxpr(lambda state: model.step(state, card.dt_s))(
        run.initial_state))


def dino_jaxpr(repo: Path, output: Path) -> str:
    captured = {}
    original = LatLonCGridOceanModel.step

    def capture(self, state, *args, **kwargs):
        captured["text"] = str(jax.make_jaxpr(
            lambda value: original(self, value, *args, **kwargs))(state))
        raise SystemExit(0)

    LatLonCGridOceanModel.step = capture
    old_argv = sys.argv
    try:
        sys.argv = [
            "run_dino.py", "--config",
            str(repo / "scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml"),
            "--grid", "latlon", "--recipe", "nemo_dino_kamm_mlf",
            "--n-lon", "50", "--nemo-faithful-grid", "--allow-multiyear",
            "--dt", "2700", "--days", "1", "--snapshot-every-days", "1",
            "--output-dir", str(output),
        ]
        sys.path.insert(0, str(repo / "scripts/run"))
        import run_dino
        try:
            run_dino.main()
        except SystemExit as exc:
            if exc.code != 0:
                raise
    finally:
        LatLonCGridOceanModel.step = original
        sys.argv = old_argv
    if "text" not in captured:
        raise RuntimeError("DINO driver never called LatLonCGridOceanModel.step")
    return captured["text"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=(*CASES, "orca2", "dino"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expect", choices=("identical", "different"))
    args = parser.parse_args(argv)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    repo = Path(__file__).resolve().parents[5]

    with tempfile.TemporaryDirectory(prefix=".l2-purity-", dir=repo) as tmp:
        deck = args.deck_root
        if args.case == "l2":
            if deck is None:
                raise ValueError("l2 requires G5 --deck-root")
            target = Path(tmp) / "l2"
            shutil.copytree(deck, target, symlinks=True)
            from l2_deck import render_text
            cfg = target / "namelist_cfg"
            cfg.write_text(render_text(cfg.read_text()))
            deck = target
        if args.case in CASES:
            if deck is None or args.record_root is None:
                raise ValueError("ladder cases require --deck-root and --record-root")
            text = ladder_jaxpr(args.case, deck, args.record_root)
        elif args.case == "orca2":
            if deck is None:
                raise ValueError("orca2 requires --deck-root")
            text = full_orca2_jaxpr(deck)
        else:
            text = dino_jaxpr(repo, Path(tmp) / "dino-output")

    normalized = normalize(text)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    same = None
    if args.expected:
        same = normalized == expected_text(
            args.expected, has_header=args.case in ("orca2", "dino"))
    if args.expected_sha256:
        if same is not None:
            raise ValueError("select only one expected trace or SHA-256")
        same = digest == args.expected_sha256
    result = {
        "case": args.case,
        "jaxpr_lines": len(text.splitlines()),
        "pure_callback": text.count("pure_callback"),
        "sha256": digest,
        "expected_identical": same,
    }
    print(json.dumps(result, sort_keys=True))
    if args.expect and same is not (args.expect == "identical"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
