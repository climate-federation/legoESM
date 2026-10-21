#!/usr/bin/env python3
"""Snapshot or compare exact two-step state leaves across the L4b lane merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
ROOTS = {
    "ORCA2G1-flat": PHASE3 / "gridladder_rounds/round1/oracle_g1_r1/smoke_2step",
    "ORCA2G2-zps": PHASE3 / "gridladder_rounds/round10/oracle_g2_r10/smoke_2step",
    "ORCA2G3-ts": PHASE3 / "gridladder_rounds/round21/oracle_g3_r21/p3_a",
    "ORCA2G4-sbc": PHASE3 / "gridladder_rounds/round21/oracle_g4_r21/p6_a",
    "ORCA2G5-ice": PHASE3 / "gridladder_rounds/round31/oracle_g5_r31/twin_a",
    "ORCA2G6-vec": PHASE3 / "gridladder_rounds/round36/oracle_g6_r36/twin_a",
    "ORCA2G7-drg": PHASE3 / "gridladder_rounds/round41/oracle_g7_r41/twin_a",
    "ORCA2G8-ldf": PHASE3 / "gridladder_rounds/round45/oracle_g8_r45/twin_a",
    "ORCA2G9-tadv": PHASE3 / "gridladder_rounds/round46/oracle_g9_r46/twin_a",
    "ORCA2L4-bbc": PHASE3 / "gridladder_rounds/round53/oracle_l4_r53/twin_a",
    "ORCA2L4b-iso-bbc": PHASE3 / "gridladder_rounds/round54/oracle_l4b_r54/twin_a",
    "ORCA2V1-tke": PHASE3 / "gridladder_vmix_rounds/round1/oracle_v1_r1/twin_a",
    "ORCA2V2-evd": PHASE3 / "gridladder_vmix_rounds/round12/oracle_v2_r12/twin_a",
    "ORCA2V3-ddm": PHASE3 / "gridladder_vmix_rounds/round13/oracle_v3_r13/twin_a",
    "ORCA2V4-iwm": PHASE3 / "gridladder_vmix_rounds/round14/oracle_v4_r14/twin_a",
    "ORCA2V5-havtb": PHASE3 / "gridladder_vmix_rounds/round15/oracle_v5_r15/twin_a",
    "ORCA2L1-bbl": PHASE3 / "gridladder_vmix_rounds/round16/oracle_l1_r16/twin_a",
    "ORCA2L2-iso": PHASE3 / "gridladder_vmix_rounds/round17/oracle_l2_r17/twin_a",
    "ORCA2L3-eiv": PHASE3 / "gridladder_vmix_rounds/round20/oracle_l3_r20/twin_a",
}
LANE_A_CASES = tuple(case for case in ROOTS if case.startswith("ORCA2G")) + (
    "ORCA2L4-bbc", "ORCA2L4b-iso-bbc",
)
LANE_B_CASES = tuple(case for case in ROOTS if case.startswith("ORCA2V")) + (
    "ORCA2L1-bbl", "ORCA2L2-iso", "ORCA2L3-eiv",
)


def _activate_repo(repo: Path) -> None:
    """Put one checkout's packages and testcase helpers first on sys.path."""

    packages = sorted((repo / "packages").glob("*"))
    helpers = repo / "scripts/validate/ocean_fidelity/testcases"
    for path in reversed([*packages, helpers, repo]):
        sys.path.insert(0, str(path))


def _digest(value) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def snapshot(repo: Path, case: str) -> dict[str, object]:
    _activate_repo(repo)
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    root = ROOTS[case]
    required = (root / "namelist_cfg", root / "domain_cfg.nc")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"{case}: incomplete admitted deck root {root}")
    card = recipe.build_nemo_testcase_card(case, deck_root=root)
    spec = card.recipe
    model = LatLonCGridOceanModel(
        spec.grid, spec.z_coord, spec.model_config,
        iwm_forcing=spec.iwm_forcing,
    )

    helpers = repo / "scripts/validate/ocean_fidelity/testcases"
    forced = case not in ("ORCA2G1-flat", "ORCA2G2-zps", "ORCA2G3-ts")
    if case == "ORCA2G4-sbc":
        sys.path.insert(0, str(helpers / "nemo_testcase_g4_orca2sbc"))
        import g4_forcing_steps as forcing_reader
    elif forced:
        sys.path.insert(0, str(helpers / "nemo_testcase_g5_orca2ice"))
        import g5_records as forcing_reader

    state = spec.initial_state
    leaves: dict[str, dict[str, object]] = {}
    for kt in (1, 2):
        kwargs = {}
        if forced:
            freshwater, surface = forcing_reader.nemo_supplied_forcing(
                card, root, kt)
            if case.startswith("ORCA2V"):
                surface = surface._replace(
                    ice_mean_thickness_hm_i=forcing_reader.read_v1_hm_i(
                        root, kt))
            kwargs = {"freshwater": freshwater, "surface_forcing": surface}
        state = jax.device_get(model.step(state, card.dt_s, **kwargs))
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            leaves[f"kt{kt}:{jax.tree_util.keystr(path)}"] = _digest(leaf)

    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "format": "nemo-testcase-l4b-merge-two-step-v1",
        "case": case,
        "repo": str(repo.resolve()),
        "git_sha": sha,
        "deck_root": str(root),
        "steps": 2,
        "leaf_count": len(leaves),
        "leaves": leaves,
    }


def compare(reference: Path, candidate: Path, *, plant: bool) -> dict[str, object]:
    ref = json.loads(reference.read_text())
    got = json.loads(candidate.read_text())
    if ref["case"] != got["case"]:
        raise RuntimeError(
            f"case mismatch: {ref['case']!r} != {got['case']!r}")
    if plant:
        first = sorted(got["leaves"])[0]
        digest = got["leaves"][first]["sha256"]
        got["leaves"][first]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    names = sorted(set(ref["leaves"]) | set(got["leaves"]))
    different = [
        name for name in names
        if ref["leaves"].get(name) != got["leaves"].get(name)
    ]
    report = {
        "format": "nemo-testcase-l4b-merge-identity-report-v1",
        "case": ref["case"],
        "reference_git_sha": ref["git_sha"],
        "candidate_git_sha": got["git_sha"],
        "leaf_count": len(names),
        "different_count": len(different),
        "first_differences": different[:8],
        "plant": plant,
        "bit_identical": not different,
    }
    if different:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--repo", type=Path, required=True)
    snap.add_argument("--case", choices=tuple(ROOTS), required=True)
    snap.add_argument("--output", type=Path, required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--reference", type=Path, required=True)
    cmp_.add_argument("--candidate", type=Path, required=True)
    cmp_.add_argument("--output", type=Path)
    cmp_.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            result = snapshot(args.repo, args.case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            result = compare(args.reference, args.candidate, plant=args.plant)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
