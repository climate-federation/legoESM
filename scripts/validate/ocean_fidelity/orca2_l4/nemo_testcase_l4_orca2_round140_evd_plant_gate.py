#!/usr/bin/env python3
"""Measure whether rung 0 consumes its stated enhanced-diffusion coefficient."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)


PLANTS = ("none", "selector", "coefficient", "profile")


class GateError(RuntimeError):
    """The EVD effectiveness readout violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "selector":
        report["scheme"] = "none"
    elif plant == "coefficient":
        report["planted_K_conv"] = report["base_K_conv"]
    elif plant == "profile":
        report["profiles_equal"]["avt"] = False

    require(report.get("claim_label") == "independent",
            "EVD plant is not an independent rung-0 readout")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("scheme") == "enhanced_diffusion",
            "rung-0 EVD selector changed")
    require(report.get("base_K_conv") == 100.0,
            "instantiated rung-0 rn_evd/K_conv changed")
    require(report.get("planted_K_conv") == 1.0e6,
            "rn_evd x1e4 plant changed")
    require(report.get("profile_dtype") == {"avt": "float64", "avm": "float64"},
            "EVD profiles are not fp64")
    require(set(report.get("profiles_equal", ())) == {"avt", "avm"},
            "EVD profile registry changed")
    require(all(bool(report["profiles_equal"][name])
                == (int(report["profile_delta"][name]["moved"]) == 0)
                for name in ("avt", "avm")),
            "EVD equality and moved-cell census disagree")
    active = not all(report["profiles_equal"].values())
    report["prediction_ledger"] = {
        "R140-P4": {
            "status": "REFUTED" if active else "CONFIRMED",
            "predicted": "rn_evd x1e4 leaves avt/avm bit-identical",
            "observed": report["profile_delta"],
        }
    }
    return {**report, "evd_active": active,
            "status": "PASS_ROUND140_EVD_EFFECTIVENESS"}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-140 EVD worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-140 EVD commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "EVD plant requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    cfg = card.recipe.model_config
    evd = cfg.physics.convection.enhanced_diffusion
    planted_evd = evd._replace(K_conv=evd.K_conv * 1.0e4)
    planted_cfg = cfg._replace(physics=cfg.physics._replace(
        convection=cfg.physics.convection._replace(
            enhanced_diffusion=planted_evd)))

    def profiles(config):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, config)
        return jax.device_get(model.diagnose_vertical_K(
            card.recipe.initial_state, card.dt_s))

    base_avt, base_avm = profiles(cfg)
    plant_avt, plant_avm = profiles(planted_cfg)
    delta = {}
    equal = {}
    for name, base, candidate in (
            ("avt", base_avt, plant_avt), ("avm", base_avm, plant_avm)):
        left, right = np.asarray(base), np.asarray(candidate)
        equal[name] = bool(np.array_equal(
            np.ascontiguousarray(left).view(np.uint64),
            np.ascontiguousarray(right).view(np.uint64)))
        moved = left.view(np.uint64) != right.view(np.uint64)
        delta[name] = {
            "moved": int(np.count_nonzero(moved)),
            "total": int(left.size),
            "max_abs": float(np.max(np.abs(right - left))),
        }
    return {
        "format": "nemo-testcase-l4-orca2-round140-evd-effectiveness-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "scheme": cfg.physics.convection.scheme,
        "base_K_conv": float(evd.K_conv),
        "planted_K_conv": float(planted_evd.K_conv),
        "profile_dtype": {"avt": str(np.asarray(base_avt).dtype),
                          "avm": str(np.asarray(base_avm).dtype)},
        "profiles_equal": equal,
        "profile_delta": delta,
        "worktree": stamp,
        "compiled_citations": {
            "dispatch": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359",
            "avt_replace": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:94-110",
            "avm_replace": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:120-135",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.expect_commit,
                    "runtime mode requires deck and commit")
            raw = measure(args.deck_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND140_EVD_EFFECTIVENESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
