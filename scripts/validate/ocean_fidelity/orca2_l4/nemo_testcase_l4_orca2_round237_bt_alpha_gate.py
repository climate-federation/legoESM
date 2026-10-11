#!/usr/bin/env python3
"""Gate Decision 115's required per-card NEMO barotropic filter alpha."""

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

from legoesm.ocean.experiments.dino import dino_config_for_recipe  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_nemo_testcase_card,
    build_overflow_zps_card,
    validate_nemo_testcase_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round235_bt_alpha_replay as r235,
)

PLANTS = ("none", "missing-card", "wrong-orca-alpha", "coefficient-bit")
EXPECTED_WEIGHTS_009 = np.asarray(
    [0.7125337174, 0.2525882038999999, 0.03722244, -0.0023443612999999985],
    dtype=np.float64,
)


class GateError(RuntimeError):
    """The card registry or source-exact interpolation moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _replace_card_alpha(card, alpha):
    """Return the same immutable card with only rn_bt_alpha changed."""

    config = card.recipe.model_config
    return card._replace(
        recipe=card.recipe._replace(
            model_config=config._replace(
                barotropic=config.barotropic._replace(
                    nemo_barotropic_filter_alpha=alpha))))


def validate_card_alpha(card, expected: float) -> None:
    """Require both the structural card gate and its explicit deck value."""

    validate_nemo_testcase_card(card)
    actual = card.recipe.model_config.barotropic.nemo_barotropic_filter_alpha
    require(actual is not None, f"{card.case} has no explicit rn_bt_alpha")
    require(actual == expected,
            f"{card.case} rn_bt_alpha moved: {actual!r} != {expected!r}")


def measure(deck_root: Path, record: Path, namelist: Path, expect_commit: str,
            *, plant: str = "none") -> dict[str, object]:
    """Validate the registry and replay the live OMT-4 statement."""

    require(plant in PLANTS, f"unknown plant {plant!r}")
    cards = {
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card(),
        "OVERFLOW-zps": build_overflow_zps_card(),
        "GYRE-zco": build_gyre_zco_card(),
        "ORCA2-zps": build_nemo_testcase_card("ORCA2-zps", deck_root=deck_root),
        "VORTEX-zco": build_nemo_testcase_card("VORTEX-zco"),
        "VORTEX_VEC-zco": build_nemo_testcase_card("VORTEX_VEC-zco"),
    }
    expected = {
        "LOCK_EXCHANGE-zco": 0.07,
        "OVERFLOW-zps": 0.0,
        "GYRE-zco": 0.07,
        "ORCA2-zps": 0.09,
        "VORTEX-zco": 0.07,
        "VORTEX_VEC-zco": 0.07,
    }
    if plant == "missing-card":
        cards["GYRE-zco"] = _replace_card_alpha(cards["GYRE-zco"], None)
    elif plant == "wrong-orca-alpha":
        cards["ORCA2-zps"] = _replace_card_alpha(cards["ORCA2-zps"], 0.07)
    for name, card in cards.items():
        validate_card_alpha(card, expected[name])

    dino = dino_config_for_recipe("nemo_dino_kamm_mlf")
    require(dino.nemo_barotropic_filter_alpha == 0.0,
            "DINO NEMO card must state inert deck rn_bt_alpha=0.0")

    replay = r235.measure(record, namelist, expect_commit)
    weights = np.asarray(replay["alpha09_weights"], dtype=np.float64)
    if plant == "coefficient-bit":
        weights = weights.copy()
        weights[0] = np.nextafter(weights[0], np.inf)
    require(np.array_equal(weights, EXPECTED_WEIGHTS_009),
            f"production alpha=0.09 weights moved: {weights.tolist()}")
    require(replay["alpha09_vs_nemo"]["unequal"] == 0,
            "production card alpha does not replay eta_pgf bit-exact")
    require(replay["alpha07_vs_nemo"]["unequal"] == 8794,
            "alpha=0.07 known-answer control moved")
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round237-bt-alpha-v1",
        "status": "PASS_R237_BT_ALPHA",
        "worktree": replay["worktree"],
        "registry": {
            **{name: expected[name] for name in sorted(expected)},
            "DINO-nemo_dino_kamm_mlf": 0.0,
            "OMT-0..4-and-ORCA2-rungs": 0.09,
        },
        "alpha09_weights": weights.tolist(),
        "alpha07_vs_nemo": replay["alpha07_vs_nemo"],
        "alpha09_vs_nemo": replay["alpha09_vs_nemo"],
        "source_statement": "dynspg_ts.f90:604-612,1522-1557",
        "predictions": {
            "R237-P1": "CONFIRMED_REQUIRED_REGISTRY",
            "R237-P2": "CONFIRMED_8794_TO_0_BIT_EXACT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--namelist", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.record, args.namelist, args.expect_commit,
            plant=args.plant,
        )
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R237_BT_ALPHA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
