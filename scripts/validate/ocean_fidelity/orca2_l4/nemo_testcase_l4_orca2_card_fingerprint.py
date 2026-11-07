#!/usr/bin/env python3
"""Print the resolved-configuration fingerprint of every NEMO card on this lane.

One number per card, so a merge or a stated field can be shown to move -- or
not to move -- a card's resolved configuration before any trajectory is run.

The digest itself is NOT re-derived here: it is the one the certified-card
test already uses (``tests/ocean/unit/test_nemo_vortex_card.py::_card_digest``),
which hashes the printed model configuration together with the initial state,
the grid's Coriolis fields, the vertical coordinate, the land mask and the
card's own deck fields.  Re-deriving it would let the two drift apart.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ORCA2_DECK = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")


def _digest_helper():
    helper = REPO_ROOT / "tests" / "ocean" / "unit" / "test_nemo_vortex_card.py"
    if not helper.is_file():                    # a tree older than that test
        helper = Path(__file__).resolve().with_name("_card_digest_helper.py")
    spec = importlib.util.spec_from_file_location("_vortex_card_tests", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._card_digest


def fingerprints(deck: Path) -> dict[str, str]:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        build_orca2_zps_card,
    )

    card_digest = _digest_helper()
    rows: dict[str, str] = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps",
                 "VORTEX-zco", "VORTEX_VEC-zco"):
        # A card a given tree does not carry is reported as absent rather than
        # skipped: run on two trees, the comparison must say which cards only
        # one of them has.
        try:
            rows[case] = card_digest(build_nemo_testcase_card(case))
        except (ValueError, KeyError, TypeError) as error:
            rows[case] = f"CARD-ABSENT: {type(error).__name__}"
    rows["ORCA2-zps"] = (
        card_digest(build_orca2_zps_card(deck)) if deck.exists()
        else "DECK-ABSENT")
    return rows


def flat_config(config, prefix: str = "") -> dict[str, str]:
    """Every leaf of a resolved card configuration, as dotted path -> repr.

    A digest says THAT a card moved; this says WHICH field, which is the only
    form in which a moved digest may be re-pinned.
    """
    rows: dict[str, str] = {}
    fields = getattr(config, "_fields", None)
    if fields is None:
        rows[prefix.rstrip(".")] = repr(config)
        return rows
    for name in fields:
        value = getattr(config, name)
        if getattr(value, "_fields", None) is not None:
            rows.update(flat_config(value, f"{prefix}{name}."))
        else:
            rows[f"{prefix}{name}"] = repr(value)
    return rows


def config_dump(deck: Path) -> dict[str, dict[str, str]]:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        build_orca2_zps_card,
    )

    out: dict[str, dict[str, str]] = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps",
                 "VORTEX-zco", "VORTEX_VEC-zco"):
        try:
            card = build_nemo_testcase_card(case)
        except (ValueError, KeyError, TypeError):
            continue
        out[case] = flat_config(card.recipe.model_config)
    if deck.exists():
        out["ORCA2-zps"] = flat_config(
            build_orca2_zps_card(deck).recipe.model_config)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, default=ORCA2_DECK)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config-dump", action="store_true",
        help="dump every resolved config leaf instead of the digests")
    args = parser.parse_args(argv)
    if args.config_dump:
        text = json.dumps(config_dump(args.deck_root), indent=1,
                          sort_keys=True)
        if args.output is not None:
            args.output.write_text(text + "\n")
        else:
            print(text)
        return 0
    rows = fingerprints(args.deck_root)
    text = json.dumps(rows, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
