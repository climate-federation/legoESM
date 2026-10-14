#!/usr/bin/env python3
"""Round 278: Decision-122 landing census for ladder-certified statements.

Measurement and row classification are round 277's (itself round 276's
ladder and classifier).  This adds one arm per candidate unit and the
``--decide`` mode that applies Decision 122 (c) and (d) to a unit's census
files: no exact row lost, and the first over-bar row unchanged or toward,
on every required card/label.  The toward/away census is registered only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round277_rung_census_gate as r277,
)

r276 = r277.r276
GateError = r276.GateError
require = r276.require

# Ladder-certified units whose cited line ORCA2 executes.  G1's EEN
# vorticity order is excluded: its certified zwz line is the np_COR case
# (ORCA2G1 dynvor.f90:727), while ORCA2 vector form runs np_CRV.
UNITS: dict[str, tuple[str, ...]] = {
    "u_g2_momentum_divisors": ("zdf_momentum_nemo_face_thickness",),
    "u_g2_tracer_diagonal": ("zdf_tracer_nemo_rounded_thickness",),
    "u_g2_rhs_ascending_sum": ("barotropic_rhs_ascending_level_sum",),
    "u_g3_divisors_stage_r3u": (
        "zdf_momentum_nemo_face_thickness", r277.STAGE_R3U),
    "u_een_fold_pair": ("barotropic_een_ratio_fold_exchange",
                        "barotropic_een_fe3mask_pivot_row"),
    "u_een_ratio_written_order": ("barotropic_een_ratio_written_order",),
    "u_stage1_rhs": ("rk3_stage1_rhs_nemo_literal",),
    "u_stage_velocity": ("rk3_stage_velocity_written_order",),
    "u_zdf_barotropic_removal": ("zdf_barotropic_removal_nemo_literal",),
    "u_fold_exchange_top_v_row": ("rk3_fold_exchange_top_v_row",),
    "u_stage_r3f": ("rk3_stage_r3f_nemo_literal",),
    "u_stage_r3t": ("rk3_stage_r3t_nemo_literal",),
    "u_stored_r3t_product": ("nemo_r3t_stored_product",),
    "u_eos_gdept_product": ("nemo_eos_gdept_rounded_product",),
    "u_stage_r3uv": ("rk3_stage_r3uv_nemo_literal",),
}
# Units that met (c)+(d) singly, less the top-V fold exchange, whose ten-step
# SSH maximum grows ~20x on rung 0 and is held for a user decision.
HELD = ("u_fold_exchange_top_v_row",)
FINAL_SET = tuple(sorted({name for unit, names in UNITS.items()
                          if unit not in HELD for name in names}))
r276.ARMS.update(UNITS)
r276.ARMS["d122_final_set"] = FINAL_SET

CARDS = (("rung0", "independent"), ("omt1", "independent"),
         ("omt2", "independent"), ("omt3", "independent"),
         ("omt4", "independent"), ("rung10", "given_nemo_entry"),
         ("rung10", "independent"))


def decide(census_paths: list[Path]) -> dict:
    """Decision 122 (c)+(d) over one arm's seven card/label censuses."""

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in census_paths]
    require(all(d.get("status") == "PASS_R276_EARLY_LATE_CENSUS"
                for d in docs), "a census is not admitted")
    arms = {d["arm"] for d in docs}
    require(len(arms) == 1, f"censuses mix arms: {sorted(arms)}")
    commits = {d["worktree_commit"] for d in docs}
    require(len(commits) == 1, "censuses come from different trees")
    got = sorted((d["rung"], d["label"]) for d in docs)
    require(got == sorted(CARDS), f"card/label set is not Decision 122's: {got}")
    cards = []
    for d in sorted(docs, key=lambda d: CARDS.index((d["rung"], d["label"]))):
        c = d["rms_direction_census"]
        lost = d["exact_row_losses"]
        first = d["first_over_bar"]["rms_direction"]
        cards.append({
            "rung": d["rung"], "label": d["label"],
            "exact_rows_lost": len(lost),
            "first_over_bar": d["first_over_bar"],
            "c_no_exact_loss": not lost,
            "d_first_over_bar_ok": first in ("toward", "equal"),
            "registered_census": {
                k: [c[k]["toward"], c[k]["away"], c[k]["equal"]]
                for k in ("all", "early_kt1_3", "late_kt4_plus")},
            "first_away_kt": d["first_away_kt"],
            "final_stage3_ssh_max": d["final_stage3_ssh_max"],
        })
    return {
        "format": "nemo-testcase-l4-orca2-round278-d122-decision-v1",
        "status": "PASS_R278_D122_DECISION",
        "arm": arms.pop(), "worktree_commit": commits.pop(),
        "cards": cards,
        "d122_c_and_d": all(c["c_no_exact_loss"] and c["d_first_over_bar_ok"]
                            for c in cards),
    }


def main() -> int:
    if "--decide" in sys.argv:
        args = sys.argv[sys.argv.index("--decide") + 1:]
        require("--output" in args, "--decide needs --output")
        out = Path(args[args.index("--output") + 1])
        paths = [Path(a) for a in args[:args.index("--output")]]
        try:
            result = decide(paths)
        except (GateError, OSError, KeyError, ValueError) as error:
            print(f"STATUS REFUSE: {error}")
            return 2
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"STATUS {result['status']} d122_c_and_d="
              f"{result['d122_c_and_d']}")
        return 0
    return r277.main()


if __name__ == "__main__":
    raise SystemExit(main())
