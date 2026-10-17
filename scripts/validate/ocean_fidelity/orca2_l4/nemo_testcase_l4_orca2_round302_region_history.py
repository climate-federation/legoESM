#!/usr/bin/env python3
"""Per-kt, per-stage, per-region error history of the OMT-4 free run.

Scores the card's completed stage states (``s0`` = step entry, ``s1``/``s2``
from separately compiled exposing models, ``s3`` = the ordinary step) against
round 222's admitted NEMO stage frames, fold band (last three T rows) and
interior separately, for T, S, u, v and ssh on NEMO's own wet masks.

Arms: ``card`` (the OMT-4 card, no private hooks) and ``unit`` (rounds
298-301's run: round 249's private atomic fold unit plus round 245's
barotropic surface-mask arm).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round298_stage7_writer as r298,
    nemo_testcase_l4_orca2_round300_w_leaf_split as r300,
    nemo_testcase_l4_orca2_round301_u_transport_factors as r301,
)

FIELDS = ("u", "v", "ssh", "T", "S")
MASK_OF = {"u": "umask", "v": "vmask", "T": "tmask", "S": "tmask", "ssh": "tmask"}
ARMS = ("card", "unit")
PLANTS = ("none", "entry-bit")
BAR = 1e-15
EXIT_REFUSE, EXIT_PLANT_GREEN = 2, 3
GateError = r301.GateError
require = r301.require


def nemo_masks(record_root: Path) -> dict[str, np.ndarray]:
    masks = r301.read_mesh(record_root, ("tmask", "umask", "vmask"))
    return {f: (masks[MASK_OF[f]][..., 0] if f == "ssh" else masks[MASK_OF[f]]) > 0.0
            for f in FIELDS}


def _region(candidate, oracle, support) -> dict[str, object]:
    row = r298._score(candidate, oracle, support)
    signal = np.asarray(oracle, np.float64)[np.asarray(support, bool)]
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal * signal)))
    row["oracle_max"], row["oracle_rms"] = peak, rms
    row["relative_rms"] = (row["rms"] / rms if rms > 0.0
                           else float("inf") if row["rms"] > 0.0 else 0.0)
    row["over_bar"] = bool(row["max_abs"] > BAR * peak) if peak > 0.0 else row["max_abs"] > 0.0
    return row


def score_stage(candidate: dict, oracle: dict, masks: dict) -> dict[str, dict]:
    """``{field: {fold_band, interior}}``; fold band = last three T rows."""
    out = {}
    for field in FIELDS:
        c, o, m = (np.asarray(candidate[field], np.float64),
                   np.asarray(oracle[field], np.float64), masks[field])
        require(c.shape == o.shape == m.shape, f"{field}: shape moved")
        out[field] = {"interior": _region(c[:-3], o[:-3], m[:-3]),
                      "fold_band": _region(c[-3:], o[-3:], m[-3:])}
    return out


def first_over_bar(history: list[dict]) -> dict[str, dict]:
    """Earliest (kt, stage) over the bar, per field and region."""
    out: dict[str, dict] = {}
    for field in FIELDS:
        for region in ("fold_band", "interior"):
            out.setdefault(field, {})[region] = next(
                ({"kt": h["kt"], "stage": h["stage"]} for h in history
                 if h["fields"][field][region]["over_bar"]), None)
    return out


def _models(card, arm: str):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )

    def hooks(stage):
        if arm == "unit":
            return r300.r249._hooks(card, True, stage=stage)
        return (_NEMOWSRK3TestHooks(expose_momentum_stage=stage,
                                    expose_tracer_stage=stage) if stage else None)

    return tuple(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks(stage)) for stage in (1, 2, 0))


def run(card, record_root: Path, arm: str, kts: int = 7,
        plant: str = "none") -> dict[str, object]:
    """Free run from the card's initial state; refuses unless kt-1 entry is exact."""
    import jax

    require(arm in ARMS and plant in PLANTS, f"unknown arm/plant {arm}/{plant}")
    fields = r300.omt4.rung0.candidate_fields
    frame = r300.omt4.rung0.assemble_frame
    masks = nemo_masks(record_root)
    state = card.recipe.initial_state
    entry = frame(record_root, 1, 0)
    if plant == "entry-bit":
        entry["u"][147, 47, 0] = np.nextafter(entry["u"][147, 47, 0], np.inf)
    for name, value in fields(state).items():
        require(np.array_equal(np.asarray(value, np.float64).view(np.uint64),
                               entry[name].view(np.uint64)),
                f"kt-1 entry {name} is not NEMO's: the labels are not one run")
    forcing = r300.omt0.rung0_ladder._zero_forcing((148, 180))
    models = _models(card, arm)
    arm_context = (r300.r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
        if arm == "unit" else contextlib.nullcontext())
    history = []
    with arm_context:
        for kt in range(1, kts + 1):
            history.append({"kt": kt, "stage": 0, "fields": score_stage(
                fields(state), frame(record_root, kt, 0), masks)})
            outs = [jax.device_get(m.step(state, card.dt_s, freshwater=forcing[0],
                                          surface_forcing=forcing[1]))
                    for m in models]
            for stage, out in enumerate(outs, start=1):
                history.append({"kt": kt, "stage": stage, "fields": score_stage(
                    fields(out), frame(record_root, kt, stage), masks)})
            print(f"PROGRESS {arm} kt={kt}", file=sys.stderr, flush=True)
            state = outs[-1]
    return {"arm": arm, "case": card.case, "history": history,
            "first_over_bar": first_over_bar(history)}


def table(report: dict, field: str = "u") -> list[str]:
    lines = []
    for h in report["history"]:
        f, i = h["fields"][field]["fold_band"], h["fields"][field]["interior"]
        lines.append(f"kt{h['kt']} s{h['stage']} {field}: fold {f['unequal']}/{f['support']} "
                     f"rel {f['relative_rms']:.4g} max {f['max_abs']:.4g} | interior "
                     f"{i['unequal']}/{i['support']} rel {i['relative_rms']:.4g} "
                     f"max {i['max_abs']:.4g}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="card")
    parser.add_argument("--kts", type=int, default=7)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import jax
        from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

        stamp = worktree_stamp()
        require(stamp["clean"] and stamp["commit"].lower() == args.expect_commit.lower(),
                "measurement requires its clean committed tree")
        policy = PrecisionPolicy.fp64(transcendentals="libm")
        set_policy(policy)
        require(get_policy() == policy and bool(jax.config.jax_enable_x64)
                and jax.default_backend() == "cpu", "fp64/libm CPU not active")
        card = r300.omt4.build_omt4_card(args.deck_root)
        r300.omt4.validate_omt4_card(args.deck_root, card)
        report = run(card, args.deck_root, args.arm, args.kts, args.plant)
    except (GateError, OSError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return EXIT_REFUSE
    if args.plant != "none":
        print(f"STATUS PLANT-GREEN {args.plant}: the plant did not refuse")
        return EXIT_PLANT_GREEN
    report["worktree"] = stamp
    if args.output:
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for field in FIELDS:
        print("\n".join(table(report, field)))
    print("FIRST_OVER_BAR", json.dumps(report["first_over_bar"], sort_keys=True))
    print(f"STATUS MEASURED {args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
