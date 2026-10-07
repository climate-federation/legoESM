#!/usr/bin/env python3
"""Round 210 / VORTEX: legoESM vs NEMO over NEMO's shipped run length.

Decision 87 / note CB (2026-10-03): the VORTEX bit-walk is stopped at its
milestones.  This round asks a different question -- not "where is the first
bit debt" but "what does that debt become over NEMO's own shipped run
length" (``nn_itend = 3000`` steps of ``rn_Dt = 2880`` s = 100 days), on both
30 km cards (flux-form UP3 and vector-invariant).  No physics or option
change: the NEMO runs use every namelist value the shipped deck pins, and
legoESM runs the certified card unmodified.

Two reused pieces, not re-derived (pre-impl search: grepped
``scripts/validate/ocean_fidelity/testcases`` for "year", "daily",
"snapshot" per the round order):

* ``nemo_testcase_phase3_trajectory_gate.lego_fields`` / ``expected_masks`` --
  the SAME field extraction and wet-face masks the certified kt=1..10 ladder
  uses for these two cards (``run()`` in that module), so the T/u/v/ssh
  convention here is identical to the one the registries already certify.
* NEMO's own restart machinery (``nn_stock``), the same mechanism
  ``nemo_testcase_l2_gyre_year_fromrest.py`` uses for GYRE's 360-day member
  runs (``_load_nemo``) -- read here with the identical variable names
  (``tn``/``sn``/``un``/``vn``/``sshn``) and axis contract
  (time_counter, nav_lev, y, x) -> transpose(1, 2, 0).

A sanity check runs first: the SAME two NEMO binaries, run from this script's
namelist, also fire the existing oracle_step_entry writer for kt=1..60 (an
unconditional write in stprk3.F90, never edited for this round), so the
trajectory gate's own ``run()`` is invoked against this round's oracle_root
at max_step=10 and required to reproduce the certified round-207/208
registries row for row -- the day-0.33 point (kt=10, i.e. 9 completed
model.step calls) is exactly that ladder's kt=10 row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    expected_masks,
    lego_fields,
    require,
    run as trajectory_run,
)

STEPS_PER_DAY = 30
N_DAYS = 100
TABLE_DAYS = (1, 2, 5, 10, 20, 30, 60, 100)
CARDS = {
    "flux": ("VORTEX-zco",
             Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round210/"
                  "nemo_run_flux")),
    "vec": ("VORTEX_VEC-zco",
            Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round210/"
                 "nemo_run_vec")),
}
CERTIFIED_LADDER = {
    "flux": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round207/"
                 "card_VORTEX-zco.json"),
    "vec": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round207/"
                "card_VORTEX_VEC-zco.json"),
}
# ROUND 216 / VORTEX_SMT round 6.  The seamount pair, scored by the SAME
# definitions: the same field extraction, the same wet masks (taken from the
# SMT card's own geometry through expected_masks, so the mask is the
# seamount's), the same NEMO restart reader and the same table days.  NEMO's
# 100-day seamount runs are round 3's admitted ones.  Their kt=1..10 sanity
# reference is this round's OWN registry, because this round moves those rows:
# a pre-move reference would have to fail.
SMT_CARDS = {
    "smtflux": ("VORTEX_SMT-zps",
                Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                     "vortex_smt/round3/VORTEX_SMT_R3_OMIP_L1_P3/day100")),
    "smtvec": ("VORTEX_SMT_VEC-zps",
               Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                    "vortex_smt/round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/"
                    "day100")),
}
SMT_LADDER = {
    "smtflux": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                    "vortex_smt/round6/after2/after_VORTEX_SMT-zps.json"),
    "smtvec": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                   "vortex_smt/round6/after2/after_VORTEX_SMT_VEC-zps.json"),
}
CARDS.update(SMT_CARDS)
CERTIFIED_LADDER.update(SMT_LADDER)

# Round 226 / mini-ladder SMT-3.  This is the same vector seamount card with
# only ORCA2 rung-0's tracer-LDF module enabled; its admitted NEMO run and the
# round's own ten-step registry use the same reader/scorer as every card above.
CARDS["smt3"] = (
    "VORTEX_SMT3_VEC-zps",
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/"
         "oracle_vortex_smt3/day100"),
)
CERTIFIED_LADDER["smt3"] = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round226/"
    "smt3_ladder.json")

CARDS["smt4"] = (
    "VORTEX_SMT4_VEC-zps",
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
         "oracle_vortex_smt4/day100"),
)
CERTIFIED_LADDER["smt4"] = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round238/"
    "smt4_ladder.json")

# Round 241 / mini-ladder SMT-1 deferred 100-day comparison.  The NEMO
# trajectory is acquired into this round's evidence root by the existing
# hash-checked ``smt1vec100d`` driver arm.  The short-run reference is the
# latest production-tree census (round 237); rounds 238-240 changed no model
# code, so this is also the current certified SMT-1 registry.
CARDS["smt1"] = (
    "VORTEX_SMT1_VEC-zps",
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/"
         "oracle_vortex_smt1/day100"),
)
CERTIFIED_LADDER["smt1"] = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/inert/"
    "VORTEX_SMT1_VEC-zps.json")

# Round 243 / mini-ladder SMT-2 deferred 100-day comparison.  The NEMO
# trajectory is acquired into this round's evidence root by the existing
# hash-checked ``smt2vec100d`` driver arm.  Round 237's inert-card census is
# the latest SMT-2 registry; rounds 238-242 changed no production model code.
CARDS["smt2"] = (
    "VORTEX_SMT2_VEC-zps",
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/"
         "oracle_vortex_smt2/day100"),
)
CERTIFIED_LADDER["smt2"] = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/inert/"
    "VORTEX_SMT2_VEC-zps.json")

class ComparisonError(RuntimeError):
    pass


def sanity_check_kt1_10(
    tag: str,
    case: str,
    nemo_dir: Path,
    *,
    ladder_reference: Path | None = None,
) -> dict:
    """This round's own first 10 steps must equal the certified ladder."""
    report = trajectory_run(case, nemo_dir, max_step=10,
                             continue_after_first=True)
    reference = CERTIFIED_LADDER[tag] if ladder_reference is None else ladder_reference
    certified = json.loads(reference.read_text())
    mismatches = []
    for new_step, old_step in zip(report["steps"], certified["steps"]):
        require(new_step["kt"] == old_step["kt"], "kt sequence misaligned")
        for new_row, old_row in zip(new_step["rows"], old_step["rows"]):
            if (new_row["normalized_max_abs"] != old_row["normalized_max_abs"]
                    or new_row["status"] != old_row["status"]):
                mismatches.append((new_row["name"],
                                    old_row["normalized_max_abs"],
                                    new_row["normalized_max_abs"]))
    return {
        "status": "REPRODUCED" if not mismatches else "MISMATCH",
        "kt10_is_day_0_33": TABLE_DAYS[0] * STEPS_PER_DAY // STEPS_PER_DAY,
        "mismatches": mismatches,
        "ladder_reference": str(reference),
        "certified_first_over_bar": certified["first_over_bar"],
        "this_round_first_over_bar": report["first_over_bar"],
    }


def run_lego_card(card, lego_dir: Path, *, model_hooks=None,
                  snapshot_days=range(1, N_DAYS + 1)) -> dict:
    """Step one resolved card for 100 days through the production closure."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        **({} if model_hooks is None else {"_nemo_ws_test_hooks": model_hooks}))
    state = card.recipe.initial_state
    lego_dir.mkdir(parents=True, exist_ok=True)
    snapshot_days = frozenset(int(day) for day in snapshot_days)
    day9_T = None
    for step in range(1, N_DAYS * STEPS_PER_DAY + 1):
        state = model.step(state, dt=card.dt_s)
        if step == 9:
            # The kt=10 "before" entry in the ladder is the state after 9
            # completed steps; keep it for the sanity cross-check.
            day9_T = np.asarray(state.T.data, dtype=np.float64).copy()
        if step % STEPS_PER_DAY == 0 and step // STEPS_PER_DAY in snapshot_days:
            day = step // STEPS_PER_DAY
            fields = lego_fields(state)
            np.savez(lego_dir / f"day{day:03d}.npz",
                     **{k: np.asarray(v, dtype=np.float64)
                        for k, v in fields.items()})
    return {"card": card, "day9_T": day9_T}


def run_lego(case: str, lego_dir: Path) -> dict:
    """Step the certified named card 3000 times; snapshot every day."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    return run_lego_card(build_nemo_testcase_card(case), lego_dir)


def load_nemo(nemo_dir: Path, day: int, nlev: int) -> dict:
    import netCDF4

    step = day * STEPS_PER_DAY
    matches = sorted(nemo_dir.glob(f"*_{step:08d}_restart.nc"))
    require(len(matches) == 1,
            f"expected exactly one NEMO restart for step {step} in "
            f"{nemo_dir}, found {len(matches)}")
    with netCDF4.Dataset(matches[0]) as handle:
        recorded = int(np.asarray(handle.variables["kt"][...]))
        require(recorded == step, f"{matches[0]}: kt={recorded}, expected {step}")

        def xyz(name):
            require(handle.variables[name].dimensions
                    == ("time_counter", "nav_lev", "y", "x"),
                    f"{name}: axes {handle.variables[name].dimensions}")
            return np.asarray(handle.variables[name][0],
                              dtype=np.float64).transpose(1, 2, 0)[..., :nlev]

        require(handle.variables["sshn"].dimensions
                == ("time_counter", "y", "x"),
                f"sshn: axes {handle.variables['sshn'].dimensions}")
        fields = {"T": xyz("tn"), "S": xyz("sn"), "u": xyz("un"),
                  "v": xyz("vn"),
                  "ssh": np.asarray(handle.variables["sshn"][0],
                                    dtype=np.float64)}
    for name, values in fields.items():
        require(bool(np.all(np.isfinite(values))),
                f"{matches[0]}: NEMO field {name} is not finite")
    return fields


def load_lego(lego_dir: Path, day: int) -> dict:
    path = lego_dir / f"day{day:03d}.npz"
    with np.load(path) as handle:
        return {key: np.asarray(handle[key], dtype=np.float64)
                for key in handle.files}


def _rms(diff, mask) -> float:
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def _max(diff, mask) -> float:
    return float(np.max(np.abs(diff[mask])))


def score_day(lego_d: dict, nemo_d: dict, masks: dict) -> dict:
    out = {}
    for field in ("T", "u", "v", "ssh"):
        diff = lego_d[field] - nemo_d[field]
        mask = masks[field]
        out[f"{field}_rms"] = _rms(diff, mask)
        out[f"{field}_max"] = _max(diff, mask)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round210"))
    ap.add_argument("--cards", default="flux,vec",
                     help=("comma-separated tags: flux, vec, smtflux, smtvec, "
                           "smt1, smt2, smt3, smt4"))
    ap.add_argument("--skip-run", action="store_true",
                     help="scoring only; legoESM snapshots already written")
    ap.add_argument(
        "--ladder-reference", type=Path,
        help=("one-card measurement arm: bind the ten-step sanity check and "
              "kt=10 cross-check to this just-produced candidate ladder "
              "instead of the immutable certified ladder"))
    args = ap.parse_args()

    report = {"format": "nemo-testcase-l1-vortex-round210-100day-v1",
              "steps_per_day": STEPS_PER_DAY, "n_days": N_DAYS,
              "table_days": list(TABLE_DAYS), "cards": {}}

    selected = [tag.strip() for tag in args.cards.split(",") if tag.strip()]
    unknown = [tag for tag in selected if tag not in CARDS]
    if unknown:
        raise ComparisonError(f"unknown card tag(s) {unknown}; "
                              f"expected from {sorted(CARDS)}")
    if args.ladder_reference is not None and len(selected) != 1:
        raise ComparisonError("--ladder-reference requires exactly one card")
    for tag in selected:
        case, nemo_dir = CARDS[tag]
        print(f"\n################ {tag} ({case})")
        sanity = sanity_check_kt1_10(
            tag, case, nemo_dir, ladder_reference=args.ladder_reference)
        print(f"  kt1-10 sanity vs certified ladder: {sanity['status']}")
        require(sanity["status"] == "REPRODUCED",
                f"{tag}: this round's first 10 steps diverge from the "
                f"certified ladder: {sanity['mismatches']}")

        lego_dir = args.out / f"lego_{tag}"
        if args.skip_run and (lego_dir / "day100.npz").is_file():
            from legoesm.ocean.fidelity.nemo_testcase_recipe import (
                build_nemo_testcase_card,
            )
            card = build_nemo_testcase_card(case)
            day9_T = None
        else:
            run_out = run_lego(case, lego_dir)
            card = run_out["card"]
            day9_T = run_out["day9_T"]

        masks = expected_masks(card)
        nlev = card.recipe.z_coord.n_levels

        # kt=10 cross-check: step=9 legoESM state must equal the T field the
        # certified ladder's kt=10 row scored (same precision, same build).
        kt10_check = None
        if day9_T is not None:
            reference = (CERTIFIED_LADDER[tag] if args.ladder_reference is None
                         else args.ladder_reference)
            certified = json.loads(reference.read_text())
            kt10_row = next(r for r in certified["steps"][9]["rows"]
                             if r["name"].endswith(".T"))
            require(certified["steps"][9]["kt"] == 10,
                    "certified ladder's 10th entry is not kt=10")
            kt10_check = {
                "kt": 10, "day_equivalent": 10.0 / STEPS_PER_DAY,
                "certified_row_status": kt10_row["status"],
                "certified_row_normalized_max_abs":
                    kt10_row["normalized_max_abs"],
                "note": ("this round's step=9 legoESM T field is the exact "
                          "input the certified kt=10 row scored; the row's "
                          "own status/error above IS this round's day-0.33 "
                          "point, reused rather than rederived"),
            }

        days, rows = [], {}
        for day in range(1, N_DAYS + 1):
            lego_d = load_lego(lego_dir, day)
            nemo_d = load_nemo(nemo_dir, day, nlev)
            rows[str(day)] = score_day(lego_d, nemo_d, masks)
            days.append(day)
            if day % 20 == 0 or day in TABLE_DAYS:
                print(f"  day {day:>3}  T_rms={rows[str(day)]['T_rms']:.6e}  "
                      f"u_rms={rows[str(day)]['u_rms']:.6e}  "
                      f"ssh_rms={rows[str(day)]['ssh_rms']:.6e}")

        report["cards"][tag] = {
            "case": case, "kt1_10_sanity": sanity, "kt10_cross_check":
                kt10_check, "days": days, "rows": rows,
            "nemo_restart_dir": str(nemo_dir),
            "lego_snapshot_dir": str(lego_dir),
        }

    out_json = args.out / "round210_scores.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nWROTE {out_json}")

    # ---- table ----
    print("\n==== TABLE (days {}) ====".format(TABLE_DAYS))
    header = f"{'day':>4}"
    for tag in selected:
        header += f"  {tag+'_T_rms':>14}{tag+'_u_rms':>14}{tag+'_ssh_rms':>14}"
    print(header)
    for day in TABLE_DAYS:
        line = f"{day:>4}"
        for tag in selected:
            r = report["cards"][tag]["rows"][str(day)]
            line += (f"  {r['T_rms']:>14.6e}{r['u_rms']:>14.6e}"
                      f"{r['ssh_rms']:>14.6e}")
        print(line)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for field, ax, title in zip(("T_rms", "u_rms", "ssh_rms"), axes,
                                  ("T rms [K]", "u rms [m/s]", "ssh rms [m]")):
        for tag in selected:
            days = report["cards"][tag]["days"]
            values = [report["cards"][tag]["rows"][str(d)][field]
                      for d in days]
            ax.plot(days, values, label=tag)
        ax.set_yscale("log")
        ax.set_xlabel("day")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("VORTEX 100-day from-rest: legoESM vs NEMO (round 210)")
    fig.tight_layout()
    png_path = args.out / "round210_curves.png"
    fig.savefig(png_path, dpi=130)
    print(f"WROTE {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
