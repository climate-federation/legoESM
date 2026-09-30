#!/usr/bin/env python3
"""Name the term inside NEMO's pre-stage momentum right-hand side.

Round 4's walk proved, by substitution, that the WHOLE of the vector-EEN
VORTEX card's first-stage momentum error lives in the completed
three-dimensional right-hand side ``stp_2D`` finishes before the RK3 stages
start: handing legoESM NEMO's own copy of it puts that stage at 1.1e-16
against a bar of 1e-15, from 1.7e-05.  Which of the routines that build it
owns the difference could not be read from that record, which carries only
the completed total.

This reads the per-term record acquired for exactly that question -- the
accumulator dumped after each contributing routine, so the per-term
increments are the DIFFERENCES between consecutive records -- and reports:

  * each term's own increment, and the ORDERED control that the
    vertical-velocity call leaves the accumulator bit-identical (it computes
    ``ww``; it must not touch the momentum trend);
  * the right-hand-side difference the measured stage-1 error REQUIRES,
    which is that error divided by the first stage's clock;
  * therefore which terms are EXCLUDED, because a term cannot own a
    difference larger than the whole of the term;
  * legoESM's own completed pre-stage right-hand side against NEMO's, and
    that difference compared with each surviving term.

Note BD: the record is parsed from its own header by the shared checker; no
size or header tuple is predicted here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _strip3, _u_full, _v_full, read_bt_frame, read_entry,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, require,
)

DEFAULT_TERMS = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round4_rhsterms")
DEFAULT_WALK = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round3")
CASE = "VORTEX_VEC-zco"
# NEMO's execution order inside stp_2D; the increments are the differences
# between consecutive dumps.  ``wzv`` is the ordered control, not a term.
BOUNDARIES = ("hpg", "ldf", "vor", "wzv", "keg", "zad")
CONTROL = "wzv"


def read_terms(root: Path) -> dict:
    """Parse every boundary with the acquisition's own checker."""
    import importlib.util

    path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location("l1_check_records", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    magic, n_header, layout, _ = checker._FAMILIES["oracle_rhsterm_kt"]
    require(layout == "groups", "the per-term family is not group-structured")

    out = {}
    for term in BOUNDARIES:
        record = root / f"oracle_rhsterm_kt00000001_{term}.bin"
        require(record.is_file(), f"the acquisition did not write {record.name}")
        parsed = checker.parse_record(record)        # refuses on any mismatch
        raw = record.read_bytes()
        offset = 16 + 4 * n_header
        fields = {}
        for name, meta in parsed["groups"].items():
            offset += 32
            count = meta["doubles"]
            values = np.frombuffer(raw[offset:offset + 8 * count],
                                   dtype=np.float64).copy()
            offset += 8 * count
            fields[name] = values.reshape(tuple(meta["shape"]), order="F")
        out[term] = {"parsed": parsed, "fields": fields}
    return out


def increments(terms: dict) -> dict:
    """Each routine's own contribution, and the ordered control."""
    rows, previous = {}, None
    for name in BOUNDARIES:
        u = terms[name]["fields"]["uu_rhs"]
        v = terms[name]["fields"]["vv_rhs"]
        if previous is None:
            du, dv = u, v
        else:
            du, dv = u - previous[0], v - previous[1]
        rows[name] = {
            "max_abs_du": float(np.max(np.abs(du))),
            "max_abs_dv": float(np.max(np.abs(dv))),
            "bit_identical_to_previous": bool(
                previous is not None and np.array_equal(u, previous[0])
                and np.array_equal(v, previous[1])),
            "_du": du, "_dv": dv,
        }
        previous = (u, v)
    return rows


def run(terms_root: Path, walk_root: Path, *, stage1_error: float,
        allow_dirty: bool = False, plant: str | None = None) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    record = read_terms(terms_root)
    rows = increments(record)
    require(rows[CONTROL]["bit_identical_to_previous"],
            "ORDERED CONTROL FAILED: the vertical-velocity call moved the "
            "momentum accumulator, so the dumps are not in the order this "
            "probe assumes and no increment below is a term")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(walk_root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(walk_root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    frame = read_bt_frame(walk_root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )

    seen = {}

    def observe_u(values):
        seen["u"] = np.asarray(values)

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            slow_forcing_rhs_observer=observe_u,
            slow_forcing_rhs_observer_face="u"))
    model.step(_seed_from_record(card.recipe.initial_state, entry1, nlev),
               dt=card.dt_s)
    require("u" in seen, "the right-hand-side observer never fired")

    # NEMO's completed total on the same faces the model carries: the record
    # keeps the halo, the model does not, and the model's zonal array holds
    # one redundant west face the comparison drops (the same convention
    # lego_fields uses).
    halo = 2
    nemo_total = record["zad"]["fields"]["uu_rhs"][
        halo:-halo, halo:-halo].transpose(1, 0, 2)[..., :nlev]
    ours = np.asarray(seen["u"])[:, 1:, :nlev]
    require(ours.shape == nemo_total.shape,
            f"shape mismatch: legoESM {ours.shape} against NEMO "
            f"{nemo_total.shape}")
    umask = np.asarray(masks["u"], dtype=bool)
    if plant == "rhs-agrees":
        ours = nemo_total.copy()
    delta = np.zeros_like(ours)
    delta[umask] = ours[umask] - nemo_total[umask]
    delta_max = float(np.max(np.abs(delta)))

    # A term cannot own a right-hand-side difference bigger than the whole of
    # the term.  The difference the STAGE error requires is that error over
    # the first stage's clock.
    stage_clock_s = float(card.dt_s) / 3.0
    required = stage1_error / stage_clock_s
    verdict = []
    for name in BOUNDARIES:
        if name == CONTROL:
            continue
        own = rows[name]["max_abs_du"]
        verdict.append({
            "term": name,
            "max_abs_increment": own,
            "excluded_by_magnitude": bool(own < required),
            "excluded_because_identically_zero": bool(own == 0.0),
            "delta_over_term": (float(delta_max / own) if own else None),
        })

    report = {
        "case": CASE, "legoesm_git_sha": sha, "plant": plant,
        "terms_root": str(terms_root), "walk_root": str(walk_root),
        "ordered_control_wzv_leaves_the_accumulator_untouched": True,
        "stage1_clock_s": stage_clock_s,
        "measured_stage1_velocity_error": stage1_error,
        "required_rhs_difference": required,
        "legoesm_minus_nemo_prestage_rhs_max_abs": delta_max,
        "terms": verdict,
    }
    survivors = [row["term"] for row in verdict
                 if not row["excluded_by_magnitude"]]
    report["surviving_terms"] = survivors
    report["status"] = "MEASURED" if survivors else "NO_TERM_CAN_OWN_IT"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms-dir", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--walk-dir", type=Path, default=DEFAULT_WALK)
    parser.add_argument("--stage1-error", type=float, default=1.7073783406e-05,
                        help="the walk's measured stage-1 velocity error")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("rhs-agrees",),
                        help="force legoESM's right-hand side to NEMO's; the "
                             "difference MUST collapse, or the comparison is "
                             "not reading what it claims")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.terms_dir, args.walk_dir,
                     stage1_error=args.stage1_error,
                     allow_dirty=args.allow_dirty, plant=args.plant)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    for row in report["terms"]:
        print("{term:4s} increment {max_abs_increment:.6e}  "
              "excluded={excluded_by_magnitude}".format(**row))
    print("required right-hand-side difference: "
          f"{report['required_rhs_difference']:.6e}")
    print("legoESM minus NEMO, completed pre-stage right-hand side: "
          f"{report['legoesm_minus_nemo_prestage_rhs_max_abs']:.6e}")
    print("surviving terms:", report["surviving_terms"])
    print("status:", report["status"])
    if args.plant:
        collapsed = (report["legoesm_minus_nemo_prestage_rhs_max_abs"] == 0.0)
        print(f"PLANT rhs-agrees {'COLLAPSED' if collapsed else 'DID NOT'}")
        return 1 if collapsed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
