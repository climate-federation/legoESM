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

Round 5 adds the other half of the comparison, which is what actually names
the owner: legoESM's OWN per-term breakdown at the SAME boundary.  It comes
from the ``MomentumTendencyDiagnostics`` the step's own ``tendencies`` call
already builds -- with this step's stage face thicknesses, lateral-diffusion
thickness operands and continuity clock -- reported through a WRITE-only
observer next to the one that reports the total.  It is NOT a second call to
a public wrapper, which would drop those overrides and measure a different
array.  The rows are mapped to NEMO's increments exactly as legoESM's own
bundling allows:

    NEMO vor        == lego vortcor_u                       (een_total: both
                                                             halves, as NEMO)
    NEMO zad        == lego vertadv_u
    NEMO hpg + keg  == lego KE_PGF_u                        (a GROUP: legoESM
                                                             bundles the KE
                                                             gradient with the
                                                             pressure gradient
                                                             and cannot split
                                                             it)
    NEMO ldf        == lego Ah_lap + Bh_bilap + Cs + Cl

and every remaining diagnostic component must be identically zero or the map
is incomplete and the probe REFUSES.

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

# The row map, written out in the preregistration before it was run.  Each
# entry is (row name, the NEMO increments it covers, the legoESM diagnostic
# components it covers).  A row is EXACT when it covers the same physics on
# both sides; ``KE_PGF`` is a GROUP because legoESM bundles the kinetic-energy
# gradient with the pressure gradient in one field.
ROW_MAP = (
    ("vor", ("vor",), ("vortcor_u",)),
    ("keg+hpg", ("hpg", "keg"), ("KE_PGF_u",)),
    ("zad", ("zad",), ("vertadv_u",)),
    ("ldf", ("ldf",), ("Ah_lap_u", "Bh_bilap_u", "Cs_smag_u", "Cl_leith_u")),
)
# Everything the map does not use.  NEMO's pre-stage program contains none of
# these on this deck, so each must be identically zero or the map is wrong.
UNMAPPED_MUST_BE_ZERO = (
    "Dterm_u", "botdrag_u", "Av_vert_u", "phys_u", "surface_stress_u",
    "sponge_u",
)
PLANT_ROWS = tuple(name for name, _, _ in ROW_MAP)
# A plant's size: far above the total being explained, far below the terms.
PLANT_DELTA = 1.0e-06


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
        case: str = CASE,
        allow_dirty: bool = False, plant: str | None = None,
        substitute_zad_w: bool = False) -> dict:
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

    halo = 2
    record = read_terms(terms_root)
    rows = increments(record)
    require(rows[CONTROL]["bit_identical_to_previous"],
            "ORDERED CONTROL FAILED: the vertical-velocity call moved the "
            "momentum accumulator, so the dumps are not in the order this "
            "probe assumes and no increment below is a term")

    card = build_nemo_testcase_card(case)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(walk_root / "oracle_step_entry_kt00000001.bin", case,
                        expect_interior=interior)
    entry2 = read_entry(walk_root / "oracle_step_entry_kt00000002.bin", case,
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

    def observe_terms(values):
        seen["terms"] = {k: np.asarray(v) for k, v in values.items()}

    # ONE-VARIABLE ARM.  ``dyn_zad``'s only non-geometric operand is the
    # vertical velocity the preceding ``wzv`` call produced, and the per-term
    # record carries NEMO's own copy of it.  Substituting it alone -- through
    # the SAME seam the GYRE rounds used, which reaches nothing but this call
    # -- splits "legoESM's vertical advection is spelt differently" from
    # "legoESM's vertical velocity is different".
    _zad_w_override = None
    if substitute_zad_w:
        _zad_w_override = jnp.asarray(
            record["zad"]["fields"]["ww"][halo:-halo, halo:-halo]
            .transpose(1, 0, 2))

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            slow_forcing_rhs_observer=observe_u,
            slow_forcing_rhs_observer_face="u",
            stage1_zad_w_override=_zad_w_override,
            slow_forcing_rhs_term_observer=observe_terms))
    model.step(_seed_from_record(card.recipe.initial_state, entry1, nlev),
               dt=card.dt_s)
    require("u" in seen, "the right-hand-side observer never fired")
    require("terms" in seen, "the per-term observer never fired")

    # PASSIVITY, MEASURED.  Asking the same call for its decomposition adds
    # graph nodes, and two differently-compiled graphs of one statement
    # re-associate a sum in the last bit -- the compiled-rounding floor this
    # campaign has measured many times (operator notes L, AL, AS).  So the
    # observer is not required to be bit-passive; it is required to move the
    # model by an amount NEGLIGIBLE against the difference being attributed,
    # and the amount is reported rather than assumed.
    passive_seen = {}
    passive_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            slow_forcing_rhs_observer=lambda v: passive_seen.__setitem__(
                "u", np.asarray(v)),
            slow_forcing_rhs_observer_face="u"))
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)
    passive_out = passive_model.step(seed, dt=card.dt_s)
    observed_out = model.step(seed, dt=card.dt_s)
    passivity = {"right_hand_side": float(np.max(np.abs(
        seen["u"] - passive_seen["u"]))) if not substitute_zad_w else 0.0}
    for name in ("T", "S", "u", "v", "eta"):
        passivity[name] = (0.0 if substitute_zad_w else float(np.max(np.abs(
            np.asarray(getattr(observed_out, name).data)
            - np.asarray(getattr(passive_out, name).data)))))

    # NEMO's completed total on the same faces the model carries: the record
    # keeps the halo, the model does not, and the model's zonal array holds
    # one redundant west face the comparison drops (the same convention
    # lego_fields uses).
    nemo_total = record["zad"]["fields"]["uu_rhs"][
        halo:-halo, halo:-halo].transpose(1, 0, 2)[..., :nlev]
    ours = np.asarray(seen["u"])[:, 1:, :nlev]
    require(ours.shape == nemo_total.shape,
            f"shape mismatch: legoESM {ours.shape} against NEMO "
            f"{nemo_total.shape}")
    umask = np.asarray(masks["u"], dtype=bool)
    # The closure control below reads the array the model ACTUALLY produced,
    # so a plant on the total cannot make the decomposition look broken.
    ours_observed = ours
    if plant == "rhs-agrees":
        ours = nemo_total.copy()
    delta = np.zeros_like(ours)
    delta[umask] = ours[umask] - nemo_total[umask]
    delta_max = float(np.max(np.abs(delta)))
    # Negligible against what is being attributed, or itself under the
    # comparison floor -- once the difference IS the floor there is nothing
    # left for the instrument to be negligible against.
    require(passivity["right_hand_side"] <= max(delta_max * 1.0e-6, 1.0e-18),
            "PASSIVITY FAILED: the per-term observer moved the right-hand "
            f"side by {passivity['right_hand_side']:.3e}, which is not "
            f"negligible against the {delta_max:.3e} being attributed")

    # ---- legoESM's OWN per-term breakdown, on the same faces -------------
    def on_model_faces(field):
        return np.asarray(field)[:, 1:, :nlev]

    lego_terms = {name: on_model_faces(values)
                  for name, values in seen["terms"].items()
                  if name.endswith("_u")}
    for name in UNMAPPED_MUST_BE_ZERO:
        require(name in lego_terms,
                f"the diagnostics carry no component named {name}; the row "
                "map was written against a different decomposition")
        peak = float(np.max(np.abs(lego_terms[name][umask])))
        require(peak == 0.0,
                f"MAP INCOMPLETE: the component {name} is not identically "
                f"zero on this deck (peak {peak:.6e}), so NEMO's five "
                "increments do not cover legoESM's right-hand side")

    # CLOSURE: the rows must decompose the very array the total observer
    # reported, or they are a second opinion about it rather than a split of
    # it.  Read BEFORE any plant is applied.
    covered = [component for _, _, components in ROW_MAP
               for component in components]
    lego_sum = sum(lego_terms[name] for name in covered)
    closure = float(np.max(np.abs((lego_sum - ours_observed)[umask])))
    total_peak = float(np.max(np.abs(ours_observed[umask])))
    require(closure <= 1.0e-18,
            f"CLOSURE FAILED: the mapped components sum to within {closure:.3e} "
            f"of the observed right-hand side (peak {total_peak:.3e}); the "
            "rows are not a decomposition of the number being explained")

    nemo_increments = {name: rows[name]["_du"][halo:-halo, halo:-halo]
                       .transpose(1, 0, 2)[..., :nlev]
                       for name in BOUNDARIES if name != CONTROL}

    row_table = []
    for row_name, nemo_parts, lego_parts in ROW_MAP:
        ours_row = sum(lego_terms[name] for name in lego_parts)
        if plant == row_name:
            ours_row = ours_row + PLANT_DELTA
        theirs_row = sum(nemo_increments[name] for name in nemo_parts)
        diff = np.zeros_like(ours_row)
        diff[umask] = ours_row[umask] - theirs_row[umask]
        row_table.append({
            "row": row_name,
            "nemo_terms": list(nemo_parts),
            "legoesm_components": list(lego_parts),
            "nemo_row_peak": float(np.max(np.abs(theirs_row[umask]))),
            "legoesm_row_peak": float(np.max(np.abs(ours_row[umask]))),
            "legoesm_minus_nemo": float(np.max(np.abs(diff))),
        })

    # How big a DIFFERENCE a term can own.  Round 4 bounded it by NEMO's own
    # increment alone -- "a term cannot own a difference bigger than the whole
    # of the term" -- and that bound is WRONG.  The difference is
    # legoESM's version of the term minus NEMO's, so its bound is the SUM of
    # the two sides' peaks: a term whose two versions carry opposite signs at
    # the same cell produces a difference larger than either.  That is exactly
    # what the vertical advection of momentum does here, and the one-sided
    # bound excluded it.  The bound below is two-sided wherever legoESM's map
    # gives a counterpart; where legoESM bundles two of NEMO's terms in one
    # field the GROUP peak is used, which is conservative and can only fail to
    # exclude.  The difference the STAGE error requires is that error over the
    # first stage's clock.
    stage_clock_s = float(card.dt_s) / 3.0
    required = stage1_error / stage_clock_s
    lego_counterpart = {}
    for row_name, nemo_parts, lego_parts in ROW_MAP:
        peak = float(np.max(np.abs(
            sum(lego_terms[name] for name in lego_parts)[umask])))
        for nemo_part in nemo_parts:
            lego_counterpart[nemo_part] = peak
    verdict = []
    for name in BOUNDARIES:
        if name == CONTROL:
            continue
        own = rows[name]["max_abs_du"]
        counterpart = lego_counterpart[name]
        bound = own + counterpart
        verdict.append({
            "term": name,
            "max_abs_increment": own,
            "legoesm_counterpart_peak": counterpart,
            "two_sided_bound": bound,
            "excluded_by_magnitude": bool(bound < required),
            "excluded_because_identically_zero": bool(bound == 0.0),
            "delta_over_term": (float(delta_max / own) if own else None),
        })

    # The preregistered decision rule, applied without re-reading the rows.
    ranked = sorted(row_table, key=lambda r: r["legoesm_minus_nemo"],
                    reverse=True)
    top, runner_up = ranked[0], ranked[1]
    owner = None
    if (0.5 * delta_max <= top["legoesm_minus_nemo"] <= 2.0 * delta_max
            and top["legoesm_minus_nemo"]
            >= 10.0 * runner_up["legoesm_minus_nemo"]):
        owner = top["row"]

    report = {
        "case": case, "legoesm_git_sha": sha, "plant": plant,
        "zad_w_substituted_from_nemo": substitute_zad_w,
        "per_term_observer_perturbation": passivity,
        "terms_root": str(terms_root), "walk_root": str(walk_root),
        "ordered_control_wzv_leaves_the_accumulator_untouched": True,
        "stage1_clock_s": stage_clock_s,
        "measured_stage1_velocity_error": stage1_error,
        "required_rhs_difference": required,
        "legoesm_minus_nemo_prestage_rhs_max_abs": delta_max,
        "terms": verdict,
        "row_closure_max_abs": closure,
        "rows": row_table,
        "comparison_floor": runner_up["legoesm_minus_nemo"],
        "owner": owner,
    }
    survivors = [row["term"] for row in verdict
                 if not row["excluded_by_magnitude"]]
    report["surviving_terms"] = survivors
    if not survivors:
        report["status"] = "NO_TERM_CAN_OWN_IT"
    else:
        report["status"] = "OWNER_NAMED" if owner else "NO_SINGLE_OWNER"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=CASE,
                        choices=(CASE, "VORTEX_SMT_VEC-zps"),
                        help="which card; a seamount card needs its own "
                             "--terms-dir and --walk-dir")
    parser.add_argument("--terms-dir", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--walk-dir", type=Path, default=DEFAULT_WALK)
    parser.add_argument("--stage1-error", type=float, default=1.7073783406e-05,
                        help="the walk's measured stage-1 velocity error")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("rhs-agrees",) + PLANT_ROWS,
                        help="'rhs-agrees' forces legoESM's right-hand side "
                             "to NEMO's and the difference MUST collapse; a "
                             "ROW name adds a known offset to that row alone "
                             "and the discriminator must flag exactly that "
                             "row and no other")
    parser.add_argument("--substitute-zad-w", action="store_true",
                        help="hand the vertical advection NEMO's own vertical "
                             "velocity and nothing else")
    parser.add_argument("--baseline", type=Path,
                        help="the unplanted report, for a row plant's "
                             "'no other row moved' check")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.terms_dir, args.walk_dir,
                     stage1_error=args.stage1_error,
                     allow_dirty=args.allow_dirty, plant=args.plant,
                     substitute_zad_w=args.substitute_zad_w, case=args.case)
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
    print(f"row closure: {report['row_closure_max_abs']:.3e}")
    print("per-row, legoESM minus NEMO [m/s^2]:")
    for row in report["rows"]:
        print("  {row:8s} lego {legoesm_row_peak:.6e}  nemo "
              "{nemo_row_peak:.6e}  diff {legoesm_minus_nemo:.6e}".format(**row))
    print("owner:", report["owner"])
    print("comparison floor (the next-largest row): "
          f"{report['comparison_floor']:.6e}")
    print("status:", report["status"])
    if args.plant == "rhs-agrees":
        collapsed = (report["legoesm_minus_nemo_prestage_rhs_max_abs"] == 0.0)
        print(f"PLANT rhs-agrees {'COLLAPSED' if collapsed else 'DID NOT'}")
        return 1 if collapsed else 0
    if args.plant:
        # The plant must move EXACTLY its own row.  ``--baseline`` carries the
        # unplanted rows so "no other row moved" is measured, not asserted.
        if not args.baseline:
            print("REFUSE: a row plant needs --baseline <unplanted json>",
                  file=sys.stderr)
            return 2
        base = {r["row"]: r["legoesm_minus_nemo"]
                for r in json.loads(args.baseline.read_text())["rows"]}
        moved = [r["row"] for r in report["rows"]
                 if r["legoesm_minus_nemo"] != base[r["row"]]]
        ok = (moved == [args.plant])
        print(f"PLANT {args.plant}: rows that moved {moved} -> "
              f"{'VISIBLE and ISOLATED' if ok else 'NOT ISOLATED'}")
        return 1 if ok else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
