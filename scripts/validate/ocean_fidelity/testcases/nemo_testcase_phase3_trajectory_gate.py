#!/usr/bin/env python3
"""First-divergence trajectory gate for the certified NEMO testcases.

The NEMO records are the Nbb/before entry state.  legoESM's card initial state
therefore maps to kt=1 and one completed ``model.step`` maps to kt=2.  The
default gate stops immediately after first debt.  ``--continue-after-first``
is the explicit owner-exhausted sweep arm: it preserves that first-divergence
record and walks the remaining registered states without pretending they have
an exact entering prefix.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)


def git_sha(*, allow_dirty: bool = False) -> str:
    """Exact legoESM producer revision (fails closed on tracked dirt)."""
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp

    try:
        return _stamp(allow_dirty=allow_dirty)
    except RuntimeError as error:
        raise GateError(f"cannot stamp legoESM git SHA: {error}") from error

BAR = 1.0e-15
DEFAULT_ORACLE_ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
    "VORTEX-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round2"),
    # The same experiment with the ORCA2/GYRE momentum scheme set
    # (decision 73); its own NEMO run, beside the flux card's.
    "VORTEX_VEC-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round3"),
    # Decision 74's resolution ladder (round 208, operator note BZ).  Each rung
    # is the SAME certified executable reading a deck refined by NEMO's own
    # rule (AGRIF_FixedGrids.in:2 ratio 3; 1_namelist_cfg:21-22,43), so a row
    # that moves between rungs is a grid-size dependence in the transcription
    # and nothing else.
    "VORTEX-15km-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_ladder/15km/flx"),
    "VORTEX_VEC-15km-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_ladder/15km/vec"),
    "VORTEX-10km-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_ladder/10km/flx"),
    "VORTEX_VEC-10km-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_ladder/10km/vec"),
    # Decision 88's seamount pair (round 211 acquired, round 212 scores): the
    # same 30 km deck with a Gaussian seamount and z partial bottom cells,
    # built through NEMO's own usrdef_zgr hook on key_vco_1d3d.
    "VORTEX_SMT-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
        "round3/VORTEX_SMT_R3_OMIP_L1_P3/kt1_10"),
    "VORTEX_SMT_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
        "round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/kt1_10"),
    # Decision 93's seamount mini-ladder, rung 1 (round 220): the same deck
    # with namzdf at ORCA2 rung 0's values.
    "VORTEX_SMT1_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
        "round9/VORTEX_SMT1_VEC_R8_OMIP_L1_P3/kt1_10"),
    # Rung 2 (round 222): the same deck with namdrg's linear bottom drag.
    "VORTEX_SMT2_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/"
        "round10/VORTEX_SMT2_VEC_R8_OMIP_L1_P3/kt1_10"),
    "VORTEX_SMT3_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/"
        "oracle_vortex_smt3/kt1_10"),
    "VORTEX_SMT4_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
        "oracle_vortex_smt4/kt1_10"),
    # SMT-5 (Decision 107): SMT-4 + ORCA2 rung 1's T/S damping; the card reads
    # NEMO's dumped inputs from this same directory.
    "VORTEX_SMT5_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/"
        "round2/oracle_vortex_smt5/kt1_10"),
    # SMT-6 / SMT-6b (Decision 110): SMT-5 + ORCA2 rung 2's BBL and geothermal
    # heating; SMT-6b adds the cold-flank anomaly.  Same dumped-input reading.
    "VORTEX_SMT6_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/"
        "round5/oracle_vortex_smt6/kt1_10"),
    "VORTEX_SMT6B_VEC-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/"
        "round5/oracle_vortex_smt6b/kt1_10"),
}

# NEMO writes its records with a halo of this width on every side; the gate
# strips it to reach the local interior.  It is the same for all three cases.
_HALO = 2


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def read_entry(path: Path, case: str, *, expect_interior=None) -> dict:
    """Parse the record's OWN header; never predict its shape (note BD).

    This used to carry a hard-coded ``(nx, ny, nz)`` tuple per case, and a
    third case would have meant a third tuple.  Five acquisitions in a row have
    now been refused by a checker that predicted a size by hand, so the shape
    is read from the header and only the parts that are genuinely fixed -- the
    magic string, the format version, the tracer count and the word size -- are
    asserted.  ``expect_interior``, when given, is the INTERIOR shape the
    caller's card carries, which is a claim about the card rather than about
    the record and is checked separately and loudly.
    """
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack(
            "=8i", fh.read(32))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require((version, ntr, bits) == (1, 2, 64),
            f"{path}: unsupported record format "
            f"(version={version}, ntr={ntr}, bits={bits})")
    require(min(nx, ny, nz) > 0, f"{path}: nonpositive extent in header")
    require(nx > 2 * _HALO and ny > 2 * _HALO,
            f"{path}: {nx}x{ny} is not wider than two halos on each side")
    count = nx * ny * nz
    require(data.size == 4 * count + nx * ny,
            f"{path}: payload is {data.size} doubles, but its own header "
            f"({nx}x{ny}x{nz}, {ntr} tracers) asks for {4 * count + nx * ny}")
    if expect_interior is not None:
        interior = (ny - 2 * _HALO, nx - 2 * _HALO)
        require(tuple(expect_interior) == interior,
                f"{path}: record interior {interior} does not match the "
                f"card's {tuple(expect_interior)}")

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[
            _HALO:-_HALO, _HALO:-_HALO].transpose(1, 0, 2)

    return {
        "step": step,
        "Nbb": nbb,
        "nz": nz,
        "T": xyz(data[:count]),
        "S": xyz(data[count:2 * count]),
        "u": xyz(data[2 * count:3 * count]),
        "v": xyz(data[3 * count:4 * count]),
        "ssh": data[4 * count:].reshape((nx, ny), order="F")[
            _HALO:-_HALO, _HALO:-_HALO].T,
    }


def expected_masks(card):
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    v = active & np.roll(active, -1, axis=0)
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def lego_fields(state):
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        # NEMO's local interior stores one x record per T column.  The phase-2
        # geometry gate established these stagger mappings exactly.
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }


def score(
    name: str, oracle, candidate, mask, *, plant=False,
    allow_empty_no_active_face=False,
) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    no_active_face = not bool(use.any())
    if no_active_face:
        require(allow_empty_no_active_face, f"{name}: empty mask")
        # Inventory the structurally absent face array and retain a gross
        # nonzero control over all stored points. It is not an alignment row.
        use = np.ones(oracle.shape, dtype=bool)
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(use)[0])] += 1.0
    require(np.all(np.isfinite(candidate[use])), f"{name}: candidate nonfinite")
    from legoesm.ocean.fidelity.ulp_move_gate import record_residual_field
    record_residual_field(name, oracle, candidate, use)
    exact = bool(np.array_equal(oracle[use], candidate[use]))
    scale = max(float(np.max(np.abs(oracle[use]))), 1.0)
    error = float(np.max(np.abs(candidate[use] - oracle[use]))) / scale
    status = "AT-BAR" if error <= BAR else "DEBT"
    row = {
        "name": name,
        "status": status,
        "exact": exact,
        "normalized_max_abs": error,
        "bar": BAR,
        "n": int(use.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if no_active_face and status == "AT-BAR":
        row["status"] = "UNMEASURED"
        row["reason"] = (
            "no active meridional velocity face in the three-row closed tank; "
            "all stored values were nevertheless checked against zero")
    return row


def mark_uninformative(row: dict, field: str, kt: int, reference, mask) -> dict:
    """Downgrade vacuous at-bar controls without hiding a real failure."""
    if row["status"] != "AT-BAR" or kt < 2:
        return row
    wet_values = np.asarray(reference)[np.asarray(mask, dtype=bool)]
    if field == "S" and np.unique(wet_values).size == 1:
        row["status"] = "UNINFORMATIVE"
        row["reason"] = (
            "oracle salinity is spatially uniform (n_unique=1); "
            "this row cannot detect transport/time-level errors")
    elif field == "ssh" and np.count_nonzero(wet_values) == 0:
        row["status"] = "UNINFORMATIVE"
        row["reason"] = (
            "oracle SSH is identically zero; the row only bounds "
            "candidate absolute noise and cannot corroborate alignment")
    return row


def characterize_growth(steps: list[dict]) -> dict:
    """Measure ratio trend and polynomial/exponential fit, field by field."""
    result = {}
    for field in ("T", "u", "ssh"):
        points = []
        for step in steps:
            row = next(row for row in step["rows"]
                       if row["name"].endswith(f".{field}"))
            if step["kt"] >= 2 and row["status"] == "DEBT":
                points.append((step["kt"], row["normalized_max_abs"]))
        if len(points) < 4 or any(error <= 0.0 for _, error in points):
            result[field] = {
                "status": "UNMEASURED",
                "reason": "fewer than four positive DEBT samples",
            }
            continue
        kt = np.asarray([point[0] for point in points], dtype=np.float64)
        errors = np.asarray([point[1] for point in points], dtype=np.float64)
        ratios = errors[1:] / errors[:-1]
        ratios_monotone_decreasing = bool(np.all(np.diff(ratios) < 0.0))
        n_tail = max(4, len(points) // 2)
        x = kt[-n_tail:]
        y = np.log(errors[-n_tail:])

        def fit(abscissa):
            slope, intercept = np.polyfit(abscissa, y, 1)
            predicted = intercept + slope * abscissa
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            return float(slope), 1.0 - ss_res / ss_tot if ss_tot else 1.0

        power, polynomial_r2 = fit(np.log(x))
        exponential_rate, exponential_r2 = fit(x)
        if exponential_rate <= 0.0:
            classification = "BOUNDED_OR_DECAYING_NO_AMPLIFYING_MODE"
        elif polynomial_r2 >= exponential_r2:
            classification = "POLYNOMIAL_FIT_PREFERRED"
        else:
            classification = "EXPONENTIAL_FIT_PREFERRED_OPEN_MODE_QUESTION"
        result[field] = {
            "status": "MEASURED",
            "classification": classification,
            "steps": [int(value) for value in kt],
            "normalized_errors": [float(value) for value in errors],
            "successive_step_ratios": [float(value) for value in ratios],
            "ratios_monotone_decreasing": ratios_monotone_decreasing,
            "tail_ratios_monotone_decreasing": bool(
                np.all(np.diff(ratios[-max(3, n_tail - 1):]) < 0.0)),
            "tail_window_steps": [int(value) for value in x],
            "tail_power_law_exponent_p": power,
            "tail_polynomial_loglog_r2": polynomial_r2,
            "tail_exponential_rate_per_step": exponential_rate,
            "tail_exponential_semilog_r2": exponential_r2,
        }
    return result


def run(
    case: str, oracle_root: Path, max_step: int, *, plant=False,
    continue_after_first=False, diagnostic_disable_bbl=False,
    owner_controls=False, allow_dirty=False, arm_literal_stage_wzv=False,
    arm_legacy_seed_faces=False, arm_legacy_hadv_min_face_thickness=False,
    arm_legacy_2d_stage_face_mask=False,
    arm_legacy_live_stage_mean_weights=False,
    after_ssh_form=None,
) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        with_first_wzv_after_ssh,
    )

    # Stamp FIRST so a dirty tree refuses before any compute (fail closed).
    allow_dirty_stamps(allow_dirty)
    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(
        case, deck_root=(oracle_root if case in (
            "VORTEX_SMT5_VEC-zps", "VORTEX_SMT6_VEC-zps",
            "VORTEX_SMT6B_VEC-zps") else None))
    # fld_read interpolates the damping target at the step's elapsed time.
    damped = card.recipe.model_config.nemo_tracer_damping is not None
    # Measurement arm only -- the card still STATES its own form; this scores
    # the same card under the other one so the pair is one run's numbers.
    arm_config = with_first_wzv_after_ssh(
        card.recipe.model_config, after_ssh_form)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, arm_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            disable_bbl=diagnostic_disable_bbl,
            literal_stage_wzv=arm_literal_stage_wzv,
            legacy_seed_min_rule_faces=arm_legacy_seed_faces,
            legacy_hadv_min_face_thickness=arm_legacy_hadv_min_face_thickness,
            legacy_2d_stage_face_mask=arm_legacy_2d_stage_face_mask,
            legacy_live_stage_mean_weights=arm_legacy_live_stage_mean_weights))
    state = card.recipe.initial_state
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    steps = []
    exact_prefix = True
    first_over_bar = None

    for kt in range(1, max_step + 1):
        path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing {path}")
        oracle = read_entry(
            path, case,
            expect_interior=np.asarray(
                card.recipe.initial_state.T.data).shape[:2])
        require(oracle["step"] == kt, f"{path}: step mismatch")
        # NEMO's record carries jpk levels where the card executes jpkm1 of
        # them; the last is the permanently dry dummy bottom.  Assert the
        # relation rather than trimming blindly, so a record with the WRONG
        # number of levels is a refusal and not a silent slice.
        require(oracle["nz"] == nlev + card.dummy_bottom_records,
                f"{path}: {oracle['nz']} levels, card executes {nlev} plus "
                f"{card.dummy_bottom_records} dummy bottom record(s)")
        candidate = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(oracle[field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{case}.kt{kt}.before.{field}", reference,
                candidate[field], masks[field], plant=plant and kt == 1
                and field == "T", allow_empty_no_active_face=field == "v"))
            if field in ("u", "v"):
                rows[-1].update({
                    "frame": "instantaneous_prognostic_Nbb",
                    "staggering_and_reduction": (
                        f"oracle {field}{field}(:,:,:,Nbb) and legoESM "
                        f"prognostic {field} are both C-grid {field.upper()}-face "
                        "instantaneous 3-D fields; both use the same wet-face "
                        "mask and an elementwise L-infinity reduction, with no "
                        "depth or substep-time averaging"),
                })
            rows[-1] = mark_uninformative(
                rows[-1], field, kt, reference, masks[field])
        exact_here = all(row["exact"] for row in rows)
        over = [row["name"].rsplit(".", 1)[-1] for row in rows
                if row["status"] == "DEBT"]
        steps.append({
            "kt": kt,
            "Nbb": oracle["Nbb"],
            "exact_prefix_entering": exact_prefix,
            "exact_at_step": exact_here,
            "rows": rows,
        })
        exact_prefix = exact_prefix and exact_here
        if over:
            if first_over_bar is None:
                first_over_bar = {"kt": kt, "fields": over}
            if not continue_after_first:
                break
        if kt < max_step:
            state = model.step(
                state, dt=card.dt_s,
                **({"t_seconds": (kt - 1) * card.dt_s} if damped else {}))

    cfg = card.recipe.model_config
    bbl_attribution = None
    overflow_t_owner_hunt = None
    if case == "OVERFLOW-zps":
        from legoesm.ocean.physics.bbl_adv import (
            bbl_static_geometry,
            bbl_transports,
        )
        geom = bbl_static_geometry(
            card.recipe.z_coord.h_partial,
            card.recipe.initial_state.land_mask.data)
        utr, vtr = bbl_transports(
            card.recipe.initial_state.T.data,
            card.recipe.initial_state.S.data,
            geom,
            np.asarray(card.recipe.grid.dy_u)[:, 1:-1],
            np.asarray(card.recipe.grid.dx_v)[1:-1, :],
            gamma_s=cfg.bbl_gamma_s,
            rho_0=cfg.rho_0,
        )
        faithful_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg)
        control_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(disable_bbl=True))
        faithful_kt2 = faithful_model.step(
            card.recipe.initial_state, dt=card.dt_s)
        control_kt2 = control_model.step(
            card.recipe.initial_state, dt=card.dt_s)
        delta = np.asarray(faithful_kt2.T.data) - np.asarray(control_kt2.T.data)
        bbl_attribution = {
            "classification": "ONE_SIDED",
            "legoesm_side": "CONFIRMED_INACTIVE_AT_KT2",
            "nemo_side": "PLAUSIBLE_INACTIVE_FROM_GEOMETRY_ONLY",
            "scaling_check_before_owner_label": True,
            "max_abs_initial_utr_m3_s": float(np.max(np.abs(np.asarray(utr)))),
            "max_abs_initial_vtr_m3_s": float(np.max(np.abs(np.asarray(vtr)), initial=0.0)),
            "max_abs_kt2_temperature_movement_K": float(np.max(np.abs(delta))),
            "bit_identical_bbl_on_off_kt2_T": bool(np.array_equal(
                np.asarray(faithful_kt2.T.data), np.asarray(control_kt2.T.data))),
            "reason": (
                "legoESM's shipped initial density front does not intersect "
                "an active downslope face, so its option-2 transport is zero. "
                "The same NEMO conclusion is geometrically plausible but is "
                "not confirmed because utr_bbl was not read from NEMO"),
            "source": (
                "NEMO trabbl.F90:342-353,415-454; stage-3 calls at "
                "stprk3_stg.F90:468,588"),
        }
        if owner_controls:
            # Every read_entry call gets the card's own shape and the level
            # relation, not only the one in the walk: a reviewer found that
            # this owner-controls path could admit a record with the right
            # interior but the WRONG number of levels, which the old
            # hard-coded tuple would have refused.
            oracle2 = read_entry(
                oracle_root / "oracle_step_entry_kt00000002.bin", case,
                expect_interior=np.asarray(
                    card.recipe.initial_state.T.data).shape[:2])
            require(oracle2["nz"] == nlev + card.dummy_bottom_records,
                    "owner-controls kt=2 record has "
                    f"{oracle2['nz']} levels, card executes {nlev} plus "
                    f"{card.dummy_bottom_records} dummy bottom record(s)")
            oracle_T = oracle2["T"][..., :nlev]
            active_T = masks["T"]
            faithful_T = np.asarray(faithful_kt2.T.data)

            def t_abs_error(values):
                return float(np.max(np.abs(
                    np.asarray(values)[active_T] - oracle_T[active_T])))

            faithful_error = t_abs_error(faithful_T)
            require(faithful_error > 0.0, "OVERFLOW owner hunt is vacuous")
            controls = []

            def add_control(name, state_control, reference, changed_operand):
                values = np.asarray(state_control.T.data)
                movement = float(np.max(np.abs(values[active_T] - faithful_T[active_T])))
                error = t_abs_error(values)
                ratio = movement / faithful_error
                improves = error < faithful_error
                if error / max(float(np.max(np.abs(oracle_T[active_T]))), 1.0) <= BAR:
                    owner_label = "CONFIRMED_OWNER"
                elif improves and ratio >= 0.1:
                    owner_label = "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
                elif ratio < 0.1 or not improves:
                    owner_label = "REFUTED_AS_PRIMARY_OWNER"
                else:
                    owner_label = "UNMEASURED"
                controls.append({
                    "name": name,
                    "classification": "DIAGNOSTIC_ONE_VARIABLE_ARM",
                    "reference_arm": reference,
                    "changed_operand": changed_operand,
                    "faithful_absolute_max_error_K": faithful_error,
                    "control_absolute_max_error_K": error,
                    "candidate_movement_K": movement,
                    "movement_over_faithful_error": ratio,
                    "error_change_K": error - faithful_error,
                    "owner_label": owner_label,
                    "reason": (
                        "scale and movement only; a legacy/private one-sided "
                        "candidate control cannot establish two-model ownership"),
                })

            hook_arms = (
                ("legacy_velocity_primary_average", _NEMOWSRK3TestHooks(
                    primary_transport_average=False),
                 "private ablation of NEMO dynspg_ts.F90:823-834,956-979",
                 "flux-form primary transport average only"),
                ("legacy_one_step_fct", _NEMOWSRK3TestHooks(
                    two_step_fct_predictor=False),
                 "legoESM legacy, pre-existing one-step FCT",
                 "two-step FCT predictor only"),
                ("legacy_frozen_final_tracer_transport", _NEMOWSRK3TestHooks(
                    kmm_tracer_transports=False),
                 "legoESM legacy, pre-existing frozen-final transport",
                 "Kmm tracer transport time level only"),
                ("omit_stage_barotropic_correction", _NEMOWSRK3TestHooks(
                    stage_barotropic_correction=False),
                 "legoESM legacy, pre-existing post-stage split",
                 "per-stage primary velocity correction only"),
                ("omit_transport_reconcile", _NEMOWSRK3TestHooks(
                    momentum_transport_reconcile=False),
                 "private harness ablation of NEMO stprk3_stg.F90:257-274",
                 "un_adv/H advecting-transport reconcile only"),
            )
            for name, hooks, reference, operand in hook_arms:
                control_state = LatLonCGridOceanModel(
                    card.recipe.grid, card.recipe.z_coord, cfg,
                    _nemo_ws_test_hooks=hooks).step(
                        card.recipe.initial_state, dt=card.dt_s)
                add_control(name, control_state, reference, operand)

            transport_cfg = cfg._replace(
                barotropic=cfg.barotropic._replace(
                    barotropic_reconcile_target="transport_avg"))
            add_control(
                "wrong_prognostic_transport_frame",
                LatLonCGridOceanModel(
                    card.recipe.grid, card.recipe.z_coord, transport_cfg).step(
                        card.recipe.initial_state, dt=card.dt_s),
                "NEMO un_adv tracer transport, not a prognostic state frame",
                "prognostic primary velocity replaced by time-mean transport")
            overflow_t_owner_hunt = {
                "status": "UNMEASURED_AFTER_REGISTERED_ARMS",
                "first_over_bar": "kt=2 T",
                "faithful_absolute_max_error_K": faithful_error,
                "faithful_normalized_max_error": faithful_error / max(
                    float(np.max(np.abs(oracle_T[active_T]))), 1.0),
                "scaling_check_before_owner_label": True,
                "one_variable_controls": controls,
            }
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-phase3-trajectory-v1",
        "legoesm_git_sha": legoesm_git_sha,
        # Which after-SSH arm produced these rows; None = the card.
        "after_ssh_form_arm": after_ssh_form,
        "after_ssh_form_resolved": arm_config.nemo_first_wzv_after_ssh,
        "case": case,
        "status": "AT-BAR" if first_over_bar is None else "DEBT",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "bar": BAR,
        "oracle_root": str(oracle_root),
        "selectors": {
            "eos": cfg.eos,
            "eos_depth": cfg.eos_depth,
            "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
            "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
            "vertical_momentum_scheme": cfg.vertical_momentum_scheme,
            "tracer_time_integrator": cfg.tracer_time_integrator,
            "rk3_ws_scheme_identity": (
                "nemo_kmm+two_step_fct+stage_correction+transport_reconcile"),
            "bbl_adv_option": cfg.bbl_adv_option,
            "bbl_gamma_s": cfg.bbl_gamma_s,
            "diagnostic_disable_bbl_test_hook": diagnostic_disable_bbl,
            "arm_literal_stage_wzv_test_hook": arm_literal_stage_wzv,
            "arm_legacy_seed_faces_test_hook": arm_legacy_seed_faces,
            "arm_legacy_hadv_min_face_thickness_test_hook": arm_legacy_hadv_min_face_thickness,
            "arm_legacy_2d_stage_face_mask_test_hook": arm_legacy_2d_stage_face_mask,
        },
        "first_over_bar": first_over_bar,
        "bbl_attribution": bbl_attribution,
        "overflow_t_owner_hunt": overflow_t_owner_hunt,
        "growth_characterization": characterize_growth(steps),
        "steps": steps,
        "unmeasured": [
            "NEMO per-term tendencies at the first divergent step",
            *(["remaining OVERFLOW kt=2 T owner after registered arms"]
              if case == "OVERFLOW-zps" else []),
            *([] if case == "OVERFLOW-zps" else
              ["stage-coupled OVERFLOW BBL transport and tendency"]),
            *([] if continue_after_first else
              ["trajectory after the first over-bar step"]),
        ],
    }


@scoped_allow_dirty
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(DEFAULT_ORACLE_ROOTS), required=True)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--continue-after-first", action="store_true")
    parser.add_argument("--diagnostic-disable-bbl", action="store_true")
    parser.add_argument("--owner-controls", action="store_true")
    parser.add_argument(
        "--arm-literal-stage-wzv", action="store_true",
        help=("one-variable S-19 arm: route the WS-RK3 stage cross-level "
              "velocity through nemo_qco_wzv_operands (sshwzv.F90:331-336 "
              "as called at stprk3_stg.F90:297) instead of the generic "
              "diagnose_w_from_flux_div. NEMO has no such switch"))
    parser.add_argument(
        "--arm-legacy-seed-faces", action="store_true",
        help=("one-variable arm: restore the min-of-stretched-cells rescale "
              "in the barotropic loop-entry seed instead of NEMO's "
              "e3u_0*(1+r3u) (dynspg_ts.F90:487 / stprk3_stg.F90:440). "
              "NEMO has no such switch"))
    parser.add_argument(
        "--arm-legacy-hadv-min-face-thickness", action="store_true",
        help=("one-variable ablation of the stage momentum-advection face "
              "thickness (private _NEMOWSRK3TestHooks control; NEMO has no "
              "such switch): restore tendencies()' min-of-stretched-T rule "
              "instead of NEMO's e3u(Kmm) = e3u_0*(1+r3u)"))
    parser.add_argument(
        "--arm-legacy-2d-stage-face-mask", action="store_true",
        help=("one-variable ablation of the stage face-mask rank (private "
              "_NEMOWSRK3TestHooks control; NEMO has no such switch): "
              "restore the 2-D state.u_mask broadcast over levels instead of "
              "NEMO's 3-D umask(ji,jj,jk) (stprk3_stg.F90:367,375,382,444,273)"))
    parser.add_argument(
        "--arm-legacy-live-stage-mean-weights", action="store_true",
        help=("one-variable ablation of the stage depth-mean WEIGHTS (private "
              "_NEMOWSRK3TestHooks control; NEMO has no such switch): restore "
              "the live h_u_pre/H_u_pre weighting instead of NEMO's reference "
              "SUM(e3u_0*uu)*r1_hu_0 (stprk3_stg.F90:440, domain.F90:145)"))
    parser.add_argument(
        "--after-ssh-form", default=None,
        help="measurement arm: score this card with NEMO's first-wzv "
             "after-SSH form overridden (rk3_extrapolated | "
             "rk3_extrapolated_carried | leapfrog_continuity). The card "
             "still states its own form; omitting this flag is the card.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    from legoesm.ocean.fidelity.ulp_move_gate import (
        add_ulp_compare_arguments, capture_residual_fields,
        comparison_exit_code, persist_ulp_comparison, run_ulp_comparison,
        write_residual_artifact,
    )
    add_ulp_compare_arguments(parser)
    args = parser.parse_args()
    require(args.max_step >= 1, "max-step must be positive")
    with capture_residual_fields() as residuals:
        report = run(
            args.case, args.oracle_dir or DEFAULT_ORACLE_ROOTS[args.case],
            args.max_step, plant=args.plant,
            continue_after_first=args.continue_after_first,
            diagnostic_disable_bbl=args.diagnostic_disable_bbl,
            owner_controls=args.owner_controls, allow_dirty=args.allow_dirty,
            arm_literal_stage_wzv=args.arm_literal_stage_wzv,
            arm_legacy_seed_faces=args.arm_legacy_seed_faces,
            arm_legacy_hadv_min_face_thickness=args.arm_legacy_hadv_min_face_thickness,
            arm_legacy_2d_stage_face_mask=args.arm_legacy_2d_stage_face_mask,
            arm_legacy_live_stage_mean_weights=args.arm_legacy_live_stage_mean_weights,
            after_ssh_form=args.after_ssh_form)
    if args.output:
        write_residual_artifact(report, args.output, residuals)
    elif args.compare_to:
        raise GateError("--compare-to requires --output for the residual sidecar")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.compare_to:
        # The exit status now reports the oracle-relative cellwise comparison,
        # not this run's own AT-BAR/DEBT verdict.
        comparison = run_ulp_comparison(args, report)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        print(persist_ulp_comparison(args, comparison))
        code = comparison_exit_code(comparison)
        if code == 2:
            print("PLANTED CONTROL DID NOT PRODUCE ITS REQUIRED VERDICT: "
                  f"{comparison['plant']}",
                  file=sys.stderr)
        return code
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
