#!/usr/bin/env python3
"""Which ``stp2d.F90`` statement produces LOCK's external-mode bit.

Round 26 localized LOCK's one external-mode bit to the ``slow_u`` frame
(1 / 127 wet U faces, absolute max ``2.168404344971009e-19`` at cell
``[1, 64]``) and left its producing statement PLAUSIBLE between two
candidates, saying "no dumped frame separates them".  One does not have to.

NEMO's slow forcing under ``key_RK3`` is one statement, ``zu_frc(:,:) =
Ue_rhs(:,:)`` at ``dynspg_ts.F90:282`` (``dyn_cor_2D`` at ``:296`` subtracts
exactly zero: ``usrdef_hgr.F90:103-104`` sets ``f = 0``).  ``Ue_rhs`` has four
unconditional writers on this card:

* ``stp2d.F90:172`` -- ``dyn_adv_up3(..., pUe=Ue_rhs)``.  The routine ZEROES
  ``pUe``/``pVe`` on entry, and every later write subtracts a term built from
  ``zFu``/``zFv``/``puu(:,:,:,Kbb)``, each a product with the entry velocity.
  This probe ASSERTS that NEMO's dumped kt=1 entry velocity is exactly ``0.0``,
  so ``Ue_rhs`` after ``:172`` is exactly ``0.0`` and ``:172`` cannot produce a
  nonzero bit.
* ``stp2d.F90:185`` -- ``Ue_rhs = Ue_rhs + SUM(e3u_0*uu(:,:,:,Krhs)*umask) *
  r1_hu_0``, the cumulated depth mean of the 3-D momentum RHS.
* ``:196`` ``dyn_drg_init`` and ``:199-202`` wind, both exactly zero here
  (``ln_drg_OFF = .true.``; ``usrdef_sbc`` sets ``utau = 0``).
* THREE more writers exist and the first draft of this docstring missed them:
  ``:207`` under ``ln_apr_dyn``, ``:223`` under ``ln_ice_embd`` and ``:235``
  under ``ln_bern_srfc``.  All three are ``.false.`` here, and the probe now
  ASSERTS that from the run's own ``ocean.output`` and namelists rather than
  from the phrase "four unconditional writers".  A sixth, ``:180``, is the
  vector-form arm of the same ``SELECT CASE`` and is excluded by
  ``ln_dynadv_up3 = .true.``.

So the depth mean is the only statement left that can make a bit, and the
question is whether legoESM's counterpart makes it from its OWN 3-D RHS or
from the reduction itself.  Three arms, one variable, no NEMO run:

* ``model_captured`` -- the production ``slow_u`` frame the barotropic gate
  scores, from the model's own step.  Reproduces the round-26 row.
* ``statement_model_rhs`` -- this probe's reproduction of the model's
  depth-mean statement, fed the model's own momentum tendency.  It MUST equal
  ``model_captured`` bit for bit; if it does not, the probe is not measuring
  the model's statement and says so rather than reporting a number.
* ``statement_nemo_rhs`` -- the same statement fed NEMO's DUMPED
  ``uu(:,:,:,Krhs)`` (``oracle_rhs_kt00000001.bin``).  This is the transplant.

Reading the result: if the transplant still differs from NEMO's ``slow_u``,
the reduction makes the bit given NEMO's own 3-D RHS, and ``:185``'s
counterpart is the producer.  If the transplant is bit-exact, the reduction is
exact and the bit came in with legoESM's own RHS instead.

SCOPE OF THE ``nemo_reciprocal`` ARM -- read this before quoting it.  It is
measured at kt=1 on LOCK, and TWO things that differ in general coincide there:

* ``r1_hu_0`` is not ``1/hu_0``.  ``domain.F90:159`` builds it as
  ``ssumask/(hu_0 + 1 - ssumask)``.  On a wet LOCK face that is
  ``1/(20 + 1 - 1)``, and LOCK has exactly ONE distinct column depth,
  ``20.0``, for which ``H + 1 - 1 == H`` and ``1/(H+1-1)`` is bit-identical to
  ``1/H`` (both measured).  On a card with a depth where the ``+1-1`` shift
  rounds, the arm as written is not NEMO's expression.
* NEMO weights with the REFERENCE ``e3u_0`` and divides by the REFERENCE
  ``hu_0``; legoESM weights with LIVE thicknesses from ``eta``.  LOCK's kt=1
  entry ``ssh`` is exactly ``0.0`` on every cell (measured), so the two
  coincide bitwise here and the arm isolates the association alone.  At
  kt >= 2, ``eta != 0`` and reference-versus-live is a SEPARATE and much
  larger difference, O(eta/H), which this probe does not measure.

So "the association is the whole residual" is CONFIRMED for this frame and
REFUTED as a general statement about the statement's fidelity.  The next check
is named and not run: re-run these arms at kt=2, plus a fifth arm weighted by
``e3u_0`` with ``r1_hu_0``, and expect the reciprocal arm to stop being 0/127.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from pathlib import Path

import numpy as np
from nemo_testcase_phase3_stage_sweep_gate import (
    GateError,
    git_sha,
    require,
    sha256,
)
import nemo_testcase_overflow_barotropic_gate as bt
from nemo_testcase_overflow_barotropic_gate import (
    capture_legoesm_trace,
    read_oracle_trace,
    score_frame,
    state_from_oracle_entry,
)
from nemo_testcase_rule12_hpg_eligibility import read_entry_full, read_rhs
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)

CASE = "LOCK_EXCHANGE-zco"
RHS_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10")
EXTERNAL_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round25_lock_external_oracle")


def depth_mean_statement(du_dt, eta, h_bathy, z_coord, config, grid,
                         face_mask, *, tag: str = "u",
                         association: str = "model"):
    """The model's own statement, called through the model's own helpers.

    ``ocean_model_latlon_cgrid`` builds the barotropic slow forcing as
    ``compute_layer_thickness`` -> ``min_cell_to_uface``/``min_cell_to_vface``
    -> a fused ``sum(h_f)``/``sum(du_dt*h_f)`` pair -> divide -> mask.  Nothing
    here is re-derived; the arm-equality check against the captured production
    frame is what proves that.

    This is the ONE implementation: the GYRE weighting probe imports it rather
    than carrying a second copy (round 28; duplicated numerics are forbidden
    in this repo).  ``tag`` selects the face and ``grid`` is required by
    ``min_cell_to_vface``.
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
        nemo_source_round,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    # Fail early on the static selection, before any work: an unknown
    # association or face must never fall through to a default arm.
    if association not in ("model", "nemo_reciprocal"):
        raise ValueError(f"unknown association {association!r}")
    if tag not in ("u", "v"):
        raise ValueError(f"unknown face {tag!r}")

    h_k = compute_layer_thickness(
        eta, h_bathy, z_coord,
        min_water_column_m=config.min_water_column_m)
    h_f = min_cell_to_uface(h_k) if tag == "u" else min_cell_to_vface(h_k, grid)
    pair = jnp.sum(jnp.stack([h_f, jnp.asarray(du_dt) * h_f], axis=-1), axis=-2)
    depth = jnp.maximum(pair[..., 0], 1e-10)
    if association == "model":
        mean = pair[..., 1] / depth
    else:
        # stp2d.F90:185 multiplies by the PRECOMPUTED reciprocal r1_hu_0, it
        # does not divide.  x*(1/H) and x/H differ by up to one ULP.
        mean = pair[..., 1] * nemo_source_round(1.0 / depth)
    return np.asarray(mean * face_mask)


def run(*, plant: bool = False, allow_dirty: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from nemo_testcase_phase3_trajectory_gate import expected_masks, read_entry

    allow_dirty_stamps(allow_dirty)

    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    # The barotropic reader and capture helper are OVERFLOW-defaulted at
    # module level and re-bound by their own main(); this probe re-binds them
    # to LOCK's card and LOCK's record header (134, 7, 1, 19, 64) so nothing
    # is read or captured under OVERFLOW's constants.
    bt.CASE = CASE
    bt.EXPECTED = bt.CASE_EXPECTED[CASE]

    rhs_path = RHS_ROOT / "oracle_rhs_kt00000001.bin"
    entry_path = RHS_ROOT / "oracle_step_entry_kt00000001.bin"
    trace_path = (EXTERNAL_ROOT
                  / "oracle_overflow_bt_substeps_kt00000001_call1.bin")
    for path in (rhs_path, entry_path, trace_path):
        require(path.is_file(), f"missing {path}")

    rhs = read_rhs(rhs_path, CASE)
    entry = read_entry_full(entry_path, CASE)

    # The fact that removes stp2d.F90:172 from contention.
    rest = {"oracle_entry_abs_max_u": float(np.max(np.abs(entry["u"]))),
            "oracle_entry_abs_max_v": float(np.max(np.abs(entry["v"])))}
    require(rest["oracle_entry_abs_max_u"] == 0.0
            and rest["oracle_entry_abs_max_v"] == 0.0,
            f"NEMO's kt=1 entry is not at rest: {rest}; dyn_adv_up3's 2-D "
            "cumulation would then be nonzero and stp2d.F90:172 is back in "
            "contention")

    # Precondition: no OTHER Ue_rhs writer is live.  stp2d.F90 has six
    # assignments to it; :180 is the unselected vector-form arm and :207/:223/
    # :235 are guarded by ln_apr_dyn / ln_ice_embd / ln_bern_srfc.
    nemo_root = pathlib.Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    card_dir = nemo_root / "tests/LOCK_EXCHANGE_OMIP_L1_P3"
    output = (RHS_ROOT / "ocean.output").read_text(errors="replace")
    cfg_text = (card_dir / "EXP00/namelist_cfg").read_text()
    ref_text = (nemo_root / "cfgs/SHARED/namelist_ref").read_text()
    other_writers = {}
    for flag in ("ln_apr_dyn", "ln_ice_embd", "ln_bern_srfc"):
        printed = re.search(rf"{flag}\s*=\s*([TF])", output)
        if printed is not None:
            other_writers[flag] = (printed.group(1) == "T", "ocean.output")
        else:
            text = cfg_text if flag in cfg_text else ref_text
            on = bool(re.search(rf"^\s*{flag}\s*=\s*\.true\.", text, re.M))
            other_writers[flag] = (on, "namelist_cfg" if flag in cfg_text
                                   else "namelist_ref")
    live = {k: v for k, v in other_writers.items() if v[0]}
    require(not live,
            f"another stp2d.F90 Ue_rhs writer is live ({live}); the owner "
            "walk's statement inventory is incomplete")
    up3 = bool(re.search(r"^\s*ln_dynadv_up3\s*=\s*\.true\.", cfg_text,
                         re.M))
    require(up3, "ln_dynadv_up3 is not .true.; stp2d.F90:180's vector-form "
                 "arm would then own Ue_rhs instead of :185")

    oracle = read_oracle_trace(trace_path)
    oracle_slow_u = oracle["substeps"][0]["slow_u"]

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    seeded = read_entry(entry_path, CASE)
    captured = capture_legoesm_trace(flux_form_override=None, kt=1,
                                     reseed_entry=seeded)
    candidate_captured = captured["substeps"][0]["slow_u"]
    u_mask_full = np.asarray(
        card.recipe.initial_state.u_mask.data, dtype=np.float64)
    mask = captured["masks"]["U"]

    # The model's own kt=1 momentum tendency, from NEMO's own entry state.
    state = state_from_oracle_entry(
        card.recipe.initial_state, seeded, expected_masks(card))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    tendency = model.tendencies(state, dt=card.dt_s, momentum_only=True)
    model_du_dt = np.asarray(tendency.du_dt.data)

    nlev = card.recipe.z_coord.n_levels
    nemo_du_dt = np.zeros_like(model_du_dt)
    nemo_du_dt[:, 1:, :] = rhs["u"][..., :nlev]
    planted_cell = None
    if plant:
        # Plant on the LARGEST wet-face operand, +1.0, the same convention the
        # rest of this campaign's controls use.  Two weaker plants were tried
        # first and are recorded because each is a lesson: the FIRST wet face
        # carries an RHS of exactly 0.0 at kt=1, so nextafter there produced a
        # denormal that underflowed in the h-weighting (a control that
        # perturbs a zero is not a control); and one ULP even on the largest
        # operand (|RHS| ~ 2.2e-3) moves the depth mean by ~2e-20, which is
        # below one ULP of the ~1.1e-3 mean, so the probe cannot resolve it.
        # That second number is the probe's operand resolution and it bounds
        # what this probe can claim.
        wet = np.zeros_like(nemo_du_dt, dtype=bool)
        wet[:, 1:, :] = mask[..., None]
        flat = int(np.argmax(np.where(wet, np.abs(nemo_du_dt), -np.inf)))
        planted_cell = [int(v) for v in np.unravel_index(flat, nemo_du_dt.shape)]
        require(nemo_du_dt[tuple(planted_cell)] != 0.0,
                "planted operand is exactly zero")
        nemo_du_dt[tuple(planted_cell)] += 1.0

    arms = {}
    for name, du_dt, association in (
            ("statement_model_rhs", model_du_dt, "model"),
            ("statement_nemo_rhs", nemo_du_dt, "model"),
            # Ordered walk INSIDE the statement: NEMO's :185 ends in
            # ``* r1_hu_0``, a precomputed reciprocal, where legoESM divides.
            ("statement_nemo_rhs_reciprocal", nemo_du_dt, "nemo_reciprocal")):
        arms[name] = depth_mean_statement(
            du_dt, state.eta.data, state.H_bathy.data,
            card.recipe.z_coord, cfg, card.recipe.grid, u_mask_full,
            tag="u", association=association)[:, 1:]

    rows = [score_frame(f"{CASE}.kt1.substep1.slow_u.model_captured",
                        oracle_slow_u, candidate_captured, mask)]
    for name, candidate in arms.items():
        rows.append(score_frame(f"{CASE}.kt1.substep1.slow_u.{name}",
                                oracle_slow_u, candidate, mask))

    # Instrument calibration: the reproduced statement must BE the model's.
    reproduces = bool(np.array_equal(
        arms["statement_model_rhs"][mask], candidate_captured[mask]))
    transplant_equals_model = bool(np.array_equal(
        arms["statement_nemo_rhs"][mask], arms["statement_model_rhs"][mask]))
    rhs_equal_on_wet_faces = int(np.count_nonzero(
        nemo_du_dt[:, 1:, :][mask] != model_du_dt[:, 1:, :][mask]))

    transplant = next(r for r in rows if r["name"].endswith("statement_nemo_rhs"))
    reciprocal = next(
        r for r in rows if r["name"].endswith("statement_nemo_rhs_reciprocal"))
    if not reproduces:
        owner = "UNMEASURED_INSTRUMENT_DOES_NOT_REPRODUCE_THE_MODEL_STATEMENT"
    elif transplant["exact"]:
        owner = ("stp2d.F90:185 counterpart is EXACT given NEMO's own 3-D RHS; "
                 "the bit entered with legoESM's momentum tendency")
    else:
        owner = ("CONFIRMED stp2d.F90:185 counterpart (the depth-mean "
                 "reduction): it makes the bit from NEMO's OWN 3-D RHS. "
                 "stp2d.F90:172 REFUTED -- Ue_rhs after it is exactly 0.0 at "
                 "this rest entry")
        if reciprocal["exact"]:
            owner += (". Walked one statement deeper: at THIS frame the "
                      "divide-vs-precomputed-reciprocal association is the "
                      "whole residual -- x*(1/H) reproduces NEMO exactly "
                      "where x/H does not. SCOPED, not general: LOCK's kt=1 "
                      "eta is exactly 0.0 so live and reference thicknesses "
                      "coincide, and its single column depth 20.0 makes "
                      "NEMO's masked r1_hu_0 expression exactly 1/H. "
                      "Reference-vs-live weighting is a separate, larger "
                      "difference at kt>=2 that this probe does not measure")
        else:
            owner += (". The divide-vs-reciprocal association is NOT the whole "
                      f"residual (reciprocal arm still "
                      f"{reciprocal['n_unequal']}/{reciprocal['n']} at "
                      f"{reciprocal['absolute_max']!r})")

    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-lock-slow-forcing-owner-v1",
        "case": CASE,
        "legoesm_git_sha": legoesm_git_sha,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "transcendentals": "libm",
        "jax_backend": jax.default_backend(),
        "artifacts": {path.name: sha256(path)
                      for path in (rhs_path, entry_path, trace_path)},
        "precondition_oracle_at_rest": rest,
        "precondition_other_ue_rhs_writers_off": {
            k: {"live": v[0], "read_from": v[1]}
            for k, v in other_writers.items()},
        "precondition_flux_form_up3_selected": up3,
        "nemo_source": [
            "dynspg_ts.F90:282 zu_frc(:,:) = Ue_rhs(:,:) under key_RK3",
            "dynspg_ts.F90:296 dyn_cor_2D, exactly zero (usrdef_hgr.F90:103-104"
            " sets f=0)",
            "stp2d.F90:172 dyn_adv_up3 with pUe (zeroed on entry, then "
            "products of the entry velocity)",
            "stp2d.F90:185 cumulated depth mean of uu(:,:,:,Krhs)",
            "stp2d.F90:196 dyn_drg_init and :199-202 wind, both zero here",
        ],
        "instrument_reproduces_model_statement": reproduces,
        "resolution_demonstration": {
            "note": "the +1.0 plant proves the probe READS its operand, not "
                    "that it resolves the 2.168e-19 effect it attributes -- "
                    "one ULP on the largest operand moves the mean by ~2e-20, "
                    "below one ULP of the mean. What demonstrates the "
                    "resolution is the ARM SEPARATION: the divide and "
                    "reciprocal arms differ from each other by exactly the "
                    "bit under attribution.",
            "divide_arm_unequal_vs_nemo": transplant["n_unequal"],
            "reciprocal_arm_unequal_vs_nemo": reciprocal["n_unequal"],
            "arms_differ_by": abs(transplant["absolute_max"]
                                  - reciprocal["absolute_max"]),
        },
        "transplant_equals_model_arm": transplant_equals_model,
        "nemo_minus_model_rhs_unequal_wet_faces": rhs_equal_on_wet_faces,
        "rows": rows,
        "owner": owner,
        "planted_control": plant,
        "planted_cell": planted_cell,
    }
    if plant:
        # The transplant arm is ALREADY non-exact, so asserting on it would be
        # a vacuous control.  The RECIPROCAL arm is bit-exact unplanted, so a
        # one-ULP perturbation of its own operand must break it.
        require(not reciprocal["exact"],
                "planted RHS perturbation left the reciprocal arm bit-exact; "
                "the probe cannot see its own operand")
    return report


@scoped_allow_dirty
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(plant=args.plant, allow_dirty=args.allow_dirty)
    except GateError as error:
        print(json.dumps({"status": "GATE-ERROR", "error": str(error)},
                         indent=2))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        return 1
    return 0 if report["instrument_reproduces_model_statement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
