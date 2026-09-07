#!/usr/bin/env python3
"""Round-40 GYRE stage-3 momentum-RHS operator walk.

WHERE THE RESIDUAL IS, AND WHERE IT IS NOT.  The stage-3 momentum RHS legoESM
hands the implicit vertical solve differs from NEMO's by ``2.06e-16`` on u
(``nemo_testcase_l2_gyre_round30_stage3_owner.py --mode pre_ldf``), and round
33 measured that ``rDt`` times exactly that residual is the whole ``dyn_zdf``
entry difference, so it is what remains of GYRE's kt=2 velocity failure.

NEMO's stage-3 momentum RHS, in call order, with the deck's own resolved
switches (``round38_oracle_trazdf_kt2/ocean.output``):

  ``stprk3_stg.F90:266-268``  zub/zvb barotropic velocity correction (np_HYB)
  ``stprk3_stg.F90:273-274``  zFu/zFv advective transport at Kmm
  ``stprk3_stg.F90:290``      wzv at np_velocity (ln_dynadv_vec = T, :798)
  ``stprk3_stg.F90:322``      eos( ts, Kmm, rhd, rhop )
  ``stprk3_stg.F90:324``      dyn_hpg -> hpg_sco (ln_hpg_sco = T, :834); under
                              key_RK3 it OVERWRITES Krhs (dynhpg.F90:359-363,
                              :383-387), so nothing before it survives
  ``stprk3_stg.F90:327``      dyn_vor -> vor_ene (ln_dynvor_ene = T, :810)
  ``stprk3_stg.F90:331``      dyn_adv -> dyn_keg (nn_dynkeg = 0, :799) + dyn_zad
  ``stprk3_stg.F90:400``      dyn_ldf   <-- the pre-ldf frame is taken HERE
  ``stprk3_stg.F90:430``      dyn_zdf

``dyn_spg`` is NOT in the stage loop: ln_dynspg_ts = T (:846) but the only
``dyn_spg_ts`` call is ``stp2d.F90:281``, before the stages.

TWO MODES, and neither of them names an owner.

``--mode inputs`` scores the STATE stage 3 is handed -- the stage-2 Kaa
velocity legoESM reads as uu(Kmm), and the stage tracers and ssh the stage
eos/dyn_hpg reads -- against NEMO's own ``oracle_stage_kt00000001_s2.bin``.
That separates "the operators inherited a wrong operand" from "the operators
generated it".

``--mode split`` exposes legoESM's OWN hpg / vorticity / advection buckets at
stage 3 and reports their MAGNITUDES.  NEMO's stage-3 split does not exist as
a record -- ``oracle_rkstage2_terms`` is header-locked to kstg = 2
(``nemo_testcase_l2_gyre_phase3_gate.py:286``) -- so these are SCALES, never
scores: what they buy is the relative error each operator would have to carry
to produce the measured residual, which RANKS the suspects and nothing more.
The frame spec and run.sh for the record that would score them are committed
in ``nemo_testcase_l2_gyre_round40_stage3_terms/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_stage,
    require,
    score,
    sha256,
)
from nemo_testcase_l2_gyre_round30_stage3_owner import (
    ORACLE_ROOT,
    PRE_LDF_RECORD,
    read_pre_ldf,
)

# The V2 root, the one every round-29..39 arm scores against.  ``gyre_kt1_10``
# carries a DIFFERENT oracle_stage_kt00000001_s2.bin (55e780b8d56e vs
# e29972359b9f), so it is not interchangeable here.
STAGE2_RECORD = "oracle_stage_kt00000001_s2.bin"
OPERATORS = ("hpg", "vorticity", "advection")


def _card_and_forcing():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    return card, cfg, freshwater, surface


def _run_hooks(card, cfg, freshwater, surface, **hooks):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks),
    ).step(card.recipe.initial_state, dt=card.dt_s,
           freshwater=freshwater, surface_forcing=surface)
    state = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state)
    return lego_fields(state)


def _precision_preflight():
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; JAX_DISABLE_JIT is "
            "forbidden")
    return jax.default_backend()


def run_inputs(oracle_root: Path, *, plant: bool = False) -> dict:
    """The state stage 3 is handed, against NEMO's stage-2 Kaa record."""
    backend = _precision_preflight()
    record = oracle_root / STAGE2_RECORD
    oracle = read_stage(record, 2)
    card, cfg, freshwater, surface = _card_and_forcing()
    masks = expected_masks(card)
    fields = _run_hooks(card, cfg, freshwater, surface,
                        expose_momentum_stage=2, expose_tracer_stage=2)
    rows = []
    for name, where in (("u", "uu(Kmm) read by dyn_vor/dyn_adv"),
                        ("v", "vv(Kmm) read by dyn_vor/dyn_adv"),
                        ("T", "ts(Kmm) read by eos at stprk3_stg.F90:322"),
                        ("S", "ts(Kmm) read by eos at stprk3_stg.F90:322"),
                        ("ssh", "ssh(Kmm) behind e3t/e3u/e3v(Kmm)")):
        candidate = fields[name]
        reference = oracle[name]
        if reference.ndim == 3:
            reference = reference[..., :candidate.shape[-1]]
        row = score(f"{CASE}.kt1.stage3.input.{name}", reference, candidate,
                    masks[name], plant=plant and name == "u")
        row["nemo_operand"] = where
        rows.append(row)
    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT",
                "planted stage-3 input violation did not fire")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round40-stage3-operators-v1",
        "mode": "inputs",
        "case": CASE,
        "boundary": ("stprk3_stg.F90:218 -- Kmm at stage 3 is the stage-2 "
                     "Kaa state"),
        "record": str(record),
        "record_sha256": sha256(record),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


def run_split(oracle_root: Path, *, plant: bool = False) -> dict:
    """legoESM's OWN stage-3 operator buckets, as SCALES, plus the closure."""
    backend = _precision_preflight()
    record = oracle_root / PRE_LDF_RECORD
    oracle = read_pre_ldf(record)
    card, cfg, freshwater, surface = _card_and_forcing()
    masks = expected_masks(card)

    total = _run_hooks(card, cfg, freshwater, surface,
                       expose_stage3_momentum_rhs="pre_ldf")
    buckets = {
        name: _run_hooks(card, cfg, freshwater, surface,
                         expose_momentum_operator=name,
                         expose_momentum_operator_stage=3)
        for name in OPERATORS
    }
    if plant:
        buckets["hpg"] = dict(buckets["hpg"])
        bumped = np.array(buckets["hpg"]["u"], copy=True)
        bumped[..., 0] = bumped[..., 0] + 1.0
        buckets["hpg"]["u"] = bumped

    rows, closure = [], []
    for face in ("u", "v"):
        mask = masks[face]
        reference = oracle[face][..., :total[face].shape[-1]]
        residual = np.abs(reference - total[face])[mask > 0]
        summed = sum(buckets[name][face] for name in OPERATORS)
        unequal = int(np.count_nonzero(
            (summed != total[face])[mask > 0]))
        closure.append({
            "face": face,
            "n": int(np.count_nonzero(mask > 0)),
            "cells_unequal": unequal,
            "max_abs": float(np.max(np.abs(summed - total[face])[mask > 0])),
            "exact": unequal == 0,
        })
        for name in OPERATORS:
            magnitude = float(np.max(np.abs(buckets[name][face])[mask > 0]))
            rows.append({
                "name": f"{CASE}.kt1.stage3.bucket.{name}.{face}",
                "kind": "SCALE-NOT-A-SCORE",
                "legoesm_absolute_max": magnitude,
                "residual_absolute_max": float(np.max(residual)),
                "required_relative_error": (
                    float(np.max(residual) / magnitude) if magnitude > 0.0
                    else None),
                "nemo_reference": ("NONE -- oracle_rkstage2_terms is locked to "
                                   "kstg=2; the stage-3 record's frame spec is "
                                   "in nemo_testcase_l2_gyre_round40_stage3_"
                                   "terms/"),
            })
    exact = all(entry["exact"] for entry in closure)
    if plant:
        require(not exact, "planted bucket violation did not break the closure")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round40-stage3-operators-v1",
        "mode": "split",
        "case": CASE,
        "boundary": "stprk3_stg.F90:400 -- the Krhs dyn_ldf receives",
        "record": str(record),
        "record_sha256": sha256(record),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "closure": closure,
        "closure_exact": exact,
        "status": "CLOSED" if exact else "OPEN",
        "rows": rows,
        "planted_control": plant,
        "note": ("every bucket row is a SCALE.  No operator is scored, and "
                 "none may be named the owner from this arm."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("inputs", "split"))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = (run_inputs(args.oracle_root, plant=args.plant)
              if args.mode == "inputs"
              else run_split(args.oracle_root, plant=args.plant))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if report["mode"] == "inputs":
        for row in report["rows"]:
            print(f"{row['status']:<8} {row['name']:<40} "
                  f"bit_unequal {row['n_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.17g}")
    else:
        for entry in report["closure"]:
            print(f"CLOSURE {entry['face']} cells_unequal "
                  f"{entry['cells_unequal']}/{entry['n']} "
                  f"max {entry['max_abs']:.17g}")
        for row in report["rows"]:
            print(f"SCALE    {row['name']:<44} "
                  f"max {row['legoesm_absolute_max']:.6e}  "
                  f"required_rel {row['required_relative_error']}")
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] in ("AT-BAR", "CLOSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
