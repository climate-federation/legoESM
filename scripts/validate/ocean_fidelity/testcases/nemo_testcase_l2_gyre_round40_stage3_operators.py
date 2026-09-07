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
    DIMS,
    _surface_forcings,
    _xy,
    _xyz,
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
# The round-40 acquisition the OPERATOR ran on this agent's request.  Its own
# admission and its source admission both report PASS with zero violations.
TERMS_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                  "round40_oracle_stage3_terms")
TERMS_RECORD = "oracle_rkstage3_terms_kt00000001.bin"


def read_stage3_terms(path: Path) -> dict:
    """Read ``NEMO_L2_RKTS3_1``: the stage-3 per-operator momentum frames.

    Self-describing: a 16-char name and ``rank, n1, n2, n3`` precede every
    payload, so the reader never assumes the write order and a renamed or
    reordered array is a KeyError rather than a silently mislabelled frame.
    """
    import struct

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=16i", handle.read(64))
        require(magic == "NEMO_L2_RKTS3_1", f"{path}: bad magic {magic!r}")
        (version, kt, kstg, kbb, kmm, krhs, kaa,
         nx, ny, nz, jpkm1, ntsi, ntei, ntsj, ntej, bits) = header
        require(
            (version, kt, kstg, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
            == (1, 1, 3, 1, 2, 3, 3, *DIMS, 64),
            f"{path}: bad header {header}")
        arrays = {}
        while True:
            name_raw = handle.read(16)
            if not name_raw:
                break
            require(len(name_raw) == 16, f"{path}: truncated array name")
            name = name_raw.decode("ascii").rstrip()
            rank, n1, n2, n3 = struct.unpack("=4i", handle.read(16))
            count = 1 if rank == 0 else (n1 * n2 if rank == 2 else n1 * n2 * n3)
            payload = np.frombuffer(handle.read(8 * count), dtype=np.float64)
            require(payload.size == count, f"{path}: truncated array {name!r}")
            require(np.all(np.isfinite(payload)),
                    f"{path}: non-finite payload in {name!r}")
            if rank == 0:
                arrays[name] = float(payload[0])
            elif rank == 2:
                arrays[name] = _xy(payload, n1, n2)
            else:
                arrays[name] = _xyz(payload, n1, n2, n3)
    require("after_adv_u" in arrays and "after_hpg_u" in arrays,
            f"{path}: the operator frames are missing")
    return {"header": header, "arrays": arrays}


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


def run_terms(oracle_root: Path, *, plant: bool = False) -> dict:
    """THE STAGE-3 OPERATOR SPLIT, scored against NEMO's own frames.

    ``dyn_hpg`` OVERWRITES ``Krhs`` under ``key_RK3`` (``dynhpg.F90:359-363``,
    ``dynhpg.F90:383-387``), so its contribution IS ``after_hpg``; the other
    two are differences of consecutive frames.  Each is scored against the
    legoESM bucket the model's own stage-3 call produced.
    """
    backend = _precision_preflight()
    record = TERMS_ROOT / TERMS_RECORD
    rec = read_stage3_terms(record)
    arrays = rec["arrays"]
    card, cfg, freshwater, surface = _card_and_forcing()
    masks = expected_masks(card)

    # THE RECORD'S OWN CLOSURE, checked before anything is read off it: its
    # last frame must reproduce the pre-dyn_ldf record every round-30..40 arm
    # already scores, bit for bit.  Two instruments, one quantity.
    pre_ldf = read_pre_ldf(oracle_root / PRE_LDF_RECORD)
    closure = {}
    for face in ("u", "v"):
        a = arrays[f"after_adv_{face}"]
        b = pre_ldf[face]
        closure[face] = {
            "cells_unequal": int(np.count_nonzero(a != b)),
            "max_abs": float(np.max(np.abs(a - b))),
        }
    record_consistent = all(v["cells_unequal"] == 0 for v in closure.values())

    buckets = {
        name: _run_hooks(card, cfg, freshwater, surface,
                         expose_momentum_operator=name,
                         expose_momentum_operator_stage=3)
        for name in OPERATORS
    }
    if plant:
        bumped = np.array(buckets["vorticity"]["u"], copy=True)
        bumped[..., 0] = bumped[..., 0] + 1.0
        buckets["vorticity"] = dict(buckets["vorticity"], u=bumped)

    rows = []
    for face in ("u", "v"):
        nlev = buckets["hpg"][face].shape[-1]
        hpg = arrays[f"after_hpg_{face}"][..., :nlev]
        vor = arrays[f"after_vor_{face}"][..., :nlev] - hpg
        adv = arrays[f"after_adv_{face}"][..., :nlev] - \
            arrays[f"after_vor_{face}"][..., :nlev]
        for name, reference in (("hpg", hpg), ("vorticity", vor),
                                ("advection", adv)):
            row = score(f"{CASE}.kt1.stage3.operator.{name}.{face}",
                        reference, buckets[name][face], masks[face])
            row["nemo_frame"] = {
                "hpg": "after_hpg (dyn_hpg OVERWRITES Krhs)",
                "vorticity": "after_vor - after_hpg",
                "advection": "after_adv - after_vor"}[name]
            rows.append(row)

    # The vorticity divisor, given NEMO's own ssh: legoESM's shared literal
    # e3f_vor against NEMO's own dumped one.  vor_ene divides by it at
    # BLD/ppsrc/nemo/dynvor.f90:556 under key_qco.
    import jax.numpy as jnp

    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid
    stage2 = read_stage(oracle_root / STAGE2_RECORD, 2)
    nemo_eta_kmm = np.asarray(stage2["ssh"], dtype=np.float64)
    lego_e3f = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
        jnp.asarray(nemo_eta_kmm), card.recipe.z_coord, jnp.float64,
        nn_e3f_typ=0), dtype=np.float64)
    nemo_e3f = arrays["e3f_vor_Kmm"]
    nlev = min(lego_e3f.shape[-1], nemo_e3f.shape[-1])
    fmask = arrays["fmask"][..., :nlev] > 0.0
    # legoESM stores NEMO's F(i,j) at vertex [j+1,i+1]; strip the added
    # south/west walls so the two are the same points.
    lego_native = lego_e3f[1:, 1:, :nlev]
    geom_rows = [{
        "name": f"{CASE}.kt1.stage3.operand.e3f_vor",
        "n": int(fmask.sum()),
        "cells_unequal": int(np.count_nonzero(
            (lego_native != nemo_e3f[..., :nlev])[fmask])),
        "max_abs": float(np.max(
            np.abs(lego_native - nemo_e3f[..., :nlev])[fmask])),
        "nemo_absolute_max": float(np.max(np.abs(nemo_e3f[..., :nlev])[fmask])),
        "statement": ("e3f_0vor*(1+r3f*fe3mask), "
                      "BLD/ppsrc/nemo/dynvor.f90:556"),
    }]

    over = [r for r in rows if r["status"] != "AT-BAR"]
    first = None
    for name in OPERATORS:
        if any(r["status"] != "AT-BAR" and f".{name}." in r["name"]
               for r in rows):
            first = name
            break
    status = "AT-BAR" if not over else "DEBT"
    if plant:
        require(status == "DEBT" and first == "vorticity",
                "the planted bucket violation did not surface as the "
                "vorticity operator")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round40-stage3-operators-v1",
        "mode": "terms",
        "case": CASE,
        "record": str(record),
        "record_sha256": sha256(record),
        "record_closure_against_pre_ldf": closure,
        "record_self_consistent": record_consistent,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "status": status,
        "first_operator_over_bar": first,
        "rows": rows,
        "operand_rows": geom_rows,
        "planted_control": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("inputs", "split", "terms"))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    runner = {"inputs": run_inputs, "split": run_split,
              "terms": run_terms}[args.mode]
    report = runner(args.oracle_root, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if report["mode"] == "terms":
        for face, entry in report["record_closure_against_pre_ldf"].items():
            print(f"RECORD-CLOSURE {face} cells_unequal "
                  f"{entry['cells_unequal']} max {entry['max_abs']:.17g}")
        for row in report["rows"]:
            print(f"{row['status']:<8} {row['name']:<46} "
                  f"bit_unequal {row['n_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.6g} "
                  f"rel {row['max_relative']:.6g}")
        for row in report["operand_rows"]:
            print(f"OPERAND  {row['name']:<46} "
                  f"unequal {row['cells_unequal']}/{row['n']} "
                  f"max {row['max_abs']:.6g} "
                  f"nemo_absmax {row['nemo_absolute_max']:.6g}")
        print(f"FIRST-OPERATOR-OVER-BAR {report['first_operator_over_bar']}")
    elif report["mode"] == "inputs":
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
