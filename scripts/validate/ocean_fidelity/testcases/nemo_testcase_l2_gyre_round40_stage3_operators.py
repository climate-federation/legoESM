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

``--mode terms`` scores each stage-3 operator's OWN contribution against
NEMO's own frames, out of the round-40 acquisition
``oracle_rkstage3_terms_kt00000001.bin``.

A ``split`` mode existed briefly in this file and is DELETED.  It added three
host-array buckets and compared them to a total from a fourth run, so its
"closure" could never be bit-exact -- the model accumulates inside one fused
graph -- and its plant could not fail against a baseline that was already
red.  The record supersedes it entirely: there is no reason to rank suspects
by a proxy once NEMO's own per-operator frames are on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    BAR,
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
# The round-40 ldf_slp acquisition, kt = nit000 + 1 -- the first step whose
# BEFORE state carries the horizontal structure step 1 created, and therefore
# the first record in which NEMO's own slopes are not identically zero.
SLOPES_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                   "round40_oracle_ldfslp")
SLOPES_RECORD = "oracle_ldfslp_kt00000002.bin"


def read_stage3_terms(path: Path, *, expect_kt: int = 1) -> dict:
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
        require((version, kt, kstg, nx, ny, nz, bits)
                == (1, expect_kt, 3, *DIMS, 64),
                f"{path}: bad header {header}")
        require(all(level in (1, 2, 3) for level in (kbb, kmm, krhs, kaa))
                and krhs == kaa,
                f"{path}: invalid RK level tuple {kbb,kmm,krhs,kaa}")
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


def read_ldfslp(path: Path) -> dict:
    """Read ``NEMO_L2_LDFSL_1``: every statement of ``ldf_slp``.

    Self-describing, same discipline as the terms reader: a renamed or
    reordered array is a KeyError, never a mislabelled frame.
    """
    import struct

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=14i", handle.read(56))
        require(magic == "NEMO_L2_LDFSL_1", f"{path}: bad magic {magic!r}")
        version, kt, kbb, kmm, nx, ny, nz = header[:7]
        require((version, nx, ny, nz, header[-1]) == (1, *DIMS, 64),
                f"{path}: bad header {header}")
        arrays = {}
        while True:
            name_raw = handle.read(16)
            if not name_raw:
                break
            require(len(name_raw) == 16, f"{path}: truncated array name")
            name = name_raw.decode("ascii").rstrip()
            rank, n1, n2, n3 = struct.unpack("=4i", handle.read(16))
            count = {0: 1, 1: n1, 2: n1 * n2}.get(rank, n1 * n2 * n3)
            payload = np.frombuffer(handle.read(8 * count), dtype=np.float64)
            require(payload.size == count, f"{path}: truncated array {name!r}")
            require(np.all(np.isfinite(payload)),
                    f"{path}: non-finite payload in {name!r}")
            if rank == 0:
                arrays[name] = float(payload[0])
            elif rank == 1:
                arrays[name] = payload.copy()
            elif rank == 2:
                arrays[name] = _xy(payload, n1, n2)
            else:
                arrays[name] = _xyz(payload, n1, n2, n3)
    for required in ("nmln", "hmlp", "uslp", "wslpi", "prd", "pn2"):
        require(required in arrays, f"{path}: missing {required!r}")
    return {"header": header, "kt": kt, "arrays": arrays}


def run_slopes(oracle_root: Path, *, plant: bool = False) -> dict:
    """THE MIXED-LAYER INDEX, scored against NEMO's own nmln and hmlp.

    ``ldf_slp`` reads ``nmln`` twice -- ``zhmlpt = gdept(nmln-1,Kmm)*ssmask``
    (``LDF/ldfslp.F90:143``) and ``r1_hmlw`` off ``hmlp``
    (``LDF/ldfslp.F90:161``) -- and both come from ``zdf_mxl``'s N-SQUARED
    INTEGRAL criterion (``ZDF/zdfmxl.F90:95-104``).  This arm scores legoESM's
    own mixed-layer helper against NEMO's dumped pair, under BOTH criteria,
    on NEMO's OWN before state.  It is a statement-level score, not a
    downstream one: the fold arm can only say the product moved.
    """
    import jax.numpy as jnp

    backend = _precision_preflight()
    record = SLOPES_ROOT / SLOPES_RECORD
    rec = read_ldfslp(record)
    arrays = rec["arrays"]
    require(rec["kt"] == 2, f"{record}: the discriminating record is kt = 2")
    nemo_nmln = np.asarray(arrays["nmln"], dtype=np.float64)
    nemo_hmlp = np.asarray(arrays["hmlp"], dtype=np.float64)
    ssmask = np.asarray(arrays["ssmask"], dtype=np.float64) > 0.0
    require(float(np.abs(np.asarray(arrays["uslp"])).max()) > 0.0,
            f"{record}: NEMO's own uslp is identically zero, so this record "
            "cannot discriminate any transcription")

    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_mld)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    init = card.recipe.initial_state
    # NEMO's OWN before state, from this record: prd and pn2 are dumped, but
    # the mixed-layer helper takes T/S, so the tracers come from the kt=2
    # trazdf record's Kbb frames, the same source round 39's fold arm used.
    from nemo_testcase_l2_gyre_round35_trazdf_matrix import (
        _box, read_trazdf_matrix)
    kt2 = read_trazdf_matrix(
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
             "round38_oracle_trazdf_kt2/oracle_trazdf_matrix_kt00000002.bin"),
        expect_kt=2)
    jpkm1 = kt2["header"]["jpkm1"]
    tr = lambda a: np.ascontiguousarray(a.transpose(1, 0, 2))
    T_bb = tr(_box(kt2, "T_Kbb_in", jpkm1))
    S_bb = tr(_box(kt2, "S_Kbb_in", jpkm1))
    from legoesm.ocean.eos import make_eos_fn
    eos_fn = make_eos_fn(cfg.eos, cfg.eos_linear)

    rows = []
    for criterion in ("rho_c", "n2_integral"):
        hml, m_base = _nemo_mld(
            criterion, jnp.asarray(T_bb), jnp.asarray(S_bb),
            jnp.asarray(init.land_mask.data), card.recipe.z_coord, eos_fn,
            cfg.gm_redi.mld_rho_c, g=cfg.constants.g,
            rho_0=cfg.constants.rho_0,
            active_3d=jnp.asarray(card.recipe.z_coord.is_active))
        # legoESM's 0-based first stratified cell is NEMO's 1-based nmln - 1.
        lego_nmln = np.asarray(m_base, dtype=np.float64) + 2.0
        lego_hml = np.asarray(hml, dtype=np.float64)
        if plant and criterion == "n2_integral":
            lego_nmln = lego_nmln + 1.0
        nz = int(ssmask.sum())
        rows.append({
            "name": f"{CASE}.kt2.ldfslp.nmln.{criterion}",
            "n": nz,
            "cells_unequal": int(np.count_nonzero(
                (lego_nmln != nemo_nmln)[ssmask])),
            "max_abs": float(np.max(np.abs(lego_nmln - nemo_nmln)[ssmask])),
            "statement": "ZDF/zdfmxl.F90:99 nmln = MIN(jk,ikt)+1",
        })
        rows.append({
            "name": f"{CASE}.kt2.ldfslp.hmlp.{criterion}",
            "n": nz,
            "cells_unequal": int(np.count_nonzero(
                (lego_hml != nemo_hmlp)[ssmask])),
            "max_abs": float(np.max(np.abs(lego_hml - nemo_hmlp)[ssmask])),
            "nemo_absolute_max": float(np.max(np.abs(nemo_hmlp)[ssmask])),
            "statement": "ZDF/zdfmxl.F90:104 hmlp = gdepw(nmln,Kmm)*ssmask",
        })
    default_row = next(r for r in rows
                       if r["name"].endswith("nmln.rho_c"))
    nemo_row = next(r for r in rows
                    if r["name"].endswith("nmln.n2_integral"))
    if plant:
        require(nemo_row["cells_unequal"] > 0,
                "the planted mixed-layer index did not move its own row")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round40-stage3-operators-v1",
        "mode": "slopes",
        "case": CASE,
        "record": str(record),
        "record_sha256": sha256(record),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "rows": rows,
        "card_default_criterion": cfg.gm_redi.mld_criterion,
        "status": ("AT-BAR" if nemo_row["cells_unequal"] == 0 else "DEBT"),
        "columns_the_card_default_gets_wrong": default_row["cells_unequal"],
        "planted_control": plant,
    }


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
            # THE BAR, IN THE GATE.  The campaign's shared scorer normalises
            # by max(|reference|, 1), so for a field whose own maximum is
            # 1.8e-13 an AT-BAR verdict is VACUOUS -- it would pass an
            # operator carrying a per-cent relative error.  A per-operator row
            # is AT-BAR only if it also clears the bar RELATIVE to the
            # operator's own magnitude, which is the quantity a transcription
            # defect lives in.
            row["relative_bar"] = BAR
            if row["n_unequal"] and row["relative_max_abs"] > BAR:
                row["status"] = "DEBT"
                row["debt_reason"] = (
                    "relative to the operator's own maximum, not to 1.0")
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
                        choices=("inputs", "terms", "slopes"))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    runner = {"inputs": run_inputs, "terms": run_terms,
              "slopes": run_slopes}[args.mode]
    report = runner(args.oracle_root, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if report["mode"] == "slopes":
        for row in report["rows"]:
            print(f"SLOPE-OPERAND {row['name']:<44} "
                  f"unequal {row['cells_unequal']}/{row['n']} "
                  f"max {row['max_abs']:.6g}")
        print(f"CARD-DEFAULT {report['card_default_criterion']} gets "
              f"{report['columns_the_card_default_gets_wrong']} columns wrong")
    elif report["mode"] == "terms":
        for face, entry in report["record_closure_against_pre_ldf"].items():
            print(f"RECORD-CLOSURE {face} cells_unequal "
                  f"{entry['cells_unequal']} max {entry['max_abs']:.17g}")
        for row in report["rows"]:
            print(f"{row['status']:<8} {row['name']:<46} "
                  f"bit_unequal {row['n_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.6g} "
                  f"rel {row['relative_max_abs']:.6g}")
        for row in report["operand_rows"]:
            print(f"OPERAND  {row['name']:<46} "
                  f"unequal {row['cells_unequal']}/{row['n']} "
                  f"max {row['max_abs']:.6g} "
                  f"nemo_absmax {row['nemo_absolute_max']:.6g}")
        print(f"FIRST-OPERATOR-OVER-BAR {report['first_operator_over_bar']}")
    else:
        for row in report["rows"]:
            print(f"{row['status']:<8} {row['name']:<40} "
                  f"bit_unequal {row['n_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.17g}")
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] in ("AT-BAR", "CLOSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
