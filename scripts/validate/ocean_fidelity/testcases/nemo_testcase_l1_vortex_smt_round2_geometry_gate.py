#!/usr/bin/env python3
"""Round 212 / VORTEX_SMT round 2 -- the cards' geometry IS NEMO's, bit for bit.

Decision 88 deliverable (2): before a single time step is scored, the two
seamount cards' resolved geometry must equal the geometry NEMO resolved --
not to a tolerance, to zero ULP.  This gate reads NEMO's own
``mesh_mask.nc`` (the SMT deck sets ``ln_meshmask = .true.``) and the cards
built by ``build_nemo_testcase_card``, and refuses on any difference.

What is compared, and why each one:
  * ``mbathy``   -- the zps bottom level.  One wrong column is a different
                    experiment, and the ze3min rule and the T-point rule
                    disagree for a whole band of bathymetries.
  * ``e3t_0``    -- the partial T thickness INCLUDING the below-bottom copy
                    NEMO writes at ik+1 and INCLUDING the land ring, which
                    ``dom_zgr`` masks only after ``usr_def_zgr`` returns.
  * ``e3u_0``, ``e3v_0``, ``e3f_0`` -- the min-of-neighbours faces, which is
                    the one place OVERFLOW's 1-D shortcut would have been
                    wrong and where e3f is built from e3v, not from e3u.
  * ``tmask``, ``umask``, ``vmask`` -- the wet set the ladder is scored on.

Pre-impl search (RULE 4): grepped scripts/validate/ocean_fidelity for an
existing mesh_mask reader; round 1's ``nemo_testcase_l1_vortex_smt_round1_
nemo_sanity.geometry`` reads the same file for a different question (the
seamount's shape, not an identity), and ``require`` from the shared
trajectory gate is reused here rather than a second refusal helper written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import netCDF4

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

CASES = {
    "VORTEX_SMT-zps": "VORTEX_SMT_OMIP_L1_P3",
    "VORTEX_SMT_VEC-zps": "VORTEX_SMT_VEC_R8_OMIP_L1_P3",
    "VORTEX_SMT1_VEC-zps": "VORTEX_SMT1_VEC_R8_OMIP_L1_P3",
    "VORTEX_SMT2_VEC-zps": "VORTEX_SMT2_VEC_R8_OMIP_L1_P3",
    "VORTEX_SMT3_VEC-zps": "VORTEX_SMT3_VEC_R8_OMIP_L1_P3",
}


def compare(case: str, run_dir: Path) -> dict:
    # The ladder scores these cards under the fp64 policy; a card built under
    # any other one would be a different card (the certified VORTEX deck's
    # uniform 500 m is exact in fp32 and a partial cell is not), so the gate
    # pins it and says so rather than inheriting the process default.
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card, vortex_smt_partial_cell_geometry,
        vortex_horizontal_coordinates, _VORTEX_RESOLUTIONS,
    )
    card = build_nemo_testcase_card(case)
    res = _VORTEX_RESOLUTIONS["30km"]
    source = vortex_horizontal_coordinates(res)
    _, k_bot, e3t, e3u, e3v, e3f = vortex_smt_partial_cell_geometry(source)
    ops = card.recipe.z_coord.nemo_een_barotropic
    nlev = e3t.shape[-1] - 1

    with netCDF4.Dataset(run_dir / "mesh_mask.nc") as h:
        def f3(name):                      # (z,y,x) -> (y,x,z)
            return np.transpose(
                np.asarray(h.variables[name][0], dtype=np.float64), (1, 2, 0))
        nemo = {
            "mbathy": np.asarray(h.variables["mbathy"][0], dtype=np.int64),
            "e3t_0": f3("e3t_0"), "e3u_0": f3("e3u_0"),
            "e3v_0": f3("e3v_0"), "e3f_0": f3("e3f_0"),
            "tmask": f3("tmask"), "umask": f3("umask"),
            "vmask": f3("vmask"),
            # Under key_vco_1d3d NEMO has NO three-dimensional e3w: the mesh
            # file carries e3w_1d alone, and domzgr_substitute.h90:80 makes
            # e3w_0(i,j,k) = e3w_1d(k) at every column, partial bottom cells
            # included.  That 1-D ladder is the divisor trazdf.F90:219-220
            # uses, so the card's own e3w_0 is compared against it rather
            # than left as the one unchecked field (round 220's reviewer).
            "e3w_1d": np.asarray(h.variables["e3w_1d"][0],
                                 dtype=np.float64).reshape(-1),
        }

    rows = []

    def row(name, lego, ref, *, integer=False):
        lego = np.asarray(lego)
        ref = np.asarray(ref)
        require(lego.shape == ref.shape,
                f"{case}/{name}: card {lego.shape} vs NEMO {ref.shape}")
        same = bool(np.array_equal(lego, ref))
        diff = np.abs(lego.astype(np.float64) - ref.astype(np.float64))
        rows.append({
            "field": name, "shape": list(lego.shape), "bit_identical": same,
            "n_differing": int((diff != 0).sum()),
            "max_abs_difference": (int(diff.max()) if integer
                                   else float(diff.max())),
        })

    # The hook's own arrays carry jpk = nlev+1 records; mesh_mask carries the
    # same, so the below-bottom copy is compared too and not quietly dropped.
    row("k_bot (mbathy)", k_bot, nemo["mbathy"], integer=True)
    row("e3t_0", e3t, nemo["e3t_0"])
    row("e3u_0", e3u, nemo["e3u_0"])
    row("e3v_0", e3v, nemo["e3v_0"])
    row("e3f_0", e3f, nemo["e3f_0"])
    # ... and the arrays the CARD hands the solver, which are the first nlev
    # records of those.  A card that built the geometry right and then wired
    # the wrong slice into its operands would pass the rows above and fail
    # these.
    row("card e3t_0 (operand)",
        np.asarray(card.recipe.z_coord.nemo_e3t_0), nemo["e3t_0"][..., :nlev])
    row("card e3u_0 (operand)", np.asarray(ops.e3u_0),
        nemo["e3u_0"][..., :nlev])
    row("card e3v_0 (operand)", np.asarray(ops.e3v_0),
        nemo["e3v_0"][..., :nlev])
    row("card e3f_0 (operand)", np.asarray(ops.e3f_0),
        nemo["e3f_0"][..., :nlev])
    # ROUND 214: and the arrays the SOLVER RESOLVES.  The rows above prove the
    # card CARRIES NEMO's faces; they say nothing about what the shared qco
    # operand builder hands wzv/div_hor, which until this round aliased
    # e3u_0 = e3v_0 = e3t_0 on every raw-mesh card.  These two rows are that
    # statement's identity proof, and they are the non-vacuous ones: on this
    # seamount e3u_0 != e3t_0 on 1 084 wet U faces.
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands
    import jax.numpy as jnp
    umask_red = np.concatenate(
        [np.zeros_like(np.asarray(ops.umask)[:, :1]), np.asarray(ops.umask)],
        axis=1)
    vmask_red = np.concatenate(
        [np.zeros_like(np.asarray(ops.vmask)[:1]), np.asarray(ops.vmask)],
        axis=0)
    resolved = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid, jnp.asarray(umask_red),
        jnp.asarray(vmask_red), jnp.float64, nlev)
    row("RESOLVED e3u_0 (qco arm)", np.asarray(resolved.e3u_0),
        nemo["e3u_0"][..., :nlev])
    row("RESOLVED e3v_0 (qco arm)", np.asarray(resolved.e3v_0),
        nemo["e3v_0"][..., :nlev])
    # The northern row, named explicitly because the round-213 repair zeroed
    # it (630 cells of e3v_0 at j = nj-1 came back 0.0 where NEMO carries
    # 500.0).  Counted, not asserted in prose.
    north_bad = int(np.count_nonzero(
        (np.asarray(resolved.e3v_0)[-1] == 0.0)
        & (nemo["e3v_0"][..., :nlev][-1] != 0.0)))
    rows.append({"field": "northern row e3v_0 wrongly zeroed",
                 "bit_identical": north_bad == 0,
                 "n_differing": north_bad, "max_abs_difference": 0.0})
    # The same question for the east column, which the round-213 repair got
    # wrong by a periodic WRAP where this mesh is closed.
    east_bad = int(np.count_nonzero(
        np.asarray(resolved.e3u_0)[:, -1] != nemo["e3u_0"][..., :nlev][:, -1]))
    rows.append({"field": "east column e3u_0 differing",
                 "bit_identical": east_bad == 0,
                 "n_differing": east_bad, "max_abs_difference": 0.0})
    # Non-vacuity: if the alias were still in place these two rows would be
    # e3t_0, so the gate must see them DIFFER from e3t_0 somewhere.
    n_alias = int(np.count_nonzero(
        np.asarray(resolved.e3u_0) != nemo["e3t_0"][..., :nlev]))
    # "bit_identical" means "this row is as it must be", which for a
    # non-vacuity row means the counter is NON-zero; the field is named so it
    # cannot be read as "1 164 cells differ and that is bit-identical".
    rows.append({"field": "non-vacuity: n cells where resolved e3u_0 != e3t_0",
                 "bit_identical": n_alias > 0, "row_is_a_counter": True,
                 "n_differing": n_alias, "max_abs_difference": 0.0})
    # The field section 6 of the round-220 receipt rests on: legoESM's
    # implicit-solve divisor e3w_0, which nemo_e3w_kmm reads off the card
    # when the coordinate carries a mesh reference.
    card_e3w0 = getattr(card.recipe.z_coord, "nemo_e3w_0", None)
    require(card_e3w0 is not None,
            f"{case}: the card carries no nemo_e3w_0 for the implicit solve")
    row("card e3w_0 vs NEMO e3w_1d",
        np.asarray(card_e3w0)[..., :nlev],
        np.broadcast_to(nemo["e3w_1d"][:nlev],
                        np.asarray(card_e3w0)[..., :nlev].shape))
    row("card umask", np.asarray(ops.umask), nemo["umask"][..., :nlev])
    row("card vmask", np.asarray(ops.vmask), nemo["vmask"][..., :nlev])
    tmask_card = (np.asarray(card.recipe.z_coord.is_active)
                  .astype(np.float64))
    row("card tmask (is_active)", tmask_card, nemo["tmask"][..., :nlev])
    return {"case": case, "rows": rows,
            "all_bit_identical": all(r["bit_identical"] for r in rows)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records-root", type=Path, required=True)
    p.add_argument("--json", type=Path)
    # The mini-ladder's rungs (decision 93) live under their own round
    # directory, so a run that scores one card cannot share a records root
    # with the others.  Default: every case, exactly as before.
    p.add_argument("--case", choices=tuple(CASES), action="append")
    a = p.parse_args()
    out = {}
    ok = True
    selected = {c: CASES[c] for c in (a.case or list(CASES))}
    for case, build in selected.items():
        res = compare(case, a.records_root / build / "kt1_10")
        out[case] = res
        ok = ok and res["all_bit_identical"]
        print(f"=== {case} ===")
        for r in res["rows"]:
            print(f"  {r['field']:28s} {'EXACT' if r['bit_identical'] else 'DIFFERS'}"
                  f"  n_diff={r['n_differing']:6d}  max={r['max_abs_difference']}")
    out["verdict"] = "GEOMETRY IDENTICAL" if ok else "GEOMETRY DIFFERS"
    print(out["verdict"])
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
