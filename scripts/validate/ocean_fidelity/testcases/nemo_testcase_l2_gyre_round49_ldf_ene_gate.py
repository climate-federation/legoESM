#!/usr/bin/env python3
"""Round-49 GYRE kt=2 LDF/ENE operand and statement discriminator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import _surface_forcings, require
from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
    ROOT, _owned2, _owned3, read_stage, sha256,
)
from nemo_testcase_l2_gyre_round40_stage3_operators import read_stage3_terms


def _u_face(a):
    a = _owned3(a)
    return np.concatenate([a[:, -1:, :], a], axis=1)


def _v_face(a, *, fill=0.0):
    a = _owned3(a)
    return np.concatenate([np.full_like(a[:1], fill), a], axis=0)


def _u_metric(a):
    a = _owned2(a)
    return np.concatenate([a[:, -1:], a], axis=1)


def _v_metric(a):
    a = _owned2(a)
    return np.concatenate([np.zeros_like(a[:1]), a], axis=0)


def _score(name, got, ref, mask):
    full_delta = np.asarray(got) - np.asarray(ref)
    delta = full_delta[mask]
    indices = np.argwhere((full_delta != 0.0) & mask)
    return {
        "name": name,
        "n": int(mask.sum()),
        "n_unequal": int(np.count_nonzero(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "posthoc_first_indices": indices[:8].tolist(),
        "posthoc_unequal_by_lat": np.count_nonzero(
            (full_delta != 0.0) & mask, axis=(1, 2)).tolist(),
        "posthoc_unequal_by_lon": np.count_nonzero(
            (full_delta != 0.0) & mask, axis=(0, 2)).tolist(),
    }


def run(root: Path, *, expect_commit: str, plant: str | None,
        claim: str = "all") -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "round49 requires x64")
    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(stamp["clean"] and stamp["commit"].lower() == expected,
            f"commit stamp mismatch: {stamp} != {expected}")

    records = {}
    hashes = {}
    for stage in (1, 2, 3):
        path = root / f"oracle_momstage_kt00000002_s{stage}.bin"
        parse_plant = plant if stage == 1 and plant in {"header", "truncation"} else None
        records[stage] = read_stage(path, plant=parse_plant)["arrays"]
        hashes[path.name] = sha256(path)
    stage3_terms_path = root / "oracle_rkstage3_terms_kt00000002.bin"
    stage3_terms = read_stage3_terms(stage3_terms_path, expect_kt=2)["arrays"]
    hashes[stage3_terms_path.name] = sha256(stage3_terms_path)

    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    ocean = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    rows = []
    for stage, a in records.items():
        ju, jv = _u_face(a["u_Kmm"]), _v_face(a["v_Kmm"])
        jub, jvb = _u_face(a["u_Kbb"]), _v_face(a["v_Kbb"])
        st = card.recipe.initial_state._replace(
            u=card.recipe.initial_state.u.replace(data=jnp.asarray(ju)),
            v=card.recipe.initial_state.v.replace(data=jnp.asarray(jv)),
            T=card.recipe.initial_state.T.replace(data=jnp.asarray(_owned3(a["T_Kmm"]))),
            S=card.recipe.initial_state.S.replace(data=jnp.asarray(_owned3(a["S_Kmm"]))),
            eta=card.recipe.initial_state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kmm"]))),
        )
        _, surface = _surface_forcings(card, st, 2)
        hu_mm, hv_mm = _u_face(a["e3u_Kmm"]), _v_face(a["e3v_Kmm"], fill=1.0)
        hu_bb, hv_bb = _u_face(a["e3u_Kbb"]), _v_face(a["e3v_Kbb"], fill=1.0)
        e3t_bb = _owned3(a["e3t_Kbb"])
        e3f = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
            st.eta.data, card.recipe.z_coord, jnp.float64, nn_e3f_typ=0))
        if stage == 3:
            # stprk3_stg.f90:256 materializes r3f(Kmm) as a source-ordered
            # interpolation of r3fb/r3fa.  Re-diagnosing it from interpolated
            # ssh is not bitwise equivalent; inject the admitted consumed
            # e3f operand instead.  Native F(i,j) maps to vertex [j+1,i+1].
            e3f = np.array(e3f, copy=True)
            e3f[1:, 1:, :30] = stage3_terms["e3f_vor_Kmm"][..., :30]
        ldf_ops = (e3t_bb, hu_bb, hv_bb, e3f, hu_mm, hv_mm)
        reciprocal = (_u_metric(a["r1_e1u"]), _v_metric(a["r1_e2v"]))
        if plant == "ldf" and stage == 1:
            ldf_ops = tuple(np.array(x, copy=True) for x in ldf_ops)
            wet_thickness = ldf_ops[0] > 0.0
            require(bool(np.any(wet_thickness)), "LDF plant has no wet target")
            ldf_ops[0][wet_thickness] *= 2.0
        if plant == "ene" and stage == 2:
            reciprocal = tuple(np.array(x, copy=True) for x in reciprocal)
            # Deliberately corrupt every positive U-face reciprocal.  The
            # earlier single-element plant landed on an unscored boundary;
            # this mask necessarily intersects the admitted wet U rows.
            wet_reciprocal = reciprocal[0] > 0.0
            require(bool(np.any(wet_reciprocal)), "ENE plant has no wet target")
            reciprocal[0][wet_reciprocal] *= 2.0

        def evaluate(state, exact_ldf, exact_ene):
            return ocean.tendencies(
                state, surface, dt=card.dt_s, momentum_only=True,
                skip_lateral_viscosity=(stage == 2),
                ldf_state=(state.T.data, state.S.data,
                           jnp.asarray(jub), jnp.asarray(jvb)),
                momentum_flux_face_thickness=(jnp.asarray(hu_mm), jnp.asarray(hv_mm)),
                ldf_thickness_operands=(
                    tuple(jnp.asarray(x) for x in ldf_ops) if exact_ldf else None),
                ene_metric_reciprocals=(
                    tuple(jnp.asarray(x) for x in reciprocal) if exact_ene else None),
                zad_continuity_dt=np.float64(1.0 / a["r1_Dt"]),
                nemo_operator_association=True,
                return_nemo_operator_components=True,
            )

        arms = (
            ("baseline", False, False),
            ("ldf_thickness", True, False),
            ("ldf_thickness_ene_reciprocal", True, True),
        )
        if claim == "ene":
            arms = arms[-1:]
        for arm, exact_ldf, exact_ene in arms:
            _, diagnostics, components = jax.jit(
                lambda state: evaluate(state, exact_ldf, exact_ene))(st)
            if stage in (1, 3):
                prior_name = "after_hpg" if stage == 1 else "after_adv"
                for face in ("u", "v"):
                    term = sum(np.asarray(getattr(diagnostics, f"{name}_{face}").data)
                               for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith"))
                    term = term[:, 1:, :] if face == "u" else term[1:, :, :]
                    prior = _owned3(a[f"{prior_name}_{face}"])
                    got = np.asarray(jax.jit(
                        lambda x, y: jax.lax.optimization_barrier(x + y))(
                            jnp.asarray(prior), jnp.asarray(term)))
                    ref = _owned3(a[f"after_ldf_{face}"])
                    mask = _owned3(a[f"{face}mask"]) > 0.5
                    rows.append(_score(
                        f"kt2.s{stage}.{arm}.post_ldf.{face}", got, ref, mask))
            for face in ("u", "v"):
                term = np.asarray(components[f"vorticity_{face}"].data)
                term = term[:, 1:, :] if face == "u" else term[1:, :, :]
                prior_name = "after_ldf" if stage == 1 else "after_hpg"
                prior = _owned3(a[f"{prior_name}_{face}"])
                got = np.asarray(jax.jit(
                    lambda x, y: jax.lax.optimization_barrier(x + y))(
                        jnp.asarray(prior), jnp.asarray(term)))
                ref = _owned3(a[f"after_vor_{face}"])
                mask = _owned3(a[f"{face}mask"]) > 0.5
                rows.append(_score(
                    f"kt2.s{stage}.{arm}.post_vor.{face}", got, ref, mask))

    final = [r for r in rows if ".ldf_thickness_ene_reciprocal." in r["name"]]
    if claim == "ene":
        final = [r for r in final if ".post_vor." in r["name"]]
    # LDF exact is evaluated only at stages 1/3; ENE exact at all stages.
    expected_rows = 6 if claim == "ene" else 10
    require(len(final) == expected_rows, f"missing final rows: {len(final)}")
    exact = all(r["n_unequal"] == 0 for r in final)
    if plant == "ldf":
        planted_rows = [r for r in final
                        if ".s1." in r["name"] and ".post_ldf." in r["name"]]
        require(planted_rows and max(r["max_abs"] for r in planted_rows) > 1.0e-12,
                "LDF thickness plant did not move a scored wet row")
    if plant == "ene":
        planted_rows = [r for r in final
                        if ".s2." in r["name"] and ".post_vor.u" in r["name"]]
        require(planted_rows and planted_rows[0]["max_abs"] > 1.0e-12,
                "ENE reciprocal plant did not move a scored wet row")
    return {
        "format": "nemo-testcase-l2-gyre-round49-ldf-ene-v1",
        "worktree": stamp,
        "record_sha256": hashes,
        "compiled_source": {
            "ldf": "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140",
            "ene": "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90:536-573",
        },
        "plant": plant,
        "claim": claim,
        "rows": rows,
        "status": ("PLANT_FIRED" if plant else
                   ("CONFIRMED" if exact else "REFUTED")),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--expect-commit", required=True)
    p.add_argument("--plant", choices=("header", "truncation", "stamp", "ldf", "ene"))
    p.add_argument("--claim", choices=("all", "ene"), default="all")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = run(args.root, expect_commit=args.expect_commit, plant=args.plant,
                 claim=args.claim)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
