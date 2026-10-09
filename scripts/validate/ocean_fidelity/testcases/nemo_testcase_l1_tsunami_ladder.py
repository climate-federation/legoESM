#!/usr/bin/env python3
"""TSUNAMI round 3 -- the kt=1..10 ladder against the RK3 record.

Two labels, never mixed (Decision 52):
  INDEPENDENT      the card's step chained from the card's own initial state;
  GIVEN-NEMO-ENTRY the card handed NEMO's recorded entry state (whole step),
                   or NEMO's recorded external handoff and stage entry
                   (stage-local), one boundary at a time.

One shared harness: the card and its model are the VORTEX ones (the shared
LatLonCGridOceanModel and its private ``_NEMOWSRK3TestHooks``), rows are scored
by the trajectory gate's ``score``, the substep boundaries and their order are
the VORTEX walk's ``SUBSTEP_ORDER``.  Only the TSUNAMI record reader is new,
and it is ``check_records.parse`` (the admission parser).

Arms (``--arm``): independent, given_entry, rhs (stp_2D right-hand side),
spgts (dyn_spg_ts substeps, the card's own entry forcing and NEMO's),
handoff (the external handoff, read at stage 1), stages (stage-local).

Every row: bit-identical, count and first unequal cell (row-major, 0-based
interior), max abs, rms, normalised max (trajectory gate's bar 1e-15).
``uu_b``/``vv_b`` of an exposed stage are the handed-in external values, so
they are listed as pass-through and not scored.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_phase3_trajectory_gate import BAR, require, score  # noqa: E402

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round2/"
    "oracle_tsunami_r2/p3")
ARMS = ("independent", "given_entry", "rhs", "spgts", "handoff", "stages",
        "b6_scaling", "front")
B6_KTS = (2, 5, 10)
B6_LAMBDAS = (0.0, 0.25, 0.5, 1.0)
PLANTS = ("score", "entry", "external", "stage_entry", "forcing")
_HALO = 2


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tools():
    cr = _load("tsunami_check_records",
               HERE / "nemo_testcase_l1_tsunami" / "check_records.py")
    walk = _load("vortex_spgts_walk",
                 HERE / "nemo_testcase_l1_vortex_round196_spgts_walk.py")
    return cr, walk


def interior(a):
    return np.asarray(a)[_HALO:-_HALO, _HALO:-_HALO]


def read_step(cr, root: Path, kt: int) -> dict:
    path = root / f"oracle_tsustep_kt{kt:08d}.bin"
    require(path.is_file(), f"missing {path}")
    head, groups = cr.parse(path, cr.STEP_MAGIC, cr.STEP_NINT,
                            cr.STEP_BOUNDS_AT)
    require(head[1] == kt, f"{path}: header step {head[1]}")
    return {k: interior(v) for k, v in groups.items()}


SEAM_PAD = "wrap"      # --seam-pad: the redundant west/south face record


def pad_u(a):
    """One redundant west record: the periodic wrap of the east face."""
    a = np.asarray(a)
    first = a[:, -1:] if SEAM_PAD == "wrap" else np.zeros_like(a[:, :1])
    return np.concatenate([first, a], axis=1)


def pad_v(a):
    a = np.asarray(a)
    first = a[-1:] if SEAM_PAD == "wrap" else np.zeros_like(a[:1])
    return np.concatenate([first, a], axis=0)


def bump(a, size=1.0e-3):
    a = np.array(a, dtype=np.float64)
    a[tuple(d // 2 for d in a.shape)] += size
    return a


def row(name, ref, cand, *, plant=False):
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    r = score(name, ref, cand, np.ones(ref.shape, dtype=bool), plant=plant)
    d = cand - ref
    bad = np.argwhere(d != 0)
    r.update({
        "n_unequal": int(bad.shape[0]),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(np.abs(d))),
        "rms": float(np.sqrt(np.mean(d * d))),
        "ref_max_abs": float(np.max(np.abs(ref))),
        "bit_identical": bool(bad.shape[0] == 0 and not plant),
    })
    return r


def first_unequal(rows):
    for r in rows:
        if not r["bit_identical"]:
            return {k: r[k] for k in ("name", "n_unequal", "first_unequal_cell",
                                      "max_abs", "normalized_max_abs", "status")}
    return None


def state_fields(st):
    return {
        "ssh": np.asarray(st.eta.data),
        "uu_b": np.asarray(st.uu_b.data)[:, 1:],
        "vv_b": np.asarray(st.vv_b.data)[1:, :],
        "u": np.asarray(st.u.data)[:, 1:, 0],
        "v": np.asarray(st.v.data)[1:, :, 0],
        "T": np.asarray(st.T.data)[..., 0],
        "S": np.asarray(st.S.data)[..., 0],
    }


def after_step_reference(cr, root, kt, g):
    """NEMO's after-step state of step kt: f_ groups; T, S from entry kt+1."""
    ref = {"ssh": g["f_ssh_bb"], "uu_b": g["f_uu_b_bb"], "vv_b": g["f_vv_b_bb"],
           "u": g["f_uu_k1_bb"], "v": g["f_vv_k1_bb"]}
    if kt < cr.FULL_STEPS:
        nxt = read_step(cr, root, kt + 1)
        ref["T"], ref["S"] = nxt["e_tn_k1_bb"], nxt["e_sn_k1_bb"]
    return ref


def run(root: Path, *, arm: str, kt_max: int = 10, plant: str | None = None,
        allow_dirty: bool = False, eos_depth: str | None = None) -> dict:
    require(arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}")
    require(plant is None or plant in PLANTS,
            f"unknown plant {plant!r}; expected one of {PLANTS}")
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.halo_latlon import meridional_periodicity
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_tsunami_zco_card)
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "the ladder must run on CPU")
    cr, walk = _tools()
    if arm == "front":
        rows = []
        for kt in range(1, kt_max + 1):
            a = np.abs(read_step(cr, root, kt)["f_ssh_bb"]) > 1e-12
            j, i = np.where(a.any(1))[0], np.where(a.any(0))[0]
            rows.append({"kt": kt, "j_range": [int(j.min()), int(j.max())],
                         "i_range": [int(i.min()), int(i.max())],
                         "n_cells": int(a.sum()),
                         "cells_to_j_seam": int(min(j.min(), a.shape[0] - 1 - j.max())),
                         "cells_to_i_seam": int(min(i.min(), a.shape[1] - 1 - i.max()))})
        return {"arm": arm, "label": "NEMO record only (|f_ssh_bb| > 1e-12)",
                "legoesm_git_sha": sha, "root": str(root), "per_kt": rows}
    card = build_tsunami_zco_card()
    s0 = card.recipe.initial_state
    nlev = int(card.recipe.z_coord.n_levels)
    require(nlev == 1, "the TSUNAMI card executes one level")
    kts = range(1, kt_max + 1)
    require(1 <= kt_max <= cr.FULL_STEPS,
            "kt_max must lie in 1..10 (the full-frame steps)")

    def seed(g, lam=1.0):
        eta = bump(g["e_ssh_bb"]) if plant == "entry" else g["e_ssh_bb"]
        sc = lambda a: lam * a      # noqa: E731 -- velocity scale (b6_scaling)
        return s0._replace(
            T=s0.T.replace(data=jnp.asarray(g["e_tn_k1_bb"][..., None])),
            S=s0.S.replace(data=jnp.asarray(g["e_sn_k1_bb"][..., None])),
            u=s0.u.replace(data=jnp.asarray(pad_u(sc(g["e_uu_k1_bb"]))[..., None])),
            v=s0.v.replace(data=jnp.asarray(pad_v(sc(g["e_vv_k1_bb"]))[..., None])),
            eta=s0.eta.replace(data=jnp.asarray(eta)),
            uu_b=s0.uu_b.replace(data=jnp.asarray(pad_u(sc(g["e_uu_b_bb"])))),
            vv_b=s0.vv_b.replace(data=jnp.asarray(pad_v(sc(g["e_vv_b_bb"])))))

    # MEASUREMENT ARM, not a card change: None runs the card as it ships.
    cfg = card.recipe.model_config
    if eos_depth is not None:
        require(eos_depth in ("insitu", "geometric"),
                f"unknown eos_depth {eos_depth!r}")
        cfg = cfg._replace(eos_depth=eos_depth)

    def model(hooks=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            **({} if hooks is None else {"_nemo_ws_test_hooks": hooks}))

    records = {kt: read_step(cr, root, kt) for kt in kts}
    seam = {kt: max(float(np.max(np.abs(records[kt][k][:, [0, -1]])))
                    for k in ("e_uu_k1_bb", "e_uu_b_bb"))
            for kt in kts}
    seam |= {f"{kt}_j": max(float(np.max(np.abs(records[kt][k][[0, -1], :])))
                            for k in ("e_vv_k1_bb", "e_vv_b_bb"))
             for kt in kts}
    out = {"arm": arm, "label": None, "legoesm_git_sha": sha, "plant": plant,
           "seam_pad": SEAM_PAD,
           "eos_depth_arm": eos_depth or f"card's own ({card.recipe.model_config.eos_depth})",
           "bar": BAR, "root": str(root), "kts": list(kts),
           "seam_column_max_abs_velocity_in_entry": {
               str(k): v for k, v in seam.items()},
           "per_kt": []}

    with meridional_periodicity(card.j_periodic):
        if arm in ("independent", "given_entry"):
            m = model()
            out["label"] = ("INDEPENDENT" if arm == "independent"
                            else "GIVEN-NEMO-ENTRY (whole step)")
            state = s0
            for kt in kts:
                g = records[kt]
                ref = after_step_reference(cr, root, kt, g)
                state = m.step(state if arm == "independent" else seed(g),
                               dt=card.dt_s)
                cand = state_fields(state)
                rows = []
                for f in ("ssh", "uu_b", "vv_b", "u", "v", "T", "S"):
                    if f not in ref:
                        continue
                    c = cand[f].copy()
                    if plant == "score" and kt == 1 and f == "ssh":
                        c = bump(c)
                    rows.append(row(f"{arm}.kt{kt}.{f}", ref[f], c))
                out["per_kt"].append({"kt": kt, "rows": rows,
                                      "first_unequal": first_unequal(rows)})
        elif arm == "rhs":
            out["label"] = "GIVEN-NEMO-ENTRY (stp_2D right-hand side)"
            for kt in kts:
                g = records[kt]
                seen = {}
                for face in ("u", "v"):
                    h = _NEMOWSRK3TestHooks(
                        slow_forcing_rhs_observer=(
                            lambda v, face=face: seen.__setitem__(
                                face, np.asarray(v))),
                        slow_forcing_rhs_observer_face=face)
                    jax.block_until_ready(model(h).step(seed(g), dt=card.dt_s))
                    jax.effects_barrier()   # the observer is an async callback
                rows = [row(f"rhs.kt{kt}.u", g["b_uu_rhs_k1"],
                            seen["u"][:, 1:, 0]),
                        row(f"rhs.kt{kt}.v", g["b_vv_rhs_k1"],
                            seen["v"][1:, :, 0])]
                out["per_kt"].append({"kt": kt, "rows": rows,
                                      "first_unequal": first_unequal(rows)})
        elif arm == "b6_scaling":
            out["label"] = ("GIVEN-NEMO-ENTRY (stp_2D right-hand side, entry "
                            "velocity scaled by lambda; T, S, ssh unscaled)")
            require(eos_depth == "geometric",
                    "b6_scaling needs --eos-depth geometric: the pressure "
                    "term must already agree with the replay")
            import nemo_testcase_l1_tsunami_hpg_replay as rep
            c = rep.parse_constants(root.parent / "ref" / "ocean.output")
            mesh = rep.mesh_from_mask(root / "mesh_mask.nc")
            seen = {}
            h = _NEMOWSRK3TestHooks(
                slow_forcing_rhs_observer=lambda v: seen.__setitem__(
                    "u", np.asarray(v)),
                slow_forcing_rhs_observer_face="u")
            m = model(h)
            for kt in B6_KTS:
                g = records[kt]
                hpg, _ = rep.replay_from_record(g, c, mesh)
                vor = g["b_uu_rhs_k1"] - hpg
                res = []
                for lam in B6_LAMBDAS:
                    seen.clear()
                    jax.block_until_ready(m.step(seed(g, lam), dt=card.dt_s))
                    jax.effects_barrier()   # the observer is an async callback
                    require("u" in seen, "the right-hand-side observer never fired")
                    pred = hpg + lam * vor
                    res.append({"lambda": lam, "max_abs_card_minus_prediction":
                                float(np.max(np.abs(seen["u"][:, 1:, 0] - pred)))})
                e = {r["lambda"]: r["max_abs_card_minus_prediction"] for r in res}
                out["per_kt"].append({
                    "kt": kt, "rows": [], "lambdas": res,
                    "ratio_E1_over_E0.5": e[1.0] / e[0.5] if e[0.5] else None,
                    "ratio_E0.5_over_E0.25": e[0.5] / e[0.25] if e[0.25] else None,
                    "reading": "ratio 4 = quadratic in velocity (momentum "
                               "advection); ratio 2 = linear (Coriolis/vorticity)"})
        elif arm == "spgts":
            out["label"] = ("GIVEN-NEMO-ENTRY (dyn_spg_ts substeps; 'own' = "
                            "the card's entry forcing, 'nemo' = NEMO's)")
            for kt in kts:
                g = records[kt]
                meta, gr = walk.read_spgts(root, kt)
                for jn in range(1, meta["icycle"] + 1):
                    pre = f"j{jn:03d}_"
                    prev = f"j{jn - 1:03d}_"
                    gr[pre + "sshn_e_eff"] = (gr["i000_sshn_e"] if jn == 1
                                              else gr[prev + "ssha_e"])
                    gr[pre + "un_e_eff"] = (gr["i000_un_e"] if jn == 1
                                            else gr[prev + "ua_new"])
                    gr[pre + "vn_e_eff"] = (gr["i000_vn_e"] if jn == 1
                                            else gr[prev + "va_new"])
                entry = {"kt": kt, "icycle": meta["icycle"]}
                for label in ("own", "nemo"):
                    override = None
                    if label == "nemo":
                        zu = gr["i000_zu_frc"]
                        if plant == "forcing":
                            zu = bump(zu, 1.0e-6)
                        override = (jnp.asarray(pad_u(zu)),
                                    jnp.asarray(pad_v(gr["i000_zv_frc"])))
                    h = _NEMOWSRK3TestHooks(
                        expose_barotropic_substeps=True,
                        barotropic_slow_forcing_override=override)
                    res = jax.device_get(model(h).step(seed(g), dt=card.dt_s))
                    tr = {k: np.asarray(v) for k, v in res.substeps.items()}
                    require(int(next(iter(tr.values())).shape[0])
                            == meta["icycle"],
                            f"kt {kt}: card ran a different substep count")
                    rows, first = [], None
                    for name, grp, key, stg in walk.ENTRY_ORDER:
                        rows.append(row(f"spgts.{label}.kt{kt}.{name}",
                                        gr["i000_" + grp],
                                        walk._lego_plane(tr[key][0], stg)))
                    for jn in range(1, meta["icycle"] + 1):
                        for name, grp, key, stg in walk.SUBSTEP_ORDER:
                            r = row(f"spgts.{label}.kt{kt}.j{jn:03d}.{name}",
                                    gr[f"j{jn:03d}_" + grp],
                                    walk._lego_plane(tr[key][jn - 1], stg))
                            rows.append(r)
                    unequal = [r for r in rows if not r["bit_identical"]]
                    entry[label] = {
                        "first_unequal": first_unequal(rows),
                        "n_unequal_boundaries": len(unequal),
                        "n_boundaries": len(rows),
                        "n_over_bar": sum(r["status"] == "DEBT" for r in rows),
                        "max_normalized_over_all_boundaries": max(
                            r["normalized_max_abs"] for r in rows),
                        "rows_unequal": [
                            {k: r[k] for k in ("name", "n_unequal", "max_abs",
                                               "normalized_max_abs", "status")}
                            for r in unequal[:12]],
                    }
                out["per_kt"].append(entry)
        elif arm == "handoff":
            out["label"] = ("GIVEN-NEMO-ENTRY (external handoff read at "
                            "stage 1: uu_b/vv_b(Kaa) = the N+1 values, "
                            "ssh(Kaa) = 2/3 ssh(Kbb) + 1/3 N+1)")
            for kt in kts:
                g = records[kt]
                h = _NEMOWSRK3TestHooks(expose_momentum_stage=1,
                                        expose_tracer_stage=1)
                cand = state_fields(model(h).step(seed(g), dt=card.dt_s))
                rows = [row(f"handoff.kt{kt}.{f}", g[nm], cand[f])
                        for f, nm in (("uu_b", "1_uu_b_aa"),
                                      ("vv_b", "1_vv_b_aa"),
                                      ("ssh", "1_ssh_aa"))]
                out["per_kt"].append({"kt": kt, "rows": rows,
                                      "first_unequal": first_unequal(rows)})
        else:   # stages
            out["label"] = ("GIVEN-NEMO-ENTRY (stage-local: NEMO's external "
                            "handoff and NEMO's stage entry)")
            for kt in kts:
                g = records[kt]
                ssh_ext = bump(g["b_ssh_aa"]) if plant == "external" \
                    else g["b_ssh_aa"]
                ext = (jnp.asarray(ssh_ext), jnp.asarray(pad_u(g["b_uu_b_aa"])),
                       jnp.asarray(pad_v(g["b_vv_b_aa"])),
                       jnp.asarray(pad_u(g["b_un_adv"])),
                       jnp.asarray(pad_v(g["b_vn_adv"])))
                stages = []
                for stage in (1, 2, 3):
                    ent = None
                    if stage > 1:
                        p = f"{stage - 1}_"
                        u_in = g[p + "uu_k1_aa"]
                        if plant == "stage_entry" and stage == 2:
                            u_in = bump(u_in)
                        ent = (stage, jnp.asarray(pad_u(u_in)[..., None]),
                               jnp.asarray(pad_v(g[p + "vv_k1_aa"])[..., None]),
                               jnp.asarray(g[p + "tn_k1_aa"][..., None]),
                               jnp.asarray(g[p + "sn_k1_aa"][..., None]),
                               jnp.asarray(g[p + "ssh_aa"]))
                    last = stage == 3
                    h = _NEMOWSRK3TestHooks(
                        stage_barotropic_output_override=ext,
                        stage_entry_override=ent,
                        expose_momentum_stage=0 if last else stage,
                        expose_tracer_stage=0 if last else stage)
                    cand = state_fields(model(h).step(seed(g), dt=card.dt_s))
                    p = f"{stage}_"
                    rows = [row(f"stage{stage}.kt{kt}.{f}", g[p + nm], cand[f])
                            for f, nm in (("ssh", "ssh_aa"), ("u", "uu_k1_aa"),
                                          ("v", "vv_k1_aa"), ("T", "tn_k1_aa"),
                                          ("S", "sn_k1_aa"))]
                    stages.append({"stage": stage, "rows": rows,
                                   "first_unequal": first_unequal(rows),
                                   "passthrough_not_scored": ["uu_b", "vv_b"]})
                out["per_kt"].append({"kt": kt, "stages": stages})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--kt-max", type=int, default=10)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plant", choices=PLANTS)
    ap.add_argument("--seam-pad", choices=("wrap", "zero"), default="wrap",
                    help="value of the redundant west/south face record when "
                         "seeding (default: the periodic wrap)")
    ap.add_argument("--eos-depth", choices=("insitu", "geometric"),
                    help="measurement arm: override the card's eos_depth for "
                         "this run only (the card itself is not changed)")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    global SEAM_PAD
    SEAM_PAD = a.seam_pad
    try:
        out = run(a.root, arm=a.arm, kt_max=a.kt_max, plant=a.plant,
                  allow_dirty=a.allow_dirty, eos_depth=a.eos_depth)
    except Exception as e:      # noqa: BLE001 -- refusals print and exit 2
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for k in out["per_kt"]:
        if "stages" in k:
            for s in k["stages"]:
                fu = s["first_unequal"]
                print(f"kt{k['kt']} stage{s['stage']} first unequal: "
                      f"{fu['name'] if fu else 'NONE (bit-identical)'}")
        elif "own" in k:
            for lab in ("own", "nemo"):
                fu = k[lab]["first_unequal"]
                print(f"kt{k['kt']} {lab}: "
                      f"{fu['name'] if fu else 'ALL BOUNDARIES BIT-IDENTICAL'} "
                      f"({k[lab]['n_unequal_boundaries']}/"
                      f"{k[lab]['n_boundaries']} unequal)")
        elif "j_range" in k:
            print(f"kt{k['kt']} j {k['j_range']} i {k['i_range']} "
                  f"cells to j-seam {k['cells_to_j_seam']}")
        elif "lambdas" in k:
            print(f"kt{k['kt']} E(lambda): " + ", ".join(
                f"{r['lambda']}:{r['max_abs_card_minus_prediction']:.3e}"
                for r in k["lambdas"])
                + f"  E(1)/E(.5)={k['ratio_E1_over_E0.5']:.6f}")
        else:
            fu = k["first_unequal"]
            print(f"kt{k['kt']} first unequal: "
                  f"{fu['name'] if fu else 'NONE (bit-identical)'}"
                  + (f" max={fu['max_abs']:.3e}" if fu else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
