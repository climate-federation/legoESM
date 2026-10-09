#!/usr/bin/env python3
"""SMT-RUNGS round 3: SMT-5 (T/S damping) inputs, ladders and damping chain.

Three sections, one report (all fp64 libm, CPU):

``inputs``  SMT-5 mesh_mask equals SMT-4's, bit for bit, in every variable;
            the card's loaded damping arrays equal the files NEMO read (read
            here independently with plain netCDF4); target record 1 equals the
            analytical initial T on wet cells, S = 35 tmask, 12 records
            identical, resto = tmask/86400; the three files are byte-equal
            across the smoke / kt / 100-day arms.
``ladder``  kt=1..10 per field first unequal cell / n unequal / max abs / rms,
            for SMT-5 and the SMT-4 control, under both labels:
            INDEPENDENT (legoESM's own trajectory) and GIVEN-NEMO-ENTRY (one
            legoESM step from NEMO's entry kt scored against NEMO's entry
            kt+1).  The two labels are never mixed.
``damping`` per given-entry step, legoESM damping ON minus OFF (one variable:
            ``nemo_tracer_damping``) is legoESM's increment; where NEMO's
            SMT-5 and SMT-4 entries are bit-equal, NEMO's SMT-5 minus SMT-4
            next entry is NEMO's increment, and the two are compared cell by
            cell.

``--plant`` moves one wet T cell one ULP in the independent kt=1 row (bit-exact
unplanted) and the
run must exit nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    expected_masks, lego_fields, read_entry, require,
)
from nemo_testcase_l1_vortex_kt2_walk import _seed_from_record  # noqa: E402

ROUND2 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
              "smtrungs_rounds/round2/oracle_vortex_smt5")
SMT4 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
            "oracle_vortex_smt4/kt1_10")
SMT4_MESH = SMT4 / "mesh_mask.nc"
FIELDS = ("T", "S", "u", "v", "ssh")
FILES = ("data_1m_potential_temperature_nomask.nc",
         "data_1m_salinity_nomask.nc", "resto.nc")


def field_stats(reference, candidate, mask) -> dict:
    """Unequal-cell count, first unequal cell (C order), max abs, rms."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == candidate.shape == mask.shape, "shape mismatch")
    if not mask.any():
        mask = np.ones(mask.shape, dtype=bool)
    require(bool(np.all(np.isfinite(candidate[mask]))), "candidate nonfinite")
    unequal = (reference != candidate) & mask
    diff = (candidate - reference)[mask]
    first = np.argwhere(unequal)
    return {
        "n": int(mask.sum()),
        "n_unequal": int(unequal.sum()),
        "first_unequal_cell": ([int(i) for i in first[0]]
                               if len(first) else None),
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(math.sqrt(float(np.mean(diff ** 2)))),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_section() -> dict:
    import netCDF4
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    rows = {}
    mesh = []
    with netCDF4.Dataset(ROUND2 / "kt1_10" / "mesh_mask.nc") as a, \
            netCDF4.Dataset(SMT4_MESH) as b:
        require(sorted(a.variables) == sorted(b.variables),
                "mesh_mask variable lists differ")
        for name in sorted(a.variables):
            x, y = np.asarray(a.variables[name][:]), np.asarray(b.variables[name][:])
            same = x.shape == y.shape and bool(np.array_equal(x, y))
            mesh.append({"var": name, "shape": list(x.shape), "bit_identical": same})
    rows["mesh_mask_smt5_vs_smt4"] = {
        "n_variables": len(mesh),
        "n_differing": sum(not m["bit_identical"] for m in mesh),
        "differing": [m["var"] for m in mesh if not m["bit_identical"]],
    }
    hashes = {d: {f: sha256(ROUND2 / d / f) for f in FILES}
              for d in ("smoke", "kt1_10", "day100")}
    rows["files_identical_across_arms"] = all(
        hashes[d] == hashes["kt1_10"] for d in hashes)
    rows["file_sha256"] = hashes["kt1_10"]

    root = ROUND2 / "kt1_10"
    card = build_nemo_testcase_card("VORTEX_SMT5_VEC-zps", deck_root=root)
    dmp = card.recipe.model_config.nemo_tracer_damping
    nlev = int(card.recipe.z_coord.n_levels)
    wet3 = np.asarray(card.recipe.z_coord.is_active)
    with netCDF4.Dataset(root / FILES[0]) as h:
        raw_T = np.asarray(h.variables["votemper"][:], dtype=np.float64)
    with netCDF4.Dataset(root / FILES[1]) as h:
        raw_S = np.asarray(h.variables["vosaline"][:], dtype=np.float64)
    with netCDF4.Dataset(root / FILES[2]) as h:
        raw_r = np.asarray(h.variables["resto"][:], dtype=np.float64)
    tmask = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
    T0 = np.asarray(card.recipe.initial_state.T.data)
    S0 = np.asarray(card.recipe.initial_state.S.data)
    rows["card_arrays_equal_files"] = {
        "target_T": bool(np.array_equal(
            np.asarray(dmp.target_T), np.moveaxis(raw_T[:, :nlev], 1, -1))),
        "target_S": bool(np.array_equal(
            np.asarray(dmp.target_S), np.moveaxis(raw_S[:, :nlev], 1, -1))),
        "resto": bool(np.array_equal(
            np.asarray(dmp.resto), np.moveaxis(raw_r[:nlev], 0, -1))),
    }
    rows["records_identical"] = bool(
        np.all(raw_T == raw_T[:1]) and np.all(raw_S == raw_S[:1]))
    t1 = np.moveaxis(raw_T[0, :nlev], 0, -1)
    s1 = np.moveaxis(raw_S[0, :nlev], 0, -1)
    rows["target_T_vs_initial_T_wet"] = field_stats(T0, t1, wet3)
    rows["target_S_vs_35_wet"] = field_stats(np.full_like(S0, 35.0), s1, wet3)
    rows["initial_S_vs_35_wet"] = field_stats(np.full_like(S0, 35.0), S0, wet3)
    rows["resto_vs_tmask_over_86400"] = field_stats(
        tmask / 86400.0, np.moveaxis(raw_r[:nlev], 0, -1),
        np.ones(wet3.shape, bool))
    ok = (rows["mesh_mask_smt5_vs_smt4"]["n_differing"] == 0
          and rows["files_identical_across_arms"]
          and all(rows["card_arrays_equal_files"].values())
          and rows["records_identical"]
          and rows["target_T_vs_initial_T_wet"]["n_unequal"] == 0
          and rows["target_S_vs_35_wet"]["n_unequal"] == 0
          and rows["initial_S_vs_35_wet"]["n_unequal"] == 0
          and rows["resto_vs_tmask_over_86400"]["n_unequal"] == 0)
    rows["status"] = "IDENTICAL" if ok else "DIFFERS"
    return rows


def _models(deck_root: Path):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card5 = build_nemo_testcase_card("VORTEX_SMT5_VEC-zps", deck_root=deck_root)
    card4 = build_nemo_testcase_card("VORTEX_SMT4_VEC-zps")

    def build(card, cfg=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config if cfg is None else cfg)

    off_cfg = card5.recipe.model_config._replace(nemo_tracer_damping=None)
    return {"card5": card5, "card4": card4,
            "m5": build(card5), "m4": build(card4),
            "m5_off": build(card5, off_cfg)}


def _read(root: Path, case: str, interior, nlev, dummy, kt):
    e = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", case,
                   expect_interior=interior)
    require(e["step"] == kt and e["nz"] == nlev + dummy, f"entry {kt} shape")
    return e


def ladder_and_damping(deck_root: Path, plant: bool) -> dict:
    ms = _models(deck_root)
    out = {"independent": {}, "given_entry": {}, "damping": {}}
    entries = {}
    for tag, case, root in (("smt5", "VORTEX_SMT5_VEC-zps", ROUND2 / "kt1_10"),
                            ("smt4", "VORTEX_SMT4_VEC-zps", SMT4)):
        card = ms["card" + tag[-1]]
        model = ms["m" + tag[-1]]
        damped = tag == "smt5"
        masks = expected_masks(card)
        nlev = int(card.recipe.z_coord.n_levels)
        dummy = card.dummy_bottom_records
        interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
        dt = card.dt_s
        ent = {kt: _read(root, case, interior, nlev, dummy, kt)
               for kt in range(1, 11)}
        entries[tag] = ent

        def advance(state, kt):
            return model.step(
                state, dt=dt,
                **({"t_seconds": (kt - 1) * dt} if damped else {}))

        def ref(e, f):
            r = np.asarray(e[f])
            return r if f == "ssh" else r[..., :nlev]

        ind, giv = {}, {}
        state = card.recipe.initial_state
        for kt in range(1, 11):
            cand = lego_fields(state)
            if plant and tag == "smt5" and kt == 1:
                cand = dict(cand)
                T = np.array(cand["T"])
                where = tuple(np.argwhere(masks["T"])[0])
                T[where] = np.nextafter(T[where], np.inf)
                cand["T"] = T
            ind[kt] = {f: field_stats(ref(ent[kt], f), cand[f], masks[f])
                       for f in FIELDS}
            if kt < 10:
                state = advance(state, kt)
        # The model carries barotropic/time-filter history that a NEMO entry
        # record does not hold, so a seed built on the card's INITIAL state is
        # valid at kt=1 only.  Every later step re-seeds the prognostic
        # fields on the carried (stepped) state and keeps its history.
        carried = card.recipe.initial_state
        for kt in range(1, 10):
            seed = _seed_from_record(carried, ent[kt], nlev)
            carried = advance(seed, kt)
            cand = lego_fields(carried)
            giv[kt] = {f: field_stats(ref(ent[kt + 1], f), cand[f], masks[f])
                       for f in FIELDS}
        out["independent"][tag] = ind
        out["given_entry"][tag] = giv

    # damping chain: SMT-5 NEMO entries, damping ON vs OFF, one step each.
    card5 = ms["card5"]
    nlev = int(card5.recipe.z_coord.n_levels)
    masks = expected_masks(card5)
    dt = card5.dt_s
    e5, e4 = entries["smt5"], entries["smt4"]
    carried = card5.recipe.initial_state
    for kt in range(1, 10):
        seed = _seed_from_record(carried, e5[kt], nlev)
        carried = ms["m5"].step(seed, dt=dt, t_seconds=(kt - 1) * dt)
        on = lego_fields(carried)
        off = lego_fields(ms["m5_off"].step(seed, dt=dt))
        entry_equal = all(np.array_equal(e5[kt][f], e4[kt][f]) for f in FIELDS)
        row = {"nemo_entry_smt5_equals_smt4": entry_equal}
        for f in ("T", "S"):
            d_lego = on[f] - off[f]
            row[f"lego_increment_{f}"] = {
                "n_nonzero": int(np.count_nonzero(d_lego[masks[f]])),
                "max_abs": float(np.max(np.abs(d_lego[masks[f]]))),
            }
            if entry_equal:
                d_nemo = (e5[kt + 1][f] - e4[kt + 1][f])[..., :nlev]
                row[f"nemo_increment_{f}"] = {
                    "n_nonzero": int(np.count_nonzero(d_nemo[masks[f]])),
                    "max_abs": float(np.max(np.abs(d_nemo[masks[f]]))),
                }
                row[f"increment_lego_vs_nemo_{f}"] = field_stats(
                    d_nemo, d_lego, masks[f])
                spacing = np.spacing(np.abs(e5[kt + 1][f][..., :nlev]))
                ratio = (np.abs(d_nemo - d_lego) / spacing)
                row[f"increment_diff_in_ulps_{f}"] = float(
                    np.max(ratio[masks[f]]))
                # Four roundings bound a same-binade difference by 2 ulp; a
                # cell above that is either a binade crossing or a real
                # sub-ulp difference.  Report both T4 and T5 binades.
                over = (ratio > 2.0) & masks[f]
                a4 = np.abs(e4[kt + 1][f][..., :nlev])[over]
                a5 = np.abs(e5[kt + 1][f][..., :nlev])[over]
                row[f"cells_over_2ulp_{f}"] = {
                    "n": int(over.sum()),
                    "crosses_binade": int(np.count_nonzero(
                        np.floor(np.log2(np.where(a4 > 0, a4, 1.0)))
                        != np.floor(np.log2(np.where(a5 > 0, a5, 1.0))))),
                }
        out["damping"][kt] = row
        if kt == 2:
            # Absorption law of the S increment: lego's S response to the
            # damping rate scaled by k (one variable: resto), against the
            # count of cells whose S differs from 35 (nonzero rate).
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            cfg = card5.recipe.model_config
            dmp = cfg.nemo_tracer_damping
            nonzero_rate = int(np.count_nonzero(
                (np.asarray(seed.S.data) != 35.0)[masks["S"]]))
            law = {"cells_with_S_not_35": nonzero_rate}
            for scale in (1.0, 10.0, 100.0):
                scaled = LatLonCGridOceanModel(
                    card5.recipe.grid, card5.recipe.z_coord,
                    cfg._replace(nemo_tracer_damping=dmp._replace(
                        resto=np.asarray(dmp.resto) * scale)))
                S_on = lego_fields(
                    scaled.step(seed, dt=dt, t_seconds=(kt - 1) * dt))["S"]
                law[f"scale_{scale:g}"] = {
                    "n_cells_S_changed": int(np.count_nonzero(
                        (S_on != off["S"])[masks["S"]])),
                    "max_abs": float(np.max(np.abs(
                        (S_on - off["S"])[masks["S"]]))),
                }
            out["s_absorption_law"] = law
    return out


def verdict(report: dict) -> str:
    return "INPUTS-" + report["inputs"]["status"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, default=ROUND2 / "kt1_10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = {"format": "nemo-testcase-l1-vortex-smtrungs-round3-v1",
              "inputs": inputs_section()}
    # Time interpolation is unobservable on this deck (the 12 records are
    # identical), so the model-time offset is checked only against NEMO's
    # printed fld_read lines, not by these numbers.
    report.update(ladder_and_damping(args.deck_root, args.plant))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    kt1 = report["independent"]["smt5"][1]["T"]["n_unequal"]
    if args.plant:
        require(kt1 == 1, f"plant moved {kt1} cells, expected exactly 1")
        print("PLANT-FIRED independent smt5 kt=1 T n_unequal", kt1)
        return 1
    print(verdict(report), "kt1 T n_unequal", kt1)
    return 0 if report["inputs"]["status"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
