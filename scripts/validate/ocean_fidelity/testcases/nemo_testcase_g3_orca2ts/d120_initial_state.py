#!/usr/bin/env python3
"""Decision 120, offline: the G3/G4 cards' initial T/S against NEMO's.

NEMO's full-depth initial state with the ORCA_R2 hand edits (ln_tsd_dmp = T,
dtatsd.f90:219) is the ORCA2 lane's admitted kt = 1 entry frame: same
climatology files, same domain file, same calendar and time step (each
required below).  Every wet cell, every level, T and S, bit for bit.

Controls: the hand-edit-free arm must differ from NEMO on exactly the edited
cells (1,283 T / 720 S), and the as-built G3 record's level-1 entry (damping
off) must equal the card at level 1 (no edit reaches level 1).

Refusals exit 73; a fired plant exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

ORCA2_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
                  "acquisition/orca1ice_surface_entry_every_step_a_np2")
G3_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round15/"
                 "oracle_g3_r15/p3_a_10step")
# deck key -> NEMO's resolved key (output.namelist.dyn) and the deck's value text
SAME = {"namrun.nn_it000": "NN_IT000", "namrun.nn_date0": "NN_DATE0",
        "namrun.nn_leapy": "NN_LEAPY", "namdom.rn_dt": "RN_DT",
        "namtsd.ln_tsd_init": "LN_TSD_INIT", "namtsd.ln_tsd_dmp": "LN_TSD_DMP"}
TSD = {"SN_TEM": "data_1m_potential_temperature_nomask", "SN_SAL": "data_1m_salinity_nomask"}
NEMO_DOMAIN = "ORCA_R2_zps_domcfg.nc"
FILES = ("domain_cfg.nc", "data_1m_potential_temperature_nomask.nc",
         "data_1m_salinity_nomask.nc")


def require(ok: bool, message: str) -> None:
    if not ok:
        print(f"REFUSE: {message}", file=sys.stderr)
        raise SystemExit(73)


def _bits(a, b, wet):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return int(((a.view("<u8") != b.view("<u8")) & wet).sum())


def run(deck_roots: dict, plant: bool = False) -> dict:
    import hashlib

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as orca2,
    )
    import nemo_testcase_g1_orca2grid_ladder as g1

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    import re
    values = recipe._orca2g1_namelist_values
    resolved = (ORCA2_ROOT / "output.namelist.dyn").read_text()

    def nemo_value(key):
        hit = re.search(rf"^ *{key}= *([^,]+),", resolved, re.M)
        require(hit is not None, f"NEMO's resolved namelist has no {key}")
        text = hit.group(1).strip()
        return {"T": True, "F": False}.get(text) if text in "TF" else float(text)

    for key, name in TSD.items():
        for field, want in (("CLNAME", f'"{name}'), ("FREQH", "-1.0"), ("LN_TINT", "T"),
                            ("LN_CLIM", "T")):
            hit = re.search(rf"^ *{key}%{field}= *([^,]+),", resolved, re.M)
            require(hit is not None and hit.group(1).strip().startswith(want),
                    f"NEMO's run {key}%{field} is not {want}")
    for name in FILES:
        source = NEMO_DOMAIN if name == "domain_cfg.nc" else name
        digest = {hashlib.sha256((ORCA2_ROOT / source).read_bytes()).hexdigest()} | {
            hashlib.sha256((Path(r) / name).read_bytes()).hexdigest() for r in deck_roots.values()}
        require(len(digest) == 1, f"{name} differs between NEMO's run and a deck")
    nemo = orca2.assemble_state_fields(ORCA2_ROOT, 1, stage=None)
    out = {"nemo_root": str(ORCA2_ROOT), "cases": {}}
    for case, root in deck_roots.items():
        deck = values(Path(root) / "namelist_cfg")
        for name, key in SAME.items():
            require(deck.get(name) == nemo_value(key),
                    f"{case}: {name} = {deck.get(name)!r}, NEMO's run {nemo_value(key)!r}")
        for key, name in TSD.items():
            require(name in str(deck.get(f"namtsd.{key.lower()}")), f"{case}: {key} file")
        require(deck.get("namtra_dmp.ln_tradmp") is False, f"{case}: restoring on")
        card = recipe.build_nemo_testcase_card(case, deck_root=root)
        state = card.recipe.initial_state
        wet = np.asarray(card.recipe.z_coord.is_active) > 0.5
        nk = nemo["T"].shape[-1]
        require(wet[..., :nk].shape == nemo["T"].shape, "card and NEMO shapes differ")
        nemo_wet = (nemo["T"] != 0) | (nemo["S"] != 0)
        require(not np.any(nemo_wet & ~wet[..., :nk]), "NEMO has T/S on a card dry cell")
        w = wet[..., :nk]
        T = np.array(state.T.data)[..., :nk]
        S = np.array(state.S.data)[..., :nk]
        if plant:
            T[np.argwhere(w)[0][0], np.argwhere(w)[0][1], np.argwhere(w)[0][2]] += 1e-15
        edits = g2.hand_edit_cells(Path(root), card)
        off_T, off_S = recipe.build_orca2_initial_ts(
            *(Path(root) / n for n in FILES[1:]), wet, apply_hand_alterations=False)
        row = {
            "wet_cells": int(w.sum()),
            "unequal_T": _bits(T, nemo["T"], w), "unequal_S": _bits(S, nemo["S"], w),
            "edited_cells_T": int(edits["T"][..., :nk].sum()),
            "edited_cells_S": int(edits["S"][..., :nk].sum()),
            "control_no_edits_unequal_T": _bits(np.asarray(off_T)[..., :nk], nemo["T"], w),
            "control_no_edits_unequal_S": _bits(np.asarray(off_S)[..., :nk], nemo["S"], w),
            "nemo_level_beyond_card": int(np.asarray(state.T.data).shape[-1] - nk),
        }
        g3 = g1.read_step(G3_RECORD, 1)
        w1 = wet[..., 0]
        row["as_built_g3_level1_unequal_T"] = _bits(T[..., 0], g3["e_tn_k1_bb"], w1)
        row["as_built_g3_level1_unequal_S"] = _bits(S[..., 0], g3["e_sn_k1_bb"], w1)
        row["bit_identical"] = row["unequal_T"] == 0 and row["unequal_S"] == 0
        out["cases"][case] = row
    return out


import nemo_testcase_g2_orca2bathy_ladder as g2  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--g3-deck", type=Path, required=True)
    p.add_argument("--g4-deck", type=Path, required=True)
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", action="store_true")
    a = p.parse_args(argv)
    out = run({"ORCA2G3-ts": a.g3_deck, "ORCA2G4-sbc": a.g4_deck}, plant=a.plant)
    text = json.dumps(out, indent=1)
    print(text)
    if a.json:
        a.json.write_text(text + "\n")
    for row in out["cases"].values():
        require(row["control_no_edits_unequal_T"] == row["edited_cells_T"] > 0
                and row["control_no_edits_unequal_S"] == row["edited_cells_S"] > 0,
                "the no-edits control does not differ exactly on the edited cells")
    return 0 if all(r["bit_identical"] for r in out["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
