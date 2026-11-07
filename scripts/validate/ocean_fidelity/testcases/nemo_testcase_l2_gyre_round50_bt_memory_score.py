#!/usr/bin/env python3
"""Score legoESM's kt=1 carried barotropic memory against round-48 NEMO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import _surface_forcings, require
from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/oracle_bt_memory")


def _owned(value):
    return np.asarray(value)[2:-2, 2:-2]


def _score(name, candidate, oracle):
    candidate, oracle = np.asarray(candidate), np.asarray(oracle)
    require(candidate.shape == oracle.shape, f"{name}: shape mismatch")
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    numeric_unequal = candidate != oracle
    return {
        "name": name,
        "n": int(candidate.size),
        "n_unequal": int(np.count_nonzero(unequal)),
        "n_numeric_unequal": int(np.count_nonzero(numeric_unequal)),
        "n_signed_zero_only": int(np.count_nonzero(unequal & ~numeric_unequal)),
        "max_abs": float(np.max(np.abs(candidate - oracle))),
    }


def run(root: Path, expect_commit: str, plant: bool = False):
    stamp = worktree_stamp()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"] == expect_commit, "producer commit mismatch")
    require(jax.config.jax_enable_x64, "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    end = read_record(root / "oracle_bt_memory_kt00000001_end.bin")["arrays"]
    start = read_record(root / "oracle_bt_memory_kt00000002_start.bin")["arrays"]
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    state = card.recipe.initial_state
    freshwater, surface = _surface_forcings(card, state, 1)
    after = model.step(state, dt=card.dt_s, freshwater=freshwater,
                       surface_forcing=surface)
    require(after.bt_hist is not None, "model returned no barotropic history")
    require(after.uu_b is not None and after.vv_b is not None,
            "model returned no current external mode")

    uf = np.asarray(after.uu_b.data)[:, 1:]
    vf = np.asarray(after.vv_b.data)[1:, :]
    ef = np.asarray(after.eta.data)
    h = tuple(np.asarray(value) for value in after.bt_hist)
    lego = {
        "ubb_e": uf - h[1][:, 1:], "ub_e": uf - h[0][:, 1:],
        "vbb_e": vf - h[3][1:, :], "vb_e": vf - h[2][1:, :],
        "sshbb_e": ef - h[5], "sshb_e": ef - h[4],
        "un_e": uf, "vn_e": vf, "sshn_e": ef,
        "un_adv": np.zeros_like(uf), "vn_adv": np.zeros_like(vf),
        "ubar_Kmm": uf, "vbar_Kmm": vf, "ssh_Kmm": ef,
        "ubar_Kaa": np.zeros_like(uf), "vbar_Kaa": np.zeros_like(vf),
        "ssh_Kaa": np.zeros_like(ef),
    }
    if plant:
        planted = lego["ubb_e"].copy()
        planted.reshape(-1).view(np.uint64)[0] ^= np.uint64(1)
        lego["ubb_e"] = planted
    rows = []
    for name in ("ubb_e", "ub_e", "vbb_e", "vb_e", "sshbb_e", "sshb_e"):
        rows.append(_score(f"carry.{name}.kt1_end", lego[name], _owned(end[name])))
        rows.append(_score(f"carry.{name}.kt2_start", lego[name], _owned(start[name])))
    for name in ("un_e", "vn_e", "sshn_e"):
        rows.append(_score(f"current.{name}.kt1_end", lego[name], _owned(end[name])))
        rows.append(_score(f"current.{name}.kt2_start", lego[name], _owned(start[name])))
    for name in ("ubar_Kmm", "vbar_Kmm", "ssh_Kmm", "un_adv", "vn_adv",
                 "ubar_Kaa", "vbar_Kaa", "ssh_Kaa"):
        rows.append(_score(f"kt2_start.{name}", lego[name], _owned(start[name])))
    for name, candidate in (("ubar_Kaa", uf), ("vbar_Kaa", vf), ("ssh_Kaa", ef)):
        rows.append(_score(f"kt1_end.{name}", candidate, _owned(end[name])))
    exact = all(row["n_unequal"] == 0 for row in rows)
    require(not plant or not exact, "memory plant stayed green")
    return {
        "format": "nemo-testcase-l2-gyre-round50-bt-memory-score-v1",
        "status": "CONFIRMED" if exact else "DEBT",
        "rows": rows,
        "worktree": stamp,
        "source": {
            "history_rotation": "dynspg_ts.f90:760-761",
            "final_commit": "dynspg_ts.f90:824-827",
            "legoesm_deviation_reconstruction": "barotropic_latlon_cgrid.py:2061-2077",
            "legoesm_deviation_commit": "barotropic_latlon_cgrid.py:2871-2885",
        },
        "plant": plant,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(args.root, args.expect_commit, args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
