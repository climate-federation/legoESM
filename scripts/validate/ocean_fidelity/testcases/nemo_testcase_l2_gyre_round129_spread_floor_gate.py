#!/usr/bin/env python3
"""Round-129 GYRE infinitesimal-initial-condition spread-floor gate.

This scorer deliberately does not modify or duplicate the certified year
integrator.  It imports the existing from-rest harness for its NEMO reader,
mesh reconciliation, perturbation transcription, and RMS definition, then
scores the operator-directed split-root ensemble.

Plants are fail-closed and must print ``STATUS PLANT-FIRED`` and exit nonzero:

* ``initial-temperature-ulp`` moves one consumed wet initial T bit;
* ``pair-registry`` removes one of the six required within-model pairs.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np


def _load_year_harness():
    path = Path(__file__).with_name(
        "nemo_testcase_l2_gyre_year_fromrest.py")
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l2_gyre_year_fromrest_round129", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load certified year harness {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


YEAR = _load_year_harness()
GateError = YEAR.GateError
require = YEAR.require

ROUND129_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
ROUND129_PAIRS = tuple(itertools.combinations(YEAR.SEEDS, 2))
ROUND129_RATIO_BAR = np.float64(0.3)
ROUND129_MEMBER0_COMMIT = (
    "4d250301588d3ed0ad83fb20d6bf520e175d576e")
ROUND129_NEW_MEMBER_COMMIT = (
    "007affce297763a9594aea19babf8c9eace4883a")
# Rounds 134-135 extended only private, default-off daily-reset diagnostics in
# the certified harness.  Round 135's fresh unnudged control reproduced all
# five saved fields bit-for-bit at every one of 360 daily boundaries before
# this pin moved.
ROUND129_ORIGINAL_YEAR_HARNESS_SHA256 = (
    "7a679711c8ce02191e839f9f4359f21e753e8e2b6014a19a2302fb69ca168f9c")
ROUND129_YEAR_HARNESS_SHA256 = (
    "2996d361ec62465e16e98c7b7a96163132dc088291757bac1efb26ce2e81d55f")
ROUND129_PHASE3_GATE_SHA256 = (
    "e57fe1c475a1d386f30856f1841a2efb65162df6968b74b1a200f8850bdd9112")
ROUND129_MESH_SHA256 = (
    "3bf5d10e36dc52336b9797b13eb1efb4f02d25e0fb3a0c450ac6ce69e65471df")
ROUND129_NEMO_BINARY_SHA256 = (
    "578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a")
ROUND129_PRIOR_VERDICT_SHA256 = (
    "257bdb20032791164593cbf021071696733abf38e17ad3e01af01e2ffe3e1f64")
ROUND129_CONTROL_GAPS_K = {
    30: np.float64(6.890484901489568e-5),
    240: np.float64(1.6446741930292448e-2),
    360: np.float64(1.1223573910167267e-2),
}

PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
DEFAULT_MEMBER0_ROOT = PHASE3 / "year_equivalence/gyre"
DEFAULT_MEMBERS_ROOT = PHASE3 / "round129"
DEFAULT_NEMO_ROOT = PHASE3 / "year_fromrest"
DEFAULT_PRIOR_VERDICT = PHASE3 / "year_fromrest_head/verdict_year.json"
NEMO_SOURCE = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
YRPERT_SOURCE = NEMO_SOURCE / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo"
USRDEF_ISTATE = YRPERT_SOURCE / "usrdef_istate.f90"
USRDEF_NAM = YRPERT_SOURCE / "usrdef_nam.f90"


def _runtime_card():
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu",
            f"Round 129 requires CPU, got {jax.default_backend()}")
    card = build_nemo_testcase_card(YEAR.CASE)
    require(card.dt_s == YEAR.DT_S,
            f"card dt {card.dt_s} != registered {YEAR.DT_S}")
    return card, policy, jax.default_backend()


def _member_directory(seed: int, member0_root: Path,
                      members_root: Path) -> Path:
    root = Path(member0_root) if seed == 0 else Path(members_root)
    return root / f"lego_seed{seed}_year"


def _load_lego_member(seed: int, day: int, member0_root: Path,
                      members_root: Path) -> tuple[dict, dict]:
    path = _member_directory(seed, member0_root, members_root) / (
        f"day{day:03d}.npz")
    require(path.is_file(), f"missing legoESM snapshot {path}")
    with np.load(path) as handle:
        require(set(handle.files) == {"T", "S", "u", "v", "ssh"},
                f"{path}: fields {sorted(handle.files)} are not the certified "
                "T/S/u/v/ssh snapshot")
        arrays = {key: np.asarray(handle[key], dtype=np.float64)
                  for key in handle.files}
    for name, values in arrays.items():
        require(bool(np.all(np.isfinite(values))),
                f"{path}: non-finite legoESM {name}")
    return arrays, {"path": str(path), "sha256": YEAR.sha256(path)}


def _source_admission() -> dict:
    require(USRDEF_ISTATE.is_file(), f"missing {USRDEF_ISTATE}")
    require(USRDEF_NAM.is_file(), f"missing {USRDEF_NAM}")
    istate = USRDEF_ISTATE.read_text()
    nam = USRDEF_NAM.read_text()
    fragments = {
        "seed_guard": "IF( nn_pert_seed /= 0 ) THEN",
        "temperature_assignment": (
            "pts(:,:,jk,jp_tem) = pts(:,:,jk,jp_tem) + 1.e-10_wp * SIN"),
        "runtime_marker": "TINY PERTURBATION SEED = ",
    }
    for name, fragment in fragments.items():
        require(fragment in istate,
                f"compiled usrdef_istate lost {name}: {fragment}")
    for name, fragment in {
            "declaration": "nn_pert_seed = 0",
            "namelist": "NAMELIST/namusr_def/ nn_GYRE, ln_bench, jpkglo, nn_pert_seed",
            "control_print": "from-rest ensemble perturbation seed"}.items():
        require(fragment in nam, f"compiled usrdef_nam lost {name}: {fragment}")
    year_path = Path(YEAR.__file__)
    gate, gate_sha = YEAR._gate_module()
    del gate
    require(YEAR.sha256(year_path) == ROUND129_YEAR_HARNESS_SHA256,
            "the certified year harness moved after the registered members ran")
    require(gate_sha == ROUND129_PHASE3_GATE_SHA256,
            "the certified phase-3 stepping gate moved after the members ran")
    return {
        "compiled_usrdef_istate": str(USRDEF_ISTATE),
        "compiled_usrdef_istate_sha256": YEAR.sha256(USRDEF_ISTATE),
        "compiled_usrdef_nam": str(USRDEF_NAM),
        "compiled_usrdef_nam_sha256": YEAR.sha256(USRDEF_NAM),
        "year_harness": str(year_path),
        "year_harness_sha256": ROUND129_YEAR_HARNESS_SHA256,
        "phase3_gate_sha256": gate_sha,
    }


def admit_lego_members(member0_root: Path, members_root: Path) -> dict:
    records = {}
    for seed in YEAR.SEEDS:
        directory = _member_directory(seed, member0_root, members_root)
        path = directory / "manifest.json"
        require(path.is_file(), f"missing member manifest {path}")
        record = json.loads(path.read_text())
        expected_commit = (ROUND129_MEMBER0_COMMIT if seed == 0
                           else ROUND129_NEW_MEMBER_COMMIT)
        require(record.get("format")
                == "nemo-testcase-l2-gyre-year-fromrest-member-v1",
                f"{path}: wrong format")
        require(record.get("case") == YEAR.CASE,
                f"{path}: case {record.get('case')}")
        require(int(record.get("seed", -1)) == seed,
                f"{path}: seed {record.get('seed')} != {seed}")
        require(record.get("tag") == "year", f"{path}: tag is not year")
        require(int(record.get("days", -1)) == YEAR.YEAR_DAYS,
                f"{path}: not a 360-day member")
        require(int(record.get("steps", -1)) == YEAR.YEAR_STEPS,
                f"{path}: not a 2160-step member")
        require(float(record.get("dt_s", -1)) == YEAR.DT_S,
                f"{path}: wrong timestep")
        require(int(record.get("snapshot_step_interval", -1))
                == YEAR.STEPS_PER_DAY,
                f"{path}: snapshots are not daily")
        require(record.get("snapshot_days")
                == list(range(1, YEAR.YEAR_DAYS + 1)),
                f"{path}: daily snapshot registry is incomplete")
        require(record.get("phase3_gate_sha256")
                == ROUND129_PHASE3_GATE_SHA256,
                f"{path}: wrong stepping gate")
        require(record.get("operands", {}).get("mesh_sha256")
                == ROUND129_MESH_SHA256,
                f"{path}: wrong mesh")
        stamp = record.get("worktree", {})
        require(stamp.get("commit") == expected_commit,
                f"{path}: producer commit {stamp.get('commit')} != "
                f"{expected_commit}")
        require(stamp.get("clean") is True,
                f"{path}: producer worktree was dirty")
        for day in ROUND129_DAYS:
            snapshot = directory / f"day{day:03d}.npz"
            require(snapshot.is_file(), f"missing scored snapshot {snapshot}")
        records[str(seed)] = {
            "directory": str(directory),
            "manifest": str(path),
            "manifest_sha256": YEAR.sha256(path),
            "producer_commit": expected_commit,
            "producer_clean": True,
            "snapshot_days": 360,
            "phase3_gate_sha256": record["phase3_gate_sha256"],
            "mesh_sha256": record["operands"]["mesh_sha256"],
            "wall_seconds": float(record["wall_seconds"]),
        }
    return records


def admit_nemo_members(nemo_root: Path) -> dict:
    records = {}
    for seed in YEAR.SEEDS:
        directory = Path(nemo_root) / f"nemo_seed{seed}"
        binary = directory / "binary.sha256"
        namelist = directory / "namelist_cfg"
        output = directory / "ocean.output"
        entry = directory / "oracle_step_entry_kt00000001.bin"
        for path in (binary, namelist, output, entry):
            require(path.is_file(), f"missing NEMO evidence {path}")
        binary_hash = binary.read_text().split()[0]
        require(binary_hash == ROUND129_NEMO_BINARY_SHA256,
                f"seed {seed}: binary {binary_hash} is not the admitted one")
        selected = re.findall(
            r"^\s*nn_pert_seed\s*=\s*(-?\d+)", namelist.read_text(), re.M)
        require(selected == [str(seed)],
                f"seed {seed}: namelist selectors are {selected}")
        text = output.read_text(errors="replace")
        printed = re.findall(
            r"from-rest ensemble perturbation seed\s+nn_pert_seed\s*=\s*(-?\d+)",
            text)
        require(printed and set(printed) == {str(seed)},
                f"seed {seed}: runtime selector print is {printed}")
        tiny = re.findall(r"TINY PERTURBATION SEED\s*=\s*(-?\d+)", text)
        require((not tiny if seed == 0 else bool(tiny) and set(tiny) == {str(seed)}),
                f"seed {seed}: executed perturbation markers are {tiny}")
        records[str(seed)] = {
            "directory": str(directory),
            "binary_sha256": binary_hash,
            "namelist": str(namelist),
            "namelist_sha256": YEAR.sha256(namelist),
            "runtime_output": str(output),
            "runtime_output_sha256": YEAR.sha256(output),
            "selected_seed": seed,
            "perturbation_branch_executed": seed != 0,
            "entry": str(entry),
            "entry_sha256": YEAR.sha256(entry),
        }
    require(len({row["binary_sha256"] for row in records.values()}) == 1,
            "the four NEMO members do not use one binary")
    return records


def initial_identity(card, nemo_root: Path, mesh_path: Path, *,
                     plant: str | None = None) -> dict:
    import jax.numpy as jnp

    gate, _ = YEAR._gate_module()
    mesh = YEAR.nemo_operands(mesh_path)
    operands = YEAR.reconcile_operands(card, mesh)
    nlev = card.recipe.z_coord.n_levels
    masks = gate.expected_masks(card)
    active = np.asarray(card.recipe.z_coord.is_active) & (
        np.asarray(card.recipe.land_mask) > 0.5)[..., None]
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    latitude = np.broadcast_to(
        np.asarray(card.recipe.grid.native_lat_T_deg,
                   dtype=np.float64)[..., None], depth.shape)
    rows = {}
    for seed in YEAR.SEEDS:
        perturbation = YEAR.nemo_istate_perturbation(
            depth, latitude, active.astype(np.float64), seed)
        state = card.recipe.initial_state
        state = state._replace(T=state.T.replace(
            data=jnp.asarray(np.asarray(state.T.data) + perturbation,
                             dtype=jnp.float64)))
        candidate = gate.lego_fields(state)
        if plant == "initial-temperature-ulp" and seed == 1:
            moved = np.asarray(candidate["T"]).copy()
            index = tuple(np.argwhere(np.asarray(masks["T"], bool))[0])
            moved[index] = np.nextafter(moved[index], np.inf)
            candidate["T"] = moved
        entry = (Path(nemo_root) / f"nemo_seed{seed}" /
                 "oracle_step_entry_kt00000001.bin")
        oracle = gate.read_entry(entry)
        require((oracle["kt"], oracle["Nbb"]) == (1, 1),
                f"{entry}: not the step-1 BEFORE state")
        fields = {}
        for name in ("T", "S", "u", "v", "ssh"):
            reference = (oracle[name] if name == "ssh"
                         else oracle[name][..., :nlev])
            right = np.asarray(candidate[name], dtype=np.float64)
            left = np.asarray(reference, dtype=np.float64)
            mask = np.asarray(masks[name], dtype=bool)
            require(left.shape == right.shape == mask.shape,
                    f"seed {seed} {name}: shape mismatch")
            difference = np.abs(left[mask] - right[mask])
            unequal = int(np.count_nonzero(left[mask] != right[mask]))
            fields[name] = {
                "cells": int(np.count_nonzero(mask)),
                "cells_unequal": unequal,
                "max_abs_difference": float(difference.max()),
                "exact": unequal == 0,
            }
        failed = [name for name, row in fields.items() if not row["exact"]]
        require(not failed,
                f"seed {seed}: matched initial state is not BIT on {failed}")
        rows[str(seed)] = {
            "entry": str(entry),
            "entry_sha256": YEAR.sha256(entry),
            "fields": fields,
        }
    return {"members": rows, "operand_reconciliation": operands,
            "all_fields_all_members_bit_exact": True}


def pairwise_t3d(states: dict[int, dict], wet: np.ndarray, *,
                 plant: str | None = None) -> dict:
    pairs = list(ROUND129_PAIRS)
    if plant == "pair-registry":
        pairs.remove((2, 3))
    require(tuple(pairs) == ROUND129_PAIRS,
            f"pair registry {pairs} != required {list(ROUND129_PAIRS)}")
    rows = {}
    for left, right in pairs:
        value = YEAR._rms(
            np.asarray(states[left]["T"]) - np.asarray(states[right]["T"]),
            wet)
        require(np.isfinite(value), f"pair {left}-{right}: non-finite RMS")
        rows[f"{left}-{right}"] = value
    require(len(rows) == 6, f"pair registry has {len(rows)} rows, not 6")
    winner = max(rows, key=rows.get)
    return {"pairs_K": rows, "maximum_K": rows[winner],
            "maximum_pair": winner, "pair_count": len(rows)}


def classify_ratio(ratio: float) -> str:
    ratio = np.float64(ratio)
    require(bool(np.isfinite(ratio)) and ratio >= 0.0,
            f"invalid day-240 ratio {ratio}")
    return "MET" if ratio >= ROUND129_RATIO_BAR else "NOT_MET"


def _loglog_fit(days, values) -> dict:
    x = np.asarray(days, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    require(x.size >= 2 and np.all(x > 0.0) and np.all(y > 0.0),
            f"cannot fit positive log-log growth: days={x}, values={y}")
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    fitted = intercept + slope * np.log(x)
    observed = np.log(y)
    residual = float(np.sum((observed - fitted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return {"exponent": float(slope), "intercept": float(intercept),
            "r_squared": r2, "days": [int(value) for value in x],
            "values_K": [float(value) for value in y]}


def growth_diagnostic(rows: dict) -> dict:
    early_days = tuple(day for day in ROUND129_DAYS if day <= 240)
    spread = [rows[str(day)]["lego_spread"]["maximum_K"]
              for day in early_days]
    fit = _loglog_fit(early_days, spread)
    factor = float(rows["240"]["lego_spread"]["maximum_K"]
                   / rows["30"]["lego_spread"]["maximum_K"])
    exponent = fit["exponent"]
    if abs(exponent - 2.3) <= 0.25 * 2.3:
        label = "t^2.3-like"
    elif abs(exponent) <= 0.25:
        label = "flat"
    else:
        label = "neither"
    gap_fit = _loglog_fit(
        ROUND129_DAYS,
        [rows[str(day)]["matched_seed_gaps_K"]["0"]
         for day in ROUND129_DAYS])
    return {
        "spread_fit_through_day240": fit,
        "spread_day240_over_day30": factor,
        "classification": label,
        "registered_nonflat_prediction": "CONFIRMED" if factor > 10.0
        else "REFUTED",
        "seed0_gap_fit_all_eight_days": gap_fit,
    }


def score(member0_root: Path, members_root: Path, nemo_root: Path,
          prior_verdict: Path, mesh_path: Path, *,
          plant: str | None = None) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, policy, backend = _runtime_card()
    source = _source_admission()
    lego_admission = admit_lego_members(member0_root, members_root)
    nemo_admission = admit_nemo_members(nemo_root)
    initial = initial_identity(card, nemo_root, mesh_path, plant=plant)
    mesh, wet3, _, _, _, _, _ = YEAR._geometry(card, mesh_path)
    wet_cells = int(np.count_nonzero(wet3))
    require(wet_cells == 18000,
            f"wet T3D population {wet_cells} != registered 18000")
    require(Path(prior_verdict).is_file(),
            f"missing prior NEMO spread artifact {prior_verdict}")
    require(YEAR.sha256(prior_verdict) == ROUND129_PRIOR_VERDICT_SHA256,
            f"prior NEMO spread artifact {prior_verdict} moved")
    prior = json.loads(Path(prior_verdict).read_text())

    rows = {}
    for day in ROUND129_DAYS:
        lego, lego_inputs = {}, {}
        nemo, nemo_inputs = {}, {}
        for seed in YEAR.SEEDS:
            lego[seed], lego_inputs[str(seed)] = _load_lego_member(
                seed, day, member0_root, members_root)
            loaded = YEAR._load_nemo(nemo_root, seed, day,
                                     card.recipe.z_coord.n_levels)
            nemo[seed] = {name: loaded[name]
                          for name in ("T", "S", "u", "v", "ssh")}
            nemo_inputs[str(seed)] = {
                "path": loaded["path"], "sha256": loaded["sha256"]}
        lego_spread = pairwise_t3d(lego, wet3, plant=plant)
        nemo_spread = pairwise_t3d(nemo, wet3)
        prior_spread = float(
            prior["rows"]["T3D"]["days"][str(day)]["spread_nemo"])
        require(nemo_spread["maximum_K"] == prior_spread,
                f"day {day}: NEMO spread {nemo_spread['maximum_K']!r} "
                f"does not reproduce pinned {prior_spread!r}")
        gaps = {}
        for seed in YEAR.SEEDS:
            gaps[str(seed)] = YEAR._rms(
                np.asarray(lego[seed]["T"]) - np.asarray(nemo[seed]["T"]),
                wet3)
            require(np.isfinite(gaps[str(seed)]),
                    f"day {day} seed {seed}: non-finite matched gap")
        rows[str(day)] = {
            "lego_spread": lego_spread,
            "nemo_spread": nemo_spread,
            "matched_seed_gaps_K": gaps,
            "inputs": {"lego": lego_inputs, "nemo": nemo_inputs},
        }

    gap_controls = {}
    for day, expected in ROUND129_CONTROL_GAPS_K.items():
        measured = np.float64(rows[str(day)]["matched_seed_gaps_K"]["0"])
        require(measured == expected,
                f"day {day}: seed-0 gap {measured!r} != control {expected!r}")
        gap_controls[str(day)] = {
            "expected_K": float(expected), "measured_K": float(measured),
            "bit_identical_float64": True}

    lego_240 = float(rows["240"]["lego_spread"]["maximum_K"])
    nemo_240 = float(rows["240"]["nemo_spread"]["maximum_K"])
    gap_240 = float(rows["240"]["matched_seed_gaps_K"]["0"])
    require(lego_240 > 0.0 and gap_240 > 0.0,
            "day-240 primary spread and gap must both be positive")
    ratio = lego_240 / gap_240
    verdict = classify_ratio(ratio)
    nemo_over_lego = nemo_240 / lego_240
    systematic = (verdict == "NOT_MET"
                  and YEAR.MARGINAL_BAND[0] <= nemo_over_lego
                  <= YEAR.MARGINAL_BAND[1])
    growth = growth_diagnostic(rows)
    with np.load(_member_directory(0, member0_root, members_root)
                 / "day030.npz") as handle:
        candidate_dtype = handle["T"].dtype
    return {
        "schema": "nemo-testcase-l2-gyre-round129-spread-floor-v1",
        "case": YEAR.CASE,
        "metric": "unweighted fp64 RMS of T3D over NEMO tmask",
        "days": list(ROUND129_DAYS),
        "seeds": list(YEAR.SEEDS),
        "wet_cells": wet_cells,
        "precision": {"policy": str(policy), "state_dtype": str(candidate_dtype),
                      "platform": backend},
        "source_admission": source,
        "lego_member_admission": lego_admission,
        "nemo_member_admission": nemo_admission,
        "matched_initial_identity": initial,
        "prior_nemo_spread_artifact": {
            "path": str(prior_verdict),
            "sha256": ROUND129_PRIOR_VERDICT_SHA256,
            "all_eight_rows_reproduced_bit_for_bit": True,
        },
        "seed0_gap_controls": gap_controls,
        "rows": rows,
        "day240_verdict": {
            "lego_spread_K": lego_240,
            "nemo_spread_K": nemo_240,
            "seed0_gap_K": gap_240,
            "ratio_lego_spread_over_seed0_gap": ratio,
            "bar": float(ROUND129_RATIO_BAR),
            "criterion": "MET iff ratio >= 0.3",
            "year_bar": verdict,
            "directional_prediction": "CONFIRMED" if verdict == "MET"
            else "REFUTED",
            "nemo_spread_over_lego_spread": nemo_over_lego,
            "nemo_and_lego_spreads_similar_by_registered_band": (
                YEAR.MARGINAL_BAND[0] <= nemo_over_lego
                <= YEAR.MARGINAL_BAND[1]),
            "systematic_by_registered_secondary_rule": systematic,
        },
        "growth": growth,
        "same_binary_reproducibility_caveat": (
            "Previously measured exactly zero; this verdict uses the registered "
            "1e-10 K initial-condition ensemble, not stochastic reruns."),
        "worktree": worktree_stamp(),
    }


def self_check() -> int:
    bar = ROUND129_RATIO_BAR
    require(classify_ratio(np.nextafter(bar, -np.inf)) == "NOT_MET",
            "the binary64 value below the bar did not fail")
    require(classify_ratio(bar) == "MET", "exact equality did not pass")
    require(classify_ratio(np.nextafter(bar, np.inf)) == "MET",
            "the binary64 value above the bar did not pass")
    wet = np.ones((1, 1, 1), dtype=bool)
    states = {seed: {"T": np.full((1, 1, 1), float(seed))}
              for seed in YEAR.SEEDS}
    result = pairwise_t3d(states, wet)
    require(result["pair_count"] == 6 and result["maximum_K"] == 3.0,
            "six-pair maximum is wrong")
    try:
        pairwise_t3d(states, wet, plant="pair-registry")
    except GateError:
        pass
    else:  # pragma: no cover
        raise AssertionError("pair-registry plant cannot fail")
    fit = _loglog_fit((1, 2, 4, 8), np.asarray((1, 2, 4, 8)) ** 2.3)
    require(abs(fit["exponent"] - 2.3) < 1.0e-12,
            "log-log growth fit cannot recover a known exponent")
    print("SELF-CHECK OK: exact binary64 verdict boundary, six-pair registry "
          "with a firing omission, and known log-log exponent")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--member0-root", type=Path,
                        default=DEFAULT_MEMBER0_ROOT)
    parser.add_argument("--members-root", type=Path,
                        default=DEFAULT_MEMBERS_ROOT)
    parser.add_argument("--nemo-root", type=Path, default=DEFAULT_NEMO_ROOT)
    parser.add_argument("--prior-verdict", type=Path,
                        default=DEFAULT_PRIOR_VERDICT)
    parser.add_argument("--mesh", type=Path, default=YEAR.DEFAULT_NEMO_MESH)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plant", choices=("initial-temperature-ulp",
                                             "pair-registry"))
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.score:
        parser.error("--score or --self-check is required")
    try:
        report = score(args.member0_root, args.members_root, args.nemo_root,
                       args.prior_verdict, args.mesh, plant=args.plant)
    except GateError as error:
        if args.plant is not None:
            print(f"REFUSE Round-129 {args.plant}: {error}", file=sys.stderr)
            print("STATUS PLANT-FIRED")
            return 1
        raise
    if args.plant is not None:
        print(f"REFUSE Round-129 plant {args.plant} did not fire",
              file=sys.stderr)
        return 3
    target = args.json or (args.members_root / "spread_floor.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print("day   lego_spread_K   nemo_spread_K   gap_seed0_K   gap_seed1_K   "
          "gap_seed2_K   gap_seed3_K")
    for day in ROUND129_DAYS:
        row = report["rows"][str(day)]
        gaps = row["matched_seed_gaps_K"]
        print(f"{day:3d}  {row['lego_spread']['maximum_K']:.12e}  "
              f"{row['nemo_spread']['maximum_K']:.12e}  "
              f"{gaps['0']:.12e}  {gaps['1']:.12e}  "
              f"{gaps['2']:.12e}  {gaps['3']:.12e}")
    verdict = report["day240_verdict"]
    print(f"DAY240 ratio={verdict['ratio_lego_spread_over_seed0_gap']:.12e} "
          f"bar={verdict['bar']:.1f} verdict={verdict['year_bar']}")
    print(f"STATUS YEAR-BAR-{verdict['year_bar']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        sys.exit(2)
