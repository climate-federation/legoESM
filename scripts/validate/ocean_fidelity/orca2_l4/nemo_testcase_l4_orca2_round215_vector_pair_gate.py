#!/usr/bin/env python3
"""Admit and replay OMT-1's slow-V/raw-ssvmask cancelling pair."""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as r15,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round197_vector_v_update as r197,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round205_omt0_substep_walk as r205,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round213_vector_pre_lbc_acquisition as acquisition,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)

PLANTS = ("none", "pair-closure", "zv-half", "mask-half", "source-order")
NAMES = ("zv_frc", "ssvmask", "va_pre_lbc")


class GateError(RuntimeError):
    """The admitted vector pair does not satisfy its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _payloads(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    raw = path.read_bytes()
    require(raw[:16] == acquisition.check_record.MAGIC, f"{path.name}: bad magic")
    header = struct.unpack_from("=16i", raw, 16)
    keys = ("version", "kt", "jn", "kmm", "krhs", "rank", "nx", "ny",
            "nimpp", "njmpp", "ntsi", "ntsj", "ntei", "ntej", "bits", "nfields")
    meta = dict(zip(keys, header))
    require((meta["version"], meta["kt"], meta["jn"], meta["kmm"],
             meta["krhs"], meta["bits"], meta["nfields"])
            == (1, 1, 1, 1, 3, 64, 3), f"{path.name}: header moved")
    offset = 80
    fields: dict[str, np.ndarray] = {}
    for expected in NAMES:
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        require(name == expected and (ndim, n3) == (2, 1),
                f"{path.name}: field header moved")
        count = n1 * n2
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name}")
        fields[name] = np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset,
        ).reshape((n1, n2), order="F").copy()
        offset = end
    require(offset == len(raw), f"{path.name}: trailing bytes")
    return meta, fields


def _assemble(root: Path) -> dict[str, np.ndarray]:
    result = {name: np.empty((148, 180), dtype=np.float64) for name in NAMES}
    coverage = np.zeros((148, 180), dtype=np.int8)
    for rank in (0, 1):
        path = root / f"oracle_r213_vector_rank{rank:04d}_kt00000001_jn001.bin"
        meta, fields = _payloads(path)
        require(meta["rank"] == rank, f"{path.name}: rank moved")
        i0 = meta["nimpp"] + meta["ntsi"] - 4
        j0 = meta["njmpp"] + meta["ntsj"] - 4
        ni = meta["ntei"] - meta["ntsi"] + 1
        nj = meta["ntej"] - meta["ntsj"] + 1
        coverage[j0:j0 + nj, i0:i0 + ni] += 1
        for name, array in fields.items():
            owned = (array if name == "zv_frc" else
                     array[meta["ntsi"] - 1:meta["ntei"],
                           meta["ntsj"] - 1:meta["ntej"]])
            require(owned.shape == (ni, nj), f"{path.name}: {name} owned shape")
            result[name][j0:j0 + nj, i0:i0 + ni] = owned.T
    require(bool(np.all(coverage == 1)), "rank coverage moved")
    return result


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "pair-closure":
        report["replays"]["pair"]["bit_exact"] = False
    elif plant == "zv-half":
        report["replays"]["zv_frc_only"]["bit_exact"] = True
    elif plant == "mask-half":
        report["replays"]["ssvmask_only"]["bit_exact"] = True
    elif plant == "source-order":
        report["source_order"] = ["ssvmask", "zv_frc"]
    require(report["admission_status"] == "PASS_R213_OMT1_VECTOR_PRE_LBC_ADMISSION",
            "record admission moved")
    require(report["source_order"] == ["zv_frc", "ssvmask", "va_pre_lbc"],
            "source order moved")
    require(report["preceding_statement"] == {
        "name": "continuity_forcing", "numerical_error": 0.0,
        "verdict": "AT_BAR_NOT_EXACT",
    }, "preceding signed-zero statement moved")
    require(report["replays"]["pair"]["bit_exact"],
            "recorded pair does not close va_pre_lbc")
    require(not report["replays"]["zv_frc_only"]["bit_exact"],
            "zv_frc half closes alone")
    require(not report["replays"]["ssvmask_only"]["bit_exact"],
            "ssvmask half closes alone")
    require(report["operand_rows"]["zv_frc"]["differing_cells"] == 35,
            "zv_frc cell census moved")
    require(report["operand_rows"]["ssvmask"]["differing_cells"] == 35,
            "ssvmask cell census moved")
    report["status"] = "PASS_R215_OMT1_VECTOR_PAIR_REPLAY"
    return report


def measure(args) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower().startswith(
        args.expect_commit.lower()), "round-215 measurement requires clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-215 replay requires production JIT on CPU")

    admission = acquisition.check_record.run(args.record_root, args.baseline_root)
    recorded = _assemble(args.record_root)
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    require(prior["status"] == "PASS_R212_OMT1_VECTOR_WALK",
            "round-212 source-order gate moved")
    require(prior["first_nonbit"]["name"] == "continuity_forcing"
            and prior["first_nonbit"]["absolute_max"] == 0.0,
            "preceding signed-zero boundary moved")
    require(prior["first_over_floor"]["name"] == "slow_v",
            "first numerical source boundary moved")

    card = omt1.build_omt1_card(args.deck_root)
    omt1.validate_omt1_card(card)
    state = card.recipe.initial_state
    freshwater, surface = rung0_ladder._zero_forcing(state.eta.data.shape)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    model.prime_step_caches(state)
    slow_u, slow_v = r205._slow_forcing(model, card, state, surface)
    dt = np.float64(card.dt_s / 65)
    zero_eta = np.zeros(state.eta.data.shape, dtype=np.float64)

    traced = jax.device_get(jax.jit(lambda seed, f_eta, f_u, f_v:
        barotropic_substeps_latlon_cgrid(
            seed, dt, 65, card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config, F_slow_eta=f_eta, F_slow_u=f_u,
            F_slow_v=f_v, add_barotropic_coriolis=True,
            u_now=seed.u.data, v_now=seed.v.data,
            _nemo_substep_trace_test_hook=True,
        ))(state, jnp.asarray(zero_eta), jnp.asarray(slow_u), jnp.asarray(slow_v)))
    trace = traced[2]
    substeps = phase3.read_bt_substeps(
        args.baseline_root / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    candidate = {
        "vn_e": r205._native_v(trace["v_entry"][0]),
        "rDt_e": dt,
        "zv_spg": r205._native_v(trace["pgf_v"][0]),
        "zv_trd": r205._native_v(trace["trd_v"][0]),
        "zv_frc": r205._native_v(trace["slow_v"][0]),
        "ssvmask": r205._native_v(np.asarray(state.v_mask.data)),
    }
    reference = {
        "vn_e": np.asarray(substeps["v_entry"][0]),
        "rDt_e": dt,
        "zv_spg": np.asarray(substeps["pgf_v"][0]),
        "zv_trd": np.asarray(substeps["trd_v"][0]),
        "zv_frc": recorded["zv_frc"],
        "ssvmask": recorded["ssvmask"],
    }
    target = recorded["va_pre_lbc"]
    operand_rows = {
        name: r197._row(candidate[name], reference[name])
        for name in ("zv_frc", "ssvmask")
    }
    variants = {
        "baseline": candidate,
        "zv_frc_only": {**candidate, "zv_frc": reference["zv_frc"]},
        "ssvmask_only": {**candidate, "ssvmask": reference["ssvmask"]},
        "pair": {**candidate, "zv_frc": reference["zv_frc"],
                 "ssvmask": reference["ssvmask"]},
    }
    replays = {
        name: r197._row(r197._literal_terms(values)["raw_va_e"], target)
        for name, values in variants.items()
    }
    reference_replay = r197._row(
        r197._literal_terms(reference)["raw_va_e"], target)
    require(reference_replay["bit_exact"], "NEMO operands do not replay NEMO target")
    raw = {
        "format": "nemo-testcase-l4-orca2-round215-vector-pair-v1",
        "claim_labels": ["independent OMT-1", "given NEMO's entry OMT-1"],
        "execution": "offline-production-pure-jit-cpu-fp64-x64-libm",
        "worktree": stamp, "admission_status": admission["status"],
        "source_order": ["zv_frc", "ssvmask", "va_pre_lbc"],
        "preceding_statement": {
            "name": "continuity_forcing", "numerical_error": 0.0,
            "verdict": "AT_BAR_NOT_EXACT",
        },
        "operand_rows": operand_rows, "replays": replays,
        "reference_replay": reference_replay,
        "compiled_source": {
            "vector_update": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682",
            "association": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756",
            "fold_v": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718",
        },
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="measure")
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.record_root, args.baseline_root, args.deck_root,
                         args.prior_report, args.expect_commit)),
                    "measurement arguments missing")
            require(args.plant == "none", "measurement does not accept plants")
            result = measure(args)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            acquisition.check_record.Refusal, omt1.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
