#!/usr/bin/env python3
"""Discriminate OMT-4 fold transport from its tracer halo operands."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round229_fold_transport_acquisition import (
    check_record as admission,
)

FIELDS = ("zFv_after_trp", "T_Kmm", "S_Kmm", "e3t_Kmm", "tmask")
PLANTS = ("none", "transport-bit", "halo-bit", "correction-sign")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _exact(left, right) -> dict[str, object]:
    return r228._difference(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _read(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Parse the writer's self-describing header and named payloads."""

    raw = path.read_bytes()
    offset = 0
    magic = raw[offset:offset + 16].decode("ascii").rstrip()
    offset += 16
    values = struct.unpack_from("=15i", raw, offset)
    offset += 60
    keys = (
        "version", "kt", "stage", "rank", "nx", "ny", "nz",
        "origin_x", "origin_y", "i0", "j0", "i1", "j1", "bits", "count",
    )
    header = dict(zip(keys, values))
    require(magic == "NEMO_L4_R229FLD1", f"{path.name}: bad magic")
    require(
        (header["version"], header["kt"], header["stage"], header["bits"], header["count"])
        == (1, 1, 1, 64, 5),
        f"{path.name}: bad provenance header",
    )
    arrays = {}
    for expected in FIELDS:
        name = raw[offset:offset + 16].decode("ascii").rstrip()
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        require((name, ndim) == (expected, 3), f"{path.name}: bad field {name!r}")
        size = n1 * n2 * n3
        require(offset + 8 * size <= len(raw), f"{path.name}: truncated {name}")
        arrays[name] = np.frombuffer(
            raw, np.float64, count=size, offset=offset,
        ).reshape((n1, n2, n3), order="F")
        offset += 8 * size
    require(offset == len(raw), f"{path.name}: trailing payload")
    return header, arrays


def assemble_record(root: Path) -> dict[str, np.ndarray]:
    """Assemble owned global fields and the first northern T halo row."""

    admission.validate(root, "none")
    owned = {name: np.empty((148, 180, 31), np.float64) for name in FIELDS}
    north_halo = {name: np.empty((180, 31), np.float64) for name in FIELDS[1:]}
    coverage = np.zeros(180, dtype=bool)
    for rank in (0, 1):
        path = root / f"oracle_r229_fold_rank{rank:04d}_kt00000001_s1.bin"
        header, arrays = _read(path)
        require(header["rank"] == rank, f"{path.name}: rank moved")
        x0 = header["origin_x"] - 1
        width = header["i1"] - header["i0"] + 1
        x1 = x0 + width
        require(not np.any(coverage[x0:x1]), "rank-owned x slabs overlap")
        coverage[x0:x1] = True
        local_i = slice(header["i0"] - 1, header["i1"])
        local_j = slice(header["j0"] - 1, header["j1"])
        halo_j = header["j1"]
        for name in FIELDS:
            slab = arrays[name][local_i, local_j, :].transpose(1, 0, 2)
            owned[name][:, x0:x1, :] = slab
            if name != "zFv_after_trp":
                north_halo[name][x0:x1, :] = arrays[name][local_i, halo_j, :]
    require(np.all(coverage), "rank-owned x coverage is incomplete")
    return {**owned, **{f"{name}_north_halo": value for name, value in north_halo.items()}}


def _literal_fold_correction(
    card, stage, candidate_zfv, candidate_sum, nemo_sum, *, plant: str,
):
    """Replace only the northern CEN2 tracer face product, offline."""

    from legoesm.core.source_rounding import nemo_source_round

    sr = lambda value: np.asarray(nemo_source_round(value), np.float64)
    if plant == "correction-sign":
        nemo_sum = -nemo_sum
    p_v = np.asarray(candidate_zfv[-1], np.float64)
    old_flux = sr(sr(0.5 * p_v) * candidate_sum)
    new_flux = sr(sr(0.5 * p_v) * nemo_sum)
    delta_flux = sr(new_flux - old_flux)
    area = np.asarray(card.recipe.grid.area_T[-1], np.float64)[:, None]
    h_after = r228._quantities(card, stage)["e3t"][-1, :, :-1]
    delta = sr(-(card.dt_s / 3.0) * sr(delta_flux / area) / h_after)
    return delta


def measure(deck_root: Path, frame_root: Path, operand_root: Path,
            label: str, expect_commit: str, *, plant: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(plant in PLANTS, f"unknown plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-231 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-231 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-231 measurement requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    oracle_stage = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 1))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r228._hooks(card, True))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r228._hooks(card, True, live=True))
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live-stage trace has the wrong return type")
    passivity = r228.passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live-stage trace is not passive")
    values = traced.stage_outputs[0]
    stage = state._replace(
        u=state.u.replace(data=values[0]),
        v=state.v.replace(data=values[1]),
        T=state.T.replace(data=values[2]),
        S=state.S.replace(data=values[3]),
        eta=state.eta.replace(data=values[4]),
    )
    candidate_zfv = np.asarray(traced.stage_geometry[0][8], np.float64)[1:]
    record = assemble_record(operand_root)

    # NEMO constructs and consumes zFv only through jpkm1; the final stored
    # level is outside the compiled loop just as the admitted halo work cell
    # is.  T/S/e3t still carry all jpk levels in the same record.
    recorded_zfv = np.asarray(record["zFv_after_trp"][..., :-1], np.float64)
    if plant == "transport-bit":
        recorded_zfv = recorded_zfv.copy()
        recorded_zfv[-1, 0, 0] = np.nextafter(recorded_zfv[-1, 0, 0], np.inf)
    transport = _exact(candidate_zfv, recorded_zfv)

    from legoesm.core.source_rounding import nemo_source_round
    fold = card.recipe.grid.fold
    perm_t = np.asarray(fold.perm_T)
    state_t = np.asarray(state.T.data, np.float64)
    state_s = np.asarray(state.S.data, np.float64)
    candidate_t_sum = np.asarray(
        interp_to_v_points(state_t, card.recipe.grid, nemo_source_sum=True)[-1],
        np.float64,
    )[..., :-1]
    candidate_s_sum = np.asarray(
        interp_to_v_points(state_s, card.recipe.grid, nemo_source_sum=True)[-1],
        np.float64,
    )[..., :-1]
    nemo_t_halo = record["T_Kmm_north_halo"]
    nemo_s_halo = record["S_Kmm_north_halo"]
    if plant == "halo-bit":
        nemo_t_halo = nemo_t_halo.copy()
        nemo_t_halo[0, 0] = np.nextafter(nemo_t_halo[0, 0], np.inf)
    nemo_t_sum = np.asarray(
        nemo_source_round(record["T_Kmm"][-1] + nemo_t_halo))[..., :-1]
    nemo_s_sum = np.asarray(
        nemo_source_round(record["S_Kmm"][-1] + nemo_s_halo))[..., :-1]

    corrected_t = np.asarray(stage.T.data, np.float64).copy()
    corrected_s = np.asarray(stage.S.data, np.float64).copy()
    corrected_t[-1, :, :-1] += _literal_fold_correction(
        card, stage, candidate_zfv, candidate_t_sum, nemo_t_sum, plant=plant)
    corrected_s[-1, :, :-1] += _literal_fold_correction(
        card, stage, candidate_zfv, candidate_s_sum, nemo_s_sum, plant=plant)
    oracle_t = np.asarray(oracle_stage.T.data, np.float64)
    oracle_s = np.asarray(oracle_stage.S.data, np.float64)

    result = {
        "format": "nemo-testcase-l4-orca2-round231-fold-operand-v1",
        "status": "PASS_R231_TRACER_FOLD_OPERAND_NAMED",
        "label": label,
        "worktree": stamp,
        "passivity": passivity,
        "transport": transport,
        "entry_owned": {
            "T": _exact(state_t, record["T_Kmm"]),
            "S": _exact(state_s, record["S_Kmm"]),
            "e3t": _exact(r228._quantities(card, state)["e3t"], record["e3t_Kmm"]),
            "tmask": _exact(np.asarray(card.recipe.z_coord.is_active), record["tmask"]),
        },
        "nemo_fold_identity": {
            "T_halo": _exact(nemo_t_halo, record["T_Kmm"][-2, perm_t]),
            "S_halo": _exact(nemo_s_halo, record["S_Kmm"][-2, perm_t]),
            "e3t_halo": _exact(
                record["e3t_Kmm_north_halo"], record["e3t_Kmm"][-2, perm_t]),
            "tmask_halo": _exact(
                record["tmask_north_halo"], record["tmask"][-2, perm_t]),
        },
        "consumer_north_sum": {
            "T": _exact(candidate_t_sum, nemo_t_sum),
            "S": _exact(candidate_s_sum, nemo_s_sum),
        },
        "endpoint": {
            "before_T": _exact(np.asarray(stage.T.data)[-3:], oracle_t[-3:]),
            "after_T": _exact(corrected_t[-3:], oracle_t[-3:]),
            "before_S": _exact(np.asarray(stage.S.data)[-3:], oracle_s[-3:]),
            "after_S": _exact(corrected_s[-3:], oracle_s[-3:]),
        },
    }
    require(transport["unequal"] == 0, "post-consumer zFv is non-bit")
    require(result["nemo_fold_identity"]["T_halo"]["unequal"] == 0,
            "NEMO T halo does not satisfy its compiled fold identity")
    require(result["nemo_fold_identity"]["S_halo"]["unequal"] == 0,
            "NEMO S halo does not satisfy its compiled fold identity")
    require(result["consumer_north_sum"]["T"]["unequal"] > 0,
            "candidate T consumer already has NEMO's fold operand")
    require(result["consumer_north_sum"]["S"]["unequal"] > 0,
            "candidate S consumer already has NEMO's fold operand")
    require(result["endpoint"]["after_T"]["max_abs"] <= 0.0013606315900794863,
            "literal T fold correction did not reach the frozen unit-OFF bound")
    require(result["endpoint"]["after_S"]["max_abs"] <= 0.0009639248797768118,
            "literal S fold correction did not reach the frozen unit-OFF bound")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--operand-root", type=Path, required=True)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"), required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.frame_root, args.operand_root, args.label,
            args.expect_commit, plant=args.plant,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
