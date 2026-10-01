#!/usr/bin/env python3
"""Walk ORCA2's kt=10 tracer extremes through NEMO's recorded K consumers.

All trajectory numbers are labelled ``given NEMO's entry``: Decision 52 replaces
only the card's initial SSH.  The directed arms replace NEMO ``avt`` and ``avm``
at legoESM's production implicit-solve boundary, one variable at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round79b_vertical_mixing_gate as vmix,
)

FIELD_ORDER = ("T", "S", "u", "v", "ssh")
ARMS = ("ordinary", "fixed_none", "avt", "avm", "avt_avm")
PLANTS = ("none", "baseline", "passive-seam", "avt-inert", "avm-inert", "claim-label")
EXPECTED_REFERENCE_SHA256 = (
    "8347540880af2aac58ff6d3f0af240d676c7f9f36dda55867e4c4d35c9780749"
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _owned(values: np.ndarray) -> np.ndarray:
    """Remove NEMO's four-point halo and change (i,j,k) to (j,i,k)."""
    return values[2:-2, 2:-2].transpose(1, 0, 2)


def _global_record(root: Path, kt: int) -> dict[str, np.ndarray]:
    ranks = []
    for rank in vmix.RANKS:
        path = root / f"oracle_zdf_vmix_kt{kt:08d}_r{rank:04d}.bin"
        ranks.append(vmix.read_record(path)["fields"])
    return {
        name: np.concatenate([_owned(ranks[0][name]), _owned(ranks[1][name])], axis=1)
        for name in ranks[0]
        if ranks[0][name].shape[:2] == (94, 152)
    }


def _interfaces(values: np.ndarray) -> np.ndarray:
    # NEMO jk=2..jpkm1 are the 29 interfaces between the 30 prognostic cells.
    result = values[..., 1:30]
    require(result.shape == (148, 180, 29), f"consumer profile shape {result.shape}")
    return result


def _reference_row(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing landed reference {path}")
    require(sha256(path) == EXPECTED_REFERENCE_SHA256,
            "round-79b landed reference digest changed")
    document = json.loads(path.read_text())
    rows = [row for row in document["candidate_trajectory"]["checkpoints"]
            if row["kt"] == 10 and row["checkpoint"] == "stage3"]
    require(len(rows) == 1, "landed reference has no unique kt10 stage3 row")
    return rows[0]


def _override(fields: dict[str, np.ndarray], selectors: tuple[bool, bool]):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOVerticalSolveTestInput,
    )

    heat = _interfaces(fields["avt_after_iwm"])
    viscosity = _interfaces(fields["avm_after_iwm"])
    zeros_i = np.zeros_like(heat)
    zeros_c = np.zeros((148, 180, 30), dtype=np.float64)
    replace = np.zeros(6, dtype=np.bool_)
    replace[0], replace[1] = selectors
    return _NEMOVerticalSolveTestInput(
        heat_K=jnp.asarray(heat),
        viscosity_K=jnp.asarray(viscosity),
        formed_K=jnp.asarray(zeros_i),
        tracer_e3w=jnp.asarray(zeros_i),
        tracer_e3t=jnp.asarray(zeros_c),
        temperature_content=jnp.asarray(zeros_c),
        replace=jnp.asarray(replace),
    )


def _argmax(actual: np.ndarray, expected: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.unravel_index(
        np.argmax(np.abs(actual - expected)), actual.shape))


def _location(index: tuple[int, int, int], card, fields: dict[str, np.ndarray]) -> dict:
    j, i, k = index
    wet = fields["tmask"][..., :30] > 0.0
    neighbours = []
    for jj, ii in ((j - 1, i), (j + 1, i), (j, (i - 1) % 180), (j, (i + 1) % 180)):
        if 0 <= jj < 148:
            neighbours.append(bool(wet[jj, ii, k]))
    interface_ids = [x for x in (k - 1, k) if 0 <= x < 29]
    contributions = {
        "river_mouth": fields["avt_after_rnf"] - fields["avt_after_tke"],
        "convection": fields["avt_after_evd"] - fields["avt_after_rnf"],
        "double_diffusive_heat": fields["avt_after_ddm"] - fields["avt_after_evd"],
        "double_diffusive_salt_extra": fields["avs_after_ddm"] - fields["avt_after_ddm"],
        "internal_wave_heat": fields["avt_after_iwm"] - fields["avt_after_ddm"],
        "internal_wave_momentum": fields["avm_after_iwm"] - fields["avm_after_ddm"],
    }
    process = {}
    for name, values in contributions.items():
        interfaces = _interfaces(values)
        process[name] = float(max(abs(interfaces[j, i, q]) for q in interface_ids))
    lat = float(np.rad2deg(np.asarray(card.recipe.grid.lat_T))[j, i])
    lon = float(np.rad2deg(np.asarray(card.recipe.grid.lon_T))[j, i])
    river = fields["rnfmsk"][j, i, 0]
    return {
        "index_jik": [j, i, k],
        "latitude_deg": lat,
        "longitude_deg": lon,
        "fold_row": j == 147,
        "land_adjacent": not all(neighbours),
        "river_mouth": bool(river != 0.0),
        "convection_site": process["convection"] != 0.0,
        "bounding_interface_indices": interface_ids,
        "process_contribution_abs_max_m2_s": process,
        "process_ranking": sorted(process, key=lambda name: (-process[name], name)),
    }


def measure(deck_root: Path, ten_step_root: Path, vmix_root: Path,
            landed_reference: Path) -> dict[str, object]:
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    admission = vmix.run_gate(vmix_root)
    require(admission["status"] == "PASS", "vertical-mixing record was not admitted")
    reference = _reference_row(landed_reference)
    _, card = ladder.card_fields(deck_root)
    require(card.recipe.model_config.outer_integrator == "forward_euler",
            "consumer seam is only valid for the card's forward-Euler program")
    entry = ladder.assemble_state_fields(ten_step_root, 1, stage=None)
    initial = card.recipe.initial_state
    initial = initial._replace(
        eta=initial.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    states = {name: initial for name in ARMS}
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing)
    last_fields = None
    for kt in range(1, 11):
        surface_fields = ladder.assemble_surface_fields(ten_step_root, kt)
        freshwater, surface = ladder._surface_forcings(card, deck_root, surface_fields, kt)
        recorded = _global_record(vmix_root, kt)
        last_fields = recorded
        next_states = {}
        for name, selectors in {
            "ordinary": None,
            "fixed_none": (False, False),
            "avt": (True, False),
            "avm": (False, True),
            "avt_avm": (True, True),
        }.items():
            kwargs = {} if selectors is None else {
                "_vertical_K_test_override": _override(recorded, selectors)
            }
            next_states[name] = model.step(
                states[name], dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface, **kwargs)
        states = next_states
    require(last_fields is not None, "no mixing record was consumed")
    oracle = ladder.read_state_frame(
        ten_step_root / "oracle_stage_kt00000010_s3.bin", kt=10, stage=3)
    candidate = {name: ladder._rank0_fields(ladder._candidate_fields(state))
                 for name, state in states.items()}
    comparisons = {name: ladder.compare_fields(values, oracle)
                   for name, values in candidate.items()}
    require(comparisons["ordinary"] == {
        key: reference[key] for key in ("rows", "ranked_non_bit_by_max_abs", "first_non_bit_field")
    }, "ordinary kt10 row differs from the landed round-79b reference")
    passive = all(np.array_equal(candidate["ordinary"][name], candidate["fixed_none"][name])
                  for name in FIELD_ORDER)
    require(passive, "all-false fixed-shape consumer seam is not passive")
    extremes = {}
    for field in ("T", "S"):
        index = _argmax(candidate["ordinary"][field], oracle[field])
        values = {}
        for name in ARMS:
            residual = candidate[name][field] - oracle[field]
            values[name] = {
                "global_max_abs": float(np.max(np.abs(residual))),
                "residual_at_ordinary_argmax": float(residual[index]),
            }
        extremes[field] = {
            "ordinary_argmax": _location(index, card, last_fields),
            "arms": values,
        }
    return {
        "format": "nemo-testcase-l4-orca2-round80-extremes-v1",
        "claim_label": "given NEMO's entry",
        "initial_mode": "decision52_ssh_bridge",
        "steps_completed": 10,
        "record_admission": {"status": admission["status"],
                             "producer_commit": admission["producer_commit"]},
        "landed_reference": {"path": str(landed_reference),
                             "sha256": sha256(landed_reference)},
        "passive_seam_bit_identical": passive,
        "comparisons": comparisons,
        "extremes": extremes,
        "compiled_citations": {
            "process_order": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381",
            "tracer_consumer": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:173-235",
            "momentum_consumer": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:183-206",
        },
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "baseline":
        report["comparisons"]["ordinary"]["rows"]["T"]["max_abs"] += 1.0
    elif plant == "passive-seam":
        report["passive_seam_bit_identical"] = False
    elif plant == "avt-inert":
        report["comparisons"]["avt"] = report["comparisons"]["ordinary"]
    elif plant == "avm-inert":
        report["comparisons"]["avm"] = report["comparisons"]["ordinary"]
    elif plant == "claim-label":
        report["claim_label"] = "independent"
    require(report["claim_label"] == "given NEMO's entry", "claim label changed")
    require(report["passive_seam_bit_identical"], "passive seam control changed the state")
    expected_t = 1.2367457128331782
    expected_s = 0.2871061346986039
    require(report["comparisons"]["ordinary"]["rows"]["T"]["max_abs"] == expected_t,
            "ordinary T maximum no longer matches the preregistered baseline")
    require(report["comparisons"]["ordinary"]["rows"]["S"]["max_abs"] == expected_s,
            "ordinary S maximum no longer matches the preregistered baseline")
    require(report["comparisons"]["avt"] != report["comparisons"]["ordinary"],
            "recorded avt arm was inert")
    require(report["comparisons"]["avm"] != report["comparisons"]["ordinary"],
            "recorded avm arm was inert")
    report["status"] = "PASS_ROUND80_EXTREMES"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--ten-step-root", type=Path)
    parser.add_argument("--vmix-root", type=Path)
    parser.add_argument("--landed-reference", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.deck_root, args.ten_step_root, args.vmix_root,
                             args.landed_reference)),
                    "--classify-json cannot be combined with run inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "plants classify an existing result")
            require(all((args.deck_root, args.ten_step_root, args.vmix_root,
                         args.landed_reference)), "run mode requires all record inputs")
            raw = measure(args.deck_root, args.ten_step_root, args.vmix_root,
                          args.landed_reference)
        result = classify(raw, args.plant)
    except (GateError, ladder.GateError, vmix.GateError, OSError, ValueError,
            KeyError, TypeError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND80_EXTREMES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
