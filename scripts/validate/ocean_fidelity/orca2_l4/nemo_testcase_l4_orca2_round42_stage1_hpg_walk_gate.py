#!/usr/bin/env python3
"""Round 42: walk ORCA2 stage-1 HPG inputs and compiled statements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hpg_gate as hpg_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate as round38,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round40_rhs_family_gate as round40,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round41_rhs_family_score_gate as round41_score,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round41_stage1_hpg_gate as round41_record,
)

INPUT_ORDER = ("rhd", "e3w", "gdept_z0", "r1_e1u")
STATEMENT_ORDER = ("zhpi_u", "zuap_u", "sum_u")
PLANTS = ("header", "truncation", "swapped-rank", "parent-byte", "restart-byte")
_PP = "ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo"
CITATIONS = {
    "eos_rhd": f"{_PP}/eosbn2.f90:810-844",
    "hpg_surface": f"{_PP}/dynhpg.f90:404-426",
    "hpg_interior": f"{_PP}/dynhpg.f90:440-464",
    "hpg_record": f"{_PP}/dynhpg.f90:477-490",
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_non_bit(rows: dict[str, dict]) -> dict | None:
    """Return the first input or statement boundary in registered order."""
    for name in INPUT_ORDER + STATEMENT_ORDER:
        row = rows[name]
        if not row["bit_exact"]:
            return {"boundary": name, **row}
    return None


def _owned(records: list[dict], name: str) -> np.ndarray:
    """Assemble the two 90-column owned blocks into the 180-column field."""
    blocks = [record["fields"][name][2:-2, 2:-2, ...] for record in records]
    require(all(block.shape[1] == round38.RANK_COLUMNS for block in blocks),
            f"{name} owned-column census changed")
    return np.concatenate(blocks, axis=1)


def _admit_hpg(root: Path, family_root: Path) -> tuple[list[dict], dict]:
    admission = json.loads((root / "round41_stage1_hpg_admission.json").read_text())
    require(admission.get("status") == "PASS_ACQUISITION",
            "round-41 HPG acquisition is not admitted")
    require(admission.get("label") == "given NEMO's entry",
            "round-41 HPG acquisition label changed")
    records = []
    for rank in range(2):
        name = round41_record.RECORD_TEMPLATE.format(rank=rank)
        path = root / name
        require(_sha256(path) == admission["record_sha256"][str(rank)],
                f"rank-{rank} HPG digest moved")
        stamp = (root / f"{name}.stamp").read_text().split()
        require(len(stamp) == 3 and stamp[0] == _sha256(path)
                and stamp[2] == name, f"rank-{rank} HPG stamp changed")
        records.append(round41_record.read_record_bytes(path.read_bytes(), rank))
        parent = round40.RECORD_TEMPLATE.format(rank=rank)
        require((root / parent).read_bytes() == (family_root / parent).read_bytes(),
                f"rank-{rank} inherited RHS-family stream moved")
    stage2 = "oracle_rkstage2_hpg_literal_kt00000001.bin"
    require((root / stage2).read_bytes() == (family_root / stage2).read_bytes(),
            "inherited stage-2 HPG stream moved")
    for name in round41_record.RESTARTS:
        require((root / name).read_bytes() == (family_root / name).read_bytes(),
                f"restart moved: {name}")
    for plant in PLANTS:
        result = json.loads((root / f"round41_{plant}_plant.json").read_text())
        log = (root / f"round41_{plant}_plant.log").read_text()
        require(result.get("status") == "PLANT-FIRED"
                and "STATUS PLANT-FIRED" in log,
                f"round-41 {plant} plant did not fire")
    return records, {
        "status": "ADMITTED_ROUND41_STAGE1_HPG",
        "producer_commit": admission["worktree"]["commit"],
        "record_sha256": admission["record_sha256"],
        "plants_fired": list(PLANTS),
    }


def _literal_from_inputs(
    inputs: dict[str, np.ndarray], g: float, grid=None
) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_hpg_sco_literal_cgrid,
    )

    if grid is None:
        r1_e1u = inputs["r1_e1u"]
        r1_e2v = inputs["r1_e2v"]
        dx_u = np.ones((r1_e1u.shape[0], r1_e1u.shape[1] + 1), np.float64)
        dy_v = np.ones((r1_e2v.shape[0] + 1, r1_e2v.shape[1]), np.float64)
        dx_u[:, 1:] = np.divide(
            1.0, r1_e1u, out=np.ones_like(r1_e1u), where=r1_e1u != 0.0)
        dy_v[1:, :] = np.divide(
            1.0, r1_e2v, out=np.ones_like(r1_e2v), where=r1_e2v != 0.0)
        grid = hpg_gate.LocalMetricGrid(jnp.asarray(dx_u), jnp.asarray(dy_v))
    values = tuple(np.asarray(value) for value in nemo_hpg_sco_literal_cgrid(
        jnp.asarray(inputs["rhd"]), jnp.asarray(inputs["e3w"]),
        jnp.asarray(inputs["gdept_z0"]), grid, g,
        return_components=True))
    return dict(zip(hpg_gate.COMPONENTS, values, strict=True))


def _candidate_inputs(model, state) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        nemo_e3w0_reference,
    )

    grid = model.grid
    z_coord = model.z_coord
    config = model.config
    e3w0 = nemo_e3w0_reference(z_coord)
    require(e3w0 is not None, "ORCA2 card lost its recorded e3w_0")

    @jax.jit
    def _compiled_inputs(source_state):
        # The production HPG evaluates this bundle inside ``_step_jitted``.
        # Keep the diagnostic in one compiled graph as well; eager EOS
        # evaluation can otherwise expose a different last-bit association.
        geom = pe.compute_frozen_geom_density(
            source_state, grid, z_coord, config)
        active = jnp.asarray(z_coord.is_active)
        rhd = (geom[4] if geom[4] is not None else
               jnp.where(active, geom[2], jnp.zeros_like(geom[2]))
               / config.rho_0)
        eta = source_state.eta.data
        min_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
        eta_safe = (jnp.maximum(
            eta, min_col - source_state.H_bathy.data)
            * source_state.land_mask.data)
        stretch = nemo_r3t_stretch(
            z_coord, eta_safe, source_state.H_bathy.data,
            evaluation="nemo_reciprocal")[..., jnp.newaxis]
        t_depth = jnp.asarray(z_coord.t_depth_ref)
        gdept_z0 = pe._nemo_qco_gdept_z0(
            t_depth[jnp.newaxis, jnp.newaxis, :], stretch, eta_safe)
        one = jnp.asarray(1.0, dtype=rhd.dtype)
        return (
            rhd,
            jnp.asarray(e3w0) * stretch,
            gdept_z0,
            nemo_source_round(one / grid.dx_u[:, 1:]),
            nemo_source_round(one / grid.dy_v[1:, :]),
        )

    values = jax.device_get(_compiled_inputs(state))
    return dict(zip(
        ("rhd", "e3w", "gdept_z0", "r1_e1u", "r1_e2v"),
        (np.asarray(value) for value in values), strict=True))


def _support_pair(field: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """West/east T operands of the periodic last native U face."""
    return np.stack((field[rows, -1, :], field[rows, 0, :]), axis=1)


def run(
    deck_root: Path,
    boundary_root: Path,
    hpg_root: Path,
    family_root: Path,
    baseline_root: Path,
    round41_json: Path,
    json_out: Path | None,
    *,
    plant: str = "none",
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round-42 gate is CPU-only")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    hpg_records, hpg_admission = _admit_hpg(hpg_root, family_root)
    family_records, parents, family_admission = round41_score.admit(
        family_root, baseline_root)
    prior = json.loads(round41_json.read_text())
    prior_support = prior["rank1_disputed_u_rows"]
    support_rows = np.asarray(prior_support["rows"], dtype=np.int64)
    require(support_rows.shape == (64,) and np.unique(support_rows).size == 64,
            "round-41 support-row census changed")

    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(boundary_root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(boundary_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            nemo_stage_rhs_accumulation_order_arm=True))
    trace = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    parts = trace.operator_operands[0]

    candidate_hpg = round38._native_rank(
        np.asarray(parts["after_hpg_u"].data), "u", 1)
    oracle_hpg = family_records[1]["fields"]["after_hpg_u"][
        2:-2, 2:-2, : round40.NZ - 1]
    support_mask = np.asarray(parents[1]["umask"])[support_rows, -1, :] > 0.0
    reproduction = round14.compare(
        candidate_hpg[support_rows, -1, :],
        oracle_hpg[support_rows, -1, :], support_mask)
    historical_hpg = prior_support["first_non_bit"]
    historical_reproduced = (
        reproduction["differing_cells"] == historical_hpg["differing_cells"]
        and reproduction["scored_cells"] == historical_hpg["scored_cells"]
        and reproduction["absolute_max"] == historical_hpg["absolute_max"])
    hpg_closed = reproduction["bit_exact"]
    hpg_improved = (
        reproduction["scored_cells"] == historical_hpg["scored_cells"]
        and reproduction["differing_cells"] < historical_hpg["differing_cells"]
        and reproduction["absolute_max"] < historical_hpg["absolute_max"])
    require(historical_reproduced or hpg_closed or hpg_improved,
            "HPG support is neither the admitted round-41 row, improved, nor "
            f"bit-exact: {reproduction}")
    for family in round41_score.FAMILIES[1:]:
        candidate = round38._native_rank(
            np.asarray(parts[f"{round41_score.PART_KEYS[family]}_u"].data),
            "u", 1)[support_rows, -1, :]
        oracle = family_records[1]["fields"][f"after_{family}_u"][
            2:-2, 2:-2, : round40.NZ - 1][support_rows, -1, :]
        require(round40._identity(
            candidate - oracle,
            candidate_hpg[support_rows, -1, :] -
            oracle_hpg[support_rows, -1, :])["bit_exact"],
            f"{family} changed the round-41 HPG residual")

    recorded_inputs = {
        name: _owned(hpg_records, name)
        for name in ("rhd", "e3w", "gdept_z0", "r1_e1u", "r1_e2v")
    }
    recorded_statements = {
        name: _owned(hpg_records, name)
        for name in hpg_gate.COMPONENTS
    }
    g = model.config.g
    recorded_replay = _literal_from_inputs(recorded_inputs, g)
    full_u_mask = np.concatenate(
        [np.asarray(parent["umask"]) for parent in parents], axis=1) > 0.0
    self_replay = {}
    for name in STATEMENT_ORDER:
        replay_native = recorded_replay[name][:, 1:, : round40.NZ - 1]
        oracle = recorded_statements[name][..., : round40.NZ - 1]
        trial = np.array(oracle, copy=True)
        if plant == "recorded-replay" and name == "sum_u":
            location = tuple(np.argwhere(full_u_mask)[0])
            trial[location] = np.nextafter(trial[location], np.inf)
        self_replay[name] = round14.compare(
            replay_native, trial, full_u_mask)
    recorded_replay_plant_fires = None
    if plant == "recorded-replay":
        recorded_replay_plant_fires = not self_replay["sum_u"]["bit_exact"]
        require(not self_replay["sum_u"]["bit_exact"],
                "recorded-replay plant did not fire")

    candidate_inputs = _candidate_inputs(model, state)
    candidate_literal = _literal_from_inputs(
        candidate_inputs, g, grid=model.grid)
    input_rows = {}
    for name in INPUT_ORDER[:-1]:
        candidate_pair = _support_pair(candidate_inputs[name], support_rows)[..., :30]
        recorded_pair = _support_pair(recorded_inputs[name], support_rows)[..., :30]
        pair_mask = np.broadcast_to(support_mask[:, None, :], candidate_pair.shape)
        input_rows[name] = round14.compare(
            candidate_pair, recorded_pair, pair_mask)
    input_rows["r1_e1u"] = round14.compare(
        candidate_inputs["r1_e1u"][support_rows, -1],
        recorded_inputs["r1_e1u"][support_rows, -1],
        support_mask[:, 0])

    statement_rows = {}
    for name in STATEMENT_ORDER:
        candidate = candidate_literal[name][:, 1:, :30][support_rows, -1, :]
        oracle = recorded_statements[name][..., :30][support_rows, -1, :]
        statement_rows[name] = round14.compare(candidate, oracle, support_mask)
    production_literal_identity = round14.compare(
        candidate_literal["sum_u"][:, 1:, :30][support_rows, -1, :],
        candidate_hpg[support_rows, -1, :], support_mask)
    instrument_valid = (
        all(row["bit_exact"] for row in self_replay.values())
        and production_literal_identity["bit_exact"])

    walk = {**input_rows, **statement_rows}
    measured_first = first_non_bit(walk)
    plant_fires = recorded_replay_plant_fires
    if plant == "first-boundary":
        planted = {name: dict(row) for name, row in walk.items()}
        if measured_first is None:
            planted[INPUT_ORDER[0]] = {
                "bit_exact": False, "differing_cells": 1,
                "scored_cells": walk[INPUT_ORDER[0]]["scored_cells"],
                "absolute_max": float.fromhex("0x1p-1074"), "ulp_max": 1,
            }
        else:
            planted[measured_first["boundary"]] = {
                "bit_exact": True, "differing_cells": 0,
                "scored_cells": measured_first["scored_cells"],
                "absolute_max": 0.0, "ulp_max": 0,
            }
        planted_first = first_non_bit(planted)
        plant_fires = (
            planted_first is not None and measured_first is None
            or planted_first is None and measured_first is not None
            or (planted_first is not None and measured_first is not None
                and planted_first["boundary"] != measured_first["boundary"]))
        require(plant_fires, "first-boundary plant did not move selector")

    result = {
        "gate": "nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate",
        "status": (
            "PLANT_FIRED" if plant != "none" else
            "MEASURED" if instrument_valid else "STOP_INSTRUMENT"
        ),
        "label": "given NEMO's entry",
        "provenance": worktree_stamp(),
        "execution": {
            "backend": jax.default_backend(),
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
            "support": "64 rank-1 periodic-east U faces, 1,754 wet layers",
        },
        "citations": CITATIONS,
        "hpg_admission": hpg_admission,
        "family_admission": family_admission,
        "round41_reproduction": reproduction,
        "round41_historical_hpg": historical_hpg,
        "round41_historical_reproduced": historical_reproduced,
        "hpg_improved": hpg_improved,
        "hpg_closed": hpg_closed,
        "recorded_input_self_replay": self_replay,
        "candidate_literal_matches_production_hpg": production_literal_identity,
        "rank1_disputed_u_walk": walk,
        "first_non_bit": measured_first,
        "plant": plant,
        "plant_fires": plant_fires,
        "single_statement_eligible": (
            hpg_closed and measured_first is None and instrument_valid),
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--hpg-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--round41-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--plant", choices=("none", "recorded-replay", "first-boundary"),
        default="none")
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.boundary_root, args.hpg_root,
            args.family_root, args.baseline_root, args.round41_json,
            args.json_out, plant=args.plant)
    except (GateError, round40.GateError, round41_score.GateError,
            round41_record.GateError, OSError, ValueError, KeyError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "execution": result["execution"],
        "round41_reproduction": result["round41_reproduction"],
        "recorded_input_self_replay": result["recorded_input_self_replay"],
        "rank1_disputed_u_walk": result["rank1_disputed_u_walk"],
        "first_non_bit": result["first_non_bit"],
        "plant": result["plant"],
        "plant_fires": result["plant_fires"],
    }, indent=2, sort_keys=True))
    if args.plant != "none":
        return 1
    return 2 if result["status"] == "STOP_INSTRUMENT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
