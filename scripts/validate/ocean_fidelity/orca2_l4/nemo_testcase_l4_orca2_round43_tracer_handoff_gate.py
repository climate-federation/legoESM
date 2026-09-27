#!/usr/bin/env python3
"""Walk ORCA2's kt=1 external-mode to stage-1 tracer handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
    validate_nemo_testcase_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_phase2l_tracer_gate as phase2l,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)

OWNED_NX, OWNED_NY, NLEV = 90, 148, 30
ORDER = (
    "metric_zFu", "metric_zFv", "metric_zFw",
    "after_advection_T", "after_advection_S",
    "after_sbc_T", "after_sbc_S", "stage1_T", "stage1_S",
)


class GateError(RuntimeError):
    """A fail-closed round-43 condition was not met."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bit_score(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            f"score shape mismatch {actual.shape}, {expected.shape}, {mask.shape}")
    left, right = actual[mask], expected[mask]
    require(left.size > 0, "empty scored support")
    require(np.isfinite(left).all() and np.isfinite(right).all(),
            "non-finite scored value")
    unequal = left.view(np.uint64) != right.view(np.uint64)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "bit_exact": not bool(unequal.any()),
        "unequal": int(unequal.sum()),
        "count": int(left.size),
        "max_abs": float(np.abs(left - right).max(initial=0.0)),
    }


def first_non_bit(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in ORDER if name in rows and not rows[name]["bit_exact"]), None)


def _support_masks(card) -> dict[str, np.ndarray]:
    cell = np.asarray(card.recipe.z_coord.is_active)[:, :OWNED_NX, :NLEV]
    safe = np.zeros((OWNED_NY, OWNED_NX), dtype=bool)
    safe[:-1, 1:-1] = True
    cell = cell & safe[..., None]
    u2 = np.asarray(card.recipe.initial_state.u_mask.data)[:, 1:OWNED_NX + 1] > 0.5
    v2 = np.asarray(card.recipe.initial_state.v_mask.data)[1:OWNED_NY + 1, :OWNED_NX] > 0.5
    u = np.broadcast_to(u2[..., None], cell.shape)
    v = np.broadcast_to(v2[..., None], cell.shape)
    return {
        "T": cell,
        "u": u & safe[..., None],
        "v": v & safe[..., None],
        "w": np.pad(cell[..., :-1] & cell[..., 1:], ((0, 0), (0, 0), (0, 1))),
    }


def _endpoint_override(card, entry: dict[str, np.ndarray], root: Path):
    eta_after = np.asarray(entry["ssh"], dtype=np.float64).copy()
    eta_after[:, :OWNED_NX] = phase2l.read_final_ssh(
        root / phase2l.STAGE3_RECORD)
    un_adv, vn_adv = phase2l.read_external_transports(root / phase2l.BT_RECORD)
    hu_avg = np.zeros(card.recipe.initial_state.u.data.shape[:2], np.float64)
    hv_avg = np.zeros(card.recipe.initial_state.v.data.shape[:2], np.float64)
    hu_avg[:, 1:OWNED_NX + 1] = un_adv
    hv_avg[1:OWNED_NY + 1, :OWNED_NX] = vn_adv
    return jnp.asarray(eta_after), jnp.asarray(hu_avg), jnp.asarray(hv_avg)


def _run(card, state, freshwater, surface, *, endpoint, exposure: str):
    kwargs: dict[str, object] = {"external_mode_result_override": endpoint}
    if exposure == "transport":
        kwargs["expose_tracer_transport_stage"] = 1
    elif exposure in ("after_advection", "after_sbc"):
        kwargs["expose_tracer_stage1_boundary"] = exposure
    elif exposure == "stage1":
        kwargs["expose_tracer_stage"] = 1
    else:
        raise GateError(f"unknown exposure {exposure!r}")
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**kwargs),
    )
    return model.step(
        state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)


def validate(deck_root: Path, record_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-43 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    require(cfg.tracer_advection == "fct2", "ORCA2 card no longer selects FCT2")
    require(cfg.momentum_advection == "vector_invariant",
            "ORCA2 card no longer selects vector momentum")
    require(cfg.nemo_stage_momentum_wzv_split is True,
            "Decision 58 is not selected on the ORCA2 card")

    tracer_path = record_root / phase2l.TRACER_RECORD
    stage_path = record_root / "oracle_stage_kt00000001_s1.bin"
    for path in (tracer_path, stage_path, record_root / phase2l.STAGE3_RECORD,
                 record_root / phase2l.BT_RECORD):
        require(path.is_file(), f"missing admitted record {path}")
    tracer = phase2l.read_tracer(tracer_path)
    oracle_stage = ladder.read_state_frame(stage_path, kt=1, stage=1)
    entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(card, deck_root, surface_fields, 1)

    # Decision 52: only SSH is bridged.  T/S/u/v remain the independently
    # constructed, already-exact card entry.
    state = card.recipe.initial_state._replace(
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    entry_rows = ladder.compare_fields(ladder._candidate_fields(state), entry)
    require(all(entry_rows["rows"][name]["bit_identical"]
                for name in ("T", "S", "u", "v", "ssh")),
            "Decision-52 kt=1 entry bridge is not bit-exact")

    masks = _support_masks(card)
    expected = {
        "metric_zFu": np.asarray(tracer["zFu"])[..., :NLEV],
        "metric_zFv": np.asarray(tracer["zFv"])[..., :NLEV],
        "metric_zFw": np.asarray(tracer["zFw"])[..., :NLEV],
        "after_advection_T": np.asarray(tracer["after_advection_T"])[..., :NLEV],
        "after_advection_S": np.asarray(tracer["after_advection_S"])[..., :NLEV],
        "after_sbc_T": np.asarray(tracer["after_sbc_T"])[..., :NLEV],
        "after_sbc_S": np.asarray(tracer["after_sbc_S"])[..., :NLEV],
        "stage1_T": np.asarray(oracle_stage["T"]),
        "stage1_S": np.asarray(oracle_stage["S"]),
    }
    row_masks = {
        "metric_zFu": masks["u"], "metric_zFv": masks["v"],
        "metric_zFw": masks["w"],
        **{name: masks["T"] for name in ORDER[3:]},
    }

    production: dict[str, np.ndarray] = {}
    out = _run(card, state, freshwater, surface, endpoint=None, exposure="transport")
    production.update({
        "metric_zFu": np.asarray(out.u.data)[:, 1:OWNED_NX + 1, :NLEV],
        "metric_zFv": np.asarray(out.v.data)[1:OWNED_NY + 1, :OWNED_NX, :NLEV],
        "metric_zFw": np.asarray(out.T.data)[:, :OWNED_NX, :NLEV],
    })
    out = _run(card, state, freshwater, surface, endpoint=None, exposure="stage1")
    production.update({
        "stage1_T": np.asarray(out.T.data)[:, :OWNED_NX, :NLEV],
        "stage1_S": np.asarray(out.S.data)[:, :OWNED_NX, :NLEV],
    })

    endpoint = _endpoint_override(card, entry, record_root)
    endpoint_arm: dict[str, np.ndarray] = {}
    out = _run(card, state, freshwater, surface, endpoint=endpoint, exposure="transport")
    endpoint_arm.update({
        "metric_zFu": np.asarray(out.u.data)[:, 1:OWNED_NX + 1, :NLEV],
        "metric_zFv": np.asarray(out.v.data)[1:OWNED_NY + 1, :OWNED_NX, :NLEV],
        "metric_zFw": np.asarray(out.T.data)[:, :OWNED_NX, :NLEV],
    })
    for boundary in ("after_advection", "after_sbc"):
        out = _run(card, state, freshwater, surface, endpoint=endpoint,
                   exposure=boundary)
        endpoint_arm[f"{boundary}_T"] = np.asarray(out.T.data)[:, :OWNED_NX, :NLEV]
        endpoint_arm[f"{boundary}_S"] = np.asarray(out.S.data)[:, :OWNED_NX, :NLEV]
    out = _run(card, state, freshwater, surface, endpoint=endpoint, exposure="stage1")
    endpoint_arm.update({
        "stage1_T": np.asarray(out.T.data)[:, :OWNED_NX, :NLEV],
        "stage1_S": np.asarray(out.S.data)[:, :OWNED_NX, :NLEV],
    })

    production_rows = {
        name: bit_score(values, expected[name], row_masks[name])
        for name, values in production.items()
    }
    endpoint_rows = {
        name: bit_score(endpoint_arm[name], expected[name], row_masks[name])
        for name in ORDER
    }
    if plant:
        planted = expected["stage1_T"].copy()
        target = tuple(np.argwhere(row_masks["stage1_T"])[0])
        planted[target] = np.nextafter(planted[target], np.inf)
        row = bit_score(planted, expected["stage1_T"], row_masks["stage1_T"])
        require(row["unequal"] == 1, "one-ULP stage-1 plant did not fire once")
        raise GateError("planted stage-1 tracer cell rejected through scorer")

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "claim_label": "GIVEN_NEMO_ENTRY_DECISION52",
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
        },
        "record": {
            "root": str(record_root), "tracer_sha256": sha256(tracer_path),
            "stage1_sha256": sha256(stage_path), "schema": tracer["header"],
        },
        "decision52_entry": entry_rows,
        "support": {name: int(mask.sum()) for name, mask in row_masks.items()},
        "production_rows": production_rows,
        "production_first_non_bit": first_non_bit(production_rows),
        "endpoint_rows": endpoint_rows,
        "endpoint_first_non_bit": first_non_bit(endpoint_rows),
        "endpoint_closes_all": all(row["bit_exact"] for row in endpoint_rows.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.record_root, plant=args.plant)
    except (GateError, OSError, ValueError, IndexError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
