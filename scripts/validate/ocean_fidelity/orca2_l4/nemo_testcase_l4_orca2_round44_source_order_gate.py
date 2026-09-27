#!/usr/bin/env python3
"""Walk ORCA2's stage-1 EMP and runoff source statements in source order."""

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
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
    validate_nemo_testcase_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round12_runoff_gate as runoff_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round43_tracer_handoff_gate as handoff,
)

OWNED_NX, NLEV = 90, 30


class GateError(RuntimeError):
    """A fail-closed round-44 condition was not met."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_operands(card, deck_root: Path, record_root: Path):
    tracer_path = record_root / handoff.phase2l.TRACER_RECORD
    tracer = handoff.phase2l.read_tracer(tracer_path)
    entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    surface = ladder.assemble_surface_fields(record_root, 1)
    mesh = runoff_gate._mesh(record_root)
    for name, value in {**entry, **surface, **mesh}.items():
        require(np.isfinite(np.asarray(value)).all(), f"non-finite operand {name}")

    tmask = np.asarray(mesh["tmask"], dtype=np.float64)
    e3t0 = np.asarray(mesh["e3t_0"], dtype=np.float64)
    ssmask = np.max(tmask, axis=-1)
    ht0 = np.sum(e3t0 * tmask, axis=-1)
    r1_ht0 = ssmask / (ht0 + 1.0 - ssmask)
    r3t = np.asarray(entry["ssh"], dtype=np.float64) * r1_ht0
    h_top = e3t0[..., 0] * (1.0 + r3t * tmask[..., 0])
    require(np.all(h_top[tmask[..., 0] > 0.5] > 0.0),
            "non-positive wet top-cell thickness")
    return tracer_path, tracer, entry, surface, tmask, h_top


def _literal_replay(after_adv, kbb, emp, runoff_content, h_top, r1_rho0):
    """trasbc.f90:284-286,321-324, preserving statement order."""
    out = np.array(after_adv, dtype=np.float64, copy=True)
    z1_rho0_e3t = np.asarray(r1_rho0, np.float64) / h_top
    out[..., 0] = out[..., 0] - emp * kbb[..., 0] * z1_rho0_e3t
    zdep = np.asarray(1.0, np.float64) / h_top
    out[..., 0] = out[..., 0] + runoff_content * zdep
    return out


def _model_spelling_replay(
    after_adv, kbb, emp, runoff_content, h_top, rho0, *, multiply_reciprocal,
):
    """Replay the current legoESM association, with one reciprocal switch."""
    if multiply_reciprocal:
        fw_eta = (-emp) * (np.asarray(1.0, np.float64) / rho0)
    else:
        fw_eta = (-emp) / rho0
    emp_rate = (fw_eta * kbb[..., 0]) / h_top
    runoff_rate = runoff_content * (np.asarray(1.0, np.float64) / h_top)
    out = np.array(after_adv, dtype=np.float64, copy=True)
    out[..., 0] = out[..., 0] + emp_rate
    out[..., 0] = out[..., 0] + runoff_rate
    return out


def validate(deck_root: Path, record_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-44 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    require(cfg.tracer_advection == "fct2", "ORCA2 no longer selects FCT2")
    require(not card.recipe.z_coord.linear_free_surface,
            "ORCA2 unexpectedly selects linear free surface")

    (tracer_path, tracer, entry, surface_fields, tmask, h_top) = (
        _source_operands(card, deck_root, record_root))
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    state = card.recipe.initial_state._replace(
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    entry_rows = ladder.compare_fields(ladder._candidate_fields(state), entry)
    require(all(entry_rows["rows"][name]["bit_identical"]
                for name in ("T", "S", "u", "v", "ssh")),
            "Decision-52 kt=1 entry bridge is not bit-exact")

    endpoint = handoff._endpoint_override(card, entry, record_root)
    expected = {
        boundary: {
            tracer_name: np.asarray(
                tracer[f"{boundary}_{tracer_name}"], dtype=np.float64
            )[..., :NLEV]
            for tracer_name in ("T", "S")
        }
        for boundary in ("after_advection", "after_sbc")
    }
    shape = state.T.data.shape
    zfu = np.zeros(shape, dtype=np.float64)
    zfv = np.zeros(shape, dtype=np.float64)
    zfw = np.zeros(shape[:-1] + (shape[-1] + 1,), dtype=np.float64)
    zfu[:, :OWNED_NX] = np.asarray(tracer["zFu"])[..., :NLEV]
    zfv[:, :OWNED_NX] = np.asarray(tracer["zFv"])[..., :NLEV]
    zfw[:, :OWNED_NX] = np.asarray(tracer["zFw"])
    transport = tuple(map(jnp.asarray, (zfu, zfv, zfw)))
    production: dict[str, np.ndarray] = {}
    for boundary in ("after_advection", "after_sbc"):
        out = handoff._run(
            card, state, freshwater, surface, endpoint=endpoint,
            exposure=boundary, transport_override=transport)
        production[f"{boundary}_T"] = np.asarray(out.T.data)[:, :OWNED_NX, :NLEV]
        production[f"{boundary}_S"] = np.asarray(out.S.data)[:, :OWNED_NX, :NLEV]

    masks = handoff._support_masks(card)
    mask = masks["T"]
    production_rows = {
        name: handoff.bit_score(
            values, expected[name.rsplit("_", 1)[0]][name.rsplit("_", 1)[1]], mask)
        for name, values in production.items()
    }
    require(production_rows["after_advection_T"]["bit_exact"]
            and production_rows["after_advection_S"]["bit_exact"],
            "recorded-W arm does not close centered advection")

    emp = np.asarray(surface_fields["emp"], dtype=np.float64)[:, :OWNED_NX]
    rnf_tsc = np.asarray(surface_fields["rnf_tsc"], dtype=np.float64)[:, :OWNED_NX]
    h_rank = h_top[:, :OWNED_NX]
    rho0 = np.asarray(cfg.rho_0, dtype=np.float64)
    r1_rho0 = np.asarray(1.0, dtype=np.float64) / rho0
    replay_rows: dict[str, dict[str, dict[str, object]]] = {}
    replay_match_production: dict[str, dict[str, object]] = {}
    for tracer_index, name in enumerate(("T", "S")):
        adv = expected["after_advection"][name]
        kbb = np.asarray(tracer[f"Kbb_{name}"], dtype=np.float64)[..., :NLEV]
        oracle = expected["after_sbc"][name]
        literal = _literal_replay(
            adv, kbb, emp, rnf_tsc[..., tracer_index], h_rank, r1_rho0)
        divide = _model_spelling_replay(
            adv, kbb, emp, rnf_tsc[..., tracer_index], h_rank, rho0,
            multiply_reciprocal=False)
        multiply = _model_spelling_replay(
            adv, kbb, emp, rnf_tsc[..., tracer_index], h_rank, rho0,
            multiply_reciprocal=True)
        if plant and name == "T":
            target = tuple(np.argwhere(mask)[0])
            literal[target] = np.nextafter(literal[target], np.inf)
        replay_rows[name] = {
            "nemo_literal": handoff.bit_score(literal, oracle, mask),
            "model_divide_rho0": handoff.bit_score(divide, oracle, mask),
            "model_multiply_r1_rho0": handoff.bit_score(multiply, oracle, mask),
        }
        replay_match_production[name] = handoff.bit_score(
            divide, production[f"after_sbc_{name}"], mask)

    if plant:
        require(replay_rows["T"]["nemo_literal"]["unequal"] == 1,
                "one-ULP literal-source plant did not fire once")
        raise GateError("planted source-order tracer cell rejected through scorer")

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "claim_label": "GIVEN_NEMO_ENTRY_DECISION52",
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
        },
        "record": {
            "root": str(record_root), "tracer_sha256": sha256(tracer_path),
            "schema": tracer["header"],
        },
        "resolved": {
            "stage": 1, "nonlinear_free_surface": True,
            "qns_sfx": "STRUCTURALLY_INACTIVE_AT_STAGE1",
            "source_order": ["EMP_DILUTION", "RIVER_RUNOFF"],
            "rho0": float(rho0), "r1_rho0": float(r1_rho0),
        },
        "decision52_entry": entry_rows,
        "production_rows": production_rows,
        "replay_rows": replay_rows,
        "model_divide_replay_matches_production": replay_match_production,
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
