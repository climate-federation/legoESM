#!/usr/bin/env python3
"""Replay rung-10's two live Redi calls against the admitted R333 record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
    nemo_testcase_l4_orca2_round333_tracer_replay as outer,
    nemo_testcase_l4_orca2_round333_traldf_gate as record_gate,
)


class GateError(RuntimeError):
    """The production replay is incomplete or moved its ordinary state."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


DIAGNOSTICS = (
    "actual_tendency",
    "dit", "djt", "dkt", "A11", "A22", "hmsku", "hmskv", "A13", "A23",
    "zfu", "zfv", "vmsku", "vmskv", "ahu_w", "ahv_w", "A31", "A32",
    "zfw_top", "zfw_kp1", "tendency", "r1_e1e2t", "e3t", "tmask",
)


def _global(records: list[dict[str, np.ndarray]], name: str,
            tracer: int | None = None) -> np.ndarray:
    owned = []
    for fields in records:
        value = fields[name]
        if tracer is not None:
            value = value[..., tracer]
        if value.ndim == 3:
            value = value[2:-2, 2:-2].transpose(1, 0, 2)
        elif value.ndim == 2:
            value = value[2:-2, 2:-2].T
        else:
            raise GateError(f"{name}: unsupported owned rank {value.ndim}")
        owned.append(value)
    return np.concatenate(owned, axis=1)


def _score(candidate: np.ndarray, oracle: np.ndarray,
           support: np.ndarray) -> dict[str, object]:
    candidate = np.broadcast_to(candidate, oracle.shape)
    require(candidate.shape == oracle.shape == support.shape, "score shapes moved")
    cand = np.ascontiguousarray(candidate[support])
    ref = np.ascontiguousarray(oracle[support])
    delta = cand - ref
    return {
        "count": int(cand.size),
        "bit_unequal": int(np.count_nonzero(
            cand.view(np.uint64) != ref.view(np.uint64))),
        "value_unequal": int(np.count_nonzero(cand != ref)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def _legacy_high_level(original):
    def legacy(*args, **kwargs):
        kwargs["redi_divisor_thickness_override"] = None
        return original(*args, **kwargs)
    return legacy


def measure(label: str, arm: str, deck_root: Path, record_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as gm

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "internal replay requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "internal replay requires production JIT on CPU")
    require(label in ("given_nemo_entry", "independent"), "unknown label")
    require(arm in ("control", "candidate"), "unknown arm")

    producer = (record_root / "producer_commit.txt").read_text().strip()
    admission = record_gate.run_gate(record_root, expect_commit=producer)
    require(admission["status"] == "PASS_R333_TRALDF_INTERNAL_ACQUISITION",
            "internal record is not admitted")
    records = [record_gate.read_record(record_root / record_gate.record_name(rank))["fields"]
               for rank in (0, 1)]

    _, shipped = rung10.card_fields(deck_root)
    card = r329.selected_card(shipped, arm="recordice", rung="rung10")
    state0 = card.recipe.initial_state
    if label == "given_nemo_entry":
        entry = rung10.assemble_state_fields(record_root, 1, stage=None)
        state0 = state0._replace(
            eta=state0.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))

    original_high = model_module.gm_redi_tracer_tendency_latlon
    if arm == "control":
        model_module.gm_redi_tracer_tendency_latlon = _legacy_high_level(original_high)
    try:
        base = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing)
        f1, s1, d1 = outer._surface(card, deck_root, record_root, 1)
        state1 = base.step(state0, dt=card.dt_s, freshwater=f1,
                           surface_forcing=s1, **d1)
        state1 = jax.tree.map(lambda value: value.block_until_ready(), state1)
        f2, s2, d2 = outer._surface(card, deck_root, record_root, 2)
        ordinary = base.step(state1, dt=card.dt_s, freshwater=f2,
                             surface_forcing=s2, **d2)
        ordinary = jax.tree.map(lambda value: value.block_until_ready(), ordinary)

        captured: list[dict[str, np.ndarray]] = []
        original_low = gm.nemo_iso_lap_tracer_tendency_latlon_cgrid

        def store(*values) -> None:
            captured.append({name: np.array(value, copy=True)
                             for name, value in zip(DIAGNOSTICS, values, strict=True)})

        def observed_low(*args, **kwargs):
            result = original_low(*args, **kwargs)
            # The production R333 call requests the through-FCT bolus transport,
            # so its ordinary return is ``(tendency, bolus_transport)``.  Record
            # only the live tendency while returning that tuple untouched.
            actual_tendency = result[0] if isinstance(result, tuple) else result
            probe = dict(kwargs)
            probe.update(return_bolus=False, return_diagnostics=True,
                         return_operand_diagnostics=True)
            _tendency, diagnostics = original_low(*args, **probe)
            jax.debug.callback(
                store, actual_tendency,
                *(diagnostics[name] for name in DIAGNOSTICS[1:]),
                ordered=True)
            return result

        gm.nemo_iso_lap_tracer_tendency_latlon_cgrid = observed_low
        try:
            observed_model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
                iwm_forcing=card.recipe.iwm_forcing)
            observed = observed_model.step(
                state1, dt=card.dt_s, freshwater=f2, surface_forcing=s2, **d2)
            observed = jax.tree.map(lambda value: value.block_until_ready(), observed)
        finally:
            gm.nemo_iso_lap_tracer_tendency_latlon_cgrid = original_low
    finally:
        model_module.gm_redi_tracer_tendency_latlon = original_high

    passive, changed = outer._tree_bit_identical(ordinary, observed)
    require(passive and changed == 0, "diagnostic replay moved the ordinary state")
    require(len(captured) == 2, "expected one T and one S diagnostic call")

    oracle_stage3 = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3)
    rows: dict[str, list[dict[str, object]]] = {}
    first_owner = {}
    for tracer_index, tracer in enumerate(("T", "S")):
        diag = captured[tracer_index]
        hseen = _global(records, "horizontal_seen", tracer_index)[..., :30] == 1.0
        vseen = _global(records, "vertical_seen", tracer_index)[..., :30] == 1.0
        ordinary_v = vseen.copy(); ordinary_v[..., -2:] = False
        targets_rank0 = np.asarray(observed._asdict()[tracer].data)[:, :90, :30] != oracle_stage3[tracer]
        expected_targets = 18 if tracer == "T" else 13
        if arm == "control":
            require(int(np.count_nonzero(targets_rank0)) == expected_targets,
                    f"{tracer}: registered target census moved")
        targets = np.zeros_like(vseen)
        targets[:, :90] = targets_rank0

        div = _global(records, "divergence", tracer_index)[..., :30]
        tmask = _global(records, "tmask")[..., :30]
        stretched = (_global(records, "e3t_3d")[..., :30]
                     * (1.0 + _global(records, "r3t_kmm")[..., None] * tmask))
        r1 = _global(records, "r1_e1e2t")[..., None]
        raw = ((div * r1) / stretched) * tmask
        rhs_post = (_global(records, "rhs_preadd", tracer_index)[..., :30]
                    + diag["tendency"][..., :30])

        specs = (
            ("dit", "dit", "dit", hseen), ("djt", "djt", "djt", hseen),
            ("dkt", "dkt", "dkt", hseen), ("A11", "A11", "A11", hseen),
            ("A22", "A22", "A22", hseen),
            ("hmsku", "hmsku", "hmsku", hseen),
            ("hmskv", "hmskv", "hmskv", hseen),
            ("A13", "A13", "A13", hseen), ("A23", "A23", "A23", hseen),
            ("zfu", "zfu", "zfu", hseen), ("zfv", "zfv", "zfv", hseen),
            ("vmsku", "vmsku", "vmsku", ordinary_v),
            ("vmskv", "vmskv", "vmskv", ordinary_v),
            ("zahu_w", "ahu_w", "zahu_w", ordinary_v),
            ("zahv_w", "ahv_w", "zahv_w", ordinary_v),
            ("A31", "A31", "A31", ordinary_v),
            ("A32", "A32", "A32", ordinary_v),
            ("zfw_lower", "zfw_top", "zfw_lower", vseen),
            ("zfw_upper", "zfw_kp1", "zfw_upper", vseen),
        )
        tracer_rows = []
        for name, candidate_name, oracle_name, support in specs:
            candidate = diag[candidate_name][..., :30]
            oracle = _global(
                records, oracle_name,
                tracer_index if records[0][oracle_name].ndim == 4 else None)[..., :30]
            tracer_rows.append({
                "statement": name,
                "everywhere_executed": _score(candidate, oracle, support),
                "registered_targets": _score(candidate, oracle, targets & support),
            })
        derived = (
            ("r1_e1e2t", np.broadcast_to(diag["r1_e1e2t"][..., None], raw.shape),
             np.broadcast_to(r1, raw.shape)),
            ("e3t_divisor", diag["e3t"][..., :30], stretched),
            ("actual_scaled_tendency", diag["actual_tendency"][..., :30], raw),
            ("scaled_tendency", diag["tendency"][..., :30], raw),
            ("rhs_add", rhs_post,
             _global(records, "rhs_postadd", tracer_index)[..., :30]),
        )
        for name, candidate, oracle in derived:
            tracer_rows.append({
                "statement": name,
                "everywhere_executed": _score(candidate, oracle, vseen),
                "registered_targets": _score(candidate, oracle, targets),
            })
        rows[tracer] = tracer_rows
        first_owner[tracer] = next(
            row["statement"] for row in tracer_rows
            if row["everywhere_executed"]["value_unequal"])

    require(arm != "control" or first_owner == {
        "T": "e3t_divisor", "S": "e3t_divisor"},
        f"first owner moved: {first_owner}")
    return {
        "format": "nemo-testcase-l4-orca2-round333-traldf-replay-v1",
        "status": "PASS_R333_TRALDF_REPLAY",
        "label": label, "arm": arm, "worktree": stamp,
        "record_admission": admission["status"],
        "observer_passivity": {"bit_identical": passive,
                               "changed_leaves": changed, "calls": len(captured)},
        "first_value_owner": first_owner,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("given_nemo_entry", "independent"), required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.label, args.arm, args.deck_root,
                         args.record_root, args.expect_commit)
    except (GateError, record_gate.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
