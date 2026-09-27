#!/usr/bin/env python3
"""Walk ORCA2's stage-1 QCO/RK tracer assignment in compiled order."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.eos import nemo_r3t_stretch  # noqa: E402
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
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round43_tracer_handoff_gate as handoff,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round44_source_order_gate as source_gate,
)

NX, NY, NZ = 94, 152, 31
OWNED_NX, NLEV = 90, 30


class GateError(RuntimeError):
    """A fail-closed round-45 condition was not met."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _rk_operands(path: Path) -> dict[str, np.ndarray | tuple[int, ...]]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    expected = (1, 1, 1, 1, 1, 3, 3, NX, NY, NZ, 64)
    require(magic == "NEMO_L2_RKTRA_1", f"bad RK magic {magic!r}")
    require(header == expected, f"bad RK header {header}")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 15 * n3 + 3 * n2, "bad RK payload size")
    require(np.isfinite(values).all(), "non-finite RK payload")
    names3 = (
        "Krhs_entry_T", "Krhs_entry_S", "zFu", "zFv", "zFw",
        "Krhs_after_advection_T", "Krhs_after_advection_S",
        "Krhs_after_sbc_T", "Krhs_after_sbc_S", "Kbb_T", "Kbb_S",
        "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result: dict[str, np.ndarray | tuple[int, ...]] = {
        name: _xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate(names3)
    }
    cursor = len(names3) * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        result[name] = _xy(values[cursor + index * n2:cursor + (index + 1) * n2])
    result["header"] = header
    return result


def _numpy_literal(kbb, krhs, r3bb, r3mm, r3aa, tmask, stage_dt):
    """stprk3_stg.f90:674-677, one NumPy operation per source operator."""
    one = np.asarray(1.0, np.float64)
    left = (one + r3bb[..., None]) * kbb
    right = np.asarray(stage_dt, np.float64) * (one + r3mm[..., None])
    right = right * krhs
    right = right * tmask
    numerator = left + right
    return numerator / (one + r3aa[..., None])


@jax.jit
def _jax_fused(kbb, krhs, r3bb, r3mm, r3aa, tmask, stage_dt):
    return (
        (1.0 + r3bb[..., None]) * kbb
        + stage_dt * (1.0 + r3mm[..., None]) * krhs * tmask
    ) / (1.0 + r3aa[..., None])


@jax.jit
def _jax_source_ordered(kbb, krhs, r3bb, r3mm, r3aa, tmask, stage_dt):
    """Same assignment with compiler barriers at Fortran statement boundaries."""
    one = jnp.asarray(1.0, dtype=kbb.dtype)
    qbb = nemo_source_round(one + r3bb[..., None])
    qmm = nemo_source_round(one + r3mm[..., None])
    qaa = nemo_source_round(one + r3aa[..., None])
    left = nemo_source_round(qbb * kbb)
    right = nemo_source_round(stage_dt * qmm)
    right = nemo_source_round(right * krhs)
    right = nemo_source_round(right * tmask)
    numerator = nemo_source_round(left + right)
    return nemo_source_round(numerator / qaa)


def _score2(actual: np.ndarray, expected: np.ndarray, mask2: np.ndarray) -> dict[str, object]:
    mask3 = np.broadcast_to(mask2[..., None], actual.shape)
    return handoff.bit_score(actual, expected, mask3)


def validate(deck_root: Path, record_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-45 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    require(card.recipe.model_config.tracer_advection == "fct2",
            "ORCA2 no longer selects FCT2")
    record_path = record_root / phase2l.TRACER_RECORD
    stage_path = record_root / "oracle_stage_kt00000001_s1.bin"
    final_path = record_root / phase2l.STAGE3_RECORD
    for path in (record_path, stage_path, final_path):
        require(path.is_file(), f"missing admitted record {path}")
    operands = _rk_operands(record_path)
    oracle_stage = ladder.read_state_frame(stage_path, kt=1, stage=1)
    source = source_gate.validate(deck_root, record_root, plant=False)
    require(source["production_rows"]["after_sbc_T"]["bit_exact"]
            and source["production_rows"]["after_sbc_S"]["bit_exact"],
            "round-44 exact after-SBC boundary no longer reproduces")

    mask3 = handoff._support_masks(card)["T"]
    mask2 = np.any(mask3, axis=-1)
    tmask = np.asarray(card.recipe.z_coord.is_active, np.float64)[:, :OWNED_NX, :NLEV]
    entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    eta_final = phase2l.read_final_ssh(final_path)
    eta_final_full = np.asarray(entry["ssh"], np.float64).copy()
    eta_final_full[:, :OWNED_NX] = eta_final
    eta_one_third = (
        np.asarray(entry["ssh"], np.float64)
        + (eta_final_full - np.asarray(entry["ssh"], np.float64)) / 3.0
    )
    qbb_model = np.asarray(nemo_r3t_stretch(
        card.recipe.z_coord, jnp.asarray(entry["ssh"]),
        card.recipe.initial_state.H_bathy.data, evaluation="nemo_reciprocal"))[:, :OWNED_NX]
    qaa_model = np.asarray(nemo_r3t_stretch(
        card.recipe.z_coord, jnp.asarray(eta_one_third),
        card.recipe.initial_state.H_bathy.data, evaluation="nemo_reciprocal"))[:, :OWNED_NX]

    recorded_r3 = {
        name: np.asarray(operands[name], np.float64)[:, :OWNED_NX]
        for name in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")
    }
    recorded_q = {name: 1.0 + value for name, value in recorded_r3.items()}
    q_rows = {
        "q_Kbb": _score2(qbb_model, recorded_q["r3t_Kbb"], mask2),
        "q_Kmm": _score2(qbb_model, recorded_q["r3t_Kmm"], mask2),
        "q_Kaa": _score2(qaa_model, recorded_q["r3t_Kaa"], mask2),
    }

    stage_dt = np.asarray(card.dt_s / 3.0, np.float64)
    replays: dict[str, dict[str, object]] = {}
    for tracer_name in ("T", "S"):
        kbb = np.asarray(operands[f"Kbb_{tracer_name}"], np.float64)[:, :OWNED_NX, :NLEV]
        krhs = np.asarray(operands[f"Krhs_after_sbc_{tracer_name}"], np.float64)[:, :OWNED_NX, :NLEV]
        kaa = np.asarray(operands[f"Kaa_{tracer_name}"], np.float64)[:, :OWNED_NX, :NLEV]
        oracle = np.asarray(oracle_stage[tracer_name], np.float64)
        require(np.array_equal(kaa, oracle),
                f"RK record Kaa_{tracer_name} differs from stage record")
        args = (kbb, krhs, recorded_r3["r3t_Kbb"], recorded_r3["r3t_Kmm"],
                recorded_r3["r3t_Kaa"], tmask, stage_dt)
        literal = _numpy_literal(*args)
        fused = np.asarray(_jax_fused(*map(jnp.asarray, args)))
        ordered = np.asarray(_jax_source_ordered(*map(jnp.asarray, args)))
        mixed_kbb = _numpy_literal(
            kbb, krhs, qbb_model - 1.0, recorded_r3["r3t_Kmm"],
            recorded_r3["r3t_Kaa"], tmask, stage_dt)
        mixed_kmm = _numpy_literal(
            kbb, krhs, recorded_r3["r3t_Kbb"], qbb_model - 1.0,
            recorded_r3["r3t_Kaa"], tmask, stage_dt)
        mixed_kaa = _numpy_literal(
            kbb, krhs, recorded_r3["r3t_Kbb"], recorded_r3["r3t_Kmm"],
            qaa_model - 1.0, tmask, stage_dt)
        if plant and tracer_name == "T":
            target = tuple(np.argwhere(mask3)[0])
            literal[target] = np.nextafter(literal[target], np.inf)
        replays[tracer_name] = {
            "numpy_literal": handoff.bit_score(literal, kaa, mask3),
            "jax_fused": handoff.bit_score(fused, kaa, mask3),
            "jax_source_ordered": handoff.bit_score(ordered, kaa, mask3),
            "model_q_Kbb_only": handoff.bit_score(mixed_kbb, kaa, mask3),
            "model_q_Kmm_only": handoff.bit_score(mixed_kmm, kaa, mask3),
            "model_q_Kaa_only": handoff.bit_score(mixed_kaa, kaa, mask3),
            "production_stage1": source["production_rows"][f"stage1_{tracer_name}"],
        }

    if plant:
        require(replays["T"]["numpy_literal"]["unequal"] == 1,
                "one-ULP QCO/RK plant did not fire once")
        raise GateError("planted QCO/RK tracer cell rejected through scorer")

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "claim_label": "GIVEN_NEMO_ENTRY_DECISION52",
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
        },
        "compiled_statement": "stprk3_stg.f90:670-681",
        "record": {
            "root": str(record_root), "rk_sha256": sha256(record_path),
            "stage1_sha256": sha256(stage_path), "header": list(operands["header"]),
        },
        "resolved": {"stage": 1, "stage_dt": float(stage_dt), "qco": True},
        "q_operand_rows": q_rows,
        "replay_rows": replays,
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
