#!/usr/bin/env python3
"""Walk the held QCO change through the admitted OVERFLOW kt=3 record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.eos import nemo_r3t_stretch
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l1_overflow_round50_pair_gate import (
    GateError,
    admit,
    read_record,
    require,
)
from nemo_testcase_phase3_trajectory_gate import (
    expected_masks,
    lego_fields,
    read_entry,
)

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round50/acquisition/oracle_overflow_kt3_pair"
)
PRODUCER_COMMIT = "932cbfa9ec2f2fbcbf51a03ca8e46e5b39b78c62"
SOURCE_ORDER = (
    "kt3.entry.T", "kt3.entry.S", "kt3.entry.u", "kt3.entry.v",
    "kt3.entry.ssh",
    "s1.tracer.Kaa.T", "s1.tracer.Kaa.S", "s1.tracer.Kaa.ssh",
    "s2.entry.T", "s2.entry.S", "s2.entry.ssh",
    "s2.hpg.u", "s2.hpg.v",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _nemo_owned(value: np.ndarray) -> np.ndarray:
    """Map the round-50 owned Fortran (x,y,z) payload to (y,x,z)."""
    value = np.asarray(value)
    require(value.ndim in (2, 3), f"unexpected record rank {value.ndim}")
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[:, :, 0].T
    axes = (1, 0) if value.ndim == 2 else (1, 0, 2)
    return value.transpose(axes)


def _u(value) -> np.ndarray:
    return np.asarray(value)[:, 1:, ...]


def _v(value) -> np.ndarray:
    return np.asarray(value)[1:, ...]


def _physical_levels(value, mask) -> np.ndarray:
    """Select the card's physical levels from a state with a bottom halo."""
    value = np.asarray(value)
    mask = np.asarray(mask)
    require(value.ndim == mask.ndim == 3, "physical-level arrays must be 3-D")
    require(value.shape[:2] == mask.shape[:2], "physical-level horizontal drift")
    require(value.shape[-1] in (mask.shape[-1], mask.shape[-1] + 1),
            "unexpected physical-level extent")
    return value[..., :mask.shape[-1]]


def _score(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape,
            f"{name}: shape mismatch {oracle.shape}/{candidate.shape}/{active.shape}")
    require(candidate.dtype == np.float64,
            f"{name}: candidate dtype is {candidate.dtype}")
    if not active.any():
        return {
            "name": name, "status": "UNMEASURED_NO_ACTIVE_FACE",
            "exact": None, "n": 0, "n_unequal": None,
        }
    tested = candidate.copy() if plant else candidate
    if plant:
        at = tuple(np.argwhere(active)[0])
        tested[at] = np.nextafter(tested[at], np.float64(np.inf))
        require(tested[at] != candidate[at], f"{name}: plant did not move")
    require(np.isfinite(tested[active]).all(), f"{name}: non-finite candidate")
    unequal = tested[active].view(np.uint64) != oracle[active].view(np.uint64)
    n_unequal = int(np.count_nonzero(unequal))
    return {
        "name": name,
        "status": "BIT_EXACT" if n_unequal == 0 else "NON_BIT",
        "exact": n_unequal == 0,
        "n": int(active.sum()),
        "n_unequal": n_unequal,
        "absolute_max": float(np.max(np.abs(tested[active] - oracle[active]))),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
        "plant": plant,
    }


def _active_values(value, mask) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    require(value.shape == active.shape, "sidecar mask shape mismatch")
    return np.ascontiguousarray(value[active])


def _record_pair(root: Path) -> tuple[dict, dict]:
    momentum = {
        stage: read_record(
            root / f"oracle_r50_momentum_kt00000003_s{stage}.bin",
            "momentum", stage,
        )["fields"]
        for stage in (1, 2, 3)
    }
    tracer = {
        stage: read_record(
            root / f"oracle_r50_tracer_kt00000003_s{stage}.bin",
            "tracer", stage,
        )["fields"]
        for stage in (1, 2, 3)
    }
    return momentum, tracer


def _given_input_rows(card, momentum: dict, tracer: dict, masks: dict,
                      plant: str | None) -> list[dict]:
    """Score the QCO, EOS, and HPG statements on NEMO-recorded operands."""
    from legoesm.core.source_rounding import nemo_source_round

    s1 = tracer[1]
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    tmask = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)

    @jax.jit
    def qco(kbb, rhs, qbb, qmm, qaa, stage_dt):
        left = nemo_source_round(qbb[..., None] * kbb)
        right = nemo_source_round(stage_dt * qmm[..., None])
        right = nemo_source_round(right * rhs)
        right = nemo_source_round(right * tmask)
        return nemo_source_round(
            nemo_source_round(left + right) / qaa[..., None])

    ssh = {
        name: _nemo_owned(s1[name])
        for name in ("Kbb_ssh", "Kmm_ssh", "Kaa_ssh")
    }
    depth = np.asarray(card.recipe.initial_state.H_bathy.data)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    require(depth.shape == ssh["Kbb_ssh"].shape,
            "card column-depth layout does not match the recorded SSH")
    q = {
        name: np.asarray(nemo_r3t_stretch(
            card.recipe.z_coord, jnp.asarray(value),
            jnp.asarray(depth),
            evaluation="nemo_reciprocal",
        ))
        for name, value in ssh.items()
    }
    rows = []
    for field in ("T", "S"):
        got = np.asarray(qco(
            jnp.asarray(_nemo_owned(s1[f"Kbb_{field}"])),
            jnp.asarray(_nemo_owned(s1[f"after_sbc_{field}"])),
            jnp.asarray(q["Kbb_ssh"]), jnp.asarray(q["Kmm_ssh"]),
            jnp.asarray(q["Kaa_ssh"]),
            jnp.asarray(card.dt_s / 3.0, dtype=jnp.float64),
        ))
        rows.append(_score(
            f"given.s1.qco.{field}",
            _nemo_owned(s1[f"Kaa_{field}"]), got, masks[field],
        ))

    # Production-JIT momentum component evaluation from the exact stage-2
    # NEMO T/S/ssh bundle. HPG does not consume velocity; zero velocity makes
    # that exclusion explicit while the same production tendencies path runs.
    s2 = momentum[2]
    zero_u = jnp.zeros_like(card.recipe.initial_state.u.data)
    zero_v = jnp.zeros_like(card.recipe.initial_state.v.data)
    state = card.recipe.initial_state._replace(
        T=card.recipe.initial_state.T.replace(
            data=jnp.asarray(_nemo_owned(s2["Kmm_T"]))),
        S=card.recipe.initial_state.S.replace(
            data=jnp.asarray(_nemo_owned(s2["Kmm_S"]))),
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(_nemo_owned(s2["Kmm_ssh"]))),
        u=card.recipe.initial_state.u.replace(data=zero_u),
        v=card.recipe.initial_state.v.replace(data=zero_v),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    @jax.jit
    def operators(value):
        return model.tendencies(
            value, None, dt=card.dt_s, momentum_only=True,
            skip_lateral_viscosity=True, nemo_operator_association=True,
            return_nemo_operator_components=True,
        )

    _, _, components = operators(state)
    rhd = np.asarray(components["operand_rho_prime"] / card.recipe.model_config.rho_0)
    rows.append(_score(
        "given.s2.eos.rhd", _nemo_owned(s2["rhd"]), rhd, masks["T"],
        plant=plant == "rhd",
    ))
    rows.append(_score(
        "given.s2.hpg.u", _nemo_owned(s2["after_hpg_u"]),
        _u(components["after_hpg_u"].data), masks["u"],
    ))
    rows.append(_score(
        "given.s2.hpg.v", _nemo_owned(s2["after_hpg_v"]),
        _v(components["after_hpg_v"].data), masks["v"],
    ))
    require(wet.shape == ssh["Kbb_ssh"].shape,
            "record/card horizontal layout drift")
    return rows


def _live_arrays(card, momentum: dict, tracer: dict) -> tuple[dict, list[dict]]:
    masks = expected_masks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    for _ in range(2):
        state = ordinary.step(state, dt=card.dt_s)

    # The full live-operand observer is deliberately GYRE-specific: it also
    # requires that card's TKE and slow-forcing bundles.  OVERFLOW uses the
    # narrower already-certified WRITE-only hooks from round 49 instead of
    # weakening that guard or fabricating absent operands.
    pair_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_stage=1,
            expose_momentum_operator="hpg",
            expose_momentum_operator_stage=2),
    )
    pair = pair_model.step(state, dt=card.dt_s)

    s2_m = momentum[2]
    s1_t = tracer[1]
    entry = read_entry(ORACLE_ROOT / "oracle_step_entry_kt00000003.bin",
                       "OVERFLOW-zps")
    entry_values = lego_fields(state)

    values = {
        "kt3.entry.T": entry_values["T"],
        "kt3.entry.S": entry_values["S"],
        "kt3.entry.u": entry_values["u"],
        "kt3.entry.v": entry_values["v"],
        "kt3.entry.ssh": entry_values["ssh"],
        "s1.tracer.Kaa.T": np.asarray(pair.T.data),
        "s1.tracer.Kaa.S": np.asarray(pair.S.data),
        "s1.tracer.Kaa.ssh": np.asarray(pair.eta.data),
        "s2.entry.T": np.asarray(pair.T.data),
        "s2.entry.S": np.asarray(pair.S.data),
        "s2.entry.ssh": np.asarray(pair.eta.data),
        "s2.hpg.u": _u(pair.u.data),
        "s2.hpg.v": _v(pair.v.data),
    }
    references = {
        "kt3.entry.T": entry["T"], "kt3.entry.S": entry["S"],
        "kt3.entry.u": entry["u"], "kt3.entry.v": entry["v"],
        "kt3.entry.ssh": entry["ssh"],
        "s1.tracer.Kaa.T": _nemo_owned(s1_t["Kaa_T"]),
        "s1.tracer.Kaa.S": _nemo_owned(s1_t["Kaa_S"]),
        "s1.tracer.Kaa.ssh": _nemo_owned(s1_t["Kaa_ssh"]),
        "s2.entry.T": _nemo_owned(s2_m["Kmm_T"]),
        "s2.entry.S": _nemo_owned(s2_m["Kmm_S"]),
        "s2.entry.ssh": _nemo_owned(s2_m["Kmm_ssh"]),
        "s2.hpg.u": _nemo_owned(s2_m["after_hpg_u"]),
        "s2.hpg.v": _nemo_owned(s2_m["after_hpg_v"]),
    }
    mask_for = {
        name: masks["ssh" if name.endswith("ssh") else
                    "u" if name.endswith(".u") else
                    "v" if name.endswith(".v") else
                    "T" if name.endswith(".T") else "S"]
        for name in SOURCE_ORDER
    }
    for name in SOURCE_ORDER:
        if mask_for[name].ndim == 3:
            values[name] = _physical_levels(values[name], mask_for[name])
            references[name] = _physical_levels(references[name], mask_for[name])
    rows = [_score(name, references[name], values[name], mask_for[name])
            for name in SOURCE_ORDER]
    active = {name: _active_values(values[name], mask_for[name])
              for name in SOURCE_ORDER if mask_for[name].any()}
    return active, rows


def _write_sidecar(output: Path, arrays: dict[str, np.ndarray]) -> dict:
    sidecar = output.with_suffix(".arrays.npz")
    np.savez_compressed(sidecar, **arrays)
    return {
        "path": str(sidecar), "sha256": _sha256(sidecar),
        "fields": list(arrays),
    }


def _load_sidecar(report: dict) -> dict[str, np.ndarray]:
    meta = report["sidecar"]
    path = Path(meta["path"])
    require(path.is_file(), f"missing sidecar {path}")
    require(_sha256(path) == meta["sha256"], f"hash drift in {path}")
    with np.load(path) as stored:
        require(stored.files == meta["fields"], "sidecar field order drift")
        return {name: np.asarray(stored[name]) for name in stored.files}


def compare(reference_path: Path, candidate: dict) -> dict:
    reference = json.loads(reference_path.read_text())
    require(reference["format"] == candidate["format"], "report format drift")
    before, after = _load_sidecar(reference), _load_sidecar(candidate)
    require(before.keys() == after.keys(), "sidecar key drift")
    rows = []
    for name in SOURCE_ORDER:
        if name not in before:
            continue
        require(before[name].shape == after[name].shape, f"{name}: shape drift")
        unequal = before[name].view(np.uint64) != after[name].view(np.uint64)
        rows.append({
            "name": name, "exact": bool(np.array_equal(before[name], after[name])),
            "n": int(before[name].size),
            "n_unequal": int(np.count_nonzero(unequal)),
            "max_abs_move": float(np.max(np.abs(after[name] - before[name]))),
        })
    first = next((row for row in rows if not row["exact"]), None)
    p3 = first is not None and first["name"] in {
        "s1.tracer.Kaa.T", "s1.tracer.Kaa.S"}
    given = {row["name"]: row for row in candidate["given_input_rows"]}
    p4 = (
        given["given.s1.qco.T"]["exact"]
        and given["given.s1.qco.S"]["exact"]
        and given["given.s2.eos.rhd"]["exact"]
        and given["given.s2.hpg.u"]["exact"] is False
    )
    return {
        "status": "MEASURED",
        "reference": str(reference_path),
        "reference_commit": reference["worktree"]["commit"],
        "candidate_commit": candidate["worktree"]["commit"],
        "first_moved_boundary": None if first is None else first,
        "R51-P2": "CONFIRMED" if all(
            row["exact"] for row in rows if row["name"].startswith("kt3.entry."))
            else "REFUTED",
        "R51-P3": "CONFIRMED" if p3 else "REFUTED",
        "R51-P4": "CONFIRMED" if p4 else "REFUTED",
        "rows": rows,
    }


def run(root: Path, expect_commit: str, output: Path,
        reference: Path | None, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"] == expect_commit,
            f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    require(jax.default_backend() == "cpu", "round-51 gate is CPU-only")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    admission = admit(root, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-50 record is not admitted")
    momentum, tracer = _record_pair(root)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    cfg = card.recipe.model_config
    require((cfg.outer_integrator, cfg.momentum_time_integrator,
             cfg.tracer_time_integrator, cfg.pgf_scheme)
            == ("forward_euler", "rk3_ws", "rk3_ws", "nemo_sco"),
            "resolved OVERFLOW program drift")
    masks = expected_masks(card)
    given = _given_input_rows(card, momentum, tracer, masks, plant)
    if plant == "rhd":
        planted = next(row for row in given if row["name"] == "given.s2.eos.rhd")
        require(planted["n_unequal"] >= 1,
                "one-ULP rhd plant did not make the exact row refuse")
        return {
            "format": "nemo-testcase-l1-overflow-round51-pair-v1",
            "status": "PLANTED_REFUSAL",
            "case": "OVERFLOW-zps", "kt": 3,
            "worktree": stamp,
            "precision": "cpu-fp64-libm-production-jit",
            "record_root": str(root),
            "record_admission": {
                "status": admission["status"],
                "producer_commit": admission["producer_commit"],
                "records": len(admission["records"]),
            },
            "given_input_rows": given,
            "plant": plant,
        }
    arrays, live_rows = _live_arrays(card, momentum, tracer)
    report = {
        "format": "nemo-testcase-l1-overflow-round51-pair-v1",
        "status": "PLANTED_REFUSAL" if plant else "ARM_MEASURED",
        "case": "OVERFLOW-zps", "kt": 3,
        "worktree": stamp,
        "precision": "cpu-fp64-libm-production-jit",
        "record_root": str(root),
        "record_admission": {
            "status": admission["status"],
            "producer_commit": admission["producer_commit"],
            "records": len(admission["records"]),
        },
        "source_order": list(SOURCE_ORDER),
        "given_input_rows": given,
        "live_rows": live_rows,
        "plant": plant,
    }
    report["sidecar"] = _write_sidecar(output, arrays)
    if reference is not None:
        report["comparison"] = compare(reference, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--plant", choices=("rhd",))
    args = parser.parse_args(argv)
    try:
        report = run(
            args.record_dir, args.expect_commit, args.output,
            args.reference, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        if args.plant:
            return 2
        return 0
    except (GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
