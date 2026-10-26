#!/usr/bin/env python3
"""Compare SWG_DG stage records and legoESM paths with per-step restarts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Fail closed without importing legoESM at module import.

    The GPU short ladder loads this module after it has selected the GPU, so
    a module-level checkout bootstrap (which defaults to CPU) would refuse
    there; only :func:`main` binds the checkout and platform.
    """
    if not condition:
        raise RuntimeError(message)

ENTRY_RESTART_FIELDS = (
    ("ssh_k", "sshn", "T"),
    ("u_k", "un", "U"),
    ("v_k", "vn", "V"),
    ("u_before", "un", "U"),
    ("v_before", "vn", "V"),
    ("utau_before", "utau_b", "U"),
    ("vtau_before", "vtau_b", "V"),
)
OUTPUT_RESTART_FIELDS = (
    ("ssh_k", "sshn", "T"),
    ("u_k", "un", "U"),
    ("v_k", "vn", "V"),
    ("u_before", "un", "U"),
    ("v_before", "vn", "V"),
    ("utau_now", "utau_b", "U"),
    ("vtau_now", "vtau_b", "V"),
)
FORCING_SIGNED_ZERO_FIELDS = frozenset({
    "utau_before", "vtau_before", "utau_now", "vtau_now",
    "utau_b", "vtau_b", "emp_b", "qns_b", "sfx_b",
})


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    require(spec is not None and spec.loader is not None,
            f"cannot load sibling module {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array_sha256(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_row(reference, candidate, mask, *, name: str, ladder) -> dict:
    """Summarize exact words under one native mask and localize every debt."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == candidate.shape == mask.shape,
            f"{name}: comparison shapes differ")
    require(reference.dtype == candidate.dtype == np.dtype("float64"),
            f"{name}: exact comparison requires float64")
    word = np.dtype("uint64")
    different_bits = (
        np.ascontiguousarray(reference).view(word)
        != np.ascontiguousarray(candidate).view(word)) & mask
    different_values = (reference != candidate) & mask
    signed_zero = ((reference == 0.0) & (candidate == 0.0)
                   & (np.signbit(reference) != np.signbit(candidate)) & mask)
    delta = np.abs(candidate - reference)
    bad = np.argwhere(different_bits)
    return {
        "name": name,
        "bit_identical": not bool(bad.size),
        "status": "AT-BAR" if not bad.size else "DEBT",
        "bit_unequal": int(bad.shape[0]),
        "numeric_unequal": int(np.count_nonzero(different_values)),
        "signed_zero_only": int(np.count_nonzero(signed_zero)),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(delta[mask])) if np.any(mask) else 0.0,
        "reference_sha256": _array_sha256(reference[mask]),
        "candidate_sha256": _array_sha256(candidate[mask]),
        "regions": ladder._localized_comparison(  # noqa: SLF001
            reference, candidate, mask),
    }


def compare_named_fields(reference: dict, candidate: dict, specifications,
                         masks, *, prefix: str, ladder) -> list[dict]:
    """Compare a declared field mapping and refuse incomplete coverage."""
    rows = []
    for candidate_name, reference_name, grid_kind in specifications:
        require(reference_name in reference,
                f"{prefix}: missing reference field {reference_name}")
        require(candidate_name in candidate,
                f"{prefix}: missing candidate field {candidate_name}")
        rows.append(exact_row(
            reference[reference_name], candidate[candidate_name],
            masks[grid_kind], name=f"{prefix}.{candidate_name}", ladder=ladder))
    return rows


def advance_stage_chain(model, boundary, dt):
    """Execute the same three materialized stages used by production step()."""
    stage1 = model.replay_swe_rk3_stage(boundary, dt, 1, boundary)
    stage2 = model.replay_swe_rk3_stage(stage1, dt, 2, boundary)
    stage3 = model.replay_swe_rk3_stage(stage2, dt, 3, boundary)
    return stage1, stage2, stage3


def _stage_item(fields):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry

    return SWERK3StageEntry(**{
        name: jnp.asarray(fields[name], dtype=jnp.float64)
        for name in SWERK3StageEntry._fields
    })


def _stage_dict(stage) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in stage._asdict().items()}


def _restart_specs(restart: dict) -> tuple[tuple[str, str, str], ...]:
    """Map every admitted restart field to itself and its native grid kind."""
    u_fields = {"un", "uu_n", "utau_b"}
    v_fields = {"vn", "vv_n", "vtau_b"}
    return tuple(
        (name, name, "U" if name in u_fields else "V" if name in v_fields else "T")
        for name in restart
    )


def _rows_exact(rows) -> bool:
    return bool(rows) and all(row["bit_identical"] for row in rows)


def _first_debt(groups):
    for group in groups:
        for row in group.get("rows", ()):
            if not row["bit_identical"]:
                return {"kt": group["kt"], **row}
    return None


# Maintainer ruling of 2026-10-13: the bar stays exact for prognostic fields;
# the sign of zero in forcing fields is the one exception, and this check
# (with its planted prognostic violation in the tests) is what enforces it.
SIGNED_ZERO_POLICY = (
    "maintainer_ruling_2026-10-13: prognostic fields exact; "
    "sign of zero in forcing fields excepted")


def signed_zero_scope(comparisons: dict[str, list[dict]]) -> dict:
    """Classify signed-zero words; rows keep their exact word-level status."""
    observed = []
    violations = []
    for comparison, groups in comparisons.items():
        for group in groups:
            for row in group.get("rows", ()):
                if row.get("signed_zero_only", 0):
                    field = row["name"].rsplit(".", 1)[-1]
                    item = {
                        "comparison": comparison,
                        "kt": group["kt"],
                        "field": field,
                        "words": row["signed_zero_only"],
                    }
                    observed.append(item)
                    if field not in FORCING_SIGNED_ZERO_FIELDS:
                        violations.append(item)
    return {
        "policy": SIGNED_ZERO_POLICY,
        "forcing_field_allowlist": sorted(FORCING_SIGNED_ZERO_FIELDS),
        "observed": observed,
        "prognostic_or_unknown_violations": violations,
        "forcing_only_observation": bool(observed) and not violations,
    }


def run(root: Path, *, plant: bool = True) -> dict:
    """Run all four preregistered comparisons on certified CPU arithmetic."""
    import jax
    from legoesm.core.precision import get_policy
    from legoesm.ocean.fidelity.provenance import git_sha

    ladder = _load_sibling(
        "nemo_testcase_l1_swg_ladder_round7",
        "nemo_testcase_l1_swg_ladder.py")
    namelist = ladder._deck_namelist("maddison")  # noqa: SLF001
    card, masks, model, forcing = ladder._swg_production_case(  # noqa: SLF001
        "round-7 restart discriminator", namelist=namelist)
    require(jax.default_backend() == "cpu", "round 7 requires CPU")
    entries = ladder.read_stage_entries(root)
    restarts = ladder.read_restarts(
        root, kts=tuple(range(1, 11)), namelist=namelist)

    # (i) A restart is the completed NEMO step; the next stage-1 record is the
    # next step's Nbb boundary after sbc.  kt=11 was not acquired.
    record_vs_restart = []
    for kt in range(1, 10):
        rows = compare_named_fields(
            restarts[kt], entries[kt + 1, 1], ENTRY_RESTART_FIELDS, masks,
            prefix=f"record_kt{kt + 1}_stage1_vs_restart_kt{kt}", ladder=ladder)
        record_vs_restart.append({"kt": kt, "rows": rows})

    # (ii) Production from rest, and (iv) its materialized stage-1 output.
    production_vs_restart = []
    production_stage1_vs_record = []
    production_boundary_vs_record = []
    state = card.recipe.initial_state
    for kt in range(1, 11):
        boundary_fields = ladder._stage_candidate(  # noqa: SLF001
            state, forcing, card)
        boundary = _stage_item(boundary_fields)
        boundary_rows = [
            exact_row(entries[kt, 1][name], value,
                      ladder._field_mask(name, masks),  # noqa: SLF001
                      name=f"production_boundary.kt{kt}.{name}", ladder=ladder)
            for name, value in boundary_fields.items()
        ]
        production_boundary_vs_record.append({"kt": kt, "rows": boundary_rows})
        stage1 = model.replay_swe_rk3_stage(
            boundary, card.dt_s, 1, boundary)
        jax.block_until_ready(stage1.ssh_k)
        stage1_fields = _stage_dict(stage1)
        stage1_rows = [
            exact_row(entries[kt, 2][name], value,
                      ladder._field_mask(name, masks),  # noqa: SLF001
                      name=f"production_stage1.kt{kt}.{name}", ladder=ladder)
            for name, value in stage1_fields.items()
        ]
        production_stage1_vs_record.append({"kt": kt, "rows": stage1_rows})

        state = model.step(state, card.dt_s, surface_forcing=forcing)
        jax.block_until_ready(state.eta.data)
        candidate = ladder._restart_candidate(state)  # noqa: SLF001
        restart_rows = compare_named_fields(
            restarts[kt], candidate, _restart_specs(restarts[kt]), masks,
            prefix=f"production_state_vs_restart.kt{kt}", ladder=ladder)
        production_vs_restart.append({"kt": kt, "rows": restart_rows})

    # (iii) The ladder's admitted-entry chain, independent of public state.
    chained_vs_restart = []
    current = _stage_item(entries[1, 1])
    for kt in range(1, 11):
        _, _, current = advance_stage_chain(model, current, card.dt_s)
        jax.block_until_ready(current.ssh_k)
        rows = compare_named_fields(
            restarts[kt], _stage_dict(current), OUTPUT_RESTART_FIELDS, masks,
            prefix=f"chained_stage3_vs_restart.kt{kt}", ladder=ladder)
        chained_vs_restart.append({"kt": kt, "rows": rows})

    control = None
    if plant:
        baseline = _stage_item(entries[2, 1])
        planted = {name: value.copy() for name, value in entries[2, 1].items()}
        candidates = np.argwhere(masks["U"] & (planted["u_k"] != 0.0))
        require(candidates.size > 0, "one-ULP control has no nonzero wet U")
        at = tuple(candidates[np.argmin(np.abs(
            planted["u_k"][tuple(candidates.T)]))])
        planted["u_k"][at] = np.nextafter(
            planted["u_k"][at], np.float64(np.inf))
        clean = model.replay_swe_rk3_stage(baseline, card.dt_s, 1, baseline)
        moved_entry = _stage_item(planted)
        moved = model.replay_swe_rk3_stage(
            moved_entry, card.dt_s, 1, moved_entry)
        control = exact_row(
            np.asarray(clean.ssh_k), np.asarray(moved.ssh_k), masks["T"],
            name="control.one_ulp_u_changes_stage1_ssh", ladder=ladder)
        require(not control["bit_identical"],
                "one-ULP control did not change stage-1 SSH")
        control.update({
            "input_field": "u_k",
            "input_cell": [int(index) for index in at],
        })

    comparisons = {
        "record_next_stage1_vs_restart": record_vs_restart,
        "production_boundary_vs_record_stage1": production_boundary_vs_record,
        "production_state_vs_restart": production_vs_restart,
        "chained_stage3_vs_restart": chained_vs_restart,
        "production_stage1_vs_record_stage2": production_stage1_vs_record,
    }
    policy = get_policy()
    return {
        "label": "SWG-DG-ROUND7-PER-STEP-RESTART-DISCRIMINATOR",
        "legoesm_git_sha": git_sha(),
        "root": str(root.resolve()),
        "deck": "maddison",
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": {
            "compute": str(policy.compute),
            "transcendentals": policy.transcendentals,
        },
        "namelist": namelist._asdict(),
        "stage_record_hashes": ladder._pinned_stage_hashes(root),  # noqa: SLF001
        "restart_hashes": {
            str(kt): ladder._sha256(  # noqa: SLF001
                root / f"SWG_{kt:08d}_restart.nc")
            for kt in range(1, 11)
        },
        "coverage": {
            "record_next_stage1_vs_restart": (
                "N=1..9; u_before/v_before map to restart un/vn after the "
                "next step's steady forcing update; uu_n/vv_n are not compared; "
                "kt=11 not acquired"),
            "production_state_vs_restart": "N=1..10; all 10 restart fields",
            "chained_stage3_vs_restart": "N=1..10; seven common fields",
            "production_stage1_vs_record_stage2": "kt=1..10; all 13 fields",
        },
        "comparisons": comparisons,
        "signed_zero_scope": signed_zero_scope(comparisons),
        "first_debt": {
            name: _first_debt(groups) for name, groups in comparisons.items()
        },
        "all_exact": {
            name: all(_rows_exact(group["rows"]) for group in groups)
            for name, groups in comparisons.items()
        },
        "control": control,
        "oracle_capture": {
            "record": "stprk3.f90:220-235 records Nnn after stage-1 LBC/swap",
            "between_record_and_use": (
                "stprk3.f90:238-255 zeros Nrhs only; Nnn is first consumed "
                "unchanged by dyn_adv/dyn_vor/pressure/ssh_nxt"),
            "restart": (
                "stprk3.f90:360-376 swaps Naa to Nbb then calls "
                "rst_write(kstp,Nbb,Nbb)"),
            "limitation": (
                "restart uu_n/vv_n are written from uu_b/vv_b and are not the "
                "stage-1 u_before/v_before capture; the exact 63-row mapping "
                "therefore does not prove identical capture instants"),
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=encode) + "\n")


def main(argv=None) -> int:
    bootstrap_checkout(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--no-plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run(args.root, plant=not args.no_plant)
        _write_json(args.json, payload)
    except Exception as exc:  # noqa: BLE001 - fail-closed measurement CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(payload["label"])
    for name, exact in payload["all_exact"].items():
        debt = payload["first_debt"][name]
        suffix = "NONE" if debt is None else (
            f"kt{debt['kt']}.{debt['name']} bits={debt['bit_unequal']} "
            f"max={debt['max_abs']:.17e}")
        print(f"{name}: all_exact={exact} first_debt={suffix}")
    print("control_binding=" + str(
        payload["control"] is not None
        and not payload["control"]["bit_identical"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
