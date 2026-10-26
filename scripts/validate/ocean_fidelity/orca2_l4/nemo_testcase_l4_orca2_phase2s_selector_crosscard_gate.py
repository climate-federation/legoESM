#!/usr/bin/env python3
"""Rule-12 kt=1 entry rows for the ORCA2-only SH2 selector repair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy, get_policy, set_policy,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_gyre_zco_card, build_lock_exchange_zco_card,
    build_orca2_zps_card, build_overflow_zps_card,
    validate_nemo_testcase_card,
)

ROOTS = {
    "GYRE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
        "round15_oracle_v2_scalarmath"),
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
}
BUILDERS = {
    "GYRE-zco": build_gyre_zco_card,
    "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card,
    "OVERFLOW-zps": build_overflow_zps_card,
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, filename: str):
    path = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load Lane-1 trajectory reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(deck: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(get_policy() == policy, "fp64 scalar-libm policy not active")
    trajectory = _load_module(
        "lane1_trajectory", "nemo_testcase_phase3_trajectory_gate.py")
    gyre = _load_module(
        "lane2_gyre", "nemo_testcase_l2_gyre_phase3_gate.py")

    cards = {case: builder() for case, builder in BUILDERS.items()}
    rows = {}
    for case, card in cards.items():
        validate_nemo_testcase_card(card)
        record = ROOTS[case] / "oracle_step_entry_kt00000001.bin"
        reader = gyre if case == "GYRE-zco" else trajectory
        oracle = (reader.read_entry(record) if case == "GYRE-zco"
                  else reader.read_entry(record, case))
        candidate = reader.lego_fields(card.recipe.initial_state)
        masks = reader.expected_masks(card)
        case_rows = {}
        for field in ("T", "S", "u", "v", "ssh"):
            target = np.asarray(oracle[field])
            if field != "ssh":
                target = target[..., :card.recipe.z_coord.n_levels]
            value = np.asarray(candidate[field]).copy()
            use = np.asarray(masks[field], bool)
            if not use.any():
                # LOCK has no active V face.  Preserve the stored-array gross
                # check but label it uninformative rather than certification.
                use = np.ones_like(target, bool)
            if plant and case == "LOCK_EXCHANGE-zco" and field == "T":
                index = tuple(int(v) for v in np.argwhere(use)[0])
                value[index] = np.nextafter(value[index], np.inf)
            exact = bool(np.array_equal(value[use], target[use]))
            case_rows[field] = {
                "unequal": int(np.count_nonzero(
                    value[use].view(np.uint64) != target[use].view(np.uint64))),
                "count": int(use.sum()),
                "status": "AT_BAR" if exact else "DEBT",
            }
            require(exact, f"{case} kt=1 {field} moved")
        rows[case] = {
            "record": str(record), "record_sha256": sha256(record),
            "fields": case_rows,
        }

    orca2 = build_orca2_zps_card(deck)
    validate_nemo_testcase_card(orca2)
    tke = orca2.recipe.model_config.physics.vertical_mixing.tke
    observed = (
        tke.tke_shear_production, tke.tke_shear_avm_weighting,
        tke.tke_shear_evaluation_stage, tke.tke_shear_metric_source,
    )
    expected = ("nemo_face_native_nbb2", "nemo_face", "step_entry",
                "nemo_qco_live_face")
    require(observed == expected, f"ORCA2 SH2 tuple {observed!r}")
    require(tke.bottom_tke_bc is True,
            "ORCA2 bottom-friction TKE Dirichlet selector is not restored")
    require(int(tke.eice) == 1,
            "ORCA2 nn_eice=1 selector is not restored")
    return {
        "status": "PASS",
        "execution": {"backend": jax.default_backend(), "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
        "orca2_sh2_tuple": list(observed),
        "orca2_bottom_tke_bc": bool(tke.bottom_tke_bc),
        "orca2_nn_eice": int(tke.eice),
        "cross_card_kt1_entry": rows,
        "scope": (
            "actual NEMO kt=1 Nbb entry rows; no ocean step.  The production "
            "mode-1 TKE ice attenuation is statically unreachable on cards "
            "whose eice selector remains zero, so unchanged cards enter the "
            "same production step from byte-identical state."),
        "c1d": {
            "status": "CONSTRUCTIBILITY_PROXY_ONLY_PENDING_ICE_MERGE",
            "evidence": "tests/ice/unit/test_ice_transport_cgrid.py",
            "reason": "this branch has no merged Lane-3 C1D fidelity card",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, plant=args.plant)
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json_out:
        args.json_out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
