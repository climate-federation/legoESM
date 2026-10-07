#!/usr/bin/env python3
"""Locate round 164's first invalid live ``e3w`` without changing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)

PLANTS = ("none", "terminal", "entry", "source-product")
EXPECTED_ERROR = "raw-mesh e3w_int must contain only finite values > 0"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _install_named_e3w_trace(trace: list[dict[str, object]]) -> None:
    """Reuse round 81's scalar callback, adding the bound consumer name."""
    import jax
    import jax.numpy as jnp

    from legoesm.ocean import eos

    original = eos.compute_buoyancy_frequency_nemo_bn2

    def wrapper(label: str):
        def traced(*args, **kwargs):
            e3w = kwargs.get("e3w_int")
            if e3w is not None:
                values = jnp.asarray(e3w)
                valid = jnp.isfinite(values) & (values > 0.0)
                score = jnp.where(jnp.isfinite(values), values, -jnp.inf)

                def capture(bad, minimum, flat_index, invalid_count):
                    if bool(bad) and not trace:
                        trace.append({
                            "consumer": label,
                            "value": float(minimum),
                            "flat_index": int(flat_index),
                            "invalid_count": int(invalid_count),
                            "shape": list(values.shape),
                        })

                jax.debug.callback(
                    capture,
                    ~jnp.all(valid),
                    jnp.min(score),
                    jnp.argmin(score),
                    jnp.count_nonzero(~valid),
                    ordered=True,
                )
            return original(*args, **kwargs)

        return traced

    # Some consumers bind the function at import time.  Replace only exact
    # references to the original, as round 81 does, and label each binding.
    for name, module in tuple(sys.modules.items()):
        if module is not None and getattr(
                module, "compute_buoyancy_frequency_nemo_bn2", None) is original:
            setattr(module, "compute_buoyancy_frequency_nemo_bn2", wrapper(name))


def _control_terminal(path: Path) -> dict[str, object]:
    text = path.read_text()
    return {
        "kt7_stage3_completed": "PROGRESS kt=7 completed stage 3" in text,
        "kt8_stages12_exposed": "PROGRESS kt=8 exposed stages 1-2" in text,
        "kt8_stage3_completed": "PROGRESS kt=8 completed stage 3" in text,
        "error": EXPECTED_ERROR in text,
    }


def _boundary_row(
    eta: np.ndarray,
    H: np.ndarray,
    raw_e3w: np.ndarray,
    index: tuple[int, int, int],
) -> dict[str, object]:
    j, i, k = index
    # eos.nemo_r3t_stretch(..., evaluation="nemo_reciprocal"), including its
    # finite positive floor.  The source product is evaluated array-wide so
    # invalid_count describes the same field the callback saw.
    reciprocal = np.divide(
        np.float64(1.0), H,
        out=np.ones_like(H, dtype=np.float64), where=H > 0.0,
    )
    r3t = np.where(H > 0.0, eta * reciprocal, 0.0)
    stretch = np.maximum(np.float64(1.0) + r3t, np.float64(1.0e-6))
    product = raw_e3w * stretch[..., None]
    valid = np.isfinite(product) & (product > 0.0)
    return {
        "eta_m": float(eta[j, i]),
        "bathymetry_m": float(H[j, i]),
        "raw_e3w0_m": float(raw_e3w[j, i, k]),
        "r3t": float(r3t[j, i]),
        "stretch": float(stretch[j, i]),
        "source_product_m": float(product[j, i, k]),
        "invalid_count": int(np.count_nonzero(~valid)),
        "all_finite_positive": bool(np.all(valid)),
    }


def measure(deck_root: Path, record_root: Path, control_log: Path) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "measurement requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    # Instantiate once before installing the callback so every already-bound
    # bn2 consumer is loaded and can be named.  No step is run here.
    LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    trace: list[dict[str, object]] = []
    _install_named_e3w_trace(trace)

    boundaries: dict[str, np.ndarray] = {}

    def observe(kt, entry_state, stage_states):
        if kt == 8:
            boundaries["entry"] = np.asarray(entry_state.eta.data).copy()
            boundaries["stage1"] = np.asarray(stage_states[0].eta.data).copy()
            boundaries["stage2"] = np.asarray(stage_states[1].eta.data).copy()

    error = None
    try:
        ladder.run(
            deck_root,
            record_root,
            external_mode_association=True,
            raw_reference_depth=True,
            unmasked_v_transport=True,
            materialize_v_transport=True,
            stage_observer=observe,
        )
    except Exception as caught:  # JAX wraps eqx.error_if in a runtime exception.
        error = str(caught)

    require(error is not None and EXPECTED_ERROR in error,
            f"observed run did not reproduce the expected refusal: {error!r}")
    require(trace, "invalid e3w callback did not fire")
    require(set(boundaries) == {"entry", "stage1", "stage2"},
            f"kt=8 boundary observer is incomplete: {tuple(boundaries)}")

    observed = trace[0]
    shape = tuple(int(value) for value in observed["shape"])
    index = tuple(int(value) for value in np.unravel_index(
        int(observed["flat_index"]), shape))
    raw = np.asarray(card.recipe.z_coord.nemo_e3w_0, dtype=np.float64)[..., 1:]
    H = np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)
    rows = {
        name: _boundary_row(eta, H, raw, index)
        for name, eta in boundaries.items()
    }
    callback_value = float(observed["value"])
    matching = [
        name for name, row in rows.items()
        if (np.isnan(callback_value) and np.isnan(row["source_product_m"]))
        or row["source_product_m"] == callback_value
    ]
    return {
        "format": "nemo-testcase-l4-orca2-round165-vertical-boundary-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "private_arm": {
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "unobserved_control": _control_terminal(control_log),
        "observed_terminal": {
            "kt8_stages12_exposed": True,
            "kt8_stage3_completed": False,
            "error": EXPECTED_ERROR,
        },
        "first_invalid": {
            **observed,
            "index_jik": list(index),
            "matching_boundaries": matching,
        },
        "boundaries": rows,
        "worktree": worktree_stamp(),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "terminal":
        report["observed_terminal"]["kt8_stage3_completed"] = True
    elif plant == "entry":
        report["boundaries"]["entry"]["all_finite_positive"] = False
    elif plant == "source-product":
        report["first_invalid"]["matching_boundaries"] = []

    control = report["unobserved_control"]
    observed = report["observed_terminal"]
    require(control == {
        "kt7_stage3_completed": True,
        "kt8_stages12_exposed": True,
        "kt8_stage3_completed": False,
        "error": True,
    }, f"unobserved control terminal changed: {control}")
    require(observed == {
        "kt8_stages12_exposed": True,
        "kt8_stage3_completed": False,
        "error": EXPECTED_ERROR,
    }, f"passive observer changed the terminal: {observed}")
    require(report["boundaries"]["entry"]["all_finite_positive"],
            "kt=8 entry geometry is already invalid")
    require(report["first_invalid"]["matching_boundaries"],
            "no observed source product reproduces the callback value")
    report["status"] = "PASS_ROUND165_VERTICAL_BOUNDARY"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--control-log", type=Path)
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in is not None:
            report = json.loads(args.report_in.read_text())
        else:
            require(
                args.deck_root is not None
                and args.record_root is not None
                and args.control_log is not None,
                "measurement requires --deck-root, --record-root and --control-log",
            )
            report = measure(args.deck_root, args.record_root, args.control_log)
        report = classify(report, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, ValueError) as error:
        label = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {label} {args.plant}: {error}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND165_VERTICAL_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
