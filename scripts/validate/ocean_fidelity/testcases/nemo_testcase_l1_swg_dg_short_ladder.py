#!/usr/bin/env python3
"""Score 1--10-step legoESM states against single-rank NEMO restarts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
REFERENCE_BASELINE = (
    REPO_ROOT / "docs/ocean/fidelity/testcases/"
    "swg_dg_short_ladder_reference.json")
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import JAX_PLATFORMS_FOR, bootstrap_checkout  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Fail closed without importing legoESM before backend selection."""
    if not condition:
        raise RuntimeError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_trajectory(restarts, trajectory, masks, *, exact_row) -> list[dict]:
    """Return one exact-comparison row per step and prognostic field."""
    rows = []
    mapping = (("sshn", "ssh_m", "T"), ("un", "u_m_s", "U"),
               ("vn", "v_m_s", "V"))
    for step, candidate in enumerate(trajectory, start=1):
        for nemo_name, lego_name, grid in mapping:
            row = exact_row(
                restarts[step][nemo_name], candidate[lego_name], masks[grid],
                name=f"step{step}.{lego_name}:{nemo_name}")
            row.update({"step": step, "field": lego_name,
                        "nemo_field": nemo_name, "grid": grid})
            rows.append(row)
    return rows


def _configure_backend(backend: str, gpu_device: str | None) -> None:
    if backend == "gpu":
        require(gpu_device is not None and gpu_device.isdigit(),
                "GPU ladder requires --gpu-device with a physical index")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        os.environ["JAX_PLATFORMS"] = JAX_PLATFORMS_FOR["gpu"]
    else:
        require(gpu_device is None, "--gpu-device is valid only with --backend gpu")
        os.environ["JAX_PLATFORMS"] = JAX_PLATFORMS_FOR["cpu"]


def run(root: Path, *, composition: str, backend: str,
        gpu_device: str | None, plant: bool = True,
        checkout_import: dict | None = None) -> dict:
    _configure_backend(backend, gpu_device)
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.provenance import git_sha

    legoesm_sha = git_sha()
    require(jax.default_backend() == backend,
            f"requested {backend}, JAX selected {jax.default_backend()}")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(
        transcendentals="native" if backend == "gpu" else "libm"))
    ladder = _load("swg_dg_short_ladder_base",
                   HERE / "nemo_testcase_l1_swg_ladder.py")
    round7 = _load("swg_dg_short_ladder_round7",
                   HERE / "nemo_testcase_l1_swg_dg_round7.py")
    composition_tool = _load(
        "swg_dg_short_ladder_composition",
        HERE / "nemo_testcase_l1_swg_dg_composition.py")
    driver = _load("swg_dg_short_ladder_driver",
                   REPO_ROOT / "scripts/run/run_swg_double_gyre.py")
    namelist = ladder._deck_namelist("maddison")  # noqa: SLF001
    if backend == "gpu":
        # The certified ladder builder requires CPU libm arithmetic; the GPU
        # form scores the long-run driver's own case under its native policy.
        card, model, forcing = driver.build_case(namelist)
        masks = ladder.card_masks(card)
    else:
        card, masks, model, forcing = ladder._swg_production_case(  # noqa: SLF001
            "short restart ladder", namelist=namelist)
    shape = tuple(card.recipe.land_mask.shape)
    restarts = {
        step: composition_tool._load_single_rank_restart(  # noqa: SLF001
            root, step, shape)[0]
        for step in range(1, 11)
    }
    state = driver.seed_steady_forcing_carry(card.recipe.initial_state, forcing)
    trajectory = []
    if composition == "outer-scan":
        def body(carry, _):
            carry = model.step(carry, card.dt_s, surface_forcing=forcing)
            fields = (carry.eta.data, carry.u.data[:, 1:, 0],
                      carry.v.data[1:, :, 0])
            return carry, fields

        @jax.jit
        def scanned(carry):
            return jax.lax.scan(body, carry, None, length=10)

        _, arrays = scanned(state)
        jax.block_until_ready(arrays[0])
        for index in range(10):
            trajectory.append({
                "ssh_m": np.asarray(arrays[0][index]),
                "u_m_s": np.asarray(arrays[1][index]),
                "v_m_s": np.asarray(arrays[2][index]),
            })
    else:
        for _ in range(10):
            state = model.step(state, card.dt_s, surface_forcing=forcing)
            jax.block_until_ready(state.eta.data)
            trajectory.append({
                "ssh_m": np.asarray(state.eta.data),
                "u_m_s": np.asarray(state.u.data[:, 1:, 0]),
                "v_m_s": np.asarray(state.v.data[1:, :, 0]),
            })

    def exact(reference, candidate, mask, *, name):
        return round7.exact_row(
            reference, candidate, mask, name=name, ladder=ladder)

    rows = score_trajectory(restarts, trajectory, masks, exact_row=exact)
    control = None
    if plant:
        moved = trajectory[0]["u_m_s"].copy()
        wet = np.argwhere(masks["U"] & (moved != 0.0))
        require(wet.size > 0, "one-ULP control has no nonzero wet U")
        at = tuple(wet[0])
        moved[at] = np.nextafter(moved[at], np.float64(np.inf))
        control = exact(restarts[1]["un"], moved, masks["U"],
                        name="control.step1.u_m_s:un")
        require(not control["bit_identical"], "one-ULP control did not bind")
    first = next((row for row in rows if not row["bit_identical"]), None)
    return {
        "schema": "swg-dg-short-restart-ladder-v1",
        "label": "SWG_DG_SHORT_LADDER",
        "checkout_import": checkout_import,
        "legoesm_git_sha": legoesm_sha,
        "nemo_root": str(root.resolve()),
        "backend": backend,
        "gpu_device": gpu_device,
        "composition": composition,
        "precision_policy": (
            "fp64-native" if backend == "gpu" else "fp64-libm"),
        "rows": rows,
        "all_exact": first is None,
        "first_difference": first,
        "control": control,
    }


def _row_signature(row: dict) -> list[int | float]:
    return [
        int(row["bit_unequal"]),
        int(row["numeric_unequal"]),
        int(row["signed_zero_only"]),
        float(row["max_abs"]),
    ]


def _expected_signatures(form: dict) -> dict[tuple[int, str], list]:
    fields = ("ssh_m", "u_m_s", "v_m_s")
    if "uniform_signature" in form:
        return {
            (step, field): form["uniform_signature"]
            for step in range(1, 11) for field in fields
        }
    return {
        (step, field): form["per_step"][str(step)][field]
        for step in range(1, 11) for field in fields
    }


def compare_reference_reports(
        outer_report: Path, materialized_report: Path,
        baseline_path: Path = REFERENCE_BASELINE) -> dict:
    """Compare both colleague reports with the measured reference behavior."""
    baseline = json.loads(baseline_path.read_text())
    require(baseline["schema"] == "swg-dg-short-ladder-reference-v1",
            "short-ladder reference schema differs")
    reports = {
        "outer-scan": json.loads(outer_report.read_text()),
        "materialized": json.loads(materialized_report.read_text()),
    }
    results = {}
    for composition, report in reports.items():
        require(report.get("schema") == "swg-dg-short-restart-ladder-v1",
                f"{composition} report schema differs")
        require(report.get("composition") == composition,
                f"expected {composition} report, got {report.get('composition')}")
        observed = {}
        for row in report.get("rows", []):
            key = (int(row["step"]), row["field"])
            require(key not in observed,
                    f"duplicate {composition} row step{key[0]}.{key[1]}")
            observed[key] = _row_signature(row)
        form = baseline["forms"][composition]
        expected = _expected_signatures(form)
        require(set(observed) == set(expected),
                f"{composition} row coverage differs: "
                f"observed={len(observed)} expected={len(expected)}")
        require(len(observed) == form["row_count"],
                f"{composition} row count differs")
        observed_all_exact = all(signature[0] == 0 for signature in observed.values())
        require(bool(report.get("all_exact")) == observed_all_exact,
                f"{composition} report all_exact contradicts its rows")
        first = next((
            {
                "step": step,
                "field": field,
                "expected": expected[(step, field)],
                "observed": observed[(step, field)],
            }
            for step in range(1, 11)
            for field in ("ssh_m", "u_m_s", "v_m_s")
            if observed[(step, field)] != expected[(step, field)]
        ), None)
        if first is None and observed_all_exact != form["all_exact"]:
            first = {
                "step": 1,
                "field": "report.all_exact",
                "expected": form["all_exact"],
                "observed": observed_all_exact,
            }
        results[composition] = {
            "same_as_reference": first is None,
            "first_departure": first,
        }
    return {
        "schema": "swg-dg-short-ladder-reference-comparison-v1",
        "baseline": str(baseline_path.resolve()),
        "forms": results,
        "all_same_as_reference": all(
            result["same_as_reference"] for result in results.values()),
    }


def main(argv=None) -> int:
    argument_vector = sys.argv[1:] if argv is None else list(argv)
    if "-h" in argument_vector or "--help" in argument_vector:
        # The decoy-checkout guard exercises help on machines without GPUs.
        bootstrap_checkout(__file__, jax_platform="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path)
    parser.add_argument("--composition", choices=("outer-scan", "materialized"),
                        default="outer-scan")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-device")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--no-plant", action="store_true")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--outer-report", type=Path)
    parser.add_argument("--materialized-report", type=Path)
    args = parser.parse_args(argument_vector)
    try:
        if args.compare_reference:
            bootstrap_checkout(__file__, jax_platform="cpu")
            require(args.outer_report is not None,
                    "--compare-reference requires --outer-report")
            require(args.materialized_report is not None,
                    "--compare-reference requires --materialized-report")
            comparison = compare_reference_reports(
                args.outer_report, args.materialized_report)
            for composition, result in comparison["forms"].items():
                if result["same_as_reference"]:
                    print(
                        "SWG_DG_SHORT_LADDER_BASELINE "
                        f"{composition} same as reference")
                else:
                    first = result["first_departure"]
                    print(
                        "SWG_DG_SHORT_LADDER_BASELINE "
                        f"{composition} departs from reference at "
                        f"step={first['step']} field={first['field']} "
                        f"expected={first['expected']} "
                        f"observed={first['observed']}")
            return 0 if comparison["all_same_as_reference"] else 1
        require(args.nemo_root is not None,
                "scoring requires --nemo-root")
        require(args.json is not None, "scoring requires --json")
        checkout_import = bootstrap_checkout(
            __file__, jax_platform=args.backend,
            gpu_device=args.gpu_device)
        require(not args.json.exists(), f"refusing to overwrite {args.json}")
        payload = run(
            args.nemo_root, composition=args.composition,
            backend=args.backend, gpu_device=args.gpu_device,
            plant=not args.no_plant, checkout_import=checkout_import)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    for row in payload["rows"]:
        print(
            f"step={row['step']} field={row['field']} "
            f"bit_unequal={row['bit_unequal']} max_abs={row['max_abs']:.17e}")
    first = payload["first_difference"]
    print(
        f"SWG_DG_SHORT_LADDER {'EXACT' if first is None else 'DEBT'} "
        f"composition={payload['composition']} backend={payload['backend']} "
        "first_difference=" + (
            "NONE" if first is None else
            f"step{first['step']}.{first['field']} bits={first['bit_unequal']}"))
    return 0 if payload["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
