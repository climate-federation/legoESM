#!/usr/bin/env python3
"""Compare ten-step outer-scan and materialized SWG_DG compositions to NEMO."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Fail closed without importing legoESM at module import.

    The GPU short ladder loads this module after selecting the GPU; only
    :func:`main` binds the checkout and its CPU platform.
    """
    if not condition:
        raise RuntimeError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field_totals(rows: list[dict]) -> dict:
    prognostic = [row for row in rows if row["field"] in {"sshn", "un", "vn"}]
    return {
        "rows": len(rows),
        "bit_unequal": sum(row["bit_unequal"] for row in rows),
        "numeric_unequal": sum(row["numeric_unequal"] for row in rows),
        "signed_zero_only": sum(row["signed_zero_only"] for row in rows),
        "prognostic_bit_unequal": sum(row["bit_unequal"] for row in prognostic),
        "prognostic_bit_identical": all(row["bit_identical"] for row in prognostic),
        "all_bit_identical": all(row["bit_identical"] for row in rows),
    }


def _load_single_rank_restart(root: Path, step: int, shape: tuple[int, int]):
    """Read one unpartitioned shallow-water restart and retain its digest."""
    import netCDF4

    path = root / f"SWG_{step:08d}_restart.nc"
    require(path.is_file(), f"missing single-rank restart {path}")
    require(not list(root.glob(f"SWG_{step:08d}_restart_*.nc")),
            "composition discriminator requires the one-rank restart")
    names = ("sshn", "un", "vn", "uu_n", "vv_n", "utau_b", "vtau_b",
             "emp_b", "qns_b", "sfx_b")
    restart = {}
    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        require(int(np.asarray(dataset["kt"][:]).item()) == step,
                "restart kt differs")
        require(float(np.asarray(dataset["rdt"][:]).item()) == 50.0,
                "restart rdt differs")
        for name in names:
            require(name in dataset.variables, f"restart is missing {name}")
            value = np.asarray(dataset[name][0])
            require(value.dtype == np.dtype("float64"),
                    f"restart {name} is not float64")
            if name in {"un", "vn"}:
                require(value.shape[0] == 2 and np.all(value[1] == 0.0),
                        f"restart {name} dummy level differs")
                value = value[0]
            require(value.shape == shape and np.all(np.isfinite(value)),
                    f"restart {name} shape/non-finite values differ")
            restart[name] = value
    return restart, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run(root: Path, *, steps: int = 10, plant: bool = True) -> dict:
    """Run both production compositions from one state and compare restart N."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.provenance import git_sha

    require(steps == 10, "the admitted composition discriminator is step 10")
    require(jax.default_backend() == "cpu", "composition discriminator requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    ladder = _load("swg_dg_composition_ladder",
                   HERE / "nemo_testcase_l1_swg_ladder.py")
    round7 = _load("swg_dg_composition_round7",
                   HERE / "nemo_testcase_l1_swg_dg_round7.py")
    driver = _load("swg_dg_composition_driver",
                   REPO_ROOT / "scripts/run/run_swg_double_gyre.py")
    namelist = ladder._deck_namelist("maddison")  # noqa: SLF001
    card, masks, model, forcing = ladder._swg_production_case(  # noqa: SLF001
        "outer-scan composition discriminator", namelist=namelist)
    restart, restart_provenance = _load_single_rank_restart(
        root, steps, tuple(card.recipe.land_mask.shape))
    initial = driver.seed_steady_forcing_carry(
        card.recipe.initial_state, forcing)

    scanned_step = driver.build_scanned_stepper(
        model, card.dt_s, forcing, steps)
    scanned = scanned_step(initial)
    jax.block_until_ready(scanned.eta.data)
    materialized = driver.run_python_steps(
        model, initial, card.dt_s, forcing, steps)
    jax.block_until_ready(materialized.eta.data)

    specifications = round7._restart_specs(restart)  # noqa: SLF001
    compositions = {}
    for name, state in (("outer_jit_lax_scan", scanned),
                        ("materialized_python_loop", materialized)):
        candidate = ladder._restart_candidate(state)  # noqa: SLF001
        rows = round7.compare_named_fields(
            restart, candidate, specifications, masks,
            prefix=f"{name}.kt{steps}", ladder=ladder)
        for row, (_, reference_name, _) in zip(rows, specifications, strict=True):
            row["field"] = reference_name
        compositions[name] = {"rows": rows, "summary": _field_totals(rows)}

    control = None
    if plant:
        candidate = ladder._restart_candidate(scanned)  # noqa: SLF001
        moved = dict(candidate)
        moved["un"] = np.asarray(candidate["un"]).copy()
        wet = np.argwhere(masks["U"] & (moved["un"] != 0.0))
        require(wet.size > 0, "one-ULP control has no nonzero wet U")
        at = tuple(wet[0])
        moved["un"][at] = np.nextafter(moved["un"][at], np.float64(np.inf))
        control = round7.exact_row(
            restart["un"], moved["un"], masks["U"],
            name="control.one_ulp_outer_scan_un", ladder=ladder)
        require(not control["bit_identical"],
                "one-ULP control did not trip exact comparison")
        control["input_cell"] = [int(index) for index in at]

    scan_summary = compositions["outer_jit_lax_scan"]["summary"]
    materialized_summary = compositions["materialized_python_loop"]["summary"]
    return {
        "schema": "swg-dg-composition-discriminator-v1",
        "label": "SWG-DG-TEN-STEP-COMPOSITION-DISCRIMINATOR",
        "legoesm_git_sha": git_sha(),
        "root": str(root.resolve()),
        "restart": restart_provenance,
        "step": steps,
        "backend": jax.default_backend(),
        "precision_policy": "fp64-libm",
        "programs": {
            "outer_jit_lax_scan": "one jax.jit enclosing lax.scan of ten model.step calls",
            "materialized_python_loop": "ten separately materialized model.step calls",
        },
        "compositions": compositions,
        "classification": {
            "outer_scan_prognostic_exact": scan_summary["prognostic_bit_identical"],
            "outer_scan_all_restart_words_exact": scan_summary["all_bit_identical"],
            "materialized_prognostic_exact": materialized_summary[
                "prognostic_bit_identical"],
            "signed_zero_policy": round7.SIGNED_ZERO_POLICY,
        },
        "control": control,
    }


def _write_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    bootstrap_checkout(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing admitted step-10 restart")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--no-plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run(args.root, plant=not args.no_plant)
        _write_json(args.json, payload)
    except Exception as exc:  # noqa: BLE001 - fail-closed measurement boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(payload["label"])
    for name, value in payload["compositions"].items():
        summary = value["summary"]
        print(
            f"{name}: prognostic_bit_unequal="
            f"{summary['prognostic_bit_unequal']} "
            f"all_bit_unequal={summary['bit_unequal']} "
            f"signed_zero_only={summary['signed_zero_only']}")
    print(f"control_binding={not payload['control']['bit_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
