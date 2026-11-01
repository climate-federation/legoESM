#!/usr/bin/env python3
"""Read-only O17 acquisition composition checks.

The component schemas and gates are imported from their admitted tools.  This
file owns only the new full-step call-order/read-set union; it intentionally
does not duplicate a record reader, extent table, or physical formula.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path

import numpy as np

COMPONENTS = (
    "scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate",
    "scripts.validate.ocean_fidelity.testcases.nemo_si3_o5_orca2_sas_gate",
    "scripts.validate.ocean_fidelity.testcases.nemo_si3_o7_orca2_sas_gate",
    "scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate",
    "scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_readset",
    "scripts.validate.ocean_fidelity.orca2_l4.nemo_acquisition_extents",
)

ORDER = (
    "si3_blk_ice_1",
    "si3_ice_sbc_tau_lbc",
    "si3_orca2_ice_strength",
    "si3_orca2_aevp_solver",
    "si3_cgrid_deformation",
    "si3_dyn_orca2",
    "si3_ice_dyn_rdgrft",
    "si3_ice_cor_dyn",
    "si3_var_glo2eqv",
    "si3_var_agg",
    "si3_ice_alb",
    "si3_blk_ice_2",
    "si3_ice_flx_other",
    "si3_ice_thd_frazil",
    "si3_ice_thd_o12_stages",
    "si3_ice_itd_rem",
    "si3_ice_thd_do",
    "si3_o15_step",
    "si3_o16_step",
    "si3_ice_update_tau",
)


def _calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "si3_o17_step"
    )
    out = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.append((node.lineno, fn.id))
    calls = [name for _, name in sorted(out)]
    # The production function contains an explicit swap plant in one ``if``
    # arm followed by the real call in its ``else`` arm.  AST walking sees
    # both; the later call is the unplanted baseline selected at runtime.
    adv = [i for i, name in enumerate(calls) if name == "si3_dyn_orca2"]
    if len(adv) == 2:
        calls.pop(adv[0])
    return calls


def preflight(repo: Path, plant: str | None = None) -> dict:
    imported = {name: importlib.import_module(name).__file__ for name in COMPONENTS}
    calls = _calls(repo / "packages/ice/legoesm/ice/si3_o17.py")
    if plant == "swap-rheology-advection":
        left = calls.index("si3_orca2_aevp_solver")
        right = calls.index("si3_dyn_orca2")
        calls[left], calls[right] = calls[right], calls[left]
    elif plant == "drop-redistribution":
        calls.remove("si3_ice_dyn_rdgrft")
    positions = []
    start = 0
    for name in ORDER:
        try:
            index = calls.index(name, start)
        except ValueError as exc:
            raise SystemExit(f"STATUS PLANT-FIRED: missing/out-of-order {name}") from exc
        positions.append(index)
        start = index + 1
    return {
        "status": "PASS_O17_READ_ONLY_PREFLIGHT",
        "imported_component_tools": imported,
        "call_order": list(ORDER),
        "steps": 100,
        "dt_seconds": 10800,
        "required_records_each_step": [
            "complete_step_entry",
            "raw_o7_meteorology",
            "o7_products",
            "named_call_boundaries",
            "exchange_outputs",
            "restart",
        ],
        "labels": ["INDEPENDENT", "GIVEN-NEMO-ENTRY"],
    }


def numerical_preflight(master: Path, plant: str | None = None) -> dict:
    """Execute live O17 boundaries and require each planted change to be numeric."""

    import jax
    from legoesm.ice import si3_o17 as o17
    from legoesm.ice.si3_o17 import SI3O17LiveForcing
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import _case

    entry, forcing, card = _case(master)
    card = card._replace(aevp=card.aevp._replace(n_subcycles=1))

    class Boundary(RuntimeError):
        def __init__(self, values):
            self.values = values

    original = o17.si3_ice_dyn_rdgrft

    def capture(state, **kwargs):
        raise Boundary((state.a_i, kwargs["divu_i"], kwargs["delta_i"]))

    def run(**plants):
        o17.si3_ice_dyn_rdgrft = capture
        try:
            with jax.disable_jit():
                o17.si3_o17_step(entry, SI3O17LiveForcing(1, *forcing[1:]), card, **plants)
        except Boundary as stop:
            return tuple(np.asarray(value) for value in stop.values)
        finally:
            o17.si3_ice_dyn_rdgrft = original
        raise RuntimeError("O17 numerical boundary was not reached")

    baseline = run()
    if plant == "swap-rheology-advection":
        candidate = run(_plant_swap_rheology_advection=True)
        fired = not np.array_equal(candidate[0], baseline[0])
    elif plant == "drop-deformation-fold":
        candidate = run(_plant_drop_deformation_fold=True)
        fired = any(not np.array_equal(a, b) for a, b in zip(candidate[1:], baseline[1:], strict=True))
    elif plant is None:
        fired = True
    else:
        raise ValueError(f"unknown numerical plant {plant}")
    if not fired:
        raise SystemExit(f"STATUS PLANT-SURVIVED: {plant}")
    return {
        "status": "PASS_O17_NUMERICAL_PREFLIGHT",
        "plant": plant,
        "active_ice_values": int(np.count_nonzero(baseline[0])),
        "deformation_values": int(np.count_nonzero(baseline[1])) + int(np.count_nonzero(baseline[2])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plant", choices=("swap-rheology-advection", "drop-redistribution"))
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--master", type=Path)
    parser.add_argument("--numerical-plant", choices=("swap-rheology-advection", "drop-deformation-fold"))
    args = parser.parse_args()
    result = (
        numerical_preflight(args.master.resolve(), args.numerical_plant)
        if args.numerical
        else preflight(args.repo.resolve(), args.plant)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
