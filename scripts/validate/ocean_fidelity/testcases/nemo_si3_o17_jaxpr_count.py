#!/usr/bin/env python3
"""Count recursive O17 JAXPR equations by their source name-stack owner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core

from legoesm.ice import si3_o17 as o17_module
from legoesm.ice.si3_o17 import SI3O17LiveForcing, si3_o17_step
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import DEFAULT_MASTER, _case

REVIEWED_EA3E570 = {
    "recursive_total": 576_364,
    "_thermo_category_loop": 342_178,
    "si3_ice_thd_do": 53_490,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 132_371,
}
# Parent 9998276020, before DECISION 194 traced the O9--O12 loops.
PRE_D194_9998276020 = {
    "recursive_total": 577_982,
    "_thermo_category_loop": 342_170,
    "si3_ice_thd_do": 53_489,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 133_998,
}
# DECISION 194 (48526f950c): category scan, layer fori_loops, shared vremap fori_loop.
D194_48526F950C = {
    "recursive_total": 227_840,
    "_thermo_category_loop": 35_858,
    "si3_ice_thd_do": 9_659,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 133_998,
}
# DECISION 194 resume 1 (b17f14b73c): vremap thickness and SSL transmission barriers.
D194_RESUME1_B17F14B73C = {
    "recursive_total": 228_050,
    "_thermo_category_loop": 35_918,
    "si3_ice_thd_do": 9_809,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 133_998,
}
# DECISION 198 round 1 (743146b590): O6 redistribution donor loop as a fori_loop.
D198_R1_743146B590 = {
    "recursive_total": 198_323,
    "_thermo_category_loop": 35_918,
    "si3_ice_thd_do": 9_809,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 104_271,
}
# DECISION 198 round 2 (56b768300a): O4 Prather subcycles as one checkpointed scan.
D198_R2_56B768300A = {
    "recursive_total": 178_146,
    "_thermo_category_loop": 35_918,
    "si3_ice_thd_do": 9_809,
    "si3_dyn_orca2": 28_148,
    "remaining_named_and_inlined": 104_271,
}
# DECISION 198 round 3: O6 ice_cor rebinning, salinity and zapsmall loops as fori_loops.
EXPECTED = {
    "recursive_total": 162_751,
    "_thermo_category_loop": 35_918,
    "si3_ice_thd_do": 9_809,
    "si3_dyn_orca2": 28_148,
    "remaining_named_and_inlined": 88_876,
}
OWNERS = ("_thermo_category_loop", "si3_ice_thd_do", "si3_dyn_orca2")


def _nested_jaxprs(value: Any) -> Iterable[jax_core.Jaxpr]:
    if isinstance(value, jax_core.ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, jax_core.Jaxpr):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _nested_jaxprs(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _nested_jaxprs(child)


def _owner(eqn: jax_core.JaxprEqn, inherited: str) -> str:
    scalar_params = {key: value for key, value in eqn.params.items()
                     if not tuple(_nested_jaxprs(value))}
    stack = f"{eqn.source_info.name_stack} {eqn.primitive.name} {scalar_params}"
    for name in OWNERS:
        if name in stack:
            return name
    return inherited


def _walk(
    jaxpr: jax_core.Jaxpr,
    counts: Counter[str],
    inherited: str = "remaining_named_and_inlined",
) -> None:
    for eqn in jaxpr.eqns:
        owner = _owner(eqn, inherited)
        counts[owner] += 1
        counts["recursive_total"] += 1
        for child in _nested_jaxprs(eqn.params):
            _walk(child, counts, owner)


def count(master: Path) -> dict[str, Any]:
    entry, forcing, card = _case(master)
    inputs = entry, forcing[1:]

    # Add trace-only names around the three boundaries Argon measured. This
    # does not change the production source or arithmetic, and makes nested
    # ownership explicit in source_info.name_stack.
    originals = {
        "_thermo_category_loop": o17_module.si3_ice_thd_o12_stages,
        "si3_ice_thd_do": o17_module.si3_ice_thd_do,
        "si3_dyn_orca2": o17_module.si3_dyn_orca2,
    }
    o17_module.si3_ice_thd_o12_stages = jax.named_call(
        originals["_thermo_category_loop"], name="_thermo_category_loop"
    )
    o17_module.si3_ice_thd_do = jax.named_call(
        originals["si3_ice_thd_do"], name="si3_ice_thd_do"
    )
    o17_module.si3_dyn_orca2 = jax.named_call(
        originals["si3_dyn_orca2"], name="si3_dyn_orca2"
    )

    def objective(x):
        step_entry, dynamic = x
        out, trace = si3_o17_step(step_entry, SI3O17LiveForcing(1, *dynamic), card)
        publication = (out, trace.finalize.final.fluxes, trace.finalize.final.outputs, trace.finalize.stress)
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(publication)
                  if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)]
        return sum(jnp.sum(leaf) for leaf in leaves)

    try:
        closed = jax.make_jaxpr(objective)(inputs)
    finally:
        o17_module.si3_ice_thd_o12_stages = originals["_thermo_category_loop"]
        o17_module.si3_ice_thd_do = originals["si3_ice_thd_do"]
        o17_module.si3_dyn_orca2 = originals["si3_dyn_orca2"]
    counts: Counter[str] = Counter()
    _walk(closed.jaxpr, counts)
    measured = {name: int(counts[name]) for name in EXPECTED}
    return {
        "status": "PASS" if measured == EXPECTED else "COUNT_CHANGED",
        "measured": measured,
        "expected": EXPECTED,
        "reviewed_ea3e570_snapshot": REVIEWED_EA3E570,
        "pre_d194_9998276020_snapshot": PRE_D194_9998276020,
        "d194_48526f950c_snapshot": D194_48526F950C,
        "d194_resume1_b17f14b73c_snapshot": D194_RESUME1_B17F14B73C,
        "d198_r1_743146b590_snapshot": D198_R1_743146B590,
        "d198_r2_56b768300a_snapshot": D198_R2_56B768300A,
        "delta_from_pre_d194": {
            name: measured[name] - PRE_D194_9998276020[name] for name in EXPECTED
        },
        "delta_from_reviewed_ea3e570": {
            name: measured[name] - REVIEWED_EA3E570[name] for name in EXPECTED
        },
        "method": "recursive JAXPR equation count attributed by source_info.name_stack",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-count-change", action="store_true")
    args = parser.parse_args()
    result = count(args.master)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" or args.allow_count_change else 2


if __name__ == "__main__":
    raise SystemExit(main())
