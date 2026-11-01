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

from legoesm.ice.si3_o17 import SI3O17LiveForcing, si3_o17_step
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import DEFAULT_MASTER, _case

EXPECTED = {
    "recursive_total": 576_364,
    "_thermo_category_loop": 342_178,
    "si3_ice_thd_do": 53_490,
    "si3_dyn_orca2": 48_325,
    "remaining_named_and_inlined": 132_371,
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

    def objective(x):
        step_entry, dynamic = x
        out, trace = si3_o17_step(step_entry, SI3O17LiveForcing(1, *dynamic), card)
        publication = (out, trace.finalize.final.fluxes, trace.finalize.final.outputs, trace.finalize.stress)
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(publication)
                  if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)]
        return sum(jnp.sum(leaf) for leaf in leaves)

    closed = jax.make_jaxpr(objective)(inputs)
    counts: Counter[str] = Counter()
    _walk(closed.jaxpr, counts)
    measured = {name: int(counts[name]) for name in EXPECTED}
    return {
        "status": "PASS" if measured == EXPECTED else "COUNT_CHANGED",
        "measured": measured,
        "expected": EXPECTED,
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
