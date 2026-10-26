"""Parent-vs-current control for the shared SI3 category sum (DECISION 180).

``run --mode capture`` patches ``jax.jit`` before the driver imports anything and
stores every concrete floating output leaf of every production-jit call, in call
order.  Run the same driver once with the parent tree and once with the current
tree on PYTHONPATH, then ``compare`` the two directories bit by bit.

``run --mode census`` (current tree) wraps ``fortran_sum_categories`` with a
``jax.debug.callback`` that counts, per calling function, the cells whose
summed categories are all ``-0`` (the only cells where a ``+0`` seed changes the
sum).  It is a separate run so the callback cannot touch the captured bits.

Usage:
  nemo_si3_catsum_caller_census.py run --mode capture|census --out DIR -- (--pytest ARGS... | SCRIPT ARGS...)
  nemo_si3_catsum_caller_census.py compare PARENT_DIR CURRENT_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _leaves_to_save(out) -> dict[str, np.ndarray]:
    import jax

    saved = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(out):
        if isinstance(leaf, jax.core.Tracer):
            return {}  # nested trace, not a production output
        if hasattr(leaf, "dtype") and np.issubdtype(leaf.dtype, np.floating):
            saved[jax.tree_util.keystr(path)] = np.asarray(leaf)
    return saved


def install_capture(out_dir: Path) -> None:
    import jax

    out_dir.mkdir(parents=True, exist_ok=True)
    real_jit = jax.jit
    index = open(out_dir / "index.jsonl", "w")
    counter = [0]

    class _Captured:
        def __init__(self, jitted, name):
            self._jitted, self._name = jitted, name

        def __getattr__(self, attr):
            return getattr(self._jitted, attr)

        def __call__(self, *args, **kwargs):
            out = self._jitted(*args, **kwargs)
            leaves = _leaves_to_save(out)
            if leaves:
                k = counter[0]
                counter[0] += 1
                np.savez(out_dir / f"call_{k:06d}.npz", **{f"l{i}": a for i, a in enumerate(leaves.values())})
                index.write(json.dumps({"call": k, "fn": self._name, "leaves": list(leaves)}) + "\n")
                index.flush()
            return out

    def jit(fun=None, *args, **kwargs):
        if fun is None:
            return lambda f: jit(f, *args, **kwargs)
        name = f"{getattr(fun, '__module__', '?')}.{getattr(fun, '__qualname__', repr(fun))}"
        return _Captured(real_jit(fun, *args, **kwargs), name)

    jax.jit = jit


def install_census(out_dir: Path) -> None:
    import jax
    import jax.numpy as jnp
    import legoesm.ice.si3_orca2_state as state

    out_dir.mkdir(parents=True, exist_ok=True)
    real = state.fortran_sum_categories
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # calls, cells, all -0 cells

    def _record(site, n_all_negzero, n_cells):
        c = counts[site]
        c[0] += 1
        c[1] += int(n_cells)
        c[2] += int(n_all_negzero)
        (out_dir / "census.json").write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")

    def wrapped(arr, axis=-1):
        caller = inspect.stack()[1]
        site = f"{Path(caller.filename).name}:{caller.function}"
        all_negzero = jnp.all((arr == 0) & jnp.signbit(arr), axis=axis)
        jax.debug.callback(
            lambda n, m, s=site: _record(s, n, m),
            jnp.sum(all_negzero),
            all_negzero.size,
        )
        return real(arr, axis)

    state.fortran_sum_categories = wrapped


def compare(parent: Path, current: Path) -> dict:
    def index(d):
        return [json.loads(line) for line in (d / "index.jsonl").read_text().splitlines()]

    ip, ic = index(parent), index(current)
    if not ip:
        raise SystemExit("no jit outputs captured: the comparison would be vacuous")
    if [(e["fn"], e["leaves"]) for e in ip] != [(e["fn"], e["leaves"]) for e in ic]:
        raise SystemExit("call sequences differ between parent and current: not a controlled comparison")
    changed, n_leaves = [], 0
    for e in ip:
        name = f"call_{e['call']:06d}.npz"
        with np.load(parent / name) as zp, np.load(current / name) as zc:
            for i, leaf in enumerate(e["leaves"]):
                a, b = zp[f"l{i}"], zc[f"l{i}"]
                n_leaves += 1
                if a.shape != b.shape:
                    raise SystemExit(f"shape differs at call {e['call']} {leaf}")
                ba, bb = a.view(np.uint64 if a.itemsize == 8 else np.uint32), b.view(np.uint64 if b.itemsize == 8 else np.uint32)
                diff = ba != bb
                if diff.any():
                    zero_swap = diff & (a == 0) & (b == 0)
                    changed.append({
                        "call": e["call"], "fn": e["fn"], "leaf": leaf,
                        "cells_changed": int(diff.sum()),
                        "signed_zero_swaps": int(zero_swap.sum()),
                        "neg_to_pos_zero": int((zero_swap & np.signbit(a) & ~np.signbit(b)).sum()),
                        "first_index": [int(x) for x in np.argwhere(diff)[0]],
                        "parent_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "current_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                    })
    return {"calls": len(ip), "leaves_compared": n_leaves, "leaves_changed": len(changed), "changed": changed}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        sep = argv.index("--")
        ap = argparse.ArgumentParser()
        ap.add_argument("--mode", choices=["capture", "census"], required=True)
        ap.add_argument("--out", type=Path, required=True)
        ns = ap.parse_args(argv[1:sep])
        driver = argv[sep + 1 :]
        (install_capture if ns.mode == "capture" else install_census)(ns.out)
        if driver[:1] == ["--pytest"]:
            import pytest

            return int(pytest.main(driver[1:]))
        sys.argv = driver
        sys.path.insert(0, str(Path.cwd()))
        try:
            runpy.run_path(driver[0], run_name="__main__")
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare"])
    ap.add_argument("parent", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument("--json", type=Path)
    ns = ap.parse_args(argv)
    report = compare(ns.parent, ns.current)
    text = json.dumps(report, indent=1, sort_keys=True)
    if ns.json:
        ns.json.write_text(text + "\n")
    print(f"CATSUM_COMPARE calls={report['calls']} leaves={report['leaves_compared']} changed={report['leaves_changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
