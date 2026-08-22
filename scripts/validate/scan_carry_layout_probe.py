#!/usr/bin/env python
"""Can a layout constraint stop XLA converting a scan carry back and forth?

WHY. The compiled lat-lon step converts between (level, lat, lon) and
(lat, lon, level) because the state is stored lat-leading and XLA:GPU
pipelines level-leading. Serialising the halo's edge payloads level-first
removed 26 of the 28 conversions and 4.0% of the step at 128 GPUs. The two
survivors sit at the SCAN CARRY and act on the full field, and the arithmetic
says they plausibly carry more time than the 26 that went -- they are the
largest single named item left on that lane.

Changing the model's state to level-leading would touch every body that reads
the carry. `jax.experimental.layout.with_layout_constraint` claims to pin an
array's DEVICE layout without changing its logical axis order, which would be
a far smaller change. Whether it survives a scan, a shard_map and this
backend is not something to find out inside the dycore.

WHAT THIS MEASURES. A scan whose carry is a (rows, lon, lev) array. The body
does a vertical cumulative sum (which wants the level axis major) and a
horizontal three-point stencil (which wants it minor) -- the same conflict the
dycore has. Arm A is the plain program. Arm B pins the carry's device layout
to level-major with `with_layout_constraint`. The number is the count of
non-bitcast transposes in the post-optimisation HLO.

READ THE RESULT AS: fewer transposes in B means the constraint reaches the
carry and the lever is real, and the next step is the same experiment inside
the step builder. The SAME count means XLA re-inserts what the constraint
removes, and changing the state's axis order is the only way -- which is a
much larger change and should not be started on this evidence.

It prints counts and configuration, and no verdict.
"""
from __future__ import annotations

import argparse
import re
import sys

import jax
import jax.numpy as jnp


def build(rows: int, lon: int, lev: int, n_steps: int, pin: bool):
    """A scan with the dycore's layout conflict, optionally with the carry pinned."""
    from jax.experimental.layout import Layout, with_layout_constraint

    # (rows, lon, lev) -> level-major. Read from jax/_src/layout.py: Layout
    # takes major_to_minor, so (2, 0, 1) puts the level axis outermost.
    level_major = Layout((2, 0, 1))

    def body(carry, _x):
        if pin:
            carry = with_layout_constraint(carry, level_major)
        # Vertical closure: wants the level axis major.
        col = jnp.cumsum(carry, axis=-1)
        # Horizontal stencil: wants the level axis minor.
        east = jnp.roll(col, 1, axis=1)
        west = jnp.roll(col, -1, axis=1)
        out = carry + 0.25 * (east + west - 2.0 * col)
        return out, None

    def run(x):
        out, _ = jax.lax.scan(body, x, xs=None, length=n_steps)
        return out

    return jax.jit(run)


_TRANSPOSE = re.compile(
    r"^\s*%?[\w.\-]+\s*=\s*[a-z0-9]+\[[0-9,]*\](?:\{[0-9,:TE ]*\})?\s*"
    r"transpose\([^)]*\)([^\n]*)$", re.MULTILINE)


def count_transposes(text: str) -> tuple[int, int]:
    """(non-bitcast transposes, bitcast transposes) in one HLO module."""
    total = bitcast = 0
    for m in _TRANSPOSE.finditer(text):
        total += 1
        if "bitcast" in (m.group(1) or "").lower():
            bitcast += 1
    return total - bitcast, bitcast


def _selftest() -> int:
    real, bit = count_transposes(
        "  %a = f32[4,8,2]{2,1,0} transpose(%p), dimensions={1,0,2}\n"
        "  %b = f32[2,4,8]{2,1,0} transpose(%p), dimensions={2,0,1}, metadata={bitcast}\n"
        "  %c = f32[64]{0} add(%a, %b)\n")
    assert (real, bit) == (1, 1), (real, bit)
    assert count_transposes("ROOT %p = f32[4]{0} parameter(0)\n") == (0, 0)
    print("selftest OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--lon", type=int, default=256)
    p.add_argument("--lev", type=int, default=26)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()

    print(f"# jax={jax.__version__} devices={jax.devices()}")
    print(f"# carry=({a.rows}, {a.lon}, {a.lev}) scan_length={a.steps}")
    x = jnp.ones((a.rows, a.lon, a.lev), jnp.float32)
    ref = None
    print(f"{'arm':<14} {'transposes':>11} {'bitcasts':>9} {'instructions':>13}")
    for pin in (False, True):
        try:
            fn = build(a.rows, a.lon, a.lev, a.steps, pin)
            text = fn.lower(x).compile().as_text()
        except Exception as exc:  # the constraint may simply not be supported
            print(f"{'pinned' if pin else 'plain':<14} REFUSED: "
                  f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        real, bit = count_transposes(text)
        n_instr = len(re.findall(r"^\s*%?[\w.\-]+\s*=\s", text, re.MULTILINE))
        print(f"{'pinned' if pin else 'plain':<14} {real:>11} {bit:>9} {n_instr:>13}")
        got = fn(x)
        if ref is None:
            ref = got
        else:
            # A layout constraint must not change the answer.
            same = bool(jnp.array_equal(ref, got))
            print(f"# pinned arm bit-identical to plain: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
