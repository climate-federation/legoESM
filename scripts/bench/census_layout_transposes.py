#!/usr/bin/env python
"""Name the transposes the GPU backend inserted into a compiled step.

Device traces of the lat-lon atmosphere step at 32 and 128 GPUs showed every
class of kernel scaling 3.2-3.5x for a 4x device increase EXCEPT transposes,
which scale 1.38x and carry two thirds of the whole local scaling loss. A CPU
dump of the same step contains ZERO transposes before or after optimisation,
so they are the GPU backend's layout assignment rather than anything the model
asks for. A trace can only say "a fusion whose root is a transpose ran"; the
post-optimisation HLO says on WHAT SHAPE, with WHAT permutation, and whether
the compiler was able to make it free.

WHAT IT REPORTS
  * every transpose in the module, grouped by (shape, permutation), with a
    count -- these are the candidates a layout change would remove;
  * whether each is a real data movement or a BITCAST, which costs nothing
    and must not be counted as a win;
  * reshapes and copies alongside, because on this lane the program contains
    288 reshapes and no transposes, so the reshapes are the suspected origin;
  * the module's total instruction count, so the shares are readable.

WHAT IT IS NOT. Not a timing. The dump says what the executable contains,
which is deterministic; how long each takes is the trace's job. Counts from a
small grid transfer to a large one because layout assignment is a property of
the program and the backend; times do not, and are not taken here.

USE
  1. Run the bench with ``XLA_FLAGS=--xla_dump_to=<dir>``.
  2. ``python scripts/bench/census_layout_transposes.py --dump-dir <dir>
     --module-re jit__scan_run``
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from census_xla_dump import find_module  # noqa: E402  (module selection rules)

# `%fusion.3 = f32[16,4096,26]{2,1,0} transpose(%p), dimensions={1,0,2}`
_TRANSPOSE = re.compile(
    r"^\s*%?(?P<name>[\w.\-]+)\s*=\s*"
    r"(?P<shape>[a-z0-9]+\[[0-9,]*\])(?P<layout>\{[0-9,:TE ]*\})?\s*"
    r"transpose\((?P<operands>[^)]*)\)(?P<attrs>[^\n]*)$",
    re.MULTILINE)
_DIMS = re.compile(r"dimensions=\{([0-9,]*)\}")


def census(text: str) -> dict:
    """Transposes, reshapes and copies in one HLO module."""
    groups: dict[tuple, dict] = collections.OrderedDict()
    for m in _TRANSPOSE.finditer(text):
        dims = _DIMS.search(m.group("attrs") or "")
        key = (m.group("shape"), dims.group(1) if dims else "?")
        # A transpose the compiler turned into a bitcast is a relabelling of
        # the same bytes. Counting it as removable work would inflate the
        # prize for a layout change by however many of them there are.
        bitcast = "bitcast" in (m.group("attrs") or "").lower()
        g = groups.setdefault(key, {"shape": key[0], "dimensions": key[1],
                                    "count": 0, "bitcast": 0,
                                    "examples": []})
        g["count"] += 1
        g["bitcast"] += int(bitcast)
        if len(g["examples"]) < 3:
            g["examples"].append(m.group("name"))
    n_instr = len(re.findall(r"^\s*%?[\w.\-]+\s*=\s", text, re.MULTILINE))
    return {
        "n_instructions": n_instr,
        "n_transpose": sum(g["count"] for g in groups.values()),
        "n_transpose_bitcast": sum(g["bitcast"] for g in groups.values()),
        "n_reshape": len(re.findall(r"=\s*[^=\n]*?\breshape\(", text)),
        "n_copy": len(re.findall(r"=\s*[^=\n]*?\bcopy\(", text)),
        "groups": sorted(groups.values(), key=lambda g: -g["count"]),
    }


_SELFTEST = """
ENTRY main {
  %p = f32[4,8,2]{2,1,0} parameter(0)
  %t1 = f32[8,4,2]{2,1,0} transpose(%p), dimensions={1,0,2}
  %t2 = f32[8,4,2]{2,1,0} transpose(%p), dimensions={1,0,2}
  %t3 = f32[2,4,8]{2,1,0} transpose(%p), dimensions={2,0,1}, metadata={bitcast}
  %r = f32[64]{0} reshape(%t1)
  %c = f32[64]{0} copy(%r)
  ROOT %out = f32[64]{0} add(%r, %c)
}
"""


def _selftest() -> int:
    """The gate must SEE what it claims to see, and separate bitcasts."""
    c = census(_SELFTEST)
    assert c["n_transpose"] == 3, c
    assert c["n_transpose_bitcast"] == 1, c
    assert c["n_reshape"] == 1 and c["n_copy"] == 1, c
    assert c["groups"][0]["count"] == 2, c
    assert c["groups"][0]["dimensions"] == "1,0,2", c
    # A module with no transposes must report zero rather than crash: a
    # census that only works on the case it was written for is not a census.
    empty = census("ENTRY main {\n  ROOT %p = f32[4]{0} parameter(0)\n}\n")
    assert empty["n_transpose"] == 0 and empty["groups"] == [], empty
    print("selftest OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-dir", type=Path)
    p.add_argument("--module-re", default="jit__scan_run")
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    if a.dump_dir is None:
        p.error("--dump-dir is required unless --selftest")
    path = find_module(a.dump_dir, a.module_re)
    c = census(path.read_text())
    c["module"] = path.name
    print(f"{path.name}: {c['n_instructions']} instructions, "
          f"{c['n_transpose']} transposes "
          f"({c['n_transpose_bitcast']} of them bitcasts, i.e. free), "
          f"{c['n_reshape']} reshapes, {c['n_copy']} copies")
    if not c["groups"]:
        print("  no transposes in this module")
    for g in c["groups"]:
        print(f"  {g['count']:4d} x {g['shape']:<24} dimensions={{{g['dimensions']}}}"
              f"{'  [bitcast]' if g['bitcast'] == g['count'] else ''}"
              f"   e.g. {g['examples'][0]}")
    if a.out:
        a.out.write_text(json.dumps(c, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
