"""Dump legoESM's CAM-default CLUBB parameters and flag table to JSON.

Stage 1 of the CLUBB-core oracle fixture (run in the legoESM environment):
the Fortran oracle is then run with exactly these parameters and flags by
``gen_clubb_core_fixture.py``.  The flag table is parsed from the reference
table at the end of ``turbulence/clubb.py`` (the flags are hard-wired in the
port, so that table is the only record of them).

Usage: python dump_cam_settings.py OUT_DIR
Writes OUT_DIR/cam_params.json and OUT_DIR/cam_flags.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_TABLE_MARKER = "CAM-default CLUBB model-flag values (reference table)"
_FLAG_RE = re.compile(r"#\s+(\w+)\s*=\s*(True|False|-?\d+)\b")


def parse_flag_table(source: str) -> dict:
    """Flag name -> bool/int from the reference table in clubb.py's source."""
    flags, on = {}, False
    for ln in source.splitlines():
        if _TABLE_MARKER in ln:
            on = True
        if not on:
            continue
        m = _FLAG_RE.match(ln)
        if m:
            v = m.group(2)
            flags[m.group(1)] = (v == "True") if v in ("True", "False") else int(v)
    if not flags:
        raise ValueError("CLUBB flag reference table not found in clubb.py")
    return flags


def dump(out: Path) -> tuple[dict, dict]:
    from legoesm.atmosphere.physics.turbulence import clubb as C
    params = {k: float(v) for k, v in C.CLUBBParams()._asdict().items()}
    flags = parse_flag_table(Path(C.__file__).read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "cam_params.json").write_text(json.dumps(params, indent=1))
    (out / "cam_flags.json").write_text(json.dumps(flags, indent=1))
    return params, flags


if __name__ == "__main__":
    p, f = dump(Path(sys.argv[1]))
    print("params", len(p), "flags", len(f))
