"""Build tests/atmosphere/oracle_fixtures/ifs_cloudsc_49r1.npz from the 49r1 oracle runs.

Columns (vote V22d): greedy coverage of the oracle's stage-dump INCREMENTS -- a column is
added while it changes a (stage, field, entry) the chosen set has not yet exercised,
up to NCOL_WHOLE columns for the whole-call test; the first NCOL_STAGE of them carry the
stage dumps.  CLOUDSC columns are independent, so a column subset is an exact sub-problem.
usage: make_fixture.py <cloudsc run dir> <out.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloudsc_io as io  # noqa: E402

NCOL_WHOLE = 25
NCOL_STAGE = 6
ARMS = ("base", "detr")


def activity(st):
    """column -> set of (stage, field, entry) whose value CHANGES at that stage (vs the
    previous stage holding the same field; vs zero for its first holder), any level."""
    order = sorted({t.split("_", 1)[0] for t in st}, key=lambda t: _STAGE_ORDER.index(t))
    sig, prev = {}, {}
    for stg in order:
        for tag in sorted(t for t in st if t.split("_", 1)[0] == stg):
            fld = tag.split("_", 1)[1]
            a = st[tag]
            p = prev.get(fld, np.zeros_like(a))
            act = np.any(a != p, axis=0)
            act = act.reshape(act.shape[0], -1)
            for j in range(act.shape[0]):
                sig.setdefault(j, set()).update((tag, e) for e in np.flatnonzero(act[j]))
            prev[fld] = a
    return sig


_STAGE_ORDER = ("s1", "s32", "s33", "s34", "s41", "s42", "s43", "s44", "s45", "s47", "s48",
                "s412", "s413", "s414", "s49", "s415", "s521", "s522", "s6")


def main(run, outp):
    run = Path(run)
    cfg, s = io.config(run / "out_base/settings_49r1.txt")
    ncol, nlev = io.read_txt(run / "run/input/PT.txt").shape
    nrec = nlev - cfg.ncldtop + 1
    stages = {arm: io.read_stages(run / f"dump_{arm}", ncol, nrec) for arm in ARMS}
    sig = {}
    for arm in ARMS:
        for j, v in activity(stages[arm]).items():
            sig.setdefault(j, set()).update((arm,) + e for e in v)
    chosen, have = [], set()
    while len(chosen) < NCOL_WHOLE:
        j = max((j for j in sig if j not in chosen), key=lambda j: (len(sig[j] - have), -j))
        chosen.append(j)
        have |= sig[j]
    total = set().union(*sig.values())
    print(f"coverage {len(have)}/{len(total)} with columns {chosen}")
    cols = np.array(chosen)
    out = {"columns": cols, "ncldtop": cfg.ncldtop, "settings_names": np.array(list(io.SETTINGS)),
           "coverage": np.array([len(have), len(total)])}
    th = io.constants(run / "run/input")
    for f in th._fields:
        out[f"const_{f}"] = getattr(th, f)
    for f in cfg._fields:
        out[f"cfg_{f}"] = getattr(cfg, f)
    for arm in ARMS:
        r = io.raw_inputs(run / "run/input", run / f"out_{arm}")
        for k, v in r.items():
            out[f"{arm}_in_{k}"] = v[cols] if np.ndim(v) else v
        for k, v in io.read_oracle(run / f"out_{arm}", cfg.ncldtop).items():
            out[f"{arm}_out_{k}"] = v[cols]
        for tag, a in stages[arm].items():
            out[f"{arm}_stage_{tag}"] = a[:, cols[:NCOL_STAGE]]
    np.savez_compressed(outp, **out)
    print("wrote", outp, Path(outp).stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
