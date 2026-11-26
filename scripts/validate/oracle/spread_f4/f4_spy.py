"""F4 spread instrument, test side (pytest plugin, -p f4_spy).

Points the committed test module's fixture-path constant at F4_FIX_PATH and wraps
the test's OWN metric function(s): every call is passed through unchanged (same
return value, so the test's assertions are untouched) and recorded as
(node id, call index, caller line, metric value, the 2nd argument = the oracle
array).  analyze.py pairs base and seed records by (node, index, line)."""
from __future__ import annotations

import linecache
import os
import pickle
import sys
from pathlib import Path

import numpy as np

FIX_ATTR = os.environ["F4_FIX_ATTR"]
FIX_PATH = os.environ["F4_FIX_PATH"]
METRICS = os.environ["F4_METRICS"].split(",")
OUT = os.environ["F4_OUT"]
NEUTRAL = os.environ.get("F4_NEUTRAL") == "1"
REC: list = []
STATE = {"node": None, "i": 0}


def _spy(orig, name):
    def f(a, b, *args, **kw):
        v = orig(a, b, *args, **kw)
        fr = sys._getframe(1)
        src = linecache.getline(fr.f_code.co_filename, fr.f_lineno).strip()
        REC.append(dict(node=STATE["node"], i=STATE["i"], line=fr.f_lineno, metric=name, src=src,
                        args=list(args), kw=dict(kw), value=float(np.max(np.asarray(v))),
                        oracle=np.array(np.asarray(b), copy=True)))
        STATE["i"] += 1
        # seed runs: keep the test running past its own TOL asserts so every
        # comparison is recorded (the value recorded above is the true one)
        return (np.zeros_like(np.asarray(v)) if np.ndim(v) else 0.0) if NEUTRAL else v
    f.__wrapped__ = orig
    return f


def pytest_collection_modifyitems(session, config, items):
    for m in {it.module for it in items}:
        setattr(m, FIX_ATTR, Path(FIX_PATH))
        for name in METRICS:
            if hasattr(m, name) and not hasattr(getattr(m, name), "__wrapped__"):
                setattr(m, name, _spy(getattr(m, name), name))


def pytest_runtest_setup(item):
    STATE["node"], STATE["i"] = item.nodeid, 0


def pytest_sessionfinish(session, exitstatus):
    with open(OUT, "wb") as fh:
        pickle.dump(dict(records=REC, exitstatus=int(exitstatus)), fh)
