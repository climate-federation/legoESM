#!/usr/bin/env python3
"""WHERE are land columns held?  Serial CPU replay of an AMIP run from one of its
checkpoints, logging the per-column hold mask of every land step.

The production log only prints rank-local held COUNTS, so it cannot say where the
held columns are.  This replays the run's own command line (from its
run_manifest.json; --distributed dropped, output and restart redirected to <out>)
with two wrappers around the shipped land step, nothing else changed:

* ``gather_land_columns`` records the packed land-column index (global cell ids);
* ``_hold_unsolved_columns`` returns its hold mask and the solver's converged flag
  through ``jax.debug.callback``.

Writes <out>/held_log.npz: held (n_land_steps, n_packed) bool, converged (same),
cells (n_packed,) global cell ids.  Same instrumentation pattern as
amip_runs/_campaign_0929/autopsy/land_replay_cert.py.  Run from the repo root of
the run's code with PYTHONPATH set, JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1.

Usage: nh_held_columns_replay.py <src_run_dir> <restart_day> <out_dir> <days> <wallclock_s>
(the run stops gracefully at the wallclock limit; the log keeps every land step seen)
"""
import json
import os
import runpy
import shlex
import shutil
import sys

import numpy as np

src, day, out, ndays, wall = (sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4],
                              sys.argv[5])
os.makedirs(out, exist_ok=True)
ck = f"{out}/checkpoint_day_{day:04d}.npz"
if not os.path.exists(ck):
    shutil.copyfile(f"{src}/checkpoint_day_{day:04d}.npz", ck)

import jax  # noqa: E402
import legoesm.land.multilayer_land as ml  # noqa: E402

IDX, HELD, CONV = {}, [], []
_gather, _hold = ml.gather_land_columns, ml._hold_unsolved_columns


def gather(tree, idx, ncol):
    if "cells" not in IDX:
        IDX["cells"] = np.asarray(idx)
        print("PACKED land columns", IDX["cells"].size, flush=True)
    return _gather(tree, idx, ncol)


def _cb(bad, conv):
    HELD.append(np.asarray(bad))
    CONV.append(np.asarray(conv))


def hold(state, new_state, response, surface_out, forcing, config, ncol, **kw):
    res = _hold(state, new_state, response, surface_out, forcing, config, ncol, **kw)
    conv = (surface_out.converged if surface_out.converged is not None
            else res[3] * False)
    jax.debug.callback(_cb, res[3], conv)
    return res


ml.gather_land_columns, ml._hold_unsolved_columns = gather, hold

argv = shlex.split(json.load(open(f"{src}/run_manifest.json"))["run"]["command_line"])[1:]


def setopt(name, val):
    while name in argv:
        i = argv.index(name)
        del argv[i:i + 2]
    argv.extend([name, val])


while "--distributed" in argv:
    argv.remove("--distributed")
setopt("--output", out)
setopt("--restart-from", ck)
setopt("--days", ndays)
setopt("--checkpoint-days", "10")
setopt("--max-wallclock-seconds", wall)
sys.argv = ["run_amip.py"] + argv
sys.path.insert(0, "scripts/run")
try:
    runpy.run_path("scripts/run/run_amip.py", run_name="__main__")
finally:
    if HELD:
        np.savez(f"{out}/held_log.npz", held=np.stack(HELD), converged=np.stack(CONV),
                 cells=IDX.get("cells", np.arange(HELD[0].size)))
        print("HELD LOG", np.stack(HELD).shape, flush=True)
