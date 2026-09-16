#!/usr/bin/env python
"""TWIN ADMISSION for a re-acquired NEMO kt=1 record.

A new record is only usable as a substitute for the certified one if it
describes the SAME RUN.  The DINO_KT1_RANKDUMP config copy exists to add
rank-tagged writers to dyn_spg_ts so the barotropic substep streams stop
being a 16-way filename race; a writer must not change a single number.
This checks that claim the only way it can be checked -- every variable of
every restart tile, byte for byte, against the certified record -- and exits
non-zero if anything moved.

It is the PREREQUISITE for scoring anything against the new record: a
substep ladder measured on a record that drifted would be a measurement of
the drift.

Usage::

    python scripts/validate/ocean_fidelity/dino_1226/\
        kt1_record_twin_admission.py [CERTIFIED_GLOB NEW_GLOB]

Defaults to RUN_FROMREST_KT1 vs the rankdump record.
"""
import glob, sys, numpy as np, netCDF4 as nc
CERTIFIED = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
             "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
NEW = ("/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump/"
       "DINO_00000001_restart_*.nc")
A = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 2 else CERTIFIED))
B = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else NEW))
print(f"tiles: certified {len(A)}  rankdump {len(B)}")
if not A or not B or len(A) != len(B):
    raise SystemExit(f"tile counts differ or are empty: {len(A)} vs {len(B)}")
nvar = 0; moved = {}
for a, b in zip(A, B):
    da, db = nc.Dataset(a), nc.Dataset(b)
    ka, kb = set(da.variables), set(db.variables)
    if ka != kb:
        print(f"  VARIABLE SET DIFFERS on {a[-7:-3]}: only-certified={sorted(ka-kb)} only-rankdump={sorted(kb-ka)}")
    for k in sorted(ka & kb):
        x = np.asarray(da[k][:]); y = np.asarray(db[k][:])
        nvar += 1
        if x.shape != y.shape:
            moved.setdefault(k, []).append(("shape", x.shape, y.shape)); continue
        d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float) if x.dtype.kind in "fciu" else None
        if d is None:
            if not np.array_equal(x, y): moved.setdefault(k, []).append(("neq", None, None))
        else:
            n = int(np.count_nonzero(d))
            if n: moved.setdefault(k, []).append(("cells", n, float(np.nanmax(np.abs(d)))))
    da.close(); db.close()
print(f"compared {nvar} (variable, tile) pairs")
if not moved:
    print("ADMITTED: every variable of every tile is BIT-IDENTICAL to the certified record")
    sys.exit(0)
print(f"NOT bit-identical on {len(moved)} variables:")
for k, v in sorted(moved.items()):
    tot = sum(x[1] for x in v if x[0] == "cells")
    mx = max((x[2] for x in v if x[0] == "cells"), default=float("nan"))
    print(f"  {k:14s} tiles={len(v):2d} cells={tot:8d} max|d|={mx:.4e}")
sys.exit(1)
