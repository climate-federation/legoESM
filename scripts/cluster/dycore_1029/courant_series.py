"""#1029 probe: vertical Courant number vs column departure over the last 500 steps.

Input: a ``burst_anatomy_probe.py --keep 500`` directory (the rerun of job
28432774).  Each saved state goes once through ``temperature_budget.step_budget``
(the committed Courant instrument, commit 01df88eb06, unchanged).  Per step:

  C    hot column, RK stage 1 (the saved state), max over all levels
  C3   hot column, max over the 3 RK stages and all levels
  Cq   quiet column (same row, 180 deg away), stage 1, max over levels
  A    hot-column swing amplitude (``swing_amplitude``)
  E    max A over the other columns of the row, hot column +-2 left out
  ident  max over stages of max|V dt - C dfu| / max|V dt|  (instrument identity)
  floor  hot column has a level at T <= T_min + 0.01 K
  k_C, k_A  levels (0 = top) of the stage-1 max C and of the max swing term

Also per level: C_lev (stage 1), C3_lev (max over stages), Cq_lev (quiet);
``at_swing_level`` repeats the C/C3 onsets at the fixed level k_A(t_D).

and the onsets t_D (A > E sustained to the last saved step), t_D2 (A > 2E),
t_C (C >= 1 sustained), t_C3, plus first-ever C >= 1 and first floor step.
The reading rule is pre-registered on #1029 (comment 6435566464); this
script prints numbers only, no verdict.

Usage: python courant_series.py <probe_dir> --out series.json --expect-trip N
       --ref-dir <earlier probe dir> [--n 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temperature_budget import (load_probe, load_state, step_budget,  # noqa: E402
                                swing_amplitude)

EXCLUDE = 2    # neighbours left out of the envelope on each side (pre-registered)


def sustained_onset(steps, flags):
    """First step from which ``flags`` holds at every later step; None if the last fails."""
    onset = None
    for s, f in zip(steps, flags):
        if not f:
            onset = None
        elif onset is None:
            onset = s
    return onset


def first_true(steps, flags):
    return next((s for s, f in zip(steps, flags) if f), None)


def row_envelope(T_row, j, exclude=EXCLUDE):
    """Max swing amplitude over the columns of ``T_row`` (nlon, nlev), periodic
    in longitude, leaving out column ``j`` and ``exclude`` neighbours each side."""
    n = T_row.shape[0]
    skip = {(j + d) % n for d in range(-exclude, exclude + 1)}
    return max(swing_amplitude(T_row[m]) for m in range(n) if m not in skip)


def series_row(model, state, target, ij):
    """Per-step numbers for column ``ij`` of one saved state."""
    i, j = ij
    _, _, info = step_budget(model, state, target)
    Cs = [np.asarray(c) for c in info["courant_stages"]]
    jq = (j + Cs[0].shape[1] // 2) % Cs[0].shape[1]
    T = np.asarray(state.T)
    A = swing_amplitude(T[i, j])
    E = row_envelope(T[i], j)
    return dict(
        C=float(np.max(Cs[0][i, j])), C3=float(max(np.max(c[i, j]) for c in Cs)),
        Cq=float(np.max(Cs[0][i, jq])), A=A, E=E,
        ident=max(float(np.max(np.abs(np.asarray(x)))) / sc
                  for x, sc in zip(info["courant_ident"], info["courant_scale"])),
        k_C=int(np.argmax(Cs[0][i, j])),
        k_A=1 + int(np.argmax(np.abs(T[i, j, 1:-1] - 0.5 * (T[i, j, :-2] + T[i, j, 2:])))),
        floor=bool(np.any(T[i, j] <= model.config.T_min + 0.01)),
        Tmax=float(np.max(T[i, j])),
        C_lev=Cs[0][i, j].tolist(), C3_lev=np.max([c[i, j] for c in Cs], 0).tolist(),
        Cq_lev=Cs[0][i, jq].tolist())


def onsets(rows):
    st = [r["step"] for r in rows]
    return dict(
        t_D=sustained_onset(st, [r["A"] > r["E"] for r in rows]),
        t_D2=sustained_onset(st, [r["A"] > 2.0 * r["E"] for r in rows]),
        t_C=sustained_onset(st, [r["C"] >= 1.0 for r in rows]),
        t_C3=sustained_onset(st, [r["C3"] >= 1.0 for r in rows]),
        first_C=first_true(st, [r["C"] >= 1.0 for r in rows]),
        first_C3=first_true(st, [r["C3"] >= 1.0 for r in rows]),
        first_floor=first_true(st, [r["floor"] for r in rows]),
        window=[st[0], st[-1]], n=len(st),
        K3_max_ident=max(r["ident"] for r in rows),
        K3_pass=all(r["ident"] <= 1e-9 for r in rows),
        K4_max_Cq=max(r["Cq"] for r in rows),
        K4_pass=all(r["Cq"] < 0.2 for r in rows))


def at_swing_level(rows, t_D):
    """Same Courant series at one fixed level: k_A at the departure step t_D
    (the level of largest swing there) instead of the column maximum.  Onsets
    use the unchanged definitions; max_Cq_k is the quiet column at that level.
    No departure (t_D None) -> no level, returns None (the rule makes no claim)."""
    if t_D is None:
        return None
    k = next(r["k_A"] for r in rows if r["step"] == t_D)
    st = [r["step"] for r in rows]
    C = [r["C_lev"][k] for r in rows]
    C3 = [r["C3_lev"][k] for r in rows]
    return dict(k=k, t_D=t_D,
                t_C=sustained_onset(st, [c >= 1.0 for c in C]),
                t_C3=sustained_onset(st, [c >= 1.0 for c in C3]),
                first_C=first_true(st, [c >= 1.0 for c in C]),
                first_C3=first_true(st, [c >= 1.0 for c in C3]),
                C_at_t_D=C[st.index(t_D)], C3_at_t_D=C3[st.index(t_D)],
                max_Cq_k=max(r["Cq_lev"][k] for r in rows), C=C, C3=C3)


def readable(o):
    """The rule reads the Courant onsets only if K3 and K4 hold at every step."""
    return dict(o, C_readable=bool(o["K3_pass"] and o["K4_pass"]))


def check_window(steps, trip, n, expect_trip):
    """K1 + window: trip == expect_trip, saved steps == trip-n .. trip-1 exactly."""
    if trip is None or trip != expect_trip:
        raise SystemExit(f"K1 FAILED: probe trip {trip} != expected {expect_trip}")
    if steps != list(range(trip - n, trip)):
        raise SystemExit(f"saved steps {steps[:1]}..{steps[-1:]} (n={len(steps)}) "
                         f"are not a contiguous window ending at trip-1, trip={trip}")


def compare_states(ref_dir, probe_dir):
    """K2: max |difference| of u, v, T, p_s for every state saved in ``ref_dir``,
    which must all also be saved in ``probe_dir`` (missing or empty = error)."""
    refs = sorted(Path(ref_dir).glob("clean_*.npz"))
    if not refs:
        raise SystemExit(f"K2: no saved states in {ref_dir}")
    out = {}
    for r in refs:
        x, y = np.load(r), np.load(Path(probe_dir) / r.name)
        out[r.stem] = {f: float(np.max(np.abs(x[f] - y[f])))
                       for f in ("u", "v", "T", "p_s")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-trip", type=int, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="K2: earlier probe dir whose saved states must overlap")
    # K1's test-runner half (runner trip == probe trip) is the launcher's
    # CONTROL line; --expect-trip is that runner trip.
    a = ap.parse_args(argv)
    meta, grid, _, model, _, target, clean = load_probe(a.probe_dir)
    check_window([int(c.stem[6:]) for c in clean], meta["trip_step"], a.n,
                 a.expect_trip)
    k2 = compare_states(a.ref_dir, a.probe_dir)     # K2, reported only
    print("K2", json.dumps(k2), flush=True)
    last = load_state(clean[-1])
    i, j, _ = np.unravel_index(int(np.argmax(np.asarray(last.T))), last.T.shape)
    lat = float(np.degrees(np.asarray(grid.lat))[i])
    lon = float(np.degrees(np.asarray(grid.lon))[j])
    print(f"seed {meta['seed']} hot column ({i},{j}) = {lat:.2f}N {lon:.2f}E, "
          f"{len(clean)} states", flush=True)
    rows = []
    for c in clean:
        r = dict(step=int(c.stem[6:]), **series_row(model, load_state(c), target, (i, j)))
        rows.append(r)
        print(json.dumps(r), flush=True)
    res = dict(meta=meta, column=[int(i), int(j), lat, lon], rows=rows,
               onsets=readable(onsets(rows)), K2=k2)
    res["at_swing_level"] = at_swing_level(rows, res["onsets"]["t_D"])
    print("ONSETS", json.dumps(res["onsets"]), flush=True)
    lv = res["at_swing_level"]
    print("AT_SWING_LEVEL", json.dumps(lv and {x: v for x, v in lv.items()
                                               if x not in ("C", "C3")}), flush=True)
    a.out.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
