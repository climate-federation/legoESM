"""F4 spread instrument, analysis.  Controls (BRIEF C1-C5) then spread + class.

C1 base generator rerun -> fixture.npz equal to the committed fixture bitwise
   (every array key; `meta` reported separately).
C2 base vs base2 fixture bitwise; seed-0 vs seed-0 rerun (s0r) bitwise.
C3 every oracle call of every run: Fortran list-directed readback == intended text
   values bitwise; per seed, moved floats = one adjacent step from base, rest identical.
C4 the port rel is the committed test's own metric value on its own call
   (recorded by the spy on the committed fixture); the spy on the base-rerun fixture
   must record bitwise the same values (same oracle bytes -> same numbers).
C5 compiler + flags from the build log.

spread per comparison = max over seeds of metric(seed oracle, base oracle) with the
test's own metric and its own extra args.  A comparison whose oracle array changes
shape under a seed gets a non-finite spread (UNMEASURED).

usage: analyze.py FAMILY RUNDIR BUILDLOG
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread")
import perturb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from families import FAMILIES  # noqa: E402


def npz_equal(a, b):
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    bad = {}
    if set(za.files) != set(zb.files):
        bad["_keys"] = sorted(set(za.files) ^ set(zb.files))
    for k in sorted(set(za.files) & set(zb.files)):
        x, y = za[k], zb[k]
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            if x.shape == y.shape and np.issubdtype(x.dtype, np.number):
                bad[k] = float(np.max(np.abs(x.astype(float) - y.astype(float))))
            else:
                bad[k] = f"{x.dtype}{x.shape} vs {y.dtype}{y.shape}"
    return bad


def recs(path):
    d = pickle.load(open(path, "rb"))
    return d["exitstatus"], {(r["node"], r["i"], r["line"]): r for r in d["records"]}


def main(family, rundir, buildlog):
    fam = FAMILIES[family]
    R = Path(rundir)
    spec = importlib.util.spec_from_file_location("t", fam["test"])
    T = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "/burg/glab/users/pg2328/wt_f4")
    spec.loader.exec_module(T)
    metric = getattr(T, fam["metrics"][0])
    notes, ctl = [], {}

    b1 = npz_equal(R / "base/fixture.npz", fam["fixture"])
    meta_diff = {k: v for k, v in b1.items() if "meta" in k}
    b1 = {k: v for k, v in b1.items() if "meta" not in k}
    local = fam.get("local_ref", False)     # VOTE V9 (b): committed fixture built elsewhere
    if local:
        notes.append(f"V9 local reference: committed fixture differs from this machine's rerun by {b1}; "
                     f"the local base run is the reference (C1 = base vs base2 bitwise)")
        b1 = npz_equal(R / "base/fixture.npz", R / "base2/fixture.npz")
    ctl["C1"] = "PASS" if not b1 else "FAIL"
    notes.append(f"C1 base rerun vs committed fixture: {'bitwise' if not b1 else b1}; "
                 f"meta keys differing: {sorted(meta_diff)}")
    c2a = npz_equal(R / "base/fixture.npz", R / "base2/fixture.npz")
    c2b = npz_equal(R / "s0/fixture.npz", R / "s0r/fixture.npz")
    ctl["C2"] = "PASS" if not c2a and not c2b else "FAIL"
    notes.append(f"C2 base vs base2: {c2a or 'bitwise'}; s0 vs s0 rerun: {c2b or 'bitwise'}")

    calls_b = json.load(open(R / "base/calls.json"))["calls"]
    c3 = []
    ok3 = bool(calls_b) and all(c["sources"] for c in calls_b) and all(
        s["readback_bitwise"] for c in calls_b for s in c["sources"])
    for sd in perturb.SEEDS:
        cs = json.load(open(R / f"s{sd}/calls.json"))["calls"]
        if len(cs) != len(calls_b) or any(len(a["sources"]) != len(b["sources"]) or not a["sources"]
                                           for a, b in zip(cs, calls_b)):
            ok3 = False
            c3.append((sd, "call count differs"))
            continue
        n_moved = 0
        for i, (cb, cp) in enumerate(zip(calls_b, cs)):
            for j, (sb, sp) in enumerate(zip(cb["sources"], cp["sources"])):
                vb = np.load(R / f"base/call{i}_{j}_floats.npy")
                vp = np.load(R / f"s{sd}/call{i}_{j}_floats.npy")
                mv = vb != vp
                step = np.nextafter(vb[mv], np.where(vp[mv] > vb[mv], np.inf, -np.inf))
                good = (sp["readback_bitwise"] and vb.shape == vp.shape
                        and step.tobytes() == vp[mv].tobytes()
                        and np.all(vp[vb == 0] == 0) and mv.sum() == np.sum(np.isfinite(vb) & (vb != 0)))
                ok3 &= bool(good)
                n_moved += int(mv.sum())
        c3.append((sd, n_moved))
    ctl["C3"] = "PASS" if ok3 else "FAIL"
    notes.append(f"C3 readback bitwise + one-step moves (all finite nonzero floats moved), "
                 f"n_moved per seed {c3}; base calls {len(calls_b)}")

    st_c, rc = recs(R / "spy_committed.pkl")
    st_b, rb = recs(R / "spy_base.pkl")
    same = rc.keys() == rb.keys() and all(
        np.float64(rc[k]["value"]).tobytes() == np.float64(rb[k]["value"]).tobytes() for k in rc)
    if local:
        notes.append("V9: committed-fixture port rel per comparison: "
                     + str({f"{k[0].split('::', 1)[1]}#{k[1]}": rc[k]["value"] for k in rc}))
        same = bool(rb) and st_b == 0
        rc = rb
        st_c = st_b
    ctl["C4"] = "PASS" if (same and st_c == 0 and rc) else "FAIL"
    notes.append(f"C4 test metric on committed vs base-rerun fixture: {len(rc)} comparisons, "
                 f"bitwise same={same}; pytest exit {st_c}/{st_b}")

    log = Path(buildlog).read_text(errors="replace")
    if (R / "base/genbuild.log").exists():      # generator built the oracle itself (traced)
        log += "\n" + (R / "base/genbuild.log").read_text(errors="replace")
        notes.append("C5 flags include the generator's own traced build (base/genbuild.log)")
    comp = sorted(set(re.findall(r"GNU Fortran[^\n]*", log)))
    if not comp:
        mz = np.load(R / "base/fixture.npz", allow_pickle=True)
        comp = sorted({c for k in mz.files if "meta" in k for c in re.findall(r"GNU Fortran[^\"\'\n]*", str(mz[k]))})
        log += " ".join(str(mz[k]) for k in mz.files if "meta" in k)
    fast = bool(re.search(r"-ffast-math|-Ofast|-funsafe-math|-fassociative-math|-freciprocal-math|-ffinite-math-only|-fno-signed-zeros|-fexcess-precision=fast", log))
    cmds = sorted({l.strip() for l in log.splitlines() if l.startswith("+") and l.lstrip("+ ").startswith("gfortran ")})
    flags = sorted({t for c in cmds for t in c.split() if t.startswith("-") and not t.startswith(("-o", "-I", "-J", "-c", "--version"))})
    ctl["C5"] = "PASS" if comp and flags and not fast else "FAIL"
    notes.append(f"C5 compiler {comp}; build flags {flags}; fast-math={fast}")

    seeds = {}
    for sd in perturb.SEEDS:
        st, r = recs(R / f"spy_s{sd}.pkl")
        seeds[sd] = r
        extra = set(r) - set(rb)
        if extra:
            notes.append(f"seed {sd}: {len(extra)} metric calls not in base (pairing diverged)")
    base_f, runs, port = {}, {sd: {} for sd in perturb.SEEDS}, {}
    shape_bad = []
    excl = fam.get("exclude_src")
    dropped = [k for k, r in rb.items() if excl and re.search(excl, r["src"])]
    if dropped:
        notes.append(f"excluded (not an oracle output, family exclude_src={excl!r}): {len(dropped)} calls")
    for k, r in rb.items():
        if k in dropped:
            continue
        lab = f"{k[0].split('::', 1)[1]}#{k[1]}@L{k[2]}"
        base_f[lab] = r["oracle"]
        port[lab] = rc[k]["value"]
        for sd in perturb.SEEDS:
            x = seeds[sd].get(k)
            if x is None or x["oracle"].shape != r["oracle"].shape:
                shape_bad.append((lab, sd))
                runs[sd][lab] = np.full(r["oracle"].shape, np.nan)
            else:
                runs[sd][lab] = x["oracle"]

    zb = np.load(R / "base/fixture.npz", allow_pickle=True)
    flips = {}
    for sd in perturb.SEEDS:
        zs = np.load(R / f"s{sd}/fixture.npz", allow_pickle=True)
        for k in zb.files:
            a, b = zb[k], zs[k] if k in zs.files else None
            if b is None or a.dtype == object or not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
                continue
            fl = np.issubdtype(a.dtype, np.floating)
            # integer-valued float arrays (flags, counts, level indices stored as float) are discrete too
            def _int(x):
                return bool(x.size) and bool(np.all(np.isfinite(x))) and bool(np.all(x == np.round(x)))
            # integer-valued in BOTH runs and different: a discrete jump (an echoed float input
            # that merely moved one ulp off an integer is not)
            intlike = fl and a.shape == b.shape and _int(a) and _int(b)
            if a.shape != b.shape or ((not fl or intlike) and not np.array_equal(a, b)) \
                    or (fl and not np.array_equal(a == 0, b == 0)):
                flips.setdefault(k, []).append(sd)
    discrete = bool(flips)
    notes.append(f"discrete changes under seeds (int/bool arrays, shapes, or zero patterns of float "
                 f"arrays): {({k: v[:5] for k, v in flips.items()} if flips else 'none')}")

    def m(lab):
        rr = rb[[k for k in rb if f"{k[0].split('::', 1)[1]}#{k[1]}@L{k[2]}" == lab][0]]
        fn = getattr(T, rr["metric"])
        fn = getattr(fn, "__wrapped__", fn)
        return lambda a, b: float(np.max(np.asarray(fn(a, b, *rr["args"], **rr["kw"]))))
    spread, per_seed = {}, {}
    for lab in base_f:
        mf = m(lab)
        vals = []
        for sd in perturb.SEEDS:
            try:
                vals.append(float(mf(runs[sd][lab], base_f[lab])))
            except Exception:
                vals.append(float("nan"))
        per_seed[lab] = vals
        spread[lab] = max(vals) if all(np.isfinite(v) for v in vals) else None
    if shape_bad:
        notes.append(f"oracle array shape/record changed under a seed (spread UNMEASURED): "
                     f"{sorted(set(l for l, _ in shape_bad))}")
    notes.append(f"pytest exit per seed: {[recs(R / f'spy_s{sd}.pkl')[0] for sd in perturb.SEEDS]}")
    comp_flip = {lab for lab in base_f for sd in perturb.SEEDS
                 if runs[sd][lab].shape == base_f[lab].shape
                 and not np.array_equal(runs[sd][lab] == 0, base_f[lab] == 0)}
    notes.append(f"comparisons whose own oracle array changed zero pattern: {sorted(comp_flip)}")
    rule = os.environ.get("F4_DISCRETE_RULE", "family")   # VOTE V7
    hit = (set(base_f) if discrete else set()) if rule == "family" else comp_flip
    if hit:
        notes.append(f"DISCRETE CHANGE under 1 ulp (rule {rule}): fields with port rel > 1e-12 among "
                     f"{len(hit)} comparisons are reported UNMEASURED, not certified A by spread")
        spread = {k: (None if (k in hit and port[k] > perturb.ROUNDOFF) else v) for k, v in spread.items()}
        per_seed = {k: (v if spread[k] is not None else [float("nan")] * len(v)) for k, v in per_seed.items()}
    verdict = perturb.artifact(
        R / f"spread_{family}.json", family=family, unit=family, fields_spread=spread,
        per_seed=per_seed, port_rel=port, compiler="; ".join(comp), controls=ctl,
        cases=[str(fam["fixture"])],
        inputs_perturbed="every float token of every oracle input text (file/stdin)",
        inputs_untouched="integer tokens; exact zeros/NaN/inf", notes=" | ".join(notes))
    print("CONTROLS", ctl)
    for n in notes:
        print(n[:2000])
    for lab in sorted(base_f):
        if port[lab] > perturb.ROUNDOFF or verdict[lab] != "A":
            print(f"{lab:70s} port {port[lab]:.3e} spread "
                  f"{'None' if spread[lab] is None else format(spread[lab], '.3e')} {verdict[lab]}")
    cnt = {}
    for v in verdict.values():
        cnt[v] = cnt.get(v, 0) + 1
    print("CLASS_COUNTS", family, cnt)


if __name__ == "__main__":
    main(*sys.argv[1:4])
