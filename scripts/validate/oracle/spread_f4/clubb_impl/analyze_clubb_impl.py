"""Stage B (legoESM env): controls C1-C4, spread with the committed test's own
metric, port rel, artifact.  Inputs: plainA.npz, plainB.npz (generator as
committed, SPREAD_MODE unset) and pert.npz (SPREAD_MODE=pert) from gen_spread.py.

The spread metric is tests/atmosphere/test_clubb_core_oracle.py::max_rel_errors
itself, run with its module-level ``_step`` replaced by a function returning
the perturbed oracle output of step n (so the perturbed oracle sits in the
"ours" slot and the committed/unperturbed oracle s_{n+1} is the reference).

Usage: python analyze_clubb.py RUNDIR OUT_JSON
"""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread")
import perturb  # noqa: E402

WT = "/burg/glab/users/pg2328/wt_f4"
FIX = f"{WT}/tests/atmosphere/oracle_fixtures/clubb_core_cam_implemented.npz"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bitwise_diff(a: dict, b: dict, keys):
    bad = {}
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            try:
                bad[k] = float(np.max(np.abs(x.astype(float) - y.astype(float))))
            except (ValueError, TypeError):
                bad[k] = f"{x.dtype}{x.shape} vs {y.dtype}{y.shape}"
    return bad


def main(rundir, out_json):
    import jax
    jax.config.update("jax_enable_x64", True)
    T = load("clubb_core_test", f"{WT}/tests/atmosphere/test_clubb_core_oracle.py")
    G = load("gen_spread", "/burg/glab/users/pg2328/oracle_audit/f4/clubb_impl/gen_spread_impl.py")
    fx0 = dict(np.load(FIX))
    pa = dict(np.load(f"{rundir}/plainA.npz"))
    pb = dict(np.load(f"{rundir}/plainB.npz"))
    pp = dict(np.load(f"{rundir}/pert.npz"))
    cases = [str(c) for c in fx0["meta/cases"]]
    n_steps = int(fx0["meta/n_steps"])
    fields = tuple(G.STATE) + tuple(G.CARRIED)
    controls = {}

    # C1: unperturbed generator rerun (and the base trajectory of the perturbed run)
    # == committed fixture, every key the fixture holds
    ks = sorted(fx0)
    missing = [k for k in ks if k not in pa or k not in pp]
    c1a = bitwise_diff(pa, fx0, [k for k in ks if k in pa])
    c1b = bitwise_diff(pp, fx0, [k for k in ks if k in pp])
    ok1 = not missing and not c1a and not c1b
    controls["C1"] = (f"{'PASS' if ok1 else 'FAIL'}: {len(ks)} committed fixture keys; plain rerun "
                      f"diffs {c1a or 'none'}; base trajectory of perturbed run diffs {c1b or 'none'}; "
                      f"missing {missing or 'none'}")
    print("C1", controls["C1"], flush=True)
    if not ok1:
        json.dump(controls, open(out_json + ".stop.json", "w"), indent=1, default=str)
        raise SystemExit("C1 FAIL: STOP (ASK)")

    # C2: two unperturbed reruns; same-seed rerun
    c2a = bitwise_diff(pa, pb, sorted(set(pa) | set(pb)) if set(pa) == set(pb) else [])
    same_keys = set(pa) == set(pb)
    c2b = {}
    for c in cases:
        for n in range(1, n_steps + 1):
            c2b.update(bitwise_diff({k: pp[f"{c}/seed0/s{n}/{k}"] for k in fields},
                                    {k: pp[f"{c}/seedrerun0/s{n}/{k}"] for k in fields}, fields))
    ok2 = same_keys and not c2a and not c2b
    controls["C2"] = (f"{'PASS' if ok2 else 'FAIL'}: plain A vs B ({len(pa)} keys) diffs "
                      f"{c2a or 'none'} (key sets equal {same_keys}); seed-0 vs seed-0 rerun "
                      f"(all cases/steps/fields) diffs {c2b or 'none'}")
    print("C2", controls["C2"], flush=True)

    # C3: boundary readback at the compiled-extension call (spy on f2py_advance_clubb_core)
    c3_fail, moved_per_seed, absent = [], {}, set()
    for c in cases:
        for n in range(1, n_steps + 1):
            for sd in list(perturb.SEEDS) + ["rerun0"]:
                r = json.loads(str(pp[f"{c}/c3/s{n}/seed{sd}"]))
                nm = 0
                for k, (ok, moved, size) in r.items():
                    if ok == "absent":
                        absent.add(k)
                    elif not ok:
                        c3_fail.append((c, n, sd, k))
                    else:
                        nm += moved
                moved_per_seed.setdefault(str(sd), []).append(nm)
                if nm == 0:
                    c3_fail.append((c, n, sd, "nothing moved"))
    ok3 = not c3_fail and not absent
    inv = json.loads(str(pp["bomex/inventory"]))
    r0 = json.loads(str(pp[f"bomex/c3/s1/seed0"]))
    n_ext = sum(1 for k in r0 if not k.endswith("(fortran)"))
    n_ftn = sum(1 for k in r0 if k.endswith("(fortran)"))
    unchecked = sorted(k for k in inv["perturbed"] if "." in k and k + "(fortran)" not in r0)
    controls["C3"] = (f"{'PASS' if ok3 else 'FAIL'}: per call, {n_ext} top-level float inputs as handed to "
                      f"the compiled f2py extension and {n_ftn} grid/nu fields read back from Fortran module "
                      f"storage after the call == intended bitwise, moved elements one adjacent step, rest "
                      f"identical; derived-type fields perturbed but not read back: {unchecked}; "
                      f"failures {c3_fail[:10] or 'none'}; "
                      f"absent {sorted(absent) or 'none'}; min moved elements per call "
                      f"{min(min(v) for v in moved_per_seed.values())}")
    controls["C3_inventory_bomex"] = inv
    print("C3", controls["C3"], flush=True)

    # spread with the test's own max_rel_errors
    orig_step = T._step
    vals = {sd: {} for sd in perturb.SEEDS}
    try:
        for c in cases:
            for sd in perturb.SEEDS:
                it = iter(range(n_steps))

                def fake_step(st, rcm, forcing, env, gr, cfg, extra=(), diag_extra=(), c=c, sd=sd, it=it, **_kw):
                    n = next(it)
                    return {k: pp[f"{c}/seed{sd}/s{n + 1}/{k}"] for k in fields}
                T._step = fake_step
                w = T.max_rel_errors(fx0, c)
                if set(w) != set(fields):
                    raise SystemExit(f"field set {sorted(w)}")
                for k, v in w.items():
                    vals[sd][f"{c}/{k}"] = v
            print("spread done", c, flush=True)
    finally:
        T._step = orig_step
    base = {k: None for k in vals[0]}
    # metric: the per-seed value is already the test's max_rel_errors output
    spread, per_seed = perturb.spread(base, vals, lambda p, b: p)

    # port rel + C4: max_rel_errors called by the committed test itself (spy) vs direct call
    port, test_seen = {}, {}
    orig_mre = T.max_rel_errors

    def spy(fx, case, cfg=T._CFG, **kw):
        w = orig_mre(fx, case, cfg, **kw)
        test_seen[case] = dict(w)
        return w
    T.max_rel_errors = spy
    try:
        for c in cases:
            T.test_core_matches_fortran_implemented(fx0, c)
    finally:
        T.max_rel_errors = orig_mre
    for c in cases:
        w = orig_mre(fx0, c, T._CFG_HARD)
        for k, v in w.items():
            port[f"{c}/{k}"] = v
    c4 = all(np.float64(test_seen[c][k]).tobytes() == np.float64(port[f"{c}/{k}"]).tobytes()
             for c in cases for k in fields)
    controls["C4"] = (f"{'PASS' if c4 else 'FAIL'}: port rel from max_rel_errors as called inside "
                      f"test_core_matches_fortran_implemented == direct call, bitwise={c4}; per-case worst "
                      + ", ".join(f"{c} {max(test_seen[c].values()):.2e}" for c in cases))
    print("C4", controls["C4"], flush=True)
    for c in cases:
        for k in fields:
            s = spread[f"{c}/{k}"]
            print(f"{c:14s} {k:10s} port {port[f'{c}/{k}']:.3e} spread "
                  f"{'None' if s is None else format(s, '.3e')}")
    json.dump(dict(spread=spread, per_seed=per_seed, port=port, controls=controls,
                   test_seen=test_seen), open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
