"""End-to-end check of the fp64 twin-snapshot artifact contract, and the
storage-precision RESOLUTION demonstration.

Verification instrument for the fp64-snapshot change in
:mod:`kamm_twin_90d` (the ``--fp64-3d`` flag plus the always-on fp64 reduced
series).  It is committed rather than pasted into a shell because a number
quoted from an uncommitted heredoc cannot be re-run against a changed model
and hides its own bugs.

It does three things, and the third is the point:

  1. CONTRACT.  Every key the change promises is present, with the promised
     dtype, shape and finiteness, and the per-field storage stamp says what
     was actually written.

  2. IDENTITY.  The stored fp64 reduced series must equal, EXACTLY, what the
     recorded scorers (``verdict360.all_metrics``, which is
     ``acceptance_gate_90d.metrics`` plus the six transport reductions)
     compute from the same snapshot.  If it does not, the series is a second
     spelling of eleven reductions rather than the same quantity, which is
     this campaign's most expensive defect class.

  3. RESOLUTION, as a CONTROLLED comparison with exactly ONE variable -- the
     STORAGE precision.  Both members are read from the same ``--fp64-3d``
     artifacts, so the two trajectories are FIXED; the only thing that changes
     between the two printed columns is whether the 3-D block is round-tripped
     through float32 before being reduced.  A metric that reads
     ``|m1 - m2| == 0`` in the float32 column and nonzero in the float64
     column was UNMEASURABLE purely because of the npz dtype.

     The comparison is refused unless the artifacts really were written at
     float64 -- on a float32 artifact the "float32 column" would not be a
     controlled arm, it would be the artifact itself.

Usage
-----
    python fp64_snapshot_contract_check.py <m1.npz> <m2.npz> [--day 5]
    python fp64_snapshot_contract_check.py --self-check
"""
import argparse
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import acc_thermal_wind as A          # noqa: E402  mesh + masks
import kamm_twin_90d as K             # noqa: E402  the stamp reader + key list
import verdict360 as V                # noqa: E402  the recorded reductions


def contract(path, day):
    """Assert the artifact contract; return the parsed storage stamp."""
    d = np.load(path)
    sto = K.snapshot_storage_dtypes(d)
    print(f"\n=== CONTRACT {path}")
    print(f"  storage_dtypes        {json.dumps(sto, sort_keys=True)}")
    print(f"  control_dtype         {str(d['control_dtype'])}  "
          f"(the precision the arm was BUILT at -- a different thing)")
    status = str(d["snapshot_reduction_status"])
    print(f"  snapshot_reduction_status {status}")
    if status != "ok":
        raise SystemExit(f"{path} carries no reduced series: {status}")
    for f in ("T3d", "S3d", "eta3d", "u3d", "v3d"):
        key = f"{f}_day{day}"
        if key not in d.files:
            raise SystemExit(f"{path} has no {key}")
        if d[key].dtype.name != sto[f]:
            raise SystemExit(f"{key} is {d[key].dtype.name} but the stamp "
                             f"says {sto[f]} -- the stamp is not honest")
    days = list(d["reduced_days"])
    print(f"  reduced_days          {days}")
    if day not in days:
        raise SystemExit(f"{path} has no reduced series on day {day}")
    for k in K.REDUCED_KEYS:
        a = d[f"reduced_{k}"]
        if a.dtype != np.float64 or a.shape != (len(days),):
            raise SystemExit(f"reduced_{k} is {a.dtype} {a.shape}")
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"reduced_{k} is not finite")
    rows = d[f"reduced_{K.REDUCED_ROW_KEY}"]
    if rows.dtype != np.float64 or rows.shape != (len(days), A.NY):
        raise SystemExit(f"the per-row profile is {rows.dtype} {rows.shape}")
    if not np.all(np.isfinite(rows)):
        raise SystemExit("the per-row profile is not finite")
    print(f"  reduced_* keys        {len(K.REDUCED_KEYS)} scalars, all float64 "
          f"and finite; per-row profile {rows.shape}")
    print(f"  artifact size         {os.path.getsize(path) / 1e6:.1f} MB")
    return sto


def metrics_from_storage(path, day, cast):
    """The recorded scorers' metrics, reduced from the STORED 3-D block, with
    the block optionally round-tripped through ``cast`` first.  ``cast=None``
    is the block exactly as written."""
    d = np.load(path)

    def f(name):
        a = d[f"{name}_day{day}"]
        return (a if cast is None else a.astype(cast)).astype(np.float64)

    u = f("u3d")[:, 1:A.NX + 1, :]
    mask = d["land_mask"].astype(np.float64)
    st = {"T": f("T3d"), "S": f("S3d"), "u": u, "land_mask": mask}
    return V.all_metrics(st, A.tmask & (mask > 0.5)[:, :, None])


def stored_series(path, day):
    d = np.load(path)
    i = list(d["reduced_days"]).index(day)
    return {k: float(d[f"reduced_{k}"][i]) for k in K.REDUCED_KEYS}


def resolution_table(m1, m2, day):
    """``(rows, n_gained)`` -- |member1 - member2| per metric, reduced from
    float32-round-tripped storage and from the stored fp64 series.

    ``n_gained`` counts ONLY metrics the fp64 series RESOLVES and float32
    storage TIES.  A metric that ties in both columns is not a gain -- it is a
    metric the perturbation never reached -- and counting it would make this
    instrument report a win for doing nothing.  (It did, in its first run:
    perturbing only ``u`` left the four density metrics tied at both
    precisions and they were scored as gains.  Corrected here.)
    """
    f32 = [metrics_from_storage(q, day, np.float32) for q in (m1, m2)]
    f64 = [stored_series(q, day) for q in (m1, m2)]
    rows, gained = [], 0
    for k in K.REDUCED_KEYS:
        d32 = abs(f32[0][k] - f32[1][k])
        d64 = abs(f64[0][k] - f64[1][k])
        gained += (d32 == 0.0 and d64 > 0.0)
        rows.append((k, d32, d64))
    return rows, gained


def report(m1, m2, day):
    # BOTH members are checked, and they must AGREE. Reading the loop variable
    # after the loop checked only the SECOND member, so a mixed fp32/fp64 pair
    # passed the refusal below and the "one variable" claim was void -- the
    # first member's float32 column would have been the artifact itself rather
    # than a controlled arm (both reviews).
    stos = [contract(q, day) for q in (m1, m2)]
    fields = ("T3d", "S3d", "u3d")
    got = {tuple(st.get(f, "float32") for f in fields) for st in stos}
    if len(got) != 1:
        raise SystemExit(
            f"the two members stored their 3-D blocks at different precisions "
            f"{sorted(got)} -- the comparison below would then have TWO "
            f"variables, not one")
    if got.pop() != ("float64",) * len(fields):
        raise SystemExit(
            "these artifacts were NOT written with --fp64-3d, so the float32 "
            "column below would not be a controlled arm -- it would be the "
            "artifact itself. Re-run the pair with --fp64-3d.")

    print(f"\n=== IDENTITY at day {day}: the stored fp64 series against the "
          f"recorded scorers reducing the same block")
    worst = 0.0
    for q in (m1, m2):
        got, want = stored_series(q, day), metrics_from_storage(q, day, None)
        for k in K.REDUCED_KEYS:
            worst = max(worst, abs(got[k] - want[k]))
    print(f"  max |stored - recomputed| over both members x "
          f"{len(K.REDUCED_KEYS)} metrics: {worst:.3e}")
    if worst != 0.0:
        raise SystemExit("the stored series is NOT the scorers' own quantity")

    print(f"\n=== RESOLUTION at day {day}: |member1 - member2|, the SAME two "
          f"trajectories, storage precision the only variable")
    print(f"  {'metric':<38}{'fp32-stored':>15}{'fp64 series':>15}   verdict")
    rows, gained = resolution_table(m1, m2, day)
    for k, d32, d64 in rows:
        note = ("TIE at fp32, RESOLVED at fp64" if d32 == 0.0 and d64 > 0.0
                else "tie at BOTH" if d32 == 0.0 else "resolved at both")
        print(f"  {V.LABELS[k]:<38}{d32:>15.6e}{d64:>15.6e}   {note}")
    d1, d2 = np.load(m1), np.load(m2)
    for f in ("T3d", "u3d"):
        x1, x2 = d1[f"{f}_day{day}"], d2[f"{f}_day{day}"]
        n64 = int((x1 != x2).sum())
        n32 = int((x1.astype(np.float32) != x2.astype(np.float32)).sum())
        print(f"  {f} cells differing between the members: {n32} of {x1.size} "
              f"at float32 storage, {n64} at float64")
    print(f"\n  {gained} of {len(K.REDUCED_KEYS)} metrics are UNMEASURABLE "
          f"from float32 storage on this pair AND measurable from the fp64 "
          f"series (a metric tied in BOTH columns is not counted -- the "
          f"perturbation never reached it).")
    return gained


def _self_check():
    """Drive the whole report on two SYNTHETIC artifacts whose difference is
    planted strictly BELOW the float32 quantum, so the instrument is exercised
    end to end without a GPU run -- and so a change that broke the float32
    round-trip (making the two columns identical, i.e. the demonstration
    vacuous) fails here rather than in a report."""
    import tempfile
    rng = np.random.default_rng(0)
    ny, nx, nz = A.tmask.shape
    z = np.arange(nz, dtype=np.float64)
    # Both members are built on EXACTLY-representable float32 values, so
    # adding a quarter of an ulp is GUARANTEED to round back to the same
    # float32 under round-to-nearest.  Perturbing an arbitrary float64 by
    # "0.4 ulp" does not guarantee that -- the first version of this
    # self-check did exactly that and 152005 of 379692 stored faces differed,
    # i.e. the perturbation was not sub-quantum at all.
    def f32_base(x):
        return np.asarray(x, dtype=np.float32).astype(np.float64)

    base = {
        "T": f32_base(4.0 + 16.0 * np.exp(-z / 6.0)[None, None, :]
                      + 0.05 * rng.standard_normal((ny, nx, nz))),
        "S": f32_base(34.5 + 0.5 * np.exp(-z / 10.0)[None, None, :]
                      + 0.01 * rng.standard_normal((ny, nx, nz))),
        "u": f32_base(0.05 * rng.standard_normal((ny, nx + 1, nz))),
    }
    mask = np.asarray(A.tmask[:, :, 0], dtype=np.float64)
    day = 5
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            # Perturb EVERY field, each cell by a quarter of its own float32
            # ulp: invisible to float32 storage, visible to an fp64 integral
            # over ~370k cells.  All three are perturbed because the density
            # metrics read T/S and the transports read u -- perturbing only u
            # leaves four of the eleven metrics untouched and unmeasurable by
            # construction rather than by precision.
            st = {k: (v + 0.25 * np.spacing(v.astype(np.float32)
                                            ).astype(np.float64) if i else v)
                  for k, v in base.items()}
            red, status = K.build_snapshot_reducer(os.path.dirname(
                A.mm.filepath()))
            if red is None:
                raise SystemExit(f"self-check needs the NEMO mesh: {status}")
            m = red({**st, "land_mask": mask})
            kw = {f"{n}3d_day{day}": st[k].astype(np.float64)
                  for n, k in (("T", "T"), ("S", "S"), ("u", "u"))}
            kw[f"eta3d_day{day}"] = np.zeros((ny, nx), dtype=np.float64)
            kw[f"v3d_day{day}"] = np.zeros((ny + 1, nx, nz), dtype=np.float64)
            kw["land_mask"] = mask.astype(np.float64)
            kw["control_dtype"] = np.str_("float64")
            kw["snapshot_reduction_status"] = np.str_(status)
            # THE HARNESS'S OWN STAMP, not a hand-written copy: a second
            # spelling here could not notice a change to the real one, which
            # is precisely what this instrument exists to check (code review).
            kw["storage_dtypes"] = np.str_(K.storage_stamp("float64",
                                                           "float64"))
            # the harness's own series assembly, for the same reason
            kw.update(K.reduced_series_kwargs({day: m}, [day]))
            q = os.path.join(td, f"m{i}.npz")
            np.savez(q, **kw)
            paths.append(q)
        # the storage MUST be blind to the perturbation, or the demonstration
        # below is not about precision at all
        a0, a1 = np.load(paths[0]), np.load(paths[1])
        for f in ("T3d", "S3d", "u3d"):
            n = int((a0[f"{f}_day{day}"].astype(np.float32)
                     != a1[f"{f}_day{day}"].astype(np.float32)).sum())
            if n:
                raise SystemExit(
                    f"SELF-CHECK FAILED: the planted perturbation moved {n} "
                    f"{f} cells at float32 storage, so it is NOT sub-quantum "
                    f"and this instrument would be demonstrating nothing")
        gained = report(paths[0], paths[1], day)
    if gained == 0:
        raise SystemExit(
            "SELF-CHECK FAILED: the planted sub-quantum perturbation was "
            "resolved even from float32 storage, so this instrument cannot "
            "demonstrate what it claims to")
    print("\n  CAVEAT on the numbers above, because they are synthetic: the "
          "planted\n  perturbation moves every cell in the SAME direction "
          "(away from zero), which is\n  a COHERENT bias. A real ensemble's "
          "sub-quantum differences are random-signed\n  and partly cancel, so "
          "this table proves the MECHANISM, not the magnitude of\n  any spread "
          "a real ensemble will show. Quote a real pair for that (physics "
          "review).")
    print(f"\nSELF-CHECK OK: the contract, the exact identity against the "
          f"recorded scorers, a perturbation VERIFIED invisible to float32 "
          f"storage on every field, and {gained} metric(s) that float32 "
          f"storage ties and the fp64 series resolves.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("m1", nargs="?")
    p.add_argument("m2", nargs="?")
    p.add_argument("--day", type=int, default=5)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    if a.self_check:
        return _self_check()
    if not (a.m1 and a.m2):
        raise SystemExit("need two member npz paths, or --self-check")
    gained = report(a.m1, a.m2, a.day)
    if gained == 0:
        # Exit-zero on a run that resolved nothing would be read as a
        # demonstration (code review). It is not a failure of the code -- the
        # pair may simply have separated at both precisions -- so it is a
        # distinct non-zero status with the reason printed.
        print("\nNOTE: no metric on this pair was tied at float32 and "
              "resolved at float64, so this run demonstrates nothing about "
              "storage precision (it is not necessarily a defect: the members "
              "may separate at both precisions).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
