"""Shared instrument: empirical round-off spread of an oracle (20 seeds).

perturb_inputs(inputs, seed): every floating-point array/scalar in `inputs` (dict name ->
value) moves by exactly one ulp of its OWN dtype, toward +inf or -inf, with an independent
fair-coin sign per element.  Exact zeros, NaN and inf are left unchanged (zero structure is
held fixed).  Integer, bool and string inputs are returned unchanged.

spread(base, runs, metric): per field, max over seeds of metric(perturbed, base) where
`metric` is the committed test's own rel function (perturbed in the "ours" slot, the
unperturbed oracle in the "oracle" slot).  No safety factor.
"""
from __future__ import annotations

import json
import numpy as np

SEEDS = tuple(range(20))
LABEL = "empirical round-off spread (20 seeds)"
REQUIRED_CONTROLS = ("C1", "C2", "C3", "C4", "C5")
ROUNDOFF = 1.0e-12   # class A = rel <= 1e-12 OR rel <= the computed spread


def _ulp_step(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    up = rng.random(x.shape) < 0.5
    target = np.where(up, np.inf, -np.inf).astype(x.dtype)
    moved = np.nextafter(x, target)
    keep = (x == 0) | ~np.isfinite(x)
    out = np.where(keep, x, moved).astype(x.dtype)
    if not np.array_equal(np.isfinite(out), np.isfinite(x)):
        raise ValueError("1-ulp step overflowed a finite input to inf: ASK")
    if np.any((out == 0) & (x != 0)):
        raise ValueError("1-ulp step turned a subnormal into an exact zero: ASK")
    return out


def perturb_inputs(inputs: dict, seed: int) -> dict:
    """One ulp = the ADJACENT representable value (np.nextafter), so the step below a power
    of two is half the step above it."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in sorted(inputs):           # sorted: draw order independent of dict order
        v = inputs[name]
        a = np.asarray(v)
        if a.dtype == object or a.dtype.names is not None:
            raise TypeError(f"object/structured input {name!r}: flatten it first; ASK if unsure")
        if np.issubdtype(a.dtype, np.complexfloating):
            raise TypeError(f"complex input {name!r}: not covered by the approved method; ASK")
        if np.issubdtype(a.dtype, np.floating):
            p = _ulp_step(a, rng)
            if isinstance(v, np.ndarray):
                out[name] = p
            elif isinstance(v, np.generic):
                out[name] = a.dtype.type(p)          # keep the NumPy scalar type
            else:
                out[name] = p.item() if a.ndim == 0 else p
        else:
            out[name] = np.copy(v) if isinstance(v, np.ndarray) else v   # no aliasing
    return out


def spread(base: dict, runs: dict, metric) -> tuple[dict, dict]:
    """runs: seed -> outputs, exactly SEEDS, each with exactly base's fields.
    field -> max over seeds of metric(run[field], base[field]), and the per-seed values.
    A non-finite metric value makes the field's spread None (UNMEASURED), never a bound."""
    if tuple(sorted(runs)) != SEEDS:
        raise ValueError(f"runs must be exactly seeds {SEEDS}, got {sorted(runs)}")
    if not base:
        raise ValueError("empty base")
    for sd, r in runs.items():
        if set(r) != set(base):
            raise ValueError(f"seed {sd}: fields {sorted(set(r) ^ set(base))} differ from base")
    res, per_seed = {}, {}
    for k in base:
        vals = [float(metric(runs[sd][k], base[k])) for sd in SEEDS]
        per_seed[k] = vals
        res[k] = max(vals) if all(np.isfinite(v) for v in vals) else None
    return res, per_seed


def classify(port_rel: float | None, bound: float | None) -> str:
    if port_rel is None or not np.isfinite(port_rel):
        return "C"
    if port_rel < 0 or (bound is not None and bound < 0):
        raise ValueError(f"negative rel/bound: {port_rel}, {bound}")
    if port_rel <= ROUNDOFF:                 # user rule: the spread only EXTENDS 1e-12
        return "A"
    if bound is None or not np.isfinite(bound):
        return "UNMEASURED (non-finite spread)"
    return "A" if port_rel <= bound else "C"


def _enc(x):
    if isinstance(x, float) and not np.isfinite(x):
        return repr(x)                       # "nan"/"inf" as strings: strict JSON
    if isinstance(x, dict):
        return {k: _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    return x


def artifact(path, *, family, unit, fields_spread, per_seed, port_rel, compiler, controls,
             seeds=SEEDS, cases, inputs_perturbed, inputs_untouched, notes=""):
    if tuple(seeds) != SEEDS or not cases or not fields_spread:
        raise ValueError("artifact needs seeds 0..19, non-empty cases and fields")
    if set(port_rel) != set(fields_spread) or set(per_seed) != set(fields_spread):
        raise ValueError(f"port/spread/per_seed field sets differ")
    for k, vals in per_seed.items():
        if len(vals) != len(SEEDS):
            raise ValueError(f"{k}: per_seed has {len(vals)} values, not 20")
        want = max(vals) if all(np.isfinite(v) for v in vals) else None
        if want != fields_spread[k]:
            raise ValueError(f"{k}: spread {fields_spread[k]} != max of per_seed {want}")
    if set(controls) != set(REQUIRED_CONTROLS):
        raise ValueError(f"controls must be exactly {REQUIRED_CONTROLS}, got {sorted(controls)}")
    ok = all(controls[c] == "PASS" for c in REQUIRED_CONTROLS)   # exact; details go in notes
    verdict = {k: (classify(port_rel[k], fields_spread[k]) if ok else "CONTROLS FAILED")
               for k in fields_spread}
    json.dump(_enc(dict(label=LABEL, family=family, unit=unit, seeds=list(seeds),
                   zero_structure="held fixed (exact zeros, NaN, inf not perturbed)",
                   compiler=compiler, cases=cases, inputs_perturbed=inputs_perturbed,
                   inputs_untouched=inputs_untouched, spread=fields_spread,
                   per_seed=per_seed, controls=controls,
                   port_vs_oracle_rel=port_rel, field_class=verdict, notes=notes)),
              open(path, "w"), indent=1, sort_keys=True, allow_nan=False)
    return verdict


if __name__ == "__main__":
    x = {"a": np.array([1.0, 0.0, -2.5, np.nan, 1e-310]), "f32": np.float32([3.0]),
         "s": 2.0, "n": np.array([1, 2])}
    p = perturb_inputs(x, 0)
    d = np.abs(p["a"] - x["a"])
    assert d[1] == 0 and np.isnan(p["a"][3])                    # zero, NaN fixed
    assert d[0] == np.spacing(1.0) or d[0] == np.spacing(1.0) / 2  # one ulp (below 1.0 is half)
    assert p["a"][4] != x["a"][4]                                 # subnormal moves
    assert p["f32"].dtype == np.float32 and p["f32"][0] != 3.0
    assert p["n"] is not x["n"] and np.array_equal(p["n"], x["n"]) and p["s"] != 2.0
    assert perturb_inputs(x, 0)["s"] == p["s"]                    # deterministic per seed
    signs = np.sign(perturb_inputs({"z": np.ones(10000)}, 1)["z"] - 1.0)
    assert abs(signs.mean()) < 0.05                               # fair coin
    m = lambda a, b: np.max(np.abs(a - b)) / np.max(np.abs(b))  # noqa: E731
    runs = {sd: {"f": np.ones(3) * (1 + sd)} for sd in SEEDS}
    runs[3] = {"f": np.array([1, np.nan, 1.0])}
    sp, ps = spread({"f": np.ones(3)}, runs, m)
    assert sp["f"] is None and classify(1.0, sp["f"]).startswith("UNMEASURED")
    assert classify(1e-13, 1e-12) == "A" and classify(2e-12, 1e-12) == "C"
    assert classify(5e-13, 0.0) == "A" and classify(2e-12, 0.0) == "C"   # spread never tightens
    assert classify(5e-13, None) == "A"
    assert classify(None, 1e-12) == "C" and classify(None, None) == "C"
    assert classify(1.0, float("inf")).startswith("UNMEASURED")
    import tempfile, os
    ps20 = {"f": [1e-13] * 20}
    with tempfile.TemporaryDirectory() as td:
        v = artifact(os.path.join(td, "a.json"), family="t", unit="t", fields_spread={"f": 1e-13},
                     per_seed=ps20, port_rel={"f": 5e-14}, compiler="x", controls=dict.fromkeys(REQUIRED_CONTROLS, "PASS"),
                     cases=["c"], inputs_perturbed=["a"], inputs_untouched=[])
        assert v == {"f": "A"}
        v = artifact(os.path.join(td, "b.json"), family="t", unit="t", fields_spread={"f": 1e-13},
                     per_seed=ps20, port_rel={"f": 5e-14}, compiler="x", controls={**dict.fromkeys(REQUIRED_CONTROLS, "PASS"), "C3": "PASSING? NO"},
                     cases=["c"], inputs_perturbed=["a"], inputs_untouched=[])
        assert v == {"f": "CONTROLS FAILED"}
        for bad in (dict(seeds=[0]), dict(fields_spread={"f": 2e-13}), dict(controls={"x": "PASS"})):
            kw = dict(family="t", unit="t", fields_spread={"f": 1e-13}, per_seed=ps20,
                      port_rel={"f": 0.0}, compiler="x", controls=dict.fromkeys(REQUIRED_CONTROLS, "PASS"), cases=["c"],
                      inputs_perturbed=[], inputs_untouched=[]); kw.update(bad)
            try:
                artifact(os.path.join(td, "c.json"), **kw); raise AssertionError
            except ValueError:
                pass
    runs[3] = {"f": np.ones(3)}
    assert spread({"f": np.ones(3)}, runs, m)[0]["f"] == 19.0
    for bad in ({sd: runs[sd] for sd in SEEDS[:19]}, {**runs, 3: {"g": np.ones(3)}}):
        try:
            spread({"f": np.ones(3)}, bad, m); raise AssertionError
        except ValueError:
            pass
    assert isinstance(perturb_inputs({"q": np.float32(2.0)}, 0)["q"], np.float32)
    try:
        perturb_inputs({"t": np.array([5e-324] * 64)}, 0); raise AssertionError
    except ValueError:
        pass
    try:
        perturb_inputs({"c": np.array([1 + 1j])}, 0); raise AssertionError
    except TypeError:
        pass
    try:
        perturb_inputs({"m": np.array([np.finfo(np.float64).max] * 64)}, 0); raise AssertionError
    except ValueError:
        pass
    print("perturb self-check OK")
