"""Smooth-window scales of the IFS CUMASTRN port's RULE 5 switches (#2281).

usage: python switch_scales.py [fixture.npz]

User rule (as CLOUDSC): a smooth window is 0.01 x a characteristic scale of
the SWITCHED quantity; the scale is the 2nd percentile of |x| > 1e-14 over the
oracle fixture.  This runs CUMASTRN with the exact (OpenIFS) switches on every
fixture column (arm L, the legoESM constants) and records, per quantity class,
the first operand of every IfsSmooth call: the quantity being floored, capped
or tested.  A class is identified by its width constant ``_W_<CLASS>`` (set to
a sentinel while recording; ``RG * _W_<CLASS>`` counts as the same class).
Values of lanes the scheme discards (inactive levels) are recorded too.
Prints ``_S_<CLASS> = <P2>  (n = <count>)`` per class.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics import _ifs_smooth
from legoesm.atmosphere.physics._ifs_thermo import lego_thermo, satur_qsat
from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

FIXTURE = (Path(__file__).resolve().parents[4] / "tests" / "atmosphere"
           / "oracle_fixtures" / "bechtold_ifs.npz")
GROUPS = ("L32", "L137", "STRESS")
MODULES = (M,)
_ONE_ARG = ("pos", "neg1", "abs")
_TWO_ARG = ("max", "min", "gt", "ge", "lt", "le", "on", "off")
_FLOOR = 1.0e-14


def _classes():
    """{(module, width name): class name} for every _W_<X> with a _S_<X>."""
    out = {}
    for mod in MODULES:
        for name in dir(mod):
            if name.startswith("_S_") and hasattr(mod, "_W_" + name[3:]):
                out[(mod, "_W_" + name[3:])] = name[3:]
    return out


def measure(fix, groups=GROUPS):
    """{class: |x| samples} from exact-switch CUMASTRN runs on ``groups``."""
    classes = _classes()
    th = lego_thermo()
    sentinel = {}
    saved = {}
    for i, (key, cls) in enumerate(sorted(classes.items(), key=lambda kv: kv[1])):
        s = 2.0 ** -(200 + i)
        mod, wname = key
        saved[key] = getattr(mod, wname)
        setattr(mod, wname, s)
        sentinel[s] = cls
        sentinel[th.RG * s] = cls
    store = defaultdict(list)

    def tag(w):
        return sentinel.get(w) if isinstance(w, float) else None

    def record(cls, a):
        if cls is None or isinstance(a, (int, float)):
            return
        jax.debug.callback(lambda v, c=cls: store[c].append(np.abs(np.asarray(v)).ravel()), a)

    orig = {n: getattr(_ifs_smooth.IfsSmooth, n) for n in _ONE_ARG + _TWO_ARG}

    def wrap(name):
        f = orig[name]
        if name in _ONE_ARG:
            def g(self, x, w, *a, **k):
                record(tag(w), x)
                return f(self, x, w, *a, **k)
        else:
            def g(self, x, b, w, *a, **k):
                record(tag(w), x)
                return f(self, x, b, w, *a, **k)
        return g

    try:
        for n in orig:
            setattr(_ifs_smooth.IfsSmooth, n, wrap(n))
        for g in groups:
            a = lambda k, g=g: jnp.asarray(fix[f"{g}_{k}"])  # noqa: E731
            zero = jnp.zeros_like(a("T"))
            M.cumastrn(th, float(fix["dt"]), pgaw=a("pgaw"), T=a("T"), q=a("q"), u=a("u"),
                       v=a("v"), plitot=a("lit"), omega=a("omega"),
                       qsat=satur_qsat(th, a("T"), a("p_full")), ahfs=a("ahfs"),
                       qhfl=a("qhfl"), p_full=a("p_full"), p_half=a("p_half"),
                       geo_full=a("geo_full"), geo_half=a("geo_half"), land=a("land"),
                       ptent=a("dT_other"), ptenq=a("dq_other"), ptenu=zero, ptenv=zero,
                       ptenta=a("dT_adv"), ptenqa=a("dq_adv"))
        jax.effects_barrier()
    finally:
        for n, f in orig.items():
            setattr(_ifs_smooth.IfsSmooth, n, f)
        for (mod, wname), v in saved.items():
            setattr(mod, wname, v)
    return {c: np.concatenate(v) for c, v in store.items()}


def p2(samples):
    """{class: (2nd percentile of |x| > 1e-14, count)}."""
    out = {}
    for c, v in samples.items():
        v = v[np.isfinite(v) & (v > _FLOOR)]
        out[c] = (float(np.percentile(v, 2.0)) if v.size else float("nan"), int(v.size))
    return out


def main(argv):
    jax.config.update("jax_enable_x64", True)   # the oracle fixture is fp64
    fix = dict(np.load(argv[1] if len(argv) > 1 else FIXTURE))
    for c, (s, n) in sorted(p2(measure(fix)).items()):
        print(f"_S_{c} = {s:.3g}  (n = {n})")


if __name__ == "__main__":
    main(sys.argv)
