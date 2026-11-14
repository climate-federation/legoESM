"""Per-site RULE 5 gradient points for the IFS CUMASTRN port (#2281).

usage: python switch_sites.py [fixture.npz]

Every smooth switch in :mod:`legoesm.atmosphere.physics.convection._ifs_cumastrn`
is a call of an :class:`~legoesm.atmosphere.physics._ifs_smooth.IfsSmooth`
method (``sel`` blends are not switches).  A SITE is identified by the stripped
source line of the call and the method name.  For each site this searches,
over the oracle fixture's columns and a set of one-parameter input
perturbations, for a point where the site is inside its smooth window AND the
model output changes when only that site is made exact; it prints the points
as a python literal (``SITE_POINTS``) for the per-site gradient test.

The windows are 0.01 x a 2nd-percentile scale, so a grid never lands in one:
the search brackets a sign change of the site's normalised position on a
coarse grid of the perturbation and bisects to the window's middle.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import linecache
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics import _ifs_smooth
from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

FIXTURE = (Path(__file__).resolve().parents[4] / "tests" / "atmosphere"
           / "oracle_fixtures" / "bechtold_ifs.npz")
GROUPS = ("L32", "L137", "STRESS")
DX = 1.0e5   # m, the grid length make_fixture.py encoded in PGAW
MODULES = (M,)
# one-sided windows: active for 0 < (a - b)/w < 1 (target 0.5); centred: |d| < 0.5
_ONE_SIDED = ("pos", "max", "min", "on", "off", "neg1")
_CENTRED = ("gt", "ge", "lt", "le")
_SITE_METHODS = _ONE_SIDED + _CENTRED
# (name, perturbed input, transform, range): one scalar each
DIRECTIONS = (
    ("tend", lambda x, s: {**x, "dT_other": s * x["dT_other"], "dq_other": s * x["dq_other"]}, (0.0, 4.0)),
    ("sflux", lambda x, s: {**x, "shf_up": s * x["shf_up"], "lhf_up": s * x["lhf_up"]}, (0.0, 4.0)),
    ("adv", lambda x, s: {**x, "dT_adv": s * x["dT_adv"], "dq_adv": s * x["dq_adv"]}, (-2.0, 4.0)),
    ("q", lambda x, s: {**x, "q": s * x["q"]}, (0.7, 1.3)),
    ("T", lambda x, s: {**x, "T": x["T"] + s}, (-4.0, 4.0)),
    ("wind", lambda x, s: {**x, "u": s * x["u"], "v": s * x["v"]}, (0.0, 4.0)),
    ("omega", lambda x, s: {**x, "omega": x["omega"] + s}, (-150.0, 50.0)),
    ("omega_moist", lambda x, s: {**x, "omega": x["omega"] + s, "q": 1.25 * x["q"]}, (-150.0, 50.0)),
)
_KEYS = ("T", "q", "p_full", "p_half", "u", "v", "shf_up", "lhf_up", "land", "dT_adv",
         "dq_adv", "dT_other", "dq_other", "omega")


def sites():
    """{(stripped line, method)} of every IfsSmooth switch call in MODULES (AST)."""
    out, n = set(), 0
    for mod in MODULES:
        src = inspect.getsource(mod)
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "sm"
                    and node.func.attr in _SITE_METHODS):
                out.add((lines[node.lineno - 1].strip(), node.func.attr))
                n += 1
    if n != len(out):
        raise ValueError(f"{n} switch calls but {len(out)} site keys: put each "
                         "same-method call on its own source line")
    return out


def _mid(name, b, w):
    """The switched operand's value at the middle of its window."""
    if name in _CENTRED:
        return b
    if name == "pos":
        return 0.5 * w
    if name == "neg1":
        return -0.5 * w
    return b + 0.5 * w


@contextlib.contextmanager
def instrument(record=None, hard_sites=(), inject=None):
    """Patch IfsSmooth: ``record(site, d)`` gets every call's normalised
    window position d (in window: 0 < d < 1 one-sided, |d| < 0.5 centred);
    calls from ``hard_sites`` take the exact form; ``inject = {site: s}``
    moves each such site's switched operand to ``mid + (s - 1) w`` (the
    offset under stop_gradient, so the operand keeps its upstream
    derivative): at s = 1 every evaluation of the site sits mid-window, and
    s spans the window in units of 1."""
    inject = dict(inject or {})
    files = {inspect.getsourcefile(m) for m in MODULES}
    hard_sites = set(hard_sites)
    hard = _ifs_smooth.IfsSmooth(True)
    orig = {n: getattr(_ifs_smooth.IfsSmooth, n) for n in _SITE_METHODS}

    def wrap(name):
        f = orig[name]

        def g(self, *args, **kw):
            fr = sys._getframe(1)
            site = None
            if fr.f_code.co_filename in files:
                site = (linecache.getline(fr.f_code.co_filename, fr.f_lineno).strip(), name)
            if site in inject and not self.hard:
                one = name in ("pos", "neg1")
                a, w = args[0], args[1] if one else args[2]
                tgt = _mid(name, None if one else args[1], w)
                a = a + jax.lax.stop_gradient(tgt - a) + (inject[site] - 1.0) * w
                args = (a,) + tuple(args[1:])
            if site is not None and not self.hard:
                if name in ("pos", "neg1"):
                    x, w = args[0], args[1]
                    d = (-x if name == "neg1" else x) / w
                else:
                    a, b, w = args[0], args[1], args[2]
                    d = (b - a) / w if name in ("lt", "le") else (a - b) / w
                    if name == "off":
                        d = (a - b) / w
                if record is not None:
                    jax.debug.callback(lambda v, s=site: record(s, np.asarray(v)), d)
                if site in hard_sites:
                    return f(hard, *args, **kw)
            return f(self, *args, **kw)
        return g

    try:
        for n in orig:
            setattr(_ifs_smooth.IfsSmooth, n, wrap(n))
        yield
    finally:
        for n, f in orig.items():
            setattr(_ifs_smooth.IfsSmooth, n, f)


def column(fix, g, c):
    """The wrapper inputs of fixture column c of group g (batch of one)."""
    return {k: jnp.asarray(fix[f"{g}_{k}"])[c:c + 1] for k in _KEYS}


def output(fix, x, hard=False):
    """Scalar the gradient tests differentiate: heating, drying, mass flux."""
    from legoesm.atmosphere.physics.convection._ifs_faithful import ifs_faithful_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput

    cfg = BechtoldConfig(use_ifs_ascent=True, dx_m=DX, ifs_hard_switches=hard)
    out, mu = ifs_faithful_convection(
        x["T"], x["q"], x["p_full"], x["p_half"], x["u"], x["v"], float(fix["dt"]), cfg,
        x["shf_up"], x["lhf_up"], x["land"], x["dT_adv"], x["dq_adv"], x["dT_other"],
        x["dq_other"], x["omega"], ConvectionOutput)
    return jnp.sum(out.dT_dt) * 1.0e4 + jnp.sum(out.dq_v_dt) * 1.0e7 + jnp.sum(mu) * 1.0e1


class _Line:
    """One (column, direction) line: jitted output and per-site positions."""

    def __init__(self, fix, g, c, direction):
        _, f, self.range = next(d for d in DIRECTIONS if d[0] == direction)
        x0 = column(fix, g, c)
        self.rec = {}
        with instrument(lambda s, d: self.rec.setdefault(s, []).append(d.ravel())):
            self._f = jax.jit(lambda s: output(fix, f(x0, s)))
            self._f(jnp.asarray(float(0.5 * sum(self.range)))).block_until_ready()
        self.rec.clear()
        self.fix, self.x = fix, lambda s: f(x0, jnp.asarray(float(s)))

    def positions(self, s):
        self.rec.clear()
        # python float: a numpy scalar would retrace outside the instrumentation
        self._f(jnp.asarray(float(s))).block_until_ready()
        jax.effects_barrier()
        if not self.rec:
            raise RuntimeError("no switch site recorded: retraced outside instrument()")
        return {k: np.concatenate(v) for k, v in self.rec.items()}

    def changes_when_hard(self, site, s):
        with instrument(hard_sites=[site]):
            vh = float(output(self.fix, self.x(s)))
        return vh != float(output(self.fix, self.x(s)))


def in_window(site, d):
    return (np.abs(d) < 0.5) if site[1] in _CENTRED else ((d > 0.0) & (d < 1.0))


def _target(site):
    return 0.0 if site[1] in _CENTRED else 0.5


def point_value(fix, g, c, direction, s):
    _, f, _ = next(d for d in DIRECTIONS if d[0] == direction)
    return f(column(fix, g, c), jnp.asarray(s))


def _bisect(line, site, e, a, b, fa, n=80):
    for _ in range(n):
        m = 0.5 * (a + b)
        dm = line.positions(m).get(site)
        if dm is None or e >= dm.size:
            return None
        if in_window(site, dm[e:e + 1])[0]:
            return m
        fm = dm[e] - _target(site)
        a, b, fa = (m, b, fm) if np.sign(fm) == np.sign(fa) else (a, m, fa)
    return None


def _fd_step(line, site, e, s):
    """A finite-difference step 1e-4 x the window's extent in the perturbation
    (the window spans one unit of d), so check_grads resolves the ramp."""
    h = max(abs(s), 1.0) * 1.0e-9
    dd = (line.positions(s + h)[site][e] - line.positions(s - h)[site][e]) / (2.0 * h)
    return float(1.0e-4 / abs(dd)) if dd != 0.0 and np.isfinite(dd) else float(h)


def search(fix, todo, cols, n_grid=24):
    """{site: (g, c, direction, s, fd_step)} for the sites of ``todo`` found:
    the site in its window and the output changed by making only that site
    exact."""
    found = {}
    for g, c in cols:
        for name, _, (lo, hi) in DIRECTIONS:
            left = [t for t in todo if t not in found]
            if not left:
                return found
            line = _Line(fix, g, c, name)
            grid = np.linspace(lo, hi, n_grid)
            pos = [line.positions(s) for s in grid]
            for site in left:
                done = False
                for i in range(n_grid - 1):
                    d0, d1 = pos[i].get(site), pos[i + 1].get(site)
                    if d0 is None or d1 is None or d0.shape != d1.shape:
                        continue
                    f0, f1 = d0 - _target(site), d1 - _target(site)
                    for e in np.nonzero(np.sign(f0) != np.sign(f1))[0]:
                        m = _bisect(line, site, e, grid[i], grid[i + 1], f0[e])
                        if m is not None and line.changes_when_hard(site, m):
                            found[site] = (g, int(c), name, float(m),
                                           _fd_step(line, site, e, m))
                            done = True
                            break
                    if done:
                        break
            del line
            jax.clear_caches()
    return found


def inject_columns(fix, todo, cols):
    """{site: (g, c)}: the first column where the injected site (s = 1)
    changes the output when only that site is made exact."""
    out = {}
    for site in todo:
        for g, c in cols:
            x = column(fix, g, c)
            with instrument(inject={site: jnp.asarray(1.0)}):
                v = float(output(fix, x))
            with instrument(inject={site: jnp.asarray(1.0)}, hard_sites=[site]):
                vh = float(output(fix, x))
            jax.clear_caches()
            if np.isfinite(v) and v != vh:
                out[site] = (g, int(c))
                break
    return out


def main(argv):
    """``switch_sites.py [fixture] [--col G:C --state found.json]``: with
    --col, search that column only for the sites not yet in the state file
    and update it (one process per column: XLA's CPU JIT runs out of code
    memory after a few hundred compilations in one process)."""
    jax.config.update("jax_enable_x64", True)   # the oracle fixture is fp64
    import json

    args = argv[1:]
    opt = dict(zip(args[::2], args[1::2])) if args and args[0].startswith("--") else {}
    fix_path = args[0] if args and not args[0].startswith("--") else FIXTURE
    if not args or not args[0].startswith("--"):
        opt = dict(zip(args[1::2], args[2::2]))
    fix = dict(np.load(fix_path))
    todo = sorted(sites())
    if "--col" in opt:
        g, c = opt["--col"].split(":")
        state = Path(opt["--state"])
        found = {tuple(json.loads(k)): tuple(v) for k, v in
                 (json.loads(state.read_text()).items() if state.exists() else ())}
        found.update(search(fix, [t for t in todo if t not in found], [(g, int(c))]))
        state.write_text(json.dumps({json.dumps(k): v for k, v in found.items()}))
        print(g, c, "found", len(found), "of", len(todo), flush=True)
        return
    cols = [(g, c) for g in GROUPS for c in range(fix[f"{g}_T"].shape[0])
            if fix[f"L0_{g}_LDCUM"][c] > 0]
    found = search(fix, todo, cols)
    print("SITE_POINTS =", repr(found))
    print("UNREACHED =", repr([t for t in todo if t not in found]))


if __name__ == "__main__":
    main(sys.argv)
