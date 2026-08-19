"""Which of ``riem_solver3``'s eight ``lax.scan`` sites breaks eager
bit-parity with the NumPy lane?

The NH-tail residual localises wholly inside ``riem_solver3``
(stage-split localiser, job 9424741: after ``update_dz_d`` the twins
are bitwise on every face; after ``riem_solver3`` they differ by
~5.8e-10 absolute per face).  The PLAUSIBLE mechanism is the one the
``edge_profile`` hunt proved for ``update_dz_d`` (jobs 9421844 ->
9424569): ``lax.scan`` compiles its body even on eager calls, and the
compiled body's contracted multiply-adds differ from the sequential
NumPy loop at ~1e-14 relative.  This probe MEASURES that instead of
assuming it, and prints no verdict.

Technique -- no re-derived numerics.  ``fv3_nh_core`` resolves ``lax``
from its module globals at call time, so a router object substituted
for ``jnh.lax`` can, per scan CALL SITE in execution order, either
delegate to the real ``lax.scan`` (XLA-compiled body) or iterate the
SAME production body closure eagerly in python (op-by-op primitives --
the exact semantics of scan, minus the compilation).  The eight sites
in execution order on the ``riem_solver3`` path:

  0 ``_pem_step``   riem_solver3   pem/peln accumulation
  1 ``_pp_fwd``     sim1_solver    pp forward Thomas sweep
  2 ``_pp_bwd``     sim1_solver    pp backward substitution
  3 ``_w_fwd``      sim1_solver    w forward Thomas sweep
  4 ``_w_bwd``      sim1_solver    w backward substitution
  5 ``_pe_step``    sim1_solver    pe time-tendency integral
  6 ``_dz_bwd``     sim1_solver    dz2 bottom-up back-out
  7 ``_seq_backbuild`` riem_solver3  zh interface rebuild

Runs, on the gate fixture's own bundle (identical operands to both
lanes; the NumPy lane's ``update_dz_d`` output feeds BOTH):

  A  router with all sites XLA (control: must equal the router-free
     call bitwise, proving the router itself changes nothing, and must
     reproduce the ~5.8e-10 baseline);
  B  all sites eager (if the scans are the whole residual, this is
     bitwise against NumPy; if not, what remains is the real site);
  C  leave-ONE-site-XLA sweep (face 4, the worst): the residual of run
     C_i is site i's own contribution.

The eager iterator is validated first against ``lax.scan`` on pure-add
bodies (forward and reverse), where no contraction is possible and any
difference is a semantics bug in the iterator itself.

MEASURED (jobs 9433880, 9435602, bit-identical across both): run B is
NOT bitwise -- eagerizing all eight scans leaves zh at 5.82e-10 on the
worst face -- so the scan hypothesis is REFUTED.  pe/peln leave the
accumulation BITWISE while pk differs at 3.55e-15 with only an
exact-rounded mul and ``exp`` between them, and run E confirms at the
primitive: eager ``jnp.exp`` vs ``np.exp`` on identical values differs
by ~1 ulp (log is bitwise); ``exp(gama*log(y))`` reproduces 4.66e-10.
The residual is the XLA-CPU-vs-libm ``exp`` implementation gap, not
scan compilation.  Run D: the router-free call is exactly repeatable
(0.0 twice, all faces).  OPEN, does not carry the verdict: run A's
router-all-XLA call differs DETERMINISTICALLY from the router-free
call by 5.24e-10 on faces 2-6 (0.0 on face 1), reproduced bit-identical
across both jobs; runs B/D/E are router-free or fully eager, so the
verdict rests only on those.

usage:  python scripts/validate/fv3_riem_solver3_scan_localiser.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax as real_lax  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "tests", "grids"))
sys.path.insert(0, _REPO)

N, NG, KM = 12, 3, 3
MA = N + 2 * NG
DT = 20.0

_SITE_NAMES = ("_pem_step", "_pp_fwd", "_pp_bwd", "_w_fwd", "_w_bwd",
               "_pe_step", "_dz_bwd", "_seq_backbuild")


def eager_scan(f, init, xs, length=None, reverse=False, unroll=1):
    """``lax.scan`` semantics with the body run EAGERLY per step.

    Same carry threading, same ys[i] <-> xs[i] correspondence under
    ``reverse`` (validated below against the real scan on pure-add
    bodies).  Every body call dispatches op-by-op primitives, which is
    the NumPy lane's association exactly (the edge_profile doctrine).
    """
    leaves = jax.tree_util.tree_leaves(xs)
    n = length if length is not None else leaves[0].shape[0]
    order = range(n - 1, -1, -1) if reverse else range(n)
    carry = init
    ys = [None] * n
    for i in order:
        x_i = jax.tree_util.tree_map(lambda a: a[i], xs)
        carry, y = f(carry, x_i)
        ys[i] = y
    ys_stacked = jax.tree_util.tree_map(
        lambda *ls: jnp.stack(ls, axis=0), *ys)
    return carry, ys_stacked


class ScanRouter:
    """Stands in for the ``lax`` module inside ``fv3_nh_core``.

    ``scan`` call number i (execution order, zero-based) delegates to
    the real ``lax.scan`` iff ``i`` is in ``xla_sites``; otherwise it
    runs :func:`eager_scan` on the same body/operands.  Everything else
    is forwarded to the real ``lax`` untouched.
    """

    def __init__(self, xla_sites):
        self.xla_sites = frozenset(xla_sites)
        self.calls = 0

    def scan(self, f, init, xs, **kw):
        i = self.calls
        self.calls += 1
        if i in self.xla_sites:
            return real_lax.scan(f, init, xs, **kw)
        return eager_scan(f, init, xs, **kw)

    def __getattr__(self, name):
        return getattr(real_lax, name)


def _validate_eager_scan() -> None:
    rng = np.random.default_rng(7)
    xs = jnp.asarray(rng.standard_normal((5, 4)))
    x0 = jnp.asarray(rng.standard_normal(4))

    def add(c, x):          # pure add: no contraction possible
        n = c + x
        return n, n

    for rev in (False, True):
        c_r, y_r = real_lax.scan(add, x0, xs, reverse=rev)
        c_e, y_e = eager_scan(add, x0, xs, reverse=rev)
        assert np.array_equal(np.asarray(c_r), np.asarray(c_e)), rev
        assert np.array_equal(np.asarray(y_r), np.asarray(y_e)), rev
    print("eager_scan semantics vs lax.scan (pure-add, fwd+rev): "
          "bitwise OK")


def _cmp(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise SystemExit(f"REFUSING TO COMPARE {name}: {a.shape} vs "
                         f"{b.shape}")
    d = float(np.abs(a - b).max())
    nbit = int((a != b).sum())
    return d, nbit


def main() -> int:
    _validate_eager_scan()

    import test_fv3_dsw_tail_3d as gate  # noqa: E402
    import legoesm.core.fv3_native_nh_core as npnh  # noqa: E402
    import legoesm.core.fv3_nh_core as jnh  # noqa: E402
    from legoesm.core.fv3_duo_stepper import (  # noqa: E402
        build_jax_duo_stepper_context,
    )
    from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
        build_six_face_duo_context,
    )
    import legoesm.core.fv3_native_dsw_tail_3d as nptail_mod  # noqa: E402

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    jctx = build_jax_duo_stepper_context(ctx)
    bd = ctx["bd"]
    bounds = (bd.is_, bd.ie, bd.js, bd.je, NG)
    fl = jctx.flags6
    state = gate._seeded_state(KM, seed=11, hydrostatic=False)
    bundle = gate.nh_bundle.__wrapped__(ctx, state)
    bundle = dict(bundle, state=state)
    carry = bundle["carry"]
    dsw, tail = bundle["dsw"], bundle["tail"]
    cswp = bundle["csw_press"]
    area6 = nptail_mod.nh_exchanged_area6(dict(ctx))
    rarea6 = [1.0 / np.asarray(a) for a in area6]

    def _np_riem(t, zh_np, ws_np):
        """NumPy lane on the shared operands; returns its outputs."""
        out = dict(
            zh=np.array(zh_np, copy=True),
            pe=np.array(carry["pe6"][t], copy=True),
            ppe=np.array(cswp[t]["pkc"], copy=True),
            pk3=np.array(carry["pk3_6"][t], copy=True),
            pk=np.array(carry["pk6"][t], copy=True),
            peln=np.array(carry["peln6"][t], copy=True),
            w=np.array(tail[t]["w"], copy=True),
            delz=np.array(bundle["state"][t]["delz"], copy=True),
        )
        npnh.riem_solver3(0, DT, bd, KM, 2.0 / 7.0, 1004.6, 100.0,
                          carry["zs6"][t], out["w"], out["delz"],
                          np.array(dsw[t]["pt"], copy=True),
                          np.array(dsw[t]["delp"], copy=True),
                          out["zh"], out["pe"], out["ppe"], out["pk3"],
                          out["pk"], out["peln"],
                          np.array(ws_np, copy=True), 0.05, 1.0,
                          use_logp=False, last_call=True, fp_out=False)
        return out

    def _jax_riem(t, zh_np, ws_np):
        """JAX lane, eager, on the SAME operands (whatever jnh.lax
        currently routes)."""
        out = jnh.riem_solver3(
            0, DT, bounds, KM, 2.0 / 7.0, 1004.6, 100.0,
            jnp.asarray(carry["zs6"][t]), jnp.asarray(tail[t]["w"]),
            jnp.asarray(bundle["state"][t]["delz"]),
            jnp.asarray(dsw[t]["pt"]), jnp.asarray(dsw[t]["delp"]),
            jnp.asarray(zh_np), jnp.asarray(carry["pe6"][t]),
            jnp.asarray(cswp[t]["pkc"]), jnp.asarray(carry["pk3_6"][t]),
            jnp.asarray(carry["pk6"][t]), jnp.asarray(carry["peln6"][t]),
            jnp.asarray(ws_np), 0.05, 1.0,
            use_logp=False, last_call=True, fp_out=False)
        names = ("w", "delz", "zh", "pe", "ppe", "pk3", "pk", "peln")
        return dict(zip(names, out))

    # Shared per-face operands: the NumPy update_dz_d output feeds BOTH
    # lanes (the twins are bitwise there -- job 9424741 -- but sharing
    # one array removes even that dependency from the comparison).
    shared = []
    for t in range(6):
        crx = np.stack([dsw[t]["levels"][k]["crx_adv"] for k in range(KM)],
                       axis=2)
        cry = np.stack([dsw[t]["levels"][k]["cry_adv"] for k in range(KM)],
                       axis=2)
        xfx = np.stack([dsw[t]["levels"][k]["xfx_adv"] for k in range(KM)],
                       axis=2)
        yfx = np.stack([dsw[t]["levels"][k]["yfx_adv"] for k in range(KM)],
                       axis=2)
        gs_nh = dict(ctx["gs6"][t])
        gs_nh["area"] = area6[t]
        gs_nh["rarea"] = rarea6[t]
        zh_np = np.array(carry["zh6"][t], copy=True)
        ws_np = np.array(carry["ws6"][t], copy=True)
        npnh.update_dz_d(np.full(KM + 1, 2.0), np.full(KM + 1, 0.12), 6,
                         bd, KM, N + 1, N + 1, area6[t], rarea6[t],
                         np.asarray(gate._DP0), carry["zs6"][t], zh_np,
                         np.array(crx, copy=True), np.array(cry, copy=True),
                         np.array(xfx, copy=True), np.array(yfx, copy=True),
                         ws_np, 1.0 / DT, gs_nh, lim_fac=1.0)
        shared.append((zh_np, ws_np))

    fields = ("zh", "delz", "w", "ppe", "pe", "peln", "pk", "pk3")
    n_sites = len(_SITE_NAMES)
    all_sites = frozenset(range(n_sites))

    ref = [_np_riem(t, *shared[t]) for t in range(6)]

    # --- run A: router with every site XLA == router-free control ----
    print("\nA. router-free vs router-all-XLA (must be bitwise: the")
    print("   router itself may change nothing), then vs NumPy:")
    plain = [_jax_riem(t, *shared[t]) for t in range(6)]
    jnh.lax = ScanRouter(all_sites)
    routed = [_jax_riem(t, *shared[t]) for t in range(6)]
    n_scan_calls = jnh.lax.calls // 6
    for t in range(6):
        worst = max(_cmp(nm, plain[t][nm], routed[t][nm])[0]
                    for nm in fields)
        dzh, nzh = _cmp("zh", routed[t]["zh"], ref[t]["zh"])
        print(f"   face {t + 1}: |router - plain| max {worst:.3e}   "
              f"zh vs numpy max|d| {dzh:.6e} ({nzh} cells differ)")
    print(f"   scan calls per riem_solver3: {n_scan_calls} "
          f"(expected {n_sites})")
    if n_scan_calls != n_sites:
        raise SystemExit(f"REFUSING: {n_scan_calls} scan sites executed, "
                         f"probe maps {n_sites} -- the site map is stale.")

    # --- run B: every site eager ------------------------------------
    print("\nB. all eight scan sites EAGER, vs NumPy (if the scans are")
    print("   the whole residual this is bitwise on every field):")
    jnh.lax = ScanRouter(frozenset())
    for t in range(6):
        got = _jax_riem(t, *shared[t])
        line = "   face %d:" % (t + 1)
        for nm in fields:
            d, nb = _cmp(nm, got[nm], ref[t][nm])
            line += f"  {nm} {d:.2e}/{nb}"
        print(line)

    # --- run C: leave one site XLA at a time (worst face) -----------
    tworst = 3                       # face 4: 5.82e-10 in job 9424741
    print(f"\nC. leave-ONE-site-XLA sweep, face {tworst + 1} "
          f"(residual = that site's own contribution):")
    for i in range(n_sites):
        jnh.lax = ScanRouter(frozenset({i}))
        got = _jax_riem(tworst, *shared[tworst])
        line = f"   only {_SITE_NAMES[i]:15s} XLA:"
        for nm in fields:
            d, nb = _cmp(nm, got[nm], ref[tworst][nm])
            if d != 0.0 or nb:
                line += f"  {nm} {d:.2e}/{nb}"
        if line.endswith("XLA:"):
            line += "  all fields bitwise"
        print(line)

    jnh.lax = real_lax

    # --- run D: repeatability of the router-free eager call ----------
    # Run A measured |router - plain| = 5.24e-10 on faces 2-6 with ALL
    # sites delegating to the real lax.scan -- i.e. two calls of the
    # SAME function on the SAME operands.  Before that number means
    # anything, the same comparison without any router in between.
    print("\nD. repeatability: router-free eager call twice, per face:")
    for t in range(6):
        r1 = _jax_riem(t, *shared[t])
        r2 = _jax_riem(t, *shared[t])
        worst = max(_cmp(nm, r1[nm], r2[nm])[0] for nm in fields)
        print(f"   face {t + 1}: max|call1 - call2| {worst:.3e}")

    # --- run E: the primitive itself -------------------------------
    # Under run B, pem/peln left the (eager) scan BITWISE while pk/pk3
    # differed at 3.55e-15, and the only arithmetic between them is
    # akap*peln2 (exact-rounded mul) and exp.  So compare np.exp and
    # eager jnp.exp ON THE LANE'S OWN VALUES, plus np.log/jnp.log on
    # the pem values, plus the sim1 pe_body composite exp(gama*log(y)).
    print("\nE. transcendental primitives on the lane's own operand "
          "values:")
    akap, gama = 2.0 / 7.0, 1.0 / (1.0 - 2.0 / 7.0)
    for t in (0, 3):
        peln_v = np.asarray(ref[t]["peln"], dtype=np.float64).ravel()
        pem_v = np.asarray(ref[t]["pe"], dtype=np.float64).ravel()
        pem_v = pem_v[pem_v > 0.0]
        x = akap * peln_v
        d_exp, n_exp = _cmp("exp", np.exp(x), jnp.exp(jnp.asarray(x)))
        d_log, n_log = _cmp("log", np.log(pem_v),
                            jnp.log(jnp.asarray(pem_v)))
        y = np.abs(pem_v) + 1.0     # generic positive values, same range
        d_cmp2, n_cmp2 = _cmp(
            "expglog", np.exp(gama * np.log(y)),
            jnp.exp(gama * jnp.log(jnp.asarray(y))))
        print(f"   face {t + 1}: exp(akap*peln) max|d| {d_exp:.3e} "
              f"({n_exp}/{x.size} differ)   log(pem) {d_log:.3e} "
              f"({n_log}/{pem_v.size})   exp(gama*log(y)) {d_cmp2:.3e} "
              f"({n_cmp2}/{y.size})")

    print("\nRIEM3_SCAN_LOCALISER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
