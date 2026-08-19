"""Characterise the UNEXPLAINED jit-vs-eager wind gap in the SW stage chain.

The stepper gate records u AND v at ~7.5e-07 relative jit-vs-eager on the
full acoustic step (job 9425294), already localised by
``test_jit_gap_localiser_stage_chain`` to ``acoustic_step_sixface`` at
5.646e-07 -- i.e. BEFORE the D-grid tail.  The per-kernel localisers
(``geopk_sw_1lev_d`` bitwise, ``one_grad_p_1lev`` ~1e-16) exonerate the
tail's kernels, so the subject is the SW chain itself.

This probe prints the measurements that discriminate the two candidate
mechanisms; it prints NO verdict.  The pre-registered readings:

  * CENSUS -- how many cells differ above each threshold, and where.
    A HANDFUL of isolated cells carrying almost the whole norm points at
    limiter/upwind selector flips (an ulp-level lowering difference
    crossing a ``smt5``/upwind switching surface produces a finite
    O(local-field) jump at exactly the flipped cells -- the fv3_nh_core
    edge_profile anatomy, here between two legal compilations).  A BROAD
    field-wide difference points at a systematic lowering defect.
  * STAGE bisection -- prefixes of the chain, each jitted whole, vs the
    eager prefix.  The first prefix whose output census jumps
    discontinuously names the stage where the compiled programs part.
    PIECEWISE -- each stage jitted alone on EAGER-produced inputs; if
    every stage alone is rounding-scale while the fused prefix is not,
    the gap needs the cross-stage fusion (or the accumulation of an
    upstream ulp through a downstream switching surface) to appear.
  * GROWTH -- the gap at n_split = 1, 2, 4, 8 substeps and over
    nsteps = 1, 2, 4, 8 full steps, jit lane vs eager lane.  FLAT (same
    order) = benign bit-flips re-selected each step; GROWING
    monotonically = a defect that compounds.
  * AMPLIFICATION control -- the SAME eager chain on the fixture vs the
    fixture with ``u`` perturbed by one ulp (relative 2^-52).  This
    calibrates how much the chain LEGALLY amplifies a rounding-level
    input difference.  If one ulp in produces O(1e-7) out at a few
    cells, the jit gap's magnitude needs no defect to explain it; if
    the response is ~2^-52, a 7.5e-07 jit gap cannot be selector
    amplification and IS a lowering defect.

Instrument controls (both hard-assert, per the validate-the-instrument
rule): the probe's stop='s6' eager prefix must be BITWISE identical to
the production ``acoustic_step_sixface`` (else the replicated chain has
drifted and every number below would be about the wrong operator), and
the full-prefix jit census must reproduce the gate's ~5.6e-07 order.

usage:  python scripts/validate/fv3_duo_jit_gap_localiser.py
"""
from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from legoesm.core import fv3_duo_stepper as jstep_mod  # noqa: E402
from legoesm.core import fv3_native_duo_stepper as npstep  # noqa: E402

N, NG = 12, 3
NPX = N + 1
DT = 450.0          # the gate module's DT
D_EXT = 0.0         # every duo deck resolves D_EXT = 0.0
_SENTINEL_FLOOR = 1.0e20
_THRESHOLDS = (1e-8, 1e-10, 1e-12, 1e-14)


def _provenance() -> None:
    try:
        sha = subprocess.run(
            ["git", "-C", _REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", _REPO, "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # pragma: no cover - provenance best effort
        sha, dirty = f"UNAVAILABLE ({e})", ""
    print(f"PROVENANCE sha={sha} dirty={bool(dirty)} jax={jax.__version__} "
          f"N={N} NG={NG} DT={DT} d_ext={D_EXT} x64="
          f"{jax.config.jax_enable_x64}", flush=True)


def _census(got, ref, name: str) -> float:
    """The gate's GLOBAL metric (``_cmp``'s scale) + a per-cell census.

    Same scale as ``test_fv3_duo_stepper._cmp`` ON PURPOSE, so the max
    printed here is digit-comparable with the gate's bounds.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) < _SENTINEL_FLOOR) \
        & (np.abs(a) < _SENTINEL_FLOOR)
    if not ok.any():
        print(f"CENSUS {name}: mask-only (0 physical cells)", flush=True)
        return 0.0
    scale = max(float(np.abs(b[ok]).max()), 1e-30)
    rel = np.where(ok, np.abs(a - b), 0.0) / scale
    mx = float(rel.max())
    counts = {t: int((rel > t).sum()) for t in _THRESHOLDS}
    cline = " ".join(f">{t:.0e}:{n}" for t, n in counts.items())
    print(f"CENSUS {name}: max_rel {mx:.3e} of {int(ok.sum())} cells | "
          f"{cline}", flush=True)
    if mx > 0.0:
        flat = np.argsort(rel, axis=None)[::-1][:8]
        for f in flat:
            idx = np.unravel_index(f, rel.shape)
            if rel[idx] == 0.0:
                break
            print(f"  top {name} idx={tuple(int(i) for i in idx)} "
                  f"rel={rel[idx]:.3e} ref={b[idx]:.6e} "
                  f"got={a[idx]:.6e}", flush=True)
    return mx


# ---------------------------------------------------------------------
# the chain, replicated stage-by-stage with a static stop
# ---------------------------------------------------------------------
# Verbatim from ``acoustic_step_sixface`` (fv3_duo_stepper.py) -- the
# bitwise control in main() is what keeps this copy honest.

_STOPS = ("entry", "csw", "s12", "s3", "bar2", "s4", "s5", "s6")


def acoustic_prefix(ctx, states, dt, stop: str = "s6"):
    if stop not in _STOPS:
        raise ValueError(f"unknown stop {stop!r}; one of {_STOPS}")
    cfg = jstep_mod._resolve_cfg(None)
    bd, npx, tab = ctx.bd, ctx.npx, ctx.tab
    m_a, ng = ctx.m_a, ctx.ng

    delp6 = jstep_mod.ext_scalar_sixface(states["delp"], tab, "A")
    pt6 = jstep_mod.ext_scalar_sixface(states["pt"], tab, "A")
    u6, v6 = jstep_mod.ext_vector_dgrid_sixface(states["u"], states["v"],
                                                tab)
    st = {**states, "delp": delp6, "pt": pt6, "u": u6, "v": v6}
    if stop == "entry":
        return {"delp": delp6, "pt": pt6, "u": u6, "v": v6}

    csw = jstep_mod.csw_step_sixface(ctx, st, 0.5 * dt)
    if stop == "csw":
        return {k: v for k, v in csw.items() if v is not None}

    s12 = jstep_mod.dsw12_step_sixface(ctx, st, csw, dt, sw_cfg=cfg)
    if stop == "s12":
        return {k: v for k, v in s12.items() if v is not None}

    s3 = jstep_mod._stack_faces([
        jstep_mod.d_sw3_duo(st["u"][t], st["v"][t], s12["uc"][t],
                            s12["vc"][t], ctx.gs6[t], ctx.flags6[t], bd,
                            npx, npx, dt=dt, hord_mt=cfg.hord_mt)
        for t in range(6)])
    if stop == "s3":
        return {k: v for k, v in s3.items() if v is not None}

    ubb6, vbbtemp6 = jstep_mod.average_shared_edge_bgrid(
        s3["ubb"], s3["vbbtemp"], tab)
    if stop == "bar2":
        return {"ubb": ubb6, "vbbtemp": vbbtemp6}

    ring = slice(ng, ng + npx)
    outs4, outs5, outs6 = [], [], []
    for t in range(6):
        kee = s3["ubbtemp"][t] * vbbtemp6[t]
        ke = jnp.zeros((m_a + 1, m_a + 1), dtype=jnp.float64)
        ke = ke.at[ring, ring].set(0.5 * (kee + ubb6[t] * s3["vbb"][t]))
        s4 = jstep_mod.d_sw4_duo(st["u"][t], st["v"][t], s12["ut"][t],
                                 s12["vt"][t], ke, ctx.flags6[t], bd,
                                 npx, npx, dt=dt)
        outs4.append(s4)
        if stop == "s4":
            continue
        s5 = jstep_mod.d_sw5_duo(s12["delp"][t], st["u"][t], st["v"][t],
                                 s12["uc"][t], s12["vc"][t],
                                 csw["ua"][t], csw["va"][t],
                                 s12["divg_d"][t], s12["crx_adv"][t],
                                 s12["cry_adv"][t], s12["xfx_adv"][t],
                                 s12["yfx_adv"][t], s12["ra_x"][t],
                                 s12["ra_y"][t], s4["ke"], ctx.gs6[t],
                                 ctx.flags6[t], bd, npx, npx, dt=dt,
                                 hord_vt=cfg.hord_vt, nord=cfg.nord,
                                 dddmp=cfg.dddmp, d2_bg=cfg.d2_bg,
                                 d4_bg=cfg.d4_bg, d_con=0.0)
        outs5.append(s5)
        if stop == "s5":
            continue
        s6 = jstep_mod.d_sw6_duo(st["u"][t], st["v"][t], s5["ut"],
                                 s5["vt"], s5["ke"], s5["wk"],
                                 s5["vortfluxx"], s5["vortfluxy"],
                                 ctx.gs6[t], ctx.flags6[t], bd, npx,
                                 npx, nord_v=cfg.nord_v,
                                 damp_v=cfg.damp_v, d_con=0.0)
        outs6.append({"delp": s12["delp"][t], "pt": s12["pt"][t],
                      "u": s6["u"], "v": s6["v"], "ke": s5["ke"],
                      "wk": s5["wk"], "divg_d": s5["divg_d"],
                      "delpc": s5["delpc"]})
    if stop == "s4":
        s = jstep_mod._stack_faces(outs4)
        return {k: v for k, v in s.items() if v is not None}
    if stop == "s5":
        s = jstep_mod._stack_faces(outs5)
        return {k: v for k, v in s.items() if v is not None}
    return jstep_mod._stack_faces(outs6)


def _tail_stage(ctx, st, csw, s12, dt):
    """d_sw3 -> barrier 2 -> d_sw4/5/6, as ONE jittable stage whose
    inputs are the (eager) outputs of the two upstream stages."""
    cfg = jstep_mod._resolve_cfg(None)
    bd, npx, tab = ctx.bd, ctx.npx, ctx.tab
    m_a, ng = ctx.m_a, ctx.ng
    s3 = jstep_mod._stack_faces([
        jstep_mod.d_sw3_duo(st["u"][t], st["v"][t], s12["uc"][t],
                            s12["vc"][t], ctx.gs6[t], ctx.flags6[t], bd,
                            npx, npx, dt=dt, hord_mt=cfg.hord_mt)
        for t in range(6)])
    ubb6, vbbtemp6 = jstep_mod.average_shared_edge_bgrid(
        s3["ubb"], s3["vbbtemp"], tab)
    ring = slice(ng, ng + npx)
    outs = []
    for t in range(6):
        kee = s3["ubbtemp"][t] * vbbtemp6[t]
        ke = jnp.zeros((m_a + 1, m_a + 1), dtype=jnp.float64)
        ke = ke.at[ring, ring].set(0.5 * (kee + ubb6[t] * s3["vbb"][t]))
        s4 = jstep_mod.d_sw4_duo(st["u"][t], st["v"][t], s12["ut"][t],
                                 s12["vt"][t], ke, ctx.flags6[t], bd,
                                 npx, npx, dt=dt)
        s5 = jstep_mod.d_sw5_duo(s12["delp"][t], st["u"][t], st["v"][t],
                                 s12["uc"][t], s12["vc"][t],
                                 csw["ua"][t], csw["va"][t],
                                 s12["divg_d"][t], s12["crx_adv"][t],
                                 s12["cry_adv"][t], s12["xfx_adv"][t],
                                 s12["yfx_adv"][t], s12["ra_x"][t],
                                 s12["ra_y"][t], s4["ke"], ctx.gs6[t],
                                 ctx.flags6[t], bd, npx, npx, dt=dt,
                                 hord_vt=cfg.hord_vt, nord=cfg.nord,
                                 dddmp=cfg.dddmp, d2_bg=cfg.d2_bg,
                                 d4_bg=cfg.d4_bg, d_con=0.0)
        s6 = jstep_mod.d_sw6_duo(st["u"][t], st["v"][t], s5["ut"],
                                 s5["vt"], s5["ke"], s5["wk"],
                                 s5["vortfluxx"], s5["vortfluxy"],
                                 ctx.gs6[t], ctx.flags6[t], bd, npx,
                                 npx, nord_v=cfg.nord_v,
                                 damp_v=cfg.damp_v, d_con=0.0)
        outs.append({"u": s6["u"], "v": s6["v"]})
    return jstep_mod._stack_faces(outs)


def _entry_only(states, tab):
    delp6 = jstep_mod.ext_scalar_sixface(states["delp"], tab, "A")
    pt6 = jstep_mod.ext_scalar_sixface(states["pt"], tab, "A")
    u6, v6 = jstep_mod.ext_vector_dgrid_sixface(states["u"], states["v"],
                                                tab)
    return {**states, "delp": delp6, "pt": pt6, "u": u6, "v": v6}


def main() -> int:
    _provenance()
    ctx = npstep.build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                            oracle_conventions=True)
    jctx = jstep_mod.build_jax_duo_stepper_context(ctx)
    states0 = npstep.w2_six_face_state(ctx)
    jstates0 = jstep_mod.states_to_jax(states0)

    # --- instrument control 1: the replicated chain is the chain ------
    ref = jstep_mod.acoustic_step_sixface(jctx, jstates0, DT)
    mine = acoustic_prefix(jctx, jstates0, DT, stop="s6")
    for k in ("delp", "pt", "u", "v"):
        assert np.array_equal(np.asarray(mine[k]), np.asarray(ref[k])), (
            f"INSTRUMENT INVALID: probe prefix s6 {k} != production "
            f"acoustic_step_sixface (the replicated chain drifted)")
    print("CONTROL prefix-s6 eager == production eager: BITWISE on "
          "delp/pt/u/v", flush=True)

    # ================================================================
    # (a) census of the full-chain jit-vs-eager gap
    # ================================================================
    print("\n=== (a) CENSUS: acoustic_step_sixface jit vs eager ===",
          flush=True)
    jit_full = jstep_mod.make_acoustic_step_sixface_jit()(jctx, jstates0,
                                                          DT)
    _seen_gap = 0.0
    for k in ("delp", "pt", "u", "v"):
        _r = _census(jit_full[k], ref[k], f"full-chain jit.{k}")
        if k in ("u", "v"):
            _seen_gap = max(_seen_gap, _r)
    # HARD-ASSERT control 2 (codex MAJOR: the docstring promised this and
    # the first version only printed). The probe's whole analysis assumes
    # it reproduces the harvest's ~5.6e-07 wind gap; a broken jit path or
    # changed backend producing 0.0 or 1e-3 here would invalidate every
    # downstream section, so it refuses instead of continuing.
    assert 1e-8 < _seen_gap < 1e-5, (
        f"instrument control: full-chain jit wind gap {_seen_gap:.3e} is "
        f"outside the expected [1e-8, 1e-5] band (harvest ~5.6e-07) -- "
        f"the probe is not measuring the phenomenon it analyses.")

    # ================================================================
    # (b) stage bisection -- prefixes jitted whole
    # ================================================================
    print("\n=== (b1) PREFIX bisection: jit(prefix) vs eager(prefix) ===",
          flush=True)
    for stop in _STOPS:
        fn = jax.jit(acoustic_prefix, static_argnums=(0,),
                     static_argnames=("stop",))
        got = fn(jctx, jstates0, DT, stop=stop)
        want = acoustic_prefix(jctx, jstates0, DT, stop=stop)
        for k in sorted(want):
            _census(got[k], want[k], f"prefix[{stop}].{k}")

    print("\n=== (b2) PIECEWISE: each stage jitted ALONE on eager "
          "inputs ===", flush=True)
    st = _entry_only(jstates0, jctx.tab)
    csw_e = jstep_mod.csw_step_sixface(jctx, st, 0.5 * DT)
    cfg = jstep_mod._resolve_cfg(None)
    s12_e = jstep_mod.dsw12_step_sixface(jctx, st, csw_e, DT, sw_cfg=cfg)

    csw_j = jstep_mod.make_csw_step_sixface_jit()(jctx, st, 0.5 * DT)
    for k in sorted(k for k, v in csw_e.items() if v is not None):
        _census(csw_j[k], csw_e[k], f"stage-alone csw.{k}")

    s12_j = jstep_mod.make_dsw12_step_sixface_jit()(jctx, st, csw_e, DT,
                                                    sw_cfg=cfg)
    for k in sorted(k for k, v in s12_e.items() if v is not None):
        _census(s12_j[k], s12_e[k], f"stage-alone dsw12.{k}")

    tail_j = jax.jit(_tail_stage, static_argnums=(0,))(jctx, st, csw_e,
                                                       s12_e, DT)
    tail_e = _tail_stage(jctx, st, csw_e, s12_e, DT)
    for k in ("u", "v"):
        _census(tail_j[k], tail_e[k], f"stage-alone tail.{k}")

    # ================================================================
    # (c) growth
    # ================================================================
    print("\n=== (c) GROWTH: jit lane vs eager lane ===", flush=True)
    jit_step = jstep_mod.make_full_acoustic_step_sixface_jit()
    for nsteps in (1, 2, 4, 8):
        e = jstep_mod.run_duo_sw(jctx, jstates0, DT, nsteps, d_ext=D_EXT)
        j = jstep_mod.run_duo_sw(jctx, jstates0, DT, nsteps, d_ext=D_EXT,
                                 step_fn=jit_step)
        for k in ("u", "v"):
            _census(j[k], e[k], f"growth nsteps={nsteps}.{k}")
    for n_split in (1, 2, 4, 8):
        e = jstep_mod.advance_duo_outer_step(jctx, jstates0,
                                             DT * n_split, n_split,
                                             d_ext=D_EXT)
        j = jstep_mod.advance_duo_outer_step(jctx, jstates0,
                                             DT * n_split, n_split,
                                             d_ext=D_EXT,
                                             step_fn=jit_step)
        for k in ("u", "v"):
            _census(j[k], e[k], f"growth n_split={n_split}.{k}")

    # ================================================================
    # (e) amplification control -- one ulp in, how much out, EAGER only
    # ================================================================
    print("\n=== (e) AMPLIFICATION: eager(x) vs eager(x*(1+2^-52)) ===",
          flush=True)
    pert = dict(jstates0)
    pert["u"] = jstates0["u"] * (1.0 + 2.0 ** -52)
    out_p = jstep_mod.acoustic_step_sixface(jctx, pert, DT)
    for k in ("delp", "pt", "u", "v"):
        _census(out_p[k], ref[k], f"one-ulp-in eager.{k}")

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
