"""Discriminators for the five standing gate failures in the JAX lane.

Committed rather than run as a heredoc, per the repo's probe discipline:
a throwaway probe's number is unmeasured, and every conclusion below has
to be re-derivable at a SHA.

Each block PRE-REGISTERS what confirms and what refutes BEFORE it runs,
prints the number, and prints NO verdict -- the interpretation belongs in
the analysis, not baked into the tool.

The five, as they stand in jobs 9408760 / 9411351 / 9414224 / 9415283
(the same five in all four, i.e. reproducible, not flaky):

  P1  test_fv3_mapz::test_profile_jax_check_grads_order2_iv_minus2_bc
      -- `assert np.float64(0.0) > 1e-09`: an off-switch PRECONDITION
      clause measured exactly zero, so check_grads never ran.
  P2  test_fv3_nh_core::test_udzd_jax_check_grads_order2_away_from_switches
      -- JVP tangent 207.603485 vs 207.599113, 2.1e-5 relative.
  P3  test_fv3_pgrad::test_one_grad_p_linear_group_is_affine_and_gap_is_roundoff
      -- affine residual 6.148e-11 against a 1e-12 bound.
  P4  test_fv3_pgrad::test_nh_p_grad_vjp_fd_gap_is_fd_truncation
      -- largest-eps gap 4.06e-4 does not clear 10x the 7.27e-5
      roundoff floor, so the ladder is in the roundoff regime.
  P5  test_fv3_tp_core::test_sw_transport_one_sided_grads_across_smt5_surface[ytp_v]
      -- the limiter flag is True at every scanned t.

usage:  python scripts/validate/fv3_gradient_gate_triage.py [P1 P2 ...]
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# BOTH paths are needed: the test modules are imported by bare name (they
# are not a package), and they themselves do `from tests.grids...`, which
# needs the repo ROOT on the path.  Job 9417309 had only the first and
# every block but P5 died with ModuleNotFoundError: No module named 'tests'.
sys.path.insert(0, os.path.join(_REPO, "tests", "grids"))
sys.path.insert(0, _REPO)


def _hdr(name, question, confirms, refutes):
    print(f"\n{'=' * 70}\n{name}: {question}")
    print(f"  CONFIRMS: {confirms}")
    print(f"  REFUTES : {refutes}\n{'-' * 70}")


# ---------------------------------------------------------------- P3
def p3_affine_scale_sweep():
    """Is one_grad_p's u/v/divg2 residual roundoff, or real curvature?

    ``_affine_residual`` forms |[f(x+2v)-f(x)] - 2[f(x+v)-f(x)]| divided
    by |f(x+2v)-f(x)|, with v a UNIT-scale normal draw.  Scaling v by s:

      * a truly affine map has numerator = pure cancellation noise,
        which is ~u_mach*|f| INDEPENDENT of s, while the denominator
        grows LINEARLY in s -- so the reported ratio falls like 1/s;
      * a map with genuine quadratic curvature has numerator ~ s^2 * |H|
        and denominator ~ s -- so the ratio GROWS like s.

    The two predictions differ by a factor s^2 over the sweep, which no
    tolerance choice can blur.
    """
    import test_fv3_pgrad as tp

    _hdr("P3", "one_grad_p u/v/divg2: roundoff or curvature?",
         "ratio falls ~1/s  => affine + cancellation (bound is wrong)",
         "ratio grows ~s     => real curvature (the gate is right)")
    km = 2
    _f_pg, f_lin, _p_pg, p_lin = tp._ogp_grad_setup(km)
    for s in (0.01, 0.1, 1.0, 10.0, 100.0):
        rng = np.random.default_rng(0)
        primals = tuple(jnp.asarray(p) for p in p_lin)
        v = tuple(jnp.asarray(s * rng.standard_normal(p.shape))
                  for p in primals)
        f0 = f_lin(*primals)
        f1 = f_lin(*[p + t for p, t in zip(primals, v)])
        f2 = f_lin(*[p + 2.0 * t for p, t in zip(primals, v)])
        lo = jax.tree_util.tree_leaves(f0)
        l1 = jax.tree_util.tree_leaves(f1)
        l2 = jax.tree_util.tree_leaves(f2)
        num = max(float(np.abs((np.asarray(b) - np.asarray(a))
                               - 2.0 * (np.asarray(c) - np.asarray(a))
                               ).max())
                  for a, b, c in zip(lo, l2, l1))
        den = max(float(np.abs(np.asarray(b) - np.asarray(a)).max())
                  for a, b in zip(lo, l2))
        fmax = max(float(np.abs(np.asarray(a)).max()) for a in lo)
        print(f"  s={s:<7g} num={num:.6e} den={den:.6e} "
              f"ratio={num / max(den, 1e-300):.6e}  |f|inf={fmax:.4e}")


# ---------------------------------------------------------------- P4
def p4_nhpg_larger_eps():
    """The failing message's own instruction: go to LARGER eps.

    The gate needs the truncation term (eps^2) to dominate the roundoff
    floor (1/eps).  Printed for four ladders so the crossover is visible
    rather than guessed, together with the floor at each top step.
    """
    import test_fv3_pgrad as tp

    _hdr("P4", "nh_p_grad pp/u/v/delp: where does truncation dominate?",
         "some ladder has all ratios in (2.5, 6) AND top gap > 10x floor",
         "no ladder does => the group is affine or the VJP is wrong")
    km = 2
    _f_pg, f_nh, _p_pg, p_nh = tp._nhpg_grad_setup(km)
    for ladder in ((4.0e-4, 2.0e-4, 1.0e-4),
                   (1.6e-3, 8.0e-4, 4.0e-4),
                   (6.4e-3, 3.2e-3, 1.6e-3),
                   (2.56e-2, 1.28e-2, 6.4e-3)):
        r = [tp._vjp_fd_projection(f_nh, p_nh, e)[0] for e in ladder]
        floor0 = tp._fd_roundoff_floor(f_nh, p_nh, ladder[0])
        ratios = [r[i] / max(r[i + 1], 1e-300) for i in range(len(r) - 1)]
        print(f"  eps={ladder} gaps={[f'{x:.4e}' for x in r]} "
              f"ratios={[f'{x:.3f}' for x in ratios]} "
              f"floor(top)={floor0:.4e} top/floor={r[0] / floor0:.2f}")


# ---------------------------------------------------------------- P2
def p2_udzd_ladder():
    """Is update_dz_d's 2.1e-5 JVP gap FD truncation or a wrong Jacobian?

    Two tolerance-free instruments, because check_grads cannot separate
    them: the eps-scaling ladder (eps^2 -> ~4 per halving; a wrong
    derivative -> ~1), and the adjoint identity <Jv,w> = <v,J^T w>,
    which uses no finite differences at all.
    """
    import test_fv3_nh_core as tn

    # ⛔ THE LABELS BELOW WERE OVERSTATED AND ARE CORRECTED (codex MAJOR
    # 7, job 9417397).  Both instruments compare the implemented program
    # with derivatives OF THAT SAME PROGRAM: `jvp` and `vjp` are two
    # transformations of it, and the FD ladder differences the program
    # itself.  A smoothly WRONG implementation -- missing term, wrong
    # coefficient, wrong index -- has an exact adjoint identity and a
    # perfect eps^2 ladder.  So neither can say "the Jacobian is right";
    # what they separate is "AD disagrees with the function it was
    # derived from" (a real AD defect) from "the finite difference
    # cannot resolve it".  Correctness of the MAP is the parity gates'
    # job, against the NumPy authority.
    _hdr("P2", "update_dz_d: is check_grads' failure the FD's doing?",
         "ladder ~4 per halving AND adjoint residual ~0 => AD is "
         "self-consistent and agrees with the FD of its own function, "
         "so the gap is FD resolution",
         "ladder ~1 OR adjoint residual large => a real AD defect "
         "(neither outcome speaks to whether the MAP matches the spec)")
    # The SAME operand mapping and loss the gate differentiates, taken
    # from the test module rather than retyped (a retyped 14-operand
    # order would measure the transcription, not the Jacobian).
    fxt = tn._udzd_grad_fixture()
    _call, ja = tn.udzd_call_and_operands(fxt)
    ndif, damp = fxt["ndif"], fxt["damp"]

    def g(zs_, area_, rarea_, dxa_, dya_, du_, dv_):
        return tn.udzd_loss(*_call(ndif, damp, *ja[:6], zs_, area_,
                                   rarea_, dxa_, dya_, du_, dv_, ja[13]))

    args = tuple(ja[6:13])

    # (a) the tolerance-free adjoint identity, on the same operands.
    rng = np.random.default_rng(7)
    v = tuple(jnp.asarray(rng.standard_normal(np.shape(a))) for a in args)
    y, jv = jax.jvp(g, tuple(jnp.asarray(a) for a in args), v)
    w = jnp.asarray(rng.standard_normal(np.shape(y)))
    _, vjp = jax.vjp(g, *[jnp.asarray(a) for a in args])
    jtw = vjp(w)
    lhs = float(jnp.sum(jv * w))
    rhs = float(sum(jnp.sum(a * b) for a, b in zip(v, jtw)))
    den = max(abs(lhs), abs(rhs), 1e-300)
    print(f"  adjoint identity: <Jv,w>={lhs:.12e}  <v,J^T w>={rhs:.12e}  "
          f"rel={abs(lhs - rhs) / den:.3e}")

    # (b) the eps ladder on a scalar projection of the SAME map.
    def proj(eps):
        ap = [jnp.asarray(a) for a in args]
        fp = g(*[a + eps * t for a, t in zip(ap, v)])
        fm = g(*[a - eps * t for a, t in zip(ap, v)])
        fd = float(jnp.sum((fp - fm) * w) / (2.0 * eps))
        return abs(fd - lhs)

    for ladder in ((4.0e-4, 2.0e-4, 1.0e-4), (1.6e-3, 8.0e-4, 4.0e-4)):
        gaps = [proj(e) for e in ladder]
        ratios = [gaps[i] / max(gaps[i + 1], 1e-300)
                  for i in range(len(gaps) - 1)]
        print(f"  eps={ladder} gaps={[f'{x:.4e}' for x in gaps]} "
              f"ratios={[f'{x:.3f}' for x in ratios]}")


# ---------------------------------------------------------------- P1
def p1_mapz_flat_arm():
    """WHICH off-switch clause measured exactly zero, and where.

    ``_assert_profile_off_switch`` asserts |AL - qbar| and |AR - qbar|
    clear ``_FLAT_EPS``; a flattened layer has them EXACTLY zero.  The
    question is whether the qs bottom BC puts a layer on the flat arm
    (fixture defect) or whether the flattening is genuine physics of
    that column (code is right, gate is in the wrong place).
    """
    import test_fv3_mapz as tm

    _hdr("P1", "mapz iv=-2 BC: which layer is flat, and is it the BC one?",
         "the zero margin sits at the BOTTOM layer only => the qs BC "
         "fixture is on the switch",
         "zeros spread over interior layers => the column itself is flat")
    im, km = 3, tm.KMP
    delp = tm._delp_col(im, km, seed=43)
    q1 = tm._smooth_column(im=im, km=km, delp=delp)
    qs = tm._smooth_column_bottom_value(delp=delp)
    a4 = tm._jax_profile("cs", q1, delp, km, -2, 9, qs=qs)
    a4 = np.asarray(a4)
    qbar, al, ar, a6 = a4[1], a4[2], a4[3], a4[4]
    print(f"  a4 shape {a4.shape}; km={km}, im={im}")
    for i in range(im):
        dl = np.abs(al[i, 1:km + 1] - qbar[i, 1:km + 1])
        dr = np.abs(ar[i, 1:km + 1] - qbar[i, 1:km + 1])
        flat = np.where((dl == 0.0) | (dr == 0.0))[0] + 1
        print(f"  col {i}: min|AL-qbar|={dl.min():.3e} at k={dl.argmin() + 1}"
              f", min|AR-qbar|={dr.min():.3e} at k={dr.argmin() + 1}"
              f", EXACT-zero layers {list(flat)}")
        da1 = ar[i, 1:km + 1] - al[i, 1:km + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(a6[i, 1:km + 1] * da1) / np.where(
                da1 ** 2 > 0, da1 ** 2, np.nan)
        print(f"           worst |A6*da1|/da1^2 = "
              f"{np.nanmax(ratio):.3f} (clamp switch at 1.0)")


# ---------------------------------------------------------------- P5
def p5_ytp_probe_cell():
    """Does the flux cell the gate reads depend on the cell it perturbs?

    The gate's flag proxy is `flux[probe] != upwind source value`.  If
    the probe cell has ZERO derivative with respect to the perturbed
    cell, the proxy is reading a flux that never moves, so the flag can
    never turn off and the fixture is not one-sided -- the INDEX MAP is
    wrong.  The xtp_u arm passes, so it is printed as the control.
    """
    import test_fv3_tp_core as tt

    # SCOPE, corrected (codex MAJOR 6, job 9417397): the margin scan
    # below shows the old boolean proxy is a CONTINUOUS quantity that
    # crosses zero, which explains how the ytp arm could report "no
    # switch anywhere".  It does NOT establish what the xtp arm's
    # passing bracket contained -- that would need the old flags, the
    # margins and the smt5 predicate printed on both sides of it.  Any
    # statement about the xtp arm is PLAUSIBLE, not confirmed.
    _hdr("P5", "ytp_v: is the probed flux cell downstream of the "
               "perturbed cell?",
         "d flux[probe]/dt == 0 for ytp_v and != 0 for xtp_u => the "
         "test's y index map is wrong",
         "both nonzero => the index map is fine, and the boolean proxy "
         "is what needs examining")
    inp = np.load(os.path.join(tt.FIX, "dswcore_input.npz"))
    rough = tt._Geo(inp, "rough")
    b = rough.bd
    fi, fj = b.is_ + 3, b.js + 3
    ci, cj = fi - b.is_, fj - b.js
    c_probe = float(rough.c_sw[ci, cj])
    print(f"  c_sw[{ci},{cj}] = {c_probe:.6e} (sign picks the upwind cell)")

    for routine in ("xtp_u", "ytp_v"):
        if routine == "xtp_u":
            field = np.array(rough.u, dtype=np.float64)
            ui = fi - 1 if c_probe > 0.0 else fi
            kk = (ui - b.isd, fj - b.jsd)

            def run(f):
                return tt._xtp_jax(rough, 5, f, rough.c_sw)
        else:
            field = np.array(rough.v, dtype=np.float64)
            vj = fj - 1 if c_probe > 0.0 else fj
            kk = (fi - b.isd, vj - b.jsd)

            def run(f):
                return tt._ytp_jax(rough, 5, f, rough.c_sw)

        out = np.asarray(run(jnp.asarray(field)))
        gsens = jax.grad(lambda t: run(jnp.asarray(field).at[kk].add(t)
                                       )[ci, cj])(0.0)
        # Where DOES the perturbed cell land?  One-hot forward difference,
        # so a wrong index map shows the right cell instead of a bare zero.
        pert = np.asarray(run(jnp.asarray(field).at[kk].add(1.0))) - out
        nz = np.argwhere(np.abs(np.nan_to_num(pert)) > 0.0)
        print(f"  {routine}: out.shape={out.shape} probe=({ci},{cj}) "
              f"perturbed={kk}")
        print(f"      d out[probe]/dt = {float(gsens):.6e}")
        print(f"      cells that MOVED under a unit bump: "
              f"{[tuple(int(x) for x in r) for r in nz[:8]]}"
              f"{' ...' if len(nz) > 8 else ''} (n={len(nz)})")

        # The gate's own flag proxy, scanned WIDER than the gate scans
        # it and printed as a MARGIN rather than a boolean.  The gate
        # only sees `flux != upwind`, so a fixture that approaches the
        # collapse without reaching it looks identical to one that never
        # approaches it at all; the margin separates them.
        def base_of(fld):
            return float(fld[kk])

        print("      t -> flux[probe] - upwind(source)  (0 => the "
              "limiter collapsed the reconstruction, i.e. flag OFF)")
        row = []
        for t in (-2.0e6, -2.0e5, -8.0e4, -8000.0, -200.0, 0.0, 200.0,
                  8000.0, 8.0e4, 2.0e5, 2.0e6):
            f = field.copy()
            f[kk] = field[kk] + t
            d = float(np.asarray(run(jnp.asarray(f)))[ci, cj]) - base_of(f)
            row.append(f"{t:+.1e}:{d:+.3e}")
        print("        " + "  ".join(row))


_BLOCKS = {"P1": p1_mapz_flat_arm, "P2": p2_udzd_ladder,
           "P3": p3_affine_scale_sweep, "P4": p4_nhpg_larger_eps,
           "P5": p5_ytp_probe_cell}

if __name__ == "__main__":
    want = sys.argv[1:] or list(_BLOCKS)
    for key in want:
        try:
            _BLOCKS[key]()
        except Exception as exc:                       # noqa: BLE001
            # A probe that dies on block 2 must still report blocks 3-5;
            # the traceback is printed in full so a harness error is not
            # mistaken for a measurement.
            import traceback
            print(f"\n{key} RAISED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("\nTRIAGE_DONE")
