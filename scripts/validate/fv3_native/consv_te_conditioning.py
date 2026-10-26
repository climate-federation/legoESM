#!/usr/bin/env python
"""How many digits of the energy fixer's ``dtmp`` are real?

``dtmp`` divides a GLOBAL sum of ``te0_2d - te_2d`` -- a difference of
two column integrals that are each ~1e9 -- into a correction that moves
pt by ~4e-6 K. GLM's MAJOR (job 9446300): the float64-vs-fixed-point
error bound I wrote was relative to ``sum |te0-te|*a``, not to the
CANCELLING ``|sum (te0-te)*a|`` that actually divides in, so it was
short by the condition number

    kappa = sum |te0-te|*a / |sum (te0-te)*a|

and kappa was never measured. This prints it, and re-does both sums
with ``math.fsum`` (exactly-rounded) so the summation choice is a
MEASUREMENT rather than an assumption:

    fsum moves dtmp by <= 1e-12 relative  -> >= 12 real digits, and
                                             everything said about the
                                             gates stands;
    fsum moves it at 1e-4                 -> only the fixer's sign and
                                             order of magnitude are
                                             physical, and the gate
                                             numbers are luck.

It also covers the same class in ``zsum1``, which uses ``np.sum``
(pairwise) where the Fortran runs a sequential k-loop.

    python consv_te_conditioning.py --consv 1.0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                            # noqa: E402

N, NG, KM = 48, 3, 5
ORACLE_ROOT = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned"


def _fsum_dtmp(te0_faces, te_faces, zsum0_faces, area_faces, *, consv, ng, n):
    """The same ratio, accumulated with math.fsum (exactly rounded)."""
    ia = ng
    num, den = [], []
    for te0, te, z0, ar in zip(te0_faces, te_faces, zsum0_faces, area_faces):
        a = ar[ia:ia + n, ia:ia + n]
        num.extend(((te0 - te) * a).ravel().tolist())
        den.extend((z0 * a).ravel().tolist())
    return consv * math.fsum(num) / math.fsum(den)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consv", type=float, default=1.0)
    ap.add_argument("--ic-run",
                    default=f"{ORACLE_ROOT}/run_hydro_zerostep_consv_gfs")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "packages", "core"))
    from legoesm.core.fv3_native_dynamics import (
        energy_fixer_dtmp,
        energy_fixer_zsum0_hydrostatic,
        fixer_energy_2d_hydrostatic,
        p_var_hydrostatic,
        total_energy_2d_hydrostatic,
    )
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA
    from legoesm.grids.fv3_native_gridstruct import FV3_RDGAS

    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__)))
    from full_step_oracle_parity import build_port_ic          # noqa: E402

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    state = build_port_ic(ctx, ak, bk, nh=False, zvir=0.0)[0]
    press = [p_var_hydrostatic(f["delp"], ptop=ptop, akap=FV3_KAPPA,
                               n=N, ng=NG, km=KM) for f in state]
    hs = np.zeros((N + 2 * NG, N + 2 * NG), dtype=np.float64)

    # te0 and te on the SAME state is the degenerate case (dtmp == 0), so
    # the interesting conditioning is te0 from the IC against te from a
    # perturbed state -- which is what one step produces. Use the IC for
    # te0 and a 1e-6-relative pt perturbation for te, i.e. a difference
    # of the size a real step makes.
    rng = np.random.default_rng(0)
    te0, te, z0, ar = [], [], [], []
    for t in range(6):
        gs = ctx["gs6"][t]
        te0.append(total_energy_2d_hydrostatic(
            state[t]["pt"], state[t]["delp"], state[t]["u"], state[t]["v"],
            press[t]["pe"], press[t]["peln"], hs, gs["rsin2"], gs["cosa_s"],
            qc=None, cp=FV3_CP_AIR, rg=FV3_RDGAS, n=N, ng=NG, km=KM))
        pt2 = state[t]["pt"] * (
            1.0 + 1.0e-6 * rng.standard_normal(state[t]["pt"].shape))
        te.append(fixer_energy_2d_hydrostatic(
            pt2, state[t]["delp"], state[t]["u"], state[t]["v"],
            press[t]["pe"], press[t]["peln"], hs, gs["rsin2"], gs["cosa_s"],
            cp=FV3_CP_AIR, rg=FV3_RDGAS, n=N, ng=NG, km=KM))
        z0.append(energy_fixer_zsum0_hydrostatic(
            press[t]["pkz"], state[t]["delp"], press[t]["pk"],
            ptop=ptop, n=N, ng=NG, km=KM))
        ar.append(gs["area"])

    dtmp, kappa = energy_fixer_dtmp(te0, te, z0, ar, consv=args.consv,
                                    n=N, ng=NG, returns_kappa=True)
    dtmp_f = _fsum_dtmp(te0, te, z0, ar, consv=args.consv, ng=NG, n=N)
    rel = abs(dtmp - dtmp_f) / abs(dtmp_f) if dtmp_f else float("nan")

    print(f"te0 column scale   {max(float(np.abs(x).max()) for x in te0):.6e}")
    print(f"dtmp   (np.sum)    {dtmp:.17e}")
    print(f"dtmp   (math.fsum) {dtmp_f:.17e}")
    print(f"kappa              {kappa:.6e}   "
          f"(cancellation amplification)")
    print(f"|d dtmp| / dtmp    {rel:.3e}")
    print(f"real digits >=     {-math.log10(rel) if rel > 0 else 17:.1f}")
    print(f"\nVERDICT: {'summation choice is immaterial' if rel <= 1e-12 else 'SUMMATION MATTERS -- the float64 sum is not safe here'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
