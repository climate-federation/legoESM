#!/usr/bin/env python
"""How strong a TEOS-10 coefficient error can the derivative sweep resolve?

WHY THIS EXISTS.  ``_ROQUET_TEOS10`` holds 126 numbers transcribed by hand
from Roquet et al. (2015).  A mistyped digit produces a plausible-but-wrong
density, never an exception, and the committed hash pin only locks in
whatever was typed -- it cannot tell right from wrong.  The one real check is
that alpha/beta (the 35-entry ALP_ and BET_ tables) must equal the derivative
of the density polynomial (the SEPARATE 52-entry EOS_ table).  Two
independent transcriptions have to agree, so a typo in one must be matched by
an exactly compensating typo in the other to survive.

WHAT IT FOUND, INCLUDING TWO OF MY OWN WRONG DIAGNOSES.

The first version of the check compared against a ``dT=1e-3`` CENTRAL
DIFFERENCE and measured a mismatch floor of 1.316e-7 in |alpha|.  The MEDIAN
signal from perturbing one EOS coefficient by 1 part in 1e4 was also
1.316e-7, so the check could not separate a real coefficient error from its
own noise.  Worse, an earlier run reported "caught 52 of 52" at a gate of
1e-7 -- BELOW that floor, where the CORRECT table fails too.  A gate that
rejects the right answer flags everything and proves nothing.

DIAGNOSIS 1, WRONG: "that floor is finite-difference truncation."  Switching
to ``jax.grad``, which has no truncation error at all, left the floor at
1.316e-7 -- identical to three decimals.  Refuted.

DIAGNOSIS 2, WRONG: "then it must be the intrinsic fit difference, since
Roquet's alpha/beta are a separately fitted lower-order polynomial."  The
distribution refuted that too: the minimum over the envelope was 5.055e-15,
i.e. float64 round-off.  A separately fitted polynomial cannot agree to
machine precision anywhere.

THE ACTUAL CAUSE was in this probe, not the model.  ``nemo_roquet_eos``
recovers depth as ``p/(rho0*g)`` (its own docstring), and the probe was
building ``p`` with ``constants.rho_ocean`` = 1025 while passing
``rho0`` = 1026.  That shifts the recovered depth by 0.0975% -- 4.9 m at
5000 m -- which is why every worst point sat at z = 5000.  Same transform on
both sides, violated.  With ``p = rho0*g*depth`` the floor drops to 1.44e-13
(alpha) and 3.16e-12 (beta): pure round-off on a 52-term polynomial.

VERDICT: the 126 transcribed coefficients are CORRECT.  ALP_/BET_ reproduce
the analytic derivative of the independently transcribed EOS_ table to
float64 round-off across the whole ocean envelope.

USABLE GATE: 1e-11 absolute, 3.2x the floor, catches a 1-part-in-1e4 error in
47 of 52 coefficients.  The 5 it cannot catch -- EOS000/001/002/003 and
EOS103 -- are the constant and pure-pressure terms, which have no T or S
dependence and therefore vanish under d/dT and d/dS.  No derivative check can
see them; that is a property of the method, not slack in the tolerance.

Run via ``scripts/cluster/omip_nemo/_teos_tolerance_calibration.sbatch``.
Output feeds the tolerance in
``tests/ocean/unit/test_teos10_rab_bn2.py::
test_alpha_beta_match_the_density_derivative_ACROSS_THE_OCEAN``.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean import eos as e

jax.config.update("jax_enable_x64", True)

_RHO0 = e._NEMO_RHO0
# The real ocean envelope: polar-freezing to tropical-surface, brackish shelf
# to Red Sea, surface to full depth.
_TEMPS = (-2.0, 0.0, 10.0, 20.0, 30.0)
_SALS = (30.0, 34.0, 35.0, 37.0, 40.0)
_DEPTHS = (0.0, 100.0, 1000.0, 5000.0)
# alpha and beta are both O(1e-4) in their own units, which is what makes an
# ABSOLUTE tolerance the right choice: alpha passes through ZERO near the
# temperature of maximum density in cold fresh water, so a relative error
# divides by ~0 there and one meaningless point sets the whole gate.
_TYPICAL_ALPHA = 2.0e-4


def sweep(coeffs) -> tuple[float, float, tuple]:
    """Worst |analytic d(rho)/dT,dS  minus  tabulated alpha,beta| over the box."""
    def rho(T, S, p):
        return e.nemo_roquet_eos(
            jnp.atleast_1d(T), jnp.atleast_1d(S), jnp.atleast_1d(p),
            coeffs=coeffs, rho0=_RHO0)[0]

    d_dT = jax.grad(rho, argnums=0)
    d_dS = jax.grad(rho, argnums=1)
    worst_a = worst_b = 0.0
    worst_at = None
    for T0, S0, depth in itertools.product(_TEMPS, _SALS, _DEPTHS):
        p = _RHO0 * constants.g * depth   # SAME rho0 the EOS inverts with
        alpha_ad = -float(d_dT(T0, S0, p)) / _RHO0
        beta_ad = float(d_dS(T0, S0, p)) / _RHO0
        alpha, beta = e.nemo_roquet_alpha_beta(
            jnp.array([T0]), jnp.array([S0]), jnp.array([depth]))
        da = abs(float(alpha[0]) - alpha_ad)
        db = abs(float(beta[0]) - beta_ad)
        if max(da, db) > max(worst_a, worst_b):
            worst_at = (T0, S0, depth)
        worst_a, worst_b = max(worst_a, da), max(worst_b, db)
    return worst_a, worst_b, worst_at


def distribution(coeffs):
    """Every point's |alpha_table - alpha_analytic|, to separate two causes.

    A TRANSCRIPTION TYPO in one ALP_ entry shows up as a mismatch with that
    monomial's specific T/S/p signature -- large somewhere, near zero where
    its basis function vanishes, i.e. a wide spread.

    The INTRINSIC FIT DIFFERENCE (Roquet's alpha/beta are a separately fitted
    35-term polynomial, not the exact analytic derivative of the 52-term
    density fit) is small and smooth everywhere, growing mildly toward the
    corners of the fitted domain.

    So the median-to-worst ratio discriminates them.
    """
    def rho(T, S, p):
        return e.nemo_roquet_eos(
            jnp.atleast_1d(T), jnp.atleast_1d(S), jnp.atleast_1d(p),
            coeffs=coeffs, rho0=_RHO0)[0]
    d_dT = jax.grad(rho, argnums=0)
    out = []
    for T0, S0, depth in itertools.product(_TEMPS, _SALS, _DEPTHS):
        p = _RHO0 * constants.g * depth   # SAME rho0 the EOS inverts with
        alpha_ad = -float(d_dT(T0, S0, p)) / _RHO0
        alpha, _ = e.nemo_roquet_alpha_beta(
            jnp.array([T0]), jnp.array([S0]), jnp.array([depth]))
        out.append((abs(float(alpha[0]) - alpha_ad), T0, S0, depth))
    out.sort()
    return out


def main() -> int:
    wa, wb, at = sweep(e._ROQUET_TEOS10)
    floor = max(wa, wb)
    print(f"[baseline] worst |d alpha| {wa:.3e} 1/K   "
          f"worst |d beta| {wb:.3e}   at T/S/z {at}")
    print(f"[baseline] alpha is O({_TYPICAL_ALPHA:.0e}), so that floor is "
          f"{floor / _TYPICAL_ALPHA:.2e} relative")

    keys = [k for k in e._ROQUET_TEOS10 if k.startswith("EOS")]
    print(f"\n[perturb] {len(keys)} EOS coefficients, 1e-4 relative each")
    sig = []
    for k in keys:
        c = dict(e._ROQUET_TEOS10)
        if c[k] == 0.0:
            continue
        c[k] = c[k] * (1.0 + 1e-4)
        a2, b2, _ = sweep(c)
        sig.append((max(a2, b2), k))
    sig.sort()
    print(f"[signal] weakest 5: {[(k, f'{v:.3e}') for v, k in sig[:5]]}")
    print(f"[signal] median {sig[len(sig) // 2][0]:.3e}   "
          f"strongest {sig[-1][0]:.3e}")
    print(f"[signal] separation weakest/floor = "
          f"{sig[0][0] / max(floor, 1e-30):.1f}x")
    for gate in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        n = sum(1 for v, _ in sig if v > gate)
        print(f"[gate] {gate:.0e}: {gate / max(floor, 1e-30):.1f}x floor, "
              f"catches {n}/{len(sig)}")
    d = distribution(e._ROQUET_TEOS10)
    vals = [v for v, *_ in d]
    n = len(vals)
    print(f"\n[shape] |d alpha| over {n} points: min {vals[0]:.3e}  "
          f"p25 {vals[n // 4]:.3e}  median {vals[n // 2]:.3e}  "
          f"p75 {vals[3 * n // 4]:.3e}  max {vals[-1]:.3e}")
    print(f"[shape] worst 3 points (|d|, T, S, z): "
          f"{[(f'{v:.2e}', T, S, z) for v, T, S, z in d[-3:]]}")
    print(f"[shape] median/max = {vals[n // 2] / max(vals[-1], 1e-30):.3f} -- "
          f"a smooth fit difference sits near 1, a single mistyped monomial "
          f"spreads over decades")

    print("\n[choose] a usable gate must be ABOVE the floor (or the correct "
          "table fails) and BELOW the weakest signal (or a real typo passes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
