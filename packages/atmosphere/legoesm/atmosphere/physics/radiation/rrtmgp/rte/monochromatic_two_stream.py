# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The radiative transfer equation solver.

Common symbols:
ssa: single-scattering albedo;
tau: optical depth;
g: asymmetry factor;
sw: shortwave;
lw: longwave;
gamma: exchange rate coefficient in the radiative transfer equation;
zenith: the zenith angle of collimated solar radiation.
"""

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import rte_utils

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66
# Guard on the adding reciprocal 1/(1 - R A); fires only where R A rounds to 1.
_EPSILON = 1e-6
# Switch between the hyperbolic (small k*tau) and exp-scaled (large k*tau) forms of
# the diffuse two-stream quantities.  Both are algebraically exact and well
# conditioned at k*tau = 1, so the switch adds no discontinuity beyond roundoff.
_KTAU_SWITCH = 1.0

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Monochromatic (single-g-point) two-stream RTE kernel: Meador-Weaver "
        "reflectance/transmittance and Planck/solar cell sources, plus the "
        "vertical adding-doubling transport recurrence to up/down/net fluxes."
    ),
    "inputs": {
        "optical_depth": "1 (layer optical depth tau)",
        "ssa": "1 (single-scattering albedo)",
        "asymmetry_factor": "1", "zenith": "rad",
        "toa_flux": "W/m^2", "planck_src": "W/m^2",
        "sfc_src": "W/m^2", "sfc_emissivity": "1", "sfc_albedo": "1",
    },
    "outputs": {
        "flux_up": "W/m^2", "flux_down": "W/m^2", "flux_net": "W/m^2",
        "r_diff": "1 (diffuse reflectance)",
        "t_diff": "1 (diffuse transmittance)",
    },
    "sign_convention": (
        "fluxes positive in their named direction (flux_up upward, flux_down "
        "downward); flux_net = flux_up - flux_down (positive-up); optical depth "
        "tau>=0, 0<=ssa<=1, |g|<=1; reflectance and transmittance in [0,1] with "
        "r+t+absorptance=1 per layer (energy-consistent). The column budget is "
        "OPEN (Planck emission and solar/surface sources), so nothing is "
        "conserved as a closed budget."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Meador & Weaver (1980), J. Atmos. Sci. 37, 630-643 (two-stream "
        "reflectance/transmittance); Fu et al. (1997) LW diffusivity secant."
    ),
    "idealized_test": (
        "tests/unit/test_two_stream_top_flux.py: conservative scattering "
        "(ssa=1) gives reflectance+transmittance ~ 1; a purely absorbing layer "
        "(ssa=0) transmits exp(-D*tau)."
    ),
}


def _shift_up(f: Array) -> Array:
  """output_i = f_{i-1}."""
  return kernel_ops.shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
  """output_i = f_{i+1}."""
  return kernel_ops.shift_from_plus(f, 2)


def lw_combine_sources(planck_srcs: StatesMap) -> StatesMap:
  """Combine the longwave source functions at each cell face.

  RRTMGP provides two source functions at each cell interface using the
  spectral mapping of each adjacent layer. These source functions are combined
  here via a geometric mean, and the result can be used for two-stream
  calculations.

  Args:
    planck_srcs: A dictionary containing the longwave Planck sources at the cell
      faces [ccf]. The `planck_src_top` 3D variable contains the Planck source
      at the top cell face derived from the cell center's spectral mapping while
      the `planck_src_bottom` 3D variable contains the Planck source at the
      bottom cell face.

  Returns:
    A map of 3D variables for the combined Planck sources at the top face and
    the bottom cell face, respectively with the same keys as the ones in the
    input `planck_srcs`.
  """
  planck_src_top = planck_srcs['planck_src_top']
  planck_src_bottom = planck_srcs['planck_src_bottom']
  # Double where: sqrt has an infinite derivative at 0, and a level source can be
  # exactly 0 (halo cell) or legitimately far below any fixed floor (a cold level
  # in a far-wing g-point).
  product = planck_src_top * _shift_down(planck_src_bottom)
  positive = product > 0.0
  combined_src_top = jnp.where(
      positive, jnp.sqrt(jnp.where(positive, product, 1.0)), 0.0)
  # Boundary levels take the adjacent layer's Planck fraction alone, with no halo
  # fraction: lev_src(sfc) = pfrac(1) B(tlev(sfc)) and lev_src(top) = pfrac(nlay)
  # B(tlev(top)) (rte-rrtmgp mo_gas_optics_rrtmgp_kernels.F90 compute_Planck_source).
  combined_src_top = combined_src_top.at[:, :, 0].set(planck_src_bottom[:, :, 1])
  combined_src_top = combined_src_top.at[:, :, -2].set(planck_src_top[:, :, -2])
  combined_src_bottom = _shift_up(combined_src_top)
  return {
      'planck_src_top': combined_src_top,
      'planck_src_bottom': combined_src_bottom,
  }


def _eps_of(x: Array) -> float:
  """Machine epsilon of ``x``'s dtype.

  Every numerical guard in this module scales with it, so the same logic is
  correct at float32 and float64 (fixed guards like the former ``1e-2`` k
  floor mean very different things at different precisions and biased the
  conservative-scattering limit at float32).
  """
  return float(jnp.finfo(jnp.result_type(x)).eps)


# The hyperbolic helper below is parameterized by ``x2 = x**2``, NOT by ``x``
# — because its only caller has ``x = k*tau`` with ``k = sqrt(k2)`` and ``k2``
# hitting *exactly* zero for conservative scattering (an f32 Rayleigh-only
# g-point rounds ssa to 1). The functions are analytically smooth in ``x2``,
# but composing them as ``f(sqrt(x2))`` puts ``d sqrt/d x2 = inf`` on the
# reverse path at ``x2 = 0`` and the chain rule manufactures ``0 * inf = NaN``
# cotangents (this NaN'd every clear-sky SW column's temperature gradient).
# Taking ``x2`` directly keeps the whole evaluation polynomial in ``x2``, so
# there is no ``sqrt`` on the differentiated path at all.


def _series_order(eps: float) -> int:
  """Highest power of ``x2`` needed to reach ``eps`` accuracy on ``[0, 1]``.

  The slowest-converging of the four quantities below is ``cosh``, whose
  truncation error after the ``x2**n`` term is the first omitted term,
  ``x2**(n+1)/(2n+2)!``, i.e. ``1/(2n+2)!`` at the top of the range. Solving
  that against the working precision gives ``n = 5`` at float32 (error
  ``1/12! = 2e-9``, well under ``eps = 1.2e-7``) and ``n = 8`` at float64
  (``1/18! = 1.6e-16`` against ``eps = 2.2e-16``).
  """
  n = 1
  while 1.0 / math.factorial(2 * n + 2) > eps:
    n += 1
  return n


def _horner(x: Array, coeffs: list[float]) -> Array:
  """Evaluate ``sum(coeffs[i] * x**i)`` by Horner's rule."""
  acc = coeffs[-1]
  for c in reversed(coeffs[:-1]):
    acc = c + x * acc
  return acc


def _hyperbolics(x2: Array) -> tuple[Array, Array, Array, Array]:
  """``sinh(x)/x``, ``sinh(x/2)/(x/2)``, ``sinh(x)/x - 1`` and ``cosh(x)``.

  All four as functions of ``x2 = x**2``, computed together because the caller
  always needs all four at the same argument, and evaluated as **truncated
  Maclaurin series in x2 with no branch at all**.

  A branch-free polynomial is admissible here only because the caller clips its
  argument: ``_diffuse_quantities`` selects this (small-``k*tau``) form solely
  for ``x2 < _KTAU_SWITCH**2 = 1`` and passes a clipped ``x2`` otherwise, so
  the domain is exactly ``[0, 1]``. Over that domain the series converge fast
  enough to be correctly rounded at ``_series_order(eps)`` terms — five at
  float32 — which is where the cost goes:

  * **no transcendentals.** The previous formulation switched between these
    series and an ``exp``-based direct form at an eps-scaled threshold, and a
    ``jnp.where`` evaluates *both* branches, so every call paid for one
    ``sqrt`` and one ``exp`` (plus four selects and their compares) to end up
    on the polynomial anyway over most of the range. On ``[0, 1]`` the direct
    form buys nothing: a handful of fused multiply-adds is both cheaper and
    *more* accurate than ``(e**x - e**-x)/2x``, which cancels mildly at the
    small end of the interval.
  * **no ``sqrt`` on the differentiated path.** ``x`` itself never appears —
    see the note above on why ``sqrt(x2)`` is poison for the adjoint at the
    conservative-scattering point ``x2 = 0``. The previous version needed a
    masked-safe operand to keep that ``sqrt`` away from zero; with the direct
    form gone there is nothing left to guard.

  ``sinh(x)/x - 1`` is returned separately (rather than left to the caller to
  subtract) because it is ``O(x2)`` and forming it as ``sinhc - 1`` would
  cancel away its leading digits; its series simply omits the constant term.
  """
  n = _series_order(_eps_of(x2))
  # sinh(x)/x = sum x2**i / (2i+1)!;  cosh(x) = sum x2**i / (2i)!.
  sinhc_c = [1.0 / math.factorial(2 * i + 1) for i in range(n + 1)]
  cosh_c = [1.0 / math.factorial(2 * i) for i in range(n + 1)]
  # sinh(x)/x - 1 = x2 * sum x2**i / (2i+3)!, i.e. the same series with the
  # constant term dropped and one factor of x2 pulled out front.
  sinhc_m1_c = [1.0 / math.factorial(2 * i + 3) for i in range(n)]

  return (
      _horner(x2, sinhc_c),
      # sinh(x/2)/(x/2) is the same series evaluated at (x/2)**2 = x2/4.
      _horner(0.25 * x2, sinhc_c),
      x2 * _horner(x2, sinhc_m1_c),
      _horner(x2, cosh_c),
  )


def _k_squared(gamma1: Array, gamma2: Array) -> Array:
  """``k^2 = (gamma1+gamma2)(gamma1-gamma2)`` (Meador & Weaver 1980).

  Non-negative for physical inputs (``gamma1 >= gamma2`` whenever
  ``ssa <= 1``); clamped at zero against roundoff. Kept as ``k^2`` — the
  small-``k*tau`` diffuse branch is written so ``k`` itself cancels, making
  the conservative-scattering limit ``k -> 0`` exact with no floor at all.
  """
  return jnp.maximum((gamma1 + gamma2) * (gamma1 - gamma2), 0.0)


def _k_fn(gamma1: Array, gamma2: Array) -> Array:
  """``k`` with an eps-scaled floor (used by the exp-branch / direct beam).

  The floor is ``sqrt(eps)`` of the working dtype — ``~3e-4`` at float32,
  ``~1.5e-8`` at float64 — replacing the former fixed ``1e-2``, which biased
  the conservative-scattering limit identically at both precisions. It is
  kept hard (not smoothed): ``k`` feeds the removable-singularity handling
  at ``k cos(zenith) = 1`` in ``_direct_quantities``, and the floor only
  engages as ssa -> 1 where the small-``k*tau`` branch (which needs no
  ``k``) carries the diffuse quantities anyway.
  """
  return jnp.sqrt(jnp.maximum(_k_squared(gamma1, gamma2), _eps_of(gamma1)))


def _expm1_over_x(x: Array) -> Array:
  """``expm1(x) / x``, exact-in-the-limit at ``x = 0`` (value 1).

  ``expm1`` itself is accurate for all arguments; only the literal ``0/0`` at
  ``x = 0`` needs the series fallback, so the switch threshold is machine eps
  (below it ``1 + x/2`` is the correctly rounded value). Safe operands inside
  the ``where`` keep reverse-mode gradients NaN-free.
  """
  near_zero = jnp.abs(x) < _eps_of(x)
  x_safe = jnp.where(near_zero, 1.0, x)
  return jnp.where(near_zero, 1.0 + 0.5 * x, jnp.expm1(x_safe) / x_safe)


def _direct_quantities(
    gamma1: Array,
    gamma2: Array,
    gamma3: Array,
    gamma4: Array,
    alpha1: Array,
    alpha2: Array,
    tau: Array,
    ssa: Array,
    zenith: float | Array,
) -> StatesMap:
  """Direct-beam reflectance and transmittance (MW80 eqs 14-15), f32-stable.

  Meador and Weaver's eqs 14-15 are catastrophically ill-conditioned in
  float32 in two regimes:

  * **thin layers** — the numerators are differences of O(1) terms whose
    residual is O(tau) (100% relative error observed below tau ~ 1e-5);
  * **the resonance ``k mu0 = 1``** — numerator and denominator both vanish
    (a removable singularity; Clough et al. 1992, Toon et al. 1989), which
    the old code papered over with a fixed-epsilon clamp on
    ``1 - k^2 mu0^2``.

  Both are cured by the same algebraic step: subtract the (exactly zero)
  ``tau = 0`` value of each numerator, which turns every exponential into an
  ``expm1``, and then factor the resonance ``eta = 1 - k^2 mu0^2`` out of
  the numerator *analytically*. With ``a2 = alpha2 - k gamma3``,
  ``c = alpha1 + k gamma4``, ``d = alpha1 - k gamma4``,
  ``tau' = tau / (mu0 (1 + k mu0))`` and the diffuse denominator ``D``:

      R = (ssa/D) [ -a2 (expm1(-2 k tau)/(1 + k mu0) + 2 k mu0 P_r)
                    - 2 k gamma3 expm1(-(k + 1/mu0) tau)/(1 + k mu0) ]
      T = (ssa/D) [ -c (1 + k mu0) P_t
                    + d e^{-k tau} expm1(-(k + 1/mu0) tau)/(1 + k mu0) ]

  where the resonance pairs

      P_r = (e^{-2 k tau} - e^{-(k + 1/mu0) tau}) / eta
      P_t = (e^{-tau/mu0} - e^{-k tau}) / eta

  are evaluated, for ``|eta tau'| < 1``, in the exactly-equivalent factored
  forms ``P_r = tau' e^{-(k + 1/mu0) tau} phi(eta tau')`` and
  ``P_t = -e^{-k tau} tau' phi(-eta tau')`` with ``phi(x) = expm1(x)/x`` —
  no division by ``eta`` anywhere, finite and smooth *through* the
  resonance. For ``|eta tau'| >= 1`` the literal quotients are used (their
  relative cancellation error is ~eps/|eta tau'| <= eps there). Thin-layer
  limits are reproduced exactly: R -> ssa gamma3 tau/mu0,
  T -> ssa gamma4 tau/mu0, and both vanish at tau = 0 and at ssa = 0 (the
  factor ssa now *multiplies*, so no safe-divide masking is needed).

  Note eq 15's full transmittance includes the unscattered beam
  ``exp(-tau/mu0)``; as before, only the diffusely-transmitted part is
  returned (the direct beam is handled separately in ``sw_cell_source``).
  """
  mu0 = jnp.cos(zenith)
  k = _k_fn(gamma1, gamma2)
  k_mu = k * mu0
  one_plus_kmu = 1.0 + k_mu

  # These exponentials come in two families and neither can be rebuilt from
  # the other, so each is evaluated in the form it is used in. `exp` carries a
  # decayed value at full *relative* precision, which `1 + expm1` cannot: by
  # k*tau = 16 float32 `expm1` has rounded to exactly -1, so `1 + expm1` gives
  # 0 where the true exponential is 1.1e-7. `expm1` in turn resolves the O(tau)
  # departure from 1 that `exp` rounds away entirely for a thin layer.
  e2m1 = jnp.expm1(-2.0 * k * tau)                # exp(-2 k tau) - 1
  denom = k * (2.0 + e2m1) - gamma1 * e2m1        # diffuse denominator D
  etm1 = jnp.expm1(-(k + 1.0 / mu0) * tau)        # exp(-(k + 1/mu0) tau) - 1
  e_ktmu = 1.0 + etm1                             # exp(-(k + 1/mu0) tau)
  e_kt = jnp.exp(-k * tau)
  t0 = jnp.exp(-tau / mu0)

  eta = 1.0 - k_mu * k_mu
  tau_p = tau / (mu0 * one_plus_kmu)
  x = eta * tau_p
  factored = jnp.abs(x) < 1.0
  # Safe operands: clip the factored-form argument (its phi/tau' pieces are
  # only well-scaled where selected) and the quotient-form divisor.
  x_f = jnp.where(factored, x, 0.0)
  eta_safe = jnp.where(factored, 1.0, eta)
  # The two resonance pairs need phi at +x and -x. One evaluation serves both:
  # phi(-x) = expm1(-x)/(-x) = (e**x - 1)/(x e**x) = phi(x) / e**x, and
  # e**x = 1 + x phi(x) identically. On the branch where this form is selected
  # |x| < 1, so e**x is in [0.37, 2.72] and that reconstruction loses at most
  # ~1.5 bits -- against a whole transcendental per g-point per cell.
  phi = _expm1_over_x(x_f)
  phi_neg = phi / (1.0 + x_f * phi)
  pair_r = jnp.where(
      factored,
      tau_p * e_ktmu * phi,
      ((1.0 + e2m1) - e_ktmu) / eta_safe,
  )
  pair_t = jnp.where(
      factored,
      -e_kt * tau_p * phi_neg,
      (t0 - e_kt) / eta_safe,
  )

  a2 = alpha2 - k * gamma3
  r_dir = (ssa / denom) * (
      -a2 * (e2m1 / one_plus_kmu + 2.0 * k_mu * pair_r)
      - 2.0 * k * gamma3 * etm1 / one_plus_kmu
  )
  c = alpha1 + k * gamma4
  d = alpha1 - k * gamma4
  t_dir = (ssa / denom) * (
      -c * one_plus_kmu * pair_t + d * e_kt * etm1 / one_plus_kmu
  )
  # `1 - exp(-tau/mu0)` is the energy-conservation cap the caller clamps
  # against. Return it from here rather than have the caller evaluate the same
  # exponential a second time. (The subtraction does cancel for a thin layer,
  # and `-expm1(-tau/mu0)` would give it exactly -- but only at the cost of
  # another transcendental, and for no observable gain: a cap that small is far
  # inside `_SW_CLIP_SHARPNESS`, so the smooth clamp drives r_dir and t_dir to
  # zero there whether or not the cap itself is resolved.)
  return {'r_dir': r_dir, 't_dir': t_dir, 'one_minus_t0': 1.0 - t0}


def _diffuse_quantities(gamma1: Array, gamma2: Array, tau: Array) -> StatesMap:
  """Diffuse two-stream quantities (MW80 eqs 25-26), float32-stable.

  Two algebraically exact evaluations are blended at ``k*tau = 1``:

  * ``k*tau < 1`` — hyperbolic, k-free form. With the identities
    ``1 - exp(-2x) = 2 exp(-x) sinh(x)`` and
    ``1 + exp(-2x) = 2 exp(-x) cosh(x)``, MW80's refactored expressions
    reduce to ``R = gamma2 sinh(k tau)/Dh``, ``T = k/Dh`` with
    ``Dh = k cosh(k tau) + gamma1 sinh(k tau)``. Factoring one ``k`` out of
    numerator and denominator (``sinh(k tau) = k tau sinhc(k tau)``) leaves

        C = cosh(k tau) + gamma1 tau sinhc(k tau)
        R = gamma2 tau sinhc(k tau) / C,      T = 1 / C

    which involves only ``k^2 = (g1+g2)(g1-g2)`` (via cosh/sinhc arguments),
    is exact at ``k = 0`` (conservative scattering — the old fixed ``1e-2``
    floor is gone) and cancellation-free for thin layers.
  * ``k*tau >= 1`` — the exp-scaled original (rte-rrtmgp) form with
    ``-expm1(-2 k tau)`` replacing ``1 - exp(-2 k tau)``; ``cosh`` would
    overflow float32 beyond ``k tau ~ 88``, and this form is
    well-conditioned for thick layers.

  Also returns the two non-cancelling combinations the longwave linear-in-tau
  source needs (Toon et al. 1989 eqs 26-27; see
  ``lw_cell_source_and_properties``):

      one_minus_r_minus_t = 1 - R - T                       (O(tau) small)
      g_minus_t           = (1 + R - T)/(tau (g1+g2)) - T   (O(tau) small)

  Assembling these from ``R`` and ``T`` would subtract numbers agreeing to
  ``O(tau)`` — exactly the float32 cancellation this refactor removes; the
  closed forms below are derived from the same hyperbolic identities
  (``cosh(x) - 1 = 2 sinh^2(x/2)``, computed as squares — never by
  subtraction) and, on the thick branch, from
  ``k (1 - exp(-k tau))^2 = k expm1(-k tau)^2``.
  """
  eps = _eps_of(gamma1)
  k2 = _k_squared(gamma1, gamma2)
  k = _k_fn(gamma1, gamma2)
  # The branch selector and the whole small branch are written in
  # x2 = (k*tau)^2 = k2*tau^2 — never through sqrt(k2), whose reverse-mode
  # derivative is infinite at the conservative-scattering point k2 = 0 (see
  # the note above the hyperbolic helpers). ``k`` itself (eps-floored, safe
  # adjoint) is used only by the large branch.
  x2 = k2 * jnp.square(tau)
  small = x2 < _KTAU_SWITCH**2
  gsum = jnp.maximum(gamma1 + gamma2, eps)

  # --- small-k*tau branch (safe operand: the argument is clipped to the switch
  # point when this branch is *not* the selected one, both so the truncated
  # series stay inside the interval they are accurate on and so their leading
  # x2**n term cannot overflow to inf for a thick layer and put a `0 * inf`
  # NaN into the reverse pass of the select below).
  x2c = jnp.where(small, x2, _KTAU_SWITCH**2)
  # All four hyperbolic quantities come from truncated series in x2 sharing one
  # clipped argument: no sqrt, no exp, no inner branch. See `_hyperbolics`.
  # Evaluating them separately was the single largest source of transcendentals
  # in this solve (issue #22), and switching to an exp-based direct form over
  # part of the range still paid for that exp on every call (issue #27).
  shc, shc_half, shc_m1, ch = _hyperbolics(x2c)
  c_small = ch + gamma1 * tau * shc
  r_small = gamma2 * tau * shc / c_small
  t_small = 1.0 / c_small
  # 1 - R - T = tau*(k^2 tau/2 * sinhc^2(x/2) + (g1-g2) sinhc(x)) / C
  omrt_small = tau * (0.5 * k2 * tau * jnp.square(shc_half)
                      + (gamma1 - gamma2) * shc) / c_small
  # G - T = (k^2 tau/(2 gsum) * sinhc^2(x/2) + sinhc_m1(x)) / C
  gmt_small = (0.5 * k2 * tau * jnp.square(shc_half) / gsum
               + shc_m1) / c_small

  # --- large-k*tau branch (exp-scaled; safe at any tau).
  ktau_big = jnp.where(small, _KTAU_SWITCH, k * tau)
  # One exp serves the whole branch. `expm1` and a second `exp` are not needed
  # here: this branch is only selected for `k*tau >= 1`, so `e1 <= exp(-1)` and
  # neither `e1 - 1` (in [-1, -0.63]) nor `1 - e1**2` (in [0.86, 1]) cancels.
  # `expm1` earns its keep near zero, which this branch never sees.
  e1 = jnp.exp(-ktau_big)
  e2 = jnp.square(e1)                  # exp(-2 k tau)
  em1 = e1 - 1.0                       # in [-1, 0)
  a = 1.0 - e2                         # 1 - exp(-2 k tau), no cancellation
  d_big = k * (1.0 + e2) + gamma1 * a
  r_big = gamma2 * a / d_big
  t_big = 2.0 * k * e1 / d_big
  # D - 2k e^{-ktau} -/+ gamma2*(1-e^{-2ktau}) = k(1-e^{-ktau})^2 + (g1-/+g2)a
  omrt_big = (k * jnp.square(em1) + (gamma1 - gamma2) * a) / d_big
  # Safe operand: on the small branch (unselected here) tau can be 0;
  # ktau >= 1 on the selected branch guarantees tau > 0.
  tau_big = jnp.where(small, 1.0, tau)
  g_big = (k * jnp.square(em1) + gsum * a) / (tau_big * gsum * d_big)
  gmt_big = g_big - t_big

  return {
      'r_diff': jnp.where(small, r_small, r_big),
      't_diff': jnp.where(small, t_small, t_big),
      'one_minus_r_minus_t': jnp.where(small, omrt_small, omrt_big),
      'g_minus_t': jnp.where(small, gmt_small, gmt_big),
  }



def lw_cell_source_and_properties(
    optical_depth: Array,
    ssa: Array,
    level_src_bottom: Array,
    level_src_top: Array,
    asymmetry_factor: Array,
    lw_diffusive_factor: float | Array = _LW_DIFFUSIVE_FACTOR,
) -> StatesMap:
  """Compute the longwave two-stream reflectance, transmittance, and sources.

  The upwelling and downwelling Planck functions and the optical properties
  (transmission and reflectance) are calculated at the cell centers. Equations
  are developed in Meador and Weaver (1980) and Toon et al. (1989).

  Args:
    optical_depth: The pointwise optical depth.
    ssa: The pointwise single-scattering albedo.
    level_src_bottom: The Planck source at the bottom cell face [W / m^2 / sr].
    level_src_top: The Planck source at the top cell face [W / m^2 / sr].
    asymmetry_factor: The pointwise asymmetry factor.
    lw_diffusive_factor: Secant of the diffusivity angle.  Scalar (e.g. the
      Fu-Liou ``1.66`` default) or an ``Array`` broadcastable against the
      ``(ncol, 1, nlev+2)`` shape of ``optical_depth``.  Upstream RRTMGP
      computes a per-band, per-column value from a polynomial fit on the
      column total optical depth (``compute_optimal_angles``); pass the
      broadcast secant from ``solve_lw`` when the gas-optics file ships
      ``optimal_angle_fit`` coefficients.

  Returns:
    A dictionary containing the following items:
      'r_diff': A 3D variable containing the pointwise reflectance.
      't_diff': A 3D variable containing the pointwise transmittance.
      'src_up': A 3D variable containing the pointwise upwelling Planck source.
      'src_down': A 3D variable with the pointwise downwelling Planck source.
  """
  # Clamp optical depth to non-negative values: halo extrapolation can produce
  # negative tau which causes exp(+|tau|*k) to explode.  The halos are stripped
  # before the recurrence, but backward-pass gradients still flow through the
  # intermediate variables and must stay finite.
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # The coefficient of the parallel irradiance in the 2-stream RTE.
  gamma1 = lw_diffusive_factor * (1 - 0.5 * ssa * (1 + asymmetry_factor))
  # The coefficient of the antiparallel irradiance in the 2-stream RTE.
  gamma2 = lw_diffusive_factor * 0.5 * ssa * (1 - asymmetry_factor)

  diffuse = _diffuse_quantities(gamma1, gamma2, optical_depth)
  r_diff = diffuse['r_diff']
  t_diff = diffuse['t_diff']

  # Cell-centre sources for the linear-in-tau Planck profile (Toon et al. 1989
  # eqs 26-27).  With S_t, S_b the top/bottom face sources, dS = S_b - S_t and
  # G = (1 + R - T) / (tau (gamma1 + gamma2)):
  #     src_up = pi (S_t (1 - R - T) + dS (G - T))
  #     src_dn = pi (S_b (1 - R - T) - dS (G - T))
  # algebraically RTE's lw_source_2str.  1 - R - T and G - T are O(tau) and come
  # from _diffuse_quantities without cancellation, so no optical-depth cutoff is
  # needed.  KNOWN DIFFERENCE: RTE zeroes these sources for tau <= 1e-8.
  delta_src = level_src_bottom - level_src_top
  src_up = math.pi * (
      level_src_top * diffuse['one_minus_r_minus_t']
      + delta_src * diffuse['g_minus_t'])
  src_down = math.pi * (
      level_src_bottom * diffuse['one_minus_r_minus_t']
      - delta_src * diffuse['g_minus_t'])
  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      'src_up': src_up,
      'src_down': src_down,
  }


def sw_cell_properties(
    zenith: float | Array, optical_depth: Array, ssa: Array, asymmetry_factor: Array
) -> StatesMap:
  """Compute shortwave reflectance and transmittance.

  Two-stream solutions to direct and diffuse reflectance and transmittance as
  a function of optical depth, single-scattering albedo, and asymmetry factor.
  Equations are developed in Meador and Weaver (1980).

  Args:
    zenith: The zenith angle of the shortwave collimated radiation.
    optical_depth: A 3D variable containing the pointwise optical depth.
    ssa: A 3D variable containing the pointwise single-scattering albedo.
    asymmetry_factor: A 3D variable containing the pointwise asymmetry factor.

  Returns:
    A dictionary containing the following items:
    't_diff': A 3D variable containing the diffuse transmittance.
    'r_diff': A 3D variable containing the diffuse reflectance.
    't_dir': A 3D variable containing the direct transmittance.
    'r_dir': A 3D variable containing the direct reflectance.
  """
  # Clamp optical depth to non-negative for AD stability (see lw comment).
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # Exchange rate coefficients from Zdunkowski et al. (1980).
  g = asymmetry_factor
  gamma1 = 0.25 * (8 - ssa * (5 + 3 * g))
  gamma2 = 0.25 * 3 * ssa * (1 - g)
  gamma3 = 0.25 * (2 - 3 * jnp.cos(zenith) * g)
  gamma4 = 1 - gamma3
  alpha1 = gamma1 * gamma4 + gamma2 * gamma3
  alpha2 = gamma1 * gamma3 + gamma2 * gamma4

  diffuse = _diffuse_quantities(gamma1, gamma2, optical_depth)
  r_diff = diffuse['r_diff']
  t_diff = diffuse['t_diff']

  direct = _direct_quantities(
      gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, optical_depth, ssa, zenith)
  r_dir_unconstrained = direct['r_dir']
  t_dir_unconstrained = direct['t_dir']

  # Constrain reflectance and transmittance to be positive and to not go above
  # physical limits by enforcing the constraint that the direct beam can
  # either be reflected, penetrate unscattetered to the bottom of the grid
  # cell, or penetrate through but be scattered on the way.

  # Direct transmittance.
  one_minus_t0 = direct['one_minus_t0']
  r_dir = jnp.clip(r_dir_unconstrained, 0, one_minus_t0)
  t_dir = jnp.clip(t_dir_unconstrained, 0, one_minus_t0 - r_dir)

  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      't_dir': t_dir,
      'r_dir': r_dir,
  }


def sw_cell_source(
    t_dir: Array,
    r_dir: Array,
    optical_depth: Array,
    toa_flux: Array,
    sfc_albedo_direct: Array,
    zenith: float | Array,
    use_scan: bool | None = None,
    mu_incident: float | Array | None = None,
) -> StatesMap:
  """Compute the monochromatic shortwave direct-beam flux and diffuse source.

  Args:
    t_dir: Direct-beam transmittance, a 3D field.
    r_dir: Direct-beam reflectance, a 3D field.
    optical_depth: Optical depth, a 3D field.
    toa_flux: The top-of-atmosphere incoming flux, a 2D field.
    sfc_albedo_direct: The surface albedo with respect to direct radiation, a 2D
      field.
    zenith: The solar zenith angle of the slant path (attenuation).
    use_scan: Whether to use scan or for loops for the recurrent operation.
    mu_incident: Cosine of the zenith angle that projects the incident beam,
      ``toa_flux * mu_incident`` (RTE sw_solver_2stream: inc_flux_dir * mu0,
      the true mu0).  ``None`` -> ``cos(zenith)``.

  Returns:
    A dictionary containing the following items:
      'src_up': A 3D field for the cell center upward source.
      'src_down': A 3D field for the cell center downward source.
      'flux_down_dir': A 3D field for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      'sfc_src': A 2D field for the shortwave source emanating from the surface.
  """
  # Clamp optical depth to non-negative for AD stability (see lw comment).
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # Transmittance of direct, unscattered beam.
  t_noscat = jnp.exp(-optical_depth / jnp.cos(zenith))
  mu = jnp.cos(zenith) if mu_incident is None else mu_incident

  # The vertical component of incident flux at the top boundary.
  # ``mu`` may be 3-D ``(ncol, 1, 1)`` for per-column zenith; reshape to
  # match the 2-D ``toa_flux`` shape ``(ncol, 1)``.
  mu_2d = mu.reshape(toa_flux.shape) if jnp.ndim(mu) > jnp.ndim(toa_flux) else mu
  flux_down_direct_bc = toa_flux * mu_2d

  # Global recurrent accumulation for the direct-beam downward flux at the
  # bottom cell face unraveling from the top of the atmosphere down to the
  # surface. The recurrence follows the simple relation:
  # flux_down_direct[i] = T_no_scatter[i] * flux_down_direct[i + 1]
  def op(carry, w):
    return w * carry, w * carry
  init = flux_down_direct_bc
  inputs = {'w': t_noscat}
  _, flux_down_direct = rte_utils.recurrent_op_with_halos(
      op, init, inputs, forward=False, use_scan=use_scan
  )

  # Upward source from direct-beam reflection at the cell center.
  src_up = r_dir * _shift_down(flux_down_direct)

  # Downward source from direct-beam transmittance at the cell center.
  src_down = t_dir * _shift_down(flux_down_direct)

  # Direct-beam flux incident on the surface.
  halo_width = 1
  flux_down_sfc = flux_down_direct[:, :, halo_width]

  # The surface source is the direct-beam downard flux that is reflected from
  # the surface.
  sfc_src = sfc_albedo_direct * flux_down_sfc

  srcs_primary = {
      'src_up': src_up,
      'src_down': src_down,
      'flux_down_dir': flux_down_direct,
      'sfc_src': sfc_src,
  }
  return srcs_primary


def _solve_rte_2stream(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_emission: Array,
    sfc_reflectance: Array,
    use_scan: bool | None = None,
) -> StatesMap:
  """Solves the monochromatic two-stream radiative transfer equation.

  Given boundary conditions for the downward flux at the top of the atmosphere
  (`toa_flux_down`) and the upward surface emission (`sfc_emission`), this
  computes the two-stream approximation of the upwelling and downwelling
  radiative fluxes at the cell faces based on the equations of Shonk and Hogan
  (2008).  All the computations here assume a single absorption interval (or 'g'
  interval in RRTM nomenclature).  This function needs to be applied to each 'g'
  interval separately.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward emission, 3D field.
    src_down: Cell center downward emission, 3D field.
    toa_flux_down: The downward component of the incoming flux at the top
      boundary of the atmosphere.  This corresponds to the downward flux at the
      wall of the domain (e.g., the face above the last interior node).  2D
      field.
    sfc_emission: The upward surface emission.  This corresponds to the wall of
      the domain (e.g., the face below the first interior node).  2D field.
    sfc_reflectance: The surface reflectance.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the bottom cell face:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
  """
  # Global recurrent accumulation for the albedo of the atmosphere below a
  # certain level, computed from the surface to the top boundary.  The
  # recurrence relation for albedo is taken from Shonk and Hogan, Eq. 9.

  def albedo_op(
      albedo_below: Array, r_diff: Array, t_diff: Array
  ) -> tuple[Array, Array]:
    """Recurrent formula for albedo solution, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    # AD-safe denominator (restored from commit 59407953): ``1 /
    # jnp.maximum(x, _EPSILON)`` has a ``-1/x**2`` VJP that overflows when
    # ``x`` is at the floor, propagating NaN gradients into every upstream
    # traced parameter whose state path touches the column optical depth
    # (e.g. C_H/C_E via boundary-layer-driven T/q_v perturbations).
    # ``safe_divide`` masks the bad branch before the divide so the
    # backward never differentiates ``1/x`` at tiny ``x``.
    denom = 1 - r_diff * albedo_below
    beta = safe_divide(
        jnp.ones_like(denom), denom, eps=_EPSILON, fill=1.0 / _EPSILON,
    )
    out = r_diff + t_diff**2 * beta * albedo_below
    return out, out  # Carry and output are the same.

  init = sfc_reflectance
  albedo_inputs = {'r_diff': r_diff, 't_diff': t_diff}
  _, albedo = rte_utils.recurrent_op_with_halos(
      albedo_op, init, albedo_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the aggregate upwelling source emission
  # computed from the surface to the top of the atmosphere.  The coefficient and
  # bias terms of the recurrent relation for emission are taken from Shonk and
  # Hogan, Eq. 11.  The upward emission is a combination of 1) the upward source
  # from the grid cell center, 2) aggregate emission from the atmosphere below,
  # transmitted through the cell, and 3) the downward source from the grid cell
  # center that is reflected from the atmosphere below and transmitted up
  # through the layer.

  def upward_emission_op(
      emission_from_below: Array,
      src_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for upward emission, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    # AD-safe denominator (see ``albedo_op`` above, restored from 59407953).
    denom = 1 - r_diff * albedo
    beta = safe_divide(
        jnp.ones_like(denom), denom, eps=_EPSILON, fill=1.0 / _EPSILON,
    )
    out = src_up + t_diff * beta * (emission_from_below + src_down * albedo)
    return out, out  # Carry and output are the same.

  init = sfc_emission
  emission_inputs = {
      'src_up': src_up,
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, emission_up = rte_utils.recurrent_op_with_halos(
      upward_emission_op, init, emission_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the downwelling radiative flux solution at
  # the bottom face, unravelling from the top of the atmosphere down to the
  # surface.  The coefficient and bias terms are taken from Shonk and Hogan,
  # Eq. 13.  The downward flux at the bottom face is a combination of 1) the
  # downward source emitted from the grid cell, 2) the downward flux from the
  # face above transmitted through the cell, and 3) the aggregate upward
  # emissions from the atmosphere below that are reflected from the cell.

  def flux_down_op(
      flux_down_from_above: Array,
      emiss_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for downwelling flux initiating at top boundar."""
    # Geometric series solution accounting for infinite reflection events.
    # AD-safe denominator (see ``albedo_op`` above, restored from 59407953).
    denom = 1 - r_diff * albedo
    beta = safe_divide(
        jnp.ones_like(denom), denom, eps=_EPSILON, fill=1.0 / _EPSILON,
    )
    out = (t_diff * flux_down_from_above + r_diff * emiss_up + src_down) * beta
    return out, out  # Carry and output are the same.

  init = toa_flux_down
  flux_down_inputs = {
      'emiss_up': _shift_up(emission_up),
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, flux_down = rte_utils.recurrent_op_with_halos(
      flux_down_op, init, flux_down_inputs, forward=False, use_scan=use_scan
  )

  # The upwelling radiative flux at the bottom face can now be computed
  # directly from the cumulative upward emissions, the cumulative albedo of
  # the atmosphere below, and the downwelling radiative flux at the same face.
  flux_up = flux_down * _shift_up(albedo) + _shift_up(emission_up)
  fluxes = {
      'flux_up': flux_up,
      'flux_down': flux_down,
  }
  return fluxes


def lw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_src: Array,
    sfc_emissivity: Array,
    use_scan: bool | None = None,
) -> StatesMap:
  """Compute the monochromatic longwave diffusive flux of the atmosphere.

  The upwelling and downwelling fluxes are computed from the equations of Shonk
  and Hogan.  The net flux is also computed at every face.  Note that the net
  flux is computed only at cell faces and does not correspond to the net flux
  into or out of the grid cell.  For the overall grid cell net flux, one must
  take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center Planck upward emission, 3D field.
    src_down: Cell center Planck downward emission, 3D field.
    toa_flux_down: The downward flux at the top boundary of the atmosphere, 2D
      field.
    sfc_src: The surface Planck source, 2D field.
    sfc_emissivity: The surface emissivity, 2D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the cell faces [W/m^2]:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  # The source of diffuse radiation is the surface emission.
  sfc_emission = math.pi * sfc_emissivity * sfc_src
  # The surface reflectance is just the complement of the surface emissivity.
  sfc_reflectance = 1 - sfc_emissivity
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      toa_flux_down,
      sfc_emission,
      sfc_reflectance,
      use_scan,
  )
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes


def sw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    sfc_src: Array,
    sfc_albedo: Array,
    flux_down_dir: Array,
    use_scan: bool | None = None,
) -> StatesMap:
  """Compute the monochromatic shortwave fluxes in a layered atmosphere.

  The direct-beam downward flux `flux_down_dir` is added to the downwelling
  diffuse flux in the final solution.

  The upwelling and downwelling diffuse fluxes are computed from the equations
  of Shonk and Hogan.  The net flux is also computed at every face.  Note that
  the net flux is computed only at cell interfaces and does not correspond to
  the net flux into or out of the cell.  For the overall grid cell net flux, one
  must take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward source, 3D field.
    src_down: Cell center downward source, 3D field.
    sfc_src: Direct-beam shortwave radiation reflected upward from the surface,
      2D field.
    sfc_albedo: The surface albedo, 2D field.
    flux_down_dir: The downwelling direct-beam radiative flux at the cell faces,
      3D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the z faces:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      jnp.zeros_like(sfc_src),
      sfc_src,
      sfc_albedo,
      use_scan,
  )

  # Add the direct-beam contribution to the downwelling flux.
  fluxes['flux_down'] = fluxes['flux_down'] + flux_down_dir

  # The net flux computed at cell faces.
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes
