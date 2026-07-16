"""Oracle-faithfulness pins for the Vreman (2004) sub-grid eddy viscosity.

Oracle: Vreman, A.W. (2004), "An eddy-viscosity subgrid-scale model for turbulent
shear flow: Algebraic theory and applications", Phys. Fluids 16, 3670
(doi:10.1063/1.1785131), Eqs. 5-7.  The single algebraic closed form in
``legoesm.atmosphere.physics.turbulence.vreman.vreman_nu_t`` is pinned to
round-off (rel 1e-12) against an INDEPENDENT numpy reimplementation typed from
the paper (NOT read back from the module — non-circular):

    α_ij = ∂u_j/∂x_i = a_ji                      (a_cd = ∂u_c/∂x_d, the 9 inputs)
    β_ij = Σ_m Δ_m² α_mi α_mj = Σ_m Δ_m² a_im a_jm      (Δ_m = Δx, Δy, Δz)
    Bβ   = β11β22 − β12² + β11β33 − β13² + β22β33 − β23²
    ν_t  = c · √( max(Bβ, 0) / (a_ij a_ij) ) + ν_floor

The scheme has NO hardcoded empirical coefficient (``c_vreman``, the filter widths
``dx/dy/dz`` and ``nu_floor`` are all caller inputs), so the pins canary the
STRUCTURE: the specific Bβ principal-minor combination (a wrong sign or a dropped
term diverges), the per-direction Δ_m² weighting (swapping Δy↔Δz diverges on an
anisotropic grid), and the ``_EPS`` rest-floor.  Complements
``test_vreman_sgs_plane.py`` (behavioral: 1-D shear ⇒ ν_floor, ν_t ≥ ν_floor,
finite gradient at rest) with the full-tensor round-off pin + structure canaries.

Rank ≤ 1, not "2-D": for filter widths Δ_m>0, Bβ = 0 ⟺ rank(a) ≤ 1 (β = M Mᵀ
with M_im = Δ_m a_im, and Bβ is the sum of β's three 2×2 principal minors, so
Bβ=0 ⟺ rank β ≤ 1 ⟺ rank a ≤ 1).  So (for c>0 and Δ_m>0) ν_t = ν_floor only
for rank ≤ 1 (rest, rank 0, or a laminar 1-D shear, rank 1); a rank-2 "2-D" flow
diag(∂u/∂x,∂v/∂y,0) gives Bβ = Δx²Δy²(∂u/∂x)²(∂v/∂y)² > 0, i.e. ν_t > ν_floor
(pinned by ``test_rank2_flow_does_not_vanish``).

DEPARTURES from the raw closed form (all deliberate, all tested):
  1. **AD-safe double-where √ (a SUBGRADIENT, not the true derivative)**: ``√x``
     has infinite slope at x=0, so a rank-1 point (Bβ=0 ⇒ arg=0) would NaN the
     adjoint.  The module masks arg≤0 to ν_t=ν_floor with a finite (0) gradient.
     But the raw √(Bβ/aa) is NON-smooth at Bβ=0: with A=diag(1,t,0) (Δ=1),
     ν_t = c·|t|/√(1+t²) — one-sided slopes ∓c at t=0 — so the returned 0 is a
     valid SUBGRADIENT, not the classical (non-existent) derivative.  Pinned:
     value everywhere + the finite 0-subgradient at the kink (both the on-kink
     subgradient and an off-kink smooth FD match in the differentiability tests).
  2. ``max(Bβ, 0)``: Bβ is a sum of 2×2 principal minors of the PSD tensor β, so
     Bβ ≥ 0 for any real gradient tensor; the clamp is a numerical safety guard
     (inert for physical inputs, keeps ν_t ≥ ν_floor for c ≥ 0).
  3. ``_EPS = 1e-30`` in the denominator (aa + _EPS): keeps arg = 0/_EPS = 0
     (⇒ ν_floor) at the aa=0 rest state instead of 0/0 = NaN.  Inert wherever
     aa ≫ 1e-30; at a PATHOLOGICALLY tiny gradient (aa ≲ 1e-30, |a| ≲ 1e-15 1/s)
     the +_EPS measurably shrinks arg, a departure from raw Eq. 5.  The oracle
     DUPLICATES ``aa + 1e-30`` so it pins the module's regularised form exactly
     — INCLUDING the aa=0 rest state (``test_rest_state_gives_floor``), where
     _EPS is load-bearing.  The round-off MATCHING tensors all have aa ~ O(1) ≫
     _EPS, so those pins probe raw Eq. 5, not the regulariser.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence import vreman
from legoesm.atmosphere.physics.turbulence.vreman import vreman_nu_t

jax.config.update("jax_enable_x64", True)


def _call(A, dx, dy, dz, c, nu_floor=0.0):
    """Call vreman_nu_t with a 3x3 gradient tensor A[c,d] = a_{c+1,d+1}."""
    a = jnp.asarray(A, dtype=float)
    return float(vreman_nu_t(
        a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2],
        a[2, 0], a[2, 1], a[2, 2], dx, dy, dz, c, nu_floor))


def _vreman_oracle(A, dx, dy, dz, c, nu_floor=0.0, *, bbeta_terms=None):
    """Independent Vreman 2004 eddy viscosity. ``bbeta_terms`` overrides the six
    (sign, i, j, k, l) Bβ minor terms for the structure canary."""
    A = np.asarray(A, dtype=float)
    d = np.array([dx ** 2, dy ** 2, dz ** 2])
    b = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            b[i, j] = sum(d[m] * A[i, m] * A[j, m] for m in range(3))
    if bbeta_terms is None:
        bbeta_terms = [
            (+1, 0, 0, 1, 1), (-1, 0, 1, 0, 1),   # b11 b22 - b12^2
            (+1, 0, 0, 2, 2), (-1, 0, 2, 0, 2),   # b11 b33 - b13^2
            (+1, 1, 1, 2, 2), (-1, 1, 2, 1, 2),   # b22 b33 - b23^2
        ]
    bbeta = sum(s * b[i, j] * b[k, ll] for (s, i, j, k, ll) in bbeta_terms)
    aa = float(np.sum(A * A))
    arg = max(bbeta, 0.0) / (aa + 1e-30)
    return c * (math.sqrt(arg) if arg > 0.0 else 0.0) + nu_floor


# Deterministic gradient tensors spanning fully-3-D turbulence.
_RNG = np.random.default_rng(20260716)
_TENSORS = [np.round(_RNG.uniform(-2.0, 2.0, size=(3, 3)), 4) for _ in range(6)]
_GRIDS = [(50.0, 50.0, 50.0), (100.0, 100.0, 20.0), (25.0, 75.0, 40.0)]


# ===================== module constant canary =====================

def test_module_eps_floor():
    assert vreman._EPS == 1e-30


# ===================== full-tensor round-off pin =====================

@pytest.mark.parametrize("A", _TENSORS)
@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("c,floor", [(0.07, 0.0), (0.07, 0.5), (0.025, 1.0e-3)])
def test_vreman_matches_paper(A, grid, c, floor):
    got = _call(A, *grid, c, floor)
    assert got == pytest.approx(_vreman_oracle(A, *grid, c, floor), rel=1e-12, abs=1e-14)


# ===================== Bβ structure canary =====================

def test_bbeta_minor_structure_canary():
    # A wrong Bβ (e.g. +β12² instead of −β12², or dropping the β22β33 minor) yields
    # a different ν_t. Proves the pin canaries the exact principal-minor combination.
    A = _TENSORS[0]
    grid = _GRIDS[1]   # anisotropic
    ref = _vreman_oracle(A, *grid, 0.07)
    assert _call(A, *grid, 0.07) == pytest.approx(ref, rel=1e-12)
    # flip the sign of the b12^2 term
    flipped = [
        (+1, 0, 0, 1, 1), (+1, 0, 1, 0, 1),   # + b12^2 (WRONG sign)
        (+1, 0, 0, 2, 2), (-1, 0, 2, 0, 2),
        (+1, 1, 1, 2, 2), (-1, 1, 2, 1, 2),
    ]
    wrong_flip = _vreman_oracle(A, *grid, 0.07, bbeta_terms=flipped)
    assert abs(ref - wrong_flip) > 1e-6
    # drop the b22 b33 minor entirely
    dropped = [
        (+1, 0, 0, 1, 1), (-1, 0, 1, 0, 1),
        (+1, 0, 0, 2, 2), (-1, 0, 2, 0, 2),
        (-1, 1, 2, 1, 2),
    ]
    wrong_drop = _vreman_oracle(A, *grid, 0.07, bbeta_terms=dropped)
    assert abs(ref - wrong_drop) > 1e-6


def test_anisotropic_filter_width_canary():
    # The per-direction Δ_m² weighting matters: on an anisotropic grid, swapping
    # Δy↔Δz gives a different ν_t (proves dx/dy/dz map to the right β terms).
    A = _TENSORS[2]
    got = _call(A, 100.0, 100.0, 20.0, 0.07)
    assert got == pytest.approx(_vreman_oracle(A, 100.0, 100.0, 20.0, 0.07), rel=1e-12)
    swapped = _vreman_oracle(A, 100.0, 20.0, 100.0, 0.07)   # dy<->dz
    assert abs(got - swapped) > 1e-6


# ===================== departures: laminar / 1-D / rest =====================

@pytest.mark.parametrize("comp", range(9))
def test_one_dimensional_shear_gives_floor(comp):
    # DEPARTURE #1/#2: a single non-zero gradient ⇒ rank-1 α ⇒ Bβ = 0 ⇒
    # ν_t = ν_floor (Vreman's rank-1 property; the √ guard returns the floor
    # exactly). NB rank-2 flow does NOT vanish — see test_rank2_flow_* below.
    A = np.zeros((3, 3))
    A[comp // 3, comp % 3] = 1.7
    floor = 0.25
    assert _call(A, 60.0, 40.0, 30.0, 0.07, floor) == pytest.approx(floor, abs=1e-14)


def test_rank2_flow_does_not_vanish():
    # Canary for the corrected "rank ≤ 1, not 2-D" docstring: a rank-2 (genuinely
    # "two-dimensional") flow diag(a,b,0) has Bβ = Δx²Δy² a²b² > 0, so ν_t is
    # strictly above the floor — only rank ≤ 1 vanishes.  Directly pins the
    # codex counterexample diag(1,1,0) → ν_t = c/√2 (Δ=1) ≠ 0.
    A = np.diag([1.3, 0.9, 0.0])
    grid = (100.0, 100.0, 20.0)
    got = _call(A, *grid, 0.07, 0.05)
    assert got == pytest.approx(_vreman_oracle(A, *grid, 0.07, 0.05), rel=1e-12)
    assert got > 0.05 + 1e-6   # strictly above ν_floor
    # unit-Δ reduction: diag(1,1,0) ⇒ Bβ=1, aa=2 ⇒ ν_t = c/√2 (no floor)
    iso = _call(np.diag([1.0, 1.0, 0.0]), 1.0, 1.0, 1.0, 0.07)
    assert iso == pytest.approx(0.07 / math.sqrt(2.0), rel=1e-12)


def test_zero_c_gives_floor_even_for_3d_strain():
    # Makes the "for c>0" caveat non-vacuous: with c_vreman=0 the model
    # degenerates to ν_t = ν_floor for EVERY tensor (incl. a fully-3-D strain),
    # so the "rank ≤ 1 ⇒ ν_floor" property is a c>0 statement.
    floor = 0.2
    for A in (_TENSORS[0], np.diag([1.0, 1.0, 0.0]), np.diag([1.0, 2.0, 3.0])):
        assert _call(A, 50.0, 50.0, 50.0, 0.0, floor) == pytest.approx(floor, abs=1e-14)


def test_rest_state_gives_floor():
    # DEPARTURE #3: a_ij a_ij = 0 ⇒ arg = 0/_EPS = 0 ⇒ ν_t = ν_floor (no 0/0 NaN).
    assert _call(np.zeros((3, 3)), 50.0, 50.0, 50.0, 0.07, 0.1) == pytest.approx(0.1, abs=1e-15)


def test_nu_t_never_below_floor():
    floor = 0.3
    for A in _TENSORS:
        for grid in _GRIDS:
            assert _call(A, *grid, 0.07, floor) >= floor - 1e-12


# ===================== differentiability =====================

def test_gradient_finite_at_rest_and_matches_fd():
    # AD-safe √: finite (zero) gradient at the laminar/rest singularity.
    def nu_of_a12(a12):
        return vreman_nu_t(0.0, a12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           50.0, 50.0, 50.0, 0.07)   # single gradient ⇒ arg=0

    g0 = float(jax.grad(nu_of_a12)(jnp.asarray(1.3)))
    assert math.isfinite(g0) and g0 == pytest.approx(0.0, abs=1e-12)   # Bβ≡0 line

    # FD check on a fully-3-D turbulent point (arg>0, smooth).
    A = _TENSORS[1]

    def nu_of_entry(a12):
        return vreman_nu_t(
            A[0, 0], a12, A[0, 2], A[1, 0], A[1, 1], A[1, 2],
            A[2, 0], A[2, 1], A[2, 2], 100.0, 100.0, 20.0, 0.07)

    x0 = float(A[0, 1])
    eps = 1e-6
    fd = (float(nu_of_entry(jnp.asarray(x0 + eps)))
          - float(nu_of_entry(jnp.asarray(x0 - eps)))) / (2.0 * eps)
    g = float(jax.grad(nu_of_entry)(jnp.asarray(x0)))
    assert g == pytest.approx(fd, rel=1e-5)


def test_kink_subgradient_is_zero_not_one_sided_derivative():
    # DEPARTURE #1 honesty: at the Bβ=0 kink the double-where returns a finite
    # SUBGRADIENT (0), NOT the classical derivative (which does not exist). With
    # A=diag(1,t,0) (Δ=1), ν_t(t) = c·|t|/√(1+t²): the one-sided slopes are ∓c at
    # t=0 but AD returns 0 — a valid subgradient (0 ∈ [-c,c]), AD-safe, no NaN.
    c = 0.07

    def nu_of_t(t):
        return vreman_nu_t(1.0, 0.0, 0.0, 0.0, t, 0.0, 0.0, 0.0, 0.0,
                           1.0, 1.0, 1.0, c)

    g0 = float(jax.grad(nu_of_t)(jnp.asarray(0.0)))
    assert math.isfinite(g0) and g0 == 0.0            # subgradient, not ±c
    # the TRUE one-sided slopes are ±c, which AD deliberately does NOT return:
    h = 1e-6
    n0 = float(nu_of_t(jnp.asarray(0.0)))
    right = (float(nu_of_t(jnp.asarray(h))) - n0) / h
    left = (n0 - float(nu_of_t(jnp.asarray(-h)))) / h
    assert right == pytest.approx(c, abs=1e-4)         # +c
    assert left == pytest.approx(-c, abs=1e-4)         # -c
    assert abs(g0 - right) > 0.06 and abs(g0 - left) > 0.06   # AD ≠ one-sided
