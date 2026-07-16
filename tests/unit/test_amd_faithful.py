"""Oracle-faithfulness pins for the AMD (Rozema 2015) sub-grid eddy viscosity.

Oracle: Rozema, Bae, Moin & Verstappen (2015), "Minimum-dissipation models for
large-eddy simulation", Phys. Fluids 27, 085107 (anisotropic scaled-gradient form
of Abkar, Bae & Moin 2016).  The single algebraic closed form in
``legoesm.atmosphere.physics.turbulence.amd.amd_nu_t`` is pinned to round-off
(rel 1e-12) against an INDEPENDENT numpy reimplementation typed from the paper
(NOT read back from the module — non-circular):

    S_ij = ½(a_ij + a_ji)                         (a_cd = ∂u_c/∂x_d, the 9 inputs)
    N    = − Σ_k Δ_k² g_kᵀ S g_k,  g_k = (a_1k, a_2k, a_3k)  (k-th COLUMN of a)
    ν_t  = C · max(N, 0) / (a_cd a_cd) + ν_floor   (a_cd a_cd = Σ_cd a_cd², UNscaled)

The scheme has NO hardcoded empirical coefficient (``c_amd``, ``dx/dy/dz`` and
``nu_floor`` are caller inputs), so the pins canary the STRUCTURE: the leading
MINUS sign (minimum-dissipation orientation), the per-direction Δ_k² weighting on
the NUMERATOR only (denominator is unscaled — swapping either diverges), the
COLUMN index g_k=(a_1k,a_2k,a_3k) not the row (∂_k u_i = a_ik), the ``max(N,0)``
clip (never-negative), and the ``_EPS`` rest floor.

AMD vanishing is a N≤0 property, NOT Vreman's rank-1: an off-diagonal 1-D shear
(a_13 only) gives N=0 ⇒ ν_floor, but a single COMPRESSIVE normal strain (a_11<0)
gives N>0 ⇒ ν_t>ν_floor — pinned by ``test_normal_strain_sign_dependence`` (the
defining minimum-dissipation behaviour, not a blanket single-gradient cutoff).

DEPARTURES from the raw closed form (all deliberate, all tested):
  1. ``max(N, 0)``: the minimum-dissipation clip — no SGS dissipation (ν_t=ν_floor)
     where the resolved flow needs none (N≤0), never negative viscosity.  Its grad
     at the N=0 KINK is a finite SUBGRADIENT, not the classical two-sided
     derivative; specifically JAX's ``maximum`` uses a BALANCED tie
     (d max(N,0)/dN = ½ at N=0), so the AD grad there is the AVERAGE of the two
     one-sided derivatives (½·the N>0 slope, since the N<0 slope is 0), pinned by
     ``test_kink_balanced_tie``.  AD-safe, no sqrt trap.
  2. ``_EPS = 1e-30`` in the denominator (aa + _EPS): keeps arg = 0/_EPS = 0
     (⇒ ν_floor) at the aa=0 rest state instead of 0/0 = NaN.  Inert wherever
     aa ≫ 1e-30; the round-off MATCHING tensors all have aa ~ O(1) ≫ _EPS.  The
     oracle DUPLICATES ``+ 1e-30`` and ``test_eps_denominator_functionally_pinned``
     pins the ACTUAL value in the aa≈_EPS regime (a wrong 1e-20 would diverge).
  3. ν_t ≥ ν_floor holds for c_amd ≥ 0; ν_t ≥ 0 / never-negative ALSO needs
     ν_floor ≥ 0 (a caller-supplied negative ν_floor is not validated here).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence import amd
from legoesm.atmosphere.physics.turbulence.amd import amd_nu_t

jax.config.update("jax_enable_x64", True)


def _call(A, dx, dy, dz, c, nu_floor=0.0):
    """Call amd_nu_t with a 3x3 gradient tensor A[c,d] = a_{c+1,d+1}."""
    a = jnp.asarray(A, dtype=float)
    return float(amd_nu_t(
        a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2],
        a[2, 0], a[2, 1], a[2, 2], dx, dy, dz, c, nu_floor))


def _amd_oracle(A, dx, dy, dz, c, nu_floor=0.0, *, sign=-1.0, scale_denom=False,
                clip=True, use_rows=False, eps=1e-30):
    """Independent Rozema-2015 AMD eddy viscosity. Flags override the structure
    for the canaries: ``sign`` (the leading −), ``scale_denom`` (Δ-weight the
    denominator), ``clip`` (max(N,0)), ``use_rows`` (g_k = row k, not column),
    ``eps`` (the denominator floor — duplicates the module's ``_EPS``)."""
    A = np.asarray(A, dtype=float)
    S = 0.5 * (A + A.T)
    cols = A.T if use_rows else A          # g_k = column k of `cols`
    d = np.array([dx ** 2, dy ** 2, dz ** 2])
    N = 0.0
    for k in range(3):
        g = cols[:, k]
        N += sign * d[k] * float(g @ S @ g)
    if scale_denom:
        aa = float(np.sum(d[None, :] * (A * A)))   # WRONG: Δ_d²-weight the denom
    else:
        aa = float(np.sum(A * A))
    Nc = max(N, 0.0) if clip else N
    return c * Nc / (aa + eps) + nu_floor


# Deterministic gradient tensors spanning fully-3-D turbulence (asymmetric, so
# the column-vs-row canary is non-vacuous).
_RNG = np.random.default_rng(20260716)
_TENSORS = [np.round(_RNG.uniform(-2.0, 2.0, size=(3, 3)), 4) for _ in range(6)]
_GRIDS = [(50.0, 50.0, 50.0), (100.0, 100.0, 20.0), (25.0, 75.0, 40.0)]


# ===================== module constant canary =====================

def test_module_eps_floor():
    assert amd._EPS == 1e-30


# ===================== full-tensor round-off pin =====================

@pytest.mark.parametrize("A", _TENSORS)
@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("c,floor", [(0.3, 0.0), (0.3, 0.5), (0.1, 1.0e-3)])
def test_amd_matches_paper(A, grid, c, floor):
    got = _call(A, *grid, c, floor)
    assert got == pytest.approx(_amd_oracle(A, *grid, c, floor), rel=1e-12, abs=1e-14)


# ===================== structure canaries =====================

# _TENSORS[0], [1], [3] give N>0 (ν_t>0) — the regime that exercises the numerator
# structure; [2], [4], [5] give N<0 (ν_t clipped to floor). Canaries pick N>0
# tensors on ANISOTROPIC grids and assert ν_t>0 so the mutation is non-vacuous.

def test_minus_sign_canary():
    # The leading − orients minimum-dissipation. Dropping it (sign=+1) picks the
    # opposite N-regime through max(·,0) ⇒ a different ν_t on a 3-D field.
    A, grid = _TENSORS[0], _GRIDS[1]
    ref = _amd_oracle(A, *grid, 0.3)
    assert ref > 1e-6                                   # N>0 regime (non-vacuous)
    assert _call(A, *grid, 0.3) == pytest.approx(ref, rel=1e-12)
    wrong = _amd_oracle(A, *grid, 0.3, sign=+1.0)
    assert abs(ref - wrong) > 1e-6


def test_anisotropic_numerator_weight_canary():
    # Δ_k² weights the NUMERATOR per direction: swapping Δy↔Δz gives a different
    # ν_t on an anisotropic grid (proves dx/dy/dz map to the right columns).
    A, grid = _TENSORS[1], _GRIDS[2]            # (25, 75, 40) — all Δ distinct
    got = _call(A, *grid, 0.3)
    assert got > 1e-6
    assert got == pytest.approx(_amd_oracle(A, *grid, 0.3), rel=1e-12)
    swapped = _amd_oracle(A, grid[0], grid[2], grid[1], 0.3)   # dy<->dz
    assert abs(got - swapped) > 1e-6


def test_denominator_is_unscaled_canary():
    # The denominator a_cd a_cd is UNscaled by Δ (only the numerator carries Δ_k²).
    # A Δ-weighted denominator gives a different ν_t on an anisotropic grid.
    A, grid = _TENSORS[3], _GRIDS[1]
    ref = _amd_oracle(A, *grid, 0.3)
    assert ref > 1e-6
    assert _call(A, *grid, 0.3) == pytest.approx(ref, rel=1e-12)
    wrong = _amd_oracle(A, *grid, 0.3, scale_denom=True)
    assert abs(ref - wrong) > 1e-6


def test_column_not_row_index_canary():
    # g_k = (a_1k, a_2k, a_3k) is the k-th COLUMN (∂_k u_i = a_ik). Using rows
    # (a_k1,a_k2,a_k3) diverges on an asymmetric tensor + anisotropic grid — pins
    # the index order.
    A, grid = _TENSORS[0], _GRIDS[2]            # (25, 75, 40)
    ref = _amd_oracle(A, *grid, 0.3)
    assert ref > 1e-6
    assert _call(A, *grid, 0.3) == pytest.approx(ref, rel=1e-12)
    wrong = _amd_oracle(A, *grid, 0.3, use_rows=True)
    assert abs(ref - wrong) > 1e-6


def test_max_clip_canary():
    # max(N,0): an expanding normal strain (a_11>0 ⇒ N<0) is clipped to ν_floor;
    # an unclipped form would give NEGATIVE viscosity. Pins the never-negative clip.
    A = np.zeros((3, 3))
    A[0, 0] = 0.4                       # N = -Δ1²·a11³ < 0
    grid = (60.0, 40.0, 30.0)
    assert _call(A, *grid, 0.3, 0.25) == pytest.approx(0.25, abs=1e-14)   # = floor
    unclipped = _amd_oracle(A, *grid, 0.3, 0.25, clip=False)
    assert unclipped < 0.25 - 1e-6                                        # negative + floor


# ===================== AMD vanishing: N<=0, not rank-1 =====================

@pytest.mark.parametrize("comp", ["a13", "a23", "a12", "a21", "a31", "a32"])
def test_offdiagonal_one_d_shear_gives_floor(comp):
    # A pure OFF-DIAGONAL 1-D shear ⇒ N=0 ⇒ ν_t=ν_floor (min-dissipation).
    idx = {"a12": (0, 1), "a13": (0, 2), "a21": (1, 0),
           "a23": (1, 2), "a31": (2, 0), "a32": (2, 1)}[comp]
    A = np.zeros((3, 3))
    A[idx] = 1.3
    floor = 0.2
    assert _call(A, 60.0, 40.0, 30.0, 0.3, floor) == pytest.approx(floor, abs=1e-14)


def test_normal_strain_sign_dependence():
    # UNIQUE to AMD (vs Vreman): a single NORMAL strain is sign-dependent —
    # expanding (a_11>0 ⇒ N<0) gives ν_floor, compressing (a_11<0 ⇒ N>0) gives
    # ν_t>ν_floor, pinned exactly to the oracle.
    grid = (60.0, 40.0, 30.0)
    exp = np.zeros((3, 3)); exp[0, 0] = 0.4
    comp = np.zeros((3, 3)); comp[0, 0] = -0.4
    assert _call(exp, *grid, 0.3, 0.1) == pytest.approx(0.1, abs=1e-14)
    got = _call(comp, *grid, 0.3, 0.1)
    assert got == pytest.approx(_amd_oracle(comp, *grid, 0.3, 0.1), rel=1e-12)
    assert got > 0.1 + 1e-6


# ===================== rest / floor =====================

def test_rest_state_gives_floor():
    assert _call(np.zeros((3, 3)), 50.0, 50.0, 50.0, 0.3, 0.1) == pytest.approx(0.1, abs=1e-15)


def test_nu_never_below_floor():
    floor = 0.3
    for A in _TENSORS:
        for grid in _GRIDS:
            assert _call(A, *grid, 0.3, floor) >= floor - 1e-12


# ===================== differentiability =====================

def test_gradient_matches_fd_on_positive_field():
    # Smooth region (N>0): AD grad matches central FD.
    A = _TENSORS[1]

    def nu_of_entry(a11):
        return amd_nu_t(a11, A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2],
                        A[2, 0], A[2, 1], A[2, 2], 100.0, 100.0, 20.0, 0.3)

    x0 = float(A[0, 0])
    eps = 1e-6
    fd = (float(nu_of_entry(jnp.asarray(x0 + eps)))
          - float(nu_of_entry(jnp.asarray(x0 - eps)))) / (2.0 * eps)
    g = float(jax.grad(nu_of_entry)(jnp.asarray(x0)))
    assert g == pytest.approx(fd, rel=1e-5)


def test_kink_balanced_tie():
    # DEPARTURE #1: at a GENUINE N=0 kink (∇N≠0) JAX's max uses a BALANCED tie,
    # so the AD grad = ½·(left + right one-sided derivatives), NOT 0 and NOT a
    # full one-sided slope. Construct A(s) = (a11=s, a13=v): N = -s·(Δ1²s² +
    # Δ3²v²) is EXACTLY 0 at s=0 (module computes 0.0 for a single off-diagonal),
    # with dN/ds|_0 = -Δ3²v² ≠ 0 — a transversal crossing (N>0 for s<0, clipped
    # for s>0), unlike the flat off-diagonal valley.
    dx, dy, dz, c, v = 50.0, 50.0, 30.0, 0.3, 1.3

    def nu_of_s(s):
        return amd_nu_t(s, 0.0, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        dx, dy, dz, c)

    assert float(nu_of_s(jnp.asarray(0.0))) == 0.0          # N=0 EXACTLY (the tie)
    ad = float(jax.grad(nu_of_s)(jnp.asarray(0.0)))
    h = 1e-6
    right = (float(nu_of_s(jnp.asarray(h))) - float(nu_of_s(jnp.asarray(0.0)))) / h
    left = (float(nu_of_s(jnp.asarray(0.0))) - float(nu_of_s(jnp.asarray(-h)))) / h
    assert right == pytest.approx(0.0, abs=1e-6)            # s>0: N<0, clipped
    assert left < -1.0                                      # s<0: N>0, real slope
    assert ad == pytest.approx(0.5 * (left + right), rel=2e-3)   # balanced ½-tie
    assert ad == pytest.approx(0.5 * left, rel=2e-3)
    assert abs(ad - left) > 0.3 * abs(left) and abs(ad) > 0.3 * abs(left)  # not 0, not full


def test_eps_denominator_functionally_pinned():
    # DEPARTURE #2: in the aa ≈ _EPS regime the denominator floor is LOAD-BEARING,
    # so the pin catches a wrong _EPS (e.g. 1e-20). A tiny compressive normal
    # strain a11=-1e-15 gives aa=1e-30 ≈ _EPS and N=Δ1²·1e-45 > 0.
    A = np.zeros((3, 3))
    A[0, 0] = -1.0e-15
    grid = (100.0, 40.0, 30.0)
    got = _call(A, *grid, 0.3)
    assert got > 0.0                                        # N>0, non-vacuous
    assert got == pytest.approx(_amd_oracle(A, *grid, 0.3), rel=1e-12)   # eps=1e-30
    wrong_eps = _amd_oracle(A, *grid, 0.3, eps=1e-20)       # broken floor
    assert abs(got - wrong_eps) > 0.4 * got                 # materially different


def test_negative_floor_can_go_negative():
    # DEPARTURE #3: "never negative" needs nu_floor >= 0 too. With nu_floor < 0
    # and a no-dissipation field (N<=0 ⇒ ν_t=nu_floor), ν_t is negative — the
    # module does NOT validate nu_floor, so the guarantee is c_amd>=0 AND floor>=0.
    A = np.zeros((3, 3))
    A[0, 0] = 0.4                       # expanding ⇒ N<0 ⇒ clip ⇒ ν_t = nu_floor
    assert _call(A, 50.0, 50.0, 50.0, 0.3, -0.5) == pytest.approx(-0.5, abs=1e-14)
