"""WENO-Z reconstruction kernels for legoESM.

Provides uniform-grid WENO-Z reconstructions at orders 5, 7, and 9 (stencil
widths 6, 8, 10).  Each kernel accepts cell-average stencil values and returns
left-biased (+) and right-biased (-) face reconstructions suitable for upwind
flux splitting.

The module also provides a **smoothness-optimised** variant
``weno_reconstruct_split`` that computes smoothness indicators from one field
*psi* while reconstructing another field *phi*.  This implements the {phi; psi}
stencil of Silvestri et al. (2024, JAMES), Eqs. 39-43.

All kernels use the WENO-Z weight formulation (Borges et al. 2008) which is
better conditioned for reverse-mode AD than the classical Jiang--Shu (JS)
formulation: ``alpha_r = c_r * (1 + tau / (beta_r + eps))`` with exponent p=1,
versus ``alpha_r = c_r / (beta_r + eps)^2`` for JS.

Coefficients are derived from the cell-average primitive-function approach
(Shu 2009) and verified against RRTMGP interpolation.py for WENO5.  WENO7
and WENO9 coefficients are verified via sympy symbolic integration.

References
----------
- Jiang & Shu 1996, JCP 126 (smoothness indicators, reconstruction coefficients)
- Borges et al. 2008, JCP 227 (WENO-Z weights)
- Shu 2009, "High Order WENO and DG Methods" (cell-average reconstruction)
- Silvestri et al. 2024, JAMES (smoothness-optimised {phi; psi} variant)
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Epsilon defaults — chosen for AD safety (see plan Section 3.3)
# ---------------------------------------------------------------------------
EPS_F64 = 1e-16
EPS_F32 = 1e-7


def _default_eps(x: jnp.ndarray) -> float:
    """Return dtype-appropriate epsilon."""
    if jnp.result_type(x) == jnp.float64:
        return EPS_F64
    return EPS_F32


# ===================================================================
#  WENO-Z weight computation (shared across all orders)
# ===================================================================

def _weno_z_weights(
    betas: list,
    optimal_weights: tuple,
    tau: jnp.ndarray,
    epsilon: float,
) -> list:
    """Compute normalised WENO-Z nonlinear weights."""
    alphas = [c * (1.0 + tau / (beta + epsilon))
              for c, beta in zip(optimal_weights, betas)]
    alpha_sum = sum(alphas)
    return [a / alpha_sum for a in alphas]


# ===================================================================
#  Reconstruction coefficients (cell-average -> face value)
# ===================================================================
#
# Derived via primitive-function Lagrange interpolation (Shu 2009).
# Left-biased sub-stencil r at face x_{j+1/2} uses cells {j-k+1+r,...,j+r}.
# Right-biased sub-stencil r uses cells {j+1-r,...,j+k-r}.

# --- WENO5 (k=3) ---
_RECON5_L = (
    (1/3, -7/6, 11/6),         # r=0
    (-1/6, 5/6, 1/3),          # r=1
    (1/3, 5/6, -1/6),          # r=2
)
_RECON5_R = (
    (11/6, -7/6, 1/3),         # r=0
    (1/3, 5/6, -1/6),          # r=1
    (-1/6, 5/6, 1/3),          # r=2
)
_C5 = (0.1, 0.6, 0.3)

# --- WENO7 (k=4) ---
_RECON7_L = (
    (-1/4, 13/12, -23/12, 25/12),
    (1/12, -5/12, 13/12, 1/4),
    (-1/12, 7/12, 7/12, -1/12),
    (1/4, 13/12, -5/12, 1/12),
)
_RECON7_R = (
    (25/12, -23/12, 13/12, -1/4),
    (1/4, 13/12, -5/12, 1/12),
    (-1/12, 7/12, 7/12, -1/12),
    (1/12, -5/12, 13/12, 1/4),
)
_C7 = (1/35, 12/35, 18/35, 4/35)

# --- WENO9 (k=5) ---
_RECON9_L = (
    (1/5, -21/20, 137/60, -163/60, 137/60),
    (-1/20, 17/60, -43/60, 77/60, 1/5),
    (1/30, -13/60, 47/60, 9/20, -1/20),
    (-1/20, 9/20, 47/60, -13/60, 1/30),
    (1/5, 77/60, -43/60, 17/60, -1/20),
)
_RECON9_R = (
    (137/60, -163/60, 137/60, -21/20, 1/5),
    (1/5, 77/60, -43/60, 17/60, -1/20),
    (-1/20, 9/20, 47/60, -13/60, 1/30),
    (1/30, -13/60, 47/60, 9/20, -1/20),
    (-1/20, 17/60, -43/60, 77/60, 1/5),
)
_C9 = (1/126, 10/63, 10/21, 20/63, 5/126)


# ===================================================================
#  Smoothness indicators
# ===================================================================
#
# Beta coefficients stored as tuples of (i, j, coeff) where
# beta = sum c * s[i] * s[j] over the sub-stencil values s.
#
# Left and right betas differ because the face position relative to
# the sub-stencil differs.  Computed via Jiang-Shu integral formula
# applied to the primitive-function reconstruction polynomial.

# --- WENO5 left betas ---
_BETA5_L = (
    ((0,0, 4/3), (0,1, -19/3), (0,2, 11/3),
     (1,1, 25/3), (1,2, -31/3), (2,2, 10/3)),
    ((0,0, 4/3), (0,1, -13/3), (0,2, 5/3),
     (1,1, 13/3), (1,2, -13/3), (2,2, 4/3)),
    ((0,0, 10/3), (0,1, -31/3), (0,2, 11/3),
     (1,1, 25/3), (1,2, -19/3), (2,2, 4/3)),
)

# --- WENO5 right betas ---
_BETA5_R = (
    ((0,0, 22/3), (0,1, -73/3), (0,2, 29/3),
     (1,1, 61/3), (1,2, -49/3), (2,2, 10/3)),
    ((0,0, 10/3), (0,1, -31/3), (0,2, 11/3),
     (1,1, 25/3), (1,2, -19/3), (2,2, 4/3)),
    ((0,0, 4/3), (0,1, -13/3), (0,2, 5/3),
     (1,1, 13/3), (1,2, -13/3), (2,2, 4/3)),
)

# --- WENO7 left betas ---
_BETA7_L = (
    ((0,0, 547/240), (0,1, -3882/240), (0,2, 4642/240), (0,3, -1854/240),
     (1,1, 7043/240), (1,2, -17246/240), (1,3, 7042/240),
     (2,2, 11003/240), (2,3, -9402/240), (3,3, 2107/240)),
    ((0,0, 267/240), (0,1, -1642/240), (0,2, 1602/240), (0,3, -494/240),
     (1,1, 2843/240), (1,2, -5966/240), (1,3, 1922/240),
     (2,2, 3443/240), (2,3, -2522/240), (3,3, 547/240)),
    ((0,0, 547/240), (0,1, -2522/240), (0,2, 1922/240), (0,3, -494/240),
     (1,1, 3443/240), (1,2, -5966/240), (1,3, 1602/240),
     (2,2, 2843/240), (2,3, -1642/240), (3,3, 267/240)),
    ((0,0, 2107/240), (0,1, -9402/240), (0,2, 7042/240), (0,3, -1854/240),
     (1,1, 11003/240), (1,2, -17246/240), (1,3, 4642/240),
     (2,2, 7043/240), (2,3, -3882/240), (3,3, 547/240)),
)

# --- WENO7 right betas ---
_BETA7_R = (
    ((0,0, 7107/240), (0,1, -33802/240), (0,2, 27042/240), (0,3, -7454/240),
     (1,1, 40643/240), (1,2, -65726/240), (1,3, 18242/240),
     (2,2, 26843/240), (2,3, -15002/240), (3,3, 2107/240)),
    ((0,0, 2107/240), (0,1, -9402/240), (0,2, 7042/240), (0,3, -1854/240),
     (1,1, 11003/240), (1,2, -17246/240), (1,3, 4642/240),
     (2,2, 7043/240), (2,3, -3882/240), (3,3, 547/240)),
    ((0,0, 547/240), (0,1, -2522/240), (0,2, 1922/240), (0,3, -494/240),
     (1,1, 3443/240), (1,2, -5966/240), (1,3, 1602/240),
     (2,2, 2843/240), (2,3, -1642/240), (3,3, 267/240)),
    ((0,0, 267/240), (0,1, -1642/240), (0,2, 1602/240), (0,3, -494/240),
     (1,1, 2843/240), (1,2, -5966/240), (1,3, 1922/240),
     (2,2, 3443/240), (2,3, -2522/240), (3,3, 547/240)),
)

# --- WENO9 left betas ---
_BETA9_L = (
    ((0,0, 22658/5040), (0,1, -208501/5040), (0,2, 364863/5040),
     (0,3, -288007/5040), (0,4, 86329/5040),
     (1,1, 482963/5040), (1,2, -1704396/5040), (1,3, 1358458/5040),
     (1,4, -411487/5040),
     (2,2, 1521393/5040), (2,3, -2462076/5040), (2,4, 758823/5040),
     (3,3, 1020563/5040), (3,4, -649501/5040),
     (4,4, 107918/5040)),
    ((0,0, 6908/5040), (0,1, -60871/5040), (0,2, 99213/5040),
     (0,3, -70237/5040), (0,4, 18079/5040),
     (1,1, 138563/5040), (1,2, -464976/5040), (1,3, 337018/5040),
     (1,4, -88297/5040),
     (2,2, 406293/5040), (2,3, -611976/5040), (2,4, 165153/5040),
     (3,3, 242723/5040), (3,4, -140251/5040),
     (4,4, 22658/5040)),
    ((0,0, 6908/5040), (0,1, -51001/5040), (0,2, 67923/5040),
     (0,3, -38947/5040), (0,4, 8209/5040),
     (1,1, 104963/5040), (1,2, -299076/5040), (1,3, 179098/5040),
     (1,4, -38947/5040),
     (2,2, 231153/5040), (2,3, -299076/5040), (2,4, 67923/5040),
     (3,3, 104963/5040), (3,4, -51001/5040),
     (4,4, 6908/5040)),
    ((0,0, 22658/5040), (0,1, -140251/5040), (0,2, 165153/5040),
     (0,3, -88297/5040), (0,4, 18079/5040),
     (1,1, 242723/5040), (1,2, -611976/5040), (1,3, 337018/5040),
     (1,4, -70237/5040),
     (2,2, 406293/5040), (2,3, -464976/5040), (2,4, 99213/5040),
     (3,3, 138563/5040), (3,4, -60871/5040),
     (4,4, 6908/5040)),
    ((0,0, 107918/5040), (0,1, -649501/5040), (0,2, 758823/5040),
     (0,3, -411487/5040), (0,4, 86329/5040),
     (1,1, 1020563/5040), (1,2, -2462076/5040), (1,3, 1358458/5040),
     (1,4, -288007/5040),
     (2,2, 1521393/5040), (2,3, -1704396/5040), (2,4, 364863/5040),
     (3,3, 482963/5040), (3,4, -208501/5040),
     (4,4, 22658/5040)),
)

# --- WENO9 right betas ---
_BETA9_R = (
    ((0,0, 471008/5040), (0,1, -2964751/5040), (0,2, 3597813/5040),
     (0,3, -2004757/5040), (0,4, 429679/5040),
     (1,1, 4724963/5040), (1,2, -11584896/5040), (1,3, 6499258/5040),
     (1,4, -1399537/5040),
     (2,2, 7159893/5040), (2,3, -8079576/5040), (2,4, 1746873/5040),
     (3,3, 2288963/5040), (3,4, -992851/5040),
     (4,4, 107918/5040)),
    ((0,0, 107918/5040), (0,1, -649501/5040), (0,2, 758823/5040),
     (0,3, -411487/5040), (0,4, 86329/5040),
     (1,1, 1020563/5040), (1,2, -2462076/5040), (1,3, 1358458/5040),
     (1,4, -288007/5040),
     (2,2, 1521393/5040), (2,3, -1704396/5040), (2,4, 364863/5040),
     (3,3, 482963/5040), (3,4, -208501/5040),
     (4,4, 22658/5040)),
    ((0,0, 22658/5040), (0,1, -140251/5040), (0,2, 165153/5040),
     (0,3, -88297/5040), (0,4, 18079/5040),
     (1,1, 242723/5040), (1,2, -611976/5040), (1,3, 337018/5040),
     (1,4, -70237/5040),
     (2,2, 406293/5040), (2,3, -464976/5040), (2,4, 99213/5040),
     (3,3, 138563/5040), (3,4, -60871/5040),
     (4,4, 6908/5040)),
    ((0,0, 6908/5040), (0,1, -51001/5040), (0,2, 67923/5040),
     (0,3, -38947/5040), (0,4, 8209/5040),
     (1,1, 104963/5040), (1,2, -299076/5040), (1,3, 179098/5040),
     (1,4, -38947/5040),
     (2,2, 231153/5040), (2,3, -299076/5040), (2,4, 67923/5040),
     (3,3, 104963/5040), (3,4, -51001/5040),
     (4,4, 6908/5040)),
    ((0,0, 6908/5040), (0,1, -60871/5040), (0,2, 99213/5040),
     (0,3, -70237/5040), (0,4, 18079/5040),
     (1,1, 138563/5040), (1,2, -464976/5040), (1,3, 337018/5040),
     (1,4, -88297/5040),
     (2,2, 406293/5040), (2,3, -611976/5040), (2,4, 165153/5040),
     (3,3, 242723/5040), (3,4, -140251/5040),
     (4,4, 22658/5040)),
)


def _beta_from_coeffs(s: list, coeffs: tuple) -> jnp.ndarray:
    """Compute smoothness indicator from coefficient matrix."""
    beta = 0.0
    for i, j, c in coeffs:
        beta = beta + c * s[i] * s[j]
    return beta


def _compute_betas_left(f: list, k: int, beta_table: tuple) -> list:
    """Left-biased smoothness indicators. Sub-stencil r uses f[r:r+k]."""
    return [_beta_from_coeffs(f[r:r + k], beta_table[r]) for r in range(k)]


def _compute_betas_right(f: list, k: int, beta_table: tuple) -> list:
    """Right-biased smoothness indicators. Sub-stencil r uses f[k-r:2k-r]."""
    return [_beta_from_coeffs(f[k - r:2 * k - r], beta_table[r])
            for r in range(k)]


# ===================================================================
#  Sub-stencil reconstruction
# ===================================================================

def _reconstruct_left(coeffs: tuple, f: list) -> list:
    """Left-biased reconstructions. Sub-stencil r uses f[r:r+k]."""
    k = len(coeffs[0])
    return [sum(c * f[r + j] for j, c in enumerate(cs))
            for r, cs in enumerate(coeffs)]


def _reconstruct_right(coeffs: tuple, f: list, k: int) -> list:
    """Right-biased reconstructions. Sub-stencil r uses f[k-r:2k-r]."""
    return [sum(c * f[k - r + j] for j, c in enumerate(cs))
            for r, cs in enumerate(coeffs)]


def _tau_z(betas: list) -> jnp.ndarray:
    """Global smoothness indicator: |beta_0 - beta_{k-1}|."""
    return jnp.abs(betas[0] - betas[-1])


# ===================================================================
#  Internal dispatch helper
# ===================================================================

_CONFIGS = {
    5: (_RECON5_L, _RECON5_R, _BETA5_L, _BETA5_R, _C5, 3),
    7: (_RECON7_L, _RECON7_R, _BETA7_L, _BETA7_R, _C7, 4),
    9: (_RECON9_L, _RECON9_R, _BETA9_L, _BETA9_R, _C9, 5),
}


def _weno_z_core(f: list, order: int, epsilon: float) -> tuple:
    """Core WENO-Z computation for a given order."""
    if order not in _CONFIGS:
        raise ValueError(f"Unsupported WENO order {order}; must be 5, 7, or 9")
    recon_l, recon_r, beta_l_tab, beta_r_tab, copt, k = _CONFIGS[order]

    betas_l = _compute_betas_left(f, k, beta_l_tab)
    betas_r = _compute_betas_right(f, k, beta_r_tab)

    w_l = _weno_z_weights(betas_l, copt, _tau_z(betas_l), epsilon)
    w_r = _weno_z_weights(betas_r, copt, _tau_z(betas_r), epsilon)

    rec_l = _reconstruct_left(recon_l, f)
    rec_r = _reconstruct_right(recon_r, f, k)

    f_plus = sum(w * r for w, r in zip(w_l, rec_l))
    f_minus = sum(w * r for w, r in zip(w_r, rec_r))
    return f_plus, f_minus


# ===================================================================
#  Public API
# ===================================================================

def weno5_z(
    stencil: list | tuple,
    epsilon: float | None = None,
) -> tuple:
    """WENO5-Z left/right reconstruction at face i+1/2.

    Parameters
    ----------
    stencil : sequence of 6 arrays
        Cell-average values [f_{i-2}, f_{i-1}, f_i, f_{i+1}, f_{i+2}, f_{i+3}].
    epsilon : float, optional
        Regularisation constant.  Defaults to dtype-aware value.

    Returns
    -------
    (f_plus, f_minus) : tuple of arrays
        Left-biased and right-biased reconstructions at face i+1/2.
    """
    f = list(stencil)
    if len(f) != 6:
        raise ValueError(f"WENO5 requires 6 stencil values, got {len(f)}")
    if epsilon is None:
        epsilon = _default_eps(f[0])
    return _weno_z_core(f, 5, epsilon)


def weno7_z(
    stencil: list | tuple,
    epsilon: float | None = None,
) -> tuple:
    """WENO7-Z left/right reconstruction at face i+1/2.

    Parameters
    ----------
    stencil : sequence of 8 arrays
        Cell-average values [f_{i-3}, ..., f_{i+4}].
    epsilon : float, optional
        Regularisation constant.

    Returns
    -------
    (f_plus, f_minus) : tuple of arrays
        Left-biased and right-biased reconstructions at face i+1/2.
    """
    f = list(stencil)
    if len(f) != 8:
        raise ValueError(f"WENO7 requires 8 stencil values, got {len(f)}")
    if epsilon is None:
        epsilon = _default_eps(f[0])
    return _weno_z_core(f, 7, epsilon)


def weno9_z(
    stencil: list | tuple,
    epsilon: float | None = None,
) -> tuple:
    """WENO9-Z left/right reconstruction at face i+1/2.

    Parameters
    ----------
    stencil : sequence of 10 arrays
        Cell-average values [f_{i-4}, ..., f_{i+5}].
    epsilon : float, optional
        Regularisation constant.

    Returns
    -------
    (f_plus, f_minus) : tuple of arrays
        Left-biased and right-biased reconstructions at face i+1/2.
    """
    f = list(stencil)
    if len(f) != 10:
        raise ValueError(f"WENO9 requires 10 stencil values, got {len(f)}")
    if epsilon is None:
        epsilon = _default_eps(f[0])
    return _weno_z_core(f, 9, epsilon)


# ===================================================================
#  Smoothness-optimised {phi; psi} variant
# ===================================================================

def weno_reconstruct_split(
    phi_stencil: list | tuple,
    psi_stencil: list | tuple,
    order: int = 5,
    epsilon: float | None = None,
) -> tuple:
    """Smoothness-optimised WENO-Z reconstruction {phi; psi}.

    Computes smoothness indicators from *psi* (a smoother field, typically
    velocity) but reconstructs *phi* (the target field, typically vorticity
    or divergence).  This is the key innovation of Silvestri et al. (2024),
    Eqs. 39-43.

    Parameters
    ----------
    phi_stencil : sequence of arrays
        Stencil values of the field to reconstruct.
    psi_stencil : sequence of arrays
        Stencil values of the field used for smoothness assessment.
        Must be same length as *phi_stencil*.
    order : {5, 7, 9}
        WENO order.
    epsilon : float, optional
        Regularisation constant.

    Returns
    -------
    (f_plus, f_minus) : tuple of arrays
        Left-biased and right-biased reconstructions of *phi* at i+1/2.
    """
    phi = list(phi_stencil)
    psi = list(psi_stencil)
    if order not in _CONFIGS:
        raise ValueError(f"Unsupported WENO order {order}; must be 5, 7, or 9")
    recon_l, recon_r, beta_l_tab, beta_r_tab, copt, k = _CONFIGS[order]
    expected_len = 2 * k
    if len(phi) != expected_len:
        raise ValueError(f"WENO{order} requires {expected_len} stencil values, "
                         f"got {len(phi)}")
    if len(psi) != expected_len:
        raise ValueError(f"psi stencil length {len(psi)} != expected {expected_len}")
    if epsilon is None:
        epsilon = _default_eps(phi[0])

    # Weights from psi smoothness
    betas_l = _compute_betas_left(psi, k, beta_l_tab)
    betas_r = _compute_betas_right(psi, k, beta_r_tab)
    w_l = _weno_z_weights(betas_l, copt, _tau_z(betas_l), epsilon)
    w_r = _weno_z_weights(betas_r, copt, _tau_z(betas_r), epsilon)

    # Reconstruction from phi values
    rec_l = _reconstruct_left(recon_l, phi)
    rec_r = _reconstruct_right(recon_r, phi, k)

    f_plus = sum(w * r for w, r in zip(w_l, rec_l))
    f_minus = sum(w * r for w, r in zip(w_r, rec_r))
    return f_plus, f_minus


# ===================================================================
#  Convenience: upwind flux from left/right reconstructions
# ===================================================================

def weno_upwind(
    f_plus: jnp.ndarray,
    f_minus: jnp.ndarray,
    velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Select upwind reconstruction based on velocity sign.

    Parameters
    ----------
    f_plus : array
        Left-biased (positive velocity) reconstruction.
    f_minus : array
        Right-biased (negative velocity) reconstruction.
    velocity : array
        Advecting velocity at the face.

    Returns
    -------
    array
        Upwind face value.
    """
    return jnp.where(velocity >= 0, f_plus, f_minus)
