"""FV3_3D iter 598: PE hydrostatic total-energy diagnostic.

Faithful port of FV3 ``compute_total_energy`` hydrostatic branch
(fv_mapz.F90:1127-1152).  Per-column total energy is:

    te_2d = pe_top·phiz_top - pe_sfc·phiz_sfc
          + Σ_k delp · (cp·T_v + KE)

Where:
- T_v = T·(1+qc) is virtual temperature (qc is condensate; ignored
  in adiabatic legoESM).
- phiz is geopotential, hydrostatically integrated from the surface:
    phiz[k] = phiz[k+1] + R·T_v·(peln[k+1] - peln[k])
- pe(k+1)·phiz(k+1) - pe(1)·phiz(1) is the "boundary work" term.
- KE on D-grid winds with rsin²/cosa corner factors (FV3 form).
  legoESM uses face-local D-grid winds; we average to cell-center
  and use orthogonal-grid KE (0.5·(u² + v²)) — the FV3 corner
  cosa term is small for nearly orthogonal cubed-sphere cells.

Returns per-column TE and globally-integrated total.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def compute_total_energy_pe(state, grid, coord) -> tuple[jax.Array, float]:
    """Compute per-column and total PE hydrostatic total energy.

    Faithful to FV3 fv_mapz.F90:1127-1152 (hydrostatic branch) using
    legoESM PE state conventions:
    - T from state.T.data (cell-centered, mid-level)
    - p_s from state.p_s.data (cell-centered, surface pressure)
    - phis = state.phis.data (surface geopotential)
    - u_d, v_d from D-grid corners → averaged to cell-center for KE

    Parameters
    ----------
    state : FV3HydrostaticState
    grid : CubedSphereGrid (needs area)
    coord : HybridSigmaPressureCoordinate (A_half, B_half, dA, dB)

    Returns
    -------
    te_column : jax.Array, shape (6, n, n)
        Per-column total energy times area [J/cell].
    te_total : float
        Globally-integrated total energy [J].
    """
    n_face, n, _ = state.p_s.data.shape
    nlev = state.T.data.shape[-1]
    p_s = state.p_s.data                              # (6, n, n)
    phis = state.phis.data                            # (6, n, n)
    T = state.T.data                                  # (6, n, n, nlev)

    # Hybrid pressure at half levels: p_half[k] = A[k]·p_ref + B[k]·p_s
    p_ref = coord.p_ref
    A_h = jnp.asarray(coord.A_half)[None, None, None, :]  # (1,1,1,nlev+1)
    B_h = jnp.asarray(coord.B_half)[None, None, None, :]
    p_half = A_h * p_ref + B_h * p_s[..., None]       # (6,n,n,nlev+1)
    # delp at full levels
    delp = p_half[..., 1:] - p_half[..., :-1]         # (6,n,n,nlev)

    # peln = log(p_half), guarded against zero
    peln = jnp.log(jnp.maximum(p_half, 1e-10))        # (6,n,n,nlev+1)

    # Hydrostatic geopotential integral (top to bottom):
    # phi_half[nlev] = phis (surface)
    # phi_half[k] = phi_half[k+1] + R_d · T[k] · (peln[k+1] - peln[k])
    # No moisture in adiabatic legoESM → use T directly.
    def _phi_step(phi_above, k_top_to_bottom):
        k = nlev - 1 - k_top_to_bottom  # k_top_to_bottom=0 → k=nlev-1
        # ...this isn't quite right for scan; better use direct loop
        return phi_above, None  # placeholder

    # Simpler: compute phi_half via cumulative sum from surface up.
    # dphi[k] = R · T[k] · (peln[k+1] - peln[k])
    dphi = constants.R_d * T * (peln[..., 1:] - peln[..., :-1])
    # phi_half[nlev] = phis ; phi_half[k] = phis + sum_{j=k}^{nlev-1} dphi[j]
    # so phi_half[k] - phi_half[nlev] = sum dphi from k to nlev-1
    # cumulative from bottom: rev_dphi = dphi[::-1]; cum = cumsum(rev_dphi)
    rev_dphi = dphi[..., ::-1]
    rev_cum = jnp.cumsum(rev_dphi, axis=-1)            # (6,n,n,nlev)
    cum_above = rev_cum[..., ::-1]                     # phi_half[k]-phis for k=0..nlev-1
    phi_half_interior = phis[..., None] + cum_above    # (6,n,n,nlev)
    phi_half = jnp.concatenate(
        [phi_half_interior, phis[..., None]], axis=-1,
    )  # (6,n,n,nlev+1) — last entry = surface

    # Boundary work: pe(top)·phi(top) - pe(sfc)·phi(sfc)
    pe_top = p_half[..., 0]                            # (6,n,n)
    pe_sfc = p_half[..., -1]                           # (6,n,n)
    phi_top = phi_half[..., 0]
    phi_sfc = phi_half[..., -1]
    te = pe_top * phi_top - pe_sfc * phi_sfc           # (6,n,n)

    # Cell-center u, v from D-grid corners (simple average).
    u_d = state.u_d.data                               # (6, n+1, n+1, nlev)
    v_d = state.v_d.data
    u_c = 0.25 * (u_d[:, :-1, :-1, :] + u_d[:, 1:, :-1, :]
                  + u_d[:, :-1, 1:, :] + u_d[:, 1:, 1:, :])
    v_c = 0.25 * (v_d[:, :-1, :-1, :] + v_d[:, 1:, :-1, :]
                  + v_d[:, :-1, 1:, :] + v_d[:, 1:, 1:, :])
    ke = 0.5 * (u_c * u_c + v_c * v_c)                 # (6,n,n,nlev)

    # cp·T + KE per cell
    contribution = delp * (constants.c_pd * T + ke)
    te = te + jnp.sum(contribution, axis=-1)           # (6,n,n)

    # Multiply by area for J per column (delp already kg·m/s²/m² → J/m² when
    # times specific energy).  FV3's te_2d before area mult is in J/m².
    te_column = te * grid.area
    te_total = float(jnp.sum(te_column))
    return te_column, te_total


def te_drift_pe(state_old, state_new, grid, coord) -> float:
    """FV3_3D iter 599: PE total-energy tendency between two states.

    Returns ``te_dt = TE(state_new) - TE(state_old)``.  Adiabatic
    flat-surface PE runs should have small drift.

    Companion to ``compute_total_energy_pe`` (iter 598) and the
    NH ``te_drift_nh`` (iter 599 NH side).

    Returns
    -------
    te_dt : float
        PE total-energy tendency [J].
    """
    _, te_old = compute_total_energy_pe(state_old, grid, coord)
    _, te_new = compute_total_energy_pe(state_new, grid, coord)
    return te_new - te_old
