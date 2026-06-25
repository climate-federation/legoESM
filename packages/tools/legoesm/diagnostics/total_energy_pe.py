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


def compute_total_energy_pe(state, grid, coord) -> tuple[jax.Array, jax.Array]:
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
    # iter-45: promote state arrays to the fp64 budget accumulator
    # before the per-cell te = delp·(c_p·T + ke) computation.  The
    # pre-iter-45 path did u·u + v·v, T-products, and the final
    # global sum in fp32 storage dtype — the same fp32-field bug
    # iter-42/43/44 fixed across the conservation helpers.
    from legoesm.core.conservation import conservation_accumulator
    acc = conservation_accumulator()
    p_s = state.p_s.data.astype(acc)                  # (6, n, n)
    phis = state.phis.data.astype(acc)                # (6, n, n)
    T = state.T.data.astype(acc)                      # (6, n, n, nlev)

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
    # Compute phi_half via cumulative sum from surface up.
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

    # Boundary work: pe(sfc)·phi(sfc) - pe(top)·phi(top)
    # FV3 fv_mapz.F90:1142: ``te_2d = pe(km+1)·phiz(km+1) - pe(1)·phiz(1)``
    # where k=1 is top (small p), k=km+1 is surface (large p).  Matches:
    # iter 603 fix: sign-corrected from iter 598's te = pe_top·phi_top - pe_sfc·phi_sfc.
    pe_top = p_half[..., 0]                            # (6,n,n)
    pe_sfc = p_half[..., -1]                           # (6,n,n)
    phi_top = phi_half[..., 0]
    phi_sfc = phi_half[..., -1]
    te = pe_sfc * phi_sfc - pe_top * phi_top           # (6,n,n)

    # Cell-center u, v from D-grid corners (simple average).
    u_d = state.u_d.data.astype(acc)                   # (6, n+1, n+1, nlev)
    v_d = state.v_d.data.astype(acc)
    u_c = 0.25 * (u_d[:, :-1, :-1, :] + u_d[:, 1:, :-1, :]
                  + u_d[:, :-1, 1:, :] + u_d[:, 1:, 1:, :])
    v_c = 0.25 * (v_d[:, :-1, :-1, :] + v_d[:, 1:, :-1, :]
                  + v_d[:, :-1, 1:, :] + v_d[:, 1:, 1:, :])
    ke = 0.5 * (u_c * u_c + v_c * v_c)                 # (6,n,n,nlev) fp64

    # cp·T + KE per cell
    c_pd = jnp.asarray(constants.c_pd, dtype=acc)
    contribution = delp * (c_pd * T + ke)
    te = te + jnp.sum(contribution, axis=-1)           # (6,n,n) fp64

    # Multiply by area for J per column (delp already kg·m/s²/m² → J/m² when
    # times specific energy).  FV3's te_2d before area mult is in J/m².
    te_column = te * grid.area.astype(acc)
    # Return the total as a TRACED 0-d array (no host ``float()``), so the
    # energy-correction Newton solve below stays jit/grad-safe. Callers wanting
    # a host scalar do ``float(te_total)`` (e.g. ``te_drift_pe``).
    te_total = jnp.sum(te_column)
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
    return float(te_new - te_old)


# Fixed Newton sweeps for the PE total-energy correction. A COUNT (never
# config/trainable); the energy is ~linear in a uniform ΔT so this converges in
# 1-2 iters and the rest are ~no-ops — a fixed loop keeps the correction
# jit/grad-safe (no data-dependent early-out).
_TE_NEWTON_ITERS = 5


def apply_te_correction_pe(state_old, state_new, grid, coord):
    """FV3_3D iter 602: PE total-energy-conserving correction.

    Faithful port of FV3 ``consv_te > 0`` correction for hydrostatic
    branch (fv_dynamics.F90:359-378).  Analog of iter 601's NH
    version but uses **cp** (constant-pressure heat) instead of cv
    because PE is hydrostatic.

    Enforces ``TE(corrected) ≈ TE(state_old)`` by adding a uniform
    temperature increment ΔT:

        te_dt = TE(state_new) - TE(state_old)
        ΔT = -te_dt / (cp · total_dry_mass)
        T_corrected = T + ΔT  (uniform across all cells)

    Total dry mass = Σ delp · area / g where delp = A·p_ref + B·p_s.

    KE, p_s, phis NOT adjusted — only T (matches FV3 hydrostatic
    consv_te which targets the thermodynamic state).

    Returns
    -------
    state_corrected : FV3HydrostaticState
        State with T adjusted; other fields unchanged.

    Notes
    -----
    Bit-for-bit identical to state_new when te_dt = 0.
    Differentiable end-to-end.
    """
    _, te_target = compute_total_energy_pe(state_old, grid, coord)  # traced scalar

    # PE TE includes a hydrostatic boundary-work term that depends on T (via phi
    # from hydrostatic integration), so the effective heat capacity is NOT
    # simply cp · total_mass. Use a FIXED-count Newton sweep with a
    # numerical-Jacobian estimate of dTE/dT — no data-dependent ``break`` and no
    # host ``float()``, so the whole correction is jit/grad-safe.
    dT_probe = 1.0e-3

    def _newton_step(_, state_curr):
        _, te_curr = compute_total_energy_pe(state_curr, grid, coord)
        residual = te_curr - te_target
        state_probe = state_curr._replace(
            T=state_curr.T.replace(data=state_curr.T.data + dT_probe),
        )
        _, te_probe = compute_total_energy_pe(state_probe, grid, coord)
        dTE_dT = (te_probe - te_curr) / dT_probe
        # Guard a (near-)singular Jacobian with a where-select, NOT a break.
        dT_step = jnp.where(jnp.abs(dTE_dT) < 1e-30, 0.0, -residual / dTE_dT)
        # Cast the (fp64-budget) increment to the state dtype so the fori_loop
        # carry keeps a stable dtype (input == output).
        return state_curr._replace(
            T=state_curr.T.replace(
                data=state_curr.T.data + dT_step.astype(state_curr.T.data.dtype),
            ),
        )

    return jax.lax.fori_loop(0, _TE_NEWTON_ITERS, _newton_step, state_new)
