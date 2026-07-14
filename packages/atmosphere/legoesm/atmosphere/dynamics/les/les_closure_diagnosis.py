"""Diagnose closure coefficients from LES-resolved turbulent fluxes.

Stage 6 of ``docs/COMPARE_REANALYSIS.md`` (gap #6): the **inverse** problem.
Given the LES-resolved fluxes from
:func:`legoesm.atmosphere.dynamics.crm.rce_diagnostics.resolved_turbulent_fluxes_plane`
and the LES domain-mean profiles, diagnose a physically-meaningful closure
coefficient — eddy diffusivity ``K``, mixing length ``ℓ``, or the boundary-layer
**entrainment velocity** ``w_e`` — that becomes the per-column (possibly
height-varying) field fed back to the GCM.

These are *diagnostics that invert a closure*, distinct from the forward
turbulence schemes (CLUBB, mass-flux) that *prescribe* one — so there is no
shared numerics to reuse; the formulae here are the closure definitions
themselves:

* Down-gradient eddy diffusivity ``K = -w'φ' / (∂⟨φ⟩/∂z)`` (K-theory).
* Prandtl mixing length ``ℓ = sqrt(K_m / |∂U/∂z|)`` from ``K_m = ℓ² |∂U/∂z|``.
* Mixed-layer entrainment ``w_e = -(w'θ_v')_inv / Δθ_v`` at the inversion
  (Lilly 1968 / Stull): the entrainment buoyancy flux balances ``-w_e Δθ_v``.

All inversions are ill-posed where the denominator vanishes (zero gradient /
zero shear / no inversion), so every diagnosis returns a ``valid`` mask and a
safe (zero) value there rather than a NaN — the caller selects only valid
levels for the feedback field.  Pure-JAX / AD-safe / vmap-friendly.

Vertical convention (**hard precondition**): ``φ_full`` / ``z_full`` are
``(nlev,)`` strictly **ascending in height** — the plane height-coordinate
convention (``height_coord.z_full``).  ``∂/∂z`` and the inversion jump are taken
in array-index order, so this ordering is assumed, not checked (these are
jit/vmap kernels); a descending coordinate would flip gradient/jump signs and is
caller error.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# --- ill-posedness guards (diagnostic regularizers, not tunable physics) -----
# Below these magnitudes the corresponding inversion is ill-posed and the
# coefficient is flagged invalid rather than amplified by a near-zero denom.
_MIN_ABS_GRADIENT_DEFAULT = 1.0e-12  # |∂⟨φ⟩/∂z| floor (φ-units per metre)
_MIN_SHEAR_DEFAULT = 1.0e-6  # |∂U/∂z| floor [1/s]
_MIN_DELTA_THETAV_DEFAULT = 1.0e-3  # inversion θ_v jump floor [K]
_WP2_FLOOR_DEFAULT = 1.0e-4  # w'² floor [m²/s²] (velocity scale √wp2 ≳ 0.01 m/s)
_KH_FLOOR_DEFAULT = 1.0e-3  # heat diffusivity floor [m²/s] (Pr_t ill-posed below)
# C_eps sanity ceiling: a tiny-wp2 / huge-production level blows the unclipped
# C_eps up far past its (0.06, 0.6) range — a transport-dominated / non-equilibrium
# level whose local production≠dissipation inversion is meaningless. 10× the upper
# bound flags such a level invalid rather than letting the clamp average it in.
_C_EPS_SANITY_MAX_DEFAULT = 6.0


class EddyDiffusivityDiagnosis(NamedTuple):
    """Down-gradient eddy diffusivity per interior interface."""

    K: jax.Array          # (nlev-1,) eddy diffusivity [m²/s]
    dphi_dz: jax.Array    # (nlev-1,) mean gradient at interfaces [φ-units/m]
    valid: jax.Array      # (nlev-1,) bool: significant gradient AND K>=0


class EntrainmentDiagnosis(NamedTuple):
    """Mixed-layer entrainment velocity at the diagnosed inversion."""

    inversion_index: jax.Array          # scalar int: interior-interface index
    z_inversion: jax.Array              # scalar: inversion height [m] (or nan)
    entrainment_buoyancy_flux: jax.Array  # scalar: (w'θ_v')_inv [K m/s]
    delta_thetav: jax.Array             # scalar: θ_v jump across inversion [K]
    w_entrainment: jax.Array            # scalar: entrainment velocity [m/s]
    valid: jax.Array                    # scalar bool


def mean_gradient_at_interfaces(
    phi_full: jax.Array, z_full: jax.Array
) -> jax.Array:
    """``∂⟨φ⟩/∂z`` at the ``nlev-1`` interior interfaces from full-level means.

    ``(⟨φ⟩[k+1] - ⟨φ⟩[k]) / (z[k+1] - z[k])`` — co-located with the resolved
    fluxes (which live at the same interior interfaces).
    """
    phi_full = jnp.asarray(phi_full)
    z_full = jnp.asarray(z_full, dtype=phi_full.dtype)
    dz = z_full[1:] - z_full[:-1]
    return (phi_full[1:] - phi_full[:-1]) / dz


def eddy_diffusivity_from_flux(
    flux: jax.Array,
    phi_full: jax.Array,
    z_full: jax.Array,
    *,
    min_abs_gradient: float = _MIN_ABS_GRADIENT_DEFAULT,
) -> EddyDiffusivityDiagnosis:
    """Diagnose ``K = -w'φ' / (∂⟨φ⟩/∂z)`` per interior interface.

    ``flux`` is the resolved ``(nlev-1,)`` eddy flux ``⟨w'φ'⟩``; ``phi_full`` /
    ``z_full`` are the ``(nlev,)`` domain-mean profile and heights.  Where the
    mean gradient is below ``min_abs_gradient`` (closure ill-posed) or the
    diagnosed ``K`` is negative (counter-gradient flux — down-gradient K-theory
    does not apply, e.g. the convective interior), ``K`` is set to zero and
    ``valid=False``.  AD-safe: the denominator is masked before division.
    """
    flux = jnp.asarray(flux)
    dphi_dz = mean_gradient_at_interfaces(phi_full, z_full).astype(flux.dtype)
    significant = jnp.abs(dphi_dz) > jnp.asarray(min_abs_gradient, dtype=flux.dtype)
    denom = jnp.where(significant, dphi_dz, jnp.ones_like(dphi_dz))
    K = jnp.where(significant, -flux / denom, jnp.zeros_like(flux))
    valid = significant & (K >= 0.0)
    K = jnp.where(valid, K, jnp.zeros_like(K))
    return EddyDiffusivityDiagnosis(K=K, dphi_dz=dphi_dz, valid=valid)


def momentum_diffusivity_from_fluxes(
    w_u: jax.Array,
    w_v: jax.Array,
    u_full: jax.Array,
    v_full: jax.Array,
    z_full: jax.Array,
    *,
    min_shear: float = _MIN_SHEAR_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Shear-projected down-gradient momentum diffusivity ``K_m`` at the
    ``nlev-1`` interior interfaces from the resolved momentum fluxes.

    For a vector eddy flux ``F = (⟨w'u'⟩, ⟨w'v'⟩)`` and mean shear
    ``S = (∂⟨u⟩/∂z, ∂⟨v⟩/∂z)``, the scalar ``K_m`` minimizing ``|F + K_m·S|²``
    (the best down-gradient scalar viscosity for a possibly-misaligned flux) is
    the least-squares projection ``K_m = −(F·S)/|S|²``.  Down-gradient momentum
    flux opposes the shear (``F·S < 0``), so ``K_m > 0`` for normal shear-driven
    mixing; the cross-shear flux component a scalar viscosity cannot represent is
    discarded by the projection.  ``valid`` where ``|S|² > min_shear²`` AND
    ``K_m ≥ 0`` (counter-gradient momentum transport ⇒ K-theory inapplicable).
    Inputs are ascending-``z`` and co-located; AD-safe (denominator masked
    before division).
    """
    w_u = jnp.asarray(w_u)
    w_v = jnp.asarray(w_v, dtype=w_u.dtype)
    du_dz = mean_gradient_at_interfaces(u_full, z_full).astype(w_u.dtype)
    dv_dz = mean_gradient_at_interfaces(v_full, z_full).astype(w_u.dtype)
    shear_sq = du_dz**2 + dv_dz**2
    min_sq = jnp.asarray(min_shear, dtype=w_u.dtype) ** 2
    ok = shear_sq > min_sq
    denom = jnp.where(ok, shear_sq, jnp.ones_like(shear_sq))
    flux_dot_shear = w_u * du_dz + w_v * dv_dz
    K_m_raw = jnp.where(ok, -flux_dot_shear / denom, jnp.zeros_like(shear_sq))
    valid = ok & (K_m_raw >= 0.0)
    K_m = jnp.where(valid, K_m_raw, jnp.zeros_like(K_m_raw))
    return K_m, valid


def clubb_coefficient_from_diffusivity(
    K_m: jax.Array,
    K_m_valid: jax.Array,
    l_mix: jax.Array,
    wp2: jax.Array,
    *,
    wp2_floor: float = _WP2_FLOOR_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Dimensionless CLUBB-lite eddy coefficient ``C_K = K_m/(ℓ·√wp2)`` — the
    exact inverse of the GCM closure ``K_m = C_K·ℓ·√wp2`` (``clubb_lite.py``).

    ``K_m`` [m²/s], the mixing length ``l_mix`` [m] and the vertical-velocity
    variance ``wp2`` [m²/s²] are co-located at the same interfaces.  ``C_K`` is
    dimensionless: ``[m²/s] / ([m]·[m/s]) = [1]``.  ``valid`` where ``K_m_valid``
    AND ``wp2 > wp2_floor`` AND ``l_mix > 0``.  AD-safe via the double-where idiom
    at the ``√wp2`` (infinite VJP at 0) and the division.

    NOTE on transferability: the GCM evaluates ``√wp2`` from its OWN prognostic
    wp2 budget, whereas this uses the LES resolved ``w'²`` (the "truth" the GCM
    wp2 approximates under the shared large-scale forcing).  When the GCM
    equilibrium wp2 departs from the LES ``w'²`` (e.g. deep convection) the
    diagnosed ``C_K`` carries an O(1) offset; the loop's iter-43 gate / iter-44
    line search and the bias monitor are the safeguards.  Co-tuning ``C_eps`` to
    align the GCM wp2 with the LES truth — the assumption-relaxing step — IS
    implemented in :func:`c_eps_from_budget`: it inverts the clubb_lite *local
    no-transport* steady-state wp2 balance, so the diagnosed ``C_eps`` drives the
    GCM equilibrium wp2 toward the LES ``w'²`` wherever vertical transport of
    ``wp2`` is small relative to local production/dissipation (it does not fully
    cancel the offset where ``diff(wp2)`` dominates).  The SIMULTANEOUS
    multi-coefficient correction (``CorrectionSpec`` with ``C_K`` + ``C_eps``)
    applies both together.
    """
    K_m = jnp.asarray(K_m)
    l_mix = jnp.asarray(l_mix, dtype=K_m.dtype)
    wp2 = jnp.asarray(wp2, dtype=K_m.dtype)
    floor = jnp.asarray(wp2_floor, dtype=K_m.dtype)
    wp2_ok = wp2 > floor
    safe_wp2 = jnp.where(wp2_ok, wp2, jnp.ones_like(wp2))
    sqrt_wp2 = jnp.where(wp2_ok, jnp.sqrt(safe_wp2), jnp.zeros_like(wp2))
    l_ok = l_mix > 0.0
    valid = jnp.asarray(K_m_valid, dtype=bool) & wp2_ok & l_ok
    denom = l_mix * sqrt_wp2
    denom_safe = jnp.where(valid, denom, jnp.ones_like(denom))
    C_K = jnp.where(valid, K_m / denom_safe, jnp.zeros_like(K_m))
    return C_K, valid


def prandtl_number_from_diffusivities(
    K_m: jax.Array,
    K_m_valid: jax.Array,
    K_h: jax.Array,
    K_h_valid: jax.Array,
    *,
    kh_floor: float = _KH_FLOOR_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Turbulent Prandtl number ``Pr_t = K_m/K_h`` — the inverse of the GCM
    relation ``K_h = K_m/Pr_t`` (``clubb_lite.py``).

    A DIMENSIONLESS ratio of the LES momentum (``K_m``) and heat (``K_h``)
    diffusivities at the same interfaces; unlike ``C_K`` it needs no velocity
    scale, so it carries NO wp2-identification assumption.  ``valid`` where both
    diffusivities are valid AND ``K_h > kh_floor`` (the ratio is ill-posed for a
    near-zero heat diffusivity).  AD-safe (denominator masked before division).
    """
    K_m = jnp.asarray(K_m)
    K_h = jnp.asarray(K_h, dtype=K_m.dtype)
    floor = jnp.asarray(kh_floor, dtype=K_m.dtype)
    kh_ok = jnp.asarray(K_h_valid, dtype=bool) & (K_h > floor)
    valid = jnp.asarray(K_m_valid, dtype=bool) & kh_ok
    denom = jnp.where(valid, K_h, jnp.ones_like(K_h))
    Pr_t = jnp.where(valid, K_m / denom, jnp.zeros_like(K_m))
    return Pr_t, valid


def c_eps_from_budget(
    K_m: jax.Array,
    K_m_valid: jax.Array,
    K_h: jax.Array,
    K_h_valid: jax.Array,
    shear_sq: jax.Array,
    N_sq: jax.Array,
    l_mix: jax.Array,
    wp2: jax.Array,
    *,
    wp2_floor: float = _WP2_FLOOR_DEFAULT,
    c_eps_sanity_max: float = _C_EPS_SANITY_MAX_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Dimensionless wp2-dissipation coefficient ``C_eps`` from the steady-state
    CLUBB-lite ``w'²`` budget — the inverse of ``dissipation = C_eps·√wp2/l``.

    The lite ``wp2`` budget (``clubb_lite.py``) balances production against the
    semi-implicit dissipation, so at steady state (NEGLECTING vertical transport)
    ``P = C_eps·wp2^{3/2}/ℓ`` with the net production ``P = K_m·S² − K_h·N²``
    (shear minus buoyancy destruction).  Inverting:
        ``C_eps = P·ℓ / wp2^{3/2}`` .
    Diagnosed so the GCM's equilibrium ``wp2`` tracks the LES ``w'²``, which makes
    the iter-46 ``C_K = K_m/(ℓ·√wp2)`` transfer correct (it removes the
    wp2-identification offset).  ``valid`` where both diffusivities are valid, ``wp2
    > wp2_floor``, ``ℓ > 0``, the net production ``P > 0`` (a layer NOT sustaining
    turbulence by this balance is excluded), AND the raw ``C_eps`` is below
    ``c_eps_sanity_max`` (a tiny-wp2 / transport-dominated blow-up is flagged, not
    averaged in).  Double-where AD-safe at ``wp2^{3/2}`` (infinite VJP at 0).

    LIMITATION: this inverts the LOCAL production=dissipation balance and DROPS the
    GCM budget's turbulent-transport term, which in a convective BL redistributes
    ``wp2`` (so the local balance can be off near the surface / inversion).  The
    ``P>0`` + sanity masks, the registered ``(0.06, 0.6)`` bounds clamp, and the
    monotonic gate are the backstops — a rejected C_eps correction does no harm.
    """
    K_m = jnp.asarray(K_m)
    dtype = K_m.dtype
    K_h = jnp.asarray(K_h, dtype=dtype)
    shear_sq = jnp.asarray(shear_sq, dtype=dtype)
    N_sq = jnp.asarray(N_sq, dtype=dtype)
    l_mix = jnp.asarray(l_mix, dtype=dtype)
    wp2 = jnp.asarray(wp2, dtype=dtype)
    floor = jnp.asarray(wp2_floor, dtype=dtype)

    production = K_m * shear_sq - K_h * N_sq
    wp2_ok = wp2 > floor
    safe_wp2 = jnp.where(wp2_ok, wp2, jnp.ones_like(wp2))
    wp2_32 = safe_wp2 * jnp.sqrt(safe_wp2)                  # wp2^{3/2}, finite VJP
    base_valid = (
        jnp.asarray(K_m_valid, dtype=bool)
        & jnp.asarray(K_h_valid, dtype=bool)
        & wp2_ok
        & (l_mix > 0.0)
        & (production > 0.0)
    )
    denom = jnp.where(base_valid, wp2_32, jnp.ones_like(wp2_32))
    c_eps_raw = jnp.where(base_valid, production * l_mix / denom, jnp.zeros_like(K_m))
    valid = base_valid & (c_eps_raw <= jnp.asarray(c_eps_sanity_max, dtype=dtype))
    c_eps = jnp.where(valid, c_eps_raw, jnp.zeros_like(c_eps_raw))
    return c_eps, valid


def mixing_length_from_momentum_diffusivity(
    K_m: jax.Array,
    shear: jax.Array,
    *,
    min_shear: float = _MIN_SHEAR_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Prandtl mixing length ``ℓ = sqrt(K_m / |∂U/∂z|)`` from ``K_m = ℓ²|∂U/∂z|``.

    ``K_m`` and ``shear`` (``∂U/∂z``) are at the same interfaces.  Returns
    ``(ell, valid)``; ``ell`` is zero where the shear is below ``min_shear`` or
    ``K_m`` is negative (closure inapplicable).  AD-safe.
    """
    K_m = jnp.asarray(K_m)
    shear = jnp.asarray(shear, dtype=K_m.dtype)
    shear_mag = jnp.abs(shear)
    ok = (shear_mag > jnp.asarray(min_shear, dtype=K_m.dtype)) & (K_m >= 0.0)
    denom = jnp.where(ok, shear_mag, jnp.ones_like(shear_mag))
    ell_sq = jnp.where(ok, K_m / denom, jnp.zeros_like(K_m))
    # Gradient-safe sqrt: ``jnp.sqrt`` has an infinite VJP at 0, and a plain
    # ``where`` does not protect it (both branches are differentiated).  Route a
    # strictly-positive argument through the sqrt and zero the ``ell_sq<=0``
    # branch (double-where idiom).
    sqrt_ok = ok & (ell_sq > 0.0)
    safe_arg = jnp.where(sqrt_ok, ell_sq, jnp.ones_like(ell_sq))
    ell = jnp.where(sqrt_ok, jnp.sqrt(safe_arg), jnp.zeros_like(ell_sq))
    return ell, ok


def entrainment_velocity_from_buoyancy_flux(
    w_thetav: jax.Array,
    thetav_full: jax.Array,
    z_half_interior: jax.Array,
    *,
    min_delta_thetav: float = _MIN_DELTA_THETAV_DEFAULT,
) -> EntrainmentDiagnosis:
    """Diagnose ``w_e = -(w'θ_v')_inv / Δθ_v`` at the entrainment inversion.

    The inversion interface is the most negative resolved buoyancy flux
    (entrainment-flux minimum, the classic mixed-layer definition).  ``Δθ_v`` is
    the virtual-potential-temperature jump across that interface from the
    full-level mean ``thetav_full`` ``(nlev,)``.  The diagnosis is valid only for
    a stable jump (``Δθ_v > min_delta_thetav``) with a negative entrainment
    buoyancy flux; otherwise ``w_entrainment`` is zero and ``valid=False`` (so a
    column with no capping inversion fails safe).
    """
    w_thetav = jnp.asarray(w_thetav)
    thetav_full = jnp.asarray(thetav_full, dtype=w_thetav.dtype)
    z_half_interior = jnp.asarray(z_half_interior, dtype=w_thetav.dtype)
    delta_thetav = thetav_full[1:] - thetav_full[:-1]  # (nlev-1,)

    # Inversion = the entrainment-flux minimum.  ``argmin`` breaks ties by
    # picking the FIRST minimum (documented); its gradient w.r.t. the index is
    # zero, but the diagnosed ``w_e`` still differentiates through the gathered
    # ``flux_inv`` value.
    n_iface = w_thetav.shape[0]
    k_inv = jnp.argmin(w_thetav)
    flux_inv = w_thetav[k_inv]
    dthv_inv = delta_thetav[k_inv]
    z_inv = z_half_interior[k_inv]

    stable = dthv_inv > jnp.asarray(min_delta_thetav, dtype=w_thetav.dtype)
    # A capping inversion cannot sit on a domain-boundary interface (a surface-
    # or top-flux minimum is not entrainment): require an interior interface.
    interior = (k_inv > 0) & (k_inv < n_iface - 1)
    valid = stable & (flux_inv < 0.0) & interior
    denom = jnp.where(stable, dthv_inv, jnp.ones_like(dthv_inv))
    w_e = jnp.where(valid, -flux_inv / denom, jnp.zeros_like(flux_inv))
    return EntrainmentDiagnosis(
        inversion_index=k_inv,
        z_inversion=jnp.where(valid, z_inv, jnp.asarray(jnp.nan, dtype=w_thetav.dtype)),
        entrainment_buoyancy_flux=flux_inv,
        delta_thetav=dthv_inv,
        w_entrainment=w_e,
        valid=valid,
    )
