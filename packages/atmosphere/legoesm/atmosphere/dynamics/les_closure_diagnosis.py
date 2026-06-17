"""Diagnose closure coefficients from LES-resolved turbulent fluxes.

Stage 6 of ``docs/COMPARE_REANALYSIS.md`` (gap #6): the **inverse** problem.
Given the LES-resolved fluxes from
:func:`legoesm.atmosphere.dynamics.rce_diagnostics.resolved_turbulent_fluxes_plane`
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
