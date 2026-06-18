"""Diagnose a closure coefficient from a finished column-LES run.

Stage 6 composition of ``docs/COMPARE_REANALYSIS.md``: after the spun-off LES for
a flagged column reaches statistical steady state, turn its resolved turbulent
fluxes into the per-column closure coefficient that feeds the spatially-varying
parameter field.  This is the glue between the two already-tested leaves:

* resolved fluxes —
  :func:`legoesm.atmosphere.dynamics.rce_diagnostics.resolved_turbulent_fluxes_plane`,
* closure inversion — :mod:`legoesm.atmosphere.dynamics.les_closure_diagnosis`.

Two methods (selected by ``method``; an unknown method raises — dispatch
hardening):

* ``"eddy_diffusivity"`` — the down-gradient eddy diffusivity ``K`` profile from
  ``w'θ'`` and the domain-mean ``θ`` gradient.
* ``"entrainment"`` — the boundary-layer entrainment velocity ``w_e`` from the
  resolved buoyancy flux ``w'θ_v'`` at the inversion (the doc's headline
  coefficient).

Vertical-orientation handling: the plane state / height coordinate store fields
**top-down** (index 0 = model top), but the closure functions require ascending
height; this module reverses the relevant arrays consistently before calling
them (resolved fluxes at the ``nlev-1`` interior interfaces reverse to the same
ascending interfaces as the reversed mean-profile gradient).  Pure-JAX, AD-safe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
    EntrainmentDiagnosis,
    c_eps_from_budget,
    clubb_coefficient_from_diffusivity,
    eddy_diffusivity_from_flux,
    entrainment_velocity_from_buoyancy_flux,
    mean_gradient_at_interfaces,
    momentum_diffusivity_from_fluxes,
    prandtl_number_from_diffusivities,
)
from legoesm.atmosphere.dynamics.rce_diagnostics import (
    resolved_turbulent_fluxes_plane,
    vertical_velocity_variance_plane,
)
from legoesm.atmosphere.physics._shared import mixing_length

from legoesm import constants

_METHODS = (
    "eddy_diffusivity", "entrainment", "clubb_coefficient", "prandtl_number",
    "c_eps",
)

# Minimum number of valid interior interfaces for a trustworthy column C_K
# (a single anomalous-shear interface should not define the whole column).
_MIN_VALID_CK_LEVELS = 3

# "Turbulence developed" threshold [m²/s²] (velocity scale √floor ≈ 0.03 m/s): a
# finished LES whose peak resolved w'² is below this never developed turbulence,
# so it carries no trustworthy turbulence diagnosis.  Regime-sensitive (see
# column_les_realism); a campaign can tune it via ColumnLESConfig.
_REALISM_WP2_FLOOR = 1.0e-3

# Thermodynamic-drift threshold [K]: the RMS over levels of the LES HORIZONTAL-MEAN
# θ′ (its drift from the GCM column reference θ(z) — ``theta_prime = θ − hc.theta_ref``
# and ``hc.theta_ref`` is the GCM column θ interpolated to the LES grid, so a
# CONSISTENT LES has mean θ′ ≈ 0). Above this the LES mean state wandered off the
# column it represents → distrust its diagnosis. RMS (not max) is robust to a single
# sharp inversion level while catching a broad drift. Conservative default; the
# downstream bounds + line-search + monotonic gate backstop a marginally-bad value,
# so erring strict (a false-reject merely keeps the background) is the safe bias.
# Regime-sensitive (stable / convective differ) and configurable via ColumnLESConfig.
_THETA_DRIFT_RMS_MAX_K = 3.0

# Moisture physical-sanity cap [kg/kg]: q_v RUNNING AWAY — above _Q_V_MAX or below
# −_Q_V_NEG_TOL — flags a finite-but-unphysical moisture blow-up the finite check
# misses. 0.05 = 50 g/kg is above ANY physical atmospheric water-vapor mixing ratio
# (extreme tropical surface ~40 g/kg), so a legitimate LES — however convective —
# never trips the upper cap (no false-reject). The column LES uses CENTERED vertical
# tracer advection (not van_leer), which can overshoot slightly NEGATIVE, so q_v ≥ 0
# is NOT guaranteed; the negative tolerance rejects only a large-negative blow-up, not
# a small numerical undershoot.  WHY a raw cap, not a drift/RH check: a
# drift-from-reference check is ill-posed (unlike θ, the LES q is NOT relaxed toward
# the GCM column — it evolves freely via large-scale advection + microphysics, so
# drifting off q_v_init is EXPECTED physics); and a T-dependent RH supersaturation cap
# (the physically-fuller check) needs realistic test fixtures (the current mocks use an
# unphysical uniform q, grossly supersaturated aloft) — a documented follow-up.
_Q_V_MAX_KG_KG = 0.05
_Q_V_NEG_TOL_KG_KG = 1.0e-3   # allow tiny advective undershoot; reject a large negative


def _thermo_drift_rms(theta_prime: jax.Array) -> jax.Array:
    """RMS over levels of the horizontal-mean θ′ — the LES mean's drift from the
    GCM column reference (``mean(θ′, horizontal)`` per level, then RMS over levels)."""
    mean_thp = jnp.mean(theta_prime, axis=(0, 1))      # (nlev,) horizontal mean
    return jnp.sqrt(jnp.mean(mean_thp ** 2))


def column_les_realism(les_state, height_coord, *, wp2_floor: float = _REALISM_WP2_FLOOR,
                       theta_drift_rms_max_K: float = _THETA_DRIFT_RMS_MAX_K,
                       q_v_max: float = _Q_V_MAX_KG_KG, qv_slot: int = 0):
    """Coarse TRUST gate on a finished column LES (``docs/COMPARE_REANALYSIS.md`` §9):
    did it develop turbulence, stay finite, stay thermodynamically near the column it
    represents, AND keep moisture physical?  Returns a TRACED scalar bool.

    A dead/laminar LES (peak resolved ``w'²`` below ``wp2_floor``), a blown-up one
    (any non-finite field), a DRIFTED one (its horizontal-mean θ′ wandered off the
    GCM column reference θ(z) by more than ``theta_drift_rms_max_K``), OR one whose
    water-vapor mixing ratio ran away (``q_v`` above ``q_v_max`` or large-negative)
    carries no trustworthy turbulence signal, so the loop keeps the column's BACKGROUND
    coefficient rather than inject a finite-but-meaningless diagnosis (which the
    bounds + non-finite guards would not otherwise catch).

    The thermodynamic term (iter 66) closes the TEMPERATURE/MSE-drift part of the
    documented §9 gap; the moisture cap (iter 67) catches a finite-but-runaway q_v.
    The CORRECTED moisture finding (iter 67): a q drift/CWV/MSE check against the GCM
    column is ILL-POSED for a final-state gate — unlike θ, the LES q is NOT relaxed
    toward the column (it evolves freely via large-scale advection + microphysics),
    and a CWV-plateau check needs the time series.  So the valid final-state moisture
    term is a physical-sanity cap (too-high OR large-negative — centered tracer
    advection can overshoot, so q_v ≥ 0 is not guaranteed); a
    T-dependent RH supersaturation cap (the fuller check) is a follow-up needing
    realistic test fixtures.  A legitimately quiescent stable column correctly FAILS
    the turbulence term (no signal ⇒ the background is right), a correct outcome, not
    a false negative.  Thresholds are regime-sensitive and configurable.  Pure-JAX (a
    boolean mask; no NaN grad — ``&`` is logical-and on bool scalars, and ``NaN < thr``
    is ``False`` so a non-finite field still rejects).
    """
    thp = jnp.asarray(les_state.theta_prime.data)
    wp2 = vertical_velocity_variance_plane(les_state, height_coord)
    q_v = jnp.asarray(les_state.tracers.data)[..., qv_slot]
    turbulent = jnp.max(wp2) > jnp.asarray(wp2_floor, dtype=wp2.dtype)
    finite = (jnp.all(jnp.isfinite(wp2)) & jnp.all(jnp.isfinite(thp))
              & jnp.all(jnp.isfinite(q_v)))
    thermo_consistent = (
        _thermo_drift_rms(thp) < jnp.asarray(theta_drift_rms_max_K, dtype=thp.dtype))
    moisture_physical = (
        (jnp.max(q_v) < jnp.asarray(q_v_max, dtype=q_v.dtype))
        & (jnp.min(q_v) > jnp.asarray(-_Q_V_NEG_TOL_KG_KG, dtype=q_v.dtype)))
    return turbulent & finite & thermo_consistent & moisture_physical


def gate_diagnosis_realism(diagnosis, realistic):
    """AND a scalar LES-realism flag into a diagnosis's ``valid`` mask (the value
    is left as-is — the reduce already double-where-masks invalid values, so an
    unrealistic/blown-up column reduces to invalid ⇒ the loop keeps the background)."""
    return diagnosis._replace(
        valid=jnp.asarray(diagnosis.valid, dtype=bool) & realistic
    )


class EddyDiffusivityProfile(NamedTuple):
    """Down-gradient eddy diffusivity ``K`` per interior interface (ascending z)."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    K: jax.Array       # (nlev-1,) eddy diffusivity [m²/s]
    valid: jax.Array   # (nlev-1,) bool: significant gradient AND K>=0


def _domain_mean_theta(les_state, height_coord) -> jax.Array:
    """Domain-mean potential temperature profile ``⟨θ⟩(z)`` [K], top-down."""
    return height_coord.theta_ref + jnp.mean(
        les_state.theta_prime.data, axis=(0, 1)
    )


def diagnose_eddy_diffusivity(
    les_state, height_coord, qv_slot: int = 0
) -> EddyDiffusivityProfile:
    """Diagnose ``K = -w'θ'/(∂⟨θ⟩/∂z)`` per interior interface from the LES.

    Reverses the top-down resolved heat flux + mean-θ + heights to ascending so
    the flux and the gradient are co-located at the same ascending interfaces.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    theta_mean = _domain_mean_theta(les_state, height_coord)        # (nlev,) top-down
    z_full = jnp.asarray(height_coord.z_full)                       # (nlev,) top-down
    # Reverse to ascending z; the interior-interface flux reverses to the same
    # ascending interfaces as the reversed full-level gradient.
    diag = eddy_diffusivity_from_flux(
        fluxes.w_theta[::-1], theta_mean[::-1], z_full[::-1]
    )
    return EddyDiffusivityProfile(
        z_m=fluxes.z_half_interior[::-1], K=diag.K, valid=diag.valid
    )


def diagnose_entrainment(
    les_state, height_coord, qv_slot: int = 0
) -> EntrainmentDiagnosis:
    """Diagnose the entrainment velocity ``w_e`` from the LES buoyancy flux.

    Builds the domain-mean virtual potential temperature ``⟨θ_v⟩`` (same ``ε``
    convention as :func:`compute_cape`), reverses to ascending, and inverts the
    buoyancy-flux minimum at the inversion.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    theta_total = height_coord.theta_ref + les_state.theta_prime.data
    q_v = les_state.tracers.data[..., qv_slot]
    coeff = 1.0 / constants.epsilon - 1.0
    theta_v = theta_total * (1.0 + coeff * q_v)
    thetav_mean = jnp.mean(theta_v, axis=(0, 1))                    # (nlev,) top-down
    return entrainment_velocity_from_buoyancy_flux(
        fluxes.w_thetav[::-1], thetav_mean[::-1], fluxes.z_half_interior[::-1]
    )


class ClubbCoefficientProfile(NamedTuple):
    """Dimensionless CLUBB-lite eddy coefficient ``C_K`` per interior interface."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    C_K: jax.Array     # (nlev-1,) dimensionless C_K = K_m/(ℓ·√wp2)
    valid: jax.Array   # (nlev-1,) bool


def diagnose_clubb_coefficient(
    les_state,
    height_coord,
    *,
    l_mix_max: float,
    qv_slot: int = 0,
    min_valid_levels: int = _MIN_VALID_CK_LEVELS,
) -> ClubbCoefficientProfile:
    """Diagnose the DIMENSIONLESS CLUBB-lite coefficient ``C_K = K_m/(ℓ·√wp2)``.

    Unlike :func:`diagnose_eddy_diffusivity` (which returns a *dimensional* heat
    diffusivity ``K`` [m²/s]), this returns the actual GCM parameter: the
    dimensionless coefficient that, with the GCM's own mixing length and velocity
    scale, reproduces the LES MOMENTUM diffusivity — the exact inverse of
    ``Km = C_K·ℓ·√wp2`` (``clubb_lite.py``).  Using the momentum flux (not heat)
    avoids conflating ``C_K`` with the turbulent Prandtl number ``Pr_t``.

    Pipeline (all co-located at the ``nlev-1`` interior interfaces, reversed
    top-down→ascending together — the plane convention):
      * ``K_m`` from the resolved ``⟨w'u'⟩, ⟨w'v'⟩`` + mean-wind shear
        (:func:`momentum_diffusivity_from_fluxes`);
      * ``ℓ`` from the SAME Blackadar :func:`mixing_length` the GCM uses, with
        the GCM's ``l_mix_max``;
      * ``√wp2`` from the resolved vertical-velocity variance
        (:func:`vertical_velocity_variance_plane`, interior half levels);
      * ``C_K`` and the valid mask from
        :func:`clubb_coefficient_from_diffusivity`.
    Fewer than ``min_valid_levels`` valid interfaces ⇒ the WHOLE column is
    flagged invalid (a single anomalous-shear interface must not define ``C_K``);
    the loop then keeps the background.  Pure-JAX, AD-safe.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    wp2_half = vertical_velocity_variance_plane(les_state, height_coord)  # (nlev+1,) top-down
    u_mean = jnp.mean(les_state.u.data, axis=(0, 1))                      # (nlev,) top-down
    v_mean = jnp.mean(les_state.v.data, axis=(0, 1))
    z_full = jnp.asarray(height_coord.z_full)                            # (nlev,) top-down
    # Reverse EVERY array top-down→ascending together so K_m's flux + shear,
    # √wp2 and ℓ(z) are all co-located at the same ascending interior interfaces.
    w_u = fluxes.w_u[::-1]
    w_v = fluxes.w_v[::-1]
    u_asc = u_mean[::-1]
    v_asc = v_mean[::-1]
    z_asc = z_full[::-1]
    wp2_interior = wp2_half[1:-1][::-1]                                  # (nlev-1,) ascending
    z_m = fluxes.z_half_interior[::-1]                                   # (nlev-1,) ascending
    K_m, km_valid = momentum_diffusivity_from_fluxes(
        w_u, w_v, u_asc, v_asc, z_asc
    )
    l_mix = mixing_length(z_m, l_mix_max)
    C_K, valid = clubb_coefficient_from_diffusivity(
        K_m, km_valid, l_mix, wp2_interior
    )
    # Column-level trust guard: too few valid interfaces ⇒ invalidate the column.
    enough = jnp.sum(valid) >= int(min_valid_levels)
    valid = valid & enough
    return ClubbCoefficientProfile(z_m=z_m, C_K=C_K, valid=valid)


class PrandtlProfile(NamedTuple):
    """Turbulent Prandtl number ``Pr_t = K_m/K_h`` per interior interface."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    Pr_t: jax.Array    # (nlev-1,) dimensionless Pr_t = K_m/K_h
    valid: jax.Array   # (nlev-1,) bool


def diagnose_prandtl_number(
    les_state,
    height_coord,
    *,
    qv_slot: int = 0,
    min_valid_levels: int = _MIN_VALID_CK_LEVELS,
) -> PrandtlProfile:
    """Diagnose the DIMENSIONLESS turbulent Prandtl number ``Pr_t = K_m/K_h``.

    The ratio of the LES MOMENTUM diffusivity ``K_m`` (shear-projected, from
    ``⟨w'u'⟩,⟨w'v'⟩``) to the HEAT diffusivity ``K_h`` (from ``⟨w'θ'⟩``), both at
    the same interior interfaces — the inverse of the GCM ``K_h = K_m/Pr_t``.
    Being a pure ratio, ``Pr_t`` carries NO wp2-identification assumption (unlike
    ``C_K``), so it transfers cleanly.  ``K_h`` reuses
    :func:`diagnose_eddy_diffusivity` (ascending, co-located with the momentum
    interfaces).  Fewer than ``min_valid_levels`` valid interfaces ⇒ the column
    is flagged invalid.  Pure-JAX, AD-safe.
    """
    kh = diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)  # heat K, ascending
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    u_mean = jnp.mean(les_state.u.data, axis=(0, 1))
    v_mean = jnp.mean(les_state.v.data, axis=(0, 1))
    z_full = jnp.asarray(height_coord.z_full)
    K_m, km_valid = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1],
        u_mean[::-1], v_mean[::-1], z_full[::-1],
    )
    Pr_t, valid = prandtl_number_from_diffusivities(
        K_m, km_valid, kh.K, kh.valid
    )
    enough = jnp.sum(valid) >= int(min_valid_levels)
    return PrandtlProfile(z_m=kh.z_m, Pr_t=Pr_t, valid=valid & enough)


class CEpsProfile(NamedTuple):
    """Dimensionless wp2-dissipation coefficient ``C_eps`` per interior interface."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    C_eps: jax.Array   # (nlev-1,) dimensionless C_eps = P·ℓ/wp2^{3/2}
    valid: jax.Array   # (nlev-1,) bool


def diagnose_c_eps_coefficient(
    les_state,
    height_coord,
    *,
    l_mix_max: float,
    qv_slot: int = 0,
    min_valid_levels: int = _MIN_VALID_CK_LEVELS,
) -> CEpsProfile:
    """Diagnose the DIMENSIONLESS wp2-dissipation coefficient ``C_eps`` so the
    GCM's equilibrium ``wp2`` tracks the LES ``w'²`` — closing the iter-46
    ``C_K`` wp2-identification gap (when ``C_eps`` and ``C_K`` are both corrected,
    the GCM ``wp2 → w'²_LES`` makes ``Km = C_K·ℓ·√wp2`` reproduce the LES K_m).

    Inverts the steady-state lite ``w'²`` budget ``C_eps = P·ℓ/wp2^{3/2}`` with the
    net production ``P = K_m·S² − K_h·N²``, all co-located at the ``nlev-1``
    ascending interior interfaces (the same convention as ``C_K`` / ``Pr_t``):
      * ``K_m`` (momentum) + ``K_h`` (heat) reuse the iter-46/47 inversions;
      * ``S² = (∂⟨u⟩/∂z)² + (∂⟨v⟩/∂z)²`` (mean-wind shear);
      * ``N² = (g/⟨θ_v⟩) ∂⟨θ_v⟩/∂z`` (co-located interface ``⟨θ_v⟩``, same ε
        convention as :func:`diagnose_entrainment`);
      * ``ℓ`` = the SAME Blackadar :func:`mixing_length`; ``wp2`` = resolved
        ``w'²`` (interior).
    See :func:`c_eps_from_budget` for the validity masks + the dropped-transport
    LIMITATION.  Pure-JAX, AD-safe.
    """
    kh = diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)  # heat K, ascending
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    wp2_half = vertical_velocity_variance_plane(les_state, height_coord)
    u_mean = jnp.mean(les_state.u.data, axis=(0, 1))
    v_mean = jnp.mean(les_state.v.data, axis=(0, 1))
    z_full = jnp.asarray(height_coord.z_full)
    # θ_v mean (same ε convention as diagnose_entrainment / compute_cape).
    theta_total = height_coord.theta_ref + les_state.theta_prime.data
    q_v = les_state.tracers.data[..., qv_slot]
    coeff = 1.0 / constants.epsilon - 1.0
    thetav_mean = jnp.mean(theta_total * (1.0 + coeff * q_v), axis=(0, 1))
    # Reverse all to ascending, co-located at the interior interfaces.
    u_asc, v_asc, z_asc = u_mean[::-1], v_mean[::-1], z_full[::-1]
    thetav_asc = thetav_mean[::-1]
    K_m, km_valid = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1], u_asc, v_asc, z_asc)
    du_dz = mean_gradient_at_interfaces(u_asc, z_asc)
    dv_dz = mean_gradient_at_interfaces(v_asc, z_asc)
    shear_sq = du_dz ** 2 + dv_dz ** 2
    dthetav_dz = mean_gradient_at_interfaces(thetav_asc, z_asc)
    thetav_iface = 0.5 * (thetav_asc[1:] + thetav_asc[:-1])          # co-located ⟨θ_v⟩
    N_sq = constants.g * dthetav_dz / jnp.maximum(thetav_iface, 1.0)
    z_m = fluxes.z_half_interior[::-1]
    c_eps, valid = c_eps_from_budget(
        K_m, km_valid, kh.K, kh.valid, shear_sq, N_sq,
        mixing_length(z_m, l_mix_max), wp2_half[1:-1][::-1])
    enough = jnp.sum(valid) >= int(min_valid_levels)
    return CEpsProfile(z_m=z_m, C_eps=c_eps, valid=valid & enough)


def diagnose_column_coefficient(
    les_state,
    height_coord,
    *,
    method: str = "eddy_diffusivity",
    qv_slot: int = 0,
    l_mix_max: float | None = None,
):
    """Dispatch to the chosen closure-coefficient diagnosis (raises on unknown).

    Returns an :class:`EddyDiffusivityProfile` for ``"eddy_diffusivity"``, an
    :class:`~legoesm.atmosphere.dynamics.les_closure_diagnosis.EntrainmentDiagnosis`
    for ``"entrainment"``, a :class:`ClubbCoefficientProfile` for
    ``"clubb_coefficient"`` (which requires the GCM ``l_mix_max``), or a
    :class:`PrandtlProfile` for ``"prandtl_number"``.
    """
    if method == "eddy_diffusivity":
        return diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)
    if method == "entrainment":
        return diagnose_entrainment(les_state, height_coord, qv_slot)
    if method == "prandtl_number":
        return diagnose_prandtl_number(les_state, height_coord, qv_slot=qv_slot)
    if method in ("clubb_coefficient", "c_eps"):
        if l_mix_max is None:
            raise ValueError(
                f"method={method!r} requires l_mix_max (the GCM "
                "CLUBBLiteConfig.l_mix_max) to evaluate the mixing length."
            )
        if method == "clubb_coefficient":
            return diagnose_clubb_coefficient(
                les_state, height_coord, l_mix_max=l_mix_max, qv_slot=qv_slot
            )
        return diagnose_c_eps_coefficient(
            les_state, height_coord, l_mix_max=l_mix_max, qv_slot=qv_slot
        )
    raise ValueError(
        f"Unknown column-LES diagnosis method {method!r}; choose from {_METHODS}."
    )
