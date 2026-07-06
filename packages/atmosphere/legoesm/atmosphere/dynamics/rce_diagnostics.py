"""RCE / RCEMIP diagnostic helpers for the plane NH CRM.

Plane-state diagnostic suite for analysing RCE-style integrations:

* :func:`column_water_vapor_plane` — Σ_k q_v · rho_total · dz [kg/m²].
* :func:`column_total_water_plane` — same but Σ over selected tracer
  slots (q_v + q_c + q_r + ...). Reuses the moist-mass weighting.
* :func:`column_moist_static_energy_plane` — Σ (c_pd·T + L_v·q_v + g·z)·
  rho_total·dz [J/m²]. Uses ``rho_ref + rho_prime`` for the column
  weight (height-coordinate convention, not sigma).
* :func:`domain_mean_profiles_plane` — horizontal mean of T(z), q_v(z),
  theta_e(z) for plot/inspection.
* :func:`cloud_fraction_profile_plane` — per-level fraction of cells
  with q_c > threshold (default 1e-6 kg/kg).
* :func:`pseudo_equivalent_potential_temperature` — simplified θ_e proxy
  per cell. NOT the full Bolton (1980) formulation.

All functions are pure JAX (JIT-traceable) and reuse
``legoesm.constants`` for c_pd, L_v, g.

Scope guards (Codex iter-2)
---------------------------
Every public function validates the plane-shape contract
``(ny, nx, nlev)`` for 3D state fields and ``(ny, nx, nlev, n_tracers)``
for tracers. Every tracer-slot argument is range-checked against
``n_tracers`` and rejects negative indices (which would silently wrap
to the end of the tracer axis and return the wrong species).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from legoesm import constants


def _validate_plane_state(state, height_coord) -> None:
    """Plane-only shape contract — rejects non-3D state fields, 4D
    tracers, and mismatched height-coordinate arrays."""
    rho_p = state.rho_prime.data
    if rho_p.ndim != 3:
        raise ValueError(
            f"RCE diagnostics are plane-only: expected rho_prime "
            f"shape (ny, nx, nlev); got ndim={rho_p.ndim} "
            f"shape={rho_p.shape}."
        )
    ny, nx, nlev = rho_p.shape
    theta_p = state.theta_prime.data
    if theta_p.shape != (ny, nx, nlev):
        raise ValueError(
            f"theta_prime shape {theta_p.shape} != "
            f"rho_prime shape {(ny, nx, nlev)}."
        )
    tracers = state.tracers.data
    if tracers.ndim != 4 or tracers.shape[:3] != (ny, nx, nlev):
        raise ValueError(
            f"tracers shape {tracers.shape} incompatible with plane "
            f"layout (ny={ny}, nx={nx}, nlev={nlev}, n_tracers)."
        )
    for name in ("rho_ref", "theta_ref", "exner_ref", "dz", "z_full"):
        arr = getattr(height_coord, name)
        if arr.shape != (nlev,):
            raise ValueError(
                f"height_coord.{name} shape {arr.shape} != (nlev={nlev},)."
            )


def _validate_slot(slot: int, n_tracers: int, name: str) -> None:
    """Reject non-int / negative / out-of-range tracer slot indices.
    Negative indices would silently wrap to the end of the tracer
    axis + return the wrong species."""
    if not isinstance(slot, int):
        raise ValueError(
            f"{name} must be int; got {slot!r} of type "
            f"{type(slot).__name__}."
        )
    if slot < 0 or slot >= n_tracers:
        raise ValueError(
            f"{name}={slot} out of range [0, {n_tracers}); negative "
            f"indices are not allowed (would silently wrap)."
        )


def _validate_slots(
    slots: Sequence[int], n_tracers: int, name: str,
) -> None:
    """Reject empty / duplicate / negative / out-of-range slot lists."""
    if len(slots) == 0:
        raise ValueError(f"{name} must contain at least one slot.")
    if len(set(slots)) != len(slots):
        raise ValueError(
            f"{name} contains duplicates: {slots} — double-counting "
            f"would corrupt the column integral."
        )
    for idx in slots:
        _validate_slot(idx, n_tracers, f"{name} entry")


def column_water_vapor_plane(state, height_coord, qv_slot: int = 0) -> jax.Array:
    """Column-integrated water vapor [kg/m²] per horizontal cell.

    ``CWV(j, i) = Σ_k q_v[j,i,k] · rho_total[j,i,k] · dz[k]``.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    rho_total = height_coord.rho_ref + state.rho_prime.data
    q_v = state.tracers.data[..., qv_slot]
    return jnp.sum(q_v * rho_total * height_coord.dz, axis=-1)


def column_total_water_plane(
    state, height_coord,
    water_slot_indices: Sequence[int] = (0, 1, 2),
) -> jax.Array:
    """Column total water [kg/m²]: ``Σ_slot Σ_k q · rho · dz``."""
    _validate_plane_state(state, height_coord)
    _validate_slots(
        water_slot_indices, state.tracers.data.shape[-1],
        "water_slot_indices",
    )
    rho_total = height_coord.rho_ref + state.rho_prime.data
    q_total = jnp.zeros_like(rho_total)
    for idx in water_slot_indices:
        q_total = q_total + state.tracers.data[..., idx]
    return jnp.sum(q_total * rho_total * height_coord.dz, axis=-1)


def column_moist_static_energy_plane(
    state, height_coord, qv_slot: int = 0,
) -> jax.Array:
    """Column-integrated MSE [J/m²] per horizontal cell.

    ``MSE(j, i) = Σ_k (c_pd·T + L_v·q_v + g·z) · rho_total · dz``

    Uses height-coordinate z_full + Exner-derived T:
    ``T = (theta_ref + theta_prime) · exner_ref``.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    theta_total = height_coord.theta_ref + state.theta_prime.data
    T = theta_total * height_coord.exner_ref
    q_v = state.tracers.data[..., qv_slot]
    z = height_coord.z_full   # (nlev,) broadcasts
    rho_total = height_coord.rho_ref + state.rho_prime.data
    mse_density = (
        constants.c_pd * T
        + constants.L_v * q_v
        + constants.g * z
    )
    return jnp.sum(mse_density * rho_total * height_coord.dz, axis=-1)


def moist_static_energy_3d_plane(
    state, height_coord, qv_slot: int = 0,
) -> jax.Array:
    """3D moist static energy [J/kg] per grid cell.

    ``MSE(j, i, k) = c_pd·T(j,i,k) + L_v·q_v(j,i,k) + g·z(k)``

    Returns the per-cell MSE *density* (specific MSE) on the
    (ny, nx, nlev) grid; integrating
    ``MSE_density · rho_total · dz`` along k recovers
    :func:`column_moist_static_energy_plane`.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    theta_total = height_coord.theta_ref + state.theta_prime.data
    T = theta_total * height_coord.exner_ref
    q_v = state.tracers.data[..., qv_slot]
    z = height_coord.z_full   # (nlev,) broadcasts
    return constants.c_pd * T + constants.L_v * q_v + constants.g * z


def temperature_3d_plane(state, height_coord) -> jax.Array:
    """3D temperature [K] from Exner conversion of theta_total."""
    _validate_plane_state(state, height_coord)
    theta_total = height_coord.theta_ref + state.theta_prime.data
    return theta_total * height_coord.exner_ref


def pseudo_equivalent_potential_temperature(
    state, height_coord, qv_slot: int = 0,
) -> jax.Array:
    """Simplified θ_e PROXY per cell [K] — NOT the full Bolton (1980).

    Computes the simplest reversible-pseudo-adiabatic form

    ``θ_e_proxy = θ · exp(L_v · q_v / (c_pd · T))``

    where ``q_v`` is SPECIFIC HUMIDITY (not mixing ratio) and ``T``
    is the cell temperature (NOT the LCL temperature). This is a
    parcel-energy proxy suitable for RCE convective-stability
    inspection only.

    Differences vs. Bolton (1980) eq. 43
    ------------------------------------
    * Bolton uses ``r`` (mixing ratio) and ``T_L`` (temperature at
      the lifting condensation level), not ``q_v`` and ``T``.
    * Bolton's pre-exponential factor includes a ``(1000/p)^κ`` term
      with κ adjusted for moist air, not the dry-air θ used here.
    * The Bolton exponent has the form
      ``(3036/T_L - 1.78) · r · (1 + 0.448·r)``, NOT
      ``L_v · q_v / (c_pd · T)``.

    Typical disagreement with the full Bolton expression is 1–5 K
    in tropical RCE conditions — use this for relative inspection
    (e.g., spotting CAPE evolution), not absolute θ_e reporting.

    Invalid-state handling (Codex iter-3)
    -------------------------------------
    Cells with ``T <= 0`` (unphysical — indicates broken
    thermodynamic state) return NaN rather than silently producing
    inf via a clipped denominator. The caller's downstream
    isfinite-check then surfaces the bad columns. The denominator
    is computed once + reused.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    theta_total = height_coord.theta_ref + state.theta_prime.data
    T = theta_total * height_coord.exner_ref
    q_v = state.tracers.data[..., qv_slot]
    denom = constants.c_pd * T
    return jnp.where(
        denom > 0.0,
        theta_total * jnp.exp(
            constants.L_v * q_v / jnp.where(denom > 0.0, denom, 1.0),
        ),
        jnp.nan,
    )


class DomainMeanProfiles(NamedTuple):
    """Per-level horizontal means + standard deviations."""
    T: jax.Array              # (nlev,)
    q_v: jax.Array            # (nlev,)
    theta_e_proxy: jax.Array  # (nlev,) — simplified θ_e (NOT Bolton)
    T_std: jax.Array          # (nlev,)
    q_v_std: jax.Array        # (nlev,)


def domain_mean_profiles_plane(
    state, height_coord, qv_slot: int = 0,
) -> DomainMeanProfiles:
    """Horizontal mean + std-dev of T(z), q_v(z), θ_e_proxy(z).

    Note: ``theta_e_proxy`` is the simplified
    :func:`pseudo_equivalent_potential_temperature` — NOT Bolton (1980).
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    theta_total = height_coord.theta_ref + state.theta_prime.data
    T = theta_total * height_coord.exner_ref         # (ny, nx, nlev)
    q_v = state.tracers.data[..., qv_slot]
    theta_e_proxy = pseudo_equivalent_potential_temperature(
        state, height_coord, qv_slot=qv_slot,
    )
    return DomainMeanProfiles(
        T=jnp.mean(T, axis=(0, 1)),
        q_v=jnp.mean(q_v, axis=(0, 1)),
        theta_e_proxy=jnp.mean(theta_e_proxy, axis=(0, 1)),
        T_std=jnp.std(T, axis=(0, 1)),
        q_v_std=jnp.std(q_v, axis=(0, 1)),
    )


def cloud_fraction_profile_plane(
    state, height_coord, qc_slot: int = 1, qi_slot: int | None = 3,
    threshold: float = 1.0e-6,
) -> jax.Array:
    """Per-level cloud fraction: fraction of cells with CLOUD CONDENSATE
    (cloud water q_c + cloud ice q_i) > threshold.

    Including cloud ICE is essential for deep convection: the upper-
    tropospheric ANVIL is almost entirely ice, so a q_c-only count misses it
    and badly under-estimates the RCE cloud fraction (RCEMIP masks on q_c+q_i).
    Threshold default 1 mg/kg matches the RCEMIP cloud-mask convention
    (Wing 2018 Tab A2). ``qi_slot=None`` (or a slot beyond the tracer count,
    e.g. a warm-only Kessler run) counts cloud water only.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qc_slot, state.tracers.data.shape[-1], "qc_slot")
    q_cloud = state.tracers.data[..., qc_slot]   # (ny, nx, nlev)
    n_tr = state.tracers.data.shape[-1]
    if qi_slot is not None and qi_slot < n_tr:
        q_cloud = q_cloud + state.tracers.data[..., qi_slot]
    is_cloud = (q_cloud > threshold).astype(jnp.float64)
    return jnp.mean(is_cloud, axis=(0, 1))


def precipitation_rate_proxy_plane(
    state, height_coord,
    qr_slot: int = 2,
    fall_speed: float = 5.0,
) -> jax.Array:
    """Surface precipitation-rate PROXY [kg/m²/s] per horizontal cell.

    Diagnostic-only approximation: assumes ``q_r`` at the lowest
    model level falls at a fixed terminal velocity ``fall_speed``.
    ``P(j, i) = q_r[j,i,k_sfc] · rho_total[j,i,k_sfc] · fall_speed``.

    Convert to RCEMIP mm/day: ``P_mm_day = P [kg/m²/s] · 86400``
    (using rho_water = 1000 kg/m³ → 1 mm = 1 kg/m²).

    This is NOT a tracking of microphysical precipitation flux —
    use the microphysics scheme's ``precipitation`` output for an
    exact rate. Use this for quick sanity checks when the
    microphysics output is not pipelined into the diagnostics path.
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qr_slot, state.tracers.data.shape[-1], "qr_slot")
    k_sfc = -1
    q_r_sfc = state.tracers.data[..., k_sfc, qr_slot]
    rho_sfc = (
        height_coord.rho_ref[k_sfc]
        + state.rho_prime.data[..., k_sfc]
    )
    return q_r_sfc * rho_sfc * fall_speed


# ---------------------------------------------------------------------------
# Convective-intensity diagnostics (the comparison-protocol "anomaly
# magnitudes": resolved w variance, updraft mass flux, condensate profiles).
# The task compares MAGNITUDES OF ANOMALIES + PROFILES against SAM, not
# snapshots — these are the CRM signatures SAM reports (W2, MFU, QC/QP).
# ---------------------------------------------------------------------------

def vertical_velocity_variance_plane(state, height_coord) -> jax.Array:
    """Resolved vertical-velocity variance ``w'²(z)`` [m²/s²] at HALF levels.

    SAM's primary convective-intensity diagnostic (``statistics.f90`` ``W2``).
    ``w' = w − horizontal mean``; variance over the doubly-periodic (y, x)
    plane at each level. Returned on the ``w`` grid (``nlev+1`` interface
    levels) where ``w`` natively lives — NO interpolation to full levels, so
    the variance is unbiased by half-level averaging (averaging w to cell
    centres before the variance would smooth + underestimate the peak).
    """
    _validate_plane_state(state, height_coord)
    w = state.w.data                       # (ny, nx, nlev+1), half levels
    return jnp.var(w, axis=(0, 1))         # (nlev+1,)


def _validate_w_half(state) -> None:
    """Reject a ``w`` field that is not on the ``nlev+1`` interface grid."""
    rho_p = state.rho_prime.data
    ny, nx, nlev = rho_p.shape
    w = state.w.data
    if w.shape != (ny, nx, nlev + 1):
        raise ValueError(
            f"resolved-flux diagnostics expect w on half levels "
            f"shape (ny={ny}, nx={nx}, nlev+1={nlev + 1}); got {w.shape}."
        )


def _resolved_flux_interfaces(
    w_half: jax.Array, phi_full: jax.Array
) -> jax.Array:
    """Domain-mean resolved eddy flux ``<w'φ'>`` at interior interfaces.

    ``w_half`` is ``(ny, nx, nlev+1)`` (native ``w`` grid); ``phi_full`` is a
    full-level scalar ``(ny, nx, nlev)``.  The flux naturally lives at the
    ``nlev-1`` interior interfaces between cells, so the scalar is averaged to
    those interfaces (``φ_{k+1/2}=½(φ_k+φ_{k+1})``) while ``w`` stays on its
    native grid — consistent with :func:`vertical_velocity_variance_plane`
    (averaging ``w`` to cell centres would smooth and bias the flux).
    Perturbations are taken from the doubly-periodic ``(y, x)`` horizontal mean
    at each level, so any horizontally-uniform reference (``theta_ref`` etc.)
    cancels.  The two rigid boundary interfaces carry ``w=0`` and hence zero
    flux; they are excluded.  Returns ``(nlev-1,)``.
    """
    w_int = w_half[..., 1:-1]                                  # (ny,nx,nlev-1)
    phi_iface = 0.5 * (phi_full[..., :-1] + phi_full[..., 1:])  # (ny,nx,nlev-1)
    w_pert = w_int - jnp.mean(w_int, axis=(0, 1), keepdims=True)
    phi_pert = phi_iface - jnp.mean(phi_iface, axis=(0, 1), keepdims=True)
    return jnp.mean(w_pert * phi_pert, axis=(0, 1))           # (nlev-1,)


class ResolvedTurbulentFluxes(NamedTuple):
    """LES-resolved turbulent fluxes at the ``nlev-1`` interior interfaces.

    All fluxes are *kinematic* domain means over the doubly-periodic plane, on
    the ``z_half_interior`` interface grid where ``w`` natively lives.  These
    are the quantities a closure-coefficient diagnosis (entrainment / eddy
    diffusivity) consumes — see ``docs/COMPARE_REANALYSIS.md`` stage 6.
    """

    z_half_interior: jax.Array  # (nlev-1,) interior interface heights [m]
    w_theta: jax.Array          # (nlev-1,) resolved potential-temp flux [K m/s]
    w_qv: jax.Array             # (nlev-1,) resolved water-vapor flux [(kg/kg) m/s]
    w_u: jax.Array              # (nlev-1,) resolved zonal-momentum flux [m²/s²]
    w_v: jax.Array              # (nlev-1,) resolved merid-momentum flux [m²/s²]
    w_thetav: jax.Array         # (nlev-1,) resolved buoyancy (virtual-θ) flux [K m/s]


def resolved_turbulent_fluxes_plane(
    state, height_coord, qv_slot: int = 0
) -> ResolvedTurbulentFluxes:
    """Resolved ``w'θ'``, ``w'q_v'``, ``w'u'``, ``w'v'``, ``w'θ_v'`` profiles.

    Computes the LES-resolved kinematic eddy fluxes (perturbations from the
    horizontal mean) at the interior interfaces.  The buoyancy flux uses the
    virtual potential temperature ``θ_v = θ (1 + (1/ε − 1) q_v)`` with the same
    ``ε`` convention as :func:`compute_cape` (no re-derived constant).  The
    horizontally-uniform ``theta_ref`` cancels in the perturbation, so the full
    ``θ = theta_ref + theta_prime`` is used directly.
    """
    _validate_plane_state(state, height_coord)
    _validate_w_half(state)
    _validate_slot(qv_slot, state.tracers.data.shape[-1], "qv_slot")
    ny, nx, nlev = state.rho_prime.data.shape
    if jnp.asarray(height_coord.z_half).shape != (nlev + 1,):
        raise ValueError(
            f"height_coord.z_half shape {jnp.asarray(height_coord.z_half).shape}"
            f" != (nlev+1={nlev + 1},)."
        )
    for name in ("u", "v"):
        arr = getattr(state, name).data
        if arr.shape != (ny, nx, nlev):
            raise ValueError(
                f"state.{name} shape {arr.shape} != full-level layout "
                f"(ny={ny}, nx={nx}, nlev={nlev})."
            )
    w = state.w.data
    theta_total = height_coord.theta_ref + state.theta_prime.data
    q_v = state.tracers.data[..., qv_slot]
    coeff = 1.0 / constants.epsilon - 1.0
    theta_v = theta_total * (1.0 + coeff * q_v)
    return ResolvedTurbulentFluxes(
        z_half_interior=jnp.asarray(height_coord.z_half)[1:-1],
        w_theta=_resolved_flux_interfaces(w, theta_total),
        w_qv=_resolved_flux_interfaces(w, q_v),
        w_u=_resolved_flux_interfaces(w, state.u.data),
        w_v=_resolved_flux_interfaces(w, state.v.data),
        w_thetav=_resolved_flux_interfaces(w, theta_v),
    )


def updraft_mass_flux_plane(
    state, height_coord, w_threshold: float = 0.0,
) -> jax.Array:
    """Convective updraft mass flux ``M_up(z)`` [kg/m²/s] on the w HALF grid.

    SAM's ``MFU`` convention (``statistics.f90`` accumulates ``rhow·w`` on the
    w levels): the upward convective mass transport per unit TOTAL area,

        ``M_up(k) = < rho_ref_half(k) · w(k) · H(w(k) − w_threshold) >_{y,x}``

    (domain mean of ``rhow·w`` restricted to upward cells, NOT a conditional
    in-updraft mean). Computed NATIVELY on the ``w`` interface grid
    (``nlev+1``) — no half→full averaging — to match SAM's ``rhow·w``
    accumulation and co-locate with :func:`vertical_velocity_variance_plane`.
    The base-state reference density ``rho_ref_half`` is SAM's ``rhow`` (the
    anelastic base-state ρ at w levels), not the perturbed ``rho_total``.
    ``w_threshold=0`` counts all upward motion; SAM also reports a ``w>1 m/s``
    core flux (pass ``w_threshold=1.0``).
    """
    _validate_plane_state(state, height_coord)
    w = state.w.data                                   # (ny, nx, nlev+1)
    rho_w = jnp.asarray(height_coord.rho_ref_half)     # (nlev+1,), SAM rhow
    is_up = (w > w_threshold).astype(w.dtype)
    return jnp.mean(rho_w * w * is_up, axis=(0, 1))    # (nlev+1,)


class CondensateProfiles(NamedTuple):
    """Horizontal-mean condensate mixing-ratio profiles [kg/kg]."""
    q_cloud: jax.Array   # (nlev,) suspended cloud condensate (q_c + q_i)
    q_precip: jax.Array  # (nlev,) precipitating condensate (q_r + q_s + q_g)
    q_total: jax.Array   # (nlev,) q_cloud + q_precip


def condensate_profile_plane(
    state, height_coord,
    cloud_slots: Sequence[int] = (1, 3),
    precip_slots: Sequence[int] = (2, 4, 5),
) -> CondensateProfiles:
    """Horizontal-mean cloud + precipitating condensate profiles [kg/kg].

    Tracer layout (fixed, MUST hold): ``[0]q_v [1]q_c [2]q_r [3]q_i [4]q_s
    [5]q_g`` ⇒ suspended cloud condensate ``q_c+q_i`` (slots 1, 3) and
    precipitating condensate ``q_r+q_s+q_g`` (slots 2, 4, 5). Slots at or
    beyond the tracer count are dropped (warm-rain runs carry only
    ``q_c``/``q_r``), so the same call works for warm-rain and double-moment
    Morrison. Negative slots raise — a scheme that REORDERS the fixed layout
    must pass explicit ``cloud_slots``/``precip_slots`` (codex iter-43 C: the
    silent tail-drop is only safe under the documented layout).
    """
    _validate_plane_state(state, height_coord)
    n_tracers = state.tracers.data.shape[-1]
    for name, slots in (("cloud_slots", cloud_slots),
                        ("precip_slots", precip_slots)):
        if any(s < 0 for s in slots):
            raise ValueError(f"{name} must be non-negative, got {tuple(slots)}")
    cloud = tuple(s for s in cloud_slots if s < n_tracers)
    precip = tuple(s for s in precip_slots if s < n_tracers)
    tr = state.tracers.data
    q_cloud = jnp.zeros_like(tr[..., 0])
    for s in cloud:
        q_cloud = q_cloud + tr[..., s]
    q_precip = jnp.zeros_like(tr[..., 0])
    for s in precip:
        q_precip = q_precip + tr[..., s]
    return CondensateProfiles(
        q_cloud=jnp.mean(q_cloud, axis=(0, 1)),
        q_precip=jnp.mean(q_precip, axis=(0, 1)),
        q_total=jnp.mean(q_cloud + q_precip, axis=(0, 1)),
    )


class CRMComparisonProfiles(NamedTuple):
    """Full CRM-vs-SAM comparison bundle: profiles + bulk scalars.

    The deliverable the task's comparison protocol needs — the magnitudes of
    the resolved convective anomalies + the mean profiles, NOT snapshots.
    Vertical-grid note: the ``*_half`` fields are on the ``w`` interface grid
    (``nlev+1``); every other profile is on the FULL levels (``nlev``).
    """
    z_full: jax.Array                   # (nlev,) full-level heights [m]
    z_half: jax.Array                   # (nlev+1,) interface heights [m]
    T: jax.Array                        # (nlev,) mean temperature [K]
    q_v: jax.Array                      # (nlev,) mean vapor [kg/kg]
    T_std: jax.Array                    # (nlev,) T anomaly magnitude [K]
    q_v_std: jax.Array                  # (nlev,) q_v anomaly magnitude [kg/kg]
    w_var_half: jax.Array               # (nlev+1,) resolved w'² [m²/s²]
    updraft_mass_flux_half: jax.Array   # (nlev+1,) M_up [kg/m²/s]
    q_cloud: jax.Array                  # (nlev,) cloud condensate [kg/kg]
    q_precip: jax.Array                 # (nlev,) precip condensate [kg/kg]
    cloud_fraction: jax.Array           # (nlev,) fractional cloud cover
    cwv_mean: jax.Array                 # scalar domain-mean CWV [kg/m²]
    max_w: jax.Array                    # scalar max|w| [m/s]
    precip_mean: jax.Array              # scalar domain-mean sfc precip [kg/m²/s]


def crm_comparison_profiles_plane(
    state, height_coord, qv_slot: int = 0, qc_slot: int = 1,
    cloud_threshold: float = 1.0e-6,
) -> CRMComparisonProfiles:
    """Assemble the full CRM comparison bundle from a plane state.

    One call → every profile + bulk scalar the GATE/LBA/RCE-vs-SAM comparison
    compares (``CRM_faithful_SAM.md`` protocol). Pure composition of the
    leaf diagnostics in this module — no new numerics.
    """
    _validate_plane_state(state, height_coord)
    means = domain_mean_profiles_plane(state, height_coord, qv_slot=qv_slot)
    cond = condensate_profile_plane(state, height_coord)
    return CRMComparisonProfiles(
        z_full=jnp.asarray(height_coord.z_full),
        z_half=jnp.asarray(height_coord.z_half),
        T=means.T, q_v=means.q_v, T_std=means.T_std, q_v_std=means.q_v_std,
        w_var_half=vertical_velocity_variance_plane(state, height_coord),
        updraft_mass_flux_half=updraft_mass_flux_plane(state, height_coord),
        q_cloud=cond.q_cloud, q_precip=cond.q_precip,
        cloud_fraction=cloud_fraction_profile_plane(
            state, height_coord, qc_slot=qc_slot, threshold=cloud_threshold),
        cwv_mean=jnp.mean(
            column_water_vapor_plane(state, height_coord, qv_slot=qv_slot)),
        max_w=jnp.max(jnp.abs(state.w.data)),
        precip_mean=jnp.mean(
            precipitation_rate_proxy_plane(state, height_coord)),
    )
