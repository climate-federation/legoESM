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
    state, height_coord, qc_slot: int = 1,
    threshold: float = 1.0e-6,
) -> jax.Array:
    """Per-level cloud fraction: fraction of cells with q_c > threshold.

    Threshold default 1 mg/kg matches the standard RCEMIP cloud-mask
    convention (Wing 2018 Tab A2).
    """
    _validate_plane_state(state, height_coord)
    _validate_slot(qc_slot, state.tracers.data.shape[-1], "qc_slot")
    q_c = state.tracers.data[..., qc_slot]   # (ny, nx, nlev)
    is_cloud = (q_c > threshold).astype(jnp.float64)
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
