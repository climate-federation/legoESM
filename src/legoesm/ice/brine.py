"""Sea-ice bulk salinity + brine rejection.

Tracks a single bulk-mean salinity ``S_ice`` per ice category and
computes the salt mass flux to the ocean implied by the per-process
ice-mass budget over a thermodynamic step:

- **Lead freezing**: open-water ice forms at ``S_ice_new`` (low
  bulk salinity, default 4 PSU).  The remainder of the ocean's
  salt stays in the ocean → brine rejection into ocean.
- **Basal / surface melt and sublimation**: ice salt is returned
  to the ocean at the ice's current ``S_ice``.
- **Snow-ice flooding**: new white ice forms with salinity
  ``S_white = pore_frac · S_ocean``.  The seawater that filled
  the snow pores carried this salt out of the ocean column → salt
  uptake into ice (negative salt flux).

Net salt flux to ocean:
    salt_flux  = − Δ(S_ice · V_ice · ρ_ice) / dt          [kg(salt)/m²/s]
              = + brine_release_from_freezing
                + salt_release_from_melt
                − salt_uptake_in_white_ice

with ``S_ice`` in PSU (≈ g/kg → ×1e-3 to get kg of salt per kg of ice).

When ``BrineConfig.enabled = False`` the helpers degenerate to
zero salinity / zero salt flux, preserving the legacy
freshwater-only convention.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


PSU_TO_KG_PER_KG = 1.0e-3  # 1 PSU ≈ 1 g/kg = 1e-3 kg of salt per kg of seawater


class SaltBudgetResult(NamedTuple):
    """Output of :func:`update_salinity_and_salt_flux`.

    ``S_ice_new`` is the post-step bulk ice salinity per category;
    ``salt_flux_to_ocean`` is positive when salt enters the ocean
    (brine rejection during freezing or salt release during melt).
    """
    S_ice_new: jnp.ndarray
    salt_flux_to_ocean: jnp.ndarray   # [kg(salt)/m²/s]


def update_salinity_and_salt_flux(
    S_ice_old: jnp.ndarray,
    V_ice_old: jnp.ndarray,
    V_ice_new: jnp.ndarray,
    delta_V_lead_freeze: jnp.ndarray,
    delta_V_white_ice: jnp.ndarray,
    *,
    rho_ice: float,
    dt: float,
    S_lead_ice: float,
    S_white_ice: float,
    S_ice_min: float = 0.0,
    S_ice_max: float = 12.0,
) -> SaltBudgetResult:
    """Update bulk ice salinity and emit ocean salt-flux diagnostic.

    Mass-budget convention: ``V_ice`` is ice volume per unit
    grid-cell area ``[m_ice]`` (i.e. ``h_ice * a_ice``).
    ``delta_V_lead_freeze`` and ``delta_V_white_ice`` are
    non-negative; the implied basal + surface melt + sublimation
    contribution is the residual

        ΔV_melt_release = V_old + ΔV_freeze + ΔV_white − V_new   (≥ 0 when net melt)

    The post-step salt content per area is

        Salt_new = S_ice_old · V_remain
                 + S_lead   · ΔV_freeze
                 + S_white  · ΔV_white                 [kg salt / m²]

    with ``V_remain = V_old − ΔV_melt_release`` (the ice that
    *survived* and kept its old salinity), and
    ``Salt`` converted to mass via ``× rho_ice × 1e-3``.

    Salt flux to ocean is the *negative* time-derivative of the
    ice column's salt mass — positive when ice loses salt to ocean.

    Parameters
    ----------
    S_ice_old : array
        Pre-step bulk salinity per category [PSU].
    V_ice_old, V_ice_new : array
        Pre / post ice volume per area per category [m].
    delta_V_lead_freeze : array
        Volume of new ice formed in leads this step [m, ≥ 0].
    delta_V_white_ice : array
        Volume of white ice formed by snow-ice flooding [m, ≥ 0].
    rho_ice : float
        Ice density [kg/m³].
    dt : float
        Time step [s].
    S_lead_ice : float
        Salinity of newly frozen lead ice [PSU] (typically 4).
    S_white_ice : float
        Salinity of white ice from flooding [PSU] (typically
        ``0.5 · S_ocean_ref ≈ 17``).
    S_ice_min, S_ice_max : float
        Numerical clamp [PSU].

    Returns
    -------
    result : :class:`SaltBudgetResult`
    """
    # Pre-step salt mass per area in the ice column [kg salt / m²].
    salt_old = S_ice_old * V_ice_old * rho_ice * PSU_TO_KG_PER_KG

    # Implied melt-release volume: ice that disappeared between
    # ``V_old`` and ``V_new`` after accounting for freeze gains.
    delta_V_melt = jnp.maximum(
        V_ice_old + delta_V_lead_freeze + delta_V_white_ice - V_ice_new,
        0.0,
    )
    V_remain = jnp.maximum(V_ice_old - delta_V_melt, 0.0)

    # Post-step salt mass per area: melt releases ice that had the
    # *old* bulk salinity; freezing adds ice at the (fixed) low
    # ``S_lead_ice``; flooding adds white ice with elevated
    # ``S_white_ice``.
    salt_new = (
        S_ice_old * V_remain
        + S_lead_ice * delta_V_lead_freeze
        + S_white_ice * delta_V_white_ice
    ) * rho_ice * PSU_TO_KG_PER_KG

    # Recover bulk ice salinity per category.
    V_safe = jnp.where(V_ice_new > 1e-12, V_ice_new, 1.0)
    S_ice_new = jnp.where(
        V_ice_new > 1e-12,
        salt_new / (V_safe * rho_ice * PSU_TO_KG_PER_KG),
        0.0,
    )
    S_ice_new = jnp.clip(S_ice_new, S_ice_min, S_ice_max)

    # Salt flux to ocean (per-cat per-area): positive = INTO ocean.
    salt_flux = (salt_old - salt_new) / dt
    return SaltBudgetResult(S_ice_new=S_ice_new, salt_flux_to_ocean=salt_flux)


def aggregate_salt_flux(
    salt_flux_per_cat: jnp.ndarray,
    *,
    axis: int = -1,
) -> jnp.ndarray:
    """Sum per-category salt flux to the cell-mean salt flux.

    Parameters
    ----------
    salt_flux_per_cat : array (..., n_cat) or (..., )
    axis : int
        Category axis to reduce over.  Default last.

    Returns
    -------
    salt_flux_cell : array (...)
        Total salt mass flux per grid-cell area [kg/m²/s].
    """
    if salt_flux_per_cat.ndim == 0 or salt_flux_per_cat.shape[axis] == 1:
        return jnp.squeeze(salt_flux_per_cat, axis=axis) if salt_flux_per_cat.ndim > 0 else salt_flux_per_cat
    return jnp.sum(salt_flux_per_cat, axis=axis)
