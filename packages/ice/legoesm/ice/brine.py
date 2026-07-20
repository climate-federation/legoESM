"""Sea-ice bulk salinity + brine rejection.

Tracks a single bulk-mean salinity ``S_ice`` per ice category and
computes the salt mass flux to the ocean as the drop in the ice column's
stored salt over a thermodynamic step:

    salt_flux = (salt_old - salt_stored) / dt   [kg(salt)/m²/s, +ve = INTO ocean]

so salt is conserved BY CONSTRUCTION (``salt_old = salt_stored + salt_flux*dt``).
Read each process through this single definition:

- **Lead freezing**: new ice forms at the low bulk salinity ``S_lead_ice``
  (default 4 PSU), so the column's stored salt INCREASES and this term makes
  ``salt_flux`` NEGATIVE — i.e. the ocean loses the (small) salt that the new
  low-salinity ice took.  Brine REJECTION still occurs at the ocean: it is the
  NET of this small salt loss and the paired freshwater flux, which removes
  ``ρ_ice·ΔV`` of (nearly fresh) water from the mixed layer.  Removing
  low-salinity ice from the ocean leaves it saltier, the physically correct
  brine-rejection outcome — but the *salt-channel* sign of the freezing term
  itself is NEGATIVE, not a positive "brine release".
- **Basal / surface melt and sublimation**: ice salt is returned to the ocean
  at the ice's current ``S_ice`` → stored salt DROPS → POSITIVE ``salt_flux``.
  (Sublimation retains its salt in the surviving ice until the column fully
  sublimates or exceeds ``S_ice_max``, then the residual is rejected.)
- **Snow-ice flooding**: white ice forms with ``S_white = pore_frac · S_ocean``;
  the seawater that filled the snow pores carried this salt out of the ocean →
  stored salt INCREASES → NEGATIVE ``salt_flux`` (salt uptake into ice).

with ``S_ice`` in PSU (≈ g/kg → ×1e-3 to get kg of salt per kg of ice).

When ``BrineConfig.enabled = False`` the helpers degenerate to
zero salinity / zero salt flux, preserving the legacy
freshwater-only convention.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.ice.config import BrineConfig

PSU_TO_KG_PER_KG = 1.0e-3  # 1 PSU ≈ 1 g/kg = 1e-3 kg of salt per kg of seawater
# Canonical brine defaults (single source of truth callers pass in).
_BRINE_DEFAULTS = BrineConfig()


class SaltBudgetResult(NamedTuple):
    """Output of :func:`update_salinity_and_salt_flux`.

    ``S_ice_new`` is the post-step bulk ice salinity per category.
    ``salt_flux_to_ocean`` follows ``+ve = salt INTO the ocean``
    (== the DROP in the ice column's stored salt, ``(salt_old -
    salt_stored)/dt``).  Sign by process (see module docstring): MELT /
    sublimation-rejection give POSITIVE flux (ice releases its salt);
    LEAD FREEZING and snow-ice flooding give NEGATIVE flux (the new
    low-salinity ice takes salt from the ocean).  Net brine rejection at
    the ocean during freezing is the combination of this small negative
    salt flux and the paired freshwater removal — it is NOT a positive
    salt-channel flux here.
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
    S_lead_ice: float | jnp.ndarray,
    S_white_ice: float | jnp.ndarray,
    delta_V_basal_freeze: jnp.ndarray | float = 0.0,
    S_basal_ice: float | jnp.ndarray | None = None,
    delta_V_sublim: jnp.ndarray | float = 0.0,
    delta_V_fresh_refreeze: jnp.ndarray | float = 0.0,
    S_fresh_ice: float = 0.0,
    S_ice_min: float = 0.0,
    S_ice_max: float = _BRINE_DEFAULTS.S_ice_max,
    S_drain_target: float | None = None,
    tau_drain_s: float | None = None,
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
    S_lead_ice : float or array
        Salinity of newly frozen lead ice [PSU].  Scalar 4 is the legacy
        CICE-style bulk constant; the NEMO SI3 ``rn_sinew`` entrapment mode
        passes the per-cell array ``f_entrap · SSS_local`` (broadcast to the
        ``V`` shape by the caller) so new shelf ice captures most of the local
        salt and the brine is released GRADUALLY by drainage, not
        front-loaded at formation.
    S_white_ice : float or array
        Salinity of white ice from flooding [PSU] (typically
        ``0.5 · S_ocean_ref ≈ 17``).
    delta_V_basal_freeze : array or float
        Volume of new ice formed by basal congelation this step
        [m, ≥ 0, per grid-cell area].  Treated as a salty-ice source
        symmetric with lead freezing so basal growth removes salt
        from the ocean consistently with the freshwater it extracts.
        Default ``0.0`` preserves the legacy budget.
    S_basal_ice : float or None
        Salinity of basal congelation ice [PSU].  ``None`` (default)
        uses ``S_lead_ice``.
    delta_V_sublim : array or float
        Volume of ice removed by sublimation this step [m, ≥ 0, per
        grid-cell area].  Sublimation is a vapour-phase loss to the
        ATMOSPHERE, not the ocean: its salt stays behind, concentrating
        the remaining ice.  Excluded from the ocean salt-release
        residual so a dry sublimating column does not emit spurious
        positive salt flux with no accompanying water exchange.  Default
        ``0.0`` preserves the legacy budget.
    delta_V_fresh_refreeze : array or float
        Volume of NEW ice formed this step by refreezing of fresh
        meltwater (e.g. refrozen melt-pond water) [m, ≥ 0, per grid-cell
        area].  This crystallises at ``S_fresh_ice`` (≈ 0), NOT from
        surviving old ice, so it is counted as a freeze gain in the melt
        residual and added to the stored salt at its own (fresh) salinity
        — never absorbing the old ice's salt.  Without this, refrozen pond
        ice would be mistaken for surviving old ice and bury old salt
        (under-reporting the ocean salt release).  Default ``0.0``.
    S_fresh_ice : float
        Salinity of the fresh refrozen-meltwater ice [PSU], default 0.
    S_ice_min, S_ice_max : float
        Numerical clamp [PSU].
    S_drain_target, tau_drain_s : float or None
        Gravity-drainage relaxation (NEMO SI3 ``nn_icesal=2`` style): after
        the freeze/melt salt accounting, the bulk salinity relaxes toward
        ``S_drain_target`` (the mature multi-year value, typically the legacy
        4 PSU) with e-folding time ``tau_drain_s``:

            S -> target + (S - target) · exp(-dt/tau)

        applied ONLY where it would LOWER the salinity (drainage removes
        brine; it never salts the ice back up toward a higher target).  The
        drained salt reaches the ocean automatically through the
        ``(salt_old - salt_stored)/dt`` residual — no separate flux term, so
        conservation stays exact BY CONSTRUCTION.  ``None`` (default) disables
        drainage — bit-identical to the legacy budget.  Pair with the
        entrapment ``S_lead_ice`` array: entrapment holds the salt at
        formation, drainage releases it over ``tau`` instead of the legacy
        instant (S_ocean − 4) front-load that erodes the halocline in
        month 1.

    Returns
    -------
    result : :class:`SaltBudgetResult`
    """
    # Pre-step salt mass per area in the ice column [kg salt / m²].
    salt_old = S_ice_old * V_ice_old * rho_ice * PSU_TO_KG_PER_KG

    # Basal congelation growth (seawater freezing onto the ice base)
    # is a salty-ice source just like lead freezing.  When the caller
    # does not supply a distinct basal-ice salinity, use the lead-ice
    # salinity (first-year congelation and frazil ice have similar low
    # bulk salinities).  Without this term, basal growth would dilute
    # ``S_ice`` toward zero and emit zero salt flux, so the ocean would
    # lose freshwater (via the ice-mass budget) but keep all its salt —
    # breaking joint freshwater/salt closure for congelation growth.
    S_basal = S_lead_ice if S_basal_ice is None else S_basal_ice

    # Implied OCEAN melt-release volume: ice that disappeared between
    # ``V_old`` and ``V_new`` after accounting for ALL freeze gains
    # (lead, white-ice flooding, basal congelation) AND for the
    # sublimation loss, which leaves to the atmosphere rather than the
    # ocean.  ``delta_V_sublim`` is therefore subtracted from the
    # residual so it is NOT counted as ocean salt release.
    delta_V_sublim_pos = jnp.maximum(delta_V_sublim, 0.0)
    delta_V_fresh_pos = jnp.maximum(delta_V_fresh_refreeze, 0.0)
    delta_V_melt = jnp.maximum(
        V_ice_old + delta_V_lead_freeze + delta_V_white_ice
        + delta_V_basal_freeze + delta_V_fresh_pos
        - delta_V_sublim_pos - V_ice_new,
        0.0,
    )
    # Ice that survived (kept its old bulk salinity).  Both ocean melt
    # and sublimation remove ice from the old column; the difference is
    # only WHERE the salt goes (ocean vs retained in ice).
    V_remain = jnp.maximum(
        V_ice_old - delta_V_melt - delta_V_sublim_pos, 0.0
    )

    # Salt retained by sublimation: the vapour carries no salt, so the salt of
    # the sublimated ice stays behind and concentrates the SURVIVING OLD ice
    # (``V_remain``) — it must NOT ride on ice that newly formed this step
    # (lead/white/basal), which crystallises from ocean water at its own low
    # salinity.  The surviving old ice can hold at most ``S_ice_max`` PSU, so
    # the retained sublimation salt is capped at the old ice's remaining
    # headroom ``(S_ice_max - S_ice_old) * V_remain``.  Any excess — including
    # the ENTIRE residual when the old ice fully sublimates (``V_remain -> 0``,
    # no surviving carrier) — is rejected to the ocean via the salt-flux
    # residual below (brine drainage; the water already left to the
    # atmosphere).  Without this cap, a step that fully sublimates old ice
    # while freezing new lead ice would bury the old salt in the new ice
    # (spuriously raising its salinity) and under-report the ocean salt flux.
    # Operator-split note (drainage interplay): this headroom is evaluated
    # against the PRE-drain ``S_ice_old``, while the gravity-drainage relax
    # below can lower the final salinity and thus leave more end-of-step
    # capacity than assumed here.  In the corner case (salty sublimating
    # column + drainage active) some sublimation salt therefore reaches the
    # ocean THIS step that a drain-first ordering would have retained one more
    # step.  This is a deliberate first-order split: the ledger stays exact
    # (post-drain/post-clamp storage + residual == salt_old), only the release
    # timing shifts by <= one step in that corner (codex MED, documented).
    old_salt_headroom = jnp.maximum(
        (S_ice_max - S_ice_old) * V_remain, 0.0
    )
    salt_retained_from_sublim = jnp.minimum(
        S_ice_old * delta_V_sublim_pos, old_salt_headroom,
    ) * rho_ice * PSU_TO_KG_PER_KG

    # Post-step salt mass per area: melt releases ice that had the
    # *old* bulk salinity TO THE OCEAN; freezing adds ice at the (fixed)
    # low ``S_lead_ice``; flooding adds white ice with elevated
    # ``S_white_ice``; basal congelation adds ice at ``S_basal``;
    # sublimation retains its salt in the column.
    salt_new = (
        S_ice_old * V_remain
        + S_lead_ice * delta_V_lead_freeze
        + S_white_ice * delta_V_white_ice
        + S_basal * delta_V_basal_freeze
        + S_fresh_ice * delta_V_fresh_pos
    ) * rho_ice * PSU_TO_KG_PER_KG + salt_retained_from_sublim

    # Recover bulk ice salinity per category, then clamp to physical
    # bounds.  The salt flux MUST be computed from the salt mass the ice
    # state actually stores AFTER clamping — otherwise a saturating cap
    # (e.g. 17-PSU white ice vs a 12-PSU S_ice_max) would remove salt
    # from the ocean that the ice never retains, breaking conservation.
    V_safe = jnp.where(V_ice_new > 1e-12, V_ice_new, 1.0)
    S_ice_unclamped = jnp.where(
        V_ice_new > 1e-12,
        salt_new / (V_safe * rho_ice * PSU_TO_KG_PER_KG),
        0.0,
    )
    # --- Gravity drainage (NEMO SI3 nn_icesal=2 style; opt-in) ---
    # Relax the bulk salinity toward the mature target, DOWNWARD ONLY
    # (drainage removes brine; it must never re-salt fresher ice up toward
    # the target).  Applied BEFORE the clamp so the drained state is what the
    # ice stores; the drained salt then reaches the ocean through the
    # (salt_old - salt_stored)/dt residual below — exact conservation with no
    # separate flux term.  Static Python gate: None -> bit-identical legacy.
    if S_drain_target is not None and tau_drain_s is not None:
        _decay = jnp.exp(-dt / tau_drain_s)
        S_drained = S_drain_target + (S_ice_unclamped - S_drain_target) * _decay
        S_ice_unclamped = jnp.where(
            S_ice_unclamped > S_drain_target, S_drained, S_ice_unclamped
        )
    S_ice_new = jnp.clip(S_ice_unclamped, S_ice_min, S_ice_max)

    # Actual stored salt mass in the post-step ice (consistent with the
    # clamped salinity and the new volume).
    salt_stored = S_ice_new * V_ice_new * rho_ice * PSU_TO_KG_PER_KG

    # Salt flux to ocean (per-cat per-area): positive = INTO ocean.  Defined
    # as the drop in the ice column's stored salt, so salt is conserved BY
    # CONSTRUCTION: ``salt_old = salt_stored + salt_flux*dt`` always.
    #
    # Sublimation interaction: the sublimated ice's salt is folded into
    # ``salt_new`` (``salt_retained_from_sublim``) so it CONCENTRATES the
    # remaining ice and emits NO ocean flux WHILE ice remains and the
    # concentrated salinity stays below ``S_ice_max``.  But the vapour carried
    # the WATER to the atmosphere, not the salt, so when the column fully
    # sublimates (``V_ice_new -> 0`` forces ``S_ice_new = 0``, ``salt_stored =
    # 0``) — or when concentration would exceed ``S_ice_max`` — the residual
    # salt has no ice to occupy and is rejected to the OCEAN (brine drainage):
    # ``salt_flux = salt_old/dt`` at full sublimation.  This is salt-conserving
    # (the salt cannot follow the water into the vapour phase) and pairs with
    # the freshwater budget, which correctly sends the sublimated WATER to the
    # atmosphere (excluded from the ocean freshwater flux), so the ocean gains
    # salt without water (salinity up) — the physically correct outcome.
    salt_flux = (salt_old - salt_stored) / dt
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
