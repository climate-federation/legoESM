"""Production-grade surface salinity restoring for OMIP/CMIP runs.

OMIP-2 protocol (Tsujino et al. 2020, Griffies et al. 2016):
relax model SSS toward an observational climatology over a
piston-velocity timescale ``v_piston = dz_1 / tau_restore``.  The
restoring flux is expressed as a freshwater flux into the ocean —
salt is conserved, only water mass is exchanged with a virtual
atmospheric reservoir:

    F_FW = - (rho_0 · dz_1 / tau_restore) · (S_model - S_target) / S_target
           [kg/m²/s, positive INTO ocean]

The salt-mass-conserving convention is equivalent to applying

    dS/dt|_restore = - (S_model - S_target) / tau_restore

in the surface layer (z_1 thick), tagging the FW reservoir with
the model's instantaneous salinity.  Region-dependent
``tau_restore`` lets marginal-sea / Arctic restoring be strong
(few days) while interior basins stay weakly constrained
(decadal) — the standard OMIP-2 choice that keeps AMOC stable
without nudging the interior to climatology.

Sea-ice gating
--------------
Where sea ice covers a cell at concentration ``conc``, the
brine/melt salt flux from the ice tile (added in the sea-ice
extensions) already drives the local salt budget.  This module
linearly down-weights the SSS restoring by ``(1 − conc)`` so the
two pathways do not double-count.  ``ice_gate_threshold`` lets the
user disable restoring entirely above some ice fraction (default
0.9 — fully ice-covered cells skip restoring).

Differentiable
--------------
The module is pure JAX so ``jax.grad`` flows through tau and the
region masks for parameter tuning.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


class RegionMaskSpec(NamedTuple):
    """Region geometric specifier used to build a smooth lat/lon mask.

    Each region is the intersection of a latitude band
    ``[lat_min, lat_max]`` and an optional longitude band
    ``[lon_min, lon_max]`` (with wrap-around in [0, 360)).  Masks
    are smooth (tanh-blended over ``edge_width_deg``) so that the
    restoring strength is differentiable and there are no
    grid-imprint discontinuities.
    """
    name: str
    lat_min: float
    lat_max: float
    lon_min: float = 0.0
    lon_max: float = 360.0
    edge_width_deg: float = 2.0
    # ``tau_restore_days`` overrides the global default inside this
    # region.  Set ``None`` to inherit the global value.
    tau_restore_days: float | None = None


# Canonical OMIP-2 region list with literature-based timescales.
# Marginal seas use few-day restoring (Tsujino 2020 default);
# interior basins use multi-year piston velocities.
DEFAULT_OMIP2_REGIONS: tuple[RegionMaskSpec, ...] = (
    # Arctic (north of 70°N) — strong restoring (1 month)
    RegionMaskSpec("Arctic", lat_min=70.0, lat_max=90.0,
                   tau_restore_days=30.0),
    # Nordic Seas (60-80°N, -45-30°E approx) — strong restoring
    RegionMaskSpec("Nordic", lat_min=60.0, lat_max=80.0,
                   lon_min=315.0, lon_max=360.0,
                   tau_restore_days=30.0),
    RegionMaskSpec("Nordic_E", lat_min=60.0, lat_max=80.0,
                   lon_min=0.0, lon_max=30.0,
                   tau_restore_days=30.0),
    # Labrador / North Atlantic deep-water region — moderate
    RegionMaskSpec("Labrador", lat_min=50.0, lat_max=65.0,
                   lon_min=290.0, lon_max=320.0,
                   tau_restore_days=90.0),
    # Mediterranean (rough mask) — very strong (closed basin)
    RegionMaskSpec("Mediterranean", lat_min=30.0, lat_max=46.0,
                   lon_min=355.0, lon_max=360.0,
                   tau_restore_days=10.0),
    RegionMaskSpec("Mediterranean_E", lat_min=30.0, lat_max=46.0,
                   lon_min=0.0, lon_max=37.0,
                   tau_restore_days=10.0),
    # Southern Ocean marginal sea ice zone (60-80°S) — moderate
    RegionMaskSpec("SO_marginal", lat_min=-80.0, lat_max=-60.0,
                   tau_restore_days=180.0),
    # Enclosed / semi-enclosed marginal seas that are UNRESOLVED at ~1° (narrow
    # straits) and accumulate river runoff -> a strong fresh bias vs NEMO (the
    # SSS-map Δ extremes).  Few-day restoring pins them to WOA, the standard
    # OMIP-2 marginal-sea treatment (cf. the Mediterranean above).
    RegionMaskSpec("Baltic", lat_min=53.0, lat_max=66.0,
                   lon_min=9.0, lon_max=31.0, tau_restore_days=15.0),
    RegionMaskSpec("Black_Sea", lat_min=40.0, lat_max=48.0,
                   lon_min=27.0, lon_max=42.0, tau_restore_days=15.0),
    RegionMaskSpec("Hudson_Bay", lat_min=50.0, lat_max=66.0,
                   lon_min=264.0, lon_max=295.0, tau_restore_days=20.0),
    RegionMaskSpec("Okhotsk_NWPac", lat_min=43.0, lat_max=62.0,
                   lon_min=133.0, lon_max=163.0, tau_restore_days=30.0),
)


class SSSRestoringConfig(NamedTuple):
    """Surface salinity restoring configuration.

    Standard OMIP-2 settings:
        * ``tau_restore_days_default = 365`` (1 year interior)
        * Surface piston layer ``z1_m = 10`` (top model layer)
        * Ice gating on, threshold 0.9 ice fraction
        * Default region list = ``DEFAULT_OMIP2_REGIONS``

    To use a global uniform tau (no region masks), pass
    ``regions=()`` and set ``tau_restore_days_default``.
    """
    enabled: bool = False
    tau_restore_days_default: float = 365.0
    z1_m: float = 10.0
    ice_gate: bool = True
    ice_gate_mode: str = "tanh"      # "tanh" | "nemo_linear"
    ice_gate_threshold: float = 0.9
    ice_gate_softness: float = 0.1   # tanh width around threshold
    regions: tuple[RegionMaskSpec, ...] = DEFAULT_OMIP2_REGIONS
    S_floor: float = 1.0             # avoid division by zero in PSU
    # Denominator of the salinity->freshwater conversion.
    #   "s_target" (default, bit-identical to every run before 2026-08-12):
    #       F_FW = -rho_0*z1*dS/dt / max(S_target, S_floor).
    #   "live_s" — NEMO `sbcssr` nn_sssr=2 (sbcssr.F90:132-134), which ORCA1
    #       runs: zerp = zsrp*coefice*(sss_m - sss_target)/MAX(sss_m, 1e-20),
    #       i.e. divided by the LIVE surface salinity, not the target.  The
    #       two differ wherever the model is far from the climatology --
    #       exactly the Arctic cells that carry the +1.4 psu common-mode bias.
    # NOTE the magnitudes already agree: z1_m/tau = 10 m / 45.5 d = 219.8
    # mm/day against NEMO's rn_deds = -220 mm/day, and the OMIP deck's 4
    # mm/day bound matches rn_sssr_bnd = 4.
    normalization: str = "s_target"  # "s_target" | "live_s"
    # River-mouth gate (NEMO sbcssr: restoring damped by (1-2*rnfmsk) -> ZERO
    # at river mouths): when a per-cell ``river_runoff`` field is passed to
    # ``compute_sss_restoring_flux``, cells whose runoff exceeds this
    # threshold get NO restoring -- otherwise the restoring fights the river
    # plume toward the coarse WOA climatology (the Amazon SSS artifact).
    # ~1e-6 kg/m2/s ~ 0.086 mm/day, far below any river-mouth cell.
    river_gate_threshold_kg_m2_s: float = 1.0e-6
    # Cap the maximum FW flux to ±200 mm/day-equivalent so a
    # pathological S_target − S_model jump does not blow up the
    # ocean surface budget.
    max_flux_kg_m2_s: float = 200e-3 / SECONDS_PER_DAY * constants.rho_water


def _smooth_box_mask(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    spec: RegionMaskSpec,
) -> jnp.ndarray:
    """Smooth tanh-blended lat/lon box indicator in [0, 1].

    Returns a JAX array of the same shape as ``lat``/``lon`` that
    transitions from 0 outside the box to 1 inside with width
    ``edge_width_deg`` at the boundaries.  Longitude wrap-around
    in [0, 360) is handled by allowing ``lon_min > lon_max`` to
    indicate the region spans the prime meridian.
    """
    w = jnp.maximum(spec.edge_width_deg, 1e-6)

    lat_inside = (
        0.5 * (jnp.tanh((lat - spec.lat_min) / w) + 1.0)
        * 0.5 * (jnp.tanh((spec.lat_max - lat) / w) + 1.0)
    )

    if spec.lon_min <= spec.lon_max:
        lon_inside = (
            0.5 * (jnp.tanh((lon - spec.lon_min) / w) + 1.0)
            * 0.5 * (jnp.tanh((spec.lon_max - lon) / w) + 1.0)
        )
    else:
        # Wrap-around band: lon < lon_max OR lon > lon_min.
        lon_inside = jnp.clip(
            0.5 * (jnp.tanh((lon - spec.lon_min) / w) + 1.0)
            + 0.5 * (jnp.tanh((spec.lon_max - lon) / w) + 1.0),
            0.0, 1.0,
        )

    return lat_inside * lon_inside


def build_region_masks(
    lat_deg: jnp.ndarray,
    lon_deg: jnp.ndarray,
    config: SSSRestoringConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-cell restoring inverse-timescale field.

    Each region contributes ``mask_r · (1 / tau_r − 1 / tau_default)``;
    summed and added to ``1 / tau_default`` to give an effective
    ``1 / tau_eff`` field that smoothly switches between interior
    and marginal-sea timescales.  Where regions overlap the maximum
    inverse-tau (= shortest tau) wins.

    Returns
    -------
    inv_tau_eff : array (same shape as ``lat_deg``)
        Effective 1 / tau in s⁻¹ at each grid cell.
    region_active : array (same shape)
        Total summed region indicator ∈ [0, 1], useful for
        diagnostic plots.
    """
    tau_default_s = config.tau_restore_days_default * SECONDS_PER_DAY
    inv_tau_default = 1.0 / tau_default_s

    inv_tau_eff = jnp.full_like(lat_deg, inv_tau_default)
    region_active = jnp.zeros_like(lat_deg)

    for spec in config.regions:
        mask = _smooth_box_mask(lat_deg, lon_deg, spec)
        tau_r = (
            spec.tau_restore_days
            if spec.tau_restore_days is not None
            else config.tau_restore_days_default
        ) * SECONDS_PER_DAY
        inv_tau_r = 1.0 / tau_r
        # Take max(inv_tau_eff, mask * inv_tau_r + (1-mask) * inv_tau_eff).
        # Equivalent to a max blend at the cell.
        candidate = mask * inv_tau_r + (1.0 - mask) * inv_tau_eff
        inv_tau_eff = jnp.maximum(inv_tau_eff, candidate)
        region_active = jnp.maximum(region_active, mask)

    return inv_tau_eff, region_active


def compute_sss_restoring_flux(
    S_model_top: jnp.ndarray,
    S_target: jnp.ndarray,
    lat_deg: jnp.ndarray,
    lon_deg: jnp.ndarray,
    ice_concentration: jnp.ndarray,
    config: SSSRestoringConfig,
    river_runoff: jnp.ndarray | None = None,
) -> dict:
    """Compute the OMIP-2 SSS restoring fluxes.

    The relaxation toward ``S_target`` is converted to a virtual
    freshwater flux:

        dz1 · dS/dt|_restore = − (z1 / tau) · (S_model − S_target)

        F_FW = −rho_0 · (z1 / tau) · (S_model − S_target) / S_target
               [kg/m²/s, positive INTO ocean]

    Equivalently, the salt-mass change rate in the surface layer
    is

        dM_salt/dt = − rho_0 · z1 · (S_model − S_target) / tau · 1e-3
                     [kg salt / m² / s, positive INTO ocean]

    Both diagnostics are returned.

    Parameters
    ----------
    S_model_top : array
        Surface-layer salinity from the ocean state [PSU].
    S_target : array
        Climatological SSS target [PSU] interpolated to the model grid.
    lat_deg, lon_deg : array
        Cell-centred lat / lon in degrees.  Must broadcast against
        ``S_model_top``.
    ice_concentration : array
        Cell ice fraction in [0, 1].  Used to down-weight the
        restoring under sea ice (which has its own salt-flux
        channel via brine).
    config : :class:`SSSRestoringConfig`

    Returns
    -------
    result : dict
        ``"freshwater_flux"`` — virtual FW flux into ocean
        [kg/m²/s, positive INTO ocean].
        ``"salt_flux"`` — equivalent salt-mass exchange
        [kg(salt)/m²/s, NEGATIVE during fresh-bias restoring
        because the FW dilution removes ocean salt indirectly via
        the surface layer's volume change].
        ``"dS_dt_top"`` — direct salinity tendency in the surface
        layer [PSU/s] for callers that want to apply restoring on
        the tracer side rather than via FW.
        ``"inv_tau_eff"`` — effective 1/τ field [1/s].
        ``"region_active"`` — diagnostic region indicator [0, 1].
    """
    inv_tau_eff, region_active = build_region_masks(lat_deg, lon_deg, config)

    # Ice gating: down-weight restoring under sea ice.
    #   "tanh"        — legoESM default: ``(1 - fr_i)`` linear weight times an
    #                   extra soft tanh cutoff above ``ice_gate_threshold``.
    #   "nemo_linear" — exact NEMO ``sbcssr`` ``nn_sssr_ice = 0``: the under-ice
    #                   relaxation coefficient is ``coefice = 1 - fr_i`` (zero
    #                   under full ice), with NO tanh cutoff.  ORCA1 RUN_REF
    #                   uses nn_sssr_ice = 0, so this is the faithful setting.
    if config.ice_gate:
        if config.ice_gate_mode == "nemo_linear":
            ice_factor = jnp.clip(1.0 - ice_concentration, 0.0, 1.0)
        elif config.ice_gate_mode == "tanh":
            soft = jnp.maximum(config.ice_gate_softness, 1e-6)
            ice_factor = 0.5 * (
                1.0 - jnp.tanh((ice_concentration - config.ice_gate_threshold) / soft)
            )
            ice_factor = ice_factor * (1.0 - ice_concentration)
        else:
            raise ValueError(
                f"unknown ice_gate_mode {config.ice_gate_mode!r} "
                "(expected 'tanh' or 'nemo_linear')"
            )
    else:
        ice_factor = jnp.ones_like(ice_concentration)

    inv_tau_eff = inv_tau_eff * ice_factor

    # River-mouth gate: NO restoring at river mouths, so the relaxation does
    # not fight the river plume toward the coarse WOA climatology.  Hard gate
    # at the threshold — river-mouth cells carry runoff orders of magnitude
    # above it.
    #
    # ⚠ THIS IS A legoESM DEVIATION, NOT NEMO PARITY (corrected 2026-08-07).
    # An earlier version of this comment cited "NEMO sbcssr (1-2*rnfmsk)".
    # That term exists (sbcssr.F90:119) but is INACTIVE in the ORCA1 deck we
    # match: sbcrnf.F90 populates rnfmsk from sn_cnf ('socoefr') only inside
    # `IF( ln_rnf_mouth )`, and its ELSE branch sets rnfmsk = 0 everywhere.
    # RUN_GATEWAY never sets ln_rnf_mouth, so the namelist_ref default .false.
    # applies, rnfmsk == 0, and (1-2*rnfmsk) == 1 -- i.e. NEMO applies FULL SSS
    # restoring at river mouths there, while this gate zeroes it.
    # Measured consequence on the Barents/Kara shelf (job 9335176): vertical S
    # range 0.188 vs NEMO 1.266, column-mean S 33.408 vs 34.508 (-1.10 psu),
    # with the T range nearly correct -- a salinity-only signature.  Compounds
    # with legoESM's horizontal runoff spread passes (NEMO has none), which
    # push more cells above this gate's threshold.
    if river_runoff is not None:
        river_factor = jnp.where(
            jnp.asarray(river_runoff) > config.river_gate_threshold_kg_m2_s,
            0.0, 1.0)
        inv_tau_eff = inv_tau_eff * river_factor

    S_diff = S_model_top - S_target
    # Salinity tendency in the surface layer [PSU/s] (pre-cap).
    dS_dt_top = -S_diff * inv_tau_eff

    # Equivalent FW flux.  With normalization="s_target" this is the
    # virtual-salt convention:
    #     F_FW · S_target / (rho_0 · z1) = − dS/dt  →
    #     F_FW = − rho_0 · z1 · dS/dt / S_target
    # With normalization="live_s" the denominator is the live surface
    # salinity instead, reproducing NEMO sbcssr nn_sssr=2.
    #
    # ⚠ SCOPE (codex 9383572 RED, 2026-08-12): in THIS code path the flux is a
    # DIAGNOSTIC -- `apply_sss_restoring_step*` consume `dS_dt_top` and edit
    # the tracer directly.  Because `dS_dt_top` is re-derived below as
    # `-freshwater_flux * S_safe / (rho_0*z1)`, `S_safe` CANCELS the division
    # that produced the flux, so the choice of denominator changes what the
    # ocean actually sees ONLY where the flux cap binds (there the applied
    # tendency scales with S_safe, i.e. live_s restores MORE strongly in a
    # too-salty cell).  It does NOT turn restoring into a water flux, and does
    # NOT make it compatible with the real_freshwater closure.
    # Dispatch hardening: a typo must not silently run the legoESM form.
    if config.normalization == "s_target":
        S_norm = S_target
    elif config.normalization == "live_s":
        # NEMO sbcssr nn_sssr=2: divide by the LIVE surface salinity.
        S_norm = S_model_top
    else:
        raise ValueError(
            f"unknown SSSRestoringConfig.normalization "
            f"{config.normalization!r}; expected 's_target' or 'live_s'.")
    S_safe = jnp.maximum(S_norm, config.S_floor)
    rho_0 = constants.rho_ocean
    freshwater_flux = -rho_0 * config.z1_m * dS_dt_top / S_safe
    freshwater_flux = jnp.clip(
        freshwater_flux, -config.max_flux_kg_m2_s, config.max_flux_kg_m2_s,
    )

    # The applied tendency is re-derived from the CAPPED flux so the flux
    # bound (NEMO ``ln_sssr_bnd``/``rn_sssr_bnd`` semantics when configured
    # to ±4 mm/day-equivalent) actually limits what reaches the ocean.
    # Previously ``dS_dt_top`` bypassed the clip, so the τ-restoring
    # appliers (``apply_sss_restoring_step*``, which consume ``dS_dt_top``)
    # saw an UNBOUNDED restoring while only the unused flux diagnostics
    # were capped.
    dS_dt_top = -freshwater_flux * S_safe / (rho_0 * config.z1_m)

    # Salt-mass flux: dM_salt/dt = rho_0 · z1 · dS/dt · 1e-3
    # (PSU·kg/m³·m/s · g/kg / 1000 = kg(salt)/m²/s), from the capped
    # tendency so all outputs stay mutually consistent.
    salt_flux = rho_0 * config.z1_m * dS_dt_top * 1.0e-3

    return {
        "freshwater_flux": freshwater_flux,
        "salt_flux": salt_flux,
        "dS_dt_top": dS_dt_top,
        "inv_tau_eff": inv_tau_eff,
        "region_active": region_active,
    }


def interp_woa_sss_to_grid(
    sss_woa: jnp.ndarray,
    lat_woa: jnp.ndarray,
    lon_woa: jnp.ndarray,
    lat_target: jnp.ndarray,
    lon_target: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear-interp the 1° WOA SSS onto a target lat/lon grid.

    Both grids are assumed to span the full globe.  Longitude is
    handled in [0, 360); latitude in ascending order.  Out-of-range
    cells (e.g. near pole singularities on grids that exceed the
    WOA latitude bounds) get the nearest-edge value.

    Pure JAX so the result is differentiable + JIT-compatible.

    Parameters
    ----------
    sss_woa : array ``(n_lat_woa, n_lon_woa)`` [PSU]
    lat_woa : array ``(n_lat_woa,)`` degrees, ascending
    lon_woa : array ``(n_lon_woa,)`` degrees in [0, 360)
    lat_target : array of shape ``S`` — degrees of the model grid
    lon_target : array of shape ``S`` — degrees of the model grid
        Both must share the same shape ``S``.

    Returns
    -------
    sss_on_grid : array of shape ``S`` [PSU]
    """
    # Indices via linear search (small WOA grid; vectorised).
    nlat_w = lat_woa.shape[0]
    nlon_w = lon_woa.shape[0]
    dlat = (lat_woa[-1] - lat_woa[0]) / (nlat_w - 1)
    dlon = 360.0 / nlon_w

    fi_lat = jnp.clip((lat_target - lat_woa[0]) / dlat, 0.0, nlat_w - 1.001)
    fi_lon = jnp.mod(lon_target, 360.0) / dlon
    i0 = jnp.floor(fi_lat).astype(jnp.int32)
    j0 = jnp.floor(fi_lon).astype(jnp.int32)
    di = fi_lat - i0
    dj = fi_lon - j0
    i1 = jnp.minimum(i0 + 1, nlat_w - 1)
    j1 = jnp.mod(j0 + 1, nlon_w)

    s00 = sss_woa[i0, j0]
    s01 = sss_woa[i0, j1]
    s10 = sss_woa[i1, j0]
    s11 = sss_woa[i1, j1]

    return (
        (1 - di) * (1 - dj) * s00
        + (1 - di) * dj * s01
        + di * (1 - dj) * s10
        + di * dj * s11
    )
