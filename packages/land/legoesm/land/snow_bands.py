"""Sub-grid elevation-band snow (VIC snow bands / CESM glacier elevation classes).

A coarse land cell in mountainous terrain spans kilometres of sub-grid relief: the
cell-MEAN temperature can be above freezing while high sub-grid elevations stay below
it year-round.  A single cell-mean snowpack therefore (a) partitions mountain
precipitation as rain that should fall as snow on the high fractions, (b) melts snow
that would survive on ridges, and (c) misses permanent (perennial) snow/ice entirely —
a large winter/spring warm bias vs ERA5 over Tibet, the Rockies and the Andes.

This module implements the classic sub-grid ELEVATION-BAND snow scheme (Liang et al.
1994 / Nijssen et al. 2001, VIC "snow bands"; Lipscomb et al. 2013, CESM glacier
multiple elevation classes):

* the cell is partitioned into ``n_bands`` equal-area elevation bands whose mean
  elevation anomalies come from the sub-grid topography standard deviation (CLM
  surfdata ``STD_ELEV``) under a Gaussian sub-grid hypsometry,
* air/surface temperature is downscaled to each band with a fixed lapse rate,
* precipitation phase, snow accumulation and energy-limited melt are computed PER
  BAND (reusing :func:`legoesm.land.snow_budget.update_snow`),
* the aggregate snow cover fraction for the albedo blend is the area-weighted band
  cover — so a warm cell keeps bright, snow-covered high fractions,
* band SWE is capped (CLM5 snow capping) with the excess routed to runoff; a band
  pinned at the cap is PERMANENT snow/ice (a sub-grid glacier accumulation zone).

Simplification vs full VIC (disclosed): all bands share the single cell soil column
(one shared substrate temperature).  The snow mass budget, precipitation phase,
snow-cover/albedo aggregation AND the surface RADIATION balance are banded
(:func:`band_net_radiation`): each band carries its own albedo, skin temperature
(lapse-slaved) and elevation-lapsed SW/LW down, so the per-band net radiation drives
per-band melt and the area-weighted aggregate drives the shared soil column — this
captures the elevation x albedo covariance (bright high bands reflect the enhanced
high-altitude insolation) that a single cell-mean radiation balance averages away.
The band elevation anomalies are zero-mean (equal-area Gaussian quintiles), so the
linear SW/LW elevation lapse is a pure sub-grid REDISTRIBUTION that conserves the
cell-mean down-flux (no spurious energy source in a coupled run).  Still cell-mean:
the TURBULENT fluxes (SH/LH) — the lapse shifts a band's air AND skin temperature
together, so the air-skin difference the bulk flux sees is ~band-invariant; per-band
bulk exchange is the deferred upgrade (~n_bands x cost) if the residual warrants it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.snow_budget import update_snow


# --- Gaussian equal-area band anomalies (standard-normal quintile conditional means) ---
# E[Z | quintile k] for Z ~ N(0,1), k = 1..5: (phi(a) - phi(b)) / 0.2 over each
# quintile (a, b) with boundaries Phi^-1(0.2, 0.4, 0.6, 0.8).  Multiplying by the
# sub-grid elevation std gives equal-area band mean-elevation anomalies under a
# Gaussian sub-grid hypsometry.
_BAND_ANOM_K5 = (-1.39981, -0.53190, 0.0, 0.53190, 1.39981)

__param_spec__ = {
    "ElevationSnowBandConfig": {
        "scheme_key": "land.snow_bands",
        "excluded": {
            "swe_cap": "convention: CLM5 ice-reservoir ceiling (water-budget bound, "
                       "albedo is saturated far below it)",
            "swe_snow_cap": "convention: seasonal-snow firnification threshold "
                            "(mass-budget bound, not a radiative closure)",
            "tau_ice_discharge_s": "mass-timescale: multi-decadal glacier discharge, "
                                   "unconstrained by an offline surface-flux fit",
            "ice_expose_kg_m2": "numerics: ice-exposure threshold for the albedo "
                                "switch (regulariser, not a physical closure)",
        },
        "params": {
            "lapse_rate_K_m": {
                "units": "K/m", "bounds": (4.5e-3, 8.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Lipscomb et al. 2013 (CESM MEC, 6 K/km); "
                             "VIC snow bands (6.5 K/km)",
                "shape": None,
            },
            "sw_elev_grad_per_m": {
                "units": "1/m", "bounds": (0.0, 1.2e-4), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "clear-sky global horizontal irradiance elevation "
                             "gradient ~3-10 %/km (thinner atmosphere)",
                "shape": None,
            },
            "lw_elev_lapse_W_m2_per_m": {
                "units": "W/m2/m", "bounds": (0.0, 6.0e-2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Marty et al. 2002 clear-sky LW down elevation "
                             "lapse ~29 W/m2/km (colder, drier air aloft)",
                "shape": None,
            },
            "alpha_glacier_ice": {
                "units": "1", "bounds": (0.15, 0.45), "tunable_tier": 1,
                "transform": "sigmoid", "category": "albedo",
                "reference": "CLM5 exposed glacier-ice (ablation-zone) albedo ~0.3; "
                             "bare-ice observations 0.2-0.4",
                "shape": None,
            },
            "sky_view_min": {
                "units": "1", "bounds": (0.5, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "terrain sky-view factor of the deepest valley band "
                             "(gap 5); 1 = flat/open sky",
                "shape": None,
            },
            "blow_snow_wind_thresh_ms": {
                "units": "m/s", "bounds": (4.0, 12.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "blowing-snow mobilisation wind threshold "
                             "(Pomeroy / CROCUS, gap 5)",
                "shape": None,
            },
            "blow_snow_subl_rate": {
                "units": "kg/m2/s/(m/s)", "bounds": (0.0, 1.0e-5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "blowing-snow sublimation rate over threshold (gap 5); "
                             "0 = off",
                "shape": None,
            },
            "refreeze_frac": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "fraction of rain refreezing into a sub-freezing snow band "
                             "(gap 6 cold content)",
                "shape": None,
            },
        },
    },
}


class ElevationSnowBandConfig(NamedTuple):
    """Sub-grid elevation-band snow configuration.

    ``band_dz`` carries the per-column band mean-elevation anomalies (build with
    :func:`band_elevation_anomalies` from the CLM ``STD_ELEV`` map).  Bands are
    EQUAL-AREA by construction, so no fraction array is needed.
    """
    band_dz: jax.Array          # (ncol, n_bands) band elevation anomaly [m]
    # T-downscaling lapse rate [K/m]: CESM/CLM glacier elevation classes 6.0e-3
    # (Lipscomb et al. 2013); VIC snow bands use 6.5e-3.  Tier-2 tunable.
    lapse_rate_K_m: float = 6.0e-3
    # --- Radiative elevation lapse (band net-radiation downscaling) ---
    # SW down elevation gradient [1/m]: clear-sky surface insolation rises with
    # elevation (thinner atmosphere), sw_down_band = sw_down*(1 + grad*dz).  3 %/km.
    sw_elev_grad_per_m: float = 3.0e-5
    # LW down elevation lapse [W/m2/m]: down-welling longwave falls with elevation
    # (colder, drier air aloft), lw_down_band = lw_down - lapse*dz.  ~29 W/m2/km.
    lw_elev_lapse_W_m2_per_m: float = 2.9e-2
    # --- Firn / glacier-ice reservoir (gap 4: perennial-ice mass balance) ---
    # Seasonal-snow cap [kg/m2]: seasonal snow above this compacts into the firn/ice
    # reservoir (delayed glacier storage) rather than becoming same-step runoff.  A
    # few m w.e. of snow before it densifies to ice (Lipscomb et al. 2013).
    swe_snow_cap: float = 1.0e3
    # Glacier-ice discharge timescale [s]: the ice reservoir runs off slowly as
    # ice/tau (outflow / basal melt) — DELAYED discharge, not instantaneous.  ~30 yr.
    tau_ice_discharge_s: float = 1.0e9
    # Exposed ablation-zone glacier-ice albedo: darker than aged firn, seen where the
    # seasonal snow has melted off the ice (CLM ~0.3) — the bright-firn/dark-ice
    # contrast a single glacier albedo misses.
    alpha_glacier_ice: float = 0.30
    # Ice-present threshold [kg/m2] above which a band's snow-free surface shows the
    # glacier-ice albedo instead of the soil/vegetation base.
    ice_expose_kg_m2: float = 10.0
    # Ice-reservoir ceiling [kg/m2] = 10 m w.e. (CLM5 snow-capping bound, Oleson et
    # al. 2013): ice above this is shed to runoff, bounding perennial storage.
    swe_cap: float = 1.0e4
    # --- Terrain radiation (gap 5): sub-grid sky-view / shading ---
    # Minimum sky-view fraction of the most-shielded (lowest) band [0-1]: rough sub-grid
    # terrain blocks part of the sky, and the blocked fraction is filled by the
    # surrounding slopes radiating at ~the band's own skin T, so the band's NET longwave
    # loss scales by its per-band sky-view V_band (1 at the ridge, sky_view_min in the
    # valley) — valleys cool less at night (terrain shielding).  1.0 disables it.
    sky_view_min: float = 1.0
    # --- Blowing snow (gap 5): wind-driven sublimation loss ---
    # 10 m wind threshold [m/s] above which snow is mobilised (Pomeroy / CROCUS).
    blow_snow_wind_thresh_ms: float = 7.0
    # Blowing-snow sublimation rate [kg/m2/s per (m/s) over threshold]; 0 = off (opt-in,
    # needs site wind/fetch calibration).  Removes SWE from the exposed high bands.
    blow_snow_subl_rate: float = 0.0
    # --- Snowpack cold content (gap 6): rain-on-snow refreezing ---
    # Fraction of rain that refreezes into a sub-freezing snow band, releasing L_f into
    # the surface energy budget (delays runoff, warms the pack).  The frozen-SOIL latent
    # heat is handled by SoilThermalConfig.enable_freeze_thaw (zero-curtain).
    refreeze_frac: float = 1.0


def band_elevation_anomalies(std_elev: jnp.ndarray, n_bands: int = 5) -> jnp.ndarray:
    """Equal-area band mean-elevation anomalies from the sub-grid elevation std.

    Parameters
    ----------
    std_elev : jnp.ndarray
        Sub-grid standard deviation of elevation [m], shape (ncol,) (CLM surfdata
        ``STD_ELEV``).
    n_bands : int
        Number of equal-area bands.  Only 5 is supported (standard-normal quintile
        table); other values raise.

    Returns
    -------
    band_dz : jnp.ndarray
        (ncol, n_bands) mean elevation anomaly of each band relative to the cell
        mean [m].  Area-weighted mean over bands is 0 by construction.
    """
    if n_bands != 5:
        raise ValueError(f"n_bands={n_bands} unsupported; only 5 equal-area bands "
                         "(standard-normal quintile table) are implemented.")
    anom = jnp.asarray(_BAND_ANOM_K5, dtype=jnp.asarray(std_elev).dtype)
    return jnp.asarray(std_elev)[:, None] * anom[None, :]


def band_precip_snow(
    T_air: jnp.ndarray,
    precip_total: jnp.ndarray,
    precip_snow: jnp.ndarray,
    cfg: ElevationSnowBandConfig,
) -> jnp.ndarray:
    """Per-band snowfall rate: downscale the CELL snow fraction by sub-grid relief.

    The band scheme must NOT re-partition phase on a flat cell — it must reproduce
    the incoming ``precip_snow`` EXACTLY at zero relief, whatever convention the
    atmosphere used to set it (mixed-phase microphysics, a smooth ramp, or a hard
    threshold).  Relief shifts the phase by the difference between the band and the
    cell freezing indicators::

        snowfall_k = clip(precip_snow
                          + precip_total * (1[T_band_k < Tf] - 1[T_air < Tf]),
                          0, precip_total)

    so a cold high band above a warm cell converts its rain to snow, and a warm low
    band below a cold cell converts its snow to rain.  At ``dz = 0`` the indicator
    difference is zero -> ``snowfall_k = precip_snow`` (exact no-relief identity,
    independent of the forcing's phase convention).  Total precip is conserved per
    band (band rain = ``precip_total - snowfall_k``).

    Parameters
    ----------
    T_air : jnp.ndarray
        Cell lowest-level air temperature [K], shape (ncol,).
    precip_total, precip_snow : jnp.ndarray
        Cell total and snow-phase precipitation rate [kg/m2/s], shape (ncol,); the
        band scheme redistributes the snow FRACTION and preserves it at zero relief.
    cfg : ElevationSnowBandConfig

    Returns
    -------
    snowfall : jnp.ndarray
        (ncol, n_bands) snowfall rate [kg/m2/s].
    """
    T_band = T_air[:, None] - cfg.lapse_rate_K_m * cfg.band_dz
    Tf = constants.T_freeze
    dtype = precip_total.dtype
    snow_ind_band = (T_band < Tf).astype(dtype)
    snow_ind_cell = (T_air < Tf).astype(dtype)[:, None]
    shift = precip_total[:, None] * (snow_ind_band - snow_ind_cell)
    return jnp.clip(precip_snow[:, None] + shift, 0.0, precip_total[:, None])


class SnowBandStep(NamedTuple):
    """Result of one banded snow + firn/ice budget step (masses [kg/m2])."""
    swe_bands: jax.Array       # (ncol, n_bands) updated seasonal-snow SWE
    swe_total: jax.Array       # (ncol,) area-weighted aggregate seasonal SWE
    ice_bands: jax.Array       # (ncol, n_bands) updated firn/glacier-ice reservoir
    ice_total: jax.Array       # (ncol,) area-weighted aggregate ice
    snow_age_bands: jax.Array  # (ncol, n_bands) per-band snow age [s]
    snow_age: jax.Array        # (ncol,) SWE-weighted aggregate age [s] (diagnostic)
    snow_melt: jax.Array       # (ncol,) area-weighted seasonal-snow melt [kg/m2] (-> infiltration)
    ice_melt: jax.Array        # (ncol,) area-weighted ablation ice melt [kg/m2] (-> runoff)
    ice_runoff: jax.Array      # (ncol,) frozen glacier discharge + ceiling [kg/m2/s] (-> runoff)
    refreeze: jax.Array        # (ncol,) rain refrozen into the pack [kg/m2] (gap 6; -> +L_f to G, NOT infiltration)
    blow_subl: jax.Array       # (ncol,) blowing-snow sublimation SWE loss [kg/m2/s] (gap 5; -> atmosphere)


def step_snow_bands(
    swe_bands: jnp.ndarray,
    snow_age_bands: jnp.ndarray,
    ice_bands: jnp.ndarray,
    T_sfc: jnp.ndarray,
    snowfall_bands: jnp.ndarray,
    dt: float,
    *,
    Q_net: jnp.ndarray,
    cfg: ElevationSnowBandConfig,
    T_snow_melt: float = constants.T_freeze,
    precip_rain_bands: jnp.ndarray | None = None,
    wind: jnp.ndarray | None = None,
) -> SnowBandStep:
    """Advance the banded seasonal-snow + firn/glacier-ice budget one step.

    Per band, in sequence (all gated by the band-downscaled surface temperature
    ``T_sfc - lapse*dz``, reusing :func:`update_snow`):

    1. **Seasonal snow** accumulates from the band snowfall and melts (energy-limited).
       ``Q_net`` is the melt-energy ceiling — the scalar cell energy balance
       ``(ncol,)`` broadcast to every band, or the PER-BAND net radiation
       ``(ncol, n_bands)`` from :func:`band_net_radiation` (the banded SEB).
    2. **Firnification**: seasonal snow above ``swe_snow_cap`` compacts into the ice
       reservoir (``to_ice``) — DELAYED glacier storage, not same-step runoff.
    3. **Ablation**: melt energy left over after the seasonal snow is exhausted melts
       exposed ice (``ice_melt``, bounded by the ice mass) — the ablation zone.
    4. **Discharge**: the ice reservoir sheds ``ice/tau_ice_discharge_s`` per step plus
       any excess above the ``swe_cap`` ceiling, both as FROZEN runoff (``ice_runoff``).

    Seasonal-snow age resets on GROSS band snowfall (fresh snow always brightens the
    surface — codex finding: perennial ice must still brighten when it snows); the
    perennial darkening now comes from the SEPARATE ice reservoir's exposed-ice albedo
    (:func:`band_albedo`), not from freezing the age clock.  Per-band age lets fresh
    snow on one band rejuvenate only that band.

    Sign/budget conventions (all positive out of the frozen store): the per-band water
    budget closes as ``d(SWE)/dt = snowfall - snow_melt - to_ice`` and
    ``d(ice)/dt = to_ice - ice_melt - ice_runoff``, so summing,
    ``d(SWE+ice)/dt = snowfall - snow_melt - ice_melt - ice_runoff``.  The caller sends
    ``snow_melt`` to infiltration and ``ice_melt`` + ``ice_runoff`` to runoff, and
    charges ``(snow_melt + ice_melt)*L_f`` to the surface energy budget (ice_runoff is
    frozen outflow — no fusion).
    """
    # Cast the band geometry + trainable (traced) config scalars to the state dtype so a
    # float64 config cannot promote the returned float32 band mass/age outputs (coupled
    # lax.scan carry-dtype mismatch).
    dt_work = swe_bands.dtype
    dz = cfg.band_dz.astype(dt_work)
    _lapse = jnp.asarray(cfg.lapse_rate_K_m, dt_work)
    _snow_cap = jnp.asarray(cfg.swe_snow_cap, dt_work)
    _ice_cap = jnp.asarray(cfg.swe_cap, dt_work)
    _tau = jnp.asarray(cfg.tau_ice_discharge_s, dt_work)
    _refreeze_frac = jnp.asarray(cfg.refreeze_frac, dt_work)
    _blow_rate_c = jnp.asarray(cfg.blow_snow_subl_rate, dt_work)
    _blow_thresh = jnp.asarray(cfg.blow_snow_wind_thresh_ms, dt_work)
    T_sfc_band = T_sfc[:, None] - _lapse * dz
    # Q_net may be the scalar cell energy balance (ncol,) broadcast to every band, or
    # the per-band net radiation (ncol, n_bands) from band_net_radiation (banded SEB).
    Q_net_bands = Q_net[:, None] if jnp.ndim(Q_net) == 1 else Q_net

    # 1. Seasonal snow accumulate + energy-limited melt.  update_snow resets the age on
    #    GROSS band snowfall, so fresh snow rejuvenates the surface even over a glacier.
    swe_after_melt, age_new_bands, snow_melt = update_snow(
        swe_bands, snow_age_bands, T_sfc_band, snowfall_bands, dt,
        Q_net=Q_net_bands, T_snow_melt=T_snow_melt,
    )
    # 1b. Rain-on-snow refreezing (gap 6 cold content): rain onto a sub-freezing band
    #     with snow freezes into the pack, releasing L_f (the caller adds it to G) and
    #     adding to SWE — so it does NOT run off/infiltrate.  Single-layer band: no pack
    #     liquid store, so this is the dominant cold-content path (multi-layer retained
    #     liquid is the upgrade).
    if precip_rain_bands is not None:
        _cold_snow = (T_sfc_band < T_snow_melt) & (swe_after_melt > 1e-6)
        # ponytail: cap refreeze at the pack cold content so the released
        # L_f can't exceed swe*c_ice*(Tf - T_skin).  Uncapped, 20 mm/hr rain
        # on cold thin snow released ~1850 W/m2 of spurious surface heating.
        # Single-layer band has no pack T, so T_sfc_band is the proxy; the
        # multi-layer retained-liquid store is the upgrade.
        _refreeze_max = (
            swe_after_melt * constants.c_pi
            * jnp.maximum(T_snow_melt - T_sfc_band, 0.0) / constants.L_f
        )
        refreeze_bands = jnp.where(
            _cold_snow,
            jnp.minimum(_refreeze_frac * precip_rain_bands * dt, _refreeze_max),
            0.0)
        swe_after_melt = swe_after_melt + refreeze_bands
    else:
        refreeze_bands = jnp.zeros_like(swe_after_melt)
    # 2. Firnification: seasonal snow above the snow cap densifies into ice storage.
    to_ice = jnp.maximum(swe_after_melt - _snow_cap, 0.0)
    swe_new = swe_after_melt - to_ice
    ice_after_firn = ice_bands + to_ice
    # 3. Ablation: melt energy the seasonal snow could not consume ablates exposed ice.
    above = T_sfc_band >= T_snow_melt
    e_avail = jnp.where(above, jnp.maximum(Q_net_bands, 0.0) * dt, 0.0)  # J/m2 for melt
    e_residual = jnp.maximum(e_avail - snow_melt * constants.L_f, 0.0)   # after snow
    ice_melt = jnp.minimum(e_residual / constants.L_f, ice_after_firn)
    ice_after_melt = ice_after_firn - ice_melt
    # 4. Slow discharge (ice/tau) + hard ceiling overflow, both frozen runoff.  Bound
    #    the discharge by the available ice so a pathological dt/tau > 1 cannot drive the
    #    reservoir negative (runoff > storage).
    discharge = jnp.minimum(ice_after_melt * (dt / _tau),
                            ice_after_melt)
    ice_after_disch = ice_after_melt - discharge
    ceil_excess = jnp.maximum(ice_after_disch - _ice_cap, 0.0)
    ice_new = ice_after_disch - ceil_excess

    n_bands = swe_bands.shape[-1]
    # 5. Blowing-snow sublimation (gap 5): above the mobilisation wind threshold, snow
    #    sublimes from suspension — an SWE sink to the atmosphere (opt-in; rate 0 = off),
    #    bounded by the band SWE.  Removes preferentially from the exposed high bands via
    #    their (larger) SWE weighting.
    # Branch ONLY on the static ``wind is not None`` (Python) — the rate is a traced
    # trainable scalar, so a ``rate > 0`` Python branch would raise under jax.grad; a
    # zero rate just yields zero sublimation.
    if wind is not None:
        _blow_rate = _blow_rate_c * jnp.maximum(
            jnp.reshape(wind, (-1, 1)) - _blow_thresh, 0.0)
        _blow_mass = jnp.minimum(_blow_rate * dt, swe_new)
        swe_new = swe_new - _blow_mass
        blow_subl = jnp.sum(_blow_mass, axis=-1) / n_bands / dt
    else:
        blow_subl = jnp.zeros(swe_new.shape[0], dtype=swe_new.dtype)

    swe_sum = jnp.sum(swe_new, axis=-1)
    ok = swe_sum > 1e-12
    age_agg = jnp.where(
        ok, jnp.sum(age_new_bands * swe_new, axis=-1) / jnp.where(ok, swe_sum, 1.0), 0.0,
    )
    return SnowBandStep(
        swe_bands=swe_new,
        swe_total=swe_sum / n_bands,                          # equal-area bands
        ice_bands=ice_new,
        ice_total=jnp.sum(ice_new, axis=-1) / n_bands,
        snow_age_bands=age_new_bands,
        snow_age=age_agg,
        snow_melt=jnp.sum(snow_melt, axis=-1) / n_bands,
        ice_melt=jnp.sum(ice_melt, axis=-1) / n_bands,
        ice_runoff=jnp.sum(discharge + ceil_excess, axis=-1) / n_bands / dt,
        refreeze=jnp.sum(refreeze_bands, axis=-1) / n_bands,
        blow_subl=blow_subl,
    )


def band_snow_cover(
    swe_bands: jnp.ndarray,
    snow_cover_fn,
) -> jnp.ndarray:
    """Area-weighted aggregate snow-cover fraction over equal-area bands.

    ``snow_cover_fn`` maps SWE -> cover fraction (pass a closure over the albedo
    config, e.g. ``lambda s: snow_cover_fraction(s, config.land_albedo)``) — the
    banded aggregate of the SAME cover curve the cell-mean path uses.
    """
    return jnp.mean(snow_cover_fn(swe_bands), axis=-1)


def band_snow_albedo(
    swe_bands: jnp.ndarray,
    snow_age_bands: jnp.ndarray,
    snow_cover_fn,
    snow_albedo_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Area-weighted snow cover and per-band snow-albedo contribution.

    Each band blends its OWN age-decayed snow albedo (fresh snow bright, perennial
    firn darker), so the caller forms the cell albedo as::

        alpha = alpha_veg * (1 - f_snow) + snow_contrib

    with ``f_snow = mean_k cover(swe_k)`` and ``snow_contrib =
    mean_k [ snow_albedo(age_k) * cover(swe_k) ]``.  At uniform bands (flat cell,
    single age) this reduces to the cell-mean ``alpha_snow * f_snow``.

    ``snow_cover_fn`` maps SWE -> cover fraction and ``snow_albedo_fn`` maps age ->
    snow albedo (pass closures over the albedo config).

    Returns
    -------
    (f_snow, snow_contrib) : each (ncol,).
    """
    f = snow_cover_fn(swe_bands)                              # (ncol, n_bands)
    contrib = snow_albedo_fn(snow_age_bands) * f
    return jnp.mean(f, axis=-1), jnp.mean(contrib, axis=-1)


def band_albedo(
    swe_bands: jnp.ndarray,
    snow_age_bands: jnp.ndarray,
    base_albedo: jnp.ndarray,
    snow_cover_fn,
    snow_albedo_fn,
    *,
    ice_bands: jnp.ndarray | None = None,
    cfg: ElevationSnowBandConfig | None = None,
) -> jnp.ndarray:
    """Per-band land albedo: snow-free ground blended with each band's snow albedo.

    ``alpha_k = ground_k * (1 - cover(swe_k)) + snow_albedo(age_k) * cover(swe_k)``.

    The snow-free ``ground_k`` is the soil/vegetation ``base_albedo`` where there is no
    glacier ice, transitioning to the dark ablation-ice albedo ``cfg.alpha_glacier_ice``
    where the firn/ice reservoir is present (``ice_bands`` above ``cfg.ice_expose_kg_m2``,
    a smooth ramp).  This gives the bright-accumulation-firn / dark-ablation-ice contrast
    (gap 4): a high band keeps its bright seasonal snow, while a snow-free glaciated band
    shows dark bare ice.  With ``ice_bands=None`` the ground is just ``base_albedo`` and
    the band mean equals the aggregate ``base*(1 - f_snow) + snow_contrib`` from
    :func:`band_snow_albedo`.  ``base_albedo`` is the snow-free base (CLM PFT + dry-soil
    brightening), scalar or ``(ncol,)``.
    """
    base = jnp.reshape(base_albedo, (-1, 1)) if jnp.ndim(base_albedo) == 1 else base_albedo
    cover = snow_cover_fn(swe_bands)                          # (ncol, n_bands)
    if ice_bands is not None and cfg is not None:
        ice_frac = jnp.clip(ice_bands / cfg.ice_expose_kg_m2, 0.0, 1.0)
        ground = base * (1.0 - ice_frac) + cfg.alpha_glacier_ice * ice_frac
    else:
        ground = base
    return ground * (1.0 - cover) + snow_albedo_fn(snow_age_bands) * cover


class BandRadiation(NamedTuple):
    """Banded surface radiation balance (fluxes [W/m2], positive into the surface)."""
    Rn_bands: jax.Array     # (ncol, n_bands) net radiation per band (drives band melt)
    sw_net_agg: jax.Array   # (ncol,) area-weighted net SW absorbed (into shared soil)
    lw_net_agg: jax.Array   # (ncol,) area-weighted net LW (into shared soil)
    lw_up_agg: jax.Array    # (ncol,) area-weighted up-welling LW (diagnostic)
    alpha_eff: jax.Array    # (ncol,) SW-flux-weighted aggregate albedo (atmosphere)


def band_net_radiation(
    T_sfc: jnp.ndarray,
    alpha_bands: jnp.ndarray,
    sw_down: jnp.ndarray,
    lw_down: jnp.ndarray,
    emissivity: jnp.ndarray,
    cfg: ElevationSnowBandConfig,
) -> BandRadiation:
    """Per-band surface radiation balance with elevation-lapsed SW/LW down.

    Each band k gets its own skin temperature (lapse-slaved,
    ``T_skin_k = T_sfc - lapse*dz_k``), down-welling SW/LW scaled by the elevation
    lapse, and its own albedo::

        SW_down_k = sw_down * (1 + sw_elev_grad_per_m * dz_k)     (thinner air aloft)
        LW_down_k = lw_down - lw_elev_lapse_W_m2_per_m * dz_k     (colder air aloft)
        Rn_k      = SW_down_k*(1 - alpha_k)
                    + emissivity*LW_down_k - emissivity*sigma*T_skin_k^4

    Because the band anomalies ``dz_k`` are zero-mean (equal-area Gaussian bands), the
    linear SW/LW lapse conserves the cell-mean down-flux: ``mean_k SW_down_k = sw_down``
    exactly.  The AREA-WEIGHTED net SW/LW therefore drive the shared soil column while
    ``Rn_bands`` drives per-band melt, and the covariance between elevation and albedo
    (bright high bands intercept the enhanced high-altitude sun) is retained — the
    physical high-elevation cooling a single cell-mean radiation balance averages away.

    The atmosphere-facing aggregate albedo ``alpha_eff`` is SW-FLUX weighted
    (``mean_k SW_down_k*alpha_k / mean_k SW_down_k``) so reflected + absorbed = incident
    exactly (energy-conserving in a coupled run); it reduces to the plain band-mean
    albedo when ``sw_elev_grad_per_m = 0``.

    Sign convention: all fluxes positive INTO the surface; ``sigma`` T^4 is emission
    (out), entering ``lw_net`` with a minus sign.
    """
    # Cast the band geometry / emissivity to the working (state/forcing) dtype so a
    # float64 CLM band_dz cannot promote the returned radiation — and hence the land
    # state — to float64 under a float32 coupled lax.scan carry (dtype-carry mismatch).
    dt_work = T_sfc.dtype
    dz = cfg.band_dz.astype(dt_work)                         # (ncol, n_bands)
    n_bands = dz.shape[-1]
    # alpha_bands comes from band_albedo, which mixes the (possibly float64, trainable)
    # ice_expose_kg_m2 / alpha_glacier_ice config scalars — cast it so a promoted albedo
    # cannot promote the returned radiation under a float32 carry.
    alpha_bands = jnp.asarray(alpha_bands, dtype=dt_work)
    emis = jnp.asarray(emissivity, dtype=dt_work)
    emis = jnp.reshape(emis, (-1, 1)) if emis.ndim == 1 else emis
    # Cast EVERY config scalar entering the arithmetic to the working dtype: these are
    # trainable (traced) scalars that may be float64 while the coupled state is float32,
    # and an un-cast scalar would promote the returned radiation (hence the land state).
    _lapse = jnp.asarray(cfg.lapse_rate_K_m, dt_work)
    _lw_lapse = jnp.asarray(cfg.lw_elev_lapse_W_m2_per_m, dt_work)
    _sw_grad = jnp.asarray(cfg.sw_elev_grad_per_m, dt_work)
    _sky_min = jnp.asarray(cfg.sky_view_min, dt_work)
    T_skin = T_sfc[:, None] - _lapse * dz
    # Elevation lapse of the down-flux, kept EXACTLY zero-mean (conservation): the band
    # anomalies dz are zero-mean, and a per-COLUMN scalar slope times dz is still
    # zero-mean, so mean_k(SW/LW down_k) == the cell forcing.  Cap each column's slope
    # so no band's down-flux can go negative at extreme relief (LW-down is smallest at
    # the highest band dz_hi; SW-down at the lowest band dz_lo < 0) — a floor on the flux
    # itself would break conservation, so we bound the SLOPE instead.
    dz_hi = jnp.max(dz, axis=-1, keepdims=True)             # highest band (dz >= 0)
    dz_lo = jnp.min(dz, axis=-1, keepdims=True)             # lowest band (dz <= 0)
    lw_slope = jnp.minimum(_lw_lapse,
                           jnp.maximum(lw_down[:, None], 0.0) / jnp.maximum(dz_hi, 1e-6))
    sw_slope = jnp.minimum(_sw_grad, 1.0 / jnp.maximum(-dz_lo, 1e-6))
    sw_down_b = sw_down[:, None] * (1.0 + sw_slope * dz)
    lw_down_b = lw_down[:, None] - lw_slope * dz
    sw_net_b = sw_down_b * (1.0 - alpha_bands)
    lw_up_b = emis * constants.sigma_sb * T_skin ** 4
    lw_net_b = emis * lw_down_b - lw_up_b
    # Terrain sky-view (gap 5): a shielded (valley) band exchanges the blocked sky
    # fraction with surrounding slopes at ~its own skin T (no net exchange there), so
    # its NET longwave scales by the per-band sky-view V_band — 1 at the ridge
    # (dz_hi), sky_view_min in the deepest valley (dz_lo).  V==1 (sky_view_min=1) is a
    # no-op.  Sub-grid geometry proxy from the band elevation spread.
    dz_span = jnp.maximum(dz_hi - dz_lo, 1e-6)
    shielding = jnp.clip((dz_hi - dz) / dz_span, 0.0, 1.0)     # 0 ridge -> 1 valley
    v_band = 1.0 - (1.0 - _sky_min) * shielding
    lw_net_b = v_band * lw_net_b
    # Effective TOTAL UPWARD LW for the atmosphere, in the repo convention (see
    # surface_energy.surface_radiation_fluxes: lw_up = emit + reflected = lw_down -
    # lw_net), CONSISTENT with the sky-view-scaled net.  This also corrects the plain
    # (V=1) case, which must report emit + (1-emis)*lw_down, not emit alone.
    lw_up_b = lw_down_b - lw_net_b
    Rn_b = sw_net_b + lw_net_b
    # SW-flux-weighted aggregate albedo: reflected / incident.  mean_k sw_down_b is
    # exactly sw_down (zero-mean dz), so alpha_eff = mean_k sw_down_b*alpha / sw_down.
    refl = jnp.mean(sw_down_b * alpha_bands, axis=-1)        # (ncol,)
    inc = jnp.mean(sw_down_b, axis=-1)                       # (ncol,) == sw_down
    _lit = inc > 1e-6
    alpha_eff = jnp.where(
        _lit, refl / jnp.where(_lit, inc, 1.0), jnp.mean(alpha_bands, axis=-1),
    )
    return BandRadiation(
        Rn_bands=Rn_b,
        sw_net_agg=jnp.sum(sw_net_b, axis=-1) / n_bands,
        lw_net_agg=jnp.sum(lw_net_b, axis=-1) / n_bands,
        lw_up_agg=jnp.sum(lw_up_b, axis=-1) / n_bands,
        alpha_eff=alpha_eff,
    )
