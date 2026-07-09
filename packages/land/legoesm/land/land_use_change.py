"""Land-use-change carbon bookkeeping (E_LUC) — BLUE / Houghton style.

When land cover changes (LUH2 transitions: forest -> crop, pasture abandonment,
...), carbon moves between the land and the atmosphere.  This module is a
**bookkeeping** estimate of that flux (E_LUC), following the BLUE model (Hansis
et al. 2015, *Global Biogeochemical Cycles*, doi:10.1002/2014GB004997) and the
Houghton bookkeeping tradition: a separate carbon-accounting layer driven by the
change in PFT cover fractions, leaving the column biophysics + DALEC untouched.

Sign convention
---------------
``E_LUC`` is **positive = a source to the atmosphere** (carbon leaving the land).
Deforestation (high-biomass forest -> low-biomass crop) is a source; regrowth on
abandoned land is a sink (negative contribution).

The model
---------
Per bookkeeping step (naturally annual, matching LUH2), given the PFT cover the
step before (``frac_prev``) and now (``frac_now``) and a per-PFT reference
vegetation carbon density ``veg_ref`` [gC/m2 of fully-covered patch]:

  - **Cleared** area (a PFT that lost cover) releases its vegetation carbon,
    ``cleared_C``, split three ways: a burn fraction to the atmosphere at once, a
    slash fraction retained in the land (litter/soil), and the remainder into
    1/10/100-yr **product pools** that decay first-order to the atmosphere.
  - **Gained** area (a PFT that gained cover) accrues carbon toward its reference
    as it regrows, modelled as a first-order **regrowth-debt** relaxation (a
    delayed sink).

  ``E_LUC = burn + sum(product-pool decay) - regrowth uptake``   [gC/m2 per step]

Conservation
------------
The bookkeeping conserves carbon by construction: product pools only decay what
was added (non-negative, first-order), and integrating a single deforestation
pulse to equilibrium emits exactly ``(1 - slash) * cleared_C`` while the pools
drain to zero (the analytic test anchor, closed to ~1e-9).

Tiling seam (future work)
-------------------------
``veg_ref`` is an INPUT, not baked in: v1 uses the fixed reference table
:func:`default_veg_carbon_density`, but a future full per-PFT sub-grid tiling
replaces it with the model's own prognostic per-tile vegetation carbon — the
:func:`eluc_step` signature does not change.

Pure JAX; differentiable; no host side effects.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# --- Reference vegetation carbon density per CLM5 PFT [gC/m2] ---
# Order-of-magnitude standing biomass carbon (above + below ground) for a
# fully-covered patch of each PFT, aligned to CLM5_PFT_NAMES.  A v1 FIXED
# reference (Houghton 2003, Tellus B; IPCC AR5 WG1 biomass-C densities); the
# tiling seam (see module docstring) later replaces it with prognostic per-tile
# vegetation carbon.  Forests hold ~1-2 orders of magnitude more than grass/crop.
_VEG_C_DENSITY_GC_M2 = (
    0.0,        # bare_soil
    12000.0,    # needleleaf_evergreen_temperate
    8000.0,     # needleleaf_evergreen_boreal
    6000.0,     # needleleaf_deciduous_boreal
    18000.0,    # broadleaf_evergreen_tropical
    14000.0,    # broadleaf_evergreen_temperate
    14000.0,    # broadleaf_deciduous_tropical
    12000.0,    # broadleaf_deciduous_temperate
    7000.0,     # broadleaf_deciduous_boreal
    3000.0,     # broadleaf_evergreen_shrub
    3000.0,     # broadleaf_deciduous_temperate_shrub
    2500.0,     # broadleaf_deciduous_boreal_shrub
    400.0,      # c3_arctic_grass
    500.0,      # c3_grass
    600.0,      # c4_grass
    400.0,      # crop_c3
    500.0,      # crop_c4
)

# Valid E_LUC schemes (dispatch discipline: unknown -> raise).
LUC_SCHEMES = ("none", "bookkeeping")

# Grams of carbon per petagram (gC -> PgC for the global E_LUC diagnostic).
_GC_PER_PG = 1.0e15


__param_spec__ = {
    "LandUseChangeConfig": {
        "scheme_key": "land.land_use_change",
        "excluded": {},
        "params": {
            "clear_burn_frac": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "BLUE (Hansis et al. 2015) / Houghton bookkeeping", "shape": None},
            "clear_slash_frac": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "BLUE (Hansis et al. 2015) / Houghton bookkeeping", "shape": None},
            "prod_frac_1yr": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "BLUE (Hansis et al. 2015) product pools", "shape": None},
            "prod_frac_10yr": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "BLUE (Hansis et al. 2015) product pools", "shape": None},
            "prod_frac_100yr": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "BLUE (Hansis et al. 2015) product pools", "shape": None},
            "tau_1yr_years": {"units": "year", "bounds": (0.5, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Houghton product-pool lifetimes (1/10/100 yr)", "shape": None},
            "tau_10yr_years": {"units": "year", "bounds": (5.0, 20.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Houghton product-pool lifetimes (1/10/100 yr)", "shape": None},
            "tau_100yr_years": {"units": "year", "bounds": (50.0, 200.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Houghton product-pool lifetimes (1/10/100 yr)", "shape": None},
            "tau_regrow_years": {"units": "year", "bounds": (10.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "secondary-forest regrowth timescale (Houghton bookkeeping)", "shape": None},
        },
    },
}


class LandUseChangeConfig(NamedTuple):
    """E_LUC bookkeeping configuration.

    scheme="none"        : disabled (default).
    scheme="bookkeeping" : BLUE/Houghton bookkeeping (:func:`eluc_step`).

    Cleared vegetation carbon splits into ``clear_burn_frac`` (immediate
    emission), ``clear_slash_frac`` (retained in land), and the remainder into
    the 1/10/100-yr product pools by ``prod_frac_*`` (which sum to 1).
    """
    scheme: str = "none"
    # Cleared-vegetation carbon partition (burn + slash + products, where
    # products = 1 - burn - slash; burn + slash must be <= 1).
    clear_burn_frac: float = 0.2
    clear_slash_frac: float = 0.3
    # Split of the product portion across the three product pools (sum to 1).
    prod_frac_1yr: float = 0.5
    prod_frac_10yr: float = 0.3
    prod_frac_100yr: float = 0.2
    # First-order product-pool lifetimes [years].
    tau_1yr_years: float = 1.0
    tau_10yr_years: float = 10.0
    tau_100yr_years: float = 100.0
    # Secondary-vegetation regrowth timescale [years].
    tau_regrow_years: float = 30.0


class ELUCState(NamedTuple):
    """Bookkeeping carbon in transit + cumulative flux.  All fields ``(ncol,)`` [gC/m2].

    Product pools hold cleared wood/product carbon decaying to the atmosphere;
    ``regrow_debt`` is the carbon secondary vegetation will still take up as it
    matures; ``cum_eluc`` accumulates the emitted E_LUC (diagnostic + the
    conservation anchor).
    """
    prod_1yr: jax.Array
    prod_10yr: jax.Array
    prod_100yr: jax.Array
    regrow_debt: jax.Array
    cum_eluc: jax.Array


def init_eluc_state(ncol: int) -> ELUCState:
    """Zeroed bookkeeping state for ``ncol`` columns."""
    z = jnp.zeros(ncol)
    return ELUCState(prod_1yr=z, prod_10yr=z, prod_100yr=z,
                     regrow_debt=z, cum_eluc=z)


def default_veg_carbon_density() -> jax.Array:
    """Per-CLM5-PFT reference vegetation carbon density ``(17,)`` [gC/m2]."""
    return jnp.asarray(_VEG_C_DENSITY_GC_M2)


def validate_luc_config(cfg: LandUseChangeConfig) -> None:
    """Fail fast on a malformed E_LUC config (static Python values).

    Raises on an unknown scheme (dispatch discipline) or a partition that does
    not conserve carbon (burn + slash > 1, or product fractions not summing to 1).
    """
    if cfg.scheme not in LUC_SCHEMES:
        raise ValueError(
            f"unknown land-use-change scheme {cfg.scheme!r}; expected one of {LUC_SCHEMES}")
    if cfg.scheme == "none":
        return
    if cfg.clear_burn_frac < 0.0 or cfg.clear_slash_frac < 0.0:
        raise ValueError("clear_burn_frac and clear_slash_frac must be >= 0")
    if cfg.clear_burn_frac + cfg.clear_slash_frac > 1.0:
        raise ValueError(
            f"clear_burn_frac + clear_slash_frac must be <= 1 "
            f"(got {cfg.clear_burn_frac} + {cfg.clear_slash_frac})")
    # Each product fraction must be non-negative: a negative fraction that still
    # summed to 1 (e.g. -1, 2, 0) would drive a product pool negative.
    for name in ("prod_frac_1yr", "prod_frac_10yr", "prod_frac_100yr"):
        if getattr(cfg, name) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    prod_sum = cfg.prod_frac_1yr + cfg.prod_frac_10yr + cfg.prod_frac_100yr
    if abs(prod_sum - 1.0) > 1e-6:
        raise ValueError(f"prod_frac_1yr+10yr+100yr must sum to 1 (got {prod_sum})")
    for name in ("tau_1yr_years", "tau_10yr_years", "tau_100yr_years", "tau_regrow_years"):
        if getattr(cfg, name) <= 0.0:
            raise ValueError(f"{name} must be > 0")


def eluc_step(state, frac_prev, frac_now, veg_ref, cfg, dt_years):
    """Advance the E_LUC bookkeeping one step; return ``(new_state, eluc_flux)``.

    Parameters
    ----------
    state : :class:`ELUCState`.
    frac_prev, frac_now : ``(ncol, npft)`` PFT cover fractions, before and after.
    veg_ref : ``(npft,)`` or ``(ncol, npft)`` reference vegetation C density [gC/m2].
    cfg : :class:`LandUseChangeConfig` (scheme must be ``"bookkeeping"``).
    dt_years : bookkeeping step length [years], **>= 0** (LUH2 is annual -> 1.0).
        A negative value is unphysical and would overflow the decay exponentials;
        the caller (the annual driver hook) always passes a fixed positive value.

    Returns
    -------
    ``(new_state, eluc_flux)`` where ``eluc_flux`` is ``(ncol,)`` [gC/m2 over the
    step], **positive = source to the atmosphere**.
    """
    # Area that lost / gained cover this step, and its reference vegetation carbon.
    cleared = jnp.maximum(frac_prev - frac_now, 0.0)               # (ncol, npft)
    gained = jnp.maximum(frac_now - frac_prev, 0.0)
    cleared_C = jnp.sum(cleared * veg_ref, axis=-1)                # (ncol,) gC/m2
    gained_C = jnp.sum(gained * veg_ref, axis=-1)

    # Partition cleared vegetation carbon: burn (atmosphere now) / slash (retained
    # in land) / products (1 - burn - slash), the latter into the three pools.
    burn = cfg.clear_burn_frac * cleared_C
    to_products = (1.0 - cfg.clear_burn_frac - cfg.clear_slash_frac) * cleared_C
    add_1 = cfg.prod_frac_1yr * to_products
    add_10 = cfg.prod_frac_10yr * to_products
    add_100 = cfg.prod_frac_100yr * to_products

    # First-order product-pool decay to the atmosphere.
    d1 = state.prod_1yr * (1.0 - jnp.exp(-dt_years / cfg.tau_1yr_years))
    d10 = state.prod_10yr * (1.0 - jnp.exp(-dt_years / cfg.tau_10yr_years))
    d100 = state.prod_100yr * (1.0 - jnp.exp(-dt_years / cfg.tau_100yr_years))
    product_emission = d1 + d10 + d100

    # Regrowth: gained area accrues carbon toward its reference (a delayed sink).
    regrow_debt = state.regrow_debt + gained_C
    uptake = regrow_debt * (1.0 - jnp.exp(-dt_years / cfg.tau_regrow_years))

    eluc = burn + product_emission - uptake                       # gC/m2 (source +)
    new_state = ELUCState(
        prod_1yr=state.prod_1yr - d1 + add_1,
        prod_10yr=state.prod_10yr - d10 + add_10,
        prod_100yr=state.prod_100yr - d100 + add_100,
        regrow_debt=regrow_debt - uptake,
        cum_eluc=state.cum_eluc + eluc,
    )
    return new_state, eluc


def annual_eluc_series(pft_frac_years, cell_area, cfg, *, veg_ref=None):
    """Run the annual E_LUC bookkeeping over a transient cover series.

    Loops the bookkeeping over consecutive native-year cover slices (E_LUC is
    annual and independent of the sub-daily biophysics scan), area-weighting each
    year's per-cell flux to a global total.  A post-run diagnostic, not a hot-loop
    routine (it host-syncs once per year).

    Parameters
    ----------
    pft_frac_years : ``(nyear, ncol, npft)`` fraction-of-gridcell PFT cover
        (``GlobalSurfaceData.pft_frac``).
    cell_area : ``(ncol,)`` grid-cell area [m2] (``GlobalSurfaceData.cell_area``);
        NaNs (ocean) count as zero area.
    cfg : :class:`LandUseChangeConfig` (``scheme="bookkeeping"``).
    veg_ref : optional ``(npft,)`` reference vegetation C density [gC/m2]
        (default :func:`default_veg_carbon_density`).

    Returns
    -------
    ``(eluc_pgc, final_state)`` where ``eluc_pgc`` is ``(nyear,)`` global E_LUC
    [PgC/yr], positive = source; ``eluc_pgc[0] = 0`` (no prior year).  Product
    pools not yet drained at the last year carry committed future emissions.

    ponytail: Python loop over years — fine for the ~10^2-year horizons LUH2
    covers; a lax.scan would matter only for far longer series.
    """
    validate_luc_config(cfg)
    pft = jnp.asarray(pft_frac_years)
    nyear, ncol = pft.shape[0], pft.shape[1]
    if veg_ref is None:
        veg_ref = default_veg_carbon_density()
    area = jnp.nan_to_num(jnp.asarray(cell_area))
    st = init_eluc_state(ncol)
    eluc_pgc = [0.0]                              # year 0: no prior-year transition
    for i in range(1, nyear):
        st, flux = eluc_step(st, pft[i - 1], pft[i], veg_ref, cfg, 1.0)
        eluc_gc = jnp.sum(jnp.nan_to_num(flux) * area)        # gC/yr (ocean NaN -> 0)
        eluc_pgc.append(float(eluc_gc / _GC_PER_PG))
    return jnp.asarray(eluc_pgc), st
