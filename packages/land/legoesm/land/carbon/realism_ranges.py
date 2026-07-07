"""Published biome-scale carbon realism ranges (single source of truth).

Order-of-magnitude realism gates for the land-carbon validators -- NOT a
calibration target.  Shared by:

* ``scripts/validate/land_carbon_equilibrium.py`` (per-pixel realism audit), and
* ``scripts/validate/global_carbon_ic_map.py`` (per-PFT realism of the global
  carbon initial-condition map).

Keeping the table here (rather than re-declaring it in each validator) means a
range is edited once and both harnesses see the change.

Sources
-------
* GPP -- Beer et al. 2010, Science 329.
* NPP, live biomass -- Saugier, Roy & Mooney 2001 (Terrestrial Global
  Productivity).
* Soil organic C to ~1 m -- Jobbagy & Jackson 2000, Ecol. Appl. 10.
* Peak LAI -- GLASS / MODIS LAI climatology.

All literals below live at MODULE scope (a plain data table), never inside a
function body, so the inline-coefficient ratchet (``tests/
test_no_inline_physics_coeffs.py``) is satisfied without a ``# coeff-ok``.
"""

from __future__ import annotations

# annual GPP / NPP [gC/m2/yr], live biomass C [kgC/m2], soil organic C to ~1 m
# [kgC/m2], peak LAI [m2/m2], keyed by biome.
LITERATURE_BIOME_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "tropical_forest":  dict(gpp=(2500, 3500), npp=(900, 1500), biomass=(15, 25), soc=(8, 15),  lai=(4.5, 7.0)),
    "savanna":          dict(gpp=(1000, 2000), npp=(400, 900),  biomass=(2, 8),   soc=(4, 12),  lai=(1.0, 3.0)),
    "temperate_forest": dict(gpp=(1200, 2000), npp=(600, 1000), biomass=(8, 18),  soc=(8, 20),  lai=(3.0, 6.0)),
    "grassland":        dict(gpp=(500, 1300),  npp=(200, 600),  biomass=(0.2, 1.5), soc=(6, 20), lai=(1.0, 3.0)),
    "boreal_forest":    dict(gpp=(600, 1200),  npp=(200, 500),  biomass=(4, 12),  soc=(10, 30), lai=(1.5, 4.0)),
    "shrubland":        dict(gpp=(300, 900),   npp=(100, 400),  biomass=(0.5, 4), soc=(3, 10),  lai=(0.5, 2.0)),
    "tundra":           dict(gpp=(150, 600),   npp=(50, 250),   biomass=(0.2, 1.5), soc=(15, 40), lai=(0.3, 1.5)),
}

# CLM5 PFT name -> the biome whose published range best brackets it, so a
# per-PFT realism check can look up SOC / biomass bounds from the biome table
# above.  Trees map to their thermal-zone forest; shrubs -> shrubland;
# grasses / crops -> the herbaceous bucket by thermal zone (arctic grass ->
# tundra, C4 grass -> savanna, C3 grass / crops -> grassland).  ``bare_soil``
# carries no vegetation, so it has no biome range (``None``).
PFT_BIOME: dict[str, str | None] = {
    "bare_soil": None,
    "needleleaf_evergreen_temperate": "temperate_forest",
    "needleleaf_evergreen_boreal": "boreal_forest",
    "needleleaf_deciduous_boreal": "boreal_forest",
    "broadleaf_evergreen_tropical": "tropical_forest",
    "broadleaf_evergreen_temperate": "temperate_forest",
    "broadleaf_deciduous_tropical": "tropical_forest",
    "broadleaf_deciduous_temperate": "temperate_forest",
    "broadleaf_deciduous_boreal": "boreal_forest",
    "broadleaf_evergreen_shrub": "shrubland",
    "broadleaf_deciduous_temperate_shrub": "shrubland",
    "broadleaf_deciduous_boreal_shrub": "shrubland",
    "c3_arctic_grass": "tundra",
    "c3_grass": "grassland",
    "c4_grass": "savanna",
    "crop_c3": "grassland",
    "crop_c4": "grassland",
}
