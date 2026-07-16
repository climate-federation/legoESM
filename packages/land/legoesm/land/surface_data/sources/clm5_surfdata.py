"""CLM5 surfdata source: cover + PFT + LAI for the **v1** harmonized surfdata.

Why this exists (v1 — read before changing)
-------------------------------------------
An observational cover/PFT/LAI pipeline is the intended long-term path, but a
CLM5 surfdata file is the right v1 source for two reasons:

  1. **PFT-specific LAI consistency.** A canopy needs LAI *per PFT patch*.  CLM5
     ``MONTHLY_LAI`` is **internally consistent** with its ``PCT_NAT_PFT`` cover
     (same source, same distribution, same PFT axis); transplanting a model LAI
     onto a *different* cover map gives zero LAI exactly where the two disagree.
  2. **Climate-zoned PFTs.** CLM5's PFTs already carry the climate zone, so no
     temperature climatology / classification crosswalk is needed.

So **v1 uses CLM5 surfdata for cover + PFT + LAI** (soil is overridden by the
better, observational HWSD product — see
:mod:`legoesm.land.surface_data.sources.hwsd2`).  A v2 swap-in would replace this
single source module with an observational cover/PFT/LAI source.

Landunit / PFT-axis reconstruction
----------------------------------
CLM stores cover in *landunits*: natural-veg (``PCT_NATVEG``) with ``natpft=15``
fractions (``PCT_NAT_PFT``), and crop (``PCT_CROP``) with ``cft=2`` fractions
(``PCT_CFT``).  ``MONTHLY_LAI`` is on ``lsmpft=17`` = 15 natural + 2 crop.  The
per-gridcell 17-PFT weight aligned with LAI is therefore::

    w[0:15] = (PCT_NATVEG/100) * PCT_NAT_PFT     # natural PFTs (bare at index 0)
    w[15:17] = (PCT_CROP/100)  * PCT_CFT         # crops

which matches :data:`legoesm.land.surface_params.CLM5_PFT_NAMES` order exactly.

Host-side only; not traced.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CLM5SurfdataConfig(NamedTuple):
    """Variable names for a CLM5 surfdata NetCDF (defaults match c250612 files)."""

    nat_pft_var: str = "PCT_NAT_PFT"     # (natpft, lat, lon) % within natural veg
    natveg_var: str = "PCT_NATVEG"       # (lat, lon) % gridcell natural veg
    crop_var: str = "PCT_CROP"           # (lat, lon) % gridcell crop
    cft_var: str = "PCT_CFT"             # (cft, lat, lon) % within crop
    lake_var: str = "PCT_LAKE"
    glacier_var: str = "PCT_GLACIER"
    landfrac_var: str = "LANDFRAC_PFT"   # (lat, lon) gridcell land fraction [0,1]
    color_var: str = "SOIL_COLOR"
    lai_var: str = "MONTHLY_LAI"         # (time, lsmpft, lat, lon)
    sai_var: str = "MONTHLY_SAI"
    htop_var: str = "MONTHLY_HEIGHT_TOP"
    hbot_var: str = "MONTHLY_HEIGHT_BOT"
    lat2d_var: str = "LATIXY"            # 2-D; reduced to 1-D centres
    lon2d_var: str = "LONGXY"
    year: int = 2015                     # stamp for the (stationary, single) slice


def assert_cover_within_landfrac(f_land, f_lake, f_glacier, landfrac, *, tol=1e-6):
    """Tripwire: the soil/veg + lake + glacier cover [percent-of-gridcell] must not
    exceed the gridcell land fraction ``landfrac`` [0,1] times 100.

    CLM landunit percentages (PCT_NATVEG/CROP/LAKE/GLACIER) sum to 100 *of the land
    part of the cell* and are populated EVEN OVER OCEAN (mksurfdata fills a land
    template everywhere; ``LANDFRAC_PFT`` gates which cells are active).  Using them
    un-gated as percent-of-gridcell smears land into coastal/ocean cells (the c250617
    +27% land over-count).  After gating by ``landfrac`` the cover is bounded by
    100*landfrac; this asserts that, so a regression that drops the gating fails loudly.
    Raises ``ValueError`` on the worst offending cell."""
    total = np.asarray(f_land) + np.asarray(f_lake) + np.asarray(f_glacier)
    bound = 100.0 * np.asarray(landfrac) + tol
    over = total - bound
    if np.any(over > 0):
        k = int(np.nanargmax(over))
        tot_flat = total.ravel()[k]
        lf_flat = np.asarray(landfrac).ravel()[k]
        raise ValueError(
            f"cover exceeds land fraction at cell {k}: f_land+f_lake+f_glacier="
            f"{tot_flat:.2f}% > 100*LANDFRAC={100 * lf_flat:.2f}% (excess "
            f"{tot_flat - 100 * lf_flat:.2f}%). CLM landunit percentages are "
            f"percent-of-LAND; they must be gated by LANDFRAC_PFT before use as "
            f"percent-of-gridcell (else land smears into ocean).")


def reconstruct_clm5_pft_frac(natveg, crop, nat_pft, cft):
    """CLM landunit cover → 17-PFT weight [% of gridcell], ``CLM5_PFT_NAMES`` order.

    CLM stores cover in *landunits*: natural-veg (``PCT_NATVEG``) split over
    ``natpft=15`` fractions (``PCT_NAT_PFT``), and crop (``PCT_CROP``) split over
    ``cft=2`` fractions (``PCT_CFT``).  ``MONTHLY_LAI`` (and any per-PFT quantity)
    lives on the combined ``lsmpft=17`` = 15 natural + 2 crop axis, so the aligned
    per-gridcell weight is::

        w[0:15]  = (PCT_NATVEG/100) * PCT_NAT_PFT     # natural PFTs (bare at index 0)
        w[15:17] = (PCT_CROP/100)   * PCT_CFT         # crops (c3, c4)

    Parameters
    ----------
    natveg, crop : (lat, lon) — gridcell natural-veg / crop cover [percent].
    nat_pft : (15, lat, lon) — natural-PFT split within natveg [percent].
    cft : (2, lat, lon) — crop-functional-type split within crop [percent].

    Returns
    -------
    (17, lat, lon) percent-of-gridcell weight (sums to ``PCT_NATVEG+PCT_CROP``
    where vegetated).  Host-side; not traced.  This is the single canonical
    landunit→PFT reconstruction — reused by :func:`read_clm5_cover_veg` and by the
    coupled-AMIP LAI loader (``legoesm.land.clm_surface_map``); do not re-derive.
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    natveg = np.asarray(natveg, dtype=np.float64)
    crop = np.asarray(crop, dtype=np.float64)
    nat_pft = np.asarray(nat_pft, dtype=np.float64)
    cft = np.asarray(cft, dtype=np.float64)
    # Landunit axes must be EXACTLY 15 natural + 2 crop in CLM5 order; a reordered or
    # merged source (e.g. natpft=16, cft=1) would still sum to 17 but silently
    # mis-align the concatenated per-PFT rows.  Guard the axis lengths + spatial shapes.
    if nat_pft.shape[0] != N_PFT_CLM5 - 2 or cft.shape[0] != 2:
        raise ValueError(
            f"expected {N_PFT_CLM5 - 2} natural PFTs + 2 crop CFTs (CLM5 17-PFT), "
            f"got natpft={nat_pft.shape[0]} + cft={cft.shape[0]}.")
    if not (natveg.shape == crop.shape == nat_pft.shape[1:] == cft.shape[1:]):
        raise ValueError(
            f"cover spatial shapes disagree: natveg{natveg.shape}, crop{crop.shape}, "
            f"nat_pft{nat_pft.shape[1:]}, cft{cft.shape[1:]}.")
    nat_w = (natveg[None, :, :] / 100.0) * nat_pft          # (15, lat, lon)
    crop_w = (crop[None, :, :] / 100.0) * cft               # (2, lat, lon)
    pft_frac = np.concatenate([nat_w, crop_w], axis=0)      # (17, lat, lon) %
    if pft_frac.shape[0] != N_PFT_CLM5:
        raise ValueError(
            f"reconstructed {pft_frac.shape[0]} PFTs, expected {N_PFT_CLM5}; "
            f"check natpft({nat_pft.shape[0]})+cft({cft.shape[0]}).")
    return pft_frac


def read_clm5_cover_veg(
    path: str | None = None,
    config: CLM5SurfdataConfig = CLM5SurfdataConfig(),
    *,
    dataset=None,
) -> dict:
    """Read cover + 17-PFT composition + monthly veg from a CLM5 surfdata file.

    Returns 1-D ``lat``/``lon`` [deg], cover fractions ``f_land``/``f_lake``/
    ``f_glacier`` in **percent** ``(lat, lon)``, ``pft_frac`` in percent of
    gridcell ``(npft=17, lat, lon)`` (sums to ``f_land`` where vegetated), the
    monthly veg fields ``(12, 17, lat, lon)`` in physical units, ``soil_color``,
    ``pft_names`` (the CLM5 17), and ``year``.  Soil is intentionally **not**
    returned — v1 takes soil from HWSD.
    """
    import xarray as xr  # noqa: F401
    from legoesm.land.surface_params import CLM5_PFT_NAMES

    ds = dataset if dataset is not None else xr.open_dataset(path, decode_times=False)
    try:
        lat = np.asarray(ds[config.lat2d_var].values, dtype=np.float64)[:, 0]
        lon = np.asarray(ds[config.lon2d_var].values, dtype=np.float64)[0, :]

        def arr(name):
            return np.asarray(ds[name].values, dtype=np.float64)

        natveg = arr(config.natveg_var)          # (lat, lon) % of LAND
        crop = arr(config.crop_var)              # (lat, lon) % of LAND
        nat_pft = arr(config.nat_pft_var)        # (natpft, lat, lon) % within natveg
        cft = arr(config.cft_var)                # (cft, lat, lon) % within crop
        # LANDFRAC_PFT [0,1] gates the landunit percentages to percent-of-GRIDCELL.
        # PCT_NATVEG/CROP/LAKE/GLACIER sum to 100 of the LAND part and are populated
        # even over ocean (mksurfdata fills a land template everywhere); without this
        # gate coastal/ocean cells acquire spurious land (the c250617 +27% over-count).
        landfrac = np.clip(arr(config.landfrac_var), 0.0, 1.0)   # (lat, lon)

        # 17-PFT weight as percent of gridcell, aligned to CLM5_PFT_NAMES order (single
        # canonical reconstruction — shared with the coupled-AMIP LAI loader). Gated by
        # landfrac so pft_frac.sum(0) == f_land (both percent-of-gridcell).
        pft_frac = (reconstruct_clm5_pft_frac(natveg, crop, nat_pft, cft)
                    * landfrac[None, :, :])       # (17, lat, lon) % of gridcell

        # Cover fractions as percent-of-gridcell (landunit % * landfrac).
        f_land = (natveg + crop) * landfrac       # soil/veg land (wetland/urban: v2)
        f_lake = arr(config.lake_var) * landfrac
        f_glacier = arr(config.glacier_var) * landfrac
        # Tripwire: gated cover must be bounded by 100*landfrac (fires if gating regresses).
        assert_cover_within_landfrac(f_land, f_lake, f_glacier, landfrac)

        # Monthly veg already on lsmpft=17 (time, lsmpft, lat, lon). Physical per-PFT
        # values (LAI/SAI/height) -- NOT fractions, so NOT gated by landfrac.
        veg = {k: arr(v) for k, v in (
            ("monthly_lai", config.lai_var), ("monthly_sai", config.sai_var),
            ("monthly_height_top", config.htop_var), ("monthly_height_bot", config.hbot_var),
        )}

        return {
            "lat": lat, "lon": lon,
            "f_land": f_land,                         # soil/veg land (wetland/urban: v2)
            "f_lake": f_lake,
            "f_glacier": f_glacier,
            "pft_frac": pft_frac,                     # (17, lat, lon) % of gridcell
            "soil_color": arr(config.color_var),
            "pft_names": np.array(CLM5_PFT_NAMES),
            "year": float(config.year),
            **veg,                                    # (12, 17, lat, lon)
        }
    finally:
        if dataset is None:
            ds.close()
