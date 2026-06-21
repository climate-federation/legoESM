"""USDA soil texture classes, van-Genuchten parameters, and the texture triangle.

Shared between the single-column driver (``run_lmip``) and the global CLM
reference-soil-map provider (``clm_surface_map``).  The van-Genuchten retention
parameters are the Carsel & Parrish (1988) class averages; the classifier is the
standard USDA texture triangle on (% sand, % clay).

van_genuchten_theta from ``soil_hydraulics`` is reused to derive the wilting-point
and field-capacity water contents from the per-class VG curve (no re-derivation).
"""
from __future__ import annotations

import jax.numpy as jnp

# --- USDA texture classes + Carsel & Parrish (1988) van-Genuchten params ------
# Order is the canonical class index used by the texture triangle below.
# Fields: theta_r, theta_sat [m3/m3], alpha_vg [1/m], n_vg [-], K_sat [m/s].
USDA_TEXTURES = (
    "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam",
    "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay",
    "silty_clay", "clay",
)
SOIL_TEXTURE_VG = {
    "sand":            dict(theta_r=0.045, theta_sat=0.430, alpha_vg=14.5, n_vg=2.68, K_sat=8.25e-5),
    "loamy_sand":      dict(theta_r=0.057, theta_sat=0.410, alpha_vg=12.4, n_vg=2.28, K_sat=4.05e-5),
    "sandy_loam":      dict(theta_r=0.065, theta_sat=0.410, alpha_vg=7.5,  n_vg=1.89, K_sat=1.22e-5),
    "loam":            dict(theta_r=0.078, theta_sat=0.430, alpha_vg=3.6,  n_vg=1.56, K_sat=2.89e-6),
    "silt_loam":       dict(theta_r=0.067, theta_sat=0.450, alpha_vg=2.0,  n_vg=1.41, K_sat=1.25e-6),
    "sandy_clay_loam": dict(theta_r=0.100, theta_sat=0.390, alpha_vg=5.9,  n_vg=1.48, K_sat=3.64e-6),
    "clay_loam":       dict(theta_r=0.095, theta_sat=0.410, alpha_vg=1.9,  n_vg=1.31, K_sat=7.22e-7),
    "silty_clay_loam": dict(theta_r=0.089, theta_sat=0.430, alpha_vg=1.0,  n_vg=1.23, K_sat=1.94e-7),
    "sandy_clay":      dict(theta_r=0.100, theta_sat=0.380, alpha_vg=2.7,  n_vg=1.23, K_sat=3.33e-6),
    "silty_clay":      dict(theta_r=0.070, theta_sat=0.360, alpha_vg=0.5,  n_vg=1.09, K_sat=5.56e-8),
    "clay":            dict(theta_r=0.068, theta_sat=0.380, alpha_vg=0.8,  n_vg=1.09, K_sat=5.56e-8),
}
_N_TEX = len(USDA_TEXTURES)
_TEX_IDX = {name: i for i, name in enumerate(USDA_TEXTURES)}

# --- soil-water potentials for plant-available water [m] (published conventions) -
_PSI_WILTING_M = -150.0      # wilting point ~ -1.5 MPa (FAO/CLM)
_PSI_FIELD_CAP_M = -3.36     # field capacity ~ -33 kPa


def usda_texture_index(pct_sand, pct_clay):
    """USDA texture-triangle class index (into :data:`USDA_TEXTURES`) from
    percent sand and percent clay (each in [0,100], same shape).  Vectorized.

    Implements the standard 12-class triangle (USDA Soil Survey Manual); the silt
    class (no Carsel-Parrish average of its own here) is folded into ``silt_loam``.
    All numeric thresholds below are the published triangle boundaries in
    % sand/clay/silt, not tunable coefficients.  Returns int32 indices.
    """
    sand = jnp.asarray(pct_sand, dtype=jnp.float64)
    clay = jnp.asarray(pct_clay, dtype=jnp.float64)
    silt = 100.0 - sand - clay  # coeff-ok: percentages sum to 100
    I = _TEX_IDX
    idx = jnp.full(sand.shape, I["loam"], dtype=jnp.int32)

    def pick(cond, name, cur):
        return jnp.where(cond, jnp.int32(I[name]), cur)

    # Order matters: assign coarse first, then override with finer clay classes.
    idx = pick((silt + 1.5 * clay) < 15, "sand", idx)  # coeff-ok: USDA triangle boundary
    idx = pick(((silt + 1.5 * clay) >= 15) & ((silt + 2 * clay) < 30), "loamy_sand", idx)  # coeff-ok: USDA triangle boundary
    idx = pick(((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2 * clay) >= 30)) | ((clay < 7) & (silt < 50) & ((silt + 2 * clay) >= 30)), "sandy_loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52), "loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick(((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12)), "silt_loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((silt >= 80) & (clay < 12), "silt_loam", idx)  # coeff-ok: USDA triangle boundary (silt->silt_loam)
    idx = pick((clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45), "sandy_clay_loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45), "clay_loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 27) & (clay < 40) & (sand <= 20), "silty_clay_loam", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 35) & (sand > 45), "sandy_clay", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 40) & (silt >= 40), "silty_clay", idx)  # coeff-ok: USDA triangle boundary
    idx = pick((clay >= 40) & (sand <= 45) & (silt < 40), "clay", idx)  # coeff-ok: USDA triangle boundary
    return idx


def vg_params_from_index(idx):
    """Per-cell van-Genuchten arrays (theta_r, theta_sat, alpha_vg, n_vg, K_sat)
    from texture indices (any shape)."""
    table = {k: jnp.asarray([SOIL_TEXTURE_VG[t][k] for t in USDA_TEXTURES],
                            dtype=jnp.float64)
             for k in ("theta_r", "theta_sat", "alpha_vg", "n_vg", "K_sat")}
    i = jnp.asarray(idx, dtype=jnp.int32)
    return {k: v[i] for k, v in table.items()}


def wilting_field_capacity(vg):
    """(theta_wp, theta_fc) from a per-cell VG dict, via the van-Genuchten curve
    at psi=-150 m (wilting, ~-1.5 MPa) and psi=-3.36 m (field capacity, -33 kPa).
    Reuses ``soil_hydraulics.van_genuchten_theta`` with per-cell array config
    fields (no re-derived retention curve)."""
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, van_genuchten_theta
    # Per-cell config: the VG retention fields are arrays; van_genuchten_theta
    # broadcasts over them (the other fields keep their scalar defaults, unused).
    cfg = SoilHydraulicsConfig(
        theta_r=vg["theta_r"], theta_sat=vg["theta_sat"],
        alpha_vg=vg["alpha_vg"], n_vg=vg["n_vg"], K_sat=vg["K_sat"])
    theta_wp = van_genuchten_theta(jnp.asarray(_PSI_WILTING_M), cfg)
    theta_fc = van_genuchten_theta(jnp.asarray(_PSI_FIELD_CAP_M), cfg)
    return theta_wp, theta_fc
