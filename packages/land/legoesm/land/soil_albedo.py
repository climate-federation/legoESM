"""Background soil albedo from CLM soil-colour class + top-layer wetness.

The surfdata carries an integer **soil colour class** (1-20); this module turns it
into a *background soil albedo* — the albedo of the bare ground beneath the
canopy, which the surface energy balance needs and which the latitude-band
vegetation albedo in :mod:`legoesm.surface_albedo` does not provide.

Following CLM (Bonan 1996; Lawrence & Chase 2007; CLM4 Tech Note Table 3.3 and
eq. 3.52), each colour class has dry and saturated albedos in the visible and
near-infrared bands, and the actual albedo darkens with soil wetness::

    alpha_Lambda = min(alpha_sat_Lambda + Delta, alpha_dry_Lambda)
    Delta        = max(0.11 - 0.40 * theta_top, 0)              # theta_top = vol. water content, top layer

so a fully dry soil reaches ``alpha_dry`` and a wet soil approaches
``alpha_sat``.  Functions are pure JAX and differentiable in ``theta_top``.

Faithfulness
------------
FAITHFUL to CLM/CTSM: the closed form above and the 20-class ``SOIL_COLOR_ALBEDO``
table reproduce CTSM ``src/biogeophys/SurfaceAlbedoMod.F90``
(``SurfaceAlbedoInitTimeConst``: ``inc = max(0.11 - 0.40*h2osoi_vol, 0)``,
``alb = min(albsat + inc, albdry)`` per band; the module's dry/sat VIS/NIR values
equal the CTSM ``albsat``/``albdry`` arrays exactly).  DEPARTURE / SURROGATE:
``soil_albedo_broadband`` collapses (vis, nir) with ``vis_fraction=0.5`` — a
legoESM convenience, NOT a CLM quantity (CLM keeps the two bands separate for the
two-stream canopy solver and has no single broadband soil albedo).  Round-off
pins (rel 1e-12) of the closed form + the CTSM table + coefficient/structure
canaries: ``tests/land/test_soil_albedo_faithful.py``.

References
----------
- Bonan, G. B. (1996): A land surface model (LSM v1.0). NCAR/TN-417+STR.
- Lawrence, P. J. and Chase, T. N. (2007): Representing a new MODIS consistent
  land surface in CLM 3.0. J. Geophys. Res., 112, G01023.
  https://doi.org/10.1029/2006JG000168
- Oleson et al. (2010): CLM4 Technical Note, Table 3.3 + eq. 3.52.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# CLM4 Table 3.3 — dry/saturated soil albedo per colour class (classes 1..20).
# Columns: dry_vis, dry_nir, sat_vis, sat_nir.  (Documented category lookup
# table, exempt from the no-literals rule — cf. the CLM5 PFT table.)
SOIL_COLOR_ALBEDO = (
    (0.36, 0.61, 0.25, 0.50),  # 1
    (0.34, 0.57, 0.23, 0.46),  # 2
    (0.32, 0.53, 0.21, 0.42),  # 3
    (0.31, 0.51, 0.20, 0.40),  # 4
    (0.30, 0.49, 0.19, 0.38),  # 5
    (0.29, 0.48, 0.18, 0.36),  # 6
    (0.28, 0.45, 0.17, 0.34),  # 7
    (0.27, 0.43, 0.16, 0.32),  # 8
    (0.26, 0.41, 0.15, 0.30),  # 9
    (0.25, 0.39, 0.14, 0.28),  # 10
    (0.24, 0.37, 0.13, 0.26),  # 11
    (0.23, 0.35, 0.12, 0.24),  # 12
    (0.22, 0.33, 0.11, 0.22),  # 13
    (0.20, 0.31, 0.10, 0.20),  # 14
    (0.18, 0.29, 0.09, 0.18),  # 15
    (0.16, 0.27, 0.08, 0.16),  # 16
    (0.14, 0.25, 0.07, 0.14),  # 17
    (0.12, 0.23, 0.06, 0.12),  # 18
    (0.10, 0.21, 0.05, 0.10),  # 19
    (0.08, 0.16, 0.04, 0.08),  # 20
)
N_SOIL_COLOR = len(SOIL_COLOR_ALBEDO)


class SoilAlbedoConfig(NamedTuple):
    """Coefficients of the CLM soil-wetness albedo relation (eq. 3.52)."""

    delta_intercept: float = 0.11      # dry-soil brightening at theta=0
    delta_slope: float = 0.40          # darkening per unit volumetric water
    vis_fraction: float = 0.5          # VIS share for the broadband helper


# CLM tech-note eq. 3.52 soil-wetness albedo coefficients (fixed published fit)
# plus a VIS/NIR broadband-split convention — not trained model closures.
__param_spec__ = {
    "SoilAlbedoConfig": {
        "scheme_key": "land.soil_albedo",
        "excluded": {
            "delta_intercept": "CLM tech note eq. 3.52 soil-wetness fit (fixed)",
            "delta_slope": "CLM tech note eq. 3.52 soil-wetness fit (fixed)",
            "vis_fraction": "convention: VIS/NIR broadband split",
        },
        "params": {},
    },
}


def _gather(color_class: jnp.ndarray) -> jnp.ndarray:
    """Look up the ``(..., 4)`` dry/sat VIS/NIR row for each colour class (1..20)."""
    table = jnp.asarray(SOIL_COLOR_ALBEDO)                       # (20, 4)
    idx = jnp.clip(jnp.asarray(color_class).astype(jnp.int32) - 1, 0, N_SOIL_COLOR - 1)
    return table[idx]                                           # (..., 4)


def soil_albedo_bounds(
    color_class: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(dry_vis, dry_nir, sat_vis, sat_nir)`` for each CLM colour class (1..20)."""
    row = _gather(color_class)
    return row[..., 0], row[..., 1], row[..., 2], row[..., 3]


def wet_soil_albedo(
    alb_dry: jnp.ndarray,
    alb_sat: jnp.ndarray,
    theta_top: jnp.ndarray,
    config: SoilAlbedoConfig = SoilAlbedoConfig(),
) -> jnp.ndarray:
    """One band of the CTSM soil albedo: ``min(alb_sat + inc, alb_dry)``.

    ``inc = max(delta_intercept - delta_slope * theta_top, 0)``.  With
    ``alb_dry == alb_sat`` (glacier, bare fallback) the result is exactly that
    value at any wetness.  Differentiable in ``theta_top``.
    """
    delta = jnp.maximum(config.delta_intercept - config.delta_slope * theta_top, 0.0)
    return jnp.minimum(alb_sat + delta, alb_dry)


def soil_albedo(
    color_class: jnp.ndarray,
    theta_top: jnp.ndarray,
    config: SoilAlbedoConfig = SoilAlbedoConfig(),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Background soil albedo ``(vis, nir)`` for a colour class + top-layer wetness.

    ``color_class`` is the integer CLM class (1..20; values are clipped into
    range).  ``theta_top`` is the volumetric water content of the top soil layer.
    Differentiable in ``theta_top``.
    """
    dry_vis, dry_nir, sat_vis, sat_nir = soil_albedo_bounds(color_class)
    return (wet_soil_albedo(dry_vis, sat_vis, theta_top, config),
            wet_soil_albedo(dry_nir, sat_nir, theta_top, config))


def rewet_soil_bands(land_params, theta_top: jnp.ndarray,
                     config: SoilAlbedoConfig = SoilAlbedoConfig()):
    """Recompute canopy soil band albedos ``ALB_VIS``/``ALB_NIR`` at ``theta_top``.

    CTSM evaluates the soil albedo from the CURRENT top-layer water at every
    albedo call.  Parameter sets built from soil colour carry the per-column
    dry/saturated bounds (``ALB_VIS_DRY`` ...); for them the bands follow the
    live soil water.  Parameter sets without bounds carry a PRESCRIBED albedo
    (e.g. an eddy-covariance site's measured albedo) and are returned unchanged.

    With bounds set, the incoming ``ALB_VIS``/``ALB_NIR`` are overwritten, so a
    gradient w.r.t. them is zero by construction; train the bounds instead.

    DEPARTURES from CTSM: ``theta_top`` is the land model's top-layer total
    volumetric water (CTSM converts liquid and ice with their own densities),
    and the top layer is whatever the configured soil grid makes it (CTSM's
    20-layer grid starts at 2 cm), so a shallow wetting pulse is damped here.
    """
    bounds = [getattr(land_params, f, None) for f in
              ("ALB_VIS_DRY", "ALB_VIS_SAT", "ALB_NIR_DRY", "ALB_NIR_SAT")]
    if all(b is None for b in bounds):
        return land_params
    if any(b is None for b in bounds):
        raise ValueError("soil albedo bounds must be all set or all None, got "
                         f"{[b is not None for b in bounds]}")
    return land_params._replace(
        ALB_VIS=wet_soil_albedo(land_params.ALB_VIS_DRY, land_params.ALB_VIS_SAT,
                                theta_top, config),
        ALB_NIR=wet_soil_albedo(land_params.ALB_NIR_DRY, land_params.ALB_NIR_SAT,
                                theta_top, config))


def soil_albedo_broadband(
    color_class: jnp.ndarray,
    theta_top: jnp.ndarray,
    config: SoilAlbedoConfig = SoilAlbedoConfig(),
) -> jnp.ndarray:
    """Single broadband background soil albedo = VIS/NIR weighted by ``vis_fraction``."""
    vis, nir = soil_albedo(color_class, theta_top, config)
    return config.vis_fraction * vis + (1.0 - config.vis_fraction) * nir
