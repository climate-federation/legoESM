"""legoESM-Veros recipe: ACC channel (Phase G.0c).

Builds the lat-lon C-grid equivalent of ``veros.setups.acc.acc.ACCSetup``:

- 30 × 42 × 15 grid at 2° horizontal resolution, ~1900 m deep
- Stretched z-coordinate (276 m at depth → 20 m at surface)
- ``y_origin = -40°``, ``x_origin = 0°``; lon wraps at 60°, lat to +44°
- Single-column western wall north of -20° latitude — re-entrant
  channel in the south, closed sub-tropical basin in the north
- Linear T(z) initial 0 (bottom) → 15 °C (surface), S = 35 uniform
- Surface wind stress: sinusoidal in the southern band, 1 - cos in
  the northern band
- T*  surface restoring (30-day timescale), zero in the tropics
- Veros canonical TKE + GM/Redi + linear bottom friction + implicit
  vertical viscosity, cos(lat) lateral viscosity

This module returns the legoESM analog with every scheme + parameter
pinned to the Veros source. Use it inside an
``override_constants(**VEROS_CONSTANTS)`` context so legoESM's
constants module also matches Veros.

Source of truth: ``veros/setups/acc/acc.py`` in the
team-ocean/veros repository.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import LatLonGrid, create_regional_latlon_grid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.eos import VerosNonlin2Config
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig, HarmonicConfig, LateralMixingConfig,
)
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import OceanZStarCoordinate, create_ocean_z_star


# ---------------------------------------------------------------------------
# Constants pulled verbatim from veros/setups/acc/acc.py
# ---------------------------------------------------------------------------

NX = 30
NY = 42
NZ = 15

DXT_DEG = 2.0
DYT_DEG = 2.0
X_ORIGIN_DEG = 0.0
Y_ORIGIN_DEG = -40.0

# Vertical layer-thickness profile (Veros: ddz[::-1] / 2.5).
# ddz published in Veros source:
_ACC_DDZ = np.array([
    50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0,
    390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0,
])
ACC_DZT = _ACC_DDZ[::-1] / 2.5   # shape (15,), dzt[0] = deepest = 276 m, dzt[-1] = surface = 20 m

DT_MOM_S = 4800.0
DT_TRACER_S = 43200.0       # = 86400 / 2

# Coriolis ω: Veros sets settings.omega — we leave that to legoESM's
# canonical constants.Omega, which the recipe-loader will override to
# 7.292115e-5 (Veros value) via the constants-override context manager.

# Lateral viscosity: A_h = (2 · degtom)^3 · 2e-11, with degtom = R_earth · π / 180.
def _degtom(r_earth: float = constants.R_earth) -> float:
    return r_earth * float(np.pi) / 180.0

def acc_A_h(r_earth: float = constants.R_earth) -> float:
    """Veros: ``A_h = (2 * degtom) ** 3 * 2e-11``. Units: m^4/s² × m = m²/s
    after the cos²(lat) scaling and ∇^4 stencil; matches MOM6 / MITgcm
    biharmonic-equivalent harmonic-viscosity convention used in ACC.

    ``r_earth`` defaults to legoESM's; build_acc_model_config passes the Veros
    value so the grid/A_h are pinned via config rather than the
    override_constants monkey-patch (G-C4)."""
    return (2.0 * _degtom(r_earth)) ** 3 * 2.0e-11

# Bottom drag: linear, ``r_bot = 1e-5``
R_BOT = 1.0e-5

# Veros TKE knobs (verbatim from ACCSetup)
ACC_TKE_CONFIG = TKEConfig(
    c_k=0.1,
    c_eps=0.7,
    alpha_tke=30.0,
    mxl_min=1.0e-8,
    tke_mxl_choice=2,
    kappaM_min=2.0e-4,
    kappaH_min=2.0e-5,
    enable_kappaH_profile=True,
)

# Veros GM/Redi knobs (verbatim from ACCSetup)
ACC_GM_REDI_CONFIG = GMRediConfig(
    kappa_GM=1000.0,
    kappa_Redi=1000.0,
    S_max=0.01,                  # ↔ iso_slopec
    taper_width_frac=0.5,        # = iso_dslope / iso_slopec = 0.005 / 0.01
)

# Surface restoring timescale
T_RESTORING_DAYS = 30.0


# ---------------------------------------------------------------------------
# Grid / vertical coordinate / bathymetry
# ---------------------------------------------------------------------------


def build_acc_grid() -> LatLonGrid:
    """Build legoESM lat-lon C-grid whose interior centres coincide with
    Veros ACC's xt/yt EXACTLY.

    Veros's u-centred grid (verified from ``veros/core/numerics.py`` +
    a live ``ACCSetup``): the first interior centre sits half a cell below
    the origin, so
        xt = [-1, 1, 3, ..., 57]   (= x_origin - dx/2 + i·dx)
        yt = [-41, -39, ..., 41]   (= y_origin - dy/2 + j·dy)
    ``create_regional_latlon_grid`` places interior cell ``i`` (i=1..n) at
    ``lower + (i-0.5)·d`` and adds N/S wall rows at index 0 and -1. To make
    the interior centres land ON Veros's xt/yt — so the analytic kbot lands
    Veros's land cells identically AND the per-process tendency comparison
    is at the same physical points — offset the bounds accordingly. Getting
    this wrong (a half-cell lon offset + a one-row lat offset) was the Phase
    G tier-2 residual: density still matched (point-wise in T,S) but the
    western wall mis-aligned and momentum metrics (Coriolis f(lat)) were off.
    """
    lon_west = X_ORIGIN_DEG - DXT_DEG / 2.0        # -1  -> centres -1,1,...,57
    lon_east = lon_west + NX * DXT_DEG             # 59
    lat_south = Y_ORIGIN_DEG - DYT_DEG             # -42 -> interior -41,...,41
    lat_north = lat_south + NY * DYT_DEG           # +42
    grid, _wall_mask = create_regional_latlon_grid(
        n_lat=NY, n_lon=NX,
        lat_south=lat_south, lat_north=lat_north,
        lon_west=lon_west, lon_east=lon_east,
        # Pin Earth radius / rotation to Veros's values via config (so the grid
        # metrics + Coriolis are correct without relying on override_constants).
        radius=VEROS_CONSTANTS_CONFIG.R_earth,
        omega=VEROS_CONSTANTS_CONFIG.Omega,
        periodic_x=True,
    )
    return grid


def build_acc_z_coord() -> OceanZStarCoordinate:
    """Veros ACC vertical coordinate, built from the exact dzt array
    in Veros's ACC source — ``ddz[::-1] / 2.5`` with ddz published as
    ``[50, 70, ..., 690]``.

    legoESM convention is k=0 surface, k=nlev-1 deepest (opposite of
    Veros which has k=0 deepest). We reverse the Veros dzt array
    when populating the legoESM z_coord so each interior k-index
    refers to the same physical layer in both models.
    """
    # legoESM dz_ref: surface (k=0) → bottom (k=nlev-1) = ddz / 2.5 directly.
    dz_ref = _ACC_DDZ / 2.5   # shape (15,), [20, 28, 40, ..., 276]
    dz_ref = jnp.asarray(dz_ref, dtype=jnp.float64)
    H_max = float(jnp.sum(dz_ref))   # 1900 m exactly

    # z_half_ref: interfaces; surface (k=0) at z=0, bottom (k=nlev) at z=-H_max
    z_half_ref = jnp.concatenate([
        jnp.zeros(1, dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    # z_full_ref: cell centres
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
    # dz_half_ref: distance between adjacent cell centres
    dz_half_ref = jnp.abs(z_full_ref[:-1] - z_full_ref[1:])

    return OceanZStarCoordinate(
        n_levels=NZ,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


def build_acc_land_mask(grid: LatLonGrid) -> jnp.ndarray:
    """Veros: ``kbot = (x > 1.0 | y < -20)``.

    In legoESM-speak: a single-column western wall north of -20°. The
    lon=0 column is land at y >= -20 and wet south of y=-20 (so the
    Drake-Passage band is re-entrant).
    """
    lat_deg = np.degrees(np.asarray(grid.lat))  # cell-centre latitudes (n_lat,)
    lon_deg = np.degrees(np.asarray(grid.lon))  # cell-centre longitudes (n_lon,)
    # Wet = (lon > 1.0) OR (lat < -20). With lon[0] = 1.0 + 0 = 1.0 (Veros's
    # cell-centre x for column 0 is x_origin + dx/2 = 1.0), this includes
    # x[0]=1.0 — but the Veros condition is *strict* (x > 1.0), so x=1.0
    # is land. We match that.
    wet_lon = lon_deg > 1.0                   # (n_lon,)
    wet_lat = lat_deg < -20.0                 # (n_lat,)
    wet = wet_lon[None, :] | wet_lat[:, None]  # (n_lat, n_lon)
    # build_acc_grid places the interior centres ON Veros's xt/yt, so the
    # kbot above lands Veros's 2-wide western wall (xt=-1,1 <= 1 -> land for
    # lat>=-20) exactly. create_regional_latlon_grid still adds non-physical
    # N/S wall rows at index 0 and -1 (the bridge zero-fills them -> T=S=0 ->
    # rho ~ 997); they are outside Veros's 42-row domain, so mark them LAND.
    # Leaving any of these mis-handled was the Phase G tier-2 density residual.
    wet[0, :] = False
    wet[-1, :] = False
    return jnp.asarray(wet.astype(np.float64))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def build_acc_state(grid: LatLonGrid,
                     z_coord: OceanZStarCoordinate,
                     ) -> LatLonCGridOceanState:
    """Veros ACC initial conditions:

    - T(z) linear from 0 at bottom to 15 °C at surface
    - S uniform 35 PSU
    - u, v, eta zero
    - Bathymetry depth = full H_max where wet, zero where land
    """
    H_max = float(np.sum(ACC_DZT))
    land_mask = build_acc_land_mask(grid)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=15.0, T_deep=0.0,
        S_uniform=35.0, H_max=H_max,
        land_mask_override=land_mask,
    )
    return state


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


class ACCRecipe(NamedTuple):
    """All the pieces needed to run legoESM-Veros ACC:

    - ``model_config``: LatLonCGridOceanConfig matching Veros ACC settings.
    - ``physics_config``: OceanPhysicsConfig matching Veros ACC physics.
    - ``grid``: lat-lon C-grid at Veros ACC's native resolution.
    - ``z_coord``: stretched-z vertical coordinate.
    - ``land_mask``: cell-centre wet mask matching Veros's kbot pattern.
    - ``initial_state``: ocean state ready for the first step.

    Use inside ``override_constants(**VEROS_CONSTANTS)`` so legoESM's
    physical constants also match Veros.
    """
    model_config: LatLonCGridOceanConfig
    physics_config: OceanPhysicsConfig
    grid: LatLonGrid
    z_coord: OceanZStarCoordinate
    land_mask: jnp.ndarray
    initial_state: LatLonCGridOceanState


def build_acc_physics_config() -> OceanPhysicsConfig:
    """Veros ACC physics: TKE + GM/Redi + linear bottom drag (via model
    config), implicit vertical viscosity. EKE / IDEMIX are NOT mapped
    — EKE is a Veros-only closure (no legoESM equivalent yet) and
    IDEMIX is disabled in the ACC adapter."""
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=ACC_TKE_CONFIG),
        # GM/Redi on the lat-lon C-grid is a DYNAMICS-level process applied via
        # the top-level config.gm_redi field (see build_acc_model_config); the
        # physics-pathway lateral-mixing factory is cubed-sphere-only and raises
        # on lat-lon. So lateral_mixing is "none" here. (Setting GM/Redi ONLY in
        # physics.lateral_mixing — as before — left it INACTIVE for the ACC
        # recipe: config.gm_redi defaults None so the model skipped it, and the
        # tendency probe never invokes the physics pipeline.)
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="prescribed"),
        bottom_drag=BottomDragConfig(scheme="none"),   # see model config bottom_drag_r
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def build_acc_model_config() -> LatLonCGridOceanConfig:
    """Veros ACC dynamics: harmonic lateral viscosity with cos(lat)
    scaling, linear bottom drag, implicit vertical viscosity,
    Veros's nonlin3 EOS."""
    return LatLonCGridOceanConfig(
        # All physical constants pinned to Veros via config (G-C4): g/rho_0 are
        # read by the PE core, the ConstantsConfig by the de-mirrored physics,
        # and R_earth feeds acc_A_h — so the recipe no longer needs the
        # override_constants monkey-patch for these.
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=acc_A_h(VEROS_CONSTANTS_CONFIG.R_earth),
        A_h_lat_scaling=True,
        A_h_cos_power=1,
        bottom_drag_r=R_BOT,
        # ``eq_of_state_type=3`` in Veros dispatches to nonlinear_eq2.py
        # (Vallis 2008) — NOT nonlinear_eq3.py despite the file name.
        # Verified against ``veros/core/density/get_rho.py``.
        eos="veros_nonlin2",
        implicit_vertical_mixing=True,
        # GM/Redi is a TOP-LEVEL (dynamics) field on the lat-lon C-grid — this
        # is what the model actually reads (ocean_model_latlon_cgrid.py:998).
        # Setting it only in physics.lateral_mixing left GM/Redi inactive.
        gm_redi=ACC_GM_REDI_CONFIG,
        physics=build_acc_physics_config(),
    )


def build_acc_recipe() -> ACCRecipe:
    """One-stop constructor. Use as::

        from legoesm.ocean.fidelity.recipe_constants import (
            VEROS_CONSTANTS, override_constants,
        )
        from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe

        with override_constants(**VEROS_CONSTANTS):
            recipe = build_acc_recipe()
            # ... run / probe with recipe.initial_state, .model_config, etc.
    """
    grid = build_acc_grid()
    z_coord = build_acc_z_coord()
    land_mask = build_acc_land_mask(grid)
    initial_state = build_acc_state(grid, z_coord)
    return ACCRecipe(
        model_config=build_acc_model_config(),
        physics_config=build_acc_physics_config(),
        grid=grid,
        z_coord=z_coord,
        land_mask=land_mask,
        initial_state=initial_state,
    )


__all__ = (
    "ACC_DZT",
    "ACC_GM_REDI_CONFIG",
    "ACC_TKE_CONFIG",
    "ACCRecipe",
    "DT_MOM_S",
    "DT_TRACER_S",
    "NX",
    "NY",
    "NZ",
    "R_BOT",
    "T_RESTORING_DAYS",
    "X_ORIGIN_DEG",
    "Y_ORIGIN_DEG",
    "acc_A_h",
    "build_acc_grid",
    "build_acc_land_mask",
    "build_acc_model_config",
    "build_acc_physics_config",
    "build_acc_recipe",
    "build_acc_state",
    "build_acc_z_coord",
)
