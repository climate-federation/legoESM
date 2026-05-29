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
pinned to the Veros source. All physical constants (g, rho_0, Omega,
R_earth, c_sw, and the derived A_h) are pinned through config
(``ConstantsConfig`` / ``VEROS_CONSTANTS_CONFIG``), so no monkey-patch
of legoESM's constants module is needed (G-C4).

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
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.state import (
    LatLonCGridOceanConfig, LatLonCGridOceanState, OceanSurfaceForcing,
)
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
    value so the grid/A_h are pinned via config (G-C4)."""
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
_SECONDS_PER_DAY = 86400.0

# Veros ACC surface-forcing profile parameters (verbatim from
# veros/setups/acc/acc.py:126-134). Band edges in degrees latitude.
_ACC_TAUX_AMP = 0.1          # wind-stress amplitude [N/m^2]
_ACC_TSTAR_AMP = 15.0        # T* amplitude [degC]
_ACC_WIND_LAT_S = -20.0      # southern edge: sin westerly band below this
_ACC_WIND_LAT_N = 10.0       # northern edge: (1-cos) band above this
_ACC_TSTAR_LAT_S = -20.0     # T* ramps down south of this
_ACC_TSTAR_LAT_N = 20.0      # T* ramps down north of this
# Veros ACC grid latitude extents INCLUDING its 2 ghost cells each side — the
# forcing formulas reference global_min/max(yt) / (yu). Derived from the recipe
# grid constants and VERIFIED against a live ACCSetup grid (yt in [-45, 45]
# step 2 -> 46 cells = NY+4; yu = yt + dy/2 in [-44, 46]).
_VEROS_YT_MIN = Y_ORIGIN_DEG - 2.5 * DYT_DEG            # -45.0
_VEROS_YT_MAX = _VEROS_YT_MIN + (NY + 3) * DYT_DEG      # +45.0  (NY+4 cells)
_VEROS_YU_MIN = _VEROS_YT_MIN + 0.5 * DYT_DEG           # -44.0
_VEROS_YU_MAX = _VEROS_YT_MAX + 0.5 * DYT_DEG           # +46.0
# Salinity is NOT restored in Veros ACC (T-only heat-flux forcing); a huge
# timescale makes the restoring tendency negligibly small (effectively off).
_ACC_NO_SALT_RESTORE_TAU_S = 1.0e30


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
        # metrics + Coriolis are correct, pinned purely through config).
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
# Surface forcing (free-run): Veros ACC wind stress + T* restoring
# ---------------------------------------------------------------------------


def _acc_taux_profile(lat_deg: np.ndarray) -> np.ndarray:
    """Veros ACC zonal wind stress ``taux`` [N/m^2] as a function of cell-centre
    latitude, reproducing ``veros/setups/acc/acc.py:126-129`` exactly. The band
    is selected by the T-point latitude ``yt`` (= ``lat_deg``) while the value
    uses the v-point latitude ``yu = yt + dy/2`` (Veros's discretisation). This
    is the OCEAN-SIDE stress (eastward-positive), applied with a ``+`` sign in
    Veros (``du += surface_taux/(rho_0·dz)``)."""
    yt = np.asarray(lat_deg, dtype=np.float64)
    yu = yt + 0.5 * DYT_DEG                       # v-point lat, as Veros uses
    taux = np.zeros_like(yt)
    south = yt < _ACC_WIND_LAT_S
    north = yt > _ACC_WIND_LAT_N
    taux = np.where(
        south,
        _ACC_TAUX_AMP * np.sin(
            np.pi * (yu - _VEROS_YU_MIN) / (_ACC_WIND_LAT_S - _VEROS_YT_MIN)),
        taux)
    taux = np.where(
        north,
        _ACC_TAUX_AMP * (1.0 - np.cos(
            2.0 * np.pi * (yu - _ACC_WIND_LAT_N) / (_VEROS_YU_MAX - _ACC_WIND_LAT_N))),
        taux)
    return taux


def build_acc_wind_stress(grid: LatLonGrid) -> OceanSurfaceForcing:
    """Veros ACC surface wind stress as an :class:`OceanSurfaceForcing`.

    Veros applies ``surface_taux`` (the ocean-side, eastward-positive stress)
    with a ``+`` sign. legoESM's external surface-forcing path treats ``tau_x``
    as the ATMOSPHERIC stress and flips it (``-tau_x``) to get the ocean
    reaction (``ocean_pe_latlon_cgrid.py:1922``). So we pass ``tau_x = -taux``
    here, which after legoESM's internal flip reproduces Veros's ``+taux`` —
    i.e. westerlies (taux>0) accelerate the ACC eastward. Verified by the sign
    of the spun-up channel jet (must be eastward)."""
    lat_deg = np.degrees(np.asarray(grid.lat))    # cell-centre lat (n_lat,)
    taux = _acc_taux_profile(lat_deg)             # ocean-side stress (n_lat,)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    tau_x = jnp.asarray(
        np.broadcast_to(-taux[:, None], (n_lat, n_lon)))   # negated: see docstring
    tau_y = jnp.zeros((n_lat, n_lon), dtype=tau_x.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def build_acc_t_star(grid: LatLonGrid) -> jnp.ndarray:
    """Veros ACC surface restoring target ``t_star`` [degC] (2-D, broadcast over
    longitude), reproducing ``veros/setups/acc/acc.py:132-134``: 15 degC in the
    band [-20, 20], ramping linearly to 0 at the meridional walls."""
    lat = np.degrees(np.asarray(grid.lat))
    ts = np.full_like(lat, _ACC_TSTAR_AMP)
    south = lat < _ACC_TSTAR_LAT_S
    north = lat > _ACC_TSTAR_LAT_N
    ts = np.where(
        south,
        _ACC_TSTAR_AMP * (lat - _VEROS_YT_MIN) / (_ACC_TSTAR_LAT_S - _VEROS_YT_MIN),
        ts)
    ts = np.where(
        north,
        _ACC_TSTAR_AMP * (1.0 - (lat - _ACC_TSTAR_LAT_N) / (_VEROS_YT_MAX - _ACC_TSTAR_LAT_N)),
        ts)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    return jnp.asarray(np.broadcast_to(ts[:, None], (n_lat, n_lon)))


def build_acc_restoring_config(grid: LatLonGrid) -> RestoringConfig:
    """Veros ACC surface heat-flux forcing as a :class:`RestoringConfig`.

    Veros: ``forc_temp_surface = (dz_surf/(30 d)) · (t_star - T_surf)`` so the
    surface tendency is ``dT/dt = (t_star - T)/(30 d)`` — the layer thickness
    cancels, matching legoESM's ``restoring_surface_forcing`` form
    ``-(T - T*)/tau_T`` exactly with ``tau_T = 30 d``. Salinity is NOT restored
    in Veros ACC, so ``tau_S`` is huge (effectively off); ``S_star`` defaults to
    35 PSU (a no-op given the huge timescale)."""
    return RestoringConfig(
        tau_T=T_RESTORING_DAYS * _SECONDS_PER_DAY,
        tau_S=_ACC_NO_SALT_RESTORE_TAU_S,
        T_star_array=build_acc_t_star(grid),
        implicit=False,    # Veros uses explicit forward-Euler restoring; dt << 2·tau_T
    )


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

    All Veros physical constants are pinned through ``model_config.constants``
    (G-C4) — no monkey-patch of legoESM's constants module is needed.
    """
    model_config: LatLonCGridOceanConfig
    physics_config: OceanPhysicsConfig
    grid: LatLonGrid
    z_coord: OceanZStarCoordinate
    land_mask: jnp.ndarray
    initial_state: LatLonCGridOceanState
    # Free-run wind stress (OceanSurfaceForcing) — passed to ``model.step`` as
    # the ``surface_forcing`` arg. ``None`` unless built with surface forcing
    # (the frozen-state tendency probe does not use it).
    wind_forcing: object = None


def build_acc_physics_config(grid: LatLonGrid | None = None, *,
                             with_surface_forcing: bool = False,
                             ) -> OceanPhysicsConfig:
    """Veros ACC physics: TKE + GM/Redi + linear bottom drag (via model
    config), implicit vertical viscosity. IDEMIX is disabled in the ACC
    adapter (``enable_idemix=False`` in Veros) so it is not mapped.

    EKE: legoESM now HAS the Eden-Greatbatch prognostic-EKE closure
    (``lateral_mixing/eke.py`` + ``GMRediConfig.eke``), and its kappa_GM
    formula ``c_k·L·√E`` + parameters (c_k=0.4, c_eps=0.5, k_max=1e4,
    lmin=100, superbee advection) were verified against Veros's ``K_gm``
    to machine precision (corr=1.0 on the developed ACC state, given
    Veros's own ``eke``+``eke_len``; see the strategy doc §8 EKE ledger,
    gate E9). EKE is NOT yet flipped on in the recipe (``GMRediConfig.eke``
    stays None) because legoESM's mixing length ``L`` (Visbeck first-
    baroclinic Rossby radius, ~200 km here) does not yet match Veros's
    ``eke_len = max(lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines))``
    (~8 km; the eddy-energy-dependent Rhines limiting is missing) — using
    it now would make the prognostic kappa_GM ~25× too large. ADOPTION is
    deferred behind the documented ``eke_len`` mixing-length variant
    (strategy doc §8, "extend L" follow-up).

    Set ``with_surface_forcing=True`` (free-run harness) to activate Veros ACC's
    T* surface restoring via the physics pipeline; ``grid`` is then required (for
    the latitude-dependent T* target). Default ``False`` keeps the prior
    prescribed-zero forcing so the frozen-state tendency probe + committed tier-2
    report are unchanged. The wind stress is applied SEPARATELY via the
    ``OceanSurfaceForcing`` argument to ``model.step`` (see
    :func:`build_acc_wind_stress`), not through this config — the two paths are
    disjoint (restoring -> dT, wind -> du/dv)."""
    if with_surface_forcing:
        if grid is None:
            grid = build_acc_grid()
        surface_forcing = SurfaceForcingConfig(
            scheme="restoring", restoring=build_acc_restoring_config(grid))
    else:
        surface_forcing = SurfaceForcingConfig(scheme="prescribed")
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
        surface_forcing=surface_forcing,
        bottom_drag=BottomDragConfig(scheme="none"),   # see model config bottom_drag_r
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def build_acc_model_config(grid: LatLonGrid | None = None, *,
                           with_surface_forcing: bool = False,
                           ) -> LatLonCGridOceanConfig:
    """Veros ACC dynamics: harmonic lateral viscosity with cos(lat)
    scaling, linear bottom drag, implicit vertical viscosity,
    Veros's nonlin3 EOS.

    ``with_surface_forcing`` (free-run) threads through to the physics config to
    activate T* restoring; default ``False`` is the frozen-state-probe config."""
    return LatLonCGridOceanConfig(
        # All physical constants pinned to Veros via config (G-C4): g/rho_0 are
        # read by the PE core, the ConstantsConfig by the de-mirrored physics,
        # and R_earth feeds acc_A_h — pinned purely through config.
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=acc_A_h(VEROS_CONSTANTS_CONFIG.R_earth),
        A_h_lat_scaling=True,
        A_h_cos_power=1,
        # Veros ACC momentum advection is flux-form (core/momentum.py), centered
        # 2nd-order — so the recipe selects flux_form/centered for true
        # apples-to-apples (doctrine rule H). Validated: vs the developed-flow
        # Veros snapshot this lifts du_adv corr 0.584->0.654 and dv_adv
        # -0.336->0.715 (vector-invariant was anti-correlated). See the §8 ledger.
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
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
        physics=build_acc_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
    )


def build_acc_recipe(*, with_surface_forcing: bool = False) -> ACCRecipe:
    """One-stop constructor. Use as::

        from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe

        recipe = build_acc_recipe()
        # ... run / probe with recipe.initial_state, .model_config, etc.

    All Veros constants (g, rho_0, Omega, R_earth, c_sw, derived A_h) are
    pinned through config — ``build_acc_model_config`` sets ``constants=
    VEROS_CONSTANTS_CONFIG`` and the grid radius/omega — so no monkey-patch
    or ``override_constants`` context is required (G-C4).

    ``with_surface_forcing=True`` (free-run harness) activates Veros ACC's
    surface forcing: T* restoring through ``physics_config`` and the wind stress
    as ``recipe.wind_forcing`` (pass to ``model.step(..., surface_forcing=
    recipe.wind_forcing)``). Default ``False`` is the frozen-state-probe recipe
    (no forcing), so the committed tier-2 tendency comparison is unchanged.
    """
    grid = build_acc_grid()
    z_coord = build_acc_z_coord()
    land_mask = build_acc_land_mask(grid)
    initial_state = build_acc_state(grid, z_coord)
    wind_forcing = build_acc_wind_stress(grid) if with_surface_forcing else None
    return ACCRecipe(
        model_config=build_acc_model_config(
            grid, with_surface_forcing=with_surface_forcing),
        physics_config=build_acc_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
        grid=grid,
        z_coord=z_coord,
        land_mask=land_mask,
        initial_state=initial_state,
        wind_forcing=wind_forcing,
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
    "build_acc_restoring_config",
    "build_acc_state",
    "build_acc_t_star",
    "build_acc_wind_stress",
    "build_acc_z_coord",
)
