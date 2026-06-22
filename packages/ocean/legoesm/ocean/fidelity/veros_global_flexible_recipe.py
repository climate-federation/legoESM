"""legoESM-Veros recipe: global_flexible (the STRETCHED-GRID global transfer).

``veros.setups.global_flexible.global_flexible.GlobalFlexibleResolutionSetup``
is Veros's ARBITRARY-RESOLUTION global world ocean: a Vinokur-stretched
meridional grid (equator-refined, factor 0.5), Vinokur-stretched z levels,
ETOPO5 topography with marginal-sea removal, monthly climatological forcing
(MIT-grid wind stress, qnet + qnec·(sst−T) heat feedback, 30-day SSS
restoring, simple ice mask) and — new vs global_4deg — a PENETRATIVE
SHORTWAVE channel (``qsol`` deposited through the column with the two-band
Jerlov profile, ice-gated).  This recipe maps every Veros setting onto
EXISTING canonical legoESM config options (SCOPING_three_setups.md §B table;
recipe = pure config selecting canonical blocks, NO bespoke numerics).

Comparison resolution (the oracle-cost mitigation, scoping §E): the stock
360×160×60 oracle is CPU-infeasible (~16 wall-days/model-yr), and the class
is EXPLICITLY rescalable — the setup builds its grid from ``settings.nx/ny/
nz`` and re-interpolates all forcing.  The banked oracle and this recipe run
the class at:

  ``NX, NY = 90, 40``  (= stock/4, the global_4deg cost class; uniform
      dx = 4°, equatorial dy = 0.5·160/NY = 2°)
  ``NZ = 20``  (NOT 15 = stock/4: the setup's own ``Vinokur(nz, 5400, 10)``
      + ``u_centered_grid`` recursion is DEGENERATE at nz=15 — the surface
      centre lands at −16.2 m BELOW the next centre at −10.0 m, gsw NaNs,
      Veros itself diverges at iteration 6 (measured) — and at nz=18 the
      surface centre pops ABOVE z=0 (+0.44 m).  nz=20 is the smallest level
      count with a monotone zt and all dzw > 0 (min dzw 10.5 m, surface cell
      12.1 m, deepest 766 m).  ``build_global_flexible_z_coord`` guards this
      LOUDLY for any other nz.)
  ``dt_mom = 1800 s``  (NOT the CFL-consistent 900×4 = 3600: |f|·dt_mom at
      the most poleward wet row (the Antarctic wet rows (-77.3°, |f|=1.42e-4)) would be ~0.5, ON the explicit-AB2
      Coriolis margin; 1800 gives ~0.25 — the proven global_4deg value.)
  ``dt_tracer = 14400 s`` → ``dt_mom_ratio = 8``  (budget-driven judgment,
      measured: the 10-yr oracle at the CFL-consistent dt_tracer=3600 costs
      11.6 h CPU / 6.5 h Veros-JAX-GPU; 14400 on GPU is ~1.6 h and sits 6×
      BELOW the global_4deg-proven dt_tracer=86400 on this exact 90×40 grid.
      The stock setup runs SYNCHRONOUS dt_mom=dt_tracer=900; the async pair
      is native Veros machinery (global_4deg: 1800/86400) and legoESM's
      faithful ``dt_mom_ratio``.  Both sides of the comparison use the SAME
      1800/14400 pair — documented loudly as the one structural deviation
      from the stock dt.)

Config deltas vs the matched global_4deg recipe (source of truth:
``veros/setups/global_flexible/global_flexible.py`` + veros/settings.py
defaults; scoping §B):

  1. **Stretched meridional grid** (the EXT-F1 surface):
     ``dyt = get_vinokur_grid_steps(ny, 160, 0.5·160/ny, two_sided=True)``
     → :func:`legoesm.grids.latlon.create_stretched_latlon_grid` with the
     TRANSCRIBED Vinokur steps (:func:`veros_vinokur_grid_steps`, anchored on
     the Veros docstring values + live-oracle dyt).  Veros places
     ``yu[2] = y_origin`` ⇒ ``lat_south = y_origin − dyt[0]``.
  2. **Vinokur z**: ``dzt = Vinokur(nz, 5400, 10, refine_towards='lower')``
     (Veros k=0 deepest); legoESM ``dz_ref = dzt[::-1]``; centres via the
     shared ``veros_u_centered_z_centres`` (Veros dzw metric), with the
     monotonicity guard above.
  3. GM/Redi: ``K_iso_steep = 50`` (4deg: 1000), ``iso_dslope = iso_slopec
     = 0.005`` → ``S_max = 5e-3``, ``taper_width_frac = 1.0`` (4deg:
     1e-3/4.0).
  4. **EKE FLIP BACK: ``isopycnal_diffusion = True``** (the setup sets
     ``enable_eke_isopycnal_diffusion = True``, like the ACC and unlike
     global_4deg's settings-default False) ⇒ the Redi tracer diffusivity
     follows the prognostic GM coefficient ``K_iso = K_gm``.
  5. Lateral viscosity ``A_h = 5e4`` (a setup LITERAL — not the (dx·degtom)³
     formula the 2°/4° setups use), same cos¹(lat) scaling.
  6. **Penetrative shortwave** (EXT-F2, the new q_solar channel):
     ``FluxFeedbackConfig.penetrative_shortwave=True`` with
     ``shortwave_water_type="I"`` (≡ the setup literals R=0.58, ζ1=0.35 m,
     ζ2=23.0 m).  Heat-ownership contract: the harness passes
     ``q_prescribed = qnet − qsol`` (non-solar remainder) and
     ``q_solar = qsol``; legoESM's I(0)=1 full-column deposit is
     cell-by-cell identical to Veros's pen(0)=0 zero-column-sum
     redistribution on the solar-inclusive qnet (see
     ``OceanSurfaceForcing.q_solar``).  Ice gating (Veros ``ice[..., None]``
     zeroes the 3-D solar source) is inside the scheme.
  7. Topography: ETOPO5 + gaussian smoothing + nearest interp + kbot =
     1 + argmin|z−zt| (NOT the 4deg count rule) + **marginal-sea removal**
     (binary dilate → fill holes → erode), replicated verbatim including the
     Veros ghost-row/cyclic-boundary handling
     (:func:`replicate_veros_kbot_flexible`).
  8. Wind stress on the MIT grid: the Veros kernel shifts taux one cell in x
     and tauy one cell in y when loading ``surface_taux/y``
     (:func:`veros_mit_tau_shift`).  Same half-cell u-face/T-point mimicry
     note as the 4deg harness (climate-negligible).
  9. Everything else — TKE block (incl. superbee advection, mxl_choice=2),
     prognostic 3-D EKE sources, eos="veros_gsw", rigid lid + faithful AB2
     stack, zero bottom drag, cp_0 kernel literal 3991.86795711963,
     30-day SSS restore, ice mask, 360-day forcing year — is IDENTICAL to
     the matched global_4deg recipe and is imported/reused from it (THE
     RULE: factored, not copied).

``enable_eke_diss_surfbot=True/0.2`` and the idemix flags in the setup are
INERT with ``enable_idemix=False`` (4deg-verified) — unmapped.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.grids.latlon import LatLonGrid, create_stretched_latlon_grid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG

# Shared canonical blocks from the matched recipes (factored, not copied):
# the ACCRecipe container, the global_4deg TKE/EKE/GM blocks this setup
# repeats, the Veros kernel cp_0 literal, the 360-day-year month weights and
# the u_centered vertical-centre construction.
from legoesm.ocean.fidelity.veros_acc_recipe import ACCRecipe
from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
    GLOBAL4_EKE_CONFIG,
    GLOBAL4_GM_REDI_CONFIG,
    GLOBAL4_TKE_CONFIG,
    T_REST_S,
    VEROS_GLOBAL4_CP0,
    get_periodic_interval_weights,
)

# Shared global-recipe builders (state seeding, GM/EKE delta, area weights).
from legoesm.ocean.fidelity.veros_global_common import (
    build_veros_global_state,
    gm_redi_eke_isopycnal_on,
    veros_area_t_generic,
)

# Shape-generic layout bridges, shared via fidelity.veros_layout (the 1deg
# recipe's DEDUP NOTE); re-exported under the recipe's ``_flex`` names.
from legoesm.ocean.fidelity.veros_layout import (
    veros_xy_to_legoesm as veros_xy_to_legoesm_flex,
)
from legoesm.ocean.fidelity.veros_layout import (
    veros_xyz_to_legoesm as veros_xyz_to_legoesm_flex,
)
from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres
from legoesm.ocean.fidelity.veros_stepping import veros_faithful_stepping
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import (
    FluxFeedbackConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    create_partial_cell_coordinate,
)

__all__ = (
    "DT_MOM_RATIO",
    "DT_MOM_S",
    "DT_TRACER_S",
    "EQ_SPACING_FACTOR",
    "GLOBAL_FLEX_A_H",
    "GLOBAL_FLEX_EKE_CONFIG",
    "GLOBAL_FLEX_GM_REDI_CONFIG",
    "GLOBAL_FLEX_TKE_CONFIG",
    "LAT_SPAN_DEG",
    "MAX_DEPTH_M",
    "MIN_DEPTH_M",
    "NX", "NY", "NZ",
    "X_ORIGIN_DEG",
    "Y_ORIGIN_DEG",
    "build_global_flexible_grid",
    "build_global_flexible_model_config",
    "build_global_flexible_physics_config",
    "build_global_flexible_recipe",
    "build_global_flexible_state",
    "build_global_flexible_z_coord",
    "get_periodic_interval_weights",
    "global_flexible_dyt_deg",
    "global_flexible_dzt_veros",
    "kbot_to_mask_and_h_bathy_flexible",
    "prepare_global_flexible_topography",
    "replicate_veros_kbot_flexible",
    "veros_area_t_flexible",
    "veros_fill_holes",
    "veros_full_axes",
    "veros_interpolate",
    "veros_mit_tau_shift",
    "veros_vinokur_grid_steps",
    "veros_xy_to_legoesm_flex",
    "veros_xyz_to_legoesm_flex",
)


# ---------------------------------------------------------------------------
# Constants pulled verbatim from veros/setups/global_flexible/global_flexible.py
# (class attributes + set_parameter), with the documented rescale
# ---------------------------------------------------------------------------

# Comparison resolution (rescaled; stock is 360, 160, 60 — see module doc).
NX = 90
NY = 40
NZ = 20

X_ORIGIN_DEG = 90.0
Y_ORIGIN_DEG = -80.0
LAT_SPAN_DEG = 160.0                  # the set_grid Vinokur total
EQ_SPACING_FACTOR = 0.5               # equatorial_grid_spacing_factor
MIN_DEPTH_M = 10.0                    # class attr min_depth
MAX_DEPTH_M = 5400.0                  # class attr max_depth

# Documented rescaled dt pair (module doc); stock is dt_mom=dt_tracer=900.
DT_MOM_S = 1800.0
DT_TRACER_S = 14400.0
DT_MOM_RATIO = DT_TRACER_S / DT_MOM_S    # 8.0

GLOBAL_FLEX_A_H = 5.0e4               # setup literal A_h [m²/s]

# Veros ETOPO5 land threshold: smoothed topo reset to 0 where raw >= -1 m.
_TOPO_LAND_THRESHOLD_M = -1.0


# ---------------------------------------------------------------------------
# Veros tools transcriptions (pure; bit-anchored by unit tests + the live
# oracle dump).  Transcribed so the recipe/harness do not import veros.
# ---------------------------------------------------------------------------


def _veros_sinhc_inverse(y: float) -> float:
    """Approximate inverse of sinh(y)/y (veros/tools/setup.py
    ``approximate_sinhc_inverse``, verbatim)."""
    if y < 2.7829681:
        ybar = y - 1.0
        inv = np.sqrt(6 * ybar) * (
            1
            - 0.15 * ybar
            + 0.057321429 * ybar**2
            - 0.024907295 * ybar**3
            + 0.0077424461 * ybar**4
            - 0.0010794123 * ybar**5
        )
    else:
        v = np.log(y)
        w = 1.0 / y - 0.028527431
        inv = (
            v
            + (1 + 1.0 / v) * np.log(2 * v)
            - 0.02041793
            + 0.24902722 * w
            + 1.9496443 * w**2
            - 2.6294547 * w**3
            + 8.56795911 * w**4
        )
    assert abs(1 - np.sinh(inv) / inv / y) < 1e-2, "precision error"
    return float(inv)


def veros_vinokur_grid_steps(
    n_cells: int,
    total_length: float,
    lower_stepsize: float,
    two_sided_grid: bool = False,
    refine_towards: str = "upper",
) -> np.ndarray:
    """``veros.tools.get_vinokur_grid_steps`` transcribed for the
    global_flexible cases (NO ``upper_stepsize`` — the setup never passes
    one).  Covers BOTH setup calls:

    - meridional: ``(ny, 160, 0.5·160/ny, two_sided_grid=True)`` (default
      ``refine_towards='upper'``; symmetric, finest at the centre/equator);
    - vertical:   ``(nz, 5400, 10, refine_towards='lower')`` (one-sided;
      steps DECREASE from the first = deepest cell to the last = the ~10 m
      surface cell, matching Veros's k=0-deepest dzt orientation).

    Both calls land in the sinh branch (``s0 > 1``); the sinc branch
    (``s0 ≤ 1``, refinement coarser than uniform) is NOT transcribed and
    raises loudly.
    """
    if refine_towards not in ("upper", "lower"):
        raise ValueError('refine_towards must be "upper" or "lower"')
    if two_sided_grid:
        if n_cells % 2:
            raise ValueError(
                f"number of grid points must be an even integer (given: {n_cells})"
            )
        n_cells = n_cells // 2
    n_cells += 1

    target_sum = total_length
    if two_sided_grid:
        target_sum *= 0.5

    s0 = float(target_sum) / float(lower_stepsize * n_cells)
    if s0 <= 1.0:
        raise ValueError(
            f"s0 = {s0} <= 1: the sinc branch of get_vinokur_grid_steps is "
            "not transcribed (unreachable for the global_flexible setup)"
        )
    stretching_factor = _veros_sinhc_inverse(s0) * 0.5
    stretched_grid = 1 + np.tanh(
        stretching_factor * np.linspace(0.0, 1.0, n_cells)
    ) / np.tanh(stretching_factor)

    steps = np.diff(stretched_grid * target_sum)
    if refine_towards == "upper":
        steps = steps[::-1]
    if two_sided_grid:
        steps = np.concatenate((steps[::-1], steps))
    assert abs(1 - np.sum(steps) / total_length) < 1e-5, "precision error"
    return steps


def global_flexible_dyt_deg(ny: int = NY) -> np.ndarray:
    """The setup's meridional cell heights [deg]: Vinokur two-sided over
    160° with equatorial spacing ``0.5·160/ny`` (set_grid verbatim)."""
    eq_spacing = EQ_SPACING_FACTOR * LAT_SPAN_DEG / ny
    return veros_vinokur_grid_steps(
        ny, LAT_SPAN_DEG, eq_spacing, two_sided_grid=True)


def global_flexible_dzt_veros(nz: int = NZ) -> np.ndarray:
    """The setup's vertical thicknesses [m] in VEROS orientation (k=0
    deepest ≈ largest, k=nz−1 surface ≈ 10 m): ``Vinokur(nz, 5400, 10,
    refine_towards='lower')`` (set_grid verbatim)."""
    return veros_vinokur_grid_steps(
        nz, MAX_DEPTH_M, MIN_DEPTH_M, refine_towards="lower")


def veros_fill_holes(data: np.ndarray) -> np.ndarray:
    """``veros.tools.fill_holes`` verbatim: in-paint NaNs with the nearest
    finite value by alternating axis sweeps."""
    data = np.array(data)
    dim = data.ndim
    flag = ~np.isnan(data)
    slcs = [slice(None)] * dim
    while np.any(~flag):
        for i in range(dim):
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)
            slcs1 = tuple(slcs1)
            slcs2 = tuple(slcs2)
            repmask = np.logical_and(~flag[slcs1], flag[slcs2])
            data[slcs1][repmask] = data[slcs2][repmask]
            flag[slcs1][repmask] = True
            repmask = np.logical_and(~flag[slcs2], flag[slcs1])
            data[slcs2][repmask] = data[slcs1][repmask]
            flag[slcs2][repmask] = True
    return data


def veros_interpolate(coords, var, interp_coords, kind: str = "linear",
                      fill: bool = True) -> np.ndarray:
    """``veros.tools.interpolate`` for the global_flexible uses (regular
    1-D coords, no missing_value): ``scipy.interpolate.interpn`` with NaN
    fill outside the hull, then nearest-value in-painting."""
    import scipy.interpolate  # setup-time data prep only

    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError("Dimensions of coordinates and values do not match")
    interp_grid = np.rollaxis(
        np.array(np.meshgrid(*interp_coords, indexing="ij")),
        0, len(interp_coords) + 1)
    coords = tuple(np.array(c, dtype="float64") for c in coords)
    out = scipy.interpolate.interpn(
        coords, np.array(var, dtype="float64"),
        np.array(interp_grid, dtype="float64"),
        bounds_error=False, fill_value=np.nan, method=kind)
    if fill:
        out = veros_fill_holes(out)
    return out


def veros_full_axes(nx: int = NX, ny: int = NY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Veros ``calc_grid`` horizontal axes INCLUDING the 2 ghost rows each
    side: ``(xt, yt, yu)`` with shapes ``(nx+4,)``/``(ny+4,)``.

    x: uniform ``dxt = 360/nx``, ``u_centered_grid`` + the ``xu[2] =
    x_origin`` shift, then the CYCLIC ghost overwrite (``xt[:2] =
    xt[-4:-2]`` — ghost VALUES from the far side, not extrapolated).
    y: Vinokur dyt with edge-extended ghosts, ``u_centered_grid`` + the
    ``yu[2] = y_origin`` shift.  Needed by the harness for the verbatim
    ``set_initial_conditions`` data-subset slicing (which uses min/max over
    the FULL axes) and the per-row metric weights."""
    dx = 360.0 / nx
    xt = X_ORIGIN_DEG - dx / 2.0 + dx * (np.arange(nx + 4) - 2.0)
    xt[-2:] = xt[2:4]
    xt[:2] = xt[-4:-2]

    dyt_int = global_flexible_dyt_deg(ny)
    dyt = np.concatenate([
        dyt_int[:1], dyt_int[:1], dyt_int, dyt_int[-1:], dyt_int[-1:]])
    n = dyt.size
    yu = np.zeros(n)
    yu[1:] = np.cumsum(dyt[1:])
    yt = np.zeros(n)
    yt[0] = yu[0] - dyt[0] * 0.5
    yt[1:] = 2.0 * yu[:-1]
    alt = np.ones(n)
    alt[::2] = -1.0
    yt = alt * np.cumsum(alt * yt)
    yt = yt + Y_ORIGIN_DEG - yu[2]
    yu = yu + Y_ORIGIN_DEG - yu[2]
    return xt, yt, yu


# ---------------------------------------------------------------------------
# Grid / vertical coordinate
# ---------------------------------------------------------------------------


def build_global_flexible_grid(ny: int = NY, nx: int = NX) -> LatLonGrid:
    """Stretched global lat-lon C-grid whose interior centres land EXACTLY
    on Veros xt/yt (cyclic x; ``create_stretched_latlon_grid`` adds one
    wall row each side ↔ Veros's first ghost rows — the placement is
    test-anchored against the transcribed ``u_centered_grid`` recursion AND
    the live oracle dump)."""
    dyt = global_flexible_dyt_deg(ny)
    dx = 360.0 / nx
    lon_west = X_ORIGIN_DEG - dx / 2.0          # first interior centre (4deg pattern)
    lon_east = lon_west + nx * dx
    grid, _wall_mask = create_stretched_latlon_grid(
        dyt,
        n_lon=nx,
        lat_south=Y_ORIGIN_DEG - float(dyt[0]),  # yu[2] = y_origin (Veros)
        lon_west=lon_west,
        lon_east=lon_east,
        radius=VEROS_CONSTANTS_CONFIG.R_earth,
        omega=VEROS_CONSTANTS_CONFIG.Omega,      # full 2Ω·sin(lat) Coriolis
        periodic_x=True,                          # enable_cyclic_x
    )
    return grid


def build_global_flexible_z_coord(nz: int = NZ) -> OceanZStarCoordinate:
    """Vinokur z-coordinate from the setup's own construction (legoESM k=0
    surface ⇒ ``dz_ref = dzt_veros[::-1]``); centres/dz_half via Veros's
    ``u_centered_grid`` recursion (the shared bridge construction — every
    ∂/∂z, slope and implicit solve sees the oracle's dzw).

    LOUD degeneracy guard: the recursion is non-monotone for too-coarse nz
    (15: surface centre below the next centre; 18: above z=0) — reject any
    nz whose centres do not interleave the interfaces."""
    dzt_veros = global_flexible_dzt_veros(nz)
    dz_ref_np = dzt_veros[::-1].copy()           # k=0 surface (~10 m)
    z_full_np = veros_u_centered_z_centres(dz_ref_np)
    iface = -np.concatenate([[0.0], np.cumsum(dz_ref_np)])   # 0 .. -H
    if not (np.all(z_full_np < iface[:-1]) and np.all(z_full_np > iface[1:])):
        raise ValueError(
            f"global_flexible Vinokur z at nz={nz} is degenerate: the Veros "
            "u_centered_grid centre recursion leaves a cell centre outside "
            "its cell (measured at nz=15 and nz=18).  nz in {15, 16, 18} is degenerate (non-monotone u_centered column); nz=20 is the smallest robust choice (17/19 also interleave but are untested)."
        )
    dz_ref = jnp.asarray(dz_ref_np, dtype=jnp.float64)
    z_half_ref = jnp.concatenate([
        jnp.zeros(1, dtype=dz_ref.dtype), -jnp.cumsum(dz_ref)])
    z_full_ref = jnp.asarray(z_full_np, dtype=dz_ref.dtype)
    dz_half_ref = jnp.abs(z_full_ref[:-1] - z_full_ref[1:])
    return OceanZStarCoordinate(
        n_levels=nz, H_max=float(np.sum(dz_ref_np)),
        z_full_ref=z_full_ref, z_half_ref=z_half_ref,
        dz_ref=dz_ref, dz_half_ref=dz_half_ref,
    )


# ---------------------------------------------------------------------------
# Topography / kbot (set_topography verbatim; pure given arrays)
# ---------------------------------------------------------------------------


def prepare_global_flexible_topography(
    topo_x: np.ndarray,
    topo_y: np.ndarray,
    topo_z: np.ndarray,
    xt_interior: np.ndarray,
    yt_interior: np.ndarray,
    nx: int = NX,
    ny: int = NY,
) -> np.ndarray:
    """Veros ``set_topography`` data prep (global_flexible.py:174-191),
    verbatim: clamp to ≤ 0, gaussian-smooth at the grid-matched sigma
    ``(0.5·len(x)/nx, 0.5·len(y)/ny)``, reset to 0 where the RAW topo is
    ≥ −1 m, shift the longitude axis to start at the model's western edge,
    nearest-interpolate (no fill) to the interior centres.  Returns
    ``z_interp`` (nx, ny)."""
    import scipy.ndimage  # setup-time data prep only

    topo_z = np.minimum(np.asarray(topo_z, dtype=np.float64), 0.0)
    gaussian_sigma = (0.5 * len(topo_x) / nx, 0.5 * len(topo_y) / ny)
    topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
    topo_z_smoothed = np.where(
        topo_z >= _TOPO_LAND_THRESHOLD_M, 0.0, topo_z_smoothed)

    # _shift_longitude_array verbatim (xt.min() over the FULL axis equals
    # the interior min on the cyclic grid).
    xt_min = float(np.min(xt_interior))
    wrap_i = np.where((topo_x[:-1] < xt_min) & (topo_x[1:] >= xt_min))[0][0]
    topo_x_shifted = np.concatenate(
        (topo_x[wrap_i:-1], topo_x[:wrap_i] + 360.0))
    topo_z_shifted = np.concatenate(
        (topo_z_smoothed[wrap_i:-1, ...], topo_z_smoothed[:wrap_i, ...]))

    return veros_interpolate(
        (topo_x_shifted, topo_y), topo_z_shifted,
        (xt_interior, yt_interior), kind="nearest", fill=False)


def replicate_veros_kbot_flexible(
    z_interp_xy: np.ndarray,
    nz: int = NZ,
) -> np.ndarray:
    """Veros global_flexible kbot from the interpolated topography
    (set_topography:193-204 + the surrounding ghost/boundary semantics),
    verbatim:

    1. ``depth_levels = 1 + argmin|z_interp − zt|`` (nearest LEVEL — not the
       4deg count rule), wet where ``z_interp < 0``;
    2. ``kbot = 0`` where ``kbot == nz`` (only-surface-cell columns are
       land);
    3. cyclic-x ghost exchange (``enforce_boundaries``);
    4. **marginal-sea removal**: ``binary_erosion(binary_fill_holes(
       binary_dilation(kbot == 0)))`` on the FULL ghosted array (the ghost
       frame is what makes enclosed seas 'holes'), then ``kbot = 0`` there.

    Parameters: ``z_interp_xy`` (nx, ny) interior topography from
    :func:`prepare_global_flexible_topography`.  Returns 1-based interior
    ``kbot`` (nx, ny); 0 = land.  Bit-target: the live oracle's ``vs.kbot``
    (asserted by the harness, the proven 4deg gate)."""
    import scipy.ndimage  # setup-time data prep only

    nx, ny = z_interp_xy.shape
    dzt_veros = global_flexible_dzt_veros(nz)
    zt = veros_u_centered_z_centres(dzt_veros[::-1])[::-1]   # k=0 deepest

    z_full = np.zeros((nx + 4, ny + 4))
    z_full[2:-2, 2:-2] = z_interp_xy
    depth_levels = 1 + np.argmin(
        np.abs(z_full[:, :, None] - zt[None, None, :]), axis=2)
    kbot = np.zeros((nx + 4, ny + 4), dtype=np.int64)
    kbot[2:-2, 2:-2] = np.where(z_full < 0.0, depth_levels, 0)[2:-2, 2:-2]
    kbot = np.where(kbot < nz, kbot, 0)
    # enforce_boundaries (cyclic x)
    kbot[-2:, :] = kbot[2:4, :]
    kbot[:2, :] = kbot[-4:-2, :]
    marginal = scipy.ndimage.binary_erosion(
        scipy.ndimage.binary_fill_holes(
            scipy.ndimage.binary_dilation(kbot == 0)))
    kbot = np.where(marginal, 0, kbot)
    return kbot[2:-2, 2:-2]


def kbot_to_mask_and_h_bathy_flexible(
    kbot_xy: np.ndarray,
    nz: int = NZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Veros kbot → legoESM ``(land_mask, H_bathy)`` on the (n_lat, n_lon)
    grid including the two wall rows, with H_bathy SNAPPED to interface
    depths ``Σ dz_ref[:n_wet]`` (``n_wet = nz − kbot + 1``) so the partial-
    cell coordinate degenerates to FULL cells — the 4deg pattern on the
    flexible vertical."""
    dz_ref = global_flexible_dzt_veros(nz)[::-1]
    iface_depth = np.concatenate([[0.0], np.cumsum(dz_ref)])    # (nz+1,)
    n_wet = np.where(kbot_xy > 0, nz - kbot_xy + 1, 0)          # (x, y)
    H_xy = iface_depth[n_wet]
    mask_xy = (n_wet > 0).astype(np.float64)
    nx, ny = kbot_xy.shape
    land_mask = np.zeros((ny + 2, nx))
    land_mask[1:-1, :] = mask_xy.T
    H_bathy = np.zeros((ny + 2, nx))
    H_bathy[1:-1, :] = H_xy.T
    return land_mask, H_bathy


# ---------------------------------------------------------------------------
# Layout bridges + forcing prep helpers (pure)
# ---------------------------------------------------------------------------


def veros_mit_tau_shift(taux_xym: np.ndarray,
                        tauy_xym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Veros set_forcing_kernel MIT-grid index shift
    (global_flexible.py:368-369): ``surface_taux[i] = taux[i+1]`` (one cell
    in x) and ``surface_tauy[j] = tauy[j+1]`` (one cell in y), applied ONCE
    at data-prep time on the INTERIOR monthly stacks (x, y, 12).

    With cyclic-x ghosts (``taux[nx+2] = taux[2]``) the x-shift is a roll by
    −1; the y-shift pulls the zero NORTH GHOST row into the last interior
    row (Veros's ``enforce_boundaries`` never fills y ghosts ⇒ tauy's
    northernmost surface_tauy row is 0) — replicated, not smoothed."""
    taux_shift = np.roll(taux_xym, -1, axis=0)
    tauy_shift = np.concatenate(
        [tauy_xym[:, 1:, :], np.zeros_like(tauy_xym[:, :1, :])], axis=1)
    return taux_shift, tauy_shift


def veros_area_t_flexible(
    yt_deg: np.ndarray,
    dyt_deg: np.ndarray,
    nx: int = NX,
    r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth,
) -> np.ndarray:
    """Veros T-cell area column weights ``dxt·dyt·cost`` [m²] on the
    STRETCHED grid (per-latitude row; broadcast over x)."""
    dx_deg = 360.0 / nx
    return veros_area_t_generic(yt_deg, dx_deg, dyt_deg, r_earth)


# ---------------------------------------------------------------------------
# Physics configs (scoping §B; deltas vs global_4deg documented at each line)
# ---------------------------------------------------------------------------

# EKE: identical parameter block to global_4deg/ACC with the FLIP BACK
# (isopycnal_diffusion=True ⇒ K_iso = K_gm), plus the GM/Redi deltas
# (S_max=5e-3, taper_width_frac=1.0, K_iso_steep=50).  This is the exact same
# delta pair as global_1deg, shared via gm_redi_eke_isopycnal_on.
GLOBAL_FLEX_GM_REDI_CONFIG, GLOBAL_FLEX_EKE_CONFIG = gm_redi_eke_isopycnal_on(
    GLOBAL4_GM_REDI_CONFIG, GLOBAL4_EKE_CONFIG,
)

# TKE: the setup repeats the global_4deg block VERBATIM (c_k=0.1, c_eps=0.7,
# alpha_tke=30, mxl_min=1e-8, tke_mxl_choice=2, kappaM_min=2e-4,
# kappaH_min=2e-5, kappaH profile, superbee advection) and r_bot=0 again ⇒
# reuse the 4deg config object unchanged.
GLOBAL_FLEX_TKE_CONFIG = GLOBAL4_TKE_CONFIG


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def build_global_flexible_state(
    grid: LatLonGrid,
    z_coord,
    land_mask: np.ndarray,
    H_bathy: np.ndarray,
    T_init: np.ndarray | None = None,
    S_init: np.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Initial state: interpolated file T/S (legoESM order/shape) masked by
    the active cells, rest velocity, rigid-lid eta ≡ 0, TKE/EKE carry
    fields seeded (the 4deg seeding convention — see its docstring)."""
    return build_veros_global_state(
        grid, z_coord, land_mask, H_bathy,
        tke_config=GLOBAL_FLEX_TKE_CONFIG,
        eke_config=GLOBAL_FLEX_EKE_CONFIG,
        T_init=T_init, S_init=S_init,
    )


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def build_global_flexible_physics_config() -> OceanPhysicsConfig:
    """global_flexible physics: prognostic TKE (superbee advection,
    Richardson Prandtl) + flux_feedback surface forcing WITH the
    penetrative-shortwave q_solar channel (EXT-F2).  GM/Redi rides the
    top-level dynamics config; IDEMIX off."""
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke", tke=GLOBAL_FLEX_TKE_CONFIG),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(
            scheme="flux_feedback",
            flux_feedback=FluxFeedbackConfig(
                # Veros kernel cp_0 hardcode (same literal as the 4deg kernel,
                # global_flexible.py:362) — NOT constants.c_sw.
                c_sw=VEROS_GLOBAL4_CP0,
                rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
                tau_restore_s=T_REST_S,              # t_rest = 30 d
                ice_mask=True,
                # *** EXT-F2: the qsol penetrative column.  Jerlov "I" ≡ the
                # setup literals rpart=0.58, efold1=0.35, efold2=23.0. ***
                penetrative_shortwave=True,
                shortwave_water_type="I",
            ),
        ),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,    # solar is OWNED by flux_feedback here
    )


def build_global_flexible_model_config() -> LatLonCGridOceanConfig:
    """global_flexible dynamics config — the matched-4deg faithful dycore
    stack (same Veros core ⇒ same options) with the setup's parameter
    deltas (A_h literal, GM/Redi block, the documented rescaled dt pair)."""
    return LatLonCGridOceanConfig(
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=GLOBAL_FLEX_A_H,                        # setup literal 5e4
        A_h_lat_scaling=True,                       # cos¹(lat) scaling
        A_h_cos_power=1,
        lateral_viscosity_operator="flux_divergence",
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        vertical_momentum_scheme="centered_full",
        tracer_advection="centered",
        bottom_drag_r=0.0,                          # Veros r_bot default 0
        eos="veros_gsw",                            # eq_of_state_type = 5
        implicit_vertical_mixing=True,
        K_v=0.0,
        gm_redi=GLOBAL_FLEX_GM_REDI_CONFIG,
        surface_forcing_implicit=True,              # Veros source placement
        # Shared bundle via veros_stepping.veros_faithful_stepping (#433);
        # dt_mom_ratio=DT_MOM_RATIO=8 (dt arg IS dt_tracer; |f|·dt_mom ≈ 0.25 @72°).
        **veros_faithful_stepping(with_surface_forcing=True,
                                  dt_mom_ratio=DT_MOM_RATIO),
        physics=build_global_flexible_physics_config(),
    )


def build_global_flexible_recipe(
    z_interp_topo_xy: np.ndarray,
    temp_xyz: np.ndarray | None = None,
    salt_xyz: np.ndarray | None = None,
) -> ACCRecipe:
    """One-stop constructor from PREPARED interior arrays (the harness owns
    the netCDF reads + the Veros-verbatim interpolation; unit tests
    fabricate tiny ones).

    Parameters: ``z_interp_topo_xy`` (nx, ny) — the nearest-interpolated
    smoothed topography from :func:`prepare_global_flexible_topography`;
    ``temp_xyz``/``salt_xyz`` (nx, ny, nz) — the trilinearly interpolated
    initial conditions in VEROS z-order (k=0 deepest).

    Returns the shared ``ACCRecipe`` container; ``wind_forcing`` is None —
    the monthly forcing is composed per-step by the harness
    (``scripts/validate/ocean_fidelity/run_global_flexible_freerun.py``)."""
    grid = build_global_flexible_grid()
    z_ref = build_global_flexible_z_coord()
    kbot = replicate_veros_kbot_flexible(z_interp_topo_xy)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy_flexible(kbot)
    z_coord = create_partial_cell_coordinate(z_ref, jnp.asarray(H_bathy))
    T_init = (veros_xyz_to_legoesm_flex(temp_xyz)
              if temp_xyz is not None else None)
    S_init = (veros_xyz_to_legoesm_flex(salt_xyz)
              if salt_xyz is not None else None)
    initial_state = build_global_flexible_state(
        grid, z_coord, land_mask, H_bathy, T_init=T_init, S_init=S_init)
    return ACCRecipe(
        model_config=build_global_flexible_model_config(),
        physics_config=build_global_flexible_physics_config(),
        grid=grid,
        z_coord=z_coord,
        land_mask=jnp.asarray(land_mask),
        initial_state=initial_state,
        wind_forcing=None,
    )
