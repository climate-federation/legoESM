"""Component factory for the legoESM driver.

Resolves configured atmosphere, ocean, land, ice, and coupler
components into concrete model instances.  The driver delegates all
component construction here, so that ``ModelDriver`` never hardcodes
a particular dynamical core or surface component class.

Design
------
* The atmosphere factory uses the two-axis resolver already in
  ``atmosphere.dynamics`` (``resolve_solver_name`` / ``create_model``),
  but wraps it with driver-specific logic:
  - Computes physical diffusion coefficients from grid properties.
  - Validates that the grid type is compatible with the chosen solver.
  - Returns the model instance ready for time-stepping.
* ``create_ocean_component``, ``create_land_component``, and
  ``create_ice_component`` delegate to the canonical component
  constructors in ``legoesm.ocean``, ``legoesm.land``, and
  ``legoesm.ice``.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import jax.numpy as jnp

from legoesm.driver.config import ExperimentConfig, DycoreConfig

logger = logging.getLogger("legoesm.driver.factory")


# =========================================================================
# Supported-combination matrix
# =========================================================================

# (model_type, discretization, grid_type) -> flat solver name
#
# Only combinations listed here are supported through the driver.  Any
# other combination fails fast with a precise diagnostic message.

_DRIVER_SUPPORTED: dict[tuple[str, str, str], str] = {
    # --- Cubed-sphere C-D grid ---
    ("shallow_water", "centered",       "cubed_sphere"): "cdgrid_shallow_water",
    ("shallow_water", "cdgrid",         "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "centered",       "cubed_sphere"): "cdgrid_primitive_equations",
    ("hydrostatic",   "cdgrid",         "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","centered",       "cubed_sphere"): "cdgrid_compressible_euler",
    ("nonhydrostatic","cdgrid",         "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Cubed-sphere finite-volume (alias) ---
    ("shallow_water", "finite_volume",  "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "finite_volume",  "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","finite_volume",  "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Spectral (Gaussian grid) ---
    ("shallow_water", "spectral",       "gaussian"):     "spectral_shallow_water",
    ("hydrostatic",   "spectral",       "gaussian"):     "spectral_primitive_equations",
    ("nonhydrostatic","spectral",       "gaussian"):     "spectral_compressible_euler",

    # --- Lat-lon C-grid ---
    ("shallow_water", "finite_volume",  "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "finite_volume",  "latlon"):       "latlon_cgrid_primitive_equations",
    ("shallow_water", "centered",       "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "centered",       "latlon"):       "latlon_cgrid_primitive_equations",
    ("shallow_water", "latlon_cgrid",   "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "latlon_cgrid",   "latlon"):       "latlon_cgrid_primitive_equations",

    # --- MPAS / SCVT Voronoi mesh + TRiSK discretization
    # (Ringler 2010, Thuburn 2009).  Canonical grid_type = "mpas"
    # (matching the ocean side); legacy aliases voronoi /
    # icosahedral / mpas_voronoi are normalised at the config
    # boundary via driver.config.normalize_grid_type. ---
    ("hydrostatic",   "mpas",           "mpas"):        "mpas_primitive_equations",
    ("nonhydrostatic","mpas",           "mpas"):        "mpas_compressible_euler",

    # --- Doubly-periodic plane (CRM rollout, PR2c) ---
    # Plane only supports the non-hydrostatic compressible Euler dycore.
    # All other (model_type, plane) combinations fall through to
    # ``_fail_unsupported`` so users see a clear error pointing at the
    # PR2c roadmap rather than a quiet construction crash.
    ("nonhydrostatic","plane",          "plane"):        "plane_compressible_euler",

    # --- SFNO data-driven ---
    # SFNO is a spherical-harmonic operator: it reads ``grid.n_sh`` and
    # runs SH synthesis/analysis, so it can only construct on the Gaussian
    # spectral grid.  (These rows previously advertised ``cubed_sphere``,
    # which has no SH transform — the build crashed with AttributeError.)
    ("shallow_water", "sfno",           "gaussian"):     "sfno_shallow_water",
    ("hydrostatic",   "sfno",           "gaussian"):     "sfno_primitive_equations",

    # --- U-Cast data-driven (convolutional U-Net emulator) ---
    # The U-Net itself is grid-agnostic, but the PE bridge packs/unpacks
    # the spectral primitive-equation state (see ucast_pe.py docstring:
    # ``grid : GaussianGrid``), so it is Gaussian-only too.
    ("hydrostatic",   "u_cast",         "gaussian"):     "ucast_primitive_equations",
}


def supported_matrix() -> list[dict[str, str]]:
    """Return the driver-supported (model_type, discretization, grid) matrix."""
    return [
        {"model_type": k[0], "discretization": k[1], "grid_type": k[2],
         "solver": v}
        for k, v in sorted(_DRIVER_SUPPORTED.items())
    ]


# =========================================================================
# Diffusion coefficient helpers
# =========================================================================

class DiffusionCoeffs(NamedTuple):
    """Physical diffusion coefficients computed from grid properties."""
    A_h: float         # Laplacian viscosity [m^2/s]
    hyperdiff: float   # Biharmonic hyperdiffusion [m^4/s]
    div_damp: float    # Divergence damping [m^2/s]


def _grid_min_dx(grid) -> float:
    """Minimum grid spacing [m] used to scale diffusion coefficients.

    MPAS/Voronoi reports the cell-to-cell edge distance directly; lat-lon's
    smallest cell is the pole cell (``min(dx)/2``).  Falls back to 1e5 m for
    grids exposing neither attribute.
    """
    if hasattr(grid, 'dcEdge'):
        # Voronoi/MPAS: dcEdge is the cell-to-cell distance along each edge
        return float(jnp.min(jnp.asarray(grid.dcEdge)))
    if hasattr(grid, 'dx'):
        # Lat-lon: min(dx)/2 is the POLE-cell spacing.  This bounds the scalar
        # Laplacian A_h to be CFL-stable at the pole (A_h = 0.05*dx_pole^2/dt);
        # an equatorial-dx A_h is ~3000x larger and makes the EXPLICIT diffusion
        # operator CFL-UNSTABLE at the pole (the polar filter truncates wave
        # modes, not the diffusion stencil) -> immediate blow-up.  NOTE: this
        # pole-bounded scalar A_h is too weak to damp midlatitude grid-scale
        # noise at fine (<=2deg) resolution; a latitude-dependent viscosity
        # (A_h ~ local dx^2) or biharmonic hyperdiffusion in the latlon_cgrid
        # dycore is needed for stable 2deg global runs (TODO).
        return float(jnp.min(jnp.asarray(grid.dx))) / 2.0
    return 1e5


def compute_diffusion(grid, dc: DycoreConfig) -> DiffusionCoeffs:
    """Compute physical diffusion coefficients from grid and config.

    Parameters
    ----------
    grid : Grid object
        Must expose a ``dx`` attribute (minimum grid spacing in metres).
    dc : DycoreConfig
        Dycore configuration with ``dt``, ``hyperdiff_scale``,
        ``div_damp_scale``.

    Returns
    -------
    DiffusionCoeffs
    """
    dx_min = _grid_min_dx(grid)
    DT = dc.dt

    # Laplacian viscosity (2nd-order, scale-NON-selective).  The earlier
    # default 0.05*dx^2/DT was ~5-10x too strong: it damped a 3000 km
    # baroclinic eddy in ~5 h (faster than its ~1-2 day growth), crushing the
    # midlatitude eddy-driven jets and producing spurious equatorial
    # super-rotation (dry Held-Suarez jet 2.9 vs spectral 37 m/s).  Reduced
    # ~16x to a level whose eddy-scale (~4000 km) damping time is ~days while
    # still damping 2*dx grid noise in a few hours; the scale-selective
    # 4th-order hyperdiff below is the primary grid-noise control.  a_h_scale
    # exposes it for tuning (0 = rely on hyperdiff alone).
    A_h = dc.a_h_scale * 3.0e-3 * dx_min ** 2 / DT

    # Biharmonic: e-folding time for grid-scale noise
    tau_efold = 24.0 * 3600.0
    hyperdiff = dc.hyperdiff_scale * dx_min ** 4 / tau_efold

    # Divergence damping: damps external gravity wave mode
    c_grav = 300.0
    div_damp = dc.div_damp_scale * c_grav * dx_min / (2.0 * 3.14159)

    return DiffusionCoeffs(A_h=A_h, hyperdiff=hyperdiff, div_damp=div_damp)


# Solvers that apply the EXPLICIT biharmonic hyperdiff / divergence damping
# (forward the raw ``diff.hyperdiff`` / ``diff.div_damp`` to a finite-volume
# operator with no implicit/spectral inversion).  Spectral dycores recompute
# their own implicit-stable hyperdiff (and zero the FV one); the plane dycore
# zeroes hyperdiff; lat-lon clamps A_h itself — none belong here, so the guard
# never cries wolf about a coefficient the selected solver discards.
_EXPLICIT_HYPERDIFF_SOLVERS = frozenset({
    "cdgrid_shallow_water",
    "cdgrid_primitive_equations",
    "cdgrid_compressible_euler",
    "mpas_primitive_equations",
})
_EXPLICIT_DIVDAMP_SOLVERS = frozenset({"cdgrid_primitive_equations"})


def warn_if_diffusion_unstable(solver_name: str, diff: DiffusionCoeffs,
                               grid, dt: float) -> None:
    """Warn (no clamp) if an explicit diffusion coeff exceeds the stable max.

    Only the lat-lon C-grid clamps its Laplacian A_h; the biharmonic hyperdiff
    and divergence damping reach the cubed-sphere / MPAS finite-volume dycores
    with no runtime guard, so an over-large ``hyperdiff_scale`` / ``div_damp_scale``
    (or ``dt``) blows up downstream as a silent NaN.  This converts that into a
    clear init-time diagnostic.

    The threshold is the model's OWN ``adaptive_hyperdiff_coeff(..., safety=1.0)``
    — the largest coefficient legoESM considers stable for the grid+dt.  Using
    that (rather than an idealized forward-Euler Cartesian bound) keeps the guard
    consistent with the validated coefficient picker and the multi-stage SSP-RK
    integrators these dycores use, so it fires only on gross over-specification,
    not on marginal calibration (which depends on the integrator stability region
    and cube metric and is validated empirically, not knowable at config time).
    Coefficients are NOT clamped — that would silently alter the user's tuned
    scale-selective damping.
    """
    from legoesm.core.cfl import adaptive_hyperdiff_coeff
    dx_min = _grid_min_dx(grid)
    checks = []
    if solver_name in _EXPLICIT_HYPERDIFF_SOLVERS:
        checks.append(("hyperdiff (biharmonic)", diff.hyperdiff, 4))
    if solver_name in _EXPLICIT_DIVDAMP_SOLVERS:
        checks.append(("div_damp", diff.div_damp, 2))
    for name, coeff, order in checks:
        coeff_max = adaptive_hyperdiff_coeff(dx_min, dt, order=order, safety=1.0)
        if coeff > coeff_max:
            logger.warning(
                "Diffusion %s=%.3e for solver %s exceeds the max stable "
                "coefficient %.3e (grid dx_min=%.1f m, dt=%.1f s) by %.1fx; "
                "the run may go unstable. Reduce hyperdiff_scale/div_damp_scale "
                "or dt.",
                name, coeff, solver_name, coeff_max, dx_min, dt,
                coeff / coeff_max,
            )


# =========================================================================
# Atmosphere factory
# =========================================================================

def create_atmosphere_dycore(
    config: ExperimentConfig,
    grid,
    sigma,
) -> Any:
    """Resolve and instantiate the configured atmosphere dynamical core.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    grid
        Horizontal grid (cubed-sphere, Gaussian, lat-lon, etc.).
    sigma
        Vertical coordinate.

    Returns
    -------
    model
        The instantiated dynamical core, ready for ``step_with_physics``.

    Raises
    ------
    ValueError
        If the (model_type, discretization, grid_type) combination is
        not supported.
    """
    dc = config.dycore
    gc = config.grid

    model_type = dc.model_type
    discretization = dc.discretization
    # Defensive canonical-name normalization at the factory entry: callers
    # that bypass run_amip's argparse postprocessor (direct test fixtures,
    # ad-hoc scripts, older YAML loaders) might still pass ``voronoi`` /
    # ``icosahedral`` / ``mpas_voronoi`` for the SCVT mesh.  The dispatch
    # table below speaks only the canonical ``mpas`` so we normalise
    # here too — the cost is one dict lookup.
    from legoesm.driver.config import normalize_grid_type
    grid_type = normalize_grid_type(gc.grid_type)

    key = (model_type, discretization, grid_type)

    if key not in _DRIVER_SUPPORTED:
        _fail_unsupported(model_type, discretization, grid_type)

    solver_name = _DRIVER_SUPPORTED[key]
    diff = compute_diffusion(grid, dc)
    warn_if_diffusion_unstable(solver_name, diff, grid, dc.dt)

    logger.info(
        "Atmosphere: model_type=%s, discretization=%s, grid=%s -> %s",
        model_type, discretization, grid_type, solver_name,
    )

    # ----- Cubed-sphere C-D grid solvers -----
    if solver_name == "cdgrid_shallow_water":
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel, CDGridShallowWaterConfig,
        )
        cfg = CDGridShallowWaterConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CDGridShallowWaterConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
        )
        return CDGridShallowWaterModel(grid, cfg)

    if solver_name == "cdgrid_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        )
        cfg = CDGridPrimitiveEquationConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff, hyperdiff_ps_coeff=diff.hyperdiff,
            div_damp_coeff=diff.div_damp,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
            # Issue #273 Phase 3: forward the implicit gravity-wave
            # damping switches from the canonical driver config.
            # Default off (both 0/False) keeps the explicit path
            # bit-exact for existing call sites.
            implicit_grav_wave_use_pcg=dc.implicit_grav_wave_use_pcg,
            implicit_grav_wave_damping=dc.implicit_grav_wave_damping,
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CDGridPrimitiveEquationConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
            # #771: flux-form moisture transport (needs moisture attached via
            # ExperimentConfig.moisture_advection; default off = advective).
            moisture_flux_form=getattr(dc, "moisture_flux_form", False),
        )
        return CDGridPrimitiveEquationModel(grid, sigma, cfg)

    if solver_name == "cdgrid_compressible_euler":
        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
        )
        cfg = CDGridCompressibleEulerConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
            # Forward the driver-level mass fixer (mirrors the plane / NH
            # branches).  Previously dropped: DycoreConfig.fix_mass=True was
            # silently ignored on the cubed-sphere NH path.
            fix_mass=dc.fix_mass,
            anchor_mass_to_initial=dc.fix_mass,
        )
        # Non-hydrostatic requires height coordinate and terrain metric.
        # Expect grid to provide these or construct defaults.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                # Build flat (zero topography) height coordinate.
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                H_top = 30_000.0  # metres
                height_coord = create_height_coordinate(nlev, H_top)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.lat)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        return CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)

    # ----- Spectral solvers (Gaussian grid) -----
    if solver_name == "spectral_shallow_water":
        from legoesm.atmosphere.dynamics.spectral_sw import SpectralShallowWaterModel
        return SpectralShallowWaterModel(grid=grid)

    if solver_name == "spectral_primitive_equations":
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
        )
        # Compute hyperdiffusion from truncation: 0.5-hour e-folding at max wavenumber
        n_max = grid.n_max
        a = grid.radius
        eig_max = n_max * (n_max + 1) / (a * a)
        hyperdiff = 1.0 / (0.5 * 3600.0 * eig_max ** 2)
        # Spectral hydrostatic PE requires the 5-stage SSP-RK54 (spectral
        # stability); it does not support the other integrators.  Tolerate the
        # global default (no deliberate choice) but REJECT any other EXPLICIT
        # time_integrator with a clear message instead of silently overriding it.
        if dc.time_integrator not in ("ssp_rk54", "auto",
                                      type(dc)().time_integrator):
            raise ValueError(
                f"spectral hydrostatic PE supports only time_integrator='ssp_rk54' "
                f"(spectral stability); got dycore.time_integrator="
                f"{dc.time_integrator!r} — that integrator is not implemented for "
                f"the spectral path."
            )
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=hyperdiff,
            hyperdiff_order=2,
            time_integrator="ssp_rk54",
            p_floor=200.0,
            dealiasing_fraction=0.667,
            # Forward the driver-level mass fixer (mirrors the plane / NH
            # branches).  Previously dropped: DycoreConfig.fix_mass=True was
            # silently ignored on the spectral hydrostatic path.
            fix_mass=dc.fix_mass,
            anchor_mass_to_initial=dc.fix_mass,
        )
        return SpectralPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=pe_config,
        )

    if solver_name == "spectral_compressible_euler":
        from legoesm.atmosphere.dynamics.spectral_nh import (
            SpectralCompressibleEulerModel, SpectralNHConfig,
        )
        # Non-hydrostatic uses a height (z-star) coordinate, not sigma:
        # build the flat-topography defaults exactly like the CDGrid-NH
        # branch above.  z_s must be (n_lat, n_lon) on the Gaussian grid
        # (grid.lat is 1-D), hence zeros_like(lat2d).
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                height_coord = create_height_coordinate(nlev, 30_000.0)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.lat2d)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        nh_cfg = SpectralNHConfig(
            fix_mass=dc.fix_mass,
            anchor_mass_to_initial=dc.fix_mass,
        )
        return SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, nh_cfg,
        )

    # ----- MPAS icosahedral -----
    if solver_name == "mpas_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
        )
        # Integrator: ``"auto"`` (run_amip CLI default) resolves to the
        # MPAS dycore's own default (ssp_rk54_scan).  The MPAS
        # biharmonic hyperdiffusion eigenvalues at production time
        # steps fall OUTSIDE ssp_rk3's stability region
        # (primitive_eq_mpas.py stability notes: ssp_rk3 diverges
        # within ~3 steps at dt=600 with hyperdiff ON, ssp_rk54 stays
        # bounded) — silently forwarding the global "ssp_rk3" default
        # was the "MPAS hidden CFL constraint" that forced the AMIP
        # smoke test down to dt=60 (AMIP.md Known issues #2).  Any
        # explicit scheme name (including ssp_rk3, for deliberate
        # integrator-sensitivity runs) is forwarded verbatim.
        _ti = dc.time_integrator
        if _ti == "auto":
            _ti = MPASPrimitiveEquationConfig().time_integrator
        cfg = MPASPrimitiveEquationConfig(
            nu_del2=diff.A_h,
            nu_del4=diff.hyperdiff,
            nu_del4_ps=diff.hyperdiff,
            K_h=diff.A_h,
            fix_mass=dc.fix_mass,
            time_integrator=_ti,
            # #930 cure: vertical biharmonic damping of the 2Δσ T checkerboard.
            nu_vert4_T=dc.mpas_nu_vert4_T,
        )
        return MPASPrimitiveEquationModel(mesh=grid, sigma_coord=sigma, config=cfg)

    if solver_name == "mpas_compressible_euler":
        from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
            MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
        )
        # Non-hydrostatic uses a height (z-star) coordinate, not sigma:
        # flat-topography defaults as in the CDGrid-NH branch; the MPAS
        # surface field is 1-D over cells.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                height_coord = create_height_coordinate(nlev, 30_000.0)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.latCell)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        nh_cfg = MPASCompressibleEulerConfig(
            nu_del2=diff.A_h,
            nu_del4=diff.hyperdiff,
            K_h=diff.A_h,
            fix_mass=dc.fix_mass,
            anchor_mass_to_initial=dc.fix_mass,
        )
        return MPASCompressibleEulerModel(grid, height_coord, terrain_metric, nh_cfg)

    # ----- Doubly-periodic plane -----
    if solver_name == "plane_compressible_euler":
        from legoesm.atmosphere.dynamics.compressible_euler import (
            CompressibleEulerConfig,
        )
        from legoesm.atmosphere.dynamics.compressible_euler_plane import (
            PlaneCompressibleEulerModel,
            make_flat_plane_terrain_metric,
        )
        # PR2c MVP: every plane setup is dry and uses flat terrain. The
        # plane grid carries its own ``f_y`` (Coriolis) field built at
        # ``create_plane_grid`` time; the driver does not override it.
        # Build a flat terrain metric if the grid did not preattach one,
        # and a default height coordinate if the driver did not pass a
        # vertical coordinate that exposes ``z_full`` / ``z_half``.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None:
            from legoesm.grids.vertical import create_height_coordinate
            nlev = grid.nlev
            # 30 km model top is the same default used by the
            # cubed-sphere NH branch above.
            height_coord = create_height_coordinate(nlev, H=30_000.0)
        if terrain_metric is None:
            terrain_metric = make_flat_plane_terrain_metric(grid, height_coord)
        # PR2c keeps the driver path strict: hyperdiff / sponge knobs
        # come from the dycore config but the plane dycore rejects
        # them per ``validate_plane_config`` until PR3. The driver
        # therefore constructs a minimal config that is safe for
        # ``PR2c`` use; users wanting sponge enabled can construct
        # ``PlaneCompressibleEulerModel`` directly with a custom
        # ``CompressibleEulerConfig``.
        cfg = CompressibleEulerConfig(
            sponge_coeff=0.0,
            hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0,
            hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False,
            use_coriolis=False,
            fix_mass=dc.fix_mass,
            anchor_mass_to_initial=dc.fix_mass,
        )
        return PlaneCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)

    # ----- Lat-lon C-grid solvers -----
    if solver_name in ("latlon_cgrid_shallow_water",
                       "latlon_cgrid_primitive_equations"):
        # The lat-lon C-grid configs support A_h and fix_mass but not
        # The lat-lon C-grid solver uses Laplacian viscosity (A_h) only —
        # it has no biharmonic hyperdiffusion operator.  However, the
        # driver still uses compute_diffusion().hyperdiff for moisture
        # smoothing, so hyperdiff_scale is NOT rejected here.
        #
        # Divergence damping is not used anywhere on lat-lon, so values
        # > 1.0 (requesting amplified damping) are rejected.
        if dc.div_damp_scale > 1.0:
            raise ValueError(
                f"Lat-lon C-grid solver does not support divergence "
                f"damping. div_damp_scale={dc.div_damp_scale} was "
                f"requested but this mechanism is not implemented. "
                f"Set div_damp_scale=1.0 (default) or 0.0 (disabled)."
            )

        # Honor conservation_fixer: when explicitly False, disable fix_mass
        # even if dc.fix_mass is True.
        _fix_mass = dc.fix_mass
        if dc.conservation_fixer is False:
            _fix_mass = False
            if dc.fix_mass:
                logger.info(
                    "Lat-lon C-grid solver: conservation_fixer=False "
                    "overrides fix_mass=True → mass fixer disabled",
                )

        # ---- Pole-cell CFL safeguards ----
        # The explicit C-grid solver on a lat-lon grid has its smallest
        # cell at the poles: dx_pole = R * dlon * cos(π/2 - dlat/2).
        # Both the advective CFL (dt < dx / c_grav) and the diffusive
        # CFL (A_h < 0.4 * dx² / dt) must be satisfied there.
        #
        # Stage 3-E: when ``dc.use_polar_filter`` is True, the Fourier
        # polar filter truncates Fourier modes in longitude that would
        # violate CFL at high latitudes, so the dynamics is stable at
        # ``dt`` set by the equatorial CFL instead of the pole CFL.
        # ``dx_equator`` = R * dlon ≫ dx_pole at all but the lowest
        # resolutions, so this typically lifts the clamp by ~100x at
        # n_lat=180 and ~10x at n_lat=90 — enough to make a 100-y AMIP
        # at 1° feasible within a chained 72-h SLURM budget.
        from legoesm.core.cfl import (
            pole_cell_dx, cfl_max_dt, max_laplacian_viscosity,
        )
        dx_pole = pole_cell_dx(grid)
        c_grav = 300.0  # gravity wave speed [m/s]
        dt_max_advective_pole = cfl_max_dt(
            dx_pole, c_grav, cfl_number=0.8, ndim=1,
        )
        _effective_dt = dc.dt

        if dc.use_polar_filter:
            # Filter on → equatorial CFL is the effective limit.
            # dx_equator = R * dlon = circumference / n_lon.
            import math as _math
            dx_equator = float(2.0 * _math.pi * grid.radius / grid.n_lon)
            dt_max_advective = cfl_max_dt(
                dx_equator, c_grav, cfl_number=0.8, ndim=1,
            )
            dx_for_diffusion = dx_equator
            _clamp_dx_label = "equatorial"
        else:
            dt_max_advective = dt_max_advective_pole
            dx_for_diffusion = dx_pole
            _clamp_dx_label = "pole-cell"

        if _effective_dt > dt_max_advective:
            logger.warning(
                "Lat-lon C-grid: dt=%.1f s exceeds %s advective "
                "CFL limit (%.1f s); clamping to %.1f s. "
                "Set dycore.dt <= %.1f for this grid.",
                _effective_dt, _clamp_dx_label, dt_max_advective,
                dt_max_advective, dt_max_advective,
            )
            _effective_dt = dt_max_advective

        A_h_max = max_laplacian_viscosity(dx_for_diffusion, _effective_dt)
        _A_h = min(diff.A_h, A_h_max)
        if diff.A_h > A_h_max:
            logger.warning(
                "Lat-lon C-grid: A_h=%.2e exceeds %s diffusive "
                "CFL limit (%.2e); clamping.",
                diff.A_h, _clamp_dx_label, A_h_max,
            )

    if solver_name == "latlon_cgrid_shallow_water":
        from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
        )
        cfg = CGridLatLonShallowWaterConfig(
            A_h=_A_h,
            fix_mass=_fix_mass,
            # Stage 3-E: pass polar-filter parameters through (mirrors the
            # PE branch below).  Previously dropped: the dt/CFL relaxation
            # above already assumed the filter was ON (dt lifted to the
            # equatorial CFL) while the model silently ran WITHOUT it.
            use_polar_filter=dc.use_polar_filter,
            polar_filter_cutoff_deg=dc.polar_filter_cutoff_deg,
            polar_filter_max_wave_speed=dc.polar_filter_max_wave_speed,
        )
        model = CGridLatLonShallowWaterModel(grid, cfg, dt=_effective_dt)
        model.effective_dt = _effective_dt
        return model

    if solver_name == "latlon_cgrid_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        )
        cfg = CGridLatLonPrimitiveEquationConfig(
            A_h=_A_h,
            fix_mass=_fix_mass,
            # Stage 3-E: pass polar-filter parameters through.  When
            # use_polar_filter is False (default) the model's filter
            # mask is None and no FFT is applied — bit-identical to
            # pre-Stage-3-E behaviour.
            # Direct attribute access (these are guaranteed DycoreConfig fields);
            # getattr-with-literal fallbacks would mask a rename behind a stale
            # default instead of failing (CLAUDE.md).
            use_polar_filter=dc.use_polar_filter,
            polar_filter_cutoff_deg=dc.polar_filter_cutoff_deg,
            polar_filter_max_wave_speed=dc.polar_filter_max_wave_speed,
            # #836 top sponge (Rayleigh damping toward the lid).  Default
            # sponge_coeff=0 -> OFF -> the tendency is bit-identical to the
            # pre-#836 dycore.  Direct DycoreConfig attribute access (guaranteed
            # fields; no getattr-literal fallback that would mask a rename).
            sponge_coeff=dc.sponge_coeff,
            sponge_width_m=dc.sponge_width_m,
            sponge_shape=dc.sponge_shape,
            sponge_scale_height_m=dc.sponge_scale_height_m,
            # Task #25: time integrator (default ssp_rk3, opt into
            # ssp_rk3_scan for ~1.5× JIT compile speedup at scale).
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CGridLatLonPrimitiveEquationConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
        )
        model = CGridLatLonPrimitiveEquationModel(
            grid, sigma, cfg, dt=_effective_dt)
        model.effective_dt = _effective_dt
        return model

    # ----- SFNO data-driven -----
    if solver_name == "sfno_shallow_water":
        from legoesm.atmosphere.dynamics.sfno_sw import SFNOShallowWaterModel
        return SFNOShallowWaterModel(grid=grid)

    if solver_name == "sfno_primitive_equations":
        from legoesm.atmosphere.dynamics.sfno_pe import SFNOPrimitiveEquationModel
        return SFNOPrimitiveEquationModel(grid=grid, sigma_coord=sigma)

    # ----- U-Cast data-driven (convolutional U-Net emulator) -----
    if solver_name == "ucast_primitive_equations":
        from legoesm.atmosphere.dynamics.ucast_pe import (
            UCastPrimitiveEquationConfig,
            UCastPrimitiveEquationModel,
        )
        from legoesm.ml.channel_packing import PE3DChannelSpec

        # The default UCastConfig is sized for the 13-level ERA5 pressure
        # set (4*13+2 = 54 channels).  The driver knows the experiment's
        # actual nlev, so size the network to the PE channel count here —
        # the model ctor fail-fasts on any mismatch.
        base = UCastPrimitiveEquationConfig()
        n_ch = PE3DChannelSpec(nlev=sigma.n_levels).n_channels
        cfg = base._replace(
            ucast_config=base.ucast_config._replace(
                in_channels=n_ch, out_channels=n_ch,
            )
        )
        return UCastPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=cfg)

    # Should be unreachable — the key check above guarantees this.
    raise RuntimeError(f"Internal error: unhandled solver {solver_name!r}")


def _fail_unsupported(model_type: str, discretization: str, grid_type: str):
    """Raise a precise diagnostic for an unsupported combination."""
    # Gather what IS available for each partial match.
    available_for_grid = sorted({
        (k[0], k[1]) for k in _DRIVER_SUPPORTED if k[2] == grid_type
    })
    available_for_model = sorted({
        (k[1], k[2]) for k in _DRIVER_SUPPORTED if k[0] == model_type
    })

    msg_parts = [
        f"Unsupported atmosphere configuration: "
        f"model_type={model_type!r}, discretization={discretization!r}, "
        f"grid_type={grid_type!r}.",
    ]
    if available_for_grid:
        combos = ", ".join(f"({m}, {d})" for m, d in available_for_grid)
        msg_parts.append(
            f"  On grid_type={grid_type!r}, supported (model_type, discretization) "
            f"pairs are: {combos}."
        )
    else:
        all_grids = sorted({k[2] for k in _DRIVER_SUPPORTED})
        msg_parts.append(
            f"  grid_type={grid_type!r} has no driver-supported solvers. "
            f"Supported grid types: {all_grids}."
        )
    if available_for_model:
        combos = ", ".join(f"({d}, {g})" for d, g in available_for_model)
        msg_parts.append(
            f"  For model_type={model_type!r}, supported (discretization, grid_type) "
            f"pairs are: {combos}."
        )
    raise ValueError("\n".join(msg_parts))


# =========================================================================
# Ocean, land, ice, and coupler component factories
# =========================================================================

def create_ocean_component(
    config: ExperimentConfig,
    grid,
    vertical_coord=None,
    *,
    ocean_config=None,
    sst_map=None,
):
    """Create the configured ocean component.

    This factory supports two modes:

    1. **Simple ocean** (default): returns a step-function created by
       ``legoesm.ocean.simple_ocean.make_ocean``.  Pass an optional
       ``SimpleOceanConfig`` via *ocean_config*; if omitted, the
       default fixed-SST configuration is used.
    2. **Full ocean model**: if *ocean_config* is an ``OceanConfig``,
       instantiates a full ``OceanModel`` from ``legoesm.ocean``.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object.
    vertical_coord
        Vertical coordinate (needed for the full ocean model; ignored
        for the simple ocean).
    ocean_config
        A ``SimpleOceanConfig``, an ``OceanConfig``, or an
        ``OceanComplexity`` rung.  If *None*, defaults to
        ``SimpleOceanConfig()`` (fixed SST at 300 K).  An
        ``OceanComplexity`` selects the simple-ocean *mode* via the
        components taxonomy (``fixed_sst``/``slab``/``slab_multilayer``);
        ``OceanComplexity.FULL_3D`` is rejected here because the full 3D
        model needs an explicit ``OceanConfig``.
    sst_map : array-like, optional
        Spatial SST map for fixed/slab modes.

    Returns
    -------
    step_fn or OceanModel
        A callable step function (simple ocean) or an ``OceanModel``
        instance (full ocean).
    """
    from legoesm.components import OceanComplexity, ocean_simple_mode
    from legoesm.ocean.simple_ocean import SimpleOceanConfig, make_ocean
    from legoesm.ocean.state import OceanConfig

    if ocean_config is None:
        ocean_config = SimpleOceanConfig()

    # An OceanComplexity rung selects the simple-ocean mode via the components
    # taxonomy (the single source for fixed_sst/slab/slab_multilayer -> mode).
    # full_3d is a prognostic model needing a full OceanConfig, not a rung.
    if isinstance(ocean_config, OceanComplexity):
        if ocean_config is OceanComplexity.FULL_3D:
            raise ValueError(
                "full_3d ocean is a prognostic 3D model that needs its "
                "parameters: pass ocean_config=OceanConfig(...) rather than "
                "the OceanComplexity.FULL_3D rung."
            )
        ocean_config = SimpleOceanConfig(mode=ocean_simple_mode(ocean_config))

    if isinstance(ocean_config, SimpleOceanConfig):
        logger.info(
            "Ocean: simple mode=%s (h_mix=%.0f m)",
            ocean_config.mode, ocean_config.h_mix,
        )
        return make_ocean(ocean_config, sst_map=sst_map)

    if isinstance(ocean_config, OceanConfig):
        from legoesm.ocean import OceanModel
        # full_3d OceanModel is cubed-sphere-only (it calls create_cubed_sphere_cdgrid).
        # Guard the grid family with a clear message instead of an AttributeError
        # deep inside cdgrid construction (mirrors the resolve_model_complexity guard).
        from legoesm.driver.config import normalize_grid_type
        from legoesm.grids.cubed_sphere import CubedSphereGrid
        # full_3d OceanModel is cubed-sphere-only (it calls
        # create_cubed_sphere_cdgrid).  Detect the family from the GRID OBJECT,
        # not ``config.grid`` — ``config`` is logging-only here and is legitimately
        # ``None`` on the complexity-builder path (test_component_complexity passes
        # config=None with a real grid).  Mirrors the isinstance(grid,
        # CubedSphereGrid) convention used throughout ocean/atmosphere physics.
        if not isinstance(grid, CubedSphereGrid):
            _gt = (
                normalize_grid_type(config.grid.grid_type)
                if config is not None and getattr(config, "grid", None) is not None
                else type(grid).__name__
            )
            raise ValueError(
                f"full_3d (OceanConfig) ocean is implemented only for grid_type in "
                f"{sorted(_FULL_OCEAN_GRID_TYPES)} (the cubed-sphere OceanModel); got "
                f"{_gt!r}. Lat-lon and MPAS full ocean require their grid-specific "
                f"models/configs — that wiring is not yet implemented here."
            )
        if vertical_coord is None:
            raise ValueError(
                "full OceanModel needs a vertical_coord (an ocean z-star "
                "coordinate from legoesm.ocean.vertical.create_ocean_z_star)."
            )
        logger.info("Ocean: full OceanModel")
        # OceanModel's vertical-coordinate parameter is named ``z_coord``.
        return OceanModel(grid=grid, z_coord=vertical_coord, config=ocean_config)

    raise TypeError(
        f"ocean_config must be SimpleOceanConfig or OceanConfig, "
        f"got {type(ocean_config).__name__!r}"
    )


def create_land_component(config: ExperimentConfig, grid, *, land_config=None):
    """Create the configured land surface model.

    Returns a ``step_land`` callable (slab land) or a
    ``step_multilayer_land`` callable, depending on the config type.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (used for determining spatial shape).
    land_config
        A ``LandConfig``, a ``MultiLayerLandConfig``, or a
        ``LandComplexity`` rung.  If *None*, defaults to ``LandConfig()``.
        A ``LandComplexity`` selects the default config of that complexity
        (``slab`` -> ``LandConfig``, ``multilayer``/column ->
        ``MultiLayerLandConfig``).

    Returns
    -------
    step_fn : callable
        The land surface step function.
    """
    from legoesm.components import LandComplexity
    from legoesm.land import (
        LandConfig, MultiLayerLandConfig,
        step_land, step_multilayer_land,
        TwoLeafCanopyConfig,
    )

    if land_config is None:
        land_config = LandConfig()

    # A LandComplexity rung selects the default config of that complexity.
    # Land has no string "mode" (the config *type* is the complexity), so the
    # rung -> config resolution lives here in the factory (which owns the
    # concrete land configs) rather than in the pure components taxonomy.
    if isinstance(land_config, LandComplexity):
        # Explicit, exhaustive rung dispatch with a final raise (CLAUDE.md: never a
        # silent else:default — a new LandComplexity rung must fail loudly here,
        # not silently degrade to the slab LandConfig).  Mirrors ice_complexity_config.
        if land_config is LandComplexity.MULTILAYER:
            land_config = MultiLayerLandConfig()
        elif land_config is LandComplexity.SLAB:
            land_config = LandConfig()
        else:
            raise ValueError(
                f"unhandled LandComplexity rung: {land_config!r} — add its "
                f"config mapping in create_land_component (not yet implemented)."
            )

    if isinstance(land_config, MultiLayerLandConfig):
        scheme_name = type(land_config.surface_scheme).__name__
        logger.info(
            "Land: multilayer model (n_layers=%d, surface_scheme=%s)",
            land_config.soil_grid.n_layers, scheme_name,
        )
        return step_multilayer_land

    if isinstance(land_config, LandConfig):
        scheme_name = type(land_config.surface_scheme).__name__
        logger.info("Land: slab model (surface_scheme=%s)", scheme_name)
        return step_land

    raise TypeError(
        f"land_config must be LandConfig or MultiLayerLandConfig, "
        f"got {type(land_config).__name__!r}"
    )


class SeaIceComponent(NamedTuple):
    """A built sea-ice brick: the step function bound to its resolved config.

    Unlike the ocean (whose step closure binds its config) and land (whose two
    distinct step functions encode the rung), the sea-ice model has a *single*
    step function that takes its config at call time.  Returning the resolved
    ``SeaIceConfig`` alongside ``step`` keeps the selected complexity from being
    silently dropped: the caller wires ``component.step`` with
    ``component.config`` and a dynamic rung cannot degrade to thermodynamic
    defaults downstream.
    """

    step: object  # step_sea_ice(state, forcing, config, dt) -> SeaIceState
    config: object  # the resolved SeaIceConfig to pass to step


def create_ice_component(config: ExperimentConfig, grid, *, ice_config=None):
    """Create the configured sea-ice model.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (needed when ice dynamics are enabled).
    ice_config
        A ``SeaIceConfig`` instance, or an ``IceComplexity`` rung.  If *None*,
        defaults to ``SeaIceConfig()`` (thermodynamic slab).  An
        ``IceComplexity`` is resolved to a ``SeaIceConfig`` via
        :func:`ice_complexity_config`.

    Returns
    -------
    SeaIceComponent
        ``(step, config)`` — ``step_sea_ice`` and the *resolved* ``SeaIceConfig``
        the caller must pass to it (so the selected rung is never dropped).
    """
    from legoesm.components import IceComplexity
    from legoesm.ice import SeaIceConfig, step_sea_ice

    if ice_config is None:
        ice_config = SeaIceConfig()
    if isinstance(ice_config, IceComplexity):
        ice_config = ice_complexity_config(ice_config)

    logger.info(
        "Ice: dynamics=%s, n_categories=%d",
        ice_config.dynamics, ice_config.n_categories,
    )
    return SeaIceComponent(step=step_sea_ice, config=ice_config)


def ice_complexity_config(complexity):
    """Resolve an :class:`~legoesm.components.IceComplexity` rung to a SeaIceConfig.

    ``thermodynamic`` -> the default single-category slab (``SeaIceConfig()``,
    ``dynamics="none"``); ``dynamic`` -> EVP rheology + 5-category CICE ITD
    (``dynamics="evp", n_categories=5``).  Orthogonal sub-physics
    (snow/brine/ridging/ponds) stay at their opt-in defaults — they are gates,
    not rungs.  Lives here (not the pure components taxonomy) because it builds a
    concrete ``SeaIceConfig``, which the driver owns.  Raises ``ValueError`` on an
    unknown rung.
    """
    from legoesm.components import IceComplexity
    from legoesm.ice import SeaIceConfig

    c = IceComplexity(complexity)  # raises ValueError on an unknown rung
    if c is IceComplexity.THERMODYNAMIC:
        return SeaIceConfig()  # default single-category slab (dynamics="none")
    if c is IceComplexity.DYNAMIC:
        return SeaIceConfig(dynamics="evp", n_categories=5)
    # Explicit branches (no implicit default): a future IceComplexity rung added
    # without a mapping here must fail loudly, not silently fall back to slab.
    raise ValueError(f"unhandled IceComplexity rung: {c!r}")


class ModelComplexitySpec(NamedTuple):
    """Concrete, factory-ready inputs for every component at one complexity level.

    The pure components taxonomy (``model_complexity_rungs``) yields complexity
    *enums*; this driver-level resolver turns them into values the component
    factories actually accept — crucially a concrete ``OceanConfig`` for the
    ``full_3d`` ocean (``create_ocean_component`` rejects the bare ``FULL_3D``
    rung because the 3-D model needs its parameters).  So a driver builds every
    component at a chosen complexity through one resolution, full ocean included.
    """

    atmosphere_model_type: str  # -> ExperimentConfig.dycore.model_type
    ocean_config: object  # create_ocean_component accepts: OceanComplexity rung | OceanConfig
    land_config: object  # create_land_component accepts: LandComplexity rung
    ice_config: object  # the resolved SeaIceConfig (pass to step_sea_ice)


#: Grid families whose full-3-D ocean is wired through ``create_ocean_component``
#: today (``OceanConfig`` -> cubed-sphere ``OceanModel``).  Lat-lon C-grid and
#: MPAS full ocean use their own grid-specific models/configs
#: (``LatLonCGridOceanModel`` / MPAS) that are not yet routed through this
#: resolver — see :func:`resolve_model_complexity`.
_FULL_OCEAN_GRID_TYPES: frozenset[str] = frozenset({"cubed_sphere"})


def resolve_model_complexity(level, *, grid_type: str = "cubed_sphere"):
    """Resolve a model-wide complexity *level* to concrete, factory-ready configs.

    ``resolve_model_complexity("full")`` -> a :class:`ModelComplexitySpec` whose
    ``ocean_config`` is an ``OceanConfig`` (the default 3-D ocean; pass an
    explicit ``OceanConfig`` for custom parameters), so every field can be handed
    straight to its factory:

        spec = resolve_model_complexity(level, grid_type=cfg.grid.grid_type)
        dycore   = create_atmosphere_dycore(cfg_with(spec.atmosphere_model_type), grid, sigma)
        ocean    = create_ocean_component(cfg, grid, z_coord, ocean_config=spec.ocean_config)
        land     = create_land_component(cfg, grid, land_config=spec.land_config)
        ice      = create_ice_component(cfg, grid, ice_config=spec.ice_config)

    The simple-ocean and land fields stay as their complexity *rungs* (which the
    factories already resolve), so this resolver does not duplicate that logic; it
    only synthesises the one concrete config the factory cannot derive from a bare
    rung (the full 3-D ocean).

    *grid_type* selects which full-3-D ocean to synthesise.  The full ocean MODEL
    is grid-dependent: only ``"cubed_sphere"`` is wired through
    ``create_ocean_component`` (``OceanConfig`` -> ``OceanModel``, the cdgrid
    ocean).  Lat-lon C-grid and MPAS full ocean use their own grid-specific
    models/configs not yet routed here, so ``full`` complexity on those grids
    raises ``ValueError`` rather than silently synthesising a cubed-sphere
    ``OceanConfig`` that would build the wrong ocean class.  Simple ocean rungs
    are grid-agnostic, so ``idealized``/``intermediate`` work on any grid.  Also
    raises ``ValueError`` on an unknown *level*.
    """
    from legoesm.components import (
        OceanComplexity,
        atmosphere_model_type,
        model_complexity_rungs,
    )
    from legoesm.ocean.state import OceanConfig

    rungs = model_complexity_rungs(level)
    if rungs.ocean is OceanComplexity.FULL_3D:
        if grid_type not in _FULL_OCEAN_GRID_TYPES:
            raise ValueError(
                f"full_3d ocean via resolve_model_complexity is currently wired "
                f"only for grid_type in {sorted(_FULL_OCEAN_GRID_TYPES)} "
                f"(OceanConfig -> cubed-sphere OceanModel); got {grid_type!r}. "
                f"Lat-lon and MPAS full ocean use their grid-specific "
                f"models/configs — build them explicitly."
            )
        # Full-3D cube ocean uses the FV3 C-D grid (the only cube backend) with
        # the FV3-faithful fv3sw barotropic — NOT the default a_grid barotropic
        # (forbidden by the never-A-grid / FV3-faithfulness directive).  Import
        # the SW core provider so its registry entry exists before construction.
        import legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid  # noqa: F401
        ocean_config = OceanConfig(barotropic_staggering="fv3sw")
    else:
        ocean_config = rungs.ocean  # simple rung — grid-agnostic, factory resolves
    return ModelComplexitySpec(
        atmosphere_model_type=atmosphere_model_type(rungs.atmosphere),
        ocean_config=ocean_config,
        land_config=rungs.land,
        ice_config=ice_complexity_config(rungs.ice),
    )
