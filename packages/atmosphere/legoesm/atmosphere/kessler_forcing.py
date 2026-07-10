"""Kessler warm-rain microphysics as an MPAS operator-split physics forcing.

Adapter that applies the shared column-local Kessler scheme
(:func:`legoesm.atmosphere.physics.microphysics.kessler.kessler_microphysics`)
to an MPAS / Voronoi hydrostatic state and returns physics tendencies in the
operator-split ``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
convention consumed by ``MPASPrimitiveEquationModel._step_jit`` (single rank)
and ``make_voronoi_mpi_step`` (multi rank).  No radiation, no convection — this
is the moisture + condensation forcing for a *moist baroclinic wave*.

Why an adapter and not a new scheme: Kessler itself
(``physics/microphysics/kessler.py``) is purely COLUMN-local — it operates on
``(ncol, nlev)`` arrays with no horizontal coupling.  So this wrapper only has
to (1) pull ``q_v``/``q_c``/``q_r`` out of ``state.tracers``, (2) build the
column thermodynamic inputs Kessler needs (``p_full``, ``p_half``, ``rho``,
``dz``) from the SHARED helpers in ``atmosphere.physics._shared`` and
``grids.vertical`` (nothing re-derived), and (3) repackage the column output as
``MPASHydrostaticTendencies``.  Because the physics is column-local, ``mesh`` is
accepted for signature compliance but unused, and the forcing is correct
per-rank under MPI with NO extra halo exchange — the dycore already
halo-exchanges and advects the tracers mass-consistently.

Operator-split timing: the MPAS step evaluates ``physics_fn`` once on the
post-dynamics state and applies the tendencies forward over ``dt``
(``state += dt · tendency``; see ``_step_jit``).  Kessler defines its
saturation adjustment as an increment/``dt`` rate, so a single forward
application over the SAME ``dt`` recovers the intended adjustment.  ``dt`` is
not part of the ``physics_fn`` signature, so it is bound at build time via
:func:`make_kessler_forcing_mpas`.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticTendencies, MPASHydrostaticTendencies
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
)
from legoesm.atmosphere.physics._shared import compute_layer_dz, compute_rho
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors

_DIMS_CELL = ("nCells", "level")
_DIMS_EDGE = ("nEdges", "level")
_DIMS_CELL_2D = ("nCells",)
_DIMS_LATLON_3D = ("lat", "lon", "level")
_DIMS_LATLON_2D = ("lat", "lon")
_DIMS_CUBE_3D = ("face", "x", "y", "level")
_DIMS_CUBE_2D = ("face", "x", "y")
_REQUIRED_TRACERS = ("q_v", "q_c", "q_r")


def kessler_column_tendencies(T, p_s, q_v, q_c, q_r, sigma_coord, *, dt, config):
    """Shared column-local Kessler tendencies for every grid adapter.

    The grid-agnostic core of the warm-rain forcing: it lives in ONE place so
    the moisture numerics are identical across the MPAS, spectral and lat-lon
    adapters (each only flattens its horizontal to ``(ncol, nlev)``, calls this,
    and repackages the rates into its grid-specific tendency container).

    Pure ``(ncol, nlev)`` column physics:
      * clips the tracer INPUTS to ``>= 0`` (tracer advection is not
        positive-definite and the dycores apply the prognostic floor only AFTER
        physics, so a tiny post-advection undershoot must not feed a negative
        ``q_c`` into Kessler's accretion);
      * builds the column thermo from the SHARED helpers
        (``pressure_from_sigma`` / ``compute_rho`` / ``compute_layer_dz``);
      * runs the shared ``kessler_microphysics`` core.

    Parameters
    ----------
    T : jax.Array, shape ``(ncol, nlev)``
        Temperature [K].
    p_s : jax.Array, shape ``(ncol,)``
        Surface pressure [Pa].
    q_v, q_c, q_r : jax.Array, shape ``(ncol, nlev)``
        Vapor / cloud / rain mixing ratios [kg/kg] (raw; clipped here).
    sigma_coord : SigmaCoordinate
        Sigma vertical coordinate (callers reject hybrid).
    dt : float
        Physics step [s].
    config : KesslerConfig

    Returns
    -------
    (dT_dt, dq_v_dt, dq_c_dt, dq_r_dt) : tuple of jax.Array ``(ncol, nlev)``
        Latent-heating and vapor / cloud / rain mixing-ratio rates.
    """
    q_v = jnp.maximum(q_v, 0.0)
    q_c = jnp.maximum(q_c, 0.0)
    q_r = jnp.maximum(q_r, 0.0)

    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)   # (ncol, nlev)
    p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)   # (ncol, nlev+1)
    rho = compute_rho(T, p_full, q_v)        # moist ideal-gas density
    dz = compute_layer_dz(T, p_half, q_v)    # moist hypsometric thickness

    ncol, nlev = T.shape
    hydro = make_zero_hydrometeors(ncol, nlev, dtype=T.dtype)._replace(
        q_c=q_c, q_r=q_r,
    )
    out = kessler_microphysics(
        T=T, q_v=q_v, hydrometeors=hydro,
        p_full=p_full, p_half=p_half, rho=rho, dz=dz,
        dt=dt, config=config,
    )
    return out.dT_dt, out.dq_v_dt, out.dq_c_dt, out.dq_r_dt


def make_kessler_forcing_mpas(dt, config: KesslerConfig | None = None):
    """Build an operator-split MPAS ``physics_fn`` applying Kessler over ``dt``.

    Parameters
    ----------
    dt : float
        Physics step [s].  Bound into the returned closure because the
        dycore's ``physics_fn`` convention passes no timestep.  Must be > 0.

        CONTRACT: the MPAS step applies these tendencies forward with the step's
        OWN ``dt`` (``state += dt_step * tendency``).  Kessler's saturation
        adjustment is an increment/``dt``-here rate, so the caller MUST step the
        model with the SAME ``dt`` used here, or the adjustment is mis-scaled by
        ``dt_step / dt_here``.  (The non-saturation processes are true
        dt-independent rates and are unaffected.)  The benchmark builds and
        steps with one ``dt`` per case, so they always match; the returned
        callable carries ``_bound_dt`` so a wrapper can assert this.
    config : KesslerConfig, optional
        Microphysics configuration (defaults to ``KesslerConfig()``).

    Returns
    -------
    callable
        ``physics_fn(state, mesh, sigma_coord, *, phys_state=None,
        forcing=None) -> MPASHydrostaticTendencies`` carrying ``dT_dt`` (latent
        heating) and ``tracer_tendencies`` for ``q_v``/``q_c``/``q_r``.  Wind
        and surface-pressure tendencies are zero (warm-rain microphysics has no
        direct momentum or surface-pressure source).
    """
    cfg = config if config is not None else KesslerConfig()
    if not (dt > 0.0):
        raise ValueError(f"kessler forcing dt must be > 0, got {dt!r}")
    dt = float(dt)

    def kessler_forcing_mpas(
        state, mesh, sigma_coord, *, phys_state=None, forcing=None,
    ):
        if state.tracers is None or any(
            k not in state.tracers for k in _REQUIRED_TRACERS
        ):
            have = None if state.tracers is None else sorted(state.tracers)
            raise ValueError(
                "kessler_forcing_mpas requires state.tracers with keys "
                f"{_REQUIRED_TRACERS}; got {have}. Initialize the state with "
                "baroclinic_wave_init_mpas(..., moist=True) (or attach the "
                "tracers before stepping)."
            )
        if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
            raise NotImplementedError(
                "kessler_forcing_mpas supports sigma coordinates only "
                "(half-level hybrid pressure is not wired); got "
                f"{type(sigma_coord).__name__}."
            )

        # MPAS state is already (nCells, nlev) = (ncol, nlev): no flatten.
        # The shared column core clips the tracer inputs to >=0 and runs the
        # column thermo + microphysics (see kessler_column_tendencies).
        T = state.T.data                  # (nCells, nlev)
        p_s = state.p_s.data              # (nCells,)
        dT_dt, dq_v_dt, dq_c_dt, dq_r_dt = kessler_column_tendencies(
            T, p_s,
            state.tracers["q_v"].data,
            state.tracers["q_c"].data,
            state.tracers["q_r"].data,
            sigma_coord, dt=dt, config=cfg,
        )

        zeros_edge = jnp.zeros_like(state.u.data)
        zeros_ps = jnp.zeros_like(p_s)
        return MPASHydrostaticTendencies(
            du_dt=Field(data=zeros_edge, name="du_dt_kessler",
                        dims=_DIMS_EDGE, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_kessler",
                        dims=_DIMS_CELL, units="K/s"),
            dp_s_dt=Field(data=zeros_ps, name="dp_s_dt_kessler",
                          dims=_DIMS_CELL_2D, units="Pa/s"),
            dphis_dt=Field(data=zeros_ps, name="dphis_dt_kessler",
                           dims=_DIMS_CELL_2D, units="m^2/s^3"),
            tracer_tendencies={
                "q_v": Field(data=dq_v_dt, name="dq_v_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
                "q_c": Field(data=dq_c_dt, name="dq_c_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
                "q_r": Field(data=dq_r_dt, name="dq_r_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
            },
        )

    # Introspectable bound timestep so a step wrapper can assert the caller
    # steps with the same dt the saturation-adjustment rate was scaled by.
    kessler_forcing_mpas._bound_dt = dt
    # Warm-rain microphysics is purely column-local (reads only its own column),
    # so the MPI step may SKIP the pre-physics halo exchange for it.
    kessler_forcing_mpas._column_local = True
    return kessler_forcing_mpas


def make_kessler_forcing_spectral(dt, config: KesslerConfig | None = None):
    """Build a spectral-PE ``physics_fn`` applying Kessler warm-rain over ``dt``.

    Mirrors :func:`make_kessler_forcing_mpas` for the global spectral primitive
    equation dycore (:class:`SpectralPrimitiveEquationModel`).  The dycore's
    ``physics_fn`` contract is ``physics_fn(state, grid, sigma_coord,
    forcing_data)`` returning a ``SpectralHydrostaticState`` whose fields are
    *tendencies*: the spectral tendency adds ``.T_hat`` (spectral) and
    ``.tracers[name]`` (grid-space) to the dynamics tendency each RK stage.

    The microphysics is grid-agnostic and column-local, so this adapter only
    bridges representations:
      * inverse-SH ``T_hat`` -> grid temperature and ``p_s = exp(synthesis(
        lnps_hat))`` (tracers ``q_v``/``q_c``/``q_r`` are ALREADY grid-space in
        ``state.tracers``);
      * flatten the horizontal ``(n_lat, n_lon, nlev) -> (ncol, nlev)`` so the
        SAME shared column thermo (``pressure_from_sigma`` / ``compute_rho`` /
        ``compute_layer_dz``) and ``kessler_microphysics`` core used by the MPAS
        adapter apply unchanged;
      * forward-SH the latent-heating rate ``dT/dt`` back to ``T_hat`` (sign as
        produced by Kessler — condensation warms; identical convention to the
        MPAS path which returns ``dT_dt`` directly); tracer rates stay grid-space.

    CONTRACT (same as MPAS): ``dt`` is bound into the closure (the physics_fn
    convention passes no timestep) and the caller MUST step the model with the
    SAME ``dt`` — Kessler's saturation adjustment is an increment/``dt`` rate.
    ``_bound_dt`` carries it for assertion.  Spectral is SINGLE-DEVICE (no MPI);
    this is a physics-capability adapter, not a scaling path.

    Parameters
    ----------
    dt : float
        Physics step [s].  Must be > 0.
    config : KesslerConfig, optional

    Returns
    -------
    callable
        ``physics_fn(state, grid, sigma_coord, forcing_data=None) ->
        SpectralHydrostaticState`` tendency (zero vor/div/lnps/phis tendencies;
        ``T_hat`` latent heating; grid-space ``q_v``/``q_c``/``q_r`` rates).
    """
    cfg = config if config is not None else KesslerConfig()
    if not (dt > 0.0):
        raise ValueError(f"kessler forcing dt must be > 0, got {dt!r}")
    dt = float(dt)

    def kessler_forcing_spectral(state, grid, sigma_coord, forcing_data=None):
        # Deferred imports: SpectralHydrostaticState + the Gaussian SH transforms
        # live in heavier modules; keep them out of module import time and avoid
        # any dynamics<->forcing import cycle.
        from legoesm.grids.gaussian import (
            sh_analysis_3d, sh_synthesis, sh_synthesis_3d,
        )

        if state.tracers is None or any(
            k not in state.tracers for k in _REQUIRED_TRACERS
        ):
            have = None if state.tracers is None else sorted(state.tracers)
            raise ValueError(
                "kessler_forcing_spectral requires state.tracers with keys "
                f"{_REQUIRED_TRACERS}; got {have}. Initialize the spectral state "
                "with a moist q_v (and zero q_c/q_r) before stepping."
            )
        if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
            raise NotImplementedError(
                "kessler_forcing_spectral supports sigma coordinates only "
                "(half-level hybrid pressure is not wired); got "
                f"{type(sigma_coord).__name__}."
            )

        # Spectral -> grid for the prognostics the column scheme needs.
        T_grid = sh_synthesis_3d(grid, state.T_hat.data)   # (n_lat, n_lon, nlev)
        lnps_grid = sh_synthesis(grid, state.lnps_hat.data)  # (n_lat, n_lon)
        p_s_grid = jnp.exp(lnps_grid)
        n_lat, n_lon, nlev = T_grid.shape

        # Flatten the horizontal so the SHARED (ncol, nlev) column core applies
        # unchanged (same contract as the MPAS / lat-lon adapters).  Tracers are
        # grid-space; the core clips the physics INPUT to >=0.
        T = T_grid.reshape(-1, nlev)
        p_s = p_s_grid.reshape(-1)
        dT_dt, dq_v_dt, dq_c_dt, dq_r_dt = kessler_column_tendencies(
            T, p_s,
            state.tracers["q_v"].data.reshape(-1, nlev),
            state.tracers["q_c"].data.reshape(-1, nlev),
            state.tracers["q_r"].data.reshape(-1, nlev),
            sigma_coord, dt=dt, config=cfg,
        )

        # Unflatten back to the spectral grid layout.
        dT_dt_grid = dT_dt.reshape(n_lat, n_lon, nlev)
        dq_v_grid = dq_v_dt.reshape(n_lat, n_lon, nlev)
        dq_c_grid = dq_c_dt.reshape(n_lat, n_lon, nlev)
        dq_r_grid = dq_r_dt.reshape(n_lat, n_lon, nlev)

        # Latent heating -> spectral T tendency; momentum / surface pressure have
        # no warm-rain source (zero spectral tendencies).
        dT_hat = sh_analysis_3d(grid, dT_dt_grid)
        return state.__class__(
            vor_hat=state.vor_hat.replace(data=jnp.zeros_like(state.vor_hat.data)),
            div_hat=state.div_hat.replace(data=jnp.zeros_like(state.div_hat.data)),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(
                data=jnp.zeros_like(state.lnps_hat.data)),
            phis_hat=state.phis_hat.replace(
                data=jnp.zeros_like(state.phis_hat.data)),
            tracers={
                "q_v": state.tracers["q_v"].replace(data=dq_v_grid),
                "q_c": state.tracers["q_c"].replace(data=dq_c_grid),
                "q_r": state.tracers["q_r"].replace(data=dq_r_grid),
            },
        )

    kessler_forcing_spectral._bound_dt = dt
    kessler_forcing_spectral._column_local = True
    return kessler_forcing_spectral


def make_kessler_forcing_gridspace(
    dt, *, dims_3d, dims_2d, config: KesslerConfig | None = None,
):
    """Build a cell-centred grid-space Kessler ``physics_fn`` (any FV grid).

    The SHARED warm-rain adapter for every dycore whose ``physics_fn`` receives
    a cell-centred :class:`HydrostaticState` carrying grid-space tracer
    ``Field``\\ s and consumes a :class:`HydrostaticTendencies` — currently the
    lat-lon Arakawa C-grid (:class:`CGridLatLonPrimitiveEquationModel`, dims
    ``("lat","lon","level")``) and the cubed-sphere FV3 C-D grid
    (:class:`CDGridPrimitiveEquationModel`, dims ``("face","x","y","level")``).
    Both call ``physics_fn(state, grid, sigma_coord)`` per RK stage and ADD the
    returned tendency, including ``tracer_tendencies[name]`` for any tracer
    already prognostic in the state; the dynamics advects ``q_v``/``q_c``/``q_r``
    so this adapter supplies only the column microphysics source.

    Grid-agnostic + column-local, so it just flattens the leading horizontal
    axes to ``(ncol, nlev)`` via ``reshape(-1, nlev)`` (works for the lat-lon 3-D
    ``(n_lat,n_lon,nlev)`` and the cube 4-D ``(6,n,n,nlev)`` layouts alike), runs
    the SAME shared column core (:func:`kessler_column_tendencies`), then
    reshapes the rates back to the state's own shape.  The thin
    :func:`make_kessler_forcing_latlon` / :func:`make_kessler_forcing_cube`
    wrappers bind ``dims_3d`` / ``dims_2d`` for their grid.

    CONTRACT (same as MPAS / spectral): ``dt`` is bound into the closure (the
    ``physics_fn`` convention passes no timestep) and the caller MUST step the
    model with the SAME ``dt`` — Kessler's saturation adjustment is an
    increment/``dt`` rate.  ``_bound_dt`` carries it for assertion.  Warm-rain
    is column-local (``_column_local = True``), so an MPI step may skip a
    pre-physics halo exchange for it.

    Parameters
    ----------
    dt : float
        Physics step [s].  Must be > 0.
    dims_3d : tuple[str, ...]
        Dim labels for the returned 3-D tendency Fields (e.g.
        ``("lat","lon","level")`` or ``("face","x","y","level")``).
    dims_2d : tuple[str, ...]
        Dim labels for the returned 2-D tendency Fields.
    config : KesslerConfig, optional

    Returns
    -------
    callable
        ``physics_fn(state, grid, sigma_coord, *, phys_state=None,
        forcing=None) -> HydrostaticTendencies`` carrying ``dT_dt`` (latent
        heating) and ``tracer_tendencies`` for ``q_v``/``q_c``/``q_r``.  Wind
        (``du_dt``/``dv_dt``), surface-pressure and geopotential tendencies are
        zero (warm-rain microphysics has no momentum / surface source).
    """
    cfg = config if config is not None else KesslerConfig()
    if not (dt > 0.0):
        raise ValueError(f"kessler forcing dt must be > 0, got {dt!r}")
    dt = float(dt)

    def kessler_forcing_gridspace(
        state, grid, sigma_coord, *, phys_state=None, forcing=None,
    ):
        if state.tracers is None or any(
            k not in state.tracers for k in _REQUIRED_TRACERS
        ):
            have = None if state.tracers is None else sorted(state.tracers)
            raise ValueError(
                "kessler_forcing requires state.tracers with keys "
                f"{_REQUIRED_TRACERS}; got {have}. Initialize the state with "
                "the moist baroclinic-wave init (baroclinic_wave_init[_latlon]"
                "(..., moist=True)) or attach the tracers before stepping."
            )
        if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
            raise NotImplementedError(
                "kessler_forcing supports sigma coordinates only "
                "(half-level hybrid pressure is not wired); got "
                f"{type(sigma_coord).__name__}."
            )

        T_grid = state.T.data            # (..., nlev) cell-centred
        p_s_grid = state.p_s.data        # (...) cell-centred
        out_shape = T_grid.shape
        nlev = out_shape[-1]

        # Flatten the leading horizontal axes so the SHARED (ncol, nlev) column
        # core applies unchanged (same contract as MPAS / spectral).  The core
        # clips the physics INPUT to >=0.
        T = T_grid.reshape(-1, nlev)
        p_s = p_s_grid.reshape(-1)
        dT_dt_col, dq_v_col, dq_c_col, dq_r_col = kessler_column_tendencies(
            T, p_s,
            state.tracers["q_v"].data.reshape(-1, nlev),
            state.tracers["q_c"].data.reshape(-1, nlev),
            state.tracers["q_r"].data.reshape(-1, nlev),
            sigma_coord, dt=dt, config=cfg,
        )

        # Unflatten the rates back to the state's own grid layout.
        dT_dt = dT_dt_col.reshape(out_shape)
        dq_v = dq_v_col.reshape(out_shape)
        dq_c = dq_c_col.reshape(out_shape)
        dq_r = dq_r_col.reshape(out_shape)

        zeros_3d = jnp.zeros_like(T_grid)
        zeros_2d = jnp.zeros_like(p_s_grid)
        return HydrostaticTendencies(
            du_dt=Field(data=zeros_3d, name="du_dt_kessler",
                        dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_kessler",
                        dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=zeros_2d, name="dp_s_dt_kessler",
                          dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=zeros_2d, name="dphis_dt_kessler",
                           dims=dims_2d, units="m^2/s^3"),
            dv_dt=Field(data=zeros_3d, name="dv_dt_kessler",
                        dims=dims_3d, units="m/s^2"),
            tracer_tendencies={
                "q_v": Field(data=dq_v, name="dq_v_dt_kessler",
                             dims=dims_3d, units="kg/kg/s"),
                "q_c": Field(data=dq_c, name="dq_c_dt_kessler",
                             dims=dims_3d, units="kg/kg/s"),
                "q_r": Field(data=dq_r, name="dq_r_dt_kessler",
                             dims=dims_3d, units="kg/kg/s"),
            },
        )

    kessler_forcing_gridspace._bound_dt = dt
    kessler_forcing_gridspace._column_local = True
    return kessler_forcing_gridspace


def make_kessler_forcing_latlon(dt, config: KesslerConfig | None = None):
    """Lat-lon C-grid Kessler ``physics_fn`` — thin wrapper over
    :func:`make_kessler_forcing_gridspace` binding the lat-lon dim labels.

    The lat-lon C-grid dycore already advects ``q_v``/``q_c``/``q_r`` in
    mass-weighted flux form (``cgrid_latlon_hydrostatic_tendencies``), so this
    supplies only the column microphysics source.
    """
    return make_kessler_forcing_gridspace(
        dt, dims_3d=_DIMS_LATLON_3D, dims_2d=_DIMS_LATLON_2D, config=config,
    )


def make_kessler_forcing_cube(dt, config: KesslerConfig | None = None):
    """Cubed-sphere FV3 C-D grid Kessler ``physics_fn`` — thin wrapper over
    :func:`make_kessler_forcing_gridspace` binding the cube dim labels.

    The FV3 C-D grid dycore advects ``q_v``/``q_c``/``q_r`` in advective form
    (consistent with its temperature transport); this supplies only the column
    microphysics source.
    """
    return make_kessler_forcing_gridspace(
        dt, dims_3d=_DIMS_CUBE_3D, dims_2d=_DIMS_CUBE_2D, config=config,
    )


def make_kessler_column_physics_fn(sigma_coord, dt,
                                   config: KesslerConfig | None = None):
    """Per-tile Kessler ``column_physics_fn`` for the TILED cube steps.

    The tiled moist steps (``make_tiled_fv3_hydrostatic_moist_step_stage_2d``
    / ``make_tiled_fv3_hydrostatic_step_blocked_2d``) inject column physics
    with the raw-array contract ``fn(T_t, p_s_t, q_v, q_c, q_r) -> (dT,
    dq_v, dq_c, dq_r)`` (each leading-shape-preserving, tile-local).  This
    factory is the Kessler bridge for that contract: it flattens the tile's
    horizontal axes to ``(ncol, nlev)``, runs the SAME shared
    :func:`kessler_column_tendencies` core that
    :func:`make_kessler_forcing_cube` (the np<=6 ``physics_fn`` lane) uses,
    and reshapes back — so the face-sharded and sub-face-tiled lanes run
    IDENTICAL microphysics (the controlled-comparison requirement for the
    cube device ladder).

    Same ``dt`` contract as every Kessler adapter: bound into the closure;
    the caller MUST step with this ``dt`` (saturation adjustment is an
    increment/``dt`` rate).
    """
    if not (float(dt) > 0.0):
        raise ValueError(
            f"make_kessler_column_physics_fn: dt must be a positive physics "
            f"step [s]; got {dt}")
    cfg = config if config is not None else KesslerConfig()

    def fn(T_t, p_s_t, q_v, q_c, q_r):
        shp = T_t.shape
        nlev = shp[-1]
        dT, dq_v, dq_c, dq_r = kessler_column_tendencies(
            T_t.reshape(-1, nlev), p_s_t.reshape(-1),
            q_v.reshape(-1, nlev), q_c.reshape(-1, nlev),
            q_r.reshape(-1, nlev),
            sigma_coord, dt=dt, config=cfg)
        return (dT.reshape(shp), dq_v.reshape(shp), dq_c.reshape(shp),
                dq_r.reshape(shp))

    return fn
