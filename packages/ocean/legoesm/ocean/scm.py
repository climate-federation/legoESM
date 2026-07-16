"""Single-column model (SCM) driver for the legoESM ocean.

A minimal, dynamics-free driver that exercises the ocean *column* physics
pipeline on a single water column.  It is the ocean counterpart of
:class:`legoesm.atmosphere.forcing.scm.scm.SingleColumnModel` and is built for the same
purposes: parameterization integration tests, vertical-mixing / convection
process studies, and stability sweeps.

Shared strategy with the atmosphere SCM
---------------------------------------
The two SCMs deliberately share one strategy so a user can drive either the
same way:

* **Time stepping** — the explicit integrators (forward Euler, RK2, RK4,
  AB2 first-step) and the integrator registry live once in
  :mod:`legoesm.core.column_stepping` and are reused here; only the
  ocean-specific ``_apply_ocean_tendencies`` / ``_average_ocean_tendencies``
  state operators are injected.
* **Column physics** — assembled by the canonical ocean physics factory
  :func:`legoesm.ocean.physics.combined.make_ocean_physics` (KPP / Richardson
  / constant vertical mixing, convective adjustment, optional shortwave
  penetration), exactly as in the 3-D model — no scheme is re-implemented.
* **External forcing** — a closure-style :class:`OceanSCMForcing`
  (wind stress, surface heat / freshwater flux, Coriolis) mirrors the
  atmosphere's ``SCMForcing``.

Vertical mixing
---------------
The default (``implicit_vertical_mixing=True``) mirrors the 3-D ocean's
``implicit_vertical_mixing`` path and is the unconditionally-stable choice:

  1. column physics runs with ``apply_vertical_diffusion=False`` and returns
     the *explicit* remainder (surface fluxes, Coriolis, KPP non-local
     counter-gradient flux, convective non-local term) plus the ``K_v`` /
     ``A_v`` interface profiles;
  2. that explicit tendency is advanced one forward-Euler step;
  3. a backward-Euler tridiagonal solve
     (:func:`implicit_vertical_diffusion_ocean`) applies the vertical
     diffusion of ``T, S, u, v`` with zero-flux boundaries.

This removes the explicit-diffusion CFL limit ``dt < dz²/(2K)`` that the
large convective diffusivity (``K_conv ≈ 1 m²/s``) would otherwise impose.
Setting ``implicit_vertical_mixing=False`` reproduces the atmosphere-SCM
structure exactly (pure explicit integration with any registered
integrator), at the cost of that CFL limit.

Level convention
----------------
Ocean columns are indexed **surface-to-bottom**: ``k=0`` is the surface,
``k=nlev-1`` is the deepest layer (opposite the atmosphere's top-to-bottom
ordering, but each is internally self-consistent).

Example
-------
>>> import jax.numpy as jnp
>>> from legoesm.ocean.scm import OceanColumnModel
>>> from legoesm.ocean.scm_forcing import OceanSCMForcing
>>> from legoesm.ocean.physics.combined import OceanPhysicsConfig
>>> from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
>>> cfg = OceanPhysicsConfig(
...     vertical_mixing=VerticalMixingConfig(scheme="kpp"),
... )
>>> forcing = OceanSCMForcing(f_c=1e-4, tau_x=lambda t: 0.1)
>>> scm = OceanColumnModel.create(
...     physics_config=cfg, nlev=40, dt=900.0,
...     T_profile=jnp.linspace(18.0, 4.0, 40),  # warm surface, cold deep
...     forcing=forcing,
... )
>>> final_state, history = scm.run(nsteps=192, save_every=8)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.column_stepping import (
    ab2_effective_tendency,
    build_explicit_integrators,
    register_integrator,
)
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_ocean_jacobian,
    create_ocean_z_star,
)
from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
from legoesm.ocean.physics.vertical_mixing import (
    build_dz_half,
    implicit_vertical_diffusion_ocean,
)
from legoesm.ocean.scm_forcing import (
    OceanSCMForcing,
    build_ocean_surface_forcing_struct,
    compute_ocean_forcing_tendencies,
    default_ocean_forcing,
    validate_ocean_forcing,
)


_DIMS_3D = ("face", "x", "y", "level")
_DIMS_2D = ("face", "x", "y")


class OceanSCMGrid(NamedTuple):
    """Minimal grid object for the single-column ocean model.

    Exposes the ``grid_lat`` / ``grid_lon`` / ``radius`` attributes that
    the prescribed-surface-forcing wind-stress helper reads.  Shape is
    ``(1, 1, 1)`` so column reshapes give ``ncol = 1``.
    """
    grid_lat: jax.Array  # (1, 1, 1) [rad]
    grid_lon: jax.Array  # (1, 1, 1) [rad]
    radius: float

    @property
    def grid_n_columns(self) -> int:
        return 1

    @property
    def grid_shape_2d(self) -> tuple[int, int, int]:
        return (1, 1, 1)


def make_ocean_scm_grid(
    latitude_deg: float, longitude_deg: float = 0.0,
) -> OceanSCMGrid:
    """Build a single-column ocean grid at the given latitude/longitude."""
    lat = jnp.asarray([[[jnp.deg2rad(latitude_deg)]]])
    lon = jnp.asarray([[[jnp.deg2rad(longitude_deg)]]])
    return OceanSCMGrid(grid_lat=lat, grid_lon=lon, radius=constants.R_earth)


def make_ocean_column_state(
    nlev: int,
    *,
    T_profile: jax.Array,
    z_coord: OceanZStarCoordinate,
    S_profile: jax.Array | float | None = None,
    u: float = 0.0,
    v: float = 0.0,
    eta: float = 0.0,
    H_bathy: float | None = None,
    dtype=None,
) -> OceanState:
    """Construct an :class:`OceanState` of shape ``(1, 1, 1, nlev)``.

    Profiles are indexed surface-to-bottom (``k=0`` = surface, ``k=nlev-1``
    = deepest), matching :class:`OceanZStarCoordinate`.

    Parameters
    ----------
    T_profile
        Potential-temperature profile [degC], shape ``(nlev,)``.
    z_coord
        Vertical coordinate; ``z_coord.H_max`` is the default bathymetry
        (a flat, full-depth single column).
    S_profile
        Salinity profile [PSU], shape ``(nlev,)``, or a scalar broadcast
        to every level.  Defaults to a uniform 35 PSU.
    u, v
        Initial uniform velocity components [m/s].
    eta
        Initial sea-surface height [m].
    H_bathy
        Column depth [m] (positive).  Defaults to ``z_coord.H_max``.
    """
    if dtype is None:
        dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    T_arr = jnp.asarray(T_profile, dtype=dtype)
    if T_arr.size != nlev:
        raise ValueError(
            f"T_profile must have {nlev} elements, got {T_arr.size}"
        )
    T_data = T_arr.reshape(1, 1, 1, nlev)

    if S_profile is None:
        S_data = jnp.full((1, 1, 1, nlev), 35.0, dtype=dtype)
    else:
        S_arr = jnp.asarray(S_profile, dtype=dtype)
        if S_arr.ndim == 0:
            S_data = jnp.full((1, 1, 1, nlev), S_arr, dtype=dtype)
        else:
            if S_arr.size != nlev:
                raise ValueError(
                    f"S_profile must have {nlev} elements, got {S_arr.size}"
                )
            S_data = S_arr.reshape(1, 1, 1, nlev)

    u_data = jnp.full((1, 1, 1, nlev), u, dtype=dtype)
    v_data = jnp.full((1, 1, 1, nlev), v, dtype=dtype)
    eta_data = jnp.full((1, 1, 1), eta, dtype=dtype)
    H = z_coord.H_max if H_bathy is None else H_bathy
    H_data = jnp.full((1, 1, 1), H, dtype=dtype)
    mask_data = jnp.ones((1, 1, 1), dtype=dtype)

    return OceanState(
        u=Field(data=u_data, name="u", dims=_DIMS_3D, units="m/s"),
        v=Field(data=v_data, name="v", dims=_DIMS_3D, units="m/s"),
        T=Field(data=T_data, name="T", dims=_DIMS_3D, units="degC"),
        S=Field(data=S_data, name="S", dims=_DIMS_3D, units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=_DIMS_2D, units="m"),
        H_bathy=Field(data=H_data, name="H_bathy", dims=_DIMS_2D, units="m"),
        land_mask=Field(data=mask_data, name="land_mask", dims=_DIMS_2D, units="1"),
    )


# ---------------------------------------------------------------------------
# State-specific tendency operators (injected into the shared integrators)
# ---------------------------------------------------------------------------


def _apply_ocean_tendencies(
    state: OceanState, tend: OceanTendencies, dt: float,
) -> OceanState:
    """Apply ``state + dt·tend`` for the prognostic ocean column fields.

    Static fields (``H_bathy``, ``land_mask``) are carried unchanged.  No
    clipping is applied — schemes that need positivity (e.g. salinity)
    must enforce it themselves; clipping here would hide instability and
    break AD smoothness.
    """
    return state._replace(
        u=state.u.replace(data=state.u.data + dt * tend.du_dt.data),
        v=state.v.replace(data=state.v.data + dt * tend.dv_dt.data),
        T=state.T.replace(data=state.T.data + dt * tend.dT_dt.data),
        S=state.S.replace(data=state.S.data + dt * tend.dS_dt.data),
        eta=state.eta.replace(data=state.eta.data + dt * tend.deta_dt.data),
    )


def _average_ocean_tendencies(*tends, weights) -> OceanTendencies:
    """Linear combination of :class:`OceanTendencies` with given weights.

    Combines the prognostic channels (``du, dv, dT, dS, deta``); ``K_v`` /
    ``A_v`` are dropped (they are only consumed by the implicit-mixing
    step, which never goes through a multi-stage integrator).
    """
    if len(weights) != len(tends):
        raise ValueError("weights length must match number of tendencies")

    def combine(field_list):
        return sum(w * f.data for w, f in zip(weights, field_list))

    first = tends[0]
    return OceanTendencies(
        du_dt=first.du_dt.replace(data=combine([t.du_dt for t in tends])),
        dv_dt=first.dv_dt.replace(data=combine([t.dv_dt for t in tends])),
        dT_dt=first.dT_dt.replace(data=combine([t.dT_dt for t in tends])),
        dS_dt=first.dS_dt.replace(data=combine([t.dS_dt for t in tends])),
        deta_dt=first.deta_dt.replace(data=combine([t.deta_dt for t in tends])),
        dH_bathy_dt=first.dH_bathy_dt.replace(
            data=jnp.zeros_like(first.dH_bathy_dt.data),
        ),
        dland_mask_dt=first.dland_mask_dt.replace(
            data=jnp.zeros_like(first.dland_mask_dt.data),
        ),
    )


def add_ocean_tendencies(
    a: OceanTendencies, b: OceanTendencies,
) -> OceanTendencies:
    """Field-wise sum of two :class:`OceanTendencies`.

    ``K_v`` / ``A_v`` from whichever operand carries them are preserved
    (physics populates them; forcing leaves them ``None``).
    """
    K_v = a.K_v if a.K_v is not None else b.K_v
    A_v = a.A_v if a.A_v is not None else b.A_v
    return OceanTendencies(
        du_dt=a.du_dt.replace(data=a.du_dt.data + b.du_dt.data),
        dv_dt=a.dv_dt.replace(data=a.dv_dt.data + b.dv_dt.data),
        dT_dt=a.dT_dt.replace(data=a.dT_dt.data + b.dT_dt.data),
        dS_dt=a.dS_dt.replace(data=a.dS_dt.data + b.dS_dt.data),
        deta_dt=a.deta_dt.replace(data=a.deta_dt.data + b.deta_dt.data),
        dH_bathy_dt=a.dH_bathy_dt,
        dland_mask_dt=a.dland_mask_dt,
        K_v=K_v,
        A_v=A_v,
    )


# Public registry shared by every OceanColumnModel instance — same pattern
# as the atmosphere SCM's TIME_INTEGRATORS so a custom integrator can be
# registered once (before construction) and then selected by name.  The
# Euler / RK / AB2 numerics live in legoesm.core.column_stepping; only the
# ocean-specific state operators are injected here.
OCEAN_TIME_INTEGRATORS: dict[str, Callable] = build_explicit_integrators(
    _apply_ocean_tendencies, _average_ocean_tendencies,
)


def register_ocean_time_integrator(name: str, step_fn: Callable) -> None:
    """Register a custom explicit integrator under ``name`` (ocean SCM).

    Thin wrapper over
    :func:`legoesm.core.column_stepping.register_integrator` targeting the
    module-level :data:`OCEAN_TIME_INTEGRATORS` registry.  The canonical
    signature is ``step_fn(state, aux, f, dt, t)`` returning
    ``(new_state, new_aux)``.  Custom integrators are only usable with
    ``implicit_vertical_mixing=False`` (the implicit operator split
    requires forward Euler).
    """
    register_integrator(OCEAN_TIME_INTEGRATORS, name, step_fn)


def _ocean_tendency_fn(physics_fn, grid, z_coord, forcing: OceanSCMForcing):
    """Return a closure ``(state, aux, t) -> (tend, aux)``.

    Evaluates the column physics (vertical mixing / convection / shortwave)
    with the surface-forcing struct built at stage time ``t``, then adds
    the external surface-flux + Coriolis tendency.  ``aux`` is an unused
    stateless carry (ocean column physics holds no prognostic state); it is
    threaded through so the shared integrator contract is satisfied.
    """
    def f(state, aux, t):
        horiz_shape = state.eta.data.shape
        dtype = state.T.data.dtype
        sfc = build_ocean_surface_forcing_struct(
            forcing, t, horiz_shape, dtype,
        )
        tend = physics_fn(state, grid, z_coord, surface_forcing=sfc)
        ftend = compute_ocean_forcing_tendencies(
            state, z_coord, grid, forcing, t,
        )
        return add_ocean_tendencies(tend, ftend), aux
    return f


class OceanSCMHistory(NamedTuple):
    """Stacked time-series of the ocean column state.

    Each field has a leading axis of length ``n_saved``; the vertical
    coordinate convention (surface-to-bottom) is preserved.
    """
    time: jax.Array   # (n_saved,) [s]
    T: jax.Array      # (n_saved, nlev) [degC]
    S: jax.Array      # (n_saved, nlev) [PSU]
    u: jax.Array      # (n_saved, nlev) [m/s]
    v: jax.Array      # (n_saved, nlev) [m/s]
    eta: jax.Array    # (n_saved,) [m]


# Surface-forcing schemes that double-count the externally-applied
# OceanSCMForcing wind / heat / freshwater fluxes if also routed through
# the physics factory.  ``"restoring"`` is interior tracer nudging and is
# compatible, so it (and ``"none"``) are allowed.
_CONFLICTING_SURFACE_SCHEMES = ("prescribed", "combined", "bulk_formulas")


class OceanColumnModel:
    """Driver for a single-column ocean physics integration.

    Construct via :meth:`create` (preferred).  See the module docstring for
    the shared-strategy design, the implicit-vertical-mixing operator split,
    and the surface-to-bottom level convention.
    """

    def __init__(
        self,
        *,
        physics_fn: Callable,
        state: OceanState,
        grid: OceanSCMGrid,
        z_coord: OceanZStarCoordinate,
        dt: float,
        time_integrator: str = "forward_euler",
        forcing: OceanSCMForcing | None = None,
        implicit_vertical_mixing: bool = True,
        K_v_background: float = 1.0e-5,
        A_v_background: float = 1.0e-4,
        t0_seconds: float = 0.0,
    ):
        if time_integrator not in OCEAN_TIME_INTEGRATORS:
            raise ValueError(
                f"Unknown time_integrator: {time_integrator!r}. "
                f"Available: {sorted(OCEAN_TIME_INTEGRATORS)}."
            )
        if implicit_vertical_mixing and time_integrator != "forward_euler":
            # The implicit operator split evaluates the explicit tendency
            # (and its K_v / A_v) once per step and pairs it with a single
            # backward-Euler diffusion solve; a multi-stage explicit
            # integrator would need a per-stage implicit solve that is not
            # implemented.  Reject the misleading combination rather than
            # silently ignoring the integrator choice (mirrors the
            # atmosphere SCM's stateful-physics restriction).
            raise ValueError(
                f"implicit_vertical_mixing=True requires "
                f"time_integrator='forward_euler', got {time_integrator!r}. "
                "Use forward_euler, or set implicit_vertical_mixing=False "
                "for pure-explicit integration with rk2/rk4/ab2 (subject to "
                "the explicit-diffusion CFL limit dt < dz^2/(2K))."
            )
        if forcing is not None:
            validate_ocean_forcing(forcing)
            # A custom integrator registered with the legacy 4-arg
            # signature is wrapped by core.register_integrator and only
            # samples forcing at the outer-step time; pairing it with
            # (possibly time-dependent) forcing would silently corrupt the
            # forcing trajectory.  Reject it (mirrors the atmosphere SCM).
            if getattr(OCEAN_TIME_INTEGRATORS[time_integrator],
                       "_is_legacy_4arg", False):
                raise ValueError(
                    f"time_integrator={time_integrator!r} was registered "
                    "with the legacy 4-arg signature and only samples forcing "
                    "at the outer-step time, which would corrupt "
                    "time-dependent OceanSCMForcing. Re-register it with the "
                    "5-arg ``(state, aux, f, dt, t)`` signature, or drop the "
                    "forcing argument."
                )

        self.physics_fn = physics_fn
        self.state = state
        self.grid = grid
        self.z_coord = z_coord
        self.dt = float(dt)
        self.time_integrator = time_integrator
        self.forcing = forcing if forcing is not None else default_ocean_forcing()
        self.implicit_vertical_mixing = bool(implicit_vertical_mixing)
        self.K_v_background = float(K_v_background)
        self.A_v_background = float(A_v_background)
        self.t_seconds = float(t0_seconds)
        self._step_fn = OCEAN_TIME_INTEGRATORS[time_integrator]
        self._ab2_prev_tend: OceanTendencies | None = None
        self._tend_fn = _ocean_tendency_fn(
            physics_fn, grid, z_coord, self.forcing,
        )

    @classmethod
    def create(
        cls,
        *,
        nlev: int,
        dt: float,
        T_profile: jax.Array,
        S_profile: jax.Array | float | None = None,
        physics_config: OceanPhysicsConfig | None = None,
        z_coord: OceanZStarCoordinate | None = None,
        H_max: float = 4000.0,
        dz_surface: float = 10.0,
        dz_deep: float = 200.0,
        u: float = 0.0,
        v: float = 0.0,
        eta: float = 0.0,
        H_bathy: float | None = None,
        latitude_deg: float = 0.0,
        longitude_deg: float = 0.0,
        time_integrator: str = "forward_euler",
        forcing: OceanSCMForcing | None = None,
        implicit_vertical_mixing: bool = True,
        K_v_background: float = 1.0e-5,
        A_v_background: float = 1.0e-4,
        dtype=None,
        t0_seconds: float = 0.0,
    ) -> "OceanColumnModel":
        """Build a single-column ocean model with sensible defaults.

        Parameters
        ----------
        nlev
            Number of vertical levels.
        dt
            Physics time step [s].
        T_profile
            Initial potential-temperature profile [degC], shape
            ``(nlev,)``, indexed surface→bottom.
        S_profile
            Initial salinity [PSU]: ``(nlev,)`` profile, scalar, or ``None``
            for a uniform 35 PSU column.
        physics_config
            Combined ocean physics configuration.  Defaults to constant
            vertical mixing with no convection.  Its
            ``surface_forcing.scheme`` must be ``"none"`` or ``"restoring"``
            — prescribed / combined / bulk schemes would double-count the
            :class:`OceanSCMForcing` surface fluxes and are rejected.
        z_coord
            Vertical coordinate.  When ``None`` a stretched z* coordinate
            is built from ``H_max`` / ``dz_surface`` / ``dz_deep``.
        H_max, dz_surface, dz_deep
            z* coordinate parameters (only used when ``z_coord is None``).
        u, v, eta
            Initial uniform velocities [m/s] and sea-surface height [m].
        H_bathy
            Column depth [m]; defaults to the coordinate's ``H_max``.
        latitude_deg, longitude_deg
            Column location (used for the Coriolis-free wind-stress helper).
        time_integrator
            Explicit integrator name.  With ``implicit_vertical_mixing=True``
            (default) only ``"forward_euler"`` is allowed.
        forcing
            Optional :class:`OceanSCMForcing` (wind stress, surface heat /
            freshwater flux, Coriolis).  ``None`` = no external forcing.
        implicit_vertical_mixing
            When ``True`` (default) use the backward-Euler vertical-diffusion
            operator split (unconditionally stable).  When ``False`` use
            pure explicit integration (CFL-limited).
        K_v_background, A_v_background
            Background tracer diffusivity / momentum viscosity [m²/s] added
            to the physics-derived profiles in the implicit solve.
        """
        if nlev < 2:
            # A single-level column has no vertical structure: the implicit
            # tridiagonal solve is a no-op and KPP / Richardson / convection
            # all read level index 1 (the layer below the surface).
            raise ValueError(
                f"OceanColumnModel requires nlev >= 2 (a column needs at "
                f"least two layers for vertical mixing / KPP); got {nlev}."
            )
        if z_coord is None:
            z_coord = create_ocean_z_star(
                n_levels=nlev, H_max=H_max,
                dz_surface=dz_surface, dz_deep=dz_deep,
            )
        elif z_coord.n_levels != nlev:
            raise ValueError(
                f"z_coord.n_levels ({z_coord.n_levels}) != nlev ({nlev})."
            )

        if physics_config is None:
            physics_config = OceanPhysicsConfig()
        sfc_scheme = physics_config.surface_forcing.scheme
        if sfc_scheme in _CONFLICTING_SURFACE_SCHEMES:
            raise ValueError(
                f"physics_config.surface_forcing.scheme={sfc_scheme!r} would "
                "double-count the OceanSCMForcing surface fluxes (the SCM "
                "applies wind stress / heat / freshwater itself). Set it to "
                "'none' (or 'restoring' for interior tracer nudging) and use "
                "OceanSCMForcing for the surface fluxes."
            )
        # TKE / CATKE are fallback-computed schemes: their pipeline
        # factories are deliberate no-ops (K_v=None); the K profiles are
        # computed inside the 3-D models' compute_vertical_K_profiles
        # fallback, which the SCM does not have — it consumes the pipeline
        # K directly (``tend.K_v`` in _apply_implicit_vertical_mixing).
        # Under the SCM they would run with NO closure mixing at all
        # (background floors only) — reject rather than silently degrade
        # (dispatch discipline).
        if physics_config.vertical_mixing.scheme in ("tke", "catke"):
            raise NotImplementedError(
                f"vertical_mixing.scheme="
                f"{physics_config.vertical_mixing.scheme!r} is not wired "
                "for the ocean SCM: its K profiles come from the 3-D "
                "models' compute_vertical_K_profiles fallback, which the "
                "SCM path does not call — the closure would silently "
                "contribute no mixing here. Use 'kpp', 'constant' or "
                "'richardson'."
            )
        # Lateral mixing (horizontal ∇² viscosity / diffusion) is
        # identically zero on a single column — there are no horizontal
        # neighbours — and the harmonic / biharmonic operators require a
        # real horizontal grid (halo offsets) that the 1×1 SCM grid does
        # not provide.  Force it off rather than let it crash; this is a
        # physical no-op, not a silent degradation.
        physics_config = physics_config._replace(
            lateral_mixing=physics_config.lateral_mixing._replace(scheme="none"),
        )

        grid = make_ocean_scm_grid(latitude_deg, longitude_deg)
        state = make_ocean_column_state(
            nlev, T_profile=T_profile, S_profile=S_profile, z_coord=z_coord,
            u=u, v=v, eta=eta, H_bathy=H_bathy, dtype=dtype,
        )
        # apply_vertical_diffusion=False ⇒ physics returns the explicit
        # remainder + K_v/A_v for the implicit solve.  =True ⇒ physics
        # applies the vertical diffusion explicitly (CFL-limited).
        physics_fn = make_ocean_physics(
            physics_config,
            apply_vertical_diffusion=not implicit_vertical_mixing,
        )
        return cls(
            physics_fn=physics_fn, state=state, grid=grid, z_coord=z_coord,
            dt=dt, time_integrator=time_integrator, forcing=forcing,
            implicit_vertical_mixing=implicit_vertical_mixing,
            K_v_background=K_v_background, A_v_background=A_v_background,
            t0_seconds=t0_seconds,
        )

    def _apply_implicit_vertical_mixing(
        self, state: OceanState, K_v, A_v, dt: float,
    ) -> OceanState:
        """Backward-Euler vertical diffusion of ``T, S, u, v`` (zero-flux).

        Mirrors the 3-D lat-lon C-grid model's implicit-mixing step: add
        the background floors to the physics ``K_v`` / ``A_v`` interface
        profiles and apply one tridiagonal solve per field.  ``T, S`` use
        the tracer diffusivity ``K_v``; ``u, v`` use the momentum viscosity
        ``A_v``.  Zero-flux boundaries preserve the column mean exactly
        (surface fluxes were already injected explicitly).

        Two simplifications relative to the 3-D reference are valid for the
        single column: (1) no ``land_mask`` re-masking — the SCM column is
        always wet (``make_ocean_column_state`` sets ``land_mask ≡ 1``); and
        (2) no cell→face interpolation of ``A_v`` / ``dz`` for momentum — the
        SCM uses an A-grid (``u, v`` share the tracer-cell shape), so the
        cell-centred profiles apply directly.
        """
        dtype = state.T.data.dtype
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, self.z_coord)
        dz_cell = self.z_coord.dz_ref * J[..., jnp.newaxis]  # (1,1,1,nlev)
        dz_half = build_dz_half(dz_cell)                     # (1,1,1,nlev-1)

        K_bg = jnp.asarray(self.K_v_background, dtype=dtype)
        A_bg = jnp.asarray(self.A_v_background, dtype=dtype)
        K_v_cell = K_bg if K_v is None else (K_v.astype(dtype) + K_bg)
        A_v_cell = A_bg if A_v is None else (A_v.astype(dtype) + A_bg)

        T_new = implicit_vertical_diffusion_ocean(
            state.T.data, K_v_cell, dz_cell, dz_half, dt,
        )
        S_new = implicit_vertical_diffusion_ocean(
            state.S.data, K_v_cell, dz_cell, dz_half, dt,
        )
        u_new = implicit_vertical_diffusion_ocean(
            state.u.data, A_v_cell, dz_cell, dz_half, dt,
        )
        v_new = implicit_vertical_diffusion_ocean(
            state.v.data, A_v_cell, dz_cell, dz_half, dt,
        )
        return state._replace(
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
        )

    def _apply_coriolis_rotation(self, state: OceanState) -> OceanState:
        """Crank-Nicolson Coriolis rotation of ``(u, v)`` over one step.

        Solves the rotation operator ``du/dt = f v``, ``dv/dt = −f u``
        with the trapezoidal (Crank-Nicolson) rule, which is the exact
        rotation by angle ``2·arctan(f·dt/2)`` and is energy-conserving
        and unconditionally stable for any ``f·dt`` — unlike forward
        Euler, whose amplification ``√(1+(f·dt)²) > 1`` blows up inertial
        motions over multi-day ocean integrations.  Applied as a uniform
        operator-split sub-step at the end of every model step (no-op when
        ``f_c == 0``).
        """
        f_c = self.forcing.f_c
        if f_c == 0.0:
            return state
        dtype = state.u.data.dtype
        a = jnp.asarray(0.5 * f_c * self.dt, dtype=dtype)
        denom = 1.0 + a * a
        u = state.u.data
        v = state.v.data
        u_new = ((1.0 - a * a) * u + 2.0 * a * v) / denom
        v_new = ((1.0 - a * a) * v - 2.0 * a * u) / denom
        return state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
        )

    def step(self) -> OceanState:
        """Advance the column by one step.

        Sub-step sequence (operator splitting):

        1. **Explicit physics + surface forcing.**  In implicit-mixing
           mode this is one forward-Euler update of the explicit remainder
           (surface fluxes, KPP non-local flux, convective non-local term);
           in explicit mode it is the selected registered integrator
           (forward Euler, RK2, RK4, or AB2) applied to the full tendency
           including explicit vertical diffusion.
        2. **Implicit vertical diffusion** (implicit-mixing mode only):
           backward-Euler tridiagonal solve of ``T, S, u, v``.
        3. **Coriolis rotation**: Crank-Nicolson rotation of ``(u, v)``
           (stable for any ``f·dt``; no-op when ``f_c == 0``).

        Coriolis is operator-split in **both** modes (it is never part of
        the tendency the registry integrator sees), so even with an RK / AB2
        explicit integrator the rotation is the stable CN sub-step rather
        than a forward-Euler tendency term — deliberate, because explicit
        Coriolis is unconditionally unstable.
        """
        if self.implicit_vertical_mixing:
            tend, _ = self._tend_fn(self.state, None, self.t_seconds)
            state_star = _apply_ocean_tendencies(self.state, tend, self.dt)
            new_state = self._apply_implicit_vertical_mixing(
                state_star, tend.K_v, tend.A_v, self.dt,
            )
        elif self.time_integrator == "ab2":
            # Adams-Bashforth 2 — single tendency evaluation per outer step;
            # first step falls back to forward Euler using the same tendency
            # that is cached as prev_tend (mirrors the atmosphere SCM).
            tend_n, _ = self._tend_fn(self.state, None, self.t_seconds)
            if self._ab2_prev_tend is None:
                tend_eff = tend_n
            else:
                tend_eff = ab2_effective_tendency(
                    tend_n, self._ab2_prev_tend, _average_ocean_tendencies,
                )
            new_state = _apply_ocean_tendencies(self.state, tend_eff, self.dt)
            self._ab2_prev_tend = tend_n
        else:
            new_state, _ = self._step_fn(
                self.state, None, self._tend_fn, self.dt, self.t_seconds,
            )
        new_state = self._apply_coriolis_rotation(new_state)
        self.t_seconds += self.dt
        self.state = new_state
        return self.state

    def run(
        self,
        nsteps: int,
        *,
        save_every: int = 1,
    ) -> tuple[OceanState, OceanSCMHistory]:
        """Integrate for ``nsteps`` steps and return final state + history.

        ``save_every`` controls the output cadence (1 = every step).
        """
        times, Ts, Ss, us, vs, etas = [], [], [], [], [], []
        for k in range(nsteps):
            self.step()
            if k % save_every == 0 or k == nsteps - 1:
                # self.t_seconds is advanced inside step(), so it already
                # holds the absolute time at the end of step k (incl. t0).
                times.append(self.t_seconds)
                Ts.append(self.state.T.data[0, 0, 0])
                Ss.append(self.state.S.data[0, 0, 0])
                us.append(self.state.u.data[0, 0, 0])
                vs.append(self.state.v.data[0, 0, 0])
                etas.append(self.state.eta.data[0, 0, 0])

        history = OceanSCMHistory(
            time=jnp.asarray(times),
            T=jnp.stack(Ts),
            S=jnp.stack(Ss),
            u=jnp.stack(us),
            v=jnp.stack(vs),
            eta=jnp.asarray(etas),
        )
        return self.state, history
