"""Single-column model (SCM) driver for legoESM.

Issue #277: a minimal, dycore-free driver that exercises the atmospheric
physics pipeline on a single (lat, lon) column. Intended for
parameterization integration tests, stability sweeps, and idealized
process studies (e.g. radiative-convective equilibrium, GABLS-style
boundary layer cases).

The SCM reuses the canonical hydrostatic physics factory
(`legoesm.atmosphere.physics.combined.make_physics`) so any scheme that
works in the full 3-D model also works here without modification. There
is no horizontal advection from the dynamical core, no pressure-gradient
force, and no intrinsic Coriolis — only column tendencies from
radiation, convection, turbulence, microphysics, and gravity-wave drag,
optionally augmented by a user-supplied :class:`SCMForcing` that
contributes external large-scale forcing (Coriolis + geostrophic wind,
large-scale subsidence, prescribed horizontal-advection tendencies, and
the Phase-B prescribed-surface-flux hooks).  Tendencies are then
combined and integrated with forward Euler, RK2, or RK4.

Example
-------
>>> from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
>>> from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
>>> from legoesm.atmosphere.physics import (
...     PhysicsConfig, RadiationConfig, TurbulenceConfig,
... )
>>> cfg = PhysicsConfig(
...     radiation=RadiationConfig(scheme="gray"),
...     turbulence=TurbulenceConfig(scheme="louis"),
... )
>>> forcing = SCMForcing(
...     f_c=1e-4,
...     u_geo=lambda t: jnp.full(40, 8.0),
... )
>>> scm = SingleColumnModel.create(
...     physics_config=cfg, nlev=40, dt=300.0,
...     latitude_deg=0.0, T_profile=jnp.linspace(220.0, 295.0, 40),
...     forcing=forcing,
... )
>>> final_state, history = scm.run(nsteps=288, save_every=12)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.column_stepping import (
    ab2_effective_tendency,
    build_explicit_integrators,
    register_integrator,
)
from legoesm.grids.vertical import SigmaCoordinate, create_sigma_coordinate
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.physics_state import (
    PhysicsState,
    init_physics_state,
)
from legoesm.atmosphere.forcing.scm.scm_forcing import (
    SCMForcing,
    add_tendencies,
    compute_forcing_tendencies,
    default_forcing,
    inject_prescribed_T_sfc_into_phys_state,
    inject_prescribed_surface_fluxes_into_phys_state,
    validate_forcing,
    validate_forcing_against_state,
)


_DIMS_3D = ("face", "x", "y", "level")
_DIMS_2D = ("face", "x", "y")
DEFAULT_MICROPHYSICS_SUBSTEPS = 1


class SCMGrid(NamedTuple):
    """Minimal grid object for the single-column model.

    Exposes the ``grid_lat`` / ``grid_lon`` / ``radius`` attributes
    consumed by the radiation and GWD integration helpers. Shape is
    ``(1, 1)`` so column reshapes give ``ncol = 1``.
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


def make_scm_grid(latitude_deg: float, longitude_deg: float = 0.0) -> SCMGrid:
    """Build a single-column grid at the given latitude/longitude."""
    lat = jnp.asarray([[[jnp.deg2rad(latitude_deg)]]])
    lon = jnp.asarray([[[jnp.deg2rad(longitude_deg)]]])
    return SCMGrid(grid_lat=lat, grid_lon=lon, radius=constants.R_earth)


def make_column_state(
    nlev: int,
    *,
    T_profile: jax.Array,
    q_v_profile: jax.Array | None = None,
    u: float | jax.Array = 0.0,
    v: float | jax.Array = 0.0,
    p_s: float = 1.0e5,
    phis: float = 0.0,
    dtype=None,
) -> HydrostaticState:
    """Construct a `HydrostaticState` of shape (1, 1, nlev) for the SCM.

    Profiles are indexed top-to-bottom (k=0 = model top, k=nlev-1 =
    surface), matching `HybridSigmaPressureCoordinate` / `SigmaCoordinate`
    conventions.
    """
    if dtype is None:
        dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    T_profile = jnp.asarray(T_profile, dtype=dtype)
    if T_profile.size != nlev:
        raise ValueError(
            f"T_profile must have {nlev} elements, got {T_profile.size}"
        )
    T_data = T_profile.reshape(1, 1, 1, nlev)

    def _wind_data(value, name: str):
        arr = jnp.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            return jnp.full((1, 1, 1, nlev), arr, dtype=dtype)
        if arr.size != nlev:
            raise ValueError(f"{name} must be scalar or have {nlev} elements")
        return arr.reshape(1, 1, 1, nlev)

    u_data = _wind_data(u, "u")
    v_data = _wind_data(v, "v")
    p_s_data = jnp.full((1, 1, 1), p_s, dtype=dtype)
    phis_data = jnp.full((1, 1, 1), phis, dtype=dtype)

    tracers = None
    if q_v_profile is not None:
        q_v_arr = jnp.asarray(q_v_profile, dtype=dtype)
        if q_v_arr.size != nlev:
            raise ValueError(
                f"q_v_profile must have {nlev} elements, got {q_v_arr.size}"
            )
        q_v_arr = q_v_arr.reshape(nlev)
        tracers = {
            "q_v": Field(
                data=q_v_arr.reshape(1, 1, 1, nlev),
                name="q_v", dims=_DIMS_3D, units="kg/kg",
            )
        }

    return HydrostaticState(
        u=Field(data=u_data, name="u", dims=_DIMS_3D, units="m/s"),
        T=Field(data=T_data, name="T", dims=_DIMS_3D, units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=_DIMS_2D, units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=_DIMS_2D, units="m^2/s^2"),
        v=Field(data=v_data, name="v", dims=_DIMS_3D, units="m/s"),
        tracers=tracers,
    )


def _apply_tendencies(
    state: HydrostaticState,
    tend: HydrostaticTendencies,
    dt: float,
) -> HydrostaticState:
    """Apply scaled tendencies to a column state (used by every integrator).

    Tracer behaviour:
    - Every species present in ``state.tracers`` is carried forward; if
      a matching tendency is returned it is added.
    - Any tracer tendency returned by physics whose species is *not*
      already in ``state.tracers`` is materialised from zero — this is
      what lets microphysics/convection introduce ``q_c``, ``q_r``,
      ``q_i``, etc. without the SCM silently dropping them.
    - No positivity clipping is applied here. Schemes that need positive
      mixing ratios must enforce that themselves (most do). SCM-only
      microphysics substepping, when enabled, reduces the explicit step
      seen by the scheme instead of clipping this generic state update.
    """
    new_u = state.u.replace(data=state.u.data + dt * tend.du_dt.data)
    new_T = state.T.replace(data=state.T.data + dt * tend.dT_dt.data)
    new_p_s = state.p_s.replace(data=state.p_s.data + dt * tend.dp_s_dt.data)
    new_v = state.v
    if state.v is not None and tend.dv_dt is not None:
        new_v = state.v.replace(data=state.v.data + dt * tend.dv_dt.data)

    if tend.tracer_tendencies is None and state.tracers is None:
        new_tracers = None
    else:
        new_tracers = {}
        existing = state.tracers or {}
        tend_tracers = tend.tracer_tendencies or {}
        # Carry every existing species; add tendency if returned.
        for k, fld in existing.items():
            if k in tend_tracers:
                new_tracers[k] = fld.replace(
                    data=fld.data + dt * tend_tracers[k].data
                )
            else:
                new_tracers[k] = fld
        # Materialise any new species introduced by physics tendencies.
        for k, tend_fld in tend_tracers.items():
            if k in new_tracers:
                continue
            zero = jnp.zeros_like(tend_fld.data)
            new_tracers[k] = Field(
                data=zero + dt * tend_fld.data,
                name=k,
                dims=tend_fld.dims if tend_fld.dims else _DIMS_3D,
                units="kg/kg",
            )

    return HydrostaticState(
        u=new_u, T=new_T, p_s=new_p_s, phis=state.phis,
        v=new_v, tracers=new_tracers,
    )


def apply_tendencies(
    state: HydrostaticState,
    tend: HydrostaticTendencies,
    dt: float,
) -> HydrostaticState:
    """Apply SCM tendencies to a hydrostatic column state."""
    return _apply_tendencies(state, tend, dt)


def _validate_microphysics_substeps(microphysics_substeps: int) -> int:
    """Return a validated fixed microphysics substep count."""
    if (
        isinstance(microphysics_substeps, bool)
        or int(microphysics_substeps) != microphysics_substeps
    ):
        raise ValueError(
            "microphysics_substeps must be a positive integer; "
            f"got {microphysics_substeps!r}."
        )
    microphysics_substeps = int(microphysics_substeps)
    if microphysics_substeps < 1:
        raise ValueError(
            "microphysics_substeps must be a positive integer; "
            f"got {microphysics_substeps}."
        )
    return microphysics_substeps


def _microphysics_substepped_forward_euler(
    *,
    microphysics_fn: Callable,
    get_grid: Callable[[], SCMGrid],
    get_sigma_coord: Callable[[], SigmaCoordinate],
    microphysics_substeps: int,
) -> Callable:
    """Build an SCM-only forward-Euler wrapper with microphysics substeps.

    The supplied outer tendency ``f`` is expected to exclude microphysics.
    It is evaluated once per SCM step, preserving the existing coupling for
    radiation/convection/turbulence/GWD and their ``PhysicsState`` updates.
    The microphysics tendency is re-evaluated on each fixed substep with a
    factory built at ``dt / microphysics_substeps``.
    """
    n_substeps = _validate_microphysics_substeps(microphysics_substeps)

    def step(state, phys_state, f, dt, t):
        nonmicro_tend, phys_out = f(state, phys_state, t)
        sub_dt = dt / n_substeps
        grid = get_grid()
        sigma_coord = get_sigma_coord()

        def substep(carry, _i):
            sub_state, base_tend = carry
            micro_tend = microphysics_fn(sub_state, grid, sigma_coord)
            tend = add_tendencies(base_tend, micro_tend)
            sub_state = _apply_tendencies(sub_state, tend, sub_dt)
            return (sub_state, base_tend), None

        (new_state, _), _ = jax.lax.scan(
            substep,
            (state, nonmicro_tend),
            jnp.arange(n_substeps, dtype=jnp.int32),
        )
        return new_state, phys_out

    return step


def _tendency_fn(physics_fn, grid, sigma_coord, forcing: SCMForcing | None = None):
    """Return a closure ``(state, phys_state, t) -> (tend, phys_state_out)``.

    Used by the time-integrator strategies so they all share the same
    physics-evaluation interface regardless of which scheme is active.
    ``t`` is the **stage time** in seconds since model start (RK stages
    pass intermediate values); it is forwarded to ``forcing`` callables
    so time-dependent geostrophic wind, prescribed surface forcing, etc.
    are evaluated at the correct sub-step time.  ``forcing=None`` is a
    no-op.
    """
    def f(state, phys_state, t):
        # For ``prescribe="T_s"``: write the prescribed skin temperature
        # into ``phys_state.surface_T_sfc_override`` so the turbulence
        # scheme's bulk-flux call sees ``T_sfc != T[..., -1]`` and
        # produces a non-zero sensible flux.  Lowest air temperature
        # evolves freely under that flux (do NOT mutate state.T[-1] —
        # collapses the gradient — Phase B codex iter-1 high finding).
        # Also notify radiation via ``physics_fn.set_T_sfc_override`` so
        # surface longwave emission uses the same prescribed value
        # (Phase B v2 codex iter-2 high finding — without this the
        # boundary was silently split between turbulence and radiation).
        #
        # The radiation hook is process-local mutable closure state on
        # ``physics_fn``; wrap the call in try/finally so the override
        # is always cleared, preventing leakage across SCM instances
        # that share a ``physics_fn`` (Phase B v2 codex iter-5 high
        # finding).
        rad_hook = None
        hook_armed = False
        if forcing is not None and forcing.prescribe == "T_s":
            phys_state = inject_prescribed_T_sfc_into_phys_state(
                phys_state, forcing.T_s(t),
            )
            rad_hook = getattr(physics_fn, "set_T_sfc_override", None)
            if callable(rad_hook):
                rad_hook(phys_state.surface_T_sfc_override)
                hook_armed = True

        # For ``prescribe="fluxes"`` with ``flux_to_closure``: hand THIS
        # stage's surface flux to the turbulence closure as its lower boundary
        # condition instead of injecting it as a column tendency.  Evaluated at
        # the stage time ``t``, so an RK stage and a diurnal cycle stay
        # consistent; ``compute_forcing_tendencies`` skips its own injection
        # under the same flag, so the flux is applied exactly once.
        if (forcing is not None and forcing.prescribe == "fluxes"
                and forcing.flux_to_closure):
            phys_state = inject_prescribed_surface_fluxes_into_phys_state(
                phys_state,
                wth=None if forcing.w_th_s is None else forcing.w_th_s(t),
                wqv=None if forcing.w_qv_s is None else forcing.w_qv_s(t),
            )

        try:
            tend, phys_out = physics_fn(
                state, grid, sigma_coord, phys_state=phys_state,
            )
        finally:
            if hook_armed:
                rad_hook(None)

        if forcing is not None:
            ftend = compute_forcing_tendencies(state, sigma_coord, forcing, t)
            tend = add_tendencies(tend, ftend)
        return tend, phys_out
    return f


def _average_tendencies(*tends, weights):
    """Linear combination of HydrostaticTendencies with the given weights."""
    if len(weights) != len(tends):
        raise ValueError("weights length must match number of tendencies")

    def combine(field_list):
        return sum(w * f.data for w, f in zip(weights, field_list))

    first = tends[0]
    du = combine([t.du_dt for t in tends])
    dT = combine([t.dT_dt for t in tends])
    dps = combine([t.dp_s_dt for t in tends])
    dphis = combine([t.dphis_dt for t in tends])

    dv_dt = None
    if first.dv_dt is not None and all(t.dv_dt is not None for t in tends):
        dv_dt = first.dv_dt.replace(
            data=combine([t.dv_dt for t in tends])
        )

    tracer_tendencies = None
    if first.tracer_tendencies is not None and all(
        t.tracer_tendencies is not None for t in tends
    ):
        tracer_tendencies = {}
        for k in first.tracer_tendencies:
            if all(k in t.tracer_tendencies for t in tends):
                tracer_tendencies[k] = first.tracer_tendencies[k].replace(
                    data=combine([t.tracer_tendencies[k] for t in tends])
                )

    return HydrostaticTendencies(
        du_dt=first.du_dt.replace(data=du),
        dT_dt=first.dT_dt.replace(data=dT),
        dp_s_dt=first.dp_s_dt.replace(data=dps),
        dphis_dt=first.dphis_dt.replace(data=dphis),
        dv_dt=dv_dt,
        tracer_tendencies=tracer_tendencies,
    )


# Public registry — same naming pattern as PhysicsConfig schemes so that
# swapping integrators feels like swapping a parameterization.  The
# integrator numerics (Euler / RK2 / RK4 stage weights + the AB2 first-
# step fallback) live in :mod:`legoesm.core.column_stepping` and are shared
# with the ocean single-column model; only the atmosphere-specific
# ``_apply_tendencies`` / ``_average_tendencies`` operators are injected
# here.  The true AB2 multistep update lives in
# :meth:`SingleColumnModel.step` to keep its prev-tendency cache isolated
# per SCM instance.
TIME_INTEGRATORS: dict[str, Callable] = build_explicit_integrators(
    _apply_tendencies, _average_tendencies,
)


def register_time_integrator(name: str, step_fn: Callable) -> None:
    """Register a custom time-integrator under ``name`` (atmosphere SCM).

    Thin wrapper over
    :func:`legoesm.core.column_stepping.register_integrator` targeting this
    module's :data:`TIME_INTEGRATORS` registry.  The canonical signature is
    ``step_fn(state, phys_state, f, dt, t)`` where ``f`` is the tendency
    callable returned by :func:`_tendency_fn` and expects
    ``f(state, phys_state, stage_t_seconds)``; the integrator must return
    ``(new_state, new_phys_state)``.  See
    :func:`~legoesm.core.column_stepping.register_integrator` for the 4-arg
    legacy-shim behaviour (sampling forcing at the outer-step time only,
    tagged ``_is_legacy_4arg``) and the full arity-validation rules.
    """
    register_integrator(TIME_INTEGRATORS, name, step_fn)


class SCMHistory(NamedTuple):
    """Stacked time-series of column state along a leading time axis.

    Each field has a leading axis of length ``n_saved``. The vertical
    coordinate convention (top-to-bottom) is preserved.
    """
    time: jax.Array        # (n_saved,) [s]
    T: jax.Array           # (n_saved, nlev)
    u: jax.Array           # (n_saved, nlev)
    v: jax.Array           # (n_saved, nlev)
    p_s: jax.Array         # (n_saved,)
    q_v: jax.Array | None  # (n_saved, nlev) or None


class SingleColumnModel:
    """Driver for a single-column atmospheric physics integration.

    Construct via :meth:`create` (preferred) or by passing pre-built
    `physics_fn`, `state`, `phys_state`, `grid`, and `sigma_coord`.

    Notes
    -----
    Integration is forward Euler. Schemes that are themselves implicit
    in the vertical (e.g. ``"louis"`` turbulence, ``"tke"`` PBL) handle
    their own implicit step inside the physics function — only the
    explicit *remainder* is exposed as a tendency, so forward Euler at
    the SCM level is consistent with the full-model coupling.

    Surface temperature is taken as the lowest-level air temperature
    ``T[..., -1]`` (the convention used throughout the model). To impose
    a prescribed SST, hold ``T[..., -1]`` fixed externally after each
    step (see :meth:`step`'s return value).
    """

    def __init__(
        self,
        *,
        physics_fn: Callable,
        state: HydrostaticState,
        phys_state: PhysicsState,
        grid: SCMGrid,
        sigma_coord: SigmaCoordinate,
        dt: float,
        time_integrator: str = "forward_euler",
        forcing: SCMForcing | None = None,
        t0_seconds: float = 0.0,
        physics_config: PhysicsConfig | None = None,
        microphysics_substeps: int = DEFAULT_MICROPHYSICS_SUBSTEPS,
        microphysics_fn: Callable | None = None,
        _built_by_create: bool = False,
    ):
        # Private marker set ONLY by :meth:`create`.  Direct callers
        # leave it ``False``; used below by the
        # ``prescribe='fluxes'`` validation to guarantee that
        # ``physics_fn`` was built from a single validated
        # ``physics_config`` (which only ``create`` can enforce).
        self._built_by_create = bool(_built_by_create)
        if time_integrator not in TIME_INTEGRATORS:
            raise ValueError(
                f"Unknown time_integrator: {time_integrator!r}. "
                f"Available: {sorted(TIME_INTEGRATORS)}."
            )
        microphysics_substeps = _validate_microphysics_substeps(
            microphysics_substeps,
        )
        if microphysics_substeps > 1:
            if not getattr(self, "_built_by_create", False):
                raise ValueError(
                    "microphysics_substeps > 1 requires construction via "
                    "SingleColumnModel.create(), which builds the split "
                    "non-microphysics and microphysics tendency functions."
                )
            if microphysics_fn is None:
                raise ValueError(
                    "microphysics_substeps > 1 requires a microphysics_fn "
                    "built at dt / microphysics_substeps."
                )
            if time_integrator != "forward_euler":
                raise ValueError(
                    "microphysics_substeps > 1 is currently supported only "
                    "with time_integrator='forward_euler'."
                )
        step_fn_obj = TIME_INTEGRATORS[time_integrator]
        if forcing is not None:
            validate_forcing(forcing)
            validate_forcing_against_state(forcing, state)
            # Direct ``SingleColumnModel(...)`` construction bypasses
            # :meth:`create`'s ``_validate_prescribed_fluxes_no_double_count``
            # gate.  Re-run it here when ``physics_config`` is supplied;
            # otherwise refuse ``prescribe='fluxes'`` so the user is
            # forced to either pass ``physics_config`` (validatable) or
            # route through :meth:`create` (Phase F fix #2 codex iter-1
            # medium finding).
            if forcing.prescribe == "fluxes":
                # ``physics_config`` is required AND must have been used
                # to build ``physics_fn``.  Direct construction cannot
                # verify the latter from outside, so the only safe path
                # is to refuse direct construction with prescribe='fluxes'
                # unless the caller routed through :meth:`create` (which
                # owns both objects).  ``create()`` sets a private
                # ``_built_by_create`` marker on the returned instance
                # before this check fires — direct constructor calls
                # never set it, so the guard cleanly distinguishes the
                # two paths.  Phase F fix #2 codex iter-2 medium finding.
                if not getattr(self, "_built_by_create", False):
                    raise ValueError(
                        "SCMForcing.prescribe='fluxes' requires that "
                        "``physics_fn`` was built from a validated "
                        "``physics_config`` — direct ``SingleColumnModel"
                        "(...)`` construction cannot enforce that "
                        "physics_fn ≡ make_physics(physics_config), so a "
                        "caller could pass a benign physics_config to "
                        "satisfy validation while running a physics_fn "
                        "that double-counts the prescribed flux.  Route "
                        "through ``SingleColumnModel.create(...)``, "
                        "which owns construction of both objects from a "
                        "single config."
                    )
                if physics_config is None:
                    raise ValueError(
                        "SingleColumnModel.create() must pass "
                        "physics_config through to the constructor for "
                        "no-double-count validation."
                    )
                self._validate_prescribed_fluxes_no_double_count(
                    physics_config, forcing,
                )
            if getattr(step_fn_obj, "_is_legacy_4arg", False):
                raise ValueError(
                    f"time_integrator={time_integrator!r} was registered "
                    "with the legacy 4-arg signature and is wrapped by "
                    "the compatibility shim.  That shim only samples "
                    "forcing at the outer-step time, which would "
                    "silently corrupt time-dependent SCM forcing "
                    "(geostrophic wind drift, prescribed surface T_s "
                    "trajectory, advective tendencies). Migrate the "
                    "integrator to the 5-arg ``(state, phys, f, dt, t)`` "
                    "signature, or drop the forcing argument."
                )
        self.physics_fn = physics_fn
        self.state = state
        self.phys_state = phys_state
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.dt = float(dt)
        self.time_integrator = time_integrator
        self.microphysics_substeps = microphysics_substeps
        self._microphysics_fn = microphysics_fn
        self.forcing = forcing if forcing is not None else default_forcing()
        self.t_seconds = float(t0_seconds)
        if microphysics_substeps > 1:
            self._step_fn = _microphysics_substepped_forward_euler(
                microphysics_fn=microphysics_fn,
                get_grid=lambda: self.grid,
                get_sigma_coord=lambda: self.sigma_coord,
                microphysics_substeps=microphysics_substeps,
            )
        else:
            self._step_fn = step_fn_obj
        # AB2 stores the previous-step tendency to combine with the
        # current step's tendency as ``1.5·tend_n − 0.5·tend_{n-1}``
        # (second-order linear multistep).  Per-instance so multiple
        # SCMs sharing a physics_fn cannot cross-contaminate.
        self._ab2_prev_tend: HydrostaticTendencies | None = None
        # Pass the *original* forcing arg (not the default sentinel) so
        # _tendency_fn skips the forcing branch entirely when no forcing
        # was supplied — preserves bit-exact behaviour for callers that
        # never use forcing.
        self._tend_fn = _tendency_fn(
            physics_fn, grid, sigma_coord, forcing=forcing,
        )

    @staticmethod
    def _validate_prescribed_fluxes_no_double_count(
        physics_config: PhysicsConfig, forcing,
    ) -> None:
        """Reject configurations that double-count the surface flux.

        ``SCMForcing(prescribe='fluxes')`` injects user-supplied
        kinematic surface fluxes (``w'θ'``, ``w'qv'``) as a lowest-
        cell tendency.  If the active turbulence scheme also runs its
        bulk-flux formula with a non-zero heat-transfer coefficient
        ``Ch_neutral``, the sensible / latent heat flux gets counted
        twice (once via the prescribed-flux tendency, once via the
        turbulence's vertical-diffusion bottom-BC).  Reject at SCM
        construction so the user gets a clear error instead of a
        silently warmed lowest cell.

        Deferred-item #2 from the Phase F summary.  Momentum drag
        (``Cd_neutral``) is *not* checked because the prescribed-flux
        channel injects only ``w'θ'`` / ``w'qv'`` — turbulence
        retains responsibility for surface momentum stress.
        """
        if forcing is None or forcing.prescribe != "fluxes":
            return
        turb = physics_config.turbulence
        if turb.scheme == "none":
            return
        scheme_sub = getattr(turb, turb.scheme, None)
        if scheme_sub is None:
            return
        surf = getattr(scheme_sub, "surface", None)
        if surf is None:
            return

        # MOST bulk schemes (COARE3, Large-Yeager) compute sensible /
        # latent heat fluxes from iterative MOST scaling parameters
        # and do NOT honour ``Ch_neutral`` — they always produce a
        # non-zero heat flux from any non-zero
        # ``(T_sfc − T_air)`` / ``(q_sfc − q_air)`` gradient.  Setting
        # ``Ch_neutral=0`` does not suppress them, so the only safe
        # combination with ``prescribe='fluxes'`` is the
        # ``bulk_scheme='constant'`` family + ``Ch_neutral=0``.
        # Phase F fix #2 codex iter-1 high finding.
        bulk_scheme = getattr(surf, "bulk_scheme", "constant")
        # "most" belongs here too: surface_layer.compute_surface_fluxes routes
        # ("most", "coare3", "large_yeager") to the SAME compute_most_fluxes
        # path, which derives the heat flux from MOST scaling and ignores
        # Ch_neutral. Omitting it let prescribe="fluxes" + bulk_scheme="most"
        # through, silently double-counting the prescribed surface heat flux --
        # and the Ch_neutral check below cannot catch it, because under MOST
        # Ch_neutral is inert and is legitimately left at 0.
        if bulk_scheme in ("most", "coare3", "large_yeager"):
            raise ValueError(
                "SCMForcing.prescribe='fluxes' is set, but the active "
                f"turbulence scheme {turb.scheme!r} uses "
                f"bulk_scheme={bulk_scheme!r}, which solves MOST "
                "iteratively for sensible / latent heat flux and does "
                "NOT honour Ch_neutral.  The bulk formula would "
                "double-count the prescribed surface flux.  Switch to "
                "bulk_scheme='constant' with Ch_neutral=0, or drop "
                "prescribe='fluxes'."
            )

        Ch = getattr(surf, "Ch_neutral", 0.0)
        if Ch != 0.0:
            raise ValueError(
                "SCMForcing.prescribe='fluxes' is set, but the active "
                f"turbulence scheme {turb.scheme!r} has a non-zero "
                f"surface.Ch_neutral={Ch}.  The prescribed-flux "
                "channel injects ``w'θ'`` / ``w'qv'`` at the lowest "
                "cell; turbulence's bulk-flux formula would inject "
                "the same flux through the implicit-diffusion bottom "
                "BC, double-counting it.  Set "
                f"turbulence.{turb.scheme}.surface.Ch_neutral=0.0 "
                "(retain Cd_neutral for momentum drag) or drop the "
                "prescribed-flux channel."
            )

    @staticmethod
    def _validate_integrator_compatibility(
        physics_config: PhysicsConfig, time_integrator: str,
    ) -> None:
        """Reject integrator/physics combinations that would silently lie.

        Multi-stage integrators (rk2, rk4) reuse the *stage-1* PhysicsState
        and stage-1 calendar time across every stage. That is safe only
        when (a) no scheme carries a prognostic physics state and (b) no
        scheme depends on sub-step time (diurnal radiation). Otherwise
        the integrator label would be misleading: schemes would see a
        stale carry and the result would not be a true RK update of the
        coupled (state, phys_state) system.
        """
        # Forward Euler evaluates physics once per outer step at a
        # single ``t`` and never reuses an in-flight stage carry, so
        # it is compatible with stateful and diurnal physics.
        if time_integrator in ("forward_euler",):
            return
        # AB2 also evaluates ``_tend_fn`` once per outer step (no sub-
        # stages) but the multistep formula only applies to
        # ``HydrostaticState`` tendencies — prognostic ``PhysicsState``
        # carries (MYNN qke, TKE, convective profile, GWD spectrum,
        # AR1 stochastic state) are advanced by the current step's
        # physics call alone, not by any multistep history.  That
        # mismatch silently breaks O(dt²) accuracy and benchmark
        # parity for any scheme with a prognostic carry, so we reject
        # the same combinations as the RK paths until a proper coupled
        # AB2 lands (Phase D codex iter-1 medium finding).
        stateful_turb = physics_config.turbulence.scheme in (
            "tke", "mynn25", "clubb_lite", "clubb", "edmf"
        )
        # Schemes that *read* the previous ``conv_prog_profile`` or
        # ``conv_stoch_state`` — and therefore must not be advanced
        # with a stale stage-1 carry under multi-stage integrators.
        # ``emanuel`` is prognostic: its faithful Emanuel-1991 cloud-base
        # mass-flux (CBMF) closure reads ``conv_prog_profile[:, -1]`` and
        # relaxes it each call, so it must NOT be re-evaluated freely by RK
        # stages.  The remaining schemes (``sbm``, ``dca``, ``kuo``,
        # ``kain_fritsch``) are diagnostic: they emit a fresh profile each
        # call, so RK stages may freely re-evaluate them.
        stateful_conv_schemes = (
            "mass_flux", "edmf",
            "zhang_mcfarlane", "tiedtke", "bechtold", "emanuel",
        )
        stateful_conv = physics_config.convection.scheme in stateful_conv_schemes
        throttled_conv = (
            physics_config.convection.scheme != "none"
            and getattr(physics_config.convection, "update_interval_steps", 1) != 1
        )
        stateful_gwd = physics_config.gravity_wave_drag.scheme == (
            "prognostic_spectral"
        )
        diurnal_rad = bool(
            getattr(physics_config.radiation, "diurnal_cycle", False)
        )
        bad = []
        if stateful_turb:
            bad.append(f"turbulence={physics_config.turbulence.scheme!r}")
        if stateful_conv:
            bad.append(f"convection={physics_config.convection.scheme!r}")
        if throttled_conv:
            bad.append("convection.update_interval_steps != 1")
        if stateful_gwd:
            bad.append("gravity_wave_drag='prognostic_spectral'")
        if diurnal_rad:
            bad.append("radiation.diurnal_cycle=True")
        if bad:
            raise ValueError(
                f"time_integrator={time_integrator!r} is incompatible with "
                f"stateful/non-autonomous physics: {', '.join(bad)}. "
                f"Use 'forward_euler', disable these schemes, or extend "
                f"the SCM integrator interface to advance PhysicsState "
                f"and stage time per stage."
            )

    @classmethod
    def create(
        cls,
        *,
        physics_config: PhysicsConfig,
        nlev: int,
        dt: float,
        T_profile: jax.Array,
        q_v_profile: jax.Array | None = None,
        u: float | jax.Array = 0.0,
        v: float | jax.Array = 0.0,
        p_s: float = 1.0e5,
        phis: float = 0.0,
        latitude_deg: float = 0.0,
        longitude_deg: float = 0.0,
        sigma_top: float = 0.01,
        prng_seed: int = 0,
        dtype=None,
        time_integrator: str = "forward_euler",
        forcing: SCMForcing | None = None,
        t0_seconds: float = 0.0,
        microphysics_substeps: int = DEFAULT_MICROPHYSICS_SUBSTEPS,
    ) -> "SingleColumnModel":
        """Build a single-column model with sensible defaults.

        Parameters
        ----------
        physics_config
            Combined physics configuration. Pass schemes with
            ``scheme="none"`` to disable a module.
        nlev
            Number of vertical levels.
        dt
            Physics time step [s].
        T_profile
            Initial temperature profile [K], shape ``(nlev,)``,
            indexed top→bottom.
        q_v_profile
            Optional initial water-vapor mixing ratio [kg/kg], shape
            ``(nlev,)``. When omitted no tracers are carried.
        u, v
            Initial wind components [m/s]. Scalars initialize uniform winds;
            ``(nlev,)`` profiles initialize level-varying winds, indexed
            top-to-bottom.
        p_s
            Initial surface pressure [Pa].
        phis
            Surface geopotential [m^2/s^2] (0 for sea-level column).
        latitude_deg, longitude_deg
            Column location. Used by radiation (solar zenith angle) and
            GWD (latitudinal coupling).
        sigma_top
            Top sigma value for the vertical coordinate.
        prng_seed
            Seed for the stochastic-physics PRNG carry.
        dtype
            Optional dtype override for state arrays.
        forcing
            Optional :class:`SCMForcing` carrying time-dependent
            geostrophic wind + Coriolis, large-scale subsidence,
            horizontal-advection tendencies, and prescribed surface
            forcing.  ``None`` (default) reproduces the pre-forcing
            SCM behaviour bit-for-bit.
        t0_seconds
            Initial simulation time (seconds since model start) seen by
            forcing callables.  Defaults to ``0.0``.
        microphysics_substeps
            Fixed SCM-only substep count for active microphysics.  The
            default ``1`` reproduces the original single explicit
            microphysics update.  Values greater than one split the
            combined physics into a once-per-step non-microphysics
            tendency plus re-evaluated microphysics tendencies over
            ``dt / microphysics_substeps``.  This is available for
            ``time_integrator="forward_euler"`` and does not affect the
            plane/CRM path.
        """
        microphysics_substeps = _validate_microphysics_substeps(
            microphysics_substeps,
        )
        if physics_config.microphysics.scheme == "none":
            microphysics_substeps = DEFAULT_MICROPHYSICS_SUBSTEPS
        cls._validate_integrator_compatibility(physics_config, time_integrator)
        if microphysics_substeps > 1 and time_integrator != "forward_euler":
            raise ValueError(
                "microphysics_substeps > 1 is currently supported only with "
                "time_integrator='forward_euler'."
            )
        cls._validate_prescribed_fluxes_no_double_count(
            physics_config, forcing,
        )
        grid = make_scm_grid(latitude_deg, longitude_deg)
        sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top, dtype=dtype)

        needs_moist = (
            physics_config.convection.scheme != "none"
            or physics_config.microphysics.scheme != "none"
            or physics_config.turbulence.scheme != "none"
        )
        if needs_moist and q_v_profile is None:
            q_v_profile = jnp.zeros(nlev)
        state = make_column_state(
            nlev, T_profile=T_profile, q_v_profile=q_v_profile,
            u=u, v=v, p_s=p_s, phis=phis, dtype=dtype,
        )
        # Pre-allocate condensate/precip species that microphysics or
        # convection may emit, so the first physics call sees a
        # consistent tracer registry rather than relying on the
        # auto-materialisation path in _apply_tendencies.
        if state.tracers is not None and physics_config.microphysics.scheme != "none":
            extra_keys = ("q_c", "q_r", "q_i", "q_s")
            if microphysics_substeps > 1:
                extra_keys = extra_keys + ("q_g",)
            new_tracers = dict(state.tracers)
            for k in extra_keys:
                if k not in new_tracers:
                    zeros = jnp.zeros_like(state.tracers["q_v"].data)
                    new_tracers[k] = Field(
                        data=zeros, name=k, dims=_DIMS_3D, units="kg/kg",
                    )
            state = state._replace(tracers=new_tracers)
        microphysics_fn = None
        if (
            microphysics_substeps > 1
            and physics_config.microphysics.scheme != "none"
        ):
            nonmicro_config = physics_config._replace(
                microphysics=physics_config.microphysics._replace(scheme="none"),
            )
            physics_fn = make_physics(
                nonmicro_config, model_type="hydrostatic", dt=dt,
            )
            microphysics_fn = make_microphysics_physics(
                physics_config.microphysics,
                model_type="hydrostatic",
                dt=dt / microphysics_substeps,
            )
        else:
            physics_fn = make_physics(
                physics_config, model_type="hydrostatic", dt=dt,
            )
        phys_state = init_physics_state(
            ncol=1, nlev=nlev, physics_config=physics_config,
            dtype=dtype, prng_seed=prng_seed,
        )
        return cls(
            physics_fn=physics_fn, state=state, phys_state=phys_state,
            grid=grid, sigma_coord=sigma_coord, dt=dt,
            time_integrator=time_integrator,
            forcing=forcing, t0_seconds=t0_seconds,
            physics_config=physics_config,
            microphysics_substeps=microphysics_substeps,
            microphysics_fn=microphysics_fn,
            _built_by_create=True,
        )

    def set_time(self, day_of_year: float, seconds_of_day: float) -> None:
        """Propagate calendar time to sub-physics (radiation diurnal cycle)."""
        set_time = getattr(self.physics_fn, "set_time", None)
        if callable(set_time):
            set_time(day_of_year, seconds_of_day)

    def step(self) -> HydrostaticState:
        """Advance the column by one step using the selected integrator.

        Stage time is sampled from ``self.t_seconds`` and the simulation
        clock is advanced by ``self.dt`` after the step.  Time-dependent
        SCM forcing callables are evaluated at the correct stage abscissae
        inside the integrator (see the RK2 / RK4 steps in
        :mod:`legoesm.core.column_stepping`).
        """
        if self.time_integrator == "ab2":
            # Adams-Bashforth 2 — single ``_tend_fn`` evaluation per
            # outer step (no sub-stages, no bootstrap re-evaluation).
            # First step falls back to forward Euler using the *same*
            # tendency that gets cached as ``prev_tend``; evaluating
            # ``_tend_fn`` twice would risk side-effect divergence in
            # forcing callables (iterators, counters, mutable hooks)
            # and could pair the cached tendency with a different
            # ``phys_state`` than the one actually applied — Phase D
            # codex iter-1 high finding.
            tend_n, new_phys = self._tend_fn(
                self.state, self.phys_state, self.t_seconds,
            )
            if self._ab2_prev_tend is None:
                tend_eff = tend_n
            else:
                tend_eff = ab2_effective_tendency(
                    tend_n, self._ab2_prev_tend, _average_tendencies,
                )
            new_state = _apply_tendencies(self.state, tend_eff, self.dt)
            self._ab2_prev_tend = tend_n
        else:
            new_state, new_phys = self._step_fn(
                self.state, self.phys_state, self._tend_fn, self.dt,
                self.t_seconds,
            )
        self.t_seconds += self.dt
        self.state = new_state
        if new_phys is not None:
            self.phys_state = new_phys
        # NB: No end-of-step ``forcing.T_s`` re-stamp.  The inject
        # inside ``_tend_fn`` (per-stage) is the single source of
        # truth — calling ``T_s(t)`` again here would double-fire
        # iterator / counter / file-cursor backed forcing callables
        # for every model step (Phase D codex iter-2 high finding).
        # The trade-off: zero-physics runs (no tagged_fns) cannot
        # persist the override on ``self.phys_state`` between steps
        # because ``physics_fn`` returns ``new_phys=None``.  No module
        # reads the persistent override between stages, so this is
        # observationally invisible; tests that need to inspect the
        # persistent override after ``step()`` must use an
        # active-physics config (Louis / gray rad / MYNN).
        return self.state

    def run(
        self,
        nsteps: int,
        *,
        save_every: int = 1,
        start_day_of_year: float = 0.0,
        start_seconds_of_day: float = 0.0,
    ) -> tuple[HydrostaticState, SCMHistory]:
        """Integrate for ``nsteps`` physics steps and return final state + history.

        ``save_every`` controls the output cadence (1 = every step).
        Calendar time advances by ``dt`` each step so that radiation's
        diurnal cycle is exercised correctly.
        """
        seconds_per_day = 86400.0
        times = []
        Ts, us, vs, p_ss, qvs = [], [], [], [], []
        has_qv = (self.state.tracers is not None
                  and "q_v" in self.state.tracers)

        for k in range(nsteps):
            elapsed = k * self.dt
            doy = start_day_of_year + (start_seconds_of_day + elapsed) / seconds_per_day
            sod = (start_seconds_of_day + elapsed) % seconds_per_day
            self.set_time(doy, sod)
            self.step()
            if k % save_every == 0 or k == nsteps - 1:
                times.append((k + 1) * self.dt)
                Ts.append(self.state.T.data[0, 0, 0])
                us.append(self.state.u.data[0, 0, 0])
                vs.append(self.state.v.data[0, 0, 0])
                p_ss.append(self.state.p_s.data[0, 0, 0])
                if has_qv:
                    qvs.append(self.state.tracers["q_v"].data[0, 0, 0])

        history = SCMHistory(
            time=jnp.asarray(times),
            T=jnp.stack(Ts),
            u=jnp.stack(us),
            v=jnp.stack(vs),
            p_s=jnp.asarray(p_ss),
            q_v=jnp.stack(qvs) if has_qv else None,
        )
        return self.state, history
