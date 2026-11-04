"""SCM external forcing API for the single-column model.

Closure-style time-dependent large-scale forcing that mirrors jax_scm's
``Forcing`` interface: geostrophic wind + Coriolis, prescribed surface
temperature or surface fluxes, large-scale subsidence, prescribed
horizontal-advection tendencies on theta and q_v.

Each channel is a callable ``f(t_seconds) -> jax.Array``.  ``None`` on a
field disables that channel.  Surface-flux channels (``T_s``, ``w_th_s``,
``w_qv_s``) are carried on the same NT so the SCM driver tracks one
forcing object across Phase A (this file) and Phase B (the
prescribed-flux surface bypass that will consume them).

Pytree semantics
----------------
``SCMForcing`` carries Python callables and a literal-typed ``prescribe``
string — neither is a valid JAX array leaf.  The default NamedTuple
auto-registration would expose those as pytree leaves and break
``jax.tree_util`` consumers that assume array leaves (and would crash
``jit``/``vmap`` if the forcing were ever passed as a dynamic argument).
We therefore re-register :class:`SCMForcing` as a **flat-static** pytree
(no dynamic children; the whole object is auxiliary data).  The SCM
driver itself does not ``jit`` through the forcing — forcing callables
run host-side per step — so the static treatment matches how the object
is actually used.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.forcing.idealized.large_scale_forcing import (
    mask_inflow_endpoint_tendency as _mask_inflow_endpoint_tendency,
    upwind_dphi_dz_top2bottom as _upwind_dphi_dz_top2bottom,
)
from legoesm.atmosphere.physics.thermodynamics import liquid_water_theta
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_layer_dz,
)
from legoesm.core.state import HydrostaticState, HydrostaticTendencies


ProfileFn = Callable[[float], jax.Array]
ScalarFn = Callable[[float], jax.Array]


_PRESCRIBE = ("none", "T_s", "fluxes")
_SUBSIDENCE_ON = ("theta_qv", "theta_l_qt")


class SCMForcing(NamedTuple):
    """Time-dependent large-scale forcing for SCM.

    Fields
    ------
    f_c
        Coriolis parameter [1/s].  ``0.0`` disables Coriolis + geostrophic
        relaxation entirely (regardless of u_geo / v_geo).
    u_geo, v_geo
        Geostrophic wind callables.  Each returns a ``(nlev,)`` profile
        [m/s] at the requested time.  ``None`` is treated as zero.
    subsidence_w
        Large-scale vertical velocity callable.  Returns ``(nlev,)`` at
        full levels [m/s], **positive upward** (subsidence is therefore
        negative).  Applied as first-order upwind advection on theta and
        on the tracers selected by ``subsidence_tracers``.
    subsidence_tracers
        ``None`` (default): q_v when the column carries it (a dry column has
        nothing else to subside).  A tuple names the column tracers subsided
        besides theta EXPLICITLY, e.g. ``("q_v", "q_c")`` for a case whose
        reference subsides cloud water too (DYCOMS-II); every named tracer must
        exist in the column — a missing one is an error, not a silent drop.
    theta_adv, qv_adv
        Prescribed horizontal-advection tendency callables.  Return
        ``(nlev,)`` profiles in [K/s] (potential temperature) and
        [(kg/kg)/s] (water-vapor mixing ratio) respectively.
    prescribe
        ``"none"`` (default), ``"T_s"``, or ``"fluxes"``.  Phase B consumes
        this to select between surface-temperature and surface-flux
        bulk-flux paths.  Phase A only validates the field.
    T_s
        Prescribed surface temperature callable, returns scalar [K]
        (consumed when ``prescribe == "T_s"``).
    w_th_s, w_qv_s
        Prescribed surface kinematic heat / moisture fluxes, returning
        scalars [K m/s] and [(kg/kg) m/s] (consumed when
        ``prescribe == "fluxes"``).
    flux_to_closure
        WHERE the prescribed flux is applied, when ``prescribe == "fluxes"``.

        ``False`` (default, the historical behaviour): it enters the column
        as a flux-divergence tendency at the lowest cell, and the TURBULENCE
        closure sees a surface heat flux of exactly ZERO.  That is fine for a
        local gradient-diffusion scheme but disables the defining input of
        every flux-driven nonlocal one (YSU, Holtslag-Boville, EDMF all scale
        their countergradient / mixed-layer velocity on ``w'theta'_0``).

        ``True``: the flux is instead written per step into
        ``PhysicsState.surface_wth_override`` / ``surface_wqv_override`` and
        applied by the closure as its lower boundary condition, so a nonlocal
        scheme receives it.  ``compute_forcing_tendencies`` then skips the
        column injection, because applying BOTH counts the same flux twice.

        This is what lets a case whose flux VARIES IN TIME reach the closure
        at all: ``SurfaceLayerConfig.prescribed_shflx_w_m2`` is one scalar for
        the whole run, so the config route can only carry one instant of a
        diurnal cycle (Wangara Day 33 measured 113.4 W/m^2 held for 8 h
        against a cycle peaking at 13:00).
    subsidence_on
        Which thermodynamic pair ``subsidence_w`` advects. ``"theta_qv"``
        (default): potential temperature and the tracers above. ``"theta_l_qt"``:
        liquid-water potential temperature θ_l and total water q_t = q_v + q_c
        (rain excluded), as swirl-lm's GCM forcing does; the θ_l tendency is
        applied to θ and the q_t tendency to q_v, q_c untouched (PI 2026-11-01).
        Needs a ``q_c`` tracer; ``subsidence_tracers`` must stay unset.
    nudge_rate, theta_l_target, qt_target
        Relaxation of θ_l and q_t toward target profiles: ``nudge_rate`` [1/s],
        targets [K] and [kg/kg]. Tendencies ``r (θ_l* − θ_l)`` on θ and
        ``r (q_t* − q_t)`` on q_v. All three are set together or not at all.
    wind_relax_rate, u_target, v_target
        Relaxation of u and v toward target profiles at the uniform rate
        ``wind_relax_rate`` [1/s]; ``0.0`` disables it. Targets [m/s].
    rad_T_tendency
        Prescribed radiative temperature tendency [K/s], added to dT/dt.
    surface_flux_fn
        A surface law evaluated on the column each stage instead of a time
        series: ``surface_flux_fn(state, sigma_coord, t) -> SurfaceFluxRecord``
        (kinematic theta_l, total-water and momentum fluxes, positive up, plus
        the surface density and Exner function they refer to). Needs
        ``prescribe="fluxes"`` and ``flux_to_closure=True`` with ``w_th_s`` and
        ``w_qv_s`` unset; the record reaches the closure through the same
        per-step overrides (heat as ``rho_s c_pd exner_s w_theta_l``, stress as
        ``rho_s uw`` [Pa]).
    """
    f_c: float = 0.0
    u_geo: Optional[ProfileFn] = None
    v_geo: Optional[ProfileFn] = None
    subsidence_w: Optional[ProfileFn] = None
    theta_adv: Optional[ProfileFn] = None
    qv_adv: Optional[ProfileFn] = None
    prescribe: Literal["none", "T_s", "fluxes"] = "none"
    T_s: Optional[ScalarFn] = None
    w_th_s: Optional[ScalarFn] = None
    w_qv_s: Optional[ScalarFn] = None
    flux_to_closure: bool = False
    subsidence_tracers: Optional[tuple] = None
    subsidence_on: str = "theta_qv"
    nudge_rate: Optional[ProfileFn] = None
    theta_l_target: Optional[ProfileFn] = None
    qt_target: Optional[ProfileFn] = None
    wind_relax_rate: float = 0.0
    u_target: Optional[ProfileFn] = None
    v_target: Optional[ProfileFn] = None
    rad_T_tendency: Optional[ProfileFn] = None
    surface_flux_fn: Optional[Callable] = None


class SurfaceFluxRecord(NamedTuple):
    """One stage's surface fluxes from a surface law (see ``SCMForcing``):
    kinematic, positive up; ``rho_s`` [kg/m3] and ``exner_s`` the surface
    density and Exner function the energetic forms use."""

    w_theta_l: jax.Array
    w_q_t: jax.Array
    uw: jax.Array
    vw: jax.Array
    rho_s: jax.Array
    exner_s: jax.Array


def default_forcing() -> SCMForcing:
    """All-zero forcing.  Preserves the baseline (pre-forcing) SCM behaviour."""
    return SCMForcing()


# Register SCMForcing as a flat-static pytree so JAX never treats the
# Python callables or the ``prescribe`` string as array leaves.  All
# fields live in the auxiliary data slot; the dynamic-leaf list is
# empty.  This makes ``jax.tree_util.tree_leaves(forcing) == []`` and
# keeps generic tree consumers (loss reducers, optimizer state walkers)
# safe even when the SCM driver is reachable from a traced computation.
def _flatten_scm_forcing(forcing: "SCMForcing"):
    return (), forcing


def _unflatten_scm_forcing(aux_data, _children):
    return aux_data


jax.tree_util.register_pytree_node(
    SCMForcing, _flatten_scm_forcing, _unflatten_scm_forcing,
)


def validate_forcing(forcing: SCMForcing) -> None:
    """Raise on dispatch typos and inconsistent prescribed-surface config.

    State-blind checks only — for state-dependent rules (qv_adv requires
    a ``q_v`` tracer, Coriolis requires both wind components) see
    :func:`validate_forcing_against_state`.
    """
    if forcing.prescribe not in _PRESCRIBE:
        raise ValueError(
            f"SCMForcing.prescribe={forcing.prescribe!r} invalid; "
            f"choose from {_PRESCRIBE}."
        )
    if forcing.subsidence_on not in _SUBSIDENCE_ON:
        raise ValueError(
            f"SCMForcing.subsidence_on={forcing.subsidence_on!r} invalid; "
            f"choose from {_SUBSIDENCE_ON}."
        )
    if forcing.subsidence_on == "theta_l_qt" and forcing.subsidence_tracers is not None:
        raise ValueError(
            "SCMForcing.subsidence_on='theta_l_qt' subsides total water as a whole; "
            "subsidence_tracers names individual species and must stay unset.")
    nudge = (forcing.nudge_rate, forcing.theta_l_target, forcing.qt_target)
    if any(x is None for x in nudge) and not all(x is None for x in nudge):
        raise ValueError(
            "SCMForcing.nudge_rate, theta_l_target and qt_target are set together "
            "or not at all; a partial set would leave one nudging term silently off.")
    winds = (forcing.u_target, forcing.v_target)
    if forcing.wind_relax_rate != 0.0:
        if not forcing.wind_relax_rate > 0.0:
            raise ValueError(
                f"SCMForcing.wind_relax_rate must be >= 0, got {forcing.wind_relax_rate}")
        if any(x is None for x in winds):
            raise ValueError(
                "SCMForcing.wind_relax_rate != 0 requires both u_target and v_target.")
    elif any(x is not None for x in winds):
        raise ValueError(
            "SCMForcing.u_target/v_target are set but wind_relax_rate is 0; the "
            "relaxation would be silently off.")
    if forcing.prescribe == "T_s" and forcing.T_s is None:
        raise ValueError("prescribe='T_s' requires SCMForcing.T_s callable.")
    if forcing.surface_flux_fn is not None:
        if not (forcing.prescribe == "fluxes" and forcing.flux_to_closure):
            raise ValueError(
                "SCMForcing.surface_flux_fn needs prescribe='fluxes' and "
                "flux_to_closure=True; it reaches the closure only as the "
                "per-step surface override.")
        if forcing.w_th_s is not None or forcing.w_qv_s is not None:
            raise ValueError(
                "SCMForcing.surface_flux_fn and w_th_s/w_qv_s are exclusive; "
                "two surface fluxes for one boundary would be ambiguous.")
    if forcing.prescribe == "fluxes":
        if (forcing.w_th_s is None and forcing.w_qv_s is None
                and forcing.surface_flux_fn is None):
            raise ValueError(
                "prescribe='fluxes' requires at least one of w_th_s, w_qv_s "
                "or surface_flux_fn."
            )
    elif forcing.flux_to_closure:
        # Silently inert otherwise: nothing outside the prescribe='fluxes'
        # branch reads it, so a caller that set it on the wrong forcing would
        # believe the closure had a surface flux it never received.
        raise ValueError(
            "SCMForcing.flux_to_closure=True requires prescribe='fluxes'; "
            f"got prescribe={forcing.prescribe!r}, where there is no "
            "prescribed surface flux to hand to the closure."
        )


def validate_forcing_against_state(
    forcing: SCMForcing, state: HydrostaticState,
) -> None:
    """State-dependent forcing checks.

    Catches the silent-data-loss cases where:
      * ``qv_adv`` is configured but the column has no ``q_v`` tracer
        (the moisture tendency would be silently dropped on every step);
      * ``f_c != 0`` (Coriolis / geostrophic relaxation) but the state
        carries no separate meridional wind (MPAS-style cell-centred
        normal velocity on edges) — the Coriolis tendency would
        otherwise be silently zero.

    ``subsidence_w`` on a dry state is *not* an error — subsidence on
    ``q_v`` is a state-derived no-op when no q_v exists, distinct from
    the user-explicit ``qv_adv`` channel.

    Used by both :class:`SingleColumnModel` construction and the public
    :func:`compute_forcing_tendencies` entry point so direct callers
    cannot bypass the check.
    """
    has_qv = (
        state.tracers is not None and "q_v" in state.tracers
    )
    if forcing.qv_adv is not None and not has_qv:
        raise ValueError(
            "SCMForcing.qv_adv is set but the SCM state carries no "
            "'q_v' tracer — the moisture tendency would be silently "
            "discarded.  Either construct the column with "
            "``q_v_profile=...`` (zeros are fine) or remove qv_adv "
            "from the forcing."
        )
    if (
        forcing.prescribe == "fluxes"
        and forcing.w_qv_s is not None
        and not has_qv
    ):
        raise ValueError(
            "SCMForcing.prescribe='fluxes' with w_qv_s set requires "
            "the SCM state to carry a 'q_v' tracer.  Construct the "
            "column with ``q_v_profile=...`` (zeros are fine), or "
            "remove w_qv_s from the forcing."
        )
    if forcing.subsidence_w is not None:
        carried = set(state.tracers or {})
        missing = [n for n in (forcing.subsidence_tracers or ())
                   if n not in carried]
        if missing:
            raise ValueError(
                f"SCMForcing.subsidence_tracers names {missing} but the SCM state "
                f"carries only {sorted(carried)}; their subsidence would be "
                "silently dropped.")
    needs_qc = forcing.nudge_rate is not None or (
        forcing.subsidence_w is not None and forcing.subsidence_on == "theta_l_qt")
    if needs_qc and not (has_qv and "q_c" in state.tracers):
        raise ValueError(
            "SCMForcing θ_l/q_t nudging or subsidence_on='theta_l_qt' needs 'q_v' "
            "and 'q_c' tracers in the SCM state (θ_l and q_t are built from them); "
            f"the state carries {sorted(state.tracers or {})}.")
    if forcing.wind_relax_rate != 0.0 and state.v is None:
        raise ValueError(
            "SCMForcing.wind_relax_rate != 0 requires the SCM column to carry "
            "both horizontal wind components; state.v is None.")
    if forcing.f_c != 0.0 and state.v is None:
        raise ValueError(
            "SCMForcing.f_c != 0 (Coriolis / geostrophic relaxation) "
            "requires the SCM column to carry both horizontal wind "
            "components — state.v is None (MPAS-style state without a "
            "separate cell-centred meridional wind).  Construct the "
            "column with both u and v fields, or set f_c=0 to disable "
            "Coriolis forcing."
        )


# ---------------------------------------------------------------------------
# Forcing-tendency assembly
# ---------------------------------------------------------------------------


def _eval_profile(fn, t, nlev, dtype):
    if fn is None:
        return jnp.zeros((nlev,), dtype=dtype)
    out = jnp.asarray(fn(t), dtype=dtype)
    if out.shape != (nlev,):
        raise ValueError(
            f"SCMForcing profile callable returned shape {tuple(out.shape)}, "
            f"expected ({nlev},)."
        )
    return out


def compute_forcing_tendencies(
    state: HydrostaticState,
    sigma_coord,
    forcing: SCMForcing,
    t_seconds: float,
) -> HydrostaticTendencies:
    """Assemble HydrostaticTendencies from external SCM forcing at time ``t``.

    Returned tendency carries zeros on every channel that no forcing
    touches, so the SCM driver can sum forcing and physics tendencies via
    a plain field-wise add (see :func:`scm_forcing.add_tendencies`).

    Coriolis convention: ``du/dt = +f_c (v - v_g)``,
    ``dv/dt = -f_c (u - u_g)``.  In the northern hemisphere (``f_c > 0``)
    this drives Ekman spin-down toward the geostrophic profile under
    turbulent friction.

    Subsidence and advective tendencies are applied to **potential
    temperature**, then converted back to ``T`` through the Exner
    function.  The horizontal-advection callable's output is therefore
    interpreted as ``d(theta)/dt`` for thermodynamic consistency with the
    rest of the column physics, matching the jax_scm convention.

    State-blind *and* state-dependent validation both run first
    (mirroring the checks in :class:`SingleColumnModel.__init__`) so
    direct callers of this public assembler cannot silently lose
    forcing channels through dispatch typos, empty
    ``prescribe='fluxes'`` configurations, or missing column tracers
    (Phase D codex iter-4 medium finding).
    """
    validate_forcing(forcing)
    validate_forcing_against_state(forcing, state)

    u = state.u.data
    v_fld = state.v
    v = v_fld.data if v_fld is not None else None
    T = state.T.data
    p_s = state.p_s.data
    nlev = T.shape[-1]
    sd = T.dtype

    q_v_fld = None
    q_v = None
    if state.tracers is not None and "q_v" in state.tracers:
        q_v_fld = state.tracers["q_v"]
        q_v = q_v_fld.data

    p_full = sigma_coord.pressure_at_full(p_s).reshape(1, 1, 1, nlev)
    p_half = sigma_coord.pressure_at_half(p_s).reshape(1, 1, 1, nlev + 1)

    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v) if v is not None else None
    dT_dt = jnp.zeros_like(T)
    dqv_dt = jnp.zeros_like(q_v) if q_v is not None else None

    needs_z = (forcing.subsidence_w is not None)
    if needs_z:
        T_col = T.reshape(1, nlev)
        q_v_col = None if q_v is None else q_v.reshape(1, nlev)
        z_full_col, _ = compute_heights_from_sigma(
            T_col, p_half.reshape(1, nlev + 1), q_v=q_v_col,
        )
        z_full = z_full_col.reshape(1, 1, 1, nlev)
    else:
        z_full = None

    theta_l_mode = forcing.subsidence_on == "theta_l_qt"
    needs_thl = forcing.nudge_rate is not None or (
        forcing.subsidence_w is not None and theta_l_mode)
    needs_exner = (
        forcing.subsidence_w is not None or forcing.theta_adv is not None
        or needs_thl
    )
    exner = (
        (p_full / constants.p_ref) ** constants.kappa if needs_exner else None
    )
    if needs_thl:
        q_c = state.tracers["q_c"].data
        theta_l = liquid_water_theta(T / jnp.clip(exner, 1e-6, None), q_c, exner)
        q_t = q_v + q_c

    # --- Coriolis + geostrophic relaxation -----------------------------
    if forcing.f_c != 0.0 and v is not None:
        u_g = _eval_profile(forcing.u_geo, t_seconds, nlev, sd).reshape(
            1, 1, 1, nlev,
        )
        v_g = _eval_profile(forcing.v_geo, t_seconds, nlev, sd).reshape(
            1, 1, 1, nlev,
        )
        du_dt = du_dt + forcing.f_c * (v - v_g)
        dv_dt = dv_dt - forcing.f_c * (u - u_g)

    # --- Large-scale subsidence on theta and the named tracers ---------
    other_tend: dict = {}
    if forcing.subsidence_w is not None:
        w_ls = _eval_profile(
            forcing.subsidence_w, t_seconds, nlev, sd,
        ).reshape(1, 1, 1, nlev)
        theta = theta_l if theta_l_mode else T / jnp.clip(exner, 1e-6, None)
        dtheta_dz = _upwind_dphi_dz_top2bottom(theta, z_full, w_ls)
        dtheta_dt_subs = _mask_inflow_endpoint_tendency(
            -w_ls * dtheta_dz, w_ls,
        )
        dT_dt = dT_dt + exner * dtheta_dt_subs
        if theta_l_mode:
            dqt_dz = _upwind_dphi_dz_top2bottom(q_t, z_full, w_ls)
            dqv_dt = dqv_dt + _mask_inflow_endpoint_tendency(-w_ls * dqt_dz, w_ls)
            names = ()
        else:
            names = (forcing.subsidence_tracers
                     if forcing.subsidence_tracers is not None else ("q_v",))
        if dqv_dt is not None and "q_v" in names:
            dqv_dz = _upwind_dphi_dz_top2bottom(q_v, z_full, w_ls)
            dqv_subs = _mask_inflow_endpoint_tendency(
                -w_ls * dqv_dz, w_ls,
            )
            dqv_dt = dqv_dt + dqv_subs
        for name in names:
            if name == "q_v":
                continue
            fld = state.tracers[name]
            dphi_dz = _upwind_dphi_dz_top2bottom(fld.data, z_full, w_ls)
            other_tend[name] = fld.replace(
                data=_mask_inflow_endpoint_tendency(-w_ls * dphi_dz, w_ls))

    # --- Prescribed surface kinematic fluxes (prescribe="fluxes") ----
    # ``w_th_s`` [K m/s] and ``w_qv_s`` [(kg/kg) m/s] are bulk-formula
    # bypasses: they enter the column as a flux-divergence tendency at
    # the lowest cell.  The user is responsible for setting the
    # turbulence scheme's ``surface.Ch_neutral`` to zero (Phase B v1) so
    # the bulk-formula path does not double-count the prescribed
    # sensible/latent heat flux.  Momentum drag (``Cd``) can stay live
    # — it is not duplicated by these kinematic-flux channels.
    #
    # ``flux_to_closure=True`` moves the SAME flux to the closure's lower
    # boundary condition (``PhysicsState.surface_wth_override``, written by
    # ``scm._tendency_fn``), so this injection is skipped: running both would
    # apply the flux twice.  Static Python ``if`` on a config bool.
    if forcing.prescribe == "fluxes" and not forcing.flux_to_closure:
        dz_layer = compute_layer_dz(
            T.reshape(1, nlev),
            p_half.reshape(1, nlev + 1),
            q_v=None if q_v is None else q_v.reshape(1, nlev),
        )
        dz_bot = dz_layer[0, -1]  # scalar
        # dz_layer comes back positive even for top-to-bottom layouts
        # because ``compute_layer_dz`` wraps in ``jnp.abs``.  Floor at
        # 1 m to keep AD finite — pathological zero-thickness lowest
        # cells (mis-built sigma_coord) would otherwise blow up the
        # injected tendency.
        dz_bot_safe = jnp.maximum(dz_bot, 1.0)
        if forcing.w_th_s is not None:
            wth = jnp.asarray(forcing.w_th_s(t_seconds), dtype=sd)
            dT_dt = dT_dt.at[..., -1].add(wth / dz_bot_safe)
        if forcing.w_qv_s is not None:
            if dqv_dt is None:
                # validate_forcing_against_state would have caught a
                # state-shape mismatch; the only path to None here is a
                # bug, so we surface it loudly.
                raise ValueError(
                    "prescribe='fluxes' with w_qv_s set requires a q_v "
                    "tracer in the column state."
                )
            wqv = jnp.asarray(forcing.w_qv_s(t_seconds), dtype=q_v.dtype)
            dqv_dt = dqv_dt.at[..., -1].add(wqv / dz_bot_safe)

    # --- Prescribed horizontal-advection tendencies --------------------
    if forcing.theta_adv is not None:
        theta_tend = _eval_profile(
            forcing.theta_adv, t_seconds, nlev, sd,
        ).reshape(1, 1, 1, nlev)
        dT_dt = dT_dt + exner * theta_tend
    if forcing.qv_adv is not None and dqv_dt is not None:
        qv_tend = _eval_profile(
            forcing.qv_adv, t_seconds, nlev, q_v.dtype,
        ).reshape(1, 1, 1, nlev)
        dqv_dt = dqv_dt + qv_tend

    # --- Relaxation of θ_l / q_t (onto θ and q_v) and of the winds -------
    if forcing.nudge_rate is not None:
        rate = _eval_profile(forcing.nudge_rate, t_seconds, nlev, sd).reshape(1, 1, 1, nlev)
        thl_target = _eval_profile(
            forcing.theta_l_target, t_seconds, nlev, sd).reshape(1, 1, 1, nlev)
        qt_target = _eval_profile(
            forcing.qt_target, t_seconds, nlev, q_v.dtype).reshape(1, 1, 1, nlev)
        dT_dt = dT_dt + exner * rate * (thl_target - theta_l)
        dqv_dt = dqv_dt + rate * (qt_target - q_t)
    if forcing.wind_relax_rate != 0.0:
        u_t = _eval_profile(forcing.u_target, t_seconds, nlev, sd).reshape(1, 1, 1, nlev)
        v_t = _eval_profile(forcing.v_target, t_seconds, nlev, sd).reshape(1, 1, 1, nlev)
        du_dt = du_dt + forcing.wind_relax_rate * (u_t - u)
        dv_dt = dv_dt + forcing.wind_relax_rate * (v_t - v)

    # --- Prescribed radiative temperature tendency ----------------------
    if forcing.rad_T_tendency is not None:
        dT_dt = dT_dt + _eval_profile(
            forcing.rad_T_tendency, t_seconds, nlev, sd).reshape(1, 1, 1, nlev)

    tracer_tend = dict(other_tend) or None
    if dqv_dt is not None and q_v_fld is not None:
        tracer_tend = {"q_v": q_v_fld.replace(data=dqv_dt), **other_tend}

    return HydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt),
        dT_dt=state.T.replace(data=dT_dt),
        dp_s_dt=state.p_s.replace(data=jnp.zeros_like(p_s)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(p_s)),
        dv_dt=(v_fld.replace(data=dv_dt) if v_fld is not None else None),
        tracer_tendencies=tracer_tend,
    )


def inject_prescribed_T_sfc_into_phys_state(phys_state, T_s_value):
    """Write the prescribed surface temperature into
    ``phys_state.surface_T_sfc_override``.

    The turbulence integration reads that field via
    :func:`legoesm.atmosphere.physics.turbulence.integration._resolve_T_sfc`
    and uses it as the bulk-flux boundary value, so the sensible-flux
    gradient ``T_sfc − T[..., -1]`` stays non-zero even when ``T_sfc``
    is prescribed.  This is the **correct** way to inject a prescribed
    surface temperature for ``SCMForcing(prescribe="T_s")`` — mutating
    ``state.T[..., -1]`` would collapse the gradient and silently
    suppress the surface heat flux (Phase B codex iter-1 high finding).

    Non-finite values (``NaN`` / ``Inf``) are rejected at injection
    because :func:`_resolve_T_sfc` treats ``NaN`` as the "no override"
    sentinel — a missing value in the user's ``T_s(t)`` time series
    would otherwise silently fall back to ``T_col[:, -1]`` and erase
    the prescribed surface forcing (Phase B v2 codex iter-1 high
    finding).

    ``phys_state is None`` is a no-op: the SCM driver always
    constructs a PhysicsState, but defensive None-handling keeps this
    helper composable with future runtime configurations.
    ``T_s_value`` may be a scalar (Python float, 0-D array) or a
    ``(ncol,)`` array; it is broadcast to the override shape and cast
    to its dtype.
    """
    if phys_state is None:
        return phys_state
    override_dtype = phys_state.surface_T_sfc_override.dtype
    override_shape = phys_state.surface_T_sfc_override.shape
    T_s_arr = jnp.asarray(T_s_value, dtype=override_dtype)
    # Fail-fast on non-finite values.  A NaN/Inf override fails the
    # ``override > threshold`` test in ``_resolve_T_sfc`` (``NaN > x`` is
    # False), so it would SILENTLY disable the prescribed surface forcing for
    # the affected column — and re-introduce a non-finite value into the state
    # (the #911 sentinel is finite precisely to avoid that).  Materialise to
    # NumPy for the host-side check; the SCM driver is not jit-traced so this
    # is concrete by construction.
    from legoesm.atmosphere.physics.physics_state import (
        SFC_T_OVERRIDE_VALID_MIN,
    )
    import numpy as _np
    T_s_host = _np.asarray(T_s_arr)
    # Reject anything the resolvers would treat as "no override": non-finite
    # (also re-poisons the finite state) OR <= the validity threshold.  Both
    # the turbulence and radiation resolvers use ``> SFC_T_OVERRIDE_VALID_MIN``,
    # so a value at/below it would SILENTLY disable the prescribed forcing —
    # never a valid physical surface temperature [K] anyway.
    if not (_np.all(_np.isfinite(T_s_host))
            and _np.all(T_s_host > SFC_T_OVERRIDE_VALID_MIN)):
        raise ValueError(
            "SCMForcing.T_s(t) returned value(s) that are non-finite or "
            f"<= {SFC_T_OVERRIDE_VALID_MIN} K ({T_s_host!r}).  Such values are "
            "indistinguishable from the no-override sentinel and would "
            "silently disable the prescribed surface forcing.  Supply a clean "
            "physical surface-temperature series."
        )
    new_override = jnp.broadcast_to(T_s_arr, override_shape)
    return phys_state._replace(surface_T_sfc_override=new_override)


def inject_prescribed_surface_fluxes_into_phys_state(
    phys_state, *, wth=None, wqv=None, tau_x=None, tau_y=None,
):
    """Write this step's prescribed surface KINEMATIC fluxes into PhysicsState.

    The flux sibling of :func:`inject_prescribed_T_sfc_into_phys_state`, used
    for ``SCMForcing(prescribe="fluxes", flux_to_closure=True)``.  The
    turbulence integration reads these in
    ``turbulence.integration._resolve_prescribed_surface_fluxes`` and applies
    them as the closure's lower boundary condition, which is what a nonlocal
    scheme needs: ``SurfaceLayerConfig.prescribed_shflx_w_m2`` is one scalar
    for the whole run and cannot carry a diurnal cycle.

    ``wth`` [K m/s] and ``wqv`` [(kg/kg) m/s], positive UPWARD — the same
    units and sign as ``SCMForcing.w_th_s`` / ``w_qv_s``, unconverted.  Either
    may be ``None``, which leaves that channel absent (a dry case has no
    moisture flux, and writing a zero would be a different statement).
    ``tau_x``/``tau_y`` [Pa, stress on the atmosphere] go to the stress
    overrides the closure applies in place of its own drag.

    NO host-side validation here, deliberately, and this is the one real
    difference from the ``T_s`` injector: the value comes from a callable of a
    TRACED time inside the SCM's ``lax.scan``, so ``np.asarray`` on it would
    raise.  It does not need one either — absence is signalled by ``None``,
    not by an in-band sentinel, so a ``NaN`` flux stays ``NaN`` and surfaces
    downstream instead of silently reading as "no override" (which is exactly
    the failure the ``T_s`` injector's check exists to prevent).
    """
    if phys_state is None:
        return phys_state
    # Shape and dtype from the sibling override, which is always materialised
    # as ``(ncol,)`` by ``init_physics_state``.
    shape = phys_state.surface_T_sfc_override.shape
    dtype = phys_state.surface_T_sfc_override.dtype
    updates = {}
    if wth is not None:
        updates["surface_wth_override"] = jnp.broadcast_to(
            jnp.asarray(wth, dtype=dtype), shape)
    if wqv is not None:
        updates["surface_wqv_override"] = jnp.broadcast_to(
            jnp.asarray(wqv, dtype=dtype), shape)
    if tau_x is not None:
        updates["surface_tau_x_override_pa"] = jnp.broadcast_to(
            jnp.asarray(tau_x, dtype=dtype), shape)
    if tau_y is not None:
        updates["surface_tau_y_override_pa"] = jnp.broadcast_to(
            jnp.asarray(tau_y, dtype=dtype), shape)
    if not updates:
        return phys_state
    return phys_state._replace(**updates)


def add_tendencies(
    a: HydrostaticTendencies, b: HydrostaticTendencies,
) -> HydrostaticTendencies:
    """Field-wise sum of two HydrostaticTendencies.

    Handles ``dv_dt is None`` and ``tracer_tendencies is None`` /
    asymmetric tracer dicts symmetrically.  The output preserves Field
    metadata from ``a`` for every non-None channel.
    """
    du = a.du_dt.replace(data=a.du_dt.data + b.du_dt.data)
    dT = a.dT_dt.replace(data=a.dT_dt.data + b.dT_dt.data)
    dps = a.dp_s_dt.replace(data=a.dp_s_dt.data + b.dp_s_dt.data)
    dphis = a.dphis_dt.replace(data=a.dphis_dt.data + b.dphis_dt.data)
    dv = None
    if a.dv_dt is not None and b.dv_dt is not None:
        dv = a.dv_dt.replace(data=a.dv_dt.data + b.dv_dt.data)
    elif a.dv_dt is not None:
        dv = a.dv_dt
    elif b.dv_dt is not None:
        dv = b.dv_dt
    a_tr = a.tracer_tendencies or {}
    b_tr = b.tracer_tendencies or {}
    tracers: Optional[dict] = None
    if a_tr or b_tr:
        tracers = {}
        for k in set(a_tr) | set(b_tr):
            if k in a_tr and k in b_tr:
                tracers[k] = a_tr[k].replace(
                    data=a_tr[k].data + b_tr[k].data,
                )
            elif k in a_tr:
                tracers[k] = a_tr[k]
            else:
                tracers[k] = b_tr[k]
    return HydrostaticTendencies(
        du_dt=du, dT_dt=dT, dp_s_dt=dps, dphis_dt=dphis,
        dv_dt=dv, tracer_tendencies=tracers,
    )
