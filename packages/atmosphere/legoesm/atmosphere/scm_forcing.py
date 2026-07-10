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
from legoesm.atmosphere.large_scale_forcing import (
    mask_inflow_endpoint_tendency as _mask_inflow_endpoint_tendency,
    upwind_dphi_dz_top2bottom as _upwind_dphi_dz_top2bottom,
)
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_layer_dz,
)
from legoesm.core.state import HydrostaticState, HydrostaticTendencies


ProfileFn = Callable[[float], jax.Array]
ScalarFn = Callable[[float], jax.Array]


_PRESCRIBE = ("none", "T_s", "fluxes")


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
        q_v.
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
    if forcing.prescribe == "T_s" and forcing.T_s is None:
        raise ValueError("prescribe='T_s' requires SCMForcing.T_s callable.")
    if forcing.prescribe == "fluxes":
        if forcing.w_th_s is None and forcing.w_qv_s is None:
            raise ValueError(
                "prescribe='fluxes' requires at least one of w_th_s, w_qv_s."
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

    needs_exner = (
        forcing.subsidence_w is not None or forcing.theta_adv is not None
    )
    exner = (
        (p_full / constants.p_ref) ** constants.kappa if needs_exner else None
    )

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

    # --- Large-scale subsidence on theta and q_v -----------------------
    if forcing.subsidence_w is not None:
        w_ls = _eval_profile(
            forcing.subsidence_w, t_seconds, nlev, sd,
        ).reshape(1, 1, 1, nlev)
        theta = T / jnp.clip(exner, 1e-6, None)
        dtheta_dz = _upwind_dphi_dz_top2bottom(theta, z_full, w_ls)
        dtheta_dt_subs = _mask_inflow_endpoint_tendency(
            -w_ls * dtheta_dz, w_ls,
        )
        dT_dt = dT_dt + exner * dtheta_dt_subs
        if dqv_dt is not None:
            dqv_dz = _upwind_dphi_dz_top2bottom(q_v, z_full, w_ls)
            dqv_subs = _mask_inflow_endpoint_tendency(
                -w_ls * dqv_dz, w_ls,
            )
            dqv_dt = dqv_dt + dqv_subs

    # --- Prescribed surface kinematic fluxes (prescribe="fluxes") ----
    # ``w_th_s`` [K m/s] and ``w_qv_s`` [(kg/kg) m/s] are bulk-formula
    # bypasses: they enter the column as a flux-divergence tendency at
    # the lowest cell.  The user is responsible for setting the
    # turbulence scheme's ``surface.Ch_neutral`` to zero (Phase B v1) so
    # the bulk-formula path does not double-count the prescribed
    # sensible/latent heat flux.  Momentum drag (``Cd``) can stay live
    # — it is not duplicated by these kinematic-flux channels.
    if forcing.prescribe == "fluxes":
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

    tracer_tend = None
    if dqv_dt is not None and q_v_fld is not None:
        tracer_tend = {"q_v": q_v_fld.replace(data=dqv_dt)}

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
