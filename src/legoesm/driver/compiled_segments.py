"""Compiled segment execution for atmosphere time integration.

Instead of orchestrating every time step from Python (which prevents XLA
from fusing across steps), this module compiles *segments* of N steps
into a single ``jax.lax.scan``-based kernel.  The host Python only runs
between segments to perform side effects: diagnostics, checkpoints,
external forcing updates, and stability checks.

Architecture
------------
::

    Host Python loop (segment boundaries)
    ├── update external forcing (solar, ozone, aerosol)
    ├── run_segment(carry, segment_steps)   ← compiled via jax.lax.scan
    │     ├── dynamics step
    │     ├── physics (with radiation sub-cycling via lax.cond)
    │     ├── state update + saturation adjustment
    │     ├── moisture fixer
    │     ├── hyperdiffusion smoothing
    │     └── Rayleigh friction
    ├── diagnostics collection (host-side, after block_until_ready)
    └── checkpoint save (host-side I/O)

The segment carry (:class:`SegmentCarry`) packs every array the hot loop
needs into a single flat NamedTuple so ``jax.lax.scan`` can type-check
the carry across iterations.

Segment length is chosen as the GCD of diagnostic and checkpoint
intervals — this ensures every I/O boundary falls exactly on a segment
boundary, without needing to interrupt the compiled kernel mid-segment.
"""

from __future__ import annotations

import math
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

logger = logging.getLogger(__name__)


# ======================================================================
# Segment carry — all mutable arrays for the hot loop
# ======================================================================

class SegmentCarry(NamedTuple):
    """All mutable arrays threaded through the compiled segment.

    Every field is a JAX array.  No Python objects, no file handles,
    no host-side state — only data that can live inside ``lax.scan``.

    Attributes
    ----------
    u, v, T, p_s, phis : jax.Array
        Prognostic dynamics fields.
    q_v, q_c, q_r : jax.Array
        Moisture tracers.
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc : jax.Array
        Held radiation tendencies for sub-cycling.
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa : jax.Array
        Held radiation fluxes for sub-cycling.
    step_index : jax.Array
        Scalar int32 — absolute step counter (for radiation cadence).
    target_moisture : jax.Array
        Scalar float — fixed global moisture target for the fixer,
        computed once at initialization to prevent cross-step drift.
    precip_accum : jax.Array
        Accumulated precipitation over the segment [kg/m2].
    """
    u: jax.Array
    v: jax.Array
    T: jax.Array
    p_s: jax.Array
    phis: jax.Array
    q_v: jax.Array
    q_c: jax.Array
    q_r: jax.Array
    held_dT_rad: jax.Array
    held_sw_net_sfc: jax.Array
    held_lw_net_sfc: jax.Array
    held_sw_up_toa: jax.Array
    held_lw_up_toa: jax.Array
    held_sw_down_toa: jax.Array
    step_index: jax.Array
    target_moisture: jax.Array
    precip_accum: jax.Array


def pack_carry(state, q_v, q_c, q_r,
               held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
               held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
               step_index,
               target_moisture=None, precip_accum=None):
    """Pack driver state into a SegmentCarry for the compiled kernel."""
    if target_moisture is None:
        target_moisture = jnp.float32(0.0)
    if precip_accum is None:
        precip_accum = jnp.zeros_like(state.p_s.data)
    return SegmentCarry(
        u=state.u.data,
        v=state.v.data,
        T=state.T.data,
        p_s=state.p_s.data,
        phis=state.phis.data,
        q_v=q_v,
        q_c=q_c,
        q_r=q_r,
        held_dT_rad=held_dT_rad,
        held_sw_net_sfc=held_sw_net_sfc,
        held_lw_net_sfc=held_lw_net_sfc,
        held_sw_up_toa=held_sw_up_toa,
        held_lw_up_toa=held_lw_up_toa,
        held_sw_down_toa=held_sw_down_toa,
        step_index=jnp.int32(step_index),
        target_moisture=jnp.asarray(target_moisture),
        precip_accum=precip_accum,
    )


def unpack_carry(carry, state_template):
    """Unpack a SegmentCarry back into driver state objects.

    Parameters
    ----------
    carry : SegmentCarry
    state_template : HydrostaticState
        Template for field metadata (names, units, dims).

    Returns
    -------
    state, q_v, q_c, q_r, held_tuple, step_index
    """
    new_state = state_template._replace(
        u=state_template.u.replace(data=carry.u),
        v=state_template.v.replace(data=carry.v),
        T=state_template.T.replace(data=carry.T),
        p_s=state_template.p_s.replace(data=carry.p_s),
        phis=state_template.phis.replace(data=carry.phis),
    )
    held_tuple = (
        carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
        carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
    )
    return (new_state, carry.q_v, carry.q_c, carry.q_r,
            held_tuple, int(carry.step_index),
            carry.precip_accum)


# ======================================================================
# Segment boundary computation
# ======================================================================

def compute_segment_length(
    diag_interval: int,
    checkpoint_interval: int,
    rad_update_steps: int,
) -> int:
    """Compute the optimal segment length.

    The segment length is the GCD of all cadence intervals that require
    host-side actions (diagnostics, checkpoints, radiation forcing
    updates).  This ensures every cadence boundary falls on a segment
    boundary.

    Parameters
    ----------
    diag_interval : int
        Steps between diagnostic collections.
    checkpoint_interval : int
        Steps between checkpoints (0 = disabled).
    rad_update_steps : int
        Steps between radiation forcing refreshes.

    Returns
    -------
    int
        Segment length in time steps.  Always >= 1.
    """
    intervals = [i for i in [diag_interval, checkpoint_interval, rad_update_steps]
                 if i > 0]
    if not intervals:
        return 1
    seg = intervals[0]
    for i in intervals[1:]:
        seg = math.gcd(seg, i)
    return max(seg, 1)


# ======================================================================
# Compiled segment builder
# ======================================================================

def build_segment_fn(
    model,
    step_unified,
    grid,
    sigma_full,
    dsigma,
    dt: float,
    rad_update_steps: int,
    microphysics: str,
    fix_moisture: bool,
    fric_decay,
    qv_smooth_coeff,
    sst,
    sic,
    lat,
    lon,
    day_of_year,
    seconds_of_day,
    solar_weights,
    s_0,
    o3_vmr,
    aerosol_od,
    start_day: float,
    gradient_checkpoint: bool | None = None,
    hyperdiffusion_3d_fn=None,
    tau_equator=None,
    tau_pole=None,
    sbm_tau_c=None,
    sbm_RH_ref=None,
    C_H=None,
    C_E=None,
    albedo_ice=None,
    albedo_ocean=None,
    ghg_vmr_override=None,
):
    """Build a compiled segment function.

    The returned function runs ``n_steps`` of the full atmosphere
    integration (dynamics + physics + fixers) inside ``jax.lax.scan``,
    compiled as a single XLA program.

    Parameters
    ----------
    model
        Dynamics model with ``.step()`` and ``.step_with_physics()``.
    step_unified : callable
        JIT-compiled physics step from ``PhysicsPipeline.build_step_unified()``.
    grid
        Cubed-sphere or lat-lon grid.
    sigma_full, dsigma : jax.Array
        Vertical coordinate arrays.
    dt : float
        Time step in seconds.
    rad_update_steps : int
        Radiation update cadence.
    microphysics : str
        "none" or explicit scheme name.
    fix_moisture : bool
        Whether to apply moisture fixer.
    fric_decay, qv_smooth_coeff
        Rayleigh friction and smoothing parameters.
    sst, sic : jax.Array
        Sea surface temperature and ice concentration (fixed within segment).
    lat, lon : jax.Array
        Grid coordinates.
    day_of_year, seconds_of_day : float
        Time state (fixed within segment — updated at segment boundaries).
    solar_weights, s_0 : jax.Array / float
        Solar forcing (fixed within segment).
    o3_vmr, aerosol_od : jax.Array
        External forcing (fixed within segment).
    start_day : float
        Start day for the entire run (used to compute step→day).
    gradient_checkpoint : bool or None, optional
        If True, wrap the scan body with ``jax.checkpoint`` to trade
        recomputation for O(sqrt(N)) memory during reverse-mode AD.
        If None (default), automatically enable for segments longer
        than 50 steps to prevent OOM on large grids.

    Returns
    -------
    callable
        ``run_segment(carry: SegmentCarry, n_steps: int) -> SegmentCarry``
    """
    from legoesm.core.conservation import compute_global_moisture, fix_moisture_hydrostatic

    if hyperdiffusion_3d_fn is None:
        from legoesm.core.operators_3d import hyperdiffusion_3d
    else:
        hyperdiffusion_3d = hyperdiffusion_3d_fn

    do_sat_adjust = (microphysics == "none")


    # Convert scalars to JAX tracers once.
    _dt = jnp.float32(dt)
    _day_of_year = jnp.float32(day_of_year)
    _seconds_of_day = jnp.float32(seconds_of_day)
    _s_0 = jnp.float32(s_0)
    _solar_weights = jnp.asarray(solar_weights)
    _sst = jnp.asarray(sst)
    _sic = jnp.asarray(sic)
    _o3_vmr = jnp.asarray(o3_vmr)
    _aerosol_od = jnp.asarray(aerosol_od)
    _fric_decay = jnp.asarray(fric_decay)
    _tau_equator = jnp.float32(tau_equator) if tau_equator is not None else None
    _tau_pole = jnp.float32(tau_pole) if tau_pole is not None else None
    _sbm_tau_c = jnp.float32(sbm_tau_c) if sbm_tau_c is not None else None
    _sbm_RH_ref = jnp.float32(sbm_RH_ref) if sbm_RH_ref is not None else None
    _C_H = jnp.float32(C_H) if C_H is not None else None
    _C_E = jnp.float32(C_E) if C_E is not None else None
    _albedo_ice = jnp.float32(albedo_ice) if albedo_ice is not None else None
    _albedo_ocean = jnp.float32(albedo_ocean) if albedo_ocean is not None else None
    # Convert GHG VMR override dict values to JAX arrays for tracing.
    _ghg_vmr_override = None
    if ghg_vmr_override is not None:
        _ghg_vmr_override = {
            k: jnp.float64(v) for k, v in ghg_vmr_override.items()
        }

    def _single_step(carry: SegmentCarry, _unused) -> tuple:
        """One atmosphere step: dynamics → physics → fixers."""
        step_idx = carry.step_index

        # --- Dynamics ---
        # Reconstruct a minimal state for the model.step call.
        # The model operates on Field-wrapped NamedTuples, but inside
        # lax.scan we work with raw arrays.  We rely on the model's
        # internal JIT handling array inputs.
        dyn_state = model.step(
            _rebuild_state(carry, model),
            _dt,
        )

        T_new = dyn_state.T.data
        u_new = dyn_state.u.data
        v_new = dyn_state.v.data
        p_s_new = dyn_state.p_s.data

        # --- Physics with radiation sub-cycling ---
        need_rad = jnp.where(
            rad_update_steps <= 1,
            jnp.bool_(True),
            ((step_idx + 1) % rad_update_steps) == 0,
        )

        phys_out, held_new = step_unified(
            need_rad,
            T_new, p_s_new,
            carry.q_v, carry.q_c, carry.q_r,
            u_new, v_new,
            _sst, _sic, lat, lon,
            _day_of_year, _seconds_of_day, _dt,
            _solar_weights, _s_0,
            _o3_vmr, _aerosol_od,
            carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
            carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
            tau_equator=_tau_equator, tau_pole=_tau_pole,
            sbm_tau_c=_sbm_tau_c, sbm_RH_ref=_sbm_RH_ref,
            C_H=_C_H, C_E=_C_E,
            albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
            ghg_vmr_override=_ghg_vmr_override,
        )

        # --- State update ---
        T_upd = T_new + _dt * phys_out.dT_dt
        q_v_upd = jnp.maximum(carry.q_v + _dt * phys_out.dq_v_dt, 0.0)
        q_c_upd = jnp.maximum(carry.q_c + _dt * phys_out.dq_c_dt, 0.0)
        q_r_upd = jnp.maximum(carry.q_r + _dt * phys_out.dq_r_dt, 0.0)

        # --- Saturation adjustment ---
        if do_sat_adjust:
            p_full = p_s_new[..., None] * sigma_full
            q_sat = saturation_mixing_ratio(T_upd, p_full)
            excess = jnp.maximum(q_v_upd - q_sat, 0.0)
            q_v_upd = q_v_upd - excess
            T_upd = T_upd + constants.L_v * excess / constants.c_pd

        # --- Moisture fixer (uses fixed target from initialization) ---
        if fix_moisture:
            q_v_upd = fix_moisture_hydrostatic(
                q_v_upd, carry.target_moisture,
                p_s_new, dsigma, grid,
            )

        # --- Moisture smoothing ---
        q_v_upd = jnp.maximum(
            q_v_upd + _dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
            0.0,
        )

        # --- Rayleigh friction ---
        u_upd = u_new * _fric_decay
        v_upd = v_new * _fric_decay

        # --- Accumulate precipitation ---
        precip_step = phys_out.precipitation if hasattr(phys_out, 'precipitation') else jnp.zeros_like(p_s_new)
        precip_accum = carry.precip_accum + precip_step * _dt

        # Cast all arrays back to carry input dtypes to prevent
        # float32→float64 promotion from Python float constants
        # (e.g., constants.L_v, constants.c_pd) breaking jax.lax.scan.
        def _match_dtype(new_val, ref_val):
            if hasattr(ref_val, 'dtype') and hasattr(new_val, 'dtype'):
                return new_val.astype(ref_val.dtype) if new_val.dtype != ref_val.dtype else new_val
            return new_val

        new_carry = SegmentCarry(
            u=_match_dtype(u_upd, carry.u),
            v=_match_dtype(v_upd, carry.v),
            T=_match_dtype(T_upd, carry.T),
            p_s=_match_dtype(p_s_new, carry.p_s),
            phis=carry.phis,
            q_v=_match_dtype(q_v_upd, carry.q_v),
            q_c=_match_dtype(q_c_upd, carry.q_c),
            q_r=_match_dtype(q_r_upd, carry.q_r),
            held_dT_rad=_match_dtype(held_new[0], carry.held_dT_rad),
            held_sw_net_sfc=_match_dtype(held_new[1], carry.held_sw_net_sfc),
            held_lw_net_sfc=_match_dtype(held_new[2], carry.held_lw_net_sfc),
            held_sw_up_toa=_match_dtype(held_new[3], carry.held_sw_up_toa),
            held_lw_up_toa=_match_dtype(held_new[4], carry.held_lw_up_toa),
            held_sw_down_toa=_match_dtype(held_new[5], carry.held_sw_down_toa),
            step_index=step_idx + 1,
            target_moisture=carry.target_moisture,
            precip_accum=_match_dtype(precip_accum, carry.precip_accum),
        )
        return new_carry, None

    # Optionally wrap scan body with gradient checkpointing so that
    # reverse-mode AD uses O(sqrt(N)) memory instead of O(N).
    _step_fn = _single_step
    if gradient_checkpoint:
        _step_fn = jax.checkpoint(_single_step, prevent_cse=False)

    @partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
    def run_segment(carry: SegmentCarry, n_steps: int) -> SegmentCarry:
        """Run n_steps of the atmosphere integration as a compiled kernel.

        Parameters
        ----------
        carry : SegmentCarry
            Input state.
        n_steps : int
            Number of steps to execute.  This is a **static** argument:
            ``jax.lax.scan`` requires a concrete ``length``, so changing
            ``n_steps`` triggers recompilation.  In practice the segment
            length is constant (the GCD of cadence intervals), so this
            causes at most one extra compile for the final short segment.

        Returns
        -------
        SegmentCarry
            Updated state after n_steps.
        """
        final_carry, _ = jax.lax.scan(_step_fn, carry, None, length=n_steps)
        return final_carry

    return run_segment


def _rebuild_state(carry: SegmentCarry, model):
    """Rebuild the model's expected state type from raw carry arrays.

    The dynamics model expects a NamedTuple with Field-wrapped arrays.
    Inside lax.scan we store raw arrays, so we reconstruct the state
    type here.  This is cheap — only Python object creation, no data copy.
    """
    from legoesm.core.field import Field

    # Detect state type from model
    if hasattr(model, '_state_type'):
        StateType = model._state_type
    else:
        # Fallback: use HydrostaticState (the most common case)
        from legoesm.core.state import HydrostaticState
        StateType = HydrostaticState

    # Build Fields with minimal metadata
    u_f = Field(carry.u, name="u", dims=("face", "x", "y", "level"), units="m/s")
    v_f = Field(carry.v, name="v", dims=("face", "x", "y", "level"), units="m/s")
    T_f = Field(carry.T, name="T", dims=("face", "x", "y", "level"), units="K")
    p_s_f = Field(carry.p_s, name="p_s", dims=("face", "x", "y"), units="Pa")
    phis_f = Field(carry.phis, name="phis", dims=("face", "x", "y"), units="m2/s2")

    return StateType(u=u_f, v=v_f, T=T_f, p_s=p_s_f, phis=phis_f)
