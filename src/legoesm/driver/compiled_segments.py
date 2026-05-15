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

import copy
import math
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
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
    conv_prog : jax.Array
        Prognostic convection control state for mass_flux / EDMF schemes.
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc : jax.Array
        Held radiation tendencies for sub-cycling.
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa : jax.Array
        Held radiation fluxes for sub-cycling.
    step_index : jax.Array
        Scalar int32 — absolute step counter (for radiation cadence).
    target_moisture : jax.Array
        Scalar float — fixed global moisture target for the fixer,
        computed once at initialization to prevent cross-step drift.
    target_mass : jax.Array
        Scalar float — fixed global dry mass target (∫ p_s * dA at t=0),
        used for target-anchored mass conservation in long runs.
        Zero disables the fixer.
    max_cfl : jax.Array
        Scalar float — maximum CFL number observed during the segment.
        Monitored at segment boundaries for adaptive dt.
    precip_accum : jax.Array
        Accumulated precipitation over the segment [kg/m2].
    shflx_accum : jax.Array
        Accumulated sensible heat flux [W/m2 * s] over the segment.
    lhflx_accum : jax.Array
        Accumulated latent heat flux [W/m2 * s] over the segment.
    """
    u: jax.Array
    v: jax.Array
    T: jax.Array
    p_s: jax.Array
    phis: jax.Array
    q_v: jax.Array
    q_c: jax.Array
    q_r: jax.Array
    conv_prog: jax.Array
    held_dT_rad: jax.Array
    held_sw_net_sfc: jax.Array
    held_lw_net_sfc: jax.Array
    held_sw_up_toa: jax.Array
    held_lw_up_toa: jax.Array
    held_sw_down_toa: jax.Array
    step_index: jax.Array
    target_moisture: jax.Array
    target_mass: jax.Array
    max_cfl: jax.Array
    precip_accum: jax.Array
    shflx_accum: jax.Array
    lhflx_accum: jax.Array


def pack_carry(state, q_v, q_c, q_r, conv_prog=None, *,
               held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
               held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
               step_index,
               target_moisture=None, target_mass=None,
               max_cfl=None, precip_accum=None,
               shflx_accum=None, lhflx_accum=None):
    """Pack driver state into a SegmentCarry for the compiled kernel.

    Prognostic fields are cast to at least the precision policy's storage
    dtype (upcasting only — never downcasts existing float64 arrays).
    Accumulation scalars use at least the accumulate dtype.
    """
    from legoesm.core.precision import _resolve_dtype
    storage = _resolve_dtype(None, "storage")
    accum = _resolve_dtype(None, "accumulate")

    def _promote(x, target_dt):
        """Cast *x* to the wider of its current dtype and *target_dt*."""
        if hasattr(x, 'dtype'):
            dt = jnp.result_type(x.dtype, target_dt)
            return x.astype(dt) if x.dtype != dt else x
        return jnp.asarray(x, dtype=target_dt)

    if target_moisture is None:
        target_moisture = jnp.asarray(0.0, dtype=accum)
    if target_mass is None:
        target_mass = jnp.asarray(0.0, dtype=accum)
    if max_cfl is None:
        max_cfl = jnp.asarray(0.0)
    if precip_accum is None:
        precip_accum = jnp.zeros_like(state.p_s.data)
    if shflx_accum is None:
        shflx_accum = jnp.zeros_like(state.p_s.data)
    if lhflx_accum is None:
        lhflx_accum = jnp.zeros_like(state.p_s.data)
    if conv_prog is None:
        conv_prog = jnp.zeros((state.p_s.data.size,), dtype=storage)
    return SegmentCarry(
        u=_promote(state.u.data, storage),
        v=_promote(state.v.data, storage),
        T=_promote(state.T.data, storage),
        p_s=_promote(state.p_s.data, storage),
        phis=_promote(state.phis.data, storage),
        q_v=_promote(q_v, storage),
        q_c=_promote(q_c, storage),
        q_r=_promote(q_r, storage),
        conv_prog=_promote(conv_prog, storage),
        held_dT_rad=_promote(held_dT_rad, storage),
        held_sw_net_sfc=_promote(held_sw_net_sfc, storage),
        held_lw_net_sfc=_promote(held_lw_net_sfc, storage),
        held_sw_up_toa=_promote(held_sw_up_toa, storage),
        held_lw_up_toa=_promote(held_lw_up_toa, storage),
        held_sw_down_toa=_promote(held_sw_down_toa, storage),
        step_index=jnp.int32(step_index),
        target_moisture=_promote(target_moisture, accum),
        target_mass=_promote(target_mass, accum),
        max_cfl=jnp.asarray(max_cfl),
        precip_accum=_promote(precip_accum, storage),
        shflx_accum=_promote(shflx_accum, storage),
        lhflx_accum=_promote(lhflx_accum, storage),
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
    state, q_v, q_c, q_r, conv_prog, held_tuple, step_index, precip_accum,
    shflx_accum, lhflx_accum
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
    return (new_state, carry.q_v, carry.q_c, carry.q_r, carry.conv_prog,
            held_tuple, int(carry.step_index),
            carry.precip_accum,
            carry.shflx_accum, carry.lhflx_accum)


# ======================================================================
# Segment boundary computation
# ======================================================================

def compute_segment_length(
    diag_interval: int,
    checkpoint_interval: int,
    rad_update_steps: int = 0,
) -> int:
    """Compute the optimal segment length.

    The segment length is the GCD of all cadence intervals that require
    host-side actions (diagnostics, checkpoints).  This ensures every
    cadence boundary falls on a segment boundary.

    ``rad_update_steps`` is intentionally **excluded** from the GCD
    because radiation sub-cycling is handled inside the compiled scan
    body via ``jnp.where`` / modulo — it does not require a host-side
    segment boundary.  Including it collapses the segment length to 1
    whenever ``rad_update_steps=1`` (the default), eliminating all
    ``jax.lax.scan`` batching.

    Parameters
    ----------
    diag_interval : int
        Steps between diagnostic collections.
    checkpoint_interval : int
        Steps between checkpoints (0 = disabled).
    rad_update_steps : int
        Kept for API compatibility but not used in the GCD.

    Returns
    -------
    int
        Segment length in time steps.  Always >= 1.
    """
    intervals = [i for i in [diag_interval, checkpoint_interval]
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

def _match_dtype(new_val, ref_val):
    """Cast new_val to ref_val's dtype if they differ."""
    if hasattr(ref_val, 'dtype') and hasattr(new_val, 'dtype'):
        return new_val.astype(ref_val.dtype) if new_val.dtype != ref_val.dtype else new_val
    return new_val


class SegmentForcing(NamedTuple):
    """Per-segment external forcing arrays.

    These change at every segment boundary (SST, solar, ozone, etc.)
    and are passed as explicit arguments to ``run_segment`` so that the
    compiled kernel can be reused across segments without recompilation.

    ``land_fraction`` (iter-257, AIMIP Phase 2.6) is the land grid-cell
    fraction in [0, 1].  Defaults to all-zero (pure ocean) for backward
    compatibility with analytical AMIP runs; ERA5-IC AIMIP runs
    populate it from the reanalysis land-sea mask.  When non-zero, the
    surface albedo blend in
    :meth:`legoesm.driver.physics_pipeline.PhysicsPipeline.compute_radiation_core`
    becomes a 3-way blend
    ``lf * albedo_land + (1 - lf) * (sic * albedo_ice + (1 - sic) * albedo_ocean)``.
    """
    sst: jax.Array
    sic: jax.Array
    day_of_year: jax.Array
    seconds_of_day: jax.Array
    solar_weights: jax.Array
    s_0: jax.Array
    o3_vmr: jax.Array
    aerosol_od: jax.Array
    ghg_vmr: jax.Array  # shape (n_species,); empty (0,) when inactive
    land_fraction: jax.Array  # in [0,1]; shape matches sst


# Canonical GHG species ordering for the ghg_vmr array.
GHG_SPECIES_ORDER = ("co2", "ch4", "n2o", "cfc11", "cfc12")


def ghg_dict_to_array(ghg_dict: dict | None) -> jax.Array:
    """Convert a GHG VMR dict to a flat array in canonical order.

    Returns shape ``(n,)`` where *n* is the number of species present
    in ``ghg_dict`` (in :data:`GHG_SPECIES_ORDER`), or ``(0,)`` if
    *ghg_dict* is None.
    """
    if ghg_dict is None:
        return jnp.zeros(0)
    vals = [ghg_dict[k] for k in GHG_SPECIES_ORDER if k in ghg_dict]
    return jnp.asarray(vals)


def ghg_array_to_dict(ghg_arr: jax.Array, ghg_keys: tuple[str, ...]) -> dict | None:
    """Reconstruct a GHG VMR dict from a flat array + key list.

    Parameters
    ----------
    ghg_arr : jax.Array, shape (n,)
    ghg_keys : tuple of str
        Species names in the same order used to build *ghg_arr*.

    Returns None when *ghg_keys* is empty (no GHG override).
    """
    if not ghg_keys:
        return None
    return {k: ghg_arr[i] for i, k in enumerate(ghg_keys)}


def pack_forcing(
    sst, sic, day_of_year, seconds_of_day,
    solar_weights, s_0, o3_vmr, aerosol_od,
    ghg_vmr=None,
    land_fraction=None,
) -> SegmentForcing:
    """Pack per-segment forcing into a SegmentForcing pytree.

    Parameters
    ----------
    ghg_vmr : dict, jax.Array, or None
        GHG volume mixing ratios.  Accepts a dict (auto-converted via
        :func:`ghg_dict_to_array`), a pre-packed array, or None.
    land_fraction : array-like or None
        Land grid-cell fraction in [0, 1] (iter-257, AIMIP Phase 2.6).
        ``None`` defaults to all-zero (pure ocean) for backward
        compatibility with analytical AMIP runs.  ERA5 AIMIP runs pass
        the reanalysis land-sea mask here; the surface-albedo blend in
        radiation then becomes a 3-way ocean / ice / land blend.
    """
    if ghg_vmr is None:
        _ghg = jnp.zeros(0)
    elif isinstance(ghg_vmr, dict):
        _ghg = ghg_dict_to_array(ghg_vmr)
    else:
        _ghg = jnp.asarray(ghg_vmr)
    _sst = jnp.asarray(sst)
    if land_fraction is None:
        _lf = jnp.zeros_like(_sst)
    else:
        _lf = jnp.asarray(land_fraction)
    return SegmentForcing(
        sst=_sst,
        sic=jnp.asarray(sic),
        day_of_year=jnp.asarray(day_of_year),
        seconds_of_day=jnp.asarray(seconds_of_day),
        solar_weights=jnp.asarray(solar_weights),
        s_0=jnp.asarray(s_0),
        o3_vmr=jnp.asarray(o3_vmr),
        aerosol_od=jnp.asarray(aerosol_od),
        ghg_vmr=_ghg,
        land_fraction=_lf,
    )


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
    fix_mass: bool,
    fric_decay,
    qv_smooth_coeff,
    lat,
    lon,
    start_day: float,
    gradient_checkpoint: bool | None = None,
    hyperdiffusion_3d_fn=None,
    tau_equator=None,
    tau_pole=None,
    sbm_tau_c=None,
    sbm_RH_ref=None,
    sbm_CAPE_threshold=None,
    C_H=None,
    C_E=None,
    albedo_ice=None,
    albedo_ocean=None,
    albedo_land=None,
    l_mix_max=None,
    ghg_vmr_override=None,
    owned_face_ids=None,
    hs_newtonian_relax=None,
):
    """Build a compiled segment function.

    The returned function runs ``n_steps`` of the full atmosphere
    integration (dynamics + physics + fixers) inside ``jax.lax.scan``,
    compiled as a single XLA program.

    Per-segment forcing (SST, SIC, solar, ozone, aerosol) is passed as
    a ``SegmentForcing`` argument to the returned function, **not**
    captured in the closure.  This allows the same compiled kernel to be
    reused across segments without JIT recompilation.

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
    lat, lon : jax.Array
        Grid coordinates.
    start_day : float
        Start day for the entire run (used to compute step→day).
    gradient_checkpoint : bool or None, optional
        If True, wrap the scan body with ``jax.checkpoint`` to trade
        recomputation for O(sqrt(N)) memory during reverse-mode AD.
        If None (default), automatically enable for segments longer
        than 50 steps to prevent OOM on large grids.
    owned_face_ids : jax.Array or None, optional
        Indices of faces owned by this MPI rank (e.g. ``[0, 1]``).
        Used for replicated-dynamics MPI: conservation fixers sum
        only owned faces, physics operates on owned columns only.
        ``None`` means single-rank (all faces owned).

    Returns
    -------
    callable
        ``run_segment(carry: SegmentCarry, n_steps: int,
        forcing: SegmentForcing) -> SegmentCarry``
    """
    from legoesm.core.conservation import fix_moisture_hydrostatic, fix_ps_mass_target
    from legoesm.core.cfl import estimate_min_dx_cubed_sphere

    # Precompute minimum grid spacing for CFL monitoring
    if hasattr(grid, 'n'):
        _dx_min = jnp.asarray(estimate_min_dx_cubed_sphere(grid.n))
    else:
        _dx_min = jnp.asarray(1e6)  # safe default for non-cubed-sphere grids

    if hyperdiffusion_3d_fn is None:
        from legoesm.core.operators_3d import hyperdiffusion_3d
    else:
        hyperdiffusion_3d = hyperdiffusion_3d_fn

    do_sat_adjust = (microphysics == "none")

    # Convert static scalars to JAX arrays once (these don't change per segment).
    _dt = jnp.asarray(dt)
    _fric_decay = jnp.asarray(fric_decay)
    _tau_equator = jnp.asarray(tau_equator) if tau_equator is not None else None
    _tau_pole = jnp.asarray(tau_pole) if tau_pole is not None else None
    _sbm_tau_c = jnp.asarray(sbm_tau_c) if sbm_tau_c is not None else None
    _sbm_RH_ref = jnp.asarray(sbm_RH_ref) if sbm_RH_ref is not None else None
    _sbm_CAPE_threshold = (
        jnp.asarray(sbm_CAPE_threshold) if sbm_CAPE_threshold is not None else None
    )
    _C_H = jnp.asarray(C_H) if C_H is not None else None
    _C_E = jnp.asarray(C_E) if C_E is not None else None
    _albedo_ice = jnp.asarray(albedo_ice) if albedo_ice is not None else None
    _albedo_ocean = jnp.asarray(albedo_ocean) if albedo_ocean is not None else None
    _albedo_land = jnp.asarray(albedo_land) if albedo_land is not None else None
    _l_mix_max = jnp.asarray(l_mix_max) if l_mix_max is not None else None
    # GHG VMR: species key order is static (captured in closure);
    # values are dynamic (passed via SegmentForcing.ghg_vmr).
    _ghg_keys: tuple[str, ...] = ()
    if ghg_vmr_override is not None:
        _ghg_keys = tuple(k for k in GHG_SPECIES_ORDER if k in ghg_vmr_override)

    # Build owned-face mask for MPI replicated dynamics.
    # Shape (6,) with 1.0 for owned faces, 0.0 for non-owned.
    # None when single-rank (no masking needed).
    _owned_mask = None
    if owned_face_ids is not None:
        _owned_mask = jnp.zeros(6, dtype=jnp.float32)
        _owned_mask = _owned_mask.at[owned_face_ids].set(1.0)

    # Iter 8: when the segment driver applies a target-anchored
    # ``fix_ps_mass_target`` immediately after the dycore step, the
    # dycore's *own* end-step mass fixer is redundant — both reduce
    # mass globally over the same surface-pressure field, and the
    # segment fixer overwrites whatever the dycore fixer produced.
    # That is one extra global allreduce per compiled timestep on the
    # MPI/SPMD path, *inside* the lax.scan body where it is hard to
    # hide with overlap.  Disable the inner fixer while the outer one
    # is active by working with a shallow-cloned model whose config has
    # ``fix_mass=False``.  Behaviour is unchanged because the segment
    # fixer is strictly stronger (anchored to ``carry.target_mass``).
    _dynamics_model = model
    _model_cfg = getattr(model, "config", None)
    if (
        fix_mass
        and getattr(_model_cfg, "fix_mass", False)
        and hasattr(_model_cfg, "_replace")
    ):
        # Iter 8 dropped the redundant inner-dycore mass fixer.  Iter 9
        # follow-up: turning off ``fix_mass`` on the inner copy re-enables
        # the per-RK-stage ``zero_mean_ps_tendency`` allreduce, because
        # the iter-1 gating in :mod:`primitive_eq_cdgrid` was
        # "skip per-stage zero-mean *only when* end-step fix_mass is on".
        # Closing the loop: also disable the per-stage zero-mean on the
        # inner copy so the segment driver sees zero RK-stage allreduces
        # in addition to zero end-step inner allreduces.  The outer
        # target-anchored fixer enforces conservation once per timestep.
        _dynamics_model = copy.copy(model)
        _replace_kwargs = {"fix_mass": False}
        if hasattr(_model_cfg, "zero_mean_ps_tendency"):
            _replace_kwargs["zero_mean_ps_tendency"] = False
        _dynamics_model.config = _model_cfg._replace(**_replace_kwargs)

    def _make_single_step(forcing: SegmentForcing):
        """Create the scan body closed over a specific forcing pytree.

        The forcing is passed through the scan as a constant (not
        varying per step), so closing here is equivalent to passing it
        in scan's xs — but simpler.
        """
        # Reconstruct GHG VMR dict from forcing array + static keys.
        # _ghg_keys is a Python tuple captured in the closure; its
        # length determines whether step_unified receives None or dict.
        _ghg_vmr_override = ghg_array_to_dict(forcing.ghg_vmr, _ghg_keys)

        def _single_step(carry: SegmentCarry, _unused) -> tuple:
            """One atmosphere step: dynamics → physics → fixers."""
            step_idx = carry.step_index

            # --- Dynamics ---
            dyn_state = _dynamics_model.step(
                _rebuild_state(carry, _dynamics_model),
                _dt,
            )

            T_new = dyn_state.T.data
            u_new = dyn_state.u.data
            v_new = dyn_state.v.data
            p_s_new = dyn_state.p_s.data

            # --- Dry mass fixer (target-anchored) ---
            if fix_mass:
                p_s_new = fix_ps_mass_target(
                    p_s_new, carry.target_mass, grid, owned_mask=_owned_mask,
                )

            # --- Physics with radiation sub-cycling ---
            # ``rad_update_steps`` is a Python ``int`` captured in this
            # closure — gate with a Python ``if`` so the dead branch is
            # never traced.  The previous ``jnp.where`` on a static int
            # forced both branches into the trace and added an unused
            # modulo on every scan step (CLAUDE.md JAX rules).
            if rad_update_steps <= 1:
                need_rad = jnp.bool_(True)
            else:
                need_rad = ((step_idx + 1) % rad_update_steps) == 0

            if owned_face_ids is not None:
                # MPI replicated dynamics: physics on owned faces only.
                # Dynamics state is full (6, n, n, ...) but physics inputs
                # (forcing, lat/lon) are rank-local.  Extract owned faces
                # from dynamics fields, run physics, write back.
                _ofi = owned_face_ids
                phys_out, held_new_local = step_unified(
                    need_rad,
                    T_new[_ofi], p_s_new[_ofi],
                    carry.q_v[_ofi], carry.q_c[_ofi], carry.q_r[_ofi],
                    carry.conv_prog,
                    u_new[_ofi], v_new[_ofi],
                    forcing.sst, forcing.sic, lat, lon,
                    forcing.day_of_year, forcing.seconds_of_day, _dt,
                    forcing.solar_weights, forcing.s_0,
                    forcing.o3_vmr, forcing.aerosol_od,
                    carry.held_dT_rad[_ofi], carry.held_sw_net_sfc[_ofi],
                    carry.held_lw_net_sfc[_ofi],
                    carry.held_sw_up_toa[_ofi], carry.held_lw_up_toa[_ofi],
                    carry.held_sw_down_toa[_ofi],
                    tau_equator=_tau_equator, tau_pole=_tau_pole,
                    sbm_tau_c=_sbm_tau_c, sbm_RH_ref=_sbm_RH_ref,
                    sbm_CAPE_threshold=_sbm_CAPE_threshold,
                    C_H=_C_H, C_E=_C_E,
                    albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                    albedo_land=_albedo_land,
                    land_fraction=forcing.land_fraction,
                    l_mix_max=_l_mix_max,
                    ghg_vmr_override=_ghg_vmr_override,
                )

                # Write physics tendencies back at owned indices.
                # Non-owned faces keep dynamics-only values (no physics).
                _phys_dT = phys_out.dT_dt
                if hs_newtonian_relax is not None:
                    _phys_dT = _phys_dT + hs_newtonian_relax(
                        T_new[_ofi], p_s_new[_ofi], lat[_ofi])
                T_upd = T_new.at[_ofi].set(T_new[_ofi] + _dt * _phys_dT)
                q_v_upd = carry.q_v.at[_ofi].set(
                    jnp.maximum(carry.q_v[_ofi] + _dt * phys_out.dq_v_dt, 0.0)
                )
                q_c_upd = carry.q_c.at[_ofi].set(
                    jnp.maximum(carry.q_c[_ofi] + _dt * phys_out.dq_c_dt, 0.0)
                )
                q_r_upd = carry.q_r.at[_ofi].set(
                    jnp.maximum(carry.q_r[_ofi] + _dt * phys_out.dq_r_dt, 0.0)
                )
                conv_prog_upd = phys_out.conv_prog

                # Held radiation: update at owned indices
                held_new = (
                    carry.held_dT_rad.at[_ofi].set(held_new_local[0]),
                    carry.held_sw_net_sfc.at[_ofi].set(held_new_local[1]),
                    carry.held_lw_net_sfc.at[_ofi].set(held_new_local[2]),
                    carry.held_sw_up_toa.at[_ofi].set(held_new_local[3]),
                    carry.held_lw_up_toa.at[_ofi].set(held_new_local[4]),
                    carry.held_sw_down_toa.at[_ofi].set(held_new_local[5]),
                )

                # Precip: update at owned indices
                precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new[_ofi])
                precip_accum = carry.precip_accum.at[_ofi].add(precip_step * _dt)

                # Surface heat fluxes: accumulate at owned indices
                _sh = phys_out.shflx if phys_out.shflx is not None else jnp.zeros_like(p_s_new[_ofi])
                _lh = phys_out.lhflx if phys_out.lhflx is not None else jnp.zeros_like(p_s_new[_ofi])
                shflx_accum = carry.shflx_accum.at[_ofi].add(_sh * _dt)
                lhflx_accum = carry.lhflx_accum.at[_ofi].add(_lh * _dt)
            else:
                phys_out, held_new = step_unified(
                    need_rad,
                    T_new, p_s_new,
                    carry.q_v, carry.q_c, carry.q_r, carry.conv_prog,
                    u_new, v_new,
                    forcing.sst, forcing.sic, lat, lon,
                    forcing.day_of_year, forcing.seconds_of_day, _dt,
                    forcing.solar_weights, forcing.s_0,
                    forcing.o3_vmr, forcing.aerosol_od,
                    carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
                    carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
                    tau_equator=_tau_equator, tau_pole=_tau_pole,
                    sbm_tau_c=_sbm_tau_c, sbm_RH_ref=_sbm_RH_ref,
                    sbm_CAPE_threshold=_sbm_CAPE_threshold,
                    C_H=_C_H, C_E=_C_E,
                    albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                    albedo_land=_albedo_land,
                    land_fraction=forcing.land_fraction,
                    l_mix_max=_l_mix_max,
                    ghg_vmr_override=_ghg_vmr_override,
                )

                # --- State update ---
                _phys_dT_dt = phys_out.dT_dt
                if hs_newtonian_relax is not None:
                    _phys_dT_dt = _phys_dT_dt + hs_newtonian_relax(T_new, p_s_new, lat)
                T_upd = T_new + _dt * _phys_dT_dt
                q_v_upd = jnp.maximum(carry.q_v + _dt * phys_out.dq_v_dt, 0.0)
                q_c_upd = jnp.maximum(carry.q_c + _dt * phys_out.dq_c_dt, 0.0)
                q_r_upd = jnp.maximum(carry.q_r + _dt * phys_out.dq_r_dt, 0.0)
                conv_prog_upd = phys_out.conv_prog

                # --- Accumulate precipitation ---
                precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new)
                precip_accum = carry.precip_accum + precip_step * _dt

                # --- Accumulate surface heat fluxes ---
                _sh = phys_out.shflx if phys_out.shflx is not None else jnp.zeros_like(p_s_new)
                _lh = phys_out.lhflx if phys_out.lhflx is not None else jnp.zeros_like(p_s_new)
                shflx_accum = carry.shflx_accum + _sh * _dt
                lhflx_accum = carry.lhflx_accum + _lh * _dt

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
                    owned_mask=_owned_mask,
                )

            # --- Moisture smoothing ---
            q_v_upd = jnp.maximum(
                q_v_upd + _dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
                0.0,
            )

            # --- Rayleigh friction ---
            u_upd = u_new * _fric_decay
            v_upd = v_new * _fric_decay

            # CFL is computed at segment boundary (host-side) from the
            # final carry's wind fields, not inside the hot loop.
            max_cfl = carry.max_cfl

            # Cast all arrays back to carry input dtypes to prevent
            # float32→float64 promotion from Python float constants
            # (e.g., constants.L_v, constants.c_pd) breaking jax.lax.scan.
            new_carry = SegmentCarry(
                u=_match_dtype(u_upd, carry.u),
                v=_match_dtype(v_upd, carry.v),
                T=_match_dtype(T_upd, carry.T),
                p_s=_match_dtype(p_s_new, carry.p_s),
                phis=carry.phis,
                q_v=_match_dtype(q_v_upd, carry.q_v),
                q_c=_match_dtype(q_c_upd, carry.q_c),
                q_r=_match_dtype(q_r_upd, carry.q_r),
                conv_prog=_match_dtype(conv_prog_upd, carry.conv_prog),
                held_dT_rad=_match_dtype(held_new[0], carry.held_dT_rad),
                held_sw_net_sfc=_match_dtype(held_new[1], carry.held_sw_net_sfc),
                held_lw_net_sfc=_match_dtype(held_new[2], carry.held_lw_net_sfc),
                held_sw_up_toa=_match_dtype(held_new[3], carry.held_sw_up_toa),
                held_lw_up_toa=_match_dtype(held_new[4], carry.held_lw_up_toa),
                held_sw_down_toa=_match_dtype(held_new[5], carry.held_sw_down_toa),
                step_index=step_idx + 1,
                target_moisture=carry.target_moisture,
                target_mass=carry.target_mass,
                max_cfl=max_cfl,
                precip_accum=_match_dtype(precip_accum, carry.precip_accum),
                shflx_accum=_match_dtype(shflx_accum, carry.shflx_accum),
                lhflx_accum=_match_dtype(lhflx_accum, carry.lhflx_accum),
            )
            return new_carry, None
        return _single_step

    @partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
    def run_segment_jit(carry: SegmentCarry, n_steps: int,
                        forcing: SegmentForcing) -> SegmentCarry:
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
        forcing : SegmentForcing
            Per-segment external forcing (SST, SIC, solar, ozone, etc.).
            Traced as a dynamic argument — changing forcing values does
            **not** trigger recompilation.

        Returns
        -------
        SegmentCarry
            Updated state after n_steps.
        """
        _step_fn = _make_single_step(forcing)
        if gradient_checkpoint:
            _step_fn = jax.checkpoint(_step_fn, prevent_cse=False)
        final_carry, _ = jax.lax.scan(_step_fn, carry, None, length=n_steps)
        return final_carry

    def run_segment(carry: SegmentCarry, n_steps: int,
                    forcing: SegmentForcing) -> SegmentCarry:
        """Non-JIT version for use inside jax.grad / eqx.filter_value_and_grad.

        Same as run_segment_jit but without JIT wrapping or buffer
        donation, which conflict with outer AD transforms. The outer
        grad call handles compilation.
        """
        _step_fn = _make_single_step(forcing)
        if gradient_checkpoint:
            _step_fn = jax.checkpoint(_step_fn, prevent_cse=False)
        final_carry, _ = jax.lax.scan(_step_fn, carry, None, length=n_steps)
        return final_carry

    # Attach both variants; default is the JIT version for inference
    run_segment_jit.raw = run_segment
    return run_segment_jit


def _rebuild_state(carry: SegmentCarry, model):
    """Rebuild the model's expected state type from raw carry arrays.

    The dynamics model expects a NamedTuple with Field-wrapped arrays.
    Inside lax.scan we store raw arrays, so we reconstruct the state
    type here.  This is cheap — only Python object creation, no data copy.
    """
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
