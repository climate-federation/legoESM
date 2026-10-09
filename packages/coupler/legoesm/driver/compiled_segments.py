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
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.conservation import conservative_positive_clip
from legoesm.core.field import Field
# #1028: the D-grid -> cell-centre wind view used wherever the column physics
# reads winds while the run carries FV3 D-staggered cube winds.
from legoesm.core.operators_cdgrid import dgrid_to_center_vector
from legoesm.core.tracers import make_full_moisture_registry
from legoesm.thermo import saturation_mixing_ratio
from legoesm.forcing.time_utils import day_to_calendar

logger = logging.getLogger(__name__)


# Auto-enable jax.checkpoint (rematerialization) for segments longer than this
# many steps when ``gradient_checkpoint`` is left at its default ``None``.
_CKPT_AUTO_STEPS = 50


def _resolve_checkpoint(gradient_checkpoint: bool | None, n_steps: int) -> bool:
    """Resolve the gradient-checkpoint policy for a segment of ``n_steps``.

    An explicit ``True``/``False`` always wins; ``None`` (the default) auto-
    enables rematerialization for segments longer than ``_CKPT_AUTO_STEPS`` —
    the documented contract that was previously absent, so ``None`` silently
    disabled checkpointing and a long reverse-mode-AD segment could OOM exactly
    where the docstring promised protection. ``n_steps`` is the static scan
    length, so this is a compile-time decision.
    """
    if gradient_checkpoint is None:
        return n_steps > _CKPT_AUTO_STEPS
    return gradient_checkpoint


# Above this static scan length, checkpointing switches from PER-STEP remat to
# NESTED (Griewank / sqrt-N) checkpointing: per-step remat recomputes each step's
# INTERNAL activations but ``lax.scan`` reverse mode STILL stores the per-step
# CARRY at EVERY step, so reverse-mode memory is O(n_steps) x carry regardless.
# For a long differentiated rollout (e.g. the WeatherBench scale trainer, where a
# lat-lon pole-cell CFL clamp forces dt ~3-4 s and roll_steps ~6000) that O(N)
# trajectory OOMs (~1 TB at 0.7 deg, f64) — see #841.  Nested checkpointing keeps
# only ~sqrt(N) chunk-boundary carries, recomputing each chunk's forward in the
# backward pass: O(sqrt(N)) memory, EXACT gradient, ~2x backward compute.  Short
# production diagnostic segments stay on the per-step path (byte-identical graph).
_NESTED_CKPT_STEPS = 256


def _sqrt_checkpointed_scan(step_fn, carry, n_steps):
    """Run ``n_steps`` of ``step_fn`` as a scan with O(sqrt(n_steps)) reverse-mode
    memory via nested (Griewank) checkpointing.

    Split the scan into ``n_outer ~ sqrt(n_steps)`` OUTER chunks of ``inner ~
    sqrt(n_steps)`` steps.  The INNER chunk scan is ``jax.checkpoint``-wrapped, so
    reverse mode stores only the ``n_outer`` chunk-boundary carries and RECOMPUTES
    each chunk's forward before backpropagating through it.  Per-step remat INSIDE
    a chunk keeps that chunk's own reverse scan at O(inner) carries (its steps'
    internals are recomputed too).  Peak memory ~ ``(n_outer + inner) x carry`` =
    O(sqrt(n_steps)) instead of O(n_steps).  A trailing remainder (< inner) runs
    as a final per-step-remat scan.

    ``n_steps`` is the STATIC scan length (Python int; ``lax.scan`` requires it),
    so the chunking is decided at trace time.  The result is BIT-IDENTICAL to the
    plain scan on the forward and the EXACT gradient on the backward — only the
    reverse-mode memory/compute schedule changes.
    """
    inner = max(int(math.isqrt(n_steps)), 1)
    n_outer = n_steps // inner
    remainder = n_steps - n_outer * inner

    step_ckpt = jax.checkpoint(step_fn, prevent_cse=False)

    def _chunk(c, _):
        c, _ = jax.lax.scan(step_ckpt, c, None, length=inner)
        return c, None

    chunk_ckpt = jax.checkpoint(_chunk, prevent_cse=False)
    carry, _ = jax.lax.scan(chunk_ckpt, carry, None, length=n_outer)
    if remainder > 0:
        carry, _ = jax.lax.scan(step_ckpt, carry, None, length=remainder)
    return carry


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
        Prognostic convection control state.  Shape ``(ncol,)`` for the
        scalar-carrying schemes (mass_flux ``M_c`` / EDMF ``a_u``) and
        ``(ncol, nlev)`` (the full ``conv_prog_profile``) for the
        profile-prognostic schemes (zhang_mcfarlane / kain_fritsch /
        emanuel / tiedtke / bechtold).  The shape is fixed at packing
        time — ``lax.scan`` cannot reshape the carry mid-segment, so
        seed it for the configured scheme (see
        ``convection_scheme_traits().is_profile_prognostic``).
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc : jax.Array
        Held radiation tendencies for sub-cycling.
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa : jax.Array
        Held radiation fluxes for sub-cycling.
    held_sw_up_toa_clr, held_lw_up_toa_clr : jax.Array
        Held CLEAR-SKY TOA up-fluxes for sub-cycling (#843), the clear-sky
        counterpart of ``held_sw_up_toa`` / ``held_lw_up_toa``.  Produced by a
        SECOND radiation pass with clouds off (``cloud_scheme="none"``, no
        condensate/number tracers) so CMOR ``rsutcs`` / ``rlutcs`` can be
        written (SW_CRE = rsut - rsutcs; LW_CRE = rlutcs - rlut).  ALWAYS real
        arrays — zeros when ``PhysicsPipeline._clear_sky_diag`` is off (the
        default), so the code path is byte-identical to the no-clear-sky model
        and the diagnostic collector skips the field.  Refreshed only at the
        radiation cadence (same ``need_rad`` gating as the all-sky held fields)
        and held between refreshes.
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
    evap_accum : jax.Array
        Accumulated surface water flux [kg/m2] over the segment: the
        ``PhysicsOutput.evap_sfc`` the column actually received (tiled /
        prescribed water, else the bulk L_v(T_sfc) inverse) -- the CMOR
        ``evspsbl`` and the moisture-budget closure read THIS, never
        ``lhflx_accum / L_v``.  Read directly off the carry (not part of the
        ``unpack_carry`` tuple), like the radiation accumulators.
    sw_up_toa_accum, lw_up_toa_accum, sw_down_toa_accum : jax.Array
        Time-integrated TOA radiative fluxes [W/m2 * s] over the segment
        (sign conventions unchanged: ``*_up`` positive-up, ``*_down``
        positive-down).  Each step adds ``held_flux * dt`` — a piecewise-
        constant time integral at the radiation sub-cycle cadence — so
        ``accum / segment_duration`` is the true segment-mean flux.
        Motivation: CMOR Amon rsdt/rsut/rlut were previously written from
        the segment-END instantaneous held fluxes, i.e. fixed-UTC snapshots
        whose "monthly mean" carried a full day/night diurnal alias per
        pixel (January rsdt: night hemisphere exactly 0, noon peak
        ~1400 W/m2).  Reset to zero at every segment start, like
        ``precip_accum``.
    sw_up_toa_clr_accum, lw_up_toa_clr_accum : jax.Array
        Time-integrated CLEAR-SKY TOA up-fluxes [W/m2 * s] over the segment
        (#843), same construction as ``sw_up_toa_accum`` — each step adds
        ``held_*_toa_clr * dt`` so ``accum / segment_duration`` is the
        segment-mean clear-sky flux written to CMOR ``rsutcs`` / ``rlutcs``.
        Zeros (never accumulated) when ``PhysicsPipeline._clear_sky_diag`` is
        off (the default).  Reset to zero at every segment start.
    sw_net_sfc_accum, lw_net_sfc_accum : jax.Array
        Time-integrated net surface radiative fluxes [W/m2 * s] over the
        segment (same construction) so the energy-budget diagnostics see
        segment-mean fluxes consistent with the TOA set.
    t_low_accum : jax.Array
        Time-integrated lowest-level air temperature [K * s] over the
        segment; ``accum / segment_duration`` gives the segment-mean T_low
        used for the CMOR ``tas`` field (previously a fixed-UTC snapshot
        with a local-time bias of a few K over land).  Accumulated from the
        post-physics temperature of each step (before the optional
        saturation adjustment — an O(dt) lag, negligible over a segment).
    T_land : jax.Array or None
        Slab-land skin temperature [K].  ``None`` for ocean-only runs
        (the land tile is then inert).  Prognostic — advanced once per
        radiation sub-cycle by the slab surface energy balance.
    q_i, q_s, q_g, N_c, N_r, N_i : jax.Array or None
        Optional double-moment hydrometeors threaded so (a) the coupled /
        SFNO-training radiation gets droplet-number-aware effective radii and
        (b) a double-moment microphysics (Morrison / Thompson / P3 …) evolves
        its FULL state without silent truncation. Cloud ice / snow / graupel
        mixing ratio [kg/kg]; cloud-droplet N_c + rain N_r per-VOLUME [#/m³];
        ice N_i per-MASS [#/kg]. Radiation r_eff uses only q_i/N_c/N_i; the
        remaining fields complete the microphysics prognostic state. ``None``
        for warm-rain runs (kessler / diagnostic clouds) — then byte-identical
        to the legacy carry. Evolved each step from the matching
        ``PhysicsOutput.dq_*_dt`` / ``dN_*_dt`` like q_c/q_r.
    tke, qke, gwd_spectrum : jax.Array or None
        Stateful-physics carries (issue #413), flattened-column layout
        like ``conv_prog``: prognostic turbulent energy ``(ncol, nlev)``
        for the TKE-family (``tke``) / MYNN-2.5 (``qke``) turbulence
        schemes, and the wave-action spectrum
        ``(ncol, n_azimuths, n_wavenumbers)`` for the prognostic
        spectral GWD.  ``None`` when the corresponding scheme is
        diagnostic — then byte-identical to the legacy carry.  Replaced
        each step by the updated values riding ``PhysicsOutput`` (the
        kernels return REPLACEMENT values, e.g. the implicit TKE
        solve — these are not Euler-integrated tendencies).
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
    held_sw_up_toa_clr: jax.Array
    held_lw_up_toa_clr: jax.Array
    held_sw_down_toa: jax.Array
    step_index: jax.Array
    target_moisture: jax.Array
    target_mass: jax.Array
    max_cfl: jax.Array
    precip_accum: jax.Array
    shflx_accum: jax.Array
    lhflx_accum: jax.Array
    evap_accum: jax.Array
    sw_up_toa_accum: jax.Array
    lw_up_toa_accum: jax.Array
    sw_up_toa_clr_accum: jax.Array
    lw_up_toa_clr_accum: jax.Array
    sw_down_toa_accum: jax.Array
    sw_net_sfc_accum: jax.Array
    lw_net_sfc_accum: jax.Array
    t_low_accum: jax.Array
    T_land: jax.Array = None
    q_i: jax.Array = None
    q_s: jax.Array = None
    q_g: jax.Array = None
    N_c: jax.Array = None
    N_r: jax.Array = None
    N_i: jax.Array = None
    tke: jax.Array = None
    qke: jax.Array = None
    gwd_spectrum: jax.Array = None
    # Diagnostic CLUBB sub-grid cloud fraction (ncol, nlev) carried one step
    # (radiation runs before turbulence) so ``compute_radiation_core`` can use it
    # in the cloud optics instead of the RH grid-scale fraction — the compiled-
    # rollout twin of the per-step ``_run_per_step`` carry (marine-Sc albedo
    # lever).  ``None`` (default / feature off, or a non-cf-producing closure) =>
    # byte-identical legacy carry.  Threaded exactly like ``tke``: fed to
    # step_unified via ``_dm_in`` and read back from ``phys_out.cloud_fraction``.
    cloud_fraction: jax.Array = None
    conv_precip_prev: jax.Array = None
    # Lagged (previous-step) total precip [kg/m²/s] driving the opt-in
    # convective cloud-fraction source.  Radiation runs BEFORE convection in
    # the step, so this carries last step's precip to this step's cloud
    # diagnosis.  ``None`` (warm-rain / convective_cloud off) ⇒ byte-identical
    # legacy carry; pack_carry seeds a zeros array for production runs so the
    # feature can read it when ``PhysicsPipeline._cloud_convective`` is set.
    land_ml: object = None
    # Optional MULTILAYER (Richards) land state (a MultiLayerLandState pytree) when
    # the differentiable forward runs the multilayer coupler tile instead of the
    # embedded slab.  ``None`` (the default) ⇒ slab path, byte-identical legacy carry;
    # when present it is advanced in place of the scalar ``T_land`` and supplies the
    # land surface temperature (``T_soil[:, 0]``) to the surface blend.
    w_land: jax.Array = None
    # Prognostic slab-land soil water [kg/m²] (Manabe bucket).  ``None``
    # unless the soil-water bucket is active (``PhysicsPipeline.
    # land_soil_bucket``) ⇒ byte-identical legacy carry.  Advanced each
    # physics step in ``physics_step_no_rad`` (precip source, beta-limited
    # land evaporation sink); sets the land evaporation efficiency beta that
    # limits land latent heat.  Threaded exactly like ``T_land``.
    snow: jax.Array = None
    # Prognostic slab-land snow water equivalent [kg/m²].  ``None`` unless
    # snow-albedo feedback is active (``PhysicsPipeline.snow_albedo_feedback``)
    # ⇒ byte-identical legacy carry.  Advanced each physics step in
    # ``physics_step_no_rad`` (snowfall source, degree-day melt); brightens the
    # land albedo.  Threaded exactly like ``w_land``.
    budget_ledger_accum: jax.Array = None
    # Time-integrated per-process column budget ledger (N_LEDGER, 2)
    # [water kg/m² , dry enthalpy J/m²] — the process_ledger rows summed as
    # ``rate*dt`` each step, like ``precip_accum``.  ``None`` unless the
    # static ``budget_ledger`` diagnostic gate is on ⇒ byte-identical legacy
    # carry.  Read at segment boundaries (``accum / seg_duration`` = mean
    # rates) and reset to zeros at every segment start.


def pack_carry(state, q_v, q_c, q_r, conv_prog=None, *,
               held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
               held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
               step_index,
               held_sw_up_toa_clr=None, held_lw_up_toa_clr=None,
               target_moisture=None, target_mass=None,
               max_cfl=None, precip_accum=None,
               shflx_accum=None, lhflx_accum=None, evap_accum=None,
               sw_up_toa_accum=None, lw_up_toa_accum=None,
               sw_up_toa_clr_accum=None, lw_up_toa_clr_accum=None,
               sw_down_toa_accum=None, sw_net_sfc_accum=None,
               lw_net_sfc_accum=None, t_low_accum=None,
               T_land=None, q_i=None, q_s=None, q_g=None,
               N_c=None, N_r=None, N_i=None,
               tke=None, qke=None, gwd_spectrum=None,
               cloud_fraction=None,
               conv_precip_prev=None,
               land_ml=None, w_land=None, snow=None,
               conv_prog_nlev=None, budget_ledger_accum=None):
    """Pack driver state into a SegmentCarry for the compiled kernel.

    Prognostic fields are cast to at least the precision policy's storage
    dtype (upcasting only — never downcasts existing float64 arrays).
    Accumulation scalars use at least the accumulate dtype.

    ``conv_prog=None`` allocates a zero carry: ``(ncol,)`` by default
    (scalar-carrying mass_flux/EDMF and stateless schemes), or
    ``(ncol, conv_prog_nlev)`` when ``conv_prog_nlev`` is given — pass
    it (= nlev) for the profile-prognostic convection schemes
    (zhang_mcfarlane / kain_fritsch / emanuel / tiedtke / bechtold),
    whose ``conv_prog_profile`` carry must be seeded full-shape before
    entering ``lax.scan``.

    ``held_sw_up_toa_clr`` / ``held_lw_up_toa_clr`` / ``sw_up_toa_clr_accum``
    / ``lw_up_toa_clr_accum`` (#843) default to zeros (byte-identical to the
    no-clear-sky carry); pass real arrays only when ``clear_sky_diag`` is on.
    """
    from legoesm.core.precision import resolve_dtype
    storage = resolve_dtype(None, "storage")
    accum = resolve_dtype(None, "accumulate")

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
    if evap_accum is None:
        evap_accum = jnp.zeros_like(state.p_s.data)
    # Segment-mean flux / T_low accumulators (CMOR diurnal-alias fix):
    # always real arrays, reset to zero at every segment start.
    if sw_up_toa_accum is None:
        sw_up_toa_accum = jnp.zeros_like(state.p_s.data)
    if lw_up_toa_accum is None:
        lw_up_toa_accum = jnp.zeros_like(state.p_s.data)
    # Clear-sky TOA held fluxes + accumulators (#843): always real arrays
    # (zeros when clear_sky_diag is off) so the carry pytree stays uniform.
    if held_sw_up_toa_clr is None:
        held_sw_up_toa_clr = jnp.zeros_like(state.p_s.data)
    if held_lw_up_toa_clr is None:
        held_lw_up_toa_clr = jnp.zeros_like(state.p_s.data)
    if sw_up_toa_clr_accum is None:
        sw_up_toa_clr_accum = jnp.zeros_like(state.p_s.data)
    if lw_up_toa_clr_accum is None:
        lw_up_toa_clr_accum = jnp.zeros_like(state.p_s.data)
    if sw_down_toa_accum is None:
        sw_down_toa_accum = jnp.zeros_like(state.p_s.data)
    if sw_net_sfc_accum is None:
        sw_net_sfc_accum = jnp.zeros_like(state.p_s.data)
    if lw_net_sfc_accum is None:
        lw_net_sfc_accum = jnp.zeros_like(state.p_s.data)
    if t_low_accum is None:
        t_low_accum = jnp.zeros_like(state.p_s.data)
    # Lagged convective-cloud precip: always a real array (like precip_accum /
    # T_land) so the hot loop has no None branch; read only when the convective
    # cloud feature is enabled.  Zeros at t=0 ⇒ no convective cloud on step 0.
    if conv_precip_prev is None:
        conv_precip_prev = jnp.zeros_like(state.p_s.data)
    if conv_prog is None:
        if conv_prog_nlev is not None:
            conv_prog = jnp.zeros(
                (state.p_s.data.size, int(conv_prog_nlev)), dtype=storage,
            )
        else:
            conv_prog = jnp.zeros((state.p_s.data.size,), dtype=storage)
    # T_land is always a real array in the carry (never None) so the
    # SegmentCarry pytree has no Python-object leaves.  The land tile is
    # gated by PhysicsPipeline.f_land, not by T_land being None — for
    # ocean-only runs this zeros array is carried but never read.
    if T_land is None:
        T_land = jnp.zeros_like(state.p_s.data)
    # #1028: the cube hydrostatic lane may carry FV3 D-staggered winds
    # (u_d/v_d at corners) instead of cell-centre u/v.  The carry stores the
    # arrays verbatim either way -- it is the state TYPE, rebuilt by
    # _rebuild_state, that tells the dycore which staggering they are.
    _u_leaf = state.u_d if hasattr(state, "u_d") else state.u
    _v_leaf = state.v_d if hasattr(state, "v_d") else state.v
    return SegmentCarry(
        u=_promote(_u_leaf.data, storage),
        v=_promote(_v_leaf.data, storage),
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
        held_sw_up_toa_clr=_promote(held_sw_up_toa_clr, storage),
        held_lw_up_toa_clr=_promote(held_lw_up_toa_clr, storage),
        held_sw_down_toa=_promote(held_sw_down_toa, storage),
        step_index=jnp.int32(step_index),
        target_moisture=_promote(target_moisture, accum),
        target_mass=_promote(target_mass, accum),
        max_cfl=jnp.asarray(max_cfl),
        precip_accum=_promote(precip_accum, storage),
        shflx_accum=_promote(shflx_accum, storage),
        lhflx_accum=_promote(lhflx_accum, storage),
        evap_accum=_promote(evap_accum, storage),
        sw_up_toa_accum=_promote(sw_up_toa_accum, storage),
        lw_up_toa_accum=_promote(lw_up_toa_accum, storage),
        sw_up_toa_clr_accum=_promote(sw_up_toa_clr_accum, storage),
        lw_up_toa_clr_accum=_promote(lw_up_toa_clr_accum, storage),
        sw_down_toa_accum=_promote(sw_down_toa_accum, storage),
        sw_net_sfc_accum=_promote(sw_net_sfc_accum, storage),
        lw_net_sfc_accum=_promote(lw_net_sfc_accum, storage),
        t_low_accum=_promote(t_low_accum, storage),
        T_land=_promote(T_land, storage),
        # Double-moment tracers: kept None for warm-rain runs (identical legacy
        # carry); promoted to storage dtype only when the caller supplies them.
        q_i=None if q_i is None else _promote(q_i, storage),
        q_s=None if q_s is None else _promote(q_s, storage),
        q_g=None if q_g is None else _promote(q_g, storage),
        N_c=None if N_c is None else _promote(N_c, storage),
        N_r=None if N_r is None else _promote(N_r, storage),
        N_i=None if N_i is None else _promote(N_i, storage),
        # Stateful-physics carries (issue #413): kept None for diagnostic
        # schemes (identical legacy carry).
        tke=None if tke is None else _promote(tke, storage),
        qke=None if qke is None else _promote(qke, storage),
        gwd_spectrum=(None if gwd_spectrum is None
                      else _promote(gwd_spectrum, storage)),
        cloud_fraction=(None if cloud_fraction is None
                        else _promote(cloud_fraction, storage)),
        conv_precip_prev=_promote(conv_precip_prev, storage),
        land_ml=land_ml,   # pytree (MultiLayerLandState) or None — not a scalar field
        # Soil-water bucket: None unless the bucket is active (identical
        # legacy carry); the land tile reads it only when active.
        w_land=None if w_land is None else _promote(w_land, storage),
        # Snow water equiv.: None unless snow-albedo feedback is active
        # (identical legacy carry).
        snow=None if snow is None else _promote(snow, storage),
        # Budget-ledger accumulator: None unless the diagnostic gate is on
        # (identical legacy carry); the driver seeds zeros((N_LEDGER, 2)).
        budget_ledger_accum=(None if budget_ledger_accum is None
                             else _promote(budget_ledger_accum, accum)),
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

    Notes
    -----
    The segment-mean radiation / T_low accumulators (``sw_up_toa_accum``,
    ``lw_up_toa_accum``, ``sw_down_toa_accum``, ``sw_net_sfc_accum``,
    ``lw_net_sfc_accum``, ``t_low_accum``) are NOT part of this tuple —
    read them directly off the carry (like ``q_i`` / ``tke``) to keep the
    long-standing 10-tuple signature stable.
    """
    # #1028: mirror of pack_carry -- restore onto u_d/v_d when the template
    # is the D-staggered cube state, onto u/v otherwise.
    _wind_kw = (
        {"u_d": state_template.u_d.replace(data=carry.u),
         "v_d": state_template.v_d.replace(data=carry.v)}
        if hasattr(state_template, "u_d") else
        {"u": state_template.u.replace(data=carry.u),
         "v": state_template.v.replace(data=carry.v)}
    )
    new_state = state_template._replace(
        **_wind_kw,
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


def _segment_water_flux(phys_out, zeros):
    """The surface water flux to accumulate this step [kg/m2/s, positive up].

    ``PhysicsOutput.evap_sfc`` when the column received water; zeros when the
    step had NO surface exchange at all (``lhflx`` None too, the fluxless
    configs).  A step that reports latent heat but no water is refused at
    trace time: silently accumulating zeros would publish CMOR ``evspsbl`` = 0
    beside a nonzero ``hfls`` -- the hidden fallback this channel exists to
    remove (never ``lhflx / L_v`` here).
    """
    if phys_out.evap_sfc is not None:
        return phys_out.evap_sfc
    if phys_out.lhflx is None:
        return zeros
    raise ValueError(
        "compiled_segments: PhysicsOutput carries lhflx but no evap_sfc; the "
        "segment water accumulator (CMOR evspsbl, moisture closure) needs the "
        "water the column actually received -- the physics path must publish "
        "evap_sfc beside lhflx (never derive it as lhflx / L_v).")


def segment_accum_to_rate(accum, seg_steps: int, dt: float):
    """Convert a segment accumulator to a segment-MEAN rate.

    The ``SegmentCarry`` accumulators (``precip_accum`` [kg/m2],
    ``shflx_accum``/``lhflx_accum``/``*_toa_accum`` [W/m2 * s]) are built as
    ``accum += instantaneous_rate * dt`` over exactly *seg_steps* steps, from a
    zero reseed at every segment start (the ``precip_accum=jnp.zeros(...)``
    argument in the driver's per-segment ``pack_carry`` call,
    model_driver.py:8809).  Dividing by the segment duration therefore recovers
    the time-mean of the instantaneous rate, in the SAME units and with the
    SAME sign convention as the per-step quantity (precip positive-downward,
    i.e. INTO the surface, kg/m2/s).

    Budget: ``rate * (seg_steps * dt) == accum`` to round-off, so a consumer
    integrating this rate over the SAME segment duration re-integrates the
    quantity the atmosphere produced.  NOTE this closes the interface budget
    only when the consumer's integration window equals the segment; see the
    cadence caveat on ``ModelDriver._run_compiled`` (the coupler callback fires
    on the DIAG cadence, which is a multiple of the segment length whenever
    ``compute_segment_length`` returns a GCD smaller than ``diag_interval``).

    *seg_steps* and *dt* are static Python scalars (never traced), so the
    divisor is a weak-typed Python float: dtype-preserving, JIT-invisible and
    transparent to ``jax.grad``.  The guard below is therefore a Python-level
    check on static values and never runs under a trace.
    """
    duration = seg_steps * dt
    if not duration > 0:
        raise ValueError(
            f"segment duration must be positive, got seg_steps={seg_steps} "
            f"dt={dt} (duration={duration})"
        )
    return accum / duration


# ======================================================================
# Segment boundary computation
# ======================================================================

def compute_segment_length(
    diag_interval: int,
    checkpoint_interval: int,
    rad_update_steps: int = 0,
    fallback_interval: int = 0,
) -> int:
    """Compute the optimal segment length.

    The segment length is the GCD of all cadence intervals that require
    host-side actions (diagnostics, checkpoints).  This ensures every
    cadence boundary falls on a segment boundary.

    Issue #316: when ``rad_update_steps > 1`` and ``rad_update_steps``
    happens to divide the GCD already, the segment is a clean multiple
    of the radiation cadence and :func:`build_segment_fn` can subcycle
    radiation via an outer/inner ``lax.scan`` pair (one fresh
    radiation call per ``rad_update_steps`` inner physics steps).
    Eliminating the inner ``lax.cond`` cuts XLA compile time at long
    scan lengths from O(hours) to O(minutes).  When ``rad_update_steps``
    does NOT divide the GCD this function does **not** snap the
    segment down — doing so would break the invariant that
    ``segment_length`` divides every cadence interval (diag boundaries
    would drift between segments).  Instead the GCD is returned
    unchanged and :func:`build_segment_fn` falls back to the legacy
    cond-based scan at the (mild) cost of slower JIT.

    Parameters
    ----------
    diag_interval : int
        Steps between diagnostic collections.
    checkpoint_interval : int
        Steps between checkpoints (0 = disabled).
    rad_update_steps : int
        Radiation update cadence.  Reserved for future snap-down logic
        — currently the GCD is **not** modified to fit
        ``rad_update_steps``, because snapping down breaks the
        invariant that ``segment_length`` divides every cadence
        interval.  :func:`build_segment_fn` instead checks
        ``segment_length % rad_update_steps == 0`` at run time and
        falls back to the legacy cond-based scan when it does not
        hold.  Passed through for API stability and for callers that
        log/inspect the radiation cadence alongside the segment
        length.
    fallback_interval : int
        Segment length when NO host cadence exists (``diag_interval``
        and ``checkpoint_interval`` both <= 0).  The driver passes the
        ``forcing_update_days`` cadence in steps; this is also the
        forcing re-sampling cadence for such runs (forcing updates only
        at segment boundaries).  Ignored whenever a real cadence is
        present; <= 0 keeps the legacy 1-step segment.

    Returns
    -------
    int
        Segment length in time steps.  Always >= 1.
    """
    intervals = [i for i in [diag_interval, checkpoint_interval]
                 if i > 0]
    if not intervals:
        # No host cadence at all (diagnostics + checkpoints disabled —
        # e.g. distributed_mode='spmd' milestone-1 forces both off):
        # WITHOUT a fallback the segment collapses to 1 step and EVERY
        # step pays a host boundary (stability check, CFL fetch,
        # forcing re-pack).  Serial that is just slow; multi-controller
        # each boundary is a cross-process rendezvous — measured as the
        # production-SPMD anti-scaling (107 -> 206 ms/step np1->6, job
        # 8471423, segment HLO collective-clean per census 8471432).
        # Callers pass a sane chunk (the driver uses one day of steps).
        return max(int(fallback_interval), 1)
    seg = intervals[0]
    for i in intervals[1:]:
        seg = math.gcd(seg, i)
    seg = max(seg, 1)

    # Issue #316: when rad_update_steps divides the GCD evenly the
    # segment is already a clean multiple and build_segment_fn can
    # subcycle without remainder.  When it doesn't divide, snapping
    # down to ``(seg // rad_update_steps) * rad_update_steps`` would
    # break the invariant that segment_length divides every cadence
    # interval (e.g. seg=4320, rad=7 → 4319 ∤ 4320, so diag boundaries
    # would drift between segments).  Leave seg unchanged in that case
    # — build_segment_fn falls back to the legacy cond-based scan.
    if rad_update_steps > 1 and seg % rad_update_steps != 0:
        pass  # divisibility cannot be improved without breaking interval alignment
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
    """
    sst: jax.Array
    sic: jax.Array
    day_of_year: jax.Array
    seconds_of_day: jax.Array
    solar_weights: jax.Array
    s_0: jax.Array
    o3_vmr: jax.Array
    aerosol_od: jax.Array
    # Per-layer LONGWAVE aerosol absorption optical depth (volcanic
    # stratospheric, gap #9).  Mirrors ``aerosol_od`` (the SHORTWAVE
    # extinction).  Always a concrete array of the SAME shape as
    # ``aerosol_od`` (materialised to zeros by :func:`pack_forcing` when
    # the caller passes None) so the JIT pytree is stable and a run with
    # no volcanic LW aerosol is byte-identical (zeros LW od is a no-op in
    # the RRTMGP solver, same as None).
    aerosol_lw_od: jax.Array
    ghg_vmr: jax.Array  # shape (n_species,); empty (0,) when inactive
    # Coupler-provided dynamic surface overrides — the tile-blended surface
    # albedo / skin temperature fed back each segment by a coupled driver
    # (see PhysicsPipeline.compute_radiation_core).  ``None`` for AMIP /
    # standalone / uncoupled runs ⇒ the static internal albedo/T_sfc blend is
    # used (byte-identical to the pre-feedback behaviour).  Grid-shaped, like
    # sst/sic.  Kept None (not a (0,) placeholder) so no module-scope device op
    # is created at import and the pytree carries no spurious empty leaf.
    sfc_albedo_override: jax.Array | None = None
    sfc_T_override: jax.Array | None = None
    sfc_emissivity_override: jax.Array | None = None
    # Coupler-provided SHARED surface turbulent heat fluxes — the tile-blended
    # sensible / latent heat flux [W/m2, positive UP = surface→atmosphere] the
    # coupler computed for this segment (its bulk scheme, q_sfc = 0.98·q_sat
    # mixing ratio, ocean-tile C_H/C_E).  When present, the atmosphere's surface
    # tendency consumes THESE fluxes (the bottom-level T/q kick in
    # physics_step_no_rad) instead of recomputing its own bulk SH/LH, so the
    # heat + water leaving the atmosphere equals what the coupler feeds the
    # ocean — the air-sea budget closes (single authoritative flux calc on both
    # sides).  ``None`` for AMIP / standalone / uncoupled runs ⇒ the atmosphere
    # computes its own bulk fluxes (byte-identical to the pre-shared-flux
    # behaviour).  Grid-shaped, like sst/sic.  Kept None (not a (0,)
    # placeholder) so no module-scope device op is created at import and the
    # pytree carries no spurious empty leaf.
    sfc_shflx_override: jax.Array | None = None
    sfc_lhflx_override: jax.Array | None = None
    # Prescribed surface WATER flux [kg/m2/s, positive up]: the coupler's
    # tile-blended mass flux.  With it the atmosphere's moisture source is
    # the water the tiles actually lost, and sfc_lhflx_override stays the
    # PHYSICAL latent heat (each tile's own L(T, phase)) for the heat
    # consumers.  None (ERA5-prescribed heat only) -> lhflx / L_v(T_sfc).
    sfc_evap_override: jax.Array | None = None
    # Prescribed ERA5 / coupler surface MOMENTUM fluxes — the surface stress
    # [Pa, stress ON THE ATMOSPHERE, opposite in sign to the wind — the
    # convention of surface_layer.compute_surface_fluxes] for this segment.
    # When present the stress becomes the LOWER BOUNDARY CONDITION of
    # whatever turbulence scheme runs (folded into the kernel config via
    # fold_prescribed_surface_fluxes -> config.surface.prescribed_tau_*_pa),
    # or, on the bulk-BL path, an explicit lowest-layer momentum kick in
    # physics_step_no_rad.  ``None`` (default) keeps the scheme's own / bulk
    # drag — byte-identical.  Grid-shaped, like sst/sic; kept None (not a
    # (0,) placeholder) so no module-scope device op is created at import.
    sfc_taux_override: jax.Array | None = None
    sfc_tauy_override: jax.Array | None = None
    # Prescribed ERA5 / coupler surface RADIATIVE fluxes [W/m2] — upwelling
    # LW, upwelling SW, downwelling SW at the surface.  When present,
    # PhysicsPipeline.compute_radiation_core forms the radiative surface BC
    # from them: T_rad = (LW_up / sigma_sb)**0.25 with emissivity 1, and
    # albedo = SW_up / SW_down where SW_down >= 1 W/m2 (else the run's own
    # albedo after the coupler overrides).  The TURBULENT surface temperature
    # (sst/sic/T_land blend) is deliberately NOT replaced — the turbulent
    # fluxes are prescribed too when this is used (phase-2 doctrine: one
    # authoritative flux set).  ``None`` (default) keeps the internal
    # radiative surface — byte-identical.  Grid-shaped, like sst/sic.
    sfc_lw_up: jax.Array | None = None
    sfc_sw_up: jax.Array | None = None
    sfc_sw_down: jax.Array | None = None
    # Static land fraction [0..1], grid-shaped like sst/sic.  Consumed by the
    # learned physics wrappers (phase 2, part 2) that share the step_unified
    # signature; the classical pipeline accepts and ignores it.  ``None``
    # (default) — byte-identical.
    land_frac: jax.Array | None = None
    # Transient land-use cover — the per-segment multilayer land surface params
    # (``LandSurfaceParams`` pytree: per-column albedo_veg / emissivity / LAI /
    # canopy-structure) a coupled or AMIP driver re-materialises each segment as
    # the cover map advances in time (interp_annual on the legoesm_surfdata
    # pft_frac).  Passed as a TRACED arg (SegmentForcing doctrine) so the jitted
    # production step reads the evolving cover instead of the closure-baked
    # ``pipeline.land_ml_params`` captured at first trace.  ``None`` (default)
    # for static-cover / uncoupled runs ⇒ the step falls back to the baked
    # ``self.land_ml_params`` (byte-identical to the pre-transient behaviour).
    # A nested pytree, not a plain array; left out of GRID_SHAPED_FORCING_FIELDS
    # (transient cover is the lat-lon multilayer-land path, not SPMD cube) so it
    # is pinned replicated by the segment JIT.
    land_ml_params: object | None = None
    # Tropospheric visible-band column AOD (ncol,) for the AOD->CCN proxy
    # (volcanic excluded).  ``None`` => CCN uses sum(aerosol_od) (unchanged).
    aerosol_ccn_aod: jax.Array | None = None
    # Column-mean ozone VMR above the model top (ncol,) for the RRTMGP
    # overhead layer.  ``None`` => the solver falls back to the top-layer o3.
    o3_top_vmr: jax.Array | None = None


# Canonical GHG species ordering for the ghg_vmr array.  The three halogens
# are present only when the GHG file carries them (else RRTMGP's fixed means);
# a key missing here is silently dropped from every compiled lane.
GHG_SPECIES_ORDER = ("co2", "ch4", "n2o", "cfc11", "cfc12", "cfc22", "ccl4", "cf4")


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
    aerosol_lw_od=None,
    ghg_vmr=None,
    sfc_albedo_override=None,
    sfc_T_override=None,
    sfc_emissivity_override=None,
    sfc_shflx_override=None,
    sfc_lhflx_override=None,
    sfc_evap_override=None,
    sfc_taux_override=None,
    sfc_tauy_override=None,
    sfc_lw_up=None,
    sfc_sw_up=None,
    sfc_sw_down=None,
    land_frac=None,
    land_ml_params=None,
    aerosol_ccn_aod=None,
    o3_top_vmr=None,
) -> SegmentForcing:
    """Pack per-segment forcing into a SegmentForcing pytree.

    Parameters
    ----------
    ghg_vmr : dict, jax.Array, or None
        GHG volume mixing ratios.  Accepts a dict (auto-converted via
        :func:`ghg_dict_to_array`), a pre-packed array, or None.
    sfc_albedo_override, sfc_T_override, sfc_emissivity_override : jax.Array or None
        Coupler-provided tile-blended surface albedo / skin temperature /
        emissivity for this segment (grid-shaped, like sst/sic).  ``None``
        (default) leaves the radiation's static internal blend untouched —
        byte-identical for AMIP / standalone runs.  The emissivity override
        carries the canopy's LAI-dependent eps_eff so the atmospheric LW
        boundary uses the same emissivity the land tile formed its LW_out with.
    sfc_albedo_override, sfc_T_override : jax.Array or None
        Coupler-provided tile-blended surface albedo / skin temperature for
        this segment (grid-shaped, like sst/sic).  ``None`` (default) leaves
        the radiation's static internal blend untouched — byte-identical for
        AMIP / standalone runs.
    sfc_shflx_override, sfc_lhflx_override : jax.Array or None
        Coupler-provided tile-blended sensible / latent heat flux [W/m2,
        positive UP] for this segment (grid-shaped, like sst/sic).  ``None``
        (default) leaves the atmosphere computing its own bulk surface fluxes —
        byte-identical for AMIP / standalone runs.  When present the atmosphere
        surface tendency consumes these instead, closing the air-sea budget.
    sfc_taux_override, sfc_tauy_override : jax.Array or None
        Prescribed surface momentum flux [Pa, stress ON THE ATMOSPHERE,
        opposite in sign to the wind] for this segment (grid-shaped, like
        sst/sic).  ``None`` (default) keeps the scheme's own / bulk surface
        drag.  When present the stress is the lower boundary condition of
        whatever turbulence scheme runs (folded into the kernel config), or
        the explicit lowest-layer momentum kick on the bulk-BL path.
    sfc_lw_up, sfc_sw_up, sfc_sw_down : jax.Array or None
        Prescribed surface radiative fluxes [W/m2] for this segment
        (grid-shaped, like sst/sic): upwelling LW, upwelling SW, downwelling
        SW.  ``None`` (default) keeps the internal radiative surface —
        byte-identical.  When present compute_radiation_core forms the
        radiative BC T_rad = (LW_up / sigma_sb)**0.25 (emissivity 1) and
        albedo = SW_up / SW_down where SW_down >= 1 W/m2 (else the run's own
        albedo, after the coupler overrides).
    land_frac : jax.Array or None
        Static land fraction [0..1] (grid-shaped, like sst/sic).  Consumed by
        the learned physics wrappers (phase 2, part 2) that share the
        ``step_unified`` signature; the classical pipeline accepts and
        ignores it.  ``None`` (default).
    """
    if ghg_vmr is None:
        _ghg = jnp.zeros(0)
    elif isinstance(ghg_vmr, dict):
        _ghg = ghg_dict_to_array(ghg_vmr)
    else:
        _ghg = jnp.asarray(ghg_vmr)
    # LONGWAVE volcanic aerosol absorption optical depth (gap #9).  Default
    # to zeros of the SAME shape as ``aerosol_od`` when None so the
    # SegmentForcing pytree leaf is always a concrete fixed-shape array
    # (no JIT retrace) and a run without volcanic LW aerosol is
    # byte-identical (zeros LW od is a RRTMGP no-op).
    _aer_od = jnp.asarray(aerosol_od)
    if aerosol_lw_od is None:
        _aer_lw_od = jnp.zeros_like(_aer_od)
    else:
        _aer_lw_od = jnp.asarray(aerosol_lw_od)
    return SegmentForcing(
        sst=jnp.asarray(sst),
        sic=jnp.asarray(sic),
        day_of_year=jnp.asarray(day_of_year),
        seconds_of_day=jnp.asarray(seconds_of_day),
        solar_weights=jnp.asarray(solar_weights),
        s_0=jnp.asarray(s_0),
        o3_vmr=jnp.asarray(o3_vmr),
        aerosol_od=_aer_od,
        aerosol_lw_od=_aer_lw_od,
        ghg_vmr=_ghg,
        sfc_albedo_override=(
            None if sfc_albedo_override is None
            else jnp.asarray(sfc_albedo_override)
        ),
        sfc_T_override=(
            None if sfc_T_override is None else jnp.asarray(sfc_T_override)
        ),
        sfc_emissivity_override=(
            None if sfc_emissivity_override is None
            else jnp.asarray(sfc_emissivity_override)
        ),
        sfc_shflx_override=(
            None if sfc_shflx_override is None
            else jnp.asarray(sfc_shflx_override)
        ),
        sfc_lhflx_override=(
            None if sfc_lhflx_override is None
            else jnp.asarray(sfc_lhflx_override)
        ),
        sfc_evap_override=(
            None if sfc_evap_override is None
            else jnp.asarray(sfc_evap_override)
        ),
        sfc_taux_override=(
            None if sfc_taux_override is None
            else jnp.asarray(sfc_taux_override)
        ),
        sfc_tauy_override=(
            None if sfc_tauy_override is None
            else jnp.asarray(sfc_tauy_override)
        ),
        sfc_lw_up=(
            None if sfc_lw_up is None else jnp.asarray(sfc_lw_up)
        ),
        sfc_sw_up=(
            None if sfc_sw_up is None else jnp.asarray(sfc_sw_up)
        ),
        sfc_sw_down=(
            None if sfc_sw_down is None else jnp.asarray(sfc_sw_down)
        ),
        land_frac=(
            None if land_frac is None else jnp.asarray(land_frac)
        ),
        # LandSurfaceParams pytree (or None) — passed through as-is; its leaves
        # are already jax arrays from the provider rebuild, no jnp.asarray coerce.
        land_ml_params=land_ml_params,
        aerosol_ccn_aod=(
            None if aerosol_ccn_aod is None else jnp.asarray(aerosol_ccn_aod)
        ),
        o3_top_vmr=(
            None if o3_top_vmr is None else jnp.asarray(o3_top_vmr)
        ),
    )


# The SegmentForcing fields that are GRID-shaped (face-plane on cubed-sphere,
# (n_lat, n_lon, ...) on lat-lon) and therefore shard like the state.  By
# NAME, not by shape: a shape heuristic would silently face-shard any
# incidental leading-6 leaf (e.g. ``ghg_vmr`` the day a 6th species joins
# GHG_SPECIES_ORDER).  Scalars (day_of_year, seconds_of_day, s_0) and small
# vectors (ghg_vmr) stay uncommitted -> the segment JIT pins them replicated.
GRID_SHAPED_FORCING_FIELDS = (
    "sst", "sic", "solar_weights", "o3_vmr", "aerosol_od", "aerosol_lw_od",
    "sfc_albedo_override", "sfc_T_override", "sfc_emissivity_override",
    "sfc_shflx_override", "sfc_lhflx_override", "sfc_evap_override",
    "sfc_taux_override", "sfc_tauy_override",
    "sfc_lw_up", "sfc_sw_up", "sfc_sw_down", "land_frac", "aerosol_ccn_aod",
    "o3_top_vmr",
)


def shard_forcing(forcing: SegmentForcing, device_config) -> SegmentForcing:
    """Commit the grid-shaped ``SegmentForcing`` leaves to the SAME device
    layout as the sharded state (SPMD sharding step 3 of the production
    cs_spmd design).

    ``run_segment``'s sharded JIT wrapper honours a committed
    ``NamedSharding`` on the build-time mesh per leaf and pins anything
    else REPLICATED (``_input_sharding``).  Replicated forcing means every
    device receives the FULL global forcing array at each segment
    boundary (host->device bandwidth and memory x n_devices) and GSPMD
    re-slices it against the face-sharded state inside the step.
    Committing the grid-shaped leaves with the state's own sharding rules
    (:func:`legoesm.parallel.mesh.shard_pytree` — face-first on
    cubed-sphere, lat-axis on lat-lon, tiling-aware) makes the transfer
    and memory per device ``1/n_devices`` and removes the reshard.

    No-ops (returns ``forcing`` unchanged) when: ``device_config`` is
    ``None`` (byte-identical single-device contract), the mesh has no
    face sharding (single device), or the run is mpi4jax-distributed
    (``is_distributed`` — forcing there is rank-local, never device-mesh
    sharded).  ``None`` optional fields and non-grid leaves pass through
    untouched; a grid-NAMED leaf whose shape does not match the grid rule
    (e.g. the ``(0,)`` o3 placeholder) is left alone by ``shard_pytree``'s
    shape gate.
    """
    if device_config is None or getattr(device_config, "is_distributed", False):
        return forcing
    if getattr(device_config, "face_sharding", None) is None:
        return forcing
    from legoesm.parallel.mesh import shard_pytree

    # Production cubed-sphere packs the 3-D radiation forcing FLATTENED to
    # ``(ncol=6*n*n, nlev)`` (``_precompute_external_forcing``), which the
    # leading-6 ``shard_pytree`` rule cannot see (codex 2026-07-01 Medium).
    # A flat face-major leaf shards ``P("face")`` on dim 0 — the exact
    # layout the driver already commits for flat carry leaves (see
    # ``_input_sharding``: "a flat [6*n*n, nlev] carry leaf the driver
    # placed on P('face')").  ``ncol % 6 == 0`` chunks land exactly one
    # face's columns per face-mesh slot; the face axis size divides 6 so
    # divisibility always holds.  Gate: cubed-sphere only, named grid
    # field, flat (shape[0] a positive non-6 multiple of 6).
    _flat_face = None
    if getattr(device_config, "grid_type", None) == "cubed_sphere" and \
            getattr(device_config, "mesh", None) is not None:
        from jax.sharding import NamedSharding, PartitionSpec
        _flat_face = NamedSharding(device_config.mesh, PartitionSpec("face"))

    updates = {}
    for name in GRID_SHAPED_FORCING_FIELDS:
        leaf = getattr(forcing, name)
        if leaf is None or not isinstance(leaf, (jax.Array, jnp.ndarray)):
            continue
        if (_flat_face is not None and leaf.ndim >= 1
                and leaf.shape[0] != 6 and leaf.shape[0] > 0
                and leaf.shape[0] % 6 == 0):
            # multiprocess_safe_device_put, NOT jax.device_put: under
            # multi-controller SPMD the cross-process bit-equality assert
            # in device_put trips on host-precomputed forcing (#693,
            # Levante 2-node receipt job 26030677).
            from legoesm.parallel.mesh import multiprocess_safe_device_put
            updates[name] = multiprocess_safe_device_put(leaf, _flat_face)
        else:
            updates[name] = shard_pytree(leaf, device_config)
    return forcing._replace(**updates) if updates else forcing


@dataclass(frozen=True)
class _SplitStepStatics:
    """Closure statics for the operator-split physics body, bundled so the
    single-rank physics + the shared finalizer live at MODULE SCOPE and can be
    reused by the lat-band-SPMD lane (which supplies its own sharded dynamics
    + mass fixer, then calls the SAME post-dynamics physics + tail — no
    duplication of the ~200-line write-back).  A plain frozen dataclass (NOT a
    NamedTuple) so it is never treated as a jax pytree: it is a closure constant
    captured by the scan body, never a scan carry/xs.  ``step_unified``,
    ``forcing`` and ``ghg_vmr_override`` depend on the per-segment forcing, so
    the bundle is built inside ``_make_single_step``."""
    step_unified: object
    forcing: object
    lat: object
    lon: object
    dt: object
    tau_equator: object
    tau_pole: object
    sbm_tau_c: object
    sbm_RH_ref: object
    C_H: object
    C_E: object
    albedo_ice: object
    albedo_ocean: object
    ghg_vmr_override: object
    hs_newtonian_relax: object
    energy_consistent_moisture_clip: bool
    do_sat_adjust: bool
    fix_moisture: bool
    sigma_full: object
    dsigma: object
    grid: object
    owned_mask: object
    qv_smooth_coeff: object
    fric_decay: object
    hyperdiffusion_3d: object
    # Clear-sky diagnostic (#843): the PhysicsPipeline (for the 2nd,
    # clouds-off compute_radiation_core pass) and the freshness mode for THIS
    # step variant — "hold" (no clear-sky compute: feature off OR the no-rad
    # subcycle variant), "cond" (compute fresh gated on the traced need_rad,
    # for the legacy data-dependent step_unified), or "always" (compute fresh
    # every call, for the static rad-every-call step_unified).  Defaults keep
    # the lat-band-SPMD lane (which does not pass them) byte-identical.
    pipeline: object = None
    clear_sky_fresh: str = "hold"
    # Per-process budget ledger (diagnostics.process_ledger): static Python
    # bool — when True the split body fills the dynamics/clips rows and
    # accumulates phys_out.budget_ledger into the carry.  Default False keeps
    # every lane byte-identical (feature-gating exception: Python ``if``).
    budget_ledger: bool = False


def build_operator_split_statics(
    *, step_unified, forcing, lat, lon, dt,
    tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref, C_H, C_E,
    albedo_ice, albedo_ocean, ghg_vmr_override,
    hs_newtonian_relax, energy_consistent_moisture_clip,
    do_sat_adjust, fix_moisture, sigma_full, dsigma, grid,
    owned_mask, qv_smooth_coeff, fric_decay, hyperdiffusion_3d,
    pipeline=None, clear_sky_fresh="hold", budget_ledger=False,
):
    """Bundle the operator-split closure statics (the frozen ``_SplitStepStatics``).

    PUBLIC factory so the lat-band-SPMD driver lane (``model_driver.
    _run_operator_split_spmd``) builds the SAME statics the serial
    ``build_segment_fn`` scan bundles, WITHOUT importing the private dataclass
    (CLAUDE.md: no private cross-module import).  The scalar closure constants
    (``dt``, ``tau_*``, ``sbm_*``, ``C_*``, ``albedo_*``, ``fric_decay``) are
    wrapped with ``jnp.asarray`` here — identical to the former inline literal —
    so both call sites get byte-identical leaves; ``None`` passes through
    untouched.  ``owned_mask`` / ``sigma_full`` / ``dsigma`` / ``grid`` /
    ``ghg_vmr_override`` / callables are already array-or-None and pass through.
    """
    _a = lambda x: (jnp.asarray(x) if x is not None else None)
    return _SplitStepStatics(
        step_unified=step_unified, forcing=forcing,
        lat=lat, lon=lon, dt=_a(dt),
        tau_equator=_a(tau_equator), tau_pole=_a(tau_pole),
        sbm_tau_c=_a(sbm_tau_c), sbm_RH_ref=_a(sbm_RH_ref),
        C_H=_a(C_H), C_E=_a(C_E),
        albedo_ice=_a(albedo_ice), albedo_ocean=_a(albedo_ocean),
        ghg_vmr_override=ghg_vmr_override,
        hs_newtonian_relax=hs_newtonian_relax,
        energy_consistent_moisture_clip=energy_consistent_moisture_clip,
        do_sat_adjust=do_sat_adjust, fix_moisture=fix_moisture,
        sigma_full=sigma_full, dsigma=dsigma, grid=grid,
        owned_mask=owned_mask, qv_smooth_coeff=qv_smooth_coeff,
        fric_decay=_a(fric_decay), hyperdiffusion_3d=hyperdiffusion_3d,
        pipeline=pipeline, clear_sky_fresh=clear_sky_fresh,
        budget_ledger=budget_ledger,
    )


class _SplitStepLocals(NamedTuple):
    """The exact set of per-step locals BOTH the single-rank and the
    MPI-owned-face branches produce and the shared finalizer consumes: the
    Euler-updated fields, the double-moment tracer updates, the accumulators,
    the land-tile updates, and the prognostic-physics carry values riding
    ``PhysicsOutput`` (``phys_*``).  ``u_new/v_new/p_s_new`` are the
    (post-mass-fix) dynamics fields the tail's Rayleigh friction + carry pack
    read."""
    u_new: object
    v_new: object
    p_s_new: object
    T_upd: object
    q_v_upd: object
    q_c_upd: object
    q_r_upd: object
    q_i_upd: object
    q_s_upd: object
    q_g_upd: object
    N_c_upd: object
    N_r_upd: object
    N_i_upd: object
    conv_prog_upd: object
    held_new: object
    precip_accum: object
    conv_precip_prev_new: object
    shflx_accum: object
    lhflx_accum: object
    evap_accum: object
    sw_net_sfc_accum: object
    lw_net_sfc_accum: object
    sw_up_toa_accum: object
    lw_up_toa_accum: object
    sw_up_toa_clr_accum: object
    lw_up_toa_clr_accum: object
    sw_down_toa_accum: object
    t_low_accum: object
    T_land_new: object
    land_ml_new: object
    w_land_new: object
    snow_new: object
    phys_tke: object
    phys_qke: object
    phys_gwd: object
    phys_cloud_fraction: object = None
    # Per-step (N_LEDGER, 2) budget ledger with the physics + dynamics +
    # state-update-clips rows filled; the finalizer adds the tail's clip
    # delta and accumulates ``*dt`` into the carry.  ``None`` when the
    # static gate is off.
    budget_ledger_step: object = None


_MOIST_FIELDS = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i")


def _clear_sky_toa_pass(clear_sky_fresh, need_rad,
                        held_sw_up_toa_clr, held_lw_up_toa_clr,
                        compute_fresh):
    """Return this step's ``(sw_up_toa_clr, lw_up_toa_clr)`` for the #843
    clear-sky diagnostic, mirroring how the all-sky held fields are refreshed.

    ``clear_sky_fresh`` (a static Python str selected in ``_make_single_step``):
      * ``"hold"``   — clear-sky diag off OR the no-rad subcycle variant: hold
        the carried clear-sky values (zeros when off) and run NO extra RRTMGP,
        so ``body_no_rad`` HLOs stay rrtmgp-free.  Byte-identical to the
        no-clear-sky model when off.
      * ``"cond"``   — legacy data-dependent ``step_unified``: recompute fresh
        clear-sky only when the traced ``need_rad`` fires, else hold — exactly
        the all-sky ``lax.cond(need_rad, ...)`` inside ``step_unified``.
      * ``"always"`` — static rad-every-call ``step_unified``: recompute fresh
        every step (all-sky is fresh every step too).

    ``compute_fresh`` is a zero-arg callable running the clouds-off second
    ``compute_radiation_core`` and returning the two TOA up-fluxes already cast
    to the held dtype.  ``held_*`` are the carried clear-sky held values used on
    the held (non-refresh) steps.
    """
    if clear_sky_fresh == "always":
        return compute_fresh()
    if clear_sky_fresh == "cond":
        return jax.lax.cond(
            need_rad,
            compute_fresh,
            lambda: (held_sw_up_toa_clr, held_lw_up_toa_clr),
        )
    # "hold" (and any unrecognised mode): no extra radiation pass.
    return held_sw_up_toa_clr, held_lw_up_toa_clr


def split_physics_single_rank(carry, T_new, u_new, v_new, p_s_new,
                               need_rad, doy_step, sod_step, statics,
                               moist=None, u_cc=None, v_cc=None):
    """The single-rank (``owned_face_ids is None``) operator-split physics body:
    ``step_unified`` on the whole column set, then the Euler write-back +
    accumulators.  Extraction of the former ``_single_step`` ``else`` body
    (dynamics + mass fixer done by the caller in the prologue) — reads closure
    statics off ``statics`` and returns the ``_SplitStepLocals`` bundle the
    shared finalizer consumes.

    ``moist`` (issue #771): the moisture fields the physics acts on — a dict
    over ``_MOIST_FIELDS`` (q_v/q_c/q_r + the double-moment fields).  ``None``
    (the lat-band-SPMD lane + the column-locked serial path) reads them straight
    off the carry, BYTE-IDENTICAL; the ``advect_moisture`` serial path passes the
    POST-ADVECTION aliases so the physics update acts on the transported
    moisture.  ``tke/qke/gwd_spectrum`` are physics carries (not advected) and
    always ride the carry."""
    from legoesm.core.conservation import energy_consistent_moisture_floor
    if moist is None:
        moist = {_nm: getattr(carry, _nm) for _nm in _MOIST_FIELDS}
    # #1028: physics is a COLUMN closure and reads winds at cell centres.  On
    # the persistent-D cube lane the carried winds sit at D-grid corners, so
    # the caller passes a cell-centre VIEW here, while ``u_new``/``v_new`` stay
    # the native carried arrays the finalizer writes back into the carry.  Off
    # that lane the two are the same object and everything below is
    # byte-identical.
    if u_cc is None:
        u_cc, v_cc = u_new, v_new

    # --- Budget ledger: DYNAMICS row (store delta across the dycore step,
    # incl. the dry-mass fixer's p_s adjustment and — with advect_moisture —
    # the flux-form tracer transport; column-locked tracers register only
    # the p_s-driven column-mass change).  Positive = dynamics added
    # water/dry enthalpy to the (global-mean) column store.
    if statics.budget_ledger:
        from legoesm.diagnostics.process_ledger import column_store_snapshot
        _led_q_names = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")
        _led_before = column_store_snapshot(
            carry.p_s, statics.dsigma, carry.T,
            *(getattr(carry, _n) for _n in _led_q_names),
            area=statics.grid.grid_area)
        _led_after = column_store_snapshot(
            p_s_new, statics.dsigma, T_new,
            *(moist[_n] for _n in _led_q_names),
            area=statics.grid.grid_area)
        _led_dynamics = (_led_after - _led_before) / statics.dt
    _dm_in = {}
    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        if moist[_nm] is not None:
            _dm_in[_nm] = moist[_nm]
    for _nm in ("tke", "qke", "gwd_spectrum", "cloud_fraction"):
        _fld = getattr(carry, _nm)
        if _fld is not None:
            _dm_in[_nm] = _fld
    _ret = statics.step_unified(
        need_rad,
        T_new, p_s_new,
        moist["q_v"], moist["q_c"], moist["q_r"], carry.conv_prog,
        u_cc, v_cc,
        statics.forcing.sst, statics.forcing.sic, statics.lat, statics.lon,
        doy_step, sod_step, statics.dt,
        statics.forcing.solar_weights, statics.forcing.s_0,
        statics.forcing.o3_vmr, statics.forcing.aerosol_od,
        carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
        carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
        tau_equator=statics.tau_equator, tau_pole=statics.tau_pole,
        sbm_tau_c=statics.sbm_tau_c, sbm_RH_ref=statics.sbm_RH_ref,
        C_H=statics.C_H, C_E=statics.C_E,
        albedo_ice=statics.albedo_ice, albedo_ocean=statics.albedo_ocean,
        ghg_vmr_override=statics.ghg_vmr_override,
        aerosol_lw_od=statics.forcing.aerosol_lw_od,
        aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
        o3_top_vmr=statics.forcing.o3_top_vmr,
        sfc_albedo_override=statics.forcing.sfc_albedo_override,
        sfc_T_override=statics.forcing.sfc_T_override,
        sfc_emissivity_override=statics.forcing.sfc_emissivity_override,
        sfc_shflx_override=statics.forcing.sfc_shflx_override,
        sfc_lhflx_override=statics.forcing.sfc_lhflx_override,
        sfc_evap_override=statics.forcing.sfc_evap_override,
        sfc_taux_override=statics.forcing.sfc_taux_override,
        sfc_tauy_override=statics.forcing.sfc_tauy_override,
        sfc_lw_up=statics.forcing.sfc_lw_up,
        sfc_sw_up=statics.forcing.sfc_sw_up,
        sfc_sw_down=statics.forcing.sfc_sw_down,
        land_frac=statics.forcing.land_frac,
        phis=carry.phis,
        T_land=carry.T_land, land_ml=carry.land_ml,
        land_ml_params=statics.forcing.land_ml_params,
        conv_precip=carry.conv_precip_prev,
        w_land=carry.w_land, snow=carry.snow,
        **_dm_in,
    )
    phys_out, held_new = _ret[0], _ret[1]
    # ``step_unified`` returns a 3rd value (the slab-land skin temperature,
    # #325) for the PhysicsPipeline build; legacy 2-tuple wrappers (neural /
    # SFNO training) leave the land tile inert by carrying ``T_land`` through.
    T_land_new = _ret[2] if len(_ret) > 2 else carry.T_land
    # optional 4th value: the advanced MULTILAYER land state (when active).
    land_ml_new = _ret[3] if len(_ret) > 3 else carry.land_ml
    w_land_new = (phys_out.w_land
                  if phys_out.w_land is not None
                  else carry.w_land)
    snow_new = (phys_out.snow
                if phys_out.snow is not None
                else carry.snow)

    # --- Clear-sky TOA second radiation pass (#843) ---
    # Mirror the all-sky held refresh: feed the SAME post-dynamics state
    # step_unified's all-sky radiation saw (T_new / p_s_new / moist q_v,
    # u_new / v_new), but with clouds OFF (cloud_scheme="none", no
    # condensate/number tracers).  Held between radiation refreshes exactly
    # like held_sw_up_toa.  "hold" mode (feature off, or the no-rad subcycle
    # variant) runs NO extra RRTMGP, so it stays byte-identical + keeps
    # body_no_rad HLOs rrtmgp-free.
    def _compute_fresh_clr():
        (_c0, _c1, _c2, _sw_clr, _lw_clr, _c5, _c6, _c7) = \
            statics.pipeline.compute_radiation_core(
                T_new, p_s_new, moist["q_v"],
                statics.forcing.sst, statics.forcing.sic,
                statics.lat, statics.lon, doy_step, sod_step,
                statics.forcing.solar_weights, statics.forcing.s_0,
                statics.forcing.o3_vmr, statics.forcing.aerosol_od,
                aerosol_lw_od_precomputed=statics.forcing.aerosol_lw_od,
                aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
                o3_top_vmr=statics.forcing.o3_top_vmr,
                tau_equator=statics.tau_equator, tau_pole=statics.tau_pole,
                albedo_ice=statics.albedo_ice,
                albedo_ocean=statics.albedo_ocean,
                ghg_vmr_override=statics.ghg_vmr_override,
                q_c=None, q_i=None, N_c=None, N_i=None,
                cloud_scheme="none",
                u=u_cc, v=v_cc, dt=statics.dt,
                T_land=carry.T_land, land_ml=carry.land_ml,
                land_ml_params=statics.forcing.land_ml_params,
                sfc_albedo_override=statics.forcing.sfc_albedo_override,
                sfc_T_override=statics.forcing.sfc_T_override,
                sfc_emissivity_override=statics.forcing.sfc_emissivity_override,
                sfc_lw_up=statics.forcing.sfc_lw_up,
                sfc_sw_up=statics.forcing.sfc_sw_up,
                sfc_sw_down=statics.forcing.sfc_sw_down,
                conv_precip=carry.conv_precip_prev,
                w_land=carry.w_land, snow=carry.snow,
            )
        _dt_h = carry.held_sw_up_toa_clr.dtype
        return _sw_clr.astype(_dt_h), _lw_clr.astype(_dt_h)

    _sw_up_toa_clr, _lw_up_toa_clr = _clear_sky_toa_pass(
        statics.clear_sky_fresh, need_rad,
        carry.held_sw_up_toa_clr, carry.held_lw_up_toa_clr,
        _compute_fresh_clr,
    )
    # Extend held_new to the 8-tuple carried by _SplitStepLocals (positions
    # 6/7 = clear-sky TOA up-fluxes); finalize_split_step writes them into the
    # held_*_toa_clr carry fields.
    held_new = tuple(held_new) + (_sw_up_toa_clr, _lw_up_toa_clr)

    # --- State update ---
    _phys_dT_dt = phys_out.dT_dt
    if statics.hs_newtonian_relax is not None:
        _phys_dT_dt = _phys_dT_dt + statics.hs_newtonian_relax(
            T_new, p_s_new, statics.lat)
    T_upd = T_new + statics.dt * _phys_dT_dt
    _qv_raw = moist["q_v"] + statics.dt * phys_out.dq_v_dt
    if statics.energy_consistent_moisture_clip:
        # Issue #323: keep the q_v floor moist-static-energy neutral by
        # removing the latent heat of the clipped (un-removed) vapour sink.
        # KNOWN EXCEPTION to "conserving form always": this opt-in branch
        # conserves MSE but CREATES the clipped vapour; a jointly water- and
        # energy-conserving floor is a separate limiter (codex 2026-08-16).
        q_v_upd, T_upd = energy_consistent_moisture_floor(_qv_raw, T_upd)
    else:
        # Conserving form always (owner decision 2026-08-16): column borrow,
        # never the mass-creating plain max(q, 0). Pure-sigma lane: dsigma is
        # the layer-mass weight up to the per-column p_s factor, which
        # cancels in the rescale.
        q_v_upd = conservative_positive_clip(
            _qv_raw, statics.dsigma, axis=-1)[0]
    q_c_upd = conservative_positive_clip(
        moist["q_c"] + statics.dt * phys_out.dq_c_dt, statics.dsigma,
        axis=-1)[0]
    q_r_upd = conservative_positive_clip(
        moist["q_r"] + statics.dt * phys_out.dq_r_dt, statics.dsigma,
        axis=-1)[0]
    def _dm_upd(fld, tend):
        return (None if fld is None
                else conservative_positive_clip(
                    fld + statics.dt * tend, statics.dsigma, axis=-1)[0])
    q_i_upd = _dm_upd(moist["q_i"], phys_out.dq_i_dt)
    q_s_upd = _dm_upd(moist["q_s"], phys_out.dq_s_dt)
    q_g_upd = _dm_upd(moist["q_g"], phys_out.dq_g_dt)
    N_c_upd = _dm_upd(moist["N_c"], phys_out.dN_c_dt)
    N_r_upd = _dm_upd(moist["N_r"], phys_out.dN_r_dt)
    N_i_upd = _dm_upd(moist["N_i"], phys_out.dN_i_dt)
    conv_prog_upd = phys_out.conv_prog

    # --- Budget ledger: CLIPS row (state-update part) + assembly ----------
    # Clips = post-floor stores minus the raw Euler-update stores (positive
    # water = the q >= 0 floors CREATED water; the energy delta is nonzero
    # only on the energy_consistent_moisture_clip path, which pairs the
    # floored vapour with latent heat).  The tail's clip delta (sat adjust /
    # moisture fixer / q_v-smoothing floor) is added in finalize_split_step.
    if statics.budget_ledger:
        from legoesm.diagnostics.process_ledger import (
            ROW_CLIPS, ROW_DYNAMICS, column_store_snapshot,
        )
        if phys_out.budget_ledger is None:
            raise ValueError(
                "budget_ledger statics gate is on but PhysicsOutput."
                "budget_ledger is None — the PhysicsPipeline was built "
                "without config.output.budget_ledger; both gates must come "
                "from the same OutputConfig.")
        _led_T_raw = T_new + statics.dt * _phys_dT_dt
        _led_raw = column_store_snapshot(
            p_s_new, statics.dsigma, _led_T_raw,
            _qv_raw,
            moist["q_c"] + statics.dt * phys_out.dq_c_dt,
            moist["q_r"] + statics.dt * phys_out.dq_r_dt,
            None if moist["q_i"] is None
            else moist["q_i"] + statics.dt * phys_out.dq_i_dt,
            None if moist["q_s"] is None
            else moist["q_s"] + statics.dt * phys_out.dq_s_dt,
            None if moist["q_g"] is None
            else moist["q_g"] + statics.dt * phys_out.dq_g_dt,
            area=statics.grid.grid_area)
        _led_clipped = column_store_snapshot(
            p_s_new, statics.dsigma, T_upd,
            q_v_upd, q_c_upd, q_r_upd, q_i_upd, q_s_upd, q_g_upd,
            area=statics.grid.grid_area)
        _led_clips = (_led_clipped - _led_raw) / statics.dt
        _led_step = phys_out.budget_ledger.astype(_led_clips.dtype)
        _led_step = _led_step.at[ROW_DYNAMICS].set(_led_dynamics)
        _led_step = _led_step.at[ROW_CLIPS].set(_led_clips)
    else:
        _led_step = None

    # --- Accumulate precipitation ---
    precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new)
    precip_accum = carry.precip_accum + precip_step * statics.dt
    # Lag this step's total precip for next step's convective cloud.
    conv_precip_prev_new = precip_step

    # --- Accumulate surface heat fluxes ---
    _sh = phys_out.shflx if phys_out.shflx is not None else jnp.zeros_like(p_s_new)
    _lh = phys_out.lhflx if phys_out.lhflx is not None else jnp.zeros_like(p_s_new)
    shflx_accum = carry.shflx_accum + _sh * statics.dt
    lhflx_accum = carry.lhflx_accum + _lh * statics.dt
    _ev = _segment_water_flux(phys_out, jnp.zeros_like(p_s_new))
    evap_accum = carry.evap_accum + _ev * statics.dt

    # --- Time-integrate radiative fluxes + lowest-level T ---
    # (segment-mean diagnostics; CMOR diurnal-alias fix).  held_new order:
    # (dT_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa, sw_down_toa).
    sw_net_sfc_accum = carry.sw_net_sfc_accum + held_new[1] * statics.dt
    lw_net_sfc_accum = carry.lw_net_sfc_accum + held_new[2] * statics.dt
    sw_up_toa_accum = carry.sw_up_toa_accum + held_new[3] * statics.dt
    lw_up_toa_accum = carry.lw_up_toa_accum + held_new[4] * statics.dt
    sw_down_toa_accum = carry.sw_down_toa_accum + held_new[5] * statics.dt
    t_low_accum = carry.t_low_accum + T_upd[..., -1] * statics.dt
    # Clear-sky TOA up-flux time integral (#843): held_new[6]/[7] = the
    # clear-sky sw/lw held-or-fresh values from the pass above (zeros when
    # off, so accum stays zero).
    sw_up_toa_clr_accum = carry.sw_up_toa_clr_accum + held_new[6] * statics.dt
    lw_up_toa_clr_accum = carry.lw_up_toa_clr_accum + held_new[7] * statics.dt

    return _SplitStepLocals(
        u_new=u_new, v_new=v_new, p_s_new=p_s_new,
        T_upd=T_upd, q_v_upd=q_v_upd, q_c_upd=q_c_upd, q_r_upd=q_r_upd,
        q_i_upd=q_i_upd, q_s_upd=q_s_upd, q_g_upd=q_g_upd,
        N_c_upd=N_c_upd, N_r_upd=N_r_upd, N_i_upd=N_i_upd,
        conv_prog_upd=conv_prog_upd, held_new=held_new,
        precip_accum=precip_accum, conv_precip_prev_new=conv_precip_prev_new,
        shflx_accum=shflx_accum, lhflx_accum=lhflx_accum,
        evap_accum=evap_accum,
        sw_net_sfc_accum=sw_net_sfc_accum, lw_net_sfc_accum=lw_net_sfc_accum,
        sw_up_toa_accum=sw_up_toa_accum, lw_up_toa_accum=lw_up_toa_accum,
        sw_up_toa_clr_accum=sw_up_toa_clr_accum,
        lw_up_toa_clr_accum=lw_up_toa_clr_accum,
        sw_down_toa_accum=sw_down_toa_accum, t_low_accum=t_low_accum,
        T_land_new=T_land_new, land_ml_new=land_ml_new,
        w_land_new=w_land_new, snow_new=snow_new,
        phys_tke=phys_out.tke, phys_qke=phys_out.qke,
        phys_gwd=phys_out.gwd_spectrum,
        phys_cloud_fraction=phys_out.cloud_fraction,
        budget_ledger_step=_led_step,
    )


def _apply_qv_smoothing(q_v_upd, statics):
    """Moisture ∇⁴ smoothing + positivity floor for the operator-split tail.

    Static gate: ``qv_smooth_coeff`` is a frozen-dataclass closure constant
    (NOT traced), so this ``if`` folds at compile time.  When it is 0 the
    hyperdiffusion is a ×0 no-op AND skipping it avoids calling the
    grid-specific ∇⁴ operator — ``laplacian_compact_3d`` reads the cube-only
    ``grid.halo_interp_offsets`` (absent on lat-lon), which the lat-lon training
    rollout (``qv_smooth_coeff=0.0``) would otherwise hit.  The nonzero-coeff
    path is bit-identical to the former nested ``max(q + dt·∇⁴, 0)``; only the
    q>=0 floor always applies.  Module-scope (not defined in the scan body) per
    the repo's hot-loop helper doctrine, and so fix-5 is directly testable.
    """
    if statics.qv_smooth_coeff != 0.0:
        q_v_upd = q_v_upd + statics.dt * statics.hyperdiffusion_3d(
            q_v_upd, statics.grid, statics.qv_smooth_coeff)
    return jnp.maximum(q_v_upd, 0.0)


def finalize_split_step(carry, lz, statics):
    """The shared operator-split TAIL (saturation adjustment, moisture fixer,
    moisture smoothing, Rayleigh friction, ``SegmentCarry`` pack) — VERBATIM
    extraction of the former ``_single_step`` post-``else`` tail, run by BOTH
    the single-rank and MPI-owned-face branches from ONE source.  Reads the
    branch outputs off ``lz`` and closure statics off ``statics``."""
    from legoesm.core.conservation import fix_moisture_hydrostatic
    T_upd = lz.T_upd
    q_v_upd = lz.q_v_upd
    p_s_new = lz.p_s_new

    # Budget ledger: snapshot the tail's entry (T, q_v) store so the tail's
    # water/enthalpy changes (saturation adjustment, moisture fixer, q_v
    # smoothing floor) are booked into the CLIPS row below.
    if statics.budget_ledger:
        from legoesm.diagnostics.process_ledger import column_store_snapshot
        _led_tail_before = column_store_snapshot(
            p_s_new, statics.dsigma, T_upd, q_v_upd,
            area=statics.grid.grid_area)

    # --- Saturation adjustment ---
    if statics.do_sat_adjust:
        p_full = p_s_new[..., None] * statics.sigma_full
        q_sat = saturation_mixing_ratio(T_upd, p_full)
        excess = jnp.maximum(q_v_upd - q_sat, 0.0)
        q_v_upd = q_v_upd - excess
        T_upd = T_upd + constants.L_v * excess / constants.c_pd

    # --- Moisture fixer (uses fixed target from initialization) ---
    if statics.fix_moisture:
        q_v_upd = fix_moisture_hydrostatic(
            q_v_upd, carry.target_moisture,
            p_s_new, statics.dsigma, statics.grid,
            owned_mask=statics.owned_mask,
        )

    # --- Moisture smoothing (static-gated ∇⁴ + positivity floor) ---
    q_v_upd = _apply_qv_smoothing(q_v_upd, statics)

    # Budget ledger: book the tail delta into CLIPS, then time-integrate the
    # per-step ledger into the carry accumulator (rate·dt, like precip_accum).
    if statics.budget_ledger:
        from legoesm.diagnostics.process_ledger import (
            ROW_CLIPS, column_store_snapshot,
        )
        _led_tail_after = column_store_snapshot(
            p_s_new, statics.dsigma, T_upd, q_v_upd,
            area=statics.grid.grid_area)
        _led_tail_rate = (_led_tail_after - _led_tail_before) / statics.dt
        _led_step = lz.budget_ledger_step.at[ROW_CLIPS].add(_led_tail_rate)
        _led_accum = (carry.budget_ledger_accum
                      + _led_step.astype(carry.budget_ledger_accum.dtype)
                      * statics.dt)
    else:
        _led_accum = carry.budget_ledger_accum

    # --- Rayleigh friction ---
    u_upd = lz.u_new * statics.fric_decay
    v_upd = lz.v_new * statics.fric_decay

    # CFL is computed at segment boundary (host-side) from the final carry's
    # wind fields, not inside the hot loop.
    max_cfl = carry.max_cfl

    # Cast all arrays back to carry input dtypes to prevent float32→float64
    # promotion from Python float constants breaking jax.lax.scan.
    new_carry = SegmentCarry(
        u=_match_dtype(u_upd, carry.u),
        v=_match_dtype(v_upd, carry.v),
        T=_match_dtype(T_upd, carry.T),
        p_s=_match_dtype(p_s_new, carry.p_s),
        phis=carry.phis,
        q_v=_match_dtype(q_v_upd, carry.q_v),
        q_c=_match_dtype(lz.q_c_upd, carry.q_c),
        q_r=_match_dtype(lz.q_r_upd, carry.q_r),
        conv_prog=_match_dtype(lz.conv_prog_upd, carry.conv_prog),
        held_dT_rad=_match_dtype(lz.held_new[0], carry.held_dT_rad),
        held_sw_net_sfc=_match_dtype(lz.held_new[1], carry.held_sw_net_sfc),
        held_lw_net_sfc=_match_dtype(lz.held_new[2], carry.held_lw_net_sfc),
        held_sw_up_toa=_match_dtype(lz.held_new[3], carry.held_sw_up_toa),
        held_lw_up_toa=_match_dtype(lz.held_new[4], carry.held_lw_up_toa),
        held_sw_up_toa_clr=_match_dtype(
            lz.held_new[6], carry.held_sw_up_toa_clr),
        held_lw_up_toa_clr=_match_dtype(
            lz.held_new[7], carry.held_lw_up_toa_clr),
        held_sw_down_toa=_match_dtype(lz.held_new[5], carry.held_sw_down_toa),
        step_index=carry.step_index + 1,
        target_moisture=carry.target_moisture,
        target_mass=carry.target_mass,
        max_cfl=max_cfl,
        precip_accum=_match_dtype(lz.precip_accum, carry.precip_accum),
        shflx_accum=_match_dtype(lz.shflx_accum, carry.shflx_accum),
        lhflx_accum=_match_dtype(lz.lhflx_accum, carry.lhflx_accum),
        evap_accum=_match_dtype(lz.evap_accum, carry.evap_accum),
        sw_up_toa_accum=_match_dtype(
            lz.sw_up_toa_accum, carry.sw_up_toa_accum),
        lw_up_toa_accum=_match_dtype(
            lz.lw_up_toa_accum, carry.lw_up_toa_accum),
        sw_up_toa_clr_accum=_match_dtype(
            lz.sw_up_toa_clr_accum, carry.sw_up_toa_clr_accum),
        lw_up_toa_clr_accum=_match_dtype(
            lz.lw_up_toa_clr_accum, carry.lw_up_toa_clr_accum),
        sw_down_toa_accum=_match_dtype(
            lz.sw_down_toa_accum, carry.sw_down_toa_accum),
        sw_net_sfc_accum=_match_dtype(
            lz.sw_net_sfc_accum, carry.sw_net_sfc_accum),
        lw_net_sfc_accum=_match_dtype(
            lz.lw_net_sfc_accum, carry.lw_net_sfc_accum),
        t_low_accum=_match_dtype(lz.t_low_accum, carry.t_low_accum),
        T_land=(None if carry.T_land is None
                else _match_dtype(lz.T_land_new, carry.T_land)),
        q_i=(None if carry.q_i is None
             else _match_dtype(lz.q_i_upd, carry.q_i)),
        q_s=(None if carry.q_s is None
             else _match_dtype(lz.q_s_upd, carry.q_s)),
        q_g=(None if carry.q_g is None
             else _match_dtype(lz.q_g_upd, carry.q_g)),
        N_c=(None if carry.N_c is None
             else _match_dtype(lz.N_c_upd, carry.N_c)),
        N_r=(None if carry.N_r is None
             else _match_dtype(lz.N_r_upd, carry.N_r)),
        N_i=(None if carry.N_i is None
             else _match_dtype(lz.N_i_upd, carry.N_i)),
        tke=(None if carry.tke is None
             else _match_dtype(
                 lz.phys_tke if lz.phys_tke is not None
                 else carry.tke, carry.tke)),
        qke=(None if carry.qke is None
             else _match_dtype(
                 lz.phys_qke if lz.phys_qke is not None
                 else carry.qke, carry.qke)),
        gwd_spectrum=(None if carry.gwd_spectrum is None
                      else _match_dtype(
                          lz.phys_gwd if lz.phys_gwd is not None
                          else carry.gwd_spectrum,
                          carry.gwd_spectrum)),
        cloud_fraction=(None if carry.cloud_fraction is None
                        else _match_dtype(
                            lz.phys_cloud_fraction
                            if lz.phys_cloud_fraction is not None
                            else carry.cloud_fraction,
                            carry.cloud_fraction)),
        conv_precip_prev=_match_dtype(
            lz.conv_precip_prev_new, carry.conv_precip_prev),
        land_ml=lz.land_ml_new,
        w_land=(None if carry.w_land is None
                else _match_dtype(lz.w_land_new, carry.w_land)),
        snow=(None if carry.snow is None
              else _match_dtype(lz.snow_new, carry.snow)),
        # Ledger accumulator: None (gate off) passes through unchanged so the
        # carry pytree structure is stable across scan iterations.
        budget_ledger_accum=_led_accum,
    )
    return new_carry


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
    C_H=None,
    C_E=None,
    albedo_ice=None,
    albedo_ocean=None,
    ghg_vmr_override=None,
    owned_face_ids=None,
    hs_newtonian_relax=None,
    step_unified_no_rad=None,
    device_config=None,
    energy_consistent_moisture_clip: bool = False,
    pipeline=None,
    advect_moisture: bool = False,
    tiled_step_fn=None,
    budget_ledger: bool = False,
    persistent_dgrid: bool = False,
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
        JIT-compiled physics step from
        ``PhysicsPipeline.build_step_unified()``.  When
        ``step_unified_no_rad`` is provided, this is interpreted as the
        ``static_need_rad=True`` (rad-every-call) variant; otherwise it
        keeps its legacy data-dependent ``lax.cond`` behaviour.
    step_unified_no_rad : callable, optional
        Issue #316 fix: the ``static_need_rad=False`` variant
        (held-radiation, no fresh RRTMGP call).  When this is provided
        the scan body uses a cond-free subcycled outer/inner pair (one
        radiation call per ``rad_update_steps`` inner physics steps),
        which keeps the XLA HLO graph compact and bounds JIT compile
        time at long scan lengths.  When ``None`` (default) the legacy
        ``lax.cond`` body is used regardless of segment length — for
        backward compatibility and for callers that cannot guarantee
        ``n_steps % rad_update_steps == 0``.
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
    device_config : DeviceConfig or None, optional
        Single-process multi-GPU SPMD device configuration (from
        ``legoesm.parallel.mesh.create_device_mesh``).  When provided
        AND ``device_config.mesh is not None``, the segment JITs pin
        explicit ``in_shardings``/``out_shardings`` READ DIRECTLY OFF
        the input carry/forcing the driver already sharded upstream
        (``ModelDriver`` runs ``shard_pytree`` / ``shard_state`` before
        the segment call, so every committed array leaf carries its own
        ``NamedSharding``).  The jit copies that existing layout
        verbatim — it does NOT re-derive a face-shard-vs-replicate
        policy.  This matters because the moist/AMIP carry stores
        FLATTENED cell-packed fields (the profile-prognostic
        ``conv_prog`` is ``[n_cells, nlev]`` = ``[6*n*n, nlev]``); the
        shape-``[0]``-keyed ``create_output_shardings`` classifier
        misreads those flat leaves as replicated, so the formerly
        derived ``in_sharding`` ``P()`` disagreed with the arg's actual
        sharding and ``jax.jit`` raised "Sharding passed to jit does not
        match the sharding on the respective arg" (the moist segment was
        the third single-process multi-GPU replication site).  Reading
        the actual ``.sharding`` is correct for BOTH the cubed-sphere
        ``(6, n, n, nlev)`` dycore state and the flat cell-packed carry.
        ``out_shardings`` equals the carry's input layout because the
        scan body preserves carry shape (no resize/repack).  In that
        mode the returned function's ``.raw`` attribute becomes a
        non-donating *sharded* JIT wrapper (still composable with an
        outer ``jax.grad`` — buffer donation, not JIT, is the AD
        conflict; a first ``.raw`` call under ``jax.grad`` sees abstract
        tracer leaves with no concrete sharding, so the jit infers the
        layout, and a later concrete production call gets its own pinned
        wrapper via the per-leaf sharding cache key).  With
        ``device_config=None`` (default) or a mesh-less config,
        behaviour is byte-identical to the legacy path: donating JIT
        kernels, and ``.raw`` stays the non-JIT, non-donating function
        that the training drivers differentiate through.
    energy_consistent_moisture_clip : bool, optional
        Issue #323.  When ``True``, the per-step ``max(q_v, 0)`` floor on
        the physics tracer update also removes the latent heat tied to the
        clipped (un-removed) vapour sink, so the floor conserves moist
        static energy ``c_pd*T + L_v*q_v`` instead of injecting spurious
        condensation heat.  This targets the kessler+sbm wind blow-up
        (the combined vapour sink can exceed the available ``q_v`` under
        organised convection; the existing floor truncates the sink but
        keeps the full latent heating).  Default ``False`` =>
        bit-identical to the legacy path.
    advect_moisture : bool, optional
        Issue #771.  When ``True``, the moisture tracers (q_v/q_c/q_r and
        the double-moment fields when present) are attached to the dycore
        state each step so the primitive-equation step advects them with
        the resolved wind (the MPAS Phase B design); the physics update
        then acts on the post-advection fields.  Requires a tracer-capable
        dycore (cubed-sphere cdgrid).  Default ``False`` => the legacy
        column-locked moisture (physics tendencies + hyperdiffusion
        smoothing only), byte-identical.
    pipeline : PhysicsPipeline or None, optional
        The physics pipeline.  Only consumed by the optional "un-fused
        radiation" host path (``ExperimentConfig.unfused_radiation`` —
        PRODUCTION ``_run_compiled`` only): it lets the returned object
        expose ``run_norad_scan`` (a host-callable jit of a pure no-rad
        ``rad_update_steps``-length scan) and ``run_rad`` (a
        host-callable jit of ``pipeline.compute_radiation_core``) so
        rrtmgp and the dynamics+physics scan compile as TWO SEPARATE
        XLA executables instead of one inlined ~3h graph.  ``None``
        (default) leaves both attributes ``None`` and changes nothing —
        the legacy fused ``run_segment`` path is byte-identical.

    Returns
    -------
    callable
        ``run_segment(carry: SegmentCarry, n_steps: int,
        forcing: SegmentForcing) -> SegmentCarry``
    """
    # #1028 dispatch hardening: the persistent-D wind layout is wired for the
    # single-rank cube lane.  The MPI owned-face branch slices winds with
    # ``[owned_face_ids]`` on arrays that would now be (n+1) in both horizontal
    # directions, and the tiled adapter converts to cell centres by
    # construction -- refuse LOUDLY rather than silently running the damped
    # cell-centre path this flag exists to remove.
    if persistent_dgrid:
        if owned_face_ids is not None:
            raise ValueError(
                "persistent_dgrid=True is not supported with owned_face_ids "
                "(MPI face-sharded physics): the owned-face slicing is written "
                "for cell-centre (n) arrays, not D-grid corner (n+1) arrays.")
        if tiled_step_fn is not None:
            raise ValueError(
                "persistent_dgrid=True is not supported with a tiled step: "
                "the tiled adapter enters and exits at cell centres by "
                "construction (tiled_step_adapter), so the round trip this "
                "flag removes would still happen every step.")
    from legoesm.core.conservation import (
        fix_moisture_hydrostatic, fix_ps_mass_target,
        energy_consistent_moisture_floor,
    )
    from legoesm.core.cfl import estimate_min_dx_cubed_sphere

    # Per-step solar clock precision (codex round-11 Low): the in-scan
    # ``_abs_day``/``day_to_calendar`` arithmetic is TRACED — without x64
    # it runs in float32, and ``seconds_of_day`` quantizes (~8 s ulp at
    # day ~1000), silently degrading the solar zenith on long runs.
    # Production sets JAX_ENABLE_X64; warn loudly when it is off.
    if not jax.config.jax_enable_x64:
        logger.warning(
            "build_segment_fn: JAX x64 is DISABLED — the per-step solar "
            "clock (diurnal cycle, #720) computes day/seconds-of-day in "
            "float32 inside the compiled scan; multi-year runs will "
            "accumulate solar-time quantization (~8 s at day 1000). Set "
            "JAX_ENABLE_X64=1 (production default) for exact solar time."
        )

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
    _C_H = jnp.asarray(C_H) if C_H is not None else None
    _C_E = jnp.asarray(C_E) if C_E is not None else None
    _albedo_ice = jnp.asarray(albedo_ice) if albedo_ice is not None else None
    _albedo_ocean = jnp.asarray(albedo_ocean) if albedo_ocean is not None else None
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
        if budget_ledger:
            # The ledger's dynamics/clips seams live in the single-rank
            # split body; the MPI owned-face branch does not fill them and
            # its global means would need allreduce plumbing.  Refuse loudly
            # (dispatch-hardening: no silent zero rows on a distributed run).
            raise NotImplementedError(
                "budget_ledger diagnostics are single-rank only — run the "
                "instrumented diagnosis on one device (P0.4 protocol) or "
                "extend the MPI owned-face branch first.")

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

    def _make_single_step(forcing: SegmentForcing, step_fn=None):
        """Create the scan body closed over a specific forcing pytree.

        The forcing is passed through the scan as a constant (not
        varying per step), so closing here is equivalent to passing it
        in scan's xs — but simpler.

        ``step_fn`` selects which ``build_step_unified`` variant the
        body invokes.  Issue #316: the rad-only and no-rad-only
        variants elide the inner ``lax.cond``, which is what lets the
        subcycled outer/inner scan in :func:`run_segment_jit` keep the
        WhileLoop body small.  ``None`` defaults to the legacy
        ``step_unified`` (data-dependent cond) for backward
        compatibility.
        """
        # Reconstruct GHG VMR dict from forcing array + static keys.
        # _ghg_keys is a Python tuple captured in the closure; its
        # length determines whether step_unified receives None or dict.
        _ghg_vmr_override = ghg_array_to_dict(forcing.ghg_vmr, _ghg_keys)
        _step_unified = step_fn if step_fn is not None else step_unified

        # Clear-sky diagnostic freshness mode for THIS step variant (#843).
        # Static Python (no HLO cost when off): decide once whether this
        # ``_single_step`` variant runs the clouds-off 2nd radiation pass.
        #   * no-rad subcycle variant  -> "hold"  (keep body_no_rad rrtmgp-free)
        #   * feature off              -> "hold"  (byte-identical)
        #   * static rad-every-call    -> "always" (all-sky fresh every call)
        #   * legacy data-dependent    -> "cond"   (fresh only when need_rad)
        _clr_on = (pipeline is not None
                   and getattr(pipeline, "_clear_sky_diag", False))
        _clr_is_no_rad_variant = (
            step_unified_no_rad is not None
            and _step_unified is step_unified_no_rad
        )
        if not _clr_on or _clr_is_no_rad_variant:
            _clr_fresh = "hold"
        elif step_unified_no_rad is not None:
            _clr_fresh = "always"
        else:
            _clr_fresh = "cond"

        # Bundle the operator-split closure statics ONCE so the module-scope
        # single-rank physics + shared finalizer (reused by the lat-band-SPMD
        # lane) read them off one object.  ``_step_unified`` / ``forcing`` /
        # ``_ghg_vmr_override`` depend on this forcing, so the bundle is per
        # ``_make_single_step`` call.
        # Byte-identical to the former inline ``_SplitStepStatics(...)`` literal:
        # the factory wraps the scalar constants with the SAME ``jnp.asarray``
        # (raw ``dt``/``tau_*``/``C_*``/``fric_decay`` -> the wrapped ``_dt``/
        # ``_tau_*``/... values).  Shared with the lat-band-SPMD driver lane.
        _split_statics = build_operator_split_statics(
            step_unified=_step_unified,
            forcing=forcing,
            lat=lat, lon=lon, dt=dt,
            tau_equator=tau_equator, tau_pole=tau_pole,
            sbm_tau_c=sbm_tau_c, sbm_RH_ref=sbm_RH_ref,
            C_H=C_H, C_E=C_E,
            albedo_ice=albedo_ice, albedo_ocean=albedo_ocean,
            ghg_vmr_override=_ghg_vmr_override,
            hs_newtonian_relax=hs_newtonian_relax,
            energy_consistent_moisture_clip=energy_consistent_moisture_clip,
            do_sat_adjust=do_sat_adjust,
            fix_moisture=fix_moisture,
            sigma_full=sigma_full, dsigma=dsigma, grid=grid,
            owned_mask=_owned_mask,
            qv_smooth_coeff=qv_smooth_coeff,
            fric_decay=fric_decay,
            hyperdiffusion_3d=hyperdiffusion_3d,
            pipeline=pipeline, clear_sky_fresh=_clr_fresh,
            budget_ledger=budget_ledger,
        )

        def _single_step(carry: SegmentCarry, _unused) -> tuple:
            """One atmosphere step: dynamics → physics → fixers."""
            step_idx = carry.step_index

            # --- Dynamics ---
            # P4 increment 1b: ``tiled_step_fn`` (built by the driver via
            # ``make_tiled_cc_step`` when enable_tiled_dycore + sub-face
            # tiling) replaces ONLY the dynamics core — same cc
            # HydrostaticState contract as ``_dynamics_model.step`` (dt is
            # baked into the tiled stage at build; the driver passes the
            # SAME dt to both).  Physics / fixers / tracers in this scan
            # body are untouched (they already run outside the dynamics
            # call).
            if tiled_step_fn is not None:
                dyn_state = tiled_step_fn(
                    _rebuild_state(carry, _dynamics_model,
                                   advect_moisture=advect_moisture))
            else:
                dyn_state = _dynamics_model.step(
                    _rebuild_state(carry, _dynamics_model,
                                   advect_moisture=advect_moisture,
                                   persistent_dgrid=persistent_dgrid),
                    _dt,
                )

            T_new = dyn_state.T.data
            # #1028: the cube hydrostatic lane may carry FV3 D-staggered winds.
            # ``u_new``/``v_new`` are ALWAYS the native carried arrays (they go
            # straight back into the carry); ``u_cc``/``v_cc`` are the
            # cell-centre view the column physics reads.  Off the persistent-D
            # lane they are the same arrays, so the whole body is
            # byte-identical.
            _dgrid_winds = hasattr(dyn_state, "u_d")
            if _dgrid_winds:
                u_new = dyn_state.u_d.data
                v_new = dyn_state.v_d.data
                u_cc, v_cc = dgrid_to_center_vector(u_new, v_new)
            else:
                u_new = dyn_state.u.data
                v_new = dyn_state.v.data
                u_cc, v_cc = u_new, v_new
            p_s_new = dyn_state.p_s.data

            # --- Advected moisture (issue #771) ---
            # With advect_moisture the tracers rode ``dyn_state`` through the
            # PE step (resolved-wind transport, same MPAS Phase B design); the
            # physics update below then acts on the POST-ADVECTION fields.
            # With the flag off these aliases are exactly the carry fields, so
            # every downstream line is byte-identical to the legacy
            # column-locked behaviour.
            if advect_moisture:
                _adv = dyn_state.tracers
                q_v_dyn = _adv["q_v"].data
                q_c_dyn = _adv["q_c"].data
                q_r_dyn = _adv["q_r"].data
                # Every water species advects since 2026-08-14 (all stored
                # per mass, registry-driven), so for present fields this
                # fallback is dead; it remains for OPTIONAL fields that are
                # None on warm-rain runs — those must stay None, not be
                # invented.
                q_i_dyn = _adv["q_i"].data if "q_i" in _adv else carry.q_i
                q_s_dyn = _adv["q_s"].data if "q_s" in _adv else carry.q_s
                q_g_dyn = _adv["q_g"].data if "q_g" in _adv else carry.q_g
                N_c_dyn = _adv["N_c"].data if "N_c" in _adv else carry.N_c
                N_r_dyn = _adv["N_r"].data if "N_r" in _adv else carry.N_r
                N_i_dyn = _adv["N_i"].data if "N_i" in _adv else carry.N_i
            else:
                q_v_dyn, q_c_dyn, q_r_dyn = carry.q_v, carry.q_c, carry.q_r
                q_i_dyn, q_s_dyn, q_g_dyn = carry.q_i, carry.q_s, carry.q_g
                N_c_dyn, N_r_dyn, N_i_dyn = carry.N_c, carry.N_r, carry.N_i

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

            # --- Per-step solar time (diurnal-cycle fix) ---
            # forcing.day_of_year / forcing.seconds_of_day are packed ONCE per
            # segment (at the segment-end wall clock).  Using them for the solar
            # zenith froze the sun for the whole segment: with whole-day segments
            # the segment-end time is always the same wall clock (midnight UTC for
            # 5-day segments started at day 0) so the compiled path had NO diurnal
            # cycle and a permanently mis-placed sun — unlike the per-step
            # reference driver, which recomputes day_to_calendar every step.
            # Recover the diurnal cycle by advancing the wall clock from the
            # ABSOLUTE step index (``step_idx`` is the absolute counter, carried
            # across segments).  ``(step_idx + 1)`` = end-of-step time, matching
            # the per-step driver's ``day = START_DAY + (step + 1) * DT``.  Only
            # the SOLAR position (declination + hour angle) uses this; the
            # seasonal forcing (SST / O3 / aerosol / GHG) stays per-segment
            # (those ride separate arrays, unaffected by these two scalars).
            _abs_day = start_day + (step_idx + 1) * _dt / 86400.0
            _doy_step, _sod_step = day_to_calendar(_abs_day)

            if owned_face_ids is not None:
                # MPI replicated dynamics: physics on owned faces only.
                # Dynamics state is full (6, n, n, ...) but physics inputs
                # (forcing, lat/lon) are rank-local.  Extract owned faces
                # from dynamics fields, run physics, write back.
                _ofi = owned_face_ids
                _T_land_in = (carry.T_land[_ofi]
                              if carry.T_land is not None else None)
                _w_land_in = (carry.w_land[_ofi]
                              if carry.w_land is not None else None)
                _snow_in = (carry.snow[_ofi]
                            if carry.snow is not None else None)
                # Double-moment tracers (None for warm-rain) → number-aware
                # radiation r_eff. Passed by KEYWORD so the neural/SFNO
                # step_unified wrappers (which parse the positional tail by
                # length) route them through **kwargs unchanged.
                _dm_in = {}
                for _nm, _dyn in (("q_i", q_i_dyn), ("q_s", q_s_dyn),
                                  ("q_g", q_g_dyn), ("N_c", N_c_dyn),
                                  ("N_r", N_r_dyn), ("N_i", N_i_dyn)):
                    if _dyn is not None:
                        _dm_in[_nm] = _dyn[_ofi]
                # Stateful-physics carries (issue #413): rank-local
                # owned-column layout like conv_prog — passed whole,
                # by keyword (None fields omitted; legacy wrappers
                # route unknown keywords through **kwargs unchanged).
                for _nm in ("tke", "qke", "gwd_spectrum"):
                    _fld = getattr(carry, _nm)
                    if _fld is not None:
                        _dm_in[_nm] = _fld
                _ret = _step_unified(
                    need_rad,
                    T_new[_ofi], p_s_new[_ofi],
                    q_v_dyn[_ofi], q_c_dyn[_ofi], q_r_dyn[_ofi],
                    carry.conv_prog,
                    u_new[_ofi], v_new[_ofi],
                    forcing.sst, forcing.sic, lat, lon,
                    _doy_step, _sod_step, _dt,
                    forcing.solar_weights, forcing.s_0,
                    forcing.o3_vmr, forcing.aerosol_od,
                    carry.held_dT_rad[_ofi], carry.held_sw_net_sfc[_ofi],
                    carry.held_lw_net_sfc[_ofi],
                    carry.held_sw_up_toa[_ofi], carry.held_lw_up_toa[_ofi],
                    carry.held_sw_down_toa[_ofi],
                    tau_equator=_tau_equator, tau_pole=_tau_pole,
                    sbm_tau_c=_sbm_tau_c, sbm_RH_ref=_sbm_RH_ref,
                    C_H=_C_H, C_E=_C_E,
                    albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                    ghg_vmr_override=_ghg_vmr_override,
                    aerosol_lw_od=forcing.aerosol_lw_od,
                    aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                    o3_top_vmr=forcing.o3_top_vmr,
                    sfc_albedo_override=forcing.sfc_albedo_override,
                    sfc_T_override=forcing.sfc_T_override,
                    sfc_emissivity_override=forcing.sfc_emissivity_override,
                    sfc_shflx_override=forcing.sfc_shflx_override,
                    sfc_lhflx_override=forcing.sfc_lhflx_override,
                    sfc_evap_override=forcing.sfc_evap_override,
                    sfc_taux_override=forcing.sfc_taux_override,
                    sfc_tauy_override=forcing.sfc_tauy_override,
                    sfc_lw_up=forcing.sfc_lw_up,
                    sfc_sw_up=forcing.sfc_sw_up,
                    sfc_sw_down=forcing.sfc_sw_down,
                    land_frac=forcing.land_frac,
                    phis=carry.phis[_ofi],
                    T_land=_T_land_in, land_ml=carry.land_ml,
                    land_ml_params=forcing.land_ml_params,
                    conv_precip=carry.conv_precip_prev[_ofi],
                    w_land=_w_land_in, snow=_snow_in,
                    **_dm_in,
                )
                phys_out, held_new_local = _ret[0], _ret[1]
                # 3rd value = slab-land skin T (#325); legacy 2-tuple
                # wrappers leave the land tile inert.
                _T_land_local = _ret[2] if len(_ret) > 2 else _T_land_in

                # --- Clear-sky TOA second radiation pass, owned faces (#843) ---
                # Same owned-face state _step_unified's all-sky radiation saw,
                # clouds OFF.  "hold" mode (feature off / no-rad variant) runs
                # no extra RRTMGP -> byte-identical + body_no_rad stays clean.
                def _compute_fresh_clr_owned():
                    (_c0, _c1, _c2, _sw_clr, _lw_clr, _c5, _c6, _c7) = \
                        pipeline.compute_radiation_core(
                            T_new[_ofi], p_s_new[_ofi], q_v_dyn[_ofi],
                            forcing.sst, forcing.sic, lat, lon,
                            _doy_step, _sod_step,
                            forcing.solar_weights, forcing.s_0,
                            forcing.o3_vmr, forcing.aerosol_od,
                            aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
                            aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                            o3_top_vmr=forcing.o3_top_vmr,
                            tau_equator=_tau_equator, tau_pole=_tau_pole,
                            albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                            ghg_vmr_override=_ghg_vmr_override,
                            q_c=None, q_i=None, N_c=None, N_i=None,
                            cloud_scheme="none",
                            u=u_new[_ofi], v=v_new[_ofi], dt=_dt,
                            T_land=_T_land_in, land_ml=carry.land_ml,
                            land_ml_params=forcing.land_ml_params,
                            sfc_albedo_override=forcing.sfc_albedo_override,
                            sfc_T_override=forcing.sfc_T_override,
                            sfc_emissivity_override=forcing.sfc_emissivity_override,
                            sfc_lw_up=forcing.sfc_lw_up,
                            sfc_sw_up=forcing.sfc_sw_up,
                            sfc_sw_down=forcing.sfc_sw_down,
                            conv_precip=carry.conv_precip_prev[_ofi],
                            w_land=_w_land_in, snow=_snow_in,
                        )
                    _dt_h = carry.held_sw_up_toa_clr.dtype
                    return _sw_clr.astype(_dt_h), _lw_clr.astype(_dt_h)

                _sw_clr_owned, _lw_clr_owned = _clear_sky_toa_pass(
                    _clr_fresh, need_rad,
                    carry.held_sw_up_toa_clr[_ofi],
                    carry.held_lw_up_toa_clr[_ofi],
                    _compute_fresh_clr_owned,
                )

                # Write physics tendencies back at owned indices.
                # Non-owned faces keep dynamics-only values (no physics).
                _phys_dT = phys_out.dT_dt
                if hs_newtonian_relax is not None:
                    _phys_dT = _phys_dT + hs_newtonian_relax(
                        T_new[_ofi], p_s_new[_ofi], lat[_ofi])
                _T_owned = T_new[_ofi] + _dt * _phys_dT
                _qv_raw = q_v_dyn[_ofi] + _dt * phys_out.dq_v_dt
                if energy_consistent_moisture_clip:
                    # Issue #323: keep the q_v floor moist-static-energy
                    # neutral (remove the latent heat of the clipped vapour
                    # sink) instead of injecting spurious condensation heat
                    # -> the kessler+sbm wind blow-up.  Default off =>
                    # bit-identical (plain ``max(q_v, 0)`` below).
                    _qv_owned, _T_owned = energy_consistent_moisture_floor(
                        _qv_raw, _T_owned)
                else:
                    # Conserving form always (owner decision 2026-08-16):
                    # column borrow on the owned subset (rows are whole
                    # columns, so the borrow is rank-local and MPI-safe).
                    _qv_owned = conservative_positive_clip(
                        _qv_raw, dsigma, axis=-1)[0]
                T_upd = T_new.at[_ofi].set(_T_owned)
                q_v_upd = q_v_dyn.at[_ofi].set(_qv_owned)
                q_c_upd = q_c_dyn.at[_ofi].set(conservative_positive_clip(
                    q_c_dyn[_ofi] + _dt * phys_out.dq_c_dt, dsigma,
                    axis=-1)[0])
                q_r_upd = q_r_dyn.at[_ofi].set(conservative_positive_clip(
                    q_r_dyn[_ofi] + _dt * phys_out.dq_r_dt, dsigma,
                    axis=-1)[0])
                # Double-moment tracers (None unless populated): evolve at owned
                # indices from the matching microphysics tendencies, floored by
                # the conserving borrow like q_c/q_r.  Base = the (possibly
                # advected) _dyn alias so non-owned faces keep dynamics-only
                # values.
                def _dm_upd_owned(fld, tend):
                    return (None if fld is None else fld.at[_ofi].set(
                        conservative_positive_clip(
                            fld[_ofi] + _dt * tend, dsigma, axis=-1)[0]))
                q_i_upd = _dm_upd_owned(q_i_dyn, phys_out.dq_i_dt)
                q_s_upd = _dm_upd_owned(q_s_dyn, phys_out.dq_s_dt)
                q_g_upd = _dm_upd_owned(q_g_dyn, phys_out.dq_g_dt)
                N_c_upd = _dm_upd_owned(N_c_dyn, phys_out.dN_c_dt)
                N_r_upd = _dm_upd_owned(N_r_dyn, phys_out.dN_r_dt)
                N_i_upd = _dm_upd_owned(N_i_dyn, phys_out.dN_i_dt)
                conv_prog_upd = phys_out.conv_prog

                # Held radiation: update at owned indices.  Positions 6/7 =
                # clear-sky TOA up-fluxes (#843), scattered like the all-sky
                # held fields (finalize_split_step writes them into the carry).
                held_new = (
                    carry.held_dT_rad.at[_ofi].set(held_new_local[0]),
                    carry.held_sw_net_sfc.at[_ofi].set(held_new_local[1]),
                    carry.held_lw_net_sfc.at[_ofi].set(held_new_local[2]),
                    carry.held_sw_up_toa.at[_ofi].set(held_new_local[3]),
                    carry.held_lw_up_toa.at[_ofi].set(held_new_local[4]),
                    carry.held_sw_down_toa.at[_ofi].set(held_new_local[5]),
                    carry.held_sw_up_toa_clr.at[_ofi].set(_sw_clr_owned),
                    carry.held_lw_up_toa_clr.at[_ofi].set(_lw_clr_owned),
                )

                # Precip: update at owned indices
                precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new[_ofi])
                precip_accum = carry.precip_accum.at[_ofi].add(precip_step * _dt)
                # Lag this step's total precip for next step's convective cloud.
                conv_precip_prev_new = carry.conv_precip_prev.at[_ofi].set(
                    precip_step)

                # Surface heat fluxes: accumulate at owned indices
                _sh = phys_out.shflx if phys_out.shflx is not None else jnp.zeros_like(p_s_new[_ofi])
                _lh = phys_out.lhflx if phys_out.lhflx is not None else jnp.zeros_like(p_s_new[_ofi])
                shflx_accum = carry.shflx_accum.at[_ofi].add(_sh * _dt)
                lhflx_accum = carry.lhflx_accum.at[_ofi].add(_lh * _dt)
                _ev = _segment_water_flux(phys_out, jnp.zeros_like(p_s_new[_ofi]))
                evap_accum = carry.evap_accum.at[_ofi].add(_ev * _dt)

                # Radiative fluxes + lowest-level T: time-integrate at owned
                # indices (segment-mean diagnostics; CMOR diurnal-alias fix).
                # held_new_local order: (dT_rad, sw_net_sfc, lw_net_sfc,
                # sw_up_toa, lw_up_toa, sw_down_toa) — signs unchanged.
                sw_net_sfc_accum = carry.sw_net_sfc_accum.at[_ofi].add(
                    held_new_local[1] * _dt)
                lw_net_sfc_accum = carry.lw_net_sfc_accum.at[_ofi].add(
                    held_new_local[2] * _dt)
                sw_up_toa_accum = carry.sw_up_toa_accum.at[_ofi].add(
                    held_new_local[3] * _dt)
                lw_up_toa_accum = carry.lw_up_toa_accum.at[_ofi].add(
                    held_new_local[4] * _dt)
                sw_down_toa_accum = carry.sw_down_toa_accum.at[_ofi].add(
                    held_new_local[5] * _dt)
                # Clear-sky TOA up-flux integral at owned indices (#843):
                # owned-face-local clear-sky values (zeros when off).
                sw_up_toa_clr_accum = carry.sw_up_toa_clr_accum.at[_ofi].add(
                    _sw_clr_owned * _dt)
                lw_up_toa_clr_accum = carry.lw_up_toa_clr_accum.at[_ofi].add(
                    _lw_clr_owned * _dt)
                t_low_accum = carry.t_low_accum.at[_ofi].add(
                    T_upd[_ofi][..., -1] * _dt)

                # Slab-land temperature: update at owned indices
                T_land_new = (
                    carry.T_land.at[_ofi].set(_T_land_local)
                    if carry.T_land is not None else None
                )
                # Multilayer land: capture the advanced state (step_unified's
                # optional 4th return), identical to the single-rank path
                # (_step_split_physics_and_finalize).  ``carry.land_ml`` is the
                # rank-local owned-face columns (scattered in
                # ModelDriver._setup_parallel), so _step_unified advanced THIS
                # rank's soil columns; capture them instead of discarding.  When
                # land_ml is None (slab land) the else-branch carries None.
                land_ml_new = _ret[3] if len(_ret) > 3 else carry.land_ml
                # Soil-water bucket: w_land_new rides PhysicsOutput
                # (advanced in physics_step_no_rad), scattered at owned
                # indices.  Carried through unchanged for legacy wrappers
                # that don't populate it.
                w_land_new = (
                    carry.w_land.at[_ofi].set(phys_out.w_land)
                    if (carry.w_land is not None
                        and phys_out.w_land is not None)
                    else carry.w_land
                )
                # Snow water equiv.: rides PhysicsOutput, scattered at owned
                # indices (mirror w_land).
                snow_new = (
                    carry.snow.at[_ofi].set(phys_out.snow)
                    if (carry.snow is not None
                        and phys_out.snow is not None)
                    else carry.snow
                )
                # Package the owned-face outputs into the shared bundle so the
                # SINGLE-source finalizer runs the identical tail on both paths.
                lz = _SplitStepLocals(
                    u_new=u_new, v_new=v_new, p_s_new=p_s_new,
                    T_upd=T_upd, q_v_upd=q_v_upd,
                    q_c_upd=q_c_upd, q_r_upd=q_r_upd,
                    q_i_upd=q_i_upd, q_s_upd=q_s_upd, q_g_upd=q_g_upd,
                    N_c_upd=N_c_upd, N_r_upd=N_r_upd, N_i_upd=N_i_upd,
                    conv_prog_upd=conv_prog_upd, held_new=held_new,
                    precip_accum=precip_accum,
                    conv_precip_prev_new=conv_precip_prev_new,
                    shflx_accum=shflx_accum, lhflx_accum=lhflx_accum,
                    evap_accum=evap_accum,
                    sw_net_sfc_accum=sw_net_sfc_accum,
                    lw_net_sfc_accum=lw_net_sfc_accum,
                    sw_up_toa_accum=sw_up_toa_accum,
                    lw_up_toa_accum=lw_up_toa_accum,
                    sw_up_toa_clr_accum=sw_up_toa_clr_accum,
                    lw_up_toa_clr_accum=lw_up_toa_clr_accum,
                    sw_down_toa_accum=sw_down_toa_accum,
                    t_low_accum=t_low_accum,
                    T_land_new=T_land_new, land_ml_new=land_ml_new,
                    w_land_new=w_land_new, snow_new=snow_new,
                    phys_tke=phys_out.tke, phys_qke=phys_out.qke,
                    phys_gwd=phys_out.gwd_spectrum,
                    phys_cloud_fraction=phys_out.cloud_fraction,
                )
            else:
                # Single-rank (single-controller / lat-band-SPMD) operator-split
                # physics — the module-scope body the SPMD lane reuses verbatim.
                # ``moist`` passes the post-advection aliases (issue #771); with
                # advect_moisture off they ARE carry.q_* -> byte-identical.
                lz = split_physics_single_rank(
                    carry, T_new, u_new, v_new, p_s_new,
                    need_rad, _doy_step, _sod_step, _split_statics,
                    moist={"q_v": q_v_dyn, "q_c": q_c_dyn, "q_r": q_r_dyn,
                           "q_i": q_i_dyn, "q_s": q_s_dyn, "q_g": q_g_dyn,
                           "N_c": N_c_dyn, "N_r": N_r_dyn, "N_i": N_i_dyn},
                    u_cc=u_cc, v_cc=v_cc)

            # Shared operator-split TAIL (sat-adjust, moisture fixer/smoothing,
            # Rayleigh, carry pack) — ONE source for both branches.
            return finalize_split_step(carry, lz, _split_statics), None
        return _single_step

    # Subcycling is only safe when both step_unified variants are
    # supplied AND the segment length is an exact multiple of the
    # radiation cadence — otherwise the trailing remainder would have
    # to be handled outside ``lax.scan`` (extra recompile cost) and
    # the radiation cadence within the segment would drift.
    _subcycle_available = (
        step_unified_no_rad is not None and rad_update_steps > 1
    )

    def _run_subcycled(carry: SegmentCarry, n_steps: int,
                       forcing: SegmentForcing, lead: int) -> SegmentCarry:
        """Issue #316: outer-rad × inner-no-rad nested scan.

        Reproduces the legacy ``need_rad = ((idx+1) % rad_update_steps) == 0``
        cadence -- fresh radiation on the last step of every cycle -- while
        keeping the RRTMGP/gray HLO out of the hot inner body and out of any
        ``lax.cond``.

        Phase is derived from the ABSOLUTE step index, not assumed.  Radiation
        fires at absolute ``i`` with ``(i+1) % k == 0``, so an arbitrary segment
        ``[s, s+n)`` decomposes as::

            lead no-rad | 1 rad | n_outer x (k-1 no-rad, 1 rad) | tail no-rad

        with ``lead = (-(s+1)) % k`` supplied by the caller as a STATIC Python
        int (:func:`_subcycle_lead`) — it cannot be derived here, since this
        body runs under ``jax.jit`` where ``carry.step_index`` is a tracer.
        That handles a segment whose length is
        NOT a multiple of ``k`` and one that starts mid-cycle, which the two
        earlier ``_use_subcycle`` restrictions instead punted to ``_run_single``
        -- and ``_run_single`` hands its ``need_rad`` to the
        ``static_need_rad=True`` variant, which DELETES it and radiates on EVERY
        step (physics_pipeline.py:2556).  So the punt was not a conservative
        fallback: it silently ran a different cadence than requested.  Concrete
        case that motivated this (codex adversarial review): a 151-step run at
        ``rad_update_steps=2`` is a 144-step segment plus a 7-STEP TAIL; the
        tail radiated on all of steps 144-150 where the cadence asks for
        145/147/149.

        """
        body_rad = _make_single_step(forcing, step_fn=step_unified)
        body_no_rad = _make_single_step(forcing, step_fn=step_unified_no_rad)
        if _resolve_checkpoint(gradient_checkpoint, n_steps):
            body_rad = jax.checkpoint(body_rad, prevent_cse=False)
            body_no_rad = jax.checkpoint(body_no_rad, prevent_cse=False)

        k = rad_update_steps

        def _scan_no_rad(c, length):
            if length <= 0:
                return c
            c, _ = jax.lax.scan(body_no_rad, c, None, length=length)
            return c

        # Segment ends before the first radiation step -> all held, no rad.
        if lead >= n_steps:
            return _scan_no_rad(carry, n_steps)

        carry = _scan_no_rad(carry, lead)
        carry, _ = body_rad(carry, None)

        rest = n_steps - lead - 1
        n_outer, tail = rest // k, rest % k
        n_held = k - 1

        def _outer_step(c: SegmentCarry, _):
            c = _scan_no_rad(c, n_held)
            c, _ = body_rad(c, None)
            return c, None

        if n_outer > 0:
            carry, _ = jax.lax.scan(_outer_step, carry, None, length=n_outer)
        return _scan_no_rad(carry, tail)

    def _run_single(carry: SegmentCarry, n_steps: int,
                    forcing: SegmentForcing) -> SegmentCarry:
        """Legacy single-scan body.

        Used when ``step_unified_no_rad`` is not provided, when
        ``rad_update_steps <= 1`` (always-rad, no subcycle benefit
        beyond what the rad-only variant already gives), or when
        ``n_steps`` does not divide evenly by ``rad_update_steps``.
        """
        _step_fn = _make_single_step(forcing)
        if _resolve_checkpoint(gradient_checkpoint, n_steps):
            # Long differentiated rollout -> nested (sqrt-N) checkpointing so the
            # reverse-mode carry storage is O(sqrt(n_steps)), not O(n_steps) which
            # OOMs (#841).  Short segments keep the cheap per-step remat + one scan.
            if n_steps >= _NESTED_CKPT_STEPS:
                return _sqrt_checkpointed_scan(_step_fn, carry, n_steps)
            _step_fn = jax.checkpoint(_step_fn, prevent_cse=False)
        final_carry, _ = jax.lax.scan(_step_fn, carry, None, length=n_steps)
        return final_carry

    # ------------------------------------------------------------------
    # Un-fused radiation host kernels (ExperimentConfig.unfused_radiation;
    # PRODUCTION ``_run_compiled`` ONLY).  A ``jax.jit`` placed INSIDE a
    # ``lax.scan`` body is INLINED by XLA into ONE giant executable — so
    # even the cond-free ``_run_subcycled`` above still inlines the rrtmgp
    # branch and pays the ~3h XLA compile.  Lifting the radiation-cycle
    # loop to the HOST (Python) makes the no-rad dynamics+physics scan and
    # the rrtmgp radiation compile as TWO SEPARATE executables.  These two
    # functions are the host-callable halves; the driver alternates them
    # (``run_norad_scan`` × ``run_rad``) once per radiation cycle.  Both
    # are gated OFF by default: built only when ``step_unified_no_rad`` and
    # ``pipeline`` are supplied AND ``rad_update_steps > 1``.
    # ------------------------------------------------------------------
    _unfused_available = (
        step_unified_no_rad is not None
        and pipeline is not None
        and rad_update_steps > 1
    )

    def _run_norad_scan(carry: SegmentCarry,
                        forcing: SegmentForcing) -> SegmentCarry:
        """Advance exactly ``rad_update_steps`` no-rad steps as ONE scan.

        Pure dynamics+physics with HELD radiation (``step_unified_no_rad``
        — no fresh rrtmgp call), so this HLO contains NO rrtmgp.  Identical
        per-step body to ``_run_subcycled``'s ``body_no_rad``; only the
        compile boundary differs (host-level, not inlined in an outer
        scan).
        """
        body_no_rad = _make_single_step(forcing, step_fn=step_unified_no_rad)
        # This unfused no-rad scan has length rad_update_steps (called once per
        # radiation cycle); resolve the checkpoint policy against THAT length so
        # a long rad_update_steps still auto-checkpoints under default None.
        if _resolve_checkpoint(gradient_checkpoint, rad_update_steps):
            body_no_rad = jax.checkpoint(body_no_rad, prevent_cse=False)
        final_carry, _ = jax.lax.scan(
            body_no_rad, carry, None, length=rad_update_steps,
        )
        return final_carry

    def _run_rad(carry: SegmentCarry,
                 forcing: SegmentForcing) -> SegmentCarry:
        """Recompute FRESH held radiation + T_land from the carry state.

        Radiation ONLY (no dynamics/physics), so this HLO is rrtmgp-only.
        Calls the SAME ``pipeline.compute_radiation_core`` the fused
        ``_rad_branch`` calls (physics_pipeline.py), threading ``forcing``
        identically (so ``sfc_albedo_override`` / ``sfc_T_override`` /
        column-shard handling is byte-identical), and writes the 6 held
        flux fields + ``T_land`` back at owned indices.  Every other carry
        field passes through unchanged — the post-cycle dynamics+physics
        state is untouched here.

        Owned-face extract/write-back MIRROR ``_make_single_step``
        (compiled_segments.py): on the MPI replicated-dynamics path
        physics/radiation operate on owned columns only and write back at
        ``owned_face_ids``; on the single-rank path the whole grid is
        owned.
        """
        # Reconstruct GHG VMR dict from the forcing array + static keys,
        # exactly as ``_make_single_step`` does (keys captured in closure;
        # values dynamic).
        _ghg_vmr_override = ghg_array_to_dict(forcing.ghg_vmr, _ghg_keys)
        # #1028: the surface scheme inside radiation reads winds at cell
        # centres.  With persistent D-grid winds the carry holds corner arrays,
        # so take a read-only cell-centre view; off that lane these ARE
        # carry.u/carry.v and the calls below are byte-identical.
        _u_phys, _v_phys = (
            dgrid_to_center_vector(carry.u, carry.v) if persistent_dgrid
            else (carry.u, carry.v))

        # Clear-sky diagnostic (#843): the unfused rad refresh runs once per
        # radiation cycle, so a static Python gate (no need_rad — this IS the
        # fresh-rad step) suffices.  Off (default) -> held clear-sky passes
        # through unchanged and no 2nd RRTMGP pass is compiled.
        _clr_on_rr = (pipeline is not None
                      and getattr(pipeline, "_clear_sky_diag", False))

        # Per-step solar time (diurnal-cycle fix) — mirror ``_single_step``.
        # The unfused radiation refresh recomputes held fluxes for the upcoming
        # no-rad cycle; use the CYCLE-BOUNDARY wall clock so the zenith
        # advances across cycles instead of being pinned at the segment-end
        # time.  ``carry.step_index`` here counts COMPLETED steps (the
        # preceding no-rad scan already incremented it), so the boundary time
        # is ``step_index * dt`` — matching the fused subcycle, whose fresh
        # radiation is computed IN the last cycle step at ``(step_idx+1)*dt``
        # = the same boundary.  ``+1`` would sample one dt into the future
        # (codex round-11 Medium).
        _abs_day = start_day + carry.step_index * _dt / 86400.0
        _doy_step, _sod_step = day_to_calendar(_abs_day)

        if owned_face_ids is not None:
            _ofi = owned_face_ids
            _T_land_in = (carry.T_land[_ofi]
                          if carry.T_land is not None else None)
            _w_land_in = (carry.w_land[_ofi]
                          if carry.w_land is not None else None)
            _snow_in = (carry.snow[_ofi]
                        if carry.snow is not None else None)

            def _own(fld):
                return None if fld is None else fld[_ofi]

            # 8th return = the advanced MULTILAYER land state (carry.land_ml is
            # the rank-local owned columns, scattered in
            # ModelDriver._setup_parallel).  In the UNFUSED-radiation split the
            # land tile is advanced ONLY here (radiation/surface core), never in
            # the norad scan — so capture it; discarding it froze the soil
            # (serial too).  The clear-sky second pass below discards its 8th
            # (diagnostic re-run, must not re-advance).
            (dT_dt_rad, sw_net_sfc, lw_net_sfc,
             sw_up_toa, lw_up_toa, sw_down_toa, T_land_new_local,
             land_ml_new) = \
                pipeline.compute_radiation_core(
                    carry.T[_ofi], carry.p_s[_ofi], carry.q_v[_ofi],
                    forcing.sst, forcing.sic, lat, lon,
                    _doy_step, _sod_step,
                    forcing.solar_weights, forcing.s_0,
                    forcing.o3_vmr, forcing.aerosol_od,
                    aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
                    aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                    o3_top_vmr=forcing.o3_top_vmr,
                    tau_equator=_tau_equator, tau_pole=_tau_pole,
                    albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                    ghg_vmr_override=_ghg_vmr_override,
                    q_c=_own(carry.q_c), q_i=_own(carry.q_i),
                    N_c=_own(carry.N_c), N_i=_own(carry.N_i),
                    cloud_scheme=pipeline._cloud_scheme,
                    u=carry.u[_ofi], v=carry.v[_ofi], dt=_dt,
                    T_land=_T_land_in, land_ml=carry.land_ml,
                    land_ml_params=forcing.land_ml_params,
                    sfc_albedo_override=forcing.sfc_albedo_override,
                    sfc_T_override=forcing.sfc_T_override,
                    sfc_emissivity_override=forcing.sfc_emissivity_override,
                    sfc_lw_up=forcing.sfc_lw_up,
                    sfc_sw_up=forcing.sfc_sw_up,
                    sfc_sw_down=forcing.sfc_sw_down,
                    conv_precip=_own(carry.conv_precip_prev),
                    w_land=_w_land_in, snow=_snow_in,
                )
            # Clear-sky second pass (#843): same owned-face state, clouds OFF.
            if _clr_on_rr:
                (_c0, _c1, _c2, _sw_clr, _lw_clr, _c5, _c6, _c7) = \
                    pipeline.compute_radiation_core(
                        carry.T[_ofi], carry.p_s[_ofi], carry.q_v[_ofi],
                        forcing.sst, forcing.sic, lat, lon,
                        _doy_step, _sod_step,
                        forcing.solar_weights, forcing.s_0,
                        forcing.o3_vmr, forcing.aerosol_od,
                        aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
                        aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                        o3_top_vmr=forcing.o3_top_vmr,
                        tau_equator=_tau_equator, tau_pole=_tau_pole,
                        albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                        ghg_vmr_override=_ghg_vmr_override,
                        q_c=None, q_i=None, N_c=None, N_i=None,
                        cloud_scheme="none",
                        u=carry.u[_ofi], v=carry.v[_ofi], dt=_dt,
                        T_land=_T_land_in, land_ml=carry.land_ml,
                        land_ml_params=forcing.land_ml_params,
                        sfc_albedo_override=forcing.sfc_albedo_override,
                        sfc_T_override=forcing.sfc_T_override,
                        sfc_emissivity_override=forcing.sfc_emissivity_override,
                        sfc_lw_up=forcing.sfc_lw_up,
                        sfc_sw_up=forcing.sfc_sw_up,
                        sfc_sw_down=forcing.sfc_sw_down,
                        conv_precip=_own(carry.conv_precip_prev),
                        w_land=_w_land_in, snow=_snow_in,
                    )
                _held_sw_clr = carry.held_sw_up_toa_clr.at[_ofi].set(_sw_clr)
                _held_lw_clr = carry.held_lw_up_toa_clr.at[_ofi].set(_lw_clr)
            else:
                _held_sw_clr = carry.held_sw_up_toa_clr
                _held_lw_clr = carry.held_lw_up_toa_clr
            held_new = (
                carry.held_dT_rad.at[_ofi].set(dT_dt_rad),
                carry.held_sw_net_sfc.at[_ofi].set(sw_net_sfc),
                carry.held_lw_net_sfc.at[_ofi].set(lw_net_sfc),
                carry.held_sw_up_toa.at[_ofi].set(sw_up_toa),
                carry.held_lw_up_toa.at[_ofi].set(lw_up_toa),
                carry.held_sw_down_toa.at[_ofi].set(sw_down_toa),
                _held_sw_clr, _held_lw_clr,
            )
            T_land_new = (
                carry.T_land.at[_ofi].set(T_land_new_local)
                if carry.T_land is not None else None
            )
        else:
            # 8th return = the advanced multilayer land state (see the owned-face
            # branch above): the unfused-radiation split advances the land tile
            # ONLY in this radiation/surface core, so capture it — discarding it
            # froze the soil.  land_ml is None for slab-land runs (else-carry).
            (dT_dt_rad, sw_net_sfc, lw_net_sfc,
             sw_up_toa, lw_up_toa, sw_down_toa, T_land_new, land_ml_new) = \
                pipeline.compute_radiation_core(
                    carry.T, carry.p_s, carry.q_v,
                    forcing.sst, forcing.sic, lat, lon,
                    _doy_step, _sod_step,
                    forcing.solar_weights, forcing.s_0,
                    forcing.o3_vmr, forcing.aerosol_od,
                    aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
                    aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                    o3_top_vmr=forcing.o3_top_vmr,
                    tau_equator=_tau_equator, tau_pole=_tau_pole,
                    albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                    ghg_vmr_override=_ghg_vmr_override,
                    q_c=carry.q_c, q_i=carry.q_i,
                    N_c=carry.N_c, N_i=carry.N_i,
                    cloud_scheme=pipeline._cloud_scheme,
                    u=_u_phys, v=_v_phys, dt=_dt,
                    T_land=carry.T_land, land_ml=carry.land_ml,
                    land_ml_params=forcing.land_ml_params,
                    sfc_albedo_override=forcing.sfc_albedo_override,
                    sfc_T_override=forcing.sfc_T_override,
                    sfc_emissivity_override=forcing.sfc_emissivity_override,
                    sfc_lw_up=forcing.sfc_lw_up,
                    sfc_sw_up=forcing.sfc_sw_up,
                    sfc_sw_down=forcing.sfc_sw_down,
                    conv_precip=carry.conv_precip_prev,
                    w_land=carry.w_land, snow=carry.snow,
                )
            # Clear-sky second pass (#843): whole grid, clouds OFF.
            if _clr_on_rr:
                (_c0, _c1, _c2, _sw_clr, _lw_clr, _c5, _c6, _c7) = \
                    pipeline.compute_radiation_core(
                        carry.T, carry.p_s, carry.q_v,
                        forcing.sst, forcing.sic, lat, lon,
                        _doy_step, _sod_step,
                        forcing.solar_weights, forcing.s_0,
                        forcing.o3_vmr, forcing.aerosol_od,
                        aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
                        aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                        o3_top_vmr=forcing.o3_top_vmr,
                        tau_equator=_tau_equator, tau_pole=_tau_pole,
                        albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                        ghg_vmr_override=_ghg_vmr_override,
                        q_c=None, q_i=None, N_c=None, N_i=None,
                        cloud_scheme="none",
                        u=_u_phys, v=_v_phys, dt=_dt,
                        T_land=carry.T_land, land_ml=carry.land_ml,
                        land_ml_params=forcing.land_ml_params,
                        sfc_albedo_override=forcing.sfc_albedo_override,
                        sfc_T_override=forcing.sfc_T_override,
                        sfc_emissivity_override=forcing.sfc_emissivity_override,
                        sfc_lw_up=forcing.sfc_lw_up,
                        sfc_sw_up=forcing.sfc_sw_up,
                        sfc_sw_down=forcing.sfc_sw_down,
                        conv_precip=carry.conv_precip_prev,
                        w_land=carry.w_land, snow=carry.snow,
                    )
                _held_sw_clr, _held_lw_clr = _sw_clr, _lw_clr
            else:
                _held_sw_clr = carry.held_sw_up_toa_clr
                _held_lw_clr = carry.held_lw_up_toa_clr
            held_new = (
                dT_dt_rad, sw_net_sfc, lw_net_sfc,
                sw_up_toa, lw_up_toa, sw_down_toa,
                _held_sw_clr, _held_lw_clr,
            )

        # Write ONLY the 6 held fields + 2 clear-sky held fields (#843) +
        # T_land back (matched to the carry's storage dtype, like the fused
        # write-back); every other field is the unchanged post-cycle state.
        return carry._replace(
            held_dT_rad=_match_dtype(held_new[0], carry.held_dT_rad),
            held_sw_net_sfc=_match_dtype(held_new[1], carry.held_sw_net_sfc),
            held_lw_net_sfc=_match_dtype(held_new[2], carry.held_lw_net_sfc),
            held_sw_up_toa=_match_dtype(held_new[3], carry.held_sw_up_toa),
            held_lw_up_toa=_match_dtype(held_new[4], carry.held_lw_up_toa),
            held_sw_down_toa=_match_dtype(held_new[5], carry.held_sw_down_toa),
            held_sw_up_toa_clr=_match_dtype(
                held_new[6], carry.held_sw_up_toa_clr),
            held_lw_up_toa_clr=_match_dtype(
                held_new[7], carry.held_lw_up_toa_clr),
            T_land=(None if carry.T_land is None
                    else _match_dtype(T_land_new, carry.T_land)),
            # Advanced multilayer land state (rank-local owned columns under
            # cube-face MPI; None for slab-land runs).  Without this the
            # unfused-radiation path never wrote the stepped soil back and the
            # Richards state froze across the whole run.
            land_ml=land_ml_new,
        )

    @partial(jax.jit, static_argnums=(1, 3), donate_argnums=(0,))
    def _run_subcycled_jit(carry: SegmentCarry, n_steps: int,
                           forcing: SegmentForcing, lead: int) -> SegmentCarry:
        return _run_subcycled(carry, n_steps, forcing, lead)

    @partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
    def _run_single_jit(carry: SegmentCarry, n_steps: int,
                        forcing: SegmentForcing) -> SegmentCarry:
        return _run_single(carry, n_steps, forcing)

    # Non-sharded production jits for the un-fused host kernels.  No static
    # ``n_steps`` arg (the scan length is the closure-captured
    # ``rad_update_steps``), donate the carry like the fused kernels.  Only
    # built when ``_unfused_available`` (else the attributes stay None).
    if _unfused_available:
        @partial(jax.jit, donate_argnums=(0,))
        def _run_norad_scan_jit(carry: SegmentCarry,
                                forcing: SegmentForcing) -> SegmentCarry:
            return _run_norad_scan(carry, forcing)

        @partial(jax.jit, donate_argnums=(0,))
        def _run_rad_jit(carry: SegmentCarry,
                         forcing: SegmentForcing) -> SegmentCarry:
            return _run_rad(carry, forcing)

    def _use_subcycle(carry: SegmentCarry, n_steps: int) -> bool:
        """Decide whether the subcycled scan path is valid for this call.

        ONE condition now: the cond-free no-rad variant must exist.

        This used to also require ``n_steps % rad_update_steps == 0`` and an
        aligned ``carry.step_index``, because the outer body assumed a segment
        was a whole number of cycles starting on a cycle boundary.  Both
        restrictions are gone: ``_run_subcycled`` derives the phase from the
        absolute step index and emits the leading/trailing held-radiation steps
        explicitly, so it is exact for any ``(start_step, n_steps)``.

        Removing them is a FIX, not a relaxation.  Falling back to
        ``_run_single`` was never cadence-preserving: that body computes
        ``need_rad`` and then passes it to the ``static_need_rad=True``
        variant, which discards it and radiates every step.  A ragged final
        segment or a restart at a non-aligned step therefore ran a denser
        radiation cadence than configured, silently.
        """
        return _subcycle_available

    def _subcycle_lead(carry: SegmentCarry) -> int:
        """Held-radiation steps before this segment's FIRST radiation step.

        Radiation fires at absolute ``i`` with ``(i+1) % k == 0``, so from a
        segment starting at ``s`` the first such ``i`` is ``s + (-(s+1)) % k``.
        Returned as a STATIC Python int for ``_run_subcycled``'s scan lengths:
        ``carry.step_index`` is a ``jnp.int32`` scalar, and reading it with
        ``int(...)`` blocks until prior device work completes — fine once per
        Python segment call (this is the same blocking read ``_use_subcycle``
        performed before the phase-general rewrite), impossible inside the jit.
        """
        return (-(int(carry.step_index) + 1)) % rad_update_steps

    # ------------------------------------------------------------------
    # Optional single-process multi-GPU sharding (third replication
    # site: the moist/AMIP segment path).  Active only when a
    # ``device_config`` with a live mesh is supplied — see the
    # ``device_config`` parameter docstring.  The donating production
    # kernels AND the non-donating ``.raw`` variant get explicit
    # in/out shardings READ DIRECTLY OFF the input carry/forcing that
    # the driver already sharded (``ModelDriver`` runs ``shard_pytree``
    # / ``shard_state`` upstream, so every committed array leaf carries
    # its own ``NamedSharding``).  The jit then only has to *match* the
    # layout that already exists — it never re-derives a policy.
    #
    # Why not re-derive via ``create_output_shardings``: that classifier
    # keys on ``leaf.shape[0] == 6`` to decide face-shard-vs-replicate.
    # The moist/AMIP carry stores FLATTENED cell-packed fields (e.g. the
    # profile-prognostic ``conv_prog`` is ``[n_cells, nlev]`` =
    # ``[6*n*n, nlev]``, leading dim 6*n*n, not 6).  The classifier
    # falls through to replicated ``P()`` for those, but the upstream
    # sharder may have placed the cell axis on the ``"face"`` mesh axis
    # — so the derived ``in_sharding`` ``P()`` disagreed with the arg's
    # actual ``P("face")`` and ``jax.jit`` raised "Sharding passed to
    # jit does not match the sharding on the respective arg" (jobs
    # 8457808/8457809).  Reading the actual ``.sharding`` is correct for
    # BOTH layouts: the cubed-sphere ``(6, n, n, nlev)`` dycore state
    # (its leaves are ``P("face", ...)`` from ``shard_state``, which a
    # ``jax.jit`` ndim-normalised match accepts) AND the flat cell-packed
    # carry (whatever axis the driver sharded, we copy verbatim).
    #
    # ``out_shardings`` == the carry's INPUT shardings: the scan body
    # (``_run_single`` / ``_run_subcycled``) is a ``lax.scan`` whose
    # carry-in structure/shape MUST equal carry-out (each field is
    # ``_match_dtype(x_upd, carry.field)`` — identical shape, no resize
    # or repack), so the output layout is the input layout by
    # construction.
    #
    # Pinning is GATED on the call being a top-level all-concrete
    # dispatch.  Under an outer trace (``jax.grad`` on the ``.raw`` path,
    # or an enclosing ``jit``) the carry carries tracer leaves whose
    # abstract ``.sharding`` does NOT reflect the concrete array the
    # transform feeds the inner jit at execution time — pinning from it
    # raised "Sharding passed to jit does not match the sharding on the
    # respective arg" (CPU job 8458320).  So when any leaf is a tracer we
    # build a PLAIN jit (no in/out shardings) and let the outer trace
    # propagate layout; pinning is only for the concrete production /
    # bench dispatch that has no enclosing trace to do so.
    #
    # Wrappers are cached by (single/subcycled, donate, PIN, carry
    # treedef, forcing treedef, leaf-shape signatures, and — in the
    # pinned branch only — leaf-SHARDING signatures): each ``jax.jit``
    # object owns its compile cache, so rebuilding a wrapper per segment
    # would retrace/recompile on every call (CLAUDE.md closure/recompile
    # rules).  Treedefs are in the key because forcing/carry pytree
    # structure can vary across calls (optional ``sfc_*_override`` /
    # double-moment fields flip between None and array); leaf shapes
    # because XLA specialises on shape; the ``pin`` flag because a
    # traced (unpinned) call and a concrete (pinned) call are different
    # executables; and the per-leaf sharding signature (pinned branch)
    # because two concrete calls with identical structure+shapes but
    # DIFFERENT layouts must not share a wrapper.  Because every pinned
    # leaf sharding is on the SINGLE build-time mesh (identity-honoured
    # or replicated on it), the ``(axis_names, spec)`` signature uniquely
    # identifies the layout.
    # ------------------------------------------------------------------
    _sharding_active = (
        device_config is not None
        and getattr(device_config, "mesh", None) is not None
    )
    _sharded_jit_cache: dict = {}
    # The single build-time mesh.  ``_input_sharding`` honours a leaf's
    # committed sharding ONLY when it lives on EXACTLY this mesh object —
    # see that helper for why identity (not a device-set / equivalence
    # match) is the right test.
    _mesh = device_config.mesh if _sharding_active else None

    def _shape_signature(tree) -> tuple:
        return tuple(
            tuple(getattr(leaf, "shape", ()))
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    def _replicated_sharding():
        """``NamedSharding(mesh, P())`` for the configured mesh."""
        from jax.sharding import NamedSharding, PartitionSpec
        return NamedSharding(device_config.mesh, PartitionSpec())

    def _input_sharding(leaf):
        """In-sharding for one leaf: concrete ``NamedSharding`` for every
        array leaf, ``None`` only for non-array (structural) leaves.

        This mirrors ``create_output_shardings``' invariant — every array
        leaf gets a concrete sharding, ``None`` marks the structural
        non-array slots ``jax.jit`` skips — but instead of re-deriving the
        layout it COPIES the layout the upstream driver already committed:

        * a committed ``jax.Array`` whose ``.sharding`` is a
          ``NamedSharding`` on EXACTLY the build-time mesh object → that
          exact sharding (this is the crash fix: a flat ``[6*n*n, nlev]``
          carry leaf the driver placed on ``P("face")`` is matched
          verbatim, never re-classified to ``P()``);
        * any other array leaf — abstract tracers under an outer
          ``jax.grad`` (``tracer.sharding`` lives on an ``AbstractMesh``,
          not this concrete mesh), uncommitted / single-device arrays,
          arrays committed to a DIFFERENT mesh — gets THIS mesh's
          fully-replicated ``P()`` sharding.  Replicated is valid
          regardless of the input's prior placement, keeps the AD path
          working (grad-of-jit composes; donation, not jit, is the AD
          conflict), and preserves the ``device_config=None``
          byte-identical contract because that path never reaches here
          (``_sharding_active`` is ``False``);
        * non-array leaves (Python scalars, structural ``None`` optional
          fields) → ``None``.

        The honour test is mesh-object IDENTITY ONLY (codex r1/r2 major).
        Not a device-set or ``(axis_names, spec)`` equivalence match: two
        distinct ``Mesh`` objects can share a device set yet differ in
        device ordering / shape / axis names, so admitting a non-build-
        time mesh would (a) leak a foreign mesh's sharding into
        ``in_shardings``/``out_shardings`` and (b) let the
        ``(axis_names, spec)`` cache signature collide across genuinely
        different concrete layouts.  Identity guarantees every honoured
        sharding references the SINGLE build-time mesh, which is exactly
        what the driver produces (it shards with
        ``NamedSharding(device_config.mesh, ...)`` — the same object
        passed here) and what makes the signature sufficient.

        Reading ``.sharding`` off the INPUT is safe even when the wrapper
        donates argument 0: donation invalidates a buffer only when the
        compiled call CONSUMES it, strictly after this host-side metadata
        read.
        """
        if not isinstance(leaf, jax.Array):
            return None
        from jax.sharding import NamedSharding
        # ``.sharding`` is a concrete ``NamedSharding`` on committed
        # arrays; on a tracer it returns the aval's (abstract-mesh)
        # sharding.  Some tracer types could raise instead — treat any
        # failure as "no honourable layout" and fall through to
        # replicated (never crash the host-side derivation).
        try:
            sharding = getattr(leaf, "sharding", None)
        except Exception:
            sharding = None
        # Honour the committed layout ONLY when it lives on exactly this
        # mesh object.  A foreign / abstract mesh falls through to a
        # replicated sharding pinned on the build-time mesh.
        if isinstance(sharding, NamedSharding) and sharding.mesh is _mesh:
            return sharding
        # Every other array leaf is pinned replicated on the mesh — never
        # ``None`` (a ``None`` in an array-leaf slot would make ``jax.jit``
        # mismatch the in_shardings tree against the argument tree).
        return _replicated_sharding()

    def _sharding_tree(tree):
        # No ``is_leaf`` override: structural ``None`` optional fields
        # (warm-rain ``q_i`` etc.) stay ``None`` so the resulting pytree
        # structure matches exactly what ``jax.jit`` flattens the real
        # argument into (its default registry treats ``None`` as a
        # zero-leaf node, never a value).  ``_input_sharding`` runs on the
        # genuine array leaves and returns a concrete ``NamedSharding`` for
        # each — so the flattened in_shardings has one concrete entry per
        # array leaf, exactly as ``jax.jit`` requires.
        return jax.tree_util.tree_map(_input_sharding, tree)

    def _sharding_signature(tree) -> tuple:
        """Hashable per-leaf layout key for the (pinned) wrapper cache.

        Used ONLY for pinned (all-concrete) wrappers — the unpinned
        traced branch keys on ``()`` instead.  Flattened with the SAME
        (default) registry ``jax.jit`` uses to bind ``in_shardings`` to
        the argument, so it has one entry per real array leaf (structural
        ``None`` slots are skipped both here and by ``jax.jit``).  Each
        leaf's ``(mesh axis names, spec)`` pair hashes distinctly, so two
        concrete dispatches with identical treedefs+shapes but DIFFERENT
        layouts (e.g. a fully face-sharded carry vs one whose flat
        cell-packed ``conv_prog`` is replicated) get SEPARATE cache
        entries — they are different XLA executables.  Every pinned leaf
        sharding is on the single build-time mesh, so ``(axis_names,
        spec)`` uniquely identifies the layout.
        """
        return tuple(
            (tuple(s.mesh.axis_names), s.spec)
            for s in jax.tree_util.tree_leaves(_sharding_tree(tree))
        )

    def _tree_has_tracer(*trees) -> bool:
        """True if any leaf is a JAX tracer (i.e. we are under an outer
        trace such as ``jax.grad`` / an enclosing ``jit``).

        A tracer's ``.sharding`` reports its aval's (often abstract-mesh)
        sharding, which does NOT reflect the CONCRETE array the enclosing
        transform ultimately feeds the inner jit at execution time — so
        pinning ``in_shardings`` from a tracer leaf can disagree with the
        real runtime arg and raise "Sharding passed to jit does not match
        the sharding on the respective arg".  When any leaf is traced we
        therefore skip pinning entirely and let the outer trace propagate
        layout (XLA already sees the enclosing shardings).
        """
        import jax.core as _jax_core
        for tree in trees:
            for leaf in jax.tree_util.tree_leaves(tree):
                if isinstance(leaf, _jax_core.Tracer):
                    return True
        return False

    def _get_sharded_jit(kind: str, donate: bool, carry, forcing):
        """Return the cached sharded JIT wrapper for this call signature."""
        # Pin explicit shardings ONLY for a top-level (all-concrete)
        # dispatch — production / bench call ``run_segment`` with a
        # committed sharded carry and no outer trace, so XLA needs the
        # explicit in/out shardings to avoid silently replicating the
        # whole-globe compute.  Under an outer ``jax.grad`` (the ``.raw``
        # training path) the carry carries tracer leaves; pinning is then
        # both unnecessary (the outer trace propagates layout) and unsafe
        # (a tracer's abstract sharding != the concrete runtime arg) — so
        # we build a plain jit and let inference do the work.
        pin = not _tree_has_tracer(carry, forcing)
        key = (
            kind,
            donate,
            pin,
            jax.tree_util.tree_structure(carry),
            jax.tree_util.tree_structure(forcing),
            _shape_signature(carry),
            _shape_signature(forcing),
            # Layout signature only distinguishes wrappers in the pinned
            # branch; in the unpinned branch every concrete layout shares
            # one inferred wrapper (``()`` keeps the key well-formed).
            _sharding_signature(carry) if pin else (),
            _sharding_signature(forcing) if pin else (),
        )
        fn = _sharded_jit_cache.get(key)
        if fn is None:
            # The "norad_scan" / "rad" host kinds (un-fused radiation) take
            # ``(carry, forcing)`` with NO static ``n_steps`` arg, so they
            # skip ``static_argnums``; the legacy "single" / "subcycled"
            # scan kinds take ``(carry, n_steps_static, forcing)``.  Either
            # way the DYNAMIC args are exactly ``(carry, forcing)``, so the
            # 2-tuple in/out shardings below bind identically.
            _targets = {
                "subcycled": _run_subcycled,
                "single": _run_single,
                "norad_scan": _run_norad_scan,
                "rad": _run_rad,
            }
            target = _targets[kind]
            _has_static_nsteps = kind in ("single", "subcycled")
            # "subcycled" also takes a 4th STATIC arg, the radiation-phase
            # ``lead`` (``_subcycle_lead``); it sets scan lengths, so it must
            # be static, and it stays out of the dynamic-arg tree the
            # ``in_shardings`` 2-tuple below binds against.
            _static_argnums = ((1, 3) if kind == "subcycled"
                               else (1,) if _has_static_nsteps else None)
            jit_kwargs: dict = (
                dict(static_argnums=_static_argnums)
                if _static_argnums is not None else {}
            )
            if pin:
                # NOTE: with ``static_argnums`` JAX matches
                # ``in_shardings`` against the tree of DYNAMIC args
                # only (jax 0.9.x ``pjit._process_in_axis_resources``
                # uses ``tree_without_statics``) — hence a 2-tuple for
                # the ``(carry, n_steps_static, forcing)`` signature.
                # A 3-tuple with a placeholder for ``n_steps`` raises
                # "wrong length ... for an args tuple of length 2".  The
                # ``(carry, forcing)`` host kinds already have exactly
                # those two dynamic args, so the SAME 2-tuple applies.
                # Every array leaf carries a concrete ``NamedSharding``
                # (its committed layout if on this mesh, else replicated
                # ``P()``); structural ``None`` marks only the non-array
                # slots ``jax.jit`` skips — the same invariant
                # ``create_output_shardings`` produced, so the
                # in_shardings tree binds 1:1 to the argument leaves.
                # ``out_shardings`` == the carry's input layout because
                # the scan/rad body preserves carry shape (no resize/
                # repack — ``run_rad`` writes the held fields + T_land
                # in place at owned indices).
                jit_kwargs["in_shardings"] = (
                    _sharding_tree(carry), _sharding_tree(forcing),
                )
                jit_kwargs["out_shardings"] = _sharding_tree(carry)
            if donate:
                jit_kwargs["donate_argnums"] = (0,)
            fn = jax.jit(target, **jit_kwargs)
            _sharded_jit_cache[key] = fn
        return fn

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
        if _use_subcycle(carry, n_steps):
            # STATIC phase, read once here (never under the jit) — see
            # ``_subcycle_lead``.
            lead = _subcycle_lead(carry)
            if _sharding_active:
                return _get_sharded_jit("subcycled", True, carry, forcing)(
                    carry, n_steps, forcing, lead,
                )
            return _run_subcycled_jit(carry, n_steps, forcing, lead)
        if _sharding_active:
            return _get_sharded_jit("single", True, carry, forcing)(
                carry, n_steps, forcing,
            )
        return _run_single_jit(carry, n_steps, forcing)

    def run_segment(carry: SegmentCarry, n_steps: int,
                    forcing: SegmentForcing) -> SegmentCarry:
        """Non-JIT version for use inside jax.grad / eqx.filter_value_and_grad.

        Same as run_segment_jit but without JIT wrapping or buffer
        donation, which conflict with outer AD transforms. The outer
        grad call handles compilation.

        Always routes through ``_run_single`` — the subcycle dispatch
        in :func:`_use_subcycle` reads ``int(carry.step_index)``, which
        is not traceable when ``carry.step_index`` is a JAX tracer
        (the AD entry path).  The legacy single-scan body remains
        bit-equivalent to the subcycled path; only the XLA compile-
        time scaling differs.  The AD pipeline already pays a
        recompute / activation cost dominated by physics, so the
        cond-vs-subcycle JIT-time tradeoff is irrelevant here.

        Opt-in sharded mode (``device_config`` with a live mesh): this
        becomes a non-donating *sharded* JIT wrapper so single-process
        multi-GPU benchmark/production callers of ``.raw`` keep the
        carry face-sharded instead of compiling replicated compute.
        Still grad-safe (no donation), and still dispatchable from
        inside an outer trace (no ``int(step_index)`` host read on
        this path).  With ``device_config=None`` this body is exactly
        the legacy non-JIT call.
        """
        if _sharding_active:
            return _get_sharded_jit("single", False, carry, forcing)(
                carry, n_steps, forcing,
            )
        return _run_single(carry, n_steps, forcing)

    # Host-callable un-fused-radiation entry points (PRODUCTION
    # ``_run_compiled`` only; default OFF).  Each is its OWN compiled
    # executable: ``run_norad_scan`` is a pure no-rad dynamics+physics
    # scan (no rrtmgp in its HLO); ``run_rad`` is rrtmgp-only.  They route
    # through the SAME ``_get_sharded_jit`` machinery as the fused kernels
    # so SPMD in/out shardings match the upstream-sharded carry (reading
    # ``.sharding`` off the committed carry/forcing — never re-derived).
    # Attached only when ``_unfused_available`` so the legacy path (and
    # any caller that does not supply ``pipeline`` / ``step_unified_no_rad``
    # / ``rad_update_steps>1``) sees ``None`` and the byte-identical fused
    # ``run_segment`` is used instead.
    _run_norad_scan_public = None
    _run_rad_public = None
    if _unfused_available:
        def _run_norad_scan_public(carry: SegmentCarry,           # noqa: F811
                                   forcing: SegmentForcing) -> SegmentCarry:
            """Advance ``rad_update_steps`` no-rad steps (compiled, host)."""
            if _sharding_active:
                return _get_sharded_jit("norad_scan", True, carry, forcing)(
                    carry, forcing,
                )
            return _run_norad_scan_jit(carry, forcing)

        def _run_rad_public(carry: SegmentCarry,                  # noqa: F811
                            forcing: SegmentForcing) -> SegmentCarry:
            """Recompute fresh held radiation + T_land (compiled, host)."""
            if _sharding_active:
                return _get_sharded_jit("rad", True, carry, forcing)(
                    carry, forcing,
                )
            return _run_rad_jit(carry, forcing)

    # Attach both variants; default is the JIT version for inference.
    # The sharded-wrapper cache is exposed for tests/introspection
    # (e.g. asserting "no wrapper rebuild per segment").
    run_segment_jit.raw = run_segment
    run_segment_jit._sharded_jit_cache = _sharded_jit_cache
    # Un-fused-radiation host kernels (None unless built); the driver only
    # uses them when ``ExperimentConfig.unfused_radiation`` is set.
    run_segment_jit.run_norad_scan = _run_norad_scan_public
    run_segment_jit.run_rad = _run_rad_public
    return run_segment_jit


# Moisture tracers eligible for dycore advection, in carry-field order.
# q_v/q_c/q_r are always present on moist runs; the double-moment fields
# are None for warm-rain runs (their None-ness is static pytree structure,
# so the per-name `is not None` check below is trace-safe).
# Tracers carried through the resolved-wind advective step (#771).  ALL of
# them: every species here is now stored per unit MASS — mixing ratios [kg/kg]
# and all three numbers [#/kg] — so the mass-mixing-ratio operator is the right
# conservation law for each, and the ratio q/N that sets particle size is
# transport-invariant.
#
# N_c/N_r used to be excluded because they were stored per VOLUME [#/m^3], for
# which this operator is wrong.  Excluding them was not a fix: it left the mass
# advecting while the number stayed put, so the diagnosed particle size was
# wrong by O(1) once a cloud moved further than its own width — worse than the
# density-bounded error it avoided.  Storing them per mass is the density-aware
# transport that exclusion was waiting for (2026-08-14).
_ADVECTED_TRACER_NAMES = (
    "q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i",
)


def _rebuild_state(carry: SegmentCarry, model, advect_moisture: bool = False,
                   persistent_dgrid: bool = False):
    """Rebuild the model's expected state type from raw carry arrays.

    The dynamics model expects a NamedTuple with Field-wrapped arrays.
    Inside lax.scan we store raw arrays, so we reconstruct the state
    type here.  This is cheap — only Python object creation, no data copy.

    With ``advect_moisture=True`` (issue #771) the moisture tracers ride
    the state as ``state.tracers`` so the primitive-equation step advects
    them with the resolved wind (mass-consistently, via the shared
    cubed-sphere tracer kernel — the same MPAS Phase B design).  Default
    ``False`` keeps the legacy column-locked moisture byte-identical.
    """
    # Detect state type from model
    if hasattr(model, '_state_type'):
        StateType = model._state_type
    else:
        # Fallback: use HydrostaticState (the most common case)
        from legoesm.core.state import HydrostaticState
        StateType = HydrostaticState

    # #1028: with persistent D-grid winds the carry holds CORNER arrays
    # (6, n+1, n+1, nlev) and the state that owns them is FV3HydrostaticState,
    # whose wind leaves are named u_d/v_d.  The caller decides -- the model's
    # ``_state_type`` is the cell-centre state on this dycore either way, so
    # keying off it would silently rebuild a cell-centre state around corner
    # arrays (found by the #1028 gate).
    if persistent_dgrid:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            FV3HydrostaticState,
        )
        StateType = FV3HydrostaticState
    _wname, _vname = ("u_d", "v_d") if persistent_dgrid else ("u", "v")
    # Build Fields with minimal metadata
    u_f = Field(carry.u, name=_wname, dims=("face", "x", "y", "level"), units="m/s")
    v_f = Field(carry.v, name=_vname, dims=("face", "x", "y", "level"), units="m/s")
    T_f = Field(carry.T, name="T", dims=("face", "x", "y", "level"), units="K")
    p_s_f = Field(carry.p_s, name="p_s", dims=("face", "x", "y"), units="Pa")
    phis_f = Field(carry.phis, name="phis", dims=("face", "x", "y"), units="m2/s2")

    if advect_moisture:
        # Only the tracer-capable dycore states reach this branch (the
        # driver gates on cubed_sphere+cdgrid); a state type without a
        # ``tracers`` field fails loudly here rather than silently
        # dropping the moisture.
        _units = {t.name: t.units for t in make_full_moisture_registry().tracers}
        tracers = {
            nm: Field(getattr(carry, nm), name=nm,
                      dims=("face", "x", "y", "level"),
                      units=_units.get(nm, "kg/kg"))
            for nm in _ADVECTED_TRACER_NAMES
            if getattr(carry, nm) is not None
        }
        return StateType(**{_wname: u_f, _vname: v_f},
                         T=T_f, p_s=p_s_f, phis=phis_f,
                         tracers=tracers)

    return StateType(**{_wname: u_f, _vname: v_f},
                     T=T_f, p_s=p_s_f, phis=phis_f)
