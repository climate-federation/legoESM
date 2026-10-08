"""Conservation fixers for legoESM.

These enforce hard constraints on conserved quantities (mass, energy, momentum)
after each timestep. The fixers use uniform additive corrections to preserve
gradients for automatic differentiation.

References
----------
- Sha et al. (2025): Global mass and energy conservation schemes for AI weather models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

def _tiny(x=None):
    """Smallest normal float for the given array's dtype (or active accumulate dtype)."""
    if x is not None and hasattr(x, 'dtype'):
        return float(jnp.finfo(x.dtype).tiny)
    return float(jnp.finfo(resolve_dtype(None, "accumulate")).tiny)
# Epsilon for energy fixers: prevents sqrt(0) which has infinite gradient,
# causing 0*Inf=NaN in the backward pass when jnp.maximum clamps KE_target to 0.
_EPS_ENERGY = 1e-20

from legoesm import constants
from legoesm.core.operators import global_integral, is_distributed
from legoesm.core.operators_voronoi import kinetic_energy_cell
from legoesm.core.precision import resolve_dtype, get_policy
from legoesm.core.state import ShallowWaterState, HydrostaticState
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import compute_geopotential

# legoesm.parallel.reductions (batch_allreduce_mpi, global_sum_mpi) and
# legoesm.runtime.backend (is_x64_enabled, supports_float64) are imported at
# function scope below: core/ must not import parallel/ or runtime/ at module
# top level (re-enters their __init__ mid-load, breaks isolated pytest; CLAUDE.md).


def energy_consistent_moisture_floor(q_v_raw, T):
    """Floor ``q_v`` at zero while conserving moist static energy.

    Issue #323.  The per-step physics tracer update
    ``q_v_raw = q_v + dt * dq_v_dt`` can go negative when the combined
    convective (SBM) + microphysical (Kessler) vapour sink exceeds the
    available vapour in a column.  A plain ``max(q_v_raw, 0)`` floor
    truncates the vapour sink but leaves the matching condensation latent
    heat already added to ``T`` in full, injecting spurious heat that
    accumulates under organised convection and drives the wind blow-up.

    This floors ``q_v`` AND removes the latent heat tied to the clipped
    (un-removed) vapour, so the floor is moist-static-energy neutral::

        deficit = max(-q_v_raw, 0)            # vapour the sink could not remove
        q_v_out = q_v_raw + deficit = max(q_v_raw, 0)
        T_out   = T - (L_v / c_pd) * deficit

    Hence ``c_pd*T_out + L_v*q_v_out == c_pd*T + L_v*q_v_raw`` to roundoff
    (the map is piecewise-linear in ``q_v_raw``), whereas the plain floor
    leaves a residual ``+L_v*deficit`` of spurious energy.

    Scope / limitations (deliberate; this targets the *energy* blow-up).
    ------------------------------------------------------------------
    * **Conserves moist static energy, NOT total water.**  It does not touch
      ``q_c``/``q_r``: if the over-removed vapour had been routed to condensate
      (``dq_c > 0``), that condensate is still created, so column total water
      ``q_v + q_c + q_r`` rises by ``deficit``.  Removing the spurious *heat*
      is what stops the wind blow-up; full total-water closure needs a
      per-scheme limiter that caps each scheme's vapour sink AND its matching
      condensate/rain source together (a larger change, see Issue #323).
    * **First-order latent-heat attribution.**  ``deficit`` is taken from the
      *summed* ``dq_v_dt``, which also carries non-condensational sinks
      (turbulent mixing, surface evaporation) that release no latent heat.
      Attributing the whole overshoot to condensation slightly over-cools
      when those are present.  This is a good approximation for the
      condensation-dominated drying it is meant for (kessler+sbm) and is why
      it is opt-in (default off) rather than always on.

    Pure ``jnp``; differentiable (subgradient at the ``q_v_raw = 0`` kink).
    Returns ``(q_v_out, T_out)`` with the dtypes of the inputs preserved by
    the caller's downstream ``_match_dtype`` cast.
    """
    deficit = jnp.maximum(-q_v_raw, 0.0)
    q_v_out = q_v_raw + deficit
    T_out = T - (constants.L_v / constants.c_pd) * deficit  # latent-ok: atmosphere moist-enthalpy reference L (constant by convention; surface gap booked by surface_layer.latent_enthalpy_correction)
    return q_v_out, T_out


#: Tracers eligible for the column-conserving borrow: PER-MASS fields whose
#: dsigma-weighted column integral is what mass-weighted transport conserves —
#: the water mixing ratios [kg/kg] and ALL the numbers, which are stored
#: per MASS [#/kg] since 2026-08-14 (``N_c``/``N_r`` were per-volume before;
#: the exclusion that this list used to encode was the density-aware repair
#: they were waiting for).  The microphysics still works in per-volume
#: internally; the conversion lives at the physics bridge.
#:
#: Lives here, next to the clip it gates, rather than in the MPAS PE dycore:
#: the serial (atmosphere) and MPI (parallel) lanes both need it, and the
#: parallel one importing an atmosphere module broke the "legoesm-core member
#: imports nothing above it" contract.
BORROW_ELIGIBLE_TRACERS = frozenset(
    {"q_v", "q_c", "q_r", "q_i", "q_s", "q_g",
     "N_c", "N_r", "N_i", "N_s", "N_g"})


def is_borrow_eligible_tracer(name: str) -> bool:
    """True for a per-mass tracer, tolerating a ``trc_`` prefix."""
    return str(name).removeprefix("trc_") in BORROW_ELIGIBLE_TRACERS


#: Per-species latent-heat coefficient in the frozen moist static energy
#: ``h = c_pd*T + Phi + KE + L_v*q_v − L_f*q_frozen`` the energy tracker uses
#: (LIQUID reference: ``q_c``/``q_r`` carry zero; vapour +L_v; ice −L_f).  When
#: a HARD floor raises species X by ``deficit = max(-q_raw, 0)``, holding h
#: fixed needs ``dT = -(coef_X / c_pd) * deficit``:
#:   q_v          -> −L_v/c_pd * d   (cool: undo the phantom condensation heat)
#:   q_c, q_r     -> 0               (liquid is the reference phase)
#:   q_i,q_s,q_g  -> +L_f/c_pd * d   (warm: adding −L_f-mass needs +L_f heat)
#: Signs confirmed 2026-08-25 (codex + GLM), against this exact h.  This is the
#: HARD-floor certificate only; the BORROW is h-neutral for every species with
#: no T change (it conserves each column integral), so it needs none of this.
_FLOOR_LATENT_COEF = {
    "q_v": constants.L_v,  # latent-ok: atmosphere moist-enthalpy reference L (constant by convention; surface gap booked by surface_layer.latent_enthalpy_correction)
    "q_c": 0.0, "q_r": 0.0,
    "q_i": -constants.L_f, "q_s": -constants.L_f, "q_g": -constants.L_f,  # latent-ok: atmosphere moist-enthalpy reference L (constant by convention; surface gap booked by surface_layer.latent_enthalpy_correction)
}


def energy_consistent_water_floor(tracers, T):
    """Hard-floor every water tracer to zero, conserving frozen MSE.

    Generalises :func:`energy_consistent_moisture_floor` (vapour only) to the
    condensate species.  Each species is floored ``q_out = max(q_raw, 0)``; the
    per-cell temperature is corrected by ``dT = -(coef/c_pd)*deficit`` with
    ``coef`` from :data:`_FLOOR_LATENT_COEF`, so the model's frozen MSE
    ``c_pd*T + L_v*q_v − L_f*(q_i+q_s+q_g)`` is unchanged pointwise (vapour
    cools, ice warms, liquid unchanged).  Tracers with no coefficient (number
    concentrations ``N_*``) get a plain floor and no T term.

    NOTE (GLM 2026-08-25): this conserves ENERGY, not water — the floor still
    CREATES the clipped mass.  It is the last-resort path; the column-conserving
    BORROW (:func:`conservative_positive_clip_global`) is the primary one and
    conserves both.  Do not call this the "conservative" clip.

    Pure ``jnp``; differentiable (subgradient at the ``q_raw = 0`` kink).
    Returns ``(tracers_out, T_out)`` mirroring the input container types.
    """
    inv_cpd = 1.0 / constants.c_pd
    T_data = T.data if hasattr(T, "data") else T
    dT = jnp.zeros_like(T_data)
    out = {}
    for name, f in tracers.items():
        data = f.data if hasattr(f, "data") else f
        floored = jnp.maximum(data, 0.0)
        coef = _FLOOR_LATENT_COEF.get(str(name).removeprefix("trc_"))
        if coef:  # non-zero coefficient only (liquid/None -> no heat term)
            deficit = floored - data          # = max(-data, 0) >= 0
            dT = dT - (coef * inv_cpd) * deficit
        out[name] = f.replace(data=floored) if hasattr(f, "replace") else floored
    T_out = (T.replace(data=T_data + dT) if hasattr(T, "replace")
             else T_data + dT)
    return out, T_out


def apply_water_positivity(tracers, T, dp, *, conservative, energy_consistent,
                           area, sum_fn=None):
    """Single positivity stage for every atmospheric dycore (MPAS/cube/spectral/
    lat-lon), so the three grids stay bit-equivalent by construction.

    * ``conservative=True`` (DEFAULT everywhere): column-conserving BORROW for
      per-mass species (:func:`conservative_positive_clip_global` with the
      ``dp`` layer-mass weight), plain floor for anything not borrow-eligible.
      ``T`` is returned untouched — the borrow is frozen-MSE-neutral for every
      species (it preserves each column integral).  ``area`` is the
      horizontal cell area (``dp`` minus its trailing level axis); the global
      net-negative-column residual is conserved in ``sum(area*dp*q)``, the
      physical mass.  ``sum_fn`` defaults to
      serial ``jnp.sum``; the MPI lane passes an allreduce-SUM reduction so the
      redistribution factor is decomposition-independent.  Iterates SORTED so
      every rank issues the per-tracer collectives in the same order.
    * ``conservative=False`` + ``energy_consistent=True``: hard floor with the
      per-species latent-heat T correction (:func:`energy_consistent_water_floor`).
    * ``conservative=False`` + ``energy_consistent=False``: plain ``max(q,0)``.

    Returns ``(tracers_out, T_out)``.
    """
    if conservative:
        out = {}
        for name in sorted(tracers):
            f = tracers[name]
            data = f.data if hasattr(f, "data") else f
            if is_borrow_eligible_tracer(name):
                clipped = conservative_positive_clip_global(
                    data, dp, axis=-1, sum_fn=sum_fn, area=area)[0]
            else:
                clipped = jnp.maximum(data, 0.0)
            out[name] = (f.replace(data=clipped) if hasattr(f, "replace")
                         else clipped)
        return out, T
    if energy_consistent:
        return energy_consistent_water_floor(tracers, T)
    out = {
        name: (f.replace(data=jnp.maximum(f.data, 0.0))
               if hasattr(f, "replace")
               else jnp.maximum(f, 0.0))
        for name, f in tracers.items()
    }
    return out, T


def conservative_positive_clip(q, weight, axis=-1, eps=1e-30):
    """Clip ``q`` to zero WITHOUT creating mass: borrow the deficit back.

    A plain ``max(q, 0)`` on a tracer left negative by non-monotone transport
    deletes the negative values and thereby ADDS their magnitude as spurious
    mass — a continuous, compounding source.  This clips, then rescales the
    remaining POSITIVE values so the weighted integral along ``axis`` is
    unchanged (hole-filling / borrowing; standard for positive-definite-but-
    non-monotone scalar transport)::

        integral(q_out) == integral(q_in)      (to roundoff)

    Measured motivation (2026-07-26, MPAS AMIP century): the MPAS floors stage
    clamped every water tracer with a plain ``maximum(f, 0.0)`` while
    ``tracer_transport_mpas`` carries no limiter.  Horizontal advection alone
    created **+0.0822 kg/m2/day (+30 kg/m2/yr)** of water, 96% of it from the
    spiky condensate fields ``q_i``/``q_c``.  That drove column water 23->42
    kg/m2, collapsed OLR 199->109 W/m2 and warmed the atmosphere +10 K/yr.
    The LES lane already had this fixer
    (``spectral_les_moist.conserving_positive``); this is the shared form so
    the numerics are not re-derived per dycore.

    Borrowing is LOCAL to ``axis`` (per column when ``axis`` is the vertical),
    so no global reduction is needed and the result is identical serial,
    sharded and under MPI.  Pure ``jnp``; differentiable.

    Parameters
    ----------
    q : jax.Array
        Tracer field, may contain negatives.
    weight : jax.Array
        Per-element integration weight broadcast against ``q`` along ``axis``
        (e.g. ``dsigma``).  Factors common to the whole column (``p_s/g``)
        cancel in the ratio and may be omitted.
    axis : int
        Axis to conserve along (the vertical for a column fixer).
    eps : float
        Positive-mass floor below which a column is DEGENERATE: no rescale is
        attempted (nothing meaningful to borrow from).  The effective
        threshold is ``max(eps, sqrt(finfo(q.dtype).tiny))`` so that
        ``after**2`` in the quotient VJP can never underflow — with the raw
        1e-30 a float32 column just above threshold squares to ~1e-60 -> 0 and
        the backward pass emits Inf (codex 2026-07-26 finding 3).

    Contract
    --------
    * Inputs are assumed FINITE; NaN/Inf propagate (the floors stage runs
      before the driver's bounds guards, which are the NaN tripwire).
    * ``weight`` must be positive (true for every MPAS ``dsigma``, pure-sigma
      and hybrid).  With mixed-sign weights ``before > after`` is possible and
      the factor clip would silently under-restore — out of scope.
    * Conservation is exact (to roundoff) for columns with
      ``after > eps_eff``.  Degenerate columns: a net-POSITIVE one keeps its
      plain-clipped values (error bounded by ``eps_eff`` per column — for
      float32 ~1e-19 kg/kg, ~1e-14 kg/m2 column water, 15 orders below the
      +30 kg/m2/yr defect this fixes); a net-negative one is zeroed (the
      minimum-creation choice).

    Returns
    -------
    (q_out, created) : tuple
        ``q_out`` clipped and rescaled; ``created`` is the weighted mass the
        NAIVE clip WOULD have invented — a monotonicity diagnostic that is
        zero for a monotone scheme.
    """
    # Trailing axis only: the weight broadcast ``q * w`` aligns on the LAST
    # dimension, so a non-trailing ``axis`` would pair weights with the wrong
    # dimension and silently mis-conserve (codex 2026-07-26 finding 1).
    if axis != -1 and axis != q.ndim - 1:
        raise ValueError(
            f"conservative_positive_clip conserves along the TRAILING axis "
            f"only (weight broadcasts on the last dim); got axis={axis} for "
            f"ndim={q.ndim}. Move the conserved dim last.")
    w = jnp.asarray(weight, dtype=q.dtype)
    # Host-side math: dtype is static under jit, so eps_eff is a trace-time
    # Python float (jnp.sqrt here would make it a tracer and break jit).
    eps_eff = max(float(eps), float(jnp.finfo(q.dtype).tiny) ** 0.5)
    # float16's narrow exponent range makes sqrt(tiny) = 7.8e-3 — a LARGE
    # mixing ratio, so the degenerate keep-the-clip branch could invent up to
    # ~10 kg/m2 of column water per column (codex 2026-07-26 round 2).  bf16
    # shares float32's exponent range and is fine; refuse anything coarser.
    if eps_eff > 1e-6:
        raise ValueError(
            f"conservative_positive_clip: dtype {q.dtype} has "
            f"sqrt(finfo.tiny) = {eps_eff:.2e}, too coarse for mixing-ratio "
            "conservation (degenerate columns could create O(g/kg) mass). "
            "Use float32/bfloat16 or wider.")
    q_clip = jnp.maximum(q, 0.0)
    before = jnp.sum(q * w, axis=axis, keepdims=True)
    after = jnp.sum(q_clip * w, axis=axis, keepdims=True)
    created = jnp.sum(jnp.maximum(-q, 0.0) * w)
    # factor <= 1 removes the borrowed mass from the positives.  Degenerate
    # columns (after <= eps_eff): keep the plain clip when the integral is
    # positive (a tiny-but-real column must NOT be zeroed — conservation error
    # bounded by eps_eff), zero when it is not (nothing to borrow from; the
    # minimum-creation choice).  The inner ``where`` keeps the disabled
    # branch's denominator at 1.0 so reverse-mode never sees 0/0, and eps_eff
    # keeps ``after**2`` in the quotient VJP above the underflow floor.
    live = after > eps_eff
    safe_after = jnp.where(live, after, 1.0)
    factor = jnp.where(live, before / safe_after,
                       jnp.where(before > 0.0, 1.0, 0.0))
    return q_clip * jnp.clip(factor, 0.0, 1.0), created


def conservative_positive_clip_global(q, weight, axis=-1, eps=1e-30,
                                      sum_fn=None, area=None):
    """Column-local borrow PLUS global residual redistribution.

    :func:`conservative_positive_clip` zeroes a net-negative column (nothing
    to borrow from locally) — the minimum-creation choice PER COLUMN, but
    still creation.  On smooth water mixing ratios that case is rare and
    tiny; on spiky per-mass NUMBER fields it is not: at century4 day 90 one
    advection step left 868/10242 columns net-negative in ``N_i``, and the
    zeroing alone re-created **x2.74/day** exponential field growth (the
    residual engine behind the day-803 N_i=1e193 overflow, measured
    2026-07-28).  This wrapper removes the invented residual proportionally
    from every positive cell so the ``sum_fn``-total is conserved exactly::

        sum(q_out * w) == sum(q_in * w)     whenever sum(q_in * w) >= 0

    (a globally net-negative field still floors at zero — nothing exists to
    borrow anywhere).  ``sum_fn`` defaults to ``jnp.sum`` (serial); the MPI
    lane passes an allreduce-SUM-based reduction so the redistribution
    factor is identical on every rank (decomposition-independent, and
    allreduce-SUM is the one AD-safe collective).  AD: one extra guarded
    quotient, same double-``where`` pattern as the column fixer.

    ``area`` (horizontal cell area, ``q``'s shape minus the trailing axis)
    weights the global sums so the conserved total is the physical mass on a
    non-equal-area grid; ``None`` means equal-area columns.
    """
    q_col, created = conservative_positive_clip(q, weight, axis=axis, eps=eps)
    w = jnp.asarray(weight, dtype=q.dtype)
    if area is not None:
        w = w * jnp.asarray(area, dtype=q.dtype)[..., None]
    s = sum_fn if sum_fn is not None else jnp.sum
    eps_eff = max(float(eps), float(jnp.finfo(q.dtype).tiny) ** 0.5)
    # ONE reduction per quantity (two total): pos_total reused for the
    # residual so the two nominally-identical q_col sums cannot differ by
    # reduction roundoff, and the MPI closure issues exactly two allreduces
    # per tracer (codex 2026-07-28 global-residual review).
    pos_total = s(q_col * w)
    resid = jnp.maximum(pos_total - s(q * w), 0.0)
    live = pos_total > eps_eff
    safe_total = jnp.where(live, pos_total, 1.0)
    # Degenerate-but-positive global total: KEEP the column result (error
    # bounded by eps_eff, mirroring the column fixer's tiny-positive branch)
    # — zeroing a tiny trace field violated the conservation contract
    # (codex: float32 q=[[5e-20]] came back [[0.]]).  Zero only when the
    # global total is non-positive (nothing exists to borrow anywhere).
    factor = jnp.where(live, 1.0 - resid / safe_total,
                       jnp.where(pos_total > 0.0, 1.0, 0.0))
    return q_col * jnp.clip(factor, 0.0, 1.0), created


def _accumulation_dtype():
    """Return the dtype for accumulation in conservation fixers.

    Uses the global precision policy's ``accumulate`` role when available,
    clamped to what the backend actually supports. On backends that lack
    float64 (e.g. Apple Metal) or when JAX x64 mode is disabled, the
    result is clamped to float32 even if the policy requests float64.
    """
    from legoesm.runtime.backend import is_x64_enabled, supports_float64
    target = get_policy().accumulate
    if target == jnp.float64 and not (supports_float64() and is_x64_enabled()):
        return jnp.float32
    return target


def conservation_accumulator():
    """Accumulator dtype for *budget* sums (mass, energy, tracer).

    Distinct from :func:`_accumulation_dtype` — promotes to ``float64``
    whenever JAX has x64 enabled, regardless of the active precision
    policy.  Mass/energy budgets involve subtracting two near-equal
    extensive quantities (e.g. ``mass_old - mass_new``), so even when
    storage/compute are intentionally float32 we want the budget sum
    to use the highest precision JAX is willing to give us.  This
    restores end-step :func:`fix_mass_hydrostatic` to ~machine
    precision and lets the "skip per-stage ``zero_mean_tendency`` when
    end-step fixer is on" scaling optimisation be lossless even in
    float32 storage/compute mode.

    Returns f64 whenever x64 is enabled — a higher dtype than fp32 storage. The
    mass fixers KEEP the correction in f64 and add it to ``p_s`` WITHOUT rounding
    back to storage (this exact-arithmetic correction is load-bearing: it is what
    delivers ~machine-precision mass fixing and the anchored ~1e-12 MPAS
    guarantee — rounding it to fp32 would let sub-ULP corrections vanish and
    destroy those). Consequences by mode (#1665):

    * **strict fp32** (x64 off): the f64 branch is inactive; the correction is
      float32; ``p_s`` stays fp32.
    * **fp64**: ``p_s`` is already f64; no promotion.
    * **mixed** (fp32 storage + x64): the f64 correction added to an fp32 ``p_s``
      promotes it to f64, and that promotion is now DELIBERATE and documented
      rather than accidental (#1675). ``p_s`` is one 2-D field, so keeping it at
      the accumulate dtype costs almost nothing in memory and buys the
      ~1e-12 global mass the exact correction is there for; the fp32 saving in
      ``mixed`` comes from the 3-D bulk state. What used to leak is the
      CONTAGION — the promoted ``p_s`` flowed into the tracer mass rescale and
      brought the 3-D tracers up with it — and that is closed at the step
      boundary by :func:`core.precision.finalize_to_storage`, which re-casts
      bulk state to storage while leaving ``p_s`` at accumulate.
    """
    if jax.config.read("jax_enable_x64"):
        return jnp.float64
    return _accumulation_dtype()


def cube_faces_are_whole_on_shards() -> bool:
    """True iff the active decomposition keeps each cube face WHOLE on a shard.

    The shard-invariant per-face reduction (:func:`shard_invariant_cube_face_sum`)
    is only bit-exact when no face is split across devices — i.e. single-device,
    serial, or a whole-face SPMD mesh (device count divides 6).  A ``(6, kt, kt)``
    SUB-FACE TILED mesh (``n_devices > 6``) shards the spatial axes too, so a
    per-face sum would cross tiles and the per-face partial would NOT be
    decomposition-invariant.  Read only Python module state (the halo backend +
    SPMD mesh shape) → a static trace-time branch, never traced control flow.
    """
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return True  # serial / single-device: whole faces
    mesh = get_spmd_mesh()
    if mesh is None:
        return True
    sh = tuple(mesh.devices.shape)
    is_tiled = len(sh) == 3 and sh[0] == 6 and sh[1] == sh[2] and sh[1] >= 2
    return not is_tiled


def shard_invariant_cube_face_sum(prod: jax.Array, reduce_axes=None) -> jax.Array:
    """Sum a cube ``(6, ...)`` array shard-count-invariantly (issue #852).

    A bare ``jnp.sum`` on a face-sharded array reduces each device's owned faces
    LOCALLY and then all-reduces the per-shard partials.  float32 addition is
    NON-ASSOCIATIVE, so that partitioned order differs from the single-device
    flat sum — and between device counts (1 vs 3-faces/shard vs 2-faces/shard) —
    by ~1 ulp.  A global conserved integral (mass, energy, tracer) that feeds a
    fixer's rescale then turns that ulp into a shard-count-DEPENDENT state
    correction which the chaotic flow amplifies (the symptom reported in #852,
    wrongly attributed there to the ppermute halo — the halo is bit-exact; this
    reduction is the real decomposition dependence).

    On a WHOLE-FACE decomposition each shard owns ``k = 6/n_devices`` whole
    faces, so reducing each face over ``reduce_axes`` first gives per-face
    partials that are BIT-IDENTICAL at every decomposition; combining the 6
    partials (axis 0) in a FIXED order (XLA does not reassociate float adds with
    fast-math off) yields a total identical for single-device and any 1/2/3/6-way
    face sharding.

    ``reduce_axes`` are the WITHIN-face axes to sum first (default: every axis
    but the leading face axis).  Pass a subset to keep a trailing axis, e.g. a
    batched stack ``(6, n, n, n_arrays)`` with ``reduce_axes=(1, 2)`` returns
    ``(n_arrays,)``.  The face axis (0) is always combined last in fixed order.

    Caller MUST gate on ``prod.shape[0] == 6`` AND
    :func:`cube_faces_are_whole_on_shards` (a tiled mesh splits faces and breaks
    the invariance).  AD-safe (a linear sum).
    """
    if reduce_axes is None:
        reduce_axes = tuple(range(1, prod.ndim))
    per_face = jnp.sum(prod, axis=reduce_axes)  # face axis 0 preserved
    total = per_face[0]
    for f in range(1, per_face.shape[0]):
        total = total + per_face[f]
    return total


def global_area_sum(
    array: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
    *,
    differentiable_broadcast: bool = False,
) -> jax.Array:
    """Area-weighted global sum of a raw array, distributed-aware.

    Works on any grid with a ``.area`` attribute (CubedSphereGrid,
    LatLonGrid, etc.).

    Parameters
    ----------
    array : jax.Array
        The field to integrate (e.g. surface pressure).
    grid : grid object
        Must have ``.area`` attribute.
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` boolean/float mask indicating which faces
        this rank owns.  Required for **replicated-dynamics MPI** where
        each rank holds full ``(6, n, n)`` data but only owned faces
        are authoritative.  Non-owned faces are zeroed before local
        summation; ``global_sum_mpi`` then combines owned portions.
        If ``None``, all faces are summed (single-rank or SPMD).
    differentiable_broadcast : bool, optional
        No longer changes anything: since #1814 both ``global_sum_mpi`` and
        :func:`legoesm.parallel.reductions.broadcast_allreduce_sum` allreduce
        the cotangent on the backward pass.  Kept for existing callers.

    Execution modes:

    - **Single device**: plain ``jnp.sum``.
    - **Multi-device SPMD** (NamedSharding): ``jnp.sum`` on a
      face-sharded array already produces the correct global sum --
      JAX/XLA automatically inserts an all-reduce when the reduction
      spans a sharded axis.  No explicit ``psum`` is needed.
    - **MPI distributed** (replicated dynamics): mask to owned faces,
      local sum, then ``allreduce(SUM)``.
    """
    acc = conservation_accumulator()
    prod = array.astype(acc) * grid.area.astype(acc)
    if owned_mask is not None:
        # Broadcast (n_faces,) → match prod shape: (6,) → (6,1,1,...)
        mask = owned_mask.astype(acc)
        while mask.ndim < prod.ndim:
            mask = mask[..., None]
        prod = prod * mask
    local_sum = jnp.sum(prod)
    # Lat-band SPMD (single-process shard_map): combine the band-local partial
    # across the "lat" axis BEFORE is_distributed() — under SPMD there is one
    # process (is_distributed() is False) yet each band holds only a partial
    # sum. Inert for serial/MPI/cube (returns None), so the default path is
    # byte-unchanged. Mirrors batch_global_area_sums (:241) so the SINGLE-array
    # fixers that reduce via global_area_sum (fix_ps_mass_target,
    # fix_moisture_hydrostatic) are lat-band-SPMD-correct too, not only the
    # batched callers.
    spmd_sums = _spmd_lat_psum_or_none([local_sum])
    if spmd_sums is not None:
        return spmd_sums[0]
    if is_distributed():
        from legoesm.parallel.reductions import (
            broadcast_allreduce_sum, global_sum_mpi,
        )
        if differentiable_broadcast:
            return broadcast_allreduce_sum(local_sum)
        return global_sum_mpi(local_sum)
    # Cube GSPMD / single-device: use the shard-count-invariant per-face
    # fixed-order reduction (issue #852) so a face-sharded mass integral is
    # bit-identical to single-device — a global conserved quantity must not
    # depend on the device count.  Gated on the cube face axis (leading dim 6)
    # AND a whole-face decomposition (a (6,kt,kt) tiled mesh splits faces, so
    # the per-face reduction would not be invariant — fall back to the plain
    # sum there).  lat-lon/lat-band (rows can split) and MPI keep the plain sum
    # handled above.
    if (prod.shape[0] == 6 and prod.ndim >= 3
            and cube_faces_are_whole_on_shards()):
        return shard_invariant_cube_face_sum(prod)
    return local_sum


def _spmd_lat_psum_or_none(local_sums: list[jax.Array]) -> list[jax.Array] | None:
    """If a lat-band SPMD halo backend is armed, sum ``local_sums`` across the
    ``"lat"`` device axis and return the result; otherwise ``None`` (the caller
    falls through to the MPI / serial logic).

    This is the single-controller (``shard_map``) cross-band reduction for the
    lat-lon band SPMD steps, where ``is_distributed()`` is False (one process)
    yet each band holds only a PARTIAL sum that must be combined across the
    ``"lat"`` axis. It mirrors, byte-for-byte in dispatch, the ocean barotropic
    PCG's ``barotropic_common._global_dot_batch`` (which already shipped this
    exact logic): route to ``jax.lax.psum`` (self-transposing => AD-safe) ONLY
    when the armed mesh is the lat-band one, keyed on the ``"lat"`` axis BY
    NAME so a coupled-run cube ``("face", ...)`` SPMD mesh falls through to the
    MPI/local path (cube fields are never lat-band-sharded). Inert for the
    serial and MPI backends (``get_halo_backend() != "spmd"``).
    """
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None:
        # Armed "spmd" with no mesh: an invalid state reachable only via a bare
        # set_halo_backend("spmd"). FAIL FAST rather than silently return
        # unreduced band-local partials inside a sharded step (matches the
        # ocean _global_dot_batch guard).
        raise RuntimeError(
            "_spmd_lat_psum_or_none: halo backend is 'spmd' but no SPMD mesh "
            "is set; arm it via activate_latlon_spmd_halo(mesh).")
    if "lat" in tuple(mesh.axis_names):
        from legoesm.parallel.reductions import batch_psum_spmd
        if ("lon" in tuple(mesh.axis_names)
                and int(mesh.shape["lon"]) > 1):
            # 2-D ("lat", "lon") tile mesh (M3a): every TILE holds a partial
            # sum — reduce across BOTH axes, or the "global" mass integral
            # would silently remain a per-lon-sector partial.  The 1-D band
            # mesh — and the degenerate (N, 1) tile mesh, whose lon rings
            # have one member — keep the bare "lat" psum (byte-unchanged /
            # structurally identical to the band program for the (N, 1)
            # bit-identity gate).
            return batch_psum_spmd(local_sums, ("lat", "lon"))
        return batch_psum_spmd(local_sums, "lat")
    if (tuple(mesh.axis_names) == ("face", "tile_i", "tile_j")
            and _TILED_REDUCTION_SCOPE):
        # Sub-face-tiled cube shard_map (the tiled operator-split lane):
        # each TILE holds a partial sum — combine across all three mesh
        # axes.  DOUBLE-gated (codex): the exact tiled axis tuple keeps a
        # face-only ``("face",)`` SPMD mesh on the jit-auto path, and the
        # explicit :func:`tiled_reduction_scope` context keeps a reduction
        # that merely RUNS while a tiled mesh is armed — but outside the
        # tiled shard_map body — from emitting an out-of-scope psum.
        from legoesm.parallel.reductions import batch_psum_spmd
        return batch_psum_spmd(local_sums, ("face", "tile_i", "tile_j"))
    return None


# Explicit opt-in scope for the tiled-mesh psum branch above: ONLY the tiled
# operator-split step's shard_map body runs with tile-partial sums; any other
# reduction (writers, diagnostics, another module) executing while the tiled
# mesh happens to be armed must fall through to the serial path.
_TILED_REDUCTION_SCOPE: list = []


class tiled_reduction_scope:
    """Context manager arming the tiled-mesh psum branch of
    :func:`_spmd_lat_psum_or_none` — enter ONLY around code that traces
    INSIDE a ``("face","tile_i","tile_j")`` shard_map body (the tiled
    operator-split step)."""

    def __enter__(self):
        _TILED_REDUCTION_SCOPE.append(True)
        return self

    def __exit__(self, *exc):
        _TILED_REDUCTION_SCOPE.pop()
        return False


def batch_global_area_sums(
    arrays: list[jax.Array],
    grid,
    owned_mask: jax.Array | None = None,
    *,
    differentiable_broadcast: bool = False,
) -> list[jax.Array]:
    """Compute multiple area-weighted global sums in a single MPI call.

    Same semantics as calling :func:`global_area_sum` on each array
    individually, but batches all reductions into one ``allreduce``
    when running under MPI, reducing latency from O(N) to O(1).

    Falls back to individual ``jnp.sum`` when not distributed.

    ``differentiable_broadcast``: no longer changes anything (see
    :func:`global_area_sum`; #1814).
    """
    acc = conservation_accumulator()
    area_acc = grid.area.astype(acc)
    weight = area_acc
    if owned_mask is not None:
        mask = owned_mask.astype(acc)
        while mask.ndim < area_acc.ndim:
            mask = mask[..., None]
        weight = area_acc * mask

    # All inputs share the same horizontal axes and weight ``weight``;
    # stack them along a new trailing axis and reduce once locally so
    # XLA fuses the N independent sum kernels into one.
    stacked = jnp.stack([arr.astype(acc) for arr in arrays], axis=-1)
    summed = jnp.sum(
        stacked * weight[..., None], axis=tuple(range(area_acc.ndim)),
    )
    local_sums = [summed[..., i] for i in range(len(arrays))]

    # Lat-band SPMD (single-process shard_map): combine the band-local partials
    # across the "lat" axis. Checked BEFORE is_distributed() because under SPMD
    # there is one process (is_distributed() is False) yet each band holds only
    # a partial sum. Inert for serial/MPI/cube (returns None).
    spmd_sums = _spmd_lat_psum_or_none(local_sums)
    if spmd_sums is not None:
        return spmd_sums

    if is_distributed():
        from legoesm.parallel.reductions import (
            batch_allreduce_mpi, broadcast_allreduce_sum,
        )
        if differentiable_broadcast:
            # One stacked broadcast-allreduce (allreduce fwd AND bwd) — same
            # single-message batching as batch_allreduce_mpi, but the correct
            # transpose for a reused/broadcast reduced value.
            reduced = broadcast_allreduce_sum(jnp.stack(local_sums, axis=0))
            return [reduced[i] for i in range(len(local_sums))]
        return batch_allreduce_mpi(local_sums, op="sum")
    # Cube whole-face GSPMD / single-device: shard-count-invariant per-array
    # reduction (issue #852), matching the single-array global_area_sum fix so
    # the DEFAULT (non-anchor) mass fixer — fix_ps_mass → batch_global_area_sums
    # — is decomposition-independent too.  ``area_acc.ndim == 3`` selects the
    # cube (6, n, n) grid; reduce each face over the spatial axes then combine
    # the 6 faces in fixed order (keeping the trailing per-array axis).
    if area_acc.ndim == 3 and cube_faces_are_whole_on_shards():
        inv = shard_invariant_cube_face_sum(
            stacked * weight[..., None], reduce_axes=(1, 2))
        return [inv[i] for i in range(len(arrays))]
    return local_sums


def global_face_sum_if_scattered(
    local_sum: jax.Array, area, *, differentiable_broadcast: bool = False
) -> jax.Array:
    """Allreduce a cubed-sphere per-face partial to the GLOBAL total, but ONLY
    when the 6 faces are genuinely SCATTERED across MPI ranks.

    A cube "global" reduction ``jnp.sum(area * field)`` over the face axis is the
    true whole-cube sum ONLY when this rank holds all 6 faces.  Under MPI
    face-scatter each rank's ``area`` is sliced to its OWNED faces (leading dim
    ``== n_local < 6``), so the sum is an owned-face PARTIAL that must be
    ``allreduce(SUM)``-combined across ranks.  ``global_sum_mpi`` is the ONLY
    AD-safe reduction (MAX/MIN have no meaningful gradient — repo MPI-AD
    doctrine), so this is safe inside ``jax.grad``.

    Identity (byte-unchanged) for:

    * single-rank / serial (``local`` halo backend);
    * REPLICATED cube MPI — every rank keeps the full ``(6, ...)`` state, so the
      local sum already IS the global sum (keyed off ``area.shape[0] == 6``);
    * SPMD — a top-level ``jnp.sum`` over a sharded face axis is auto-reduced by
      GSPMD (shard_map cube face-sharding is out of scope / fail-closed upstream);
    * lat-lon band MPI — the topology carries no ``local_face_ids``.

    ``local_sum`` may be any shape (scalar, ``(nlev,)``, ``(nlev, ntr)``, ...);
    ``allreduce(SUM)`` combines it element-wise.  The scatter DECISION is keyed on
    ``area`` (the sliced cube area), not on ``local_sum`` — so a numerator and a
    denominator reduced through this one predicate always agree on WHEN to reduce.
    Shared gate behind :func:`_total_area` (the mass-fixer denominator) and the
    cube flux-form moisture substep's mass reductions (#811 / #771 follow-up).

    ``differentiable_broadcast``: no longer changes anything (see
    :func:`global_area_sum`; #1814).
    """
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology

    if get_halo_backend() == "mpi":
        topo = get_mpi_topology()
        # Only the cubed-sphere face-only topology carries ``local_face_ids``;
        # lat-lon band MPI exposes a ``LatLonBandLayout`` through the same
        # accessor (no ``local_face_ids``), so guard with hasattr.  The chained
        # comparison is a static (Python-int) trace-time decision — not traced
        # control flow.
        if (topo is not None and area is not None
                and hasattr(topo, "local_face_ids")
                and area.shape[0] == len(topo.local_face_ids) < 6):
            from legoesm.parallel.reductions import (
                broadcast_allreduce_sum, global_sum_mpi,
            )
            if differentiable_broadcast:
                return broadcast_allreduce_sum(local_sum)
            return global_sum_mpi(local_sum)
    return local_sum


def _total_area(grid) -> jax.Array:
    """Total area for any grid, promoted to the fp64 budget accumulator.

    iter-13: ``grid.grid_total_area`` is computed as
    ``jnp.sum(self.area)`` on grid classes that store ``area`` in the
    storage dtype (fp32 by default).  That reduction leaks ~N·eps into
    the divisor of every mass-fixer correction — visible as a residual
    ~10^-13 drift on the cube hydro PE even with the anchored
    fixer + fp64 ``mass_target``.  Cast to the conservation
    accumulator here so every fixer division sees an fp64 denominator.

    **Cubed-sphere face-scatter (MPI):** when each rank owns only a
    SUBSET of the 6 faces (``grid`` sliced to owned faces, leading dim
    ``< 6``), ``grid.grid_total_area`` is the rank-LOCAL owned-face area,
    not the global total.  The mass-fixer numerator
    (``global_area_sum``) already allreduces owned partials to the global
    mass, so an un-reduced local denominator would scale the correction
    by ``global_area / owned_area`` (~``n_ranks``) and break conservation
    + replicated-vs-scattered equivalence.  Detect the scattered case
    (leading dim equals this rank's owned-face count AND ``< 6``) and
    ``allreduce(SUM)`` the local area to the global total.  Replicated
    dynamics (full ``(6, ...)`` on every rank) and single-rank runs
    already hold the global area, so they are left byte-identical.
    """
    acc = conservation_accumulator()
    local = grid.grid_total_area.astype(acc)
    return global_face_sum_if_scattered(local, getattr(grid, "area", None))


def fix_mass_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
) -> ShallowWaterState:
    """Fix mass conservation for shallow water equations (any grid).

    Applies a uniform additive correction to the fluid depth h
    so that the global integral of h is preserved exactly.

    The uniform correction preserves gradients of h (important
    for differentiability) while enforcing exact conservation.

    Parameters
    ----------
    state_new : ShallowWaterState
        State after time integration (may not conserve mass exactly).
    state_old : ShallowWaterState
        State before time integration (reference mass).
    grid : CubedSphereGrid or LatLonGrid
        The grid (any grid with .area and .total_area).

    Returns
    -------
    ShallowWaterState : Mass-conserving state.
    """
    mass_old, mass_new = batch_global_area_sums(
        [state_old.h.data, state_new.h.data], grid,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    h_fixed = state_new.h.replace(data=state_new.h.data + correction)

    return state_new._replace(h=h_fixed)


def fix_energy_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
    g: float = constants.g,
) -> ShallowWaterState:
    """Fix total energy conservation for shallow water equations (any grid).

    Total energy = kinetic + potential:
        E = 0.5 * h * (u^2 + v^2) + 0.5 * g * (h + h_s)^2

    Applies a uniform scaling to velocities to restore total energy.

    Parameters
    ----------
    state_new : ShallowWaterState
        State after time integration.
    state_old : ShallowWaterState
        State before time integration.
    grid : CubedSphereGrid or LatLonGrid
        The grid (any grid with .area).
    g : float
        Gravitational acceleration.

    Returns
    -------
    ShallowWaterState : Energy-conserving state.
    """
    # Compute all three energy integrals as local sums, then batch
    # into a single MPI allreduce (3 separate allreduces -> 1).
    # iter-42: promote the per-cell energy *field* computation to the
    # fp64 budget accumulator before the area-weighted sum.  The pre-
    # iter-42 path computed ``0.5 * h * (u² + v²)`` in fp32 (input
    # storage dtype), so the square+multiply lost ~7 bits of precision
    # before ``batch_global_area_sums`` ever cast to fp64 — the same
    # fp32-field bug iter-1/4/5 fixed for the mass diagnostic, just on
    # the energy path.
    acc = conservation_accumulator()
    h_old = state_old.h.data.astype(acc)
    u_old = state_old.u.data.astype(acc)
    v_old = state_old.v.data.astype(acc)
    h_s_old = state_old.h_s.data.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)
    E_old_field = 0.5 * h_old * (u_old**2 + v_old**2) + 0.5 * g_acc * (h_old + h_s_old)**2

    h_new = state_new.h.data.astype(acc)
    u_new = state_new.u.data.astype(acc)
    v_new = state_new.v.data.astype(acc)
    h_s_new = state_new.h_s.data.astype(acc)
    KE_new_field = 0.5 * h_new * (u_new**2 + v_new**2)
    PE_new_field = 0.5 * g_acc * (h_new + h_s_new)**2

    E_old, KE_new, PE_new = batch_global_area_sums(
        [E_old_field, KE_new_field, PE_new_field], grid,
    )

    KE_target = E_old - PE_new
    KE_target = jnp.maximum(KE_target, _EPS_ENERGY)
    scale = jnp.where(KE_new > _tiny(KE_new), jnp.sqrt(KE_target / KE_new), 1.0)

    u_fixed = state_new.u.replace(data=u_new * scale)
    v_fixed = state_new.v.replace(data=v_new * scale)

    return state_new._replace(u=u_fixed, v=v_fixed)


def apply_conservation_fixer(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
    fix_mass: bool = True,
    fix_energy: bool = True,
    g: float = constants.g,
) -> ShallowWaterState:
    """Apply all conservation fixers in sequence (any grid).

    Order: mass first, then energy (energy fix adjusts velocities
    without changing mass).
    """
    if fix_mass:
        state_new = fix_mass_shallow_water(state_new, state_old, grid)
    if fix_energy:
        state_new = fix_energy_shallow_water(state_new, state_old, grid, g)
    return state_new


def fix_mass_hydrostatic(
    state_new: HydrostaticState,
    state_old: HydrostaticState,
    grid: CubedSphereGrid,
) -> HydrostaticState:
    """Fix mass conservation for the hydrostatic primitive equations.

    Applies a uniform additive correction to surface pressure p_s
    so that the global dry air mass is preserved exactly.

    Global dry air mass: M = (1/g) * ∫ p_s * dA

    Since g and dA are constant, we just need ∫ p_s * dA to be conserved.

    The uniform correction preserves gradients of p_s (important
    for differentiability) while enforcing exact conservation.

    Parameters
    ----------
    state_new : HydrostaticState
        State after time integration.
    state_old : HydrostaticState
        State before time integration (reference mass).
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    mass_old, mass_new = batch_global_area_sums(
        [state_old.p_s.data, state_new.p_s.data], grid,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)

    return state_new._replace(p_s=p_s_fixed)


def zero_mean_tendency(
    tendency: jax.Array,
    grid,
) -> jax.Array:
    """Remove the area-weighted global mean from a tendency field (any grid).

    After this correction, ``sum(tendency * area) == 0`` to machine
    precision, enforcing exact conservation for explicit FV mass equations.

    Works for 2D, 3D, and 4D arrays. The area dimensions are inferred
    from ``grid.area.ndim``:
    - Cubed-sphere: area is (6,n,n), tendency is (6,n,n) or (6,n,n,nlev).
    - Lat-lon: area is (n_lat,n_lon), tendency is (n_lat,n_lon) or (n_lat,n_lon,nlev).

    For arrays with a trailing level axis, the correction is applied
    independently at each level.

    Parameters
    ----------
    tendency : jax.Array
        The mass/tracer tendency field.
    grid : CubedSphereGrid or LatLonGrid
        Grid with cell areas.

    Returns
    -------
    jax.Array : Corrected tendency with zero global integral.
    """
    acc = conservation_accumulator()
    area = grid.area
    area_acc = area.astype(acc)
    total_area_acc = jnp.sum(area_acc)
    orig_dtype = tendency.dtype
    area_ndim = area.ndim  # 3 for cubed-sphere, 2 for lat-lon

    if tendency.ndim in (area_ndim, area_ndim + 1):
        # Rank/band-partial grids (lat-band MPI, lat-band SPMD, replicated
        # cube MPI): reduce the area exactly like the numerator, in one
        # collective, or the global integral is divided by this rank's own
        # area.  ``None`` = this process holds the whole domain.
        tend_acc = tendency.astype(acc)
        if tendency.ndim == area_ndim:
            local_num = jnp.sum(tend_acc * area_acc)[None]
        else:
            local_num = jnp.sum(tend_acc * area_acc[..., None],
                                axis=tuple(range(area_ndim)))
        reduced = _reduce_rank_partials(
            jnp.concatenate([local_num, total_area_acc[None]]))
        if reduced is not None:
            corrections = reduced[:-1] / reduced[-1]
            if tendency.ndim == area_ndim:
                corrections = corrections[0]
            return (tend_acc - corrections).astype(orig_dtype)

    if tendency.ndim == area_ndim:
        # 2D tendency (lat-lon) or 3D tendency (cubed-sphere) — no level axis
        global_sum = global_area_sum(tendency, grid)
        correction = global_sum / total_area_acc
        return (tendency.astype(acc) - correction).astype(orig_dtype)
    elif tendency.ndim == area_ndim + 1:
        # Has a trailing level axis — per-level correction
        tend_acc = tendency.astype(acc)
        prod = tend_acc * area_acc[..., None]
        # Sum over all spatial axes (all except the last)
        spatial_axes = tuple(range(area_ndim))
        level_sums = jnp.sum(prod, axis=spatial_axes)  # (nlev,)
        corrections = level_sums / total_area_acc  # (nlev,)
        # Broadcast corrections to match tendency shape
        for _ in range(area_ndim):
            corrections = jnp.expand_dims(corrections, 0)
        return (tend_acc - corrections).astype(orig_dtype)
    else:
        return tendency


def _reduce_rank_partials(local: jax.Array) -> jax.Array | None:
    """Sum rank/band partials with the same dispatch as :func:`global_area_sum`;
    ``None`` when this process already holds the whole domain (serial, GSPMD).

    ``broadcast_allreduce_sum`` (allreduce VJP): the reduced value is a shared
    correction reused on every rank, so its cotangent must be summed too.
    """
    spmd = _spmd_lat_psum_or_none([local])
    if spmd is not None:
        return spmd[0]
    if is_distributed():
        from legoesm.parallel.reductions import broadcast_allreduce_sum
        return broadcast_allreduce_sum(local)
    return None


# ==============================================================================
# Moisture conservation
# ==============================================================================

def compute_global_moisture(
    q_v: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute global column-integrated water vapor.

    Integral: (1/g) * ∫∫ q_v · p_s · dσ · dA

    Parameters
    ----------
    q_v : jax.Array, shape (..., nlev)
        Specific humidity [kg/kg].
    p_s : jax.Array, shape (...)
        Surface pressure [Pa].
    dsigma : jax.Array, shape (nlev,)
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
        Grid with ``.area`` attribute.
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics.

    Returns
    -------
    jax.Array : Scalar global moisture integral [kg].
    """
    # Column water vapor: ∫ q_v dp/g = q_v * p_s * dsigma / g
    # iter-44: promote field computation to fp64 budget accumulator
    # (same fp32-field bug as iter-42/43 energy diagnostics).
    acc = conservation_accumulator()
    cwv = jnp.sum(
        q_v.astype(acc) * p_s.astype(acc)[..., None] * dsigma.astype(acc),
        axis=-1,
    ) / jnp.asarray(constants.g, dtype=acc)
    return global_area_sum(cwv, grid, owned_mask=owned_mask)


def fix_moisture_hydrostatic(
    q_v: jax.Array,
    target_moisture: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix global moisture conservation via multiplicative scaling.

    Scales q_v uniformly so that the global column-integrated water vapor
    matches *target_moisture*.  Uses multiplicative (not additive) correction
    to preserve spatial gradients and guarantee non-negativity.

    Parameters
    ----------
    q_v : jax.Array, shape (..., nlev)
        Specific humidity after physics [kg/kg].
    target_moisture : jax.Array
        Target global moisture integral [kg] (from initial state).
    p_s : jax.Array, shape (...)
        Surface pressure [Pa].
    dsigma : jax.Array, shape (nlev,)
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics.

    Returns
    -------
    jax.Array : Moisture-conserving q_v with same shape as input.
    """
    current = compute_global_moisture(q_v, p_s, dsigma, grid, owned_mask=owned_mask)
    scale = jnp.where(current > _tiny(current), target_moisture / current, 1.0)
    return q_v * scale


def diagnose_moisture_correction(
    q_v: jax.Array,
    target_moisture: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Diagnose the moisture correction that ``fix_moisture_hydrostatic``
    would apply, without actually applying it.

    Returns a dict of:
      - ``current_mass`` [kg]: current global column water-vapor integral
      - ``target_mass`` [kg]: prescribed target
      - ``correction_mass`` [kg]: target - current (positive = mass added by
        the fixer; negative = mass removed)
      - ``scale``: multiplicative factor that ``fix_moisture_hydrostatic``
        would multiply q_v by.  Equals 1.0 when target == current.

    Use to track silent corrections that would otherwise compound with
    other untracked sources/sinks (advection clipping, microphysics
    saturation adjustment, etc.) and bias the moist energy budget.
    Iter-87 audit follow-up to the deferred multiplicative-fixer
    tracking finding.
    """
    current = compute_global_moisture(q_v, p_s, dsigma, grid, owned_mask=owned_mask)
    scale = jnp.where(current > _tiny(current), target_moisture / current, 1.0)
    return {
        "current_mass": current,
        "target_mass": target_moisture,
        "correction_mass": target_moisture - current,
        "scale": scale,
    }


#: Water species whose physics tendency carries MASS in or out of the
#: column (vapour, warm-rain and ice-phase condensate).  Number
#: concentrations and passengers are per-mass tracers that ride the layer
#: mass but add none.
WATER_MASS_SPECIES = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")


def _d(x):
    """Field payload or the array itself (a bare jax Array's ``.data`` is
    its buffer, so no ``getattr(x, "data", x)``)."""
    return x.data if hasattr(x, "replace") and hasattr(x, "data") else x


def dry_surface_pressure(p_s, tracers, sigma_coord, *,
                         water_names=WATER_MASS_SPECIES):
    """Surface pressure of the DRY air alone [Pa]:
    ``p_top + sum_k dp_k (1 - Q_k)`` on sigma or hybrid layer masses
    (the coordinate's own top pressure ``p_top = p_half[0]`` -- sigma_top
    * p_s on sigma, ak[0] on hybrid -- is dry: no tracer lives above the
    top layer).  Equals ``p_s - g * column water`` since ``p_top + sum_k
    dp_k == p_s`` for every coordinate; written as the layer sum so it is
    exact in floating point against the layer masses the tests and the
    water tendencies use.

    The p_s-coordinate lanes' mass fixer conserves THIS (user decision
    2026-09-28, the FV3/IFS/CAM convention: dry air is conserved, water
    comes and goes through precipitation and evaporation).  A fixer on
    the total ``p_s`` would put every step's precipitated mass straight
    back as dry air, undoing :func:`apply_physics_water_mass`.
    ``tracers`` may be ``None`` (dry run): returns ``p_s``.
    """
    if tracers is None:
        return p_s
    Q = None
    for name in water_names:
        if name in tracers:
            q = _d(tracers[name])
            Q = q if Q is None else Q + q
    if Q is None:
        return p_s
    dp = sigma_coord.layer_thickness_dp(p_s)
    p_top = sigma_coord.pressure_at_half(p_s)[..., 0]
    return p_top + jnp.sum(dp * (1.0 - Q), axis=-1)


def apply_physics_water_mass(tracers, tracer_tendencies, p_s, sigma_coord,
                             dt, *, water_names=WATER_MASS_SPECIES):
    """Apply physics tracer tendencies WITH their mass (the FV3
    ``fv_update_phys`` nwat block on a p_s-coordinate column).

    Convention: every tracer is a SPECIFIC quantity on TOTAL air mass
    (``q_x = m_x / m_total``, the FV3/CAM/IFS convention this driver's
    column-water diagnostic ``sum(q p_s dsigma)/g`` already assumes).  A
    water tendency ``dq_x`` [1/s] from physics therefore means
    ``dt*dq_x*dp_k/g`` kg/m^2 of water added to layer k (negative for
    precipitation leaving it).  Applying it as ``q += dt*dq`` at fixed
    ``p_s`` (the previous behaviour) keeps the column's total mass: the
    precipitated water stays behind as DRY AIR.  Here the layer masses
    carry the water in and out:

        D_k     = dt * sum_x dq_x,k                       (water species only)
        dp_k    = layer mass of the coordinate at p_s      (sigma OR hybrid)
        p_s'    = p_s + sum_k dp_k * D_k                   (column water change)
        dp_k'   = layer mass of the coordinate at p_s'
        q_t,k'  = (q_t,k + dt*dq_t,k) * dp_k / dp_k'       for EVERY tracer t

    i.e. each tracer's layer MASS after the increment,
    ``(q + dt*dq)*dp_k``, is placed on the coordinate's new layer mass.
    Column totals of every tracer (water and passengers) and the column
    dry mass ``sum_k dp_k (1 - Q_k)`` are exact; the per-layer
    redistribution is the coordinate's own response to the column
    mass change (GLM/codex 2026-09-28: the unweighted form
    ``q' = (q + dq)/(1 + D_k)`` is first-order wrong on sigma).

    Returns ``(tracers_new, p_s_new)``.  Tracers with no tendency entry
    are re-weighted too (their mass is unchanged, their layer mass is
    not).  ``tracer_tendencies`` values may be Field-like (``.data``) or
    arrays; the tracers dict is returned in the same wrapping as given.
    """
    dp = sigma_coord.layer_thickness_dp(p_s)                  # (..., nlev)
    D = None
    for name in water_names:
        # a tendency for a species the state does not carry adds no mass:
        # its increment is dropped below, so it must not move p_s either
        # (codex 2026-09-28 r2)
        if name in tracer_tendencies and name in tracers:
            inc = dt * _d(tracer_tendencies[name])
            D = inc if D is None else D + inc
    if D is None:
        # no water tendency: layer masses unchanged, passengers still
        # take their own tendencies (codex 2026-09-28)
        p_s_new, ratio = p_s, None
    else:
        p_s_new = p_s + jnp.sum(dp * D, axis=-1)
        ratio = dp / sigma_coord.layer_thickness_dp(p_s_new)
    out = {}
    for name, tr in tracers.items():
        q = _d(tr)
        if name in tracer_tendencies:
            q = q + dt * _d(tracer_tendencies[name])
        elif ratio is None:
            out[name] = tr
            continue
        q_new = q if ratio is None else (q * ratio).astype(q.dtype)
        out[name] = tr.replace(data=q_new) if hasattr(tr, "replace") else q_new
    return out, p_s_new


def shift_ps_keep_tracer_mass(p_s, tracers, sigma_coord, correction):
    """Uniform ``p_s += correction`` (a mass fixer's move) with every
    tracer re-weighted onto the new layer masses so each tracer's column
    mass is untouched: ``q' = q * dp_k / dp_k'``.  The added or removed
    mass is then entirely DRY air, so the fixer's correction is exact
    for the dry integral (``dry' = dry + correction`` per column; the
    unweighted shift changes dry pressure by only ``correction * (1 -
    sum_k dB_k Q_k)`` and moves water by ``Q_bar * correction`` -- codex
    and GLM 2026-09-28).  Returns ``(p_s_new, tracers_new)``.
    """
    p_s_new = p_s + correction
    if tracers is None:
        return p_s_new, tracers
    ratio = sigma_coord.layer_thickness_dp(p_s) / sigma_coord.layer_thickness_dp(p_s_new)
    out = {}
    for name, tr in tracers.items():
        q = _d(tr)
        q_new = (q * ratio).astype(q.dtype)
        out[name] = tr.replace(data=q_new) if hasattr(tr, "replace") else q_new
    return p_s_new, out


def fix_total_water(
    tracers: dict[str, jax.Array],
    target_total_water: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    water_names: tuple[str, ...] = ("q_v", "q_c", "q_r"),
) -> dict[str, jax.Array]:
    """Fix total water (vapor + condensate) conservation.

    Scales all water tracers by a single uniform factor so that
    ``∫(q_v + q_c + q_r + ...) dp/g dA = target_total_water``.

    Parameters
    ----------
    tracers : dict[str, jax.Array]
        Tracer dict; only entries whose keys are in *water_names* are scaled.
    target_total_water : jax.Array
        Target global total water integral [kg].
    p_s : jax.Array
        Surface pressure [Pa].
    dsigma : jax.Array
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
    water_names : tuple[str, ...]
        Names of water-species tracers to include.

    Returns
    -------
    dict[str, jax.Array] : Tracers with water species scaled to conserve total water.
    """
    total_q = sum(tracers[n] for n in water_names if n in tracers)
    current = compute_global_moisture(total_q, p_s, dsigma, grid)
    scale = jnp.where(current > _tiny(current), target_total_water / current, 1.0)

    result = dict(tracers)
    for name in water_names:
        if name in result:
            result[name] = result[name] * scale
    return result


def fix_mass_hydrostatic_target(
    state_new: HydrostaticState,
    target_mass: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> HydrostaticState:
    """Fix mass conservation anchored to a fixed target mass.

    Unlike ``fix_mass_hydrostatic`` which anchors to the previous step,
    this anchors to a fixed target (typically the initial global mass),
    preventing slow drift accumulation over many steps.

    Parameters
    ----------
    state_new : HydrostaticState
        State after time integration.
    target_mass : jax.Array
        Target global mass integral (∫ p_s * dA at t=0).
    grid : CubedSphereGrid
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics — pass-through
        to ``fix_ps_mass_target`` so the mass integral correctly
        avoids double-counting under replicated MPI.  Iter-99 audit
        fix: previously this function called the Field-API
        ``global_integral`` directly without an owned_mask escape
        hatch.

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    if owned_mask is not None:
        # Use the raw-array path that handles owned_mask for MPI
        # replicated dynamics.  Same correction formula but the
        # global integral correctly weights by owned_mask.
        p_s_fixed_data = fix_ps_mass_target(
            state_new.p_s.data, target_mass, grid, owned_mask=owned_mask,
        )
        return state_new._replace(
            p_s=state_new.p_s.replace(data=p_s_fixed_data),
        )
    mass_new = global_integral(state_new.p_s, grid)
    correction = (target_mass - mass_new) / _total_area(grid)
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)


def fix_ps_mass(
    p_s_new: jax.Array,
    p_s_old: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix dry mass on raw p_s arrays — non-anchor variant.

    Raw-array equivalent of :func:`fix_mass_hydrostatic`.  Lets
    callers (e.g. the FV3 D-grid dycore) avoid round-tripping the
    full prognostic state through ``fv3_to_hydrostatic`` just to
    extract ``p_s``: D-grid → cell-centre wind interpolation is
    expensive and gets thrown away because only ``p_s`` is touched.

    Both ``mass_old`` and ``mass_new`` are computed in one
    batched allreduce, matching :func:`fix_mass_hydrostatic`'s
    communication pattern.
    """
    # differentiable_broadcast=True: ``correction`` is added to EVERY cell, so
    # under face-scatter the reduction's VJP must allreduce the cotangent (else
    # the cross-rank gradient of the shared correction is dropped — #811).
    mass_old, mass_new = batch_global_area_sums(
        [p_s_old, p_s_new], grid, owned_mask=owned_mask,
        differentiable_broadcast=True,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    return p_s_new + correction


def fix_ps_mass_target(
    p_s: jax.Array,
    target_mass: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix dry mass conservation on raw p_s array, anchored to a fixed target.

    This is the raw-array version of ``fix_mass_hydrostatic_target``,
    suitable for use inside ``jax.lax.scan`` where we work with raw
    arrays rather than Field-wrapped NamedTuples.

    Parameters
    ----------
    p_s : jax.Array, shape (6, n, n)
        Surface pressure after dynamics.
    target_mass : jax.Array (scalar)
        Target global mass integral (∫ p_s * dA at t=0).
    grid : CubedSphereGrid
    owned_mask : jax.Array, optional
        Shape ``(6,)`` float mask for MPI replicated dynamics.
        See :func:`global_area_sum` for details.

    Returns
    -------
    jax.Array : Corrected p_s with same shape.
    """
    # differentiable_broadcast=True: ``correction`` is added to EVERY cell, so
    # under face-scatter the reduction's VJP must allreduce the cotangent (else
    # the cross-rank gradient of the shared correction is dropped — #811).
    mass_new = global_area_sum(
        p_s, grid, owned_mask=owned_mask, differentiable_broadcast=True)
    correction = (target_mass - mass_new) / _total_area(grid)
    return p_s + correction


def compute_nh_dry_mass(
    rho_prime: jax.Array,
    height_coord,
    terrain_metric,
    grid: CubedSphereGrid,
) -> jax.Array:
    """Compute global dry-air mass for the non-hydrostatic model.

    M = ∫ J · rho_total · dz · dA  summed over all levels.

    Parameters
    ----------
    rho_prime : jax.Array, shape (6, n, n, nlev)
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : Scalar global dry mass [kg].
    """
    rho_total = height_coord.rho_ref + rho_prime  # (6,n,n,nlev)
    J = terrain_metric.jacobian  # (6, n, n)
    dz = height_coord.dz  # (nlev,)
    # Column mass: sum_k(J * rho_total_k * dz_k)
    col_mass = jnp.sum(
        J[..., None] * rho_total * dz[None, None, None, :],
        axis=-1,
    )  # (6, n, n)
    return global_area_sum(col_mass, grid)


def fix_mass_nonhydrostatic(
    state,
    target_mass: jax.Array,
    height_coord,
    terrain_metric,
    grid: CubedSphereGrid,
):
    """Fix dry-mass conservation for the non-hydrostatic model.

    Applies a uniform additive correction to rho_prime so that the
    global dry mass matches the target.

    Parameters
    ----------
    state : NonHydrostaticState
    target_mass : jax.Array
        Target global mass.
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    grid : CubedSphereGrid

    Returns
    -------
    NonHydrostaticState : Mass-conserving state.
    """
    rho_total = height_coord.rho_ref + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    col_mass = jnp.sum(
        J[..., None] * rho_total * dz[None, None, None, :],
        axis=-1,
    )
    col_vol = J * jnp.sum(dz)
    current_mass, total_vol = batch_global_area_sums(
        [col_mass, col_vol], grid,
    )
    correction = (target_mass - current_mass) / total_vol
    rho_fixed = state.rho_prime.replace(
        data=state.rho_prime.data + correction,
    )
    return state._replace(rho_prime=rho_fixed)


def compute_hydrostatic_energy(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord,
) -> dict[str, jax.Array]:
    """Compute energy diagnostics for the hydrostatic PE model.

    Returns
    -------
    dict with:
        'kinetic_energy': ∫ 0.5·p_s·(u²+v²)·dσ·dA / g
        'internal_energy': ∫ c_v·T·p_s·dσ·dA / g
        'potential_energy': ∫ Φ·p_s·dσ·dA / g
        'total_energy': sum of all three
    """
    # iter-43: promote energy fields to the fp64 budget accumulator
    # before the column + area-weighted sums.  Same fp32-field bug as
    # iter-1/4/5 mass path and iter-42 SW/MPAS energy fixer — the
    # 0.5·(u²+v²) square+multiply lost ~7 bits of precision when the
    # state was stored in fp32.
    acc = conservation_accumulator()
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    T = state.T.data.astype(acc)
    p_s = state.p_s.data.astype(acc)
    phis = state.phis.data.astype(acc)
    g = jnp.asarray(constants.g, dtype=acc)
    c_v = jnp.asarray(constants.c_vd, dtype=acc)
    dsigma = sigma_coord.dsigma.astype(acc)

    # Mass weight per layer: p_s * dsigma / g
    # Use [..., None] broadcasting so this works for both cubed-sphere
    # (6, n, n, nlev) and lat-lon (n_lat, n_lon, nlev) state shapes.
    mass_weight = p_s[..., None] * dsigma / g

    # Kinetic + internal + potential energy column reductions all share
    # the level axis and ``mass_weight``; stack the integrands and reduce
    # once.  ``mass_weight`` is factored into the stack so each integrand
    # contributes only its own value field.
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)
    _ke_intg = 0.5 * (u**2 + v**2)
    _ie_intg = c_v * T
    _pe_intg = Phi
    _col_triple = jnp.sum(
        jnp.stack([_ke_intg, _ie_intg, _pe_intg], axis=-1)
        * mass_weight[..., None],
        axis=-2,
    )
    ke_col = _col_triple[..., 0]
    ie_col = _col_triple[..., 1]
    pe_col = _col_triple[..., 2]

    ke, ie, pe = batch_global_area_sums([ke_col, ie_col, pe_col], grid)

    total = ke + ie + pe
    return {
        'kinetic_energy': ke,
        'internal_energy': ie,
        'potential_energy': pe,
        'total_energy': total,
    }


def compute_nh_energy(
    state,
    grid: CubedSphereGrid,
    height_coord,
    terrain_metric,
) -> dict[str, jax.Array]:
    """Compute energy diagnostics for the non-hydrostatic CE model.

    Returns
    -------
    dict with:
        'kinetic_energy': ∫ 0.5·rho·(u²+v²+w²)·J·dz·dA
        'internal_energy': ∫ c_v·T·rho·J·dz·dA
        'potential_energy': ∫ g·z·rho·J·dz·dA
        'total_energy': sum of all three
    """
    # iter-43: promote energy fields to fp64 budget accumulator (see
    # compute_hydrostatic_energy docstring above).
    acc = conservation_accumulator()
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    w = state.w.data.astype(acc)
    theta_p = state.theta_prime.data.astype(acc)
    rho_p = state.rho_prime.data.astype(acc)

    g = jnp.asarray(constants.g, dtype=acc)
    c_v = jnp.asarray(constants.c_vd, dtype=acc)
    theta_0 = height_coord.theta_ref.astype(acc)
    rho_0 = height_coord.rho_ref.astype(acc)
    exner_0 = height_coord.exner_ref.astype(acc)
    dz = height_coord.dz.astype(acc)
    z_full = height_coord.z_full.astype(acc)
    J = terrain_metric.jacobian.astype(acc)

    rho_total = rho_0 + rho_p
    theta_total = theta_0 + theta_p
    T = theta_total * exner_0  # approximate T from theta * exner_ref

    # w at full levels
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])

    weight = J[..., None] * dz[None, None, None, :] * rho_total  # (6,n,n,nlev)

    # KE+IE+PE column reductions all share the level axis and
    # ``weight``; stack the integrands and reduce once.
    _ke_intg = 0.5 * (u**2 + v**2 + w_full**2)
    _ie_intg = c_v * T
    _pe_intg = g * jnp.broadcast_to(z_full[None, None, None, :], T.shape)
    _col_triple = jnp.sum(
        jnp.stack([_ke_intg, _ie_intg, _pe_intg], axis=-1)
        * weight[..., None],
        axis=-2,
    )
    ke_col = _col_triple[..., 0]
    ie_col = _col_triple[..., 1]
    pe_col = _col_triple[..., 2]

    ke, ie, pe = batch_global_area_sums([ke_col, ie_col, pe_col], grid)

    total = ke + ie + pe
    return {
        'kinetic_energy': ke,
        'internal_energy': ie,
        'potential_energy': pe,
        'total_energy': total,
    }


def compute_conservation_diagnostics(
    state: ShallowWaterState,
    grid,
    g: float = constants.g,
) -> dict[str, jax.Array]:
    """Compute conservation diagnostic quantities (any grid).

    Returns
    -------
    dict with:
        'total_mass': Global integral of h*area
        'total_energy': Global integral of (KE + PE)*area
    """
    # iter-44: promote energy field to fp64 budget accumulator before
    # the area-sum (same fp32-field bug as iter-42/43).  The mass
    # integrand (h alone) is already correct via batch_global_area_sums
    # internal cast.
    acc = conservation_accumulator()
    h = state.h.data.astype(acc)
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    h_s = state.h_s.data.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)

    ke = 0.5 * h * (u**2 + v**2)
    pe = 0.5 * g_acc * (h + h_s)**2
    total_mass, total_energy = batch_global_area_sums(
        [h, ke + pe], grid,
    )

    return {
        'total_mass': total_mass,
        'total_energy': total_energy,
    }


# ==============================================================================
# Voronoi mesh conservation
# ==============================================================================

def global_integral_voronoi(field, mesh) -> jax.Array:
    """Area-weighted global integral on a Voronoi mesh.

    Parameters
    ----------
    field : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array : scalar
    """
    acc = conservation_accumulator()
    return jnp.sum(field.astype(acc) * mesh.areaCell.astype(acc))


def fix_mass_mpas(state_new, state_old, mesh, target_mass=None):
    """Fix mass conservation on a Voronoi mesh via a uniform additive ``h``
    correction — the canonical MPI-aware fixer shared by the MPAS shallow-water
    dycore (federation dedup: ``atmosphere.shallow_water_mpas`` imports this
    instead of carrying its own copy).

    Upcasts to the conservation accumulator (fp64 where supported, fp32 fallback
    on Metal / no-x64 — policy-aware, unlike a hardcoded ``float64`` which is
    Metal-unsafe) for the global reduction, to avoid catastrophic cancellation
    in the mass difference.  The sums are batched into ONE allreduce for
    multi-rank scaling.

    ``target_mass`` given (anchor-to-initial mode) → skip the ``state_old`` sum;
    only ``mass_new`` and ``total_area`` need a reduction.  ``target_mass=None``
    (match-previous-state mode) → reduce all three.
    """
    acc = conservation_accumulator()
    area = mesh.areaCell.astype(acc)
    if target_mass is not None:
        _h_stack = jnp.stack(
            [
                state_new.h.data.astype(acc),
                jnp.ones_like(state_new.h.data, dtype=acc),
            ],
            axis=-1,
        ) * area[..., None]
        local = jnp.sum(_h_stack, axis=tuple(range(area.ndim)))
        # is_multi_process(), NOT jax.process_count() > 1: mpi4jax reduce
        # only when each rank holds a LOCAL partition (route-A).  Under
        # multi-controller SPMD the sum above is already global via GSPMD;
        # mpi4jax here would arm the forbidden mixed stack and over-count
        # by the world size (#751 latent-bug class).
        from legoesm.parallel.reductions import is_multi_process
        if is_multi_process():
            from legoesm.parallel.reductions import global_sum_mpi
            local = global_sum_mpi(local)
        mass_new, total_area = local[0], local[1]
        mass_old = target_mass
    else:
        _h_stack = jnp.stack(
            [
                state_old.h.data.astype(acc),
                state_new.h.data.astype(acc),
                jnp.ones_like(state_new.h.data, dtype=acc),
            ],
            axis=-1,
        ) * area[..., None]
        local = jnp.sum(_h_stack, axis=tuple(range(area.ndim)))
        from legoesm.parallel.reductions import is_multi_process
        if is_multi_process():  # route-A local partitions only (see above)
            from legoesm.parallel.reductions import global_sum_mpi
            local = global_sum_mpi(local)
        mass_old, mass_new, total_area = local[0], local[1], local[2]
    correction = (mass_old - mass_new) / total_area
    # No cast-back: the fp64 correction promotes the add — load-bearing for the
    # ~1e-12 mass conservation in test_sw_mass_conservation_anchored.
    h_fixed = state_new.h.replace(data=state_new.h.data + correction)
    return state_new._replace(h=h_fixed)


def compute_nh_dry_mass_mpas(
    rho_prime: jax.Array,
    height_coord,
    terrain_metric,
    mesh,
) -> jax.Array:
    """Global dry mass on a Voronoi mesh for the non-hydrostatic model.

    M = sum_k(sum_cells J · (rho_ref_k + rho_prime_k) · dz_k · areaCell).

    Parameters
    ----------
    rho_prime : jax.Array, shape (nCells, nlev)
    height_coord : HeightCoordinate (rho_ref, dz attrs as in
        ``compute_nh_dry_mass`` cubed-sphere variant).
    terrain_metric : TerrainMetric (jacobian shape ``(nCells,)``).
    mesh : VoronoiMesh.

    Returns
    -------
    jax.Array : scalar dry mass [kg].
    """
    rho_total = height_coord.rho_ref + rho_prime  # (nCells, nlev)
    J = terrain_metric.jacobian                   # (nCells,)
    dz = height_coord.dz                          # (nlev,)
    col_mass = jnp.sum(
        J[:, None] * rho_total * dz[None, :], axis=-1,
    )                                             # (nCells,)
    return global_integral_voronoi(col_mass, mesh)


def fix_mass_nonhydrostatic_mpas(
    state,
    target_mass: jax.Array,
    height_coord,
    terrain_metric,
    mesh,
):
    """Fix dry-mass conservation on a Voronoi mesh.

    Applies a uniform additive correction to ``rho_prime`` so the
    global dry mass matches ``target_mass``.  Volume normalisation is
    ``∫ J · dz · dA`` so the correction has units of density (kg/m³)
    and is broadcast across levels — matches the cubed-sphere
    ``fix_mass_nonhydrostatic`` convention.

    Parameters
    ----------
    state : MPASNonHydrostaticState
    target_mass : jax.Array
        Target dry mass (e.g. computed once via
        ``compute_nh_dry_mass_mpas`` on the initial state).
    height_coord, terrain_metric, mesh : as for
        :func:`compute_nh_dry_mass_mpas`.
    """
    rho_total = height_coord.rho_ref + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    col_mass = jnp.sum(
        J[:, None] * rho_total * dz[None, :], axis=-1,
    )                                             # (nCells,)
    col_vol = J * jnp.sum(dz)                     # (nCells,)
    current_mass, total_vol = _batch_global_area_sums_voronoi(
        [col_mass, col_vol], mesh,
    )
    correction = (target_mass - current_mass) / total_vol
    rho_fixed = state.rho_prime.replace(
        data=state.rho_prime.data + correction,
    )
    return state._replace(rho_prime=rho_fixed)


def _batch_global_area_sums_voronoi(
    arrays: list[jax.Array],
    mesh,
) -> list[jax.Array]:
    """Voronoi analogue of :func:`batch_global_area_sums`.

    Stacks the arrays along a new trailing axis, multiplies by
    ``mesh.areaCell`` (cast to the fp64 conservation accumulator),
    reduces once locally, then applies the multi-rank allreduce when
    the mesh is sharded.
    """
    acc = conservation_accumulator()
    area_acc = mesh.areaCell.astype(acc)
    stacked = jnp.stack([arr.astype(acc) for arr in arrays], axis=-1)
    summed = jnp.sum(stacked * area_acc[..., None], axis=0)  # (n_arrays,)
    local_sums = [summed[..., i] for i in range(len(arrays))]
    if is_distributed():
        from legoesm.parallel.reductions import batch_allreduce_mpi
        return batch_allreduce_mpi(local_sums, op="sum")
    return local_sums


def fix_energy_mpas(state_new, state_old, mesh, g=constants.g):
    """Fix energy conservation on a Voronoi mesh via velocity scaling — the
    canonical MPI-aware fixer shared by the MPAS shallow-water dycore (federation
    dedup: ``atmosphere.shallow_water_mpas`` imports this).

    Rescales ``u`` so the total energy of ``state_new`` matches that of
    ``state_old``: the four contributing sums (KE/PE of old and new) are batched
    into ONE allreduce instead of four when the mesh is sharded across ranks.

    The KE/PE integrands are promoted to the precision-policy ``conservation_
    accumulator`` (fp64 where supported, fp32 fallback on Metal) BEFORE the
    area-weighted reduction — so the energy budget does not lose digits to fp32
    cancellation when the model runs in fp32 storage (same rationale as
    :func:`fix_mass_mpas`).  The resulting ``scale`` is a dimensionless ratio,
    cast back to ``u``'s storage dtype so the velocity field keeps its policy
    dtype (the accumulator only sharpens the ratio, it does not widen ``u``).
    """
    _TINY = float(jnp.finfo(jnp.float32).tiny)
    acc = conservation_accumulator()
    area = mesh.areaCell.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)

    def _ke_pe_terms(state):
        h = state.h.data.astype(acc)
        u = state.u.data.astype(acc)
        h_s = state.h_s.data.astype(acc)
        # KE and PE share ``area`` on the same horizontal axes — stack the two
        # integrands and reduce locally once.
        _stack = jnp.stack(
            [kinetic_energy_cell(u, mesh) * h, 0.5 * g_acc * (h + h_s) ** 2],
            axis=-1,
        ) * area[..., None]
        _pair = jnp.sum(_stack, axis=tuple(range(area.ndim)))
        return _pair[0], _pair[1]

    KE_old, PE_old = _ke_pe_terms(state_old)
    KE_new, PE_new = _ke_pe_terms(state_new)

    local = jnp.stack([KE_old, PE_old, KE_new, PE_new])
    # Route-A local-partition reduce only — NOT under multi-controller SPMD
    # (GSPMD already made the sums global; #751 latent-bug class).
    from legoesm.parallel.reductions import is_multi_process
    if is_multi_process():
        from legoesm.parallel.reductions import global_sum_mpi
        local = global_sum_mpi(local)
    KE_old, PE_old, KE_new, PE_new = local[0], local[1], local[2], local[3]
    E_old = KE_old + PE_old

    KE_target = jnp.maximum(E_old - PE_new, 0.0)
    scale = jnp.where(KE_new > _TINY, jnp.sqrt(KE_target / KE_new), 1.0)

    # ``scale`` is dimensionless — cast back so ``u`` keeps its storage dtype.
    scale = scale.astype(state_new.u.data.dtype)
    u_fixed = state_new.u.replace(data=state_new.u.data * scale)
    return state_new._replace(u=u_fixed)
