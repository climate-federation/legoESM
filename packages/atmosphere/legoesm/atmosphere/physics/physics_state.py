"""Explicit prognostic state for all atmospheric physics modules.

Physics state is carried through the time loop alongside the dynamical
state.  It is checkpointable, vmappable, and visible to JAX tracing.

This replaces the previous pattern of mutable ``nonlocal`` variables
captured in factory closures, which prevented checkpoint/restart of
physics state, ensemble ``vmap``, and clean JIT tracing.

Every physics module that carries prognostic variables (TKE, convective
mass-flux profile, GWD wave action spectrum) reads from and writes to
fields in this state.  Modules that are disabled (``scheme="none"``) or
stateless simply ignore the corresponding fields.

Example
-------
>>> from legoesm.atmosphere.physics.physics_state import (
...     PhysicsState, init_physics_state,
... )
>>> ps = init_physics_state(ncol=1536, nlev=40, physics_config=config)
>>> # Thread through time loop:
>>> tendencies, ps = physics_fn(state, grid, coord, ps)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# "No surface-T override" sentinel for ``surface_T_sfc_override``.  Formerly
# ``jnp.nan``, which left the state non-finite EVERY step and poisoned two
# debug tools (JAX_DEBUG_NANS tripped on step 1; the realism inspector
# false-positived every checkpoint) — see #911.  ``-1e4`` is unambiguously
# below any physical surface temperature [K] (coldest Earth surface ~180 K),
# so the resolver selects the fallback with a simple threshold while the state
# stays finite.  Kept well within fp16 range (max ~6.5e4) so the sentinel is
# representable — and stays finite — in every supported storage dtype (a
# larger magnitude overflowed fp16 to -inf; codex).
NO_SFC_T_OVERRIDE: float = -1.0e4
# A real override is a physical surface temperature (> 0 K); anything at or
# below this threshold means "no override".  Well clear of both the sentinel
# and any physical value.
SFC_T_OVERRIDE_VALID_MIN: float = 1.0


class PhysicsState(NamedTuple):
    """Prognostic physics state — passed through time loop, checkpointable.

    All fields are concrete ``jax.Array`` instances (never ``None``).
    Unused fields are zero-filled and ignored by their modules.

    Fields
    ------
    tke : jax.Array, shape (ncol, nlev)
        Turbulent kinetic energy for TKE / CLUBB-lite / EDMF schemes.
        Initialized to ``tke_min`` when active, zero otherwise.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Convection prognostic *profile*.  Layout depends on the scheme:

        * Profile-carrying schemes (Tiedtke, Bechtold) store the full
          updraft mass-flux profile :math:`M_u(k)` at every level.
        * Scalar-carrying schemes (``mass_flux``, ``edmf``) pack their
          single scalar (``M_c`` or ``a_u``) at ``[:, -1]`` (the
          surface-adjacent slot, used as a cloud-base proxy) and leave
          the rest of the column zero.
        * Stateless / diagnostic schemes (``sbm``, ``dca``, ``kuo``,
          ``zhang_mcfarlane``, ``kain_fritsch``, ``emanuel``,
          ``"none"``) carry the field as zeros and may opportunistically
          pack a diagnostic into ``[:, -1]`` for visibility.

        The unified shape ``(ncol, nlev)`` lets every scheme — current
        or future — share a single carry slot without per-scheme schema
        churn.
    conv_stoch_state : jax.Array, shape (ncol,)
        Stochastic AR1 noise state for Bechtold/IFS (Bechtold et al.
        2014).  Carried across time steps so that the perturbation has
        the prescribed temporal decorrelation.  Other schemes leave
        this zero-filled and ignore it.
    gwd_spectrum : jax.Array, shape (ncol, n_azimuths, n_wavenumbers)
        Gravity wave drag wave action spectrum for the prognostic
        spectral scheme.  Shape is ``(ncol, 1, 1)`` when inactive
        (minimal allocation).
    prng_key : jax.Array, shape (2,) uint32
        Master JAX PRNG key advanced once per outer step.  Stochastic
        sub-physics modules (currently only Bechtold/IFS when
        ``enable_stochastic=True``) fold this key with a module-id and
        consume the derived sub-key for their AR1 innovation.  When all
        stochastic modules are disabled the key is carried unchanged.
    surface_T_sfc_override : jax.Array, shape (ncol,)
        Per-column override for the surface temperature seen by the
        turbulence scheme's bulk-flux call.  The finite
        ``NO_SFC_T_OVERRIDE`` value (the default) is the sentinel for
        "no override — fall back to ``T_col[:, -1]``", preserving the
        legacy ``T_sfc = lowest air temp`` convention for every 3-D run
        while keeping the state finite (#911; was ``NaN``).  The
        single-column model populates this
        from ``SCMForcing.T_s(t)`` when ``prescribe="T_s"`` so that the
        bulk-flux gradient ``T_sfc − T[..., -1]`` is non-zero (without
        this override, anchoring ``T[..., -1]`` to the prescribed value
        collapses the sensible-flux gradient — Phase B codex iter-1
        finding).
    qke : jax.Array, shape (ncol, nlev)
        Prognostic ``qke = 2·TKE`` [m²/s²] for the MYNN-2.5 turbulence
        scheme (Nakanishi-Niino 2009).  Carried in a slot **distinct**
        from :attr:`tke` so a restart that switches schemes between
        MY-2.5 (``tke``) and MYNN-2.5 (``qke``) cannot silently feed
        the wrong moment as energy — the active scheme reads its own
        field.  Zero-filled when the active turbulence scheme is not
        MYNN-2.5 (Phase C codex iter-1 medium finding).
    clubb_moments : jax.Array, shape (ncol, 15, nlev+1)
        Prognostic CLUBB higher-order moment state (the 15-field
        :class:`~legoesm.atmosphere.physics.turbulence.clubb.CLUBBMomentState`
        packed by ``pack_clubb_moments``) for the fuller ``scheme="clubb"``
        prognostic path. zm-level fields use the full ``nlev+1`` axis; zt-level
        means/``wp3`` use the first ``nlev`` slots (trailing slot zero).
        Minimally allocated ``(ncol, 1, 1)`` when CLUBB is not the active scheme
        (like :attr:`gwd_spectrum`).  Distinct from :attr:`tke` so the diagnostic
        phase-1 CLUBB (``tke``) and the prognostic CLUBB (``clubb_moments``)
        cannot cross-feed on a restart scheme switch.
    rad_heating : jax.Array, shape (ncol, nlev)
        Cached radiative heating tendency ``dT/dt`` [K/s] from the most
        recent full radiation solve.  Used by the radiation sub-cycle: on
        steps that do NOT re-solve RRTMGP/gray (cadence ``rad_update_steps``),
        the combined physics adds this held tendency instead of recomputing
        radiation (CESM/E3SM-standard radiation cadence).  Written on each
        radiation step and carried unchanged on the intervening steps.
        Zero-filled before the first radiation solve (which is always step 0
        of the sub-cycle, so the cache is populated before any held step
        reads it).  Carried through the #413 checkpoint so a restart that
        lands mid-sub-cycle continues with the correct held tendency.
    aerosol_number : jnp.ndarray, shape (ncol, nlev)
        OPTIONAL prognostic accumulation-mode aerosol NUMBER concentration
        [1/m^3] for the opt-in prognostic-aerosol tracer
        (:mod:`~legoesm.atmosphere.physics.microphysics.prognostic_aerosol`).
        Zero-filled by default so every existing run is byte-identical and the
        pytree stays uniform.  When the option is enabled a driver advances it
        with ``step_prognostic_aerosol`` and hands it to ARG activation via the
        ``activated_nc_field(..., aerosol_number=...)`` /
        ``arg_cdnc_from_config`` HOOK — e.g. by placing this field in the
        microphysics ``forcing`` dict under key ``"aerosol_number"`` (the
        microphysics ``physics_fn`` reads it there).  The default (disabled)
        path keeps the prescribed-AOD proxy and never touches this field.
        Appended LAST (with a default) so existing direct constructors are
        unaffected.
    """
    tke: jnp.ndarray
    conv_prog_profile: jnp.ndarray
    conv_stoch_state: jnp.ndarray
    gwd_spectrum: jnp.ndarray
    prng_key: jnp.ndarray
    surface_T_sfc_override: jnp.ndarray
    qke: jnp.ndarray
    clubb_moments: jnp.ndarray
    rad_heating: jnp.ndarray
    # GLOBAL column ids, shape (ncol,) int32 — the decomposition-invariant
    # identity for per-column stochastic draws (Bechtold AR1 folds the
    # per-step sub-key with each column's GLOBAL id).  Sharding-aware by
    # construction: a lat-band SPMD shard receives its own contiguous
    # chunk, so a physical column draws the SAME innovation regardless of
    # the decomposition.  Constant data (never updated by sub-physics);
    # re-derivable as arange(ncol) — restart loaders may default it.
    col_index: jnp.ndarray
    aerosol_number: jnp.ndarray = None
    # Sub-grid LIQUID cloud fraction [-], shape (ncol, nlev), written by a
    # turbulence scheme that carries its own PDF cloud closure (CLUBB) so the
    # radiation module can consume it (cloud_scheme="clubb") instead of the
    # RH-diagnosed grid-scale one.  Always materialised as zeros (uniform
    # pytree, byte-identical for runs that never read it — radiation ignores it
    # unless cloud_scheme="clubb"); appended LAST with a default so existing
    # direct constructors are unaffected.
    cloud_fraction: jnp.ndarray = None
    # OPTIONAL per-step DYNAMICS (large-scale advective) tendencies of T [K/s]
    # and q_v [kg/kg/s], shape (ncol, nlev) — the IFS ``PTENTA``/``PTENQA``
    # analog consumed by Bechtold's RCAPQADV CAPE-advection correction
    # (``use_ifs_cape_qadv``; see convection/bechtold.py).  A process-split
    # driver writes ``(state_after_dyn - state_before_dyn)/dt`` here BEFORE the
    # convection call, AND passes convection the post-dynamics state — the leaf
    # forms its reference environment as ``state - dyn_tendency*dt``, so the
    # two must be staged consistently (the IFS ``ZTENH2 = ZTENH - PTENTA*dt``
    # convention).  These are a diagnostic INPUT to the closure (they shape
    # CAPE), NOT applied to the state by convection — the dynamics updates the
    # state separately, so there is no double count.  ``None``
    # (default) leaves the RCAPQADV path inert; the convection bridge raises at
    # trace time if ``use_ifs_cape_qadv`` is on while these are absent.
    # Appended LAST with a default so existing direct constructors are
    # unaffected.
    dyn_tendency_T: jnp.ndarray = None
    dyn_tendency_qv: jnp.ndarray = None
    # CONVECTIVE surface precipitation [kg/m^2/s], shape (ncol,), written by
    # the convection module each step (the combined-physics accumulator
    # captures the convection slot's ``precip`` tendency field).  Consumed
    # one step LAGGED by the radiation module's cloud-fraction call
    # (Slingo-1987 ``convective_cloud``) on the standalone (MPAS) path —
    # the same lagged-carry convention the FV pipeline uses for its
    # ``conv_precip`` threading.  Always materialised as zeros (uniform
    # pytree; byte-identical for runs that never read it).  Appended LAST
    # with a default so existing direct constructors are unaffected.
    conv_precip: jnp.ndarray = None


# Per-step INPUT fields (recomputed by the driver from forcing/dynamics before
# every convection call) — NOT evolving physics memory, so they are neither
# persisted in a restart checkpoint nor subject to the carry-completeness gate.
# A checkpoint legitimately lacks them; the fresh seed's ``None`` is correct.
PHYSSTATE_INPUT_FIELDS = frozenset({"dyn_tendency_T", "dyn_tendency_qv"})


def init_physics_state(
    ncol: int,
    nlev: int,
    physics_config,
    *,
    dtype=None,
    prng_seed: int = 0,
) -> PhysicsState:
    """Create initial physics state with correct shapes and defaults.

    Parameters
    ----------
    ncol : int
        Number of atmospheric columns (e.g. 6*n*n for cubed-sphere).
    nlev : int
        Number of vertical levels.
    physics_config : PhysicsConfig
        Unified physics configuration that determines which modules
        need prognostic state and what their initial values are.
    dtype : jnp.dtype or None
        Storage dtype for the persistent prognostic arrays.  When
        ``None`` (default) JAX picks its default float dtype (float64
        under x64).  Pass an explicit dtype (typically the precision
        policy's storage dtype) when running in a precision policy
        that holds the model state at f32 — defaulting otherwise
        forces every physics step to read/write the persistent state
        at f64 even though the column physics path runs at f32, which
        doubles the GPU bandwidth on the prognostic carry.

    Returns
    -------
    PhysicsState
        Initialized state ready for the time loop.

    Notes
    -----
    For the scalar-carrying ``mass_flux`` / ``edmf`` branches, the
    initial value is packed at ``[:, -1]`` (cloud-base proxy) with
    zeros aloft — the rest of the column is empty storage that profile
    schemes may write into.  This keeps the layout uniform across all
    convection schemes (current and future).
    """
    # --- Turbulence TKE (MY-2.5 family) vs qke (MYNN-2.5) ---
    # Distinct fields so a restart that switches schemes cannot
    # silently feed the wrong moment as energy: MY-2.5 / CLUBB-lite /
    # EDMF carry TKE in ``tke``; MYNN-2.5 carries ``qke = 2·TKE`` in
    # ``qke``.  Inactive slots stay zero-filled with no per-step cost
    # because the dispatcher only reads the slot tied to the active
    # scheme.  Which scheme uses which slot comes from the shared
    # ``turbulence_scheme_traits`` (function-scope import: the
    # integration module pulls the dynamics import chain).
    from legoesm.atmosphere.physics.turbulence.integration import (
        turbulence_scheme_traits,
    )
    turb_cfg = physics_config.turbulence
    _turb_traits = turbulence_scheme_traits(turb_cfg.scheme)
    if _turb_traits.energy_field == "tke":
        scheme_sub = getattr(turb_cfg, turb_cfg.scheme)
        tke_min = getattr(scheme_sub, "tke_min", 1e-6)
        tke = jnp.full((ncol, nlev), tke_min, dtype=dtype)
    else:
        tke = jnp.zeros((ncol, nlev), dtype=dtype)
    if _turb_traits.energy_field == "qke":
        qke_min = getattr(turb_cfg.mynn25, "tke_min", 1e-10)
        qke = jnp.full((ncol, nlev), qke_min, dtype=dtype)
    else:
        qke = jnp.zeros((ncol, nlev), dtype=dtype)

    # --- CLUBB prognostic higher-order moment state ---
    # Carried only for the fuller prognostic clubb path; packed (ncol, 15, nzm).
    # Minimal (ncol, 1, 1) otherwise (like gwd_spectrum) — no wasted memory.
    if turb_cfg.scheme == "clubb" and getattr(turb_cfg.clubb, "prognostic", False):
        from legoesm.atmosphere.physics.turbulence.clubb import (
            init_clubb_moments,
            pack_clubb_moments,
        )
        _dt = dtype if dtype is not None else jnp.float64
        clubb_moments = pack_clubb_moments(
            init_clubb_moments(ncol, nlev, turb_cfg.clubb, dtype=_dt))
    else:
        clubb_moments = jnp.zeros((ncol, 1, 1), dtype=dtype)

    # --- Convection prognostic profile ---
    conv_cfg = physics_config.convection
    conv_prog_profile = jnp.zeros((ncol, nlev), dtype=dtype)
    if conv_cfg.scheme == "mass_flux":
        conv_prog_profile = conv_prog_profile.at[:, -1].set(
            conv_cfg.mass_flux.M_c_init
        )
    elif conv_cfg.scheme == "edmf":
        conv_prog_profile = conv_prog_profile.at[:, -1].set(
            conv_cfg.edmf.a_u_init
        )

    # --- Convection stochastic AR1 noise state (Bechtold/IFS) ---
    conv_stoch_state = jnp.zeros((ncol,), dtype=dtype)

    # --- GWD wave action spectrum ---
    # Seeded for prognostic_spectral AND any '+'-composite that contains it
    # (issue #834) — both thread the wave-action spectrum through the carry.
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        gwd_carries_spectrum,
    )
    gwd_cfg = physics_config.gravity_wave_drag
    if gwd_carries_spectrum(gwd_cfg.scheme):
        sc = gwd_cfg.prognostic_spectral
        gwd_spectrum = jnp.full(
            (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
            dtype=dtype,
        )
    else:
        # Minimal allocation — zero-element trailing dims avoid wasted memory.
        gwd_spectrum = jnp.zeros((ncol, 1, 1), dtype=dtype)

    # --- Master JAX PRNG key (advanced once per outer step) ---
    # Imported lazily so the module load path does not pull in the
    # ``jax.random`` symbol when only ``init_physics_state`` is unused.
    import jax
    prng_key = jax.random.PRNGKey(int(prng_seed))

    # --- Surface-temperature override (SCM forcing hook) ---
    # Finite ``NO_SFC_T_OVERRIDE`` sentinel = "no override; turbulence falls
    # back to ``T[:, -1]``".  Preserves all existing 3-D behaviour (the
    # resolver still selects the fallback for every unset column) while
    # keeping the state finite (#911).  The single-column driver overwrites
    # this with a physical skin temperature when ``SCMForcing.prescribe ==
    # "T_s"``.
    surface_T_sfc_override = jnp.full((ncol,), NO_SFC_T_OVERRIDE, dtype=dtype)

    # --- Radiation sub-cycle cache (held heating tendency) ---
    # Zero before the first solve; populated on sub-cycle step 0 (which is
    # always a radiation step) before any held step reads it.
    rad_heating = jnp.zeros((ncol, nlev), dtype=dtype)

    # --- Prognostic aerosol number (opt-in tracer) ---
    # Always materialised as zeros so the pytree is uniform and existing runs
    # are byte-identical; only evolved when the prognostic-aerosol option is on.
    aerosol_number = jnp.zeros((ncol, nlev), dtype=dtype)

    # --- CLUBB sub-grid cloud fraction hand-off (turbulence -> radiation) ---
    # Zeros before the first turbulence step; a PDF turbulence scheme (CLUBB)
    # overwrites it each step and radiation reads it when cloud_scheme="clubb".
    cloud_fraction = jnp.zeros((ncol, nlev), dtype=dtype)

    return PhysicsState(
        tke=tke,
        conv_prog_profile=conv_prog_profile,
        conv_stoch_state=conv_stoch_state,
        gwd_spectrum=gwd_spectrum,
        prng_key=prng_key,
        surface_T_sfc_override=surface_T_sfc_override,
        qke=qke,
        clubb_moments=clubb_moments,
        rad_heating=rad_heating,
        col_index=jnp.arange(ncol, dtype=jnp.int32),
        aerosol_number=aerosol_number,
        cloud_fraction=cloud_fraction,
        # Dynamics-tendency inputs default None (RCAPQADV inert); a
        # process-split driver / SCM writes them per step (see the field
        # docstrings).
        dyn_tendency_T=None,
        dyn_tendency_qv=None,
        # Lagged convective surface precip for the standalone-path Slingo
        # cumulus cloud fraction; zeros before the first convection step.
        conv_precip=jnp.zeros((ncol,), dtype=dtype),
    )


def update_physics_state(phys_state, updates):
    """Rebuild a ``PhysicsState`` from per-field updates returned by sub-physics.

    Co-located with ``PhysicsState`` and ``init_physics_state`` so that adding
    a new prognostic field only requires editing this one module.

    Parameters
    ----------
    phys_state : PhysicsState or None
        Input physics state.  When ``None``, returns ``None`` (the orchestrator
        is running without a prognostic carry).
    updates : dict
        Mapping from field name (``'tke'``, ``'conv_prog_profile'``,
        ``'gwd_spectrum'``) to the updated ``jax.Array`` returned by
        the sub-physics function.  Fields absent from the dict are
        carried over unchanged.

    Returns
    -------
    PhysicsState or None
    """
    if phys_state is None:
        return None
    return PhysicsState(
        tke=updates.get("tke", phys_state.tke),
        conv_prog_profile=updates.get(
            "conv_prog_profile", phys_state.conv_prog_profile
        ),
        conv_stoch_state=updates.get(
            "conv_stoch_state", phys_state.conv_stoch_state
        ),
        gwd_spectrum=updates.get("gwd_spectrum", phys_state.gwd_spectrum),
        prng_key=updates.get("prng_key", phys_state.prng_key),
        surface_T_sfc_override=updates.get(
            "surface_T_sfc_override", phys_state.surface_T_sfc_override,
        ),
        qke=updates.get("qke", phys_state.qke),
        clubb_moments=updates.get("clubb_moments", phys_state.clubb_moments),
        rad_heating=updates.get("rad_heating", phys_state.rad_heating),
        col_index=phys_state.col_index,   # constant identity, never updated
        aerosol_number=updates.get(
            "aerosol_number", phys_state.aerosol_number
        ),
        cloud_fraction=updates.get(
            "cloud_fraction", phys_state.cloud_fraction
        ),
        # Per-step INPUTS: CONSUMED each call, never carried forward.  Default
        # to None (NOT the prior value) so a driver that forgets to refresh
        # them on a later step fails CLOSED — the bridge guard raises on a
        # None carry rather than silently pairing a STALE dynamics tendency
        # with a new post-dynamics state (RCAPQADV staging contract, codex
        # r2).  A driver re-populates them before every convection call.
        dyn_tendency_T=updates.get("dyn_tendency_T", None),
        dyn_tendency_qv=updates.get("dyn_tendency_qv", None),
        # Evolving lag carry (convection writes, radiation reads next step):
        # carried forward unchanged when the step's convection published
        # nothing (schemes without a rain split).
        conv_precip=updates.get("conv_precip", phys_state.conv_precip),
    )
