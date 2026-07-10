"""Driver adapter: cell-centre step over the tiled cube dycore (P4).

Bridges the production driver's cell-centred ``HydrostaticState`` contract to
the sub-face-tiled D-grid step
(:func:`legoesm.parallel.tiled_production_cdgrid.make_tiled_fv3_hydrostatic_step_stage_2d`),
for ``n_devices = 6*kt^2 > 6`` runs where the face-only SPMD shard cannot
decompose further.

Faithfulness: this adapter reuses the SERIAL step's own entry/exit
conversions verbatim — ``center_to_dgrid_vector`` on entry (the
rotation-aware cc→corner interp ``_step_cell_centre`` uses) and
``fv3_to_hydrostatic`` on exit — around the tiled D-grid SSP-RK3 core.  With
the driver-default config the serial ``_step_fv3`` degenerates to exactly
that core: ``_sync_dgrid_boundary`` is a no-op, the (non-implicit) sponge
lives inside ``fv3_hydrostatic_tendencies`` (already in the tiled twin), and
every post-step damp defaults to zero — the loud refusals below pin that
envelope so a config outside it can never silently run different numerics.
The mass/moisture fixers are NOT applied here: the compiled-segment driver
externalizes them (it disables the inner model's ``fix_mass`` and applies the
target-anchored fixer after dynamics), which is the only production caller.
"""

from __future__ import annotations

import jax.numpy as jnp


def dedup_tiled_corners(t, kt: int, nl: int):
    """Reassemble a tiled corner-staggered output to true global corners.

    The tiled step returns corner fields BLOCK-CONCATENATED per tile —
    tile ``(ti, tj)`` occupies rows ``[ti*(nl+1):(ti+1)*(nl+1)]`` and holds
    global corners ``[ti*nl : ti*nl + nl+1]`` — so adjacent tiles carry a
    DUPLICATED shared face (gathered shape ``(F, kt*(nl+1), kt*(nl+1),
    ...)``).  Adjacent tiles compute that shared face bit-identically (the
    capstone gate pins the overlap at ~1e-10), so deduplication is a pure
    slice-drop of each non-first block's first row/col:
    ``(F, kt*(nl+1), ...) -> (F, kt*nl + 1, ...) = (F, n+1, ...)``.
    """
    blk = nl + 1
    rows = [t[:, 0:blk]] + [t[:, i * blk + 1:(i + 1) * blk]
                            for i in range(1, kt)]
    t = jnp.concatenate(rows, axis=1)
    cols = [t[:, :, 0:blk]] + [t[:, :, i * blk + 1:(i + 1) * blk]
                               for i in range(1, kt)]
    return jnp.concatenate(cols, axis=2)


def _refuse(cond: bool, what: str) -> None:
    if cond:
        raise NotImplementedError(
            f"tiled cc step (P4 increment 1): {what} is outside the tiled "
            "base-cut envelope — the tiled D-grid step twins the DEFAULT "
            "_step_fv3 (all post-step damps zero, implicit sponge off, "
            "dynamics-only).  Disable it or run the face-only/serial path."
        )


def _validate_tiled_envelope(cfg, cdgrid, *, inner_fix_mass_ok: bool) -> None:
    """Envelope refusals shared by the single-shot cc step and the closed
    loop (dispatch-hardening — a config outside the proven base cut must
    fail LOUDLY, never silently integrate different numerics).  Defaults
    are all inside the envelope.

    ``inner_fix_mass_ok``: the closed-loop step applies the serial
    post-step dry-mass fixer IN-STAGE (telescoping ``fix_ps_mass``), so
    ``use_conservation_fixer+fix_mass`` is inside ITS envelope; the
    single-shot cc step has no fixer, so there it must be refused.
    """
    _refuse(getattr(cdgrid.base, "duogrid", None) is not None, "duogrid")
    # The serial step honours config.time_integrator; the tiled stage is
    # hardwired SSP-RK3 — only the names that dispatch to the SAME
    # integrator are accepted (codex round-13 HIGH #2).
    _refuse(getattr(cfg, "time_integrator", "ssp_rk3")
            not in ("ssp_rk3", "ssp3", "rk3"),
            f"time_integrator={getattr(cfg, 'time_integrator', None)!r} "
            "(tiled step is SSP-RK3)")
    # The inner mass fixer: the compiled-segment driver externalizes it
    # (inner model runs fix_mass=False); a DIRECT default-config caller
    # would silently lose the per-step fix_ps_mass (codex round-13 HIGH
    # #1) — refuse so the caller must disable it explicitly.  The closed
    # loop instead RUNS the tiled in-stage fixer for this config.
    if not inner_fix_mass_ok:
        _refuse(bool(getattr(cfg, "use_conservation_fixer", False))
                and bool(getattr(cfg, "fix_mass", False)),
                "the inner mass fixer (use_conservation_fixer+fix_mass; the "
                "segment driver applies the target-anchored fixer OUTSIDE "
                "the step — pass a config with fix_mass=False)")
    # Every non-default tendency/damping term the tiled base cut omits
    # (codex round-13 HIGH #3 — the tiled module's own scope note).
    for _f, _lbl in (
        ("damp_v", "damp_v del-6 damping"),
        ("damp_v_d_con", "damp_v_d_con heating"),
        ("div_damp_d_con", "div_damp_d_con"),
        ("corner_div_damp_d_con", "corner_div_damp_d_con"),
        ("div_damp_coeff", "divergence damping"),
        ("corner_div_damp_d2_bg", "corner div damping (d2)"),
        ("corner_div_damp_d4_bg", "corner div damping (d4)"),
        ("A_h", "Laplacian viscosity A_h"),
        ("smagorinsky_cs", "Smagorinsky viscosity"),
        ("hyperdiff_coeff", "hyperdiffusion"),
        ("hyperdiff_ps_coeff", "p_s hyperdiffusion"),
        ("T_diss_coeff", "T dissipation"),
        ("implicit_grav_wave_damping",
         "implicit gravity-wave damping (the p_s damp + p_floor clamp "
         "post-step)"),
        # Ray_fast post-step Rayleigh damping (u_d/v_d *= rff above
        # rf_cutoff_pa) — a serial post-step op neither tiled path runs; a
        # nonzero value would silently diverge (envelope hardening, this
        # increment).
        ("rf_tau_days", "Ray_fast Rayleigh damping (rf_tau_days)"),
    ):
        _refuse(getattr(cfg, _f, 0.0) > 0.0, _lbl)
    _refuse(bool(getattr(cfg, "sponge_implicit", False)),
            "the implicit sponge")
    _refuse(bool(getattr(cfg, "use_fv3_a2b_zeta_corner", False)),
            "use_fv3_a2b_zeta_corner")
    # Serial tracer/moisture semantics (flux-form substep, post-RK3
    # moisture handling) are outside BOTH tiled cuts — the tiled moist
    # contract is the injected column_physics_fn path only (codex).
    _refuse(bool(getattr(cfg, "moisture_flux_form", False)),
            "moisture_flux_form tracer transport")


def make_tiled_cc_step(model, mesh, kt: int, dt: float):
    """Build ``step(state) -> state`` running one ``dt`` of the cube
    hydrostatic dycore sub-face-tiled on a ``(6, kt, kt)`` mesh.

    Parameters
    ----------
    model : CDGridPrimitiveEquationModel
        The production cube model (supplies ``cdgrid``, ``sigma_coord``,
        ``grid.n`` and the config whose envelope is validated).
    mesh : jax.sharding.Mesh
        ``(6, kt, kt)`` device mesh (``create_device_mesh`` with sub-face
        tiling).
    kt : int
        Tiles per face edge (``n_devices == 6*kt**2``).
    dt : float
        Time step [s] — closed over by the tiled stage (static).

    Returns
    -------
    step : callable
        ``step(state: HydrostaticState) -> HydrostaticState`` (cell-centred
        in and out; dynamics only — no physics_fn, no fixers).
    """
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_fv3_hydrostatic_step_stage_2d,
    )
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        FV3HydrostaticState, fv3_to_hydrostatic,
    )

    cfg = model.config
    cdgrid = model.cdgrid
    _validate_tiled_envelope(cfg, cdgrid, inner_fix_mass_ok=False)

    n = int(model.grid.n)
    nlev = int(model.sigma_coord.n_levels)
    tiled = make_tiled_fv3_hydrostatic_step_stage_2d(
        mesh, cdgrid, model.sigma_coord, n, kt, nlev,
        p_floor=float(cfg.p_floor), dt=float(dt),
        # The (default-ON) non-implicit Rayleigh sponge: the serial
        # tendencies apply it in-tendency (step 13); the tiled bodies take
        # it from these knobs (measured 6.2e-6 u drift without, job 8689100).
        sponge_sigma=float(getattr(cfg, "sponge_sigma", 0.0)),
        sponge_tau_sec=float(getattr(cfg, "sponge_tau_sec", 0.0)),
    )

    from legoesm.core.precision import cast_pytree

    def step(state):
        # Entry: the SAME rotation-aware cc→corner vector interp the serial
        # _step_cell_centre uses (a scalar interp would re-inject the
        # cube-edge vorticity imprint).  Runs in the STORAGE dtype, exactly
        # like serial (the cast happens inside _step_fv3, i.e. AFTER this
        # conversion).
        u_d, v_d = center_to_dgrid_vector(
            state.u.data, state.v.data, cdgrid)
        # Mirror the serial _step_fv3 entry cast EXACTLY: it casts the
        # integrated state to the COMPUTE dtype before the RK3 core
        # (primitive_eq_cdgrid:1378, cast_pytree(state, None, "compute")).
        # Feeding the storage dtype (f64 under x64) into the tiled core
        # instead accumulated a pure precision drift vs serial (~5.6e-6
        # rel u over 3 steps, jobs 8689100/8693550).
        u_d, v_d, T_in, ps_in, phis_in = cast_pytree(
            (u_d, v_d, state.T.data, state.p_s.data, state.phis.data),
            None, "compute")
        u_d2, v_d2, T2, ps2 = tiled(u_d, v_d, T_in, ps_in, phis_in)
        # The tiled corner outputs carry the duplicated shared tile face
        # ((F, kt*(nl+1), kt*(nl+1), nlev)) — reassemble to true global
        # corners before the serial exit conversion.  cc outputs (T, p_s)
        # partition exactly (no dedup).
        nl = n // kt
        u_d2 = dedup_tiled_corners(u_d2, kt, nl)
        v_d2 = dedup_tiled_corners(v_d2, kt, nl)
        fv3_new = FV3HydrostaticState(
            u_d=state.u.replace(data=u_d2, name="u_d"),
            v_d=state.v.replace(data=v_d2, name="v_d"),
            T=state.T.replace(data=T2),
            p_s=state.p_s.replace(data=ps2),
            phis=state.phis,
            tracers=getattr(state, "tracers", None),
        )
        # Exit: the serial step's own corner→centre conversion.
        return fv3_to_hydrostatic(fv3_new, cdgrid)

    return step


#: Tracer names + packing order of the tiled moist q_pack (matches the moist
#: step stages' ``[q_v, q_c, q_r]`` contract and the Kessler bridge).
_TILED_TRACERS = ("q_v", "q_c", "q_r")


def make_tiled_cc_loop(model, mesh, kt: int, dt: float,
                       column_physics_fn=None):
    """Closed-loop tiled stepping: state stays TILE-SHARDED across steps.

    The single-shot ``make_tiled_cc_step`` re-replicates per call (its exit
    dedup is a full-cube gather) — valid for one-step parity/bench probes,
    invalid as a production loop.  This builder returns the persistent
    variant over the blocked layout
    (:func:`legoesm.parallel.tiled_production_cdgrid.
    make_tiled_fv3_hydrostatic_step_blocked_2d`):

    ``(enter, step, exit_)`` with

    * ``enter(state: HydrostaticState) -> blocked`` — ONE-TIME layout
      conversion: the serial entry conversions verbatim
      (``center_to_dgrid_vector`` + the serial compute-dtype cast), then
      corner block-expansion + device placement on the tiled mesh.
    * ``step(blocked) -> blocked`` — one ``dt``; input layout == output
      layout, so ``s = step(s)`` iterates with NO per-step gather.
    * ``exit_(blocked, template_state) -> HydrostaticState`` — dedup + the
      serial ``fv3_to_hydrostatic`` exit (``template_state`` supplies the
      Field wrappers, normally the state passed to ``enter``); for
      I/O/diagnostics only (a global gather — never call it inside the
      loop).

    MOIST (``column_physics_fn`` given — e.g. the Kessler bridge
    ``make_kessler_column_physics_fn``): the state MUST carry exactly the
    ``q_v``/``q_c``/``q_r`` tracers; ``enter`` packs them into the blocked
    ``q_pack`` (cc 5-D, ``[q_v, q_c, q_r]`` order), ``step`` threads it
    through the moist blocked step (dynamics advection + injected column
    physics + the post-step ``max(q, 0)`` floor), and ``exit_`` unpacks
    them back into the tracer Fields.  Dry mode refuses tracer-carrying
    states (silently freezing them would be divergence-by-omission).

    Unlike the single-shot adapter, the default production config's
    ``use_conservation_fixer+fix_mass`` is INSIDE this envelope: the
    blocked step applies the serial telescoping post-step ``fix_ps_mass``
    in-stage (the anchor path threads the per-call pre-step mass under an
    outer jit, which telescopes identically — primitive_eq_cdgrid step()
    docstring).  ``zero_mean_ps_tendency`` composes exactly as serial: with
    the end-step fixer active the serial step SKIPS the per-stage zero-mean
    (``_apply_zero_mean_per_stage=False``), which is the blocked body's
    base cut; without the fixer that config is outside the envelope —
    refused loudly below.
    """
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.core.precision import cast_pytree
    from legoesm.parallel.tiled_production_cdgrid import (
        expand_corners_to_blocks,
        make_tiled_fv3_hydrostatic_step_blocked_2d,
    )
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        FV3HydrostaticState, fv3_to_hydrostatic,
    )

    cfg = model.config
    cdgrid = model.cdgrid
    _validate_tiled_envelope(cfg, cdgrid, inner_fix_mass_ok=True)
    _fix_mass = (bool(getattr(cfg, "use_conservation_fixer", False))
                 and bool(getattr(cfg, "fix_mass", False)))
    # Per-stage zero-mean runs only when the end-step fixer is OFF
    # (primitive_eq_cdgrid._apply_zero_mean_per_stage); the blocked base
    # cut has no per-stage zero-mean, so that combination must refuse.
    _refuse(bool(getattr(cfg, "zero_mean_ps_tendency", False))
            and not _fix_mass,
            "zero_mean_ps_tendency without the end-step mass fixer (the "
            "serial step then zero-means dp_s/dt EVERY RK stage; the "
            "blocked base cut does not)")

    n = int(model.grid.n)
    nlev = int(model.sigma_coord.n_levels)
    nl = n // kt
    _moist = column_physics_fn is not None
    tiled = make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, model.sigma_coord, n, kt, nlev,
        p_floor=float(cfg.p_floor), dt=float(dt),
        sponge_sigma=float(getattr(cfg, "sponge_sigma", 0.0)),
        sponge_tau_sec=float(getattr(cfg, "sponge_tau_sec", 0.0)),
        column_physics_fn=column_physics_fn,
        fix_mass=_fix_mass,
    )

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    cz = NamedSharding(mesh, P("face", "tile_i", "tile_j", None))
    co = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    cz5 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None, None))

    def enter(state):
        """cc HydrostaticState -> blocked pytree (dict) on the tiled mesh."""
        tracers = getattr(state, "tracers", None)
        if _moist:
            # The moist blocked step advances EXACTLY [q_v, q_c, q_r];
            # extra/missing tracers would silently freeze/invent fields.
            got = tuple(sorted(tracers)) if tracers else ()
            if got != tuple(sorted(_TILED_TRACERS)):
                raise NotImplementedError(
                    "make_tiled_cc_loop (moist): state must carry exactly "
                    f"the {_TILED_TRACERS} tracers (the tiled q_pack "
                    f"contract); got {got or None}.")
        elif tracers:
            # Dry loop must not silently freeze a tracer dict across steps
            # while serial advances/floors it (codex BLOCKER) — refuse.
            raise NotImplementedError(
                "make_tiled_cc_loop: state carries tracers but no "
                "column_physics_fn was given — the dry loop would freeze "
                "them (divergence-by-omission). Pass the column physics "
                "(e.g. make_kessler_column_physics_fn) or drop the tracers.")
        u_d, v_d = center_to_dgrid_vector(
            state.u.data, state.v.data, cdgrid)
        # Serial _step_fv3 entry cast (see make_tiled_cc_step's note).
        u_d, v_d, T_in, ps_in, phis_in = cast_pytree(
            (u_d, v_d, state.T.data, state.p_s.data, state.phis.data),
            None, "compute")
        blocked = {
            "u_d": jax.device_put(
                expand_corners_to_blocks(u_d, kt, nl), cz),
            "v_d": jax.device_put(
                expand_corners_to_blocks(v_d, kt, nl), cz),
            "T": jax.device_put(T_in, cz),
            "p_s": jax.device_put(ps_in, co),
            "phis": jax.device_put(phis_in, co),
        }
        if _moist:
            q_pack = jnp.stack(
                [state.tracers[nm].data for nm in _TILED_TRACERS], axis=-1)
            (q_pack,) = cast_pytree((q_pack,), None, "compute")
            blocked["q_pack"] = jax.device_put(q_pack, cz5)
        return blocked

    def step(blocked):
        if _moist:
            u2, v2, T2, ps2, q2 = tiled(
                blocked["u_d"], blocked["v_d"], blocked["T"],
                blocked["p_s"], blocked["phis"], blocked["q_pack"])
            return {"u_d": u2, "v_d": v2, "T": T2, "p_s": ps2,
                    "phis": blocked["phis"], "q_pack": q2}
        u2, v2, T2, ps2 = tiled(blocked["u_d"], blocked["v_d"],
                                blocked["T"], blocked["p_s"],
                                blocked["phis"])
        return {"u_d": u2, "v_d": v2, "T": T2, "p_s": ps2,
                "phis": blocked["phis"]}

    def exit_(blocked, template_state):
        """Blocked -> cc HydrostaticState (GLOBAL GATHER — I/O only)."""
        u_d = dedup_tiled_corners(blocked["u_d"], kt, nl)
        v_d = dedup_tiled_corners(blocked["v_d"], kt, nl)
        tracers = getattr(template_state, "tracers", None)
        if _moist:
            tracers = {
                nm: template_state.tracers[nm].replace(
                    data=blocked["q_pack"][..., i])
                for i, nm in enumerate(_TILED_TRACERS)
            }
        fv3 = FV3HydrostaticState(
            u_d=template_state.u.replace(data=u_d, name="u_d"),
            v_d=template_state.v.replace(data=v_d, name="v_d"),
            T=template_state.T.replace(data=blocked["T"]),
            p_s=template_state.p_s.replace(data=blocked["p_s"]),
            phis=template_state.phis,
            tracers=tracers,
        )
        return fv3_to_hydrostatic(fv3, cdgrid)

    return enter, step, exit_


# Public alias for the tiled operator-split driver lane (the
# no-private-cross-imports ratchet's sanctioned surface).  Same object.
validate_tiled_envelope = _validate_tiled_envelope
