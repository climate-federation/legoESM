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
    # Envelope refusals (dispatch-hardening — a config outside the proven
    # base cut must fail LOUDLY, never silently integrate different
    # numerics).  Defaults are all inside the envelope.
    _refuse(getattr(cdgrid.base, "duogrid", None) is not None, "duogrid")
    _refuse(getattr(cfg, "damp_v", 0.0) > 0.0, "damp_v del-6 damping")
    _refuse(getattr(cfg, "damp_v_d_con", 0.0) > 0.0, "damp_v_d_con heating")
    _refuse(getattr(cfg, "div_damp_d_con", 0.0) > 0.0, "div_damp_d_con")
    _refuse(getattr(cfg, "corner_div_damp_d_con", 0.0) > 0.0,
            "corner_div_damp_d_con")
    _refuse(bool(getattr(cfg, "sponge_implicit", False)),
            "the implicit sponge")
    _refuse(getattr(cfg, "implicit_grav_wave_damping", 0.0) > 0.0,
            "implicit gravity-wave damping (the p_s damp + p_floor clamp "
            "post-step)")

    n = int(model.grid.n)
    nlev = int(model.sigma_coord.n_levels)
    tiled = make_tiled_fv3_hydrostatic_step_stage_2d(
        mesh, cdgrid, model.sigma_coord, n, kt, nlev,
        p_floor=float(cfg.p_floor), dt=float(dt),
    )

    def step(state):
        # Entry: the SAME rotation-aware cc→corner vector interp the serial
        # _step_cell_centre uses (a scalar interp would re-inject the
        # cube-edge vorticity imprint).
        u_d, v_d = center_to_dgrid_vector(
            state.u.data, state.v.data, cdgrid)
        u_d2, v_d2, T2, ps2 = tiled(
            u_d, v_d, state.T.data, state.p_s.data, state.phis.data)
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
