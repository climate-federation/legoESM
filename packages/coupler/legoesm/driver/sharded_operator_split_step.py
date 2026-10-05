"""Lat-band-SPMD operator-split atmosphere step.

The single-controller multi-device twin of the serial ``_single_step``
(``compiled_segments.build_segment_fn``): it runs the SAME operator-split
integration — dynamics -> dry-mass fixer -> ``step_unified`` physics -> Euler
write-back -> saturation/moisture fixer/smoothing/Rayleigh -> carry pack — on a
``SegmentCarry`` sharded by LATITUDE BAND over a 1-D ``"lat"`` mesh, keeping the
carry sharded across steps (no per-step gather).

Correctness rests on the operator split being decomposition-friendly:

* **Physics is PURELY column-local** (``split_physics_single_rank`` -> the
  ``step_unified`` column kernels): each column is independent, so a band's
  contiguous ``(ncol,)`` slice is computed with NO collectives and is
  bit-identical to the serial full-grid result (decomposition-INVARIANT).
* **The only global reductions** are OUTSIDE the physics — the dry-mass fixer
  and the moisture fixer (``global_area_sum``, now lat-band-psum-aware) and the
  moisture hyperdiffusion halo (``hyperdiffusion_3d``, backend-dispatched
  ppermute). All fire correctly inside the armed ``shard_map``.
* **The per-step cell<->C-grid round trip** the serial ``model.step``
  performs is reproduced band-local: cell->C-grid winds via the halo-aware
  :func:`cell_to_cgrid_winds_spmd` (interior cuts lifted from the neighbour
  band, physical poles zeroed), the band C-grid dynamics via
  ``_step_cgrid_impl`` (band geometry, as in ``make_sharded_atm_latlon_step``),
  and C-grid->cell via ``cgrid_to_hydrostatic`` (``_face_to_cell_v`` on the
  band's reconstructed faces = band-safe).

Lives in the driver package (not atmosphere): it composes the atm dynamics
(``_step_cgrid_impl`` + the lat-band machinery) with the driver's operator-split
physics helpers (``split_physics_single_rank`` / ``finalize_split_step``),
and driver->atmosphere is the established import direction.

The refused-loudly follow-ups mirror the serial-vs-SPMD contract: a
DECOMPOSITION-VARIANT PRNG (stochastic Bechtold ``normal(key, (ncol,))`` keyed
by global column position, or the 3-D Monte-Carlo radiation with horizontal
photon transport) is NOT column-local and must be re-keyed by absolute column
index before it can ride this lane; the caller (the driver) refuses those
upstream.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel.geometry_consistency import (
    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked, checked_replicated_put,
    coerce_bool, coerce_count, config_digest48, mesh_axis_terms, name_digest48,
    tree_schema_digest48)
from legoesm.parallel.latlon_spmd import (
    cell_to_cgrid_winds_spmd, spmd_pole_end_masks, activate_latlon_spmd_halo)
from legoesm.parallel.shard_map_compat import shard_map
from legoesm.forcing.time_utils import day_to_calendar
from legoesm.core.conservation import fix_ps_mass_target
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    build_band_grids_atm, atm_grid_array_field_names, lat_spec)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState, cgrid_to_hydrostatic)
from legoesm.driver.compiled_segments import (
    SegmentCarry, SegmentForcing, GRID_SHAPED_FORCING_FIELDS,
    split_physics_single_rank, finalize_split_step, ghg_array_to_dict)


# Carry fields that are per-column state sharded on their leading axis under the
# lat band — grid-shaped ``(n_lat, ...)`` OR flattened ``(ncol=n_lat*n_lon, ...)``
# (the lat-major flatten's contiguous split IS the band, so BOTH shard P("lat")
# on axis 0). Everything not listed is a replicated scalar / None / land_ml.
_CARRY_SCALAR_FIELDS = frozenset(
    ("step_index", "target_moisture", "target_mass", "max_cfl"))
_CARRY_REPLICATED_FIELDS = frozenset(("land_ml",))


def _carry_valid_leads(carry: SegmentCarry):
    """The two leading dims that shard P("lat") — ``(n_lat, ncol=n_lat*n_lon)`` —
    read off the grid-shaped ``T`` leaf ``(n_lat, n_lon, nlev)``. ONLY these two
    align with the lat-major band split; a coincidentally device-divisible OTHER
    axis (e.g. a ``(nlev,)`` leaf) is NOT a lat band and must replicate (codex)."""
    ref = carry.T
    n_lat, n_lon = int(ref.shape[0]), int(ref.shape[1])
    return (n_lat, n_lat * n_lon)


def _lat_shardable(arr, n_dev, valid_leads):
    """Shard P("lat") on axis 0 iff the leading dim is one of ``valid_leads``
    (n_lat / ncol) AND divisible by the device count — NOT any divisible axis."""
    return (arr.ndim >= 1 and arr.shape[0] in valid_leads
            and arr.shape[0] % n_dev == 0)


def _carry_leaf_spec(name, arr, n_dev, valid_leads):
    """PartitionSpec for one carry leaf: P() for scalars / land_ml / leaves whose
    leading dim is not a lat band, else P("lat", None, ...) (grid-shaped OR
    flattened ncol)."""
    if arr is None:
        return None
    if (name in _CARRY_SCALAR_FIELDS or name in _CARRY_REPLICATED_FIELDS
            or not _lat_shardable(arr, n_dev, valid_leads)):
        return P()
    return lat_spec(arr)


def _carry_specs(carry: SegmentCarry, n_dev) -> SegmentCarry:
    """A SegmentCarry-shaped pytree of PartitionSpecs (None where the field is
    None) for shard_map in/out_specs."""
    vl = _carry_valid_leads(carry)
    return SegmentCarry(**{
        name: _carry_leaf_spec(name, getattr(carry, name), n_dev, vl)
        for name in SegmentCarry._fields
    })


def _forcing_leaf_spec(name, arr, n_dev):
    """Only the KNOWN grid-shaped forcing fields (all ``(n_lat, n_lon, ...)``
    leading ``n_lat``) shard P("lat"); the allowlist is the guard, so a
    replicated small vector (``solar_weights (14,)``, ``ghg_vmr``) stays P()."""
    if arr is None:
        return None
    if name in GRID_SHAPED_FORCING_FIELDS and arr.shape[0] % n_dev == 0:
        return P("lat", *((None,) * (arr.ndim - 1)))
    return P()


def _forcing_specs(forcing: SegmentForcing, n_dev) -> SegmentForcing:
    return SegmentForcing(**{
        name: _forcing_leaf_spec(name, getattr(forcing, name), n_dev)
        for name in SegmentForcing._fields
    })


# Ordered flag names for the operator-split MESH+TREE gate (scatter bridges and
# the returned callable). STATIC tuple: fixed width, never rank-local.
_OPSPLIT_MESH_ENTRY_FLAGS = (
    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
    "has_tree", "tree_schema", "has_tree2", "tree2_schema",
)


def _agree_opsplit_mesh_entry(mesh, tree=None, tree2=None, *,
                              where: str) -> None:
    """Agree the mesh AND pytree LEAF SCHEMAS before a scatter / a sharded call.

    #1362 round 4, blockers 6-7.  These entries ``device_put`` full global
    arrays onto a cross-process ``NamedSharding`` — serviced by an ALL-GATHER
    (documented on ``latlon_spmd.shard_leaf``) — one per NON-``None`` leaf, and
    they return early on ``mesh is None``.  So both the mesh and the leaf
    schedule are rank-local inputs to a collective, and the returned
    ``sharded_split_step`` additionally derives its shard_map specs from the
    carry/forcing structure.
    """
    names, sizes = mesh_axis_terms(mesh)
    assert_flags_agree(_OPSPLIT_MESH_ENTRY_FLAGS, (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        float(tree is not None),
        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
        float(tree2 is not None),
        tree_schema_digest48(tree2) if tree2 is not None else FLAG_ABSENT,
    ), context=where)


def shard_operator_split_carry(carry: SegmentCarry, mesh) -> SegmentCarry:
    """Commit a SegmentCarry to the lat-band layout (grid-shaped + flattened
    leaves P("lat"); scalars / land_ml replicated). ``mesh=None`` -> unchanged."""
    # FIRST statement: the `mesh is None` return SKIPS every per-leaf put,
    # and each put is an all-gather under multi-process (#1362 r4, blocker 6).
    _agree_opsplit_mesh_entry(mesh, carry,
                              where="shard_operator_split_carry")
    if mesh is None:
        return carry
    n_dev = mesh.devices.size
    vl = _carry_valid_leads(carry)
    return SegmentCarry(**{
        name: (None if (v := getattr(carry, name)) is None
               else jax.device_put(v, NamedSharding(
                   mesh, _carry_leaf_spec(name, v, n_dev, vl))))
        for name in SegmentCarry._fields
    })


def shard_operator_split_forcing(forcing: SegmentForcing, mesh) -> SegmentForcing:
    """Commit a SegmentForcing to the lat-band layout. ``mesh=None`` -> unchanged."""
    # FIRST statement: same rank-local early return + per-leaf all-gather as
    # the carry twin (#1362 round 4, blocker 6).
    _agree_opsplit_mesh_entry(mesh, forcing,
                              where="shard_operator_split_forcing")
    if mesh is None:
        return forcing
    n_dev = mesh.devices.size
    return SegmentForcing(**{
        name: (None if (v := getattr(forcing, name)) is None
               else jax.device_put(v, NamedSharding(
                   mesh, _forcing_leaf_spec(name, v, n_dev))))
        for name in SegmentForcing._fields
    })


def _need_rad_and_time(step_index, start_day, dt, rad_update_steps):
    """Reproduce the serial prologue's radiation cadence + per-step solar time
    (compiled_segments._single_step)."""
    if rad_update_steps <= 1:
        need_rad = jnp.bool_(True)
    else:
        need_rad = ((step_index + 1) % rad_update_steps) == 0
    abs_day = start_day + (step_index + 1) * dt / 86400.0
    doy, sod = day_to_calendar(abs_day)
    return need_rad, doy, sod


# Ordered flag names for the operator-split SPMD entry gate. STATIC tuple: the
# payload width is fixed by this literal, never by rank-local data.
_OPSPLIT_SPMD_ENTRY_FLAGS = (
    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
    "grid_n_lat", "grid_n_lon", "geom_schema", "fold_active",
    "has_polar_mask", "has_polar_mask_v",
    "model_config_digest", "statics_digest", "statics_schema",
    "fix_mass", "rad_update_steps", "n_ghg_keys", "ghg_keys",
)


def _agree_opsplit_spmd_entry(model, mesh, statics, *, fix_mass,
                              rad_update_steps, ghg_keys, where: str) -> None:
    """Agree every rank-local input, as the FIRST statement of this factory.

    #1362 round 4.  This is the THIRD lat-band SPMD lane (after
    ``sharded_atm_latlon_step`` and ``sharded_ocean_step``) and it had the
    SAME two defects, unfixed: it recomputes the band geometry per process
    from ``build_band_grids_atm`` and hands it to a REPLICATED
    ``device_put`` (whose bit-identity assert is what #1362 trips), and it
    performs a rank-local ``mesh is None`` return plus a north-fold
    ``raise`` before reaching any collective.  Rounds 1-3 fixed the two
    lanes where the defect was REPORTED; per legoESM's "fix the class, not
    the instance" rule this lane gets the identical treatment.

    ``has_polar_mask`` is the sharpest entry: ``model._polar_mask`` decides
    whether the ``__polar_mask``/``__polar_mask_v`` fields EXIST in the
    stacked geometry dict, so a per-process difference changes the field
    LIST and would desynchronise the per-field gathers instead of failing
    cleanly.  ``fix_mass`` / ``rad_update_steps`` / the GHG key ORDER select
    different compiled programs (Python-level feature gating on static
    values), so processes disagreeing there run different graphs.
    """
    # Defensive attribute reads: nothing in the payload build may raise before
    # the collective (see the atm twin for the full rule).
    grid = getattr(model, "grid", None)
    fold = getattr(grid, "fold", None)
    names, sizes = mesh_axis_terms(mesh)
    problems = []

    def _count(value, label, absent=FLAG_ABSENT):
        payload, problem = coerce_count(value, absent=absent)
        if problem is not None:
            problems.append((label, problem))
        return payload

    def _flag(value, label):
        payload, problem = coerce_bool(value, absent=FLAG_ABSENT)
        if problem is not None:
            problems.append((label, problem))
        return payload

    # `ghg_keys or ()` would evaluate the truthiness of an ARRAY and raise
    # before the collective (codex round-4, blocker 2); test for None instead.
    try:
        keys = () if ghg_keys is None else tuple(
            str(k) for k in ghg_keys)
    except Exception:                       # pragma: no cover - defensive
        keys = ("<unreadable ghg_keys>",)
        problems.append(("ghg_keys", "is not an iterable of species names"))
    flags = (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
        tree_schema_digest48(grid),
        float(bool(fold is not None and getattr(fold, "is_active", False))),
        float(getattr(model, "_polar_mask", None) is not None),
        # The body conditions on `_polar_mask` and then slices
        # `_polar_mask_v` unconditionally (codex round-4, blocker 4).
        float(getattr(model, "_polar_mask_v", None) is not None),
        config_digest48(getattr(model, "config", None)),
        # `statics` was not agreed AT ALL: `statics.fix_moisture` controls a
        # global-area reduction, `qv_smooth_coeff` / `owned_mask` gate loud
        # rank-local refusals, and `dt` sets the trajectory (codex round-4,
        # blocker 3). Digest its static scalars AND its leaf schema.
        config_digest48(statics),
        tree_schema_digest48(statics),
        _flag(fix_mass, "fix_mass"),
        _count(rad_update_steps, "rad_update_steps", absent=0.0),
        float(len(keys)),
        name_digest48(keys),
    )
    assert_flags_agree(_OPSPLIT_SPMD_ENTRY_FLAGS, flags, context=where)
    # AFTER the collective only, so the refusal is symmetric on every rank
    # (see the atm twin's rationale: a raise while assembling the payload
    # kills one process while its peers block in process_allgather).
    for label, problem in problems:
        raise ValueError(f"{where}: {label} {problem}")
    if (getattr(model, "_polar_mask", None) is not None
            and getattr(model, "_polar_mask_v", None) is None):
        raise ValueError(
            f"{where}: model._polar_mask is set but model._polar_mask_v is "
            f"None; the band geometry build slices BOTH, so this would fail "
            f"mid-build instead of here.")


def make_sharded_operator_split_step(
    model, mesh, statics, *, fix_mass, rad_update_steps, start_day,
    ghg_keys=None,
):
    """Build ``sharded_split_step(carry, forcing) -> carry`` — one operator-split
    atmosphere step, lat-band-sharded (or serial when ``mesh is None``).

    ``statics`` is the GLOBAL :class:`_SplitStepStatics` (grid/lat/lon/forcing
    are swapped for the band's inside the body). ``fix_mass`` / ``rad_update_steps``
    / ``start_day`` mirror the serial ``build_segment_fn`` prologue. ``dt`` comes
    from ``statics.dt``.

    ``ghg_keys`` (optional): the GHG species order (``GHG_SPECIES_ORDER`` subset)
    used to rebuild ``ghg_vmr_override`` from the RUNTIME ``forcing.ghg_vmr`` each
    step — exactly as the serial ``_make_single_step`` does. The serial statics
    bakes ``ghg_vmr_override`` from the segment forcing; the SPMD lane threads a
    per-segment ``forcing`` through the SAME step object, so a baked override
    would go stale when GHG varies segment-to-segment (transient CMIP6). Passing
    ``ghg_keys`` recomputes it per step from the sharded forcing (correct for a
    time-varying dataset); ``None`` (default) leaves ``statics.ghg_vmr_override``
    untouched — byte-identical to a fixed-GHG / no-GHG (e.g. gray-radiation) run.

    CONTRACT — ``statics.step_unified`` MUST be built on a BAND grid, not the
    global grid: the body does NOT rebuild it per band (``dataclasses.replace``
    swaps grid/lat/lon/forcing but leaves ``step_unified``), and
    ``PhysicsPipeline.build_step_unified`` bakes ``ncol`` into the ColumnAdapter.
    A global-``ncol`` ``step_unified`` would mis-reshape the band's ``nl*n_lon``
    columns. The bands are UNIFORM, so ONE band-grid build serves every band —
    the driver (Phase 3) builds ``step_unified`` on ``build_band_grids_atm(grid,
    n_dev)[0]``. ``mesh=None`` (serial) uses the full grid, so its
    ``step_unified`` must be the global build. (The 2b-main parity test uses a
    shape-agnostic column-local mock, sidestepping this — the real-physics
    end-to-end gate is the driver subprocess parity.)
    """
    # FIRST statement: agree every rank-local input before ANY rank-local
    # check can raise or return (#1362 round 4 -- this lane was the third,
    # unfixed instance of the class).
    _agree_opsplit_spmd_entry(model, mesh, statics, fix_mass=fix_mass,
                              rad_update_steps=rad_update_steps,
                              ghg_keys=ghg_keys,
                              where="make_sharded_operator_split_step")
    dt = statics.dt
    sigma_coord = model.sigma_coord

    def _one_step(carry, band_geom, forcing_band,
                  polar_mask=None, polar_mask_v=None):
        """Dynamics -> mass fixer -> column-local physics -> finalize, on the
        band (or the full domain when band_geom is the global grid). The band's
        ``_step_cgrid_impl`` produces C-grid faces whose interior-cut boundary
        rows agree across bands to the FV-PPM cut bound (the same
        boundary-order residual the dynamics-only SPMD step carries), so the
        C-grid->cell average is band-safe with no extra reconciliation.

        ``polar_mask`` / ``polar_mask_v`` are the BAND-sliced polar-filter masks
        under sharding (``None`` -> ``_step_cgrid_impl`` resolves the model's
        GLOBAL mask, correct only for the full-domain / serial path — passing the
        band slices is required, else a ``use_polar_filter`` model would filter a
        band with the global-length mask)."""
        # --- Dynamics: cell -> C-grid (halo-aware winds) -> band step -> cell ---
        u_face, v_face = cell_to_cgrid_winds_spmd(carry.u, carry.v)
        cstate = CGridLatLonHydrostaticState(
            u=u_face, v=v_face, T=carry.T, p_s=carry.p_s, phis=carry.phis,
            tracers={})
        out_cgrid, _ = model._step_cgrid_impl(
            cstate, dt, physics_fn=None, phys_state=None,
            grid=band_geom, sigma_coord=sigma_coord,
            polar_mask=polar_mask, polar_mask_v=polar_mask_v,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        hs_out = cgrid_to_hydrostatic(out_cgrid, band_geom)
        T_new, u_new = hs_out.T.data, hs_out.u.data
        v_new, p_s_new = hs_out.v.data, hs_out.p_s.data

        # --- Dry mass fixer (global_area_sum is lat-band-psum-aware) ---
        if fix_mass:
            p_s_new = fix_ps_mass_target(
                p_s_new, carry.target_mass, band_geom, owned_mask=None)

        # --- Physics (column-local) + shared finalize tail ---
        need_rad, doy, sod = _need_rad_and_time(
            carry.step_index, start_day, dt, rad_update_steps)
        # Physics lat/lon are the 2D (nl, n_lon) CELL fields (grid_lat/grid_lon),
        # which the column adapter flattens to (ncol,) — NOT the grid's 1D lat/lon
        # axes.  The band grid carries its own sliced grid_lat/grid_lon (== the
        # global field's band rows), and for the serial path band_geom is the
        # full grid so these ARE the global 2D fields.  Using band_geom.lat (1D)
        # here silently mis-shapes the radiation column lat (flatten_2d error).
        band_statics = dataclasses.replace(
            statics, grid=band_geom,
            lat=band_geom.grid_lat, lon=band_geom.grid_lon,
            forcing=forcing_band)
        # Rebuild ghg_vmr_override from THIS segment's forcing (the serial
        # _make_single_step derives it from forcing.ghg_vmr; the baked statics
        # value would freeze GHG for the whole run). None ghg_keys -> untouched.
        if ghg_keys is not None:
            band_statics = dataclasses.replace(
                band_statics,
                ghg_vmr_override=ghg_array_to_dict(
                    forcing_band.ghg_vmr, ghg_keys))
        lz = split_physics_single_rank(
            carry, T_new, u_new, v_new, p_s_new, need_rad, doy, sod,
            band_statics)
        return finalize_split_step(carry, lz, band_statics)

    if mesh is None:                       # serial / single-device
        def serial_step(carry, forcing):
            return _one_step(carry, model.grid, forcing)
        return serial_step

    # --- lat-band SPMD ---
    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    grid = model.grid
    fold = getattr(grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise NotImplementedError(
            "operator-split SPMD: tripole north-fold is a follow-up.")

    band_grids = build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    afn = atm_grid_array_field_names(template)
    rep = NamedSharding(mesh, P())
    raw = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids], 0)
        for name in afn
    }
    # Per-band polar-filter masks (only when the filter is on): slice the global
    # masks [r*nl:(r+1)*nl] (cell) / [r*nl:r*nl+nl+1] (v-face stagger) and stack,
    # exactly as make_sharded_atm_latlon_step does. Absent otherwise -> _body
    # passes None -> _step_cgrid_impl resolves self._polar_mask (also None).
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)], 0)
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            0)
    # #1362 (round 4): the band geometry above is RECOMPUTED per process from
    # the same config, and per-process XLA autotuning on device-derived grid
    # fields makes the last ULPs differ at larger sizes -- which trips the
    # bit-identical assert inside the REPLICATED ``device_put`` below.  This
    # is the SAME defect fixed in the atm and ocean lanes; this lane reuses
    # ``build_band_grids_atm`` so it inherits the hazard verbatim.  Verify
    # cross-process agreement, then broadcast process 0's bytes -- GUARDED,
    # so a REAL divergence (different polar masks = different filtering =
    # different physics) RAISES instead of being papered over by process 0.
    # Single-process: both helpers short-circuit, so this is a no-op.
    _ordered = list(raw)
    assert_schema_agrees(_ordered, n_dev,
                         context="make_sharded_operator_split_step",
                         arrays=[raw[n] for n in _ordered])
    # checked_replicated_put keeps that guarded broadcast and places the
    # canonical bytes WITHOUT jax's whole-array device_put equality assert,
    # whose per-field all-gather (~P*N*(2s+1) bytes) made per-rank memory grow
    # with the rank count on the lat-lon lane until a 128-rank arm was
    # OOM-killed.
    stacks = {
        name: checked_replicated_put(
            raw[name], name, rep,
            context="make_sharded_operator_split_step")
        for name in _ordered
    }
    _cache = {}

    def sharded_split_step(carry, forcing):
        _agree_opsplit_mesh_entry(
            mesh, carry, forcing,
            where="make_sharded_operator_split_step.step")
        # Cache key = pytree STRUCTURE (None<->array flips) PLUS every leaf's
        # shape/dtype: the in/out_specs derive from arr.ndim + arr.shape[0], so a
        # shape/rank change under an unchanged structure would otherwise reuse a
        # stale P() vs P("lat") spec (codex).
        def _leaf_sig(pytree):
            return tuple((getattr(l, "shape", ()), getattr(l, "dtype", None))
                         for l in jax.tree.leaves(pytree))
        key = (jax.tree.structure(carry), jax.tree.structure(forcing),
               _leaf_sig(carry), _leaf_sig(forcing))
        fn = _cache.get(key)
        if fn is None:
            c_spec = _carry_specs(carry, n_dev)
            f_spec = _forcing_specs(forcing, n_dev)
            stacks_spec = jax.tree.map(lambda _x: P(), stacks)

            def _body(carry_b, forcing_b, stacks_b):
                r = jax.lax.axis_index(axis)
                band_geom = template._replace(
                    **{name: stacks_b[name][r] for name in afn})
                pmask = (stacks_b["__polar_mask"][r]
                         if "__polar_mask" in stacks_b else None)
                pmaskv = (stacks_b["__polar_mask_v"][r]
                          if "__polar_mask_v" in stacks_b else None)
                return _one_step(carry_b, band_geom, forcing_b, pmask, pmaskv)

            fn = jax.jit(shard_map(
                _body, mesh=mesh,
                in_specs=(c_spec, f_spec, stacks_spec),
                out_specs=c_spec, check_vma=False))
            _cache[key] = fn

        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh)
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        try:
            # Inside the try so a failure DURING activation still restores the
            # previous backend (codex): activate may mutate global halo state
            # before raising.
            activate_latlon_spmd_halo(mesh)
            return fn(carry, forcing, stacks)
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    return sharded_split_step


# Public alias for the tiled operator-split lane (tiled_operator_split_step)
# — the no-private-cross-imports ratchet's sanctioned surface.  Same object.
need_rad_and_time = _need_rad_and_time
