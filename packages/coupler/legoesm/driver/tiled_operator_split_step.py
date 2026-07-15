"""Sub-face-TILED operator-split atmosphere step (cube, 6*kt^2 devices).

The tiled-cube twin of :mod:`sharded_operator_split_step` (the lat-band
lane): one ``shard_map`` over the ``("face","tile_i","tile_j")`` mesh runs
the SAME operator-split integration the serial ``_single_step``
(``compiled_segments``) performs — dynamics -> target-anchored dry-mass
fixer -> ``step_unified`` column physics -> Euler write-back ->
saturation/moisture-fixer/Rayleigh tail -> carry pack — on a
:class:`SegmentCarry` sharded per TILE, kept sharded across steps (no
per-step gather).

Correctness rests on the same decomposition-friendliness:

* **Physics is PURELY column-local** (``split_physics_single_rank`` ->
  ``step_unified`` column kernels): a tile's ``nl*nl`` columns are computed
  with NO collectives, decomposition-invariant.
* **Global reductions** live OUTSIDE the physics: the target-mass fixer
  (:func:`_tile_fix_ps_mass_target` psum) and the moisture fixer
  (``fix_moisture_hydrostatic`` -> ``global_area_sum``, tile-psum-aware via
  ``conservation._spmd_lat_psum_or_none``'s tiled-mesh branch, armed by
  this module around the step call).
* **Dynamics** is the gated blocked-tile machinery: cc -> D-grid per tile
  (the shared rotating vector halo + 4-pt average), the dry
  ``_build_hydro_tile_tendency_fns`` SSP-RK3 (in-stage halos), D-grid ->
  cc via the LOCAL ``interp_corner_to_center`` — mirroring the serial
  ``model.step`` cc round trip.  The serial-vs-tiled dynamics difference
  is the documented O(1e-6 abs) face-corner wind class bounded by the
  blocked-loop gates.

TILE-AWARE CARRY LAYOUT (the PhysicsState shard): grid-shaped carry leaves
``(6, n, n, ...)`` shard ``P("face","tile_i","tile_j", ...)`` directly;
FLATTENED per-column leaves ``(ncol=6*n*n, ...)`` (``conv_prog``, the held
radiation fields, accumulators) canNOT shard on the tile mesh (the
row-major column order is not tile-contiguous) — :func:`pack_carry_tiled`
reshapes them to grid-shaped ``(6, n, n, ...)`` once at entry (a pure
layout view; ``ncol -> (6, n, n)`` row-major is exact) and
:func:`unpack_carry_tiled` restores the serial layout for writers /
serial interop.  Scalars (+ ``land_ml``) replicate.

ENVELOPE (refused loudly, never silently degraded): ``qv_smooth_coeff``
must be 0 (the ∇⁴ moisture smoothing reads a full-cube halo the tiled
in-stage pads do not carry); the ``statics.step_unified`` MUST be built at
TILE ``ncol = (n/kt)^2`` (the lat-band lane's band-build contract — tiles
are uniform, one build serves all 6*kt^2); ``owned_mask`` must be None
(single-controller lane).  The driver-side statics/forcing stacking that
feeds a REAL PhysicsPipeline through this step is the remaining
integration increment — this module + its mock-physics parity gate pin
the layout/composition contract (the lat-band lane's "2b" staging).
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map
from legoesm.parallel.cubesphere_exchange import (
    make_tiled_pad_body, make_tiled_pad_vector_body)
from legoesm.parallel.tiled_production_cdgrid import (
    build_hydro_tile_tendency_fns, sponge_rate_from_config,
    ssp_rk3_tile_step, tile_fix_ps_mass_target,
    validate_tiled_step_factory_args)
from legoesm.core.conservation import conservation_accumulator
from legoesm.driver.compiled_segments import (
    SegmentCarry, split_physics_single_rank, finalize_split_step)
from legoesm.driver.sharded_operator_split_step import need_rad_and_time

#: Carry fields that stay replicated scalars / opaque pytrees under the tile
#: shard (same partition as the lat-band lane's sets — kept local because
#: those are private to its module; the two lanes' sets must track the
#: SegmentCarry contract together).
_CARRY_SCALAR_FIELDS = frozenset(
    ("step_index", "target_moisture", "target_mass", "max_cfl"))
_CARRY_REPLICATED_FIELDS = frozenset(("land_ml",))

_TILE_AXES = ("face", "tile_i", "tile_j")

#: Carry fields the physics pipeline consumes at their FLAT per-column
#: layout (shape-validated (ncol, ...) — issue #405/#413): flattened to the
#: tile's ncol inside the body, re-gridded after the finalize pack.
_PER_COLUMN_FLAT_FIELDS = ("conv_prog", "tke", "qke", "gwd_spectrum")


class TileGridView(tuple):
    """Minimal per-tile grid view for the conservation reductions inside the
    tiled body: exposes exactly ``.area`` (the tile's slice).  The moisture /
    mass fixers read ONLY ``grid.area`` (``compute_global_moisture`` /
    ``global_area_sum``); handing them the full grid object would silently
    mix full-cube metrics into a tile-local body."""
    __slots__ = ()

    def __new__(cls, area):
        return tuple.__new__(cls, (area,))

    @property
    def area(self):
        return self[0]


def _grid_shaped(arr, n: int) -> bool:
    return arr.ndim >= 3 and tuple(arr.shape[:3]) == (6, n, n)


def _flat_ncol(arr, n: int) -> bool:
    return arr.ndim >= 1 and arr.shape[0] == 6 * n * n


def pack_carry_tiled(carry: SegmentCarry, n: int) -> SegmentCarry:
    """Serial carry layout -> tile-shardable layout.

    Reshapes every FLATTENED per-column leaf ``(ncol, ...) ->
    (6, n, n, ...)`` (row-major; exact) so it can shard
    ``P("face","tile_i","tile_j")``.  Grid-shaped leaves, scalars,
    ``land_ml`` and ``None`` pass through.  Inverse:
    :func:`unpack_carry_tiled`."""
    out = {}
    for name in SegmentCarry._fields:
        v = getattr(carry, name)
        if (v is None or name in _CARRY_SCALAR_FIELDS
                or name in _CARRY_REPLICATED_FIELDS
                or _grid_shaped(v, n) or not _flat_ncol(v, n)):
            out[name] = v
        else:
            out[name] = v.reshape((6, n, n) + tuple(v.shape[1:]))
    return SegmentCarry(**out)


def unpack_carry_tiled(carry: SegmentCarry, n: int,
                       template: SegmentCarry) -> SegmentCarry:
    """Tile layout -> the serial carry layout (writer / restart interop):
    every leaf that was flattened in ``template`` is re-flattened."""
    out = {}
    for name in SegmentCarry._fields:
        v = getattr(carry, name)
        t = getattr(template, name)
        if (v is None or t is None or name in _CARRY_SCALAR_FIELDS
                or name in _CARRY_REPLICATED_FIELDS
                or _grid_shaped(t, n) or not _flat_ncol(t, n)):
            out[name] = v
        else:
            out[name] = v.reshape((6 * n * n,) + tuple(v.shape[3:]))
    return SegmentCarry(**out)


def pack_forcing_tiled(forcing, n: int):
    """SegmentForcing with every FLAT ``(ncol, ...)`` leaf reshaped
    grid-shaped ``(6, n, n, ...)`` — the same rule as
    :func:`pack_carry_tiled`.  The real driver builds the per-column
    radiation forcing (``o3_vmr``, ``aerosol_od``, ``aerosol_lw_od``,
    flat ``T_sfc`` overrides) at ``(ncol[, nlev])`` (codex: replicating
    those into a tile body hands full-cube columns to a tile-ncol
    ``step_unified``).  Small replicated vectors (``solar_weights``,
    scalars) and ``None`` pass through.

    Returns ``(packed_forcing, flat_names)`` — ``flat_names`` records
    which leaves were COLUMN-format at pack time: the pipeline consumes
    those in column layout (``radiation_fn`` / the aerosol-CCN paths read
    them as ``(ncol[, nlev])`` WITHOUT an adapter flatten — codex), so
    the tile body must re-flatten exactly those to the tile's
    ``(nl*nl, ...)`` before the physics call, while grid-native surface
    fields (sst/sic/overrides) stay grid-shaped for the adapter."""
    out = {}
    flat_names = []
    for name in type(forcing)._fields:
        v = getattr(forcing, name)
        if v is None or not hasattr(v, "shape"):
            out[name] = v
            continue
        arr = jnp.asarray(v)
        if _grid_shaped(arr, n) or not _flat_ncol(arr, n):
            out[name] = v
        else:
            out[name] = arr.reshape((6, n, n) + tuple(arr.shape[1:]))
            flat_names.append(name)
    return type(forcing)(**out), tuple(flat_names)


def _tile_carry_spec(name, arr, n: int):
    """PartitionSpec for one PACKED carry leaf: grid-shaped leaves shard the
    three leading axes; scalars / ``land_ml`` / anything else replicate."""
    if arr is None:
        return None
    if (name in _CARRY_SCALAR_FIELDS or name in _CARRY_REPLICATED_FIELDS
            or not _grid_shaped(arr, n)):
        return P()
    return P(*_TILE_AXES, *((None,) * (arr.ndim - 3)))


def _tile_carry_specs(carry: SegmentCarry, n: int) -> SegmentCarry:
    return SegmentCarry(**{
        name: _tile_carry_spec(name, getattr(carry, name), n)
        for name in SegmentCarry._fields
    })


def shard_tiled_split_carry(carry: SegmentCarry, mesh, n: int) -> SegmentCarry:
    """Commit a (SERIAL-layout) SegmentCarry to the tiled layout: pack the
    flattened leaves grid-shaped (:func:`pack_carry_tiled`), then place every
    leaf on the ``(6, kt, kt)`` mesh per :func:`_tile_carry_spec`.  This is
    the tile-aware PhysicsState/carry shard.

    ``land_ml`` (the multilayer-land carry) is an opaque per-column pytree
    with no tile partition spec — replicating it would hand a tile-ncol
    ``step_unified`` full-cube (or divergent) land state (codex).  Refused
    until its leaves are packed tile-wise."""
    if getattr(carry, "land_ml", None) is not None:
        raise NotImplementedError(
            "shard_tiled_split_carry: the multilayer-land carry (land_ml) "
            "has no tile-wise packing yet — run the land-free config or "
            "the face-only lane.")
    packed = pack_carry_tiled(carry, n)
    return SegmentCarry(**{
        name: (None if (v := getattr(packed, name)) is None
               else jax.device_put(
                   v, NamedSharding(mesh, _tile_carry_spec(name, v, n))))
        for name in SegmentCarry._fields
    })


class _TilePhysicsGrid(tuple):
    """Shape-only grid view for building the physics pipeline at TILE ncol.

    ``make_adapter`` reads exactly ``grid_n_columns`` + ``grid_lat.shape``
    (ColumnAdapter bakes STATIC ncol/shape_2d); the pipeline's runtime
    ``self._grid`` branches (w-grid / moisture-convergence convection) are
    horizontal-operator consumers a tile cannot serve —
    :func:`build_tile_step_unified` refuses those schemes up front, so this
    view is never asked for real geometry."""
    __slots__ = ()

    def __new__(cls, nl: int):
        z = jnp.zeros((1, nl, nl))
        return tuple.__new__(cls, (z,))

    @property
    def grid_lat(self):
        return self[0]

    @property
    def grid_lon(self):
        return self[0]

    @property
    def grid_n_columns(self) -> int:
        return int(self[0].shape[1] * self[0].shape[2])


def build_tile_step_unified(grid, sigma, config, kt: int):
    """Build the unified physics ``step_unified`` at TILE ncol — the
    tiled-cube analogue of the lat-band lane's band-grid build ("ONE
    band-grid build serves every band"; cube tiles are uniform windows,
    so one tile-ncol build serves every tile).

    Refusals (loud — decomposition-VARIANT physics):
    * a land-active pipeline (``f_land`` per-column bake differs per tile);
    * convection schemes whose traits consume HORIZONTAL grid operators
      (w-grid / moisture-convergence) — those read ``pipeline._grid`` at
      runtime and are not column-local.

    Returns ``(step_unified, pipeline)``.
    """
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.atmosphere.physics.convection.integration import (
        convection_scheme_traits,
    )

    if getattr(config, "convection", "none") != "none":
        traits = convection_scheme_traits(config.convection)
        if (getattr(traits, "is_w_grid_consumer", False)
                or getattr(traits, "is_mc_consumer", False)
                or getattr(traits, "is_simple_mc_consumer", False)):
            raise NotImplementedError(
                f"build_tile_step_unified: convection={config.convection!r} "
                "consumes horizontal grid operators (w-grid / moisture "
                "convergence) — not column-local, cannot run per tile. "
                "Use a column-local convection scheme or the face-only "
                "lane.")
    if getattr(config, "shard_radiation_columns", False):
        raise NotImplementedError(
            "build_tile_step_unified: shard_radiation_columns builds a "
            "GLOBAL column mesh against the pipeline ncol — nested inside "
            "the per-tile shard_map that is either a false divisibility "
            "refusal or a wrong nested sharding (codex). The tile mesh IS "
            "the column decomposition here; disable "
            "shard_radiation_columns for the tiled lane.")
    nl = int(grid.n) // kt
    pipeline = build_physics_pipeline(_TilePhysicsGrid(nl), sigma, config)
    if pipeline.f_land is not None:
        raise NotImplementedError(
            "build_tile_step_unified: land-active pipeline (f_land is a "
            "per-column bake that differs per tile) — no tile-wise land "
            "packing yet; run land-free or the face-only lane.")
    return pipeline.build_step_unified(static_need_rad=True), pipeline


def make_tiled_operator_split_step(
    model, mesh, statics, *, fix_mass, rad_update_steps, start_day,
    kt: int, ghg_keys=None,
):
    """Build ``tiled_split_step(carry, forcing) -> carry`` — one
    operator-split atmosphere step on the ``(6, kt, kt)`` tiled mesh.

    ``carry`` is a PACKED, tile-sharded :class:`SegmentCarry`
    (:func:`shard_tiled_split_carry`); input layout == output layout, so
    ``carry = step(carry, forcing)`` iterates gather-free (the
    blocked-loop contract).  ``forcing`` is a PER-CALL traced
    :class:`SegmentForcing` (the SegmentForcing doctrine: the driver
    re-samples it per segment and the same step object serves every
    segment) — flat ``(ncol, ...)`` leaves are packed grid-shaped per
    call (:func:`pack_forcing_tiled`).  ``statics`` is a
    :class:`_SplitStepStatics` whose ``step_unified`` is built at TILE
    ncol (``build_tile_step_unified``) and whose ``lat``/``lon`` are the
    FULL-cube ``(6, n, n)`` cell fields — the body slices the tile's
    window (the lat-band lane's stacked-statics pattern, expressed as
    slicing because cube tiles are uniform windows of face-major
    arrays); ``statics.forcing`` is unused (per-call arg wins).
    ``ghg_keys`` mirrors the lat-band factory: rebuild
    ``ghg_vmr_override`` per step from ``forcing.ghg_vmr`` for transient
    GHG; ``None`` leaves the statics value untouched.

    ENVELOPE refusals (loud): ``statics.qv_smooth_coeff != 0`` (full-cube
    ∇⁴ halo), ``statics.owned_mask is not None`` (MPI-replicated
    semantics do not compose with the single-controller tile mesh).
    """
    cfg = model.config
    n = int(model.grid.n)
    nlev = int(model.sigma_coord.n_levels)
    coord = model.sigma_coord
    cdgrid = model.cdgrid
    validate_tiled_step_factory_args(
        "make_tiled_operator_split_step", mesh, cdgrid, coord, n, kt,
        nlev, cfg.p_floor, statics.dt)
    # The serial _single_step's DYNAMICS is the driver model's step — the
    # tiled base-cut RK3 omits every optional damp, so a model config with
    # any of them nonzero (driver hyperdiff_scale > 0 etc.) would silently
    # integrate different dynamics.  Same envelope as the blocked-loop
    # adapter; the fixer flags are irrelevant here (this lane externalizes
    # the fixer like the compiled-segment driver).
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        validate_tiled_envelope,
    )
    validate_tiled_envelope(cfg, cdgrid, inner_fix_mass_ok=True)
    if getattr(statics, "qv_smooth_coeff", 0.0) != 0.0:
        raise NotImplementedError(
            "make_tiled_operator_split_step: qv_smooth_coeff != 0 needs the "
            "full-cube ∇⁴ moisture halo, which the tiled in-stage pads do "
            "not carry — disable the smoothing or run the face-only lane.")
    if getattr(statics, "owned_mask", None) is not None:
        raise NotImplementedError(
            "make_tiled_operator_split_step: owned_mask (MPI replicated "
            "dynamics) does not compose with the single-controller tile "
            "mesh.")
    nl = n // kt
    dt = statics.dt
    acc = conservation_accumulator()

    # --- Static metrics (face-replicated, sliced per tile — the blocked-step
    # set + the angle fields for the cc->D-grid entry lift). ---
    grid = cdgrid.base
    area = grid.area
    total_area = jnp.sum(area.astype(acc))
    cos_a, sin_a = grid.cos_angle, grid.sin_angle
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded
    offsets = grid.halo_interp_offsets

    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)
    _tile_tendency, _slice_metrics = build_hydro_tile_tendency_fns(
        coord, cdgrid, nl, nlev, float(cfg.p_floor), scalar_body, vector_body,
        sponge_rate=sponge_rate_from_config(
            coord, float(getattr(cfg, "sponge_sigma", 0.0)),
            float(getattr(cfg, "sponge_tau_sec", 0.0))))

    m_cosa_c = cdgrid.cosa_corner
    m_dxe, m_dye = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    m_gc = (cdgrid.grad_c00, cdgrid.grad_c01, cdgrid.grad_c10,
            cdgrid.grad_c11)
    m_fco, m_cosau = cdgrid.f_corner, cdgrid.cosa_u
    m_dx, m_dy = grid.dx, grid.dy

    # Forcing is a PER-CALL traced argument (the SegmentForcing doctrine —
    # the driver re-samples SST/solar/ozone per segment and the SAME step
    # object must serve every segment without a rebuild); statics.forcing
    # stays None/unused here.  Grid-shaped forcing fields shard by tile
    # spec; FLAT (ncol,...) leaves (the driver's per-column radiation
    # forcing) are packed grid-shaped per call — same rule as the carry
    # (codex round-2).
    def _forcing_spec(v):
        if v is None:
            return None
        arr = jnp.asarray(v)
        if _grid_shaped(arr, n):
            return P(*_TILE_AXES, *((None,) * (arr.ndim - 3)))
        return P()

    fo = P("face", None, None)

    def _one_tile_step(carry_t, lat_t, lon_t, forcing_t, ar_t, mt,
                       angle_t, offs, flat_forcing_names):
        """The serial _single_step composition on ONE tile."""
        ca_t, sa_t, cap_t, sap_t = angle_t

        # --- Dynamics: cc -> D-grid (rotating vector halo + 4-pt average —
        # the gated tiled center_to_dgrid_vector kernel's stencil) ---
        u_pad, v_pad = vector_body(carry_t.u[0], carry_t.v[0], ca_t, sa_t,
                                   cap_t, sap_t, offs)
        u_pad, v_pad = u_pad[None], v_pad[None]
        u_d0 = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                       + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d0 = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                       + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])

        def _F(s):
            return _tile_tendency(s[0], s[1], s[2], s[3], carry_t.phis, *mt)

        ud3, vd3, T_new, ps_new = ssp_rk3_tile_step(
            (u_d0, v_d0, carry_t.T, carry_t.p_s), _F, float(dt))

        # --- D-grid -> cc (LOCAL 4-pt box; exact cc partition) ---
        from legoesm.core.operators_cdgrid import interp_corner_to_center
        u_new = interp_corner_to_center(ud3)
        v_new = interp_corner_to_center(vd3)

        # --- Target-anchored dry-mass fixer (psum across all tiles) ---
        if fix_mass:
            ps_new = tile_fix_ps_mass_target(
                ps_new, carry_t.target_mass, ar_t, total_area, acc)

        # --- Column-local physics + the shared finalize tail ---
        need_rad, doy, sod = need_rad_and_time(
            carry_t.step_index, start_day, dt, rad_update_steps)
        tile_statics = dataclasses.replace(
            statics, grid=TileGridView(ar_t),
            lat=lat_t, lon=lon_t, forcing=forcing_t)
        if ghg_keys is not None:
            # Transient GHG: rebuild the override from THIS segment's
            # forcing (the lat-band factory's contract — a baked statics
            # value would freeze GHG for the whole run).
            from legoesm.driver.compiled_segments import ghg_array_to_dict
            tile_statics = dataclasses.replace(
                tile_statics,
                ghg_vmr_override=ghg_array_to_dict(
                    forcing_t.ghg_vmr, ghg_keys))

        # The pipeline's per-column-NATIVE fields travel the tile mesh
        # grid-shaped (the only shardable layout) — flatten them to the
        # tile's (nl*nl, ...) for the physics call and (for carry fields)
        # re-grid after the finalize pack, so the carry out-spec stays
        # tile-shaped:
        #   * carry: tke/qke/gwd_spectrum (shape-VALIDATED at (ncol, ...)
        #     in physics_step_no_rad) + conv_prog (consumed flat);
        #   * forcing: the leaves that were COLUMN-format at pack time
        #     (o3_vmr / aerosol_od / aerosol_lw_od / flat overrides) —
        #     radiation_fn + the aerosol-CCN paths read them WITHOUT an
        #     adapter flatten (codex).
        def _flt(v):
            return (None if v is None
                    else v.reshape((v.shape[1] * v.shape[2],)
                                   + tuple(v.shape[3:])))

        def _grd(v):
            return (None if v is None
                    else v.reshape((1, nl, nl) + tuple(v.shape[1:])))

        if flat_forcing_names:
            forcing_flat = forcing_t._replace(**{
                nm: _flt(getattr(forcing_t, nm))
                for nm in flat_forcing_names})
            tile_statics = dataclasses.replace(
                tile_statics, forcing=forcing_flat)

        carry_phys = carry_t._replace(**{
            nm: _flt(getattr(carry_t, nm))
            for nm in _PER_COLUMN_FLAT_FIELDS})
        lz = split_physics_single_rank(
            carry_phys, T_new, u_new, v_new, ps_new, need_rad, doy, sod,
            tile_statics)
        out = finalize_split_step(carry_phys, lz, tile_statics)
        return out._replace(**{
            nm: _grd(getattr(out, nm))
            for nm in _PER_COLUMN_FLAT_FIELDS})

    _cache = {}

    def tiled_split_step(carry: SegmentCarry, forcing) -> SegmentCarry:
        if getattr(carry, "land_ml", None) is not None:
            raise NotImplementedError(
                "make_tiled_operator_split_step: land_ml has no tile-wise "
                "packing yet (see shard_tiled_split_carry).")
        f_named, flat_forcing_names = pack_forcing_tiled(forcing, n)

        def _leaf_sig(pytree):
            return tuple((getattr(x, "shape", ()), getattr(x, "dtype", None))
                         for x in jax.tree.leaves(pytree))
        key = (jax.tree.structure(carry), _leaf_sig(carry),
               jax.tree.structure(f_named), _leaf_sig(f_named),
               flat_forcing_names)
        fn = _cache.get(key)
        if fn is None:
            c_spec = _tile_carry_specs(carry, n)
            f_spec = type(f_named)(**{
                nm: _forcing_spec(getattr(f_named, nm))
                for nm in type(f_named)._fields})

            def _body(carry_b, forcing_b, lat_b, lon_b, ar_b,
                      cosa_c_b, dxe_b, dye_b, gc0, gc1, gc2, gc3,
                      fco_b, cosau_b, dx_b, dy_b, ca_b, sa_b, capf_b,
                      sapf_b, offs_b):
                a_i = jax.lax.axis_index("tile_i") * nl
                a_j = jax.lax.axis_index("tile_j") * nl

                def _s(arr, si, sj):
                    arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
                    return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

                m = _slice_metrics(_s, cosa_c_b, dxe_b, dye_b, ar_b,
                                   gc0, gc1, gc2, gc3, fco_b, cosau_b,
                                   dx_b, dy_b, ca_b, sa_b, capf_b, sapf_b)
                mt = (m["cosa_c_t"], m["dxe_t"], m["dye_t"], m["ar_t"],
                      m["gc"], m["fco_t"], m["cosau_t"], m["dx_t"],
                      m["dy_t"], m["ca_t"], m["sa_t"], m["cap_t"],
                      m["sap_t"], offs_b)
                angle_t = (m["ca_t"], m["sa_t"], m["cap_t"], m["sap_t"])
                lat_t = _s(lat_b, nl, nl)
                lon_t = _s(lon_b, nl, nl)
                return _one_tile_step(
                    carry_b, lat_t, lon_t, forcing_b, m["ar_t"], mt,
                    angle_t, offs_b, flat_forcing_names)

            fn = jax.jit(shard_map(
                _body, mesh=mesh,
                in_specs=(c_spec, f_spec,
                          fo, fo,                      # lat, lon
                          fo,                          # area
                          fo, fo, fo,                  # cosa_c, dxe, dye
                          fo, fo, fo, fo,              # gc00..gc11
                          fo, fo,                      # f_corner, cosa_u
                          fo, fo,                      # dx, dy
                          fo, fo, fo, fo,              # cos_a, sin_a, cap, sap
                          P()),                        # offsets
                out_specs=c_spec, check_vma=False))
            _cache[key] = fn

        from legoesm.core.conservation import tiled_reduction_scope
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh)
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        try:
            # Arm the tiled mesh + the EXPLICIT tiled-reduction scope so the
            # conservation reductions traced inside the body (the moisture
            # fixer's global_area_sum) route to the tiled psum branch —
            # and ONLY those (an unrelated reduction running while the mesh
            # is armed falls through; codex).
            set_halo_backend("spmd")
            set_spmd_mesh(mesh)
            with tiled_reduction_scope():
                return fn(carry, f_named, statics.lat, statics.lon, area,
                          m_cosa_c, m_dxe, m_dye, *m_gc, m_fco, m_cosau,
                          m_dx, m_dy, cos_a, sin_a, cap, sap, offsets)
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    return tiled_split_step
