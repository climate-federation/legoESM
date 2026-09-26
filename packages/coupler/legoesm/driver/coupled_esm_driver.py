"""Fully coupled Earth System Model driver.

Orchestrates atmosphere + slab/dynamic ocean + land (slab/Richards')
+ sea ice + lake + optional carbon cycle into a single integration.

Builds on ``ModelDriver`` for atmosphere and ``make_coupler`` for
surface exchange.  The coupler + ocean step runs at segment boundaries
via a callback from ``ModelDriver.run()``.

Configuration is via ``CoupledConfig`` (see ``coupled_config.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.driver.air_sea_consistency import validate_air_sea_consistency
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.coupled_config import CoupledConfig
from legoesm.diagnostics.energy_budget import area_weighted_mean
from legoesm.driver.model_driver import ModelDriver

from legoesm import constants

logger = logging.getLogger("legoesm.driver.coupled_esm")

# Exact time-unit conversion for the day-valued restoring timescales.
_SECONDS_PER_DAY = 86400.0


def _total_ice_sic(ice_state):
    """Total sea-ice concentration [0-1] on the grid, from either a scalar
    :class:`SeaIceState` (``concentration`` is grid-shaped) or a multi-category
    :class:`DynamicSeaIceState` (``concentration`` carries a trailing
    ``category`` axis, summed to the aggregate areal fraction)."""
    conc = ice_state.concentration.data
    if "category" in tuple(ice_state.concentration.dims):
        conc = conc.sum(axis=-1)
    return conc


def enable_diurnal_surface_land(land_cfg):
    """Switch a ``MultiLayerLandConfig`` to the coupled DIURNAL surface model:
    Monin-Obukhov (MOST) surface exchange + Farquhar photosynthesis-stomata coupling.

    Farquhar (the coupled stomatal-conductance path) only fires when the carbon scheme
    is ``differland`` (it reads the prognostic LAI = C_fol/LCMA); under any other scheme
    ``compute_effective_beta`` silently falls back to the Jarvis model, which ignores
    Vc_max25.  So any non-``differland`` scheme is upgraded to ``differland`` here to
    guarantee the photosynthesis params are actually used (the coupler initialises +
    threads the carbon state from this config).  Pure -> unit-testable; the driver
    gates it on ``CoupledConfig.land_diurnal_surface``.

    NOTE: MOST references fluxes to ``land_cfg.z_ref`` (default 10 m).  When the
    atmosphere's lowest model level sits at a different height this is a (bounded) bias
    the coupled feedback absorbs; threading the true lowest-level height is a follow-up.
    Setting ``land_diurnal_surface=False`` reverts to the constant-Ch + soil-beta land."""
    from legoesm.land.carbon.config import CarbonConfig
    cfg = land_cfg._replace(
        bulk_scheme="most", stomata=land_cfg.stomata._replace(enabled=True))
    if cfg.carbon.scheme != "differland":   # Farquhar needs the differland LAI
        cfg = cfg._replace(carbon=CarbonConfig(scheme="differland"))
    return cfg


def assert_land_tile_reachable(land_mode, f_land_mode, f_land) -> None:
    """Raise if a land model is configured but its tile has zero area everywhere.

    The driver-level invariant on the MATERIALIZED land fraction, the residue the
    CONFIG-level CLI guard (``run_coupled.apply_land_runoff_scheme``) documents it
    cannot see: ``land_mode != "none"`` builds land physics, yet ``f_land`` comes
    out identically zero, so the entire land tile is silently dead. ``f_land`` is
    a fraction in ``[0, 1]``, so ``max(f_land) == 0`` iff there is no land in any
    cell -- the ``from_ocean`` all-wet-ocean-mask path and the ``analytical``
    degenerate-grid (no latitude) fall-to-zeros path, neither visible to any
    config predicate.

    ``f_land_mode == "zero"`` is EXCLUDED: that is the explicit
    aquaplanet-with-slab-land request (``--preset aquaplanet --land-scheme slab``;
    ``run_coupled.py``), where a zero land area is intended, not an accident.

    Pure -> unit-testable; the driver calls it once at coupler-init after
    materialising ``f_land``."""
    if land_mode == "none" or f_land_mode == "zero":
        return
    # f_land >= 0 everywhere by construction, so max <= 0 <=> all cells zero.
    if float(jnp.max(f_land)) <= 0.0:
        raise ValueError(
            f"land_mode={land_mode!r} builds a land model but the materialized "
            f"land fraction is zero everywhere (f_land_mode={f_land_mode!r}): "
            f"the land tile is silently dead. This is the from_ocean "
            f"all-wet-ocean-mask or the analytical degenerate-grid (no latitude) "
            f"residue that no config-level predicate can see. Set "
            f"f_land_mode='zero' if you intend an aquaplanet, or supply a grid "
            f"latitude / an ocean mask that actually contains land."
        )


def _flatten_pytree_to_npz(state, prefix: str) -> dict:
    """Flatten a registered pytree (a NamedTuple of ``Field`` leaves) into a
    flat ``{prefix+path: np.ndarray}`` dict for ``np.savez`` serialization.

    Leaves are keyed by their stringified pytree path so a structural change
    between save and restore is *detectable* rather than silently mis-mapped
    (see ``_restore_pytree_from_npz``).  ``None`` leaves (an inactive optional
    Field — e.g. ``T_som``/``eke``/``tke`` when the scheme is off) are simply
    absent from JAX's pytree traversal, so they are neither saved nor restored;
    a fresh init with the same config reproduces the identical None structure.
    """
    out = {}
    for path_parts, leaf in jax.tree_util.tree_leaves_with_path(state):
        out[prefix + ".".join(str(p) for p in path_parts)] = np.asarray(leaf)
    return out


def _restore_pytree_from_npz(state, data, prefix: str, ckpt_name: str,
                             strict: bool = False):
    """Restore a registered pytree from npz leaves written by
    ``_flatten_pytree_to_npz``.

    Every leaf is matched by its stringified path; each restored leaf's shape is
    validated against the fresh-init leaf (and cast to its dtype) so a corrupted
    or mismatched array fails LOUDLY instead of being placed by ``tree_unflatten``
    (which does not check shapes).  Leaves are restored atomically in one
    ``tree_unflatten`` — for the C-grid ocean this keeps the ``land_mask`` /
    ``u_mask`` / ``v_mask`` triple mutually consistent (a partial mask restore
    would leak mass through walls; CLAUDE.md "Land/face masks").

    Structure drift (a leaf expected now but absent from the checkpoint, or an
    unused checkpoint leaf) means the config changed since the save.  With
    ``strict=True`` (the prognostic 3D ocean) this RAISES — silently reseeding a
    toggled EKE/TKE/SOM/AB2 history leaf would corrupt the restart.  With
    ``strict=False`` (the surface state) it warns and keeps the absent leaf's
    fresh-init value.  Returns ``(new_state, restored)``; ``restored`` is False
    when the checkpoint carried no ``prefix`` leaves (nothing to do).
    """
    keys = [k for k in data.files if k.startswith(prefix)]
    if not keys:
        return state, False
    leaves_with_path = jax.tree_util.tree_leaves_with_path(state)
    expected = {
        prefix + ".".join(str(p) for p in pp) for pp, _ in leaves_with_path
    }
    saved = set(keys)
    missing, extra = expected - saved, saved - expected
    if missing or extra:
        msg = (
            f"Coupled checkpoint {ckpt_name}: '{prefix}' structure drift — "
            f"{len(missing)} expected leaf(s) absent from the checkpoint, "
            f"{len(extra)} unused checkpoint leaf(s).  The config likely "
            f"changed since the checkpoint was written.")
        if strict:
            raise ValueError(
                msg + "  Refusing to restore a structurally-mismatched "
                "prognostic state.")
        logger.warning(msg + "  Absent leaves keep their fresh-init value.")
    new_leaves = []
    for pp, leaf in leaves_with_path:
        key = prefix + ".".join(str(p) for p in pp)
        if key not in saved:
            new_leaves.append(leaf)
            continue
        arr = data[key]
        if tuple(arr.shape) != tuple(jnp.shape(leaf)):
            raise ValueError(
                f"Coupled checkpoint {ckpt_name}: leaf '{key}' shape "
                f"{tuple(arr.shape)} != current {tuple(jnp.shape(leaf))}; "
                f"refusing to restore a mismatched leaf.")
        # Cast to the fresh-init leaf's dtype so a checkpoint saved under a
        # different x64 setting does not drift the restored state's dtype.
        new_leaves.append(jnp.asarray(arr, dtype=jnp.asarray(leaf).dtype))
    return (
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(state), new_leaves),
        True,
    )


class CoupledESMDriver:
    """Coupled atmosphere + ocean + land + carbon driver.

    Parameters
    ----------
    atm_config : ExperimentConfig
        Atmosphere experiment configuration.
    coupled_config : CoupledConfig
        Ocean/land/carbon mode selection.
    output_dir : str or Path, optional
    """

    def __init__(
        self,
        atm_config: ExperimentConfig,
        coupled_config: CoupledConfig | None = None,
        coupler_config=None,
        ice_config=None,
        lake_config=None,
        ocean_grid=None,
        output_dir=None,
    ):
        self.atm_config = atm_config
        self.coupled_cfg = coupled_config or CoupledConfig()
        validate_air_sea_consistency(atm_config, coupler_config)
        self._atm = ModelDriver(atm_config, output_dir=output_dir)
        # This driver's _build_atm_forcing READS held_sw_net_sfc /
        # held_lw_net_sfc / seg_precip out of the atmosphere's _carry_aux, so
        # atmosphere lanes that never write them must refuse to run rather
        # than silently force the surface with zeros.  The marker (not the
        # presence of a segment_callback, which is also used by uncoupled
        # diagnostic samplers) is what ModelDriver._reject_coupled_lane gates
        # on.
        self._atm._requires_surface_flux_export = True
        self._coupler_config = coupler_config
        self._ice_config = ice_config
        self._lake_config = lake_config
        # Ocean grid: None => ocean lives on the atmosphere grid (single-grid
        # coupling, no remap).  A distinct grid object enables differentiable
        # atm<->ocean regridding through the coupler (see coupler.grid_remap).
        self._ocean_grid_arg = ocean_grid

        # Populated during setup()
        self._step_surface = None
        self._sfc_state = None
        self._ocean_state = None
        self._ocean_step = None
        # WOA surface T/S restoring targets (set by the dynamic-ocean WOA init;
        # None => no target => relaxation skipped).
        self._ocean_T_target = None
        self._ocean_S_target = None
        self._last_sfc_response = None
        self._coupled_diag = []
        self._sst_mean_init = None  # set on first diag — SST-drift reference
        # F2 water-conservation tripwire state (diagnostics-only, host-side).
        # ``_cwv_prev``: previous-segment column-water-vapour field, for the
        # global atmospheric moisture-budget tendency (catches a mis-scaled
        # coupler-DELIVERED precip, C1).  ``_last_atm_precip``: the precip RATE
        # handed to the coupler this segment.  Runoff export/applied integrals
        # (catch runoff discarded by the ocean wet mask, M2) are stashed in
        # ``_assemble_ocean_forcing`` (dynamic ocean only).  All rank-local.
        self._cwv_prev = None
        self._last_atm_precip = None
        self._last_runoff_export_integral_ranklocal = None
        self._last_runoff_applied_integral_ranklocal = None
        # F2 closed water-INVENTORY residual (tripwire C, catches the H2 class:
        # ice-fraction precip destroyed).  ``_water_store_prev_integral_ranklocal``
        # mirrors ``_cwv_prev``/``_sst_mean_init`` first-call seeding: the
        # segment-boundary area integral of W_atm[+condensate]+W_land+W_ice on
        # the atm grid [kg], stored between segments so the residual first
        # appears on the SECOND diag.  ``_last_f_ocean_applied_integral_ranklocal``
        # is the ocean-grid freshwater-applied integral [kg/s], stashed in
        # ``_assemble_ocean_forcing`` (dynamic ocean only).  All rank-local.
        self._water_store_prev_integral_ranklocal = None
        self._last_f_ocean_applied_integral_ranklocal = None

    @property
    def output_dir(self) -> Path:
        return self._atm.output_dir

    @property
    def grid(self):
        """The atmosphere grid (delegates to the atm driver).

        Exposes the same public ``grid`` as :class:`ModelDriver` so a coupled
        (CMIP) run is grid-introspectable like an atm-only run — e.g. the column
        comparison reconstructs the MPAS cell wind from ``driver.grid`` (the
        ``VoronoiMesh``) for both AMIP and CMIP.
        """
        return self._atm.grid

    @property
    def sigma(self):
        """The atmosphere vertical coordinate (delegates to the atm driver).

        Exposes the same public ``sigma`` as :class:`ModelDriver` so the column
        comparison can synthesize the grid winds from a spectral CMIP state
        (``spectral_pe_to_grid`` needs the grid + sigma) for both AMIP and CMIP.
        """
        return self._atm.sigma

    # ==================================================================
    # Setup
    # ==================================================================

    def setup(self) -> None:
        """Initialize all components."""
        # 1. Atmosphere
        self._atm.setup()

        # 2. Ocean (slab / two-layer)
        self._init_ocean()

        # 3. Coupler (land + ice + lake + ocean tile blending)
        self._init_coupler()

        # 3b. Bake the ocean wet mask into the cross-grid remap weights (H3/H4).
        #     AFTER _init_coupler so the setup-time from_ocean ocean-fraction remap
        #     (which area-averages the 0/1 mask and needs the UNMASKED weights)
        #     already ran; identity (shared-grid) remapper -> no-op.
        self._attach_ocean_wet_masks()

        # 4. Carbon / CO2 tracer (if active)
        self._init_carbon()

        # 5. Override SST source: slab ocean instead of file
        self._override_sst()

        # 6. Optionally feed the coupler's dynamic surface albedo / skin
        #    temperature back to the atmosphere's radiation (opt-in).
        self._override_sfc()

        # 7. Optionally feed the coupler's blended surface SH/LH fluxes back to
        #    the atmosphere surface tendency (opt-in) so the air-sea budget
        #    closes (single shared flux calc on both sides).
        self._override_sfc_fluxes()

        logger.info("CoupledESM: all components initialized")
        logger.info(f"  ocean_mode={self.coupled_cfg.ocean_mode}, "
                    f"land_mode={self.coupled_cfg.land_mode}, "
                    f"carbon_active={self.coupled_cfg.carbon_active}")

    def _attach_ocean_wet_masks(self):
        """Bake the ocean wet mask into the cross-grid remap weights (H3/H4).

        Rebuilds the o2a (ocean->atm STATE) weights to EXCLUDE ocean-grid land
        source cells from every coastal atm SST/current average (H4) -- the land
        fill value must not bleed into coastal atm cells.  The a2o (atm->ocean
        FLUX) weights stay the conservative partition-of-unity remap; the
        open-water fraction that multiplies every a2o flux GATES them to the
        ocean's own wet domain (H3, _assemble_ocean_forcing), which conserves the
        flux integral over the wet ocean without delivering to inert land cells.

        No-op for the identity (shared-grid) remapper -> byte-identical
        single-grid run.  The traced apply stays a pure segment_sum in all cases.
        """
        from legoesm.coupler.grid_remap import attach_wet_masks
        rem = getattr(self, "_grid_remapper", None)
        if rem is None or rem.identity:
            return
        owet = getattr(self, "_ocean_land_mask", None)
        if owet is None:
            return
        self._grid_remapper = attach_wet_masks(
            rem, self._atm.grid, self._ocean_grid, owet)

    def _init_ocean(self):
        """Initialize the slab/two-layer ocean (on the ocean grid)."""
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        from legoesm.ocean.simple_ocean import init_slab_state, make_ocean

        cfg = self.coupled_cfg
        # The ocean may live on a DIFFERENT grid than the atmosphere.  Build the
        # differentiable atm<->ocean remap once here (host).  When the ocean
        # defaults to the atmosphere grid the remapper is the identity and every
        # remap below is a byte-identical pass-through (standard single-grid run).
        self._ocean_grid = self._ocean_grid_arg or self._atm.grid
        self._grid_remapper = make_grid_remapper(self._atm.grid, self._ocean_grid)
        shape_2d = self._ocean_grid.grid_shape_2d
        # Per-cell ocean area weights for the global-mean SST / drift metric.
        # ``jnp.mean`` over a lat-lon ocean grid over-weights the cold polar
        # rows, so an unweighted SST mean read a much larger cold drift than the
        # area-weighted ocean actually experiences (see ``area_weighted_mean``).
        self._ocean_area_w = getattr(self._ocean_grid, "grid_area", None)

        # Initial SST from the atmosphere's SST source (day 0), remapped onto
        # the ocean grid (identity => unchanged).
        sst_init_atm, _ = self._atm.get_sst_sic(0.0)
        sst_init = remap_field(sst_init_atm, self._grid_remapper.a2o)
        T_sfc_mean = float(jnp.mean(sst_init))

        # Dispatch on ocean_mode (explicit; ValueError on unknown — no silent
        # else->slab, per the CLAUDE.md dispatch-hardening rule).
        if cfg.ocean_mode in ("slab", "two_layer", "fixed"):
            # ocean_mode selects THIS thermodynamic branch, but the physics
            # actually run is make_ocean(cfg.ocean_config) — two mode fields.
            # Their agreed mapping is _OCEAN_MODE_LABEL (fixed/slab -> "slab",
            # two_layer -> "two_layer"; run_coupled + preset_complexity both
            # build through it).  Reject any other pairing loudly
            # (ocean_mode="two_layer" + SimpleOceanConfig(mode="fixed")
            # logged "two_layer" while running fixed-SST physics; 2026-07-21
            # audit GAP-5).
            from legoesm.driver.coupled_config import ocean_mode_label
            _sub_mode = getattr(cfg.ocean_config, "mode", None)
            if (_sub_mode is not None
                    and ocean_mode_label(_sub_mode) != cfg.ocean_mode):
                raise ValueError(
                    f"CoupledConfig.ocean_mode={cfg.ocean_mode!r} is "
                    f"inconsistent with ocean_config.mode={_sub_mode!r} "
                    f"(expected ocean_mode={ocean_mode_label(_sub_mode)!r}); "
                    "the coupler dispatches on ocean_mode while make_ocean "
                    "runs ocean_config.mode — keep them consistent."
                )
            self._is_dynamic_ocean = False
            self._ocean_state = init_slab_state(
                shape_2d, T_sfc_init=T_sfc_mean,
            )
            # Override with the spatially varying AMIP SST
            from legoesm.core.field import Field
            self._ocean_state = self._ocean_state._replace(
                T_sfc=Field(
                    data=jnp.array(
                        sst_init, dtype=self._ocean_state.T_sfc.data.dtype),
                    name="T_sfc", dims=self._ocean_state.T_sfc.dims,
                    units="K",
                ),
            )
            self._ocean_step = make_ocean(cfg.ocean_config)
            # Optional spatially+seasonally varying q-flux climatology: load
            # once here (regridded to the ocean grid), interpolated per
            # coupling interval in _step_ocean and threaded into the slab step.
            # ``None`` (no path) => the scalar config.Q_flux, byte-identical.
            self._qflux_forcing = None
            _qfp = getattr(cfg.ocean_config, "q_flux_path", "")
            if _qfp:
                from legoesm.ocean.forcing.qflux import load_qflux_climatology
                self._qflux_forcing = load_qflux_climatology(
                    _qfp, self._ocean_grid)
                logger.info(
                    "  Ocean q-flux climatology: %s (%d records, regridded to "
                    "the ocean grid; +into mixed layer)",
                    _qfp, int(self._qflux_forcing.times.shape[0]))
            logger.info(f"  Ocean: mode={cfg.ocean_mode}, "
                        f"h_mix={cfg.ocean_config.h_mix}m, "
                        f"T_sfc_init={T_sfc_mean:.1f}K")
        elif cfg.ocean_mode == "dynamic":
            # Prognostic 3D ocean (LatLonCGridOceanModel) stepped by the coupler
            # on a SHARED lat-lon grid (no cross-grid remap).  Phase 1 of
            # docs/ocean/coupled_3d_ocean_plan.md.
            self._init_dynamic_ocean(T_sfc_mean)
        else:
            raise ValueError(
                f"unknown ocean_mode {cfg.ocean_mode!r}; expected one of "
                f"'slab', 'two_layer', 'fixed', 'dynamic'.")

    def _init_dynamic_ocean(self, T_sfc_mean: float):
        """Build the prognostic 3D ``LatLonCGridOceanModel`` (ocean_mode=
        'dynamic') on the shared lat-lon grid with the OMIP-validated stable
        cold-start stack.  See docs/ocean/coupled_3d_ocean_plan.md (Phase 1)."""
        # Voronoi/MPAS ocean: a co-located 3-D MPAS ocean on the atmosphere's
        # Voronoi mesh has its own init recipe + edge/cell TRiSK staggering (NOT
        # the lat-lon C-grid stack below).  Dispatch early, mirroring the tripole
        # early-return.
        from legoesm.grids.voronoi import VoronoiMesh
        if isinstance(self._ocean_grid, VoronoiMesh):
            self._init_mpas_dynamic_ocean(T_sfc_mean)
            return
        from legoesm.grids.latlon import LatLonGrid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            idealized_bathymetry_latlon_cgrid,
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        cfg = self.coupled_cfg
        # Accept EITHER a regular lat-lon ocean grid (co-located with the
        # atmosphere, identity remap) OR a tripole (active-fold) curvilinear
        # ocean grid (a DIFFERENT grid coupled to the lat-lon atm via the
        # Phase-2 cross-grid conservative remap, make_grid_remapper).  Cube /
        # MPAS ocean grids stay unsupported (the cross-family overlap generator
        # is still deferred).  ``fold.is_active`` is a static Python bool set at
        # geometry-build time.
        _fold = getattr(self._ocean_grid, "fold", None)
        _is_tripole_grid = _fold is not None and getattr(_fold, "is_active", False)
        if not (isinstance(self._ocean_grid, LatLonGrid) or _is_tripole_grid):
            raise ValueError(
                "ocean_mode='dynamic' requires a regular lat-lon ocean grid "
                "(lat-lon atm + lat-lon 3D ocean co-located) OR a tripole "
                "(active-fold) ocean grid coupled via the Phase-2 cross-grid "
                f"remap; got ocean grid {type(self._ocean_grid).__name__}. "
                "Cube / MPAS ocean grids need the deferred cross-family "
                "spherical-overlap remap.")
        if _is_tripole_grid:
            # Tripole ocean: build with the OMIP-validated cold-start recipe
            # (NOT the lat-lon smc03/rk3 recipe below — they differ and mixing
            # them blows the cold start up; see top-risk #2 in the build plan).
            self._init_tripole_dynamic_ocean()
            return

        # The 3D ocean needs the full LatLonCGridOceanConfig; build a default
        # one if the caller passed a SimpleOceanConfig.
        _oc = cfg.ocean_config
        if not isinstance(_oc, LatLonCGridOceanConfig):
            _oc = LatLonCGridOceanConfig.from_flat()
        # Force the OMIP-validated cold-start stack (the defaults
        # explicit_substep barotropic + euler momentum give O(30 m/s) day-1
        # transients on a WOA/strat cold start; see omip_latlon_75lev_solved /
        # omip_rk3_coldstart_solve).  C_smag_lap + smag_cfl_safety are the
        # CFL-capped Laplacian-Smagorinsky eddy viscosity that damps the
        # geostrophic-adjustment transient off the (rest-velocity) WOA density
        # field — WITHOUT it the standalone ocean blows up from the WOA IC alone
        # (max|u| 21 m/s, eta 84 m in 2 h, NaN by 10 h; the OMIP latlon
        # cold-start recipe uses exactly --C-smag-lap 3.0 --smag-cfl-safety
        # 0.125; see omip_smag_cap_stabilizer).
        # replace_flat routes the flat names into their nested sub-configs
        # (barotropic_solver -> BarotropicConfig #640; C_smag_lap /
        # smag_cfl_safety -> LateralViscosityConfig #661) exactly like the
        # hand-nested ``_replace`` it replaced — leaf-identical by probe.
        ocfg = _oc.replace_flat(
            barotropic_solver="implicit_cn",
            momentum_time_integrator="rk3",
            pgf_scheme="smc03",
            implicit_vertical_mixing=True,
            # Additive Laplacian-Smagorinsky (the "none" lateral-friction
            # closure); force scheme="none" so a caller-supplied om4p25/qg_leith
            # config does not trip the model's "no additive A_h/C_smag with a
            # closure scheme" validation (codex LOW).
            lateral_friction_scheme="none",
            C_smag_lap=3.0,
            smag_cfl_safety=0.125,
        )

        if cfg.ocean_dt_s <= 0.0:
            raise ValueError(
                f"ocean_dt_s must be > 0, got {cfg.ocean_dt_s!r}")
        from legoesm.core.precision import get_policy
        _sd = get_policy().storage
        z_star = create_ocean_z_star(cfg.ocean_nlev, H_max=cfg.ocean_H_max_m)
        # WOA surface T/S restoring targets (set in the ocean_ic=="woa" branches;
        # None => no restoring target available => relaxation skipped).
        self._ocean_T_target = None
        self._ocean_S_target = None

        if cfg.ocean_ic == "woa":
            # REALISTIC cold start: WOA18 reanalysis T/S + WOA-derived continents
            # AND WOA-derived bathymetry on an OMIP-validated PARTIAL-CELL coord.
            # The SAME ocean mask seeds the ocean land_mask AND the atmosphere
            # f_land (_init_coupler, f_land_mode='from_ocean') on the shared grid,
            # so the wet masks agree exactly (codex Phase-1 HIGH).
            #
            # A stable WOA cold start REQUIRES all of the following — each one is
            # empirically necessary (the flat-bottom z-star coord blows up from
            # the WOA IC alone; this geometry survives at max|u| ~ 1 m/s):
            #   * REAL WOA bathymetry (not flat H_max) + make_partial_cell_latlon
            #     bathy smoothing + thin-cell snap + min_levels masking;
            #   * the PARTIAL-CELL coordinate, which ACTIVATES the smc03 density-
            #     Jacobian PGF (a plain z-star coord silently falls back to the
            #     centered-difference PGF whose truncation error on the sharp WOA
            #     pycnocline seeds the geostrophic-adjustment blow-up);
            #   * apply_balanced_init (geostrophic / level-of-no-motion balance);
            #   * the CFL-capped Laplacian-Smagorinsky viscosity (set in ocfg).
            from legoesm.ocean.init_woa import (
                woa_ocean_mask, woa_ocean_bathymetry, init_ocean_from_woa,
            )
            from legoesm.ocean.init_latlon_cgrid import (
                apply_balanced_init, make_partial_cell_latlon,
            )
            if not cfg.woa_t_path or not cfg.woa_s_path:
                raise ValueError(
                    "ocean_ic='woa' requires woa_t_path and woa_s_path "
                    f"(got woa_t_path={cfg.woa_t_path!r}, "
                    f"woa_s_path={cfg.woa_s_path!r}).")
            ocean_mask = woa_ocean_mask(self._ocean_grid, cfg.woa_t_path)
            H_woa = woa_ocean_bathymetry(
                self._ocean_grid, cfg.woa_t_path, H_max=cfg.ocean_H_max_m)
            # Smooth + thin-cell snap + shallow-mask -> partial-cell coordinate
            # (this is the coord the MODEL steps on, NOT the plain z-star).
            model_z_coord, H_snap, land_np = make_partial_cell_latlon(
                z_star, H_woa, ocean_mask,
            )
            land_mask = jnp.asarray(land_np, dtype=_sd)   # 1=ocean, 0=land
            _wet = land_np > 0.5
            # Bathymetry on the state: snapped depth on wet cells, H_max on land
            # (land cells are inert — gated by land_mask, not depth).
            H_state = np.where(_wet, H_snap, cfg.ocean_H_max_m)
            base_state = rest_state_latlon_cgrid_ocean(
                self._ocean_grid, z_star,
                land_mask_override=land_mask,
                H_bathy_override=jnp.asarray(H_state, dtype=_sd),
            )
            # Observed T [degC] / S [PSU] on model levels, masking cells below
            # the local bathymetry so deep cells get the abyssal fill (not a
            # surface-extended profile).  bathymetry_depth must be strictly
            # positive everywhere (land uses H_max — inert).
            T_woa, S_woa = init_ocean_from_woa(
                self._ocean_grid, z_star,
                T_path=cfg.woa_t_path, S_path=cfg.woa_s_path, interp="bilinear",
                bathymetry_depth=H_state,
            )
            _expect = base_state.T.data.shape
            if tuple(T_woa.shape) != tuple(_expect):
                raise ValueError(
                    f"WOA T/S shape {tuple(T_woa.shape)} != ocean state "
                    f"{tuple(_expect)} — grid/level mismatch.")
            self._ocean_state = base_state._replace(
                T=base_state.T.replace(data=jnp.asarray(T_woa, dtype=_sd)),
                S=base_state.S.replace(data=jnp.asarray(S_woa, dtype=_sd)),
            )
            # WOA surface restoring targets (top model level [degC]/[PSU]) — the
            # climatology the optional Newtonian relaxation anchors the surface
            # toward during the coupled spin-up (CoupledConfig.ocean_restore_*).
            self._ocean_T_target = jnp.asarray(T_woa[..., 0], dtype=_sd)
            self._ocean_S_target = jnp.asarray(S_woa[..., 0], dtype=_sd)
            # Balanced cold start (MANDATORY for WOA): the rest-velocity state
            # leaves WOA's baroclinic PGF UNBALANCED -> a violent geostrophic
            # adjustment that goes nonlinear (the standalone ocean blows from
            # the WOA IC alone: max|u| 21 m/s, eta 84 m in 2 h, NaN by 10 h).
            # apply_balanced_init seeds u/v/eta in geostrophic / level-of-no-
            # motion balance so there is no adjustment shock (OMIP-validated
            # cold-start; see omip_smag_cap_stabilizer / omip_rk3_coldstart).
            # Runs on the PARTIAL-CELL coord the model steps on (model_z_coord).
            self._ocean_state = apply_balanced_init(
                self._ocean_state, self._ocean_grid, model_z_coord, ocfg,
            )
        elif cfg.ocean_ic == "rest":
            # Idealized aquaplanet rest state: flat bottom, ALL-OCEAN (no polar
            # land caps), matching an f_land=0 atmosphere so the same-grid wet
            # masks AGREE (codex HIGH: caps vs all-ocean atm leak atm-side
            # fluxes onto ocean-masked cells).
            model_z_coord = z_star
            H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
                self._ocean_grid, H_max=cfg.ocean_H_max_m,
                land_lat_threshold=90.0,
            )
            self._ocean_state = rest_state_latlon_cgrid_ocean(
                self._ocean_grid, z_star,
                land_mask_override=land_mask, H_bathy_override=H_bathy,
            )
        else:
            raise ValueError(
                f"ocean_ic must be 'rest' or 'woa', got {cfg.ocean_ic!r}.")

        self._ocean_model = LatLonCGridOceanModel(
            self._ocean_grid, model_z_coord, ocfg,
        )
        self._ocean_step = self._ocean_model.step
        self._ocean_z_coord = model_z_coord
        self._ocean_land_mask = land_mask
        self._is_dynamic_ocean = True
        _ocean_frac = float(jnp.mean(land_mask))
        logger.info(
            "  Ocean: mode=dynamic (3D LatLonCGridOceanModel), "
            f"ic={cfg.ocean_ic}, ocean_frac={_ocean_frac:.2f}, "
            f"nlev={cfg.ocean_nlev}, ocean_dt={cfg.ocean_dt_s}s, "
            f"barotropic={ocfg.barotropic.barotropic_solver}, "
            f"momentum={ocfg.momentum_time_integrator}, pgf={ocfg.pgf_scheme}")

    def _build_tripole_ocean_config(self):
        """The OMIP-validated tripole ``LatLonCGridOceanConfig`` — cloned VERBATIM
        from the standalone forced-ocean recipe (``run_omip.py`` tripole branch,
        the config validated by the 20-yr production run): adcroft PGF +
        implicit_cn barotropic (30 substeps) + implicit vertical mixing + KPP +
        enhanced-diffusion convection + GM/Redi 600 + C_smag_lap 0.33 + tvd
        tracer advection + 1e-3 bottom drag + virtual-salt freshwater.  The
        coupled run delivers atm fluxes as an ``OceanSurfaceForcing`` consumed by
        the SAME dynamics-core external-tau block the OMIP run uses, so the
        surface-forcing scheme is "none" (the coupler is the forcing source)."""
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig, KPPConfig,
        )
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig, GMRediConfig, VisbeckConfig,
        )
        physics = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            vertical_mixing=VerticalMixingConfig(
                scheme="kpp", kpp=KPPConfig(K_conv=1.0),
            ),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
            ),
            shortwave_penetration=None,
        )
        # A_h=None = DERIVE from the mesh's narrowest wet cell, the same
        # rule the forced-ocean recipe uses (resolved below, once the mask is
        # read).  A fixed 1e5 m^2/s is a ~1 degree value: at 1/12 degree it is
        # 26x over the explicit Laplacian limit and a cold start diverges at
        # step 10, and this driver accepts an arbitrary tripole mesh.
        return LatLonCGridOceanConfig.from_flat(
            A_h=None, A_v=1.0e-4, K_v=1.0e-5, B_h=0.0,
            C_smag_lap=0.33,
            n_barotropic_substeps=30,
            barotropic_solver="implicit_cn",
            barotropic_implicit_pcg_tol=1e-10,
            barotropic_implicit_pcg_maxiter=300,
            pgf_scheme="adcroft",
            implicit_vertical_mixing=True,
            tracer_advection="tvd",
            bottom_drag_r=1e-3,
            bottom_drag_bbl_thickness=100.0,
            bottom_drag_bg_velocity=0.1,
            freshwater_closure="virtual_salt_flux",
            gm_redi=GMRediConfig(
                kappa_GM=600.0, kappa_Redi=600.0, S_max=0.005,
                visbeck=VisbeckConfig(enabled=False),
                slope_scheme="centered",
            ),
            physics=physics,
        )

    def _init_tripole_dynamic_ocean(self):
        """Build the 3D ``LatLonCGridOceanModel`` on a tripole (eORCA) grid with
        the OMIP-validated cold-start stack: NEMO mesh land mask + bathymetry ->
        partial-cell coordinate -> WOA18 (or rest) IC -> balanced init, stepped
        with the cloned OMIP tripole config.  Geometry/IC are READ from the SAME
        NEMO mesh_mask the tripole geometry was built from (``tripole_mesh_path``)
        so the wet domain is identical to the standalone OMIP run we already
        validated as stable (memory: omip_latlon_75lev_solved /
        omip_rk3_coldstart_solve — do NOT re-tune; reuse)."""
        import numpy as np
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean, apply_balanced_init,
            make_partial_cell_latlon,
        )
        from legoesm.ocean.init_tripole import (
            read_mesh_mask_bathy, compute_woa_3d,
        )
        from legoesm.core.precision import get_policy

        cfg = self.coupled_cfg
        if not cfg.tripole_mesh_path:
            raise ValueError(
                "a tripole dynamic ocean requires tripole_mesh_path (the NEMO "
                "eORCA mesh_mask file the land mask + bathymetry are read from).")
        if cfg.ocean_dt_s <= 0.0:
            raise ValueError(f"ocean_dt_s must be > 0, got {cfg.ocean_dt_s!r}")
        _sd = get_policy().storage

        ocfg = self._build_tripole_ocean_config()
        z_star = create_ocean_z_star(cfg.ocean_nlev, H_max=cfg.ocean_H_max_m)

        # NEMO land mask (1=ocean) + total wet-column bathymetry [m].
        land_np, H_nemo = read_mesh_mask_bathy(cfg.tripole_mesh_path)
        # Smooth + thin-cell snap + shallow-mask -> partial-cell coordinate
        # (ACTIVATES the adcroft density-Jacobian PGF; a plain z-star silently
        # falls back to the centered-diff PGF that blows up on sharp bathymetry).
        model_z_coord, H_snap, land_np = make_partial_cell_latlon(
            z_star, np.asarray(H_nemo), np.asarray(land_np),
        )
        land_mask = jnp.asarray(land_np, dtype=_sd)        # 1=ocean, 0=land

        # Resolve the mesh-derived lateral viscosity now that the grid and its
        # (snapped) land mask are both in hand; the model refuses an
        # unresolved one.
        if ocfg.lateral_viscosity.A_h is None:
            from legoesm.grids.tripole import DEFAULT_MIN_DX_M
            from legoesm.ocean.state import (
                resolution_scaled_lateral_viscosity,
                wet_min_spacing,
            )
            _lv = ocfg.lateral_viscosity
            _dx_min = wet_min_spacing(self._ocean_grid, land_np,
                                      clamp_floor_m=DEFAULT_MIN_DX_M)
            ocfg = ocfg._replace(lateral_viscosity=_lv._replace(
                A_h=resolution_scaled_lateral_viscosity(_dx_min, _lv),
                A_h_dx_m=_dx_min))
            print(f"  Coupled tripole ocean: lateral viscosity from the mesh, "
                  f"A_h={ocfg.lateral_viscosity.A_h:.6g} m2/s "
                  f"(narrowest wet cell {_dx_min:.1f} m)")
        _wet = np.asarray(land_np) > 0.5
        H_state = np.where(_wet, H_snap, cfg.ocean_H_max_m)
        base_state = rest_state_latlon_cgrid_ocean(
            self._ocean_grid, z_star,
            land_mask_override=land_mask,
            H_bathy_override=jnp.asarray(H_state, dtype=_sd),
        )

        if cfg.ocean_ic == "woa":
            if not cfg.woa_t_path or not cfg.woa_s_path:
                raise ValueError(
                    "ocean_ic='woa' requires woa_t_path and woa_s_path "
                    f"(got {cfg.woa_t_path!r}, {cfg.woa_s_path!r}).")
            # WOA18 T/S on the partial-cell levels (NaN-aware, flood-filled for
            # NEMO-ocean cells WOA lacks data for, deep-filled below seafloor) —
            # the shared OMIP loader so the coupled IC matches the standalone.
            T_woa, S_woa = compute_woa_3d(
                self._ocean_grid, model_z_coord,
                cfg.woa_t_path, cfg.woa_s_path, H_state, land_np,
            )
            _expect = base_state.T.data.shape
            if tuple(T_woa.shape) != tuple(_expect):
                raise ValueError(
                    f"WOA T/S shape {tuple(T_woa.shape)} != ocean state "
                    f"{tuple(_expect)} — grid/level mismatch.")
            self._ocean_state = base_state._replace(
                T=base_state.T.replace(data=jnp.asarray(T_woa, dtype=_sd)),
                S=base_state.S.replace(data=jnp.asarray(S_woa, dtype=_sd)),
            )
            # WOA surface restoring targets (top level [degC]/[PSU]) — see the
            # co-located path; anchors the surface during the coupled spin-up.
            self._ocean_T_target = jnp.asarray(T_woa[..., 0], dtype=_sd)
            self._ocean_S_target = jnp.asarray(S_woa[..., 0], dtype=_sd)
            self._ocean_state = apply_balanced_init(
                self._ocean_state, self._ocean_grid, model_z_coord, ocfg,
            )
        elif cfg.ocean_ic == "rest":
            # Rest state on the NEMO geometry (exponential T / uniform S from
            # rest_state_latlon_cgrid_ocean above) — no balanced init needed.
            self._ocean_state = base_state
        else:
            raise ValueError(
                f"ocean_ic must be 'rest' or 'woa', got {cfg.ocean_ic!r}.")

        self._ocean_model = LatLonCGridOceanModel(
            self._ocean_grid, model_z_coord, ocfg,
        )
        self._ocean_step = self._ocean_model.step
        self._ocean_z_coord = model_z_coord
        self._ocean_land_mask = land_mask
        self._is_dynamic_ocean = True
        _ocean_frac = float(jnp.mean(land_mask))
        logger.info(
            "  Ocean: mode=dynamic (3D LatLonCGridOceanModel, TRIPOLE), "
            f"ic={cfg.ocean_ic}, ocean_frac={_ocean_frac:.2f}, "
            f"nlev={cfg.ocean_nlev}, ocean_dt={cfg.ocean_dt_s}s, "
            f"barotropic={ocfg.barotropic.barotropic_solver}, pgf={ocfg.pgf_scheme}, "
            f"mesh={cfg.tripole_mesh_path}")

    def _init_mpas_dynamic_ocean(self, T_sfc_mean: float):
        """Build the prognostic 3-D MPAS ocean (``MPASOceanModel``) CO-LOCATED on
        the atmosphere's Voronoi mesh, from the OMIP-validated NEMO-match dycore
        recipe (the single source of truth ``nemo_match_mpas_model_config``, the
        same config scripts/run/run_omip.py steps) + a stratified rest cold
        start.

        MPAS uses TRiSK C-grid staggering — edge-normal velocity, cell-centred
        T/S/eta — unlike the lat-lon Arakawa-C ocean, so SST + surface currents
        are read back through the Perot edge->cell reconstruction in the MPAS
        branch of :meth:`_ocean_surface_KuvC`.  Surface fluxes are supplied
        EXTERNALLY by the coupler (``surface_forcing`` scheme='none' — the
        dynamics-core external tau/q_net block is the sole consumer), exactly as
        the OMIP JRA55 forcing mode does; the recipe's idealized prescribed-wind
        + restoring forcing is off."""
        from legoesm.grids.voronoi import VoronoiMesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_mpas_model_config,
        )
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm import constants

        cfg = self.coupled_cfg
        if cfg.ocean_dt_s <= 0.0:
            raise ValueError(f"ocean_dt_s must be > 0, got {cfg.ocean_dt_s!r}")
        # Co-located: the ocean mesh IS the atmosphere's Voronoi mesh (identity
        # remap) — reuse it; never rebuild at a mismatched resolution.
        mesh = self._ocean_grid
        if not isinstance(mesh, VoronoiMesh):
            raise ValueError(
                "_init_mpas_dynamic_ocean requires a VoronoiMesh (MPAS) ocean "
                f"grid; got {type(mesh).__name__}.")
        z_coord = create_ocean_z_star(cfg.ocean_nlev, H_max=cfg.ocean_H_max_m)
        # Proven OMIP NEMO-match MPAS dycore + coefficients (locked by
        # tests/ocean/unit/test_recipes.py), with ONLY the two coupled-run
        # overlays:
        #   * surface_forcing scheme='none' so the coupler's air-sea tau/q_net
        #     (assembled in _assemble_ocean_forcing) is the SOLE surface forcing,
        #     NOT the recipe's idealized prescribed-wind + restoring default;
        #   * normalize_freshwater=True (the CORE-II net ~+0.65 Sv P-E+R input
        #     would otherwise accumulate as a global fresh drift) — matching the
        #     lat-lon/tripole from_flat(normalize_freshwater=True) path.
        base = nemo_match_mpas_model_config()
        config = base._replace(
            physics=base.physics._replace(
                surface_forcing=base.physics.surface_forcing._replace(
                    scheme="none")),
            normalize_freshwater=True,
        )
        self._ocean_model = MPASOceanModel(mesh, z_coord, config)
        # Stratified rest cold start (zero velocity/SSH, exponential T, uniform
        # S).  land_lat_threshold=90 => all-ocean, matching a co-located
        # aquaplanet atmosphere (f_land=0) so the wet masks agree — the same
        # rationale as the lat-lon ocean_ic='rest' branch.  The surface layer
        # starts near the atmosphere's mean SST (T_sfc_mean [K] -> degC) so the
        # cold start is not shocked by a large air-sea temperature jump.
        T_surf_C = float(T_sfc_mean) - float(constants.T_freeze)
        self._ocean_state = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=T_surf_C, H_max=cfg.ocean_H_max_m,
            land_lat_threshold=90.0,
        )
        self._ocean_step = self._ocean_model.step
        self._ocean_z_coord = z_coord
        self._ocean_land_mask = self._ocean_state.land_mask.data
        self._ocean_is_mpas = True
        self._is_dynamic_ocean = True
        # No climatological restoring target on the idealized cold start; the
        # optional surface relaxation (_apply_ocean_restoring) stays a no-op
        # unless a target is configured.
        self._ocean_T_target = None
        self._ocean_S_target = None
        _ocean_frac = float(jnp.mean(self._ocean_land_mask))
        logger.info(
            "  Ocean: mode=dynamic (3D MPASOceanModel, voronoi), ic=rest, "
            f"ocean_frac={_ocean_frac:.2f}, nlev={cfg.ocean_nlev}, "
            f"ocean_dt={cfg.ocean_dt_s}s, surface_forcing=none(coupled), "
            "normalize_freshwater=True")

    def _init_coupler(self):
        """Initialize coupler, land, ice, lake surface states."""
        from legoesm.core.precision import get_policy
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.coupler.coupler import init_surface_state, make_coupler
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.ice.config import SeaIceConfig
        from legoesm.land.config import LandConfig, MultiLayerLandConfig

        cfg = self.coupled_cfg
        shape_2d = self._atm.grid.grid_shape_2d
        _sd = get_policy().storage

        coupler_cfg = self._coupler_config or CouplerConfig()
        self._coupler_cfg = coupler_cfg  # reused by the 3D-ocean flux assembly
        ice_cfg = self._ice_config or SeaIceConfig()
        lake_cfg = self._lake_config or LakeConfig()

        # Select land config based on mode
        if cfg.land_mode == "none":
            land_cfg = LandConfig()  # won't be used (f_land=0)
        elif cfg.land_mode == "multilayer":
            if isinstance(cfg.land_config, MultiLayerLandConfig):
                land_cfg = cfg.land_config
            else:
                land_cfg = MultiLayerLandConfig()
        elif cfg.land_mode == "slab":
            land_cfg = cfg.land_config if isinstance(cfg.land_config, LandConfig) else LandConfig()
        else:
            raise ValueError(
                f"Unknown land_mode {cfg.land_mode!r}; "
                "expected 'none', 'slab', or 'multilayer'."
            )

        # Enable carbon in land config if carbon_active + differland
        if cfg.carbon_active and cfg.carbon_land == "differland":
            from legoesm.land.carbon.config import CarbonConfig
            carbon_cfg = CarbonConfig(scheme="differland")
            land_cfg = land_cfg._replace(carbon=carbon_cfg)

        # Calibrated config-level land parameters on the CLM default path (the
        # per-cell PFT params come from the provider; these are the global snow/ice
        # + bulk-transfer values tuned vs ERA5 under physical bounds).
        if (cfg.land_mode != "none"
                # getattr: optional-config compat gate (land_param_source selector)
                and getattr(cfg, "land_param_source", "analytical") == "clm"):
            import legoesm.land.clm_surface_map as _csm
            if cfg.land_mode == "multilayer":
                # Multilayer carries the full trainable snow feedback (cover threshold +
                # fresh/aged brightness + age decay) from the 2026-07 full-grid recalibration.
                ch = _csm.TUNED_CH_MULTILAYER
                alb = land_cfg.land_albedo._replace(
                    alpha_snow_max=_csm.TUNED_SNOW_ALBEDO_MAX_MULTILAYER,
                    alpha_snow_min=_csm.TUNED_SNOW_ALBEDO_MIN_MULTILAYER,
                    snow_depth_crit=_csm.TUNED_SNOW_DCRIT_MULTILAYER,
                    tau_snow_decay=_csm.TUNED_SNOW_TAU_DAYS_MULTILAYER * 86400.0,  # days -> s
                    soil_dry_albedo_boost=_csm.TUNED_SOIL_DRY_BOOST_MULTILAYER)   # deserts
            else:
                ch = _csm.TUNED_CH
                alb = land_cfg.land_albedo._replace(alpha_snow_max=_csm.TUNED_SNOW_ALBEDO_MAX)
            land_cfg = land_cfg._replace(
                Ch_land=ch, Cd_land=ch, snow_albedo_feedback=True, land_albedo=alb)
            logger.info(f"  Land: ERA5-calibrated Ch/snow params "
                        f"({cfg.land_mode} CLM default path)")

        # Spatial soil hydraulics from the CLM reference map (per-column van-
        # Genuchten retention) for the Richards multilayer land.
        if (cfg.land_mode == "multilayer"
                # getattr: optional-config compat gate (land_param_source selector)
                and getattr(cfg, "land_param_source", "analytical") == "clm"
                and self._atm._grid_lat is not None):
            from legoesm.land.clm_surface_map import (
                download_clm_surfdata, load_clm_surface, clm_hydraulics_config,
                clm_multilayer_thermal_config, clm_multilayer_ch,
                TUNED_PFT_SNOWMASK_MULTILAYER)
            lat = self._atm._grid_lat; lon = self._atm._grid_lon
            lat_d = np.asarray(jnp.rad2deg(jnp.broadcast_to(lat, shape_2d)).ravel())
            lon_d = np.asarray(jnp.rad2deg(jnp.broadcast_to(lon, shape_2d)).ravel())
            smap = load_clm_surface(download_clm_surfdata(), lat_d, lon_d)
            # per-cell calibrated soil hydraulics (van-Genuchten), thermal inertia
            # (C_soil/k_solid -> seasonal cycle) and bulk exchange Ch.  Cast to the
            # storage dtype so the (float64) PFT-table matmuls do not silently down-
            # cast into the (possibly float32) land state on every scatter update.
            _c = lambda x: x.astype(_sd) if isinstance(x, jnp.ndarray) else x
            cast = lambda t: jax.tree.map(_c, t)   # cast only the array fields
            ch_cell = clm_multilayer_ch(smap).astype(_sd)
            # v7 per-PFT canopy snow masking (per-cell scale on the snow-cover
            # fraction; glacier blends to 1 — no canopy on ice).  Set here, where
            # the surface map exists: the scalar snow-constant block above cannot
            # carry a per-cell field (codex: coupled runs were missing the mask).
            _mask_cell = (
                (1.0 - jnp.asarray(smap["glacier_frac"]))
                * (jnp.asarray(smap["pft_fractions"])
                   @ jnp.asarray(TUNED_PFT_SNOWMASK_MULTILAYER))
                + jnp.asarray(smap["glacier_frac"])).astype(_sd)
            land_cfg = land_cfg._replace(
                hydraulics=cast(clm_hydraulics_config(smap)),
                thermal=cast(clm_multilayer_thermal_config(smap)),
                land_albedo=land_cfg.land_albedo._replace(
                    snow_cover_scale=_mask_cell),
                Ch_land=ch_cell, Cd_land=ch_cell)
            logger.info("  Soil: CLM reference VG + per-PFT thermal/Ch map (per-column)")

            # Sub-grid elevation-band snow (opt-in): band elevations from the CLM
            # STD_ELEV map so warm cells keep bright snow on their cold high fractions.
            if getattr(cfg, "land_elev_bands", False):
                from legoesm.land.snow_bands import (
                    ElevationSnowBandConfig, band_elevation_anomalies)
                band_dz = band_elevation_anomalies(
                    jnp.asarray(smap["std_elev"], dtype=_sd))
                land_cfg = land_cfg._replace(
                    elev_bands=ElevationSnowBandConfig(band_dz=band_dz))
                logger.info("  Snow: sub-grid elevation-band scheme (CLM STD_ELEV, "
                            "5 equal-area bands)")

        # Coupled DIURNAL surface model (default ON for the multilayer land): the
        # coupled atmosphere supplies a fully-resolved diurnal cycle at a single,
        # consistent lowest-model-level height, so the surface exchange can be the
        # physical Monin-Obukhov (MOST) scheme (roughness-driven, stability-dependent)
        # and transpiration the Farquhar photosynthesis-stomata coupling — both of
        # which are ill-posed against the crude offline single-column forcing but
        # well-posed here.  Carbon must run (differland) so Farquhar has a prognostic
        # LAI; the coupler already initialises + threads the carbon state.
        if (cfg.land_mode == "multilayer"
                and cfg.land_diurnal_surface):
            land_cfg = enable_diurnal_surface_land(land_cfg)
            logger.info("  Land surface: MOST exchange + Farquhar stomata "
                        "(coupled diurnal model)")

        self._land_cfg = land_cfg  # store for diagnostics

        # PFT parameter provider (if requested and land is active)
        land_param_provider = None
        if cfg.use_pft and cfg.land_mode != "none":
            land_param_provider = self._build_pft_provider(shape_2d)

        # Optional spun-up land carbon IC (the finidat global_carbon_ic.npz):
        # INGEST the seeded per-cell 8-pool CarbonState + the per-cell permafrost
        # phi so the coupled run starts carbon at its mapped equilibrium and
        # MAINTAINS the seeded permafrost SOC (phi -> make_coupler's f_perma
        # protection), instead of cold-starting carbon and decomposing the seed.
        # Only meaningful for the multilayer differland carbon column; a carbon IC
        # with slab / carbon-off land is a caller error (fail loud, never a silent
        # no-op).  ""/None (default) => cold-start + no protection (byte-identical).
        carbon_override = None
        land_soil_frozen_fraction = None
        carbon_ic_path = getattr(cfg, "carbon_ic_path", "")
        if carbon_ic_path:
            import math

            from legoesm.land.carbon.global_init import load_finidat_carbon_ic
            from legoesm.land.config import MultiLayerLandConfig as _MLLC
            if not (isinstance(land_cfg, _MLLC)
                    and land_cfg.carbon.scheme == "differland"):
                raise ValueError(
                    f"carbon_ic_path={carbon_ic_path!r} requires multilayer land "
                    "with the differland carbon scheme (land_mode='multilayer', "
                    f"carbon active); got land_config {type(land_cfg).__name__} / "
                    f"carbon scheme {land_cfg.carbon.scheme!r}.")
            if self._atm._grid_lat is None or self._atm._grid_lon is None:
                raise ValueError(
                    "carbon_ic_path needs the atmosphere grid lat/lon to grid-"
                    "match the finidat; the atmosphere exposes none.")
            ncol = int(math.prod(shape_2d))
            lat_deg = np.rad2deg(np.asarray(self._atm._grid_lat)).reshape(-1)
            lon_deg = np.rad2deg(np.asarray(self._atm._grid_lon)).reshape(-1)
            carbon_override, land_soil_frozen_fraction = load_finidat_carbon_ic(
                carbon_ic_path, expect_ncol=ncol,
                target_lat_deg=lat_deg, target_lon_deg=lon_deg)
            if carbon_override is None:
                raise ValueError(
                    f"carbon_ic_path={carbon_ic_path!r} is not a carbon finidat "
                    "(missing the 8 CarbonState pool fields); expected a "
                    "global_carbon_ic.npz from build_global_carbon_ic.py.")
            logger.info(
                "  Land carbon IC: seeded %d-column CarbonState from finidat %s "
                "(permafrost phi %s)", ncol, carbon_ic_path,
                "threaded" if land_soil_frozen_fraction is not None else "absent")

        # Build coupler step function
        self._step_surface = make_coupler(
            coupler_cfg, land_cfg, ice_cfg, lake_cfg,
            lat=self._atm._grid_lat,
            grid=self._atm.grid,
            land_param_provider=land_param_provider,
            land_soil_frozen_fraction=land_soil_frozen_fraction,
        )

        # Initialize surface state.  Optionally warm-start the soil at the
        # atmosphere's lat-structured near-surface air temperature (t=0) — the
        # same spatial source the slab SST uses — so tropical land does not
        # cold-spin from a uniform 280 K (default off => byte-identical).
        soil_kwargs = {}
        if cfg.warm_start_soil:
            soil_kwargs["T_soil_init"] = self._atm.state.T.data[..., -1]
            logger.info("  Soil warm-start: T_soil init = atm near-surface air T")
        self._sfc_state = init_surface_state(
            shape_2d, land_config=land_cfg, carbon_override=carbon_override,
            ice_config=ice_cfg, **soil_kwargs,
        )

        # Tile fractions
        if cfg.f_land_mode == "zero":
            f_land = jnp.zeros(shape_2d, dtype=_sd)
        elif cfg.f_land_mode == "analytical":
            # Use driver's land mask if non-trivial; otherwise generate one
            # based on latitude (simple continents approximation).
            if (self._atm._f_land is not None
                    and float(jnp.max(self._atm._f_land)) > 0):
                f_land = self._atm._f_land
            elif cfg.land_mode != "none":
                # Generate analytical land mask: ~30% land by area
                # Land at |lat| > 20 in two longitude sectors
                lat = self._atm._grid_lat
                if lat is not None:
                    if lat.ndim < len(shape_2d):
                        lat_2d = jnp.broadcast_to(
                            lat.reshape(lat.shape + (1,) * (len(shape_2d) - lat.ndim)),
                            shape_2d,
                        )
                    else:
                        lat_2d = lat
                    abs_lat = jnp.abs(lat_2d) * 180.0 / jnp.pi
                    # Land where |lat| > 25 degrees (crude polar/midlat continents)
                    f_land = jnp.where(abs_lat > 25.0, 0.5, 0.0).astype(_sd)
                else:
                    f_land = jnp.zeros(shape_2d, dtype=_sd)
            else:
                f_land = jnp.zeros(shape_2d, dtype=_sd)
        elif cfg.f_land_mode == "from_ocean":
            # f_land from the dynamic-ocean WOA-derived wet mask (1=ocean): the
            # atmosphere land fraction and the 3D-ocean wet mask come from ONE
            # source on the shared lat-lon grid, so they agree exactly (no atm
            # surface flux leaks onto an ocean-masked cell; codex Phase-1 HIGH).
            # _init_ocean runs before _init_coupler so the mask is available.
            ocean_mask = getattr(self, "_ocean_land_mask", None)
            if ocean_mask is None:
                raise ValueError(
                    "f_land_mode='from_ocean' requires the prognostic 3D ocean "
                    "(ocean_mode='dynamic' + ocean_ic='woa') — no ocean wet "
                    "mask was initialised.")
            if tuple(ocean_mask.shape) != tuple(shape_2d):
                # Cross-grid (tripole ocean): the ocean wet mask lives on the
                # ocean grid, NOT the atm grid.  Remap it to the atm grid with
                # the conservative ocean->atm remapper to get the atm-cell OCEAN
                # FRACTION (a 0/1 mask area-averaged onto each atm cell -> a
                # fraction in [0,1]); f_land is its complement.  This keeps the
                # atm land fraction and the 3D-ocean wet mask consistent across
                # the two grids (no flux leak), the cross-grid analogue of the
                # shared-grid identity below.
                from legoesm.coupler.grid_remap import remap_field
                rem = getattr(self, "_grid_remapper", None)
                if rem is None or rem.o2a is None:
                    raise ValueError(
                        "from_ocean f_land on a non-shared ocean grid needs an "
                        "ocean->atm remapper, but none was built "
                        f"(ocean mask shape {tuple(ocean_mask.shape)} != atm "
                        f"{tuple(shape_2d)}).")
                ocean_frac = remap_field(
                    jnp.asarray(ocean_mask, dtype=_sd), rem.o2a)
                ocean_frac = jnp.clip(ocean_frac, 0.0, 1.0)
                f_land = (1.0 - ocean_frac).astype(_sd)
            else:
                f_land = (1.0 - ocean_mask).astype(_sd)
        else:
            # No silent fallback to f_land=0: a typo (e.g. 'from-ocean') would
            # make the atmosphere treat WOA land cells as ocean while the 3D
            # ocean still masks them dry — the exact mask-consistency leak this
            # mode exists to prevent (codex; CLAUDE.md dispatch-hardening).
            raise ValueError(
                f"unknown f_land_mode {cfg.f_land_mode!r}; expected one of "
                "'zero', 'analytical', 'from_ocean'.")

        self._tile_config = TileConfig(
            f_land=f_land,
            f_lake=jnp.zeros(shape_2d, dtype=_sd),
        )

        # Driver-level invariant on the MATERIALIZED land fraction (the residue
        # the CONFIG-level CLI helper apply_land_runoff_scheme documents it
        # cannot catch): a land model configured yet f_land identically zero =
        # a silently dead land tile.
        assert_land_tile_reachable(cfg.land_mode, cfg.f_land_mode, f_land)

        land_frac = float(jnp.mean(f_land))
        pft_str = " (PFT)" if land_param_provider is not None else ""
        logger.info(f"  Land: mode={cfg.land_mode}{pft_str}, "
                    f"f_land_mean={land_frac:.2f}")

    def _build_pft_provider(self, shape_2d):
        """Create the spatial land-parameter provider.

        ``land_param_source='clm'`` → CLM reference surfdata (real PFT map +
        reference soil); ``'analytical'`` → latitude-band PFT fractions."""
        import math

        from legoesm.land.param_providers import PFTParamProvider

        lat = self._atm._grid_lat
        if lat is None:
            logger.warning("  PFT requested but no latitude available; "
                           "falling back to scalar params")
            return None

        # getattr: optional-config compat gate (land_param_source selector)
        source = getattr(self.coupled_cfg, "land_param_source", "analytical")
        # Transient cover only rides the CLM path (it needs the CLM soil map to
        # freeze around).  Fail loudly rather than silently ignore the request.
        if (getattr(self.coupled_cfg, "transient_land_cover", False)
                and getattr(self.coupled_cfg, "land_cover_surfdata", "")
                and source != "clm"):
            raise ValueError(
                "transient_land_cover with land_cover_surfdata requires "
                f"land_param_source='clm' (got {source!r}); transient cover overlays "
                "the CLM reference soil map.")
        if source == "clm":
            from legoesm.land.clm_surface_map import clm_surface_provider
            lon = self._atm._grid_lon
            lat_deg = np.asarray(jnp.rad2deg(jnp.broadcast_to(lat, shape_2d)).ravel())
            lon_deg = np.asarray(jnp.rad2deg(jnp.broadcast_to(lon, shape_2d)).ravel())
            # Use the calibration matched to the active land scheme (each tuned its
            # surface-energy params against a different soil forward).
            variant = ("multilayer" if self.coupled_cfg.land_mode == "multilayer"
                       else "slab")
            # Transient land-use cover (opt-in): the vegetation params re-weight per
            # segment from a transient legoesm_surfdata cover; soil frozen.  Base +
            # per-year both go through clm_provider_rebuild so their albedo treatment
            # is consistent (a normal run without land_cover_surfdata is unchanged).
            cover_path = getattr(self.coupled_cfg, "land_cover_surfdata", "")
            if getattr(self.coupled_cfg, "transient_land_cover", False) and cover_path:
                from legoesm.land.clm_surface_map import (
                    clm_provider_rebuild, download_clm_surfdata, load_clm_surface,
                    load_transient_cover_on_columns, TransientCoverProvider)
                clm_path = getattr(self.coupled_cfg, "clm_surfdata_path", "") \
                    or download_clm_surfdata()
                m = load_clm_surface(clm_path, lat_deg, lon_deg)
                # Match the non-transient coupled provider (clm_surface_provider omits
                # the soil-colour albedo) so an unchanged cover slice is a no-op.
                rebuild = clm_provider_rebuild(m, variant=variant,
                                               include_soil_albedo=False)
                cover, years = load_transient_cover_on_columns(
                    cover_path, lat_deg, lon_deg)
                if cover.shape[1] != np.asarray(m["pft_fractions"]).shape[0]:
                    raise ValueError(
                        f"transient cover has {cover.shape[1]} columns but the CLM "
                        f"map has {np.asarray(m['pft_fractions']).shape[0]}; mismatch.")
                provider = TransientCoverProvider(
                    base=rebuild(m["pft_fractions"]), cover=cover, years=years,
                    _rebuild=rebuild)
                logger.info(
                    f"  Land params: CLM soil + TRANSIENT cover ({cover_path}, "
                    f"{cover.shape[0]} years {int(years[0])}-{int(years[-1])}, "
                    f"{variant} tuning), {lat_deg.size} columns")
                return provider
            provider = clm_surface_provider(lat_deg, lon_deg, variant=variant)
            logger.info(f"  Land params: CLM reference surfdata (real PFT map + "
                        f"reference soil, {variant} tuning), {lat_deg.size} columns")
            return provider
        if source != "analytical":
            raise ValueError(
                f"land_param_source must be 'analytical' or 'clm', got {source!r}.")

        # Flatten to (ncol,)
        lat_flat = jnp.ravel(lat) if lat.ndim > 1 else lat
        ncol = lat_flat.shape[0]
        if lat.ndim > 1:
            ncol = math.prod(shape_2d)
            lat_flat = jnp.broadcast_to(lat, shape_2d).ravel()

        abs_lat_deg = jnp.abs(lat_flat) * 180.0 / jnp.pi

        # 17 CLM5 PFTs — assign analytical fractions by latitude band
        # 0=bare_soil, 1=NET_temperate, 2=NET_boreal, 3=NDT_boreal,
        # 4=BET_tropical, 5=BET_temperate, 6=BDT_tropical, 7=BDT_temperate,
        # 8=BDT_boreal, 9=BES, 10=BDS_temperate, 11=BDS_boreal,
        # 12=C3_arctic, 13=C3_non_arctic, 14=C4, 15=crop, 16=bare_soil_2
        n_pft = 17
        fracs = jnp.zeros((ncol, n_pft))

        # Tropical broadleaf (|lat| < 15)
        tropical = (abs_lat_deg < 15.0).astype(jnp.float32)
        fracs = fracs.at[:, 4].set(0.7 * tropical)   # BET_tropical
        fracs = fracs.at[:, 14].set(0.3 * tropical)   # C4 grass

        # Temperate (15-45)
        temperate = ((abs_lat_deg >= 15.0) & (abs_lat_deg < 45.0)).astype(jnp.float32)
        fracs = fracs.at[:, 7].set(0.3 * temperate)   # BDT_temperate
        fracs = fracs.at[:, 13].set(0.4 * temperate)  # C3_non_arctic
        fracs = fracs.at[:, 15].set(0.3 * temperate)  # crop

        # Boreal (45-65)
        boreal = ((abs_lat_deg >= 45.0) & (abs_lat_deg < 65.0)).astype(jnp.float32)
        fracs = fracs.at[:, 2].set(0.5 * boreal)    # NET_boreal
        fracs = fracs.at[:, 12].set(0.3 * boreal)   # C3_arctic
        fracs = fracs.at[:, 0].set(0.2 * boreal)    # bare_soil

        # Polar (>65)
        polar = (abs_lat_deg >= 65.0).astype(jnp.float32)
        fracs = fracs.at[:, 0].set(0.7 * polar)     # bare_soil
        fracs = fracs.at[:, 12].set(0.3 * polar)    # C3_arctic

        provider = PFTParamProvider.from_defaults(fracs)
        logger.info(f"  PFT: {n_pft} types, {ncol} columns, "
                    f"analytical latitude-band fractions")
        return provider

    def _init_carbon(self):
        """Set up CO2 tracer in the atmosphere if carbon is active."""
        cfg = self.coupled_cfg
        if not cfg.co2_tracer:
            return

        # CO2 as a prognostic atmospheric tracer.
        # Mixing ratio: co2_ppmv * 1e-6 * (M_CO2 / M_air).
        # Molar masses come from ``legoesm.constants`` per CLAUDE.md
        # (no hardcoded physical constants in production code).
        co2_init_kgkg = (
            cfg.co2_ppmv_init * 1.0e-6
            * (constants.M_CO2 / constants.M_air)
        )

        # Get 3D shape from atmosphere state
        T_data = self._atm.state.T.data
        shape_3d = T_data.shape

        self._co2_field = jnp.full(shape_3d, co2_init_kgkg,
                                   dtype=T_data.dtype)
        logger.info(f"  CO2 tracer: init={cfg.co2_ppmv_init:.1f} ppmv "
                    f"({co2_init_kgkg:.6e} kg/kg), shape={shape_3d}")

    def _override_sst(self):
        """Replace the atmosphere's file-based SST with slab ocean SST."""
        # Store the original for fallback SIC
        self._original_get_sst_sic = self._atm.get_sst_sic

        def _coupled_get_sst_sic(day):
            from legoesm.coupler.grid_remap import remap_field
            # SST from the ocean, remapped onto the atmosphere grid (identity
            # remapper => unchanged) so the atm physics always sees atm-grid SST
            # even when the ocean runs on a different grid.
            sst = remap_field(self._ocean_surface_KuvC()[0],
                              self._grid_remapper.o2a)
            # SIC from prognostic sea ice (already on the atm grid, where the
            # coupler runs) or the file fallback.
            if (self._sfc_state is not None
                    and hasattr(self._sfc_state, 'ice')
                    and self._sfc_state.ice is not None):
                sic = _total_ice_sic(self._sfc_state.ice)
            else:
                _, sic = self._original_get_sst_sic(day)
            return sst, sic

        self._atm.get_sst_sic = _coupled_get_sst_sic
        logger.info("  SST override: atmosphere reads SST from slab ocean")

    def _override_sfc(self):
        """Feed the coupler's tile-blended dynamic surface albedo + skin
        temperature back to the atmosphere radiation each segment.

        Gated on ``CoupledConfig.couple_surface_radiation`` (default False ⇒
        no-op, existing coupled runs byte-identical).  When enabled, the
        sea-ice albedo feedback / zenith ocean albedo / snow brightening and
        the prognostic ice/land skin temperature — otherwise silently dropped
        — replace the atmosphere's frozen surface-property scalars.

        Returns grid-shaped arrays on EVERY segment (a static-blend seed
        before the first coupler step) so the ``SegmentForcing`` pytree
        structure is stable across the run — ``model_driver`` compiles the
        segment kernel once.
        """
        if not self.coupled_cfg.couple_surface_radiation:
            return

        from legoesm.forcing.surface_utils import (
            blend_surface_temperature,
            snow_for_albedo_deblend,
            surface_temperature_for_lw_boundary,
        )

        # Only correlated-k schemes (RRTMGP/RRTMG) honour the PAIRED dynamic
        # (T_rad, eps_grid) surface boundary, so they get the flux-conserving
        # radiative-equivalent temperature + dynamic emissivity.  Gray/none use
        # an idealized black surface (eps=1) and ignore the dynamic emissivity;
        # handing them T_rad (which is defined WITH eps_grid via
        # eps_grid*sigma*T_rad^4 = blended emission) would make sigma*T_rad^4
        # over-emit by 1/eps_grid, so they keep the area-weighted skin
        # temperature instead.  See physics_pipeline gray radiation_fn.
        _conservative_lw = getattr(
            self.atm_config, "radiation", "gray") in ("rrtmgp", "rrtmg")

        def _seed_blend(day):
            # Same static blend the atmosphere radiation would use, as
            # grid-shaped arrays — used only until the first sfc_response.
            #
            # This seed is an OVERRIDE: compute_radiation_core REPLACES its own
            # internal blend with it, so an ocean/ice-only albedo here does not
            # merely mis-report, it makes radiation reflect ~0.06 instead of
            # ~0.20 from every land column until the first coupler response
            # lands (#1556).  Take the pipeline's own ocean/ice/land blend —
            # the same single source of truth the emissivity seed below already
            # uses.  Note this REPLACES radiation's internal blend, so under
            # dynamic_albedo / a multilayer land tile the seed is the static
            # approximation static_surface_albedo documents, for the segments
            # before the first response.
            sst, sic = self._atm.get_sst_sic(day)
            acfg = self.atm_config
            _phys_seed = self._atm.physics
            alb = _phys_seed.static_surface_albedo(
                sic, land_active=_phys_seed.f_land is not None,
                lat=self._atm._grid_lat,
                # Empty before the first segment (there is no snow carry yet),
                # which is exactly when the seed is in charge; present on later
                # segments if the response is still absent.
                snow=snow_for_albedo_deblend(
                    getattr(self._atm, "_carry_aux", {}).get("snow"),
                    getattr(self._atm, "_ensemble_size", 1)))
            T = blend_surface_temperature(sst, sic, acfg.T_ice)
            if not _conservative_lw:
                # Gray/none never take an emissivity override (eps=1); return
                # None so the override leaf is CONSISTENTLY None across seed and
                # response — matching ``_coupled_get_sfc_override`` so no
                # None->array pytree transition / recompile occurs.
                return alb, T, None
            # Seed a GRID-SHAPED emissivity — NOT None.  A None->array transition
            # at the first coupler response would change the SegmentForcing pytree
            # structure (None has no leaf, an array does) and force a recompile,
            # violating the no-recompile invariant.  Use the EXACT static blend
            # the radiation pipeline emits with (ocean/ice/land, configured
            # emissivity_* values) so the seeded boundary that radiation uses
            # before the first response is the SAME emissivity the lw_net_sfc
            # inversion reconstructs with (otherwise land cells bias the initial
            # lw_down).
            _phys = self._atm.physics
            eps = _phys.static_surface_emissivity(
                sic, land_active=_phys.f_land is not None)
            return alb, T, eps

        def _coupled_get_sfc_override(day):
            r = self._last_sfc_response
            if r is None or getattr(r, "albedo", None) is None:
                return _seed_blend(day)
            if not _conservative_lw:
                # Gray/none emit as an idealized BLACK surface (eps = 1) and
                # cannot honour the canopy's eps_col, so feed them a black-surface
                # BRIGHTNESS temperature derived from the full upward flux
                # (sigma*T_bb^4 = LW_out).  Feeding T_rad would emit
                # sigma*T_rad^4 = LW_emit/eps_col and overstate canopy emission by
                # ~1/eps_col.  No emissivity override (gray keeps eps = 1).  NOT
                # the aerodynamic/sensible-heat T_sfc (the canopy air-space Tc).
                T_bb = surface_temperature_for_lw_boundary(
                    "gray", T_rad=getattr(r, "T_rad", r.T_sfc), lw_up=r.lw_up)
                return r.albedo, T_bb, None
            eps = getattr(r, "emissivity", None)
            if eps is None:
                # Keep the override leaf a constant-shape array (no recompile).
                eps = _seed_blend(day)[2]
            # RRTMGP/RRTMG: the radiative-equivalent T_rad + dynamic eps_grid
            # (flux-conserving tile blend) make eps*sigma*T_rad^4 + (1-eps)*La
            # equal the area-weighted sum of tile lw_up exactly for mixed cells.
            return r.albedo, getattr(r, "T_rad", r.T_sfc), eps

        self._atm.get_sfc_override = _coupled_get_sfc_override
        logger.info(
            "  Surface-radiation feedback: dynamic albedo + skin T + emissivity"
            " -> radiation"
        )

    def _override_sfc_fluxes(self):
        """Feed the SHARED air-sea surface SH/LH fluxes back to the atmosphere
        surface tendency each segment so the air-sea heat+water budget closes.

        Gated on ``CoupledConfig.couple_surface_fluxes`` (default False => no-op,
        existing coupled runs byte-identical).  When enabled, the atmosphere
        consumes the coupler's TILE-BLENDED surface ``shflx``/``lhflx`` (its bulk
        scheme, q_sfc = 0.98*q_sat mixing ratio, ocean-tile C_H/C_E) instead of
        recomputing its OWN INDEPENDENT bulk fluxes.  This is the blended TOTAL
        surface flux (f_ocean*ocean + f_ice*ice + f_land*land + f_lake*lake), the
        correct forcing for the atmosphere bottom-level tendency over ALL tiles.

        Air-sea budget closure: the blended flux fed to the atmosphere and the
        ocean-tile ``tile.shflx``/``tile.lhflx`` that the dynamic ocean q_net
        SINKS BOTH derive from the SAME coupler ``ocean_tile_response`` -- the
        atmosphere no longer runs its own independent surface bulk calc, so over
        the ocean fraction the two are the same partition (blended ocean
        component = f_ocean*ocean_tile; the ocean column receives ocean_tile over
        its wet area).  For an all-ocean cell blended == ocean_tile EXACTLY, so
        heat/water leaving the ocean == heat/water entering the atmosphere
        (proven in tests/unit/test_air_sea_flux_coupling_conservation.py).  Over
        a MIXED cell the atmosphere is correctly forced by the blended flux while
        the ocean gets the ocean component -- the air-sea (ocean) exchange still
        closes; land/ice/lake heat goes to those reservoirs, not the ocean.

        Sign convention: ``shflx``/``lhflx`` are [W/m2, positive UP =
        surface->atmosphere], exactly the convention the atmosphere's bulk
        ``shflx``/``lhflx`` use; the ocean ``q_net = ... - shflx - lhflx`` SINKS
        the SAME positive-up fluxes, so a unit of heat leaving the ocean arrives
        as a unit of heat into the atmosphere (and evap mass leaving the ocean
        arrives as vapour into the atmosphere).

        Returns ``(None, None)`` until the first coupler step populates
        ``_last_sfc_response`` (segment 0 falls back to the atmosphere's own bulk
        fluxes, as before).  The hook being installed (or not) is fixed for the
        whole run; within a flux-coupled run ``model_driver`` packs ``None`` on
        segment 0 and arrays thereafter -- a one-time recompile when the override
        first appears, identical to the ``couple_surface_radiation`` lag.
        """
        if not self.coupled_cfg.couple_surface_fluxes:
            return

        # Air-sea closure precondition (codex HIGH): the blended flux fed to the
        # atmosphere closes the air-sea exchange against the ocean q_net ONLY when
        # the atmosphere's land/ocean partition (TileConfig.f_land) and the
        # dynamic ocean's wet mask are the SAME partition.  For a dynamic ocean
        # with CONTINENTS that holds only under f_land_mode='from_ocean' (both
        # masks from one WOA source); the default 'analytical' land (crude lat
        # bands) diverges from the WOA wet mask over coastal cells, so the
        # atmosphere would be force-balanced against a flux the ocean never
        # received there.  Refuse it LOUDLY rather than silently break the budget
        # (CLAUDE.md: no silent degradation).  An aquaplanet dynamic ocean (no
        # land tile) is fine: the partitions trivially agree (all ocean).
        # Divergence can come from EITHER partition: the atmosphere having land
        # (TileConfig.f_land > 0) OR the dynamic OCEAN masking dry cells
        # (continents) while the atmosphere does not (e.g. f_land_mode='zero'
        # with a WOA/tripole ocean -- the atm is all-ocean but the ocean masks
        # land, so the atm gets shared flux over cells the ocean never wetted).
        # Only f_land_mode='from_ocean' ties BOTH masks to one source.
        _tc = getattr(self, "_tile_config", None)
        _has_atm_land = (
            _tc is not None
            and float(jnp.max(jnp.abs(jnp.asarray(_tc.f_land)))) > 0.0
        )
        # Ocean wet mask is 1=ocean, 0=land (set by the dynamic-ocean init); any
        # cell < 1 is a dry (continent) cell.  None / all-wet aquaplanet => no
        # dry cells => the partitions trivially agree.
        _omask = getattr(self, "_ocean_land_mask", None)
        _ocean_has_dry = (
            _omask is not None
            and float(jnp.min(jnp.asarray(_omask))) < 1.0
        )
        if (getattr(self, "_is_dynamic_ocean", False)
                and (_has_atm_land or _ocean_has_dry)
                and self.coupled_cfg.f_land_mode != "from_ocean"):
            raise ValueError(
                "couple_surface_fluxes=True with a dynamic ocean that has land "
                "(atmosphere f_land>0 and/or a masked-dry ocean wet mask) "
                "requires f_land_mode='from_ocean' so the atmosphere land/ocean "
                "partition matches the ocean wet mask (otherwise the shared-flux "
                "air-sea budget does not close over coastal/continental cells); "
                f"got f_land_mode={self.coupled_cfg.f_land_mode!r}. Set "
                "f_land_mode='from_ocean' (dynamic+WOA/tripole), or use an "
                "all-ocean aquaplanet (ocean_ic='rest', no land tile), or "
                "disable couple_surface_fluxes."
            )

        def _coupled_get_sfc_flux_override(day):
            # The atmosphere gets the coupler's TILE-BLENDED surface flux
            # (f_ocean*ocean + f_ice*ice + f_land*land + f_lake*lake) -- the
            # correct TOTAL surface flux for the atmosphere bottom-level tendency
            # over ALL tiles, so a mixed (land/ice) cell is forced by its own
            # tiles, not by the ocean-tile flux.  It is already on the ATMOSPHERE
            # grid (step_surface runs on the atm-grid forcing), so no remap.  The
            # AIR-SEA budget closes because this blended flux and the ocean
            # q_net's ``tile.shflx``/``tile.lhflx`` BOTH derive from the SAME
            # coupler ``ocean_tile_response`` (no longer the atmosphere's old
            # INDEPENDENT bulk calc): over the ocean fraction the blended flux's
            # ocean component is f_ocean*ocean_tile and the ocean column receives
            # ocean_tile over its wet area -- the same partition.  For an
            # all-ocean cell blended == ocean_tile exactly (proven in the
            # conservation tests).
            r = self._last_sfc_response
            if r is None or getattr(r, "shflx", None) is None:
                return None, None
            return r.shflx, r.lhflx

        self._atm.get_sfc_flux_override = _coupled_get_sfc_flux_override
        logger.info(
            "  Shared air-sea fluxes: coupler blended SH/LH -> atmosphere surface"
        )

    # ==================================================================
    # Coupling step
    # ==================================================================

    def _build_atm_forcing(self, day: float):
        """Build AtmToSurface from atmosphere state and physics."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.forcing.surface_utils import (
            blend_surface_temperature,
            snow_for_albedo_deblend,
            surface_emissivity_for_lw_inversion,
            surface_temperature_for_lw_boundary,
        )

        state = self._atm.state
        # Low-level winds/T/q/p_s for the ocean surface stress + bulk turbulent
        # fluxes.  Three atm-state layouts:
        #   * cube/latlon  -> cell-centered u/v/T/p_s Fields, read directly.
        #   * MPAS/voronoi -> edge-normal velocity (state.v is None); Perot-
        #     reconstruct the cell-centered (zonal, meridional) winds.
        #   * SPECTRAL     -> the state is spectral COEFFICIENTS, so synthesize
        #     every field to grid (spectral_pe_to_grid) before extracting the
        #     lowest level.  Moisture is grid-space in the state's tracers dict
        #     (NOT in spectral_pe_to_grid); zeros on a dry spectral run.
        if hasattr(state, "vor_hat"):
            from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                spectral_pe_to_grid,
            )
            _f = spectral_pe_to_grid(state, self._atm.grid, self._atm.sigma)
            p_s = _f["p_s"]
            T_low = _f["T"][..., -1]
            u_low = _f["u"][..., -1]
            v_low = _f["v"][..., -1]
            _tr = getattr(state, "tracers", None)
            _qg = _tr.get("q_v") if _tr else None
            _qgd = (_qg.data if hasattr(_qg, "data") else _qg)  # Field -> array
            q_low = (_qgd[..., -1] if _qgd is not None
                     else jnp.zeros_like(T_low))
        else:
            q_v = self._atm.q_v
            p_s = state.p_s.data
            T_low = state.T.data[..., -1]
            if state.v is not None:
                u_low = state.u.data[..., -1]
                v_low = state.v.data[..., -1]
            else:
                from legoesm.grids.voronoi import reconstruct_cell_velocity
                _u_cell, _v_cell = reconstruct_cell_velocity(
                    state.u.data, self._atm.grid)
                u_low = _u_cell[..., -1]
                v_low = _v_cell[..., -1]
            q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        p_low = self._atm.sigma.pressure_at_full(p_s)[..., -1]
        # Moist-air density: rho = p / (R_d * T_v), T_v = T*(1 + (1/eps - 1)*q).
        # The dry form rho = p/(R_d*T) underestimates density by ~0.6% in the
        # tropics (q_v ~ 17 g/kg) and biases every downstream bulk-flux surface
        # stress / turbulent flux that reads forcing.rho_lowest -- the SAME T_v
        # correction the canonical extract_atm_to_surface uses (single
        # convention across the coupling-field extractors).
        T_v_low = T_low * (1.0 + (1.0 / constants.epsilon - 1.0) * q_low)
        rho_low = p_low / (constants.R_d * T_v_low)

        # Radiation and precipitation from last atmosphere physics.  STRICT:
        # a lane that advanced a segment with an ACTIVE radiation /
        # precipitation source MUST have stashed these.  Silently defaulting a
        # missing key to zeros forces the surface with sw_down=0 / precip=0
        # while has_radiation=1 below asserts the forcing is valid -- the
        # silent wrong-physics fallback dispatch-hardening doctrine forbids.
        # Gated on the atmosphere config, NOT on bare key presence: zero is the
        # CORRECT value for a dry run or radiation="none".  Safe to raise
        # unconditionally here -- _build_atm_forcing has exactly one caller,
        # _segment_hook (the segment callback itself), so it never runs before
        # an atmosphere segment has advanced.
        from legoesm.core.coupling_fields import require_surface_radiation_aux
        aux = getattr(self._atm, '_carry_aux', {})
        _acfg = self.atm_config
        require_surface_radiation_aux(
            aux,
            radiation_active=(
                getattr(_acfg, "radiation", "none") not in (None, "none")),
            precip_active=(
                getattr(_acfg, "microphysics", "none") not in (None, "none")
                or getattr(_acfg, "convection", "none") not in (None, "none")),
            lane=type(self._atm).__name__,
        )
        sw_net_sfc = aux.get("held_sw_net_sfc", jnp.zeros_like(p_s))
        lw_net_sfc = aux.get("held_lw_net_sfc", jnp.zeros_like(p_s))
        seg_precip = aux.get("seg_precip", jnp.zeros_like(p_s))

        # Reconstruct gross downward fluxes from net
        acfg = self.atm_config
        sst, sic = self._atm.get_sst_sic(day)
        # When the dynamic surface-radiation feedback is active, radiation
        # produced the held net fluxes using the coupler's blended albedo AND
        # skin temperature (the value _last_sfc_response held when this
        # segment's forcing was packed — still current here, since
        # _step_surface updates it only after this call).  Invert sw_down AND
        # lw_down with the SAME albedo / T_sfc so the reconstructed gross
        # fluxes stay consistent; otherwise the frozen-scalar de-blend biases
        # the surface forcing.  Falls back to the static blend when the
        # feedback is off or before the first coupler step.
        _resp = self._last_sfc_response
        _dyn_sfc = (
            self.coupled_cfg.couple_surface_radiation
            and _resp is not None
            and getattr(_resp, "albedo", None) is not None
        )
        # Invert the held lw_net_sfc back to gross lw_down with the SAME (eps, T)
        # pair radiation EMITTED the boundary with, or the round trip leaks an
        # O(1 W/m^2) surface-energy bias.  Temperature: RRTMGP/RRTMG use the
        # radiative-equivalent T_rad; gray/none use a black-surface brightness
        # temperature (sigma*T_bb^4 = LW_out) — NOT the aerodynamic/sensible-heat
        # T_sfc (the canopy air-space Tc over vegetated cells).
        _radiation = getattr(self.atm_config, "radiation", "gray")
        _phys = self._atm.physics
        if _dyn_sfc:
            albedo_eff = _resp.albedo
            T_sfc = surface_temperature_for_lw_boundary(
                _radiation, T_rad=getattr(_resp, "T_rad", _resp.T_sfc),
                lw_up=_resp.lw_up)
        else:
            # The pipeline's own ocean/ice/land blend, not an ocean/ice
            # approximation of it.  ``blend_surface_property(sic, ice, ocean)``
            # carries NO land fraction, so over land this inverted
            # sw_net/(1-alpha) with alpha ~ 0.06 instead of ~0.20 and handed
            # the surface ~36 W/m^2 too little shortwave, with no error and a
            # surface energy budget that did not close (#1556).  Same
            # single-source-of-truth argument as the emissivity inversion
            # immediately below.  See static_surface_albedo for the two terms
            # it still cannot see (zenith ocean albedo, multilayer albedo_veg);
            # this is the first-order land term, not an exact round trip.
            # lat + the SNOW carry are passed, not defaulted: with
            # snow_albedo_feedback on (the ERA5-calibrated land config sets it)
            # radiation brightens the land albedo by snow cover, so a deblend
            # that fell back to the bare vegetation albedo would re-open this
            # same gap over every snow-covered column.
            albedo_eff = _phys.static_surface_albedo(
                sic, land_active=_phys.f_land is not None,
                lat=self._atm._grid_lat,
                snow=snow_for_albedo_deblend(
                    aux.get("snow"),
                    getattr(self._atm, "_ensemble_size", 1)))
            T_sfc = blend_surface_temperature(sst, sic, acfg.T_ice)
        sw_down = sw_net_sfc / jnp.maximum(1.0 - albedo_eff, 0.01)
        # Emissivity matching the emission:
        #   * RRTMGP/RRTMG + dynamic feedback -> the coupler's tile-blended eps_col
        #     (incl. the canopy's LAI-dependent eps_eff);
        #   * RRTMGP/RRTMG static / pre-first-response -> the EXACT ocean/ice/land
        #     emissivity blend the radiation pipeline emitted with (configured
        #     emissivity_* values, not a constant ocean/ice approximation);
        #   * gray/none -> an idealized black surface (eps = 1.0).
        eps_sfc = surface_emissivity_for_lw_inversion(
            _radiation,
            dynamic_emissivity=(
                getattr(_resp, "emissivity", None) if _dyn_sfc else None),
            static_sfc_emissivity=_phys.static_surface_emissivity(
                sic, land_active=_phys.f_land is not None),
        )
        lw_up_sfc = eps_sfc * constants.sigma_sb * T_sfc ** 4
        lw_down = (lw_net_sfc + lw_up_sfc) / jnp.maximum(eps_sfc, 0.01)

        precip_total = jnp.maximum(seg_precip, 0.0)
        # Smooth snow fraction (Wigmosta 1994 / Dai 2008) via shared helper —
        # single source of truth with the earth-system driver. Replaces the
        # prior hard step, which killed d(snow)/d(T_low) and miscounted
        # mixed-phase precip in the 0–4 °C band.
        from legoesm.forcing.surface_utils import snow_fraction
        snow_frac = snow_fraction(T_low, constants.T_freeze)
        precip_snow = precip_total * snow_frac

        # Cosine zenith — route through the atmosphere's seasonal insolation seam (iter
        # 449/461) so the coupler's ocean/surface insolation runs the SAME season as the
        # atmosphere (config.insolation_start_doy); offset 0 (default) == day_to_calendar(day),
        # byte-identical. Without this the coupled ocean surface saw JANUARY insolation while
        # the atmosphere saw the aligned season — a physically inconsistent sun.
        doy, _ = self._atm._calendar_for_radiation(day)
        lat = self._atm._grid_lat
        if lat is not None:
            from legoesm.atmosphere.physics.radiation.solar import (
                daily_mean_insolation, earth_orbit, earth_sun_distance_factor,
            )
            # Solar constant from legoesm.constants per CLAUDE.md.
            # ``acfg.S_0`` allows override for sensitivity studies.
            S_0 = acfg.S_0
            # Realistic orbit (Berger 1978) when enabled; None ⇒ circular.
            _orbit = (earth_orbit()
                      if getattr(acfg, "orbital_insolation", False) else None)
            _eccf = (earth_sun_distance_factor(float(doy), _orbit)
                     if _orbit is not None else 1.0)
            Q_daily = daily_mean_insolation(lat, float(doy), S_0=S_0,
                                            orbit=_orbit)
            # cos_zen is a geometric optical-path cosine: use the orbital
            # declination but divide out the (a/r)^2 flux factor so it stays
            # <= 1.  _eccf == 1.0 on the circular orbit ⇒ bit-identical.
            cos_zen = jnp.clip(Q_daily / (_eccf * S_0), 0.0, 1.0)
        else:
            cos_zen = jnp.full_like(p_s, 0.5)

        # CO2: prognostic or constant
        if self.coupled_cfg.co2_tracer and hasattr(self, '_co2_field'):
            co2_lowest = self._co2_field[..., -1]
            co2_ppmv = (
                co2_lowest / (constants.M_CO2 / constants.M_air) * 1.0e6
            )
        else:
            co2_ppmv = jnp.full_like(p_s, self.atm_config.co2_ppmv)

        from legoesm.core.coupling_fields import lowest_level_height
        return AtmToSurface(
            z_lowest=lowest_level_height(
                T_low, self._atm.sigma.pressure_at_half(p_s),
                self._atm.sigma.pressure_at_full(p_s)),
            sw_down=sw_down,
            lw_down=lw_down,
            precip_total=precip_total,
            precip_snow=precip_snow,
            T_lowest=T_low,
            q_lowest=q_low,
            u_lowest=u_low,
            v_lowest=v_low,
            p_lowest=p_low,
            p_surface=p_s,
            rho_lowest=rho_low,
            cos_zenith=cos_zen,
            co2_ppmv=co2_ppmv,
            has_radiation=jnp.ones_like(p_s),
            has_precipitation=jnp.where(precip_total > 0, 1.0, 0.0),
        )

    def _step_ocean(self, atm_forcing, dt, q_flux=None):
        """Advance the slab ocean one coupling step.

        ``q_flux`` (optional, [W/m2], +INTO the mixed layer, on the ocean grid)
        is the calendar-month-interpolated q-flux climatology map for this
        step; ``None`` falls back to the scalar ``config.Q_flux`` in the slab
        step (byte-identical when no climatology is loaded).

        The slab's own open-ocean atmospheric fluxes are applied over the
        ice-free fraction only (``_slab_open_water_frac``, lagged one coupling
        step like the 3D path); under ice the slab receives only ``q_flux``.

        **One-way ice -> ocean coupling (intentional for the slab ocean).**
        ``step_sea_ice`` populates ice -> ocean back-reaction channels on
        the surface response (``freshwater_flux``, ``ocean_heat_extraction``,
        ``salt_flux``, ``ocean_stress_x``/``ocean_stress_y``), but this
        driver advances the :class:`SimpleOcean` slab, a thermodynamic
        mixed-layer model with no prognostic salinity and no prognostic
        momentum.  It therefore cannot consume those feedbacks:

        * ``salt_flux`` / ``freshwater_flux`` -> no salinity prognostic;
        * ``ocean_stress_x``/``ocean_stress_y`` -> no momentum prognostic
          (the slab returns ``u_sfc = v_sfc = 0``);
        * ``ocean_heat_extraction`` -> the slab already diagnoses its own
          ``Q_freeze`` (the heat removed by its freezing clamp); adding the
          ice's basal heat extraction on top would double-count against it.

        The channels remain available on ``self._last_sfc_response`` for a
        full prognostic ocean (salinity + momentum + a two-way
        ``Q_freeze`` <-> ice-seeding contract), which is tracked as separate
        feature work; they are deliberately NOT applied to the slab here.
        """
        if self._ocean_step is None:
            return
        if not getattr(self, "_is_dynamic_ocean", False):
            self._ocean_state, sst_new, u_sfc, v_sfc = self._ocean_step(
                self._ocean_state, atm_forcing, dt, q_flux=q_flux,
                open_water_frac=self._slab_open_water_frac(),
            )
            self._ocean_u_sfc = u_sfc
            self._ocean_v_sfc = v_sfc
            return
        # --- Prognostic 3D ocean (LatLonCGridOceanModel) ---
        # Hold the atm flux over the coupling step (standard explicit coupling)
        # and SUBSTEP the ocean at ocean_dt_s (the 3D ocean CFL forbids stepping
        # at coupling_dt; OMIP runs ~300 s at 1°).  SST/currents are read from
        # the 3D state by ``_ocean_surface_KuvC`` (no stored slab tuple).
        sf, fw = self._assemble_ocean_forcing(atm_forcing)
        # CEIL (not round) so the actual substep odt <= ocean_dt_s — never
        # exceed the ocean CFL for a non-divisor coupling_dt (codex MED).
        import math
        n_o = max(1, math.ceil(dt / self.coupled_cfg.ocean_dt_s))
        odt = dt / n_o
        for _ in range(n_o):
            self._ocean_state = self._ocean_step(
                self._ocean_state, odt, freshwater=fw, surface_forcing=sf,
            )
        # Optional WOA surface T/S restoring (coupled spin-up anchor) — applied
        # once per coupling step over the full dt; no-op when the restoring
        # timescales are 0 or no WOA target was loaded (byte-identical).
        self._apply_ocean_restoring(dt)

    def _slab_open_water_frac(self):
        """Ice-free fraction of the slab's water area, on the ocean grid.

        Sea-ice concentration is relative to the water area (tile fractions:
        ``f_ice = f_water * sic``), so the per-unit-water-area slab column
        receives the open-ocean atmospheric fluxes over ``1 - sic`` of its
        area; the ice-covered part is forced by the ice tile.  ``1.0`` when no
        ice state exists.
        """
        from legoesm.coupler.grid_remap import remap_field
        _sfc = getattr(self, "_sfc_state", None)
        if _sfc is None or getattr(_sfc, "ice", None) is None:
            return 1.0
        sic = _total_ice_sic(_sfc.ice)
        _rem = getattr(self, "_grid_remapper", None)
        if _rem is not None and getattr(_rem, "a2o", None) is not None:
            sic = remap_field(sic, _rem.a2o)
        return 1.0 - jnp.clip(sic, 0.0, 1.0)

    def _apply_ocean_restoring(self, dt):
        """Relax the 3D-ocean surface T/S toward the WOA-climatology IC.

        The standard coupled spin-up anchor: a free 3D ocean started from
        realistic WOA T/S cold-collapses when the dry cold-start atmosphere
        radiates away the warm ocean's heat faster than it can re-equilibrate
        (measured -92 K/yr at 15 d even with the air-sea gustiness fix).
        Newtonian relaxation of the surface layer toward the WOA initial state
        keeps the surface near observed climatology while the atmosphere spins
        up.  No-op unless ocean_mode=='dynamic' + ocean_ic=='woa' AND a positive
        restoring timescale is configured (CoupledConfig.ocean_restore_*).
        """
        cfg = self.coupled_cfg
        tau_T_days = getattr(cfg, "ocean_restore_sst_tau_days", 0.0)
        tau_S_days = getattr(cfg, "ocean_restore_sss_tau_days", 0.0)
        if tau_T_days <= 0.0 and tau_S_days <= 0.0:
            return
        if self._ocean_T_target is None or self._ocean_S_target is None:
            return
        from legoesm.ocean.forcing.surface_relaxation import (
            apply_surface_relaxation_step,
        )
        self._ocean_state = apply_surface_relaxation_step(
            self._ocean_state,
            T_target=self._ocean_T_target,
            S_target=self._ocean_S_target,
            dt=dt,
            tau_T_s=tau_T_days * _SECONDS_PER_DAY,
            tau_S_s=tau_S_days * _SECONDS_PER_DAY,
        )

    def _ocean_surface_KuvC(self):
        """Ocean surface (SST [K], u_sfc, v_sfc [m/s]) at cell centres on the
        ocean grid.  Slab: ``T_sfc`` [K] + zero currents.  Dynamic 3D: top-level
        ``T`` [°C → K] + C-grid face-averaged top-level currents.  Reconciles
        the slab(K) vs 3D-ocean(°C) SST-convention divergence so the atmosphere
        always sees SST in Kelvin."""
        if not getattr(self, "_is_dynamic_ocean", False):
            sst = self._ocean_state.T_sfc.data
            z = jnp.zeros_like(sst)
            return sst, z, z
        if getattr(self, "_ocean_is_mpas", False):
            # 3-D MPAS ocean: SST = top-level cell T [°C -> K]; surface currents
            # = Perot area-weighted edge->cell reconstruction of the top-level
            # edge-normal velocity, returned DIRECTLY in the geographic
            # (east, north) basis (reconstruct_cell_velocity) — the basis the
            # o2a coupler expects, so NO rotation is needed (unlike the tripole
            # cap below).  All returns are shape (nCells,).
            from legoesm import constants
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            T_mpas = self._ocean_state.T.data            # (nCells, nlev) [°C]
            sst = T_mpas[..., 0] + constants.T_freeze    # top level -> K
            u_east, v_north = reconstruct_cell_velocity(
                self._ocean_state.u.data[..., 0], self._ocean_grid)
            return sst, u_east, v_north
        from legoesm import constants
        T = self._ocean_state.T.data                 # (n_lat, n_lon, nlev) [°C]
        sst = T[..., 0] + constants.T_freeze         # top level → K
        u = self._ocean_state.u.data                 # (n_lat, n_lon+1, nlev)
        v = self._ocean_state.v.data                 # (n_lat+1, n_lon, nlev)
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])      # lon-faces → centres (grid-i)
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])      # lat-faces → centres (grid-j)
        # Tripole bipolar cap: the prognostic currents are GRID-ALIGNED (i, j);
        # the atmosphere expects geographic east/north.  Rotate on the ocean
        # grid BEFORE the o2a remap (the remap is per-component scalar only once
        # both components share the geographic basis).  ``fold.is_active`` is a
        # static Python bool (geometry build time) → a Python ``if`` is correct
        # here (the JAX feature-gating exception; not data-dependent).  Below
        # the cap (and for a regular lat-lon ocean with no rotation angles) this
        # is the identity → byte-identical to the pre-tripole path.
        grid = getattr(self, "_ocean_grid", None)
        fold = getattr(grid, "fold", None)
        cos_a_u = getattr(grid, "cos_alpha_u", None)
        sin_a_u = getattr(grid, "sin_alpha_u", None)
        if (fold is not None and getattr(fold, "is_active", False)
                and cos_a_u is not None and sin_a_u is not None):
            from legoesm.coupler.grid_remap import (
                rotate_tpoint_currents_to_geographic,
            )
            u_c, v_c = rotate_tpoint_currents_to_geographic(
                u_c, v_c, cos_a_u, sin_a_u)
        return sst, u_c, v_c

    def _assemble_ocean_forcing(self, atm_forcing):
        """Build ``(OceanSurfaceForcing, FreshwaterForcing)`` for the 3D ocean
        from the atm forcing + the ocean-tile bulk fluxes — the audited OMIP
        assembly (``run_omip.py`` ocean-tile path): ``q_net = sw_net + lw_down -
        lw_up - SH - LH`` (positive into ocean, INCLUDING shortwave); ``sw_down``
        is the downwelling SW for sub-surface penetration; ``tau`` is in the
        ATMOSPHERIC convention (the ocean core applies ``-tau`` + grid rotation);
        freshwater is passed via the ``freshwater=`` arg (NOT ``sf.freshwater``,
        which is the cube-only channel).  Over an ice-covered cell the open-water
        atmospheric fluxes (q_net, sw, tau, evap, precip) are scaled by the
        open-water fraction ``f_ocean = f_water*(1 - ice_concentration)`` and the
        ice→ocean back-reaction (basal heat, brine salt, ice-ocean stress) that
        the ice model already computed on ``self._last_sfc_response`` is added —
        so the ocean is NOT forced as ice-free and the melt/freeze freshwater
        arrives WITH its heat and salt.  The KPP ice-buoyancy ``sf.freshwater``
        channel stays cube-only (left None; the lat-lon P-E enters via the
        ``freshwater=`` FreshwaterForcing arg)."""
        from legoesm.coupler.config import CouplerConfig
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.grid_remap import remap_field
        from legoesm.coupler.tile_fractions import compute_tile_fractions
        from legoesm.ocean.freshwater import FreshwaterForcing
        from legoesm.ocean.state import OceanSurfaceForcing

        sst_K, u_o, v_o = self._ocean_surface_KuvC()
        ccfg = getattr(self, "_coupler_cfg", None) or CouplerConfig()
        tile = ocean_tile_response(atm_forcing, sst_K, u_o, v_o, ccfg)
        # Net surface shortwave into the OCEAN tile (positive INTO ocean).  With
        # the ``f_ocean`` open-water scaling below, the correct per-tile share is
        # ``f_ocean * sw_down * (1 - alpha_ocean)`` (ocean-tile albedo): the
        # ocean and ice tile shares sum to ``sw_down * (1 - alpha_blended)`` =
        # the radiation net SW, so the SW budget closes by construction.  Do NOT
        # use the radiation aux ``held_sw_net_sfc`` here — that is already the
        # TILE-BLENDED net SW, and multiplying it by ``f_ocean`` would
        # double-apply the partition and under-heat the open ocean over
        # ice-covered cells (codex).  The pre-``f_ocean`` code used the blended
        # net as a full-cell workaround for the (then) missing ice fraction; the
        # scaling makes the ocean-tile albedo the right choice.  ``sw_down`` is
        # the reconstructed gross incident (``_build_atm_forcing``), so over an
        # all-ocean cell (f_ocean==1) this is byte-identical to the legacy net.
        _remapper = getattr(self, "_grid_remapper", None)
        # Open-water net shortwave uses the OCEAN-TILE albedo, ``sw_down*(1 -
        # alpha_ocean)``, and is scaled by ``f_ocean`` below.  Do NOT use the
        # radiation aux ``held_sw_net_sfc`` here: that is the TILE-BLENDED net SW
        # (f_ocean*ocean + f_ice*ice partition already applied), so multiplying
        # it by ``f_ocean`` again would double-count the partition and under-heat
        # the open ocean over ice-covered cells (codex).  ``tile`` is the ocean
        # tile, so ``tile.albedo`` is the open-water albedo; f_ocean==1 (no ice)
        # recovers the legacy full-cell value byte-identically.
        sw_net = atm_forcing.sw_down * (1.0 - tile.albedo)
        # --- Open-water fraction: the open-ocean bulk fluxes act ONLY on the
        #     ice-free part of the cell.  The ice-covered fraction is forced by
        #     the ICE tile (its own surface energy balance); its back-reaction on
        #     the ocean (basal heat, brine salt, ice-ocean stress) is ADDED below.
        #     Without this an ice-covered cell was heated / evaporated / wind-
        #     stressed as if ICE-FREE — a full-cell energy+moisture leak. ---
        # f_ocean = f_water*(1 - ice_concentration) from the SHARED tile-fraction
        # helper (``compute_tile_fractions``) on the ATM grid — the SAME partition
        # the tile blender used to weight the ice back-reaction channels below —
        # remapped onto the ocean grid.  No ice tile (aquaplanet Phase-1) ⇒
        # concentration == 0 and (over a wet ocean cell) f_water == 1 ⇒
        # f_ocean == 1 ⇒ byte-identical to the legacy full-cell assembly.
        _sfc = getattr(self, "_sfc_state", None)
        _tile_cfg = getattr(self, "_tile_config", None)
        if (_sfc is not None and getattr(_sfc, "ice", None) is not None
                and _tile_cfg is not None):
            _sic = jnp.clip(_total_ice_sic(_sfc.ice), 0.0, 1.0)
            _cross = (_remapper is not None
                      and getattr(_remapper, "a2o", None) is not None)
            if _cross:
                # CROSS-GRID (H3): the STATIC open-water (land/sea) fraction is the
                # OCEAN's OWN wet mask -- NOT the remapped atm f_water, which would
                # impose the ATM coastline on the ocean grid and leak / starve
                # coastal ocean cells.  Only the DYNAMIC sea-ice concentration is
                # remapped.  Multiplying every a2o flux by f_ocean also GATES the
                # conservative partition-of-unity flux to the wet ocean domain
                # (land cells -> 0), so the energy/freshwater integral is conserved
                # there (sum_d out[d]*area_T[d] ~ sum_s F[s]*A_wet[s]).
                from legoesm.coupler.grid_remap import (
                    cross_grid_open_water_fraction,
                )
                _owet = getattr(self, "_ocean_land_mask", None)
                if _owet is None:
                    raise ValueError(
                        "cross-grid coupling requires the ocean wet mask "
                        "(self._ocean_land_mask) for the open-water fraction.")
                # M13 (DEFERRED -- corrected proof): the DYNAMIC sea-ice cross
                # term drops sub-cell covariance -- ocean_wet*remap(sic)*remap(F)
                # vs the ideal ocean_wet*remap(sic*F).  a2o is a partition-of-
                # unity conservative remap out[d]=Sum_s w[d,s] F[s], Sum_s w=1
                # (segment_sum), so Cov_d = remap(sic*F)-remap(sic)*remap(F) is
                # NONZERO ONLY when >=2 ATM SOURCE cells map into ONE OCEAN
                # destination cell, i.e. only when the OCEAN is COARSER than the
                # ATM.  In every reachable config the lat-lon atm (n_lat 16-90,
                # ~2-11 deg) is COARSER than the eORCA1 tripole ocean (332x362,
                # ~1 deg), so the ocean is FINER: each ocean cell nests in one
                # atm cell, a2o is a refinement copy, and Cov_d=0 EXACTLY on the
                # nested interior -- only a SECOND-ORDER residual at the thin band
                # of ocean cells straddling an atm-cell boundary (nonzero only
                # where sic AND a flux BOTH jump across that boundary; a few %
                # of ocean area, near-zero global mean).  EXACTLY zero for the
                # identity/aquaplanet path (sic=0, byte-identical).
                #   Partition-before-remap is CONSTRUCTIBLE + AD-safe for the
                # ATM-ORIGIN class-A fluxes ONLY (sw_down/lw_down/precip: form
                # (1-sic_atm)*F_atm on the atm grid, remap the product, then
                # *ocean_wet) -- but it recovers only the ~0 second-order
                # straddling residual here, so it is NOT worth the restructuring
                # (raw atm forcing is consumed only after the L2237 a2o remap)
                # nor the risk to the byte-exact shared-grid identity branch.
                # The class-B turbulent tile fluxes (lw_up/shflx/lhflx/tau) are
                # computed ON the ocean grid from ocean SST/currents, so
                # sic_ocean=remap(sic) IS already their correct LOCAL fraction --
                # they have NO atm-grid covariance to lose (the prior 'sic*F_bulk
                # cannot be formed on the atm grid' reason is TRUE only for these,
                # not for class A).  Revisit ONLY if the ocean is deliberately
                # run COARSER than the atm (fine cube/gaussian atm + a coarse
                # --ocean-grid latlon:<res>) -- the one regime where the dropped
                # covariance becomes first-order.
                sic_ocean = remap_field(_sic, _remapper.a2o)
                f_ocean = cross_grid_open_water_fraction(_owet, sic_ocean)
                # f_water = the ocean's OWN wet fraction (open water + ice);
                # land -> 0.  Needed below to deliver the WATER-fraction precip
                # (open water + ice) for the no-snow-reservoir ice path without
                # injecting precip on dry (land) ocean-grid cells.
                f_water = jnp.asarray(_owet, dtype=f_ocean.dtype)
            else:
                # Shared-grid identity: the atm f_water IS the ocean wet fraction
                # (same grid) -- keep the legacy assembly byte-identical.
                _fracs = compute_tile_fractions(_tile_cfg, _sic)
                f_ocean = _fracs.f_ocean
                # f_water = open water + ice fraction = 1 - f_land - f_lake
                # (EXCLUDES land/lake).  Used below to route the ice-fraction
                # precip to the ocean on the no-snow-reservoir path WITHOUT
                # re-adding the land/lake precip (already handled by the river-
                # runoff / lake channels -- adding it here would double-count).
                f_water = _fracs.f_ocean + _fracs.f_ice
        else:
            f_ocean = 1.0
            f_water = 1.0
        # Open-water fluxes scaled to the ice-free fraction.  Sign conventions
        # (ocean-consumer frame): q_net +into ocean; sw_pen +into ocean
        # (penetrating solar, post-albedo); tau_x/y in the ATMOSPHERIC convention
        # (the ocean core applies -tau); evap +up (removed from ocean; enters
        # ocean P-E with the -evap sign in FreshwaterForcing); precip +into ocean.
        q_net = f_ocean * (sw_net + atm_forcing.lw_down
                           - tile.lw_up - tile.shflx - tile.lhflx)
        sw_pen = f_ocean * sw_net                    # +into ocean (penetrating SW)
        tau_x = f_ocean * tile.tau_x                 # atmospheric convention (-tau)
        tau_y = f_ocean * tile.tau_y
        # Tile mass flux = lhflx over the latent heat the flux used.
        evap = f_ocean * tile.surface_mass_flux  # [kg/m²/s], +up (open water)
        # Precip over the ice fraction: WHERE it is counted depends on whether
        # the active ice model owns a snow reservoir.  Sign: +into ocean.  The
        # LAND and LAKE fractions are ALWAYS excluded here -- their precip is the
        # land/lake tile's water, returned to the ocean via the river-runoff and
        # ice_lake channels -- so the ocean direct-precip is AT MOST the WATER
        # fraction f_water = f_ocean + f_ice, NEVER the full cell (which would
        # double-count the land/lake precip against runoff, and on the cross-grid
        # path inject precip into dry land cells).
        #   * v2 / new-physics ice (``uses_new_physics``) accumulates snow in
        #     ``h_snow`` and runs the ice-fraction rain off to the ocean via
        #     ``freshwater_flux`` (returned here as ``ice_fw``); the ice tile
        #     ALREADY carries the ice-fraction (f_ice) precip, so the DIRECT
        #     channel takes only the OPEN-water share ``f_ocean`` (f_ocean==1 =>
        #     full precip) to avoid double-counting the ice-routed water.
        #   * slab / legacy ice has NO snow reservoir and its ``freshwater_flux``
        #     carries NO precip (melt/freeze only), so the ice-fraction precip
        #     ``f_ice*P`` has nowhere to be stored -- without delivery it is
        #     DROPPED (atmosphere loses it, no reservoir gains it: the H2 leak).
        #     Deliver the WATER-fraction precip ``f_water*P`` (open water + ice)
        #     to the ocean top cell, matching OMIP ``blend_ice_ocean_forcing``'s
        #     no-snow-reservoir policy on an ocean-only (f_water==1) cell, so
        #     precip is counted exactly ONCE and the land/lake fractions are
        #     never double-counted.
        from legoesm.ice import uses_new_physics
        from legoesm.ice.config import SeaIceConfig
        _ice_cfg = getattr(self, "_ice_config", None) or SeaIceConfig()
        if uses_new_physics(_ice_cfg):
            precip = f_ocean * atm_forcing.precip_total  # open-water share only
        else:
            precip = f_water * atm_forcing.precip_total  # water frac (no reservoir)
        z = jnp.zeros_like(sw_net)
        # Freshwater into the ocean, SPLIT by vertical-injection channel so each
        # term lands where it physically belongs:
        #   * precip/evap  -> ocean P−E (top cell), CURRENT (depends on current SST)
        #   * runoff       -> LAND river runoff (depth-spread over the ocean's
        #                     ``runoff_depth_spread_m``, NEMO ``rn_dep_max``)
        #   * ice_fw       -> ice melt + lake P−E (top cell)
        # ONLY the two NON-ocean channels are lagged one coupling sub-step (the
        # surface/coupler step runs AFTER the ocean step — explicit coupling).
        # Ocean P−E stays CURRENT, so a time-varying precip/evaporation forcing
        # is delivered without one-step staleness and the aquaplanet path is
        # byte-identical to the legacy ``runoff=ice_fw=0`` code (no land/ice/lake
        # tile ⇒ both lagged channels are exactly zero).  The land/ice/lake
        # exchange is read straight from the lagged blended response's dedicated
        # sub-channels — NOT reconstructed by subtracting the current ocean P−E
        # (which would leak a stale-P−E / coastal area-weight residual into
        # ice_fw).  Direct attribute access (not getattr-with-default) so a
        # malformed surface response fails loudly instead of silently routing
        # river runoff into the wrong channel.  Interior-land runoff at fully-dry
        # cells is gated out by the ocean wet mask (no river-routing map here — a
        # separate Dai-Trenberth concern); fractional coastal cells receive their
        # local f_land·runoff.
        prev = self._last_sfc_response
        if prev is None:
            river = z
            surface_extra = z
        else:
            # The lagged blended surface response lives on the ATMOSPHERE grid;
            # its river-runoff / ice-lake freshwater sub-channels must be REMAPPED
            # onto the OCEAN grid before joining the ocean-grid precip/evap in
            # FreshwaterForcing -- otherwise on a tripole (cross-grid) coupling
            # these atm-grid arrays are placed beside ocean-grid fields (shape
            # mismatch -> crash, or silent wrong-grid injection).  The
            # conservative scalar remap is sign-preserving, so ``river`` stays
            # +INTO ocean and ``ice_fw`` keeps its melt(+)/freeze(-) sign at both
            # ends.  Identity (shared-grid Phase-1 ocean / no remapper) is an
            # exact pass-through -> byte-identical to the single-grid path.
            river = prev.river_runoff_flux              # land, depth-spread
            surface_extra = prev.ice_lake_freshwater_flux  # ice melt + lake, surface
            if _remapper is not None:
                river = remap_field(river, _remapper.a2o)
                surface_extra = remap_field(surface_extra, _remapper.a2o)
        fw = FreshwaterForcing(
            precip=precip, evap=evap, runoff=river,
            ice_fw=surface_extra,
        )
        # F2 closed water-INVENTORY residual (tripwire C, H2 class): stash the
        # freshwater the OCEAN MODEL applies POST wet-mask as ONE rank-local
        # integral [kg/s], +into ocean = water LEAVING the tracked atm+land+ice
        # inventory.  Sign convention (ocean-consumer frame, +into ocean):
        #   F_ocean_applied = INT(precip) - INT(evap) + INT(runoff_applied)
        #                     + INT(ice_fw)
        # precip/evap already carry the f_ocean open-water gate (0 on dry
        # cells); runoff_applied reconstructs the ocean wet-mask gating
        # (river*clip(owet)) exactly as the M2 runoff tripwire below (river is
        # the a2o-remapped runoff, identity on the shared grid where owet is
        # None).  precip/runoff/ice_fw are +into ocean; evap is +up (removed
        # from ocean) -> enters with -evap.  Dynamic-ocean only (this assembly
        # never runs on the storage-free slab -- _step_ocean returns at the slab
        # branch before calling this).  RANK-LOCAL sums (diagnostics-only; a
        # sharded run must route the SUM through global_sum_mpi).
        _oa_fw = getattr(self, "_ocean_area_w", None)
        if _oa_fw is not None:
            from legoesm.diagnostics.water_budget import area_integral
            _owet_fw = getattr(self, "_ocean_land_mask", None)
            _runoff_applied = (river if _owet_fw is None
                               else river * jnp.clip(_owet_fw, 0.0, 1.0))
            self._last_f_ocean_applied_integral_ranklocal = (
                area_integral(precip, _oa_fw)
                - area_integral(evap, _oa_fw)
                + area_integral(_runoff_applied, _oa_fw)
                + area_integral(surface_extra, _oa_fw)
            )
        # F2 (M2) runoff-conservation tripwire: the conservative a2o remap
        # preserves the river-runoff INTEGRAL; the ocean wet mask then drops
        # interior-land runoff at fully-dry ocean cells (a silent freshwater
        # leak).  We RECONSTRUCT the horizontal wet-mask gating here as
        # ``river * clip(_owet,0,1)`` -- ``self._ocean_land_mask`` is the ocean's
        # OWN wet mask (1 = ocean; see cross_grid_open_water_fraction) so this
        # matches the gating the ocean applies for a binary interior-land
        # discard.  Stash the exported (atm-grid) vs reconstructed-applied
        # (ocean-grid) area integrals so the segment diagnostic can difference
        # them.  This is a TRIPWIRE (a small cross-grid conservative-remap
        # residual + fractional-coast wet fraction sit at its floor; the
        # balanced all-wet case is exactly 0, test-pinned), NOT a machine-zero
        # identity.  RANK-LOCAL sums (diagnostics-only; a sharded run must use
        # global_sum_mpi).  Dynamic-ocean only -> no leak on the storage-free
        # slab.  No-op for the shared-grid identity aquaplanet (no wet mask).
        _owet = getattr(self, "_ocean_land_mask", None)
        _oa = getattr(self, "_ocean_area_w", None)
        # Guard the whole chain: a minimal caller (unit-test stub, or a driver
        # built without the atmosphere wired) may lack ``_atm``; ``self._atm.grid``
        # would then AttributeError before the ``_aa is not None`` guard below.
        _aa = getattr(getattr(getattr(self, "_atm", None), "grid", None),
                      "grid_area", None)
        if (prev is not None and _owet is not None
                and _oa is not None and _aa is not None):
            from legoesm.diagnostics.water_budget import area_integral
            _applied = river * jnp.clip(_owet, 0.0, 1.0)  # kept on wet cells
            self._last_runoff_export_integral_ranklocal = (
                area_integral(prev.river_runoff_flux, _aa))
            self._last_runoff_applied_integral_ranklocal = (
                area_integral(_applied, _oa))
        # --- Ice → ocean back-reaction: the melt/freeze water in ``ice_fw`` must
        #     arrive WITH its melt/freeze HEAT + brine SALT + ice-ocean STRESS,
        #     else the ocean gets freshwater without its energy/salt (a mass↔heat
        #     inconsistency).  These blended channels are already tile-weighted by
        #     ``blend_tiles`` on the ATM grid (ocean_heat_extraction / salt_flux =
        #     f_water·ice, ocean_stress = f_ice·ice), matching
        #     ``ocean_forcing.ice_ocean_forcing_from_ice_response``; remap to the
        #     ocean grid and apply with the SAME signs.  prev is None on segment 0
        #     (fall through: no back-reaction yet); a run with no ice tile carries
        #     these channels as zeros ⇒ byte-identical to the legacy assembly. ---
        salt_flux = None
        if prev is not None:
            ohe = prev.ocean_heat_extraction        # [W/m², +ocean LOSES heat]
            salt = prev.salt_flux                   # [kg/m²/s, +INTO ocean]
            ice_tau_x = prev.ocean_stress_x         # [Pa, +eastward force ON ocean]
            ice_tau_y = prev.ocean_stress_y         # [Pa, +northward force ON ocean]
            if _remapper is not None:
                ohe = remap_field(ohe, _remapper.a2o)
                salt = remap_field(salt, _remapper.a2o)
                ice_tau_x = remap_field(ice_tau_x, _remapper.a2o)
                ice_tau_y = remap_field(ice_tau_y, _remapper.a2o)
            # ocean_heat_extraction is +ocean-LOSES; q_net is +into ocean ⇒ subtract
            # (the ocean loses this basal heat to melting ice = pairs with ice_fw).
            q_net = q_net - ohe
            # ocean_stress_* is the force ON the ocean, but the ocean consumer
            # applies external tau as (ocean force = -tau); feed -stress so the NET
            # applied force equals the on-ocean ice stress (matches the -(f_ice·
            # stress) sign in ice_ocean_forcing_from_ice_response).
            tau_x = tau_x - ice_tau_x
            tau_y = tau_y - ice_tau_y
            salt_flux = salt                        # real brine salt (+INTO ocean)
        # ``OceanSurfaceForcing.sw_down`` is the NET (post-albedo) surface SW the
        # ocean PENETRATES (``shortwave_penetration``: "net shortwave INTO the
        # ocean (post-albedo)"; the lat-lon core forms q_nonsolar = q_net -
        # 0.94*sw_down).  Passing the GROSS downwelling here treated reflected SW
        # as penetrating solar and compensated it with an artificially reduced
        # non-solar surface flux -- the column total still telescoped to q_net but
        # the vertical heating profile was wrong.  Use the SAME (f_ocean-scaled)
        # ``sw_pen`` that built q_net so the solar/non-solar split is consistent
        # (matches the canonical OMIP-2 applicator, which passes sw_net as sw_down).
        sf = OceanSurfaceForcing(
            sw_down=sw_pen, q_net=q_net,
            tau_x=tau_x, tau_y=tau_y, salt_flux=salt_flux, freshwater=None,
        )
        return sf, fw

    def _step_co2_tracer(self, dt):
        """Apply blended surface CO2 flux to lowest atmospheric level."""
        if not self.coupled_cfg.co2_tracer or not hasattr(self, '_co2_field'):
            return
        if self._last_sfc_response is None:
            return

        co2_flux = self._last_sfc_response.co2_flux  # kgCO2/m2/s, +up
        p_s = self._atm.state.p_s.data
        # Layer mass of lowest level: dp / g [kg/m2]
        dp = self._atm.sigma.layer_thickness_dp(p_s)[..., -1]
        mass_air = dp / constants.g
        dco2 = co2_flux / jnp.maximum(mass_air, 1.0) * dt
        self._co2_field = self._co2_field.at[..., -1].add(dco2)

    def _segment_hook(self, driver, day, dt_segment):
        """Callback at each segment boundary: step ocean + coupler.

        The segment may span many atmosphere time steps (e.g. an entire
        diagnostic interval).  Surface coupling — ocean, land, ice, and
        carbon — must be sub-cycled at ``coupling_dt`` (default 3600 s)
        so that the carbon cycle's forward-Euler integration remains
        stable and fluxes are physically consistent.
        """
        coupling_dt = self.coupled_cfg.coupling_dt  # default 3600 s
        n_sub = max(1, int(round(dt_segment / coupling_dt)))
        sub_dt = dt_segment / n_sub

        from legoesm.coupler.grid_remap import remap_field, remap_surface_fields

        atm_forcing = self._build_atm_forcing(day)
        # F2 (C1) tripwire: stash the precip RATE the coupler is about to
        # deliver so the segment-boundary moisture-budget residual can compare
        # it against the atmosphere's OWN column-water-vapour drain (an
        # independent witness that flags a mis-scaled coupler-delivered precip).
        self._last_atm_precip = atm_forcing.precip_total
        # Surface forcing for the ocean step lives on the OCEAN grid; remap the
        # atm-grid forcing fields onto it (identity remapper => unchanged, so the
        # standard single-grid run is byte-identical).
        ocean_forcing = remap_surface_fields(atm_forcing, self._grid_remapper.a2o)

        # Spatially+seasonally varying q-flux for this coupling segment: the
        # monthly climatology (already on the ocean grid) interpolated to the
        # current model day.  ``None`` when no climatology is loaded => the slab
        # uses the scalar config.Q_flux (byte-identical).  Interpolated once per
        # segment (it varies on a monthly scale, far slower than sub_dt).
        q_flux_now = None
        if getattr(self, "_qflux_forcing", None) is not None:
            from legoesm.ocean.forcing.qflux import qflux_at_time
            q_flux_now = qflux_at_time(self._qflux_forcing, day)

        for _ in range(n_sub):
            # Step slab ocean (on the ocean grid)
            self._step_ocean(ocean_forcing, sub_dt, q_flux=q_flux_now)

            # Ocean state is on the ocean grid; remap SST / surface currents onto
            # the atmosphere grid for the coupler / surface step (identity =>
            # pass-through).
            sst_o, u_o, v_o = self._ocean_surface_KuvC()
            sst = remap_field(sst_o, self._grid_remapper.o2a)
            u_sfc = remap_field(u_o, self._grid_remapper.o2a)
            v_sfc = remap_field(v_o, self._grid_remapper.o2a)

            # Same seasonal insolation seam as the atmosphere (iter 449/461) so the surface
            # step's day-of-year matches the atmosphere's season; offset 0 => identical.
            doy, _ = self._atm._calendar_for_radiation(day)

            # Transient land-use cover: the segment's calendar year (from the atm
            # sub-driver's start year); a year-varying land provider re-weights its
            # vegetation params, static providers ignore it (byte-identical).
            _sy = getattr(self._atm, "_start_year", None)
            cover_year = None if _sy is None else float(_sy) + day / 365.0

            self._sfc_state, sfc_response = self._step_surface(
                self._sfc_state,
                atm_forcing,
                self._tile_config,
                ocean_sst=sst,
                ocean_u_sfc=u_sfc,
                ocean_v_sfc=v_sfc,
                dt=sub_dt,
                doy=float(doy),
                year=cover_year,
            )
            self._last_sfc_response = sfc_response

            # CO2 tracer update
            self._step_co2_tracer(sub_dt)

        # Interactive carbon-radiation coupling (#3 / C4MIP): feed the prognostic
        # CO2 back to atmospheric radiation.  The next atmosphere segment's GHG
        # override uses this global-mean CO2 mole fraction, so the carbon cycle
        # changes radiative forcing.  Gated on the tracer (and inert for gray
        # radiation, which ignores GHG) => fixed-CO2 runs are byte-identical.
        if self.coupled_cfg.co2_tracer and hasattr(self, '_co2_field'):
            co2_vmr = float(jnp.mean(self._co2_field)) / (
                constants.M_CO2 / constants.M_air)
            self._atm._co2_vmr_override = co2_vmr

        # Diagnostics (once per segment, not per sub-step)
        self._log_coupled_diag(day, dt_segment)

    def _log_coupled_diag(self, day, dt_segment=0.0):
        """Record coupled diagnostics for this segment.

        Stacks all reductions into one ``jnp.stack`` and pulls them in
        a single ``np.asarray`` transfer.  Each ``float(jnp.X(...))``
        was previously its own device→host sync, serialising 3-5
        GPU stalls per coupling segment.
        """
        sst = self._ocean_surface_KuvC()[0]
        has_co2 = self.coupled_cfg.co2_tracer and hasattr(self, '_co2_field')
        has_T_sfc = self._last_sfc_response is not None

        terms = [area_weighted_mean(sst, self._ocean_area_w),
                 jnp.min(sst), jnp.max(sst)]
        # co2/T_sfc kept as unweighted means deliberately: the co2 global mean
        # mirrors the radiation-override mean (line ~1100), so area-weighting it
        # here would diverge from the value that actually forces the radiation —
        # that change is NOT diagnostics-only and needs separate validation.
        if has_co2:
            terms.append(jnp.mean(self._co2_field))
        if has_T_sfc:
            terms.append(jnp.mean(self._last_sfc_response.T_sfc))
        host = np.asarray(jnp.stack(terms))

        diag = {
            "day": float(day),
            "sst_mean": float(host[0]),
            "sst_min": float(host[1]),
            "sst_max": float(host[2]),
        }
        # SST drift vs run start — the headline slab-piControl drift metric.
        # Cheap (reuses sst_mean; no new device sync) and gives operators
        # long-run drift visibility for multi-decadal coupled runs.  A full
        # global TOA energy-residual diagnostic needs the atmosphere's TOA
        # fluxes plumbed into the coupled diag — deferred (see audit gap #4).
        if self._sst_mean_init is None:
            self._sst_mean_init = diag["sst_mean"]
        diag["sst_drift_K"] = diag["sst_mean"] - self._sst_mean_init
        idx = 3
        if has_co2:
            diag["co2_ppmv_mean"] = (
                float(host[idx])
                / (constants.M_CO2 / constants.M_air) * 1e6
            )
            idx += 1
        if has_T_sfc:
            diag["T_sfc_mean"] = float(host[idx])

        # --- F2 water-conservation tripwires (diagnostics-only, RANK-LOCAL) ---
        # Independent witnesses built from the ACTUAL per-tile conserved-water
        # flows, so a coupler water-routing bug surfaces early instead of as a
        # silent multi-year drift.  These reductions are RANK-LOCAL, matching
        # the sst_mean reduction above; a SHARDED coupled run MUST route the SUM
        # through global_sum_mpi (see legoesm.diagnostics.water_budget) — the
        # keys carry an explicit ``_ranklocal`` suffix so a global conservation
        # residual is never read off a per-rank sum.  Host-side float() off the
        # differentiated segment loss => no VJP/donation concern.  NOTE: the atm
        # residual needs a PREVIOUS-segment CWV, so it first appears on the
        # SECOND diagnostic segment (mirrors the sst_drift first-call seeding).
        # Catches C1 (atm moisture budget) and M2 (runoff wet-mask); H2 (frozen
        # precip destroyed on the ice fraction) needs the ice-storage inventory
        # term — see the F2 fork in the design note — and is NOT wired here.
        if (self._last_sfc_response is not None
                and self._last_atm_precip is not None
                and dt_segment > 0.0):
            from legoesm.diagnostics.column_integrals import column_water_vapor
            from legoesm.diagnostics.water_budget import atm_moisture_residual
            atm_area = getattr(self._atm.grid, "grid_area", None)
            cwv_now = column_water_vapor(
                self._atm.q_v,
                self._atm.state.p_s.data,
                jnp.asarray(self._atm.sigma.dsigma),
            )
            if self._cwv_prev is not None:
                diag["water_atm_residual_kg_m2_s_global"] = float(
                    atm_moisture_residual(
                        cwv_now, self._cwv_prev,
                        self._last_sfc_response.surface_mass_flux,
                        self._last_atm_precip,
                        atm_area, float(dt_segment),
                    )
                )
            self._cwv_prev = cwv_now
        _rexp = getattr(self, "_last_runoff_export_integral_ranklocal", None)
        _rapp = getattr(self, "_last_runoff_applied_integral_ranklocal", None)
        if _rexp is not None and _rapp is not None:
            diag["water_runoff_residual_kg_s_global"] = (
                float(_rexp) - float(_rapp))

        # --- F2 closed water-INVENTORY residual (tripwire C: ice-fraction
        #     precip destroyed, H2).  DIAGNOSTIC-ONLY, host-side float() off the
        #     differentiated loss, RANK-LOCAL (a sharded run must route the SUMs
        #     through global_sum_mpi -- advective moisture divergence crosses
        #     rank boundaries).  DYNAMIC-OCEAN ONLY (F_ocean_applied is built
        #     only on the dynamic path).  W_prev is seeded on the first diag
        #     (mirrors _cwv_prev / _sst_mean_init) so the residual first appears
        #     on the SECOND diagnostic segment.  Wrapped in try/except so a
        #     diagnostic can NEVER abort the model trajectory (byte-identical
        #     doctrine): any shape/attr surprise on an untested tile layout
        #     (e.g. subset-column multilayer land) just omits the key for that
        #     segment. ---
        _foa = getattr(self, "_last_f_ocean_applied_integral_ranklocal", None)
        if (getattr(self, "_is_dynamic_ocean", False)
                and self._sfc_state is not None
                and _foa is not None
                and dt_segment > 0.0):
            try:
                from legoesm.diagnostics.column_integrals import (
                    column_water_vapor,
                )
                from legoesm.diagnostics.water_budget import (
                    area_integral,
                    ice_water_content,
                    land_water_content_multilayer,
                    land_water_content_slab,
                    water_inventory_residual,
                )
                from legoesm.ice.state import DynamicSeaIceState
                from legoesm.land.state import MultiLayerLandState
                atm_area = getattr(self._atm.grid, "grid_area", None)
                if atm_area is not None and self._tile_config is not None:
                    p_s = self._atm.state.p_s.data
                    dsig = jnp.asarray(self._atm.sigma.dsigma)
                    # W_atm = column vapour + any CARRIED condensate tracers
                    # (same (1/g) INT q dp mass weighting); each accessor is
                    # None when that species is not carried (dry/kessler).
                    w_atm = column_water_vapor(self._atm.q_v, p_s, dsig)
                    for _qn in ("q_c", "q_r", "q_i", "q_s", "q_g"):
                        _qx = getattr(self._atm, _qn, None)
                        if _qx is not None:
                            w_atm = w_atm + column_water_vapor(_qx, p_s, dsig)
                    # Whole-cell tile weights on the ATM grid: the ice-tile
                    # concentration is fraction-OF-WATER so W_ice weights by
                    # f_water = 1 - f_land - f_lake; per-land-area land storage
                    # weights by f_land (matches compute_tile_fractions:
                    # f_ice = f_water*conc).
                    f_land = jnp.clip(self._tile_config.f_land, 0.0, 1.0)
                    f_lake = jnp.clip(self._tile_config.f_lake, 0.0, 1.0)
                    f_water = jnp.clip(1.0 - f_land - f_lake, 0.0, 1.0)
                    ice = self._sfc_state.ice
                    _hsnow = (ice.h_snow.data
                              if isinstance(ice, DynamicSeaIceState) else None)
                    w_ice = ice_water_content(
                        ice.h_ice.data, ice.concentration.data, f_water,
                        h_snow=_hsnow)
                    land = self._sfc_state.land
                    if isinstance(land, MultiLayerLandState):
                        from legoesm.land.soil_grid import make_soil_grid
                        dz = make_soil_grid(self._land_cfg.soil_grid).dz
                        w_land = land_water_content_multilayer(
                            land.theta_soil, dz, land.snow_depth,
                            f_land.reshape(-1),
                            surface_water=land.surface_water)
                    else:
                        w_land = land_water_content_slab(
                            land.W_bucket.data, land.snow_depth.data, f_land)
                    # W_lake = 0 (fixed-depth two-layer lake stores no water).
                    # Flatten every per-cell term so a multilayer land (ncol,)
                    # and the spatial atm/ice terms integrate uniformly on the
                    # atm grid.
                    w_cell = (w_atm.reshape(-1) + w_ice.reshape(-1)
                              + w_land.reshape(-1))
                    store_int = area_integral(
                        w_cell, atm_area.reshape(-1))
                    if self._water_store_prev_integral_ranklocal is not None:
                        diag["water_inventory_residual_kg_s_global"] = float(
                            water_inventory_residual(
                                store_int,
                                self._water_store_prev_integral_ranklocal,
                                _foa, float(dt_segment)))
                    self._water_store_prev_integral_ranklocal = store_int
            except Exception:
                # A diagnostic must NEVER break the run (byte-identical
                # trajectory); omit the key for this segment on any surprise.
                pass

        self._coupled_diag.append(diag)

    # ==================================================================
    # Run
    # ==================================================================

    def run(
        self,
        start_step: int = 0,
        start_day: float | None = None,
        segment_callback=None,
    ) -> str:
        """Run the coupled integration.

        ``segment_callback(driver, day, dt_segment)`` is an OPTIONAL extra hook
        invoked at each segment boundary AFTER the coupling step ``_segment_hook``
        (so it sees the post-coupling state, e.g. the updated ocean SST) — used to
        sample diagnostics such as the time-mean column state for ERA5 comparison.
        ``None`` (default) is byte-identical to the plain coupled run.
        """
        logger.info("Starting coupled ESM run")
        if segment_callback is None:
            hook = self._segment_hook
        else:
            def hook(driver, day, dt_segment):
                self._segment_hook(driver, day, dt_segment)  # couple first
                segment_callback(driver, day, dt_segment)    # then sample
        status = self._atm.run(
            start_step=start_step,
            start_day=start_day,
            segment_callback=hook,
            # Checkpoint the FULL coupled state (atm + ocean + surface + CO2),
            # not just the atmosphere, on periodic and wallclock-budget saves.
            checkpoint_callback=self.save_checkpoint,
        )
        logger.info(f"Coupled ESM run: {status}")
        return status

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def state(self):
        return self._atm.state

    @property
    def q_v(self):
        """Atmospheric specific humidity ``q_v`` (stored outside the dycore state)."""
        return self._atm.q_v

    @property
    def ocean_state(self):
        return self._ocean_state

    @property
    def surface_state(self):
        return self._sfc_state

    def get_sst_sic(self, day):
        """The coupled SST + SIC on the ATMOSPHERE grid (the same public signature as
        :meth:`ModelDriver.get_sst_sic`).

        Delegates to the atmosphere driver, whose ``get_sst_sic`` was overridden at setup
        (:meth:`_override_sst`) to return the slab/dynamic-ocean SST remapped onto the
        atmosphere grid via the coupler's ``o2a`` remapper — so it is atm-grid even when the
        ocean runs on a DIFFERENT grid.  Exposed (like ``grid`` / ``sigma`` / ``state`` /
        ``q_v``) so the column comparison reads the CMIP coupled SST on the atmosphere grid
        (matching the atm columns) for BOTH AMIP and CMIP — unlike ``ocean_state.T_sfc``,
        which is on the OCEAN grid and would mis-align the env tag when ``ocean_grid``
        differs (cf. the ``column_state_from_hydrostatic`` sst_K grid guard, iter 329).
        """
        return self._atm.get_sst_sic(day)

    @property
    def diagnostics(self):
        return self._atm.diagnostics

    @property
    def coupled_diagnostics(self):
        return self._coupled_diag

    # Coupled-checkpoint format version.  Bump when the saved layout changes so
    # a stale restart is detected rather than silently mis-mapped.
    #   v1: slab ocean (ocean_T_sfc/ocean_T_deep) + flattened surface state.
    #   v2: + full dynamic 3D ocean (ocean3d_* = the LatLonCGridOceanState
    #       pytree) so an ocean_mode='dynamic' run can checkpoint/restart.
    #       The v1 slab keys are unchanged, so a v1 slab checkpoint still
    #       restores under v2 (a stale-version restore only warns; it is the
    #       additive dynamic-ocean keys that a v1 reader would lack).
    #   v3: + coupling-lag surface response (sfcresp_* = the SurfaceToAtm
    #       ``_last_sfc_response`` lag buffer) + sst_mean_init (the SST-drift
    #       reference), so a --resume run's first coupled sub-step delivers
    #       the SAME lagged runoff / ice-lake-freshwater / CO2 fluxes as the
    #       uninterrupted run and sst_drift_K stays referenced to the ORIGINAL
    #       run start.  Additive: a v1/v2 checkpoint still restores; the lag
    #       buffer / drift reference then fall back to the pre-v3 resume
    #       behavior (explicit in load_coupled_checkpoint, not silent).
    _CKPT_VERSION = 3

    def save_checkpoint(self, step: int, day: float) -> None:
        """Save atmosphere + ocean + surface + CO2 state."""
        self._atm.save_checkpoint(step, day)

        # Save coupled state alongside the atmosphere checkpoint.
        elapsed_day = day - self.atm_config.start_day
        coupled_path = self.output_dir / f"coupled_day_{int(elapsed_day):04d}.npz"
        arrays = {}
        # Provenance for load-time validation (version + ocean grid shape so a
        # checkpoint from a different ocean_grid / config is caught, not silently
        # restored into a mismatched state — see load_coupled_checkpoint).
        arrays["_ckpt_version"] = np.asarray(self._CKPT_VERSION, dtype=np.int64)
        if self._ocean_state is not None:
            if getattr(self, "_is_dynamic_ocean", False):
                # Dynamic 3D ocean (ckpt v2): flatten the full
                # LatLonCGridOceanState pytree (T,S,u,v,eta,w + any active
                # AB2 / SOM / EKE / TKE history) into ``ocean3d_*`` leaves.
                # The whole pytree — including the static H_bathy and the
                # land/u/v mask triple — is saved together so the restore is
                # atomic and mask-consistent (a partial mask restore would
                # leak mass through walls; CLAUDE.md "Land/face masks").
                arrays["_ckpt_ocean3d_shape"] = np.asarray(
                    self._ocean_state.T.data.shape, dtype=np.int64)
                arrays.update(
                    _flatten_pytree_to_npz(self._ocean_state, "ocean3d_"))
            else:
                arrays["_ckpt_ocean_shape"] = np.asarray(
                    self._ocean_state.T_sfc.data.shape, dtype=np.int64)
                # Ocean state (SlabOceanState is a NamedTuple of Fields)
                arrays["ocean_T_sfc"] = np.asarray(self._ocean_state.T_sfc.data)
                arrays["ocean_T_deep"] = np.asarray(
                    self._ocean_state.T_deep.data)

        # CO2 tracer field
        if hasattr(self, '_co2_field') and self._co2_field is not None:
            arrays["co2_field"] = np.asarray(self._co2_field)

        # Surface state (land, ice, lake, accumulator, carbon) — flatten
        # the pytree into a dict of named arrays for serialization.
        if self._sfc_state is not None:
            arrays.update(_flatten_pytree_to_npz(self._sfc_state, "sfc_"))

        # Coupling-lag surface response (ckpt v3).  ``_last_sfc_response`` is
        # the one-coupling-sub-step lag buffer the NEXT segment reads for the
        # land-runoff / ice-lake-freshwater delivery, the CO2 tracer flux and
        # the radiation skin-T/albedo channels; without it a --resume run's
        # first coupled sub-step takes the prev-is-None zero-flux branch,
        # dropping one sub-step of those fluxes vs the uninterrupted run.
        if self._last_sfc_response is not None:
            arrays.update(
                _flatten_pytree_to_npz(self._last_sfc_response, "sfcresp_"))

        # SST-drift reference (ckpt v3): the run-start SST mean, so the
        # resumed sst_drift_K diagnostic stays referenced to the ORIGINAL run
        # start rather than resetting at the restart point.
        if self._sst_mean_init is not None:
            arrays["sst_mean_init"] = np.asarray(
                self._sst_mean_init, dtype=np.float64)

        if arrays:
            np.savez(coupled_path, **arrays)
            logger.info(f"  Coupled checkpoint: {coupled_path.name}")

    def load_coupled_checkpoint(self, step_day: float,
                               checkpoint_dir: str | Path | None = None) -> None:
        """Load coupled state saved alongside an atmosphere checkpoint.

        Parameters
        ----------
        step_day : float
            Elapsed day used in the filename (same as atmosphere checkpoint).
        checkpoint_dir : str or Path, optional
            Directory containing the coupled checkpoint.  Defaults to
            ``self.output_dir``.
        """
        from legoesm.core.field import Field

        base = Path(checkpoint_dir) if checkpoint_dir is not None else self.output_dir
        coupled_path = base / f"coupled_day_{int(step_day):04d}.npz"
        if not coupled_path.exists():
            logger.warning(f"No coupled checkpoint at {coupled_path}")
            return

        data = np.load(coupled_path)

        # Validate provenance FIRST — a checkpoint from a different format
        # version or a different ocean grid must NOT be mapped into a mismatched
        # state (the failure mode the grid-coupling review flagged).  A v1 slab
        # checkpoint loaded by v2 code still restores (the slab keys are
        # unchanged); the warning only flags the additive dynamic-ocean keys a
        # v1 reader would lack.
        saved_ver = int(data["_ckpt_version"]) if "_ckpt_version" in data.files else 0
        if saved_ver != self._CKPT_VERSION:
            logger.warning(
                f"Coupled checkpoint {coupled_path.name}: format version "
                f"{saved_ver} != current {self._CKPT_VERSION}; restore may be "
                f"unreliable.")

        # Dynamic 3D ocean (ckpt v2): restore the PROGNOSTIC LatLonCGridOceanState
        # leaves (T,S,u,v,eta,w + any active AB2/SOM/EKE/TKE history) from the
        # ``ocean3d_*`` keys.  A dynamic run whose checkpoint predates v2 (no
        # ocean3d_* keys) cannot have its 3D ocean restored — refuse loudly
        # rather than silently resume the ocean at its IC.
        if getattr(self, "_is_dynamic_ocean", False):
            if not any(k.startswith("ocean3d_") for k in data.files):
                raise ValueError(
                    f"Coupled checkpoint {coupled_path.name}: ocean_mode="
                    "'dynamic' but the checkpoint carries no 3D-ocean state "
                    "('ocean3d_*'; a pre-ckpt-v2 file).  The 3D ocean cannot "
                    "be restarted from it — restart from the initial condition "
                    "instead.")
            if "_ckpt_ocean3d_shape" not in data.files:
                raise ValueError(
                    f"Coupled checkpoint {coupled_path.name}: ocean3d_* leaves "
                    "present but '_ckpt_ocean3d_shape' is missing (malformed v2 "
                    "checkpoint); refusing to restore.")
            if self._ocean_state is not None:
                saved_shape = tuple(
                    int(s) for s in data["_ckpt_ocean3d_shape"])
                cur_shape = tuple(int(s) for s in self._ocean_state.T.data.shape)
                if saved_shape != cur_shape:
                    raise ValueError(
                        f"Coupled checkpoint {coupled_path.name}: saved "
                        f"3D-ocean shape {saved_shape} != current ocean shape "
                        f"{cur_shape}.  The ocean grid / vertical levels "
                        f"changed since the checkpoint; refusing to restore a "
                        f"mismatched state.")
                # Restore the prognostic state (per-leaf shape/dtype checked;
                # strict on structure drift so a toggled EKE/TKE/SOM scheme is
                # caught, not silently reseeded), then RESET the static geometry
                # (H_bathy + the land/u/v mask triple) to the deterministic
                # fresh-init values so it stays consistent with the caches the
                # ocean model + coupler built from it (_ocean_model,
                # _ocean_z_coord, _ocean_land_mask).  Geometry is config-derived
                # and identical on a clean resume; sourcing it from the fresh
                # init (not the npz) makes a same-shape config change a no-op on
                # geometry rather than a silent cache desync.
                fresh = self._ocean_state
                restored, _ = _restore_pytree_from_npz(
                    fresh, data, "ocean3d_", coupled_path.name, strict=True)
                # Reset the static geometry to the deterministic fresh-init
                # values (config-derived, identical on a clean resume).  The
                # MPAS/voronoi ocean state carries H_bathy + a cell land_mask
                # but has NO edge u/v masks (its mesh geometry lives on the
                # model, not the state), so the latlon C-grid face-mask triple
                # reset only applies where those fields exist — an unconditional
                # ``_replace(u_mask=..., v_mask=...)`` crashes MPASOceanState.
                self._ocean_state = restored._replace(
                    H_bathy=fresh.H_bathy, land_mask=fresh.land_mask)
                if hasattr(fresh, "u_mask"):
                    self._ocean_state = self._ocean_state._replace(
                        u_mask=fresh.u_mask, v_mask=fresh.v_mask)

        if "ocean_T_sfc" in data.files and self._ocean_state is not None:
            saved_shape = tuple(int(s) for s in data["ocean_T_sfc"].shape)
            cur_shape = tuple(int(s) for s in self._ocean_state.T_sfc.data.shape)
            if saved_shape != cur_shape:
                raise ValueError(
                    f"Coupled checkpoint {coupled_path.name}: saved ocean state "
                    f"shape {saved_shape} != current ocean grid shape "
                    f"{cur_shape}.  The ocean_grid / config changed since the "
                    f"checkpoint was written; refusing to restore a mismatched "
                    f"state.")

        if "ocean_T_sfc" in data and self._ocean_state is not None:
            self._ocean_state = self._ocean_state._replace(
                T_sfc=Field(
                    data=jnp.asarray(data["ocean_T_sfc"]),
                    name="T_sfc",
                    dims=self._ocean_state.T_sfc.dims,
                    units="K",
                ),
                T_deep=Field(
                    data=jnp.asarray(data["ocean_T_deep"]),
                    name="T_deep",
                    dims=self._ocean_state.T_deep.dims,
                    units="K",
                ),
            )

        if "co2_field" in data:
            self._co2_field = jnp.asarray(data["co2_field"])

        # Restore surface state (land, ice, lake, accumulator, carbon) from its
        # flattened pytree leaves — atomic + drift-warned via the same helper
        # used to save it (and to save/restore the dynamic ocean above).
        if self._sfc_state is not None:
            self._sfc_state, _ = _restore_pytree_from_npz(
                self._sfc_state, data, "sfc_", coupled_path.name)

        # Coupling-lag surface response (ckpt v3): restore the one-sub-step lag
        # buffer so the first coupled sub-step after --resume delivers the SAME
        # lagged runoff / ice-lake-freshwater / CO2 fluxes (and radiation
        # skin-T/albedo channels) as the uninterrupted run.
        if any(k.startswith("sfcresp_") for k in data.files):
            from legoesm.core.coupling_fields import SurfaceToAtm
            _struct = SurfaceToAtm(*([0] * len(SurfaceToAtm._fields)))
            _paths = ["sfcresp_" + ".".join(str(p) for p in pp)
                      for pp, _ in jax.tree_util.tree_leaves_with_path(_struct)]
            expected = set(_paths)
            saved_resp = {k for k in data.files if k.startswith("sfcresp_")}
            if saved_resp == expected:
                # Every SurfaceToAtm leaf lives on the atmosphere grid with
                # the same 2D shape as p_s (cube (6,n,n) / lat-lon
                # (nlat,nlon)) — that pins the current-run shape the restore
                # helper validates against.  The dtype comes from each SAVED
                # leaf, canonicalized by the current runtime (jnp.zeros
                # downcasts float64 -> float32 when x64 is off), so a
                # same-config resume restores the buffer BIT-IDENTICALLY (the
                # coupler emits float64 under x64 even when the storage
                # policy keeps p_s float32).
                _shape = self._atm.state.p_s.data.shape
                template = jax.tree_util.tree_unflatten(
                    jax.tree_util.tree_structure(_struct),
                    [jnp.zeros(_shape, dtype=jnp.zeros((), data[k].dtype).dtype)
                     for k in _paths])
                self._last_sfc_response, _ = _restore_pytree_from_npz(
                    template, data, "sfcresp_", coupled_path.name, strict=True)
            else:
                # SurfaceToAtm changed shape (field append/removal) since the
                # save.  A PARTIAL restore would silently zero some channels;
                # fall back to a fresh (None) lag buffer instead — exactly the
                # pre-v3 resume behavior (one zero-flux land/ice/CO2 sub-step),
                # loudly.
                logger.warning(
                    f"Coupled checkpoint {coupled_path.name}: 'sfcresp_' "
                    f"structure drift (SurfaceToAtm changed since the save); "
                    f"falling back to a fresh coupling-lag buffer (pre-v3 "
                    f"resume behavior: one zero-flux land/ice/CO2 sub-step).")
        # else: pre-v3 checkpoint — no lag buffer saved.  Keep None: the first
        # coupled sub-step after resume takes the prev-is-None zero-flux branch
        # (exactly the pre-v3 resume behavior), then rebuilds the buffer.

        # SST-drift reference (ckpt v3): keep sst_drift_K referenced to the
        # ORIGINAL run start across --resume.
        if "sst_mean_init" in data.files:
            self._sst_mean_init = float(data["sst_mean_init"])
        # else: pre-v3 checkpoint — keep None: the drift re-references at the
        # restart point (the pre-v3 resume behavior), explicit not silent.

        logger.info(f"  Loaded coupled checkpoint: {coupled_path.name}")
