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

from legoesm import constants
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.coupled_config import CoupledConfig
from legoesm.diagnostics.energy_budget import area_weighted_mean

logger = logging.getLogger("legoesm.driver.coupled_esm")


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
        self._atm = ModelDriver(atm_config, output_dir=output_dir)
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
        self._last_sfc_response = None
        self._coupled_diag = []
        self._sst_mean_init = None  # set on first diag — SST-drift reference

    @property
    def output_dir(self) -> Path:
        return self._atm.output_dir

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

        # 4. Carbon / CO2 tracer (if active)
        self._init_carbon()

        # 5. Override SST source: slab ocean instead of file
        self._override_sst()

        # 6. Optionally feed the coupler's dynamic surface albedo / skin
        #    temperature back to the atmosphere's radiation (opt-in).
        self._override_sfc()

        logger.info("CoupledESM: all components initialized")
        logger.info(f"  ocean_mode={self.coupled_cfg.ocean_mode}, "
                    f"land_mode={self.coupled_cfg.land_mode}, "
                    f"carbon_active={self.coupled_cfg.carbon_active}")

    def _init_ocean(self):
        """Initialize the slab/two-layer ocean (on the ocean grid)."""
        from legoesm.ocean.simple_ocean import make_ocean, init_slab_state
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field

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
        from legoesm.grids.latlon import LatLonGrid
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean, idealized_bathymetry_latlon_cgrid,
        )

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
            _oc = LatLonCGridOceanConfig()
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
        ocfg = _oc._replace(
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
            f"barotropic={ocfg.barotropic_solver}, "
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
        return LatLonCGridOceanConfig(
            A_h=1.0e5, A_v=1.0e-4, K_v=1.0e-5, B_h=0.0,
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
            f"barotropic={ocfg.barotropic_solver}, pgf={ocfg.pgf_scheme}, "
            f"mesh={cfg.tripole_mesh_path}")

    def _init_coupler(self):
        """Initialize coupler, land, ice, lake surface states."""
        from legoesm.coupler.coupler import make_coupler, init_surface_state
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.land.config import LandConfig, MultiLayerLandConfig
        from legoesm.ice.config import SeaIceConfig
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.core.precision import get_policy

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
                and getattr(cfg, "land_param_source", "analytical") == "clm"):
            import legoesm.land.clm_surface_map as _csm
            if cfg.land_mode == "multilayer":
                ch, snow_max = _csm.TUNED_CH_MULTILAYER, _csm.TUNED_SNOW_ALBEDO_MAX_MULTILAYER
            else:
                ch, snow_max = _csm.TUNED_CH, _csm.TUNED_SNOW_ALBEDO_MAX
            land_cfg = land_cfg._replace(
                Ch_land=ch, Cd_land=ch, snow_albedo_feedback=True,
                land_albedo=land_cfg.land_albedo._replace(alpha_snow_max=snow_max))
            logger.info(f"  Land: ERA5-calibrated Ch/snow params "
                        f"({cfg.land_mode} CLM default path)")

        # Spatial soil hydraulics from the CLM reference map (per-column van-
        # Genuchten retention) for the Richards multilayer land.
        if (cfg.land_mode == "multilayer"
                and getattr(cfg, "land_param_source", "analytical") == "clm"
                and self._atm._grid_lat is not None):
            from legoesm.land.clm_surface_map import (
                download_clm_surfdata, load_clm_surface, clm_hydraulics_config,
                clm_multilayer_thermal_config, clm_multilayer_ch)
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
            land_cfg = land_cfg._replace(
                hydraulics=cast(clm_hydraulics_config(smap)),
                thermal=cast(clm_multilayer_thermal_config(smap)),
                Ch_land=ch_cell, Cd_land=ch_cell)
            logger.info("  Soil: CLM reference VG + per-PFT thermal/Ch map (per-column)")

        # Coupled DIURNAL surface model (default ON for the multilayer land): the
        # coupled atmosphere supplies a fully-resolved diurnal cycle at a single,
        # consistent lowest-model-level height, so the surface exchange can be the
        # physical Monin-Obukhov (MOST) scheme (roughness-driven, stability-dependent)
        # and transpiration the Farquhar photosynthesis-stomata coupling — both of
        # which are ill-posed against the crude offline single-column forcing but
        # well-posed here.  Carbon must run (differland) so Farquhar has a prognostic
        # LAI; the coupler already initialises + threads the carbon state.
        if (cfg.land_mode == "multilayer"
                and getattr(cfg, "land_diurnal_surface", True)):
            land_cfg = enable_diurnal_surface_land(land_cfg)
            logger.info("  Land surface: MOST exchange + Farquhar stomata "
                        "(coupled diurnal model)")

        self._land_cfg = land_cfg  # store for diagnostics

        # PFT parameter provider (if requested and land is active)
        land_param_provider = None
        if cfg.use_pft and cfg.land_mode != "none":
            land_param_provider = self._build_pft_provider(shape_2d)

        # Build coupler step function
        self._step_surface = make_coupler(
            coupler_cfg, land_cfg, ice_cfg, lake_cfg,
            lat=self._atm._grid_lat,
            grid=self._atm.grid,
            land_param_provider=land_param_provider,
        )

        # Initialize surface state.  Optionally warm-start the soil at the
        # atmosphere's lat-structured near-surface air temperature (t=0) — the
        # same spatial source the slab SST uses — so tropical land does not
        # cold-spin from a uniform 280 K (default off => byte-identical).
        soil_kwargs = {}
        if getattr(cfg, "warm_start_soil", False):
            soil_kwargs["T_soil_init"] = self._atm.state.T.data[..., -1]
            logger.info("  Soil warm-start: T_soil init = atm near-surface air T")
        self._sfc_state = init_surface_state(
            shape_2d, land_config=land_cfg, **soil_kwargs,
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

        source = getattr(self.coupled_cfg, "land_param_source", "analytical")
        if source == "clm":
            from legoesm.land.clm_surface_map import clm_surface_provider
            lon = self._atm._grid_lon
            lat_deg = np.asarray(jnp.rad2deg(jnp.broadcast_to(lat, shape_2d)).ravel())
            lon_deg = np.asarray(jnp.rad2deg(jnp.broadcast_to(lon, shape_2d)).ravel())
            # Use the calibration matched to the active land scheme (each tuned its
            # surface-energy params against a different soil forward).
            variant = ("multilayer" if self.coupled_cfg.land_mode == "multilayer"
                       else "slab")
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
                sic = self._sfc_state.ice.concentration.data
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
        if not getattr(self.coupled_cfg, "couple_surface_radiation", False):
            return

        from legoesm.forcing.surface_utils import (
            blend_surface_property, blend_surface_temperature,
        )

        def _seed_blend(day):
            # Same static blend the atmosphere radiation would use, as
            # grid-shaped arrays — used only until the first sfc_response.
            sst, sic = self._atm.get_sst_sic(day)
            acfg = self.atm_config
            alb = blend_surface_property(
                sic, acfg.albedo_ice, acfg.albedo_ocean,
            )
            T = blend_surface_temperature(sst, sic, acfg.T_ice)
            return alb, T

        def _coupled_get_sfc_override(day):
            r = self._last_sfc_response
            if r is None or getattr(r, "albedo", None) is None:
                return _seed_blend(day)
            return r.albedo, r.T_sfc

        self._atm.get_sfc_override = _coupled_get_sfc_override
        logger.info(
            "  Surface-radiation feedback: dynamic albedo + skin T -> radiation"
        )

    # ==================================================================
    # Coupling step
    # ==================================================================

    def _build_atm_forcing(self, day: float):
        """Build AtmToSurface from atmosphere state and physics."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.forcing.surface_utils import blend_surface_temperature

        state = self._atm.state
        q_v = self._atm.q_v
        p_s = state.p_s.data
        T_low = state.T.data[..., -1]
        u_low = state.u.data[..., -1]
        v_low = state.v.data[..., -1]
        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        sigma_full = jnp.asarray(self._atm.sigma.sigma_full)
        p_low = p_s * sigma_full[-1]
        rho_low = p_low / (constants.R_d * T_low)

        # Radiation and precipitation from last atmosphere physics
        aux = getattr(self._atm, '_carry_aux', {})
        sw_net_sfc = aux.get("held_sw_net_sfc", jnp.zeros_like(p_s))
        lw_net_sfc = aux.get("held_lw_net_sfc", jnp.zeros_like(p_s))
        seg_precip = aux.get("seg_precip", jnp.zeros_like(p_s))

        # Reconstruct gross downward fluxes from net
        acfg = self.atm_config
        sst, sic = self._atm.get_sst_sic(day)
        from legoesm.forcing.surface_utils import blend_surface_property
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
            getattr(self.coupled_cfg, "couple_surface_radiation", False)
            and _resp is not None
            and getattr(_resp, "albedo", None) is not None
        )
        if _dyn_sfc:
            albedo_eff = _resp.albedo
            T_sfc = _resp.T_sfc
        else:
            albedo_eff = blend_surface_property(
                sic, acfg.albedo_ice, acfg.albedo_ocean,
            )
            T_sfc = blend_surface_temperature(sst, sic, acfg.T_ice)
        sw_down = sw_net_sfc / jnp.maximum(1.0 - albedo_eff, 0.01)
        # Surface emissivity: blend canonical ocean/ice emissivity by sea-ice
        # fraction (same blend as albedo, matching earth_system_driver). The
        # old ``getattr(coupled_cfg, "surface_emissivity", ...)`` referenced a
        # field ``CoupledDriverConfig`` never defines, so it silently pinned
        # emissivity to the ocean value and ignored the ice fraction.
        eps_sfc = blend_surface_property(
            sic, constants.emissivity_ice, constants.emissivity_ocean,
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

        # Cosine zenith
        from legoesm.forcing.time_utils import day_to_calendar
        doy, _ = day_to_calendar(day)
        lat = self._atm._grid_lat
        if lat is not None:
            from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
            # Solar constant from legoesm.constants per CLAUDE.md.
            # ``acfg.S_0`` allows override for sensitivity studies.
            S_0 = getattr(acfg, 'S_0', constants.S_0)
            Q_daily = daily_mean_insolation(lat, float(doy), S_0=S_0)
            cos_zen = jnp.clip(Q_daily / S_0, 0.0, 1.0)
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

        return AtmToSurface(
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

    def _step_ocean(self, atm_forcing, dt):
        """Advance the slab ocean one coupling step.

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
                self._ocean_state, atm_forcing, dt,
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
        which is the cube-only channel).  Ice→ocean channels (freshwater_flux /
        ocean_heat_extraction / salt_flux / ice stress) are Phase 3 — an
        aquaplanet Phase-1 run has no ice tile."""
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.config import CouplerConfig
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.freshwater import FreshwaterForcing
        from legoesm import constants

        sst_K, u_o, v_o = self._ocean_surface_KuvC()
        ccfg = getattr(self, "_coupler_cfg", None) or CouplerConfig()
        tile = ocean_tile_response(atm_forcing, sst_K, u_o, v_o, ccfg)
        sw_net = atm_forcing.sw_down * (1.0 - tile.albedo)
        q_net = (sw_net + atm_forcing.lw_down
                 - tile.lw_up - tile.shflx - tile.lhflx)
        evap = tile.lhflx / constants.L_v            # [kg/m²/s], positive up
        z = jnp.zeros_like(sw_net)
        fw = FreshwaterForcing(
            precip=atm_forcing.precip_total, evap=evap, runoff=z, ice_fw=z,
        )
        sf = OceanSurfaceForcing(
            sw_down=atm_forcing.sw_down, q_net=q_net,
            tau_x=tile.tau_x, tau_y=tile.tau_y, freshwater=None,
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
        dsigma = jnp.asarray(self._atm.sigma.dsigma)
        # Layer mass of lowest level: dp / g [kg/m2]
        dp = p_s * dsigma[-1]
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
        from legoesm.forcing.time_utils import day_to_calendar

        coupling_dt = self.coupled_cfg.coupling_dt  # default 3600 s
        n_sub = max(1, int(round(dt_segment / coupling_dt)))
        sub_dt = dt_segment / n_sub

        from legoesm.coupler.grid_remap import remap_field, remap_surface_fields

        atm_forcing = self._build_atm_forcing(day)
        # Surface forcing for the ocean step lives on the OCEAN grid; remap the
        # atm-grid forcing fields onto it (identity remapper => unchanged, so the
        # standard single-grid run is byte-identical).
        ocean_forcing = remap_surface_fields(atm_forcing, self._grid_remapper.a2o)

        for _ in range(n_sub):
            # Step slab ocean (on the ocean grid)
            self._step_ocean(ocean_forcing, sub_dt)

            # Ocean state is on the ocean grid; remap SST / surface currents onto
            # the atmosphere grid for the coupler / surface step (identity =>
            # pass-through).
            sst_o, u_o, v_o = self._ocean_surface_KuvC()
            sst = remap_field(sst_o, self._grid_remapper.o2a)
            u_sfc = remap_field(u_o, self._grid_remapper.o2a)
            v_sfc = remap_field(v_o, self._grid_remapper.o2a)

            doy, _ = day_to_calendar(day)

            self._sfc_state, sfc_response = self._step_surface(
                self._sfc_state,
                atm_forcing,
                self._tile_config,
                ocean_sst=sst,
                ocean_u_sfc=u_sfc,
                ocean_v_sfc=v_sfc,
                dt=sub_dt,
                doy=float(doy),
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
        self._log_coupled_diag(day)

    def _log_coupled_diag(self, day):
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

        self._coupled_diag.append(diag)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run the coupled integration."""
        logger.info("Starting coupled ESM run")
        status = self._atm.run(
            start_step=start_step,
            start_day=start_day,
            segment_callback=self._segment_hook,
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
    def ocean_state(self):
        return self._ocean_state

    @property
    def surface_state(self):
        return self._sfc_state

    @property
    def diagnostics(self):
        return self._atm.diagnostics

    @property
    def coupled_diagnostics(self):
        return self._coupled_diag

    # Coupled-checkpoint format version.  Bump when the saved layout changes so
    # a stale restart is detected rather than silently mis-mapped.
    _CKPT_VERSION = 1

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
        # NOTE: the prognostic 3D ocean (ocean_mode='dynamic') is NOT yet
        # checkpointed — its full LatLonCGridOceanState pytree (T,S,u,v,eta +
        # AB2 history) needs the checkpoint-v2 flatten path (deferred, see
        # docs/ocean/coupled_3d_ocean_plan.md).  Skip the slab-only T_sfc/T_deep save
        # for dynamic so a short Phase-1 run does not crash on the missing
        # T_sfc field; a dynamic run must currently restart from the IC.
        if self._ocean_state is not None and not getattr(
                self, "_is_dynamic_ocean", False):
            arrays["_ckpt_ocean_shape"] = np.asarray(
                self._ocean_state.T_sfc.data.shape, dtype=np.int64)
            # Ocean state (SlabOceanState is a NamedTuple of Fields)
            arrays["ocean_T_sfc"] = np.asarray(self._ocean_state.T_sfc.data)
            arrays["ocean_T_deep"] = np.asarray(self._ocean_state.T_deep.data)

        # CO2 tracer field
        if hasattr(self, '_co2_field') and self._co2_field is not None:
            arrays["co2_field"] = np.asarray(self._co2_field)

        # Surface state (land, ice, lake, accumulator, carbon) — flatten
        # the pytree into a dict of named arrays for serialization.
        if self._sfc_state is not None:
            leaves_with_path = jax.tree_util.tree_leaves_with_path(
                self._sfc_state,
            )
            for path_parts, leaf in leaves_with_path:
                key = "sfc_" + ".".join(str(p) for p in path_parts)
                arrays[key] = np.asarray(leaf)

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

        # The prognostic 3D ocean is NOT yet checkpointed (ckpt v2 deferred), so
        # a restart would resume the atm/surface at day N with the ocean reset
        # to its IC — a silent state inconsistency.  Refuse it loudly until full
        # 3D-ocean checkpointing exists (codex MED); a dynamic run restarts from
        # the IC.
        if getattr(self, "_is_dynamic_ocean", False):
            raise ValueError(
                "Coupled checkpoint restart is not supported for "
                "ocean_mode='dynamic' (the 3D ocean state is not checkpointed; "
                "ckpt v2 is deferred — see docs/ocean/coupled_3d_ocean_plan.md). "
                "Restart from the initial condition instead.")

        # Validate provenance BEFORE restoring — a checkpoint from a different
        # format version or a different ocean grid must NOT be silently mapped
        # into a mismatched state (this is the failure mode the grid-coupling
        # review flagged: ocean_grid != the run's ocean_grid).
        saved_ver = int(data["_ckpt_version"]) if "_ckpt_version" in data.files else 0
        if saved_ver != self._CKPT_VERSION:
            logger.warning(
                f"Coupled checkpoint {coupled_path.name}: format version "
                f"{saved_ver} != current {self._CKPT_VERSION}; restore may be "
                f"unreliable.")
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

        # Restore surface state from flattened pytree leaves.
        sfc_keys = [k for k in data.files if k.startswith("sfc_")]
        if sfc_keys and self._sfc_state is not None:
            leaves_with_path = jax.tree_util.tree_leaves_with_path(
                self._sfc_state,
            )
            # Build a lookup from stringified path -> saved array
            saved = {}
            for k in sfc_keys:
                saved[k] = data[k]

            # Detect surface-state structure drift: leaves expected now but
            # absent from the checkpoint silently keep their fresh-init value
            # (and vice-versa), which corrupts a restart if the CoupledConfig
            # changed.  Warn loudly instead of failing silently.
            expected_keys = {
                "sfc_" + ".".join(str(p) for p in path_parts)
                for path_parts, _ in leaves_with_path
            }
            missing = expected_keys - set(sfc_keys)
            extra = set(sfc_keys) - expected_keys
            if missing or extra:
                logger.warning(
                    f"Coupled checkpoint {coupled_path.name}: surface-state "
                    f"structure drift — {len(missing)} expected leaf(s) absent "
                    f"from the checkpoint (kept fresh-init), {len(extra)} "
                    f"unused checkpoint leaf(s).  The CoupledConfig likely "
                    f"changed since the checkpoint was written.")

            # Replace leaves in-order (same traversal as save)
            new_leaves = []
            for path_parts, leaf in leaves_with_path:
                key = "sfc_" + ".".join(str(p) for p in path_parts)
                if key in saved:
                    new_leaves.append(jnp.asarray(saved[key]))
                else:
                    new_leaves.append(leaf)

            self._sfc_state = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(self._sfc_state),
                new_leaves,
            )

        logger.info(f"  Loaded coupled checkpoint: {coupled_path.name}")
