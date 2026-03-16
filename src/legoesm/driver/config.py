"""Driver configuration dataclasses.

Provides structured configuration for the composable model driver,
with conversion to/from AMIPExperimentConfig for backward compatibility.
"""
from __future__ import annotations

from typing import NamedTuple


class GridConfig(NamedTuple):
    """Horizontal and vertical grid configuration."""
    grid_type: str = "cubed_sphere"  # cubed_sphere, gaussian, latlon, voronoi
    resolution: int = 16             # N for CS, n_max for spectral
    nlev: int = 40
    vertical_coord: str = "hybrid"   # sigma, hybrid
    p_top_Pa: float = 200.0
    stretching: float = 2.0


class DycoreConfig(NamedTuple):
    """Dynamical core configuration."""
    model_type: str = "hydrostatic"       # hydrostatic, nonhydrostatic, spectral_pe
    discretization: str = "centered"      # centered, finite_volume, cgrid
    dt: float = 600.0
    hyperdiff_scale: float = 1.0
    div_damp_scale: float = 1.0
    conservation_fixer: bool = True
    fix_mass: bool = True


class OutputConfig(NamedTuple):
    """Output and diagnostics configuration."""
    output_dir: str = ""
    diag_days: int = 5
    checkpoint_days: int = 0
    monthly_means: bool = False
    cmip_output: bool = False
    clear_sky_diag: bool = False
    checkpoint_format: str = "npz"  # npz, zarr


class ExperimentConfig(NamedTuple):
    """Top-level experiment configuration.

    Composes GridConfig, DycoreConfig, OutputConfig with physics
    and forcing parameters.
    """
    grid: GridConfig = GridConfig()
    dycore: DycoreConfig = DycoreConfig()
    output: OutputConfig = OutputConfig()

    # Integration
    days: int = 200
    start_day: float = 0.0

    # Forcing
    dataset: str = "analytical"
    forcing_path: str = ""
    sst_var: str = ""
    sic_var: str = ""
    sst_offset: float = 0.0
    sic_scale: float = 1.0

    # Radiation
    radiation: str = "gray"
    rad_update_steps: int = 1
    diurnal_cycle: bool = False
    co2_ppmv: float = 415.0
    ch4_ppbv: float = 1900.0
    n2o_ppbv: float = 332.0
    S_0: float = 1361.0
    ozone_source: str = "standard"

    # Clouds & Microphysics
    cloud_scheme: str = "none"
    microphysics: str = "none"

    # Topography
    topography: str = "flat"
    topo_smoothing: int = 4
    topo_edge_blend: float = 0.3

    # Surface
    T_init: float = 300.0
    RH_init: float = 0.7
    dynamic_albedo: bool = False
    carbon_cycle: str = "none"

    # CMIP
    experiment: str = ""
    start_year: int = 1979

    # Surface parameters
    C_H: float = 0.0044
    C_E: float = 0.0044
    T_ice: float = 271.35
    albedo_ice: float = 0.65
    albedo_ocean: float = 0.06
    sfc_emissivity: float = 0.97
    emissivity_ice: float = 0.95
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    sbm_tau_c: float = 7200.0
    sbm_RH_ref: float = 0.7
    sigma_b: float = 0.7
    k_BL_max_per_day: float = 1.0
    k_free_per_day: float = 0.1

    # Distributed
    distributed: bool = False
    ensemble_size: int = 1

    @staticmethod
    def from_amip_config(amip_cfg) -> ExperimentConfig:
        """Convert AMIPExperimentConfig to ExperimentConfig."""
        grid = GridConfig(
            resolution=amip_cfg.resolution,
            nlev=amip_cfg.nlev,
            vertical_coord=amip_cfg.vertical_coord,
            p_top_Pa=amip_cfg.p_top_Pa,
            stretching=amip_cfg.stretching,
        )
        dycore = DycoreConfig(
            dt=amip_cfg.dt,
            hyperdiff_scale=getattr(amip_cfg, 'hyperdiff_scale', 1.0),
        )
        output = OutputConfig(
            output_dir=amip_cfg.output_dir,
            diag_days=amip_cfg.diag_days,
            checkpoint_days=amip_cfg.checkpoint_days,
            monthly_means=amip_cfg.monthly_means,
            cmip_output=amip_cfg.cmip_output,
            clear_sky_diag=amip_cfg.clear_sky_diag,
        )
        return ExperimentConfig(
            grid=grid,
            dycore=dycore,
            output=output,
            days=amip_cfg.days,
            start_day=amip_cfg.start_day,
            dataset=amip_cfg.dataset,
            forcing_path=amip_cfg.forcing_path,
            sst_var=amip_cfg.sst_var,
            sic_var=amip_cfg.sic_var,
            sst_offset=amip_cfg.sst_offset,
            sic_scale=amip_cfg.sic_scale,
            radiation=amip_cfg.radiation,
            rad_update_steps=amip_cfg.rad_update_steps,
            diurnal_cycle=amip_cfg.diurnal_cycle,
            co2_ppmv=amip_cfg.co2_ppmv,
            ch4_ppbv=amip_cfg.ch4_ppbv,
            n2o_ppbv=amip_cfg.n2o_ppbv,
            S_0=amip_cfg.S_0,
            ozone_source=amip_cfg.ozone_source,
            cloud_scheme=amip_cfg.cloud_scheme,
            microphysics=amip_cfg.microphysics,
            topography=amip_cfg.topography,
            topo_smoothing=amip_cfg.topo_smoothing,
            topo_edge_blend=amip_cfg.topo_edge_blend,
            T_init=amip_cfg.T_init,
            RH_init=amip_cfg.RH_init,
            dynamic_albedo=amip_cfg.dynamic_albedo,
            carbon_cycle=amip_cfg.carbon_cycle,
            experiment=amip_cfg.experiment,
            start_year=amip_cfg.start_year,
            C_H=amip_cfg.C_H,
            C_E=amip_cfg.C_E,
            T_ice=amip_cfg.T_ice,
            albedo_ice=amip_cfg.albedo_ice,
            albedo_ocean=amip_cfg.albedo_ocean,
            sfc_emissivity=amip_cfg.sfc_emissivity,
            emissivity_ice=amip_cfg.emissivity_ice,
            tau_equator=amip_cfg.tau_equator,
            tau_pole=amip_cfg.tau_pole,
            sbm_tau_c=amip_cfg.sbm_tau_c,
            sbm_RH_ref=amip_cfg.sbm_RH_ref,
            sigma_b=amip_cfg.sigma_b,
            k_BL_max_per_day=amip_cfg.k_BL_max_per_day,
            k_free_per_day=amip_cfg.k_free_per_day,
            distributed=amip_cfg.distributed,
            ensemble_size=amip_cfg.ensemble_size,
        )

    def to_amip_config(self):
        """Convert to AMIPExperimentConfig for backward compatibility."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        return AMIPExperimentConfig(
            resolution=self.grid.resolution,
            nlev=self.grid.nlev,
            dt=self.dycore.dt,
            vertical_coord=self.grid.vertical_coord,
            p_top_Pa=self.grid.p_top_Pa,
            stretching=self.grid.stretching,
            start_day=self.start_day,
            days=self.days,
            diag_days=self.output.diag_days,
            checkpoint_days=self.output.checkpoint_days,
            dataset=self.dataset,
            forcing_path=self.forcing_path,
            sst_var=self.sst_var,
            sic_var=self.sic_var,
            sst_offset=self.sst_offset,
            sic_scale=self.sic_scale,
            radiation=self.radiation,
            rad_update_steps=self.rad_update_steps,
            diurnal_cycle=self.diurnal_cycle,
            co2_ppmv=self.co2_ppmv,
            ch4_ppbv=self.ch4_ppbv,
            n2o_ppbv=self.n2o_ppbv,
            S_0=self.S_0,
            ozone_source=self.ozone_source,
            cloud_scheme=self.cloud_scheme,
            microphysics=self.microphysics,
            topography=self.topography,
            topo_smoothing=self.topo_smoothing,
            topo_edge_blend=self.topo_edge_blend,
            T_init=self.T_init,
            RH_init=self.RH_init,
            dynamic_albedo=self.dynamic_albedo,
            carbon_cycle=self.carbon_cycle,
            experiment=self.experiment,
            start_year=self.start_year,
            C_H=self.C_H,
            C_E=self.C_E,
            T_ice=self.T_ice,
            albedo_ice=self.albedo_ice,
            albedo_ocean=self.albedo_ocean,
            sfc_emissivity=self.sfc_emissivity,
            emissivity_ice=self.emissivity_ice,
            tau_equator=self.tau_equator,
            tau_pole=self.tau_pole,
            sbm_tau_c=self.sbm_tau_c,
            sbm_RH_ref=self.sbm_RH_ref,
            sigma_b=self.sigma_b,
            k_BL_max_per_day=self.k_BL_max_per_day,
            k_free_per_day=self.k_free_per_day,
            monthly_means=self.output.monthly_means,
            cmip_output=self.output.cmip_output,
            clear_sky_diag=self.output.clear_sky_diag,
            distributed=self.distributed,
            ensemble_size=self.ensemble_size,
            output_dir=self.output.output_dir,
        )
