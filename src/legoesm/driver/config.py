"""Driver configuration dataclasses.

Provides the **canonical runtime configuration** for legoESM:

- ``ExperimentConfig`` is the single authoritative in-memory schema.
- ``GridConfig``, ``DycoreConfig``, ``OutputConfig`` are composed sub-configs.
- Native JSON serialization via ``experiment_config_to_dict`` /
  ``experiment_config_from_dict`` — no intermediate format needed.
- Backward-compatible conversion to/from the legacy
  ``AMIPExperimentConfig`` for checkpoint I/O is preserved but restricted
  to serialization boundaries (see ``to_amip_config`` / ``from_amip_config``).
"""
from __future__ import annotations

import json
from pathlib import Path
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
    discretization: str = "cdgrid"        # cdgrid, spectral, sfno, latlon_fv, mpas
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
    diagnostics_perf_mode: str = "auto"  # auto, always, never


class ExperimentConfig(NamedTuple):
    """Top-level experiment configuration.

    This is the **canonical runtime schema** for legoESM.  All driver,
    physics, forcing, and diagnostic code should consume this type.

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
    ozone_forcing: str = "inline"       # inline, external, off
    ozone_file: str = ""
    ghg_forcing: str = "constant"       # constant, external
    ghg_file: str = ""

    # Solar
    solar_source: str = "constant"      # constant, file, spectral_file
    solar_file: str = ""
    solar_spectral_var: str = "solar_fraction_by_gpt"

    # Aerosol
    aerosol_forcing: str = "off"        # off, external
    aerosol_file: str = ""
    aerosol_reference_aod: float = 0.03
    volcanic_aerosol_file: str = ""
    volcanic_aerosol_scale: float = 1.0

    # Clouds & Microphysics
    cloud_scheme: str = "none"
    microphysics: str = "none"

    # Convection / Turbulence / GWD
    convection: str = "sbm"            # sbm, dca, kuo, mass_flux, edmf, none
    turbulence: str = "none"           # smagorinsky, louis, tke, none
    gravity_wave_drag: str = "none"    # rayleigh, lindzen, mcfarlane, none

    # Conservation
    fix_moisture: bool = False

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

    # Performance
    precision: str = "fp32"           # fp32, fp64, or mixed
    gradient_checkpoint: bool = False  # wrap scan body with jax.checkpoint for AD

    # Distributed
    distributed: bool = False
    ensemble_size: int = 1

    def validate_strict(self) -> None:
        """Raise ValueError for invalid parameter values.

        Called before simulation start to catch configuration errors
        early, before JIT compilation or data loading.
        """
        g = self.grid
        d = self.dycore
        errors: list[str] = []
        if g.resolution <= 0:
            errors.append(f"grid.resolution must be > 0, got {g.resolution}")
        if g.nlev <= 0:
            errors.append(f"grid.nlev must be > 0, got {g.nlev}")
        if g.p_top_Pa <= 0:
            errors.append(f"grid.p_top_Pa must be > 0, got {g.p_top_Pa}")
        if d.dt <= 0:
            errors.append(f"dycore.dt must be > 0, got {d.dt}")
        if d.hyperdiff_scale < 0:
            errors.append(f"dycore.hyperdiff_scale must be >= 0, got {d.hyperdiff_scale}")
        if d.div_damp_scale < 0:
            errors.append(f"dycore.div_damp_scale must be >= 0, got {d.div_damp_scale}")
        if self.precision not in ("fp32", "fp64", "mixed"):
            errors.append(
                f"precision must be 'fp32', 'fp64', or 'mixed', got {self.precision!r}"
            )
        if self.days <= 0:
            errors.append(f"days must be > 0, got {self.days}")
        # Reject unsupported coupled/ESM modes with actionable errors.
        if self.carbon_cycle != "none":
            errors.append(
                f"carbon_cycle={self.carbon_cycle!r} is not implemented. "
                f"ModelDriver is atmosphere-only with prescribed SST/SIC. "
                f"Set carbon_cycle='none' or use a coupled driver."
            )

        if errors:
            raise ValueError(
                "Invalid ExperimentConfig:\n  " + "\n  ".join(errors)
            )

    def validate(self) -> list[str]:
        """Check for suspicious parameter combinations.

        Returns a list of warning strings (empty if config is clean).
        """
        warns: list[str] = []
        if self.radiation == "gray" and self.aerosol_forcing != "off":
            warns.append("aerosol_forcing is ignored with gray radiation")
        if self.microphysics != "none" and self.radiation == "gray":
            warns.append(
                "microphysics without RRTMGP radiation may give unrealistic results"
            )
        if self.dycore.dt > 900 and self.grid.resolution >= 32:
            warns.append(
                f"dt={self.dycore.dt}s may violate CFL at C{self.grid.resolution}"
            )
        if self.cloud_scheme != "none" and self.radiation == "gray":
            warns.append(
                "cloud_scheme is ignored with gray radiation; "
                "set radiation='rrtmgp' for cloud-radiation coupling"
            )
        if self.fix_moisture and self.microphysics != "none":
            warns.append(
                "fix_moisture with active microphysics may conflict "
                "with microphysical moisture sources/sinks"
            )
        return warns

    # ------------------------------------------------------------------
    # Legacy AMIP adapter (serialization boundary only)
    # ------------------------------------------------------------------

    @staticmethod
    def from_amip_config(amip_cfg) -> ExperimentConfig:
        """Convert AMIPExperimentConfig to ExperimentConfig.

        Used at deserialization boundaries (checkpoint load, legacy
        experiment factory) — not in core runtime paths.
        """
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
            output_dir=getattr(amip_cfg, 'output_dir', ''),
            diag_days=amip_cfg.diag_days,
            checkpoint_days=amip_cfg.checkpoint_days,
            monthly_means=getattr(amip_cfg, 'monthly_means', False),
            cmip_output=getattr(amip_cfg, 'cmip_output', False),
            clear_sky_diag=getattr(amip_cfg, 'clear_sky_diag', False),
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
            ozone_forcing=getattr(amip_cfg, 'ozone_forcing', 'inline'),
            ozone_file=getattr(amip_cfg, 'ozone_file', ''),
            solar_source=getattr(amip_cfg, 'solar_source', 'constant'),
            solar_file=getattr(amip_cfg, 'solar_file', ''),
            solar_spectral_var=getattr(amip_cfg, 'solar_spectral_var', 'solar_fraction_by_gpt'),
            aerosol_forcing=getattr(amip_cfg, 'aerosol_forcing', 'off'),
            aerosol_file=getattr(amip_cfg, 'aerosol_file', ''),
            aerosol_reference_aod=getattr(amip_cfg, 'aerosol_reference_aod', 0.03),
            volcanic_aerosol_file=getattr(amip_cfg, 'volcanic_aerosol_file', ''),
            volcanic_aerosol_scale=getattr(amip_cfg, 'volcanic_aerosol_scale', 1.0),
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
        """Convert to AMIPExperimentConfig for backward-compatible serialization.

        Used at serialization boundaries (checkpoint save, legacy config
        export) — not in core runtime paths.
        """
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
            ozone_forcing=self.ozone_forcing,
            ozone_file=self.ozone_file,
            solar_source=self.solar_source,
            solar_file=self.solar_file,
            solar_spectral_var=self.solar_spectral_var,
            aerosol_forcing=self.aerosol_forcing,
            aerosol_file=self.aerosol_file,
            aerosol_reference_aod=self.aerosol_reference_aod,
            volcanic_aerosol_file=self.volcanic_aerosol_file,
            volcanic_aerosol_scale=self.volcanic_aerosol_scale,
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


# ======================================================================
# Native JSON serialization for ExperimentConfig
# ======================================================================

_SUB_CONFIGS = {
    "grid": GridConfig,
    "dycore": DycoreConfig,
    "output": OutputConfig,
}


def experiment_config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize ExperimentConfig to a JSON-safe dict.

    Sub-configs (grid, dycore, output) are inlined as nested dicts.
    This is the canonical serialization format.
    """
    d = config._asdict()
    for key in _SUB_CONFIGS:
        sub = d[key]
        if hasattr(sub, '_asdict'):
            d[key] = sub._asdict()
    return d


def experiment_config_from_dict(d: dict) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from a dict (e.g. loaded from JSON).

    Unknown fields are silently dropped for forward-compatibility
    (so older checkpoints with removed fields still load).
    """
    # Reconstruct sub-configs
    sub_values = {}
    for key, cls in _SUB_CONFIGS.items():
        if key in d and isinstance(d[key], dict):
            known_sub = set(cls._fields)
            filtered = {k: v for k, v in d[key].items() if k in known_sub}
            sub_values[key] = cls(**filtered)

    # Filter top-level fields
    known = set(ExperimentConfig._fields)
    filtered = {k: v for k, v in d.items() if k in known}
    filtered.update(sub_values)

    return ExperimentConfig(**filtered)


def save_experiment_config(config: ExperimentConfig, path: Path | str) -> None:
    """Save ExperimentConfig to JSON file."""
    with open(path, "w") as f:
        json.dump(experiment_config_to_dict(config), f, indent=2, default=str)


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load ExperimentConfig from JSON file."""
    with open(path) as f:
        return experiment_config_from_dict(json.load(f))
