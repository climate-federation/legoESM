"""Held-Larichev eddying channel: APE -> eddy KE cascade saturation.

Reference
---------
Held, I. M., & Larichev, V. D. (1996). "A scaling theory for
horizontally homogeneous, baroclinically unstable flow on a beta
plane", J. Atmos. Sci. 53, 946-952.

What it tests
-------------
A re-entrant zonal channel with prescribed baroclinic shear --
similar geometry to Eady-uniform -- but stirred from rest by a
stochastic initial perturbation so the dominant baroclinic mode
saturates into an eddy field whose isotropic kinetic-energy
spectrum follows ``E(k) ~ k^-3`` at mesoscale wavenumbers.

The HL scaling predicts:

* Total eddy KE balances Available Potential Energy generation
  ``epsilon_APE = -<v' b'> * dU/dy * H`` against bottom-drag
  + viscous dissipation.
* Spectral slope ``-3`` in the inverse-cascade range (between the
  injection scale ~ L_d and the bottom-drag-controlled arrest
  scale L_eddy = (U_*) / beta).

Configuration
-------------
Reuses the ``EadyUniformConfig`` channel geometry (10 deg wide,
18 deg tall, H = 5500 m) but:

* Perturbation amplitude bumped to ``0.5 K`` (vs 0.1 K in Eady) and
  applied as multi-wavenumber noise so multiple baroclinic modes
  excite simultaneously and the cascade can fill up over many
  e-folding times.
* Bottom drag set to ``1e-5 s^-1`` -- standard HL value that gives
  an arrest scale L_eddy ~ 200 km consistent with the eddy field
  that the ``k^-3`` slope describes.

Acceptance
----------
* Domain-mean eddy KE saturates (not growing exponentially after
  ~ 30 e-folding times).
* Isotropic ``E(k)`` slope between ``L_d`` and the Nyquist falls in
  ``-3 +/- 0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    create_initial_conditions as _eu_create_ic,
    create_forcings as _eu_create_forcings,
)


@dataclass
class HeldLarichevConfig:
    """Configuration for the Held-Larichev eddying-channel benchmark.

    All non-overridden fields fall back to ``EadyUniformConfig``
    defaults via ``_to_eady_cfg``.
    """

    # Channel geometry (matches EadyUniformConfig).
    H_max: float = 5500.0
    lat_south: float = 16.0
    lat_north: float = 34.0
    lat_center: float = 25.0
    lon_west: float = 0.0
    lon_east: float = 10.0

    # Stratification + shear (matches EadyUniformConfig).
    N: float = 1.2e-3
    T_ref_C: float = 10.0          # reference temperature [degC]
    S_uniform: float = 35.0
    alpha_T: float = 2.0e-4
    rho_0: float = 1024.0
    U_surface: float = 0.8
    jet_width_deg: float = 5.0
    jet_depth_scale: float = 5500.0

    # Bigger perturbation seed than Eady so multiple modes excite.
    T_perturbation_K: float = 0.5
    perturbation_wavenumber: int = 3

    # Physics: HL canonical bottom drag value.
    A_h: float = 5000.0
    B_h: float = 0.0
    C_smag: float = 0.0
    K_h: float = 0.0
    K_bih: float = 0.0
    A_v: float = 1.0e-5
    K_v: float = 1.0e-5
    bottom_drag_coeff: float = 1.0e-5
    tracer_advection: str = "tvd"
    # Free-surface Laplacian: OFF. Inherited from the collocated/cubed-sphere
    # solvers (0.05 is the CUBED-SPHERE default, raised there for a
    # face-boundary feedback); this is a C-grid case, which has no such
    # checkerboard mode. At its resolution the setting implied a diffusivity
    # of order 1e6 m^2/s against 1e2-1e3 for the real ocean, and it moved far
    # more water than the flow itself.
    #
    # Measured, one variable at a time: with it OFF this case PASSES on both
    # its grids, and it also passes with the derived velocity viscosity added,
    # so no replacement is required -- unlike eady_uniform, which needed one.
    #
    # THIS CHANGES A SCIENCE NUMBER, not just stability: peak current on
    # the unstructured grid goes 1.099 -> 0.726 m/s (-34%). The OLD value
    # is the contaminated one -- it was produced with the spurious
    # diffusivity active -- but the case's acceptance band admits both, so
    # the move would otherwise have been silent. The lat-lon arm is
    # unchanged at 0.648 because that path never receives this setting.
    barotropic_diffusion_alpha: float = 0.0
    barotropic_div_damp: float = 0.05


def _to_eady_cfg(cfg: HeldLarichevConfig) -> EadyUniformConfig:
    """Build the underlying ``EadyUniformConfig`` (HL re-uses Eady IC)."""
    return EadyUniformConfig(
        H_max=cfg.H_max,
        lat_south=cfg.lat_south, lat_north=cfg.lat_north,
        lat_center=cfg.lat_center,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        N=cfg.N, T_ref_C=cfg.T_ref_C, S_uniform=cfg.S_uniform,
        alpha_T=cfg.alpha_T, rho_0=cfg.rho_0,
        U_surface=cfg.U_surface, jet_width_deg=cfg.jet_width_deg,
        jet_depth_scale=cfg.jet_depth_scale,
        T_perturbation_K=cfg.T_perturbation_K,
        perturbation_wavenumber=cfg.perturbation_wavenumber,
        A_h=cfg.A_h, B_h=cfg.B_h, C_smag=cfg.C_smag,
        K_h=cfg.K_h, K_bih=cfg.K_bih,
        A_v=cfg.A_v, K_v=cfg.K_v,
        bottom_drag_coeff=cfg.bottom_drag_coeff,
        tracer_advection=cfg.tracer_advection,
        barotropic_diffusion_alpha=cfg.barotropic_diffusion_alpha,
        barotropic_div_damp=cfg.barotropic_div_damp,
    )


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: HeldLarichevConfig | None = None):
    """HL initial state via the Eady-uniform builder (same channel IC)."""
    if config is None:
        config = HeldLarichevConfig()
    return _eu_create_ic(grid_type, grid, z_coord, _to_eady_cfg(config))


def create_forcings(grid_type: str, grid,
                    config: HeldLarichevConfig | None = None):
    """No surface forcing -- HL eddies feed off the baroclinic shear IC."""
    if config is None:
        config = HeldLarichevConfig()
    return _eu_create_forcings(grid_type, grid, _to_eady_cfg(config))


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: HeldLarichevConfig | None = None) -> tuple[bool, str]:
    """Smoke-grade validation. Full ``k^-3`` slope fit lives in a
    fidelity-tier follow-up that hooks
    :func:`legoesm.ocean.diagnostics.isotropic_energy_spectrum`."""
    import jax.numpy as jnp
    if config is None:
        config = HeldLarichevConfig()
    for name in ("u", "T", "S", "eta"):
        data = getattr(final_state, name).data
        if not bool(jnp.all(jnp.isfinite(data))):
            return False, f"NaN/Inf in final {name}"
    max_speed = diagnostics.get("max_speed", [0.0])[-1] if diagnostics.get(
        "max_speed") else 0.0
    notes = f"|u|max={max_speed:.3f}m/s (HL eddying-channel saturation)"
    ok = max_speed > 0.05 and max_speed < 10.0
    return ok, notes


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "magma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m", "max_speed": "m/s",
        "max_abs_u": "m/s", "mean_T": "degC",
    }


EXPERIMENT_CONFIG = {
    "name": "held_larichev",
    "description": "Held-Larichev eddying channel: APE->eddy KE cascade",
    "scientific_purpose": (
        "Eddy saturation + isotropic kinetic-energy spectrum E(k) ~ k^-3"
    ),
    "reference": "Held & Larichev (1996), J. Atmos. Sci. 53, 946-952",
    "config_class": HeldLarichevConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": lambda c=None: {
        "H_max": (c or HeldLarichevConfig()).H_max,
        "description": "Eady-uniform channel reused for HL saturation",
    },
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 200.0,
    "quick_duration": 30.0,
    "grid_support": {
        "cubed_sphere": False, "latlon": False, "mpas": False,
        "latlon_regional": False, "mpas_regional": False,
        "latlon_channel": True, "mpas_channel": True,
        "cs_regional": False, "spectral": False,
    },
    "expected_metrics": {
        "eddy_KE_saturation":
            "domain-mean eddy KE plateaus after ~30 e-folding times",
        "isotropic_spectrum_slope":
            "E(k) ~ k^-3 with slope -3 +/- 0.5 between L_d and Nyquist",
    },
}
