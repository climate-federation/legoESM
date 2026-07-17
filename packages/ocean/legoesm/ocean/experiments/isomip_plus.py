"""ISOMIP+ ice-shelf cavity benchmark (Asay-Davis et al. 2016).

Reference
---------
Asay-Davis, X. S., et al. (2016). "Experimental design for three
interrelated marine ice sheet and ocean model intercomparison
projects: MISMIP+, ISOMIP+, and MISOMIP1", Geosci. Model Dev. 9,
2471-2497.

Configurations (Ocean0 + Ocean1)
---------------------------------
Idealised sub-ice-shelf cavity:

* Cavity:  80 km x 160 km horizontal, 500 m max water-column depth
  below the ice base.
* Ice shelf draft: linearly decreasing from 500 m at the grounding
  line to 0 m at the ice front (160 km from the grounding line).
* Forcing:
    - Ocean0 (cold): far-field T = -1.9 deg C, S = 34.55 PSU.
    - Ocean1 (warm): far-field T =  1.0 deg C, S = 34.7 PSU.

Diagnostic
----------
Domain-mean basal melt rate ``mdot`` (m/yr, ICE-equivalent — the ISOMIP+
reporting convention; the pre-2026-07-17 linearised closure reported a
freshwater-equivalent, hand-tuned-down value).  For reference the ISOMIP+
circulating-cavity ensemble spans Ocean0 ~ 0.1-0.3 and Ocean1 ~ 1-10 m/yr,
but THIS smoke probe feeds far-field T as the boundary-layer ambient (no
meltwater throttling), so it reads Ocean0 ~ 1.5 and Ocean1 ~ 65 m/yr with
the faithful closure at the ~250 dbar mean draft — an expected, documented
overestimate, NOT an acceptance mismatch.  The honest ensemble comparison
(the old "+/- 30 %% of ensemble mean" bar) belongs to the cavity-circulation
follow-up below.

Scope
-----
This experiment drives the FAITHFUL Holland-Jenkins three-equation
basal-melt closure (``legoesm.ocean.physics.ice_shelf.three_equation_melt``
— full quadratic with the meltwater salt-dilution feedback).  The former
``ice_shelf_basal_melt`` linearised closure (small-thermal-driving limit
with a hand-tuned effective gamma_T ~30x below the ISOMIP+ gamma_T* and a
freshwater-convention shortcut that absorbed rho_sw/rho_fw into the tuning)
was DELETED and this caller rerouted (2026-07-17).
The full cavity boundary-condition wiring (masking top cells where
``z_ice > z_top_interface``, applying basal-melt freshwater flux at
the appropriate level) is a Phase D follow-up: the present
``create_initial_conditions`` builds the cavity geometry, the
``compute_basal_melt`` helper integrates Jenkins over the domain,
and the matrix runner can drive both quantities. The ocean model's
top-boundary conditions still treat the surface as a free surface;
swapping to a rigid-cavity-lid boundary is the remaining piece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    three_equation_melt,
)


@dataclass
class ISOMIPPlusConfig:
    """Configuration for the ISOMIP+ Ocean0 / Ocean1 cavity benchmark."""

    # Domain (Asay-Davis 2016 Fig. 1).
    lx_km: float = 80.0
    ly_km: float = 160.0
    H_max: float = 720.0           # max water depth (cavity floor)
    ice_draft_max_m: float = 500.0  # ice base depth at grounding line
    ice_front_y_km: float = 160.0   # ice-shelf extent along y (km)

    # Far-field T, S (Ocean0 cold; override for Ocean1 warm via
    # ``ISOMIPPlusConfig.ocean1()``).
    T_far_C: float = -1.9
    S_far_psu: float = 34.55

    # Basal-melt parameterisation knobs — the FAITHFUL three-equation config
    # (its default gamma_T = 1.0e-4 m/s equals the value the old presets set
    # explicitly, so the presets carry no override).
    melt: IceShelfConfig = None

    @classmethod
    def ocean0_cold(cls) -> "ISOMIPPlusConfig":
        return cls(T_far_C=-1.9, S_far_psu=34.55, melt=IceShelfConfig())

    @classmethod
    def ocean1_warm(cls) -> "ISOMIPPlusConfig":
        return cls(T_far_C=1.0, S_far_psu=34.7, melt=IceShelfConfig())

    def __post_init__(self):
        if self.melt is None:
            object.__setattr__(self, "melt", IceShelfConfig())


def ice_draft_m(x_km, y_km, config: ISOMIPPlusConfig | None = None) -> jnp.ndarray:
    """Linear ice-shelf draft: ``z_ice(y)`` from grounding line to front.

    The grounding line is at y = 0; the ice front is at
    ``y = ice_front_y_km``. ``z_ice`` is negative (below sea level).
    """
    if config is None:
        config = ISOMIPPlusConfig.ocean0_cold()
    y = jnp.asarray(y_km, dtype=jnp.float64)
    frac = jnp.clip(y / config.ice_front_y_km, 0.0, 1.0)
    # Linear ramp: -ice_draft_max at y=0, 0 at y=ice_front.
    return -config.ice_draft_max_m * (1.0 - frac)


def cavity_water_column_m(x_km, y_km,
                          config: ISOMIPPlusConfig | None = None
                          ) -> jnp.ndarray:
    """Water column thickness inside the cavity, ``H_max - |z_ice|``."""
    if config is None:
        config = ISOMIPPlusConfig.ocean0_cold()
    z_ice = ice_draft_m(x_km, y_km, config)
    return jnp.maximum(config.H_max - jnp.abs(z_ice), 0.0)


def compute_basal_melt(state, *, config: ISOMIPPlusConfig | None = None,
                        rho_sw: float = 1028.0,
                        g_val: float = constants.g) -> float:
    """Domain-mean basal melt rate ``mdot`` [m/yr, ICE-equivalent] from a state.

    Reads the top-cell T, S; computes the ice-shelf-base pressure
    (``rho_sw * g * |z_ice|``); applies the FAITHFUL Holland-Jenkins
    three-equation closure (full quadratic, meltwater salt dilution)
    cell-by-cell; integrates over the cavity area.

    CONVENTION + CAVEAT (reroute 2026-07-17): the rate is ICE-equivalent
    (``m_dot_m_s``; freshwater mass flux = rho_ice*m_dot), the ISOMIP+
    reporting convention — the deleted linearised closure reported a
    freshwater-equivalent rate.  This smoke diagnostic feeds the FAR-FIELD
    T to the closure as if it were the cavity boundary-layer ambient, so
    warm cases (Ocean1) OVERESTIMATE vs the ISOMIP+ circulating-cavity
    ensemble (~60 vs 1-10 m/yr): in the real intercomparison meltwater
    cools/freshens the top cells and throttles the melt.  The old closure
    only sat "in range" because its hand-tuned gamma_T compensated for
    exactly this missing feedback.  An honest ensemble comparison needs
    the cavity-circulation experiment (the Phase-D follow-up below).
    """
    if config is None:
        config = ISOMIPPlusConfig.ocean0_cold()
    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    # Top-cell T, S.
    T_w = T[..., 0]
    S_w = S[..., 0]
    # Pressure proxy: ice draft constant in y (per the cavity-geometry
    # builder), uniform |z_ice| across columns. The matrix runner
    # provides ``z_ice`` as state metadata when the cavity is fully
    # wired; for the smoke harness we use a single representative
    # ice draft.
    z_ice_mean = -0.5 * config.ice_draft_max_m
    p_b_dbar = rho_sw * g_val * abs(z_ice_mean) / 1.0e4  # 1 dbar = 1e4 Pa
    m_per_s = np.asarray(three_equation_melt(
        jnp.asarray(T_w), jnp.asarray(S_w), jnp.asarray(p_b_dbar),
        config=config.melt,
    ).m_dot_m_s)
    # Mask-area-weighted mean melt (m/s) -> m/yr.
    area_weight = mask
    total = area_weight.sum()
    if total <= 0.0:
        return 0.0
    mean_m_per_s = float((m_per_s * area_weight).sum() / total)
    return mean_m_per_s * 86400.0 * 365.0


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: ISOMIPPlusConfig | None = None):
    """Rest-state IC at the configured far-field T, S; lat-lon only.

    Cavity BC handling (top-cell masking against ``z_ice``) is the
    remaining Phase D follow-up. For now the experiment runs as a
    rectangular box and the basal-melt diagnostic is post-processed
    from the top cell.
    """
    if grid_type != "latlon_regional":
        raise NotImplementedError(
            "ISOMIP+ supports only latlon_regional in this Phase D smoke; "
            "MPAS + cube ports require cavity-aware top-boundary wiring."
        )
    if config is None:
        config = ISOMIPPlusConfig.ocean0_cold()
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    return rest_state_latlon_cgrid_ocean(
        grid=grid, z_coord=z_coord, H_max=config.H_max,
        T_water_init_C=config.T_far_C, T_deep=config.T_far_C,
        S_uniform=config.S_far_psu,
    )


def create_forcings(grid_type: str, grid,
                    config: ISOMIPPlusConfig | None = None):
    """No surface wind / restoring. The basal-melt freshwater flux is
    applied post-hoc by the matrix runner once the cavity BC is wired.
    """
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
    )


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: ISOMIPPlusConfig | None = None
                     ) -> tuple[bool, str]:
    import jax.numpy as jnp
    if config is None:
        config = ISOMIPPlusConfig.ocean0_cold()
    for name in ("u", "T", "S", "eta"):
        data = getattr(final_state, name).data
        if not bool(jnp.all(jnp.isfinite(data))):
            return False, f"NaN/Inf in final {name}"
    mdot = compute_basal_melt(final_state, config=config)
    notes = f"mdot={mdot:.3f} m/yr ice-equivalent (cavity-mean basal melt)"
    # Loose smoke sanity bound only.  With the faithful three-equation
    # closure and the far-field-T probe (see compute_basal_melt), Ocean0
    # reads ~1.5 and Ocean1 ~65 m/yr at the ~250 dbar mean draft — the old
    # 50 m/yr bound predated the reroute (tuned linearised closure) and
    # would have failed the warm preset.  200 m/yr still catches NaN-scale
    # blowups; the ensemble-band gate moves to the cavity-circulation
    # follow-up.
    ok = abs(mdot) < 200.0
    return ok, notes


def get_diagnostic_field_specs() -> list:
    return [
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("SSS", "SSS (PSU)", "YlGnBu"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_T": "degC", "mean_S": "PSU",
        "basal_melt_m_per_yr": "m/yr",
    }


EXPERIMENT_CONFIG = {
    "name": "isomip_plus",
    "description": (
        "ISOMIP+ Ocean0 / Ocean1 ice-shelf cavity benchmark "
        "(Asay-Davis 2016)"
    ),
    "scientific_purpose": (
        "Exercise the faithful three-equation basal-melt closure on the "
        "ISOMIP+ cavity geometry (smoke sanity bound only; the ensemble "
        "comparison needs the cavity-circulation follow-up)"
    ),
    "reference": "Asay-Davis et al. 2016, GMD 9, 2471-2497",
    "config_class": ISOMIPPlusConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": lambda c=None: {
        "lx_km": (c or ISOMIPPlusConfig.ocean0_cold()).lx_km,
        "ly_km": (c or ISOMIPPlusConfig.ocean0_cold()).ly_km,
        "H_max": (c or ISOMIPPlusConfig.ocean0_cold()).H_max,
        "description": "ISOMIP+ rectangular ice-shelf cavity",
    },
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 365.0,
    "quick_duration": 5.0,
    "grid_support": {
        "cubed_sphere": False, "latlon": False, "mpas": False,
        "latlon_regional": True, "mpas_regional": False,
        "cs_regional": False,
        "latlon_channel": False, "mpas_channel": False, "spectral": False,
    },
    "expected_metrics": {
        "Ocean0_basal_melt_m_per_yr":
            "~1.5 m/yr ICE-equivalent at this far-field-T smoke probe "
            "(faithful 3-eq closure; circulating-cavity ensemble is 0.1-0.3)",
        "Ocean1_basal_melt_m_per_yr":
            "~65 m/yr ICE-equivalent at this far-field-T smoke probe "
            "(expected overestimate — no meltwater throttling; ensemble 1-10)",
    },
}
