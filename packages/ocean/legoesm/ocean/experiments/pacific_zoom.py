"""Regional Pacific zoom on the lat-lon C-grid: the settings of the ENSO hindcast
deck (Isma Di Carlo's run_hind9698.py and nest parent, ENSO fork).

Selects every option the zoom needs, each named explicitly: open lateral
boundaries (MOM6 1-D Orlanski on the normal velocity, tangential ring velocity
nudged only, CROCO's 1 / 360 day nudging), the step-centred trapezoidal
Coriolis with the energy-conserving operator, the 3-D momentum face mask, and
the OM4p5 lateral friction as sole friction.  The boundary and forcing data
(OpenBoundaryData arrays, the face stress, the sponge viscosity) are inputs of
the run, not settings, and are supplied by the caller.
"""

from __future__ import annotations

from legoesm.ocean.dynamics.open_boundary_latlon import OpenBoundaryData
from legoesm.ocean.state import LatLonCGridOceanConfig, OMp25Config

_DAY = 86400.0
OPEN_BOUNDARY_TAU_IN_S = 1.0 * _DAY       # nudging on inflow (isoCROCO)
OPEN_BOUNDARY_TAU_OUT_S = 360.0 * _DAY    # nudging on outflow (isoCROCO)

OM4P5 = OMp25Config(C2=0.15, Cu2=0.01, C4=0.06, Cu4=0.01, Kh_sin_lat=2000.0, Kh_sin_power=4.0,
                    bound_coriolis_vel=6.0, resolution_function="mom6_step_wkb", res_coef=1.0,
                    gill_equatorial_ld=True, bound_coef=0.8, smooth_width=0.01)


def pacific_zoom_config(base: LatLonCGridOceanConfig) -> LatLonCGridOceanConfig:
    """``base`` with the zoom's dynamics and friction selected."""
    return base._replace(
        pgf_scheme="smc03",
        momentum_face_mask="3d",
        vertical_momentum_scheme="nemo_advective",
        fix_eta_drift=False,
        coriolis_time_scheme="trapezoidal_step",
        coriolis_baroclinic_operator="energy_conserving",
        open_boundary_smooth_width=0.01,
        lateral_friction_scheme="om4p25",
        omp25=OM4P5,
        lateral_viscosity=base.lateral_viscosity._replace(
            A_h=0.0, B_h=0.0, C_smag=0.0, C_smag_lap=0.0, C_leith=0.0),
    )


def pacific_zoom_open_boundary(**data) -> OpenBoundaryData:
    """OpenBoundaryData with the zoom's radiation and nudging selected; ``data``
    holds the boundary arrays (``west_T``, ``south_vbar``, ...)."""
    return OpenBoundaryData(tau_in=OPEN_BOUNDARY_TAU_IN_S, tau_out=OPEN_BOUNDARY_TAU_OUT_S,
                            velocity_rx_max=1.0, tangential_nudge_only=True, **data)


__all__ = ["OM4P5", "OPEN_BOUNDARY_TAU_IN_S", "OPEN_BOUNDARY_TAU_OUT_S",
           "pacific_zoom_config", "pacific_zoom_open_boundary"]
