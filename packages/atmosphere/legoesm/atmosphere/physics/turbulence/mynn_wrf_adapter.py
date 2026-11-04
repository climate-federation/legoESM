"""Host adapter: ``turbulence: mynn25`` -> the WRF v4.8.0 MYNN-EDMF port.

Maps a lane's TOP-DOWN column state onto :func:`mynnedmf_wrf.mynnedmf_column`
(BOTTOM-UP, specific humidity) and its outputs back (votes M1-M11, N1-N14 in
``oracle_audit/vote_1103_r3``):

* surface ``ust``/``hfx``/``qfx`` and the wind speed ``wspd`` come from ONE
  call of the lane's bulk surface layer (``surface_fluxes_at_lowest_level``
  with ``config.surface``; ``wspd`` is the speed that call used, M4/M11);
  ``qfx`` is its surface moisture flux (positive up), ``ts = T_sfc``,
  ``ps`` = the surface interface pressure;
* ``exner = (p/p_ref)^kappa``, ``th = T/exner``, ``rho`` the moist density,
  ``dz`` the lane's moist hydrostatic layer thickness (N5);
* ``dx = sqrt(cell area)`` (M5), ``xland = 1`` where land fraction >= 0.5
  else 2, all water without a land fraction (M6/N14);
* ``sqc = q_c``; ``sqi = q_i`` or zeros (M10); ``dT/dt = dth/dt * exner``
  (M9);
* the carry is ONE array ``(ncol, 3, nlev)`` = (qke, qsq, el), top-down
  (N1); the cold-start step passes ``initflag=True`` (the oracle's
  mym_initialize), every later step ``False`` -- a STATIC Python bool the
  caller must give (M3/N4).

The port's hard switches are kept (user-approved exception, see the port's
docstring).  Lanes route ``dq_c_dt`` / ``dq_i_dt`` to the condensate tracers
(N6); a lane without ice gets ``dq_i_dt = None`` because the port's ice
tendency is exactly zero for zero ice.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics._shared import exner_function
from legoesm.atmosphere.physics.turbulence.mynnedmf_wrf import mynnedmf_column
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    surface_fluxes_at_lowest_level,
    surface_moisture_flux,
)

__physics_contract__ = {
    "summary": (
        "Host adapter for the WRF v4.8.0 MYNN-EDMF column port: top-down lane "
        "columns + one bulk surface-layer exchange in, port tendencies, "
        "diffusivities, PBL height and the (qke, qsq, el) carry out."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg (specific)",
        "carry": "(qke m^2/s^2, qsq kg^2/kg^2, el m)", "p_full": "Pa",
        "p_half": "Pa", "z_full": "m", "z_half": "m", "T_sfc": "K",
        "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s", "w": "m/s",
        "q_c": "kg/kg", "q_i": "kg/kg", "cell_area": "m^2", "land_frac": "1",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s",
        "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "evap_sfc": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; lane arrays top-down (index -1 lowest level), port arrays "
        "bottom-up; shflx, lhflx, evap_sfc > 0 upward; tendencies d/dt of "
        "the column state."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "see mynnedmf_wrf (WRF v4.8.0 module_bl_mynnedmf.F90)",
    "idealized_test": (
        "tests/unit/test_mynn_adapter.py: adapter reproduces the WRF oracle "
        "fixture rebuilt in host layout (fp64, rtol 1e-10)."
    ),
}

# WRF's xland convention (module_bl_mynnedmf: 1 land, 2 water).
_XLAND_LAND = 1.0
_XLAND_WATER = 2.0
_LAND_THRESHOLD = 0.5  # land where the land fraction >= this (vote M6)

N_CARRY = 3  # (qke, qsq, el)


def _up(a):
    """Top-down (ncol, nlev) <-> bottom-up."""
    return a[:, ::-1]


def mynn_wrf_columns(dz, u, v, w, T, p, exner, rho, q_v, q_c, q_i, carry,
                     xland, ps, ts, ust, hfx, qfx, wspd, dt, dx, *, initflag):
    """Call the port on TOP-DOWN host columns; returns ``(MYNNOutput mapped
    top-down as a dict, new carry)``.  ``q_i`` None = no ice (zeros in, the
    ice tendency is dropped).  ``carry`` is ``(ncol, 3, nlev)`` top-down."""
    if not isinstance(initflag, bool):
        raise TypeError(
            "mynn25 needs a static Python bool initflag (the cold-start step "
            f"is True, every later step False); got {initflag!r}")
    ncol, nlev = T.shape
    if tuple(carry.shape) != (ncol, N_CARRY, nlev):
        raise ValueError(
            f"mynn25 carry must be (ncol, 3, nlev) = {(ncol, N_CARRY, nlev)} "
            f"(qke, qsq, el), got {tuple(carry.shape)}; a state from the "
            "removed MYNN-2.5 surrogate (ncol, nlev) qke cannot be restored")
    sqi = jnp.zeros_like(q_v) if q_i is None else q_i
    th = T / exner
    out = mynnedmf_column(
        _up(dz), _up(u), _up(v), _up(w), _up(th), _up(T), _up(p), _up(exner),
        _up(rho), _up(q_v), _up(q_c), _up(sqi), _up(carry[:, 0]),
        _up(carry[:, 1]), _up(carry[:, 2]), xland, ps, ts, ust, hfx, qfx,
        wspd, dt, dx, initflag=initflag)
    mapped = {
        "du": _up(out.du), "dv": _up(out.dv),
        "dT": _up(out.dth) * exner, "dqv": _up(out.dqv), "dqc": _up(out.dqc),
        "dqi": None if q_i is None else _up(out.dqi),
        "km": _up(out.km), "kh": _up(out.kh), "pblh": out.pblh,
    }
    new_carry = jnp.stack([_up(out.qke), _up(out.qsq), _up(out.el)], axis=1)
    return mapped, new_carry


def mynn_wrf_turbulence(u, v, T, q_v, qke, p_full, p_half, z_full, z_half,
                        T_sfc, q_sfc, rho, dt, config, *, w, q_c, q_i,
                        cell_area, land_frac, initflag):
    """Lane kernel for ``turbulence: mynn25`` (signature of the other
    carry-holding kernels plus the port's extra inputs, all keyword-only and
    required: ``w`` [m/s, up], ``q_c``, ``q_i`` (None = no ice),
    ``cell_area`` [m^2, (ncol,)], ``land_frac`` ((ncol,) or None = water),
    ``initflag`` (static bool)).  ``qke`` is the packed (qke, qsq, el)
    carry under the slot's name (the pipeline passes it by that keyword).
    Returns ``(TurbulenceOutput, carry_new)``.
    """
    for name, val in (("w", w), ("q_c", q_c), ("cell_area", cell_area)):
        if val is None:
            raise ValueError(
                f"turbulence='mynn25' (WRF MYNN-EDMF) needs {name}; this lane "
                "supplies none")
    tau_x, tau_y, shflx, lhflx, ust, wspd = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1], T_sfc, q_sfc, rho[:, -1],
        config.surface, z_full[:, -1] - z_half[:, -1], return_wind_speed=True)
    qfx = surface_moisture_flux(config.surface, lhflx, T_sfc)
    exner = exner_function(p_full)
    dz = z_half[:, :-1] - z_half[:, 1:]
    xland = (jnp.full(T_sfc.shape, _XLAND_WATER, T.dtype) if land_frac is None
             else jnp.where(jnp.reshape(land_frac, T_sfc.shape)
                            >= _LAND_THRESHOLD, _XLAND_LAND, _XLAND_WATER
                            ).astype(T.dtype))
    dx = jnp.sqrt(jnp.reshape(cell_area, T_sfc.shape))
    m, carry_new = mynn_wrf_columns(
        dz, u, v, w, T, p_full, exner, rho, q_v, q_c, q_i, qke, xland,
        p_half[:, -1], T_sfc, ust, shflx, qfx, wspd, dt, dx,
        initflag=initflag)
    out = TurbulenceOutput(
        du_dt=m["du"], dv_dt=m["dv"], dT_dt=m["dT"], dq_v_dt=m["dqv"],
        Km=m["km"], Kh=m["kh"], shflx=shflx, lhflx=lhflx, ustar=ust,
        h_pbl=m["pblh"], dq_c_dt=m["dqc"], dq_i_dt=m["dqi"], evap_sfc=qfx)
    return out, carry_new


def cold_start_carry(ncol, nlev, dtype):
    """Carry before the cold-start (initflag) step: zeros, which the port
    overwrites (vote N3)."""
    return jnp.zeros((ncol, N_CARRY, nlev), dtype=dtype)
