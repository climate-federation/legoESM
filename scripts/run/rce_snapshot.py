"""Snapshot saver for the RCEMIP plane-CRM run (surface + 4-level fields, precip,
and full-3D condensate + moist-static-energy dumps for 3D visualization).

Kept OUT of the faithfulness-critical ``run_rcemip_plane.py`` driver: the driver
only adds two flags + a few-line loop hook that calls :func:`save_surface_levels`
(cheap, frequent) and :func:`save_3d` (heavy, only at the requested viz days).
All physics diagnostics come from the shared
``legoesm.atmosphere.dynamics.crm.rce_diagnostics`` module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
    column_water_vapor_plane,
    moist_static_energy_3d_plane,
    precipitation_rate_proxy_plane,
    temperature_3d_plane,
)

# default 4 levels [m]: sub-cloud, freezing level, upper troposphere, anvil/outflow
_DEFAULT_HEIGHTS_M = (1000.0, 5000.0, 9000.0, 12000.0)


def _condensate_slots(n_tracers):
    cloud = [s for s in (1, 3) if s < n_tracers]          # q_c, q_i (suspended)
    precip = [s for s in (2, 4, 5) if s < n_tracers]      # q_r, q_s, q_g
    return cloud, precip


def _level_indices(z_full, targets_m):
    z = np.asarray(z_full)
    idx = [int(np.argmin(np.abs(z - t))) for t in targets_m]
    return np.array(idx, dtype=int), z[np.array(idx, dtype=int)]


def _fields(state, grid, hc, qv_slot=0):
    tr = np.asarray(state.tracers.data)                   # (ny,nx,nlev,ntr)
    ntr = tr.shape[-1]
    cloud, precip = _condensate_slots(ntr)
    qcloud = sum((tr[..., s] for s in cloud), np.zeros_like(tr[..., 0]))
    qprecip = sum((tr[..., s] for s in precip), np.zeros_like(tr[..., 0]))
    cond = qcloud + qprecip                               # total condensate [kg/kg]
    qv = tr[..., qv_slot]
    w_half = np.asarray(state.w.data)
    w = 0.5 * (w_half[..., :-1] + w_half[..., 1:])         # full-level w
    u = np.asarray(state.u.data); v = np.asarray(state.v.data)  # horizontal wind
    mse = np.asarray(moist_static_energy_3d_plane(state, hc)) / 1.0e3   # kJ/kg
    T = np.asarray(temperature_3d_plane(state, hc))
    cwv = np.asarray(column_water_vapor_plane(state, hc))               # kg/m²
    precip_rate = np.asarray(
        precipitation_rate_proxy_plane(state, hc)) * 86400.0           # mm/day
    z = np.asarray(hc.z_full)
    return dict(cond=cond, qcloud=qcloud, qv=qv, w=w, u=u, v=v, mse=mse, T=T,
                cwv=cwv, precip=precip_rate, z=z)


def save_surface_levels(out_dir, step, t_s, state, grid, hc,
                        heights_m=_DEFAULT_HEIGHTS_M, qv_slot=0,
                        surface_fields=None):
    """Cheap, frequent dump: surface fields (precip, CWV, column-max |w|, sub-grid
    near-surface qv) + horizontal cross-sections at four heights of condensate, w,
    qv, MSE. One npz per call under ``<out>/snapshots/``."""
    out_dir = Path(out_dir) / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = _fields(state, grid, hc, qv_slot)
    hidx, hz = _level_indices(f["z"], heights_m)
    cond, w, qv, mse = f["cond"], f["w"], f["qv"], f["mse"]
    u, v, T = f["u"], f["v"], f["T"]
    k_sfc = int(np.argmin(f["z"]))
    day = t_s / 86400.0
    payload = dict(
        day=day, t_s=t_s, dx=grid.dx, Lx=grid.nx * grid.dx, Ly=grid.ny * grid.dx,
        heights=hz,
        precip=f["precip"], cwv=f["cwv"],
        wcolmax=np.max(np.abs(w), axis=2),
        qv_sfc=qv[:, :, k_sfc],
        # horizontal wind (speed = hypot(u,v)) — TC tangential circulation — and
        # temperature cross-sections, at the surface + the requested heights.
        u_sfc=u[:, :, k_sfc], v_sfc=v[:, :, k_sfc],
        u_levels=np.stack([u[:, :, k] for k in hidx]),
        v_levels=np.stack([v[:, :, k] for k in hidx]),
        T_levels=np.stack([T[:, :, k] for k in hidx]),
        cond_levels=np.stack([cond[:, :, k] for k in hidx]),
        w_levels=np.stack([w[:, :, k] for k in hidx]),
        qv_levels=np.stack([qv[:, :, k] for k in hidx]),
        mse_levels=np.stack([mse[:, :, k] for k in hidx]),
    )
    if surface_fields:
        for name, value in surface_fields.items():
            payload[name] = np.asarray(value)
    np.savez(out_dir / f"sfc_{step:08d}.npz", **payload)


def save_3d(out_dir, step, t_s, state, grid, hc, qv_slot=0):
    """Heavy dump (only at the requested viz days): full 3D condensate + MSE (+ w,
    T) for the 3D rendering. One npz per call under ``<out>/snapshots3d/``."""
    out_dir = Path(out_dir) / "snapshots3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = _fields(state, grid, hc, qv_slot)
    np.savez(
        out_dir / f"vol_{step:08d}.npz",
        day=t_s / 86400.0, t_s=t_s, dx=grid.dx,
        Lx=grid.nx * grid.dx, Ly=grid.ny * grid.dx, z=f["z"],
        cond=f["cond"].astype(np.float32), qcloud=f["qcloud"].astype(np.float32),
        mse=f["mse"].astype(np.float32), w=f["w"].astype(np.float32),
        T=f["T"].astype(np.float32),
    )
