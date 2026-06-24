"""Geostrophic-adjustment comparison: legoESM vs Oceananigans (free-surface COUPLING).

A DETERMINISTIC test of the free-surface predictor-corrector coupling + Coriolis +
time-stepper -- the node the Phase-1 tendency match pointed the §5 residual at (the
momentum tendency itself is faithful). Both codes start from the IDENTICAL
unbalanced free-surface bump (eta Gaussian, u=v=0) and adjust toward geostrophic
balance; eta/u/v are compared over the (non-chaotic) adjustment.

Reference: scripts/data/generate_oceananigans_geostrophic_adjustment_reference.jl.

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref \
      JAX_ENABLE_X64=1 .venv/bin/python \
      scripts/validate/ocean_fidelity/compare_oceananigans_geostrophic_adjustment.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

_spec = importlib.util.spec_from_file_location(
    "bk", os.path.join(os.path.dirname(__file__), "compare_oceananigans_bickley_jet.py"))
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

ETA0, SIG = 0.1, 30.0   # bump amplitude + width (deg), matching the deck
DT = 0.01


def _bump(lon_deg, lat_deg):
    return ETA0 * np.exp(-((lon_deg - 0.0) ** 2 + (lat_deg - 0.0) ** 2) / (2 * SIG ** 2))


def _align(lego, ocn):
    """(lat,lon) align: search lat-offset + lon-roll, validated per-field."""
    L, O = np.asarray(lego), np.asarray(ocn)
    nlat, nlon = O.shape
    best = (-2.0, 0, 0)
    for la0 in range(0, max(1, L.shape[0] - nlat + 1)):
        for roll in range(-2, 3):
            cand = np.roll(L, roll, axis=1)[la0:la0 + nlat, :nlon]
            if cand.shape != O.shape or cand.std() < 1e-12 or O.std() < 1e-12:
                continue
            c = float(np.corrcoef(cand.ravel(), O.ravel())[0, 1])
            if c > best[0]:
                best = (c, la0, roll)
    return best


def main():
    ref = os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"]
    ds = Dataset(os.path.join(ref, "geostrophic_adjustment", "geostrophic_adjustment.nc"))
    iters = np.asarray(ds.variables["iters"][:]).astype(int)
    # Julia column-major -> Python reverses dims: stored (t,x,y) reads as (lat,lon,t).
    o_eta = np.asarray(ds.variables["eta"][:])   # (lat, lon, t)
    o_u = np.asarray(ds.variables["u"][:])        # (lat, lon, t)
    nsteps = int(iters.max())
    print(f"[oracle] iters={list(iters)} eta shape={o_eta.shape} "
          f"max|eta0|={np.abs(o_eta[:, :, 0]).max():.4f}")

    grid, wall, z, state, model = bk.build_bickley()
    lat = np.degrees(np.asarray(grid.lat))      # (n_lat,)
    lon = np.degrees(np.asarray(grid.lon))      # (n_lon,)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    eta = _bump(LON, LAT)                        # (n_lat, n_lon)
    eta = eta.reshape(np.asarray(state.eta.data).shape)  # match the eta field shape
    state = state._replace(
        eta=state.eta.replace(data=jnp.asarray(eta)),
        u=state.u.replace(data=jnp.zeros_like(state.u.data)),
        v=state.v.replace(data=jnp.zeros_like(state.v.data)))

    # Validate the IC eta alignment vs the oracle (both should be the same bump).
    c0, la0, roll = _align(np.asarray(state.eta.data), o_eta[:, :, 0])
    print(f"[align] eta IC corr {c0:+.4f} (lat0={la0} roll={roll})")

    step = jax.jit(lambda s: model.step(s, DT, surface_forcing=None))
    from legoesm.ocean.fidelity.compare import compare_field
    print("\n  iter | corr(eta) | corr(u)  | lego max|u| | oracle max|u| | eta_rmse")
    snap = {0: np.asarray(state.eta.data)}
    usnap = {0: np.asarray(state.u.data)[:, :, 0]}
    for it in range(1, nsteps + 1):
        state = step(state)
        if it in iters:
            snap[it] = np.asarray(state.u.data)  # placeholder
            le = np.asarray(state.eta.data)
            lu = np.asarray(state.u.data)[:, :, 0]
            oi = int(np.where(iters == it)[0][0])
            oe, ou = o_eta[:, :, oi], o_u[:, :, oi]
            le_a = np.roll(le, roll, axis=1)[la0:la0 + oe.shape[0], :oe.shape[1]]
            _, lau, rou = _align(lu, ou)
            lu_a = np.roll(lu, rou, axis=1)[lau:lau + ou.shape[0], :ou.shape[1]]
            me = compare_field(le_a, oe)
            mu = compare_field(lu_a, ou)
            print(f"  {it:4d} |  {me.pattern_corr:+.4f}  | {mu.pattern_corr:+.4f} | "
                  f"{np.abs(lu_a).max():.4e} | {np.abs(ou).max():.4e} | {me.nrmse:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
