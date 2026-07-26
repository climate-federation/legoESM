"""Does legoESM conserve tracer CONTENT under pure transport?

NEMO conserves sum(e3t*T) exactly by construction: the tracer update is
flux-form on thickness-weighted content (trazdf.F90:271-278), and the
advecting w carries the layer-thickness tendency (sshwzv.F90:198-206) so the
transport satisfies discrete continuity with the breathing z-star layers.

legoESM has the same two pieces -- flux-form hT update
(ocean_model_latlon_cgrid.py:3962-3964) and the z-star sigma correction
(vertical.py:942-965) -- but sources d(eta)/dt DIFFERENTLY: NEMO uses the
actual ssh increment from its barotropic solve (r3t(Kaa)-r3t(Kbb)), while
legoESM infers it from the baroclinic column divergence (w_euler[...,0]).
Those agree only if the column-integrated div(h*u) matches what the
split-explicit barotropic solver actually did to eta.

If they DON'T agree, the advecting transport is inconsistent with the
thickness evolution and tracer content leaks -- systematically, invisibly to
any instantaneous tendency comparison, and cumulatively over 34,560 steps.
That is exactly the defect profile left standing after GM, Redi, vertical
mixing and surface forcing were all eliminated as carriers of the #1226 ACC
growth deficit.

This is a BLACK-BOX test: surface forcing off, so the only thing that can
change globally-integrated content is a transport/thickness inconsistency.
Drift is reported in K per 1000 steps of volume-mean temperature, which is
directly comparable to the ~1 K abyssal signal the ACC deficit needs.

Usage:
  tracer_content_conservation.py [n_steps]
"""
import sys

import numpy as np
import jax

from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)
from legoesm.ocean.vertical import compute_layer_thickness

import dataclasses

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_TRAJ"
RESTART = f"{DINO}/RUN_Y5_REBUILD/DINO_00057600_restart.nc"
DT = 2700.0
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# Optional asselin_gamma override: gamma=0 removes the Robert-Asselin filter
# entirely, isolating whether the leak is the filter (which divides
# thickness-weighted content by a SEPARATELY filtered thickness, and is not
# exactly conservative) or the advection/thickness bookkeeping.
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else None

g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=0)
s = read_nemo_restart(RESTART, nn_hls=0)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                          lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
if GAMMA is not None:
    cfg = dataclasses.replace(cfg, asselin_gamma=GAMMA)
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
print(f"asselin_gamma = {mc.asselin_gamma}")
model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing)

st = br.state
if len(sys.argv) > 3 and sys.argv[3] == "f64":
    # Promote every float leaf of the state to float64 so the MODEL integrates
    # in double precision. If the content drift collapses when this is on, the
    # drift was float32 roundoff, not a conservation defect.
    st = jax.tree_util.tree_map(
        lambda x: x.astype(np.float64) if hasattr(x, "dtype")
        and np.issubdtype(x.dtype, np.floating) else x, st)
    print("STATE PROMOTED TO float64")
print("state dtype:", np.asarray(st.T.data).dtype)
area = np.asarray(br.geometry.cell_area) if hasattr(br.geometry, "cell_area") else None
if area is None:                       # fall back to NEMO's own e1t*e2t
    import netCDF4 as nc
    _mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    area = (np.asarray(_mm["e1t"][0]).squeeze()
            * np.asarray(_mm["e2t"][0]).squeeze())
active = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]


# Depth bands: the ACC deficit is sourced below 2000 m (memory addendum 19),
# so WHERE the leak sits decides whether a small global-mean number matters.
_zc = np.abs(np.asarray(br.z_coord.z_ref if hasattr(br.z_coord, "z_ref")
                        else np.cumsum(np.asarray(br.z_coord.dz_ref))))
BANDS = [("0-200m", 0.0, 200.0), ("200-1000m", 200.0, 1000.0),
         ("1000-2500m", 1000.0, 2500.0), ("2500m+", 2500.0, 1e9)]


def content(state):
    """Volume-integrated heat and salt content, and total volume.

    ALL accumulation in float64. The model state is float32 (eps~1.2e-7) and
    the global sum is ~1e18 over 372k cells, so a float32 reduction is itself
    only accurate to ~1e-7 relative -- the same order as the drift being
    measured. Summing in float32 measures the accumulator, not the model.
    """
    h = np.asarray(compute_layer_thickness(
        np.asarray(state.eta.data, dtype=np.float64),
        np.asarray(state.H_bathy.data, dtype=np.float64), br.z_coord,
        min_water_column_m=mc.min_water_column_m), dtype=np.float64)
    vol = h * area.astype(np.float64)[..., None] * active
    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    return (float(np.sum(vol * T, dtype=np.float64)),
            float(np.sum(vol * S, dtype=np.float64)),
            float(np.sum(vol, dtype=np.float64)))


def band_content(state):
    """Per-depth-band heat content and volume."""
    h = np.asarray(compute_layer_thickness(
        np.asarray(state.eta.data, dtype=np.float64),
        np.asarray(state.H_bathy.data, dtype=np.float64), br.z_coord,
        min_water_column_m=mc.min_water_column_m), dtype=np.float64)
    vol = h * area.astype(np.float64)[..., None] * active
    T = np.asarray(state.T.data, dtype=np.float64)
    out = {}
    for name, z0, z1 in BANDS:
        lev = (_zc >= z0) & (_zc < z1)
        m = vol * lev[None, None, :]
        out[name] = (float(np.sum(m * T, dtype=np.float64)),
                     float(np.sum(m, dtype=np.float64)))
    return out


# WIND ONLY: no surface heat/salt forcing is applied (we never call
# apply_dino_lat_lon_surface_forcing), so `sf` carries tau only and nothing
# can legitimately change globally-integrated heat/salt content.
dyn = jax.jit(lambda x: model.step(x, DT, surface_forcing=sf))

H0, S0, V0 = content(st)
B0 = band_content(st)
print(f"recipe nemo_dino_kamm_mlf, from NEMO y5 restart, {NSTEPS} steps, "
      f"NO surface heat/salt forcing")
print(f"initial: heat={H0:.10e} K.m3   salt={S0:.10e} psu.m3   vol={V0:.10e} m3\n")
print(f"{'step':>6}{'d(heat)/H0':>16}{'d(salt)/S0':>16}{'d(vol)/V0':>16}"
      f"{'mean dT [K]':>14}")
for k in range(1, NSTEPS + 1):
    st = dyn(st)
    if k % max(1, NSTEPS // 10) == 0 or k == 1:
        H, S_, V = content(st)
        print(f"{k:>6}{(H-H0)/abs(H0):>16.3e}{(S_-S0)/abs(S0):>16.3e}"
              f"{(V-V0)/abs(V0):>16.3e}{(H-H0)/V0:>14.3e}")

B1 = band_content(st)
print(f"\n{'band':<12}{'d(heat)/H0_band':>18}{'band mean dT [K]':>20}"
      f"{'-> K per 3yr run':>18}")
for name, _, _ in BANDS:
    h0b, v0b = B0[name]; h1b, v1b = B1[name]
    if abs(h0b) < 1e-30 or v0b <= 0: continue
    dT = (h1b - h0b) / v0b
    print(f"{name:<12}{(h1b-h0b)/abs(h0b):>18.3e}{dT:>20.3e}"
          f"{dT/NSTEPS*34560:>18.3e}")

H, S_, V = content(st)
dT_per_1000 = (H - H0) / V0 / NSTEPS * 1000.0
print(f"\nDRIFT over {NSTEPS} steps: heat {(H-H0)/abs(H0):+.3e} rel, "
      f"volume-mean T {(H-H0)/V0:+.3e} K")
print(f"EXTRAPOLATED: {dT_per_1000:+.3e} K per 1000 steps "
      f"=> {dT_per_1000*34.56:+.3e} K over a 3-yr run (34,560 steps)")
print("\nA conservative flux-form scheme with continuity-consistent transport "
      "drifts only at\nroundoff (~1e-15 rel). Anything above ~1e-10 rel is a "
      "real content leak.")
