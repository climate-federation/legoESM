"""Controlled comparison run (v2 — WIND FIX): legoESM nemo_dino_kamm, 180-day mean.

v1 bug: wind_through_step=True recipes SKIP the wind in
apply_dino_lat_lon_surface_forcing, expecting it through model.step(surface_forcing=).
The v1 harness never passed surface_forcing -> NO WIND for 180d. This version builds
the step surface forcing (dino_step_surface_forcing) and threads it into model.step.
Everything else byte-identical to kamm_run180.py.

Usage: kamm_run180_v2.py <nsteps> <out.npz>   (nsteps=5760 for full 180d)
"""
import os, sys, dataclasses, numpy as np, jax, jax.numpy as jnp
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
    dino_step_surface_forcing)

RECIPE = sys.argv[1]; OUT = sys.argv[2]; DT = 2700.0
NSTEPS = 11520; ACC0 = 0  # 1-year screen: full year-1 mean (matched to NEMO annual mean)
RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
# MATCHED-STATE GROWTH TEST: set DINO_INIT_RESTART=<rebuilt single-file NEMO
# restart> to start from a *developed* NEMO state instead of the analytic
# rest IC. Everything else (window, forcing, annual-mean output, metric) is
# held byte-identical to the from-rest screen, so the only variable is the
# initial state -- which makes lego's 1-year ACC growth directly comparable
# to NEMO's own growth over the same year from the same state.
INIT_RESTART = os.environ.get("DINO_INIT_RESTART")
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=0)
s = read_nemo_restart(INIT_RESTART or f"{RUN}/DINO_00000320_restart.nc", nn_hls=0)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = dataclasses.replace(dino_config_for_recipe(RECIPE),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)   # bridge-frame lon fix
if os.environ.get("DINO_VMIX"):
    # Swap-the-subsystem discriminator: "constant" uses cfg.A_v_bg/K_v_bg
    # directly, which for the kamm card are 1.2e-4 / 1.2e-5 -- byte-identical
    # to NEMO's rn_avm0/rn_avt0 under ln_zdfcst=T. Running BOTH models on this
    # zero-transcription-risk closure asks whether the growth gap is OWNED by
    # the vertical mixing scheme or merely survives it.
    cfg = dataclasses.replace(cfg, vmix_scheme=os.environ["DINO_VMIX"])
    print(f"ABLATION: vmix_scheme={cfg.vmix_scheme}")
if os.environ.get("DINO_NO_GM"):
    # Ablation / instrument-POWER control: GM is a first-order ACC lever, so a
    # run with it off bounds how much a 1-year window can move ACC at all. If
    # ACC barely shifts here, a 1-year matched-state ACC comparison has no
    # discriminating power and only the multi-year curve can be trusted.
    cfg = dataclasses.replace(cfg, use_gm_redi=False)
    print("ABLATION: use_gm_redi=False")
if INIT_RESTART:
    st = br.state          # the bridged NEMO state itself, NOT the analytic rest IC
    print(f"INIT from developed NEMO restart: {INIT_RESTART}")
else:
    st = dino_lat_lon_state(br.geometry, br.z_coord, cfg, land_mask_override=br.land_mask)
# --- HARD topo census gate: legoESM wet cells must equal NEMO tmask exactly ---
import netCDF4 as _nc
_mm = _nc.Dataset(f"{RUN}/mesh_mask.nc")   # FULL frame: NEMO 5 files are haloless
_tm = (np.moveaxis(np.asarray(_mm["tmask"][0]).squeeze(), 0, -1) > 0.5)
_wet = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]
if not np.array_equal(_wet, _tm):
    raise SystemExit(f"TOPO CENSUS FAIL: {int(np.sum(_wet != _tm))} cells differ — refusing to run")
_umN = np.moveaxis(np.asarray(_mm["umask"][0]).squeeze(), 0, -1)[:, :, 0] > 0.5
_vmN = np.moveaxis(np.asarray(_mm["vmask"][0]).squeeze(), 0, -1)[:, :, 0] > 0.5
_umL = np.asarray(st.u_mask.data)[:, 1:53] > 0.5     # lego face i+1 = east face of col i
_vmL = np.asarray(st.v_mask.data)[1:200, :] > 0.5    # lego face j+1 = north face of row j
if not (np.array_equal(_umL, _umN) and np.array_equal(_vmL, _vmN)):
    raise SystemExit("FACE-MASK CENSUS FAIL: u/v masks differ from NEMO umask/vmask")
print(f"census OK (FULL 199x52): {int(_tm.sum())} wet cells + u/v face masks EXACT")          # analytic IC == NEMO usrdef_istate
if INIT_RESTART:
    # Same hard day-0 gate as kamm_twin_90d: a "twin" that silently ran from
    # rest on NEMO topography is the defect this gate exists to prevent.
    from kamm_twin_90d import verify_day0_matches_restart
    verify_day0_matches_restart(st, s, br.land_mask)
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing)   # WIND: tau_x/taum into the dycore external-tau block
print(f"slope_scheme={mc.gm_redi.slope_scheme} kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")

dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))  # sf constant (annual tau)

# DINO_YEARS>1 runs consecutive years and writes ONE annual mean per year
# (suffix _y2, _y3, ...), so a multi-year run reproduces exactly what NEMO's
# yearly output gives -- a growth CURVE, not just an endpoint. Single-year
# behaviour and the output filename are unchanged when DINO_YEARS is unset.
YEARS = int(os.environ.get("DINO_YEARS", "1"))
kglob = 0
for year in range(1, YEARS + 1):
    acc = {k: jnp.zeros_like(getattr(st, k).data) for k in ("T", "S", "eta", "u", "v")}
    for k in range(NSTEPS):
        kglob += 1
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                t_seconds=kglob * DT)
        st = dyn(st)
        if k >= ACC0:
            for f in acc: acc[f] = acc[f] + getattr(st, f).data
        if (k + 1) % 960 == 0:
            Td = np.asarray(st.T.data); m = np.asarray(st.land_mask.data) > 0.5
            us = np.asarray(st.u.data[..., 0])
            print(f"  y{year} day {(k+1)*DT/86400:5.1f}  T[{Td[m].min():.1f},{Td[m].max():.1f}] "
                  f"usurf[{us.min():.3f},{us.max():.3f}] finite={np.isfinite(Td[m]).all()}", flush=True)
    mean = {f: np.asarray(acc[f]) / (NSTEPS - ACC0) for f in acc}
    mean["land_mask"] = np.asarray(st.land_mask.data)
    out_year = OUT if year == 1 else OUT.replace(".npz", f"_y{year}.npz")
    np.savez(out_year, **mean)
    Td = np.asarray(st.T.data); m = mean["land_mask"] > 0.5
    print(f"DONE year {year}/{YEARS} ({kglob*DT/86400:.1f}d cumulative) "
          f"STABLE={np.isfinite(Td[m]).all() and Td[m].max()<45} -> {out_year}", flush=True)
