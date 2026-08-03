"""#1317 tracer (temperature) tendency comparison: legoESM per-term probe at
the bridged NEMO day-180 (kt 5760) leap-frog-before-populated state vs NEMO's
own trdtra dump (kt 5760->5761). Mirrors momentum_budget_diff.py's pattern:
same bridge, same recipe card, same channel. NEMO ttrd_* are pure tendencies
[K/s] (verified from trdtra.F90 -- no dt scaling; trd_tra_adv_t divides by
e3t, trd_tra_mng passes straight through). ttrd_tot is NEMO's own model-total
trend (Kmm level); dumped alongside the 5761 restart it is the tendency
computed DURING the step that advanced 5760->5761, i.e. evaluated on the
5760 (Kbb/Kmm now-level) state -- the same convention momentum_budget_diff.py
already relies on.

Buckets:
  ADVECTION = ttrd_xad + ttrd_yad + ttrd_zad   (lego: explicit compute_advection_flux_div_pair)
  ISO/REDI  = ttrd_ldf                          (lego: GM/Redi tendency, incl. implicit K33 fold; K_h=0 on this card)
  VERTMIX   = ttrd_zdf + ttrd_evd               (lego: implicit_vertical_diffusion_ocean using NEMO's own avt_k,
                                                        isolating the SOLVER/discretization from the TKE-closure Kv)
  FORCING   = ttrd_qsr + ttrd_nsr               (lego: probe's surface-forcing contribution to dT_dt)
  TOTAL     = ttrd_tot                          (lego: sum of the 4 buckets above)

Caveat (stated, not hidden): VERTMIX uses NEMO's diagnosed avt_k as the
diffusivity for BOTH sides, so it tests the vertical-mixing OPERATOR match,
not the TKE turbulence-closure match (lego's own prognostic TKE state isn't
in the bridge).

#1317 time-level fix (this file + tendency_probe.py): NEMO's MLF stepper
evaluates the iso-neutral (Redi/GM) operator ENTIRELY on the leap-frog BEFORE
level (Nbb) -- stpmlf.F90:199 `CALL ldf_slp(kstp, rhd, rn2b, Nbb, Nnn)`
("before slope for standard operator"); traldf_iso_scheme.h90 gradients read
pt_in(...,Kbb); traldf.F90:97/103 pass pts(:,:,:,:,Kbb) into
traldf_iso_lap/blp -- while ADVECTION is evaluated on the NOW level
(Nnn/Kmm, traadv_fct.F90:354 `trd_tra(..., pt(:,:,:,jn,Kmm))`; the
high-order flux itself also reads pt(...,Kmm), traadv_fct.F90:187-269).
ISO/REDI below feeds the probe's GM/Redi bucket the BEFORE tracer
(state0.T_before/S_before); ADVECTION correctly keeps the NOW-level
state.T/S.data (Kmm) it always used.

GM-bolus-through-FCT fold (this card sets gm_bolus_advection="through_fct",
kappa_GM=200): NEMO folds the eddy-induced (GM) transport into the
ADVECTING velocity before tra_adv_fct (traadv.F90:210
`CALL ldf_eiv_trp(kt, kit000, pFu, pFv, pFw, Kmm, Krhs)`, called from the
SAME slopes ldf_slp computed at Nbb) -- so NEMO's own ttrd_xad/yad/zad
already includes the GM bolus contribution, and the pure-Redi ISO/REDI
bucket (gm_redi_tracer_tendency_latlon with gm_bolus_advection="through_fct")
correctly excludes it (nemo_iso_lap_tracer_tendency_latlon_cgrid: `pass` on
the through_fct branch, see gm_redi_latlon_cgrid.py:1622-1626 -- no
double-count regardless of return_bolus_transport). ADVECTION below adds the
SAME before-level bolus transport (add_bolus_to_advecting_flux, exactly the
production ocean_model_latlon_cgrid.py:3732-3741 helper) to its mass flux so
the comparison is apples-to-apples with NEMO's bolus-augmented ttrd_*ad.
"""
import dataclasses
import glob
import os

import netCDF4 as nc
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
    dino_T_star_seasonal, dino_Q_sr_seasonal,
)
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import (
    restoring_surface_forcing,
)
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig, shortwave_penetration_tendency,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    compute_advection_flux_div_pair,
    add_bolus_to_advecting_flux,
    static_kappa_redi_override,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface, min_cell_to_vface, compute_face_masks_3d,
)
from legoesm.ocean.vertical import (
    compute_layer_thickness, OceanPartialCellCoordinate,
    compute_ocean_jacobian, diagnose_w_from_flux_div,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_ocean, build_dz_half,
)

RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP"
DT = 2700.0
H = 2
CH = slice(12, 47)  # channel rows, interior idx (matches momentum_budget_diff.py)

# #1226: cfg built BEFORE the bridge so cfg.omega (NEMO's full-precision
# Earth rotation rate on the nemo_dino_kamm_mlf card) reaches
# bridge_nemo_to_legoesm_topo's grid.f construction -- omitting this left
# grid.f (and every f20 taper reference downstream) on legoESM's rounded
# constants.Omega default, silently re-running the pre-fix ldf_eiv kappa
# (aeiu) amplitude bias this script exists to validate.
cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)

# --- bridge day-0 twin state (leap-frog before-level populated: #1317 --bridge-before) ---
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=0)
s = read_nemo_restart(f"{RUN}/DINO_00005760_restart.nc", nn_hls=0)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True,
                                  omega=cfg.omega)
before = read_nemo_restart_before(f"{RUN}/DINO_00005760_restart.nc", nn_hls=0)
state0 = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)
br = br._replace(state=state0)

mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing)

# --- probe: forcing + iso/Redi (GM/Redi, incl. implicit K33) bucket ---
# #1317: the iso-neutral operator in NEMO's MLF stepper is evaluated ENTIRELY
# on the leap-frog BEFORE level (Nbb/Kbb) -- stpmlf.F90:199
# `CALL ldf_slp(kstp, rhd, rn2b, Nbb, Nnn)` ("before slope for standard
# operator"); traldf_iso_scheme.h90 gradients read pt_in(...,Kbb);
# traldf.F90:97/103 pass pts(:,:,:,:,Kbb) into traldf_iso_lap/blp -- while
# ADVECTION is evaluated on the NOW level (Nnn/Kmm, traadv_fct.F90:354
# `trd_tra(..., pt(:,:,:,jn,Kmm))`). br.state.T/S are the "tn/sn" now-level
# fields (restart.F90:195-196); the before-level twin lives on
# br.state.T_before/S_before (populated by bridge_before_state_topo above).
# Feed the BEFORE tracer to the probe's GM/Redi bucket so it matches NEMO's
# own ttrd_ldf time level; ADVECTION below correctly keeps the now-level
# state.T/S.data (Kmm), unchanged.
pr = probe_latlon_cgrid(br.state, br.geometry, br.z_coord, mc,
                        surface_forcing=sf, dt=DT,
                        gm_redi_tracer_state=(state0.T_before.data,
                                               state0.S_before.data))

# --- lego ADVECTION bucket: mirror ocean_model_latlon_cgrid.step()'s mass-flux
# build on the FROZEN state itself (no barotropic correction -- a converged
# bridged state's own u/v already IS the mass flux the model would use for a
# single-step probe; this is the same "frozen state, single evaluation" logic
# the momentum probe already uses for vertadv_u / botdrag_u). ---
state = br.state
z_coord = br.z_coord
grid = br.geometry
mask = state.land_mask.data
h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord,
                              min_water_column_m=mc.min_water_column_m)
h_u = min_cell_to_uface(h_k)
h_v = min_cell_to_vface(h_k, grid)
if isinstance(z_coord, OceanPartialCellCoordinate):
    u_mask_3d_tr, v_mask_3d_tr = compute_face_masks_3d(z_coord.is_active, grid)
    active_3d = z_coord.is_active.astype(h_k.dtype)
else:
    u_mask_3d_tr = state.u_mask.data[..., jnp.newaxis]
    v_mask_3d_tr = state.v_mask.data[..., jnp.newaxis]
    active_3d = mask[:, :, jnp.newaxis] * jnp.ones_like(h_k)
mass_flux_u = h_u * state.u.data * u_mask_3d_tr
mass_flux_v = h_v * state.v.data * v_mask_3d_tr

# GM-bolus-through-FCT fold (see module docstring): this recipe sets
# gm_bolus_advection="through_fct" with kappa_GM=200 -- NEMO's own
# ttrd_xad/yad/zad already carries the eddy-induced (GM) transport
# (ldf_eiv_trp adds usd/vsd to the advecting velocity, traadv.F90:196-197,
# BEFORE tra_adv_fct), built from the SAME before-level slopes as ldf_slp.
# Reproduce production's exact call (ocean_model_latlon_cgrid.py:3718-3741):
# gm_redi_tracer_tendency_latlon(..., return_bolus_transport=True) on the
# BEFORE tracer, then add_bolus_to_advecting_flux onto the mass flux --
# so ADVECTION below is apples-to-apples with NEMO's bolus-augmented trend.
_kappa_redi_ov, _kappa_redi_v_ov = static_kappa_redi_override(mc.gm_redi, grid)
_, _, _bolus = gm_redi_tracer_tendency_latlon(
    state0.T_before.data, state0.S_before.data, state.eta.data, state.H_bathy.data,
    grid, z_coord, mc.gm_redi,
    eos=mc.eos, eos_linear=mc.eos_linear,
    mask=mask, u_mask=state.u_mask.data, v_mask=state.v_mask.data,
    rho_0=mc.constants.rho_0, g=mc.constants.g,
    omega=mc.omega,   # #1226: see the cfg/bridge omega note above.
    kappa_redi_override=_kappa_redi_ov,
    kappa_redi_v_override=_kappa_redi_v_ov,
    return_bolus_transport=True,
    dt=DT,
)
mass_flux_u, mass_flux_v, w_baro = add_bolus_to_advecting_flux(
    _bolus, mass_flux_u, mass_flux_v, u_mask_3d_tr, v_mask_3d_tr, grid, z_coord,
)

_wall_fill_mask = active_3d if getattr(mc, "tracer_wall_neumann_fill", True) else None
(dh_T, dv_T), (dh_S, dv_S) = compute_advection_flux_div_pair(
    state.T.data, state.S.data, mc.tracer_advection,
    mass_flux_u, mass_flux_v, w_baro, h_k, h_u, h_v, grid, DT,
    recon_fill_mask=_wall_fill_mask,
    linssh_top_flux=getattr(z_coord, "linear_free_surface", False),
)
# div_hut/vert_flux_div are [tracer*m/s] thickness-weighted divergences;
# tendency = -(div_hut + vert_flux_div) / h_k (flux-form -> rate).
h_safe = jnp.maximum(h_k, 1e-10)
dT_adv = -(dh_T + dv_T) / h_safe * mask[:, :, jnp.newaxis]
dT_adv_h = -dh_T / h_safe * mask[:, :, jnp.newaxis]
dT_adv_v = -dv_T / h_safe * mask[:, :, jnp.newaxis]

def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


# --- lego VERTMIX bucket: implicit backward-Euler solve driven by NEMO's OWN
# diagnosed avt_k (isolates the solver/discretization; NOT lego's TKE closure,
# which isn't in the bridge -- stated caveat -- and NOT NEMO's separate
# convective-adjustment boost that produces ttrd_evd; see header + report). ---
rst5760 = nc.Dataset(f"{RUN}/DINO_00005760_restart.nc")
avt_k = llz(rst5760["avt_k"][0])  # (n_lat, n_lon, nlev); avt_k[...,k] = interface
# ABOVE T-cell k (NEMO trdtra.F90:158 avt(jk) multiplies (T(jk-1)-T(jk)));
# avt_k[...,0]=surface=0. The nlev-1 INTERIOR interfaces implicit_vertical_
# diffusion_ocean wants are exactly avt_k[...,1:] -- NOT a cell-centre average
# (avt_k is already AT interfaces, not at cell centres).
avt_iface = avt_k[..., 1:]  # (n_lat, n_lon, nlev-1) interior interfaces
J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
jac = jnp.maximum(J[..., jnp.newaxis], 1e-10)
dz_cell = z_coord.dz_ref * jac
dz_half = build_dz_half(dz_cell)
T_imp = implicit_vertical_diffusion_ocean(state.T.data, jnp.asarray(avt_iface), dz_cell, dz_half, DT)
dT_vmix = (T_imp - state.T.data) / DT * mask[:, :, jnp.newaxis]

# --- lego FORCING bucket: DINO routes heat/salt forcing through a SEPARATE
# "analytic post-step applicator" (apply_dino_lat_lon_surface_forcing), NOT
# through model.step(surface_forcing=...) -- that path (dino_step_surface_forcing,
# what the probe consumes) carries ONLY wind (wind_through_step=True). So
# pr.dT_dt_total has ZERO forcing contribution here; it is (lateral_diff(K_h=0)
# + GM/Redi) only. Compute the REAL forcing tendency by mirroring
# apply_dino_lat_lon_surface_forcing's own T-restoring + Jerlov-SW math
# directly (same functions, same t_seconds convention as kamm_twin_90d.py:
# t_seconds=(k+1)*DT with NEMO's 1-indexed kt -> absolute t_seconds=5761*DT).
dT_iso = pr.dT_gm_redi
assert getattr(cfg, "forcing_annual_cycle", False)
t_seconds = 5761 * DT
dz_0 = float(z_coord.dz_ref[0])
lat1 = forcing["lat_deg_1d"]
T_star_2d = jnp.broadcast_to(dino_T_star_seasonal(lat1, t_seconds, cfg)[:, None], mask.shape)
Q_sr_2d = jnp.broadcast_to(dino_Q_sr_seasonal(lat1, t_seconds, cfg)[:, None], mask.shape)
tau_T = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0)
tau_S = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0)
restoring_cfg = RestoringConfig(
    tau_T=tau_T, tau_S=tau_S, T_star_array=T_star_2d,
    S_star_array=forcing["S_star_2d"], subtract_qsr=True, implicit=True,
)


class _GridShim:
    def __init__(self, cell_mask):
        self.grid_lat = jnp.zeros_like(cell_mask)


rest_out = restoring_surface_forcing(
    state.T.data, state.S.data, _GridShim(mask), restoring_cfg,
    sw_down=Q_sr_2d, dt=DT, rho_0=cfg.rho_0, c_p=cfg.c_p, dz_0=dz_0,
)
sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
dT_dt_sw = shortwave_penetration_tendency(
    sw_down=Q_sr_2d, z_coord_dz_ref=z_coord.dz_ref,
    z_coord_z_half_ref=z_coord.z_half_ref,
    jacobian=jnp.ones_like(state.eta.data), config=sw_cfg,
    rho_0=cfg.rho_0, c_sw=cfg.c_p,
)
dT_forcing = (rest_out.dT_dt + dT_dt_sw) * mask[:, :, jnp.newaxis]

dT_total_lego = dT_adv + dT_iso + dT_vmix + dT_forcing

# --- NEMO trends (5761 dump = tendency evaluated on the 5760 state) ---
rst = nc.Dataset(f"{RUN}/DINO_00005761_restart.nc")


def nf(v):
    return llz(rst[v][0])


nemo_xad, nemo_yad, nemo_zad = nf("ttrd_xad"), nf("ttrd_yad"), nf("ttrd_zad")
nemo_adv = nemo_xad + nemo_yad + nemo_zad
nemo_adv_h = nemo_xad + nemo_yad
nemo_adv_v = nemo_zad
nemo_ldf = nf("ttrd_ldf")
nemo_zdf = nf("ttrd_zdf")
nemo_evd = nf("ttrd_evd")
nemo_vmix = nemo_zdf + nemo_evd
nemo_qsr = nf("ttrd_qsr")
nemo_nsr = nf("ttrd_nsr")
nemo_forcing = nemo_qsr + nemo_nsr
nemo_tot = nf("ttrd_tot")

# --- weights + wet mask (channel rows, interior columns like momentum script) ---
mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
tmask = llz(mm["tmask"][0]) > 0.5
e1t = np.asarray(mm["e1t"][0]).squeeze()
e2t = np.asarray(mm["e2t"][0]).squeeze()
e3 = np.asarray(mm["e3t_1d"][:]).squeeze()
Wt = (e1t * e2t)[..., None] * e3[None, None, :] * tmask

# upper-200m band (level index where cumulative e3 < 200)
z_cum = np.cumsum(e3) - 0.5 * e3
upper200 = z_cum <= 200.0

wet_ch = np.zeros_like(tmask)
wet_ch[CH] = tmask[CH]
wet_ch_u200 = wet_ch & upper200[None, None, :]


def to_np(a):
    return np.asarray(a)


def bucket_stats(name, lego, nemo, drift=None):
    l = to_np(lego)
    n = to_np(nemo)
    out = {}
    for tag, m in (("full3D", wet_ch), ("upper200m", wet_ch_u200)):
        if m.sum() == 0:
            continue
        lm, nm = l[m], n[m]
        corr = np.corrcoef(lm, nm)[0, 1] if lm.size > 1 else float("nan")
        rms_l = np.sqrt(np.mean(lm ** 2))
        rms_n = np.sqrt(np.mean(nm ** 2))
        ratio = rms_l / rms_n if rms_n > 0 else float("nan")
        proj = float("nan")
        if drift is not None:
            dm = to_np(drift)[m]
            diffm = lm - nm
            if np.std(dm) > 0 and np.std(diffm) > 0:
                # corr((lego-nemo), drift pattern): a bucket whose model
                # DISAGREEMENT is spatially aligned with the day90-day0 SST/
                # thermocline drift is the bias-source candidate.
                proj = float(np.corrcoef(diffm, dm)[0, 1])
        out[tag] = dict(corr=corr, rms_lego=rms_l, rms_nemo=rms_n, ratio=ratio,
                        drift_proj=proj, n=int(m.sum()))
    return out


# drift pattern (day90-day0 T structure) -- a term that differs lego-vs-NEMO
# AND projects strongly onto this pattern is a source-of-bias candidate.
_drift_hits = glob.glob("/tmp/claude-*/**/d90_twin_final.npz", recursive=True)
drift3d = None
if _drift_hits:
    dz = np.load(_drift_hits[0])
    drift3d = dz["T3d_day90"] - dz["T3d_day0"]
    print(f"[drift file] {_drift_hits[0]}")
else:
    print("[drift file] none found -- drift_proj will be nan")

print(f"{'='*100}")
print("TRACER (temperature) TENDENCY BUCKET COMPARISON -- day-0 twin (kt=5760->5761)")
print(f"NEMO ttrd_* units: K/s (verified trdtra.F90; no dt scaling).")
print(f"{'='*100}")

buckets = [
    ("ADVECTION (xad+yad+zad)", dT_adv, nemo_adv),
    ("  advection horiz (xad+yad)", dT_adv_h, nemo_adv_h),
    ("  advection vert (zad)", dT_adv_v, nemo_adv_v),
    ("ISO/REDI (ldf)", dT_iso, nemo_ldf),
    ("VERTMIX (zdf+evd)", dT_vmix, nemo_vmix),
    ("FORCING (qsr+nsr)", dT_forcing, nemo_forcing),
    ("TOTAL", dT_total_lego, nemo_tot),
]

results = {}
for name, lego_arr, nemo_arr in buckets:
    st = bucket_stats(name, lego_arr, nemo_arr, drift=drift3d)
    results[name] = st
    print(f"\n{name}:")
    for tag, d in st.items():
        print(f"   {tag:10s} corr={d['corr']:+.4f}  rms_lego={d['rms_lego']:.3e}  "
              f"rms_nemo={d['rms_nemo']:.3e}  ratio={d['ratio']:.3f}  "
              f"drift_proj={d['drift_proj']:+.4f}  n={d['n']}")

# sanity: sum-of-buckets vs total (both sides)
lego_sum = dT_adv + dT_iso + dT_vmix + dT_forcing
nemo_sum = nemo_adv + nemo_ldf + nemo_vmix + nemo_forcing
l_resid = np.asarray(lego_sum - dT_total_lego)[wet_ch]
n_resid = np.asarray(nemo_sum - nemo_tot)[wet_ch]
print(f"\nSANITY (sum of buckets == total):")
print(f"  lego  max|sum-total| = {np.max(np.abs(l_resid)):.3e} K/s (should be ~0 by construction)")
print(f"  NEMO  max|sum-total| = {np.max(np.abs(n_resid)):.3e} K/s "
      f"(NEMO ttrd_tot also includes atf/dmp/bbc/npc -- nonzero residual expected)")

# --- plot: zonal-mean upper-500m sections per bucket (lego, NEMO, diff) +
# a per-bucket corr / drift-projection summary bar chart ---
upper500 = z_cum <= 500.0
lat_1d = np.degrees(np.asarray(br.geometry.lat))[CH]
depth_1d = z_cum[upper500]

plot_buckets = [
    ("ADVECTION", dT_adv, nemo_adv),
    ("ISO/REDI", dT_iso, nemo_ldf),
    ("VERTMIX", dT_vmix, nemo_vmix),
    ("FORCING", dT_forcing, nemo_forcing),
]


def zonal_mean_section(arr):
    a = np.asarray(arr)[CH][..., upper500]
    m = tmask[CH][..., upper500]
    num = np.sum(np.where(m, a, 0.0), axis=1)
    den = np.maximum(np.sum(m, axis=1), 1)
    return num / den  # (n_lat_ch, nlev_upper)


fig, axes = plt.subplots(len(plot_buckets), 3, figsize=(13, 3.2 * len(plot_buckets)),
                          sharex=True, sharey=True)
for row, (name, lego_arr, nemo_arr) in enumerate(plot_buckets):
    l_sec = zonal_mean_section(lego_arr)
    n_sec = zonal_mean_section(nemo_arr)
    vmax = max(np.nanmax(np.abs(l_sec)), np.nanmax(np.abs(n_sec)), 1e-30)
    for col, (title, sec, cmap, vm) in enumerate((
        (f"{name} lego", l_sec, "RdBu_r", vmax),
        (f"{name} NEMO", n_sec, "RdBu_r", vmax),
        (f"{name} diff (lego-NEMO)", l_sec - n_sec, "PuOr_r", vmax),
    )):
        ax = axes[row, col]
        im = ax.pcolormesh(lat_1d, depth_1d, sec.T, cmap=cmap, vmin=-vm, vmax=vm,
                            shading="nearest")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        if col == 0:
            ax.set_ylabel("depth [m]")
        if row == len(plot_buckets) - 1:
            ax.set_xlabel("lat [deg]")
        plt.colorbar(im, ax=ax, shrink=0.8, label="K/s")
fig.suptitle("Tracer tendency bucket comparison -- channel zonal mean, upper 500m\n"
             "(day-0 twin, kt=5760->5761)")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = "/home/dbalwada/legoESM/scripts/validate/ocean_fidelity/dino_1226/tracer_tendency_compare.png"
fig.savefig(out_png, dpi=130)
print(f"\n[saved] {out_png}")

# summary bar chart: per-bucket corr (full3D) + drift projection
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
names = [b[0] for b in buckets]
corrs = [results[n]["full3D"]["corr"] for n in names]
projs = [results[n]["full3D"]["drift_proj"] for n in names]
x = np.arange(len(names))
w = 0.35
ax2.bar(x - w / 2, corrs, width=w, label="corr(lego, NEMO)", color="tab:blue")
ax2.bar(x + w / 2, projs, width=w, label="corr(lego-NEMO diff, drift)", color="tab:red")
ax2.axhline(0, color="k", lw=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("correlation")
ax2.set_ylim(-1, 1)
ax2.legend(fontsize=8)
ax2.set_title("Per-bucket lego-vs-NEMO match + drift-pattern projection (full 3D, channel)")
fig2.tight_layout()
out_png2 = "/home/dbalwada/legoESM/scripts/validate/ocean_fidelity/dino_1226/tracer_tendency_compare_summary.png"
fig2.savefig(out_png2, dpi=130)
print(f"[saved] {out_png2}")
