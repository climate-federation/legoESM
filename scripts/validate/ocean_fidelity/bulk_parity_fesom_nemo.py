#!/usr/bin/env python
"""Bulk parity: fesom-jax (FESOM2 C port) vs legoESM (NEMO NCAR) heat fluxes
on IDENTICAL NH-winter inputs. Both get absolute wind, same SST/air state.
Output: Q_net difference decomposed, on a small regime grid."""
import numpy as np, jax.numpy as jnp
import sys
sys.path.insert(0, '/burg-archive/glab/users/pg2328/fesom_jax_repo')
from fesom_jax import forcing as ff
from legoesm.ocean.bulk_flux_omip import air_sea_fluxes
from legoesm import constants

# NH-winter regime grid: SST 5..20 C, T_air = SST - dT (cold advection), wind 4..16
SST = np.array([5., 10., 15., 20.])
dT  = np.array([2., 5., 10.])
W   = np.array([4., 8., 12., 16.])
sst, dt_, w = np.meshgrid(SST, dT, W, indexing='ij')
sst, dt_, w = sst.ravel(), dt_.ravel(), w.ravel()
tair = sst - dt_
# 80% RH specific humidity at T_air (rough, same for both sides).
# This is NEMO/FESOM's OWN bulk saturation curve (sbcblk: 0.98 * 640380/rho *
# exp(-5107.4/T)), not ours: the whole point of this probe is that both sides
# see the identical input humidity, so substituting legoesm.thermo here would
# change what is being compared.
# satcurve-ok: the oracle's bulk qsat, held identical on both sides of a
# FESOM-vs-NEMO parity comparison.
qsat_a = 0.98*640380/1.22*np.exp(-5107.4/(tair + constants.T_freeze))*1e-3
q = 0.8 * 0.640380/1.22*np.exp(-5107.4/(tair + constants.T_freeze))
swd = np.full_like(sst, 80.0); lwd = np.full_like(sst, 280.0)

# --- fesom side ---
cd, ch, ce = ff.ncar_ocean_fluxes_mode(jnp.asarray(tair), jnp.asarray(q),
                                       jnp.asarray(w), jnp.zeros_like(w),
                                       jnp.asarray(sst), jnp.zeros_like(w), jnp.zeros_like(w))
qsr_f, qns_f, evap_f = ff.obudget(jnp.asarray(q), jnp.asarray(swd), jnp.asarray(lwd),
                                  jnp.asarray(sst), jnp.asarray(w), jnp.asarray(tair),
                                  ch, ce)
Qnet_f = np.asarray(qsr_f) - np.asarray(qns_f)     # +down into ocean

# --- legoESM/NEMO side ---
tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
    u10=jnp.asarray(w), v10=jnp.zeros_like(w),
    T_air_K=jnp.asarray(tair)+constants.T_freeze,
    q_air=jnp.asarray(q),
    T_sfc_K=jnp.asarray(sst)+constants.T_freeze, algo="ncar")
lw_up = (constants.emissivity_ocean * constants.sigma_sb
         * (np.asarray(sst) + constants.T_freeze)**4)
lw_net = constants.emissivity_ocean*lwd - lw_up
sw_net = (1.0-0.066)*swd  # NEMO ocean albedo ~0.066
Qnet_n = np.asarray(sh)+np.asarray(lh)+sw_net+lw_net

d = Qnet_f - Qnet_n
print(f"Q_net fesom-NEMO [W/m2]: mean {d.mean():+.1f} min {d.min():+.1f} max {d.max():+.1f}")
# decompose: turbulent (sh+lh) parts
sens_f = -np.asarray(ff.BULK_RHOAIR*ff.BULK_CPAIR*ch*jnp.asarray(w)*(jnp.asarray(tair)-jnp.asarray(sst)))
print("regime rows (SST, dT, wind, dQ):")
for i in range(0, len(d), 6):
    print(f"  SST {sst[i]:4.0f} dT {dt_[i]:3.0f} W {w[i]:3.0f}  dQ {d[i]:+7.1f}  "
          f"sh_n {np.asarray(sh)[i]:+6.1f} lh_n {np.asarray(lh)[i]:+7.1f}")
