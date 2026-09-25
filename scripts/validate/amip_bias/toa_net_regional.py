#!/usr/bin/env python3
"""Regional top-of-atmosphere net radiation, model and bias against CERES.

A water-cycle change is not free: it moves clouds, and clouds move the energy
budget. Both reviewers named TOA net radiation as the metric that should
decide whether a surface-flux correction is worth adopting, because a moisture
improvement bought with a larger radiative imbalance is not an improvement in
a model destined for coupled runs.

Net is rsdt - rsut - rlut, and the bias replaces the model's outgoing fluxes
with CERES-EBAF's for the same calendar months, so the incoming solar is
identical on both sides and the difference is entirely the model's own
reflection and emission.

Usage: toa_net_regional.py <run> [<run> ...]
"""
import sys

import numpy as np
sys.path.insert(0,"scripts/validate/amip_bias")
import regional_bias as rb
B={"ITCZ 10S-10N":(-10,10,0,360),"trades 10-30N":(10,30,0,360),"trades 10-30S":(-30,-10,0,360),
   "SO stormtrack":(-60,-30,0,360),"NH midlat":(30,60,0,360),"GLOBAL":(-90,90,0,360)}
print(f"{'region':<16}" + "".join(f"{r:>22}" for r in sys.argv[1:]))
rows={}
for run in sys.argv[1:]:
    d={v: rb._load_model(run,v) for v in ("rsdt","rsut","rlut")}
    lat,lon=np.asarray(d["rsdt"].lat),np.asarray(d["rsdt"].lon)
    months=rb._month_labels(d["rsdt"])
    net=(np.asarray(d["rsdt"]["rsdt"]).mean(0)-np.asarray(d["rsut"]["rsut"]).mean(0)
         -np.asarray(d["rlut"]["rlut"]).mean(0))
    o={}
    for v in ("rsut","rlut"):
        o[v]=np.asarray(rb._ref_clim(v,months,lat,lon))
    neto=np.asarray(d["rsdt"]["rsdt"]).mean(0)-o["rsut"]-o["rlut"]
    rows[run]=[(rb.region_mean(net,lat,lon,b), rb.region_mean(net-neto,lat,lon,b)) for b in B.values()]
for i,n in enumerate(B):
    print(f"{n:<16}" + "".join(f"  net={rows[r][i][0]:7.2f} bias={rows[r][i][1]:7.2f}" for r in sys.argv[1:]))
print("net = model TOA net [W/m2]; bias = model minus CERES-EBAF for the same months.")
