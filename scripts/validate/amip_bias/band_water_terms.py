#!/usr/bin/env python3
"""Band water terms with ONE source per quantity: evaporation, precipitation
and transport, model against reference, all on the same bands.

Written after a review caught the previous table mixing sources: its model
terms came from the process ledger while its "observed transport" was carried
over from a table computed on 30-90 degree bands, so the reference transport
did not equal the reference precipitation minus the reference evaporation of
the band beside it. Here every column is recomputed for the band in the row,
the model's terms all come from the ledger, and the reference transport is
exactly Pobs - Eobs, so the row is internally checkable by eye.

Usage: band_water_terms.py [<run> [<flux_run>]]
"""
import sys, json
import numpy as np
sys.path.insert(0,"scripts/validate/amip_bias")
import regional_bias as rb
from cloud_layers import mesh_coords
import water_budget_bands as wbb
B={"ITCZ 10S-10N":(-10,10),"trades 10-30N":(10,30),"trades 10-30S":(-30,-10),
   "NH midlat 30-60N":(30,60),"SO stormtrack 60-30S":(-60,-30),"global":(-90,90)}
run = sys.argv[1] if len(sys.argv) > 1 else "wv_ctl30_ldgA"
z=np.load(f"{rb.ROOT}/{run}/budget_ledger_columns.npz",allow_pickle=True)
lat,_lo,area=mesh_coords(json.load(open(f"{rb.ROOT}/{run}/experiment_config.json")))
r=np.asarray(z["ledger_rates"])[:,:,0]*86400.0; P=[str(p) for p in z["processes"]]
ff=wbb._flux_fields(sys.argv[2] if len(sys.argv) > 2 else "wv_ctl30")
pm,pr_,mlat,mlon=ff["pr"]; em,er,_,_=ff["evspsbl"]
S=86400.0
print(f"{'band':<22}{'E':>7}{'Eobs':>7}{'P':>7}{'Pobs':>7}{'T':>7}{'Tobs':>7}{'dT':>7}")
for n,(lo,hi) in B.items():
    w=area*((lat>=lo)&(lat<=hi))
    E=float((r[:,P.index("turbulence")]*w).sum()/w.sum())
    Pm=-(float((r[:,P.index("microphysics")]*w).sum()/w.sum())
         +float((r[:,P.index("convection")]*w).sum()/w.sum()))
    T=float((r[:,P.index("dynamics")]*w).sum()/w.sum())
    box=(lo,hi,0,360)
    po=rb.region_mean(pr_,mlat,mlon,box)*S; eo=rb.region_mean(er,mlat,mlon,box)*S
    print(f"{n:<22}{E:7.2f}{eo:7.2f}{Pm:7.2f}{po:7.2f}{T:7.2f}{po-eo:7.2f}{T-(po-eo):7.2f}")
