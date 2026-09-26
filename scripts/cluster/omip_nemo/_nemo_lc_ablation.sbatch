#!/bin/bash
#SBATCH --job-name=lcabl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lcabl_%j.out
# Both-sided ablation, oracle side: NEMO 2 days hourly with ln_lc=.false.
# (RUN_HOURLY_NOLC) vs ln_lc=.true. (RUN_HOURLY). Langmuir injection peaks at
# half the LC depth (~10 m) and is absent from eshear/estrat/ediss.
# PRE-REGISTERED: NEMO-noLC dawn turbocline within 3 m of OURS (22.8 m night 1)
# -> Langmuir carries the oracle's nocturnal deepening and our LC depth/source
# is the row; NEMO-noLC still >= 27 m -> Langmuir exonerated, transport next.
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
$PY - <<PYEOF
import netCDF4 as nc, numpy as np
def series(d):
    f = nc.Dataset("$B/"+d+"/ORCA1_1h_20000101_20000102_grid_T_1h.nc")
    la = f.variables["nav_lat"][:]; lo = f.variables["nav_lon"][:] % 360.0
    box = (np.abs(la) <= 2.0) & (lo >= 220.0) & (lo <= 240.0)
    mk = f.variables["mldkz5"]; r10 = f.variables["mldr10_1"]; tos = f.variables["tos"]
    return [(float(np.ma.masked_invalid(mk[h])[box].mean()), float(np.ma.masked_invalid(r10[h])[box].mean()), float(np.ma.masked_invalid(tos[h])[box].mean())) for h in range(mk.shape[0])]
A = series("RUN_HOURLY"); Bn = series("RUN_HOURLY_NOLC")
ours = {6:18.24,12:21.56,18:22.77,24:14.43,30:15.93,36:17.17,42:18.24,48:16.61}
print("hour  local | turbocline LC  noLC | mldr10 LC noLC | tos LC   noLC   d(noLC-LC) | ours turbocline (t=h)")
for h,(a,b) in enumerate(zip(A,Bn)):
    o = ours.get(h+1, None)
    print(f"{h:4d}  {(h+15.3)%24:04.1f} | {a[0]:6.2f} {b[0]:6.2f} | {a[1]:6.2f} {b[1]:6.2f} | {a[2]:7.3f} {b[2]:7.3f} {b[2]-a[2]:+.3f} | {'' if o is None else f'{o:6.2f}'}")
PYEOF
echo "=== card exit: $? ==="
