#!/bin/bash
#SBATCH --job-name=nemo_dtcmp
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo_dtcmp_%j.out
# NEMO vs NEMO, oracle-side test of two harmonization rows (2026-09-26):
#   RUN_HOURLY      rn_Dt=3600 nn_fsbc=4   (the production step, 4-h block-mean fluxes)
#   RUN_DT150       rn_Dt=150  nn_fsbc=96  (our step, same 4-h block fluxes)   -> isolates dt
#   RUN_DT150_FSBC1 rn_Dt=150  nn_fsbc=1   (our step, per-step fluxes)        -> isolates forcing cadence
# Box 220-240E |lat|<=2: hourly turbocline (mldkz5), mldr10_1, tos; T profile at hours 4/12/24/47;
# avt profile at hour 12. Hourly output files are window means in every run (same 1h file_def).
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
RUNS=${RUNS:-"RUN_HOURLY RUN_DT150 RUN_DT150_FSBC1"}
$PY - $RUNS <<PYEOF
import sys, netCDF4 as nc, numpy as np
runs = sys.argv[1:]
def box_of(f):
    la = f.variables["nav_lat"][:]; lo = f.variables["nav_lon"][:] % 360.0
    return (np.abs(la) <= 2.0) & (lo >= 220.0) & (lo <= 240.0)
T = {r: nc.Dataset(f"$B/{r}/ORCA1_1h_20000101_20000102_grid_T_1h.nc") for r in runs}
W = {r: nc.Dataset(f"$B/{r}/ORCA1_1h_20000101_20000102_grid_W_1h.nc") for r in runs}
box = box_of(T[runs[0]])
def bm(v, h): return float(np.ma.masked_invalid(v[h])[box].mean())
n = min(T[r].variables["mldkz5"].shape[0] for r in runs)
print("hourly box means; columns per run:", runs)
print("hour local | turbocline mldkz5 | mldr10_1 | tos")
for h in range(n):
    a = [bm(T[r].variables["mldkz5"], h) for r in runs]; b = [bm(T[r].variables["mldr10_1"], h) for r in runs]; c = [bm(T[r].variables["tos"], h) for r in runs]
    print(f"{h:3d} {(h+15.3)%24:04.1f} | " + " ".join(f"{x:6.2f}" for x in a) + " | " + " ".join(f"{x:6.2f}" for x in b) + " | " + " ".join(f"{x:7.3f}" for x in c))
dep = np.asarray(T[runs[0]].variables["deptht"][:]).ravel()
for h in (3, 11, 23, 46):
    if h >= n: continue
    print(f"\n## T profile, hour {h} (rec {h}, local {(h+15.3)%24:04.1f}); columns: " + " ".join(runs) + " | diffs vs " + runs[0])
    prof = [np.array([float(np.ma.masked_invalid(T[r].variables["to"][h, k])[box].mean()) for k in range(16)]) for r in runs]
    for k in range(16):
        print(f"{dep[k]:6.2f} m | " + " ".join(f"{p[k]:7.3f}" for p in prof) + " | " + " ".join(f"{p[k]-prof[0][k]:+.3f}" for p in prof[1:]))
depw = np.asarray(W[runs[0]].variables["depthw"][:]).ravel()
for h in (3, 11):
    if h >= n: continue
    print(f"\n## avt (window mean, post-EVD) and avt_k/avm_k if present, hour {h}; columns: " + " ".join(runs))
    for k in range(1, 17):
        row = []
        for r in runs:
            v = W[r].variables
            row.append(f"{bm(v['avt'], h) if False else float(np.ma.masked_invalid(v['avt'][h, k])[box].mean()):9.2e}")
            for nm in ("avt_k", "avm_k"):
                if nm in v: row.append(f"{float(np.ma.masked_invalid(v[nm][h, k])[box].mean()):9.2e}")
        print(f"{depw[k]:6.2f} m | " + " ".join(row))
PYEOF
echo "=== card exit: $? ==="
