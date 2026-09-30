#!/bin/bash
#SBATCH --job-name=nanconf
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nanconf_%j.out
# CONFIRM the NaN before acting. My probes have been at fault eight times
# today, so check the raw arrays without any mask logic, and check BOTH
# snapshots plus a KNOWN-GOOD reference snapshot through the SAME code path.
# If the reference comes back clean and the new run comes back NaN, the probe
# is exonerated and the run is genuinely sick.
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PY'
import numpy as np
NEW = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faith_d30/"
REF = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
       "trp_base2_d30/snapshot_day0030.npz")
for label, p in (("NEW day05", NEW + "snapshot_day0005.npz"),
                 ("NEW day10", NEW + "snapshot_day0010.npz"),
                 ("REF base2 day30", REF)):
    d = np.load(p)
    T = np.asarray(d["T"], float); S = np.asarray(d["S"], float)
    print(f"\n{label}  T{T.shape}")
    print(f"  T: NaN {np.isnan(T).sum():>9d} of {T.size}   "
          f"finite range {np.nanmin(T) if np.isfinite(T).any() else float('nan'):.3f}"
          f" .. {np.nanmax(T) if np.isfinite(T).any() else float('nan'):.3f}")
    print(f"  S: NaN {np.isnan(S).sum():>9d} of {S.size}")
    for k in ("u", "v"):
        if k in d:
            a = np.asarray(d[k], float)
            print(f"  {k}: NaN {np.isnan(a).sum():>9d} of {a.size}")
PY
echo DONE
