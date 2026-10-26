#!/bin/bash
#SBATCH --job-name=midhealth
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/midhealth_%j.out
# Cheap mid-run health check on the newest snapshot of the adopted-card
# baseline. The card buffers its stdout through `tail`, so snapshots are the
# only progress signal. PRE-REGISTERED: a blowup means killing the run now
# rather than waiting ~90 more minutes for a dead result.
#   REFUTES health: any NaN in T or S, |SST| outside -3..40 C, or SSS outside
#                   0..45 on WET cells.
#   CONFIRMS health: all inside range, and the fields are not frozen at the
#                    initial condition (day-5 and day-10 must DIFFER).
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PY'
import numpy as np, glob, os
R = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faith_d30"
snaps = sorted(glob.glob(os.path.join(R, "snapshot_day*.npz")))
print("snapshots:", [os.path.basename(s) for s in snaps])
if len(snaps) < 2:
    raise SystemExit("need two snapshots to check the run is advancing")
def load(p):
    d = np.load(p)
    return np.asarray(d["T"], float), np.asarray(d["S"], float), \
        np.asarray(d["land_mask"], float)
a_T, a_S, mask = load(snaps[-2])
b_T, b_S, _ = load(snaps[-1])
wet = mask > 0.5
sst = b_T[..., 0][wet]; sss = b_S[..., 0][wet]
print(f"\nday {os.path.basename(snaps[-1])}: {wet.sum()} wet cells")
print(f"  SST  {sst.min():.3f} .. {sst.max():.3f} C   NaN={np.isnan(sst).sum()}")
print(f"  SSS  {sss.min():.3f} .. {sss.max():.3f}     NaN={np.isnan(sss).sum()}")
bad = (np.isnan(b_T).any() or np.isnan(b_S).any()
       or sst.min() < -3.0 or sst.max() > 40.0
       or sss.min() < 0.0 or sss.max() > 45.0)
moved = float(np.abs(b_T - a_T).max())
print(f"  max |dT| between the last two snapshots: {moved:.4f} C")
print("\nVERDICT:", "SICK -- consider killing the run" if bad
      else ("FROZEN -- fields are not advancing" if moved < 1e-9
            else "HEALTHY so far"))
PY
echo DONE
