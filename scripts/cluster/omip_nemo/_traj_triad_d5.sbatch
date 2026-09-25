#!/bin/bash
#SBATCH --job-name=traj_triad
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/traj_triad_%j.out
# Is the 3e-2 K day-5 difference arm-vs-control MY accumulator (must be
# read-only) or pre-existing run-to-run drift? Three runs, same command line:
#   pinned  = trp_pinned_d30      (2026-09-21 tree, 5-day cadence)
#   phase4  = trp_phase4_d5       (eb06dac06 tree, 6-hourly cadence)
#   arm     = trp_statemean_d30   (60919bf31 tree, --state-accumulate)
# phase4==pinned and arm!=phase4  -> my commits changed the trajectory (DEFECT)
# phase4!=pinned                  -> drift predates the accumulator
set -u
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python - <<PYEOF
import numpy as np, json
runs={"pinned":"$R/trp_pinned_d30/snapshot_day0005.npz","phase4":"$R/trp_phase4_d5/snapshot_day0005.npz","arm":"$R/trp_statemean_d30/snapshot_day0005.npz"}
T={k:np.nan_to_num(np.load(v)["T"]) for k,v in runs.items()}
for k,v in runs.items():
    try: m=json.load(open(v.rsplit("/",1)[0]+"/run_manifest.json")); print(k, "git:", m.get("legoESM",{}).get("git_sha", m.get("reproducibility",{}).get("git_sha","?")), "| created", m["run"]["creation_time"][:16])
    except Exception as e: print(k, "manifest?", e)
names=list(T)
for i in range(3):
    for j in range(i+1,3):
        d=np.abs(T[names[i]]-T[names[j]]); k=np.unravel_index(d.argmax(), d.shape)
        print("%-7s vs %-7s max|dT| = %.3e  cells>1e-6: %d  >1e-3: %d  argmax %s"%(names[i],names[j],d.max(),(d>1e-6).sum(),(d>1e-3).sum(),k))
PYEOF
echo "=== card exit: $? ==="
