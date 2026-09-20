#!/bin/bash
#SBATCH --job-name=nino3nemo
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3nemo_%j.out
#
# Does NEMO warm its OWN top two metres in nino3 over days 25->30?
#
# Our d25->d30 nino3 split was: 0-2 m +0.415 C, 2-10 m +0.033, 10-50 m -0.283,
# 50-200 m -0.056, and the column LOST heat.  That was read as "heat failing to
# leave the surface downward".  It is only a defect if the oracle does not do
# the same thing.  One probe, one code path, one set of mesh metrics and depth
# bins for both sides.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUN=$MAIN/results/omip_nemo/trp_orca1faithtint_d30
NEMO=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY scripts/validate/ocean_fidelity/global_tracer_content.py \
    --snapshot $RUN/snapshot_day0025.npz $RUN/snapshot_day0030.npz \
    --nemo-gridt $NEMO --nemo-days 25,30 \
    --mesh-mask $MAIN/data/grids/eORCA1.2_mesh_mask.nc \
    --region=-5,5,-150,-90 \
    --depth-bin-edges 0,2,10,50,200,1000,inf \
    --json-out $MAIN/results/omip_nemo/nino3_bins_vs_nemo.json
rc=$?
echo "=== rc=$rc ==="
[ $rc -ne 0 ] && exit $rc
$PY - <<'PYEOF'
import json
r=json.load(open("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nino3_bins_vs_nemo.json"))
# Key names READ from the report, not guessed: the first version of this
# block invented "heat_bins_C_m3" and printed an EMPTY table that looked like
# a successful run.
H="heat_by_depth_bin_C_m3"; V="volume_by_depth_bin_m3"
snaps=[r["snapshots"][k] for k in sorted(r["snapshots"])]
nem=[r["nemo"][k] for k in sorted(r["nemo"])]
def dT(a,b):
    return {k:(b[H][k]/b[V][k]-a[H][k]/a[V][k]) for k in a[H]
            if a[V].get(k,0)>0 and b[V].get(k,0)>0}
o=dT(snaps[0],snaps[1]); n=dT(nem[0],nem[1])
if not o: raise SystemExit("FATAL: empty depth-bin table -- key names drifted")
print("nino3 mean T change, day 25 -> day 30 [degC]")
print(f"{'layer':12s} {'ours':>9s} {'NEMO':>9s} {'ours-NEMO':>10s}")
for k in o: print(f"{k:12s} {o[k]:+9.4f} {n[k]:+9.4f} {o[k]-n[k]:+10.4f}")
print("\nabsolute layer mean T [degC], day 30")
print(f"{'layer':12s} {'ours':>9s} {'NEMO':>9s} {'bias':>10s}")
for k in o:
    a=snaps[1][H][k]/snaps[1][V][k]; b=nem[1][H][k]/nem[1][V][k]
    print(f"{k:12s} {a:9.4f} {b:9.4f} {a-b:+10.4f}")
PYEOF
