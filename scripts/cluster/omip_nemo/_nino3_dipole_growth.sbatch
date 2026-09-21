#!/bin/bash
#SBATCH --job-name=n3growth
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n3growth_%j.out
#
# Does the nino3 vertical dipole GROW, or was it there on day 5?
#
# GLM's pre-check: if the two runs started from different states, a day-30
# difference is adjustment, not a defect. We start from NEMO's own Dec/Jan
# blended initial condition, so the answer should be a dipole near zero early
# that grows -- but that is a claim, and this measures it.
#
# PRE-REGISTERED:
#   dipole ~0 at day 5 and growing  -> a running model defect; IC parity holds.
#   dipole already large at day 5   -> initial-condition or spin-up offset;
#                                      the day-30 attribution is void.
#   dipole flat across all days     -> an offset, not an accumulation.
# Both sides, one code path, same mesh metrics and same bins.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# RUN/OUT are overridable so the SAME code path, mesh and bins can score an
# ARM without a second copy of this card -- the defaults are the baseline's
# own values, so an unparameterised submission runs exactly what it always
# ran.  To score the convection arm once its legs finish:
#   RUN=$MAIN/results/omip_nemo/trp_faithevd_d30 \
#   OUT=$MAIN/results/omip_nemo/nino3_dipole_faithevd.json \
#   sbatch scripts/cluster/omip_nemo/_nino3_dipole_growth.sbatch
# Then compare that JSON against nino3_dipole_growth.json layer by layer; the
# two are directly comparable BECAUSE they came through this one card.
RUN=${RUN:-$MAIN/results/omip_nemo/trp_orca1faithtint_d30}
NEMO=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
OUT=${OUT:-$MAIN/results/omip_nemo/nino3_dipole_growth.json}
echo "RUN=$RUN"
echo "OUT=$OUT"
# DAYS is the scoring window. The DEFAULT IS THE BASELINE'S OWN SIX RECORDS, so
# an unparameterised submission runs exactly what it always ran. It is
# overridable because the convection arm lost its day-15 snapshot: each leg
# writes snapshot_final.npz, and leg 2 overwrote leg 1's copy before it was
# renamed. Scoring the days that exist through this ONE card keeps the arm and
# the baseline comparable at every day they share; a second card would not.
# Both sides are paired day-for-day -- the same list drives our snapshots and
# NEMO's records, so a window can never be compared against a different window.
DAYS=${DAYS:-5,10,15,20,25,30}
SNAPS=$(for d in ${DAYS//,/ }; do printf "%s/snapshot_day%04d.npz " "$RUN" "$d"; done)
echo "DAYS=$DAYS"
for s in $SNAPS; do
  [ -f "$s" ] || { echo "REFUSING: $s does not exist -- scoring a window whose"; \
                   echo "          snapshots are absent would silently mis-pair days."; exit 3; }
done
$PY scripts/validate/ocean_fidelity/global_tracer_content.py \
    --snapshot $SNAPS \
    --nemo-gridt $NEMO --nemo-days $DAYS \
    --mesh-mask $MAIN/data/grids/eORCA1.2_mesh_mask.nc \
    --region=-5,5,-150,-90 \
    --depth-bin-edges 0,2,10,50,200,1000,inf \
    --json-out $OUT
rc=$?; echo "=== rc=$rc ==="; [ $rc -ne 0 ] && exit $rc
$PY - "$OUT" <<'PYEOF'
import json,sys
r=json.load(open(sys.argv[1]))
H="heat_by_depth_bin_C_m3"; V="volume_by_depth_bin_m3"
ours=[r["snapshots"][k] for k in sorted(r["snapshots"])]
nem=[r["nemo"][k] for k in sorted(r["nemo"])]
if len(ours)!=len(nem): raise SystemExit("FATAL: unequal day counts")
days=[int(k.replace("day","")) for k in sorted(r["nemo"])]
layers=[k for k in ours[0][H] if ours[0][V].get(k,0)>0]
print("nino3 layer-mean T BIAS vs NEMO [degC], by day")
print(f"{'layer':12s}" + "".join(f"{d:>9d}" for d in days))
for L in layers:
    row=[o[H][L]/o[V][L]-n[H][L]/n[V][L] for o,n in zip(ours,nem)]
    print(f"{L:12s}" + "".join(f"{v:+9.3f}" for v in row))
row=[o["mean_T_C"]-n["mean_T_C"] for o,n in zip(ours,nem)]
print(f"{'column':12s}" + "".join(f"{v:+9.3f}" for v in row))
PYEOF
