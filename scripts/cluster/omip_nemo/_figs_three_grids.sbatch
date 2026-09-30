#!/bin/bash
#SBATCH --job-name=figs3grid
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/figs3grid_%j.out
#
# THE FULL FIGURE SET, THE SAME ONE, FOR ALL THREE GRIDS.
#
# Reproduces for tripole / MPAS / FESOM2 exactly the panels that exist for a
# single arm today, each against the SAME oracle window:
#   T and S horizontal depth snapshots (0/100/300/1000 m)  [battery]
#   T and S zonal-mean sections                            [battery]
#   SST / SSS / MLD map triptychs + zonal means            [compare_omip_nemo]
#   SSH vs NEMO zos (geostrophic surface circulation)      [_diag_ssh_rec]
#
# ORACLE: GATEWAY 5-day grid_T, 0-based record DAY/5-1 = the window ENDING at
# the snapshot day (record 5 = days 25-30 for day 30), the campaign convention
# fixed in ea58d2473.  NOT the RUN_REF monthly climatology the older one-arm
# figures used -- that is a 5-year January mean of a spun-up NEMO and is not
# comparable to a 30-day cold start.
#
# CONFIGURATION CAVEAT, carried with every number these figures show: the
# tripole run is the PRODUCTION card (GM/Redi Treguier set, SI3 exchange set,
# NEMO domain mask, Dec/Jan initial-condition blend) while the MPAS and FESOM
# runs are the pre-adoption base2 cards.  Differences between the three panels
# are configuration + grid together, not grid alone.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
DAY=$(( 10#${DAY:-30} ))
[ $(( DAY % 5 )) -eq 0 ] && [ "$DAY" -ge 5 ] || { echo "FATAL: DAY=$DAY is not a positive multiple of 5"; exit 1; }
REC=$(( DAY / 5 - 1 ))
OUT=$R/figs_grids_d${DAY}_${TAG:-harmonized}
echo "=== oracle record $REC (0-based) = days $((DAY-5))-$DAY ==="

# Run directories, overridable so this card follows the campaign instead of
# naming one generation of arms.  The defaults are the HARMONIZED runs (each
# lane carrying every production flag it threads); the pre-adoption base2 dirs
# are still scorable by passing them in.
TRPDIR=${TRPDIR:-trp_orca1faithtint_d30}
MPASDIR=${MPASDIR:-mpas_orca1faithtint_d30}
FESDIR=${FESDIR:-fesom_orca1faithtint_d30}
SNAP=snapshot_day$(printf %04d $DAY).npz

# label : run directory : snapshot file
ARMS=(
  "tripole:$TRPDIR:$SNAP"
  "MPAS:$MPASDIR:$SNAP"
  "FESOM2:$FESDIR:$SNAP"
)

$PY - <<PY || exit 1
import netCDF4 as nc
ds = nc.Dataset("$GW")
print("[gateway] variables:", sorted(v for v in ds.variables
                                     if v in ("tos","sos","mldr10_1","zos")))
missing = [v for v in ("tos","sos","mldr10_1","zos") if v not in ds.variables]
if missing:
    print("FATAL: GATEWAY grid_T lacks", missing); raise SystemExit(1)
PY

# COMBINED FIGURES (default): tripole and MPAS drawn in the SAME panels --
# model1 | model2 | NEMO | delta1 | delta2 for T/S depth snapshots and the
# zonal sections, both models in one SSH figure, and the three-way scorer's
# *_maps3.png / zonal_3way.png for SST/SSS/MLD (it already draws two models
# plus NEMO on one shared cell set, with FESOM2 passed as --also-mask so the
# cell set matches the three-grid scorecard).
if [ "${COMBINE:-1}" = 1 ]; then
  TRP=$R/$TRPDIR/$SNAP
  MPS=$R/$MPASDIR/$SNAP
  FES=$R/$FESDIR/$SNAP
  for f in "$TRP" "$MPS" "$FES"; do
    [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
  done
  C=$OUT/tripole_MPAS; mkdir -p "$C"
  echo; echo "########## COMBINED tripole + MPAS ##########"
  ls -l --time-style=+%Y-%m-%d_%H:%M "$TRP" "$MPS"

  $PY -u scripts/validate/diag_omip_nemo_battery.py \
    --legoesm-snapshot "$TRP" --grid-label tripole \
    --legoesm-snapshot-2 "$MPS" --grid-label-2 MPAS \
    --nemo-gridt "$GW" --nemo-time-idx "$REC" --output-dir "$C" 2>&1 \
    | grep -vE "Warning|warn" | tail -20
  echo "  combined battery rc=$?"

  $PY -u scripts/tmp/_diag_ssh_rec.py "$TRP" "$GW" "$REC" tripole "$C/ssh.png" \
    rec "$MPS" MPAS 2>&1 | tail -4
  echo "  combined ssh rc=$?"

  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$TRP" --label-tripole tripole \
    --mpas "$MPS" --label-mpas MPAS --also-mask "$FES" --expect-day "$DAY" \
    --nemo-gridt "$GW" --nemo-time-idx "$REC" --res-deg 1.0 \
    --out-dir "$C" 2>&1 | grep -vE "Warning|warn" | tail -12
  echo "  three-way maps rc=$?"
fi

for a in "${ARMS[@]}"; do
  [ "${PERGRID:-0}" = 1 ] || break
  LAB=${a%%:*}; rest=${a#*:}; DIR=${rest%%:*}; SNAPNAME=${rest#*:}
  SNAP=$R/$DIR/$SNAPNAME
  [ -r "$SNAP" ] || { echo "FATAL: missing $SNAP"; exit 1; }
  O=$OUT/$LAB; mkdir -p "$O"
  echo; echo "########## $LAB  ($DIR/$SNAPNAME) ##########"
  ls -l --time-style=+%Y-%m-%d_%H:%M "$SNAP"

  $PY -u scripts/validate/diag_omip_nemo_battery.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GW" --nemo-time-idx "$REC" \
    --grid-label "$LAB" --output-dir "$O" 2>&1 | grep -vE "Warning|warn" | tail -20
  echo "  battery rc=$?"

  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GW" --nemo-time-idx "$REC" \
    --sss-faithful --grid-label "$LAB" --output-dir "$O" 2>&1 \
    | grep -vE "Warning|warn" | tail -25
  echo "  maps rc=$?"

  $PY -u scripts/tmp/_diag_ssh_rec.py "$SNAP" "$GW" "$REC" "$LAB" "$O/ssh.png" rec \
    2>&1 | tail -3
  echo "  ssh rc=$?"
done

echo; echo "=== figures written ==="
ls -1 $OUT/*/*.png
echo "OUT=$OUT"
echo DONE
