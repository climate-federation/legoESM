#!/bin/bash
#SBATCH --job-name=euctest
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/euctest_%j.out
#
# DOES THE TRANSPORT DIAGNOSTIC ACTUALLY RESPOND? GLM'S LAST OPEN OBJECTION.
#
# The undercurrent deficit is the session's main surviving claim, and its
# defence is that two runs of one configuration reproduce the 220E transport
# to the printed digit. GLM pointed out the obvious alternative: a diagnostic
# that reproduces because it is INSENSITIVE reproduces just as well as one
# that reproduces because the physics agrees. Four of six longitudes matching
# to four significant figures is consistent with both.
#
# Partial answers already exist and are not enough. The 260E channel moves
# between the two runs and its floor grew from 0.01 to 0.43 m2/s across ten
# days, so the diagnostic is not frozen and is not reading a shared file. But
# "it moves when the ocean moves" is weaker than "it moves by the right
# amount when the ocean moves by a known amount", and only the second licenses
# quoting a 64 m2/s deficit.
#
# THE TEST, AND ITS ARITHMETIC IS THE WHOLE POINT. The probe reports the net
# zonal transport as the integral of u over 0-400 m in a |lat|<=1 box. Add a
# UNIFORM known velocity to u everywhere inside that box and depth range, and
# the reported transport must change by exactly that velocity times the depth
# span: 0.01 m/s over 400 m is 4.00 m2/s. Nothing about the ocean state enters
# the prediction, so a mismatch is a defect in the instrument and not a fact
# about the model.
#
# PRE-REGISTERED:
#   * every longitude shifts by 4.00 +- 0.05 m2/s -> the diagnostic is
#     calibrated, the deficit is a real measurement, GLM's objection closes.
#   * shifts are proportional but the wrong size -> a depth-span or
#     unit-conversion error in the probe; the RATIO gives the factor.
#   * some longitudes shift and others do not -> the box or the mask differs
#     by longitude and the section is not what it claims to be.
#   * nothing shifts -> the diagnostic is insensitive and every transport
#     number this session has quoted is void.
# The perturbed copy is written to scratch; the original snapshot is READ
# ONLY and never modified.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
MESH=/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc
SRC=$R/trp_faithz0_d30/snapshot_day0030.npz
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_euctest_work
DU=0.01
for f in "$SRC" "$MESH" ${STEM}_grid_T.nc ${STEM}_grid_U.nc ${STEM}_grid_V.nc ${STEM}_grid_W.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$WORK" || exit 1

$PY -u - "$SRC" "$WORK/perturbed.npz" "$DU" <<'EOF'
import sys
import numpy as np

src, dst, du = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = dict(np.load(src, allow_pickle=True))
lat = np.asarray(d["lat_T"], dtype=np.float64)
u = np.asarray(d["u"], dtype=np.float64)
z = np.asarray(d["z_center_ref"], dtype=np.float64)
print(f"u shape {u.shape}; lat shape {lat.shape}; {len(z)} levels, "
      f"z from {z.min():.1f} to {z.max():.1f} m")

# The band the probe integrates over: |lat| <= 1 and the top 400 m.
in_band = np.abs(lat) <= 1.0
in_depth = np.abs(z) <= 400.0
print(f"perturbing {int(in_band.sum())} columns x {int(in_depth.sum())} "
      f"levels by du = {du} m/s")
assert in_band.any() and in_depth.any(), "empty perturbation band"

pert = u.copy()
pert[in_band[..., None] & in_depth[None, None, :]] += du
moved = int(np.sum(pert != u))
print(f"cells actually changed: {moved}")
assert moved > 0, "perturbation changed nothing -- band indexing is wrong"

d["u"] = pert
np.savez(dst, **d)
print(f"wrote {dst}")
EOF
rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: perturbation step failed rc=$rc"; exit $rc; }

for pair in "baseline:$SRC" "perturbed:$WORK/perturbed.npz"; do
  TAG=${pair%%:*}; SNAP=${pair#*:}
  echo "############ $TAG ############"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt ${STEM}_grid_T.nc \
    --nemo-t-recs "5:6" --label "resp_$TAG" \
    --nemo-ufile ${STEM}_grid_U.nc --nemo-vfile ${STEM}_grid_V.nc \
    --mesh-mask "$MESH" 2>&1 | grep -A 8 'NET ZONAL TRANSPORT'
  r=${PIPESTATUS[0]}; echo "### $TAG rc=$r"; [ "$r" -eq 0 ] || rc=$r
done
echo "PREDICTED SHIFT: du * 400 m = $(echo "$DU" | awk '{printf "%.2f", $1*400}') m2/s at every longitude."
echo "=== card exit: $rc ==="
exit $rc
