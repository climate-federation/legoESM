#!/bin/bash
#SBATCH --job-name=eucforce
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eucforce_%j.out
#
# WHAT HOLDS THE UNDERCURRENT UP, OURS AGAINST THE ORACLE, AT TWELVE MARKS.
#
# WHY. Our Pacific undercurrent runs at roughly 0.55 of NEMO's core speed at
# every mark from day 10 to day 90. The twelve-mark series (9944090-9944101)
# showed that ratio FLAT, and I read flatness as "spin-up refuted". GLM-5.2
# refuted the reading: the undercurrent's transport is a basin-scale mass
# balance needing western-boundary reflection and Rossby propagation, order
# six to twelve months, so ninety days is INSIDE the adjustment window and two
# runs with different effective mixing look exactly like this mid-transient.
# Retracted in 7fcce5615.
#
# A FORCE BALANCE DOES NOT HAVE THAT PROBLEM. It asks which term is wrong
# rather than how far the jet has got, so it reads the same at day 35 as at
# day 400. At the equator the Coriolis term vanishes and the jet is held up by
# an eastward pressure gradient against the downward transfer of westward
# surface momentum by vertical friction. Both terms are computable offline
# from files already on disk: our snapshots carry K_M_diag, and the oracle's
# grid_W publishes avm.
#
# PRE-REGISTERED READING, on the CORE level at 220E, required to hold across
# the twelve marks and not merely at one:
#   pressure gradients agree, our FRIC markedly more negative -> excess
#     vertical friction; structural; spin-up independent
#   our pressure gradient markedly weaker -> density/thermocline structure
#   both agree and RESID differs -> advection or lateral friction, AND ONLY
#     AFTER du/dt HAS BEEN SUBTRACTED. Advection belongs to the STATIONARY
#     undercurrent balance too, so "it is advection, therefore spin-up,
#     therefore no verdict" is a false exoneration. The du/dt block below
#     removes the local acceleration using the five-day spacing of the marks
#     on BOTH sides, which is what makes the remainder readable at all.
#
# EOS, AND A FINDING IT SURFACES. Our run integrates with WRIGHT (rho_0 1025,
# from the run manifest); ORCA1 runs TEOS-10 (namelist_cfg: ln_teos10=.true.),
# whose Roquet polynomial this repo already implements as "nemo_teos10". The
# default arm therefore gives each model the EOS it actually felt. The second
# arm applies NEMO's EOS to BOTH sides' T and S, which asks instead whether
# the density FIELDS differ; the gap between the two arms is the part of any
# pressure-gradient difference that is EOS formulation rather than ocean
# state. Nothing here changes a default -- the mismatch is REPORTED, not
# fixed.
#
# THE SAMPLING MISMATCH IS NOT FIXABLE HERE AND IS NOT HIDDEN. Our snapshots
# are INSTANTANEOUS; the oracle publishes FIVE-DAY MEANS, and the mean of a
# product is not the product of the means, so NEMO's own residual will not
# close even on a healthy oracle. Consequences built in: the probe gates only
# on terms LINEAR in the means, and no single mark is reportable.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_d90
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
MESH=/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_eucforce
mkdir -p "$WORK" || exit 1

# THE PROBE'S OWN OPERATORS FIRST. A derivative that is wrong by a metric
# factor produces a plausible number, not an error, so the manufactured
# solutions gate the real run: if they are red, nothing below is worth
# reading. Two of them exist only to show the others can fail.
echo "### manufactured-solution gate"
$PY -m pytest tests/validate/test_euc_force_balance.py -q 2>&1 | tail -15
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: probe operators failed their own tests"; exit 1; }

DAYS="35 40 45 50 55 60 65 70 75 80 85 90"
run_arm () {                      # $1 = tag, $2.. = extra probe flags
  local tag=$1; shift
  local fail=0
  for D in $DAYS; do
    local REC=$(( D / 5 - 1 ))
    local SNAP="$R/snapshot_day$(printf '%04d' $D).npz"
    # A run that REACHES its target writes that day as snapshot_final.npz.
    [ -r "$SNAP" ] || SNAP="$R/snapshot_final.npz"
    [ -r "$SNAP" ] || { echo "FATAL: no snapshot for day $D"; fail=1; continue; }
    echo "######## $tag day $D (oracle rec $REC) ########"
    $PY -u scripts/validate/ocean_fidelity/euc_force_balance.py \
      --legoesm-snapshot "$SNAP" \
      --nemo-gridt ${STEM}_grid_T.nc --nemo-gridu ${STEM}_grid_U.nc \
      --nemo-gridw ${STEM}_grid_W.nc --mesh-mask "$MESH" \
      --rec "$REC" --expect-day "$D" --lons 200,220,240 \
      --dump "$WORK/${tag}_d$(printf '%04d' $D).npz" "$@" 2>&1 | tail -45
    local r=${PIPESTATUS[0]}
    [ "$r" -eq 0 ] || { echo "### $tag day $D rc=$r"; fail=1; }
  done
  return $fail
}

echo; echo "############ ARM 1: each model's OWN eos (the force each FELT)"
run_arm felt; a1=$?
echo; echo "############ ARM 2: NEMO's eos on BOTH (the density FIELD question)"
run_arm field --eos-both nemo_teos10; a2=$?

# THE du/dt BLOCK. The probe's RESID lumps advection, lateral friction and the
# local acceleration. The marks are five days apart on BOTH sides, so du/dt at
# the core is a centred difference over ten days -- and subtracting it is what
# turns "the residuals differ" from an unreadable number into a statement
# about advection and lateral friction. Both sides get the SAME operator over
# the SAME interval; an acceleration differenced across mismatched windows is
# the confound this campaign has already paid for once.
echo; echo "############ core-level summary across the twelve marks"
$PY -u - "$WORK" <<'EOF'
import glob
import os
import sys

import numpy as np

work = sys.argv[1]
for arm in ("felt", "field"):
    fs = sorted(glob.glob(os.path.join(work, f"{arm}_d*.npz")))
    if not fs:
        print(f"\n[{arm}] no dumps"); continue
    print(f"\n=== {arm}: 220E core, m/s^2 (u in m/s) ===")
    print(f"{'day':>4} {'u_o':>7} {'u_n':>7} {'ratio':>6} | "
          f"{'PGF_o':>10} {'PGF_n':>10} | {'FRIC_o':>10} {'FRIC_n':>10} | "
          f"{'RES_o':>10} {'RES_n':>10}")
    days, rec = [], {}
    for f in fs:
        d = np.load(f)
        if "220_ours_core" not in d.files:
            print(f"  MISSING core row in {os.path.basename(f)}"); continue
        day = float(d["day"][0])
        o, n = d["220_ours_core"], d["220_nemo_core"]
        # core row: [z, u, pgf_baro, pgf_bc, fric, resid]
        days.append(day); rec[day] = (o, n)
        print(f"{day:4.0f} {o[1]:7.3f} {n[1]:7.3f} "
              f"{(o[1]/n[1] if n[1] else float('nan')):6.2f} | "
              f"{o[2]+o[3]:10.3e} {n[2]+n[3]:10.3e} | "
              f"{o[4]:10.3e} {n[4]:10.3e} | {o[5]:10.3e} {n[5]:10.3e}")
    days.sort()
    if len(days) >= 3:
        print(f"\n  du/dt at the core, centred over 10 days, and the residual "
              f"with it REMOVED:")
        print(f"{'day':>4} {'dudt_o':>10} {'dudt_n':>10} | "
              f"{'RES-dudt_o':>11} {'RES-dudt_n':>11}")
        for i in range(1, len(days) - 1):
            dt = (days[i + 1] - days[i - 1]) * 86400.0
            o0, n0 = rec[days[i - 1]]; o2, n2 = rec[days[i + 1]]
            oi, ni = rec[days[i]]
            do, dn = (o2[1] - o0[1]) / dt, (n2[1] - n0[1]) / dt
            print(f"{days[i]:4.0f} {do:10.3e} {dn:10.3e} | "
                  f"{oi[5]-do:11.3e} {ni[5]-dn:11.3e}")
    print("  NEMO's residual is NOT expected to close: it is a 5-day mean and "
          "mean(Kv du/dz) != mean(Kv) d(mean u)/dz. Read the OURS-vs-NEMO "
          "difference in each term, never the absolute closure.")
EOF
rc2=$?
echo "=== card exit: arm1=$a1 arm2=$a2 summary=$rc2 ==="
[ "$a1" -eq 0 ] && [ "$a2" -eq 0 ] && [ "$rc2" -eq 0 ] || exit 1
exit 0
