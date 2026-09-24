#!/bin/bash
#SBATCH --job-name=trpevdeq
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpevdeq_%j.out
#
# THE ORACLE MIXES ITS TOP METRE AND WE DO NOT. USER-APPROVED ARM.
#
# Measured on the oracle's own archive, equatorial Pacific 200-260E, |lat|<=2,
# day 30, same reduction both sides -- heat diffusivity by depth, ours against
# NEMO's: 6.34e-04 vs 3.750e+01 at 1.0 m, 4.71e-04 vs 1.667e+01 at 2.1 m,
# 2.86e-04 vs 3.335e+00 at 3.3 m. Five orders of magnitude at the top
# interface.
#
# AND ITS SOURCE IS PINNED WITH NO FREE PARAMETER. NEMO's top viscosity is
# 7.619e-03. If that diffusivity came from the turbulence closure it would
# satisfy avt = avm/Pr with Pr clamped to [1,10] (zdftke.F90:399), so it could
# not exceed 7.619e-03. The archived value is 37.5. The closure cannot produce
# it. The only ORCA1 term that writes a tracer diffusivity outside that band is
# ln_zdfevd=.true. with rn_evd=100 and nn_evdm=0 -- enhanced vertical diffusion
# on TRACERS ONLY, which also explains the shape: NEMO's top viscosity is
# ordinary while its diffusivity is four orders larger.
#
# WE RUN --convection none, by a default whose own help says it "preserves the
# validated faithful config". No OMIP card sets it. So the oracle runs a
# scheme we do not run at all, in exactly the layer where our error lives.
#
# WHAT THE ERROR LOOKS LIKE, measured today, box mean 2S-2N 220-240E at day 30,
# ours minus NEMO by depth: +2.64 K at 0.5 m, +2.53 at 2, +1.96 at 5, +1.09 at
# 10, +0.42 at 15, -0.08 at 20, -0.78 at 30, -1.11 at 40 and 50. A twenty-metre
# warm lens sitting on a cold layer. The top-to-20 m contrast is 3.49 K against
# the oracle's 0.76, and it GROWS -- 1.93 against 0.31 at day 15. All three
# grids show it (3.49 / 2.87 / 2.83), so it is shared column physics, not mesh.
# EVD fires where a column is statically unstable, which in the tropics is at
# night under surface cooling: the oracle erases its warm layer nightly and we
# accumulate ours. That is the growth.
#
# ONE VARIABLE, AND THE CONTROL ALREADY EXISTS. Built programmatically from
# _arm_trp_pinned_d30.pre_fwadopt.txt -- the command line that produced the
# trp_pinned_d30 snapshots on 2026-09-21, BEFORE both of today's adoptions --
# with the output redirected and seven convection flags appended, 119 tokens to
# 133, asserted at build time along with "exactly one --output" and "the base
# line carries neither adoption". So this arm differs from the existing control
# in the convection scheme and nothing else. It therefore tests the scheme on
# the PRE-adoption baseline, which is the correct baseline for attribution and
# is NOT the current production configuration.
#
# THE SETTINGS ARE NEMO'S, NOT CHOSEN:
#   --convection-K-conv 100.0        rn_evd = 100
#   --convection-n2-mode nemo_bn2    NEMO's eosbn2 rn2, not a smoothed in-situ N^2
#   --convection-n2-eos teos10       ORCA1 runs ln_teos10
#   --convection-trigger hard        NEMO's switch, not the sigmoid default
#   --convection-n2-threshold -1e-12 zdfevd's threshold
#   --convection-K-bg 0.0            so STABLE columns are untouched. The
#     default 1e-5 would set the diffusivity where the scheme does NOT fire,
#     and our measured value at 24.6 m is 1.77e-06 -- six times smaller -- so
#     the default would have changed the whole stable ocean and made this a
#     two-variable run. zdfevd only modifies avt where the column is unstable;
#     zero reproduces that whether the scheme's profile is added or superseded.
#   NOT --convection-two-level: the driver refuses it, this integrator has no
#     leap-frog outer step, so NEMO's MIN(rn2, rn2b) has no analogue here.
# Every previous enhanced-diffusion arm in this repo was ARCTIC and ran the
# in-situ smoothed default, because nemo_bn2 was unreachable from the command
# line. None has ever run at the equator with the oracle's trigger.
#
# PRE-REGISTERED, because this costs more than one GPU-hour. Scored against
# trp_pinned_d30 at day 30 on the same probe and the same box:
#   * top-to-20 m contrast <= 1.5 K AND Nino3 SST bias <= +0.8 K
#         -> CONFIRMED. The missing convective mixing is the leading cause of
#            the campaign's largest error, and the production question becomes
#            which settings, not whether.
#   * contrast >= 3.0 K (i.e. essentially unmoved)
#         -> REFUTED as the leading cause. The diffusivity gap is real and
#            measured, but closing it does not close the lens, and shortwave
#            penetration depth becomes the next candidate.
#   * between -> PARTIAL: a real term that does not carry the bias alone.
# Both ends are reachable: 100 m2/s over a 20 m layer homogenises it in well
# under a day where it fires, so a large move is physically available; and the
# scheme only fires at night on unstable columns, so a small move is equally
# available.
#
# WHY NOT A CHEAPER OFFLINE TEST: the answer depends on a nightly feedback
# between the convective trigger, the closure and advection, re-entered 576
# times a day for thirty days. No diagnostic on a saved state can integrate
# that; the diffusivity gap it would re-measure is already known.
#
# WATCH FOR, and report rather than absorb: a strongly negative Nino3 bias
# would mean the scheme overshoots, and a global cooling would mean it is
# firing far outside the tropics.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_arm_trp_evdeq_d30.txt)
# The command line is untracked, so the arm asserts its own identity rather
# than trusting it: the convection settings must be present, and the two
# adoptions landed later today must be ABSENT, or this stops being a
# one-variable test against the existing control.
for _flag in "--convection enhanced_diffusion" "--convection-K-conv 100.0" \
             "--convection-K-bg 0.0" "--convection-n2-mode nemo_bn2" \
             "--convection-trigger hard"; do
  case "$CMD" in
    *"$_flag"*) ;;
    *) echo "FATAL: command line lacks $_flag"; exit 1 ;;
  esac
done
for _bad in "--sss-restore-channel" "--runoff-spread-passes"; do
  case "$CMD" in
    *"$_bad"*) echo "FATAL: $_bad present -- this would no longer be one variable against trp_pinned_d30"; exit 1 ;;
  esac
done
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdeq_d30/*.npz 2>/dev/null | tail -5
exit $rc
