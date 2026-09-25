#!/bin/bash
#SBATCH --job-name=tkeprof
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkeprof_%j.out
#
# WHERE, IN DEPTH, DOES OUR TURBULENT ENERGY PART FROM THE ORACLE'S?
#
# Enhanced diffusion is refuted as the cold-tongue lever (63d495e91): the arm
# fires at night at the oracle's rate and moves the lens four percent. What
# remains is the 15-65 m gap the 2026-09-20 note already separated from it:
# there NEMO's heat diffusivity is ORDINARY closure turbulence, 1.3e-3 to
# 2.1e-3, and ours is 250 to 1000 times smaller, while our closure is known to
# be faithful GIVEN the energy (the frozen twin reproduces NEMO's own
# coefficients from NEMO's own en at x0.956). So the coefficients are not the
# question; the ENERGY is, and every comparison so far has read it through a
# five-day mean or through the coefficients.
#
# THE ORACLE ARCHIVES ITS INSTANTANEOUS TKE STATE ONCE, AND IT IS THE SAME
# INSTANT AS OUR FINAL SNAPSHOT. The RUN_GATEWAY restart at step 2160 is
# day 90 at 00:00 UTC (adatrj 90.0) and carries en, dissl, avm_k, avt_k. Our
# trp_faithz0_d90 snapshot_final is day 90 at 00:00 UTC. Instant against
# instant, same hour, same day, same box, no five-day-mean sampling on either
# side. In the cold tongue that hour is mid-afternoon on both sides.
#
# WHAT IS PRINTED, per interface down to 105 m, box medians over 220-240E
# |lat|<=2:
#   ours          en, N2, S2, Ri, l_k, l_eps, K_M, K_H  -- our snapshot state
#                 through our own closure helpers (the twin's control path)
#   ours_on_nemo  the same, with NEMO's restart state fed to our closure
#   nemo          en, dissl, avm_k, avt_k               -- NEMO's own state
# The energy ratio ours/nemo per interface is the profile that answers the
# question; the two length scales beside it say whether the closure turns
# equal energy into equal mixing.
#
# PRE-REGISTERED (as AMENDED in 4c5cc7a8d after dual review, before any
# result), on the FIRST interface, reading from the top, where ours/nemo en
# falls below 0.1:
#   * the first interior interface (~1 m) -> the SURFACE SOURCE is short: the
#     energy injected by the wind boundary condition is not arriving, and the
#     2026-09-20 "surface source is right" reading (made from viscosity, not
#     energy) is overturned.
#   * any interface in (1 m, 65 m], the top being >= 0.1 (a top ratio ABOVE 1
#     followed by a crossing belongs here too) -> the DECAY is the defect;
#     report the crossing depth, and the TKE equation's own terms between the
#     top and that depth are the next measurement.
#   * no crossing down to 65 m -> NO ORDER-OF-MAGNITUDE energy gap; report the
#     per-interface ratio, which is the finding ("energy matches" is not a
#     label this branch earns -- a ninefold deficit would sit in it).
#   * a zero or non-finite denominator at an interface -> that interface is
#     unresolved and excluded from the crossing search.
# Read the energy in LOG with l_eps beside it: K ~ e/N, so a 250-1000x
# coefficient gap needs an energy gap of order 400-1500x and the crossing will
# sit shallow. Every branch is reachable and none is favoured by the instrument.
#
# THE ONE CAVEAT NAMED IN ADVANCE: both models are read at mid-afternoon in
# the cold tongue, peak stratification, the daily minimum of near-surface
# turbulence. A ratio at that hour is a ratio of two minima; it locates the
# depth of divergence, which is what is registered on, and does not by itself
# give a daily-mean energy ratio.
#
# GATES FIRST. The twin's control gate (our closure must reproduce our own
# stored K within x1.5) and its non-vacuity gate (en x0.1 must break it) run
# before either profile; the reduction's unit tests run before everything.
# THE GATE IS TAKEN OVER THE ACTIVE INTERFACES, not the 65-105 m band: in this
# run the closure sits exactly on its momentum floor below ~7 m and the
# stored coefficient there is floor plus the additive wave-mixing background,
# so the band ratio (measured x0.476 in job 9981703) reports the background
# fraction, not whether the closure is wired correctly. Over the interfaces
# where the closure is above its floor the same job read x0.68-0.86, the
# level the twin was validated at.
#
# REUSED, NOT REBUILT: frozen_column_tke_twin.py already stitches the restart,
# builds both states with the model's own helpers, and reduces over the box;
# --profile-depth-max only prints per interface what it already computed.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "=== profile reduction unit test ==="
$PY -m pytest -q tests/ocean/unit/test_frozen_column_tke_twin_profile.py 2>&1 | tail -3
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: reduction test is red (rc=$rc)"; exit 1; }

HARNESS=scripts/validate/ocean_fidelity/frozen_column_tke_twin.py
ARM=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_d90
SNAP=$ARM/snapshot_final.npz
MANI=$ARM/run_manifest.json
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
RGLOB="$G/RUN_GATEWAY/ORCA1_00002160_restart_oce_*.nc"
SBC=$G/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc
MESH=$G/RUN_SMOKE/mesh_mask_0001.nc
for f in "$SNAP" "$MANI" "$SBC" "$MESH"; do
  [ -e "$f" ] || { echo "FATAL: missing input $f"; exit 1; }
done
ls $RGLOB >/dev/null 2>&1 || { echo "FATAL: no restart tiles match $RGLOB"; exit 1; }

# Both sides at day 90: our stamp, and the restart's own day counter.
$PY - "$SNAP" <<'STAMP' || { echo "FATAL: snapshot is not at day 90"; exit 1; }
import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
got = float(np.asarray(d["time_days"]).ravel()[0])
print(f"  ours: stamp day {got:g}")
sys.exit(0 if abs(got - 90.0) < 1e-6 else 1)
STAMP
$PY - "$G/RUN_GATEWAY/ORCA1_00002160_restart_oce_0000.nc" <<'STAMP' || { echo "FATAL: restart is not at day 90"; exit 1; }
import sys
import netCDF4
ds = netCDF4.Dataset(sys.argv[1])
got = float(ds.variables["adatrj"][:].ravel()[0])
print(f"  nemo: restart adatrj {got:g}")
sys.exit(0 if abs(got - 90.0) < 1e-6 else 1)
STAMP

BOX="--lat-halfwidth 2.0 --lon-west 220.0 --lon-east 240.0"
# The SBC file holds 17 records (last centred 24 March, days 80-85), one
# fewer than the grid files, so the day-90 wind stress is not archived.
# Record 16 is the nearest. It feeds ONLY the surface mixing-length anchor,
# which the twin already caveats as a five-day mean; the energy profiles do
# not depend on it.
REC=16

echo "===================== STEP 1: self-recovery CONTROL (+ our profile) ====="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-meshmask "$MESH" --nemo-sbc "$SBC" --rec $REC $BOX --gate-band active --profile-depth-max 105
rc1=$?
echo "[gate] control exit = $rc1"
[ "$rc1" -eq 0 ] || { echo "FATAL: control did NOT reproduce our own stored K -- profile NOT trusted"; exit 1; }

echo "===================== STEP 2: NON-VACUITY (en x0.1) ======================="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-meshmask "$MESH" --nemo-sbc "$SBC" --rec $REC $BOX --gate-band active --perturb-en 0.1
rc2=$?
echo "[gate] non-vacuity exit = $rc2  (0 = STEP-1 assertion correctly FAILED)"
[ "$rc2" -eq 0 ] || { echo "FATAL: gate is VACUOUS; NEMO side NOT run"; exit 1; }

echo "===================== STEP 3: NEMO instant state (+ both profiles) ======="
$PY -u "$HARNESS" --mode nemo --manifest "$MANI" \
    --restart-glob "$RGLOB" --nemo-meshmask "$MESH" --nemo-sbc "$SBC" \
    --rec $REC $BOX --profile-depth-max 105
rc3=$?
echo "=== card exit: $rc3 ==="
exit $rc3
