#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_eight_%j.out
set +e
# ONE parameter set per turbulence closure, fitted jointly to every LES
# reference -- four dry and four wet:
#
#   dry   ekman     neutral shear layer          scored u, v
#         gabls1    stable boundary layer        scored theta, u, v
#         cbl       Nieuwstadt free convection   scored theta
#         wangara   diurnal dry CBL (Day 33)     scored theta, u, v
#   wet   bomex     shallow trade cumulus        scored theta, qv
#         rico      precipitating trade cumulus  scored theta, qv
#         astex     transitioning stratocumulus  scored theta, qv
#         dycoms    nocturnal stratocumulus      scored theta, qv
#
# WANGARA IS BACK. It was excluded for a structural reason: its surface heat
# flux is DIURNAL, and the only path that reached the closure carried a single
# scalar in SurfaceLayerConfig. Freezing the cycle at one instant measured
# 8.6-9.9 normalized error on every closure, scheme-independent -- i.e. the
# arm, not the closures -- and driving the cycle from the SCMForcing channel
# instead left the closure seeing a surface heat flux of exactly ZERO, the
# defining input of every flux-driven nonlocal scheme, at 13.0-15.5. Both were
# a missing boundary condition, not a turbulence signal.
#
# The fix does NOT thread `t` through the nine closure signatures. It reuses
# the channel `prescribe="T_s"` already used for a time-varying skin
# temperature: the SCM writes the current kinematic flux into
# `PhysicsState.surface_wth_override` before every tendency evaluation, and the
# turbulence integration folds it into the config leaf the closure already
# reads (`turbulence/integration.py::_resolve_prescribed_surface_fluxes`), with
# the SAME rho the closure is handed -- so YSU's `shflx / (rho[:, -1] * c_pd)`
# recovers exactly the prescribed value. `SCMForcing.flux_to_closure` moves the
# flux from the column tendency to that boundary condition, so it is applied
# exactly once.
#
# Both stratocumulus decks set `dolongwave = .true., doradsimple = .true.` and
# the SCM arm runs that SAME Stevens (2005) kernel, so cloud-top radiative
# cooling is on both sides rather than only in the LES.
#
#
# ============================ 2026-08-21 RE-RUN ============================
# Re-fit at HEAD, because CLUBB's missing surface boundary condition is now
# ported. In the previous eight-case fit (job 9419739, pin c780e2f29) CLUBB
# was LAST of nine at 2.5367 and could not take a single reducing step
# ("no_reducing_step"): its Wangara gradient was 1.33e16 against a loss of 1,
# a near-singular denominator rather than a physical sensitivity. The cause
# was `calc_sfc_varnce` -- CLUBB's own surface second-moment BC, called
# unconditionally every timestep upstream -- being entirely UNPORTED (#1508).
# It is ported now (24eb46cef + d7050fcf8, 0433205ca, 7c8666f79).
#
# PRE-REGISTERED, before the queue: this run produces the joint default and
# tuned score of all nine closures over all eight cases.
#   CONFIRMS the fix  -> clubb's status is `tuned`, not `no_reducing_step`,
#                        and its joint DEFAULT score is below 1.0.
#   REFUTES it        -> clubb still cannot take a reducing step, or its
#                        joint default stays above 2.0.
# Expected from the three single-case arms already run at this pin
# (_sfcv_wangara/_sfcv_bomex/_sfcv_cbl): wangara default 15.21 -> 1.145,
# bomex 0.053, cbl 0.064, each tuning 16-22% on its own.
#
# Why not answer this offline: the three per-case arms above ARE the cheap
# test and they have already run. What they cannot produce is the deliverable
# -- ONE parameter set per closure that holds across all eight regimes -- and
# a joint objective is a mean, so per-case fits do not compose into it.
#
# This is a NEW BASELINE, not a term-by-term successor to job 9419739. Only
# CLUBB's change is under test with independent per-case evidence; the other
# eight closures also moved from c780e2f29 to this pin (the hydrostatic lane
# now exports its surface fluxes as diagnostics, and the dead-parameter freeze
# is derived from source instead of a hardcoded CLUBB-only list). Both are
# expected to be identity for these arms -- the first is a None-guarded
# diagnostic field and the second only relabels parameters the zero-gradient
# gate already froze -- but "expected" is not measured, so do not read a small
# non-CLUBB move as a physics result.
# ===========================================================================
# WALLTIME, sized from the ONE measurement available (job 9333698: 9 schemes,
# 5 cases, 8 steps, 21 h 43). That run's cost is dominated by XLA compilation
# of the per-(scheme, case) rollout graph, which scales with schemes x cases
# and not with optimizer steps: 5 -> 8 cases is 1.6x the compiles, giving
# ~35 h, plus ~13 h of execution at 40 steps over 1.47x the model steps per
# pass. ~48 h expected against a 72 h limit. The driver now writes its full
# report after EVERY scheme, so an overrun costs the unfinished schemes rather
# than the whole campaign.
#
# --steps 40, not 8: in job 9333698 six of nine schemes were still accepting
# an update at step 8, so that run reported a truncated fit as a converged one.
#
# --surface-flux-to-closure is ON. Without it every closure computes a surface
# heat flux of exactly zero on a prescribed-flux case, which disables the
# defining pathway of every flux-driven nonlocal scheme; measured as a 39%
# score change on YSU, moving it two ranks.
#
# --joint-aggregation default_relative, ONE arm only. 'mean' weights each
# regime's ABSOLUTE normalized error equally, which on the five-case run put
# 43% of the joint loss on ekman (70% for tke) against 3.1% on bomex -- the
# "joint" fit was largely an ekman fit. 'default_relative' minimizes the mean
# of s_i/s_i(default), so every regime enters with equal improvement headroom,
# which is what "one set of parameters that works across all the cases" means.
# BOTH numbers are reported either way, and score_tuned is always the plain
# mean so the headline default-vs-tuned comparison keeps its units.
: "${E_WT:?set E_WT to the pinned worktree}"
WT=$E_WT
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
CASE_LIST="ekman gabls1 cbl wangara bomex rico astex dycoms"
# SCHEMES: subset for a completion/split run. Each closure is fitted
# INDEPENDENTLY (the joint objective averages over CASES, not schemes), so
# splitting the nine across jobs is exactly equivalent to running them in
# one -- provided every other flag is held identical, which is why they are
# defaulted here rather than passed per job.
# Pass a subset PLUS-separated ("ysu+edmf"): sbatch --export treats commas
# as its own entry separator, so a comma-separated value is silently
# truncated to its first element and the rest become bogus exports --
# the run would then fit ONE scheme and report it as the whole job.
SCHEMES=${SCHEMES:-all}
SCHEMES=${SCHEMES//+/,}
STEPS=${STEPS:-40}
AGG=${AGG:-default_relative}
OUT=${OUT:-$R/scm_les_turbulence/all_eight_sfcv}

# PREFLIGHT: refuse to start if any reference is missing. This is a long job
# and discovering a missing deck after the queue wait wastes the slot.
CASES=""
for c in $CASE_LIST; do
  if [ ! -d "$R/les_ref/$c" ]; then
    echo "MISSING LES reference: $R/les_ref/$c -- refusing to start"; exit 2
  fi
  CASES="${CASES:+$CASES,}$c:$R/les_ref/$c"
done
echo "all references present"
# ...and that each one is the case it claims to be. A directory can exist and
# still be refused: the Wangara reference was stamped "cbl" by a --case-label
# default, the loader rejected it (correctly -- the two are not cross-aliased),
# and this campaign died six seconds in AFTER waiting for a GPU. The stamp is
# one npz field; check it here rather than discovering it past the queue.
for c in $CASE_LIST; do
  $PY - "$c" "$R/les_ref/$c" <<'PYEOF' || exit 3
import sys, glob
import numpy as np
case, d = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{d}/profiles/prof_*.npz"))
if not fs:
    print(f"  {case}: NO profile frames in {d}"); sys.exit(1)
wrote = str(np.load(fs[0], allow_pickle=True)["case"])
sys.path.insert(0, "packages/ml")
from legoesm.training.les_reference import _case_matches
ok = _case_matches(wrote, case)
print(f"  {case}: {len(fs)} frames, stamped {wrote!r} -> "
      f"{'ok' if ok else 'REFUSED by the reference loader'}")
sys.exit(0 if ok else 1)
PYEOF
done
echo "all stamps accepted"
echo "cases: $CASES"

rm -rf "$OUT"
t0=$SECONDS
# NO pipe to tail: a pipe buffers the driver's progress until it exits, so a
# long run shows nothing and a death shows nothing either. Without the pipe,
# $? is the driver's own status.
$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --cases "$CASES" --outdir "$OUT" --nlev 48 --steps "$STEPS" \
    --joint-aggregation "$AGG" --schemes "$SCHEMES" --surface-flux-to-closure
rc=$?
echo "driver rc=$rc wall=$((SECONDS - t0))s"

# The plots run even on rc != 0: the per-scheme flush means a killed or partly
# failed run still holds every scheme that finished, and those are exactly the
# ones worth looking at.
$PY scripts/plot/plot_scm_les_tuning_summary.py "$OUT"
echo "  summary plot rc=$?"
for c in $CASE_LIST; do
  $PY scripts/plot/plot_scm_les_turbulence.py "$OUT" --case "$c" \
      --out "$OUT/profiles_vs_les_$c.png" && echo "  plotted $c"
done
$PY scripts/validate/scm_les_score_decomposition.py "$OUT" \
    --json-out "$OUT/score_decomposition.json" | tail -90
echo "=== ranking ==="; cat "$OUT/ranking.csv"
echo "DONE $(date +%T)"
exit $rc
