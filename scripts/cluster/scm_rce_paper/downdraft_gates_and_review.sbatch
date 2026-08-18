#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=dd_gate_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/downdraft_gates_%j.out
set +e
# ---------------------------------------------------------------------------
# Run the downdraft port's own gates, then hand the commit to codex for
# adversarial review.  Gates FIRST: a review of code that does not pass its
# own orientation test would be reviewing the wrong artifact.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ downdraft gates ############"
$PY -m pytest -q $WT/tests/unit/test_emanuel_downdraft.py 2>&1 | tail -30
GATE_RC=${PIPESTATUS[0]}; echo "GATE_RC=$GATE_RC"

echo; echo "############ emanuel regression (the mixing output changed) ############"
$PY -m pytest -q $WT/tests/unit/test_emanuel.py 2>&1 | tail -15
REG_RC=${PIPESTATUS[0]}; echo "REG_RC=$REG_RC"

[ $GATE_RC -eq 0 ] && [ $REG_RC -eq 0 ] || { echo "ABORT: gates red, not reviewing"; exit 1; }

export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo; echo "CODEX_HOME=$CODEX_HOME"; codex --version || exit 92

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of the HEAD commit. Run `git show --stat HEAD` and
`git show HEAD` first.

It ports the precipitating unsaturated downdraft of Emanuel's CONVECT v4.3c
into JAX. THE ORACLE IS ON DISK at
.physics-validator/emanuel/oracle/convect43c.f -- READ IT. The downdraft loop
is lines 713-830, its initialisations are at lines 466-501, and the tendency
terms that consume it are at lines 843-849 (lowest level) and 932-934 (all
other levels). Diff the port against the Fortran line by line; do not review
it as if the reference were unavailable.

CRITICAL CONTEXT: the oracle is SURFACE-FIRST (I=1 is the lowest model level,
I+1 is ABOVE I). The new module claims to work surface-first internally. The
arrays it consumes come from _emanuel_mixing's output, which returns SOME
fields surface-last (dT_dt, dq_v_dt, dq_c_conv_dt, convective_layer_mask) and
others surface-first (ment, m_profile, and the newly added elij, clw, ep, qs,
h_moist, gz, lv). VERIFY the newly added ones really are surface-first at the
point they are assigned in _emanuel_mixing, not merely labelled so.

Priorities, most important first:

1. LINE-BY-LINE FIDELITY. For each oracle statement in 713-830, find its
   counterpart and say whether it is faithful. Pay special attention to:
   - WDTRAIN's inner sum `DO 320 J=1,I-1`: is the triangular mask the right
     orientation, and is it strict?
   - `QSM=0.5*(Q(I)+QP(I+1))` -- is QP(I+1) the CARRY (previous scan step) or
     an array read? A wrong choice here is a one-level shift that still runs.
   - `C6=(WATER(I+1)*WT(I+1)+WDTRAIN/SIGD)/WT(I)` -- WT(I+1) is the carried
     fall speed, WT(I) the current one. Check both.
   - `DHDP=(H(I)-H(I-1))/(P(I-1)-P(I))` -- units. The port converts P to hPa;
     confirm the oracle's P is hPa there and that MP's leading 100. factor is
     therefore right.
   - `FAC=20.0/(PH(I-1)-PH(I))` -- PH(I-1)-PH(I), not PH(I)-PH(I+1).
   - the `IF(I.EQ.INB)GOTO 400` skip of the QP update, and what QP(INB) is
     left at (see the initialisation at line 501).
   - the `IF(I.EQ.1)GOTO 360` skip, and whether MP(1) ends at zero.

2. THE TENDENCIES. The port uses ONE formula for every level and claims the
   oracle's i=1 expression is the general one with MP(1)=0. Verify against
   lines 843-849 vs 932-934, including the DPINV definition and the
   `(CL-CPD)*WATER(I+1)*(T(I+1)-T(I))` sensible-heat term. Is `CPN(1)` vs
   `CPINV` handled the same way? Is there a factor-of-100 error anywhere in
   the hPa/Pa conversions? State the units of every term you check.

3. WATER CONSERVATION. Does `SIGD*EVAP` in FQ exactly equal the water removed
   from the falling precipitation? If the oracle does not close it exactly,
   say where the residual goes and whether the port reproduces that.

4. DIFFERENTIABILITY. The guards claim to keep gradients finite. Attack them:
   is the double-where actually applied on BOTH branches that can divide by
   zero? Can `jnp.clip(qp, 0, qs_below)` kill a gradient that matters? Does
   the discrete argmax anchor in _taper_mass_flux_to_surface leak a wrong
   gradient rather than no gradient?

5. THE TESTS. Would `test_rain_falls_downward_not_upward` actually FAIL if the
   level ordering were inverted? Construct the counterexample if not. Are any
   of the tests vacuous -- passing for a reason other than the one stated?

6. Anything that would silently produce a WRONG NUMBER rather than an error.

Numbered findings, each with file:line, severity, and a concrete fix. Say
explicitly when a concern turns out not to be a defect.
PROMPT
)" 2>&1 | tail -250
echo "CODEX_RC=$?"
echo "COMPLETED"
