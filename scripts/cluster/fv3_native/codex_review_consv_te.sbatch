#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_consv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_consv_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of the consv_te TOTAL-ENERGY FIXER, newly ported to
both lanes of the FV3 duo-grid core. FIND THE COMMITS: `git log
--oneline -6`; the range starts at the commit whose subject begins
"feat(fv3): an oracle for consv_te" and runs to HEAD. Do not assume
HEAD is an implementation commit.

WHAT WAS DONE. The energy fixer was the last refused arm. It is now
ported to fv3_native_dynamics.py (SPEC) and fv3_dynamics.py (JAX), and
scored against a GENERATED oracle at residual 1.1866e-09 with a
response-gate figure of 1.347e-08, identical on both backends.

THE STRUCTURAL POINT YOU NEED. fv_mapz.F90 computes te_2d, reduces it
with g_sum and applies dtmp at :975 all inside ONE call, because there
a "domain" is every tile at once. This lane calls
lagrangian_to_eulerian PER FACE, so dtmp cannot be known inside one
call. The remap was therefore SPLIT at its own `if (last_step)`
boundary: `defer_close=True` skips the :975 conversion and leaves pt as
T_v, the caller reduces over six faces, then `close_out_pt` applies
`(pt + dtmp/cp*pkz)/(1+r_vir*q)`.

ATTACK IN THIS ORDER. Cite file:line.

1. IS THE SPLIT FAITHFUL? Between the point where the oracle computes
   te_2d (:638-658) and where it applies :975, does the oracle modify
   ANYTHING this port has already written back, or read anything this
   port has already changed? A deferral that moves the conversion
   across an intervening write is not a refactor, it is a different
   program. Check the write-back ordering in BOTH lanes.
2. THE TWO COLUMN INTEGRALS. The port claims te0_2d
   (compute_total_energy, :1128-1151) and te_2d (the fixer, :638-658)
   are DIFFERENT expressions -- te0_2d takes pt as TEMPERATURE and
   forms tv = pt*(1+qc) itself with phiz integrated DOWNWARD, te_2d
   takes post-remap T_v with cp*pt and gz integrated UPWARD. Verify
   both against the Fortran, verify the shared _ke_column really is
   identical in the two (:650-656 vs :1145-1150), and check the
   u(i,j+1) / v(i+1,j) neighbour slices against the staggered shapes
   (u is (m_a, m_a+1), v is (m_a+1, m_a)).
3. THE REDUCTION. g_sum(mode=0, reproduce=.true.) at :708/:712 --
   verify from fv_grid_utils.F90:2946-2996 that mode 0 is an
   area-weighted SUM (not a mean), that the port's ratio therefore
   cancels the global-area normalisation, and that using float64
   instead of BITWISE_EFP_SUM cannot matter at the gate. Check the
   port sums over the COMPUTE window with the right area slice.
4. LAYOUTS. pe is (n+2, km+1, n+2) with a one-cell halo; peln is
   (n, km+1, n) compute-only; both are (i,k,j). An earlier draft had a
   spurious .T on peln. Verify every index in the two integrals and in
   energy_fixer_zsum0_hydrostatic, including pk's (m_a, m_a, km+1).
5. THE JAX LANE REUSES THE SPEC'S INTEGRALS via np.asarray on traced
   arrays. Is that safe HERE -- are these paths ever reached under jit
   or inside lax.scan, where np.asarray on a tracer raises? Say
   precisely which call sites can be traced.
6. CAN consv_te = 0 HAVE MOVED? The certified 1.1866e-09 must be
   untouched. Check that defer_close is False there, that close_out_pt
   is not called, and that no expression on that path changed.
7. THE GATES. --consv scores against a generated deck; the response
   gate is credited with certifying the fixer because the residual
   floor is only ~12x below the 4.2e-06 K signal. Is that ratio right,
   and does the response gate actually discriminate here? What defect
   in the fixer would leave BOTH numbers passing?
8. ANY COMMENT OR DOCSTRING ADDED THAT IS NOT TRUE.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_CONSV_DONE
