#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_hs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_hs_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a GLM-authored physics port: held_suarez_tend in
packages/core/legoesm/core/fv3_native_physics_coupling.py (added in
git show fb2d8b24b) and its tests in tests/grids/test_fv3_physics_coupling.py.
The oracle is driver/solo/hswf.F90 (Held_Suarez_Tend) in the pinned tree
/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean.
Cite file:line, classify BLOCKER/MAJOR/MINOR, smallest fix, say where NOTHING found.

Attack specifically:
1. The bottom-up k-loop and teq recursion. Fortran runs k=npz..1 with
   teq(i,k)=teq(i,k+1)+dt_tropic in the strat/meso branches. Does the port's
   teq column (length npz+1, slot npz = teq(npz+1)) reproduce it EXACTLY,
   including that the bottom level is tropospheric and never reads teq(k+1)?
2. dz. I corrected GLM's dz from interface peln to h0*log(pl_{k+1}/pl_k) (layer
   mean) with a k+1 clamp at the bottom. Verify against hswf.F90 that (a) it IS
   log(pl(k+1)/pl(k)), (b) the clamp only affects masked-out cells, (c) the sign
   is -2.25*cos*dz for mesosphere and +2.25*cos*dz for stratosphere.
3. The three regime thresholds (pl<=1mb meso, 1mb<pl<=100mb strat, else tropo)
   and branch ORDER. A vectorized np.where nest must match the Fortran if/elseif
   order exactly.
4. Rayleigh friction: only in the tropospheric branch, only where (sigl-sigb)>0,
   reads ua+u_dt / va+v_dt. Does the port's fric mask (fric & ~(meso|stratm),
   then sigf where fric) reproduce that, and is u_dt(k) zero before friction so
   ua+u_dt == ua?
5. Constants: are the H&S coefficients (315,60,10,2/7,0.7,40,4,1,2.25,1mb,100mb,
   6371e3,100) faithful to hswf.F90, and is rad_ratio/kf_day/rka/rks/rkv/rms/tau
   arithmetic identical?
6. The test: is _hs_ref a genuinely independent transcription (explicit i/j/k
   loops, no shared vectorized slice), would it fail on a shifted branch or a
   wrong dz, and does test_..._all_three_regimes prove the column is non-vacuous?
   Is atol 1e-15/rtol 1e-11 defensible? Any residual HTML-escape (&amp; &gt;)?
PROMPT
)" 2>&1
echo "CODEX_HS_REVIEW_DONE"
