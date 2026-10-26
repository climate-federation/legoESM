#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_cpl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_cpl_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a GLM-authored port: `git show 784efa6fe` --
packages/core/legoesm/core/fv3_native_physics_coupling.py (update_dwinds_phys_duo
+ its JAX twin) and tests/grids/test_fv3_physics_coupling.py. Cite file:line,
classify BLOCKER / MAJOR / MINOR, give the smallest fix, and say where you found
NOTHING.

It ports the grid_type=0 bounded-domain (duo) path of FV3's update_dwinds_phys
(model/fv_grid_utils.F90:3420-3540 in the pinned tree
/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean):
v3 = u_dt*vlon + v_dt*vlat on (is-1:ie+1,js-1:je+1); ue(i,j)=v3(i,j-1)+v3(i,j);
ve(i,j)=v3(i-1,j)+v3(i,j); u += dt/2*(ue.es(:,i,j,1)); v += dt/2*(ve.ew(:,i,j,2)).
Port arrays are full data domain (m=n+2ng), component axis LAST, es1=es(:,:,:,1),
ew2=ew(:,:,:,2). The four .not.bounded_domain edge_vect blocks are claimed dead
on a duo grid and omitted.

Attack specifically:
1. INDEX ALIGNMENT. Do the numpy slices map to the Fortran loop bounds EXACTLY?
   Verify the v3 window slice(ng-1,ng+n+1), the ue sum v3[1:n+1,0:n+1]+v3[1:n+1,1:n+2],
   the ve sum, and the two projection windows u[ng:ng+n, ng:ng+n+1] and
   v[ng:ng+n+1, ng:ng+n] against the Fortran i/j ranges. An off-by-one or a
   swapped edge sum (v3(i,j-1) vs v3(i,j+1)) is the failure mode.
2. es/ew NaN MARGINS. The builder leaves es1[:,0]/[:,-1] and ew2[0,:]/[-1,:] NaN.
   Does the projection window ever touch them (for ng>=1? for ng=0?)? Is there a
   guard, and should ng<1 be rejected rather than silently reading NaN?
3. THE DEAD-BLOCK CLAIM. Is omitting the edge_vect_w/e/s/n blocks correct for a
   duo grid, i.e. is bounded_domain truly always true there and are those blocks
   the ONLY difference from the general path? Or is something else in the general
   path (a halo fill of v3, a different projection) also being dropped?
4. JAX TWIN. Is update_dwinds_phys_duo_jax numerically identical to the numpy
   authority (it uses .at[].add rather than in-place +=)? static_argnums for n/ng?
   Any float32/float64 promotion difference vs numpy?
5. THE TEST. Is _ref a genuinely INDEPENDENT transcription (no shared slicing with
   the authority), and would it actually FAIL on a shifted window? Does the FD
   grad check probe a cell whose gradient is nonzero? Is atol 1e-13 defensible?
   Does test_only_the_compute_block_moves actually pin the write region?
PROMPT
)" 2>&1
echo "CODEX_CPL_REVIEW_DONE"
