#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_c2l
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_c2l_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a GLM-authored port (Claude independently derived the same slicing): commit 7b6e6862c -- c2l_ord4_face in packages/core/legoesm/grids/fv3_native_ext_vector.py and tests/grids/test_c2l_ord4_face.py. Cite file:line, classify BLOCKER/MAJOR/MINOR, smallest fix, say where you found NOTHING.

It ports c2l_ord4 (cubed_to_latlon 4th order), DUO/bounded branch, from fv_grid_utils.F90:2407-2531 in the pinned tree /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean. Fortran bounded branch: utmp(i,j)=c2*(u(i,j-1)+u(i,j+2))+c1*(u(i,j)+u(i,j+1)); vtmp(i,j)=c2*(v(i-1,j)+v(i+2,j))+c1*(v(i,j)+v(i+1,j)); ua=a11*utmp+a12*vtmp; va=a21*utmp+a22*vtmp; c1=1.125,c2=-0.125. u is (m_a,m_b)=(n+2ng, n+2ng+1); v is (m_b,m_a). amat=(a11..a22) each (m_a,m_a). The non-bounded edge/corner cases are claimed dead on a duo grid.

ATTACK:
1. INDEX ALIGNMENT. With s=ng,e=ng+n,cs=slice(s,e): are u[cs,s-1:e-1],u[cs,s:e],u[cs,s+1:e+1],u[cs,s+2:e+2] EXACTLY u(i,j-1..j+2) for the compute cells, given u's 2nd axis is the y-face (length m_b=m_a+1)? Same for v's row stencil. An off-by-one or a u/v axis swap is the failure mode and is SHAPE-INVISIBLE on square faces. Verify against the Fortran lo=1-ng mapping.
2. DOMAIN/HALO. Output is compute-domain only [ng:ng+n], NaN elsewhere; ng>=2 enforced. Is the compute-edge stencil (Fortran i or j = 1 or n) reading valid halo indices (u columns ng-1 and ng+n+1; v rows ng-1 and ng+n+1) within bounds of m_b? Is dropping ord2's is-1..ie+1 output ring safe for the ONLY consumer (apply_held_suarez_step, which reads ua[ng:ng+n]) -- grep it?
3. THE DEAD-BLOCK CLAIM. Is omitting the non-bounded edge_interpolation/dx-dy-weighted edge cases correct for bounded_domain=.true., i.e. does the Fortran bounded branch truly use only the unweighted c1/c2 interior stencil with NO dx/dy weighting and NO corner handling?
4. THE TEST. Is _fortran_loop_ref a genuinely INDEPENDENT transcription (no shared slicing), would it FAIL on the shifted stencil (it asserts so), and is atol 1e-13 defensible? Does it verify the NaN region exactly?
5. Anything else the bounded c2l_ord4 does (a sign, a metric, a mode!=1 branch) that this omits?
PROMPT
)" 2>&1
echo "CODEX_C2L_REVIEW_DONE"
