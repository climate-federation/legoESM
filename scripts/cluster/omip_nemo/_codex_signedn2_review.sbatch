#!/bin/bash
#SBATCH --job-name=cxsgn
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cxsgn_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export CODEX_HOME=$HOME/.codex
git diff 9efc193b4~1..HEAD -- packages/ scripts/run/ tests/ > /tmp/sgn_$$.diff
codex exec --sandbox read-only -C "$WT" "$(cat <<'EOP'
Adversarially review the diff below against NEMO 5.0.1 zdftke/eosbn2 fidelity.
It does two things: (1) removes a clip on the buoyancy frequency N^2 so the
ORCA1 TKE card sees SIGNED stratification (negative N^2 in unstable columns),
(2) wires a TEOS-10 Roquet-2015 alpha/beta path into the bn2 assembly and
threads it to the tripole C-grid and MPAS bridges.

Answer ONLY these, with file:line evidence, and say REFUTED where I am wrong:

1. NEMO's eosbn2 does NOT clip bn2. But does every DOWNSTREAM consumer in our
   TKE survive a negative N^2 the way NEMO's does? Walk each use of the N2
   array in tke.py: mixing-length sqrt, Prandtl number, buoyancy production,
   the Langmuir PE integral, and the dissipation length. Name any site where a
   negative value now produces a NaN, a wrong sign, or a silently different
   answer from NEMO's zdftke.

2. NEMO computes bn2 ONCE per step in eosbn2 and zdftke reads it. We compute
   N2 at two sites (N2 and N2b). Under signed N^2, can those two disagree in a
   way that matters (different time level, different ladders, different eos
   form)? Quote both call sites.

3. The TEOS-10 alpha/beta: check the Horner nesting order and the beta division
   by zs against Roquet et al. 2015 / NEMO eosbn2.F90 rab_3d. A transposed
   coefficient or a misplaced r1_rho0 gives a plausible-but-wrong number, not
   an error.

4. The MPAS bridge now uses LIVE z-star ladders gdept_0*(1+eta/H). Confirm the
   tripole C-grid path uses the SAME ladder construction, and that the two
   grids therefore see the same N^2 for the same T/S/eta. If they differ, that
   breaks the cross-grid agreement claim this whole branch is chasing.

5. What test would FAIL if the sign fix were silently reverted? If the answer
   is "none", name the test to write.

Be specific and skeptical. Prefer "this is fine" only with evidence.
EOP
)

DIFF:
$(cat /tmp/sgn_$$.diff)" 2>&1
echo "=== CODEX RC=$? ==="
