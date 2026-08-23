#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_deard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/cdx_deard_%j.out
set +e
# Adversarial review of the GLM-authored Deardorff stable-length eddy viscosity
# (scripts/tmp/_glm_deardorff_candidate.py), which is proposed to replace the
# Lilly sqrt cutoff that makes the single-column Smagorinsky closure untunable.
# Routing per CLAUDE.md: GLM-authored -> codex + Claude. This is the codex half.
WT=/burg-archive/glab/users/pg2328/legoESM
cd "$WT" || exit 1
export CODEX_HOME=/burg-archive/home/pg2328/.codex   # NOT node-local; a wrong
                                                     # CODEX_HOME exits 0 having
                                                     # reviewed nothing
codex exec --sandbox read-only -C "$WT" "$(cat <<'PROMPT'
Adversarially review scripts/tmp/_glm_deardorff_candidate.py. It is a proposed
shared helper for a differentiable JAX climate model. Be specific and cite lines.

INTENT. The single-column Smagorinsky closure currently computes
  Ri = N2/S2  (N2>0 = STABLE)
  f  = sqrt(max(0, 1 - Ri/Pr_t))          # lilly_buoyancy_factor
  K_m = (C_s * l_mix)**2 * S * f
  K_h = K_m / Pr_t
This is UNTUNABLE: the sqrt is guarded only where its argument goes negative, so
as the argument -> 0+ the slope 0.5/sqrt(x) diverges. MEASURED |dLoss/dparam| =
7.4e18 on the stable GABLS1 case versus 0.013-0.375 on the seven other LES
cases; every optimizer step is rejected.

The candidate replaces that with the Deardorff(1980)/SAM "dosmagor" treatment:
suppress stable mixing via the mixing LENGTH, not via a vanishing factor.
Reference implementation already in this repo at
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py
(search "dosmagor"), where the caller passes an isotropic grid delta and
computes strain_mag = safe_sqrt_strain(strain_mag_sq - prandtl*n2_sgs) BEFORE
the block. The single column has no grid delta; it passes a Blackadar length
kappa*z/(1+kappa*z/l_mix_max), and Pr_t is TUNABLE over (0.33, 3.0).

ANSWER THESE, each with a verdict and the line:
1. Is the reverse-mode gradient BOUNDED (not merely finite) everywhere the
   inputs can reach? 0.5/sqrt(1e-36)=5e17 is finite and useless -- that is the
   bug being fixed. Check N2 -> 0+, both clip crossovers, zero shear, and the
   S2 = Pr*N2 surface. If you claim boundedness, give the bound.
2. PRANDTL CONVENTION. The existing scheme divides (1 - Ri/Pr_t, i.e.
   S2 - N2/Pr_t); the candidate multiplies (S2 - prandtl*N2). Do these agree?
   For which Pr_t? What happens across the tunable range if a caller passes
   config.Pr_t straight in?
3. UNSTABLE SIDE. The candidate uses jnp.maximum(n2, 0.0) in the strain
   correction. The reference does NOT clamp there. What does that change for
   N2 < 0, and does the candidate still reduce EXACTLY to the current default
   form on neutral/unstable columns as required?
4. DOUBLE CORRECTION. The candidate applies the Pr*N2 subtraction internally
   while the existing plane-LES caller already applies it. What breaks, and is
   it silent?
5. Any clamp that is INERT given the declared bounds (C_s in (0.05,0.6)), or
   that silently returns a plausible wrong number instead of raising.
6. Anything else that would produce wrong numbers rather than an error.

Do not rewrite the file. Report defects, ranked, most severe first.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo "DONE $(date +%T)"
