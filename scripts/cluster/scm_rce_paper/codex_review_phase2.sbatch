#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_codex_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt2
# Codex needs a NODE-LOCAL CODEX_HOME (shared-FS home has failed before).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version

cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW. You are reviewing commit b47595e22 in this repo (the HEAD
commit; run `git show --stat HEAD` and `git show HEAD` to see the diff).

The change threads a `subsidence_solve` selector through four atmospheric
convection schemes so a single-column RCE intercomparison can force the SAME
conservative vertical transport kernel across the whole mass-flux family.

Files changed:
  packages/atmosphere/legoesm/atmosphere/physics/convection/config.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/tiedtke.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/emanuel.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/mass_flux.py
  packages/atmosphere/legoesm/atmosphere/physics/ml_parameterization.py
  tests/unit/test_convection_subsidence_solve_threading.py

Be maximally skeptical and CONCRETE (cite file:line). Focus on:

1. CORRECTNESS OF THE SEMANTIC DIFFERENCE. `apply_mass_flux_kernel`'s
   "advective" branch emits `dq_c = delta_0*M*q_c_u/rho` with NO paired vapor
   sink; the "implicit_flux" branch books `dq_v -= dq_c` AND transports in
   telescoping flux form. Kain-Fritsch, which already uses implicit_flux, adds
   a compensating `dT_dt += (L_v/c_pd)*dq_c_conv_dt` latent-release term after
   its kernel call (kain_fritsch.py ~1266). Tiedtke/Emanuel/ZM/mass_flux do
   NOT add that term. QUESTION: when a user now selects
   subsidence_solve="implicit_flux" on those four schemes, is the resulting
   column MSE budget WRONG (short by L_v*int dq_c), or is deferring that
   latent to microphysics a self-consistent convention? Give the sign and an
   order-of-magnitude estimate. This is the single most important question.

2. Does Tiedtke's call site double-count or mis-pair anything? Note tiedtke.py
   DISCARDS the kernel's third return value and computes its own
   `dq_c_conv_dt = delta_0_eff * M_u * p_gate_qc * q_c_u / rho_safe`. Under
   implicit_flux the kernel has ALREADY debited `dq_v -= dq_c` internally --
   does Tiedtke's separate dq_c then get counted twice against a single vapor
   debit, or not at all? Same question for Emanuel, which rescales
   `dq_c_conv_dt / max(sort_multiplier, 1e-15)` after the call while passing
   `delta_0 * sort_multiplier` INTO it.

3. `mass_flux_convection_from_closure` gained `dt: float | None = None` and
   dropped its `del p_half`. Is passing `p_half`/`dt` unconditionally to the
   kernel safe on the advective path (they are ignored there)? Any AD/JIT
   concern from `dt=None` reaching the kernel under the advective default?

4. AD / JIT / pytree: the configs are NamedTuples with new str + float fields.
   Does adding fields break any positional construction, `_replace`, pytree
   flattening, or a `__param_spec__`/AST ratchet
   (tests/test_param_specs.py requires every :float-annotated field to be
   spec'd or classified -- `theta_implicit: float` is NEWLY added to four
   configs; is that a ratchet violation?). Check
   tests/_param_spec_baseline.py and tests/_inline_coeff_baseline.py.

5. THE TEST FILE. Is each gate genuinely NON-VACUOUS? Would
   test_implicit_flux_is_honoured pass if the config field were silently
   dropped? Is test_implicit_flux_reduces_column_water_residual measuring the
   right conserved quantity, with the right sign of dp (surface-LAST arrays)
   and the right units? Does the fixture actually convect (otherwise the
   gates are vacuous)? Is the jax.grad gate proving anything?

6. Anything that would silently change SHIPPED behaviour (defaults must stay
   "advective" for these four).

Output a numbered list of findings, each tagged BLOCKER / MAJOR / MINOR /
NIT, with file:line and a concrete suggested fix. If a concern is
unfounded after checking, say so explicitly rather than padding the list.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
