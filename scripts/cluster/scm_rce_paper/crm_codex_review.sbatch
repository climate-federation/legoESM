#!/bin/bash
# Codex adversarial review of the CRM Wing-seed / fetch / budget-validator work.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=crm_codex
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/crm_codex_%j.out
set +e
# Pipelines: $? after `cmd | tail` is TAIL's status, not cmd's. Use
# ${PIPESTATUS[0]} wherever a piped command's exit code is reported.
BRANCH="${BRANCH:-crm/rcemip-resolved-convection}"
WT="${WT:-/burg-archive/glab/users/pg2328/_wt_crmcodex_$SLURM_JOB_ID}"
SRC=/burg-archive/glab/users/pg2328/legoESM
git -C "$SRC" worktree add --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"

# Node-local CODEX_HOME: a shared/NFS home makes codex exit 0 having reviewed
# nothing. Copy the credentials in and PROVE they landed.
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p "$CODEX_HOME"
cp -r ~/.codex/auth.json ~/.codex/config.toml "$CODEX_HOME"/ 2>/dev/null
export PATH="$HOME/.nvm/versions/node/v24.16.0/bin:$PATH"
echo "CODEX_HOME=$CODEX_HOME exists=$([ -d $CODEX_HOME ] && echo yes || echo no) auth=$([ -s $CODEX_HOME/auth.json ] && echo yes || echo no)"
echo "codex bin: $(which codex)"

git diff --stat origin/main...HEAD
echo "=================== REVIEW ==================="
codex exec --sandbox read-only -C "$WT" "$(cat <<'PROMPT'
Adversarially review the diff of this branch against origin/main. Files changed:
  packages/atmosphere/legoesm/atmosphere/dynamics/crm/sam_case_setup.py
  scripts/run/run_rcemip_plane.py
  scripts/validate/rcemip_column_budget.py
  scripts/data/fetch_rcemip_reference.py
  tests/unit/test_sam_case_setup.py
  tests/unit/test_fetch_rcemip_reference.py
  tests/unit/test_rcemip_column_budget.py
  scripts/cluster/scm_rce_paper/*.sbatch
  docs/science/specs/CRM_implementation.md

Run `git diff origin/main...HEAD` first and review ONLY what changed.

This is a JAX Earth-system model. Prioritise, in order:

1. CORRECTNESS OF THE NEW SEED. wing2018_thermal_noise_seed claims to implement
   Wing et al. 2018 GMD Sect 3.2.3: random thermal noise in the FIVE LOWEST
   layers, amplitude 0.1 K in the lowest tapering LINEARLY to 0.02 K in the
   fifth. The plane CRM stores levels TOP-DOWN (index -1 = lowest layer). Check
   the taper is applied in the right vertical DIRECTION -- i.e. that 0.1 K lands
   on the LOWEST layer and 0.02 K on the fifth, not inverted. Check the
   interaction with the caller in run_rcemip_plane._build_rcemip_initial_state,
   especially the hydrostatic rho' = -rho_0*theta'/theta_0 construction and the
   n_seed_lev=None default resolution (5 for wing2018, 4 for the legacy kinds).
   Does any existing caller break from n_seed_lev changing from 4 to None?

2. THE DIAGNOSTIC TOOLING IS THE HIGHEST RISK. scripts/validate/rcemip_column_budget.py
   produces the numbers a physics conclusion rests on. Attack it: is
   hydrostatic_pressure's top-down/bottom-up flipping correct? Is layer_thickness
   consistent with the model's actual staggering? Is the water-budget estimator
   P = E - d(CWV)/dt sound, and is np.gradient the right differencing given the
   snapshot spacing? Is the bulk evaporation estimate using the right
   coefficients and the right q_sat (surface vs lowest-level pressure)? Would any
   of these produce a plausible-looking but WRONG mm/day?

3. SIGN CONVENTIONS AND UNITS everywhere. MSE is stored in kJ/kg by the driver
   and converted with MSE_KJ_TO_J; specific humidity vs mixing ratio in the fetch
   script; K vs C; m vs km.

4. FAIL-LOUD BEHAVIOUR. Are there paths where a broken input yields a
   plausible number instead of an abort? Is any control vacuous (would it pass
   even if the feature were removed)?

5. DISPATCH HARDENING and the repo rules: no duplicated numerics, constants from
   legoesm.constants, saturation only from legoesm.thermo, every new .py has a
   direct test.

Report findings as SEVERITY: file:line: problem -> fix. Be specific and
adversarial; say plainly if something is fine. Do not restate the diff.
PROMPT
)" 2>&1 | tail -180
echo "codex rc=${PIPESTATUS[0]}"
git -C "$SRC" worktree remove --force "$WT" >/dev/null 2>&1
rm -rf "$CODEX_HOME"
echo "ALLDONE"
