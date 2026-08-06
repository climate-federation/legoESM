#!/bin/bash
#SBATCH --job-name=ss_codex
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_codex_%j.log
# Adversarial review of the uniform super-saturation guard, on a PINNED
# detached worktree. CODEX_HOME is node-local (/tmp) per the 2026-07-27 note;
# auth is copied in and the "exists=yes auth=yes" line is echoed BEFORE the run
# so a review that ran without credentials is not mistaken for a clean verdict.
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
BASE_SHA="${BASE_SHA:?set BASE_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wtcodex_${SLURM_JOB_ID}"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT" "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
echo "PINNED sha=$(git -C "$WT" rev-parse HEAD)"

export CODEX_HOME="/tmp/codex_home_${SLURM_JOB_ID}"
mkdir -p "$CODEX_HOME"
cp -a "$HOME/.codex/." "$CODEX_HOME/" 2>/dev/null || true
echo "CODEX_HOME $CODEX_HOME exists=$([ -d "$CODEX_HOME" ] && echo yes || echo no) auth=$([ -s "$CODEX_HOME/auth.json" ] && echo yes || echo no)"
export PATH="$HOME/.nvm/versions/node/v24.16.0/bin:$PATH"
echo "codex=$(command -v codex)"

git -C "$WT" diff "$BASE_SHA".."$PINNED_SHA" -- \
  packages/atmosphere/legoesm/atmosphere/physics/microphysics/ \
  packages/coupler/legoesm/driver/config.py \
  tests/ > "$RUNS/ssguard_diff_${SLURM_JOB_ID}.patch"
echo "diff lines=$(wc -l < "$RUNS/ssguard_diff_${SLURM_JOB_ID}.patch")"

read -r -d '' PROMPT <<'EOF'
Adversarial review. This diff makes the hard (iterated) saturation-adjustment
guard UNIFORM across every microphysics scheme in legoESM and adds a CI ratchet.
Assume it is WRONG and try to prove it. Be concrete: file:line, the failure
mode, and the input that triggers it. Do not praise.

Read (in the worktree, not from memory):
  packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py
    (hard_saturation_blend, hard_saturation_drain, _hard_saturation_condensation,
     _hard_saturation_rate_limit, saturation_adjustment)
  packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py
  packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py
  packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py
  packages/coupler/legoesm/driver/config.py
  tests/test_microphysics_supersaturation_guard.py
  tests/unit/test_supersaturation_guard_behaviour.py

Attack specifically:
 1. SIGN + BUDGET. In sundqvist.py the blended rate feeds
    dq_v_dt=-cond+evap, dq_c_dt=+cond-auto, dT_dt=L_v*(cond-evap)/c_pd. Can the
    overlay make cond NEGATIVE (spurious evaporation), or break
    c_pd*T + L_v*q_v conservation, or drive q_c/q_v negative through the
    qc_avail donor cap? In ml_emulator.py the drain is computed at the POST-step
    state (T+dT_dt*dt, q_v+dq_v_dt*dt) but applied as a rate to the tendencies:
    is q_v guaranteed >= 0 after the correction? Is the enthalpy pairing right?
 2. BYTE-IDENTITY. Is the OFF path really unchanged? Look for any dtype
    promotion, retrace, or reordering introduced even when the flag is False.
 3. AD / JIT. hard_saturation_blend contains stop_gradient + a jvp; is it safe
    inside sundqvist's lax.scan-bearing function and inside the emulator's
    double vmap? Any NaN gradient at q_sat -> 0 (205 K, 150 hPa)?
 4. THE RATCHET. tests/test_microphysics_supersaturation_guard.py enumerates
    the factory by AST. Give a concrete way a NEW unguarded scheme could be
    added and still pass. Are the self-tests genuinely non-vacuous, or would
    they pass with the feature removed?
 5. THE EXEMPTIONS. sdm / fast_sbm / none are exempt. Read those modules and
    say whether the stated reasons are TRUE. If any of them can in fact leave a
    super-saturated bulk state, say so.
 6. THE BEHAVIOURAL TEST. Are its thresholds measuring what their names claim,
    or could they pass for the wrong reason (e.g. a scheme that reaches RH<=1 by
    destroying water rather than condensing it)?
Rank findings Critical / High / Medium / Low.
EOF

cd "$WT"
timeout 4200 codex exec --sandbox read-only -C "$WT" "$PROMPT" 2>&1 | tee "$RUNS/codex_out_${SLURM_JOB_ID}.txt"
echo "CODEX_RC=${PIPESTATUS[0]}"
echo "ALL DONE job=${SLURM_JOB_ID}"
