#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_reach
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/cdx_reach_%j.out
set +e
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)  CODEX_HOME=$CODEX_HOME"; codex --version || exit 91
cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
REVIEW ONE NEW PROBE, then stop. Numbered answers, defects only.

FILE: scripts/validate/check_convection_param_reachability.py

THE QUESTION IT EXISTS FOR. In an SCM-RCE tuning campaign, `tiedtke` (16
tunable params, 192 evaluations) and `kuo` (2, 48) returned a tuned score
BIT-IDENTICAL to their defaults, while bechtold/sbm/mass_flux moved 40-85 %.

ALREADY ESTABLISHED by direct check -- do not re-derive, but DO tell me if any
of it is wrong:
* the parameters exist as fields on the scheme config NamedTuple;
* the tuner's override chain writes them: fields change AND
  _config_cache_key differs, so run_cached really evaluates distinct configs;
* both schemes read the parameters in their bodies (kuo.py:205,510;
  tiedtke.py:357,361,365).

Q1. Does the probe actually isolate what it claims? It builds a
convection-ONLY PhysicsConfig and calls make_physics per trial. Is the
tendency it reads (`tend.dT_dt`, `tend.tracer_tendencies["q_v"]`) the
scheme's, or could it be zero for a reason unrelated to the parameter --
e.g. a trigger needing state the synthetic column lacks (no q_c/q_i, no
phys_state carry, tracers limited to q_v/q_c/q_r), or a scheme that needs a
`phys_state` it is being passed None for?

Q2. Is `_column` a physically valid input? Sigma-coordinate temperature and
humidity built from a dry hydrostatic height estimate, uniform RH, lapse
capped at 200 K. Would any of the five states be REJECTED by a scheme's own
sanity guards, or produce a trigger verdict that is an artifact of the
construction rather than of the physics?

Q3. Is the verdict logic sound? "DEAD" = exactly zero change in both
max|d(dT/dt)| and max|d(dq_v/dt)| across every state and every trial value.
Can a genuinely live parameter produce exactly zero under this test? Can a
genuinely dead one produce nonzero?

Q4. The sweep is 7 fractions of the declared range plus 4 multiples of the
default, clipped to the range. Is that enough to straddle a THRESHOLD
parameter, given the threshold is compared against a state-dependent quantity
the probe never computes? If not, what is the smallest addition that fixes it?

Q5. GLM-5.2 argued these two schemes are UNIDENTIFIABLE in this configuration
on physical grounds: both use moisture-convergence closures, and with no
imposed large-scale omega and E ~ P at equilibrium the convergence supply
vanishes, so the closure degenerates; and kuo's Anthes partition
`clip(1 - rhmean - anthes_rh_offset, 0, 1)` is identically 0 for any offset in
[0, 0.3] once rhmean >= 1. Read kuo.py and tiedtke.py and say whether that is
correct, partially correct, or wrong -- with the lines that decide it.

Q6. Anything else wrong with the probe, and anything it should measure that
it does not.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN"
echo ALLDONE
