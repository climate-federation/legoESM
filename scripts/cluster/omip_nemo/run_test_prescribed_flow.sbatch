#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pflow_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pflow_test_%j.out

# Unit gate for the IN-MODEL prescribed-flow lever
# (LatLonCGridOceanConfig.prescribed_flow: "zero" | "frozen"):
#   - zero:   u/v/eta == 0 after every REAL model.step; T-anomaly COM fixed
#             (no advection); T changed in the vertical (physics active)
#   - frozen: u/v/eta == entry values exactly after N steps
#   - default None byte-identity (untraced static gate)
#   - construction ValueErrors (unknown mode, implicit_unsplit; GM/Redi both
#     model-level gm_redi [incl. prognostic EKE] and pipeline
#     lateral_mixing="gm_redi"; Fox-Kemper MLE) -- codex findings 2+3
#   - outer-AB2 stale-carry neutralization (u/v_incr_prev)
#   - inner tracer-AB2: poisoned T/S_flux_div_prev neither consumed nor
#     re-written (byte-identical to euler zero-mode) -- codex finding 1
#   - ew_cyclic_overlap slaves T/S only (no u/v/eta unpin) -- codex finding 4
#   - NamedTuple tail append: positional construction unshifted -- finding 5a
#   - run_omip_core2 CLI gates (grid + --spinup-drag rejection + the
#     --no-gm-redi requirement)
# CPU-only (the test module pins JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1).
set -e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
echo "=== worktree ==="; git -C $WT log --oneline -1; git -C $WT status --short | head

$PY -m pytest tests/ocean/unit/test_prescribed_flow.py -v -x --tb=short

# Regression neighbours: the lever must not perturb the default path the
# AB2-scope suite pins byte-identically.
$PY -m pytest tests/ocean/unit/test_ab2_scope.py -v --tb=short
