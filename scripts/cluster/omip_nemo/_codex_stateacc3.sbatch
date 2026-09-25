#!/bin/bash
#SBATCH --job-name=cdx_stacc3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_stacc3_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Round 3. Review commit HEAD (git show HEAD) against your round-2 HOLD. Changes: (3) the single _flux_acc.add now sits after the last in-step state assignment (after the spin-up drag _replace, before the EVD occupancy block), with a source test forbidding any state-changing assignment between it and the snapshot; (6) accumulator sums T*h, S*h, h with h = compute_layer_thickness(eta, H_bathy, z_coord) each step and drains T_mean_hw/S_mean_hw/h_mean (NaN dry) beside the plain T_mean/S_mean; (2) probe requires the tiled windows to span exactly [5k, 5k+5] and finite durations; (4) both sub-surface regrid coverage masks enter the support.
Numbered, severity-tagged, file:line: (a) anything still open? (b) is compute_layer_thickness(eta, H_bathy, z_coord) the live e3t equivalent NEMO weights with (read packages/ocean/legoesm/ocean/vertical.py:1350) and is it the thickness the model actually used this step (eta AFTER the step)? (c) the add is now outside both lane branches: is sf defined and the applied forcing in both lanes at that point, and is the add reachable on the persistent/SPMD lane without a gather (state may be rank-local)? (d) does the extra compute_layer_thickness per step force a host transfer or retrace? (e) any remaining test vacuity." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="


