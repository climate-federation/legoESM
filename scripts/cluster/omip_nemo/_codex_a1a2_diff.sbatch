#!/bin/bash
#SBATCH --job-name=cdx_a1a2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_a1a2_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_a1a2_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review commit af4420b9e (git show af4420b9e): wires NEMO
--isf (prescribed ice-shelf melt) and --iwm (zdfiwm wave mixing) onto the
MPAS Voronoi ocean lane. 30-day arms are queued; a CRITICAL cancels them.
Attack ONLY: (1) the paired-cells regrid branches — any remaining
meshgrid/2-D assumption reachable on the (nCells,) path, incl. the
gradient-based source-area estimate at curvilinear folds; (2) the MPAS
model-level K_iwm splice — placement vs the masks and the tracer/momentum
solves, dtype, and whether the momentum carrier addition double-counts when
KPP is active; (3) the driver wiring — is iwm threaded into the zdftke card
in a way that actually flows rn_emin=1e-10/rmxl_min=1e-3, and do BOTH
config-replace rebuild sites carry the maps; (4) the isf host-loop apply on
MPAS state shapes (eta/H_bathy/compute_layer_thickness on (nCells,)); (5)
any silent behaviour change for the EXISTING tripole/latlon paths. Rank
CRITICAL/MAJOR/MINOR, one line each + file:line. No summary." > "$LOG" 2>&1
rc=$?
tail -60 "$LOG"
echo "=== codex rc=$rc ==="
echo DONE
