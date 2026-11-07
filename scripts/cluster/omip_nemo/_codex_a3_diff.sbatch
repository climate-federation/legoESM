#!/bin/bash
#SBATCH --job-name=cdx_a3bbl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_a3_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_a3_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review commit f6547d3a9 (git show f6547d3a9): edge-indexed
Voronoi port of the NEMO trabbl nn_bbl_adv=2 advective bottom boundary
layer (new packages/ocean/legoesm/ocean/physics/bbl_adv_mpas.py + driver
wiring in scripts/run/run_omip_core2.py). The structured twin is
ocean/physics/bbl_adv.py. Attack ONLY: (1) any physics/indexing divergence
from the structured twin the two-cell test would not catch (multi-edge
convergence on one deep cell, k_s/k_d selection when both columns share the
bottom level, mgrh==0 handling); (2) the driver wiring — bbl_face_widths
tuple shape for mpas vs the structured branch, the apply-site dispatch, any
path where the mpas branch leaves variables undefined; (3) the exchange cap
on edges vs faces; (4) conservation of the .at[].add scatter under
duplicate edge indices; (5) any silent behaviour change for tripole/latlon.
Rank CRITICAL/MAJOR/MINOR, one line each + file:line. No summary." > "$LOG" 2>&1
rc=$?
tail -60 "$LOG"
echo "=== codex rc=$rc ==="
echo DONE

