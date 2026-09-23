#!/bin/bash
#SBATCH --job-name=cdx_unify
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_unify_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_unify_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review docs/ocean/fidelity/three_grid_unification_design.md
(design only, no code yet). Verify its file:line claims against the repo.
Attack: (1) Part A2 — is the additive K_iwm splice point at
mpas_integration.py:909-917 actually BEFORE the masks, and does un-hardcoding
iwm=None at run_omip_core2.py:6055 flow rn_emin/rmxl_min correctly? (2) Part
A1 — any isf consumer beyond the host-loop apply site that stays 2-D? (3)
Part B3 — sign/unit conventions of fesom_jax SurfaceFluxes (bc_T, water_flux,
stress_surf) vs legoESM OceanSurfaceForcing: name each conversion the design
missed. (4) Part B — any double-counting path the design's guards miss
(runoff, SSS restore, ice fluxes). (5) Anything in the staging order that
invalidates the one-variable discipline. Rank CRITICAL/MAJOR/MINOR, one line
each + file:line evidence. No summary." > "$LOG" 2>&1
rc=$?
tail -80 "$LOG"
echo "=== codex rc=$rc ==="
grep -q "Failed to refresh token\|401 Unauthorized\|token_expired" "$LOG" && { echo "FATAL: codex auth failed — UNREVIEWED"; exit 2; }
echo DONE
