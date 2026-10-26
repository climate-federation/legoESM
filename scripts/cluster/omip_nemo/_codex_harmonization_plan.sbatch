#!/bin/bash
#SBATCH --job-name=cdx_plan
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_plan_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_plan_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review docs/ocean/fidelity/harmonization_plan_2026-09-05.md (read it) — a PLAN, no code yet — for unifying three ocean grids (tripole eORCA1, MPAS ico7, FESOM2 CORE2 via fesom_jax) against the NEMO ORCA1 oracle. Attack: (1) P1: is the NEMO file-viscosity option (--A-h-profile-file, scripts/run/run_omip_core2.py ~5394, ah_profile_from_file ~943) a faithful transcription of nn_ahm_ijk_t=-30 with eddy_viscosity_3D.nc, or only a zonal-median latitudinal shape? Does the ladder isolate ONE variable (Smagorinsky must be off in (b)? the tripole card carries --visc-schedule with Smag 3.0->0.5 over 16 days)? Which arm would you drop or add? (2) P2: is frozen_column_tke_twin.py usable for CVMix-TKE vs our TKE, and what is the decisive metric? (3) P3: is inferring FESOM's level rule from nlvls.out/elvls.out sound (name the FESOM2 rule if you know it: elvls from the element's shallowest node? nlvls = max over adjacent elements? minimum 3 levels?), and what breaks in fesom_jax with a 1 m top layer at dt 1800 s? (4) Anything in the plan that would silently change more than one variable vs the 2026-09-04 baselines? Rank CRITICAL/MAJOR/MINOR, cite file:line, under 450 words." > $LOG 2>&1
echo "codex rc=$?"; grep -vE "^\s*$" $LOG | tail -60; echo DONE
