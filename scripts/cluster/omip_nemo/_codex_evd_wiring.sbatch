#!/bin/bash
#SBATCH --job-name=cdx_evdw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_evdw_%j.out
# Diff review: the EVD trigger flags + firing-occupancy diagnostic (uncommitted).
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) + uncommitted ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_evdw_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review the UNCOMMITTED diff (git diff; new files: tests/unit/test_run_omip_core2_evd_trigger_cli.py, scripts/cluster/omip_nemo/_trp_evd_d30.sbatch). Purpose: wire NEMO zdfevd's trigger into the OMIP driver (scripts/run/run_omip_core2.py): new flags --convection-n2-mode/--convection-n2-eos/--convection-trigger/--convection-n2-threshold/--convection-two-level build an EnhancedDiffusionConfig (build_enhanced_diffusion_config) passed to build_tripole(convection_cfg=); plus --evd-occupancy-every-hours, an hourly sample in the per-step host loop of the fraction of wet columns whose FIRST interior interface carries K_conv (evd_top_interface_occupancy on _kprofiles' K_H_diag), averaged into two new diag_timeseries.csv columns. No default changes intended. Attack: (1) any path where a trigger flag is accepted but the config the model runs does not carry it (MPAS/FESOM/latlon builders, the restart path, the visc-schedule model rebuild at ~line 8810+ which rebuilds config+model -- does it keep convection_cfg?); (2) is K_H_diag from diagnose_vertical_K the diffusivity the implicit solve actually applies, including the EVD branch (k_profiles.py compute_vertical_K_profiles / enhanced-diffusion composition, vmix_background_mode), and does index 0 = the interface between cells 1 and 2; (3) the hourly sampler's cost/side effects (does _kprofiles mutate state, recompile, or run every step); (4) csv header/row misalignment when restarting a leg (append path); (5) two_level_trigger under the rk3 outer integrator (before_tracers None -> silently now-only?) -- should the card refuse or note; (6) the card's pre-registration: can it be gamed or fail to discriminate. Rank CRITICAL/MAJOR/MINOR, one line each, no summary." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -60 "$LOG"
echo DONE
