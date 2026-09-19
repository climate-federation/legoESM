#!/bin/bash
#SBATCH --job-name=cdx_evda
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_evda_%j.out
# Pre-launch review of the EVD arm card at the committed HEAD.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_evda_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "FINAL PRE-LAUNCH REVIEW of scripts/cluster/omip_nemo/_trp_evd_d30.sbatch (about to consume ~14 GPU-hours in two legs). It is the previous baseline card _trp_base2_d30.sbatch plus four flags: --convection enhanced_diffusion --convection-K-conv 100 --convection-K-bg 0 --convection-n2-mode nemo_bn2 --convection-n2-eos teos10 --convection-trigger hard --convection-n2-threshold=-1e-12 --evd-occupancy-every-hours 1. Purpose: NEMO ORCA1 runs ln_zdfevd=T rn_evd=100 nn_evdm=0 and its 5-day-mean avt in the equatorial cold tongue is 50 m2/s in the top 1-2 m with convective occupancy 0.5-1.0, i.e. it convects there every night; our runs have convection 'none' and carry a 3.8 K warm lens over the top 20 m where NEMO has 0.76 K. Answer, each in one line, ranked CRITICAL/MAJOR/MINOR: (1) diff _trp_evd_d30.sbatch against _trp_base2_d30.sbatch and confirm the ONLY differences are those flags, the job name/output/OUT dir, the header and the added scoring block -- name any other difference; (2) will K_conv=100 m2/s on tracers with dt=150 s and a 1 m top cell be stable and conservative through the implicit backward-Euler vertical solve (packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py _apply_implicit_vertical_mixing), and does K_bg=0 remove any floor the rest of the column relies on (note the config background K_v is added separately); (3) does nu_conv default 0 really leave momentum untouched on the lat-lon C-grid path; (4) the leg-2 restart: does the new csv column set break the append path in scripts/run/run_omip_core2.py (header written only on a fresh file), and does snapshot_day0015 get copied as in the base2 card; (5) the manifest gate heredoc at the top of the scoring block -- will the key path config.resolved_config.runtime_config.physics.convection.enhanced_diffusion exist in run_manifest.json for a tripole run, and will a failure there abort before scoring; (6) anything in the card that would make the pre-registered endpoint (220-240E T(0.5 m)-T(20 m) at day 30 vs the base2 arm at the same 00Z sampling, threshold 1.9 K) unmeasurable. No summary." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -40 "$LOG"
echo DONE
