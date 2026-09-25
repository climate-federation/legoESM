#!/bin/bash
#SBATCH --job-name=cdx_scorer2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_scorer2_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Round 2 on commit HEAD (git show HEAD) against your three findings on the previous commit: (P1) window length -> loader returns window_days from flux_mean_window_s, require_window() in compare_omip_nemo.py gates every loaded snapshot in both scorers against NEMO_RECORD_DAYS=5; (P2) fields_mode recorded in both reports, MLD descriptions mode-dependent; (P2) pair scorer mean-MLD path no longer needs geometry. Numbered, severity-tagged, file:line: anything still open, any new defect (e.g. the record-length constant vs --nemo-month monthly climatology mode, or require_window called before/after the oracle-record gate), and whether the extended test would fail if require_window were made a no-op." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="




