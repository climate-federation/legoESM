#!/bin/bash
#SBATCH --job-name=cdx_scorer3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_scorer3_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Round 3 on commit HEAD (git show HEAD): fields_mode now at report level in compare_omip_nemo.py; --use-mean-fields with --nemo-month is a SystemExit in both scorers. Numbered, severity-tagged, file:line: anything still open or newly broken? Is fields_mode defined on every path that reaches the report dict? One line verdict: CLEAN or HOLD." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="





