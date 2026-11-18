#!/bin/bash
#SBATCH --job-name=cdxnino3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxnino3_%j.out
#
# Adversarial review of a CLAIM, before it drives any code change.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_merge; cd $REPO || exit 1
echo "HEAD: $(git log --oneline -1)"
P=$REPO/scripts/cluster/omip_nemo/_codex_nino3_bins.prompt
[ -e "$P" ] || { echo "FATAL: missing $P"; exit 1; }
codex exec --sandbox read-only -C "$REPO" "$(cat $P)" < /dev/null 2>&1
echo "=== rc=$? ==="
