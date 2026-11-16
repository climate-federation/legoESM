#!/bin/bash
#SBATCH --job-name=cdxflux
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxflux_%j.out
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $REPO || exit 1
echo "HEAD: $(git log --oneline -1)"
echo "dirty: $(git status --porcelain | wc -l)"
P=$REPO/scripts/cluster/omip_nemo/_codex_flux_acc.prompt
[ -e "$P" ] || { echo "FATAL: missing $P"; exit 1; }
codex exec --sandbox read-only -C "$REPO" "$(cat $P)" < /dev/null 2>&1
echo "=== rc=$? ==="
