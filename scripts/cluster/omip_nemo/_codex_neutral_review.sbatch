#!/bin/bash
#SBATCH --job-name=cdxneutral
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxneutral_%j.out
#
# Adversarial review of the MPAS harmonisation NEUTRAL verdict.
# The question that matters is #2: a null measured on a package of six fields
# does not license a statement about any one of them, and the claim being
# defended here ("lateral viscosity eliminated again") is exactly such a
# statement. Asking the reviewer to name the discriminating measurement rather
# than to agree or disagree.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
Q=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_neutral_q.txt
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
codex exec --sandbox read-only -C $REPO "$(cat $Q)" < /dev/null
echo "=== CODEXRC=$? ==="
