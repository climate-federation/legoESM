#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=les_suite_campaign
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100s
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=1-23:59
#SBATCH --output=results/les_suite/campaign/slurm-%j.out
#SBATCH --mail-user=ac5006@columbia.edu
#SBATCH --mail-type=END,FAIL
#
# LES-truth suite campaign (LES_SUITE.md §7). Resumable: re-submit to continue.
# Output lands under results/les_suite/ (see docs/atmosphere/les_suite/CAMPAIGN.md).
set -e
cd /burg-archive/glab/users/ac5006/legoESM
mkdir -p results/les_suite/campaign
bash scripts/cluster/les_suite/run_les_suite_campaign.sh
