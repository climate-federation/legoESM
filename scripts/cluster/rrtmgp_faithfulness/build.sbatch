#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --job-name=rrtmgp_build
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/scratch/b/b381103/rrtmgp_audit/build_%j.log
set -euo pipefail
export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH
bash /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt_rrtmgp_audit/scripts/validate/oracle/rrtmgp_faithfulness/build.sh /scratch/b/b381103/rrtmgp_audit/ref/rte-rrtmgp /scratch/b/b381103/rrtmgp_audit/build
