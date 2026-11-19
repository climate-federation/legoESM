#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=sp_ecrad_or
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/spread/ecrad_ifs49/oracle-%j.out
set -e
module load gcc/13.0.1
export NETCDF_PREFIX=/burg-archive/glab/users/jn2808/.conda/envs/legoesm
S=/burg/glab/users/pg2328/oracle_audit/spread/ecrad_ifs49
W=/burg/glab/users/pg2328/wt_ifs_ecrad
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "C5 $(gfortran --version | head -1); node $(hostname)"; sha256sum /burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/bld49/ecrad49.x
$PY -I $W/scripts/validate/oracle/ecrad_ifs/spread_run.py oracle $S/runs
echo ORACLE_PHASE_DONE
