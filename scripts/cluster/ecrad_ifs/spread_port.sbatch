#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --job-name=sp_ecrad_port
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/spread/ecrad_ifs49/port-%j.out
W=/burg/glab/users/pg2328/wt_ifs_ecrad; cd $W
S=/burg/glab/users/pg2328/oracle_audit/spread/ecrad_ifs49
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$(ls -d $W/packages/* | tr "\n" ":")$W/src:$W
echo "HEAD $(git rev-parse HEAD) node $(hostname)"; sha256sum packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/*.py tests/atmosphere/test_ecrad_ifs_end_to_end_oracle.py
$PY $W/scripts/validate/oracle/ecrad_ifs/spread_run.py port $S/runs ${TESTLOG} 2>&1 | grep -v -i "cuda\|xla_bridge\|plugin\|Traceback\|^ *\^\|initialize()" 
