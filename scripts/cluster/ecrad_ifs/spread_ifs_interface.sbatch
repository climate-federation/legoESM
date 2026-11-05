#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/agents/ifs_interface/port/spread-%j.out
set -uo pipefail
module load gcc/13.0.1
W=/burg/glab/users/pg2328/wt_ifs_ecrad; cd $W
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/agents/ifs_interface/port
WD=$D/spread
S=$W/scripts/validate/oracle/ecrad_ifs/spread_run.py
export IFS49_BLD=$D/bld
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$(ls -d $W/packages/* | tr "\n" ":")$W/src:$W
[ "${SKIP_ORACLE:-0}" = 1 ] || $PY $S ifs_oracle $WD 2>&1 | grep -v cuInit | tail -30
$PY -m pytest -q -s -p no:cacheprovider tests/atmosphere/test_ecrad_ifs_interface_oracle.py -k carry_through > $WD/test.log 2>&1
grep -c RH_COMPOSED $WD/test.log
$PY $S ifs_port $WD $WD/test.log 2>&1 | grep -v "cuInit\|cuda\|Traceback\|File \|\^\^\|RuntimeError\|plugin\|for d in" | tail -110
