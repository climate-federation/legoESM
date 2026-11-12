#!/bin/bash
# usage: si3_prather_multicycle_seed_run.sh <rundir> <reload dir> <delayed cfl>   (tier-4 run_dyn.sh 'stop' + restart delayed-CFL edit on COPIES)
set -eu
RD=$1; REL=$2; CFL=$3
T=/burg/glab/users/pg2328/oracle_audit/tier4/dyn/run_template
rm -rf $RD; mkdir -p $RD; \cp -a $T/. $RD/
for c in rhg adv rdg cor; do touch $RD/t4_$c.on; done; touch $RD/t4_dyn.stop
for f in ORCA1_00017280_restart_oce.nc ORCA1_00017280_restart_ice.nc; do rm -f $RD/$f; \cp -fL $T/$f $RD/$f; test ! -L $RD/$f; done
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -I -c "
import netCDF4,sys
for f in ('ORCA1_00017280_restart_oce.nc','ORCA1_00017280_restart_ice.nc'):
    with netCDF4.Dataset('$RD/'+f,'r+') as d: d['DELAY__r8_cflice'][:]=float('$CFL')"
for f in $REL/t4_*_reload_*.bin; do ln -s $f $RD/$(basename $f); done
WS=/burg-archive/glab/users/pg2328/nemo_orca1
singularity exec --writable-tmpfs --bind $WS:/home/jdoe/to_host --bind /burg/glab/users/pg2328/oracle_audit:/burg/glab/users/pg2328/oracle_audit \
  $WS/morays_env_amd64.sif bash -lc "cd $RD && ulimit -s unlimited && mpirun --oversubscribe -np 32 ./nemo.exe > nemo_stdout.txt 2>&1"
grep -i 'number of cycles' $RD/ocean.output | head -2
