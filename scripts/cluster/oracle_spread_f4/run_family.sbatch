#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/f4/logs/%x_%j.out
# usage: sbatch -J <family> run_family.sbatch <family>
set -euo pipefail
FAM=$1
H=/burg/glab/users/pg2328/oracle_audit/f4
WT=/burg/glab/users/pg2328/wt_f4
export PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$(ls -d $WT/packages/* | tr "\n" ":")$WT:$H
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
R=$H/runs/$FAM
rm -rf "$R"; mkdir -p "$R/tmp" $H/build
export TMPDIR=$R/tmp   # generator mkdtemp builds land here (C5 binary check)
echo "WT $(git -C $WT rev-parse HEAD)"
gfortran --version | head -1
gfortran -O0 -o $H/build/readback.x $H/readback.F90
export F4_READBACK=$H/build/readback.x
BUILD=$($PY -c "import sys; sys.path.insert(0,'$H'); from families import FAMILIES; print(FAMILIES['$FAM']['build'])")
echo "BUILD: $BUILD"
eval "$BUILD" > $R/build.log 2>&1 || { cat $R/build.log; echo BUILD_FAILED; exit 1; }
tail -3 $R/build.log
MODES="base base2 s0r $(seq -f 's%g' 0 19)"
printf '%s\n' $MODES | xargs -P 8 -I{} sh -c "cd $R && $PY $H/gen_intercept.py $FAM {} $R > $R/gen_{}.log 2>&1 || echo GEN_FAIL {}"
grep -h "GEN_DONE\|GEN_FAIL" $R/gen_*.log | sort | uniq -c | head -30
cat $R/build.log $R/gen_base.log > $R/buildgen.log
FIXT=$($PY -c "import sys; sys.path.insert(0,'$H'); from families import FAMILIES as F; f=F['$FAM']; print(f['test'], f['fix_attr'], ','.join(f['metrics']), f['fixture'])")
read TEST ATTR MET FIXC <<< "$FIXT"
SEL=$($PY -c "import sys; sys.path.insert(0,'$H'); from families import FAMILIES as F; print(F['$FAM']['select'])")
spy() {  # tag fixture
  F4_NEUTRAL=${F4_NEUTRAL:-0} F4_FIX_ATTR=$ATTR F4_FIX_PATH=$2 F4_METRICS=$MET F4_OUT=$R/spy_$1.pkl \
    $PY -m pytest "$TEST" -p f4_spy -p no:cacheprovider -q -k "$SEL" > $R/spy_$1.log 2>&1 || true
  tail -1 $R/spy_$1.log
}
export -f spy; export R ATTR MET TEST SEL PY
spy committed $FIXC
( for t in base $(seq -f 's%g' 0 19); do echo $t; done ) | xargs -P 8 -I{} bash -c 'F4_NEUTRAL=$([ {} = base ] && echo 0 || echo 1) spy {} $R/{}/fixture.npz'
$PY $H/analyze.py $FAM $R $R/buildgen.log
echo HARNESS_DONE $FAM
