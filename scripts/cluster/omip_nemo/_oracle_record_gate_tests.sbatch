#!/bin/bash
#SBATCH --job-name=oraclerec
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/oraclerec_%j.out
#
# Gate for the ORACLE-RECORD check the user authorised on 2026-09-19.
#
# THE DEFECT: --nemo-time-idx defaults to -1, the LAST record of the oracle
# file.  A card that omitted the flag scored its day-30 state against NEMO's
# day 90, and the 70-day misalignment came back labelled an MLD regression.
# The snapshot side of the comparison has been pinned since 2026-09-15; the
# oracle side was not pinned by anything.
#
# NON-VACUITY: the gate is a NEW function, so there is no pre-fix commit to
# run it against -- the pre-fix behaviour was NO CHECK AT ALL.  Proof that it
# is not vacuous is therefore that it RAISES on the exact input the defect
# produced (day 20 wanted, -1 given, 18 records present), which is what
# test_the_default_last_record_is_refused_on_a_long_oracle_file asserts, and
# that it stays silent on the three inputs that are legitimately fine.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_merge
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO log --oneline -1)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO:$REPO/scripts/validate"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "########## the whole scorer test module, new gate included"
$PY -u -m pytest -q tests/validate/test_compare_three_way_nemo.py
RC=$?
echo "=== rc=$RC"

echo; echo "########## the scorer still parses and imports after the edit"
$PY -u -c "
import importlib.util, sys
p='$REPO/scripts/validate/ocean_fidelity/compare_three_way_nemo.py'
sys.path.insert(0,'$REPO/scripts/validate')
s=importlib.util.spec_from_file_location('c3w',p); mod=importlib.util.module_from_spec(s)
s.loader.exec_module(mod)
print('NEMO_RECORD_DAYS =', mod.NEMO_RECORD_DAYS)
print('gate is callable:', callable(mod.check_oracle_record))
"
IRC=$?
echo "=== import rc=$IRC"

echo; echo "########## NON-VACUITY: delete the CALL, the wiring test must go red"
# Codex's finding: every helper test passes with the call in main removed, so
# only a test that drives main can pin enforcement.  That claim is itself a
# claim, so prove it -- run the same test file against a copy of the scorer
# with exactly that one line deleted and require a FAILURE.
COPY=$(mktemp -d /tmp/c3w_novac_XXXX)/compare_three_way_nemo.py
grep -v '^    check_oracle_record(a\.expect_day' \
  $REPO/scripts/validate/ocean_fidelity/compare_three_way_nemo.py > "$COPY"
DELETED=$(( $(wc -l < $REPO/scripts/validate/ocean_fidelity/compare_three_way_nemo.py) \
            - $(wc -l < "$COPY") ))
echo "lines deleted from the copy: $DELETED (MUST be 1)"
[ "$DELETED" = "1" ] || { echo "FATAL: the call line was not matched"; exit 1; }
C3W_MODULE="$COPY" $PY -u -m pytest -q \
  tests/validate/test_compare_three_way_nemo.py::test_the_scorer_actually_calls_the_gate
NVRC=$?
echo "=== gate-removed rc=$NVRC (EXPECT NON-ZERO; 0 => the wiring test is VACUOUS)"
rm -rf "$(dirname "$COPY")"

echo; echo "=== ALL DONE (pytest $RC, import $IRC, non-vacuity $NVRC) ==="
exit $(( RC != 0 || IRC != 0 || NVRC == 0 ))
