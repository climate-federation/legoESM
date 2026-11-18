#!/bin/bash
#SBATCH --job-name=cdxoracle
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:35:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxoracle_%j.out
#
# Adversarial review of the oracle-record gate (3868d7f3a).  Half of the
# mandatory dual review; GLM is the other half and runs separately.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_omip_merge
cd $WT || exit 1
echo "=== HEAD $(git -C $WT log --oneline -1)"

export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="

P=$(mktemp /tmp/cdx_oracle_XXXX.txt)
cat > "$P" <<'PROMPT'
ONE REVIEW. Attack this change and stop. Do not propose follow-up work.

FILE: scripts/validate/ocean_fidelity/compare_three_way_nemo.py
NEW FUNCTION: check_oracle_record, called from main right after the NEMO file
is loaded. TESTS: tests/validate/test_compare_three_way_nemo.py, the six at
the end of the file.

WHAT IT IS FOR. The scorer takes --expect-day, the run day being scored, and
--nemo-time-idx, which record of the oracle file to read. --nemo-time-idx
defaults to -1, the last record. A card that omitted the flag scored a day-30
state against the oracle's day 90 and the 70-day misalignment was reported as
a physics regression. The module docstring states, measured from
time_counter_bounds, that record k is the mean over run days 5k to 5k+5, so
the record ending on day D is index D/5 - 1.

ANSWER ONLY THESE, quoting the lines you rely on.

1. Is the index arithmetic right, including the negative-index resolution
   against n_time? Give an input where it accepts a record it should refuse,
   or refuses one it should accept, or say there is none.
2. int(round(expect_day / 5)) - 1 on a day that is NOT a multiple of five:
   what does it do, and is silently rounding the right behaviour or should a
   non-multiple be refused outright? Name which callers can pass one.
3. The gate is skipped when nemo_month is set. Confirm from the code that the
   monthly path really does ignore nemo_time_idx; if it does not, the
   exemption is a hole.
4. Are the six tests non-vacuous? For each, say what breaks if the assertion
   is deleted. Flag any that would pass with the gate removed.
5. The call sits AFTER _load_nemo. Is there any path where main can reach a
   scored number without passing through it?

Verdict line: SOUND or one defect named, file and line.
PROMPT

codex exec --sandbox read-only -C $WT "$(cat $P)" < /dev/null
echo "=== codex rc=$? ==="
rm -f "$P"
echo DONE
