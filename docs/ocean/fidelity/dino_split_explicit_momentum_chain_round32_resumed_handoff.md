# Round-32 corrected scorer-only resume

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Blocks 1--3 are retained: OFF/ON
counts are 223/226, both runs ended `STOP 0`, the three 6,184,192-byte streams
are present, and bracket SHA-256 is
`dc64d95988af1b3092960744ace65e7f40abd78cf004a7303534f6aa18f3ae49`.

The corrected scorer intercepts the faithful card's 68-row `fori_loop` and
aligns its carry-in with NEMO's pre-update trace. It also binds NEMO trace rows
1/2 exactly to the existing entry/post-substep-1 dumps. Only this block must
rerun; do not rebuild or rerun either NEMO arm.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=a23e54f9bf57d692e0bfa6d1b90f380b569f1b7a
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"
script="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round32.py"
test "$(sha256sum "$script" | awk '{print $1}')" = 7b31f286c8dec729c223c6f509e80dab21923e60d427dffa4f3db1bc0d14dc5f
test "$(sha256sum /tmp/dino_spg_row14_substep_trace_bracket.json | awk '{print $1}')" = dc64d95988af1b3092960744ace65e7f40abd78cf004a7303534f6aa18f3ae49
test "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round31_production_replay.json | awk '{print $1}')" = 7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d
OFF=$(cat /tmp/row32-off-run.txt)
ON=$(cat /tmp/row32-on-run.txt)
test "$OFF" = /tmp/RUN_SPG_ROW14_TRACE_OFF.XcuJqr
test "$ON" = /tmp/RUN_SPG_ROW14_TRACE_ON.76V2Lw
test "$(find "$OFF" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 223
test "$(find "$ON" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 226
grep -q '^STOP 0$' "$OFF/run.attempt1.log"
grep -q '^STOP 0$' "$ON/run.attempt1.log"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$script" \
  --on-run "$ON" --off-run "$OFF" \
  --bracket /tmp/dino_spg_row14_substep_trace_bracket.json \
  --round31 /tmp/dino_split_explicit_momentum_chain_round31_production_replay.json \
  --output /tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json
```

Expected control print from a checkout-local CPU dry validation:
`disposition=NO_STRICT_SSH_FAILURE` and
`first_strict_failure_substep={'ssh': None, 'u': None, 'v': None}`. The dry
artifact is diagnostic only; the output from the block above is the formal
receipt.
