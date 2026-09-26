# Round-26 retained-bracket scorer handoff

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Blocks 1--3 are retained without qualification. Their exact bracket is
`/tmp/dino_spg_row13_bottom_update_bracket.json`, SHA-256
`f9ff8f957c3228a882339b7ce5e5e423ad22f4e0d0e7f9839dc761f242841854`;
the ON/OFF manifests contain 223/211 streams and every one of the twelve new
round-26 streams is full-halo. The failed score emitted no artifact.

The writer patch is unchanged, SHA-256
`3df88df75a243bdb2bc2e5ceb5bef6bff771b9ef00989c30067d77c832e85f8c`.
The corrected scorer recognizes only `spg_dump_zu_frc.bin` and
`spg_dump_zv_frc.bin` as interior `A2D(0)` streams, matching
`dynspg_ts.F90:168`; all new streams retain the full-halo loader.

## Corrected block 4 — CPU/fp64 score from retained runs

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=b15557ae07cd7e858db3b1c73bf6cec6d690e99e
BRACKET_SHA=f9ff8f957c3228a882339b7ce5e5e423ad22f4e0d0e7f9839dc761f242841854
OFF=$(cat /tmp/row26-off-run.txt)
ON=$(cat /tmp/row26-on-run.txt)
test "$OFF" = /tmp/RUN_SPG_ROW13_BOTTOM_UPDATE_OFF.FoO6YS
test "$ON" = /tmp/RUN_SPG_ROW13_BOTTOM_UPDATE_ON.VlHrvW
test "$(sha256sum /tmp/dino_spg_row13_bottom_update_bracket.json | awk '{print $1}')" = "$BRACKET_SHA"
test "$(find "$OFF" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 211
test "$(find "$ON" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 223
test "$(stat -c %s "$ON/spg_dump_zu_frc.bin")" -eq 82784
test "$(stat -c %s "$ON/spg_dump_zv_frc.bin")" -eq 82784
for file in "$ON"/row13_dump_*_substep1.bin; do test "$(stat -c %s "$file")" -eq 90944; done
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round26.py" \
  --on-run "$ON" --off-run "$OFF" \
  --bracket /tmp/dino_spg_row13_bottom_update_bracket.json \
  --bracket-sha "$BRACKET_SHA" --producer "$producer" \
  --round25 /tmp/dino_split_explicit_momentum_chain_round25_association_factorial.json \
  --output /tmp/dino_split_explicit_momentum_chain_round26_bottom_update.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round26_bottom_update.json
```

This block does not build NEMO, invoke a GPU, use `mpirun`, or push.
