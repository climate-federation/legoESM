# Round-41 row-5 WZV call-2 SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Producer
`205d31a61efa5484146534945970033a319e5354`.

This is a Python-only held capture from existing day-180 dumps. It does not
copy, patch, build, or run NEMO. The existing full-halo deterministic writer
already emits both `wzv_dump_ww_call1.bin` and
`wzv_dump_ww_call2.bin` through reserved unit 9103; block 1 scans the whole
NEMO tree and admits only its eight expected mirrored source/build-tree
`OPEN` lines (two branches in DINO/DINO_DBG MY_SRC/WORK). No new unit is
claimed.

Every block changes to the absolute checkout, exports the session and
checkout-first `PYTHONPATH`, pins the producer by SHA plus model-diff-zero,
uses a tracked-only porcelain gate, and keeps the `__MEASURED_` SLOT naming.
NEMO W streams are genuine full-halo `(36,203,56)` and are stripped by the
loader. The retained `mesh_mask.nc` and legoESM capture are cited genuine
interior shapes `(199,52,36)` and `(199,52,37)` respectively; neither is
silently padded or stripped twice.

## Block 1 — producer, source, unit, and Python-build admission

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=205d31a61efa5484146534945970033a319e5354
git cat-file -e "${producer}^{commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$producer" HEAD -- \
  packages/core packages/ocean \
  docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round41.md \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_capture.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_bracket.py)"

capture="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_capture.py"
bracket="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_bracket.py"
scorer="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41.py"
prereg="$repo/docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round41.md"
test "$(sha256sum "$capture" | awk '{print $1}')" = 2392d434cd48fa99a7f85194ae1d8dd65ffc51a9008cdaa84c0cd9109fa2a550
test "$(sha256sum "$bracket" | awk '{print $1}')" = 76dc4c1e7f9ba76f622885a2baa19d9acbc2f3456f8a4e08851d461687c9231b
test "$(sha256sum "$scorer" | awk '{print $1}')" = de50658cdd3d0be032fd906e70584abe3c56a89d685f3702a92659742aac88fa
test "$(sha256sum "$prereg" | awk '{print $1}')" = fe983f5a69aa98eb1f2a1d73be7bc22d67eace823eca6d41f5331560412c79da
test "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round40.json | awk '{print $1}')" = 723fe0e74724febab71537ef31cde16e32388c9da01c9058ef2c37629067258b

nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
unit_receipt=/tmp/dino-row41-unit-9103.txt
grep -R -n -E "OPEN\( *UNIT *= *9103" --include='*.F90' "$nemo" > "$unit_receipt"
test "$(wc -l < "$unit_receipt")" -eq 8
test -z "$(grep -v -E '/cfgs/(DINO|DINO_DBG)/(MY_SRC|WORK)/sshwzv.F90:.*wzv_dump_ww_call[12]\.bin' "$unit_receipt" || true)"
test "$(grep -c "wzv_dump_ww_call1.bin" "$unit_receipt")" -eq 4
test "$(grep -c "wzv_dump_ww_call2.bin" "$unit_receipt")" -eq 4

export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX=/tmp/dino-row41-pycache
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python -m py_compile "$capture" "$bracket" "$scorer"
/home/dbalwada/legoESM/.venv/bin/python "$capture" --help >/dev/null
/home/dbalwada/legoESM/.venv/bin/python "$bracket" --help >/dev/null
/home/dbalwada/legoESM/.venv/bin/python "$scorer" --help >/dev/null
printf 'SLOT __MEASURED_ROW41_PRODUCER__ VALUE=%s\n' "$producer"
printf 'SLOT __MEASURED_ROW41_UNIT_SCAN_SHA256__ VALUE=%s\n' "$(sha256sum "$unit_receipt" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW41_CAPTURE_SCRIPT_SHA256__ VALUE=%s\n' "$(sha256sum "$capture" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW41_BRACKET_SCRIPT_SHA256__ VALUE=%s\n' "$(sha256sum "$bracket" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW41_SCORER_SHA256__ VALUE=%s\n' "$(sha256sum "$scorer" | awk '{print $1}')"
```

## Block 2 — two fresh post-`dyn_zdf` captures

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW41_PRODUCER__
UNIT_SCAN_SHA=__MEASURED_ROW41_UNIT_SCAN_SHA256__
CAPTURE_SHA=__MEASURED_ROW41_CAPTURE_SCRIPT_SHA256__
for value in "$PRODUCER" "$UNIT_SCAN_SHA" "$CAPTURE_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 205d31a61efa5484146534945970033a319e5354
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$PRODUCER" HEAD -- packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_capture.py)"
test "$(sha256sum /tmp/dino-row41-unit-9103.txt | awk '{print $1}')" = "$UNIT_SCAN_SHA"
capture="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_capture.py"
test "$(sha256sum "$capture" | awk '{print $1}')" = "$CAPTURE_SHA"

stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
test "$(sha256sum "$stepdump/wzv_dump_ww_call1.bin" | awk '{print $1}')" = defad5014cc7210dd52d6373dafd46856471afedb892267cef30232fdd9baeb2
test "$(sha256sum "$stepdump/wzv_dump_ww_call2.bin" | awk '{print $1}')" = 895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634
test "$(sha256sum "$stepdump/mesh_mask.nc" | awk '{print $1}')" = 3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622
test "$(sha256sum "$stepdump/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e

run_root=/tmp/dino-split-momentum-row41-01a04e34
test ! -e "$run_root"
mkdir -p "$run_root/capture_a" "$run_root/capture_b"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX="$run_root/pycache"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
for out in "$run_root/capture_a" "$run_root/capture_b"; do
  cd "$repo"
  /home/dbalwada/legoESM/.venv/bin/python "$capture" \
    --run-stepdump "$stepdump" --run-traj "$traj" \
    --round40 /tmp/dino_split_explicit_momentum_chain_round40.json \
    --output-dir "$out"
  test "$(find "$out" -maxdepth 1 -type f | wc -l)" -eq 2
  test "$(stat -c %s "$out/wzv_call2_production.bin")" -eq 3063008
done
printf '%s\n' "$run_root" > /tmp/dino-row41-run-root.txt
printf 'SLOT __MEASURED_ROW41_CAPTURE_A_SHA256__ VALUE=%s\n' "$(sha256sum "$run_root/capture_a/capture.json" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW41_CAPTURE_B_SHA256__ VALUE=%s\n' "$(sha256sum "$run_root/capture_b/capture.json" | awk '{print $1}')"
```

## Block 3 — exact duplicate bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
BRACKET_SCRIPT_SHA=__MEASURED_ROW41_BRACKET_SCRIPT_SHA256__
CAPTURE_A_SHA=__MEASURED_ROW41_CAPTURE_A_SHA256__
CAPTURE_B_SHA=__MEASURED_ROW41_CAPTURE_B_SHA256__
for value in "$BRACKET_SCRIPT_SHA" "$CAPTURE_A_SHA" "$CAPTURE_B_SHA"; do case "$value" in __*) exit 2;; esac; done
run_root=$(cat /tmp/dino-row41-run-root.txt)
test "$(sha256sum "$run_root/capture_a/capture.json" | awk '{print $1}')" = "$CAPTURE_A_SHA"
test "$(sha256sum "$run_root/capture_b/capture.json" | awk '{print $1}')" = "$CAPTURE_B_SHA"
bracket="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_bracket.py"
test "$(sha256sum "$bracket" | awk '{print $1}')" = "$BRACKET_SCRIPT_SHA"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX="$run_root/pycache"
/home/dbalwada/legoESM/.venv/bin/python "$bracket" \
  --capture-a "$run_root/capture_a" --capture-b "$run_root/capture_b" \
  --output "$run_root/row41_bracket.json"
printf 'SLOT __MEASURED_ROW41_BRACKET_SHA256__ VALUE=%s\n' "$(sha256sum "$run_root/row41_bracket.json" | awk '{print $1}')"
```

## Block 4 — registered row-5 score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW41_PRODUCER__
SCORER_SHA=__MEASURED_ROW41_SCORER_SHA256__
BRACKET_SHA=__MEASURED_ROW41_BRACKET_SHA256__
for value in "$PRODUCER" "$SCORER_SHA" "$BRACKET_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 205d31a61efa5484146534945970033a319e5354
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$PRODUCER" HEAD -- packages/core packages/ocean \
  docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round41.md \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_capture.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41_bracket.py)"
run_root=$(cat /tmp/dino-row41-run-root.txt)
test "$(sha256sum "$run_root/row41_bracket.json" | awk '{print $1}')" = "$BRACKET_SHA"
scorer="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round41.py"
test "$(sha256sum "$scorer" | awk '{print $1}')" = "$SCORER_SHA"
stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
output=/tmp/dino_split_explicit_momentum_chain_round41.json
test ! -e "$output"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX="$run_root/pycache"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$scorer" \
  --capture "$run_root/capture_a" \
  --bracket "$run_root/row41_bracket.json" --bracket-sha "$BRACKET_SHA" \
  --run-stepdump "$stepdump" \
  --round40 /tmp/dino_split_explicit_momentum_chain_round40.json \
  --nemo-root "$nemo" --output "$output"
sha256sum "$output"
```

On `ROW5_WZV_CALL2_AT_BAR`, row 6 is released. On
`ROW5_WZV_CALL2_DIVERGED`, the scorer's production-vs-call2 residual is the
registered operand peel target and row 6 remains blocked. `INVALID` supplies
no science result.
