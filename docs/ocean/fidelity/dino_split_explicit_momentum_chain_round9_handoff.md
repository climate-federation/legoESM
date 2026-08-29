# DINO split-explicit chain: row-1.3 QCO operand SLOT handoff

Date: 2026-08-29. This package is **DESIGNED, NOT RUN**. It adds no measured
row and authorizes no physics change.

The writer snapshots the active DINO `key_qco` continuity operands in source
order at `cfgs/DINO/MY_SRC/dynspg_ts.F90:651-725`. The scorer applies the
preregistered BASE → Q-depth → F-flux-product → B-divergence/boundary ladder.
The exact bars and disposition rules are frozen in
`PREREG_split_explicit_momentum_chain_round9.md`.

## SHA slot map

Copy only the token after `VALUE=` from the unique producer line.

| placeholder | producer |
|---|---|
| `__MEASURED_SPG_QCO_BINARY_SHA256__` | build block |
| `__MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__` | build block |
| `__MEASURED_SHA256_QCO_DUMP_ZSSHP2__` | run block |
| `__MEASURED_SHA256_QCO_DUMP_ZHUP2__` | run block |
| `__MEASURED_SHA256_QCO_DUMP_ZHVP2__` | run block |
| `__MEASURED_SHA256_QCO_DUMP_ZHU__` | run block |
| `__MEASURED_SHA256_QCO_DUMP_ZHV__` | run block |
| `__MEASURED_SHA256_QCO_DUMP_ZHDIV__` | run block |
| `__MEASURED_SHA256_SPG_QCO_BRACKET_RECEIPT__` | bracket block |

## Held build block

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-spg-qco-build-xdg
cd /tmp/codex-zdf-sweep
base=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
patch_file=scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_continuity_operands.patch
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/dynspg_ts.F90" | awk '{print $1}')" = a64b32e7c9ce1c1be6c121597235b08f05c853961e83d52be6082ea7242f6275
test "$(sha256sum "$patch_file" | awk '{print $1}')" = aa3a1c99b8a32d431f027137f10fe4a02cd458929a9c8be5ef55e6636f5bed83
test "$(grep -c 'key_qco' "$base/cfgs/DINO/cpp_DINO.fcm")" -ge 1
test "$(grep -c 'key_qcoTest_FluxForm' "$base/cfgs/DINO/cpp_DINO.fcm")" -eq 0
for unit in 8980 8981 8982 8983 8984 8985; do
  test "$(grep -rE --include='*.F90' "OPEN\( UNIT=${unit}" "$base/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
done
nemo_src=$(mktemp -d /tmp/nemo-spg-qco-src.XXXXXX)
cp -a "$base"/. "$nemo_src"/
patch --fuzz=0 -p1 -d "$nemo_src" < "$patch_file"
for unit in 8980 8981 8982 8983 8984 8985; do
  test "$(grep -rE --include='*.F90' "OPEN\( UNIT=${unit}" "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
test "$(grep -c "qco continuity operands (substep jn=1) written" "$nemo_src/cfgs/DINO/MY_SRC/dynspg_ts.F90")" -eq 1
cd "$nemo_src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-spg-qco-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-spg-qco-on.exe
sha256sum /tmp/nemo-spg-qco-on.exe > /tmp/nemo-spg-qco-build.sha256
sha256sum cfgs/DINO/MY_SRC/dynspg_ts.F90 > /tmp/nemo-spg-qco-dynspg-source.sha256
printf 'SLOT __MEASURED_SPG_QCO_BINARY_SHA256__ SOURCE=/tmp/nemo-spg-qco-build.sha256:first-field VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-qco-build.sha256)"
printf 'SLOT __MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__ SOURCE=/tmp/nemo-spg-qco-dynspg-source.sha256:first-field VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-qco-dynspg-source.sha256)"
```

## Held four-step sequential run block

This invokes the sequential executable directly. It does not use `mpirun`.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-spg-qco-run-xdg OMP_NUM_THREADS=1
BIN_SHA=__MEASURED_SPG_QCO_BINARY_SHA256__
SRC_SHA=__MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__
for value in "$BIN_SHA" "$SRC_SHA"; do
  case "$value" in __MEASURED_*) echo 'replace both measured build SHA slots' >&2; exit 2;; esac
done
bin=/tmp/nemo-spg-qco-on.exe
test "$(sha256sum "$bin" | awk '{print $1}')" = "$BIN_SHA"
test "$(awk '{print $1}' /tmp/nemo-spg-qco-dynspg-source.sha256)" = "$SRC_SHA"
OFF=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
test "$(find "$OFF" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 319
test "$(sha256sum "$OFF/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
test "$(sha256sum "$OFF/DINO_00005764_restart.nc" | awk '{print $1}')" = f1a29b5df8a849a2efb3b3b797040f6676a3af23798ae00c37d978e204760fb0
ON=$(mktemp -d /tmp/RUN_SPG_QCO_ROW13_ON.XXXXXX)
cp -a "$OFF"/. "$ON"/
find "$ON" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005764_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
rm -f "$ON/nemo"
ln -s "$bin" "$ON/nemo"
sha256sum "$bin" > "$ON/.nemo_binary_sha256"
cp -p /tmp/nemo-spg-qco-dynspg-source.sha256 "$ON/.dynspg_source_sha256"
( cd "$ON" && ./nemo > run.attempt1.log 2>&1 )
grep -q '^STOP 0$' "$ON/run.attempt1.log"
test "$(sha256sum "$ON/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
test "$(sha256sum "$ON/DINO_00005764_restart.nc" | awk '{print $1}')" = f1a29b5df8a849a2efb3b3b797040f6676a3af23798ae00c37d978e204760fb0
test "$(find "$ON" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 325
for name in zsshp2 zhup2 zhvp2 zhU zhV zhdiv; do
  test "$(stat -c %s "$ON/qco_dump_${name}_substep1.bin")" -eq 90944
done
printf '%s\n' "$ON" > /tmp/spg-qco-row13-on-dir.txt
for item in \
  __MEASURED_SHA256_QCO_DUMP_ZSSHP2__:qco_dump_zsshp2_substep1.bin \
  __MEASURED_SHA256_QCO_DUMP_ZHUP2__:qco_dump_zhup2_substep1.bin \
  __MEASURED_SHA256_QCO_DUMP_ZHVP2__:qco_dump_zhvp2_substep1.bin \
  __MEASURED_SHA256_QCO_DUMP_ZHU__:qco_dump_zhU_substep1.bin \
  __MEASURED_SHA256_QCO_DUMP_ZHV__:qco_dump_zhV_substep1.bin \
  __MEASURED_SHA256_QCO_DUMP_ZHDIV__:qco_dump_zhdiv_substep1.bin; do
  slot=${item%%:*}; name=${item#*:}; value=$(sha256sum "$ON/$name" | awk '{print $1}')
  printf 'SLOT %s SOURCE=sha256sum:%s:first-field VALUE=%s\n' "$slot" "$ON/$name" "$value"
done
```

## Held 319/319 bracket block

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/spg-qco-row13-on-dir.txt)
OFF=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
PYTHONPATH=scripts/validate/ocean_fidelity/dino_1226 /home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import json, sys
from zdf_stream_bracket import files_byte_identical, manifest_sha256, one_bit_file_control, sha256, stream_manifest
on, off = map(Path, sys.argv[1:])
expected={f"qco_dump_{n}_substep1.bin" for n in ("zsshp2","zhup2","zhvp2","zhU","zhV","zhdiv")}
om,fm=stream_manifest(on),stream_manifest(off)
def exact_new_stream_set(on_manifest, off_manifest):
    return (len(on_manifest)==325 and len(off_manifest)==319
            and set(on_manifest)-set(off_manifest)==expected
            and set(off_manifest).issubset(on_manifest))
assert exact_new_stream_set(om,fm)
assert all(files_byte_identical(on/n,off/n) for n in fm)
assert all(om[n]["size_bytes"]==90944 for n in expected)
one=one_bit_file_control(on/sorted(fm)[0])
planted=dict(om); planted.pop(sorted(expected)[0])
missing=not exact_new_stream_set(planted,fm)
assert one and missing
r={"schema":"dino-spg-qco-row13-bracket-v1","on_dir":str(on.resolve()),"off_dir":str(off.resolve()),"shared_count":319,"shared_exact":True,"shared_manifest_sha256":manifest_sha256({n:om[n] for n in sorted(fm)}),"new_streams":sorted(expected),"controls":{"one_bit_file":one,"missing_stream":missing}}
p=Path('/tmp/dino_spg_qco_row13_bracket_receipt.json')
p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
value=sha256(p)
print(f'SLOT __MEASURED_SHA256_SPG_QCO_BRACKET_RECEIPT__ SOURCE=sha256sum:{p}:first-field VALUE={value}')
PY
```

## Held CPU/fp64 score block

The clean local clone is deliberate: the scorer refuses editable-install
imports from another worktree and refuses a dirty measured tree.

```bash
set -euo pipefail
ON=$(cat /tmp/spg-qco-row13-on-dir.txt)
BIN_SHA=__MEASURED_SPG_QCO_BINARY_SHA256__
SRC_SHA=__MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__
ZSSHP_SHA=__MEASURED_SHA256_QCO_DUMP_ZSSHP2__
ZHUP_SHA=__MEASURED_SHA256_QCO_DUMP_ZHUP2__
ZHVP_SHA=__MEASURED_SHA256_QCO_DUMP_ZHVP2__
ZHU_SHA=__MEASURED_SHA256_QCO_DUMP_ZHU__
ZHV_SHA=__MEASURED_SHA256_QCO_DUMP_ZHV__
ZHDIV_SHA=__MEASURED_SHA256_QCO_DUMP_ZHDIV__
BRACKET_SHA=__MEASURED_SHA256_SPG_QCO_BRACKET_RECEIPT__
for value in "$BIN_SHA" "$SRC_SHA" "$ZSSHP_SHA" "$ZHUP_SHA" "$ZHVP_SHA" "$ZHU_SHA" "$ZHV_SHA" "$ZHDIV_SHA" "$BRACKET_SHA"; do
  case "$value" in __MEASURED_*) echo 'replace every measured SHA slot from its mapped SLOT line' >&2; exit 2;; esac
done
SHADOW_GIT=/tmp/zdf-sweep-git.cJQ6wi/repo.git
PACKAGE_COMMIT=$(git --git-dir="$SHADOW_GIT" rev-parse refs/heads/fidelity/dino-zdf-sweep-codex)
printf 'SLOT __MEASURED_SPG_QCO_PACKAGE_COMMIT__ SOURCE=git:%s:refs/heads/fidelity/dino-zdf-sweep-codex VALUE=%s\n' "$SHADOW_GIT" "$PACKAGE_COMMIT"
MEASURED=$(mktemp -d /tmp/codex-spg-qco-score.XXXXXX)
git clone --no-hardlinks "$SHADOW_GIT" "$MEASURED"
git -C "$MEASURED" checkout --detach "$PACKAGE_COMMIT"
test "$(git -C "$MEASURED" rev-parse HEAD)" = "$PACKAGE_COMMIT"
test "$(git -C "$MEASURED" status --porcelain | wc -l)" -eq 0
test "$(sha256sum "$MEASURED/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round9.py" | awk '{print $1}')" = 2198b1ae958d67598958093ba1fa9aa3e1f7a212d52a1900b42a2acaecfca564
test "$(sha256sum "$MEASURED/docs/ocean/fidelity/dino_split_explicit_momentum_chain_round8_artifact.json" | awk '{print $1}')" = 16883e8e140f3e25de24866d9deedf9dd6b18189ca4f280c1448144620b48a92
export PYTHONPATH="$MEASURED/src:$MEASURED/packages/core:$MEASURED/packages/ocean:$MEASURED/scripts/validate/ocean_fidelity/dino_1226"
: "${CODEX_SESSION_ID:?set CODEX_SESSION_ID to the active session id}"
cd "$MEASURED"
DINO_1226_LANE=d180 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
/home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round9.py \
  --run "$ON" --binary-sha256 "$BIN_SHA" --dynspg-source-sha256 "$SRC_SHA" \
  --dump-sha "qco_dump_zsshp2_substep1.bin=$ZSSHP_SHA" \
  --dump-sha "qco_dump_zhup2_substep1.bin=$ZHUP_SHA" \
  --dump-sha "qco_dump_zhvp2_substep1.bin=$ZHVP_SHA" \
  --dump-sha "qco_dump_zhU_substep1.bin=$ZHU_SHA" \
  --dump-sha "qco_dump_zhV_substep1.bin=$ZHV_SHA" \
  --dump-sha "qco_dump_zhdiv_substep1.bin=$ZHDIV_SHA" \
  --bracket-receipt /tmp/dino_spg_qco_row13_bracket_receipt.json \
  --bracket-sha256 "$BRACKET_SHA" \
  --round8-artifact docs/ocean/fidelity/dino_split_explicit_momentum_chain_round8_artifact.json \
  --round8-sha256 16883e8e140f3e25de24866d9deedf9dd6b18189ca4f280c1448144620b48a92 \
  --package-commit "$PACKAGE_COMMIT" \
  --output /tmp/dino_split_explicit_momentum_chain_round9_artifact.json
artifact_sha=$(sha256sum /tmp/dino_split_explicit_momentum_chain_round9_artifact.json | awk '{print $1}')
printf 'SLOT __MEASURED_SHA256_SPG_QCO_ROUND9_ARTIFACT__ SOURCE=sha256sum:/tmp/dino_split_explicit_momentum_chain_round9_artifact.json:first-field VALUE=%s\n' "$artifact_sha"
```

The blocks intentionally contain no push, GPU command, or `mpirun`. The run
is four sequential day-180 steps because the frozen OFF stack is the existing
four-step deterministic-writer run; the writer itself samples only the first
step and first barotropic substep.
