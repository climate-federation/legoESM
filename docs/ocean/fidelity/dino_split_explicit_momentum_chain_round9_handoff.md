# DINO split-explicit chain: row-1.3 QCO operand SLOT handoff

Date: 2026-08-29. This package is **DESIGNED, NOT RUN**. It adds no measured
row and authorizes no physics change.

Host execution receipt, 2026-08-29: the cascade completed through scoring,
but required three shell-only corrections now folded into every block below:
`grep` replaces the unavailable `rg`; each block enters its own repository
directory and all cross-directory inputs use absolute paths (the former
relative patch redirect failed after the OFF build changed directory); and
`CODEX_SESSION_ID` is explicitly exported before the scorer child process.

Correction, 2026-08-29: the original build block in commit `5328e23f489`
is retired. It named the 55 GiB oracle root as a `cp -a` donor and did not
explicitly bind the cumulative writer source stack. Do not run that block.
The replacement below uses the capped row-19 deterministic-writer donor,
excludes all run/build trees, and verifies its exact 28-file cumulative source
stack before applying the new QCO writer. The existing
`RUN_ZDF18_OPERANDS_ON.tnC9wz` directory supplies only the scrubbed day-180
run inputs; fresh OFF and ON arms are both built and run from the same donor.

The writer snapshots the active DINO `key_qco` continuity operands in source
order at `cfgs/DINO/MY_SRC/dynspg_ts.F90:651-725`. The scorer applies the
preregistered BASE → Q-depth → F-flux-product → B-divergence/boundary ladder.
The exact bars and disposition rules are frozen in
`PREREG_split_explicit_momentum_chain_round9.md`.

## Required source stack and mandatory copy guard

The build order is fixed:

1. Lean donor `/tmp/nemo-row19-detwriter.kTFp14`, capped at 3,072 MiB before
   copying. Existing `RUN_*`, `BLD`, `WORK`, and `.git` are never copied.
2. The donor's already-materialized cumulative instrumentation: the DINO
   seam/operand writers through `nemo_row18_gdepw_htau.patch`, then
   `nemo_deterministic_dump_buffers.patch`, then the row-19 raw-MXL arm removed
   after `nemo_row19_raw_mxl.patch` by
   `nemo_row19_raw_mxl_remove.patch` to leave the certified 197-stream OFF
   source. All 28 resulting `MY_SRC` files—including `dino_dump_zero.F90`—are
   bound by `nemo_spg_qco_base_source_manifest.sha256`; these historical
   patches are named for provenance and are not applied a second time.
3. `nemo_spg_qco_continuity_operands.patch`, the sole new patch, applied last
   with `--fuzz=0`.

`safe_copy_nemo_source.sh` is the standard guard for this and future held
SLOT build blocks. It refuses a non-`/tmp/nemo-*` donor, refuses a donor over
3,072 MiB or less than 4,096 MiB free space, excludes `RUN_*`/`BLD`/`WORK`
and `.git`, and refuses a copied source tree over 512 MiB. Every invocation
below supplies those limits explicitly, so ambient variables cannot relax
them. The three red controls prove path, size, and free-space rejection before
the real copy, with an empty destination after every rejection.

## SHA slot map

Copy only the token after `VALUE=` from the unique producer line.

| placeholder | producer |
|---|---|
| `__MEASURED_SPG_QCO_OFF_BINARY_SHA256__` | build block |
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
repo=/tmp/codex-zdf-sweep
base=/tmp/nemo-row19-detwriter.kTFp14
oracle_root=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
stack_manifest=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_base_source_manifest.sha256
copy_guard=$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh
patch_file=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_continuity_operands.patch
row18_patch=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row18_gdepw_htau.patch
writer_patch=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_deterministic_dump_buffers.patch
row19_patch=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row19_raw_mxl.patch
row19_remove_patch=$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row19_raw_mxl_remove.patch
test "$(sha256sum "$stack_manifest" | awk '{print $1}')" = 22fdaa20eeb3ff1c04ea22fa0dc293d3db9b9e6bf8cf9638075fdfb819b97e57
test "$(sha256sum "$copy_guard" | awk '{print $1}')" = f1bfbc7e5428c2f532a26aa8197b368847c40421dae6704b33fb91afeb98e071
test "$(sha256sum "$patch_file" | awk '{print $1}')" = aa3a1c99b8a32d431f027137f10fe4a02cd458929a9c8be5ef55e6636f5bed83
test "$(sha256sum "$row18_patch" | awk '{print $1}')" = e797e5d9ba9cbc50401a7542c53182f41d7cd2ab3b148b914ccb01c15cb28fb7
test "$(sha256sum "$writer_patch" | awk '{print $1}')" = 4f3674462d629d6174854545400d35669d43857ac170e82e64e4b7406daf3292
test "$(sha256sum "$row19_patch" | awk '{print $1}')" = 3379bdeb9e6bef1729ff0ff26592abc8a55d5534589f19ddb491c37b4b56fa21
test "$(sha256sum "$row19_remove_patch" | awk '{print $1}')" = 2a913a15ed64b2e4e5d52fe4837115cd2e57f8f921f94415f8b449776c70e73d
test "$(git -C "$base" rev-parse HEAD)" = dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
test "$(git -C "$base" describe --tags --exact-match HEAD)" = 5.0.2
test -z "$(git -C "$base" status --porcelain --untracked-files=no)"
test "$(sha256sum "$base/cfgs/DINO/cpp_DINO.fcm" | awk '{print $1}')" = f72d84ceee6ec63a2bdc65354ba13fbde58b6b5611962d482d49b49b12a963dc
test "$(sha256sum "$base/arch/arch-conda.fcm" | awk '{print $1}')" = 64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/dynspg_ts.F90" | awk '{print $1}')" = 58566572b899843a768e8d6cf30cc69ae8a05830927e6ddbe919f443610b7b2d
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/dino_dump_zero.F90" | awk '{print $1}')" = ddb1abfaf4ebb3b95791f262460ce7652aee868a40b9cbd05d01c874b3128780
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/zdftke.F90" | awk '{print $1}')" = 6a33079678505c105cae1edb8a52b9d70d02012d0eb3395673f4b62fe657a9fd
test "$(find "$base/cfgs/DINO/MY_SRC" -maxdepth 1 -type f | wc -l)" -eq 28
( cd "$base" && sha256sum -c "$stack_manifest" )
test "$(grep -c 'key_qco' "$base/cfgs/DINO/cpp_DINO.fcm")" -ge 1
test "$(grep -c 'key_qcoTest_FluxForm' "$base/cfgs/DINO/cpp_DINO.fcm")" -eq 0
test "$(grep -rE --include='*.F90' 'CALL dino_dump_2d' "$base/cfgs/DINO/MY_SRC" | wc -l)" -eq 150
test "$(grep -rE --include='*.F90' 'CALL dino_dump_3d' "$base/cfgs/DINO/MY_SRC" | wc -l)" -eq 5
for unit in 8980 8981 8982 8983 8984 8985; do
  test "$(grep -rE --include='*.F90' "OPEN\( UNIT=${unit}" "$base/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
done

# Mandatory no-copy controls: each rejected invocation must leave its fresh
# destination empty. Exact limits are supplied on every invocation.
reject_path=$(mktemp -d /tmp/nemo-spg-qco-reject-path.XXXXXX)
if NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
    "$copy_guard" "$oracle_root" "$reject_path"; then
  echo 'path red control unexpectedly copied the oracle root' >&2; exit 2
fi
test -z "$(find "$reject_path" -mindepth 1 -print -quit)"
rmdir "$reject_path"

reject_size=$(mktemp -d /tmp/nemo-spg-qco-reject-size.XXXXXX)
if NEMO_SOURCE_BASE_MAX_MIB=1 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
    "$copy_guard" "$base" "$reject_size"; then
  echo 'size red control unexpectedly copied an over-cap donor' >&2; exit 2
fi
test -z "$(find "$reject_size" -mindepth 1 -print -quit)"
rmdir "$reject_size"

reject_free=$(mktemp -d /tmp/nemo-spg-qco-reject-free.XXXXXX)
filesystem_total_mib=$(df -Pm -- "$reject_free" | awk 'NR==2 {print $2}')
if NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 \
    NEMO_SOURCE_MIN_FREE_MIB=$((filesystem_total_mib + 1)) \
    "$copy_guard" "$base" "$reject_free"; then
  echo 'free-space red control unexpectedly copied below its floor' >&2; exit 2
fi
test -z "$(find "$reject_free" -mindepth 1 -print -quit)"
rmdir "$reject_free"

nemo_src=$(mktemp -d /tmp/nemo-spg-qco-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 \
  NEMO_SOURCE_MIN_FREE_MIB=4096 "$copy_guard" "$base" "$nemo_src"
( cd "$nemo_src" && sha256sum -c "$stack_manifest" )
test "$(find "$nemo_src/cfgs/DINO/MY_SRC" -maxdepth 1 -type f | wc -l)" -eq 28

# Build the manifest-exact deterministic OFF arm before adding the new writer.
cd "$nemo_src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-spg-qco-off-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-spg-qco-off.exe
sha256sum /tmp/nemo-spg-qco-off.exe > /tmp/nemo-spg-qco-off-build.sha256
printf 'SLOT __MEASURED_SPG_QCO_OFF_BINARY_SHA256__ SOURCE=/tmp/nemo-spg-qco-off-build.sha256:first-field VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-qco-off-build.sha256)"

# The QCO operand writer is the only difference in the paired ON arm.
patch --fuzz=0 -p1 -d "$nemo_src" < "$patch_file"
test "$(sha256sum "$nemo_src/cfgs/DINO/MY_SRC/dynspg_ts.F90" | awk '{print $1}')" = 25c780e0b73b32ad4c7716f45009e868ac9dafbeeccee09cd07fbe31cddf94bb
for unit in 8980 8981 8982 8983 8984 8985; do
  test "$(grep -rE --include='*.F90' "OPEN\( UNIT=${unit}" "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
test "$(grep -c "qco continuity operands (substep jn=1) written" "$nemo_src/cfgs/DINO/MY_SRC/dynspg_ts.F90")" -eq 1
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-spg-qco-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-spg-qco-on.exe
sha256sum /tmp/nemo-spg-qco-on.exe > /tmp/nemo-spg-qco-build.sha256
sha256sum cfgs/DINO/MY_SRC/dynspg_ts.F90 > /tmp/nemo-spg-qco-dynspg-source.sha256
printf 'SLOT __MEASURED_SPG_QCO_BINARY_SHA256__ SOURCE=/tmp/nemo-spg-qco-build.sha256:first-field VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-qco-build.sha256)"
printf 'SLOT __MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__ SOURCE=/tmp/nemo-spg-qco-dynspg-source.sha256:first-field VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-qco-dynspg-source.sha256)"
```

## Held one-step sequential run block

This invokes the sequential executable directly. It does not use `mpirun`.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-spg-qco-run-xdg OMP_NUM_THREADS=1
cd /tmp/codex-zdf-sweep
ON_BIN_SHA=__MEASURED_SPG_QCO_BINARY_SHA256__
OFF_BIN_SHA=__MEASURED_SPG_QCO_OFF_BINARY_SHA256__
SRC_SHA=__MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__
for value in "$ON_BIN_SHA" "$OFF_BIN_SHA" "$SRC_SHA"; do
  case "$value" in __MEASURED_*) echo 'replace all three measured build SHA slots' >&2; exit 2;; esac
done
on_bin=/tmp/nemo-spg-qco-on.exe
off_bin=/tmp/nemo-spg-qco-off.exe
test "$(sha256sum "$on_bin" | awk '{print $1}')" = "$ON_BIN_SHA"
test "$(sha256sum "$off_bin" | awk '{print $1}')" = "$OFF_BIN_SHA"
test "$(awk '{print $1}' /tmp/nemo-spg-qco-dynspg-source.sha256)" = "$SRC_SHA"
template=/tmp/RUN_ZDF18_OPERANDS_ON.tnC9wz
test "$(find "$template" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 197
test "$(sha256sum "$template/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
test "$(sha256sum "$template/DINO_00005761_restart.nc" | awk '{print $1}')" = 33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c
ON=$(mktemp -d /tmp/RUN_SPG_QCO_ROW13_ON.XXXXXX)
OFF=$(mktemp -d /tmp/RUN_SPG_QCO_ROW13_OFF.XXXXXX)
for arm in OFF ON; do
  if test "$arm" = OFF; then
    arm_dir=$OFF; arm_bin=$off_bin; expected_count=197
  else
    arm_dir=$ON; arm_bin=$on_bin; expected_count=203
  fi
  cp -a "$template"/. "$arm_dir"/
  find "$arm_dir" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
  rm -f "$arm_dir/nemo"
  ln -s "$arm_bin" "$arm_dir/nemo"
  sha256sum "$arm_bin" > "$arm_dir/.nemo_binary_sha256"
  ( cd "$arm_dir" && ./nemo > run.attempt1.log 2>&1 )
  grep -q '^STOP 0$' "$arm_dir/run.attempt1.log"
  test "$(sha256sum "$arm_dir/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
  test "$(sha256sum "$arm_dir/DINO_00005761_restart.nc" | awk '{print $1}')" = 33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c
  test "$(find "$arm_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq "$expected_count"
done
cp -p /tmp/nemo-spg-qco-dynspg-source.sha256 "$ON/.dynspg_source_sha256"
for name in zsshp2 zhup2 zhvp2 zhU zhV zhdiv; do
  test "$(stat -c %s "$ON/qco_dump_${name}_substep1.bin")" -eq 90944
done
printf '%s\n' "$ON" > /tmp/spg-qco-row13-on-dir.txt
printf '%s\n' "$OFF" > /tmp/spg-qco-row13-off-dir.txt
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

## Held 197/197 bracket block

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
repo=/tmp/codex-zdf-sweep
ON=$(cat /tmp/spg-qco-row13-on-dir.txt)
OFF=$(cat /tmp/spg-qco-row13-off-dir.txt)
PYTHONPATH=$repo/scripts/validate/ocean_fidelity/dino_1226 /home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import json, sys
from zdf_stream_bracket import files_byte_identical, manifest_sha256, one_bit_file_control, sha256, stream_manifest
on, off = map(Path, sys.argv[1:])
expected={f"qco_dump_{n}_substep1.bin" for n in ("zsshp2","zhup2","zhvp2","zhU","zhV","zhdiv")}
om,fm=stream_manifest(on),stream_manifest(off)
def exact_new_stream_set(on_manifest, off_manifest):
    return (len(on_manifest)==203 and len(off_manifest)==197
            and set(on_manifest)-set(off_manifest)==expected
            and set(off_manifest).issubset(on_manifest))
assert exact_new_stream_set(om,fm)
assert all(files_byte_identical(on/n,off/n) for n in fm)
assert all(om[n]["size_bytes"]==90944 for n in expected)
one=one_bit_file_control(on/sorted(fm)[0])
planted=dict(om); planted.pop(sorted(expected)[0])
missing=not exact_new_stream_set(planted,fm)
assert one and missing
r={"schema":"dino-spg-qco-row13-bracket-v1","on_dir":str(on.resolve()),"off_dir":str(off.resolve()),"shared_count":197,"shared_exact":True,"shared_manifest_sha256":manifest_sha256({n:om[n] for n in sorted(fm)}),"new_streams":sorted(expected),"controls":{"one_bit_file":one,"missing_stream":missing}}
p=Path('/tmp/dino_spg_qco_row13_bracket_receipt.json')
p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
value=sha256(p)
print(f'SLOT __MEASURED_SHA256_SPG_QCO_BRACKET_RECEIPT__ SOURCE=sha256sum:{p}:first-field VALUE={value}')
PY
```

## Held CPU/fp64 score block

The clean local clone is deliberate: the scorer refuses editable-install
imports from another worktree and refuses a dirty measured tree. The fresh
deterministic files intentionally have different full-file hashes from the
legacy round-8 streams because their halos are zeroed. The scorer therefore
binds each legacy file to the round-8 SHA receipt, then requires old versus
fresh bits to be exact on the registered wet T/U/V population; a one-ULP wet
cell plant must fail for every translated field.

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/spg-qco-row13-on-dir.txt)
OFF=$(cat /tmp/spg-qco-row13-off-dir.txt)
BIN_SHA=__MEASURED_SPG_QCO_BINARY_SHA256__
OFF_BIN_SHA=__MEASURED_SPG_QCO_OFF_BINARY_SHA256__
SRC_SHA=__MEASURED_SPG_QCO_DYNSPG_SOURCE_SHA256__
ZSSHP_SHA=__MEASURED_SHA256_QCO_DUMP_ZSSHP2__
ZHUP_SHA=__MEASURED_SHA256_QCO_DUMP_ZHUP2__
ZHVP_SHA=__MEASURED_SHA256_QCO_DUMP_ZHVP2__
ZHU_SHA=__MEASURED_SHA256_QCO_DUMP_ZHU__
ZHV_SHA=__MEASURED_SHA256_QCO_DUMP_ZHV__
ZHDIV_SHA=__MEASURED_SHA256_QCO_DUMP_ZHDIV__
BRACKET_SHA=__MEASURED_SHA256_SPG_QCO_BRACKET_RECEIPT__
for value in "$BIN_SHA" "$OFF_BIN_SHA" "$SRC_SHA" "$ZSSHP_SHA" "$ZHUP_SHA" "$ZHVP_SHA" "$ZHU_SHA" "$ZHV_SHA" "$ZHDIV_SHA" "$BRACKET_SHA"; do
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
test "$(sha256sum "$MEASURED/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round9.py" | awk '{print $1}')" = dfd071b5cb9a3f5f9ccca2d745785f3b8a9a9dfd2a03afceab3425617741a1c4
test "$(sha256sum "$MEASURED/docs/ocean/fidelity/dino_split_explicit_momentum_chain_round8_artifact.json" | awk '{print $1}')" = 16883e8e140f3e25de24866d9deedf9dd6b18189ca4f280c1448144620b48a92
export PYTHONPATH="$MEASURED/src:$MEASURED/packages/core:$MEASURED/packages/ocean:$MEASURED/scripts/validate/ocean_fidelity/dino_1226"
: "${CODEX_SESSION_ID:?set CODEX_SESSION_ID to the active session id}"
export CODEX_SESSION_ID
cd "$MEASURED"
DINO_1226_LANE=d180 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
/home/dbalwada/legoESM/.venv/bin/python "$MEASURED/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round9.py" \
  --run "$ON" --off-run "$OFF" \
  --binary-sha256 "$BIN_SHA" --off-binary-sha256 "$OFF_BIN_SHA" \
  --dynspg-source-sha256 "$SRC_SHA" \
  --dump-sha "qco_dump_zsshp2_substep1.bin=$ZSSHP_SHA" \
  --dump-sha "qco_dump_zhup2_substep1.bin=$ZHUP_SHA" \
  --dump-sha "qco_dump_zhvp2_substep1.bin=$ZHVP_SHA" \
  --dump-sha "qco_dump_zhU_substep1.bin=$ZHU_SHA" \
  --dump-sha "qco_dump_zhV_substep1.bin=$ZHV_SHA" \
  --dump-sha "qco_dump_zhdiv_substep1.bin=$ZHDIV_SHA" \
  --bracket-receipt /tmp/dino_spg_qco_row13_bracket_receipt.json \
  --bracket-sha256 "$BRACKET_SHA" \
  --round8-artifact "$MEASURED/docs/ocean/fidelity/dino_split_explicit_momentum_chain_round8_artifact.json" \
  --round8-sha256 16883e8e140f3e25de24866d9deedf9dd6b18189ca4f280c1448144620b48a92 \
  --package-commit "$PACKAGE_COMMIT" \
  --output /tmp/dino_split_explicit_momentum_chain_round9_artifact.json
artifact_sha=$(sha256sum /tmp/dino_split_explicit_momentum_chain_round9_artifact.json | awk '{print $1}')
printf 'SLOT __MEASURED_SHA256_SPG_QCO_ROUND9_ARTIFACT__ SOURCE=sha256sum:/tmp/dino_split_explicit_momentum_chain_round9_artifact.json:first-field VALUE=%s\n' "$artifact_sha"
```

The blocks intentionally contain no push, GPU command, or `mpirun`. They run
paired OFF and ON arms for one sequential day-180 step from the same scrubbed
input template; the new writer samples the ON step's first barotropic substep.
