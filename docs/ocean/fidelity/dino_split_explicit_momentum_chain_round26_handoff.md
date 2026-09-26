# Row-1.3 bottom-stress/vector-update SLOT handoff

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. This cascade is **DESIGNED, NOT
RUN**. It stops the ordered walk at the first missing executed operand.

The guarded lean donor contains the certified 197-stream deterministic-writer
stack. Both arms receive `nemo_spg_qco_continuity_operands.patch` and
`nemo_spg_row13_een_coefficients.patch`, producing the same 211-stream OFF
baseline used by rounds 22--25. The ON arm alone receives
`nemo_spg_row13_bottom_update_operands.patch`, adding twelve streams. The
bracket is therefore 211/211 exact with twelve named additions. The physics
producer is SHA-pinned to `b15557ae07cd7e858db3b1c73bf6cec6d690e99e`;
later packaging commits are admitted only by a zero model diff.

Every block enters its own absolute directory. Every Python invocation uses
checkout-first `PYTHONPATH`. The standard guarded lean-copy checks reject the
55-GiB oracle/archive tree, oversize donors, and insufficient free space while
excluding `RUN_*`, `BLD`, `WORK`, and `.git`. No block pushes, invokes a GPU,
or uses `mpirun`.

## Block 1 — guarded cumulative paired build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=b15557ae07cd7e858db3b1c73bf6cec6d690e99e
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"
base=/tmp/nemo-row19-detwriter.kTFp14
guard="$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh"
manifest="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_base_source_manifest.sha256"
qco_patch="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_continuity_operands.patch"
cor_patch="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_row13_een_coefficients.patch"
row26_patch="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_row13_bottom_update_operands.patch"
test "$(du -sm "$base" | awk '{print $1}')" -le 3072
test "$(sha256sum "$manifest" | awk '{print $1}')" = 22fdaa20eeb3ff1c04ea22fa0dc293d3db9b9e6bf8cf9638075fdfb819b97e57
test "$(sha256sum "$guard" | awk '{print $1}')" = f1bfbc7e5428c2f532a26aa8197b368847c40421dae6704b33fb91afeb98e071
test "$(sha256sum "$qco_patch" | awk '{print $1}')" = aa3a1c99b8a32d431f027137f10fe4a02cd458929a9c8be5ef55e6636f5bed83
test "$(sha256sum "$cor_patch" | awk '{print $1}')" = 5107a4e83e61499a1f5c54d197d99541cb85ddeb42b5e34d252de115ce4218a5
test "$(sha256sum "$row26_patch" | awk '{print $1}')" = 3df88df75a243bdb2bc2e5ceb5bef6bff771b9ef00989c30067d77c832e85f8c
test "$(find "$base/cfgs/DINO/MY_SRC" -maxdepth 1 -type f | wc -l)" -eq 28
( cd "$base" && sha256sum -c "$manifest" )

# Standard no-copy controls: wrong path, oversize donor, impossible free-space floor.
for mode in path size free; do
  reject=$(mktemp -d "/tmp/nemo-row26-reject-${mode}.XXXXXX")
  if test "$mode" = path; then donor=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2; cap=3072; floor=4096
  elif test "$mode" = size; then donor=$base; cap=1; floor=4096
  else donor=$base; cap=3072; floor=$(( $(df -Pm "$reject" | awk 'NR==2 {print $2}') + 1 )); fi
  if NEMO_SOURCE_BASE_MAX_MIB=$cap NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=$floor \
      "$guard" "$donor" "$reject"; then
    echo "copy guard red control $mode unexpectedly passed" >&2; exit 2
  fi
  test -z "$(find "$reject" -mindepth 1 -print -quit)"
  rmdir "$reject"
done

src=$(mktemp -d /tmp/nemo-row26-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
  "$guard" "$base" "$src"
( cd "$src" && sha256sum -c "$manifest" )
patch --fuzz=0 -p1 -d "$src" < "$qco_patch"
patch --fuzz=0 -p1 -d "$src" < "$cor_patch"
claimed_units='9420 9421 9422 9423 9424 9425 9426 9427 9428 9429 9430 9431'
for unit in $claimed_units; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
  test "$(grep -rE --include='*.F90' "\\b${unit}\\b" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
done
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row26-xdg
cd "$src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row26-off-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-row26-off.exe
patch --fuzz=0 -p1 -d "$src" < "$row26_patch"
for unit in $claimed_units; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row26-on-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-row26-on.exe
sha256sum "$src/cfgs/DINO/MY_SRC/dynspg_ts.F90" > /tmp/nemo-row26-source.sha256
printf '%s\n' "$src" > /tmp/nemo-row26-src.txt
printf 'SLOT __MEASURED_ROW26_OFF_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-row26-off.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW26_ON_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-row26-on.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW26_SOURCE_SHA256__ VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-row26-source.sha256)"
```

## Block 2 — fresh paired sequential runs

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF_SHA=__MEASURED_ROW26_OFF_BINARY_SHA256__
ON_SHA=__MEASURED_ROW26_ON_BINARY_SHA256__
SRC_SHA=__MEASURED_ROW26_SOURCE_SHA256__
for value in "$OFF_SHA" "$ON_SHA" "$SRC_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$(sha256sum /tmp/nemo-row26-off.exe | awk '{print $1}')" = "$OFF_SHA"
test "$(sha256sum /tmp/nemo-row26-on.exe | awk '{print $1}')" = "$ON_SHA"
test "$(awk '{print $1}' /tmp/nemo-row26-source.sha256)" = "$SRC_SHA"
template=/tmp/RUN_ZDF18_OPERANDS_ON.tnC9wz
test "$(find "$template" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 197
test "$(sha256sum "$template/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
OFF=$(mktemp -d /tmp/RUN_SPG_ROW13_BOTTOM_UPDATE_OFF.XXXXXX)
ON=$(mktemp -d /tmp/RUN_SPG_ROW13_BOTTOM_UPDATE_ON.XXXXXX)
for arm in OFF ON; do
  if test "$arm" = OFF; then run=$OFF; binary=/tmp/nemo-row26-off.exe; count=211
  else run=$ON; binary=/tmp/nemo-row26-on.exe; count=223; fi
  cp -a "$template"/. "$run"/
  find "$run" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
  rm -f "$run/nemo"
  ln -s "$binary" "$run/nemo"
  sha256sum "$binary" > "$run/.nemo_binary_sha256"
  ( cd "$run" && ./nemo > run.attempt1.log 2>&1 )
  grep -q '^STOP 0$' "$run/run.attempt1.log"
  test "$(find "$run" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq "$count"
done
cp -p /tmp/nemo-row26-source.sha256 "$ON/.dynspg_source_sha256"
names='zCdU_u zCdU_v hu_e hv_e hur_e hvr_e zu_trd_after_bottom zv_trd_after_bottom zu_spg zv_spg ua_e_after_update va_e_after_update'
for name in $names; do
  file="$ON/row13_dump_${name}_substep1.bin"
  test "$(stat -c %s "$file")" -eq 90944
  slot=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')
  printf 'SLOT __MEASURED_ROW26_SHA_%s__ VALUE=%s\n' "$slot" "$(sha256sum "$file" | awk '{print $1}')"
done
printf '%s\n' "$OFF" > /tmp/row26-off-run.txt
printf '%s\n' "$ON" > /tmp/row26-on-run.txt
```

## Block 3 — exact 211/211 bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF=$(cat /tmp/row26-off-run.txt)
ON=$(cat /tmp/row26-on-run.txt)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import files_byte_identical,manifest_sha256,one_bit_file_control,sha256,stream_manifest
on,off=map(Path,sys.argv[1:])
stems=('zCdU_u','zCdU_v','hu_e','hv_e','hur_e','hvr_e','zu_trd_after_bottom','zv_trd_after_bottom','zu_spg','zv_spg','ua_e_after_update','va_e_after_update')
expected={f'row13_dump_{name}_substep1.bin' for name in stems}
om,fm=stream_manifest(on),stream_manifest(off)
def exact(o,f): return len(o)==223 and len(f)==211 and set(o)-set(f)==expected and set(f).issubset(o)
assert exact(om,fm)
assert all(files_byte_identical(on/n,off/n) for n in fm)
assert all(om[n]['size_bytes']==90944 for n in expected)
one=one_bit_file_control(on/sorted(fm)[0])
planted=dict(om); planted.pop(sorted(expected)[0])
missing=not exact(planted,fm)
assert one and missing
receipt={'schema':'dino-spg-row13-bottom-update-bracket-v1','on_dir':str(on.resolve()),'off_dir':str(off.resolve()),'shared_count':211,'shared_exact':True,'shared_manifest_sha256':manifest_sha256({n:om[n] for n in sorted(fm)}),'new_streams':sorted(expected),'on_manifest_sha256':manifest_sha256(om),'off_manifest_sha256':manifest_sha256(fm),'controls':{'one_bit_shared_stream':one,'missing_new_stream':missing}}
p=Path('/tmp/dino_spg_row13_bottom_update_bracket.json')
p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(f'SLOT __MEASURED_ROW26_BRACKET_SHA256__ VALUE={sha256(p)}')
PY
```

## Block 4 — CPU/fp64 oracle-identity score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=b15557ae07cd7e858db3b1c73bf6cec6d690e99e
BRACKET_SHA=__MEASURED_ROW26_BRACKET_SHA256__
case "$BRACKET_SHA" in __*) exit 2;; esac
OFF=$(cat /tmp/row26-off-run.txt)
ON=$(cat /tmp/row26-on-run.txt)
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

Block 4 validates the oracle writer and remains explicitly `PRODUCTION_REPLAY_HELD`.
It does not advance row 1.4 before the literal EEN coefficient builder and its
production recurrence are present.
