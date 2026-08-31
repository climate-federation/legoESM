# Row-1.4 substep-trajectory SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. **DESIGNED, NOT RUN.** The admitted
base is the cumulative round-26 source (deterministic writer + continuity
operands + EEN coefficients + bottom/update operands); the OFF binary is its
retained SHA-pinned executable. The new patch claims units 9440--9442.

**Scorer correction:** blocks 1--3 completed and remain admitted. The block-4
scorer below incorrectly intercepts `scan`; the faithful card executes
`fori_loop`. Do not rerun this file's block 4. Use only the corrected block in
`dino_split_explicit_momentum_chain_round32_resumed_handoff.md`.

## Block 1 — guarded cumulative source copy and ON build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=bb65603854b8f6bdf7cb0c7aa59707f708bf32a2
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"
base=/tmp/nemo-row26-src.75aqmw
off=/tmp/nemo-row26-on.exe
guard="$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh"
patch_file="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_row14_substep_trace.patch"
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/dynspg_ts.F90" | awk '{print $1}')" = e203f1011f81a5e17e54c990bba547cfbdcd066acdd48913f1b3d323d4c0a0ff
test "$(sha256sum "$off" | awk '{print $1}')" = e2eb657e31486b94118a0792d14480f048e9412285a4943457141574cd55f163
test "$(sha256sum "$guard" | awk '{print $1}')" = f1bfbc7e5428c2f532a26aa8197b368847c40421dae6704b33fb91afeb98e071
test "$(sha256sum "$patch_file" | awk '{print $1}')" = 7c3a0f6c2e33c191977afada68eb5c1bc5dd7c2e21808eb8bebd1ede914d186d
test "$(du -sm "$base" | awk '{print $1}')" -le 512
for unit in 9440 9441 9442; do
  test "$(grep -rE --include='*.F90' "\\b${unit}\\b" "$base/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
done
src=$(mktemp -d /tmp/nemo-row32-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=512 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
  "$guard" "$base" "$src"
test "$(sha256sum "$src/cfgs/DINO/MY_SRC/dynspg_ts.F90" | awk '{print $1}')" = e203f1011f81a5e17e54c990bba547cfbdcd066acdd48913f1b3d323d4c0a0ff
patch --fuzz=0 -p1 -d "$src" < "$patch_file"
for unit in 9440 9441 9442; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row32-xdg
cd "$src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row32-on-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-row32-on.exe
sha256sum "$src/cfgs/DINO/MY_SRC/dynspg_ts.F90" > /tmp/nemo-row32-source.sha256
printf '%s\n' "$src" > /tmp/nemo-row32-src.txt
printf 'SLOT __MEASURED_ROW32_ON_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-row32-on.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW32_SOURCE_SHA256__ VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-row32-source.sha256)"
```

## Block 2 — fresh OFF/ON sequential runs

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
ON_SHA=__MEASURED_ROW32_ON_BINARY_SHA256__
SRC_SHA=__MEASURED_ROW32_SOURCE_SHA256__
for value in "$ON_SHA" "$SRC_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$(sha256sum /tmp/nemo-row32-on.exe | awk '{print $1}')" = "$ON_SHA"
test "$(awk '{print $1}' /tmp/nemo-row32-source.sha256)" = "$SRC_SHA"
template=/tmp/RUN_SPG_ROW13_BOTTOM_UPDATE_ON.VlHrvW
test "$(find "$template" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 223
test "$(sha256sum "$template/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
OFF=$(mktemp -d /tmp/RUN_SPG_ROW14_TRACE_OFF.XXXXXX)
ON=$(mktemp -d /tmp/RUN_SPG_ROW14_TRACE_ON.XXXXXX)
for arm in OFF ON; do
  if test "$arm" = OFF; then run=$OFF; binary=/tmp/nemo-row26-on.exe; count=223
  else run=$ON; binary=/tmp/nemo-row32-on.exe; count=226; fi
  cp -a "$template"/. "$run"/
  find "$run" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
  rm -f "$run/nemo"
  ln -s "$binary" "$run/nemo"
  sha256sum "$binary" > "$run/.nemo_binary_sha256"
  if test "$arm" = ON; then cp -p /tmp/nemo-row32-source.sha256 "$run/.dynspg_source_sha256"
  else sha256sum /tmp/nemo-row26-src.75aqmw/cfgs/DINO/MY_SRC/dynspg_ts.F90 > "$run/.dynspg_source_sha256"; fi
  ( cd "$run" && ./nemo > run.attempt1.log 2>&1 )
  grep -q '^STOP 0$' "$run/run.attempt1.log"
  test "$(find "$run" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq "$count"
done
for name in ssh u v; do
  file="$ON/row14_dump_${name}_substeps.bin"
  test "$(stat -c %s "$file")" -eq 6184192
  printf 'SLOT __MEASURED_ROW32_SHA_%s__ VALUE=%s\n' "$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')" "$(sha256sum "$file" | awk '{print $1}')"
done
printf '%s\n' "$OFF" > /tmp/row32-off-run.txt
printf '%s\n' "$ON" > /tmp/row32-on-run.txt
```

## Block 3 — exact 223/223+3 bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF=$(cat /tmp/row32-off-run.txt)
ON=$(cat /tmp/row32-on-run.txt)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import files_byte_identical,manifest_sha256,one_bit_file_control,sha256,stream_manifest
on,off=map(Path,sys.argv[1:])
expected={'row14_dump_ssh_substeps.bin','row14_dump_u_substeps.bin','row14_dump_v_substeps.bin'}
om,fm=stream_manifest(on),stream_manifest(off)
def exact(o,f): return len(o)==226 and len(f)==223 and set(o)-set(f)==expected and set(f).issubset(o)
assert exact(om,fm)
assert all(files_byte_identical(on/n,off/n) for n in fm)
assert all(om[n]['size_bytes']==6184192 for n in expected)
one=one_bit_file_control(on/sorted(fm)[0])
planted=dict(om); planted.pop(sorted(expected)[0])
missing=not exact(planted,fm)
assert one and missing
receipt={'schema':'dino-spg-row14-substep-trace-bracket-v1','on_dir':str(on.resolve()),'off_dir':str(off.resolve()),'shared_count':223,'shared_exact':True,'shared_manifest_sha256':manifest_sha256({n:om[n] for n in sorted(fm)}),'new_streams':sorted(expected),'on_manifest_sha256':manifest_sha256(om),'off_manifest_sha256':manifest_sha256(fm),'controls':{'one_bit_shared_stream':one,'missing_new_stream':missing}}
p=Path('/tmp/dino_spg_row14_substep_trace_bracket.json')
p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(f'SLOT __MEASURED_ROW32_BRACKET_SHA256__ VALUE={sha256(p)}')
PY
```

## Block 4 — CPU/fp64 trajectory score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=bb65603854b8f6bdf7cb0c7aa59707f708bf32a2
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"
test "$(sha256sum "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round32.py" | awk '{print $1}')" = 70bef026e24c25782ab7abe20d3735b9ace8a31684179858c5e776b349ee1069
test "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round31_production_replay.json | awk '{print $1}')" = 7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d
OFF=$(cat /tmp/row32-off-run.txt)
ON=$(cat /tmp/row32-on-run.txt)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round32.py" \
  --on-run "$ON" --off-run "$OFF" \
  --bracket /tmp/dino_spg_row14_substep_trace_bracket.json \
  --round31 /tmp/dino_split_explicit_momentum_chain_round31_production_replay.json \
  --output /tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json
```
