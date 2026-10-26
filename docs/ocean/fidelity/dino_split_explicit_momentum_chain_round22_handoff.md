# Row-1.3 EEN coefficient SLOT handoff

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. This cascade is **DESIGNED, NOT
RUN**. It stops the ordered walk at the first operand requiring a new NEMO
measurement.

The source stack is explicit: the guarded lean donor already contains the
certified 197-stream deterministic-writer stack; both arms then receive
`nemo_spg_qco_continuity_operands.patch`, producing a 203-stream OFF arm; the
ON arm alone receives `nemo_spg_row13_een_coefficients.patch`, producing eight
additional streams. The bracket is therefore 203/203, not merely a presence
check. Every Python invocation uses checkout-first `PYTHONPATH`, every block
enters its own directory, and the physics producer is pinned to
`a6a5908db19ffaef99fa2e837fa4064b166d2bae`; later packaging-only commits are
admitted only when their model diff from that SHA is empty.

The coefficient writer owns the documented contiguous unit range
`9400--9407`. A full post-QCO `MY_SRC` inventory found no explicit `OPEN`,
`WRITE`, `CLOSE`, or other numeric reference to any unit in that range. The
build block now proves all eight units have zero `OPEN` owners before applying
the coefficient patch and exactly one owner afterward. This replaces the
retired `8990--8997` proposal, which collided with the existing salinity-FCT
writers in `traadv_fct.F90`.

The guarded lean-copy convention below is mandatory for this and every future
held handoff. It rejects non-`/tmp/nemo-*` donors, caps the donor and copied
source sizes, verifies free space, and excludes `RUN_*`, `BLD`, `WORK`, and
`.git`. It must never be replaced by a whole-tree `cp -a`.

## Block 1 — guarded paired build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=a6a5908db19ffaef99fa2e837fa4064b166d2bae
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"
base=/tmp/nemo-row19-detwriter.kTFp14
guard="$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh"
manifest="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_base_source_manifest.sha256"
qco_patch="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_qco_continuity_operands.patch"
cor_patch="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_spg_row13_een_coefficients.patch"
test "$(du -sm "$base" | awk '{print $1}')" -le 3072
test "$(sha256sum "$manifest" | awk '{print $1}')" = 22fdaa20eeb3ff1c04ea22fa0dc293d3db9b9e6bf8cf9638075fdfb819b97e57
test "$(sha256sum "$guard" | awk '{print $1}')" = f1bfbc7e5428c2f532a26aa8197b368847c40421dae6704b33fb91afeb98e071
test "$(find "$base/cfgs/DINO/MY_SRC" -maxdepth 1 -type f | wc -l)" -eq 28
( cd "$base" && sha256sum -c "$manifest" )

# Standard no-copy controls.
for mode in path size free; do
  reject=$(mktemp -d "/tmp/nemo-spg-corcoef-reject-${mode}.XXXXXX")
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

src=$(mktemp -d /tmp/nemo-spg-corcoef-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
  "$guard" "$base" "$src"
( cd "$src" && sha256sum -c "$manifest" )
patch --fuzz=0 -p1 -d "$src" < "$qco_patch"
claimed_units='9400 9401 9402 9403 9404 9405 9406 9407'
for unit in $claimed_units; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
  test "$(grep -rE --include='*.F90' "\\b${unit}\\b" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 0
done
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-spg-corcoef-xdg
cd "$src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-spg-corcoef-off-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-spg-corcoef-off.exe
patch --fuzz=0 -p1 -d "$src" < "$cor_patch"
for unit in $claimed_units; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
  test "$(grep -rlE --include='*.F90' "\\b${unit}\\b" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-spg-corcoef-on-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-spg-corcoef-on.exe
sha256sum "$src/cfgs/DINO/MY_SRC/dynspg_ts.F90" > /tmp/nemo-spg-corcoef-source.sha256
printf '%s\n' "$src" > /tmp/nemo-spg-corcoef-src.txt
printf 'SLOT __CORCOEF_OFF_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-spg-corcoef-off.exe | awk '{print $1}')"
printf 'SLOT __CORCOEF_ON_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-spg-corcoef-on.exe | awk '{print $1}')"
printf 'SLOT __CORCOEF_SOURCE_SHA256__ VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-spg-corcoef-source.sha256)"
```

## Block 2 — fresh paired sequential runs

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF_SHA=__CORCOEF_OFF_BINARY_SHA256__
ON_SHA=__CORCOEF_ON_BINARY_SHA256__
SRC_SHA=__CORCOEF_SOURCE_SHA256__
for value in "$OFF_SHA" "$ON_SHA" "$SRC_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$(sha256sum /tmp/nemo-spg-corcoef-off.exe | awk '{print $1}')" = "$OFF_SHA"
test "$(sha256sum /tmp/nemo-spg-corcoef-on.exe | awk '{print $1}')" = "$ON_SHA"
test "$(awk '{print $1}' /tmp/nemo-spg-corcoef-source.sha256)" = "$SRC_SHA"
template=/tmp/RUN_ZDF18_OPERANDS_ON.tnC9wz
test "$(find "$template" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 197
test "$(sha256sum "$template/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
OFF=$(mktemp -d /tmp/RUN_SPG_CORCOEF_OFF.XXXXXX)
ON=$(mktemp -d /tmp/RUN_SPG_CORCOEF_ON.XXXXXX)
for arm in OFF ON; do
  if test "$arm" = OFF; then run=$OFF; binary=/tmp/nemo-spg-corcoef-off.exe; count=203
  else run=$ON; binary=/tmp/nemo-spg-corcoef-on.exe; count=211; fi
  cp -a "$template"/. "$run"/
  find "$run" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
  rm -f "$run/nemo"
  ln -s "$binary" "$run/nemo"
  sha256sum "$binary" > "$run/.nemo_binary_sha256"
  ( cd "$run" && ./nemo > run.attempt1.log 2>&1 )
  grep -q '^STOP 0$' "$run/run.attempt1.log"
  test "$(find "$run" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq "$count"
done
cp -p /tmp/nemo-spg-corcoef-source.sha256 "$ON/.dynspg_source_sha256"
for name in ffu_nw ffu_ne ffu_sw ffu_se ffv_nw ffv_ne ffv_sw ffv_se; do
  test "$(stat -c %s "$ON/corcoef_dump_${name}.bin")" -eq 90944
  printf 'SLOT __SHA_%s__ VALUE=%s\n' "$(printf %s "$name" | tr '[:lower:]' '[:upper:]')" \
    "$(sha256sum "$ON/corcoef_dump_${name}.bin" | awk '{print $1}')"
done
printf '%s\n' "$OFF" > /tmp/spg-corcoef-off-run.txt
printf '%s\n' "$ON" > /tmp/spg-corcoef-on-run.txt
```

## Block 3 — exact 203/203 bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF=$(cat /tmp/spg-corcoef-off-run.txt)
ON=$(cat /tmp/spg-corcoef-on-run.txt)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import files_byte_identical,manifest_sha256,one_bit_file_control,sha256,stream_manifest
on,off=map(Path,sys.argv[1:])
expected={f"corcoef_dump_{a}_{b}.bin" for a in ("ffu","ffv") for b in ("nw","ne","sw","se")}
om,fm=stream_manifest(on),stream_manifest(off)
def exact(o,f): return len(o)==211 and len(f)==203 and set(o)-set(f)==expected and set(f).issubset(o)
assert exact(om,fm)
assert all(files_byte_identical(on/n,off/n) for n in fm)
assert all(om[n]["size_bytes"]==90944 for n in expected)
one=one_bit_file_control(on/sorted(fm)[0])
planted=dict(om); planted.pop(sorted(expected)[0])
missing=not exact(planted,fm)
assert one and missing
receipt={"schema":"dino-spg-row13-een-coefficient-bracket-v1","on_dir":str(on.resolve()),"off_dir":str(off.resolve()),"shared_count":203,"shared_exact":True,"shared_manifest_sha256":manifest_sha256({n:om[n] for n in sorted(fm)}),"new_streams":sorted(expected),"on_manifest_sha256":manifest_sha256(om),"off_manifest_sha256":manifest_sha256(fm),"controls":{"one_bit_shared_stream":one,"missing_new_stream":missing}}
p=Path('/tmp/dino_spg_row13_een_coefficient_bracket.json')
p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(f'SLOT __CORCOEF_BRACKET_SHA256__ VALUE={sha256(p)}')
PY
```

## Block 4 — CPU/fp64 coefficient score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=a6a5908db19ffaef99fa2e837fa4064b166d2bae
BRACKET_SHA=__CORCOEF_BRACKET_SHA256__
case "$BRACKET_SHA" in __*) exit 2;; esac
OFF=$(cat /tmp/spg-corcoef-off-run.txt)
ON=$(cat /tmp/spg-corcoef-on-run.txt)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round22.py" \
  --on-run "$ON" --off-run "$OFF" \
  --bracket /tmp/dino_spg_row13_een_coefficient_bracket.json \
  --bracket-sha "$BRACKET_SHA" --producer "$producer" \
  --round21 /tmp/dino_split_explicit_momentum_chain_round21_pgf.json \
  --output /tmp/dino_split_explicit_momentum_chain_round22_corcoef.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round22_corcoef.json
```

No block pushes, invokes a GPU, or uses `mpirun`.
