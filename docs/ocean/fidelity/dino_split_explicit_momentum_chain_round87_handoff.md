# Round-87 Redi zfw component SLOT handoff

All blocks `cd` themselves, use checkout-first `PYTHONPATH`, retain the
tracked-only gate, and pin the producer by SHA. Units 9460--9463 are reserved.
The round-78 source is load-bearing and is listed in `KEEP_TMP_TREES.txt`; the
guarded copy excludes its BLD/RUN payload. The component patch has a final
source SHA SLOT gate because its historical round-78 context requires patch
fuzz 1--2, the same admitted-by-final-hash convention used by the row-30
reconstruction.

## Block 1 — guarded source copy and build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=f9032e65ec7b0854bbb2b01af6df2ef34b2429fd
git cat-file -e "$producer^{commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean scripts/validate/ocean_fidelity/dino_1226)"
base=/tmp/nemo-redi-flux-round78-src.EYzfxX
grep -Fxq "$base" "$repo/scripts/validate/ocean_fidelity/dino_1226/KEEP_TMP_TREES.txt"
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/traldf_iso.F90" | awk '{print $1}')" = f69ff46e61e7212eab381955b2a28b20b4b006fe3d9e5b065f8558f9dc0d4adf
test "$(sha256sum "$base/src/OCE/TRA/traldf_iso_scheme.h90" | awk '{print $1}')" = 65d3c21f96f395ab71ff513a15c671ae75c8a87a44d7d8e98152e1a409624598
guard="$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh"
patchfile="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_redi_zfw_components.patch"
test "$(df -Pm /tmp | awk 'NR==2 {print $4}')" -ge 4096
src=$(mktemp -d /tmp/nemo-redi-zfw-round87-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 "$guard" "$base" "$src"
for unit in 9460 9461 9462 9463; do test "$(grep -rE --include='*.F90' "\b${unit}\b" "$src" | wc -l)" -eq 0; done
patch --dry-run --fuzz=2 -p1 -d "$src" < "$patchfile"
patch --fuzz=2 -p1 -d "$src" < "$patchfile"
test "$(sha256sum "$src/cfgs/DINO/MY_SRC/traldf_iso.F90" | awk '{print $1}')" = 3cc1548fbe98b833ff0871e91cb05eeae476ef940b21b44528e924277653bf2b
test "$(sha256sum "$src/src/OCE/TRA/traldf_iso_scheme.h90" | awk '{print $1}')" = 3e55740bfdaabc0516af966ac4a05a2f728dd00564ae881536d69b30226cfe68
for unit in 9460 9461 9462 9463; do test "$(grep -rE --include='*.F90' "OPEN.*UNIT.*=${unit}([^0-9]|$)" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1; done
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-redi-zfw-round87-xdg
cd "$src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-redi-zfw-round87-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-redi-zfw-round87.exe
printf 'SLOT __MEASURED_ROW87_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-redi-zfw-round87.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW87_TRA_SOURCE_SHA256__ VALUE=%s\n' "$(sha256sum cfgs/DINO/MY_SRC/traldf_iso.F90 | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW87_SCHEME_SOURCE_SHA256__ VALUE=%s\n' "$(sha256sum src/OCE/TRA/traldf_iso_scheme.h90 | awk '{print $1}')"
```

## Block 2 — fresh one-rank run

```bash
set -euo pipefail
repo=/tmp/codex-zdf-sweep; cd "$repo"
producer=f9032e65ec7b0854bbb2b01af6df2ef34b2429fd; binary_sha=__MEASURED_ROW87_BINARY_SHA256__
git cat-file -e "$producer^{commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean scripts/validate/ocean_fidelity/dino_1226)"
test "$(sha256sum /tmp/nemo-redi-zfw-round87.exe | awk '{print $1}')" = "$binary_sha"
template=/tmp/RUN_BN2_CERT_1R; run_root=/tmp/dino-redi-zfw-round87-01a04e34
test ! -e "$run_root"; mkdir -p "$run_root/on"
cp -a "$template"/. "$run_root/on"/
find "$run_root/on" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'run*.log' \) -delete
ln -sf /tmp/nemo-redi-zfw-round87.exe "$run_root/on/nemo"
( cd "$run_root/on" && ./nemo > run.attempt1.log 2>&1 )
grep -q '^STOP 0$' "$run_root/on/run.attempt1.log"
for term in skew a33; do for tracer in tem sal; do test "$(stat -c %s "$run_root/on/redi_dump_zfw_${term}_${tracer}.bin")" -eq 3183040; done; done
printf 'SLOT __MEASURED_ROW87_RUN_DIR__ VALUE=%s\n' "$run_root/on"
```

## Block 3 — exact shared-stream bracket

```bash
set -euo pipefail
repo=/tmp/codex-zdf-sweep; cd "$repo"
export PYTHONPATH="$repo/scripts/validate/ocean_fidelity/dino_1226:$repo/packages/core:$repo/packages/ocean:$repo"
run=__MEASURED_ROW87_RUN_DIR__; out=/tmp/dino_redi_zfw_round87_bracket.json
python3 - "$run" "$out" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import stream_manifest,one_bit_file_control
run,out=map(Path,sys.argv[1:]); old=Path('/tmp/dino-redi-flux-round78-01a04e34/on')
m0,m1=stream_manifest(old),stream_manifest(run)
new={f'redi_dump_zfw_{a}_{t}.bin' for a in ('skew','a33') for t in ('tem','sal')}
shared=set(m0)&set(m1)
assert all(m0[n]==m1[n] for n in shared)
assert new <= set(m1) and not (new & set(m0))
r={'schema':'dino-redi-zfw-component-bracket-v1','new_streams':sorted(new),'new_stream_manifest':{n:m1[n] for n in sorted(new)},'controls':{'shared_exact':True,'one_bit_red':one_bit_file_control(run/sorted(new)[0])}}
out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
PY
printf 'SLOT __MEASURED_ROW87_BRACKET_SHA256__ VALUE=%s\n' "$(sha256sum "$out" | awk '{print $1}')"
```

## Block 4 — CPU/fp64 score

```bash
set -euo pipefail
repo=/tmp/codex-zdf-sweep; cd "$repo"
producer=f9032e65ec7b0854bbb2b01af6df2ef34b2429fd; run=__MEASURED_ROW87_RUN_DIR__; bracket_sha=__MEASURED_ROW87_BRACKET_SHA256__
bash "$repo/scripts/validate/ocean_fidelity/dino_1226/run_round87_redi_zfw_score.sh" "$producer" "$run" /tmp/dino_redi_zfw_round87_bracket.json "$bracket_sha" /tmp/dino_split_explicit_momentum_chain_round87.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round87.json
```
