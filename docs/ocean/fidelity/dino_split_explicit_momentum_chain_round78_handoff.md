# Round-78 Redi flux-ladder SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Physics/instrument producer
`5de6f7a6370e6e04113110cc5a29fc97b1ae1a28`.

Round 77 is `REDI_MSC_E3W_MAJORITY`, so the exact live W thickness is replayed
but not yet made the production default.  This held cascade records the first
missing operands in NEMO order: `zfu`, `zfv`, and total `zfw_kp1`, for T then
S.  The writer uses reserved units `9450--9455`; every buffer is explicitly
zeroed and every stream is genuine full-halo `(35,203,56)`.

Every block changes to the absolute checkout, exports checkout-first
`PYTHONPATH` including `packages/core`, pins the producer SHA and admits later
packaging commits only through a zero model diff, uses a tracked-only porcelain
gate, and keeps the `__MEASURED_` SLOT convention.  The guarded source copy
excludes `RUN_*`, `BLD`, `WORK`, and `.git`; block 1 also requires 8 GiB free
before it starts.  It must stop, not weaken that gate, when the host needs
cleanup.  The fresh OFF run is reduced to its cryptographic manifest before
the ON run to limit scratch use; that temporary OFF directory is then removed
and is reproducible from its pinned binary and template.

## Block 1 — guarded OFF/ON build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=5de6f7a6370e6e04113110cc5a29fc97b1ae1a28
git cat-file -e "${producer}^{commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean src)"

base=/tmp/nemo-row30-uslp.NPhyli
guard="$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh"
patch_file="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_redi_flux_ladder.patch"
test "$(du -sm "$base" | awk '{print $1}')" -le 3072
test "$(df -Pm /tmp | awk 'NR==2 {print $4}')" -ge 8192
test "$(sha256sum "$guard" | awk '{print $1}')" = f1bfbc7e5428c2f532a26aa8197b368847c40421dae6704b33fb91afeb98e071
test "$(sha256sum "$patch_file" | awk '{print $1}')" = bf5756a4fcf262d64f28267c227355085defddf8365f8a2ffcbe94d493dc8fed

src=$(mktemp -d /tmp/nemo-redi-flux-round78-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=8192 \
  "$guard" "$base" "$src"
test "$(du -sm "$src" | awk '{print $1}')" -le 512
cp -p "$src/src/OCE/TRA/traldf_iso.F90" "$src/cfgs/DINO/MY_SRC/traldf_iso.F90"

for unit in 9450 9451 9452 9453 9454 9455; do
  test "$(grep -rE --include='*.F90' "\\b${unit}\\b" "$src" | wc -l)" -eq 0
done
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-redi-flux-round78-xdg
cd "$src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-redi-flux-round78-off-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-redi-flux-round78-off.exe

patch --fuzz=0 -p1 -d "$src" < "$patch_file"
for unit in 9450 9451 9452 9453 9454 9455; do
  test "$(grep -rE --include='*.F90' "OPEN[[:space:]]*\\([[:space:]]*UNIT[[:space:]]*=[[:space:]]*${unit}([^0-9]|$)" \
    "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
  test "$(grep -rlE --include='*.F90' "\\b${unit}\\b" "$src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-redi-flux-round78-on-build.log
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-redi-flux-round78-on.exe
sha256sum "$src/cfgs/DINO/MY_SRC/traldf_iso.F90" > /tmp/nemo-redi-flux-round78-source.sha256
printf '%s\n' "$src" > /tmp/nemo-redi-flux-round78-src.txt
printf 'SLOT __MEASURED_ROW78_PRODUCER__ VALUE=%s\n' "$producer"
printf 'SLOT __MEASURED_ROW78_OFF_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-redi-flux-round78-off.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW78_ON_BINARY_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/nemo-redi-flux-round78-on.exe | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW78_SOURCE_SHA256__ VALUE=%s\n' "$(awk '{print $1}' /tmp/nemo-redi-flux-round78-source.sha256)"
```

## Block 2 — full-fresh OFF/ON sequential runs

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW78_PRODUCER__
OFF_SHA=__MEASURED_ROW78_OFF_BINARY_SHA256__
ON_SHA=__MEASURED_ROW78_ON_BINARY_SHA256__
SRC_SHA=__MEASURED_ROW78_SOURCE_SHA256__
for value in "$PRODUCER" "$OFF_SHA" "$ON_SHA" "$SRC_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 5de6f7a6370e6e04113110cc5a29fc97b1ae1a28
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$PRODUCER" HEAD -- packages/core packages/ocean src)"
test "$(sha256sum /tmp/nemo-redi-flux-round78-off.exe | awk '{print $1}')" = "$OFF_SHA"
test "$(sha256sum /tmp/nemo-redi-flux-round78-on.exe | awk '{print $1}')" = "$ON_SHA"
test "$(awk '{print $1}' /tmp/nemo-redi-flux-round78-source.sha256)" = "$SRC_SHA"

template=/tmp/RUN_BN2_CERT_1R
test "$(du -sm "$template" | awk '{print $1}')" -le 1024
test "$(sha256sum "$template/DINO_00005760_restart.nc" | awk '{print $1}')" = 0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
test "$(sha256sum "$template/mesh_mask.nc" | awk '{print $1}')" = 3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622
run_root=/tmp/dino-redi-flux-round78-01a04e34
test ! -e "$run_root"
mkdir -p "$run_root"

export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX="$run_root/pycache"
off="$run_root/off"
mkdir "$off"
cp -a "$template"/. "$off"/
find "$off" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
rm -f "$off/nemo"
ln -s /tmp/nemo-redi-flux-round78-off.exe "$off/nemo"
( cd "$off" && ./nemo > run.attempt1.log 2>&1 )
grep -q '^STOP 0$' "$off/run.attempt1.log"
/home/dbalwada/legoESM/.venv/bin/python - "$off" "$run_root/off_manifest.json" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import stream_manifest
run,out=map(Path,sys.argv[1:])
m=stream_manifest(run)
assert len(m) >= 160
out.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
PY
off_count=$(/home/dbalwada/legoESM/.venv/bin/python -c "import json; print(len(json.load(open('$run_root/off_manifest.json'))))")
case "$off" in /tmp/dino-redi-flux-round78-01a04e34/off) rm -rf "$off";; *) exit 2;; esac

on="$run_root/on"
mkdir "$on"
cp -a "$template"/. "$on"/
find "$on" -maxdepth 1 -type f \( -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o -name 'ocean.output' -o -name 'run*.log' \) -delete
rm -f "$on/nemo"
ln -s /tmp/nemo-redi-flux-round78-on.exe "$on/nemo"
( cd "$on" && ./nemo > run.attempt1.log 2>&1 )
grep -q '^STOP 0$' "$on/run.attempt1.log"
on_count=$(find "$on" -maxdepth 1 -type f -name '*.bin' | wc -l)
test "$on_count" -eq $((off_count + 6))
for tracer in tem sal; do
  for flux in zfu zfv zfw; do
    file="$on/redi_dump_${flux}_${tracer}.bin"
    test "$(stat -c %s "$file")" -eq 3183040
    slot=$(printf '%s_%s' "$flux" "$tracer" | tr '[:lower:]' '[:upper:]')
    printf 'SLOT __MEASURED_ROW78_SHA_%s__ VALUE=%s\n' "$slot" "$(sha256sum "$file" | awk '{print $1}')"
  done
done
cp -p /tmp/nemo-redi-flux-round78-source.sha256 "$on/.traldf_iso_source_sha256"
printf '%s\n' "$run_root" > /tmp/dino-redi-flux-round78-run-root.txt
printf 'SLOT __MEASURED_ROW78_OFF_MANIFEST_SHA256__ VALUE=%s\n' "$(sha256sum "$run_root/off_manifest.json" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW78_ON_DIR__ VALUE=%s\n' "$on"
```

## Block 3 — exact shared-stream bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
OFF_MANIFEST_SHA=__MEASURED_ROW78_OFF_MANIFEST_SHA256__
ON_DIR=__MEASURED_ROW78_ON_DIR__
for value in "$OFF_MANIFEST_SHA" "$ON_DIR"; do case "$value" in __*) exit 2;; esac; done
run_root=$(cat /tmp/dino-redi-flux-round78-run-root.txt)
test "$ON_DIR" = "$run_root/on"
test "$(sha256sum "$run_root/off_manifest.json" | awk '{print $1}')" = "$OFF_MANIFEST_SHA"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
/home/dbalwada/legoESM/.venv/bin/python - "$ON_DIR" "$run_root/off_manifest.json" "$run_root/redi_flux_bracket.json" <<'PY'
from pathlib import Path
import json,sys
from zdf_stream_bracket import one_bit_file_control,sha256,stream_manifest
on,off_path,out=map(Path,sys.argv[1:])
om=stream_manifest(on); fm=json.loads(off_path.read_text())
expected={f'redi_dump_{flux}_{tracer}.bin' for flux in ('zfu','zfv','zfw') for tracer in ('tem','sal')}
shared=set(fm)
exact=(set(om)-shared==expected and shared.issubset(om) and all(om[n]==fm[n] for n in shared))
assert exact and all(om[n]['size_bytes']==3183040 for n in expected)
one=one_bit_file_control(on/sorted(shared)[0])
planted=set(om); planted.remove(sorted(expected)[0])
missing=(planted-shared != expected)
assert one and missing
receipt={'schema':'dino-redi-flux-bracket-v1','on_dir':str(on.resolve()),
 'shared_count':len(shared),'shared_exact':True,'new_streams':sorted(expected),
 'new_stream_manifest':{n:om[n] for n in sorted(expected)},
 'controls':{'one_bit_shared_stream':one,'missing_new_stream':missing}}
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(f'SLOT __MEASURED_ROW78_BRACKET_SHA256__ VALUE={sha256(out)}')
PY
```

## Block 4 — CPU/fp64 registered score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW78_PRODUCER__
BRACKET_SHA=__MEASURED_ROW78_BRACKET_SHA256__
for value in "$PRODUCER" "$BRACKET_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 5de6f7a6370e6e04113110cc5a29fc97b1ae1a28
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$PRODUCER" HEAD -- packages/core packages/ocean src)"
test "$(sha256sum "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round55.py" | awk '{print $1}')" = 4498395b95d34efab12a1ed1bac62936d92442b70ca1f062058ac578ce06ae27
test "$(sha256sum "$repo/docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round78.md" | awk '{print $1}')" = c0fe235ef86abf4a639bc8ae5e1906675b6d4feae7432b55628cf50a154ebcc0
run_root=$(cat /tmp/dino-redi-flux-round78-run-root.txt)
bracket="$run_root/redi_flux_bracket.json"
test "$(sha256sum "$bracket" | awk '{print $1}')" = "$BRACKET_SHA"
held=/tmp/RUN_LATERAL_ROW8_PU_ON.Qln5u8
redi=/tmp/RUN_BN2_CERT_1R
traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
output=/tmp/dino_split_explicit_momentum_chain_round78.json
test ! -e "$output"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX="$run_root/pycache"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round55.py" \
  --held-dir "$held" --run-traj "$traj" \
  --round54 /tmp/dino_split_explicit_momentum_chain_round54.json \
  --round56 /tmp/dino_split_explicit_momentum_chain_round58_unheld.json \
  --round59 /tmp/dino_split_explicit_momentum_chain_round59_held.json \
  --round60 /tmp/dino_split_explicit_momentum_chain_round60.json \
  --round61 /tmp/dino_split_explicit_momentum_chain_round61.json \
  --round62 /tmp/dino_split_explicit_momentum_chain_round62.json \
  --round63 /tmp/dino_split_explicit_momentum_chain_round63.json \
  --round64 /tmp/dino_split_explicit_momentum_chain_round64.json \
  --round65 /tmp/dino_split_explicit_momentum_chain_round65.json \
  --round66 /tmp/dino_split_explicit_momentum_chain_round66.json \
  --round67 /tmp/dino_split_explicit_momentum_chain_round67.json \
  --round68 /tmp/dino_split_explicit_momentum_chain_round68.json \
  --round69 /tmp/dino_split_explicit_momentum_chain_round69.json \
  --round70 /tmp/dino_split_explicit_momentum_chain_round70.json \
  --round71 /tmp/dino_split_explicit_momentum_chain_round71.json \
  --round72 /tmp/dino_split_explicit_momentum_chain_round72.json \
  --round73 /tmp/dino_split_explicit_momentum_chain_round73.json \
  --round74 /tmp/dino_split_explicit_momentum_chain_round74.json \
  --round75 /tmp/dino_split_explicit_momentum_chain_round75.json \
  --round76 /tmp/dino_split_explicit_momentum_chain_round76.json \
  --round77 /tmp/dino_split_explicit_momentum_chain_round77.json \
  --hold-slow-forcing --oracle-transport --capture-cycle --direct-cycle-entry \
  --live-thickness-entry --capture-bolus-operands --capture-kappa-operands \
  --literal-kappa-reduction --capture-kappa-geometry --coupled-kappa-carry \
  --surface-kmm-carry --exact-surface-kmm-carry --post-chain-factorial \
  --rossby-factorial --zn-sqrt-factorial --exact-sqrt-production \
  --capture-redi-tail --redi-e3w-factorial --redi-flux-ladder \
  --redi-run-dir "$redi" --redi-flux-run-dir "$run_root/on" \
  --redi-flux-bracket "$bracket" --redi-flux-bracket-sha "$BRACKET_SHA" \
  --raw-artifact /tmp/dino_lateral_row8_pu_held_artifact.json \
  --nemo-root "$nemo" --output "$output"
sha256sum "$output"
```

No block pushes, invokes a GPU, or uses `mpirun`.  The scorer stops at the
first red flux in NEMO order.  If all six pass while the total tendency stays
red, the next registered target is divergence/volume-factor association; if
`zfw` is first, the next target is the A31/A32 versus explicit-A33 split.
