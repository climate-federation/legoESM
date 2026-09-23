# Preregistration: ZDF row-30 U/V operand ladder

Date: 2026-08-29. CPU-only matched day-180 state. This amendment is frozen
before the held NEMO execution. It completes the interval left open by
`PREREG_zdf_chain_sweep_round30.md`; it does not alter an earlier score.

## Ordered expressions, bars, and controls

The running DINO source is the deterministic-writer baseline
`/tmp/nemo-row19-detwriter.kTFp14/cfgs/DINO/MY_SRC/ldfslp.F90`, SHA256
`a4e65b80484423df67247e4b93ace08fd50fa929e325b2f2af113d4bb31ffd30`.
The literal source-order ladder is:

1. `ldfslp.F90:203-204,217-218` initializes and advances
   `zgru/zgrv(...,iikm1) = mask * (prd(neighbour)-prd)`; dump both the current
   `iik` and newly filled `iikm1` rolling slots. Score both U slots as the U
   gradient stage before either V slot, preventing a V-first false owner.
2. `:242-243`: `zau = zgru(...,iik)*r1_e1u` and the V analogue.
3. `:244-245`: pre-bound `zbu/zbv`, the two-face `zdzr` averages.
4. `:248-249`: post-bound `zbu/zbv`, including literal `-7.e+3/e3u,e3v`.
5. `:269-270`: raw `zwz/zww`, including the mixed-layer carry.
6. `:279-292`: post-Shapiro U/V in NEMO's written corner/cardinal and mask
   association. Capture precedes `CALL lbc_lnk` at `:366`.

Score in that order, U before V where NEMO writes U first, on Python levels
`[1:35]` (Fortran `jk=jpkm1..2`). The normalized whole-column bar is
`1.0e-15`: U must be `0/9758`, V `0/9868`, and every operand must separately
pass the four frozen southern columns `(11,1), (12,1), (13,1), (13,23)`.
The first nonzero count is `DIVERGED`; later stages cannot own it.

The twelve new write-only buffers are completely zero-initialized, halos
included, before interior assignment. Each stream must be 3,273,984 bytes
(`56 * 203 * 36 * 8`, the same as the existing haloed `zgrv` streams).
The patched arm has 209 streams, control 197, the inventory difference is
exactly the twelve named streams, and all 197 shared streams must be exact.
Both output restarts must have SHA256
`33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c`.
There is no retry. A copied shared-stream one-bit change and missing-new-stream
inventory must fail. On verified `zgrv(iik)`, a bar-scale perturbation, i-roll,
and wet NaN must fail; one ULP must add exactly one exact-unequal element.
The scorer also requires exact ON/OFF SHAs for the complete registered input
set (input restart, mesh, both namelists, and both layout files) and stamps
both logs, binary receipts, inputs, output restarts, output domains, and
`ocean.output` files.

## SHA-gated build block (human executes)

Host rules are embedded: `grep`, repo-venv Python, activated `nemo-build`, and
`makenemo -n DINO`. Codex does not execute this block.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row30-build-xdg
repo=/tmp/codex-zdf-sweep
baseline=/tmp/nemo-row19-detwriter.kTFp14
patch_file="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row30_uv_operands.patch"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  de6b46dc3fc1c347e79f9b44f49eda44f40e2edb938c7b7d0a82dfc0bbb7d110
test "$(sha256sum "$baseline/cfgs/DINO/MY_SRC/ldfslp.F90" | awk '{print $1}')" = \
  a4e65b80484423df67247e4b93ace08fd50fa929e325b2f2af113d4bb31ffd30
sha256sum "$baseline"/cfgs/DINO/MY_SRC/*.F90 | sort > \
  /tmp/nemo-row30-baseline-source-manifest.sha256
test "$(sha256sum /tmp/nemo-row30-baseline-source-manifest.sha256 | awk '{print $1}')" = \
  6b1b1fdfcb4b4ac0999f3f7008ce6fddbaf3811edefa06f279a9bfc5de3d8fe7

nemo_src=$(mktemp -d /tmp/nemo-row30-uv.XXXXXX)
cp -a "$baseline"/. "$nemo_src"/
patch --dry-run --fuzz=0 -d "$nemo_src" -p1 < "$patch_file"
patch --fuzz=0 -d "$nemo_src" -p1 < "$patch_file"
test "$(sha256sum "$nemo_src/cfgs/DINO/MY_SRC/ldfslp.F90" | awk '{print $1}')" = \
  8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29
( cd "$nemo_src" && sha256sum cfgs/DINO/MY_SRC/*.F90 | sort > \
  /tmp/nemo-row30-on-source-manifest.sha256 )
test "$(sha256sum /tmp/nemo-row30-on-source-manifest.sha256 | awk '{print $1}')" = \
  433067fd909cafe37b31d094e0f7326177c8628c4ee6d3d9ff31b528b2ec40c3
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_2d' \
  "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 162
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_3d' \
  "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 5
test "$(grep -rE --include='*.F90' \
  'WRITE\([^)]*\).*ji[[:space:]]*=[[:space:]]*1[[:space:]]*,[[:space:]]*jpi' \
  "$nemo_src/cfgs/DINO/MY_SRC" | grep -v dino_dump_zero.F90 | wc -l)" -eq 0
for name in zgru_iik zgru_iikm1 zau zav zbu_pre zbv_pre zbu_post zbv_post \
  uslp_raw vslp_raw uslp_postshapiro vslp_postshapiro; do
  test "$(grep -c "eiv_dump_${name}.bin" \
    "$nemo_src/cfgs/DINO/MY_SRC/ldfslp.F90")" -eq 1
done
for unit in $(seq 8900 8911); do
  test "$(grep -rE --include='*.F90' "UNIT[[:space:]]*=[[:space:]]*$unit" \
    "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done

cd "$nemo_src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row30-on-build.log
on_src=/tmp/ldfslp-row30-on.F90
on_bin=/tmp/nemo-row30-on.exe
cp -p cfgs/DINO/MY_SRC/ldfslp.F90 "$on_src"
cp -p cfgs/DINO/BLD/bin/nemo.exe "$on_bin"
test "$on_bin" -nt "$on_src"
sha256sum "$on_bin" > /tmp/nemo-row30-on-build.sha256
sha256sum "$on_src" "$on_bin" /tmp/nemo-row30-on-build.log
printf '%s\n' "$nemo_src" > /tmp/nemo-row30-source-dir.txt
printf 'SUBSTITUTE __MEASURED_ROW30_ON_BINARY_SHA256__=%s\n' \
  "$(sha256sum "$on_bin" | awk '{print $1}')"
```

## One-step ON/OFF block (human executes)

Replace the marked binary slot. There is deliberately no automatic retry.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row30-run-xdg
ROW30_ON_BINARY_SHA256=__MEASURED_ROW30_ON_BINARY_SHA256__
case "$ROW30_ON_BINARY_SHA256" in __MEASURED_*) echo 'replace SHA slot' >&2; exit 2;; esac
on_bin=/tmp/nemo-row30-on.exe
off_bin=/tmp/nemo-row19-detwriter-off.exe
test "$(sha256sum "$on_bin" | awk '{print $1}')" = "$ROW30_ON_BINARY_SHA256"
test "$(awk 'NR==1 {print $1}' /tmp/nemo-row30-on-build.sha256)" = "$ROW30_ON_BINARY_SHA256"
test "$(sha256sum "$off_bin" | awk '{print $1}')" = \
  4f28b00205658781d7c2a6bbc35362a5a19eb8d57d245bad28a9fbe96ebfafae
ON=$(mktemp -d /tmp/RUN_ZDF30_UV_ON.XXXXXX)
OFF=$(mktemp -d /tmp/RUN_ZDF30_UV_OFF.XXXXXX)
for spec in "ON:$ON:$on_bin:209" "OFF:$OFF:$off_bin:197"; do
  IFS=: read -r tag run_dir binary expected_count <<< "$spec"
  cp -a /tmp/RUN_ZDF19_DETWRITER_OFF.q2tSGL/. "$run_dir"/
  find "$run_dir" -maxdepth 1 \( -type f -o -type l \) \( \
    -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o \
    -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o \
    -name 'ocean.output' -o -name 'run*.log' -o -name '.nemo_binary_sha256' \) -delete
  ln -sfn "$binary" "$run_dir/nemo"
  sha256sum "$binary" > "$run_dir/.nemo_binary_sha256"
  if ! ( cd "$run_dir" && mpirun -np 1 ./nemo > run.attempt1.log 2>&1 ); then
    printf 'HOLD: %s failed; preserve %s\n' "$tag" "$run_dir" >&2; exit 1
  fi
  grep -q '^STOP 0$' "$run_dir/run.attempt1.log"
  test "$(sha256sum "$run_dir/DINO_00005761_restart.nc" | awk '{print $1}')" = \
    33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c
  test "$(find "$run_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq "$expected_count"
done
for name in zgru_iik zgru_iikm1 zau zav zbu_pre zbv_pre zbu_post zbv_post \
  uslp_raw vslp_raw uslp_postshapiro vslp_postshapiro; do
  test "$(stat -c %s "$ON/eiv_dump_${name}.bin")" -eq 3273984
done
printf '%s\n' "$ON" > /tmp/row30-on-dir.txt
printf '%s\n' "$OFF" > /tmp/row30-off-dir.txt
sha256sum "$ON"/eiv_dump_{zgru_iik,zgru_iikm1,zau,zav,zbu_pre,zbv_pre,zbu_post,zbv_post,uslp_raw,vslp_raw,uslp_postshapiro,vslp_postshapiro}.bin
```

## Strict write-only bracket

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row30-on-dir.txt); OFF=$(cat /tmp/row30-off-dir.txt)
PYTHONPATH=scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import sys
from zdf_stream_bracket import files_byte_identical, one_bit_file_control, stream_manifest
on, off = map(Path, sys.argv[1:])
new = {f"eiv_dump_{n}.bin" for n in ("zgru_iik", "zgru_iikm1", "zau", "zav",
 "zbu_pre", "zbv_pre", "zbu_post", "zbv_post", "uslp_raw", "vslp_raw",
 "uslp_postshapiro", "vslp_postshapiro")}
on_m, off_m = stream_manifest(on), stream_manifest(off)
def exact(a, b):
    return len(a) == 209 and len(b) == 197 and a-b == new and not b-a
assert exact(set(on_m), set(off_m))
for name in sorted(off_m): assert files_byte_identical(on/name, off/name), name
assert one_bit_file_control(on/sorted(off_m)[0])
assert not exact(set(on_m)-{next(iter(new))}, set(off_m))
print("row30 write-only bracket VERIFIED: 197/197 shared streams exact; controls fired")
PY
```

## Measured-SHA score block

Replace every marked slot from the run output. Unsubstituted slots are fatal.
Exit 30 is the expected owner receipt; exit 0 is a valid unexpected full pass.

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row30-on-dir.txt); OFF=$(cat /tmp/row30-off-dir.txt)
ROW30_ON_BINARY_SHA256=__MEASURED_ROW30_ON_BINARY_SHA256__
ZGRU_IIK_SHA=__MEASURED_SHA256_EIV_DUMP_ZGRU_IIK_BIN__
ZGRU_IIKM1_SHA=__MEASURED_SHA256_EIV_DUMP_ZGRU_IIKM1_BIN__
ZAU_SHA=__MEASURED_SHA256_EIV_DUMP_ZAU_BIN__
ZAV_SHA=__MEASURED_SHA256_EIV_DUMP_ZAV_BIN__
ZBU_PRE_SHA=__MEASURED_SHA256_EIV_DUMP_ZBU_PRE_BIN__
ZBV_PRE_SHA=__MEASURED_SHA256_EIV_DUMP_ZBV_PRE_BIN__
ZBU_POST_SHA=__MEASURED_SHA256_EIV_DUMP_ZBU_POST_BIN__
ZBV_POST_SHA=__MEASURED_SHA256_EIV_DUMP_ZBV_POST_BIN__
USLP_RAW_SHA=__MEASURED_SHA256_EIV_DUMP_USLP_RAW_BIN__
VSLP_RAW_SHA=__MEASURED_SHA256_EIV_DUMP_VSLP_RAW_BIN__
USLP_POST_SHA=__MEASURED_SHA256_EIV_DUMP_USLP_POSTSHAPIRO_BIN__
VSLP_POST_SHA=__MEASURED_SHA256_EIV_DUMP_VSLP_POSTSHAPIRO_BIN__
for value in "$ROW30_ON_BINARY_SHA256" "$ZGRU_IIK_SHA" "$ZGRU_IIKM1_SHA" \
 "$ZAU_SHA" "$ZAV_SHA" "$ZBU_PRE_SHA" "$ZBV_PRE_SHA" "$ZBU_POST_SHA" \
 "$ZBV_POST_SHA" "$USLP_RAW_SHA" "$VSLP_RAW_SHA" "$USLP_POST_SHA" "$VSLP_POST_SHA"; do
  case "$value" in __MEASURED_*) echo 'replace every measured SHA slot' >&2; exit 2;; esac
done
dump_manifest=/tmp/row30-measured-dump-sha256.json
/home/dbalwada/legoESM/.venv/bin/python - "$dump_manifest" \
 "$ZGRU_IIK_SHA" "$ZGRU_IIKM1_SHA" "$ZAU_SHA" "$ZAV_SHA" \
 "$ZBU_PRE_SHA" "$ZBV_PRE_SHA" "$ZBU_POST_SHA" "$ZBV_POST_SHA" \
 "$USLP_RAW_SHA" "$VSLP_RAW_SHA" "$USLP_POST_SHA" "$VSLP_POST_SHA" <<'PY'
import json, sys
names=("zgru_iik","zgru_iikm1","zau","zav","zbu_pre","zbv_pre",
 "zbu_post","zbv_post","uslp_raw","vslp_raw","uslp_postshapiro","vslp_postshapiro")
json.dump({f"eiv_dump_{n}.bin": s for n,s in zip(names,sys.argv[2:])}, open(sys.argv[1],"w"))
PY
for name in zgru_iik zgru_iikm1 zau zav zbu_pre zbv_pre zbu_post zbv_post \
 uslp_raw vslp_raw uslp_postshapiro vslp_postshapiro; do
 expected=$(/home/dbalwada/legoESM/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
  "$dump_manifest" "eiv_dump_${name}.bin")
 test "$(sha256sum "$ON/eiv_dump_${name}.bin" | awk '{print $1}')" = "$expected"
done
lane_root=/tmp/row30-lane-root; mkdir -p "$lane_root"
ln -sfn "$ON" "$lane_root/RUN_SEQDUMP_D180_1R"
repo_sha=$(git --git-dir=/tmp/zdf-sweep-git.cJQ6wi/repo.git \
 --work-tree=/tmp/codex-zdf-sweep rev-parse HEAD)
set +e
DINO_ORACLE_ROOT="$lane_root" DINO_1226_LANE=d180 \
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
GIT_DIR=/tmp/zdf-sweep-git.cJQ6wi/repo.git GIT_WORK_TREE=/tmp/codex-zdf-sweep \
PYTHONPATH=packages/atmosphere:packages/core:packages/coupler:packages/ice:\
packages/land:packages/ml:packages/ocean:packages/tools:scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python \
 scripts/validate/ocean_fidelity/dino_1226/zdf_row30_uv_operands.py \
 --run-dir "$ON" --bracket-dir "$OFF" \
 --mld-maps /tmp/dino_mld_audit_codex/mld_maps.npz \
 --nemo-source /tmp/ldfslp-row30-on.F90 \
 --bracket-nemo-source /tmp/nemo-row19-detwriter.kTFp14/cfgs/DINO/MY_SRC/ldfslp.F90 \
 --nemo-binary /tmp/nemo-row30-on.exe \
 --bracket-nemo-binary /tmp/nemo-row19-detwriter-off.exe \
 --instrument-patch scripts/validate/ocean_fidelity/dino_1226/nemo_row30_uv_operands.patch \
 --dump-sha-manifest "$dump_manifest" --expected-repo-sha "$repo_sha" \
 --on-source-manifest /tmp/nemo-row30-on-source-manifest.sha256 \
 --off-source-manifest /tmp/nemo-row30-baseline-source-manifest.sha256 \
 --expected-source-sha 8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29 \
 --expected-bracket-source-sha a4e65b80484423df67247e4b93ace08fd50fa929e325b2f2af113d4bb31ffd30 \
 --expected-binary-sha "$ROW30_ON_BINARY_SHA256" \
 --expected-patch-sha de6b46dc3fc1c347e79f9b44f49eda44f40e2edb938c7b7d0a82dfc0bbb7d110 \
 --expected-on-source-manifest-sha 433067fd909cafe37b31d094e0f7326177c8628c4ee6d3d9ff31b528b2ec40c3 \
 --expected-off-source-manifest-sha 6b1b1fdfcb4b4ac0999f3f7008ce6fddbaf3811edefa06f279a9bfc5de3d8fe7 \
 --output /tmp/dino_zdf_row30_uv_operands_artifact.json
rc=$?; set -e
test "$rc" -eq 0 -o "$rc" -eq 30
sha256sum /tmp/dino_zdf_row30_uv_operands_artifact.json
exit "$rc"
```

## Owner and rows 31--32

The first red registered stage owns the localization interval, not necessarily
a single scalar operand. Gradient, metric, and pre/post limiter stages are
atomic at the available dump resolution. Raw U/V are composites; if raw is
first red, the frozen next peel is `iku/ikv -> zfi/zfj -> zdepu/zdepv ->`
carried `zuslp_hml/zvslp_hml` -> interior division -> ML blend, one slot at a
time. Post-Shapiro is also composite; if raw passes and post-Shapiro is first
red, peel corner-pair sum -> cardinal-pair sum -> centre -> horizontal mask
factor -> vertical mask factor -> final product. Only after that peel may a
single operand be called the owner.

Any resulting production fix is a selectable literal path, faithful by
default on the two NEMO DINO cards with legacy opt-in; every other card stays
byte-identical. It requires red, JIT, AD, unchanged-card, and post-fix zero
failure tests.

Row-30 closure removes ordering blocks but does not promote targeting numbers.
No new NEMO dump is presently needed:

- Row 31 must put the already-confirmed `dynzdf.F90:199-214,340-380` literal
  matrix and ordered Thomas recurrences on the production DINO path. Promotion
  requires production U `0/9758`, V `0/9868` at `1.0e-12`, all focus passes,
  wrong-`rDt`/roll controls red, and other cards byte-identical. Its promotion
  scorer must construct both resolved NEMO DINO cards, assert and artifact the
  registered literal selector value, invoke the model's selector-dispatched
  production entry point, and capture that the literal callable ran. Directly
  calling the offline helper is not promotable; selecting legacy must be a red
  control reproducing the current generic failure.
- Only then may row 32 run the production paired tracer solve on
  `(e3t(Kbb)*T(Kbb)+rDt*e3t(Kmm)*Krhs)/e3t(Kaa)`, `avt+K33`, and live
  `e3w(Kmm)`. Its targeting T `627/9920`, S `23/9920` is not promotable.
  Promotion needs T and S both `0/9920` at `1.0e-12`, every focus pass, and
  wrong-`e3t`/roll controls red; otherwise source-order matrix/RHS/Thomas peel.
  The same resolved-card/asserted-selector/production-dispatch capture is
  mandatory for row 32, with legacy and unchanged-card byte controls.

Climate remains unauthorized until rows 30--32 are VERIFIED or waived.
