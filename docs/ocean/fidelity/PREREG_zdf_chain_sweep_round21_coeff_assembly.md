# Preregistration: ZDF row 21 base coefficient assembly

Date: 2026-08-29. Ordered predecessor rows 19 and 20 are VERIFIED. Row 21 is
the next unresolved row; nothing at row 22 or later can be promoted across it.

## Oracle execution order and registered outputs

The uninstrumented deterministic-writer baseline evaluates the active DINO
chain at `cfgs/DINO/MY_SRC/zdftke.F90:913-925` in
`/tmp/nemo-row19-detwriter.kTFp14`. Within each `jj` slice NEMO evaluates:

```fortran
zsqen = SQRT( en(ji,jj,jk) )
zav   = rn_ediff * zmxlm(ji,jk) * zsqen
p_avm(ji,jj,jk) = MAX( zav,                  avmb(jk) ) * wmask(ji,jj,jk)
p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)
dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)
```

Only after those five terms does `nn_pdl == 1` overwrite `p_avt` at line 924.
The write-only patch therefore captures, in this exact order,
`zsqen`, `zav`, base `avm`, base pre-Prandtl `avt`, and post-`tke_avn`
`dissl`. Every capture buffer is allocated over `jpi,jpj,jpk`, zeroed in full,
and filled only at the live `jk=1..jpkm1` assignment. The terminal level and
halos remain deterministic zero. The five streams contain the physical
interior only and each has registered size 2,980,224 bytes.

After applying the patch, the literal expressions are at lines 924--928, the
write-only copies at 930--934, and the Prandtl overwrite at 944.

The patch is
`scripts/validate/ocean_fidelity/dino_1226/nemo_row21_coeff_assembly.patch`,
SHA256 `cd2fcb09e1761e0bd0fa5fe8da79dc8c22adc7eb320a812135d72b09173d8666`.
It applies to `zdftke.F90` SHA256
`6a33079678505c105cae1edb8a52b9d70d02012d0eb3395673f4b62fe657a9fd`
and produces SHA256
`37f0fc774d85baf3f04082e2584a587345f4d38796037603ac390d8d1ce07ee0`.

## Registered bars and dispositions

Each term is scored over every wet level in each of 9,920 columns, plus the
four registered southern-basin focus columns. Its per-column statistic is the
maximum absolute element difference. The frozen bar is `1.0e-15` in each
term's native units. CONFIRM is zero failing columns, no nonfinite wet values,
and zero focus failures. Any failure is `DIVERGED`, and measurement stops at
the first failing term in the five-term order above. Later terms are not used
to explain an earlier failure.

The production operands are NEMO's already-verified post-row-18 `en`, final
row-20 `zmxlm/zmxld`, resolved `rn_ediff`, `avmb`, `avtb_2d*avtb`, and `wmask`.
The scorer must call the production TKE coefficient path and retain the literal
source association above. It must not compare a NEMO reconstruction with a
NEMO dump and call that a legoESM measurement.

Red controls are preregistered before the run: a one-i roll, one wet value
raised by more than the absolute bar, and one wet NaN must each fail; a one-bit
change to one shared stream and a missing shared stream must fail the same
bracket predicates used for the production receipt.

## SHA-gated rebuild and one-step bracket

Two corrections from the prior held blocks are permanent: this host has no
`rg`, so shell gates use `grep -rE --include`; and every Python invocation uses
the repository virtual environment, never bare `python`.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row21-xdg

repo=/tmp/codex-zdf-sweep
baseline=/tmp/nemo-row19-detwriter.kTFp14
patch_file="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row21_coeff_assembly.patch"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  cd2fcb09e1761e0bd0fa5fe8da79dc8c22adc7eb320a812135d72b09173d8666
test "$(sha256sum "$baseline/cfgs/DINO/MY_SRC/zdftke.F90" | awk '{print $1}')" = \
  6a33079678505c105cae1edb8a52b9d70d02012d0eb3395673f4b62fe657a9fd
sha256sum "$baseline"/cfgs/DINO/MY_SRC/*.F90 | sort > \
  /tmp/nemo-row21-baseline-source-manifest.sha256
test "$(sha256sum /tmp/nemo-row21-baseline-source-manifest.sha256 | awk '{print $1}')" = \
  6b1b1fdfcb4b4ac0999f3f7008ce6fddbaf3811edefa06f279a9bfc5de3d8fe7

nemo_src=$(mktemp -d /tmp/nemo-row21-coeff.XXXXXX)
cp -a "$baseline"/. "$nemo_src/"
cd "$nemo_src"
patch --dry-run -p1 < "$patch_file"
patch -p1 < "$patch_file"
test "$(sha256sum cfgs/DINO/MY_SRC/zdftke.F90 | awk '{print $1}')" = \
  37f0fc774d85baf3f04082e2584a587345f4d38796037603ac390d8d1ce07ee0
sha256sum cfgs/DINO/MY_SRC/*.F90 | sort > \
  /tmp/nemo-row21-on-source-manifest.sha256
test "$(sha256sum /tmp/nemo-row21-on-source-manifest.sha256 | awk '{print $1}')" = \
  82e54ec72d8a2e6f304e285896b899408399ade0480aaa604a70f8892cc84a95
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_2d' \
  cfgs/DINO/MY_SRC | wc -l)" -eq 150
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_3d' \
  cfgs/DINO/MY_SRC | wc -l)" -eq 5
test "$(grep -rE --include='*.F90' \
  'WRITE\([^)]*\).*ji[[:space:]]*=[[:space:]]*1[[:space:]]*,[[:space:]]*jpi' \
  cfgs/DINO/MY_SRC | grep -v dino_dump_zero.F90 | wc -l)" -eq 0
for name in zsqen_base zav_base avm_base avt_base dissl_postavn; do
  test "$(grep -c "tke_dump_${name}.bin" cfgs/DINO/MY_SRC/zdftke.F90)" -eq 1
done

./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row21-on-build.log
on_src=/tmp/zdftke-row21-on.F90
on_bin=/tmp/nemo-row21-on.exe
cp -p cfgs/DINO/MY_SRC/zdftke.F90 "$on_src"
cp -p cfgs/DINO/BLD/bin/nemo.exe "$on_bin"
test "$on_bin" -nt "$on_src"
sha256sum "$on_bin" > /tmp/nemo-row21-on-build.sha256
sha256sum "$on_src" "$on_bin" /tmp/nemo-row21-on-build.log
```

Run one patched arm and one fresh control from independently scrubbed copies.
There is deliberately no automatic retry.

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row21-run-xdg

on_bin=/tmp/nemo-row21-on.exe
off_bin=/tmp/nemo-row19-detwriter-off.exe
test "$(awk 'NR==1 {print $1}' /tmp/nemo-row21-on-build.sha256)" = \
  "$(sha256sum "$on_bin" | awk '{print $1}')"
test "$(sha256sum "$off_bin" | awk '{print $1}')" = \
  4f28b00205658781d7c2a6bbc35362a5a19eb8d57d245bad28a9fbe96ebfafae

ON=$(mktemp -d /tmp/RUN_ZDF21_COEFF_ON.XXXXXX)
OFF=$(mktemp -d /tmp/RUN_ZDF21_COEFF_OFF.XXXXXX)
for spec in "ON:$ON:$on_bin:202" "OFF:$OFF:$off_bin:197"; do
  IFS=: read -r tag run_dir binary expected_count <<< "$spec"
  cp -a /tmp/RUN_ZDF19_DETWRITER_OFF.q2tSGL/. "$run_dir/"
  find "$run_dir" -maxdepth 1 \( -type f -o -type l \) \( \
    -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o \
    -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o \
    -name 'ocean.output' -o -name 'run*.log' -o \
    -name '.nemo_binary_sha256' \) -delete
  ln -sfn "$binary" "$run_dir/nemo"
  sha256sum "$binary" > "$run_dir/.nemo_binary_sha256"
  if ! ( cd "$run_dir" && mpirun -np 1 ./nemo > run.attempt1.log 2>&1 ); then
    printf 'HOLD: %s failed; preserve %s\n' "$tag" "$run_dir" >&2
    exit 1
  fi
  grep -q '^STOP 0$' "$run_dir/run.attempt1.log"
  test "$(sha256sum "$run_dir/DINO_00005761_restart.nc" | awk '{print $1}')" = \
    33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c
  test "$(find "$run_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq \
    "$expected_count"
done
for name in zsqen_base zav_base avm_base avt_base dissl_postavn; do
  test "$(stat -c %s "$ON/tke_dump_${name}.bin")" -eq 2980224
done
printf '%s\n' "$ON" > /tmp/row21-on-dir.txt
printf '%s\n' "$OFF" > /tmp/row21-off-dir.txt
sha256sum "$ON"/tke_dump_{zsqen_base,zav_base,avm_base,avt_base,dissl_postavn}.bin
```

The strict write-only bracket uses the already-committed production comparator:

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row21-on-dir.txt)
OFF=$(cat /tmp/row21-off-dir.txt)
PYTHONPATH=scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import sys
from zdf_stream_bracket import (
    files_byte_identical, one_bit_file_control, stream_manifest,
)

on, off = map(Path, sys.argv[1:])
new = {
    "tke_dump_zsqen_base.bin", "tke_dump_zav_base.bin",
    "tke_dump_avm_base.bin", "tke_dump_avt_base.bin",
    "tke_dump_dissl_postavn.bin",
}
on_m = stream_manifest(on)
off_m = stream_manifest(off)

def inventory_is_exact(on_names, off_names):
    return (
        len(on_names) == 202
        and len(off_names) == 197
        and on_names - off_names == new
        and off_names - on_names == set()
    )

assert inventory_is_exact(set(on_m), set(off_m))
for name in sorted(off_m):
    assert files_byte_identical(on / name, off / name), name
plant = next(iter(sorted(off_m)))
assert one_bit_file_control(on / plant)
assert not inventory_is_exact(set(on_m) - {plant}, set(off_m))
print("row21 write-only bracket VERIFIED: 197/197 shared streams exact; controls fired")
PY
```

## Outcome (2026-08-29)

The held block completed cleanly. Binary SHA256 is
`c7b0de8a33040921bd1cd477f8ca81ff8654fbdfbc671d4a259f8b715f7a9f68`;
the ON run is `/tmp/RUN_ZDF21_COEFF_ON.cVaC2Q`. The strict bracket reports
197/197 shared streams exact with no exclusions and both controls firing.
The five-term scorer reports 0/9,920 failures, maximum error 0, exact unequal
0, and zero focus failures for every term. Row 21 is therefore VERIFIED.

The machine receipt is
`docs/ocean/fidelity/dino_zdf_row21_coeff_assembly_artifact.json`, SHA256
`84885e45ecc149082606c0b44b411271f40942a99497996b0d4b35e051b6a97a`.
The permanent held-block correction was needed again: use the repository venv
path rather than bare `python`.
