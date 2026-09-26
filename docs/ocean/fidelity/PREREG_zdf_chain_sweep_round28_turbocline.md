# Preregistration: ZDF row 28 turbocline scan

Date: 2026-08-29. Rows 19--23 and 25--27 are VERIFIED; row 24 is WAIVED
because the resolved river-mouth branch is inactive. Row 28 is the first
unresolved row. Rows 29--32 cannot be promoted across it.

## Oracle operation and direct slot

The active DINO oracle evaluates, in order:

```fortran
! cfgs/DINO/WORK/zdfmxl.F90:145-152
imld(:,:) = mbkt(T2D(0)) + 1
DO_3DS( 0, 0, 0, 0, jpkm1, nlb10, -1 )
   IF( avt(ji,jj,jk) < avt_c * wmask(ji,jj,jk) ) imld(ji,jj) = jk
END_3D
iik = imld(ji,jj)
hmld(ji,jj) = gdepw(ji,jj,iik,Kmm) * ssmask(ji,jj)
```

Here `avt_c=5e-4 m2 s-1` and `nlb10=2` on the registered mesh. The existing
composed-`avt` and frozen live-geometry operands suffice, but NEMO did not
write the resulting `imld/hmld`. The patch adds one direct
`zdf_dump_hmld_turb.bin` stream after `zdf_mxl_turb` has run. Its registered
size is 90,944 bytes (`56*203` binary64 values, with deterministic zero halos).
The scorer removes the two-cell halo before comparison. The integer `imld` is recovered
exactly by inverting the strictly monotone live `gdepw` ladder; promotion
requires exactly one matching level at every wet column. This keeps the patch
to one writer while retaining both the exact index and numeric depth gates.

The scorer calls the actual production coefficient composer for `avt`, uses
its frozen step-entry `gdepw_Kmm`, and performs the bottom-to-top loop as an
ordered JAX `fori_loop`. It does not reconstruct `avt` from NEMO fields.

## Frozen bars and controls

- Exact `imld` equality at every one of 9,920 wet columns.
- Absolute per-column `hmld` error no larger than `1e-15 m`.
- The four southern focus columns use the same gates.
- CONFIRM: zero index failures, zero depth failures, no nonfinite wet values,
  and zero focus failures. Otherwise row 28 is DIVERGED and the walk stops.
- Red controls: a depth value raised above the bar, a swap of the wet minimum-
  and maximum-depth columns, wet NaN, and one-level index change must fail.
  The swap fails closed if the field lacks two values separated by the bar.
  A one-bit shared-stream change and a missing shared stream must fail the
  production bracket predicates.

## SHA-gated build and run handoff

Permanent held-block corrections apply: use `grep -rE --include` because `rg`
is absent on the execution host; use the repository virtual environment for
every Python invocation; and use `makenemo -n DINO` for the existing config.

Build from the exact successful row-21 source tree:

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row28-build-xdg

repo=/tmp/codex-zdf-sweep
baseline=/tmp/nemo-row21-coeff.9gnuvq
patch_file="$repo/scripts/validate/ocean_fidelity/dino_1226/nemo_row28_turbocline.patch"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  cb3e408e5b7c909dafdc3f1ac94f6ceb20c73f4f17005696426bd8657f0f9f06
test "$(sha256sum "$baseline/cfgs/DINO/MY_SRC/ldftra.F90" | awk '{print $1}')" = \
  e7eef4f8b7510a66051afa429e589507ffd89c44be54ef4964d4360113eaa0e6
test "$(sha256sum /tmp/nemo-row21-on-source-manifest.sha256 | awk '{print $1}')" = \
  82e54ec72d8a2e6f304e285896b899408399ade0480aaa604a70f8892cc84a95

nemo_src=$(mktemp -d /tmp/nemo-row28-turb.XXXXXX)
cp -a "$baseline"/. "$nemo_src"/
patch --dry-run --fuzz=0 -d "$nemo_src" -p1 < "$patch_file"
patch --fuzz=0 -d "$nemo_src" -p1 < "$patch_file"
test "$(sha256sum "$nemo_src/cfgs/DINO/MY_SRC/ldftra.F90" | awk '{print $1}')" = \
  799be057d7c9afc0f06e6f055a47b8fb111d2cedb6b2c746cf1babd91cc51c95
(
  cd "$nemo_src"
  sha256sum cfgs/DINO/MY_SRC/*.F90 | sort > /tmp/nemo-row28-on-source-manifest.sha256
)
test "$(sha256sum /tmp/nemo-row28-on-source-manifest.sha256 | awk '{print $1}')" = \
  fef6296747d7b1d012798e1207f4c986a62d272d799a06da4dc1bb851a08d926
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_2d' \
  "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 151
test "$(grep -rE --include='*.F90' 'CALL[[:space:]]+dino_dump_3d' \
  "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 5
test "$(grep -rE --include='*.F90' \
  'WRITE\([^)]*\).*ji[[:space:]]*=[[:space:]]*1[[:space:]]*,[[:space:]]*jpi' \
  "$nemo_src/cfgs/DINO/MY_SRC" | grep -v dino_dump_zero.F90 | wc -l)" -eq 0
test "$(grep -c 'zdf_dump_hmld_turb.bin' \
  "$nemo_src/cfgs/DINO/MY_SRC/ldftra.F90")" -eq 1

cd "$nemo_src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row28-on-build.log
on_src=/tmp/ldftra-row28-on.F90
on_bin=/tmp/nemo-row28-on.exe
cp -p cfgs/DINO/MY_SRC/ldftra.F90 "$on_src"
cp -p cfgs/DINO/BLD/bin/nemo.exe "$on_bin"
test "$on_bin" -nt "$on_src"
sha256sum "$on_bin" > /tmp/nemo-row28-on-build.sha256
sha256sum "$on_src" "$on_bin" /tmp/nemo-row28-on-build.log
printf '%s\n' "$nemo_src" > /tmp/nemo-row28-source-dir.txt
```

Run one patched arm and a fresh row-21 control. There is deliberately no
automatic retry:

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row28-run-xdg

on_bin=/tmp/nemo-row28-on.exe
off_bin=/tmp/nemo-row21-on.exe
test "$(awk 'NR==1 {print $1}' /tmp/nemo-row28-on-build.sha256)" = \
  "$(sha256sum "$on_bin" | awk '{print $1}')"
test "$(sha256sum "$off_bin" | awk '{print $1}')" = \
  c7b0de8a33040921bd1cd477f8ca81ff8654fbdfbc671d4a259f8b715f7a9f68

ON=$(mktemp -d /tmp/RUN_ZDF28_TURB_ON.XXXXXX)
OFF=$(mktemp -d /tmp/RUN_ZDF28_TURB_OFF.XXXXXX)
for spec in "ON:$ON:$on_bin:203" "OFF:$OFF:$off_bin:202"; do
  IFS=: read -r tag run_dir binary expected_count <<< "$spec"
  cp -a /tmp/RUN_ZDF21_COEFF_ON.cVaC2Q/. "$run_dir"/
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
test "$(stat -c %s "$ON/zdf_dump_hmld_turb.bin")" -eq 90944
printf '%s\n' "$ON" > /tmp/row28-on-dir.txt
printf '%s\n' "$OFF" > /tmp/row28-off-dir.txt
sha256sum "$ON/zdf_dump_hmld_turb.bin"
```

Run the exact bracket and scorer only after both arms finish. Replace
`DUMP_SHA_FROM_THE_LINE_ABOVE` with the printed dump SHA:

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row28-on-dir.txt)
OFF=$(cat /tmp/row28-off-dir.txt)
on_bin=/tmp/nemo-row28-on.exe
off_bin=/tmp/nemo-row21-on.exe
on_sha=$(sha256sum "$on_bin" | awk '{print $1}')
build_receipt_sha=$(sha256sum /tmp/nemo-row28-on-build.sha256 | awk '{print $1}')
repo_sha=$(git --git-dir=/tmp/zdf-sweep-git.cJQ6wi/repo.git \
  --work-tree=/tmp/codex-zdf-sweep rev-parse HEAD)
lane_root=/tmp/row28-lane-root
mkdir -p "$lane_root"
ln -sfn "$ON" "$lane_root/RUN_SEQDUMP_D180_1R"

DINO_ORACLE_ROOT="$lane_root" DINO_1226_LANE=d180 \
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
LEGOESM_NEMO_E3T=both \
GIT_DIR=/tmp/zdf-sweep-git.cJQ6wi/repo.git \
GIT_WORK_TREE=/tmp/codex-zdf-sweep \
PYTHONPATH=packages/atmosphere:packages/core:packages/coupler:packages/ice:\
packages/land:packages/ml:packages/ocean:packages/tools:\
scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/zdf_row28_turbocline.py \
  --run-dir "$ON" --bracket-dir "$OFF" \
  --tail-artifact docs/ocean/fidelity/dino_zdf_chain_tail_existing_artifact.json \
  --mld-maps /tmp/dino_mld_audit_codex/mld_maps.npz \
  --nemo-source /tmp/ldftra-row28-on.F90 \
  --bracket-nemo-source /tmp/nemo-row21-coeff.9gnuvq/cfgs/DINO/MY_SRC/ldftra.F90 \
  --oracle-zdfmxl-source /tmp/nemo-row21-coeff.9gnuvq/cfgs/DINO/WORK/zdfmxl.F90 \
  --nemo-binary "$on_bin" --bracket-nemo-binary "$off_bin" \
  --build-binary-receipt /tmp/nemo-row28-on-build.sha256 \
  --on-source-manifest /tmp/nemo-row28-on-source-manifest.sha256 \
  --off-source-manifest /tmp/nemo-row21-on-source-manifest.sha256 \
  --instrument-patch scripts/validate/ocean_fidelity/dino_1226/nemo_row28_turbocline.patch \
  --expected-repo-sha "$repo_sha" \
  --expected-dump-sha DUMP_SHA_FROM_THE_LINE_ABOVE \
  --expected-source-sha 799be057d7c9afc0f06e6f055a47b8fb111d2cedb6b2c746cf1babd91cc51c95 \
  --expected-bracket-source-sha e7eef4f8b7510a66051afa429e589507ffd89c44be54ef4964d4360113eaa0e6 \
  --expected-oracle-source-sha 3a9caebc8599e6e432bbd33d23bf5ef63f7a291cf093ad077e884162c7a5f732 \
  --expected-binary-sha "$on_sha" \
  --expected-bracket-binary-sha c7b0de8a33040921bd1cd477f8ca81ff8654fbdfbc671d4a259f8b715f7a9f68 \
  --expected-build-receipt-sha "$build_receipt_sha" \
  --expected-on-manifest-sha fef6296747d7b1d012798e1207f4c986a62d272d799a06da4dc1bb851a08d926 \
  --expected-off-manifest-sha 82e54ec72d8a2e6f304e285896b899408399ade0480aaa604a70f8892cc84a95 \
  --expected-patch-sha cb3e408e5b7c909dafdc3f1ac94f6ceb20c73f4f17005696426bd8657f0f9f06 \
  --output /tmp/dino_zdf_row28_turbocline_artifact.json
```

The patch compile-check passed without running NEMO. Its log SHA256 is
`5e2c724d7fd1fa909ea026c24df5cd9a8cca79120186196e718b41c736639913`.
The compile emitted a nonfatal broken `conda-anaconda-tos` entry-point warning,
then completed successfully; this does not alter the source or binary gates.

Until the direct bracket and scorer pass, row 28 remains
`UNMEASURED-NEEDS-DUMP`, rows 29--32 remain ordered-blocked, and climate arms
are not authorized.
