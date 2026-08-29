# Preregistration: ZDF row-30 `zbu` operand peel

Date: 2026-08-29. CPU-only matched day-180 state. Frozen before measurement.
This continues immediately after the verified `zau/zav` reciprocal fix and
uses existing dumps first; it requests no NEMO rebuild or run.

## Oracle order and time levels

The running standard-slope branch is `stpmlf.F90:219,234`:
`eos(ts,Nbb,rhd)` followed by `ldf_slp(kstp,rhd,rn2b,Nbb,Nnn)`. Thus `prd`
is BEFORE-level S-EOS density and `pn2` is the step-entry BEFORE `rn2b`
computed at `stpmlf.F90:205-207` on `Nnn` geometry. NEMO then writes:

```fortran
! ldfslp.F90:226-230
zdzr = zm1_g * (prd + 1._wp) * (pn2(jk) + pn2(jk+1)) \
     * (1._wp - 0.5_wp*tmask(jk+1))
! ldfslp.F90:244
zbu = 0.5_wp * (zdzr(i,j) + zdzr(i+1,j))
```

The frozen order is: `pn2(jk)` -> `pn2(jk+1)` -> their ordered sum ->
`prd+1` -> mask factor -> `zm1_g=-1/grav` product -> east-face pair sum ->
literal `0.5_wp` multiply. No later limiter or slope is inspected before all
eight stages pass.

## Existing operands, bars, and adjudication

The probe SHA-binds `tke_dump_rn2b.bin`, `eiv_dump_prd_arg.bin`,
`eiv_dump_zbu_pre.bin`, the row-30 binary/restart/source receipts, focus map,
clean repository HEAD, CPU backend, and fp64 policy. The first three arrays
are independently written NEMO operands/output; no value is inferred from the
final slope.

Each array stage uses the normalized whole-column bar `1.0e-15`, reports its
own wet-column denominator, and scores all four southern focus columns.
`zbu` face stages must be `0/9758`. Intermediate T/W stages must have zero
failed wet columns. The first nonzero stage is `DIVERGED`; later stages are
not dispositioned.

A mandatory discrimination compares two legoESM paths before assigning a
physics owner:

1. the historical `ldf_slp_per_element.build_state()` reconstruction used by
   the row-30 scorer; and
2. the actual `LatLonCGridOceanModel._tke_step_entry_n2_bundle()` production
   path, including its below-seafloor extrapolation and carried `rn2b` slot.

If the production bundle and all eight literal stages pass but the historical
capture fails, disposition is `DIVERGED-HARNESS`: repair the scorer/helper,
not production physics, then rerun row 30. If production `pn2(jk)` is first
red, it owns the production interval and the fix must be selectable, faithful
by default on only the two DINO NEMO cards, with every other card byte-exact.
Any later first red owns only that exact arithmetic interval.

Controls must fail: bar-scale perturbation, zonal roll, wet NaN, one-ULP exact
identity, swapping `jk/jk+1`, replacing `rn2b` by `rn2`, and using current
instead of BEFORE `prd`. Missing/changed dump SHA, wrong lane, backend, dtype,
focus map, or dirty tracked tree is fatal.

Rows 31--32 and climate remain ordered-blocked until row 30 is fully VERIFIED
or admissibly waived; there is no focus-only exception.

## Frozen continuation: post-bound `zbu` limiter

Date: 2026-08-29. Frozen before the post-bound substitution measurement. The
eight-stage peel above has repaired the historical scorer and the complete
row-30 scorer now verifies the raw `zbu` at `ldfslp.F90:244` (`0/9758`) but
first diverges after the stability bounds at `ldfslp.F90:248`
(`5496/9758`, all four focus columns). This continuation uses only the existing
`zau`, `zbu_pre`, and `zbu_post` dumps plus the SHA-bound NOW restart and raw
mesh; no oracle rebuild is authorized before this ladder is exhausted.

The running source is evaluated literally, in this order:

```fortran
! ldfslp.F90:247-248
zbu = MIN( zbu, -z1_slpmax * ABS( zau ),   &
     &          -7.e+3_wp / e3u(ji,jj,jk,Kmm) * ABS( zau ) )
```

The registered stages are: captured `zbu_pre`; captured `zau`; stored
`z1_slpmax`; raw-mesh `e3u_0`; `r3u(Kmm)` reconstructed from the NOW SSH using
`domqco.F90:166-169`; live `e3u(Kmm)=e3u_0*(1+r3u*umask)` from
`domzgr_substitute.h90:129`; slope cap; live-metric cap; inner ordered `MIN`;
outer ordered `MIN`; captured `zbu_post`. Each stage uses the whole wet-U
column bar `1.0e-15` and reports the four southern focus columns.

The discrimination is frozen as follows. If substituting the literal live
`e3u(Kmm)` makes the final `MIN` `0/9758`, while the production static partial-
cell face thickness reproduces the measured `5496/9758`, the owner is the
`e3u(Kmm)` operand at `ldfslp.F90:248`. The production repair is a selectable
live-QCO face-thickness construction, faithful by default only on the two DINO
NEMO cards; the historical static face metric remains the global default and
must be byte-identical on every other card. If the live substitution is still
red, the first red literal stage owns the row and no later stage is scored.

Controls must fail: replacing live `e3u` by static partial-cell thickness,
using BEFORE rather than NOW SSH in `r3u`, a zonal face roll, a one-ULP exact-
identity perturbation, wet NaN, and bar-scale perturbation. The raw-mesh,
restart, dump, source, binary, parent-artifact, focus-map, clean-HEAD, CPU, and
fp64 gates are fatal. Rows 31--32 and climate remain blocked until the row is
closed under the original no-focus-exception registration.

## Frozen continuation: first raw U slope

Date: 2026-08-29. Frozen after the live-face fix made both post-bound rows
exact and before inspecting any raw-slope substitution. The complete scorer's
new first red stage is `uslp_raw` at original `ldfslp.F90:269`: `8387/9758`,
with all four focus columns red. Existing `dump_nmln.bin`,
`eiv_dump_gdept.bin`, `zau`, `zbu_post`, and `uslp_raw` receipts are sufficient;
no rebuild is authorized first.

The registered peel follows `ldfslp.F90:251-271`: `iku` from the two NEMO
`nmln` columns -> integer `zfi` -> live `gdept(Kmm)` face average -> surface
`e3u(miku,Kmm)` subtraction -> `zdepu` -> carried anchor at `jk=iku` ->
interior `zau/(zbu-zeps)` -> ML blend -> U mask. The whole-domain bar remains
`1.0e-15`, with the four focus columns on every numeric stage.

The first discrimination is the surface-thickness operand. The current code
subtracts a T-column `_e3_top`; NEMO subtracts one half of the live U-face
`e3u(ji,jj,miku,Kmm)` inside its outer half multiply. If replacing only that
operand by the already-verified live `e3u_k[...,0]` yields `0/9758`, it owns
the row at `ldfslp.F90:261-264`. If not, the peel stops at the first earlier
red stage. Controls replace NEMO `nmln` by a one-level shift, use the old
T-column surface thickness, roll the face thickness, inject wet NaN, and add a
bar-scale perturbation; each must fail. All parent/dump/source/restart/mesh,
clean-HEAD, CPU, fp64, and focus-map gates remain fatal.

### Live-`gdept` sub-peel amendment (frozen before measurement)

The first raw-slope probe found `gdept(Kmm)` first red (`706/9758`; focus
`0/4`) before a live-face `zdepu` substitution reduced `uslp_raw` from
`8387/9758` to `14/9758`. The next registered discrimination therefore uses
the existing canonical `nemo_r3t_stretch(..., evaluation="nemo_reciprocal")`
with raw-mesh `gdept_0`, NOW SSH, and the bridged `ht_0`. CONFIRM is both
`gdept 0/9758` and the downstream live-face `uslp_raw 0/9758`; REFUTE is any
failure at either bar. A quotient-evaluation control and a one-ULP SSH control
must fail. If confirmed, the owner is the live T-depth evaluation boundary
(`domain.F90:158` stored reciprocal -> `domqco.F90:160` multiply ->
`domzgr_substitute.h90:139` live `gdept`) feeding `ldfslp.F90:261-264`.

Retraction, 2026-08-29, before the successful receipt: the preregistered
quotient control did not fail at `1.0e-15`, and perturbing SSH by one ULP was
rounded away before the live-depth product. They are not red-capable and are
withdrawn. The replacements are the measured-red production-Jacobian depth
construction and a direct one-ULP perturbation of one wet live-`gdept` value.
This changes no scored stage, bar, focus set, or ownership criterion.

## HELD: exact raw-U operand dumps (human executes; Codex does not run NEMO)

Existing dumps are exhausted with nine non-focus columns still red. The next
receipt writes `iku -> zfi -> e3u(miku) -> zdepu -> carried anchor -> interior
quotient -> ML term -> pre-mask blend`. The patch is write-only, initializes
every full halo buffer, and guards the irrelevant pre-anchor carry slots so
uninitialized values cannot enter a stream.

Build block (the preserved row-30 deterministic-writer source is the baseline):

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row30-uslp-build-xdg
cd /tmp/codex-zdf-sweep
base=/tmp/nemo-row30-manifest.OMUakO
test "$(sha256sum "$base/cfgs/DINO/MY_SRC/ldfslp.F90" | awk '{print $1}')" = \
  8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29
test "$(sha256sum scripts/validate/ocean_fidelity/dino_1226/nemo_row30_uslp_raw_operands.patch | awk '{print $1}')" = \
  b3435410f3ce2dcd7a77d197f233682691cc8609276c2ffcae07782d0507cca9
nemo_src=$(mktemp -d /tmp/nemo-row30-uslp.XXXXXX)
cp -a "$base"/. "$nemo_src"/
patch -p1 -d "$nemo_src" < \
  scripts/validate/ocean_fidelity/dino_1226/nemo_row30_uslp_raw_operands.patch
test "$(sha256sum "$nemo_src/cfgs/DINO/MY_SRC/ldfslp.F90" | awk '{print $1}')" = \
  2d59df4697b3d16f0ee9dc2b38ce929cca707d59ef43442dee3600c8d8b1f7ca
test "$(grep -c 'dino_dump_2d' "$nemo_src/cfgs/DINO/MY_SRC/ldfslp.F90")" -eq 30
for unit in $(seq 8912 8919); do
  test "$(grep -rE --include='*.F90' "UNIT[[:space:]]*=[[:space:]]*$unit" \
    "$nemo_src/cfgs/DINO/MY_SRC" | wc -l)" -eq 1
done
cd "$nemo_src"
./makenemo -m conda -n DINO -j 8 2>&1 | tee /tmp/nemo-row30-uslp-build.log
cp -p cfgs/DINO/MY_SRC/ldfslp.F90 /tmp/ldfslp-row30-uslp-on.F90
cp -p cfgs/DINO/BLD/bin/nemo.exe /tmp/nemo-row30-uslp-on.exe
test /tmp/nemo-row30-uslp-on.exe -nt /tmp/ldfslp-row30-uslp-on.F90
sha256sum /tmp/nemo-row30-uslp-on.exe > /tmp/nemo-row30-uslp-build.sha256
sha256sum /tmp/ldfslp-row30-uslp-on.F90 /tmp/nemo-row30-uslp-on.exe \
  /tmp/nemo-row30-uslp-build.log
printf 'SUBSTITUTE __MEASURED_ROW30_USLP_BINARY_SHA256__=%s\n' \
  "$(awk '{print $1}' /tmp/nemo-row30-uslp-build.sha256)"
```

One-step run block (replace the measured binary slot; no automatic retry):

```bash
set -euo pipefail
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp XDG_CACHE_HOME=/tmp/nemo-row30-uslp-run-xdg
BIN_SHA=__MEASURED_ROW30_USLP_BINARY_SHA256__
case "$BIN_SHA" in __MEASURED_*) echo 'replace binary SHA slot' >&2; exit 2;; esac
bin=/tmp/nemo-row30-uslp-on.exe
test "$(sha256sum "$bin" | awk '{print $1}')" = "$BIN_SHA"
ON=$(mktemp -d /tmp/RUN_ZDF30_USLP_ON.XXXXXX)
cp -a /tmp/RUN_ZDF19_DETWRITER_OFF.q2tSGL/. "$ON"/
find "$ON" -maxdepth 1 \( -type f -o -type l \) \( \
  -name '*.bin' -o -name 'DINO_00005761_restart.nc' -o \
  -name 'DINO_*_grid_*.nc' -o -name 'domain_cfg_out.nc' -o \
  -name 'ocean.output' -o -name 'run*.log' -o -name '.nemo_binary_sha256' \) -delete
ln -sfn "$bin" "$ON/nemo"
sha256sum "$bin" > "$ON/.nemo_binary_sha256"
( cd "$ON" && mpirun -np 1 ./nemo > run.attempt1.log 2>&1 )
grep -q '^STOP 0$' "$ON/run.attempt1.log"
test "$(sha256sum "$ON/DINO_00005761_restart.nc" | awk '{print $1}')" = \
  33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c
test "$(find "$ON" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 217
for name in iku zfi e3u_miku zdepu zuslp_hml_pre sint_u mlterm_u blend_u; do
  test "$(stat -c %s "$ON/eiv_dump_${name}.bin")" -eq 3273984
done
printf '%s\n' "$ON" > /tmp/row30-uslp-on-dir.txt
sha256sum "$ON"/eiv_dump_{iku,zfi,e3u_miku,zdepu,zuslp_hml_pre,sint_u,mlterm_u,blend_u}.bin
```

Strict bracket (uses the already-certified deterministic OFF arm).  This gate
persists a SHA-bindable JSON receipt; the scorer refuses to run without that
receipt and independently reconstructs its 197-stream manifest:

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row30-uslp-on-dir.txt)
OFF=/tmp/RUN_ZDF30_UV_OFF.75aJsW
PYTHONPATH=scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python - "$ON" "$OFF" <<'PY'
from pathlib import Path
import sys
import json
from zdf_stream_bracket import (files_byte_identical, manifest_sha256,
                                one_bit_file_control, sha256, stream_manifest)
on, off = map(Path, sys.argv[1:])
old = {f"eiv_dump_{n}.bin" for n in ("zgru_iik","zgru_iikm1","zau","zav",
 "zbu_pre","zbv_pre","zbu_post","zbv_post","uslp_raw","vslp_raw",
 "uslp_postshapiro","vslp_postshapiro")}
new = {f"eiv_dump_{n}.bin" for n in ("iku","zfi","e3u_miku","zdepu",
 "zuslp_hml_pre","sint_u","mlterm_u","blend_u")}
om, fm = stream_manifest(on), stream_manifest(off)
assert len(om) == 217 and len(fm) == 197
assert set(om)-set(fm) == old | new and not set(fm)-set(om)
for name in sorted(fm): assert files_byte_identical(on/name, off/name), name
one_bit = one_bit_file_control(on/sorted(fm)[0])
missing = not (set(om)-{next(iter(new))}-set(fm) == old | new)
assert one_bit and missing
names = ("nemo", "mesh_mask.nc", "DINO_00005760_restart.nc",
         "DINO_00005761_restart.nc", "namelist_cfg", "run.attempt1.log")
receipt = {
 "schema": "dino-zdf-row30-uslp-bracket-v1",
 "on_dir": str(on.resolve()), "off_dir": str(off.resolve()),
 "shared_count": len(fm), "shared_exact": True,
 "shared_manifest_sha256": manifest_sha256({n: om[n] for n in sorted(fm)}),
 "controls": {"one_bit_file": one_bit, "missing_stream": missing},
 "on_sha256": {n: sha256(on/n) for n in names},
 "off_sha256": {n: sha256(off/n) for n in names},
}
path = Path("/tmp/dino_zdf_row30_uslp_bracket_receipt.json")
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print("row30 raw-U write-only bracket VERIFIED: 197/197 shared streams exact; controls fired")
print(path, sha256(path))
PY
```

Measured-SHA score block (replace all ten marked slots, including the bracket
receipt SHA printed by the preceding block):

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
ON=$(cat /tmp/row30-uslp-on-dir.txt)
BIN_SHA=__MEASURED_ROW30_USLP_BINARY_SHA256__
IKU_SHA=__MEASURED_SHA256_EIV_DUMP_IKU_BIN__
ZFI_SHA=__MEASURED_SHA256_EIV_DUMP_ZFI_BIN__
E3U_SHA=__MEASURED_SHA256_EIV_DUMP_E3U_MIKU_BIN__
ZDEPU_SHA=__MEASURED_SHA256_EIV_DUMP_ZDEPU_BIN__
ANCHOR_SHA=__MEASURED_SHA256_EIV_DUMP_ZUSLP_HML_PRE_BIN__
SINT_SHA=__MEASURED_SHA256_EIV_DUMP_SINT_U_BIN__
MLTERM_SHA=__MEASURED_SHA256_EIV_DUMP_MLTERM_U_BIN__
BLEND_SHA=__MEASURED_SHA256_EIV_DUMP_BLEND_U_BIN__
BRACKET_SHA=__MEASURED_SHA256_BRACKET_RECEIPT_JSON__
for value in "$BIN_SHA" "$IKU_SHA" "$ZFI_SHA" "$E3U_SHA" "$ZDEPU_SHA" \
 "$ANCHOR_SHA" "$SINT_SHA" "$MLTERM_SHA" "$BLEND_SHA" "$BRACKET_SHA"; do
  case "$value" in __MEASURED_*) echo 'replace every SHA slot' >&2; exit 2;; esac
done
manifest=/tmp/row30-uslp-measured-dump-sha256.json
/home/dbalwada/legoESM/.venv/bin/python - "$manifest" \
 "$IKU_SHA" "$ZFI_SHA" "$E3U_SHA" "$ZDEPU_SHA" "$ANCHOR_SHA" \
 "$SINT_SHA" "$MLTERM_SHA" "$BLEND_SHA" <<'PY'
import json,sys
names=("iku","zfi","e3u_miku","zdepu","zuslp_hml_pre","sint_u","mlterm_u","blend_u")
json.dump({f"eiv_dump_{n}.bin":s for n,s in zip(names,sys.argv[2:])},open(sys.argv[1],"w"))
PY
lane_root=$(mktemp -d /tmp/row30-uslp-lane.XXXXXX)
ln -s "$ON" "$lane_root/RUN_SEQDUMP_D180_1R"
repo_sha=$(git --git-dir=/tmp/zdf-sweep-git.cJQ6wi/repo.git \
 --work-tree=/tmp/codex-zdf-sweep rev-parse HEAD)
set +e
DINO_ORACLE_ROOT="$lane_root" DINO_1226_LANE=d180 \
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
GIT_DIR=/tmp/zdf-sweep-git.cJQ6wi/repo.git GIT_WORK_TREE=/tmp/codex-zdf-sweep \
PYTHONPATH=packages/atmosphere:packages/core:packages/coupler:packages/ice:\
packages/land:packages/ml:packages/ocean:packages/tools:scripts/validate/ocean_fidelity/dino_1226 \
/home/dbalwada/legoESM/.venv/bin/python \
 scripts/validate/ocean_fidelity/dino_1226/zdf_row30_uslp_dump_score.py \
 --run-dir "$ON" --mld-maps /tmp/dino_mld_audit_codex/mld_maps.npz \
 --dump-sha-manifest "$manifest" --nemo-source /tmp/ldfslp-row30-uslp-on.F90 \
 --bracket-receipt /tmp/dino_zdf_row30_uslp_bracket_receipt.json \
 --expected-bracket-sha "$BRACKET_SHA" \
 --nemo-binary /tmp/nemo-row30-uslp-on.exe \
 --expected-source-sha 2d59df4697b3d16f0ee9dc2b38ce929cca707d59ef43442dee3600c8d8b1f7ca \
 --expected-binary-sha "$BIN_SHA" --expected-repo-sha "$repo_sha" \
 --output /tmp/dino_zdf_row30_uslp_held_artifact.json
rc=$?; set -e
test "$rc" -eq 0 -o "$rc" -eq 30
sha256sum /tmp/dino_zdf_row30_uslp_held_artifact.json
exit "$rc"
```

The held scorer also hard-gates the production-state inputs used by
`ldf_slp_per_element.build_state()`: `mesh_mask.nc` SHA
`3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622`,
input restart SHA
`0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e`,
and `namelist_cfg` SHA
`55f17d2344e5aaa58f7c6ef23e7d5dfebd9c351888c6499d5d312131b8515355`.
The bracket receipt additionally binds both binaries, both input/output
restarts, both meshes and namelists, and both run logs.  Red controls now live
on the new ladder itself: wrong `iku`, vertically shifted `zfi`, zonally
rolled and wrong-level `e3u(miku)`, zonally rolled `zdepu`, and a wet NaN.
The blend reconstruction uses the literal Fortran multiply/divide/add
association rather than a post-hoc `where` selection.

## Frozen continuation: `zdepu` ownership

Date: 2026-08-29. Frozen after the held scorer first reported `zdepu`
**9,758/9,758**, `focus_fail=4`, and before substituting any operand. The
parent artifact SHA is
`5c01587801a5ebc10f1522e33e425e9f81b53c60e465a981ccaf469e1c4f1c74`;
the exact bracket receipt SHA is
`6eea10e2b3034ba81999c55c5dd2b37891f6e80cd48d56b43afe2b0cca45afbe`.

The executed NEMO expression is:

```fortran
! ldfslp.F90:298-301 (instrumented source; original :260-263)
zdepu = 0.5_wp * ( ( gdept(ji,jj,jk,Kmm) + gdept(ji+1,jj,jk,Kmm) ) &
   &              - 2 * MAX( risfdep(ji,jj), risfdep(ji+1,jj) )    &
   &              - e3u(ji,jj,miku(ji,jj),Kmm) )
```

DINO has no ice shelf, so the registered order is: live `gdept(Kmm)` pair ->
ordered add -> subtract independently dumped live `e3u(miku,Kmm)` -> multiply
by literal `0.5_wp`. The probe first assembles that expression from the three
independent NEMO dumps. It then substitutes the production-computable
operands: raw `gdept_0` times NOW `1 + ssh*r1_ht_0`, with the stored reciprocal
boundary, and the already verified live-QCO U-face thickness. Each stage uses
the `1.0e-15` wet-U column bar over 9,758 columns and reports all four southern
focus columns.

CONFIRM is dumped-operand literal **0/9,758**, production-computable literal
**0/9,758**, and `focus_fail=0` at both. Then the owner is the combined depth
evaluation/association boundary: the current Jacobian-derived `gdept` plus
post-average T-column subtraction is replaced by live
`gdept_0*(1+ssh*r1_ht_0)` and literal face-thickness subtraction at
`ldfslp.F90:298-301`. REFUTE is any failure at either literal stage; the first
red operand remains open and no fix lands.

The production design is selectable. `legacy_jacobian_t_surface` remains the
global/default byte-identical path. Only `nemo_dino_kamm` and
`nemo_dino_kamm_mlf` select `nemo_qco_live_literal`, which requires NOW SSH,
the local bathymetry, raw `gdept_0`, and live `e3u/e3v`; it uses the same live
stretch for `zhmlpt`, `zdepu/zdepv`, and `zck`. Unknown values fail closed.
Red controls are: subtracting `e3u` after rather than inside the half multiply,
using the current Jacobian depth, rolling the U-face thickness zonally, one
wet NaN, and a one-ULP exact-identity perturbation. JIT and finite-AD tests plus
default-vs-explicit legacy `assert_array_equal` and live-helper reachability
guards for every unchanged card are required before promotion.

The ordered stop remains row 30. Rows 31--32 and climate are blocked until the
complete raw U/V and post-Shapiro composite passes; focus membership creates
no exception.

### Compiled-lowering amendment

Frozen before measuring the compiled arm. A synthetic unit fixture showed a
one-ULP eager-versus-JIT difference in the face-depth expression, so eager
agreement alone is insufficient. The production-computable expression from
NOW SSH through stored-reciprocal stretch, raw `gdept_0`, face add, live
`e3u(miku)`, subtract, and half multiply is now also evaluated by `jax.jit`.
CONFIRM additionally requires compiled **0/9,758**, `focus_fail=0`; any compiled
failure REFUTES the proposed implementation even when the eager arm passes.
The eager/JIT synthetic unit test is exact only after the production lowering
is made stable; tolerance-based promotion is forbidden.

Retraction, 2026-08-29, after the compiled oracle arm and before production
promotion: exact eager-versus-JIT identity on an unrelated synthetic fixture
is not the registered oracle statistic and differs by at most two ULP under XLA's
vector lowering. The production compiled arm itself is **0/9,758** at the
frozen `1.0e-15` per-column oracle bar with `focus_fail=0`; that is the numeric
gate and no tolerance was changed. The unit receipt therefore bounds the
synthetic eager/JIT difference at two ULP and retains the red old-association
control. The scorer's compiled oracle bar remains mandatory.

## Frozen continuation: row-30 raw/post-Shapiro composite close

Date: 2026-08-29. Frozen after the repaired held ladder printed eight
consecutive zero-failure stages through `blend_u`; parent artifact SHA256
`8be5ec24beed64967b1a31646df516a40a1b6647f4e5c334d00852cb3e4f435e`.
No new NEMO dump is required. The existing run already carries the source-
ordered composites from `ldfslp.F90:306-307,320-333`.

The registered order is `uslp_raw` -> `vslp_raw` -> `uslp_postshapiro` ->
`vslp_postshapiro`. U stages must be **0/9,758** and V stages **0/9,868** at
the unchanged `1.0e-15` per-column bar; every stage reports the four southern
focus columns and requires `focus_fail=0`. The scorer SHA-binds the parent,
197/197 bracket receipt, mesh, input/output restarts, source, focus map, and
all four composite dumps, and requires clean HEAD/CPU/fp64. Frame capture must
be numerically inert. Each direction has expected-vs-expected exact baseline,
bar-scale, zonal-roll, and wet-NaN controls. The first nonzero stage is
`DIVERGED` and later stages are not dispositioned. If all four pass, row 30 is
VERIFIED and the ordered sweep advances to row 31.
