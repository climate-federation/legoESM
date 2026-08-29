# DINO day-180 ZDF execution-chain sweep: first divergence

Date: 2026-08-28.  Lane: CPU-only, one-rank matched day-180 state
(`RUN_SEQDUMP_D180_1R`, `kt=5761`).

## Round-15 hold: row-18 operands instrumented on paper; existing tail scored

Home storage was quota-blocked, so this round made **no NEMO build or run**.
The ordered frontier remains row 17 and climate arms remain unauthorized.
The committed write-only patch
`scripts/validate/ocean_fidelity/dino_1226/nemo_row18_gdepw_htau.patch`
(SHA256 `e797e5d9ba9cbc50401a7542c53182f41d7cd2ab3b148b914ccb01c15cb28fb7`)
applies cleanly to the active instrument source SHA256
`5eec1700605ff54dff13b76913b46f67c8cbdc7b526d9901c33c61a1196d67a5`.
It captures `gdepw(ji,jj,jk,Kmm)` and `htau(ji,jj)` separately at the live
read site and never reads the observation arrays back into the calculation.
The unchanged oracle expression is:

```fortran
etau_arg_dump(ji,jj,jk) = -gdepw(ji,jj,jk,Kmm) / htau(ji,jj)
```

Source: active DINO instrument `MY_SRC/zdftke.F90:648`; after applying the
patch the two new copies are at `:658-659` and the unchanged expression moves
to `:660`. The preregistered one-at-a-time substitution, ULP controls,
provenance stamps, and dump-disabled/enabled bracket are in
`PREREG_zdf_chain_sweep_round18_operands.md`. Rebuild and measurement are on
explicit hold until home writes recover.

The existing-dump tail was scored meanwhile. These are deliberately labeled
`PROVISIONAL-DOWNSTREAM`: they target later work but do not promote any row
past the unresolved row 18.

| Row | Operation | Qualified disposition | Whole-domain column result | Southern focus |
|---:|---|---|---|---|
| 19 | raw buoyancy length | `UNMEASURED-NEEDS-DUMP` | existing length slots are post-scan | needs raw slot |
| 20 | `nn_mxl=3` scans, row-19+20 composite | `DIVERGED` | `zmxlm` and `zmxld`: 9,920/9,920 fail; maxima `2.042006` and `2.073254` | 4/4 fail both |
| 21 | base `avm/avt/dissl` | `UNMEASURED-NEEDS-DUMP` | isolated base `avm` is exact, 0/9,920; pre-Prandtl `avt` and post-overwrite `dissl` lack slots | `avm` 4/4 exact |
| 22 | inverse-Prandtl `avt` | `VERIFIED` | 0/9,920; max `2.367838e-16` | 4/4 pass |
| 23 | closure coefficient copy | `UNMEASURED-ORACLE-SELFCHECK` | NEMO stable-subset reconstruction is exact, but legoESM composition was not invoked | preview only |
| 24 | river-mouth enhancement | `WAIVED` | `ln_rnf=F`, `ln_rnf_mouth=F` | inactive |
| 25 | EVD tracer overwrite | `UNMEASURED-ORACLE-SELFCHECK` | NEMO reconstruction is exact over 39,293 fired wet elements; legoESM EVD was not invoked | preview only |
| 26 | EVD momentum overwrite | `UNMEASURED-ORACLE-SELFCHECK` | NEMO reconstruction is exact over 39,293 fired wet elements; legoESM EVD was not invoked | preview only |
| 27 | `avs=avt`; optional enhancements | `VERIFIED` | exact source/object identity, 0/9,920; disabled DDM/SWM/IWM enhancements separately waived | 4/4 pass |
| 28 | composed-`avt` turbocline | `UNMEASURED-NEEDS-DUMP` | no same-step `imld/hmld` slot | needs slot |
| 29 | `avm` lateral boundary update | `UNMEASURED-ORACLE-SELFCHECK` | NEMO interior reconstruction exact; legoESM LBC path was not invoked | preview only |
| 30 | `ldf_slp` | `DIVERGED` | `uslp/vslp/wslpi/wslpj` fail 9,306/9,412/9,462/9,462 columns | 4/4 fail each |
| 31 | momentum implicit application | `UNMEASURED-EXISTING-BRACKET` | existing Kbb + stage-6 Krhs + stage-7 barotropic + stage-8 fields suffice; exact reconstruction is blocked by composed-`avm` uncertainty | no new dump |
| 32 | tracer implicit application | `UNMEASURED-EXISTING-BRACKET` | existing Kbb/stage-23/stage-21 fields suffice; exact reconstruction is blocked by row-30 K33 divergence | no new dump |

Rows 31--32 are the explicit large deferred stop for this held-instrumentation
round, not dump requests. Row 31 will reconstruct
`Naa_A=(Kbb+rDt*stage6_Krhs)*mask`, subtract stage-7 barotropic `Naa`, add the
dumped level-1 stress deposit, substitute `dump_avm`, and score the production
momentum matrix/drag call against stage 8. Row 32 will reconstruct the NEMO
z-star volume input
`(e3t_Kbb*T_Kbb+rDt*e3t_Kmm*stage23_Krhs)/e3t_Kaa`, assemble K33 from
`dump_avt` plus the existing slope/coefficient dumps, and score the production
T/S pair solve against stage 21. Both designs require per-column focus scores,
wrong-slot controls, and one-cell-roll controls. They are a substantial new
committed substitution probe, but require no new oracle output.

Row 20 was isolated by feeding the real production mixing-length routine
NEMO's dumped post-row-18 `en`, current `rn2`, and the captured production
geometry/configuration. Because the only NEMO length dumps are after both the
raw buoyancy calculation and the `nn_mxl=3` scans, the divergence cannot yet
be assigned between rows 19 and 20; a raw pre-limit length slot is required.
The maximum absolute misses are 290.505 m (`zmxlm`) and 476.846 m (`zmxld`),
so this is not a roundoff-only preview.

The NEMO-only previews for rows 23 and 25--26 close exactly when NEMO's source-order EVD mask
`MIN(rn2,rn2b) <= -1e-12` is used. On Fortran `jk=2..jpkm1`, copying the
closure coefficients at stable points and setting both coefficients to
`100*wmask` at fired points reproduces `dump_avt.bin` and `dump_avm.bin`
bit-for-bit. These are oracle self-consistency checks, not legoESM-vs-NEMO
measurements; the earlier `VERIFIED` labels are loudly retracted. Row 29 is
qualified the same way.

Row 30's earliest available failing operand is the dumped `prd` argument:
9,920/9,920 columns fail the `1e-15` bar, maximum column error
`2.521742e-08`, even though correlation prints 1.0. NEMO first consumes it in
the bottom horizontal gradients:

```fortran
zgru(ji,jj,iikm1) = umask(ji,jj,jpkm1) * ( prd(ji+1,jj,jpkm1) - prd(ji,jj,jpkm1) )
zgrv(ji,jj,iikm1) = vmask(ji,jj,jpkm1) * ( prd(ji,jj+1,jpkm1) - prd(ji,jj,jpkm1) )
```

Source: DINO `MY_SRC/ldfslp.F90:202-203`. This is a provisional later-row
target, not permission to jump over rows 18--20. Its next discriminator is a
literal SEOS/`prd` association check using the existing direct slot.

Machine receipt:
`docs/ocean/fidelity/dino_zdf_chain_tail_existing_artifact.json`, SHA256
`677ff32559b0855e41cecd9ecd0f5485f2e26b1e0e191ceac3c24184b1da5dc6`,
stamped probe commit `f31f09ccd26efcc04c2b9739484f26d2fadcd8a9`. All four
planted perturbation, nonfinite, and one-cell-roll control families fired.
The receipt also stamps every quoted active NEMO source. Review correction:
the former row-22 `pdlr` identity field compared a dump with itself and has
been removed; only the real production `avt` comparison supports row 22.

### Held rebuild/run commands

Run these only after home writes recover. They keep the build, run outputs,
and caches under `/tmp`; the donor restart is read through its restored
symlink and is not modified.

```bash
source /home/dbalwada/miniconda3/etc/profile.d/conda.sh
conda activate nemo-build
export TMPDIR=/tmp
export XDG_CACHE_HOME=/tmp/nemo-row18-xdg

nemo_src=$(mktemp -d /tmp/nemo-row18-operands.XXXXXX)
cp -a /tmp/nemo-tau-slot-eaef2a1e1/. "$nemo_src/"
cd "$nemo_src"
test "$(sha256sum cfgs/DINO/MY_SRC/zdftke.F90 | awk '{print $1}')" = \
  5eec1700605ff54dff13b76913b46f67c8cbdc7b526d9901c33c61a1196d67a5
patch --dry-run -p1 < \
  /tmp/codex-zdf-sweep/scripts/validate/ocean_fidelity/dino_1226/nemo_row18_gdepw_htau.patch
patch -p1 < \
  /tmp/codex-zdf-sweep/scripts/validate/ocean_fidelity/dino_1226/nemo_row18_gdepw_htau.patch
./makenemo -m conda -r DINO -j 8
sha256sum cfgs/DINO/MY_SRC/zdftke.F90 cfgs/DINO/BLD/bin/nemo.exe

run_off=$(mktemp -d /tmp/RUN_ZDF18_OPERANDS_OFF.XXXXXX)
run_on=$(mktemp -d /tmp/RUN_ZDF18_OPERANDS_ON.XXXXXX)
cp -a /tmp/RUN_ZDF18_ETAUARG.g6lJkJ/. "$run_off/"
cp -a /tmp/RUN_ZDF18_ETAUARG.g6lJkJ/. "$run_on/"
ln -sfn /tmp/nemo-tau-slot-eaef2a1e1/cfgs/DINO/BLD/bin/nemo.exe "$run_off/nemo"
ln -sfn "$nemo_src/cfgs/DINO/BLD/bin/nemo.exe" "$run_on/nemo"
( cd "$run_off" && mpirun -np 1 ./nemo > run.log 2>&1 )
( cd "$run_on" && mpirun -np 1 ./nemo > run.log 2>&1 )
test "$(stat -c %s "$run_on/tke_dump_etau_gdepw.bin")" -eq 2980224
test "$(stat -c %s "$run_on/tke_dump_etau_htau.bin")" -eq 2980224
sha256sum "$run_off/DINO_00005761_restart.nc" \
  "$run_on/DINO_00005761_restart.nc" \
  "$run_on/tke_dump_etau_gdepw.bin" \
  "$run_on/tke_dump_etau_htau.bin"

# Bracket-stream proof: every shared raw physics stream is byte-identical.
# Logs/timing/static donor files are deliberately outside the physics receipt.
find "$run_off" -maxdepth 1 -type f -name '*.bin' -printf '%P\n' \
  | sort > /tmp/row18-off-bin.files
find "$run_on" -maxdepth 1 -type f -name '*.bin' \
  ! -name tke_dump_etau_gdepw.bin ! -name tke_dump_etau_htau.bin \
  -printf '%P\n' | sort > /tmp/row18-on-shared-bin.files
cmp /tmp/row18-off-bin.files /tmp/row18-on-shared-bin.files
while IFS= read -r rel; do
  cmp "$run_off/$rel" "$run_on/$rel"
done < /tmp/row18-off-bin.files
sha256sum /tmp/row18-off-bin.files /tmp/row18-on-shared-bin.files

cd /tmp/codex-zdf-sweep
DINO_1226_LANE=d180 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=1 PYTHONPATH=packages/core:packages/ocean python - \
  "$run_on" "$run_off" <<'PY'
from pathlib import Path
import sys
from scripts.validate.ocean_fidelity.dino_1226.zdf_chain_sweep import (
    restart_numeric_identity,
)
on, off = map(Path, sys.argv[1:])
on_files = {p.name for p in on.glob("DINO_*.nc")}
off_files = {p.name for p in off.glob("DINO_*.nc")}
assert on_files == off_files, (sorted(on_files), sorted(off_files))
for name in sorted(on_files):
    receipt = restart_numeric_identity(on / name, off / name)
    print(name, receipt)
    assert receipt["pass"]
PY
```

The on-run must then be fed to the row-18 operand scorer registered above;
the operand ULP/roll controls and the one-at-a-time substitutions are not
replaced by the restart bracket.

## Round-14 result: scalar-LIBM waiver refuted; literal EXP lands; ordered stop remains at row 18

The preregistered Rule-1b discriminator used the 59 columns where JAX/XLA
`exp` missed the dumped oracle operand.  On this host the linked runtime is
glibc 2.34.  Scalar `exp` loaded from `libm.so.6` by `ctypes` did **not** own
the difference: the ownership residual failed **59/59** columns at the
`1e-16` bar (maximum `1.421263e-15`), and the planted +1-ULP argument control
fired.  Row 18 is therefore not `WAIVED-LIBM`.

The oracle's active loop imports glibc's two-lane
`_ZGVbN2v_exp@GLIBC_2.22`.  The implemented
`tke_etau_exponential_evaluation="nemo_literal"` reproduces that ordinary-
range glibc-2.34 arithmetic in pure JAX: exact 1024-entry table bits,
source-ordered range reduction and polynomial, and explicit binary64 rounding
points with identity JVPs.  It defaults only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; `jax_expression` remains byte-identical everywhere else
and is the legacy opt-in on those two cards.  The literal helper is bit-exact
on all 328,072 ordinary-range wet dumped arguments; its exceptional fallback
differs only for 430 subnormal results, maximum absolute `9.855e-314`.

The ordered rerun is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1--15 | `eos_rab` through all TKE RHS terms | `VERIFIED` | unchanged; rows 2, 4, 8, 10, 12--15 remain 0/9,920 | 4/4 pass |
| 16 | wave surface boundary | `WAIVED` | resolved DINO has `ln_wave=F` | inactive |
| 17 | literal TKE solve | `VERIFIED` | 0/9,920; max 0 | 4/4 exact |
| 18 | `nn_etau=1` penetrating TKE addition | **`DIVERGED`** | **21/9,920; max `2.277967e-15`** | **4/4 exact** |
| 19--32 | mixing length through EVD, assembly, `ldf_slp`, and both implicit applications | `UNMEASURED` | ordered stop at row 18 | ordered stop |

### Remaining row-18 operand

The EXP fix itself is verified: evaluating the literal helper on NEMO's
dumped exponent argument scores **0/9,920** (maximum `6.660194e-313`).  The
same helper on legoESM's production argument fails **217/9,920**, maximum
`1.500624e-15`.  Most decisively, substituting NEMO's dumped EXP into the
literal left-associated factor chain at `zdftke.F90:590-591` makes both the
increment and final `en` output **0/9,920** (final maximum exactly zero).

Thus the remaining first operand is the production input to EXP,
`-gdepw(Kmm)/htau`.  The current dump does not separate its numerator from
its denominator, so ownership is **not yet assigned** to either one.  The
candidate denominator is constructed at:

```fortran
htau(:,:) = MAX( 0.5_wp, MIN( 30._wp, 45._wp * &
   ABS( SIN( rpi/180._wp * gphit(A2D(0)) ) ) ) ) ! zdftke.F90:1005
```

The live `gdepw(Kmm)` construction passed preceding row-level bars, but that
does not prove bit identity at this more sensitive use.  Both a literal
`(rpi/180)*gphit` association and raw grid radians stay within the coarse
argument bar but do not close the downstream literal EXP.  Although the
oracle imports `_ZGVbN2v_sin@GLIBC_2.22`, assigning the residual to vector
sine now would be premature.  The registered next design is write-only direct
`gdepw(Kmm)` and `htau` streams immediately before line 590, with the usual
restart bit-identity bracket, then one-operand-at-a-time substitutions into
the dumped division.  Only the first failing construction gets a production
option; if it is `htau`, that option will reproduce source-ordered line 1005
and the linked sine lowering in pure JAX.  This instrumentation plus a second
vector transcendental port if indicated is too large for this round, so
ordered discipline does not promote row 19.

**CLIMATE ARMS NOT AUTHORIZED.**  Rows 19--32 remain unmeasured.  Frozen bands
remain baseline `22.479491 m`, CONFIRM `<=11.2397455 m`, REFUTE
`>=20.2775 m`, with all previously registered acceptance gates unchanged.

Receipts:

- scalar-LIBM discriminator:
  `docs/ocean/fidelity/dino_zdf_etau_libm_ownership_artifact.json`, SHA256
  `d0e62557c3d3e09b43ffb30494a219a9e47fc0b40a1730e5a33c2cbeb63f0854`;
- literal-EXP and remaining-operand sweep:
  `docs/ocean/fidelity/dino_zdf_chain_sweep_round17_artifact.json`, SHA256
  `14aa2bd2ecf52bd86ccea65c43389619fadbae9e9152e2a58c36264e803f2266`.

## Round-11 result: literal TKE solve closed; ordered stop at row 18

`tke_solver_evaluation="nemo_literal"` is implemented and defaults only on
`nemo_dino_kamm` and `nemo_dino_kamm_mlf`.  It transcribes
`zdftke.F90:547-565` as separate ordered forward-diagonal and forward-RHS JAX
scans, seeds the direct `jpkm1` solution, reverse-substitutes toward the
surface, and applies the floor and W mask after the solve.  It preserves the
division-before-multiply association in the RHS recurrence and does not solve
the held `jpk` row as an ordinary Thomas row.

`shared_thomas` remains the exact historical default for every other DINO
recipe, generic `TKEConfig`, the ORCA-oriented `_nemo_tke_config`, and both ACC
TKE cards; it is the explicit legacy opt-in on the two DINO oracle cards.  The
literal solver requires the already-literal NEMO matrix, z=0 surface row,
floor positivity, and W mask and fails closed otherwise.  The red-capable hand
case pins the first-column solution to `22/23, 19/23, 0`, distinguishes an
unequal-depth column, and proves that the shared solver, a changed RHS, and a
multiply-before-divide recurrence do not pass.  Eager/JIT equality and finite
reverse gradients are also pinned.  The focused solver/TKE suites pass 87/87;
the time-level registry suite passes 16/16.

The ordered CPU/fp64 disposition is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1--15 | `eos_rab` through all TKE RHS terms | `VERIFIED` | unchanged | 4/4 pass |
| 16 | wave surface boundary | `WAIVED` | resolved DINO has `ln_wave=F` | inactive |
| 17 | literal TKE tridiagonal solve application | **`VERIFIED`** | **0/9,920; max `0`** | **4/4 exact** |
| 18 | `nn_etau=1` penetrating TKE addition | **`DIVERGED`** | **173/9,920; max `2.277967e-15`** | **4/4 pass** |
| 19--32 | mixing length through EVD, assembly, `ldf_slp`, and both implicit applications | `UNMEASURED` | ordered stop at row 18 | ordered stop |

### Row-18 first operand

The active NEMO expression is:

```fortran
en(ji,jj,jk) = en(ji,jj,jk) + rn_efr * en(ji,jj,1) * &
   EXP( -gdepw(ji,jj,jk,Kmm) / htau(ji,jj) ) &
   * MAX( 0._wp, 1._wp - zice_fra(ji) ) * wmask(ji,jj,jk) * tmask(ji,jj,1)
```

Source: NEMO 5.0.2 `cfgs/DINO/MY_SRC/zdftke.F90:590-591` under the active
`nn_etau == 1` branch at line 588.

The pre-penetration `en` is exact (0/9,920).  A new write-only stream separates
the exponent argument from its result.  The raw `-gdepw/htau` argument is
`VERIFIED`, 0/9,920 failures with maximum `4.496170e-16`; the immediately
following NEMO `EXP(argument)` is the first numeric divergence after
substituting NEMO's dumped argument, 59/9,920 failures with maximum
`1.500624e-15`.  The unsubstituted production argument-plus-EXP composite
fails 355/9,920 with maximum `1.875780e-15`.  The complete additive increment fails
1,480/9,920 in production and the final row fails 173/9,920.  NumPy/libm and a
raw-radian latitude substitution also fail, so neither is promoted as a fix.
All four southern focus columns pass the final row (`0` through
`2.847459e-16`); the whole-domain census, not a focus-only score, finds this
roundoff-tier mismatch.

The argument/EXP/increment instrument is write-only: its output restart is
bit-identical to the dump-off control for all 131 shared numeric variables.
The active binary imports `exp@GLIBC_2.29`; source, binary, all three operand
streams, restarts, time levels, and the MLD focus map are SHA-stamped in the
artifact.  Planted perturbation, horizontal-roll, and nonfinite controls all
fire.

### Deferred large-fix design

The next option is `tke_etau_exponential_evaluation`.  `nemo_literal` will be
the correct-by-default choice only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; `jax_expression` remains byte-identical everywhere else
and becomes the explicit legacy opt-in on those two cards.  The faithful path
must reproduce the active NEMO/glibc fp64 exponential lowering from the
already-verified argument using pure JAX source-ordered range reduction and
polynomial/table evaluation; a NumPy callback is forbidden because it breaks
JIT and reverse-mode AD.  Required red tests include the direct argument and
EXP dumps, the worst day-180 columns, ULP/`nextafter` discriminators, eager/JIT
identity, finite reverse gradients, final row 18 at 0/9,920, and exact pins for
all unchanged cards.  This is a new differentiable elementary-function path
and is too large for this round, so rows 19--32 are not measured.

### Climate status

**CLIMATE ARMS NOT AUTHORIZED.**  The chain is not verified end-to-end.  The
frozen prediction is unchanged: baseline southern day-90 MLD RMS
`22.479491 m`, CONFIRM `<=11.2397455 m`, REFUTE `>=20.2775 m`, with the
legacy-baseline (`0.001 m`), acceptance-floor, 5x pass-tally, and
southern-density gates unchanged.  The eventual faithful command remains
option-free.  The eventual legacy command must add
`--tke-solver-evaluation shared_thomas` and, once the row-18 option exists, its
`jax_expression` selector to the previously frozen control command.  These
are not runnable authorization commands while row 18 remains divergent.

Machine-readable receipt:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round14_artifact.json`, SHA256
`56bd965f2625ef5196729650fc326662005f0f2263320df4cb44b21ec63d5b1d`.
The stamped committed probe/tree SHA is
`ca287822841fb04138fe6cd8d25aedffd5dcc252`.

### Round-11 adversarial review

Two independent read-only reviews raised four holds, all resolved before this
receipt was committed.  The solver review required an association
discriminator exercised through the literal solver, full production-dispatch
coverage, and behavioral shared-solver pins for every unchanged reaching
card.  The receipt review required substituting the dumped NEMO exponent
argument rather than inferring attribution from an at-bar argument composite;
that stricter measurement is the 59/9,920 result above.  The final rerun uses
the committed fixes and exits cleanly.  Residual scope: the write-only bracket
proves restart neutrality for this matched one-step state, and the literal
solver is intentionally selectable only on DINO cards whose terminal W mask
is the NEMO zero row.

## Round-10 result: literal Langmuir closed; ordered stop at row 17

`tke_langmuir_evaluation="nemo_literal"` is implemented and defaults only on
`nemo_dino_kamm` and `nemo_dino_kamm_mlf`. It transcribes the active no-Stokes
chain at `zdftke.F90:422-463` with ordered JAX scans, strict threshold
semantics, and each column's own `mbkt+1` no-crossing fallback. The final
line-463 update also preserves NEMO's `((rn_Dt*zus3)*zwlc^3)/zhlc`
association. `vectorized` remains the exact historical default on every other
DINO, generic NEMO, and ACC card and is the explicit legacy opt-in on the two
DINO oracle cards. The literal path rejects mixed generic/Langmuir sources.

The ordered CPU/fp64 disposition is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1--9 | `eos_rab` through bottom TKE boundary | `VERIFIED` | unchanged | 4/4 pass |
| 10 | complete Langmuir operation through line 463 | **`VERIFIED`** | **0/9,920; max `0`** | **4/4 exact** |
| 11 | Richardson `zri/p_pdlr` | `VERIFIED` | 0/9,920; max `2.877796e-16` | 4/4 pass |
| 12 | literal TKE matrix, including `dissl` | `VERIFIED` | 0/9,920; max `0` | 4/4 exact |
| 13--15 | shear, stratification, and dissipation RHS operands | `VERIFIED` | each 0/9,920; max `0` | 4/4 exact |
| 16 | wave surface boundary | `WAIVED` | resolved DINO has `ln_wave=F` | inactive |
| 17 | TKE tridiagonal solve application | **`DIVERGED`** | **1,374/9,920; max `0.3044579`** | **4/4 pass** |
| 18--32 | postsolve closure through EVD, assembly, `ldf_slp`, and implicit applications | `UNMEASURED` | ordered stop at row 17 | ordered stop |

### Row-10 receipt and the final two-column localization

The first literal-source rerun made the source rate itself exact but left two
post-update columns outside the `1e-15` bar (maximum
`1.997411720059583e-15`). That residual was the last operation at NEMO line
463, not another source operand: the generic solver multiplied `rn_Dt` after
the source had already been divided by `zhlc`. Carrying the separately
source-ordered post-Langmuir `en` closes the direct dump at **0/9,920**, maximum
zero, with all four southern focus columns exact. The planted unequal-bottom
control makes the historical global-bottom fallback disagree, and a hand case
makes the rate-first line-463 association differ by one bit.

The write-only instrument bracket remains exact across all 131 shared numeric
restart variables, and the source-rate, solver-source, line-463 update, direct
post-Langmuir stream, time-level, source, binary, and bracket hashes are all
stamped in the artifact. The tracked row-11 qualification remains loud: its
intermediate numerator misses 3/9,920 columns even though the subsequent
`/e3w` and complete `zri/pdlr` row pass. It is not claimed bit-identical.

### First divergence: row-17 solve application

The active oracle applies the matrix as:

```fortran
zdiag(ji,jk) = zdiag(ji,jk) - zd_lw(ji,jk) * &
               zd_up(ji,jk-1) / zdiag(ji,jk-1)       ! :548-550
zd_lw(ji,jk) = en(ji,jj,jk) - zd_lw(ji,jk) / &
               zdiag(ji,jk-1) * zd_lw(ji,jk-1)       ! :555-557
en(ji,jj,jpkm1) = zd_lw(ji,jpkm1) / zdiag(ji,jpkm1) ! :558-560
en(ji,jj,jk) = (zd_lw(ji,jk) - zd_up(ji,jk) * &
                en(ji,jj,jk+1)) / zdiag(ji,jk)       ! :561-563
en(ji,jj,jk) = MAX(en(ji,jj,jk),rn_emin)*wmask(ji,jj,jk) ! :564-565
```

Production's shared normalized Thomas solve diverges in **1,374/9,920**
columns, maximum normalized error `0.3044579035485571`; the four southern
focus columns pass (`8.40e-17` to `3.36e-16`). The matrix, composite RHS,
post-Langmuir base, bottom scatter, forward diagonal, and forward RHS each
score 0/9,920. Most decisively, the committed post-hoc transcription of the
full NEMO recurrence produces the postsolve oracle at **0/9,920**, maximum
zero. The first failing operand is therefore the production recurrence and
terminal/back-substitution application at `zdftke.F90:547-565`, not its
inputs.

The registered next option is `tke_solver_evaluation`: `nemo_literal` becomes
the default only on the two complete DINO oracle cards; `shared_thomas`
remains byte-identical elsewhere and is their legacy opt-in. The literal path
will use ordered differentiable scans for the reciprocal surface seed, the
two separate forward recurrences, the direct `jpkm1` seed, reverse
substitution, and the final floor/mask. This is a new differentiable solver
path with boundary, JIT, and gradient obligations and is too large for this
round, so the ordered sweep stops here.

Two independent adversarial reviews returned PASS. The physics reviewer
re-ran 164 CPU/fp64 tests and confirmed the literal source/update order and
row-17 recurrence localization. The scope reviewer confirmed the two-card
default boundary, unchanged-card pins, mixed-source guards, provenance,
ordered stop, and climate hold. The author's focused CPU/fp64 suite passes
180/180 tests.

### Climate status

**CLIMATE ARMS NOT AUTHORIZED.** Rows 18--32 are unmeasured. The frozen
southern day-90 MLD prediction remains baseline `22.479491 m`, CONFIRM
`<=11.2397455 m`, REFUTE `>=20.2775 m`, with the legacy-baseline (`0.001 m`),
acceptance-floor, 5x pass-tally, and southern-density gates unchanged. The
future faithful command remains option-free; the legacy control now includes
`--tke-langmuir-evaluation vectorized`, as printed below in round 9.

## Round-9 result: matrix receipt closed; ordered stop corrected to row 10

`tke_matrix_evaluation="nemo_literal"` is implemented and defaults only on
`nemo_dino_kamm` and `nemo_dino_kamm_mlf`. It carries live raw-mesh
`e3t(Kmm)` and previous-step `dissl`, uses NEMO's base `rn_Dt=2700 s` under
MLF, and assembles `zcof/zd_up/zd_lw/zdiag` in source order. `factored` stays
the byte-identical default everywhere else and remains an explicit opt-in on
the two DINO oracle cards. Red-capable tests cover nonuniform `e3t != e3w`, a
wrong-slot denominator, source-order association, literal-path JIT/reverse AD,
and default-versus-explicit-factored output identity.

The corrected ordered CPU/fp64 disposition is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1--9 | `eos_rab` through bottom TKE boundary | `VERIFIED` | unchanged | 4/4 pass |
| 10 | complete Langmuir operation through the `en` update | **`DIVERGED`** | **329/9,920; max `4.344518e-04`** | **4/4 pass at zero** |
| 11--32 | Richardson chain through EVD, assembly, `ldf_slp`, and both implicit applications | `UNMEASURED` | ordered stop at row 10 | ordered stop |

**LOUD RETRACTION:** the first round-9 report treated row 10 as only its
`rn2b` operand, marked that operand `VERIFIED`, and promoted rows 11--14 before
stopping at row 15. The preregistered row 10 is the entire active Langmuir
operation at `zdftke.F90:401-468`. The direct post-Langmuir stream proves that
row 10 itself diverges, so rows 11--32 are now `UNMEASURED`. The committed
probe can no longer print the stale row-15-first result.

### Row-12 requested receipt (post-hoc, not an ordered promotion)

Before the ordered-stop correction, the implemented matrix was rerun against
`zdftke.F90:499-510`: `zd_up`, `zd_lw`, and `zdiag` each scored **0/9,920** at
the `1e-15` bar, maximum zero, with all four focus columns exact. The carried
`dissl` operand in the line-510 diagonal also scored **0/9,920**, maximum zero.
This remains a citable post-hoc receipt from commit `7de1be2cc74` and artifact
SHA256 `8b001296c8fc8c0fa72ba202a52bc43a89b3e775bf39ade3644da196545444ec`;
it is not a row-12 `VERIFIED` disposition past the row-10 stop.

The earlier timestep diagnosis is also loudly retracted: the first draft used
2700 s to reconstruct `zcof` while production consumed the MLF 5400 s step.
Production now threads NEMO's base `rn_Dt` only to `nemo_literal`; with the
actual solver timestep scored, the matrix receipt closes exactly.

### First divergence: row-10 Langmuir source output

The active oracle source says:

```fortran
imlc(:) = mbkt(T1Di(0),jj) + 1                         ! zdftke.F90:444
DO_2Dik( 0, 0, jpkm1, 2, -1 )
   IF( zpelc(ji,jk) > zWlc2(ji) ) imlc(ji) = jk       ! :445-446
END_2D
...
en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji) * &
               ( zwlc * zwlc * zwlc ) / zhlc(ji)      ! :463
```

Against `tke_dump_en_postlc.bin`, legoESM's post-Langmuir `en` fails
**329/9,920** columns, maximum normalized column error `4.344518e-04`; all four
registered southern focus columns pass exactly. The already-fixed `rn2b`,
`gdepw(Kmm)`, and `e3w(Kmm)` inputs pass, localizing the remaining operand to
the vectorized Langmuir source composite produced at line 463. The offline
source-only reconstruction fails 1,562/9,920 and is localization evidence,
not the dispositive row score.

The write-enabled restart remains bit-identical to the write-disabled control
across all 131 shared numeric variables (0 differences). The probe now fails
closed unless the post-LC stream, instrument source/binary, and both bracket
restarts exist; their paths and SHA256 values are stamped globally even though
row 17 is not reached. `tke_dump_en_postlc.bin` is registered at the `now`
time level. Its SHA256 is
`663bbe155f9e358b224bd93cc31a0a6b15e2f6cd209a4c145b75112a49cecccb`.

### Stop design

The next option is `tke_langmuir_evaluation`. `nemo_literal` becomes the
default only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf`; `vectorized`
remains byte-identical elsewhere and is the explicit legacy opt-in on those
two cards. The literal path must transcribe the active no-Stokes `zWlc2` arm
at `zdftke.F90:422-429`, the top-down `zpelc` recurrence, and the bottom-up
`imlc` scan. Critically, it must carry each column's `bottom_level/mbkt` and
initialize no-crossing fallback to that column's `mbkt+1` exactly as line 444
does; a global deepest-interface fallback is wrong for unequal-depth columns.
It then evaluates `zhlc`, `zus/zus3`, `zwlc`, and the line-463 update in source
order using differentiable scans.

Required red tests include a hand threshold/tie case, first-versus-last
crossing, unequal shallow/deep no-crossing columns, direct post-LC census,
JIT/finite gradients, and unchanged-card byte identity. The two DINO cards
currently have generic EKE-recycling and bottom-dissipation sources disabled;
the literal implementation must keep Langmuir separate from generic external
sources or fail closed before any mixed-source card can select it. This
cumulative/reverse-selection fix is too large for this round, so work stops at
row 10.

The row-11 qualification remains tracked, without an ordered promotion: the
prior committed receipt's intermediate numerator misses 3/9,920 columns
(maximum `1.174266e-15`) even though the subsequent `/e3w` and final
`zri/pdlr` composite passed. It is not silently upgraded to bit identity.

### Adversarial review and tests

Two independent reviews returned PASS after corrections. The physics review
required the per-column `mbkt+1` no-crossing fallback, corrected active-source
line citations, and the mixed-source guard. The scope review required the loud
row-10 ordered-stop retraction, global stream/source/binary/bracket provenance,
and literal JIT/AD plus explicit-factored output pins. The focused CPU/fp64
suite passes 123/123 tests.

### Climate status

**CLIMATE ARMS NOT AUTHORIZED.** The chain is not verified end-to-end. The
registered next step is the literal Langmuir implementation and row-10 rerun,
not a GPU integration. The frozen prediction remains: baseline southern
day-90 MLD RMS `22.479491 m`; CONFIRM `<=11.2397455 m`; REFUTE
`>=20.2775 m`; the legacy-baseline (`0.001 m`), acceptance-floor, 5x pass
tally, and southern-density conditions remain as previously registered.

The future faithful arm remains option-free. The historical control command,
once the new Langmuir selector exists, must opt out of every implemented
faithful stage:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_faithful_d90.npz --days 90 --save-3d --bridge-tke

CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_legacy_d90.npz --days 90 --save-3d --bridge-tke \
  --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian \
  --dino-wind-profile-evaluation factored_smoothstep \
  --tke-n2-evaluation-stage implicit_solve_state \
  --tke-matrix-evaluation factored \
  --tke-langmuir-evaluation vectorized
```

The last flag is the registered next-round interface and does not exist yet;
therefore these are frozen handoff commands, not authorization to run them.
Round-9 artifact SHA256:
`b7c3d2de46935da9a05ab8e2e8ef483ba2f125c0774493f1bf376b716aa15229`.

## Round-8 result: row 11 closed; ordered stop at row 12

The instrumented `eosbn2.F90:1459-1468` walk and production rerun close row
11. The faithful `step_entry` path now carries the full column-dependent raw
`gdepw_0`, not the 1-D reference ladder, constructs NEMO's reciprocal-first
qco stretch, and evaluates both `eos_rab` depth and `zrw` from separately
rounded raw-mesh products. The row-11 `zri/p_pdlr` composite is **0/9,920**
at the bar, maximum `2.877796e-16`, with 4/4 focus columns passing.

The ordered rerun is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 4 | complete `zdf_sh2` | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 5 | bottom-drag coefficient | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 6 | native MLD index `nmln` | `VERIFIED` | exact; 0/9,920 | 4/4 pass |
| 7 | native MLD depth `hmlp` | `VERIFIED` | 0/9,920; max `5.670461e-16` | 4/4 pass |
| 8 | surface TKE boundary | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 9 | bottom TKE boundary | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 10 | Langmuir `rn2b` operand | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 11 | Richardson `zri/p_pdlr` | **`VERIFIED`** | **0/9,920; max `2.877796e-16`** | **4/4 pass** |
| 12 | TKE diffusion matrix | **`DIVERGED`** | **9,920/9,920; `zd_up` max `8.829184`** | **4/4 fail** |
| 13--32 | TKE RHS through both implicit applications | `UNMEASURED` | ordered stop at row 12 | ordered stop |

### Row-11 instrumentation and fix receipt

The write-only NEMO stream records `zrw -> zaw/zbw -> numerator -> /e3w` for
the first Nbb call. The dump-on and same-source dump-off one-step restarts have
131/131 numeric variables and zero differences. The older 95-variable
certified binary is explicitly not this bracket; the probe prints that claim
as retracted and identifies its pre-existing `utrd_tau` difference. The
planted wrong-field control fires.

Against the active NEMO lines:

```fortran
zrw = ( gdepw(ji,jj,jk,Kmm) - gdept(ji,jj,jk,Kmm) ) / &
      ( gdept(ji,jj,jk-1,Kmm) - gdept(ji,jj,jk,Kmm) )
zaw = pab(ji,jj,jk,jp_tem) * (1. - zrw) + pab(ji,jj,jk-1,jp_tem) * zrw
zbw = pab(ji,jj,jk,jp_sal) * (1. - zrw) + pab(ji,jj,jk-1,jp_sal) * zrw
pn2(ji,jj,jk) = grav * (zaw*dT-zbw*dS) / e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)
```

the full 3-D `gdepw_0` is essential at partial cells. The initial use of
`gdepw_1d` was rejected by the probe before a result was admitted. With the
raw W ladder and raw `gdept_0*(1+r3t)` feeding `eos_rab`, `zrw` and `zbw` are
bit-identical. `zaw`, numerator, and result retain last-bit differences in
9,425, 9,275, and 9,205 columns, respectively. `zaw` and the final result pass
their registered bars (max `4.658093e-16` and `3.979116e-16`); the intermediate
numerator misses in 3/9,920 columns (max `1.174266e-15`) before division by
the exact `e3w` operand closes the final result to 0/9,920 failures. Rows 2,
3, and 10 consequently also improve from sub-bar residuals to exact zero.

**LOUD RETRACTION:** an earlier draft of this round's result called all five
intermediate streams bit-identical. The committed artifact never supported
that statement: only `zrw` and `zbw` are exact. The intermediate numerator is
also not VERIFIED at its diagnostic bar (3 failing columns); only `zaw` and
the final `/e3w` result are VERIFIED. The row-11 disposition is based on that
final result and the downstream `zri/pdlr`, both 0/9,920 at the bar.

**TRACKED NOTE — open arithmetic-association debt:** retain the 3/9,920
intermediate-numerator misses (max `1.174266e-15`) in every later chain
handoff. They do not change row 11's VERIFIED disposition because the next
literal `/e3w` operation closes the final `bn2` result to 0/9,920, but they
must not be silently promoted to exact identity or dropped from the ledger.

Scope is narrow:

| Reachable card/config | Literal raw-mesh N2 path | Numerical change |
|---|---|---|
| `nemo_dino_kamm` | `step_entry`, faithful default | raw `gdept_0/gdepw_0` source association |
| `nemo_dino_kamm_mlf` | inherited `step_entry`, faithful default | same, with genuine Nbb tracers |
| DINO `nemo_paper`, `veros` | `implicit_solve_state` | byte-identical legacy path |
| all non-TKE DINO cards | unreachable | unchanged |
| generic/ORCA/ACC/MPAS TKE | no DINO `step_entry` selector | unchanged |

The red-capable suite includes the hand-computed matched-step source-order
case, a cancelled-ratio violation, reciprocal-versus-quotient separation,
full 3-D W-ladder bridge pins, JIT/gradient checks, and unchanged-card pins.
The post-review focused physics/bridge run reports 69/69 passing tests; no GPU
was used.

### First divergence: row-12 live `e3t` denominator is absent

NEMO assembles (`cfgs/DINO/MY_SRC/zdftke.F90:499-510`):

```fortran
zcof   = zfact1 * tmask(ji,jj,jk)
zzd_up = zcof * MAX(p_avm(ji,jj,jk+1)+p_avm(ji,jj,jk),2.e-5_wp) / &
         (e3t(ji,jj,jk,Kmm)*e3w(ji,jj,jk,Kmm))
zzd_lw = zcof * MAX(p_avm(ji,jj,jk)+p_avm(ji,jj,jk-1),2.e-5_wp) / &
         (e3t(ji,jj,jk-1,Kmm)*e3w(ji,jj,jk,Kmm))
zdiag(ji,jk) = 1._wp-zzd_lw-zzd_up + zfact2*dissl(ji,jj,jk)*wmask(ji,jj,jk)
```

The production solve receives `dz_cell=None`, so its legacy matrix branch
feeds the incoming `e3w` into the face-gradient slot where NEMO reads live
`e3t(jk,Kmm)`, then feeds a shifted/repeated `e3w` ladder into the control-
volume slot where NEMO reads `e3w(jk,Kmm)`. In coefficient evaluation order,
`zcof`, `p_avm`, and both clipped viscosity sums pass at exact zero. The first
failure is therefore the `zd_up` `e3t(jk,Kmm)` operand at the quoted line 504:
**9,920/9,920**, maximum `0.2484444`, including every focus column. The lower
`e3t(jk-1,Kmm)` slot similarly fails 9,920/9,920 (max `0.2394656`), and the
effective `e3w(jk,Kmm)` control-volume slot fails 9,920/9,920 (max
`0.4662894`). The independently captured incoming live `e3w(Kmm)` remains an
exact-zero control; it is present but production assigns it to the wrong
coefficient role. The resulting `zd_up`, `zd_lw`, and `zdiag` each fail every
column; their maxima are `8.829184`, `9.372063`, and `5.965324`, respectively.

**LOUD RETRACTION:** the first row-12 artifact draft scored one shifted
`dz_int_eff` approximation against `e3t` and reported max `0.6806799`; it did
not score every actual coefficient slot separately. The corrected committed
probe now records `zcof`, upper/lower viscosity sums, upper/lower `e3t`, the
effective control-volume `e3w`, and the incoming `e3w` control, and selects
the first failing operand separately for each coefficient. The corrected
line-504 owner is `up_e3t_jk_Kmm`, max `0.2484444`.

The later `dissl` operand also fails all columns (maximum `0.1088969`), but it
first enters `zdiag` at line 510. It cannot displace the earlier line-504
metric owner and is the registered next operand after the metric fix.

The next-round design is
`tke_matrix_evaluation="nemo_literal"`: carry live raw-mesh
`e3t_0*(1+r3t)` with the step-entry geometry and assemble `zcof`, `zd_up`,
`zd_lw`, and `zdiag` in literal source order before the unchanged Thomas
solve. It becomes the default only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; `factored` is their legacy opt-in and remains the
byte-identical default everywhere else. A dedicated selector avoids enabling
the unrelated N2, surface-volume, and mixing-length semantics bundled under
`veros_dz_slots`. Required tests use a hand-computed nonuniform column with
`e3t != e3w`, a planted shifted-W denominator, exact coefficient values,
JIT/grad checks, and pins for every unchanged card. This fix plus the ordered
`dissl` peel is too large for this round, so rows 13--32 are not measured.

### Round-8 climate status

**CLIMATE ARMS NOT AUTHORIZED.** The chain is not verified past row 12. The
frozen prediction remains baseline `22.479491 m`, CONFIRM
`<=11.2397455 m`, REFUTE `>=20.2775 m`, with the registered legacy-baseline,
acceptance-floor, pass-tally, and southern-density conditions unchanged. Do
not run either GPU arm yet.

Artifacts: round 7 SHA256
`47ce1337a23f677e391137b3c4ff971d6fdac412b39f29ba5dcf783b0ed0d108`;
round 8 SHA256
`ed49f0a54fd0bf782237c08cc5043b7126b7f6665ff6b4b7a018a399aa0fbed6`.
The round-8 parent/probe SHA is `5ebd141b0fe` and the artifact carries all
source, input, dump, environment, time-level, focus, and control stamps.

## Round-6 result: row 10 closed; ordered stop at row 11

`tke_n2_evaluation_stage="step_entry"` is now the default on the complete
`nemo_dino_kamm` and `nemo_dino_kamm_mlf` cards. It freezes one
`(rn2, rn2b, gdepw_Kmm, e3w_Kmm)` bundle at physical step entry and carries it
across the outer MLF implicit solve into `zdf_mxl` and every registered TKE
consumer. The MLF card constructs genuine Nbb `rn2b`; the FE card retains its
documented `rn2b == rn2` ceiling. `implicit_solve_state` is the unchanged
generic/default path for every other card and the explicit legacy opt-in.

The ordered CPU/fp64 rerun is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4 | complete `zdf_sh2` | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 5 | bottom-drag coefficient | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 6 | native MLD index `nmln` | `VERIFIED` | exact integer equality; 0/9,920 | 4/4 pass at zero |
| 7 | native MLD depth `hmlp` | `VERIFIED` | 0/9,920; max `5.670461e-16` | 4/4 pass |
| 8 | surface TKE boundary | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 9 | bottom TKE boundary | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 10 | Langmuir `rn2b` operand | **`VERIFIED`** | **0/9,920; max `5.968673e-16`** | **4/4 pass** |
| 11 | Richardson `zri` / inverse Prandtl `p_pdlr` | **`DIVERGED`** | `zri`: **86/9,920**, max `3.473067e-14`; `p_pdlr`: 15/9,920, max `4.152660e-13` | 4/4 pass |
| 12--32 | TKE matrix through EVD, assembly, `ldf_slp`, and both implicit applications | `UNMEASURED` | ordered stop at row 11 | ordered stop |

All planted perturbation, horizontal-roll, and nonfinite controls fire. Row
10 therefore meets its requested target exactly: **0/9,920 failures at the
registered `1e-15` bar**, with all southern focus columns retained.

### Row-11 arithmetic fix and remaining operand

The first row-11 pass exposed and fixed a literal-association difference.
NEMO evaluates (`cfgs/DINO/MY_SRC/zdftke.F90:477-495`):

```fortran
zdiv = p_sh2(ji,jj,jk) + rn_bshear
zri = rn2b(ji,jj,jk) * p_avm(ji,jj,jk) / zdiv
p_pdlr(ji,jj,jk) = MAX( 0.1_wp, ri_cri / MAX( ri_cri, zri ) )
```

The two complete DINO cards now retain those divisions and construct literal
`p_pdlr` before returning `Pr=1/p_pdlr`; the historical reciprocal-first and
algebraically collapsed path remains byte-identical under
`implicit_solve_state`. A hexadecimal hand case turns red by one ulp under
either shortcut and is exact under the faithful path. The first post-fix
probe mistakenly retained reciprocal-first arithmetic in its own diagnostic;
that output was retracted in the probe before the final artifact was made.

The final one-at-a-time substitution localizes the remaining row-11 failure
to the carried `rn2b` value: substituting NEMO `rn2b` alone gives bit-exact
`zri` in all 9,920 columns; substituting `p_avm` or `p_sh2` changes no failed
column, while those two operands are already bit-exact. The N² residual passed
row 10's own bar but is amplified by Richardson division. The source operation
is NEMO `src/OCE/TRA/eosbn2.F90:1459-1468`:

```fortran
zrw = ( gdepw(ji,jj,jk,Kmm) - gdept(ji,jj,jk,Kmm) ) / &
      ( gdept(ji,jj,jk-1,Kmm) - gdept(ji,jj,jk,Kmm) )
zaw = pab(ji,jj,jk,jp_tem) * (1. - zrw) + pab(ji,jj,jk-1,jp_tem) * zrw
zbw = pab(ji,jj,jk,jp_sal) * (1. - zrw) + pab(ji,jj,jk-1,jp_sal) * zrw
pn2(ji,jj,jk) = grav * ( zaw * (T_upper-T_lower) - zbw * (S_upper-S_lower) ) &
                 / e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)
```

The next fix is deliberately not guessed from the final `rn2b` residual. It
requires extending the existing write-only NEMO instrumentation with ordered
`zrw`, `zaw`, `zbw`, pre-division numerator, and final-division slots, proving
the instrumented one-step bracket stream bit-identical, then walking those
operands against the existing legoESM entry bundle. The first failed slot will
select a `nemo_literal` association only on `step_entry`; all
`implicit_solve_state` cards remain byte-identical. That instrumented oracle
run and its red-capable production correction are too large to complete in
this round, so the ordered sweep stops here.

### Round-6 climate status

**CLIMATE ARMS NOT AUTHORIZED.** Rows 12--32 have not been measured. The
prediction remains frozen: baseline `22.479491 m`, CONFIRM
`<=11.2397455 m`, REFUTE `>=20.2775 m`, with the previously registered
acceptance-floor, pass-tally, legacy-baseline, and southern-density conditions
unchanged. The faithful command remains option-free but is not authorized;
the eventual legacy command additionally selects
`--tke-n2-evaluation-stage implicit_solve_state`.

Round-6 artifact:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round6_artifact.json`, SHA256
`baa1d31d933d7055010c073df7f638ef708e81920350cb11e456105daf4e80e1`.
Its stamped parent SHA is `4e79e71109c`; the full artifact carries the probe,
input, dump, map, environment, dtype, time-level, and control receipts.

## Round-5 result: row 8 closed; ordered stop at row 10

`dino_wind_profile_evaluation="nemo_literal"` is now the default on the
complete `nemo_dino_kamm` and `nemo_dino_kamm_mlf` cards.  It retains NEMO's
raw degree-valued `gphiu` operand from `mesh_mask.nc` in the matched-state
harness; ordinary production runs reconstruct that source-degree operand with
the same scalar `usrdef_hgr.F90:95-107` expression rather than fall back to
`degrees(grid.lat)`.  It uses NEMO's nearest-node interval
selection and preserves the left-associated cubic at
`usrdef_sbc.F90:632`.  `factored_smoothstep` remains byte-identical by default
on all five other DINO cards and is the explicit legacy opt-in on the two
complete cards.  A matched-state dump control proves the historical
construction remains red: 154/9,920 columns fail against `sbc_dump_utau.bin`.

The ordered rerun is:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4 | complete `zdf_sh2` | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 5 | bottom-drag coefficient | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 6 | native MLD index `nmln` | `VERIFIED` | exact integer equality; 0/9,920 | 4/4 pass at zero |
| 7 | native MLD depth `hmlp` | `VERIFIED` | 0/9,920; max `5.670461e-16` | 4/4 pass at zero |
| 8 | surface TKE Dirichlet boundary | **`VERIFIED`** | **0/9,920; max `0`** | **4/4 pass at zero** |
| 9 | bottom TKE Dirichlet boundary | `VERIFIED` | 0/9,920; max `0`; independent `mbathy-1` identity 0/9,920 mismatches | 4/4 pass at zero |
| 10 | Langmuir `rn2b` operand | **`DIVERGED`** | **9,920/9,920; max `4.150232e-07`** | **4/4 fail** |
| 11--32 | Prandtl through EVD, coefficient assembly, `ldf_slp`, and both implicit solves | `UNMEASURED` | ordered stop at row 10 | ordered stop |

### First failing row-10 operand

NEMO computes and freezes both stability fields on the Nnn geometry before
entering vertical physics (`cfgs/DINO/MY_SRC/stpmlf.F90:204-210`):

```fortran
CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )
CALL eos_rab( ts(:,:,:,:,Nnn), rab_n, Nnn )
CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nnn )
CALL bn2    ( ts(:,:,:,:,Nnn), rab_n, rn2, Nnn  )
CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )
```

The Langmuir PE integral then consumes that carried `rn2b` with the same live
Nnn geometry (`cfgs/DINO/MY_SRC/zdftke.F90:436-440`):

```fortran
zpelc(ji,1) = MAX( rn2b(ji,jj,1), 0._wp ) * gdepw(ji,jj,1,Kmm) * e3w(ji,jj,1,Kmm)
zpelc(ji,jk) = zpelc(ji,jk-1) + MAX( rn2b(ji,jj,jk), 0._wp ) &
             * gdepw(ji,jj,jk,Kmm) * e3w(ji,jj,jk,Kmm)
```

legoESM instead recomputes `rn2b` when the implicit-mixing state is assembled
(`k_profiles.py:827-831`) and calls the Langmuir kernel with static
`-z_interface` and generic `dz_half` (`tke.py:2305-2322`).  The operand walk
therefore fails before any Langmuir arithmetic: `taum` is exact, but the
captured `rn2b`, `gdepw`, and `e3w` fail in all 9,920 columns, with normalized
maxima `4.150232e-07`, `6.294510e-04`, and `1.197581e-02`; all four southern
focus columns fail each operand.  Offset-zero controls are much worse, ruling
out an indexing explanation.  Perturbation, i-roll, and nonfinite controls all
fire.  The previously reported 9,155/9,920 full-source count came from an
offline reconstruction without a NEMO post-`ln_lc` stage dump; it is retained
in the artifact only as a non-dispositive diagnostic.  The ordered verdict
stops at the first independently dumped failing operand, `rn2b`, at
`zdftke.F90:436-440`.

Registered next-round design: add `tke_n2_evaluation_stage`, with
`step_entry` the faithful default on the two complete DINO NEMO cards and
`implicit_solve_state` the legacy default everywhere else and explicit opt-in
on those cards.  At step entry, compute and freeze the exact
`(rn2, rn2b, gdepw_Kmm, e3w_Kmm)` raw-mesh/live-geometry bundle and carry it
through `zdf_mxl` and every `zdf_tke` consumer.  Langmuir receives the frozen
live `gdepw/e3w`, not `-z_interface/dz_half`.  Every non-oracle card remains
byte-identical.

### Climate status

**CLIMATE ARMS NOT AUTHORIZED.**  Row 10 is a large, basin-visible divergence
and rows 11--32 remain ordered-unmeasured.  The prediction remains frozen at
baseline `22.479491 m`, CONFIRM `<=11.2397455 m`, REFUTE `>=20.2775 m`, with
the previously registered acceptance-floor, pass-tally, legacy-baseline, and
southern-density conditions unchanged.  Once row 10 is fixed and the later
rows are disposed, the faithful command remains option-free; the historical
control adds this round's selector to the three row-4 opt-outs:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_faithful_d90.npz --days 90 --save-3d --bridge-tke

CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_legacy_d90.npz --days 90 --save-3d --bridge-tke \
  --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian \
  --dino-wind-profile-evaluation factored_smoothstep
```

Round-5 artifact:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round5_artifact.json`, SHA256
`c611a4ee1961af071adee0f5edebaf8371c2a15a134a6c3907c0584d4ff3d833`.
Its stamped parent/probe SHA is
`1551b3055b2c5f34944bf13d1268330ed073fb96`.

## Round-4 result: row 4 closed; ordered stop at row 8

The registered `tke_shear_evaluation_stage=step_entry` implementation is now
production code.  On the two complete DINO NEMO cards it evaluates `p_sh2`
once at step entry with each card's configured shear formulation and carried
previous-step `p_avm`; the frozen array feeds both the TKE shear RHS and the
Prandtl denominator.  The MLF card uses the measured face-native NOW x BEFORE
form and `tke_shear_metric_source=nemo_qco_live_face`, preserving NEMO's
independent raw `e3uw_0/e3vw_0` operands and applying the two QCO face
stretches with the literal divisor and four-face association.  The FE card has
no leapfrog BEFORE velocity and freezes its pre-existing `squared_centered`
NOW formulation at entry.

The scope is deliberately narrow:

| Reachable card/config | Shear stage | Metric source | Numerical change |
|---|---|---|---|
| `nemo_dino_kamm` (FE) | `step_entry`, `squared_centered` NOW | `nemo_qco_live_face` resolved but inactive for T-point shear | stage timing changes; configured FE shear retained |
| `nemo_dino_kamm_mlf` | `step_entry`, face-native NOW x BEFORE | `nemo_qco_live_face` active | measured faithful defaults enabled |
| DINO `nemo_paper`, `veros` | `implicit_solve_state` | `tpoint_jacobian` | byte-identical legacy path |
| DINO `legoesm_default`, `mitgcm`, `oceananigans` | not this TKE path or legacy selectors | legacy | unchanged |
| generic `TKEConfig`, ORCA-oriented `nemo_recipe`, ACC/ACC-basic TKE, MPAS | `implicit_solve_state` | `tpoint_jacobian` | byte-identical legacy path |

The red-capable tests cover a hand-computed matched-step case, poisoned current
velocity, missing/shape-invalid frozen operands, invalid selectors, legacy
silent-no-op rejection, exact default-versus-explicit-legacy arrays, resolved
selectors for every DINO card and every independently constructed reachable
TKE config, live-face arithmetic, JIT, and finite gradients.  The final two
CPU/fp64 batches pass **315 tests** (134 + 181).  A real matched-state CPU
step of `nemo_dino_kamm` also completes with finite tracer and TKE arrays,
proving the resolved FE card no longer reaches the review-caught rejection.

### Ordered rerun

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920 failed; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4 | complete `zdf_sh2` | **`VERIFIED`** | **0/9,920; max `0`** | **4/4 pass at zero** |
| 5 | bottom-drag coefficient | `VERIFIED` | 0/9,920; max `0` | 4/4 pass at zero |
| 6 | native MLD index `nmln` | `VERIFIED` | exact integer equality; 0/9,920 at A-bar `1e-12` | 4/4 pass at zero |
| 7 | native MLD depth `hmlp` | `VERIFIED` | 0/9,920; max `5.670461e-16` | 4/4 pass at zero |
| 8 | surface TKE Dirichlet boundary | **`DIVERGED`** | **154/9,920; max `1.638670e-15`** | 4/4 pass at zero |
| 9 onward | bottom TKE boundary through EVD and implicit solves | `UNMEASURED` | ordered stop at row 8 | ordered stop |

The row-4 target is therefore met exactly: **0/9,920 failures at the registered
`1e-15` bar**, including every southern focus column.  Rows 5--7 also cross
their bars.  Row 8 is the next, and thus current, first divergence even though
its aggregate statistics and all four focus columns pass.  The general planted
perturbation, i-roll, and nonfinite controls fire.  Row 8 has its own exact
substitution baseline: the one-cell and nonfinite poisons fire; the i-roll is
explicitly waived because DINO's analytic wind is zonally invariant.

### First failing operand at row 8

NEMO constructs the wind and modulus at
`cfgs/DINO/MY_SRC/usrdef_sbc.F90:221-223` and preserves this arithmetic at
line 632:

```fortran
utau(ji,jj) = znl_cbc(znds_wnd_phi, znds_wnd_val, gphiu(ji,jj))
taum(ji,jj) = ABS( utau(ji,jj) )
IF( utau(ji,jj) > 0 ) taum(ji,jj) = taum(ji,jj) * 1.3_wp
pprofile = pnodes_val(ks) + ( pnodes_val(kn) - pnodes_val(ks) ) * ( 3 - 2 * zs ) * zs ** 2
```

The surface boundary then uses the carried modulus at
`cfgs/DINO/MY_SRC/zdftke.F90:334,361`:

```fortran
zbbrau = rn_ebb / rho0
en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )
```

The operand walk is decisive.  `gphiu` passes in 9,920/9,920 columns (max
`3.104929e-16`).  legoESM's production `utau` fails in 154/9,920 columns (max
`1.552068e-15`), and its derived `taum` fails in the same 154 columns (max
`1.609541e-15`).  A literal NEMO reconstruction using the same latitude and
knots matches `sbc_dump_utau.bin` exactly, 0/9,920 failures.  The row-8 oracle
is independently formed from that dump and resolved `rn_ebb/rho0/rn_emin0`;
it also matches the later post-`tke_tke` surface `en` exactly as a separately
labeled downstream-invariance check.  Substituting the dump-derived `taum`
makes row 8 exact, also 0/9,920.  The failure is therefore
the factored smoothstep in `dino_wind_stress`—`weight=(3-2s)*s**2` followed by
`delta*weight`—versus NEMO's left-associated
`delta*(3-2s)*s**2`, not geometry, knot selection, the 1.3 boost, or the TKE
boundary formula.

Next-round design: add `DINOConfig.dino_wind_profile_evaluation` with
`nemo_literal` as the faithful default only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; retain `factored_smoothstep` as the default everywhere
else and as the explicit legacy opt-in on those two cards.  The literal branch
uses NEMO's nearest-node interval selection and left-associated cubic before
`ABS` and the conditional westerly multiplier.  Required tests pin every
unchanged card byte-for-byte and hand-compute a latitude at which reassociation
changes the last bits.

### Climate prediction remains frozen; GPU is not the next step

The registered climate bands do not change: baseline `22.479491 m`, CONFIRM
`<=11.2397455 m`, and REFUTE `>=20.2775 m`, with the previously frozen
acceptance-floor, pass-tally, legacy-baseline, and southern-density conditions.
The faithful command remains option-free.  The legacy control still requires
all three implemented row-4 opt-outs:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_row4_faithful_d90.npz --days 90 --save-3d \
  --bridge-tke

CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_row4_legacy_d90.npz --days 90 --save-3d \
  --bridge-tke --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian
```

**Do not run the climate arms yet.**  The chain is not clean enough: the
registered next step is the row-8 literal-wind implementation and rerun, then
the ordered continuation at row 9.  A future implemented row-8 option will
also require its legacy selector in the control command before GPU execution.

Round-4 artifact:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round4_artifact.json`, SHA256
`f4ecff6d1fec9cbb5c0aacd15f8e5ab583276966a5452de8f9a3a08d0eaa121e`.
Derived probe/tree SHA: `9a9ebc25f1da5c140c67641b481a6ec842adabbf`.
The sandbox run exported the authoritative shadow
`GIT_DIR=/tmp/zdf-sweep-git.cJQ6wi/repo.git` and
`GIT_WORK_TREE=/tmp/codex-zdf-sweep`; the probe derived both SHAs from that
metadata and rejected dirty probe code and every effective ablation override.

## Round-3 result: carried coefficients fixed; row 4 stops again

The registered `carried_previous_step` construction is now the production
default on the complete DINO NEMO cards, `nemo_dino_kamm` and its inherited
`nemo_dino_kamm_mlf` card.  A NEMO restart bridge reads `avm_k` and `avt_k`
without inference.  During the step the carried pair feeds face-weighted shear,
the Prandtl numerator, TKE matrix diagonals, the stratification RHS, and (when
active) the wave denominator.  Only the post-solve closure output is stored as
the next step's pair; EVD and other enhancements remain downstream composition
and cannot leak into the carry.

This follows NEMO's active lifetime: `zdf_phy` calls
`zdf_sh2(Kbb,Kmm,avm_k)` before `zdf_tke(...,avm_k,avt_k)` at
`cfgs/DINO/WORK/zdfphy.F90:268,286`; `zdftke.F90:489,503-506,514,538` consumes
the incoming pair and `zdftke.F90:832-844` overwrites it after the solve.

The generic `TKEConfig` default remains `current_subiteration`.  Therefore the
partial DINO `nemo_paper` card, the DINO `veros` card, the NEMO-recipe/ORCA
path, every Veros ACC/global recipe, and MPAS when configured with this common
TKE kernel keep their previous numerical path and array values.  No non-oracle
card selects the new behavior.  In particular, `fidelity/nemo_recipe.py` is a
partial, non-DINO NEMO-oriented recipe with its own certificate; it remains on
`current_subiteration` rather than borrowing this DINO matched-state result.
The common state pytree necessarily gains
three optional carry slots, but they remain `None` and are not read on those
cards; this is a schema extension, not a numerical-path change.

The red-capable regression set includes a hand-computed two-interface case
with deliberately different carried/current coefficients.  It proves that
carried `avm=[3,5]` is observed as shear production and the momentum matrix
operand, carried `avt=[7,11]` is the stratification operand, and the post-solve
`en=4` produces the next `avm=[2,2]`, `avt=[1,0.5]`.  Missing carry, an ignored
carry on the legacy selector, and selector/surface shape violations fail
loudly.  Restart axis/halo identity, DINO-card selection, JIT, and finite
gradients are also covered.  The final focused CPU/fp64 run passed 145 tests.

### Ordered rerun

The requested composite target was **not reached**.  The corrected carried
`p_avm` operand itself is `VERIFIED` at exactly 0/9,920 failures, zero maximum
column error, and 4/4 southern focus columns passing.  Continuing to the next
operand in the same row shows that row 4 as a whole is still `DIVERGED`:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920 failed; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4a | carried `p_avm` operand | `VERIFIED` | **0/9,920; max `0`** | **4/4 pass** |
| 4 | complete `zdf_sh2` | `DIVERGED` | **9,920/9,920; max `73.29360955`** | **4/4 fail** |
| 5 onward | remaining TKE terms through implicit solves | `UNMEASURED` | ordered stop at row 4 | ordered stop |

All planted controls remain live: the baseline passes and the wet-cell
perturbation, one-i roll, and nonfinite injection each fire.

### New first operand: step-entry NOW velocities

After exact carried viscosity, NEMO's next factors are

```fortran
* ( uu(ji,jj,jk-1,Kmm) - uu(ji,jj,jk,Kmm) )
* ( uu(ji,jj,jk-1,Kbb) - uu(ji,jj,jk,Kbb) )
```

with the analogous `vv` factors at
`cfgs/DINO/WORK/zdfsh2.F90:81-82,86-87`.  NEMO evaluates `zdf_phy` at step
entry.  legoESM currently reaches the TKE closure after its explicit
dynamics/advection update, so its NOW factors are post-explicit and
pre-implicit-solve.

The directly scored BEFORE vertical-difference operands are bit-identical
(0 failed U or V face columns, zero maximum error).  The NOW difference
operand is not: 9,758/9,758 wet U columns and 9,868/9,868 wet V columns fail.
Scoring the literal `velocity(k-1)-velocity(k)` gives maxima `12.59807570`
(U) and `4.530529912` (V).  Each southern T-column focus score is the maximum
over its two surrounding wet faces; all four fail both operands.  U focus
errors are `1.1925297`, `0.9549570`, `1.1168380`, and `1.1182779`; V errors
are `0.1037876`, `0.1000737`, `0.1000737`, and `0.4359260`.  Substituting the
exact step-entry NOW velocities reduces the
complete shear maximum to `0.03098204` but does not close it (9,920/9,920
still fail), so later metric/four-face operands remain deliberately
unattributed.  The ordered walk stops at the first failing velocity operand.

Next-round design: add the static option `tke_shear_evaluation_stage`, with
`step_entry` as the faithful default on the two complete DINO NEMO cards and
`implicit_solve_state` as explicit legacy opt-in.  Evaluate and freeze
`p_sh2` from step-entry NOW/BEFORE velocities before explicit dynamics and
advection, carry that field to the closure, and use the same frozen field for
both the TKE shear RHS and Prandtl denominator.  Re-run row 4 before examining
the next `e3uw/e3vw` operand; the present substitution proves velocity timing
is first, not that it is the only remaining row-4 defect.

### Frozen climate prediction and GPU handoff

The prospective registration remains frozen even though the ordered
matched-state sweep found a later row-4 defect.  Baseline southern-basin
day-90 MLD RMS is `22.479491 m` (`22.4795 m` headline).

- `CONFIRM`: faithful RMS `<=11.2397455 m`; legacy control within `0.001 m` of
  `22.479491 m`; no acceptance error worsens versus control by more than its
  one-floor value; the 5x pass tally does not decrease; and at least one of the
  two southern surface-density errors improves by one floor.
- `REFUTE`: faithful RMS `>=20.2775 m`, or the legacy baseline control fails,
  or any acceptance metric worsens by more than one floor, or the 5x pass tally
  decreases.
- Between the RMS bands with valid controls is `PARTIAL/INDETERMINATE`.

The frozen one-floor values are ACC `0.091 Sv`, upper density contrast
`1.1e-4 kg m-3`, deep contrast `4.5e-5 kg m-3`, southern surface sigma maximum
`9.5e-5 kg m-3`, and mean `9.5e-5 kg m-3`.

Exact arm commands:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_carry_faithful_d90.npz --days 90 --save-3d \
  --bridge-tke

CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_carry_legacy_d90.npz --days 90 --save-3d \
  --bridge-tke --tke-preclosure-coeff-source current_subiteration

python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  /tmp/zdf_carry_faithful_d90.npz --level 5
python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  /tmp/zdf_carry_legacy_d90.npz --level 5
```

The exact MLD scorer remains the audit's symmetric NOW-state criterion with
its NEMO area/mask and southern region.  This round ran no GPU arm.

Round-3 artifact:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round3_artifact.json`, SHA256
`695efe3c8f1f65634795be7ddf99b82ea90afc7ae3b990ccf7d495e03191ab43`.
Probe/tree SHA: `57c6b21e3bbb61fccd915c8575c2cadcc8ac6de0`.

### Round-3 adversarial review

Two independent read-only rereviews ended `NON-HOLD`.  The measurement
reviewer reproduced the artifact byte-for-byte, checked all 32 provenance
hashes, the exact difference/focus populations, live controls, and the ordered
stop.  The physics reviewer checked the active NEMO lifetime against source,
the partial-depth cold-start mask, restart/pass-through and partial-carry
failures, resolved-card scope, legacy identity, surface consumption, JIT/AD,
and separation of the post-solve closure carry from EVD.  `dissl` remains a
later ordered operand; this round makes no claim that it is closed.

## Round-2 result (supersedes the row-2 stop below)

The registered raw-mesh fix is now production code.  A NEMO bridge preserves
`mesh_mask.nc:e3w_0`, and the default `mesh_reference` construction supplies
`e3w_0*(1+r3t)` to every `nemo_bn2` path and to the paired `zdf_mxl`
multiplication.  The former `diff(live_gdept)` construction remains available
only through the explicit `depth_difference` option.  Missing, malformed, or
nonpositive native geometry fails closed.

The row-2 rerun is `VERIFIED`: 0/9,920 wet columns fail, maximum column error
`5.968673256056548e-16`, and all four southern focus columns pass.  The
explicit legacy control reproduces the accepted baseline exactly: 4,630/9,920
fail with maximum `6.366584806460317e-15`.  Row 3
`eos_rab/bn2(Nnn)` is also `VERIFIED`, 0/9,920 failures and maximum
`5.968545325803916e-16`.

The next and therefore current first divergence is row 4, `zdf_sh2`:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920 failed | 4/4 pass |
| 2 | `bn2(Nbb)` native `e3w` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4 | `zdf_sh2` | `DIVERGED` | 9,920/9,920; max `7.329361e+01` | 4/4 fail (`3.036e-4` to `3.988e-3`) |
| 5 onward | bottom drag through implicit solves | `UNMEASURED` | ordered stop at row 4 | ordered stop |

### Row-4 first operand

NEMO starts its active no-Stokes expression with the carried pre-step
viscosity:

```fortran
zsh2u(ji,jj) = ( p_avm(ji+1,jj,jk) + p_avm(ji,jj,jk) ) &
```

Source: NEMO 5.0.2 `src/OCE/ZDF/zdfsh2.F90:80`; the full face products and
four-face assembly are at lines 80-94.  legoESM instead supplies the newly
reconstructed current sub-iteration `K_M_curr`.  That operand comparison is
already `DIVERGED` in 9,920/9,920 wet columns (max column error
`3.3486302069727913`, correlation `0.9984439417185291`, RMS ratio
`1.0146340961467346`); all four focus columns fail (`1.0224e-5` through
`6.8126e-5`).  The day-0 now/before velocity bridge is bit-identical, so the
ordered operand walk stops at `p_avm` before considering the later velocity,
`e3uw/e3vw`, and four-face operands.  Substituting NEMO's carried `p_avm`
alone does not close the composite because later operands remain non-identical;
that is recorded rather than misreported as a failed localization.

Next-round design: add `tke_preclosure_coeff_source` with faithful default
`carried_previous_step` and explicit legacy `current_subiteration`.  Carry
`avm/avt` closure fields across steps and seed a bridged run from restart
`avm/avt`.  Before `tke_avn`, carried `p_avm` must feed `zdf_sh2`, the
`rn2b*p_avm` Prandtl numerator, TKE matrix diagonals (`zdftke.F90:503-506`),
and wave surface denominator (`:538`); carried `p_avt` must feed the
`-p_avt*rn2` RHS (`:514`).  Only the post-solve `tke_avn` overwrite at line
621 and below constructs the coefficients carried into the next step.
Required red tests distinguish carried/current arrays at every consumer,
prove restart identity and next-step carry, lock post-solve sequencing and
legacy bits, and keep JIT/grad finite.  Per the ordered discipline, this is
design only; row 4 is not fixed in this round.

Round-2 machine-readable result:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round2_artifact.json`, SHA256
`bab31d7a8e322e855b187dcfdf0ad19cdb36b4232f1782c15be3c64730234384`.
The stamped probe/tree SHA is
`a58c33d5c7b7d4788f921d8626d9f499e29ff445`.  All planted controls fired.

### Round-2 adversarial review

Two independent final read-only reviews ended `NON-HOLD`.  The measurement
reviewer reran the committed CPU/fp64 sweep from the final production delta and
reproduced the artifact byte-for-byte (SHA256 above), verified every source and
input stamp, and passed 92 delta-focused tests.  The physics reviewer passed 25
prior-regression tests and mutation-tested the production MLD helper: replacing
only its multiplier with reconstructed `diff(gdept)` changes the doubled-native
case from 20 m/base 1 to 40 m/base 2, so the shared-e3w control is red-capable.
Review HOLDs on generic MPAS/ORCA geometry, the flat NEMO bridge, native operand
validation, row-4 provenance, and the full pre-`tke_avn` carry design were all
fixed before sign-off.

The remainder of this document preserves the accepted round-1 evidence and
design history; its statement that the sweep stopped at row 2 is historical.

## Verdict

The sweep stopped at row 2, `bn2(Nbb)`.  `eos_rab(Nbb)` is `VERIFIED`; `bn2`
is `DIVERGED` under its preregistered POINTWISE per-column bar.  The first
failing operand is construction/evaluation order for NEMO's live
`e3w(Kmm)` divisor, not different physical geometry.

The result is not visible in the aggregate statistics: `bn2` has correlation
`1.0` and RMS ratio `1.0`, but 4,630 of 9,920 wet columns exceed the
`1e-15` per-column bar.  This is exactly why the reset requires a whole-domain
column census rather than an aggregate verdict.

The complete 32-row execution-order table, active/dead branch evidence,
measurement classes, fixed bars, focus registry, controls, and stop rule were
committed before measurement in
`scripts/validate/ocean_fidelity/dino_1226/PREREG_zdf_chain_sweep.md`
(preregistration commit `b6c11c309ec5d42e3fa48645849e278724dca38d`).

## Rows reached

| Row | Operation | Disposition | Whole-domain per-column result | Focus-column result |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` alpha/beta | `VERIFIED` | alpha max `3.087467e-16`, beta max `0`; 0/9,920 failed | all four pass |
| 2 | `bn2(Nbb)` | `DIVERGED` | max `6.366585e-15`; 4,630/9,920 failed | all four pass (`1.321191e-17` to `2.020645e-17`) |
| 3 onward | `eos_rab/bn2(Nnn)` through TKE, EVD, `ldf_slp`, and the implicit solves | `UNMEASURED` | ordered stop at row 2 | ordered stop at row 2 |

The aggregate gates pass for both reached rows.  They do not override the
per-column failure.  Row 1 covers 342,134 wet T cells; row 2 covers 332,214
wet W interfaces.  The row-2 reference RMS is `6.811828981750718e-05 s-2`.

## Focus registry

The focus rule was derived from the committed MLD audit maps before this
measurement: all wet southern-basin day-90 columns where
`basin_legacy_base_index_day90 != nemo_base_index_day90`.  The map SHA256 is
`9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0`.
The resulting zero-based, halo-stripped `(j,i)` registry is

```text
(11,1), (12,1), (13,1), (13,23)
```

Those four output columns do not expose the `bn2` failure—their output scores
remain below the bar—while the whole-domain census does.  Conversely, the
underlying derived-`e3w` operand fails in all 9,920 wet columns, including all
four focus columns.  This is a direct demonstration that the MLD pattern audit
is useful for targeting but cannot decide term fidelity.

## Operand localization

The active NEMO expression is

```fortran
pn2 = grav * (zaw * dT - zbw * dS) / e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)
```

Source: NEMO 5.0.2 `src/OCE/TRA/eosbn2.F90:1465-1467`; DINO calls it with
BEFORE T/S and NOW geometry at `cfgs/DINO/MY_SRC/stpmlf.F90:206`.

Substitutions were applied one at a time, holding all later arithmetic fixed:

| Substitution | Failed wet columns | Max column error | Bar result |
|---|---:|---:|---|
| `mesh_mask:gdepw_0 * (1+r3t)` in `zrw` | 4,630 | `6.366585e-15` | fail |
| NEMO dumped `gdept(Kmm)` in `zrw` | 4,639 | `6.366585e-15` | fail |
| NEMO dumped alpha/beta | 4,632 | `6.366585e-15` | fail |
| `diff(gdept_0) * (1+r3t)` divisor | 1,361 | `1.790602e-15` | fail |
| raw mesh `diff(gdept_0) * (1+r3t)` divisor | 0 | `5.968673e-16` | pass |
| `mesh_mask:e3w_0 * (1+r3t)` divisor | 0 | `5.968673e-16` | pass |
| NEMO dumped live `e3w(Kmm)` divisor | 0 | `5.968673e-16` | pass |

The independently scored operand confirms the distinction:

- the bridged BEFORE T/S are bit-identical to the raw restart `tb/sb` at every
  registered wet point, and the live `gdepw` is at-bar against
  `mesh_mask:gdepw_0*(1+r3t)`; substituting that
  independently reconstructed `gdepw` leaves all 4,630 baseline failures;
- raw mesh `diff(gdept_0)` and raw mesh `e3w_0` are bit-identical over the
  scored interfaces; both raw-mesh constructions close the row after the live
  stretch;
- current `diff(gdept(Kmm))` versus the live dump: all 9,920 columns fail,
  maximum `5.641628e-15`;
- `diff(gdept_0)*(1+r3t)` versus the live dump: all 9,920 columns fail,
  maximum `3.173416e-15`;
- the mesh's independent `e3w_0*(1+r3t)` versus the live dump: 0 columns fail,
  maximum `3.526017e-16`.

Therefore this is not the already-closed question “is the divisor live?”—all
candidates above are live.  The remaining defect is operand construction and
floating evaluation order.  The bridge canonicalizes `gdept_0` by a wet-cell
horizontal mean (`nemo_state_bridge.py:360-364`), then legoESM differences the
already-live depth.  NEMO's raw `gdept_0` difference is exactly its raw
`e3w_0`, and it stretches that reference spacing directly.  Preserving either
raw mesh construction closes this row; preserving `e3w_0` is the most direct
transcription of the operand NEMO actually reads.  The old claim that the
bridged reconstruction is exact has been retracted in the measuring tool and
in `fidelity_bar_gate.py`; the gate can no longer print this row `AT BAR`.

This roundoff-tier whole-domain divergence is **not** claimed to explain the
22.5 m southern MLD pattern.  All four targeted southern `bn2` output columns
already pass before substitution.  The sweep must repair/reverify row 2 and
then continue to row 6 before any MLD-causality claim is possible.

## Next-round fix design (not implemented here)

Add one statically validated bridge/geometry selector, for example
`bn2_e3w_source`, with values `"mesh_reference"` and `"depth_difference"`.
It is one shared geometry choice, not independently selectable TKE/EVD flags
and not an environment variable.

- `"mesh_reference"` is the correct-by-default value for a NEMO bridge.  It
  consumes the raw reference `e3w_0` carried from `mesh_mask.nc` and forms live
  `e3w=e3w_0*(1+r3t)`.  Preserve raw `gdept_0` as well; validate that its
  reference difference agrees with `e3w_0` where the grid promises that
  identity.  If a NEMO-fidelity configuration lacks the selected operand or
  encounters a bad shape/nonpositive spacing, fail closed.
- `"depth_difference"` preserves the legacy `diff(gdept)` behavior and must be
  selected explicitly.  It remains useful for generic/synthetic coordinates
  that do not claim NEMO operand identity.

Implementation shape:

1. extend `NemoGrid`/the NEMO bridge and both vertical-coordinate wrappers to
   preserve raw `mesh_mask:gdept_0` and `e3w_0` as optional array pytree leaves;
   do not horizontally average the bit-uniform reference used for this arm;
2. add a helper returning the live native `e3w` without changing
   `nemo_bn2_live_ladders`' existing two-value API;
3. add an explicit canonical `e3w_int` operand to
   `compute_buoyancy_frequency_nemo_bn2` and thread the selector/operand through
   all TKE, EVD, MLD, GM/Redi, C-grid, and MPAS `nemo_bn2` consumers;
4. use that same exact `e3w_int` wherever the next operation multiplies by
   `e3w` (notably `zdf_mxl`) so the NEMO cancelling pair remains paired;
5. add reader halo/axis, shape/positivity, both coordinate-wrapper propagation,
   JIT, finite `jax.grad` with respect to T/S/eta, pytree, selector-typo,
   missing-mesh-operand, C-grid/MPAS parity, bit-identity legacy, planted
   last-bit divergence, and exact shared-`e3w` bn2-to-MLD cancellation tests.
   The decisive
   regression is the day-180 census: 4,630 failures must become zero and the
   maximum must be no larger than `1e-15` before the sweep may continue.

No production physics fix is included in this round.

## Controls and provenance

The focus set and map SHA were checked mechanically.  The alpha baseline
passes; a planted nonzero wet-cell perturbation makes it fail, and a one-cell
zonal roll also makes it fail.  A planted NaN at a registered wet point is
counted and fails rather than being censored.  Backend is CPU, JAX x64 is enabled, and the
artifact stamps source/dump/restart/mesh SHA256 values and time levels.  No new
NEMO dump slot or NEMO rerun was necessary: the existing PR #1689 dump family
already contained `eiv_dump_e3w.bin`, `eiv_dump_gdept.bin`, alpha/beta, and
`tke_dump_rn2b.bin`.

Machine-readable result:
`docs/ocean/fidelity/dino_zdf_chain_sweep_artifact.json`, SHA256
`949604577cf3900c259e033a3f53c3067ef3740cc50df2100621f50d0eda095b`.
The committed probe SHA stamped inside it is
`88d53850d3f2e9ca95210010caba9d39dbbc1af0`.

The host worktree's administrative Git directory is read-only in this
sandbox.  Commits were therefore made, in order, in writable shadow metadata
cloned from parent `782b0d7887277c88bcaa9c1be24eedad5447d9ae`; the final bundle
is created from that metadata and is the authoritative reachable branch ref.

GitHub issue #1455 could not be read or updated from this sandbox: the `gh`
request failed to connect to `api.github.com`.  This document is formatted as
the evidence record to post when connectivity is available; no claim of issue
publication is made.

## Adversarial review

Two independent read-only reviews ended `NON-HOLD`.  The measurement reviewer
independently reran the final CPU/fp64 probe byte-for-byte, verified every
artifact SHA, and confirmed the preregistration is an ancestor of the stamped
probe commit.  The design reviewer confirmed the source attribution and shared
geometry design.  Findings raised during review—missing `gdepw`/T/S operand
checks, nonfinite censoring, raw-mesh construction discrimination, paired MLD
geometry, exact row-5 lines, and overbroad dry-cell identity wording—were all
dispositioned in the committed probe/table/result before sign-off.
