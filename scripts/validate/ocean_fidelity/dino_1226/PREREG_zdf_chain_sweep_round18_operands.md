# ZDF chain sweep row-18 operand preregistration

Date frozen: 2026-08-28. Status: **not yet built or measured**.

Row 18 remains `DIVERGED`: the literal glibc vector EXP clears the dumped
NEMO argument, while the production argument leaves 21/9,920 wet columns over
the registered `1e-15` pointwise bar (all four southern focus columns pass).
The unresolved NEMO expression is
`-gdepw(ji,jj,jk,Kmm) / htau(ji,jj)` at `zdftke.F90:646` in the active
instrument source. This round separates its two operands before considering
any further production change.

## Write-only slots

Apply `nemo_row18_gdepw_htau.patch` to the already-instrumented DINO source
whose `zdftke.F90` SHA256 is recorded by the build receipt. The patch adds:

- `tke_dump_etau_gdepw.bin`: `gdepw(ji,jj,jk,Kmm)` copied immediately before
  the active division;
- `tke_dump_etau_htau.bin`: `htau(ji,jj)` copied at the same site and repeated
  over `jk` to preserve the canonical `(jk,jj,ji)` stream shape.

The arrays are observation-only. The live assignment remains textually
unchanged and does not read either dump array. Both files are interior fp64
streams of shape `(52,199,36)`, i-fastest, registered at the `now` level.

## Frozen measurement and bars

The rerun must use the day-180 one-rank CPU lane, restart
`DINO_00005760_restart.nc`, `kt=5761`, and the existing dump-loader/time-level
registry. It must reproduce the four registered focus columns before scoring.

For every wet level and column, measure in this order:

1. legoESM's carried live `gdepw(Kmm)` versus the direct NEMO gdepw dump;
2. legoESM's resolved `htau` versus the direct NEMO htau dump;
3. `-gdepw_NEMO/htau_lego` and `-gdepw_lego/htau_NEMO` versus the dumped NEMO
   argument;
4. each hybrid argument through the already-verified literal vector EXP and
   full left-associated row-18 addition.

Operands 1--2 and hybrid arguments use the pointwise column bar `1e-15` from
the parent preregistration. The full row must return `0/9920` failing columns,
with `0/4` focus failures. The first one-at-a-time NEMO substitution that
clears the full row owns the divergence; if neither clears it, row 18 remains
`DIVERGED` and the interaction is localized without assigning a false owner.

## Controls and bracket identity

Before citing either new file:

- compare a dump-disabled and dump-enabled one-step run from the identical
  donor restart; every shared normal output and restart variable must be
  bit-identical;
- record SHA256 for source, patch, executable, donor restart, every consumed
  dump, and sorted per-run file manifests;
- perturb one wet dumped gdepw value by one representable ULP and require its
  operand check to fail;
- perturb one wet htau value by one representable ULP and require its check to
  fail;
- roll one new dump by one horizontal cell and require the whole-domain census
  to fail.

Home storage is quota-blocked at preregistration time. No NEMO build or run is
authorized in this commit; the patch is deliberately stopped before that
step. Rows 19--32 measured meanwhile are labeled `PROVISIONAL-DOWNSTREAM` and
cannot advance the ordered verified frontier past row 17 while row 18 is open.
