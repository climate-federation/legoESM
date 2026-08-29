# Preregistration: ZDF row 19 raw buoyancy length

Date: 2026-08-28. Ordered predecessor: row 18 VERIFIED at the registered
`1e-15` per-column bar (0/9,920 failing columns, including all four southern
focus columns). This receipt must not be used if row 18 ceases to clear.

## Oracle line and missing slot

DINO's live override evaluates, in order:

```fortran
! cfgs/DINO/MY_SRC/zdftke.F90:831-833
zrn2 = MAX( rn2(ji,jj,jk), rsmall )
zmxlm(ji,jk) = MAX( rmxl_min, SQRT( 2._wp * en(ji,jj,jk) / zrn2 ) )
```

The existing `tke_dump_zmxlm.bin` is written only after the `nn_mxl=3`
downward/upward scans and geometric-mean assembly. Add one write-only
`tke_dump_zmxlm_raw.bin` capture immediately after the loop above and before
the first physical-limit scan. It is accumulated by `jj` into a SAVE'd 3-D
array and written in the same bracket stream convention as the existing
length dumps. Registered size: 2,980,224 bytes (56x203x36 binary64).

## Bars and dispositions

- State/lane: day 180, `kt=5761`, one rank, fp64 CPU.
- Census: every wet W element and 9,920 wet columns; the four registered
  southern-basin MLD focus columns are reported separately.
- Row 19: per-column max absolute raw-length error <= `1e-15` m. `VERIFIED`
  requires 0/9,920 failures and all focus columns pass. Otherwise `DIVERGED`
  and the first owner is localized in source order to `zrn2`, `2*en/zrn2`,
  `SQRT`, or the `MAX(rmxl_min,...)` result.
- Exact census is also reported, but is diagnostic rather than a stricter bar.
- Row 20: after substituting the NEMO raw row-19 field into legoESM's literal
  `nn_mxl=3` scans, both `zmxlm` and `zmxld` must have 0/9,920 columns above
  the registered accumulating bar `1e-12` m, with all focus columns passing.

Controls that must fire: +1 ULP in one wet raw element fails the exact census;
a one-cell i-roll fails the row-19 bar; a +`2e-12` plant in one wet final
length fails the row-20 bar. Any control miss invalidates the receipt.

## Write-only bracket

Patched and unpatched runs start from the identical donor restart. Both output
restarts must be byte-identical. Every shared `*.bin` physics stream must be
byte-identical except `cor2d_dump_zu_trd_substep1.bin`, which carries the
already-adjudicated determinism exclusion: two executions of the identical
unpatched binary differ beginning at byte 3 while their restarts are identical.
The new raw stream must be the only additional TKE dump. SHA256 stamps bind the
source, binaries, donor/output restarts, dump, maps, and active NEMO source.

## Dated adjudication amendment — 2026-08-29

This is a loud post-run amendment, not part of the original preregistration.
The first row-19 run exposed a defect in the oracle's historical debug writers:
13 shared streams (the formerly known `cor2d_zu` stream plus 12 more) write
four uninitialized halo binary64 slots. The writers all use `STATUS='REPLACE'`;
append mode and a stale copied stream are therefore refuted. Every physical
interior slot is bit-identical, and two identical-binary controls show the same
four changed indices. The original whole-file bracket is retracted for exactly
these 13 streams only.

Promotion now requires two fresh executions of the **current row-19 binary**
from independent clean copies of the same donor/template. Each run records the
binary SHA before execution. The scorer requires:

- byte-identical output restarts and exact donor/source/binary SHA bindings;
- exact byte identity for every ordinary shared stream;
- for the 13 named defective streams, exactly four changed binary64 slots,
  identical changed-index signatures between patch and current-binary control,
  and zero changes inside the `nn_hls=2` physical interior;
- a planted one-bit physical-interior change that must fail the exclusion;
- the actual JIT-compiled production `_tke_raw_mixing_length` selector, not a
  duplicated NumPy expression;
- the original **absolute** per-column bar of `1e-15 m` (not RMS-normalized),
  0/9,920 failures, all focus columns passing, and an exact census.

The source-formula NumPy calculation remains only an operand-localization
self-check. Until the same-current-binary pair exists, row 19 remains
`UNMEASURED-NEEDS-CLEAN-BRACKET`; its diagnostic 0/9,920 result is not promoted.
