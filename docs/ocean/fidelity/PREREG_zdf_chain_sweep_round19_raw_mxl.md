# Preregistration: ZDF row 19 raw buoyancy length

Outcome, 2026-08-29: the repaired deterministic bracket passed without stream
exceptions and row 19 is VERIFIED at 0/9,920 failing columns, exact unequal
wet elements 0, and 0 focus failures. The frozen receipt is
`dino_zdf_row19_raw_mxl_artifact.json`, SHA256
`f5e42f1d15cd9e823e81fa3f5b56a2b9eebd504c8ef823718c50dd9b9f3fc29b`.
The conditional hold language below is preserved as the preregistered rule
that the successful run satisfied.

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

## Second adjudication amendment — 2026-08-29

**LOUD RETRACTION:** the 13-stream/four-slot signature model above is false.
Fresh runs of the current row-19 binary produced 250 unequal values in
`sbc_dump_utau.bin`, not four. They also exposed two previously unlisted TKE
matrix streams whose complete terminal 52x199 plane (10,348 values) varied.
The defect is therefore variable-extent reads of uninitialized storage, not a
stable halo signature that can safely be exempted.

The same fresh experiment also produced one nondeterministic oracle crash:
arm B stopped with a segmentation fault after 55 of the expected 198 dumps,
then completed with all 198 dumps and the certified restart after the directory
was scrubbed and the identical binary/input was run again. This is a
human-bound receipt because the failed `run.log` was overwritten during the
retry; the surviving clean retry is `/tmp/RUN_ZDF19_DET_B.tH3A7l`. The crash
may share the uninitialized-storage cause, but that mechanism is not claimed
until the repaired writer runs discriminate it.

No stream exception is now admissible. Before row 19 can be scored:

- all 155 full-field raw writes (150 2-D records and five 3-D BN2 fields) in
  the 14 instrumented MY_SRC modules must route through zero-initialized halo
  buffers; a static census must leave no direct full-halo writer outside the
  helpers;
- TKE local work-array captures must be initialized in full, then fill only
  valid levels; the fully defined production `en` field remains a literal
  full-depth capture rather than substituting a synthetic terminal zero;
- 3-D BN2 captures must copy only defined `2:jpkm1` levels, leaving both
  terminal planes zero so `pn2`'s undefined `INTENT(out)` slots are never read;
- two row-19-ON executions and one otherwise-identical row-19-OFF execution
  must all finish normally from the certified donor and reproduce restart SHA
  `33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c`;
- every ON-A/ON-B stream must be byte-identical; every shared ON/OFF stream
  must be byte-identical; the raw MXL stream must be the sole ON addition;
- the scorer must enforce 198 ON and 197 OFF streams, require the scored run
  to be determinism arm A, and pass a copied-file one-bit plant plus a missing
  stream through the same production predicates; and
- full stream manifests, complete ON/OFF source manifests, both patches,
  ON/OFF binaries, donor/output restarts, run logs, maps, and dump are
  SHA-bound.

The prior 15-stream census is regenerated by committed probe
`zdf_dump_determinism_audit.py`; it records per-stream name/size/SHA manifests
and exercises the same copied-file exact comparator with a one-bit plant.

Until that deterministic bracket passes, row 19 is
`UNMEASURED-NEEDS-DETERMINISTIC-WRITER-BRACKET`. The unpromoted arithmetic
diagnostic remains 0/9,920, including all focus columns.
