# ORCA2 round 332 — full re-score and source-order acquisition handoff

Date: 2026-10-24

Base: `f2dcf0a8cfff`; preregistration: `77ab00003`; rescore gate:
`a80e3d0ab`.  Final disposition: **ACQUISITION_NEEDED**.  No model or
configuration value changed, and Codex did not run NEMO.

## Answer

The current-tip full-depth production-JIT ladder contains 1,560 rows.  Of
those, 207 are bit-exact, 181 are non-bit but at the registered normalized
bar, and 1,172 are over bar.  Both rung-10 labels have identical metrics on
every row.  Their first non-bit row is kt 1 stage 1 `v`: 2,032 unequal
binary64 words, maximum and RMS exactly zero, hence signed-zero differences.

The first owning statement is **PLAUSIBLE, not CONFIRMED**.  In the executing
record build, the EEN loop computes `zva` and then adds it to `pv_rhs` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90:796-802`.
legoesm computes the corresponding EEN contribution and adds it to `dv_dt` at
`ocean_pe_latlon_cgrid.py:3049`.  The admitted round-5, 326, 328 and 331
records do not contain the accumulator immediately before that add, its
`zva`, or the result immediately after it.  The certified rung is therefore
**none**: those records cannot prove that this one statement accounts for all
2,032 sign bits without moving a nonzero value.

The seven remaining post-TKE `avm/avt` cells are likewise **PLAUSIBLE**.  NEMO
literally evaluates square root, mixing-length multiplication, independent
floors and mask in
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:708-719`;
legoesm forms the raw viscosity and then its floors/Prandtl branch in
`tke.py:2262-2287`.  The admitted round-331 record contains returned energy,
final `avm/avt`, and `wmask`, but not `zmxlm` or `zmxld`.  Consequently the
round-331 phrase “downstream source-order rounding” is demoted here from a
factual label to a hypothesis: the evidence does not yet distinguish literal
NEMO operation order from a remaining transcription difference.

## Full ladder publication

The full table publishes maximum absolute difference, RMS, unequal-word
count, element count, oracle maximum, normalization scale, normalized
maximum, and class for every kt/checkpoint/field row.  It is at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/rescore/full_ladder.tsv`
(SHA-256 `66aacbaaa8828b47924dd1d249ce963cee2d8e58878d3bb70f128878971372c4`).
The machine-readable aggregate is `aggregate.json` in the same directory
(SHA-256 `6df5bf7f8eed79b513653093c08245af2163ffbffa7a0f38f11876604792af89`).
Both were produced from clean-tree commit
`a80e3d0ab90605afcc1502a2b72a1ff5ace7211d` on CPU, fp64, production JIT.

| pair / label | bit-exact | at bar, non-bit | over bar | rows |
|---|---:|---:|---:|---:|
| OMT-0 / independent | 16 | 23 | 161 | 200 |
| rung 0 / independent | 11 | 10 | 179 | 200 |
| OMT-1 / independent | 48 | 33 | 79 | 160 |
| OMT-2 / independent | 48 | 33 | 119 | 200 |
| OMT-3 / independent | 36 | 25 | 139 | 200 |
| OMT-4 / independent | 12 | 15 | 173 | 200 |
| rung 10 / given NEMO entry | 18 | 21 | 161 | 200 |
| rung 10 / independent | 18 | 21 | 161 | 200 |
| **total** | **207** | **181** | **1,172** | **1,560** |

The first non-bit row for OMT-0 through OMT-4 is kt 1 entry/restart `T`, with
151,917 unequal signed zeros and zero maximum/RMS.  Rung 0 first differs at kt
1 stage 1 `T` with the same metrics.  Both rung-10 labels first differ at kt 1
stage 1 `v`, with 2,032 unequal signed zeros and zero maximum/RMS.  Thus
R332-P1 and P2 are **CONFIRMED**.

## Offline source-order walk

For both rung-10 labels the admitted entry state is bit-exact.  The kt-1
stage-1 `T`, `S`, `u`, and `ssh` rows are also bit-exact; `v` is the first
non-bit row and differs only in zero sign.  Walking the executing NEMO order
therefore reaches EEN vorticity before any measured nonzero departure.  The
record build's scalar assignment is a credible single owner, while legoESM's
array accumulation is its shared counterpart, but subtraction of completed
stage records cannot reconstruct a signed-zero-sensitive before/add/after
triple.  R332-P3 remains **PLAUSIBLE** and triggers acquisition.

Round 331 improved post-TKE `avm/avt` from 931/929 unequal wet cells to 7/7,
with maxima `5.551115123125783e-17` and `3.469446951953614e-18`.  Exact
returned energy excludes that state as the cause, but it does not supply the
missing mixing lengths.  R332-P4 therefore also remains **PLAUSIBLE** and
triggers acquisition; “downstream source-order rounding” is not yet literally
demonstrated as NEMO's operation order.

## Operator acquisition

The thin round-332 launcher imports the reviewed round-328 acquisition driver
and its round-331 TKE extension.  It adds only write-only observers to the
exact record-build `dynvor` and `zdftke` units.  At kt 1 on both ranks it will
record:

1. `pv_rhs` immediately before the EEN `+ zva` statement, the exact `zva`,
   and `pv_rhs` immediately after it; and
2. `zmxlm` and `zmxld` at the viscosity construction boundary.

Admission first re-admits every inherited round-331 boundary and the admitted
round-328 `dyn_zdf` frames.  It then requires bitwise replay of NEMO's
before-plus-add result, a value-equal bit change somewhere across the two
ranks, active mixing-length channels, matching extents, and valid checksums.
Header, field-order, truncation, stamp, vorticity and mixing-length plants must
all refuse.

The clean preflight passed at `18af16cce`: 55 tests passed; all inherited
deterministic files matched (140); all eight writer/observer Fortran units
passed syntax; the same-program proof found 409 identical preprocessed units
and 421 identical normalized objects; and its non-observer source plant
refused.  The preflight ran no NEMO executable.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r332-93928372/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round332_source_order_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_source_order_kt1_np2`.

## Gates and disposition

- R332-P1 and P2: **CONFIRMED**.
- R332-P3 and P4: **PLAUSIBLE; ACQUISITION_NEEDED** because the registered
  before/add/after and mixing-length operands are absent.
- R332-P5: **CONFIRMED**.  With no confirmed single-statement owner, Decision
  122 is not opened and no production-call-site, cross-card, or gradient claim
  is eligible.
- Per-receipt citation gate: PASS, 4 citations, 0 unmapped, 0 failures and 0
  map-audit failures; required citation-gate pytest: 17 passed.
- Default ORCA2 landing battery plus the round-332 gate: 158 passed in
  1,039.52 seconds, including all 6 private-import ratchet tests.
- No configuration value changed: **ASKED**.
- Stop and extend the reviewed acquisition tools when a record is missing:
  **ASKED and satisfied**.

independent review: operator (Claude)

The next admissible action is the operator-only acquisition above.  Until it
is admitted, the vorticity add and the seven TKE cells have proposals, not
certified owners.
