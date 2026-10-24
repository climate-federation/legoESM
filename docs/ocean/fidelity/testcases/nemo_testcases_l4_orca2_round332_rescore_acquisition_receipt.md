# ORCA2 round 332 — full re-score and source-order acquisition handoff

Date: 2026-10-24

Base: `f2dcf0a8cfff`; preregistration: `77ab00003`; rescore gate:
`a80e3d0ab`.  Post-review acquisition repair: pending final commit below.
Final disposition: **ACQUISITION_NEEDED**.  No model or configuration value
changed, and Codex did not run NEMO.

## Answer

The current-tip full-depth production-JIT ladder contains 1,560 rows.  Of
those, 207 are bit-exact, 181 are non-bit but at the registered normalized
bar, and 1,172 are over bar.  Both rung-10 labels have identical metrics on
every row.  Their first non-bit row is kt 1 stage 1 `v`: 2,032 unequal
binary64 words, maximum and RMS exactly zero, hence signed-zero differences.

The original first-owner assignment is **REFUTED as a required NEMO flip**.
In the executing record build, the EEN loop computes `zva` and adds it to `pv_rhs` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90:796-802`.
legoesm computes the corresponding EEN contribution and adds it to `dv_dt` at
`ocean_pe_latlon_cgrid.py:3049`.  The admitted round-40 source-order record
shows instead that NEMO already has positive zero after HPG and changes no `v`
word at LDF, VOR, KEG, or ZAD on either rank.  The 2,032 stage-output
differences therefore do not establish a NEMO `vor_een` flip.  The new record
remains useful to inspect the literal EEN add, but a legitimate no-flip result
is admissible.

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
non-bit row and differs only in zero sign.  The executing order is HPG, LDF,
then VOR at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stp2d.f90:139-147`.
Reading the admitted round-40 records with their committed reader measured,
for rank 0, 216,291 positive and zero negative zeros after HPG and zero changed
words at each later boundary; rank 1 measured 254,390 positive and zero
negative zeros with the same zero-change result.  Thus the signed zeros are
present before VOR on the NEMO side.  R332-P3's mandatory VOR-flip prediction
is **REFUTED**; source-order acquisition reports changes but may not require
one.

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

1. `pv_rhs` before and after the production EEN call, plus the exact EEN
   contribution from a second call of the unchanged `vor_een` into zeroed
   scratch accumulators; and
2. `zmxlm`, `zmxld`, and `p_pdlr` immediately after mixing-length selection
   and before the viscosity construction.

Admission first re-admits every inherited round-331 boundary and the admitted
round-328 `dyn_zdf` frames.  It compares all inherited deterministic outputs,
and explicitly compares the kt-1 `pv_rhs` and stage-1 `ua/va` frames bit for
bit against the admitted record.  It then requires bitwise replay of NEMO's
before-plus-add result, active wet mixing-length channels, wet `p_pdlr` in
`[0.1,1]`, matching extents, and valid checksums.  Change and value-equal-flip
counts are reported only.  Header, field-order, truncation, stamp, wet
one-ULP vorticity, and wet one-ULP `p_pdlr` plants must all print `REFUSE`.

The original preflight was a false green because it counted calls without
checking their post-patch location.  The repaired preflight applies the
round-331 and round-332 patches in fixed order, checks the patched TKE call is
between `END SELECT` and the viscosity loop, and has a misplaced-call plant
that must refuse.  Its real gfortran fixture writes different data per rank,
preserves dry negative-zero sign bits, exercises the real writer/reader ABI,
and proves both wet one-ULP plants reach `REFUSE`.

With the record build's exact flags (`-O3 -funroll-all-loops` and no
fast-math/FMA option), the rejected per-cell observer changed `vor_een` from
2,404 to 2,509 disassembly lines.  The repaired separate replay leaves
`vor_een` at 12,739 bytes and 3,217 disassembly lines in both objects; the
only disassembly differences are three runtime-diagnostic source-line
immediates (727--729 become 740--742).  The floating-point instruction stream,
including the production add loop, is otherwise identical.  Admission still
provides the decisive passivity proof through bitwise output comparisons.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r332/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round332_source_order_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_source_order_kt1_np2`.

## Gates and disposition

- R332-P1 and P2: **CONFIRMED**.
- R332-P3 mandatory VOR flip: **REFUTED**.  The no-flip path is admissible.
- R332-P4 and the remaining literal EEN source-order question:
  **PLAUSIBLE; ACQUISITION_NEEDED** because the registered operands are absent.
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

Independent reviews: **HOLD** (Claude and Argon); every acquisition-tool
finding above is repaired or converted to a measured, fail-closed admission
check.  Crashed and refused runs are retained under `refused/`, and
`--admit-existing` for round 332 requires an explicit producer commit and
binary SHA-256 pin before it will read one.

The next admissible action is the operator-only acquisition above.  Until it
is admitted, the vorticity add and the seven TKE cells have proposals, not
certified owners.
