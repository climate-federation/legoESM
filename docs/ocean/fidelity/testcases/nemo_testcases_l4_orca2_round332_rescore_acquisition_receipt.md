# ORCA2 round 332 — full re-score and source-order acquisition handoff

Date: 2026-10-24

Base: `f2dcf0a8cfff`; preregistration: `77ab00003`; rescore gate:
`a80e3d0ab`.  Post-review acquisition repair: `c9569b8cf`.
Initial disposition: **ACQUISITION_NEEDED**.  The resume-5 disposition below
supersedes it after admitting the operator's record, and the resume-7 decision
and measurements are the terminal account.  Resume 7 changes two cited model
statements but no configuration value.  Codex did not run NEMO.

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
`tke.py:2267-2292`.  The admitted round-331 record contains returned energy,
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

## Historical operator acquisition (completed in resume 5)

The thin round-332 launcher imports the reviewed round-328 acquisition driver
and its round-331 TKE extension.  It adds only write-only observers to the
exact record-build `dynvor` and `zdftke` units.  At kt 1 on both ranks it will
record:

1. `pv_rhs` before and after the production EEN call, plus the exact EEN
   contribution from a second call of the unchanged `vor_een` into
   negative-zero-initialized scratch accumulators; and
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

Resume-4 review repaired four additional acquisition defects.  The replay
scratch now uses the full 94 by 152 by 31 explicit `vor_een` extents; a
gfortran `-fcheck=bounds` interface replica accepts those extents and its
90 by 148 plant refuses.  Scratch starts at negative zero, and the compiled
fixture proves a negative-zero contribution retains its sign bit.  The record
now carries the NEMO V-point mask and uses it, rather than the T-column mask,
for vorticity counts and plants.  Finally, retained round-332 admission reads
the producer commit and binary digest only from the retained run's own
single-entry manifests, requires the producer to be a local commit object and
an ancestor of the admission commit, and verifies the recorded digest.

With the record build's exact flags (`-O3 -funroll-all-loops` and no
fast-math/FMA option), the rejected per-cell observer changed `vor_een` from
2,404 to 2,509 disassembly lines.  The repaired separate replay leaves
`vor_een` at 12,739 bytes and 3,217 disassembly lines in both objects; the
only disassembly differences are three runtime-diagnostic source-line
immediates (727--729 become 740--742).  The floating-point instruction stream,
including the production add loop, is otherwise identical.  Admission still
provides the decisive passivity proof through bitwise output comparisons.

The clean resume-4 preflight passed: 68 tests; all 140 inherited deterministic
files matched; the misplaced-TKE and inner-domain-shape plants fired; all eight
writer/observer Fortran units passed syntax; and the same-program proof found
409 identical preprocessed units and 421 identical normalized objects before
its non-observer source plant refused.  The preflight ran no NEMO executable.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r332/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round332_source_order_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_source_order_kt1_np2`.

## Historical gates and disposition (superseded by resume 7)

- R332-P1 and P2: **CONFIRMED**.
- R332-P3 mandatory VOR flip: **REFUTED**.  The no-flip path is admissible.
- R332-P4 and the remaining literal EEN source-order question:
  **PLAUSIBLE; ACQUISITION_NEEDED** because the registered operands are absent.
- R332-P5: **CONFIRMED**.  With no confirmed single-statement owner, Decision
  122 is not opened and no production-call-site, cross-card, or gradient claim
  is eligible.
- Per-receipt citation gate: PASS, 5 citations, 0 unmapped, 0 failures and 0
  map-audit failures; the required citation-gate pytest is included below.
- Default ORCA2 landing battery: 150 passed in 1,165.53 seconds, including all
  17 citation-gate and 6 private-import ratchet tests.
- Explicit private-import ratchet: 6 passed in 5.10 seconds.
- No configuration value changed: **ASKED**.
- Stop and extend the reviewed acquisition tools when a record is missing:
  **ASKED and satisfied**.

Independent reviews: **HOLD** (Claude and Argon); every acquisition-tool
finding above is repaired or converted to a measured, fail-closed admission
check; Argon separately confirmed that the repaired replay retains `vor_een`'s
arithmetic.  Crashed and refused runs are retained under `refused/`, and
`--admit-existing` for round 332 takes its producer commit and binary SHA-256
only from the retained run's recorded manifests.

The next admissible action is the operator-only acquisition above.  Until it
is admitted, the vorticity add and the seven TKE cells have proposals, not
certified owners.

## Resume 5: admitted source-order record and terminal attribution

The operator ran the reviewed launcher with `--run`.  Its retained terminal
line was:

```text
ORCA2_ROUND328_TKE332_ACQUISITION_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_source_order_kt1_np2
```

The retained `round332_source_order_admission.json` is
`PASS_R332_SOURCE_ORDER_ADMISSION`, names producer
`a00f55b202d566cbfb1a611b378c5359548ce615`, re-admits round 331, and reports
zero changed bits, zero value-equal bit changes, and zero wet changes on both
ranks.  Its two source records have SHA-256
`bbad9acd14d5949fecefd863f97498a2b887862552621f6acd82eae12baeb1d1` and
`202f05ebefa552e3819be235c5793a2f9d80bf694d0ffa26efd65e6974089797`;
all fourteen retained-output manifest checks pass.  The record is therefore
**ADMITTED**.  Resume 5 did not execute NEMO.

### kt 1 stage-1 `v`: all 2,032 words

The entire family is rank-0 global row `j=147`, `i=0..89`, `k=0..29` (2,032
mask-selected words, first `[147,0,0]`, last `[147,89,29]`).  Every word is a
dry V-fold target (`vmask=0`), while the scored stage record contains positive
zero and the production legoESM result contains negative zero.  This does
**not** observe NEMO's live dry-cell sign: the stage writer initializes the
canonical array to `+0.0` and copies only wet words at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynspg_ts.f90:1574-1606`,
then writes V through that canonicalizer at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:343-346`.
All 172,963 dry V words in the dump are therefore positive zero by
construction.  NEMO's live sign at the 2,032 scored words is **UNOBSERVED**.

At those exact words the admitted NEMO EEN accumulator before the add contains
1,620 positive zeros, 315 negative nonzero values, and 97 positive nonzero
values.  The recorded `zva` is negative zero at all 2,032 words.  The
accumulator after the add is bit-identical to the accumulator before it at
all 2,032 words; more strongly, all 399,600 recorded words on both ranks have
zero before/after changed bits because this kt-1-from-rest add is `+/-0` at
every word.  The exclusion is therefore **CONFIRMED but vacuous**: the add
could not have changed any word.
Thus the literal EEN add at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90:796-802`, and the
legoesm add at `ocean_pe_latlon_cgrid.py:3049`, are **CONFIRMED excluded** as
the first owner.

The earlier fold-owner claim is **REFUTED/unsupported**.  The executing call
is the `l_zdfsh2` arm at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:791-793`,
not lines 795--796.  For equal signed-zero sources, NEMO's `psgn*src` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lbcnfd.f90:684-717` and
legoesm's `-1*src*mask` at `ocean_model_latlon_cgrid.py:18281-18302` produce
the same sign.  The observed difference must therefore arise either from the
stage dump's dry-cell zero fill or from an upstream mask multiplication on
the source row.  The existing record cannot distinguish them and does not
authorize changing the legoESM fold assignment.

### Seven residual `avm/avt` cells

The seven global `[j,i,k]` cells are `[24,7,0]`, `[24,73,0]`,
`[109,115,0]`, `[113,38,0]`, `[116,112,0]`, `[116,112,1]`, and
`[121,112,0]`.  Returned TKE remains bit-exact at every wet cell.  At the
seven cells, candidate-minus-NEMO ULP deltas are respectively:

| quantity | seven ULP deltas in the cell order above |
|---|---|
| final `zmxlm` | `+1, +1, -1, +1, -1, -2, -1` |
| final `zmxld` | `0, +1, 0, 0, -1, 0, 0` |
| `avm` | `+1, +2, -1, +2, -1, -2, -1` |
| `avt` | `+1, +2, -1, +2, -1, -1, -1` |

All seven recorded `p_pdlr` values are exact binary64 `0.1`.  Replaying the
host-binary64 expression `(0.1 * recorded_zmxlm) * sqrt(recorded_en)` equals
the recorded NEMO `avm` bit for bit at all seven cells; multiplying that
result by recorded `p_pdlr` equals NEMO `avt` bit for bit at all seven.
NEMO's viscosity/diffusivity statements are
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:708-719`, and the
legoesm assembly is `tke.py:2267-2329`.  Therefore the seven `avm/avt`
residuals are **CONFIRMED not** a difference in NEMO's literal operation order;
they are a remaining mixing-length transcription difference already present
in `zmxlm` (and, at two cells, `zmxld`).

The offline input replay resolves that family without another NEMO run.  At
all seven cells the binding arm is `lup`: the NEMO CASE(2) surface anchor at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:621-637` plus the
successive live `e3t_0*(1+ssh/ht_0)` terms reproduces final recorded `zmxlm`
bit for bit, and recorded `zmxld` is never below it.  NEMO's owning surface
statement is lines 623--624; legoESM's first differing statement was
`_mxl0_surface_anchor` at `tke.py:262-268`, with its materialization at
`tke.py:4220-4227`.  The pre-fix production-JIT anchor
missed all seven inputs by `+1,+1,-1,+1,-1,-1,-1` ULP (the repeated column is
listed twice); eager already matched.  Materializing the NEMO source
operations makes eager and JIT exact at every required input.  The owner is
therefore in the `lup` inputs, **not** `ldown` or the `MIN`/`SQRT` combine.

The same replay corrects the Prandtl statement.  At
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:717-720`, NEMO
multiplies by `p_pdlr`; the old legoESM path divided `K_M` by `1/pdlr`.
Across 414,119 wet cells, literal multiplication has zero NEMO mismatches,
eager reciprocal division has 148,479 mismatches (64.14581315998542% exact),
and JIT reciprocal division has zero because XLA rewrites it to a multiply.
The NEMO step-entry arm now carries `pdlr` and evaluates the literal multiply
at `tke.py:2306-2329`; every other `_prandtl_number` caller retains the
reciprocal API.

### Resume-7 decision

Two single source-cited statements are confirmed and implemented: the
CASE(2) surface-anchor evaluation order and the literal `pdlr*K_M` update.
No selector, constant, recipe, deck, default, or configuration value changes.
The remaining V-sign question requires the raw post-fold record described
below, so the terminal status is **ACQUISITION_NEEDED** rather than a fold
model change.

## Resume 7: interrupted-tree audit and raw-V handoff

Every interrupted edit was diffed against `ce3732a403`.  The useful pieces
retained are: the round-328 fixture's `jpi=5,jpj=4` extents (needed to compile
the exact raw array); the round-332 writer, layout fixture and gate after
reducing them to one raw post-fold V field; the stage patch after reducing it
to one call; the reviewed source-order `zdftke` row and its existing placement
check; and the round-328 driver/test extensions needed to compile, plant,
admit and pin that raw record.  The interrupted pre/post-fold fields, fold
mask and semantic owner assertion, `lup/ldown` observer, TKE-arm writer and
fixture fields, duplicate fold calls, fold/arm plants, TKE arithmetic-object
comparison, and their call censuses were reverted.  They either encoded the
overturned fold-owner claim or acquired inputs the offline replay has now
resolved.

The remaining operator-only observer writes exactly one full declared
`jpi*jpj*jpk` binary64 array immediately after the executing
`l_zdfsh2` fold.  It applies no mask and no canonicalizer, so dry and halo
sign bits survive.  Its compiled fixture initializes the full array to
negative zero, plants one positive-zero cell and one rank-distinguishing
cell, and is compiled with `-fcheck=bounds`.  Admission checks the header,
exact extents, field order, checksum and rank, reports only a positive-zero /
negative-zero / nonzero census, and deliberately does not alter or reinterpret
the scored comparison.  Header, truncation and checksum plants must refuse.
The landing check requires exactly one call immediately after the
`l_zdfsh2` `lbc_lnk`; its misplaced-call plant moves it past `ELSE` and must
refuse.  The same-program proof still normalizes every observer unit, and the
stage object/disassembly check proves the observer leaves the floating-point
instruction census unchanged while a non-observer arithmetic plant changes
it.  The manifest, binary and producer pins and retained-run
`--admit-existing` rules remain in force.

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_raw_postfold_v_kt1_np2`.
If it shows live NEMO negative zero too, dry-cell scoring is a policy question
for the user; this round does not change the score.

The offline replay artifact
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/resume7_evidence/r332_tke_input_replay.json`
(SHA-256 `48bf6e3b74789a10071fb79b0c45491f64a405f468dc85e1d0679f9f3c526766`)
reports
`PASS_R332_TKE_INPUT_REPLAY`: seven of seven lup recurrences and both eager/JIT
anchors are exact; literal Prandtl multiplication and JIT are exact at all
414,119 wet cells; eager reciprocal division misses 148,479.  The focused
production-call-site and acquisition tests pass (30 selected tests), including
finite reverse derivatives.  Reverting either source materialization or the
literal multiply makes its production-call-site regression fail.

### Resume-7 measurements and gates

The production-call-site regressions and the full offline replay pass.  The
replay covers the six distinct anchor inputs represented by the seven cells
(one horizontal column appears at two levels), and both eager and production
JIT now reproduce all seven recorded NEMO `lup` values.  The literal
`pdlr*K_M` form is exact at all 414,119 wet cells; the pre-fix eager reciprocal
form missed 148,479, while the pre-fix JIT happened to be exact by compiler
rewrite.  Reverse derivatives through both fixed statements are finite, and
reverting either statement makes its production-call-site test fail.

The clean-commit Decision-122 remeasurement at `b0f9eb329` covered all seven
required card/label pairs plus OMT-0.  OMT-0, rung 0 and OMT-1 through OMT-4
are byte-for-byte identical to the published pre-fix row objects.  Each
rung-10 label gains 13 bit-exact rows and loses none; 91 normalized-maximum
rows improve, while the same 12 rows on each identical label have slightly
larger normalized maxima, all with relative increases below `1e-15`.  A total
of 180 rows reduce their unequal-word count and none increases it.
The recomputed first over-bar row moves later from kt 2 stage 2 `u` to kt 2
stage 3 `T`.  The registered RMS census is 176 toward, 4 away, 0 equal on
each label, while the first registered non-bit row remains kt 1 stage 1 `v`
and is equal.  The campaign classifier returns
`PASS_R278_D122_DECISION`: zero exact-row losses and an equal first registered
row on all seven pairs.  The four later RMS-away rows are the registered
census, not a Decision-122 veto.

The established three-step faithful-DINO production probe was run on both
the exact pre-fix commit `ce3732a403` and `b0f9eb329`, CPU/fp64/production
JIT.  All 66 state-leaf/step hashes are bit-identical.  The default-grid
full-model probe correctly refuses because that grid lacks the raw NEMO
`gdept_0/gdepw_0/e3t_0` operands required by its step-entry configuration;
the direct/default and all other closure callers are covered by the 311-test
caller suite, which passes.  That suite includes the anchor API, reciprocal
API, both iterative and final `compute_K_from_tke` production call sites,
prognostic TKE, Veros conventions, NEMO defaults and DINO configuration.

The clean-commit rung-10 `N=1` comparison-card gradient probe completes in
2,447.8 seconds with peak RSS 72.6 GiB.  Every reverse cotangent is finite,
with no non-finite leaves and no dry-cell gradient.  `T_dir0` reverse/forward
relative differences are `1.819369557262068e-14` for the T loss and
`1.3868976507379975e-15` for the SSH loss.  The trainable `rn_ud` scalar's T
derivative agrees reverse/forward to `1.501929835492298e-15`; its SSH
derivative is exactly zero in both modes.  Forward production parity has zero
non-bit leaves.

Resume 8 corrects the raw-V admission's RK3 after level.  The expected level
is now derived from NEMO's initialized levels and its inter-stage swaps: the
observed stage 1 uses Kaa 3.  The compiled fixture writes that value, and a
dedicated wrong-time-level plant refuses.

The clean read-only acquisition preflight passes 77 tests, all continuation,
placement, time-level, extent, bounds, checksum and arithmetic plants, eight Fortran
syntax proofs, a normalized stage-object floating-point instruction census
of 1,367, and the same-program proof (409 identical preprocessed units and
421 identical normalized objects).  It runs no NEMO executable and stops at:

```text
ORCA2_ROUND328_TKE332_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_raw_postfold_v_kt1_np2
```

On clean commit `a30a9e67b`, the per-receipt citation gate passes 16
citations with zero failures, zero unmapped citations and zero map-audit
failures; every self-test plant fires.  The default ORCA2 landing battery
passes all 150 tests in 1,042.78 seconds, and the explicit private-import
ratchet passes all 6 tests in 5.01 seconds.  Codex review: operator.  The
independent Claude review's four disagreements are incorporated in the
corrected walk above; no overturned fold-owner or live-dry-sign claim
remains.

## Resume 9: admitted live post-fold V, source owner, and landing

The operator's retained raw record ends in
`ORCA2_ROUND328_TKE332_ACQUISITION_PASS` and admission returns
`PASS_R332_SOURCE_ORDER_ADMISSION`.  Codex did not rebuild or run NEMO.  The
rank-0/rank-1 raw-record SHA-256 values are
`36f808c1...cd4a` and `3343e813...0f4b`; their complete local sign censuses
are respectively +0/-0/nonzero = 191,507/15,530/235,891 and
228,513/16,534/197,881.  The record has producer `e73db2a2f94d`, the corrected
stage-1 after level Kaa=3, and all header, field-order, truncation, checksum,
producer, time-level and arithmetic plants remain refusing.

### Confirmed live sign census

The 2,032 registered words are the rank-0 global V row `j=147`, `i=0..89`,
`k=0..29` selected by the dry source faces that feed NEMO's north-fold image.
The result is mixed, so neither proposed answer applies to the whole family:
1,975 words are case (a), while 57 are case (b).  The production result below
is after the single-statement fix.

| population | words | NEMO live +0 / -0 / nonzero | legoESM +0 / -0 / nonzero | joint N+/L+ ; N+/L- ; N-/L+ ; N-/L- | bit exact |
|---|---:|---:|---:|---:|---:|
| registered family | 2,032 | 57 / 1,975 / 0 | 48 / 1,984 / 0 | 48 ; 9 ; 0 ; 1,975 | 2,023 |
| every other dry/halo cell | 381,993 | 361,468 / 20,525 / 0 | 367,721 / 14,272 / 0 | 361,468 ; 0 ; 6,253 ; 14,272 | 375,740 |
| all dry/halo cells | 384,025 | 361,525 / 22,500 / 0 | 367,769 / 16,256 / 0 | 361,516 ; 9 ; 6,253 ; 16,247 | 377,763 |
| wet control | 415,175 | 0 / 0 / 415,175 | 0 / 0 / 415,175 | value-sign zero classes inapplicable | 415,175 |

The clean committed census artifact is
`orca2_rounds/round332/resume9_live/live_v_census.json`, SHA-256
`c68f6523d7b530eb8ca574cc0c6f8cb3771b24a7154ea4d23a8806178c577d2f`;
its terminal status is `PASS_R332_LIVE_V_CENSUS`.

Before this fix, legoESM stored -0 at all 2,032 registered words: 1,975 were
already bit-identical case (a), and the 57 NEMO +0 words were case (b).  The
57 case-(b) targets are `i=58,k=20..29`; `i=59,k=19..29`;
`i=60,k=16..29`; `i=61,k=17..29`; and `i=64,k=0..8`.  NEMO's corresponding
fold sources, at row 146 and permuted columns 122, 121, 120, 119 and 116, are
all -0.  Thus NEMO's fold multiply at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lbcnfd.f90:684-717` and
legoESM's fold at `ocean_model_latlon_cgrid.py:18281-18302` agree: -1 times
the source -0 makes the target +0.

The pre-fold source-order record closes the owner.  At those 57 mapped source
cells, NEMO records a finite negative full stage-1 V RHS before the stage
mask, a -0 add channel, the same finite-negative RHS after the add, and a +0
V mask.  NEMO's active HPG assignments write the unmasked surface and
interior V accelerations at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynhpg.f90:402,427`; its
vector stage first multiplies the completed update by V mask at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`.
legoESM instead multiplied the completed V RHS by the literal V mask earlier
in its generic tendency boundary.  That upstream multiply was therefore the
first owning statement.

The fix changes only the `rk3_vector_raw_vmask` arm at
`ocean_pe_latlon_cgrid.py:6236-6256`: it no longer writes a prematurely masked
`dv_dt`, leaving NEMO's later stage-update product to determine the dry sign.
The production-call-site AST test fails on the exact one-line revert, and the
real stage-update-plus-fold test checks both signed-zero directions and a
finite reverse derivative.  It recovers 48 of the 57 case-(b) target bits and
12,399 dry/halo bits overall, with all 415,175 wet words unchanged and exact.
The remaining nine targets are `i=64,k=0..8`: their NEMO dry HPG values depend
on live dry tracer/density operands that the canonical entry record replaces
with zero.  No unrecorded operand or index predicate was manufactured.

This live-faithful result makes the canonical stage score's policy visible.
The writer's +0 dry fill now compares against 7,534 legoESM -0 words at the kt
1 stage-1 V row, up from 2,032 before the fix, even though live-NEMO dry/halo
bit agreement improves.  No dump zero-fill was added on the legoESM side.

### Decision 122 and complete re-score

The clean production-JIT re-score covers all 1,560 rows and passes the
aggregate completeness gate.  It contains 233 bit-exact, 161 at-bar non-bit,
and 1,166 over-bar rows.  OMT-0 is 16/23/161; rung 0 is 11/10/179; OMT-1 is
48/33/79; OMT-2 is 48/33/119; OMT-3 is 36/25/139; OMT-4 is 12/15/173; and
each rung-10 label is 31/11/158 (bit/at-bar/over-bar).

Across the seven Decision-122 card/label pairs no exact row is lost.  Rung 0
and OMT-1 through OMT-4 are max/RMS-identical to the published control, and
their first over-bar row is equal.  Both rung-10 labels gain 13 exact rows,
have the registered RMS census 176 toward / 4 away / 0 equal, and move their
first over-bar row later from kt 2 stage 2 `u` to kt 2 stage 3 `T`; the old
first over-bar row moves toward.  This is `PASS_R278_D122_DECISION` under
Decisions 122 and 139.  Both labels remain row-for-row identical.  Their first
non-bit row is kt 1 stage 1 `v`, 7,534 unequal signed zeros with max/RMS zero.
The other first non-bit rows are unchanged: OMT-0 through OMT-4 kt 1 entry
`T`, and rung 0 kt 1 stage 1 `T`, each 151,917 unequal signed zeros and
max/RMS zero.

The aggregate and full table are under
`orca2_rounds/round332/resume9_rescore/`; their SHA-256 values are
`faca29a380119171e5a8a70fbcbd1a29b967d250a4c0a4db378f7ae9b532132a`
and `3b1f5c81fdd8c8551fce0bf4dc740b3cf1cc8da4125fcd05749e4384aa79914c`.

### Cross-caller, gradient, and landing gates

The faithful-DINO three-step production probe is bit-identical to the
pre-change baseline on all 66 state-leaf/step hashes and its bottom census;
the current artifact SHA-256 is
`5f977ac145525b73d311bdaacd5c7620b4199567f62afac15d3d1c8d19b503fa`.
DINO retains the default-false raw-V arm.  The affected ORCA2-grid production
caller passes all three isolated variants; the GYRE arm, atomic-mask caller,
production/revert regression and finite local gradient pass 35 tests.  A
combined 106-test JAX process aborted during compilation after many green
cases because of accumulated compiler resources; the exact affected test was
rerun alone and passed 3/3 in 334.16 seconds.

The clean production-commit rung-10 N=1 comparison-card gradient completes in
2,871.7 seconds with peak RSS 75.0 GiB.  Production and traced-forward states
have zero non-bit leaves.  Both reverse cotangents are finite, with zero
non-finite leaves and zero nonzero dry-cell gradients.  T-direction
reverse/forward relative differences are `1.819369557262068e-14` for the T
loss and `1.3868976507379975e-15` for the SSH loss.  The trainable `rn_ud` T
derivative agrees to `1.501929835492298e-15`; its SSH derivative is exactly
zero in both modes.  Rebuilt/traced construction has zero non-bit leaves.  The
artifact SHA-256 is
`e25d9fa08f44b97fc3816a03f1e7896045f51875b84c16590b491f6a518eac8a`.

The per-receipt citation gate passes 5 citations with zero failures, zero
unmapped citations and zero map-audit failures; all nine self-test controls
fire.  The exact default ORCA2 landing battery passes 152 tests in 1,273.40
seconds, and the explicit private-import ratchet passes 6 tests in 4.99
seconds.  The production change is commit `7b03e8b97`; all later commits add
only the census, receipt and citation evidence.  No configuration value or
NEMO source changed.  Codex review: operator; independent Claude review:
operator.
