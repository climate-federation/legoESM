# ORCA2 round 333 — first wet-value departure and tracer acquisition handoff

Date: 2026-10-25

Base: `072e4aa1b1c8`; preregistration: `fb7aa3cffc`; measurement gate:
`c238f7232c`. Disposition: **HELD at Resume 3; acquisition admitted**. This round changes no
model statement or configuration value, and Codex did not run NEMO.
Signed-zero-only dry-cell rows were excluded exactly as instructed; no scoring
rule was changed.

## First wet-value and first over-bar rows

The two requested boundaries coincide for both rung-10 labels. The metrics
below are measured independently for each label in campaign row order. RMS is
reported both over the complete canonical row and over NEMO's wet support so
that the denominator is explicit.

| label | boundary | first row | max abs | full-row RMS | wet RMS | unequal wet | maximum `(i,j,k,rank)` | location |
|---|---|---|---:|---:|---:|---:|---|---|
| given NEMO entry | first wet value | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| given NEMO entry | first over bar | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| independent | first wet value | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| independent | first over bar | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |

The unique maximum is not the first unequal word: the latter is global
`[j,i,k]=[147,31,2]`. All 18 unequal words are wet and value-unequal. The
normalized maximum is `2.5511626431537103e-13`, over the registered
`1e-15` bar, and the oracle normalization scale is
`30.0963970850997 K`.

Salinity departs at that same kt-2 stage-3 boundary on both independently
constructed labels: 13 wet words are unequal, with maximum absolute difference
`9.65e-11`. It follows T in the registered row order and therefore does not
change the first-wet-row identity above. Every earlier T and S row contains
signed-zero differences only, never a wet value difference.

The clean CPU/fp64/production-JIT artifacts were produced at
`c238f7232cc6cf24a7243cd6bc554c599cc6e0fa`:

- `orca2_rounds/round333/first_wet/given_nemo_entry.json`, SHA-256
  `b4c72b84aeaa22d58d29299c9b13af18ba403e31beae7bedd0007226f2bcd930`;
- `orca2_rounds/round333/first_wet/independent.json`, SHA-256
  `09eb4aa025907933847c4b4644dc0be301b96d89ec52ccb085dc3f4f3fea266f`.

## Offline source-order walk

The admitted round-5/326 trajectory establishes the row endpoints; the
round-328, round-331, and round-332 records refine vertical-mixing, TKE, and
kt-1 source-order boundaries. None records the inner kt-2 stage-3 tracer
boundaries. In the executing record build, stage 3 first zeroes `Krhs` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:604`, then
executes advection at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:637`, surface
forcing at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:645`, solar
penetration at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:717`, lateral
mixing at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:734`, the
enabled bottom/damping calls in
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:738-740`, and
the implicit tracer solve at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:749` before the
already recorded completed stage and fold exchange.

The earliest possible owner in that order is therefore the `tra_adv` call at
line 637. Its legoESM counterpart is the production stage-3 `_flux_pair`
call at `ocean_model_latlon_cgrid.py:2946-2947`. This assignment is
**PLAUSIBLE, not CONFIRMED**: the existing kt-2 records jump from the preceding
completed stage to the completed stage-3 frame, so they cannot distinguish
advection from any later tracer statement. No single statement is eligible
for a Decision-122 change.

## Acquisition extension

The committed operator-only extension imports the reviewed round-328 driver
and extends the reviewed round-332 writer rather than replacing their
controls. At kt 2 stage 3 it records raw, signed-zero-preserving `Krhs` on both
ranks after zeroing, advection, surface forcing, solar forcing, lateral mixing,
and the enabled bottom-heat-flux/BBL/damping group. It then records pre-fold
`Kbb`, `Kmm`, `Kaa`, and `r3t` after the implicit solve and after the inherited
kt-1 dump block. Every field has declared `jpi,jpj,jpk` (or `jpi,jpj`) extents
and a checksum. The existing stage frame supplies the post-boundary-condition
endpoint; admission rechecks the inherited record and compares rank-0 `Kmm`
to the admitted stage-2 state.

The corrected driver adds exact landing checks at the six cited source-order
locations plus the post-damping boundary, with one misplacement plant per
observer. It compiles and runs the extended writer for two ranks under
`-fcheck=bounds`, including raw `-0.0` cells, and has checksum, truncation,
time-level, wet-zero, and wet-Kmm plants. The arithmetic comparison strips all
r332/r333 observer calls from its baseline and reports its floating-point
instruction census. `--admit-existing` is pinned to the retained run's producer
commit and binary SHA; all `admit-r333-*.log` files remain inside retained
evidence and final admission runs inside the refused-output retention trap.

At committed tool revision `251029d2c`, the clean `--preflight-only` run
passed 77 inherited tests and 6 round-333 tests, all seven placement plants,
the three enabled-selector plants, both-rank bounds/sign fixtures, checksum and
truncation plants, deterministic passivity across 140 inherited files, and
eight real-Fortran syntax proofs. The observer-free versus instrumented object
comparison reported an unchanged census of 1,367 floating-point instructions;
the same-program proof reported 409 identical preprocessed units and 421
identical normalized objects. It printed
`ORCA2_ROUND333_TRACER_PREFLIGHT_READY` and did not build or execute NEMO.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r333/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2`.

## Preregistered verdicts and stop

- **R333-P1 — REFUTED in its RMS clause.** The predicted row, maximum, and 18
  wet unequal words are confirmed, but the wet-support RMS is
  `1.561494903752516e-14 K`, not the published full-row RMS
  `1.2687664596813964e-14 K`; equal dry words still change the denominator.
- **R333-P2 — CONFIRMED.** Both labels have identical row identities, metrics,
  maximum location, wet census, and flags.
- **R333-P3 — CONFIRMED.** The maximum is on global fold row 147, rank 0, off
  the cyclic seam; the measured additional flag is under ice, not coast or
  open ocean.
- **R333-P4 — PLAUSIBLE; ACQUISITION_NEEDED.** The first wet departure is in
  the stage-3 tracer program, but the admitted records skip its first
  discriminating statement.
- **R333-P5 — CONFIRMED as a stop condition.** There is no confirmed
  single-statement owner, so Decision 122 is not opened and no production
  edit, cross-card claim, DINO/month claim, or gradient claim is made.

The per-receipt citation gate passes 8 citations with zero failures, zero
unmapped citations, and zero map-audit failures; all nine self-test controls
fire. The exact default ORCA2 landing battery passes 152 tests in 1,133.45
seconds, including the required receipt-citation test. The explicit
private-import ratchet passes 6 tests in 5.54 seconds.

No configuration value changed: **ASKED**. The next admissible action is the
operator-only acquisition above, followed by admission and offline replay.

Independent reviews: **HOLD** (Claude and Argon) on the superseded acquisition
tool. Every enumerated finding is repaired above; operator rereview is still
required before acquisition.

## Resume 2: retained-run admission review repair

Claude and Argon confirmed every earlier acquisition repair and narrowed their
remaining HOLD to the two retained-run entry paths. Both are repaired without
changing an observer, a model statement, a configuration value, or a score.

The round-333 wrapper now adapts both the canonical target and the refused-run
basename predicate from the inherited round-332 raw-V name to
`orca1ice_tracer_stage3_kt2_np2`. Thus an operator can set
`R328_EXISTING_RUN` to a retained round-333 refusal and the inherited driver
will validate it, admit it, and move it back to the canonical round-333 target
without another NEMO run.

Every `--admit-existing` invocation, including the in-place canonical path,
now reads its expected producer from the run's single-entry
`producer_commit.txt` and its binary digest from `nemo.sha256`. Admission
requires the producer to be a local commit object and an ancestor of the
admission commit, and hashes the run's `nemo` against that manifest. The
in-place path no longer substitutes current `HEAD` for the producer or depends
on a currently staged target-build executable.

The read-only preflight constructs a disposable, manifest-pinned round-333
run and passes it through both paths: first in-place at the canonical name,
then through `R328_EXISTING_RUN` under `refused/` with the round-333 refusal
basename. Its terminal control is
`R333_SYNTHETIC_RETAINED_ADMISSION_PASS routes=in-place,retained`. The complete
preflight remains green: 77 inherited tests and 6 round-333 tests, all seven
placement plants, both-rank bounds/sign fixtures, 140-file deterministic
passivity, eight Fortran syntax proofs, 1,367 floating-point instructions on
both object arms, and 409/421 identical same-program units. It ends at
`ORCA2_ROUND333_TRACER_PREFLIGHT_READY`; no NEMO executable ran.

The exact ORCA2 landing battery passes 152 tests in 1,139.58 seconds. The
per-receipt citation gate still passes 8 citations with zero failures, zero
unmapped citations, and zero map-audit failures; its self-test plants fire.
No DINO month run is required because no model or package file changed.
Codex review: operator. The two reviewer HOLD items are dispositioned here;
operator rereview remains required before acquisition.

Disposition remains **ACQUISITION_NEEDED** at the same operator command and
expected evidence directory above. This resume makes both recovery paths
usable; it does not admit or synthesize the missing tracer record.

## Resume 3: admitted tracer record and source-order replay

The operator returned the requested two-rank record with
`ORCA2_ROUND333_TRACER_ACQUISITION_PASS`.  Codex did not run NEMO.  The
committed `--admit-existing` path admitted that retained record and preserved
all refusals: the inherited round-332 admission, producer-commit ancestry,
binary digest, both record digests, field order/extents, checksums, time
levels, wet-zero control and rank-0 Kmm/stage-2 identity all pass.  All seven
header, truncation, checksum, stamp, zero, Kmm and time-level plants fire.
`round333_tracer_admission.json` has SHA-256
`37f7d773758d8590741a0cca3725e008f6b50a35fbe78eafdc0fab74c6b04ba6` and
status `PASS_R333_TRACER_ACQUISITION`.

The production-JIT replay observes the ordinary shared stage-3 Krhs walk and
implicit solve with ordered callbacks, then compares the result with a
separately compiled unobserved production step.  Both labels report zero
changed prognostic leaves and bit-identical observed/unobserved states.  The
walk is exact for both tracers after zeroing, advection, surface forcing and
solar forcing.  Its first wet value mismatch is immediately after lateral
mixing:

| boundary | T everywhere wet | T registered 18 | S everywhere wet | S registered 13 |
|---|---:|---:|---:|---:|
| after QSR | 0, max 0 | 0, max 0 | 0, max 0 | 0, max 0 |
| after LDF | 14, max `1.0587911840678754e-22` | 0, max 0 | 4, max `4.235164736271502e-22` | 1, max `4.235164736271502e-22` |
| after damping | 14, same maximum | 0, max 0 | 4, same maximum | 1, same maximum |
| pre-fold finish | 29, max `7.678080393702658e-12` | 18, same maximum | 22, max `9.652723065300961e-11` | 13, same maximum |
| post-boundary finish | 18, same maximum | 18, same maximum | 13, same maximum | 13, same maximum |

Counts are value-unequal wet binary64 words.  “Everywhere wet” contains
430,552 pre-fold cells and 233,341 rank-0 post-boundary cells.  Both the
given-NEMO-entry and independent label produce these same rows.  Their replay
artifacts have SHA-256
`6cd401600086842e7b2a546828de6d6c51b55f4fffc4a5782fd3b98f41937a24` and
`e0171f525f77ddb0a7790c37fefeb79201b28ed67814a5c9df5210cfe7254285`,
respectively.

### First owner: CONFIRMED at the recorded statement boundary

The first owning NEMO statement in executed source order is the record-build
`CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:734`:
the admitted before boundary is exact everywhere wet and the admitted after
boundary contains every first mismatch.  The corresponding shared legoESM
route places `_ldf_t/_ldf_s` in the third L8 group at
`ocean_model_latlon_cgrid.py:11048-11054` and writes that group into the live
accumulators at `:208-210`.  This call-boundary ownership is **CONFIRMED** for
both labels and both tracers; advection, SBC and QSR are exonerated.

The record does not contain the internal `zfu`, `zfv`, `zfw`, reciprocal and
divisor operands needed to name one expression inside `traldf_iso_lap`.
Moreover, the most direct single-statement proposal is refuted.  NEMO's
regular in-place update is at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:287-291`.
Replaying its add/sub increment seam against the live post-QSR accumulator
leaves exactly the same 14 T and 4 S mismatches and the same maxima as the
current direct add.  Thus neither the final in-place assignment nor the
recorder subtraction owns the residual magnitude.

### Disposition and unchanged first row

There is no confirmed single legoESM implementation statement to fix.  The
conditional landing gate therefore does not open: Decision 122, DINO, the
gridladder caller cards and the N=1 forward/reverse gradient probe are not run
or claimed.  No production implementation or configuration value changes.
The held proposal, if a later round is authorized, is an operator-only
acquisition of the internal `traldf_iso_lap` operands at the 18 T / 13 S
registered cells plus every one of the 14 T / 4 S first-boundary cells; until
then, changing the shared operator would be an uncited guess.

Because nothing lands, the first wet-value row is unchanged for each label:

| label | first wet-value row | wet unequal | max abs | wet RMS |
|---|---|---:|---:|---:|
| given NEMO entry | kt 2 stage 3 T | 18 | `7.678080393702658e-12 K` | `1.561494903752516e-14 K` |
| independent | kt 2 stage 3 T | 18 | `7.678080393702658e-12 K` | `1.561494903752516e-14 K` |

Salinity remains the following row at the same boundary with 13 unequal wet
words and maximum `9.652723065300961e-11`.  Resume 3 ends **HELD**, with no
acquisition or configuration decision needed for this disposition.

### Resume-3 gates

On clean commit `0d3ee15b1`, the full-receipt citation gate passes 11
citations with zero failures, zero unmapped citations and zero map-audit
failures; all self-test controls fire.  Its JSON SHA-256 is
`11959e7787c5abfbd2d663e7c677104298545e3e1fc6aac83be357175cc0d406`.

The default battery's first invocation passed 154 of 155 tests in 1,073.34
seconds.  Its sole refusal was the private-import anti-vacuity sentinel because
that command exposed only four of the repository's eight namespace package
roots; it reported no private import.  The corrected all-package invocation
passes the complete six-test private-import ratchet in 4.90 seconds.  Thus all
155 default test nodes pass across the split invocation, without weakening or
bypassing the sentinel.  The retained default and corrected-ratchet log
SHA-256 values are
`b3e492f001c7ca3a8466b32b5b84f93e433efcfee068a473925627c921a18b5f` and
`68084d0b99bade46badc42b33290319bf0689e5405ca0f7d6443c3ff5c155a88`.

## Resume 4: internal `traldf_iso_lap` acquisition handoff

The operator authorized the held proposal and fixed its scope: the first wet
departure is **CONFIRMED** at the `tra_ldf` call boundary for both labels and
both tracers, while advection, SBC and QSR are exonerated and the in-place
update hypothesis is refuted. This resume changes no model statement or
configuration value. Codex did not run NEMO.

The executing record build selects the standard isoneutral Laplacian at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf.f90:105-122`.
Inside that routine, NEMO computes `a33`, enters `jn=1,itra`, and then enters
the vertical-level loop at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:154-160`.
There is no `kpass` loop in `traldf_iso_lap`; the two executed passes are the
T and S iterations of this tracer loop, whose end is pinned at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:335-341`.

The passive observer follows the executing statement order. It copies the
masked tracer gradients formed at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:183-204`,
the horizontal tensor coefficients, reciprocal results and complete `zfu` and
`zfv` fluxes formed at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:230-246`,
and the vertical mask quotients, averaged coefficients, `A31/A32`, lower and
upper `zfw`, A33 operands, RHS reciprocal/divisor operands, and ordinary-level
increment inputs formed and consumed at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:259-291`.
The deepest-level zero-upper-flux case and its divisor operands are covered at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:297-304`.
NEMO forms each horizontal total and the vertical total in one statement, so
there are no separately assigned flux-component temporaries to observe; all
of their source operands are recorded instead.

Each rank-local stream contains 53 fields at explicit full declared extents,
not merely the 18 T / 13 S registered cells or the 14 T / 4 S first-boundary
cells. It includes both T and S passes, the input time level, RHS before and
after, the bitwise checked per-pass increment, gradients, tensor coefficients,
fluxes, source slopes/coefficient arrays, masks and metric operands, and
coverage arrays. Stream metadata pins rank, origin, extents, field order and
binary64 precision; every payload has an XOR checksum and raw IEEE signed
zeros are retained.

The new operator-only wrapper extends the reviewed round-333 tool rather than
replacing it. It retains every inherited landing check, deliberate
misplacement plant, deterministic-output passivity check, all-observer-stripped
baseline, raw per-rank records, bounds fixture, checksum/truncation/derived
time-level plants, wet-cell plants, manifest-pinned in-place and retained
`--admit-existing` paths, and refused-output retention trap. Added controls
cover internal begin, gradient, horizontal, vertical, deepest-level and finish
placement; field order, tracer count, extents, signed zero, wet coverage and
exact increment; and a two-rank `-fcheck=bounds` fixture.

The clean read-only preflight passes 77 inherited tests plus 6 outer-tracer
and 4 internal-LDF tests, the selector and every outer/internal placement
plant, both two-rank bounds/sign fixtures, 140-file inherited deterministic
passivity, and all ten Fortran syntax proofs. The stage object retains its
1,367-instruction census. For `traldf_iso.o`, the exact preprocessed unit with
all `R333_LDF_OBSERVER` lines stripped and the observed unit have identical
normalized FP instruction censuses of 404; a changed-arithmetic object plant
refuses. Same-program admission reports 408 identical ppsrc units and 420
identical normalized objects, with exactly six observer and four writer units.
The preflight ends with `ORCA2_ROUND333_TRALDF_PREFLIGHT_READY`; it neither
builds nor executes NEMO.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r333/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2`.

Disposition is **STOPPED_FOR_RECORD**. The next admissible action is the
operator-only acquisition and manifest-pinned admission, followed by offline
replay of both tracer passes. No implementation landing, D122 claim, DINO
month run or gradient claim is made. Codex review: operator.

### Resume-4 gates

The acquisition implementation is commit `a604a72e5`. Its clean
`--preflight-only` invocation produced every result reported above and stopped
at the read-only ready marker. The full-receipt citation gate on commit
`c69d58031` passes 18 citations with zero failures, zero unmapped citations
and zero map-audit failures; all nine self-test controls fire. Its JSON
SHA-256 is
`5e3a4e9b91835175368ca24bfafbbd521d4f35c3064e87063c9d4f53dcf65179`.

The exact ORCA2 landing battery passes all 152 tests in 990.20 seconds,
including the complete six-test private-import ratchet and the ORCA2 parallel
receipt gate. Its log SHA-256 is
`a7a8c8bc9cccbdd89129c9d96ef88a948f941c99bae3c7b38780d0818bfe8312`.
No DINO month gate is required because no model or package source changed.

## Resume 5: passive production-flag internal acquisition repair

Claude and Argon held the Resume-4 acquisition tool. All seven enumerated
findings are repaired in committed tool revision `9d6bf22c5`; no NEMO model
statement, package source, configuration value, score, or preregistered
threshold changed, and Codex did not run NEMO.

Every internal capture now occurs through an observer subroutine call after a
NEMO loop nest, never through an observer branch or copy inside the loop body.
The gradient recorder copies only NEMO's initialized `INN+1,INN,INN+1,INN`
horizontal range; the broader range remains confined to the initialized
vertical gradient. The independent observer-free and observed units are
compiled with the record build's unchanged production flags
`-O3 -funroll-all-loops -fno-tree-vectorize`. Their normalized floating-point
instruction censuses are **1,966 and 1,966**. The arithmetic mutation refuses.

The begin observer now precedes the executing `traldf_iso_a33` call. The
71-field record adds `e1u`, `e2v`, `rDt`, `r1_Dt`, the A33 mask quotients,
averaged coefficients, slope-square coefficient, raw coefficient, `ze3w_2`,
`zcoef0`, and final limited coefficient. For the ordinary Laplacian update it
adds the pre-add RHS, lower and upper vertical flux, pre-add divergence,
divisor, and post-add RHS. Time levels are derived by
`rk3_levels_for_step_stage`; every payload must be finite. Admission replays
the recorded A33 and RHS statements from these operands and requires bitwise
identity before the record can be scored.

Placement now brackets both ordinary updates and the deepest update, checks
the exact gradient and vertical lower bounds, inspects the independently
preprocessed `ppsrc/nemo/traldf_iso.f90`, and carries the same check into the
future `$TARGET_ROOT`. Plants cover begin/finish, all three A33 boundaries,
gradient/horizontal/ordinary/deepest placement, both lower bounds, and both
bitwise replay families.

The bounds fixture executes the complete 71-field writer for two rank-distinct
records, including raw negative zero and non-vacuous A33/RHS replays. A second
production-flag fixture executes the observer-bearing loop bodies on small
rank arrays and compares them bitwise with an independently written unpatched
copy; its arithmetic mutation fails. The same-program ledger is produced by
independently patching and preprocessing pristine NEMO source, not by diffing
the observed scratch file against itself.

The clean read-only preflight passes 77 inherited tests, 6 outer-tracer tests,
and 5 internal-LDF tests; all selector, placement, bounds, checksum,
truncation, arithmetic, replay, re-admission, and retention controls fire. It
passes 10 Fortran syntax proofs, inherited deterministic passivity over 140
files, both manifest-pinned admission routes, the 1,367-instruction inherited
stage census, the 1,966-instruction production LDF census, and same-program
identity for 408 preprocessed units and 420 normalized objects. It ends at
`ORCA2_ROUND333_TRALDF_PREFLIGHT_READY` and does not build or execute NEMO.

Disposition remains **STOPPED_FOR_RECORD**. Reviewer re-admission is still
required before the operator runs:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r333/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

Expected evidence remains
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2`.

### Resume-5 gates

The clean full-receipt citation gate passes 18 citations with zero failures,
zero unmapped citations and zero map-audit failures; all nine self-test
controls fire. The exact ORCA2 landing battery passes all 152 tests in
1,438.41 seconds, including the complete private-import ratchet and inherited
ORCA2 parallel receipt gate. The private-import ratchet also passes separately
at 6 of 6 tests in 6.62 seconds.

No DINO month gate is required because Resume 5 changes only acquisition
tools, fixtures, tests and this receipt. No model or package source changed.
The final state is **STOPPED_FOR_RECORD**: the next action is operator-only
acquisition with the command above, followed by manifest-pinned admission and
offline replay before any scientific conclusion or implementation landing.


## Resume 6: macro, shape, stretched-metric, and actual-source repair

Claude and Argon held Resume 5 on five remaining defects. All are repaired in
committed tool revisions `a1152eb3d` through `c15b2590f`; no NEMO model
statement, package source, configuration value, score, or threshold changed,
and Codex did not run NEMO.

The scheme patch now uses the cpp macro `INN` exactly as NEMO defines it at
`src/OCE/TRA/traldf_iso.F90:145`; no observer can resolve to the uninitialised
local integer `inn`. The placement gate rejects lowercase observer bounds on
the raw include and accepts only their legitimate numeric expansion in
independently preprocessed source. The horizontal observer passes the exact
asymmetric range NEMO fills, with conforming `zfu` and `zfv` sections rather
than their one-cell-larger declared arrays. That range and the executed
gradient and flux statements are pinned at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:183-204` and
`:230-246`.

The writer now records A11/A22 with the executed `r3u/r3v` stretching and the
ordinary and deepest RHS divisors with the executed `r3t*tmask` stretching.
The record retains `r3t`, `r3u`, and `r3v` at derived stage-3 `Kmm`. Admission
replays bitwise from raw operands every recorded gradient, tensor coefficient,
mask quotient, `zfu`, `zfv`, `zfw`, divergence, divisor, RHS addition, and A33
statement. The cited ordinary vertical path is
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:259-291`; the
deepest path is `:297-304`. A dedicated scheme-replay mutation refuses.

The loop control no longer contains a hand-written observed kernel. It compiles
the actual patched `traldf_iso_scheme.h90` beside an actual pristine copy under
`-O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds`, runs both ranks
with rank-distinct nonzero `r3*` data and raw negative-zero cells, compares the
states bitwise, and applies the complete admission replay to both output
records. Its arithmetic mutation refuses. Placement additionally enforces
`zdkt < gradient observer`, `A33 first observer < raw statement < limited
statement`, and `ordinary after observer < deepest before observer`; the
driver exports `PYTHONPATH` before either embedded Python use.

The clean read-only preflight passes 77 inherited tests, 6 outer-tracer tests,
and 5 internal-LDF tests; every selector, placement, lowercase-macro, shape,
bounds, signed-zero, checksum, truncation, arithmetic, full-replay,
re-admission, and retention control fires. It passes 10 Fortran syntax proofs,
140-file deterministic passivity, both manifest-pinned admission routes, the
1,367-instruction inherited stage census, and the unchanged **1,966/1,966**
production LDF census. Same-program identity remains 408 preprocessed units
and 420 normalized objects. It ends at
`ORCA2_ROUND333_TRALDF_PREFLIGHT_READY` and neither builds nor executes NEMO.

Disposition remains **STOPPED_FOR_RECORD**. Operator rereview is required, then
the operator runs:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r333/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

Expected evidence remains
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2`.
Codex review: operator.

### Resume-6 gates

The clean full-receipt citation gate passes 19 citations with zero failures,
zero unmapped citations and zero map-audit failures; all nine self-test
controls fire. The exact ORCA2 landing battery passes all 152 tests in
1,118.94 seconds, including the inherited ORCA2 parallel receipt gate. The
private-import ratchet also passes separately at 6 of 6 tests in 5.02
seconds.

No DINO month gate is required because Resume 6 changes only acquisition
tools, fixtures, tests and this receipt. No model or package source changed.
The final state is **STOPPED_FOR_RECORD**: the next action is operator-only
acquisition with the command above, followed by manifest-pinned admission and
offline replay before any scientific conclusion or implementation landing.

## Resume 7: admitted internal operands, confirmed divisor, held proposal

The operator-produced internal-LDF record was admitted in place with
`--admit-existing`; Codex did not build or execute NEMO. Admission returned
`PASS_R333_TRALDF_INTERNAL_ACQUISITION` for both T and S passes, 71 fields,
both ranks, finite payloads, complete wet coverage, signed-zero preservation,
checksums, truncation, time levels, field order, full source-statement replay,
and retained-run manifests. Every refusal remained live, including arithmetic
scheme/RHS replay, non-finite, wet-coverage, signed-zero, checksum, truncation,
and producer/binary-stamp plants. The self-consistency replay accepted both
the dry and coastal paths without repair. The admitted JSON has SHA-256
`c57f475f0d9dba30725f44220b3405d3ca34c98ea7879fade509e373d214f220`.

### Source-order operand replay

The offline production-JIT replay walks NEMO order independently for each
label and tracer: `dit`, `djt`, `dkt`, `A11`, `A22`, `hmsku`, `hmskv`, `A13`,
`A23`, `zfu`, `zfv`, `vmsku`, `vmskv`, `zahu_w`, `zahv_w`, `A31`, `A32`,
lower and upper `zfw`, `r1_e1e2t`, the T-cell divisor, scaled tendency, and RHS
addition. Earlier bit differences are signed-zero-only; every earlier value is
exact at all 18 registered T words, all 13 registered S words, and everywhere
the recorded statement executes wet. Observer passivity is bit-identical for
two calls with zero changed state leaves.

The **first value-unequal internal operand is CONFIRMED** as `e3t_divisor` for
both labels and both tracers. Everywhere executed wet it differs at 29 words,
maximum `5.684341886080802e-14`, RMS `7.537596267255227e-17`. At the registered
first-departure words it is exact at 18/18 T words and differs at 1/13 S words
by `1.7763568394002505e-15`; the later scaled-tendency and RHS rows inherit that
S difference. Thus this is the first internal operand owner, but it does not
explain the 18 T endpoint words, and the campaign's first-wet T row cannot move
under this proposal.

The owning ordinary and deepest NEMO statements divide the accumulated flux
divergence by the already stored
`e3t_3d*(1+r3t(Kmm)*tmask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:301-305`
and
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:318-322`.
The producing record build materializes the half-stage `r3t` in
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:210-229`.
legoESM already materializes the corresponding `_h_s3_kmm` at
`ocean_model_latlon_cgrid.py:8783-8797`, but the shared operator chooses its
divisor at `gm_redi_latlon_cgrid.py:3264-3269` and the production call supplies
only rebuilt half-stage sea level, not the stored thickness, at
`ocean_model_latlon_cgrid.py:10630-10643`.

The two replay artifacts return `PASS_R333_TRALDF_REPLAY` and name
`e3t_divisor` as first value owner. Their SHA-256 values are
`919e7a163543e4ed4239242525a65a148af90ee7d1a62e93343e2d09fe6324de`
(`given_nemo_entry`) and
`28f688050b3e7ca4677ce02046c99203b9e71c31d04c362d27cc54f6eaa6f175`
(`independent`).

### Proposal, measurements, and gradient falsifier

The tested single-statement proposal passed the already materialized
`_h_s3_kmm` to the one shared divisor override at the production call site.
Its exact call-site revert failed the provisional test. At candidate commit
`086a971a5357369cb54f0fc8a83440fb9e2f694e`, Decision 122 passed all seven
pairs: zero exact rows lost, every registered row equal, and both rung-10
labels equal on all 200 rows. The decision artifact SHA-256 is
`67edc2f9bc9e0d149c2662951eb5a35f13cbcec4e705894b0a223b5cff5c7017`.
The candidate's preliminary grid-ladder battery passed 111 tests, and its
three-step faithful DINO probe reproduced all 66 baseline leaf-step hashes;
the DINO artifact SHA-256 is
`5f977ac145525b73d311bdaacd5c7620b4199567f62afac15d3d1c8d19b503fa`.
These are candidate diagnostics, not landing claims.

The mandatory rung-10 N=1 gradient gate falsified the proposal. The direct
stored-divisor candidate at `1881e0df69b344b70916b00ae07ec28f39b5c855`
produced 799,200 non-finite T cotangents for both T and SSH objectives and NaN
reverse `rn_Ud` gradients, while forward `dL/dln(rn_Ud)` was finite at
`-4.511704487056347e-05`; artifact SHA-256
`63a138bf22b184e39955a7643a6d00eec06e4395ceaaf5bcbadb3c7184792393`.
A rebuilt-tangent custom JVP and an explicit stored-graph cut retained the
same all-NaN reverse result, with artifact SHA-256 values
`4444c35e5b5371dd991443630c1a87f07411fbc9dc79a984885c27b41dab523d`
and `13904193eaaf2953260f5e50fac54a82263bf2b1934422646032d49ff5a270e9`.
Finally, the ordinary-primitives straight-through form passed its isolated
forward/reverse/poisoned-graph test but failed the production scalar gate after
1,277.95 seconds: both `dL/draw` and `dL/dln(rn_Ud)` were NaN. The same-tree
legacy call-site control returned finite `dL/dln(rn_Ud)=-4.511811e-05`, so the
gate itself is discriminating. The required finite gradients and
reverse-equals-forward condition are therefore not met.

The candidate re-score leaves the first wet-value row unchanged for both
labels: kt 2, stage 3, T; 18 unequal wet words, maximum
`7.678080393702658e-12 K`, full RMS `1.2687664596813964e-14 K`, wet RMS
`1.561494903752516e-14 K`, first unequal `[147,31,2]`, unique maximum
`(i,j,k,rank)=(32,147,5,0)` on the north fold under ice. The candidate artifacts
are `given_nemo_entry` SHA-256
`1448e717915ab1eb7c2bf732041c83937582f500cdfb1098596d50b30e6df05a`
and `independent` SHA-256
`1cbea4258d1beeed550c37da00a62a89ba46d9ba7ad2d9df64a81cd00695160e`.

Per the round condition, work stopped at the failed gradient gate. No DINO
month claim is made, and the provisional production implementation and its
landing test were withdrawn. The final tree changes no package source and no
configuration value; relative to incoming `3895dbeb307c`, it retains only the
offline statement replay, the Decision-122 census tool, citation mappings,
and this receipt. Codex review: operator.

Disposition: **HELD — PROPOSAL ONLY**. The confirmed internal divisor mismatch
is recorded, but no implementation is eligible to land until a forward-exact
form also has finite rung-10 N=1 cotangents with reverse equal to forward.

### Resume-7 final gates

The full-receipt citation gate passes 25 citations with zero failures, zero
unmapped citations, and zero map-audit failures; all nine self-test controls
fire. The exact default ORCA2 landing battery passes all 152 tests in 1,030.79
seconds, including the inherited parallel-receipt gate. The private-import
ratchet also passes separately at 6 of 6 tests in 5.36 seconds. The DINO month
gate is not applicable to the final held tree because no package/model source
change remains; no month-gate claim is made for the withdrawn proposal.

## Resume 8: post-LDF owner and reverse-safe stored divisor landing

Resume 8 performed the two requested measurements in order, offline and
without rerunning NEMO. No configuration value changed.

### Measurement A: the 18 registered T words

Both the `given_nemo_entry` and `independent` replays pass. At all 18 registered
T words, the recorded divisor, actual live scaled LDF tendency, independently
recomputed scaled tendency, and LDF RHS addition are bit-exact. The outer
statement walk then remains exact at those same 18 words through the
pre-LDF accumulator, LDF term addition, live accumulator after LDF, the
live-versus-reconstructed check, and the accumulator after BBC, BBL, and
damping. All 18 first become value-unequal on return from the implicit vertical
tracer solve, with maximum `7.678080393702658e-12` and RMS
`1.8904194483901373e-12`; the subsequent recorded boundary is unchanged.

The earlier global `after_ldf` classifier is explained by 14 other wet T cells
whose live LDF RHS addition differs; none is one of the 18 registered target
words. Thus the call-boundary premise was too coarse for these targets.
NEMO's observed sequence is the LDF call at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:745-771`:
the target words are exact before and after `tra_ldf`, remain exact after the
optional BBC/BBL/damping calls, and first differ after `tra_zdf`. legoESM's
source-ordered accumulator is materialized at
`ocean_model_latlon_cgrid.py:192-212`, the active rung-10 path calls the
implicit vertical mixer at `ocean_model_latlon_cgrid.py:11706-11738`, and the
literal tracer solve dispatch is at
`ocean_model_latlon_cgrid.py:14824-14849`.

The owning statement is therefore **CONFIRMED** as NEMO's `CALL tra_zdf` and
legoESM's `_apply_implicit_vertical_mixing` / literal tracer-pair solve, not the
LDF accumulation, masking, or an intervening fold/halo step. The admitted
record does not expose a finer recurrence boundary inside that solve, so the
specific internal solve primitive that creates the ULP-scale difference is
only **PLAUSIBLE**, not claimed as confirmed. The four Measurement-A artifacts
have SHA-256 values `4e45a2f885429478fcd2eef4d11db9ffa4974e996e3587acc200f3e93dec8bd6`,
`c3ab7c94c6252800aa3cb3c2fb135b95ce946e3fcf089186562920cb04747698`,
`b6a69173a8cfff827ae883b33b5a655c162cfa57aee97092b72bf43302eedbfa`,
and `02fb8383ade7f1c12b0ae0d8bb65fec3524eb0f2772e4101e0b4a7542767579b`.

### Measurement B: first NaN and reverse-safe formulation

The smallest full-shape reproducer contains 799,200 tracer cells: 430,552
active and 368,648 inactive. The stored partial-cell divisor is nonzero at
every active cell and exactly zero at all 368,648 inactive cells, comprising
62,438 coastal/sub-bottom cells and 306,210 cells in fully dry columns. The
rebuilt divisor is nonzero on those inactive cells. The first reverse-producing
primitive is the transpose of division on the stored zero divisor: checkify
reports division by zero, `jax.debug_nans` reports NaN in `div`, and a
zero-cotangent primitive VJP produces NaN bars for numerator and divisor. The
following `tend * act` mask then also forms a NaN mask cotangent. This is the
rectangular legoESM evaluation at
`gm_redi_latlon_cgrid.py:3288-3304`; NEMO instead executes only wet tracer
cells in its divisions at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:301-305`
and
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:318-322`.
This explains why the stored path poisons reverse mode while the rebuilt path
does not. The localization artifact passes with SHA-256
`d99d0d1f912ddbf4a244d7532e93f2a21bcfcff597ebccea8ffd1528d2314b6e`.

The landed form uses the stored divisor at active cells and an inert unit
divisor at inactive cells. Its helper is
`ocean_model_latlon_cgrid.py:215-231`; the production call, including the
diagnostic-override precedence, is
`ocean_model_latlon_cgrid.py:10643-10659`. This leaves every NEMO-executed wet
word forward-exact while removing all dry zero divisions. The production
call-site test fails when this choice is reverted.

Decision 122 passes all seven control/candidate pairs and both labels: no exact
row is lost and every registered row direction remains equal. Its decision
artifact SHA-256 is
`6fcd29325220ed33cc1b1d8883d3b5a3ee540b866a8f99f0f2036f8839679dc7`.
The faithful three-step DINO probe reproduces all 66 accepted leaf-step hashes;
its artifact SHA-256 remains
`5f977ac145525b73d311bdaacd5c7620b4199567f62afac15d3d1c8d19b503fa`.

The rung-10 N=1 gradient gate completes with production/traced forward parity
and no non-bit leaves. Both T and SSH reverse cotangents are finite, with zero
non-finite cells and zero nonzero dry cells. For the T objective, the
directional reverse derivative is `-0.04157351924843481` versus forward
`-0.04157351924843511` (relative difference
`7.176982927160828e-15`); reverse `dL/dln(rn_Ud)` is
`-4.511810759103504e-05` versus forward
`-4.511810759103502e-05` (relative difference
`3.0037889175036433e-16`). The SSH directional relative difference is
`2.2687815985495016e-15` and its `rn_Ud` derivatives are exactly zero. The
gradient artifact SHA-256 is
`d5f593c4aa4c49eb53ffca14444f29146a2b1e585f2432875cdb93aa25f3426d`.

Disposition: **LANDED**. The reverse-safe stored divisor meets the forward
fidelity, DINO, and gradient conditions, with a revert-sensitive production
test and no configuration change. Codex review: operator.

### Resume-8 final gates

The Resume-8 citation gate passes all 9 citations with zero failures, zero
unmapped citations, and zero map-audit failures; all 9 self-test controls fire.
The exact default ORCA2 landing battery passes all 152 tests in 1,261.12
seconds. The private-import ratchet passes separately at 6 of 6 tests in 5.67
seconds, and the revert-sensitive production-call-site suite passes all 3
tests in 1.43 seconds. The operator-owned DINO month/push gate is not claimed
here; the requested faithful DINO probe is the passing 66-hash measurement
reported above.

## Operator amendment (2026-10-25, after Claude and Argon reviews: both LAND)

Both reviews independently confirmed the reverse-safe stored-divisor landing:
the LDF and damping accumulators are exact against NEMO at every wet cell for
T and S (the legacy divisor missed 14 T and 4 S words), no dry-cell bit
changes, the post-solve state is unchanged, reverse cotangents are finite with
rn_Ud reverse -4.5118108e-05 against a central difference -4.5118107e-05
(relative 2e-8), Decision 122 rows and DINO 66/66 are unchanged, and only the
rung-10 rk3_kmm call site changed. Claim A is confirmed at the walk level
(the 18 registered T words first differ after tra_zdf); its internal primitive
stays PLAUSIBLE. CORRECTION: the sentence above that NEMO "executes only wet
tracer cells in its divisions" is wrong — NEMO's LDF update also runs at
166,259 dry rank-0 cells, where its stored e3t is nonzero, and the result is
masked; the helper's docstring was corrected to say so (no line shift).

## Resume 9: rebase onto the lane tip after round 334

The approved round (51 commits, previously based on `072e4aa1b1c8`, tip
`d8d4104a4`) was rebased onto the lane tip `d95e9a1237` (round 334, CI ratchet
cleanup). No model change beyond the approved one; no configuration value
changed. The pre-rebase tip is kept as local branch
`backup/r333-pre-rebase-d8d4104a4`.

### Conflicts and resolutions

Only `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`
conflicted, twice, both in the same hunk at the head of `CITATION_MAP`:

| round-333 commit | hunk | round-334 side | round-333 side | resolution |
|---|---|---|---|---|
| `397bbafaf` record round333 first wet departure | 1 | 3 entries (`trabbc.F90:179`, `phycst.F90:26`, `tramle.F90:51`) | first-wet source-order walk entries | keep both: round 334's block, then round 333's |
| `437bd07a0` record held round 333 divisor proposal | 1 | same 3 entries | Resume-7 internal traldf ownership entries | keep both: round 334's block, then round 333's |

The resolver refused any conflict whose lane side was not exactly round 334's
three-entry block. Post-rebase checks: the lane tip is an ancestor of the new
tip; the round's diff outside the citation gate is byte-identical to the
pre-rebase diff; and in the citation gate, both round 333's and round 334's
changed lines are present unchanged.

### Admission replay

Re-running both admission gates on the retained records, pinned to their
producer commits, reproduces every admitted JSON digest:
`round333_tracer_admission.json` for the stage-3 tracer record
`37f7d773758d8590741a0cca3725e008f6b50a35fbe78eafdc0fab74c6b04ba6`, and for
the internal-traldf record `efaaa8396c5b2c6eb7d78269299078caf46b75aa35837e22ab946d4f01526ed6`;
`round333_traldf_admission.json`
`c57f475f0d9dba30725f44220b3405d3ca34c98ea7879fade509e373d214f220`.
All three report PASS.
