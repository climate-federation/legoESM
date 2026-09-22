# NEMO testcase L2 GYRE Round 154 receipt — developed FCT walk

Date: 2026-09-22  
Status: **STOPPED_FOR_RECORD** — no physics or configuration changed.  The
day-180 production FCT block first receives a non-bit stage-3 transport, so no
statement inside FCT is an admissible owner.  A syntax-proven acquisition for
the missing transport operands is ready for the operator.

## Frozen scope

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round154.md`, committed as
`3b9201796`.  The authoritative measurement commit is `6f1a525ea`; the
acquisition and this receipt are later diagnostic-only commits.  Evidence is
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round154/`.

No model physics, default, configuration, carried state, restart schema,
stabilizer, year harness, reconciliation gate, freshwater pair, or #1484
guard changed.  The production change is a private, write-only FCT trace whose
ordinary return is tested byte-for-byte.  Therefore no ladder, month, DINO, or
year candidate arm exists in this round.

## Round-153 refusal reconciled before use

The operator's Round-153 run reached `STOP 0` but its original post-run gate
refused one inherited private stream.  The mandatory discrimination found:

* every carried restart at steps 0, 1, 2, 10, 60, 1080 and 1081, plus
  `mesh_mask.nc`, is byte-identical to the Round-148 source run;
* the process-budget stream is byte-identical; and
* exactly 11 private diagnostic streams changed.  They are registered in the
  preregistration and in `round153_developed_fct_admission.json`.

The compiled writer adds observation calls and temporaries around the FCT
loops, which can change diagnostic materialisation while leaving all carried
state unchanged.  The corrected admission therefore requires byte identity
for all state files, the exact 11-file changed-diagnostic registry, and a
self-consistent manifest for the producing build.  Its restart-byte plant
exits nonzero.  The admitted record has 61 fields and direct non-unity limiter
coefficients for both T (624) and S (1,027).  Round 153's original prediction
that every inherited private stream would remain byte-identical is
**REFUTED**, not relabelled.

## Compiled source path

This paragraph cites only the compiled branch that produced the record.
Stage 3 establishes Kmm = N+1/2 and forms horizontal volume transports from
the live Kmm face thickness, Kmm velocity, and barotropic correction at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:260-314`.
The program calls `tra_adv_trp` at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:785` before passing
those transports to `tra_adv` at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:879`.
The resolved vector-form arm activates vertical-transport diagnosis at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:195`; `wzv` and the
metric-bearing `pFw=e1e2t*ww` write are at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:266-280`.

FCT receives Kbb/Kmm tracers and those completed `pU/pV/pW` transports at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:176-178`.
Its first upwind faces, first divergence, and midpoint write occupy
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:515-553`;
the active U/V/W coefficient selection and multiplication are at
`GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:924-953`.

## Method and controls

The existing developed-state gate loads NEMO's admitted step-1080 restart,
executes one complete `LatLonCGridOceanModel.step` through the production JIT,
and observes the shared FCT implementation without substituting any NEMO
operand.  The same values are reported under production eager and an isolated
JIT closure, with the three labels kept distinct.  Every one of the 61 record
fields is mapped and scored; NEMO `r3t` context rows are compared as the live
thickness operand `e3t_0*(1+r3t)` rather than through a lossy inverse.

Controls:

* observed versus ordinary production state: 0 unequal bytes;
* branch observer versus the existing process/vertical rows: 0 unequal cells;
* all 360 daily restarts and all 12 monthly overlaps re-admitted at the
  measurement commit;
* the production transport ULP plant changed exactly one `T.first_u` cell,
  by `2.3283064365386963e-10`, printed `STATUS PLANT-FIRED`, and exited 1;
* FCT trace unit test: 28 passed; developed registry focused test: 5 passed;
  Round-153 admission test: 8 passed; Round-154 acquisition test: 3 passed.

The first measurement attempt correctly refused an over-strong control that
required eager and production-JIT transports to be byte-identical.  That would
erase the fusion distinction the round is required to report.  Commit
`6f1a525ea` narrows the invariant to shared T/S context *within* each mode and
scores each mode's own live operands.  The rejected run produced no JSON
verdict and is not used below.

## Authoritative production-JIT table

`cells` is cells unequal / cells scored.  `max |diff|` is in the native unit
of each row.  Common `r3t_*` rows are the consumed thickness comparison
described above.

| row | cells | max \|diff\| |
|---|---:|---:|
| p2dt | 0/1 | 0 |
| transport_u | 17,400/21,780 | 1.4246544619672932 |
| transport_v | 17,100/22,080 | 2.3571955611114390 |
| transport_w | 18,000/21,824 | 3.0646979774755891 |
| e3t_3d | 0/21,120 | 0 |
| r3t_Kbb -> live thickness | 3,120/21,120 | 3.0071001721561242e2 |
| r3t_Kmm -> live thickness | 21,120/21,120 | 3.0071001721561242e2 |
| r3t_Kaa -> live thickness | 21,120/21,120 | 3.0071001721561242e2 |
| tmask | 0/21,120 | 0 |
| wmask | 0/21,120 | 0 |
| r1_e1e2t | 0/704 | 0 |
| T.base | 3,120/21,120 | 2.5387611238677501e1 |
| T.now | 19,194/21,120 | 2.5389269943902143e1 |
| T.rhs_entry | 0/21,120 | 0 |
| T.first_u | 17,400/21,780 | 3.3683910373132676e1 |
| T.first_v | 17,100/22,080 | 5.3239334610057995e1 |
| T.first_w | 17,400/21,824 | 6.2519065450178459e1 |
| T.first_div | 18,000/21,120 | 6.9536197323803062e-10 |
| T.midpoint | 20,901/21,120 | 2.5387611238677501e1 |
| T.average_u | 17,400/21,780 | 3.3666003523394465e1 |
| T.average_v | 17,100/22,080 | 5.3229927752632648e1 |
| T.average_w | 17,400/21,824 | 6.2514037907356396e1 |
| T.upstream_div | 18,000/21,120 | 6.9535399748765490e-10 |
| T.rhs_after_up | 21,120/21,120 | 3.5174328868915349e-11 |
| T.anti_pre_u | 17,400/21,780 | 5.5054651245518471e-1 |
| T.anti_pre_v | 17,100/22,080 | 1.2283097727340646 |
| T.anti_pre_w | 17,400/21,824 | 2.5198845380909916 |
| T.coef_u | 208/21,780 | 9.7885517970087899e-1 |
| T.coef_v | 230/22,080 | 9.9386952922522487e-1 |
| T.coef_w | 182/21,824 | 2.3783508691176403e-3 |
| T.anti_post_u | 17,398/21,780 | 5.5054651245518471e-1 |
| T.anti_post_v | 17,099/22,080 | 1.2283097727340646 |
| T.anti_post_w | 17,399/21,824 | 2.5198845380909916 |
| T.final_div | 17,999/21,120 | 3.9534851509745386e-10 |
| T.divisor | 21,120/21,120 | 3.0071001721561242e2 |
| T.rhs_final | 21,120/21,120 | 3.1476948279472017e-11 |
| S.base | 3,120/21,120 | 3.6878388502324135e1 |
| S.now | 16,774/21,120 | 3.6878371253703556e1 |
| S.rhs_entry | 0/21,120 | 0 |
| S.first_u | 17,400/21,780 | 5.2358926909510046e1 |
| S.first_v | 17,100/22,080 | 8.6546912700403482e1 |
| S.first_w | 17,400/21,824 | 1.1235603147745132e2 |
| S.first_div | 18,000/21,120 | 9.9809183890253537e-11 |
| S.midpoint | 19,387/21,120 | 3.6878388502324135e1 |
| S.average_u | 17,400/21,780 | 5.2358672895003110e1 |
| S.average_v | 17,100/22,080 | 8.6546777667477727e1 |
| S.average_w | 17,400/21,824 | 1.1235588533943519e2 |
| S.upstream_div | 18,000/21,120 | 9.9805547082415814e-11 |
| S.rhs_after_up | 21,120/21,120 | 9.4302984645276659e-13 |
| S.anti_pre_u | 17,400/21,780 | 1.4007572082732622e-2 |
| S.anti_pre_v | 17,100/22,080 | 3.7844206118279544e-2 |
| S.anti_pre_w | 17,400/21,824 | 3.0927238411129565e-1 |
| S.coef_u | 331/21,780 | 1.0 |
| S.coef_v | 341/22,080 | 9.9943994660446722e-1 |
| S.coef_w | 280/21,824 | 8.2831125563427949e-2 |
| S.anti_post_u | 17,372/21,780 | 1.4007572082732622e-2 |
| S.anti_post_v | 17,070/22,080 | 3.7844206118279544e-2 |
| S.anti_post_w | 17,383/21,824 | 3.0927238411129565e-1 |
| S.final_div | 17,987/21,120 | 4.9822566389132266e-11 |
| S.divisor | 21,120/21,120 | 3.0071001721561242e2 |
| S.rhs_final | 21,120/21,120 | 6.5332416077751741e-13 |

Seven of 61 rows are bit-exact.  Production eager and isolated-closure JIT
also have 7/61 exact rows and the same first non-bit context
`transport_u`, followed by `T.first_u`.  Their values are not relabelled as
production: for example eager `transport_u` has max difference
`1.4246545043279184`, while production JIT has
`1.4246544619672932`.  Isolated JIT equals production JIT for this captured
closure, but that does not turn it into a production measurement.

Direct branch activity is non-vacuous.  For T, legoESM/NEMO active face
coefficients are U 205/210, V 228/231 and W 183/183; selection disagrees at
5 U and 3 V faces.  For S they are U 325/359, V 351/371 and W 297/297;
selection disagrees at 34 U and 20 V faces.  Coefficients differ even where
selection agrees because their upstream operands already differ.

## Frozen predictions

1. Both NEMO tracers have a non-unity coefficient: **CONFIRMED** (624 T,
   1,027 S face coefficients).
2. The first production-JIT non-bit row is no later than the first horizontal
   upwind faces: **CONFIRMED**.  It is already `transport_u`.
3. At least one active limiter row differs: **CONFIRMED**; all six coefficient
   rows contain unequal values.
4. The first non-bit row owns the next walk: **CONFIRMED AS AN UPSTREAM
   BOUNDARY**, not as an FCT statement.  `T.first_u` cannot own the mismatch
   because its `pU` input is already non-bit.

## Verdict and unchanged campaign rows

**No FCT statement is named as an owner and nothing lands.**  The first
non-bit boundary is the inherited stage-3 U transport: 17,400 unequal faces,
max `1.4246544619672932`.  Walking the limiter would violate the first-non-bit
rule.  The round remains on the magnitude path because Round 152 directly
measured the completed FCT contribution as `1.0974591404090626e-8 K` one-step
RMS; this round localises that measured contribution upstream rather than
claiming the transport carries the full day-240 gap.

Because production physics is unchanged, the latest admitted campaign rows
remain: kt2 T/S at bar, kt2 U/V approximately `2.7377e-12` / `3.2849e-12`,
kt3 T approximately `8.60e-7 K`, day-30 T3D RMS
`6.888194e-5 K`, day-240 `1.644671864e-2 K`, and day-360
`1.122345086e-2 K`.  These are inherited landed-arm values, not remeasured
Round-154 candidate numbers.

Blast radius: GYRE only, diagnostic execution.  DINO, LOCK_EXCHANGE and
OVERFLOW have no production change.  ORCA2 is **UNMEASURED-WITH-SPEC**: repeat
the developed-entry production FCT and stage-transport operand walk on the
ocean-only ORCA2 card before transferring a statement verdict.

## Acquisition request

The existing record has the completed stage-3 transports but not the operands
needed to split the first U statement.  The new operator-run script is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round154_developed_transport/run.sh`.
It creates target `GYRE_OMIP_L2_P3_SM_R154TRPWALK`, copies the Round-153 source
card file by file, adds one post-loop writer call, runs NEMO from the same
day-180 entry, and admits the record only if every restart and `mesh_mask.nc`
remain byte-identical.  Private streams are hashed as one self-consistent new
record rather than incorrectly required to match a build with different
materialisation.  The local proof printed
`SYNTAX_PROOF_PASS l2_r154_transport.f90 stprk3_stg.f90`; its layout plant
printed `STATUS PLANT-FIRED` and exited 69.

## Independent review

**Independent review unavailable in-sandbox.**  The required command was
attempted exactly with `codex exec --sandbox read-only -C` against this clone,
but the reviewer stopped before reading the diff.  Its terminal verdict was:

> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

The complete attempt is `round154/codex_review.log`.  It produced neither a
SHIP nor a DO NOT SHIP verdict.

## Tests and citation gate

The focused campaign and blast-radius set reported exactly:

> 122 passed in 983.84s (0:16:23)

That set includes the FCT trace/unit tests, the Round-153 admission controls,
the Round-154 acquisition controls, the citation-gate tests, and the required
face-mask, prognostic-barotropic-state, and generic-recipe suites.  The normal
citation gate reported `PASS`, with zero failures and zero unmapped citations.
Its shifted-line plant exited 1 and reported `SYMBOL-NOT-AT-LINE` (with the
other deliberately damaged anchors also non-OK).

The mandated combined `tests/ocean/fidelity tests/ocean/unit -n 12` invocation
was also run once.  It did not complete as a valid regression suite: five JAX
workers aborted while compiling unrelated tests, pytest then hit a
`MemoryError`, and xdist ended with an internal error.  Its literal terminal
summary was:

> 327 failed, 5990 passed, 121 skipped, 2 xfailed, 56 warnings, 79 errors in
> 873.79s (0:14:33)

The cascading failures/errors therefore cannot be diffed meaningfully against
the known-red list.  The full log preserves all worker traces at
`round154/full_ocean_tests.log`; the clean 122-test focused result above is the
usable result for every path changed or required by this round.

## OPEN — Round 155

1. Operator runs the acquisition above.  Admit only if the state/restart
   comparison is bit-exact and all plants exit nonzero.
2. From the admitted day-180 stage-3 record, walk the compiled U transport in
   source order: barotropic correction `un_adv*r1_hu(Kmm)-uu_b(Kmm)`, live
   Kmm `e3u`, Kmm `uu`, mask, metric, then their written product.  Compare
   production JIT, production eager and isolated JIT; the production row owns.
3. If the first unequal operand is upstream again, follow it.  Only a single
   source-exact production-JIT statement becomes a candidate, and it must pass
   the full Decision-43/45 month, day-240, day-360, ladder, moved-row, and DINO
   gates before landing.
