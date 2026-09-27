# Round 189 receipt — production shortwave ratio attribution

**Status: HELD.**  The preregistered owner prediction is **REFUTED**.
Replacing only the production-JIT stage-3 live stretch ratio with NEMO's
step-1080 value leaves the QSR boundary non-bit in the same 9,666 of 18,000
wet cells: maximum error changes from `2.4678031493863273e-06 K` to
`2.4677698409192317e-06 K`, a removal of only
`1.3497213950746828e-05` (0.00135%), while RMS error worsens by
`2.048928180207141e-05` relative.  The upstream free-surface walk is therefore
withheld.  Round 188's inference that the first inherited non-bit operand was
the owner is retracted.  The first surviving boundary is instead the
production source-to-process-bucket association around stage-3 QSR; this is
an attribution boundary, not yet a compiled NEMO statement.  No physics,
configuration, carried state, default, or certified trajectory changed.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round189.md`, commit
`c55dc2b67`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round189/`.

## Frozen prediction and production substitution

The frozen prediction was that substituting exact NEMO `r3t(Kmm)` in the
production closure would remove at least 90% of the baseline maximum QSR
error.  The falsifier was removal below 90%, any moved registered pre-QSR
boundary, or a plant that did not fire.  The production run replayed steps
1--1,079 through `LatLonCGridOceanModel.step`, then supplied the exact NEMO
ratio only to the live stage-3 QSR call in step 1,080.  Its worktree stamp was
clean at commit `ed1615b42`.

| production-JIT row | unequal / scored | maximum (K) | RMS (K) |
|---|---:|---:|---:|
| recorded baseline | 9,666 / 18,000 | `2.4678031493863273e-06` | `2.513304102439831e-07` |
| exact NEMO `r3t(Kmm)` only | 9,666 / 18,000 | `2.4677698409192317e-06` | `2.513355598235840e-07` |

The maximum removal is 0.00135%, not the predicted 90%; the RMS removal is
negative.  All registered pre-QSR boundaries (`Tbb`, `q_Kbb`, `q_Kmm`,
`q_Kaa`, `B0`, `Badv`, and `Bsbc`) are BIT between the baseline and candidate,
so the substitution is one-variable at the observed boundary.  The producer
walk is mechanically conditional on exact closure and remained null.

NEMO computes stage-1 after-level free-surface ratios after the continuity
solve at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:172-192`,
then forms the HYB half-step ratio used by stage 3 at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:227-237`.
The ratio conversion itself is the compiled QCO reciprocal-thickness loop at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90:237-258`.
Those cited statements explain the provenance of the substituted operand;
the measurement shows that operand does **not** own the QSR process-row
magnitude.

## Corrected association ranking

An initial post-hoc diagnostic compared NEMO's cumulative RHS with legoESM's
concentration accumulator.  Those quantities have unlike units; that row is
invalid and is explicitly retracted.  The committed scorer now compares like
with like: NEMO and model cumulative concentration accumulators, direct QSR
rates in K/s, and source-ordered associated concentration outputs in K.

| input or boundary | unequal / scored | maximum | RMS |
|---|---:|---:|---:|
| surface QSR flux | 0 / 600 | 0 | 0 |
| step-entry stretch | 600 / 600 | `1.6327013074857177e-09` | `5.993713660608785e-10` |
| live stretch | 600 / 600 | `1.6345230724468252e-09` | `6.030334309176205e-10` |
| after stretch | 600 / 600 | `1.6363447263856301e-09` | `6.067196986626289e-10` |
| direct QSR rate | 10,200 / 18,000 | `5.155285191929060e-15 K/s` | `3.7245636766659584e-16 K/s` |
| preceding cumulative accumulator | 18,000 / 18,000 | `1.697256783881329e-03 K` | `6.094849968062367e-05 K` |
| direct source-order rebuild versus actual production row | 9,679 / 18,000 | `2.467770444880557e-06 K` | `2.513355999155942e-07 K` |

The direct source-order rebuild from the recorded model `Bsbc`, direct QSR
rate, `q_Kbb`, `q_Kmm`, and `q_Kaa` is itself close to NEMO: maximum
`7.433342830154288e-11 K`, 99.99699% below the actual process-row maximum.
Replacing all those isolated inputs with NEMO values leaves only
`3.552713678800501e-15 K`.  Yet that same direct rebuild differs from the
actual production process bucket by `2.467770444880557e-06 K`, essentially
the full observed row.  Therefore the production trace's QSR process row is
not simply the compiled direct-QSR rate associated with its recorded
accumulator and thickness ratios.

NEMO's direct two-band absorption and `Krhs` association are compiled at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645`.
Round 188 already proved every executed boundary in that span BIT given NEMO
operands.  This round does **not** name a defect in that statement.  The first
surviving boundary is legoESM's production source/process association or its
observer classification, which must be split before a statement can be
named.

## Controls, scope, and certified rows

The production hook is private test instrumentation.  Construction rejects
the override unless process tracing is active, and the override feeds only
the live stage-3 QSR ratio.  The ordinary production path is unchanged when
the hook is absent.  The exact substitution keeps every registered pre-QSR
boundary BIT.  The preregistered one-ULP production plant **failed its
control**: the changed ratio was rounded away before `Bqsr`, so it moved zero
cells and exited 1 with `production r3t ULP plant moved no QSR boundary
cell`.  That failure is kept; the run is not relabelled as a fired plant.
Post hoc, the same production path gained a finite `2^-20` ratio-effect
plant.  Its result is recorded below and is only a non-vacuity control for
hook reachability, not a replacement prediction or a scientific arm.

Because no executable production statement changed, the certified rows remain:

- kt2 T/S/U/V: `1.4210854715202004e-14`,
  `2.1316282072803006e-14`, `8.326672684688674e-17`, and
  `9.714451465470120e-17`;
- kt3 T/S: `4.9403105251144552e-07` and `4.0085410546453204e-08`;
- day-30/day-240/day-360 T3D RMS: `2.3276772050683987e-06`,
  `6.5861718814795174e-05`, and `2.6709923853294689e-03 K`;
- first over bar: kt3.

GYRE is the only card exercised by the diagnostic hook.  Generic NEMO-GYRE,
DINO, LOCK_EXCHANGE, and OVERFLOW execute no changed production statement.
ORCA2 is **UNMEASURED-WITH-SPEC**: transfer requires a native developed
process trace exposing the same direct-rate and process-bucket boundaries;
this GYRE attribution may not be projected onto it.

## Review, citation gate, and tests

The required read-only Codex review was unavailable in the sandbox.  Its
verbatim verdict text is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

This is **independent review unavailable in-sandbox**, not a `SHIP` verdict;
it did not issue `DO NOT SHIP`.  The round is therefore explicitly
**UNREVIEWED** under the house dual-review rule.  Its production change is
test-only instrumentation and is inert without a private hook.

The post-hoc finite-effect production plant moved 15 `Bqsr` cells, printed
`STATUS PLANT-FIRED: production-r3t-effect; unequal=15`, and exited 1.  The
failed one-ULP control also exited 1, but printed the named zero-effect refusal
rather than success.  The focused instrument, receipt-citation, and generic
NEMO recipe batch reports **`48 passed in 336.00s (0:05:35)`**.  The receipt
citation gate passes all four citations with zero failures, zero unmapped
citations, and zero map-audit failures.  Shifting the compiled stage-program
citation by two lines changes the gate to `FAIL` with
`SYMBOL-NOT-AT-LINE` and exits 1.  The full citation-gate tests also exercise
the default cumulative receipt after the required model-file re-anchoring.
`git diff --check` and Python compilation pass.  A broad ocean battery was not
run because no production model path, recipe, or physics gate changed; the
only model edit is a guarded private test hook.

## OPEN — round 190

Instrument the existing production process trace at the stage-3 QSR
source-to-bucket association.  Under production JIT, register the source
inputs `tendency_kbb`, `qsr_kbb`, `qsr_kmm`, `thickness_kbb`, and
`thickness_kmm`, plus `_process_qsr_kbb`,
`_nemo_ws_process_surface_rate`, and `_nemo_ws_process_qsr_rate`.  First
reproduce this round's actual-row/direct-rebuild split and fire an association
plant.  Then determine whether the observer assigns a non-QSR source to the
QSR bucket or whether the bridge algebra is non-bit.  Name the first compiled
statement only after that split.  Do not walk the upstream free-surface
producer: its exact substitution was magnitude-inert.  No landing is
authorized without the full Decision-43/45/55/59 trajectory and card gates.
