# GYRE, WHO OWNS THE DAY-30 GAP: the difference is BORN in STEP 2, and step 2 is not bit-exact on NEMO's own inputs

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_owners.md`,
written before any number below was read and frozen; no expectation, metric or
falsifier was changed.  It follows
`nemo_testcases_l2_gyre_year_fromrest_receipt.md`, which scored the year and
left the owner unnamed.

Harness: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_owners.py`.
Acquisition, for the operator: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_earlydays/run.sh`.
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/`,
SHA-256 of all 41 in `year_owners_artifacts.sha256` beside them.  The six the
receipt quotes:

| artifact | SHA-256 |
|---|---|
| `step_gap.json` | `f288b181aaf4ce03d6f294d4616e87a7bea268a25667fa4be7297d926c92d603` |
| `equal_input_kt2.json` | `5fdaeaf3d329198ba8e0d390faf7630c2aeece54a22f4eec691f5f24afc8c3f3` |
| `forcing_gate.json` | `d9e117acf55c777519bb7327da4264564f249f8bde318ba0637f997dbb7dd970` |
| `decompose_day030.json` | `1b16d8c82e5a41deadec97a7064da3d51cacb3713186cdbecacc21a984c16296` |
| `decompose_day360.json` | `0f2c82a343d72eb561c847c5f91b9e1a40905b31b29d52be1133b353fc5856e5` |
| `switch_trace_seeds01.json` | `a07945d26b8a1ae0fae81a57a9a17ba39e2efc7f94d459b356d69379a1ac40d8` |

CPU, fp64, `PrecisionPolicy.fp64(transcendentals="libm")`, `JAX_ENABLE_X64=1`.

**Tests, quoted rather than summarised.**  `88 passed in 54.19s` over the new
harness's own file, the year harness's, the time-level registry's and the
certified phase-3 gate's.  The new file's own self-check exits `0` and all four
forcing plants exit `1`.

## VERDICT

**The GYRE year's model-model gap is created in STEP 2.  The largest
single-step amplification anywhere in the following 2158 RESOLVED steps is
`9.3`; step 2's is `2.0e+11` on temperature.**  ("Resolved" is steps 3-60: the
oracle's per-step record stops at step 60, so the shape between day 10 and day
360 is known only from the day-30 and day-360 points.)

Wet-cell rms differences, legoESM against NEMO's own per-step record:

| field | entering step 2 (after 1 step) | entering step 3 (after 2 steps) | STEP 2's multiplier | largest multiplier of any LATER single step |
|---|---:|---:|---:|---:|
| `T` | `2.0438e-15` K | `4.1544e-04` K | **`2.03e+11`** | `2.003` (step 3) |
| `S` | `4.7867e-15` g/kg | `5.9939e-05` g/kg | **`1.25e+10`** | `1.689` (step 55) |
| `u` | `2.1720e-13` m/s | `1.1909e-05` m/s | **`5.48e+07`** | `9.310` (step 3) |
| `v` | `4.2034e-13` m/s | `1.7317e-05` m/s | **`4.12e+07`** | `9.180` (step 3) |
| `ssh` | `0.0` m | `1.1826e-07` m | from exactly zero | `3.099` (step 5) |

and the whole trajectory, one instrument on each side (section 1 proves the
three records are one legoESM run and one NEMO run):

| | `T` gap | over how many steps |
|---|---:|---:|
| entering step 1, both models at rest | `0.0` K | — |
| entering step 2 | `2.0438e-15` K | 1 |
| entering step 3 | `4.1544e-04` K | 1 more |
| entering step 60 (day 9.83) | `5.2599e-03` K | 57 more |
| day 30 (after 180 steps) | `1.4241e-02` K | 120 more |
| day 360 (after 2160 steps) | `4.0714e-01` K | 1980 more |

**A number that belongs beside the multiplier, and an independent review was
right to ask for it.**  In ABSOLUTE terms step 2 supplies `4.15e-04` K of the
day-30 gap's `1.4241e-02` K — **`2.9 %`**.  The remaining `97 %` is made by the
178 steps after it.  What "step 2 owns the year" means is therefore precise and
narrower than it sounds: step 2 is where the difference is BORN, taking it from
rounding to finite, and nothing later does anything of the kind; the later
growth is a few per cent per step compounding on what step 2 created.  As a
fraction of the signal NEMO itself has developed, step 2's contribution is
`8.4e-03` of NEMO's two-step change against `1.2e-02` of its thirty-day
change — i.e. about `70 %` of the day-30 relative error is present after two
steps.  Both framings are given because the absolute and relative readings
differ by a factor of 24.

**CONFIRMED, and section 1b upgrades it from a trajectory statement to an
EQUAL-INPUT one: given NEMO's own state entering step 2, legoESM's step 2
produces a state that differs from NEMO's by `4.15e-04` K rms on 17999 of
18000 wet cells.  Step 2 does not amplify a difference it is handed; it makes
one.**

Between step 2 and step 3 the temperature difference is
multiplied by two hundred billion.  Over the 57 steps that follow it is
multiplied by `12.7` in total, with a per-step geometric mean of `1.046` and
individual steps ranging from `0.953` to `2.003` — it does not climb
monotonically, and eight of those 57 steps make it SMALLER.

**Step 3 is a SECOND event, and it belongs to MOMENTUM only.**  It multiplies
`u` by `9.31` and `v` by `9.18` while multiplying `T` by `2.00` and `S` by
`1.46`.  That is the largest multiplier anywhere after step 2, and it is ten
orders of magnitude below step 2's.  It is named here because an earlier draft
of this receipt asserted a per-step band of `1.002` to `1.045` over the whole
year; that band is the AVERAGE over long intervals and is not what any
individual step does.  **RETRACTED and replaced by the table above** — found by
an independent review, re-measured, reproduced.

Three consequences, all of which change what the campaign should do next.

1. **The year receipt's section 7(c) is CORRECTED.**  It computed, correctly,
   that the kt=2 error compounded over 180 steps is `2.4e-12` K against a
   measured day-30 gap of `1.4e-02` K, and concluded "the per-step error must
   GROW by orders as the flow spins up".  It does not grow by orders over many
   steps; it grows by eleven orders in ONE step and then amplifies, over the
   57 later steps this record resolves, by a geometric mean of `4.6 %` per
   step.  The sentence "closing the remaining kt=2 DEBT rows cannot be assumed
   to close the year" is too weak in one direction and too strong in the
   other: closing those two rows is not sufficient (they are `3e-12` m/s and
   step 2's OUTPUT is off by `4e-04` K), but **the step-2 operators are where
   the year's gap is made** — step 2's multiplier exceeds the largest later
   single-step multiplier by `6.7` orders on `u` and `11.0` orders on `T`.
2. **The day-30 owner and the kt=2 owner are the same question.**  The kt=1..10
   ladder that rounds 8..50 of this branch have been walking is not a separate,
   smaller problem from the year.  It is the whole problem.  And the target
   inside it is now specific: not the kt=2 ENTRY rows (`u`, `v` at `3e-12` m/s,
   the ladder's standing DEBT) but the kt=2 OUTPUT, which section 1b shows is
   wrong by `4.15e-04` K on NEMO's own inputs.
3. **The surface forcing is exonerated as a statement** (section 3).  What step
   2 gets wrong is therefore an interior operator, and section 4 says where in
   the water column and the basin its consequence lands — with both of the two
   defensible normalizations reported, because they rank the fields
   differently.

## 1.  THE INSTRUMENT, AND WHY NO NEW NEMO RUN WAS NEEDED FOR DAYS 0-10

NEMO's certified GYRE card writes its whole-step entry state for
`kstp = nit000 .. nit000 + 59`
(`cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90-101`:
`WRITE(itraj) ts(:,:,:,:,Nbb), uu(:,:,:,Nbb), vv(:,:,:,Nbb), ssh(:,:,Nbb)`).
The year run's pristine member wrote all sixty months ago and they have been on
disk ever since.  They could not be READ: the campaign's time-level registry
(`ocean/fidelity/time_levels.py`, which fails closed on any unregistered dump)
covered only ten of them.  It now covers sixty and still raises at sixty-one.

**A citation defect found on the way in, and fixed.**  The registry cited the
vendored `scripts/.../nemo502_MY_SRC/stprk3.F90` and said it is "used
byte-for-byte by testcase lanes 1 and 2".  That is FALSE for lane 2.  The
`WRITE` statement is byte-identical, but the GATES are not: the vendored
lane-1 copy fires at `nit000`, the midpoint and `nitend`; the GYRE card fires
on the first sixty steps.  A reader following that citation would have
concluded the year run wrote three entry dumps, not sixty.  The lane-2 range is
now cited from the card that compiles it.

**Reconciliation before recording (Rule 1e).**  This walk is a NEW probe and it
disagrees with nothing only because it was checked against the recorded ladder
first:

| quantity | this walk | recorded, round-8 receipt | agreement |
|---|---|---|---|
| kt=2 `u`, max abs | `2.747840e-12` m/s | `2.747840e-12` m/s | all printed digits |
| kt=2 `v`, max abs | `3.305560e-12` m/s | `3.305560e-12` m/s | all printed digits |
| kt=2 `T`, normalized max | `6.048356e-16` | `6.054358e-16` | `0.9990` |
| kt=10 `T`, normalized max | `5.620512e-03` | `5.636638e-03` | `0.9971` |

The two velocity rows are the SAME SIXTEEN DIGITS as numbers this round did not
produce and could not have tuned.  The two temperature rows agree to `0.3 %`
and are expected to, not to agree exactly: the recorded sweep was taken before
the ENE vorticity commit and before the round's later operator work, and the
year receipt measures that commit moving the model.  **A prose correction the
preregistration owes**: its entering table labels `5.786374e-16` as the kt=2
`T` row.  That number is the kt=2 **`S`** row; `T` is `6.054358e-16`
(round-8 receipt, the last sweep).  No conclusion moves.

**THE THREE RECORDS ARE ONE RUN ON EACH SIDE, and that is measured, not
argued.**  The verdict's table is read off two instruments -- a free-running
per-step walk for steps 1-60 and the scored year ensemble for days 30 and 360
-- and an independent review objected, correctly, that splicing two
instruments into one curve is an unproven claim.  Closed here:

| what could differ | check | result |
|---|---|---|
| legoESM's per-step walk vs the scored year member | the walk's rows re-computed from a SEPARATE 30-day run's daily snapshots: day 1, day 5, day 9 | `2.699448e-03`, `3.889506e-03`, `4.932676e-03` K against the walk's `2.6994e-03`, `3.8895e-03`, `4.9327e-03` -- every printed digit |
| that separate run vs the SCORED year member at day 30 | bit comparison of all five saved fields | `0` cells unequal of 21120 (T, S, u, v) and 704 (ssh); max abs difference exactly `0.0` |
| NEMO's per-step record vs the SCORED year control | `cmp` of the day-30 restarts, `nemo_pristine` against `nemo_seed0` | **BYTE-IDENTICAL** |

So the per-step rows and the day-30/360 rows are the same legoESM trajectory
against the same NEMO trajectory.

**THE ORACLE HAS TWO RECORDS OF THIS CARD AND THEY ARE NOT THE SAME RECORD.
This round read one of them and the ladder's recorded rows were taken against
the other.**  Found by codex's parallel card-reconciliation work on this
branch, measured with ITS committed gate
(`nemo_testcase_l2_gyre_card_reconciliation_gate.py --oracle-floor`), and
reproduced here to the last printed digit by an independent reading through
`read_entry`.  The two records are byte-identical at kt=1 and diverge from
kt=2:

| kt | ladder record vs year record, `T` max abs [K] | wet cells unequal |
|---:|---:|---:|
| 1 | `0.0` | 0 / 18000 |
| 2 | `1.0658e-14` | 154 |
| 3 | `1.3500e-13` | 527 |
| 10 | `1.8471e-10` | 17268 |

**What it changes, and what it does not.**

* **It changes the kt=2 reconciliation row above.**  This walk's kt=2 `T` is
  `2.0438e-15` K rms against an ORACLE-vs-ORACLE spread of `4.5408e-16` K at
  the same step.  The measured gap is only about `4.5x` the oracle's own
  record-to-record spread, so the `0.9990` agreement tabulated against the
  ladder's recorded `6.054358e-16` is agreement WITHIN THE ORACLE'S OWN
  SPREAD, not a precision-level confirmation.  **The reconciliation row is
  RETRACTED as evidence at kt=2** and stands only as evidence that the two
  probes read the same field.
* **It does not touch kt=10.**  The oracle spread there is `1.8471e-10` K
  against this walk's `2.8136e-03` K gap — a factor of `1.5e+07`.  The `0.3 %`
  difference from the ladder's recorded kt=10 row therefore cannot be the two
  records; the attribution to the model's own later operator commits stands.
* **It does not touch the verdict.**  Section 1b's equal-input residual is
  `4.15e-04` K against an oracle spread of `1.35e-13` K at kt=3 — a factor of
  `3.1e+09`.  Taking the oracle spread as an additional floor on the
  denominator, step 2's temperature multiplier is bounded below by
  `4.1544e-04 / (2.0438e-15 + 4.5408e-16) = 1.66e+11` instead of `2.03e+11`.
  Eleven orders either way.
* **Every per-step row in this receipt is against the YEAR record**
  (`year_fromrest/nemo_pristine`), which is the record the day-30 and day-360
  rows come from, so the walk and the year are one NEMO trajectory.  That was
  the property section 1 needed and it is the one that holds.

**What this instrument cannot see.**  It compares STATES, so it says which STEP
owns the gap and says nothing about which OPERATOR inside that step does.  The
per-operator record exists only at `kstp == nit000` on this card, so naming the
operator needs one instrumented step launched from a step-1 restart — a
separate acquisition, named in the open questions, not taken here.
Two further limits, both raised by review and both real: **the record stops at
step 60** (`stprk3.F90:90` caps the writer at `nit000+59`), so no per-step
statement about day 30 or beyond is falsifiable without a new NEMO run; and
**the kt=1 row's exact `0.0` is a SEED check, not a step check** -- it compares
the card's initial state against the dump that IS the initial state, which is
what the year harness's A1 leg already certifies.

## 1b.  STEP 2 *CREATES* THE DIFFERENCE — AN EQUAL-INPUT MEASUREMENT, NOT AN ARGUMENT

A trajectory comparison cannot tell "step 2's operators disagree" from "step 2
amplified what step 1 left".  An earlier draft of this receipt settled that by
arguing no stable scheme multiplies a perturbation by `2e+11` in one step.  That
is an argument.  Here is the measurement.

`year_owners/equal_input_kt2.json`.  Step 2 is run three ways and each output is
scored against NEMO's own state entering step 3:

| arm | what it is given | `T` rms of the OUTPUT |
|---|---|---:|
| FREE RUN | legoESM's own state entering step 2 | `4.154394627928367e-04` K |
| EQUAL INPUT, 3-D | **NEMO's** `ts/uu/vv/ssh` entering step 2 | `4.154394627918457e-04` K |
| EQUAL INPUT, 3-D + barotropic | the same, plus `uu_b`/`vv_b` from the card's own `bt_frames` writer at kt=1, which `stprk3` swaps into `Nbb` — **this reseed wrote `0.0`; see below** | `4.154394627918457e-04` K |

**Handing legoESM NEMO's exact inputs changes the answer in the TWELFTH
significant figure** — a relative move of `2.4e-12`.

**The third arm is not a third arm, and the reason is a result.**  The
barotropic reseed wrote **exactly `0.0`** into both `uu_b` and `vv_b`:
legoESM's prognostic barotropic pair entering step 2 is ALREADY BIT-IDENTICAL
to NEMO's `bt_frames` record on every wet face.  So the arm is vacuous as a
control — and that vacuity is an independent reproduction of the round-8
result `GYRE equal-input kt1 Kaa uu_b/vv_b = 0 / 580, 0 / 570`.  It also
removes the caveat this arm was built to carry: the barotropic pair is not an
unreseeded input, because it needed no reseeding.  Measured, and stated as a
vacuous arm rather than reported as a third confirming row.

**So step 2's operators do not agree with NEMO's, given NEMO's own inputs.  The
difference is MANUFACTURED by the step; it is not carried into it.  CONFIRMED,
and it is the Rule-12 statement: the operator set of step 2 is not bit-exact on
NEMO's own inputs, by `4.15e-04` K rms and `8.74e-03` K max, on 17999 of 18000
wet cells.**

Seventeen thousand nine hundred and ninety-nine of eighteen thousand.  This is
not a localized defect in one cell or one column; step 2 disagrees essentially
everywhere.

Every row below is the EQUAL-INPUT arm, not the free run:

| field | equal-input OUTPUT rms | max abs | wet cells unequal | (free-run cells, for contrast) |
|---|---:|---:|---:|---:|
| `T` | `4.1544e-04` K | `8.7413e-03` K | 17999 / 18000 | 17999 |
| `S` | `5.9939e-05` g/kg | `1.3569e-03` g/kg | 17258 / 18000 | 17365 |
| `u` | `1.1909e-05` m/s | `7.1934e-04` m/s | 17400 / 17400 | 17400 |
| `v` | `1.7317e-05` m/s | `8.6069e-04` m/s | 17100 / 17100 | 17100 |
| `ssh` | `1.1826e-07` m | `7.0726e-07` m | 600 / 600 | 600 |

Feeding NEMO's own inputs changes the unequal-cell count on ONE row, `S`, and
by 107 cells of 18000.  Every other row is identical.

**The controls this arm needs, and what they say.**  The 3-D reseed is shown to
have WRITTEN: it moved the input by `1.4211e-14` K, `2.1316e-14` g/kg,
`2.7478e-12` m/s and `3.3056e-12` m/s (`ssh` by `0.0`, because both models'
sea surface after one step is already bit-identical), and the harness REFUSES
if the reseed moves nothing at all.  The barotropic write size is measured for
the same reason and came out `0.0` on both components — which is why the third
arm is reported above as vacuous rather than as agreement.  **A first version
of this harness measured the 3-D write and not the barotropic one**, and would
have reported "reseeding the barotropic pair changes nothing" as a physical
result when it is a statement that nothing was written.  Found and fixed before
the number was recorded.

**What this arm cannot see.**  It reseeds the five prognostic fields the card's
writer records; the barotropic pair needed no reseeding.  Anything else
legoESM carries into a step that NEMO's dump does not record (the TKE state,
the closure's carried coefficients) is NOT reseeded and stays legoESM's own.  Those carriers are a candidate for the
residual `2.4e-12` relative move between the free-run and equal-input arms, and
they are not a candidate for the `4.15e-04` K: at the entry of step 2 the flow
is two hours old and every such carrier is within rounding of NEMO's.

## 2.  THE DAY-BY-DAY TABLE

Days 0-9 are at STEP resolution from the record above; day 30 is the year
record.  Days 10-29 are the acquisition this round writes and does not run.
Wet-masked rms, fp64, NEMO's own `tmask` for `T`/`S`/`ssh` and the card's face
masks for `u`/`v`.

| day | kt | `T` [K] | `S` [g/kg] | `u` [m/s] | `v` [m/s] | `ssh` [m] | `T` ratio | leads |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | — | — |
| 1 | 7 | `2.6994e-03` | `2.0391e-04` | `5.1958e-04` | `7.4568e-04` | `1.3519e-06` | — | `S` |
| 2 | 13 | `2.7856e-03` | `1.4378e-04` | `8.6902e-04` | `1.8743e-04` | `8.1644e-06` | `1.032` | `u` |
| 3 | 19 | `2.8185e-03` | `1.5600e-04` | `5.5284e-04` | `2.2112e-04` | `1.7806e-05` | `1.012` | `u` |
| 4 | 25 | `3.3105e-03` | `1.9930e-04` | `5.7926e-04` | `2.8858e-04` | `2.8945e-05` | `1.175` | `u` |
| 5 | 31 | `3.8895e-03` | `2.2572e-04` | `7.4593e-04` | `4.7500e-04` | `4.1285e-05` | `1.175` | `u` |
| 6 | 37 | `4.1958e-03` | `2.8294e-04` | `7.6619e-04` | `3.5979e-04` | `5.4624e-05` | `1.079` | `u` |
| 7 | 43 | `4.4082e-03` | `2.3556e-04` | `7.7226e-04` | `2.2426e-04` | `6.8648e-05` | `1.051` | `u` |
| 8 | 49 | `4.6569e-03` | `2.5223e-04` | `7.1844e-04` | `4.6195e-04` | `8.3165e-05` | `1.056` | `u` |
| 9 | 55 | `4.9327e-03` | `2.6367e-04` | `7.2368e-04` | `4.3775e-04` | `9.8125e-05` | `1.059` | `u` |
| 10-29 | 61-174 | — | — | — | — | — | — | ACQUISITION PENDING |
| 30 | 181 | `1.4241e-02` | `2.2345e-03` | — | — | `4.5334e-04` | — | `S` |

The STEP table, which is what carries the verdict, is in
`year_owners/step_gap.json`; the eleven-order jump is between its `kt = 2` and
`kt = 3` rows and nowhere else in sixty steps.

The `leads` column uses normalizer **(a)** of section 4 (gap over NEMO's own
from-rest change).  Under normalizer **(b)** (gap over the field's own standard
deviation) `u` and `v` lead every day, because for a field that starts at rest
the two normalizers coincide and `u`/`v` are the only rows that do not move
between them.  Both are in section 4; the column is labelled rather than
silently carrying one choice.

| expectation | bound | measured | verdict |
|---|---|---|---|
| **E1** day-1 `T3D` above `1e-4` K | `> 1e-4` | `2.6994e-03` K | **HELD** |
| **E2** no knee: day-to-day `T` ratio below 3 | `< 3` | max `1.175` (days 1-9) | **HELD**, and it is the wrong question — the knee is INSIDE day 1, between steps 2 and 3, which a daily cadence cannot resolve and a step cadence does |
| **E3** the forcing statement is bit-exact at every clock value | 0 cells | 0 of 600, five fields, thirteen clock values | **HELD** |
| **E4** upper ocean and a tracer lead | `0-100` m more than `10x` the `1000+` band; `T` or `S` leads | `2.5079e-02` vs `1.7094e-04` K = `147x`; `S` leads | **HELD** |
| **E5** the EVD trigger masks first differ inside `[1080, 1260]` | step window | see section 6 | see section 6 |

## 3.  THE SURFACE FORCING IS EXONERATED AS A STATEMENT

`year_owners/forcing_gate.json`.  legoESM's production forcing path and a
literal transcription of the compiled `usrdef_sbc.f90` are evaluated on the
SAME state — NEMO's own restart, which IS the `Kbb` state entering the next
step (`stprk3.F90:155`, `CALL sbc( kstp, Nbb, Nbb )`) — so the inputs are
identical by construction and any difference can only be the statement.

**The rows are the MODEL-FACING quantities, not a re-assembly of them.**  The
gate CALLS `nemo_testcase_l2_gyre_phase3_gate._surface_forcings` -- the
function the certified step calls -- and compares what comes out of it:
`surface.sw_down` against `qsr`, the materialized `surface.q_net` against the
literal `qns + qsr`, `freshwater.evap` against `emp`, and
`surface.tau_i_native` / `tau_j_native` against `utau` / `vtau`.  A first
version re-assembled that composition here instead of calling it, and an
independent review proved the consequence: a transposed stress pair inside the
real function left every row bit-exact.  Retraction 9.

| clock sample | `ztime` [h] | `qsr` | `q_net` | `emp` | `utau` | `vtau` |
|---|---:|---:|---:|---:|---:|---:|
| kt = 1 | `4.0` | 0 | 0 | 0 | 0 | 0 |
| entering days 30, 60, … 330 | `724 … 7924` | 0 | 0 | 0 | 0 | 0 |
| clock sample at kt = 2161 | `8644.0` | 0 | 0 | 0 | 0 | 0 |

Cells unequal, of 600 wet, at every one of the thirteen samples: **zero**.
The samples span the entire `8640`-hour seasonal cycle, so no phase,
denominator, amplitude or calendar operand of the seasonal statements can be
wrong.  **C1 REFUTED as an owner.**

**The GEOGRAPHIC stress pair, which the five rows above do not pin.**  The
shared forcing object also carries `tau_x`/`tau_y`, the geographic rotation of
the native pair, and that is where a transposition or a sign error would live.
Recovering the native pair from it costs `1.3878e-17` Pa in binary64 (because
`cos² + sin²` is not exactly 1), so the row is scored as a DISCRIMINATION
rather than at the bit-exact bar: how much further from the literal a
TRANSPOSED pair lands.

| | identity | transposed | ratio | verdict |
|---|---:|---:|---:|---|
| `utau`, all 13 samples | `1.3878e-17` Pa | `1.3878e-17` Pa | `1.0000` | **BLIND**, structurally |
| `vtau`, all 13 samples | `1.3878e-17` Pa | `1.19e-01` … `1.78e-01` Pa | `8.55e+15` … `1.28e+16` | DISCRIMINATES |

**`utau` cannot see a transposition on this card and no correct model could
make it.**  Measured: GYRE's grid is exactly 45 degrees (`sin_alpha_u` equals
`cos_alpha_u` to `0.0`) and its wind has `vtau` equal to `-utau` to `0.0`, so
the transposed recovery of `utau` is algebraically the identity recovery --
`utau*(2 sin cos - cos² + sin²)`, which is `utau` at 45 degrees.  The gate
requires that the transposition be SEEN by SOME component and reports the blind
ones rather than hiding them, which is the same shape as the year round's A2b
leg.

**The statements that were re-read line by line to make that claim** (Rule 0;
`BLD/ppsrc/nemo/usrdef_sbc.f90`, the compiled file the card runs):

| # | NEMO, cited | legoESM | verdict |
|---|---|---|---|
| 1 | `:107-108` `ztime = REAL(kt)*rn_Dt/(rmmss*rhhmm) - (nyear-1)*rjjhh*zyydd` | `nemo_gyre_seasonal_cosines`, `t_seconds/3600` with `t_seconds = kt*dt`; **no `nyear` term** | MATCH inside year 1 — the term is `(nyear-1)*24*360` h = exactly one full period of every cosine, and inside year 1 it is `0.0`, which subtracts exactly.  **MEASURED**: forcing the literal arm to `nyear = 2` at kt = 181 moves `qsr` by `1.4211e-13` W/m2, i.e. the term is mathematically a no-op and numerically one ulp of `libm`'s `cos`.  The gate CANNOT see a missing `nyear` term in year 1, stated rather than discovered later |
| 2 | `:110-113` `ztimemax1 = 4104`, `ztimemin1 = 8424`, `ztimemax2 = 4824`, `ztimemin2 = 504` h | same four constants, same composition | MATCH |
| 3 | `:123-124` `zcos_sais1` divides by `(ztimemin1-ztimemax1)`, `zcos_sais2` by `(ztimemax2-ztimemin2)` — two DIFFERENT expressions with the same value `4320` | the same two expressions, not a shared constant | MATCH |
| 4 | `:132-134` `t_star = 28.3*(1+1/50*zcos_sais2)*COS(rpi*(gphit-5)/(53.5*(1+11/53.5*zcos_sais2)*2))` | `nemo_gyre_t_star`, operand for operand | MATCH |
| 5 | `:136` `qsr = 230*COS(3.1415*(gphit-23.5*zcos_sais1)/(0.9*180))` — the literal is **`3.1415`, not `rpi`** | `_NEMO_QSR_PI` | MATCH.  **MEASURED**: swapping it for `rpi` moves `qsr` by `9.2864e-03` W/m2, so the gate would see it |
| 6 | `:137` `qns = ztrp*(ts(ji,jj,1,jp_tem,Kbb) - t_star) - qsr`, `ztrp = -40` | `nemo_gyre_qns` on the entering tracer | MATCH, and the TIME LEVEL is `Kbb` = the step-entry tracer, which is what legoESM feeds |
| 7 | `:140` `IF( gphit >= 14.845 .AND. 37.2 >= gphit )` | `(lat >= 14.845) & (lat <= 37.2)` | MATCH |
| 8 | `:151-157` `zsumemp = glob_2Dsum(emp)/glob_2Dsum(tmask(:,:,1))`, then `emp = emp - zsumemp*tmask` | `nemo_gyre_zero_mean_emp`, NEMO's DDPDD compensated sum in `ji`-inner order | MATCH (certified bit-exact at kt=1 by round 8) |
| 9 | `:161` `qns = qns - emp*sst_m*rcp` | `nemo_gyre_qns`'s `emp_heat`, with `sst_m` = potential temperature from the entering conservative tracer | MATCH, **and the time level was the live question**: `sst_m` is built in `sbcssm.f90:84-93` from `ts(:,:,1,:,Kmm)` while the Haney term reads `Kbb`.  Those are DIFFERENT LEVELS in general.  On this card they are the same one, because `stprk3.F90:155` calls `sbc( kstp, Nbb, Nbb )` — `Kmm` IS `Nbb`.  legoESM uses one state for both.  **C2's restoring-time-level half is REFUTED on this card, by the call site, not by an argument** |
| 10 | `:185-193` `ztau = 0.105/SQRT(2.)`, `ztau_sais = 0.015`, `ztaun = ztau - ztau_sais*COS(...)`, `utau = -ztaun*SIN(rpi*(gphit-15)/(29-15))`, `vtau = +ztaun*SIN(...)` | `nemo_gyre_wind` | MATCH |

**What section 3 cannot see.**  Three things, two of them named in the
preregistration and one found by an independent review.

* **A statement misread the SAME way by both arms.**  The mitigations are that
  the literal arm is certified against NEMO's OWN kt=1 dump (round 8: all five
  fields, 0 of 600 wet cells, max abs `0`) and that the table above is a fresh
  reading of the compiled file, not a re-reading of the transcription.
* **A NAMED shared channel, and it is not hypothetical.**  Both arms compute
  the potential temperature that row 9 feeds into `-emp*sst_m*rcp` with the
  SAME legoESM function, `nemo_potential_temperature_from_conservative`; NEMO
  uses Fortran `eos_pt_from_ct` (`sbcssm.f90:92`).  A defect in that conversion
  is invisible to this gate by construction.  It is not unbounded — the kt=1
  oracle dump compares `qns`, which contains that product, against NEMO's own
  and finds `0` of 600 cells unequal — but that binds the conversion at ONE
  temperature and ONE salinity, the initial profile's surface values, not
  across the year's range.  Raised by review; the bound is stated rather than
  the exposure denied.
* **The gate scores the 600 WET surface cells and is blind to the closed
  boundary ring.**  NEMO computes `emp`, `utau` and `vtau` over a WIDER index
  range than `qsr` and `qns` (`usrdef_sbc.f90:139,156,189` carry `nn_hls`,
  `:128,159,198` do not), and on the ring `emp` keeps its raw value because the
  de-meaning multiplies `tmask`, which is zero there.  Those cells are land on
  this card and every scored quantity is masked, so a difference there cannot
  reach a scored number — but this gate does not look, and that is a property
  of the gate, not a proof about the cells.
* **The `nyear` operand is never exercised.**  All thirteen samples have
  `nyear = 1` read from the restarts' own `ndastp`, so the `-(nyear-1)` term is
  identically zero in both arms and legoESM has no such operand at all.  What
  bounds it is the plant, measured: at `nyear = 2` the literal arm's `qsr`
  moves by `1.4211e-13` W/m2, so the term is mathematically a full period and
  numerically one ulp of `libm`'s `cos`.  Twelve of the thirteen samples are
  steps the run actually takes; the thirteenth is past `nn_itend` and is
  labelled a clock sample.

## 4.  WHERE THE GAP LIVES: FIELD, DEPTH, REGION

`year_owners/decompose_day030.json`, `decompose_day360.json`.  Each cut's blind
spot is printed with it, not left to the reader.

**BY FIELD, AND THE RANKING DEPENDS ON THE NORMALIZER — BOTH ARE GIVEN.**  Five
fields in five units cannot be compared without one, and there are two
defensible choices.  They DISAGREE, and an earlier draft of this receipt
reported only the first and drew a headline from it; that headline is
**RETRACTED** and replaced by the pair.  Found by an independent review,
re-measured, reproduced.

| field | day 30 gap | (a) gap / NEMO's own from-rest CHANGE | rank | (b) gap / NEMO's field STANDARD DEVIATION | rank |
|---|---:|---:|---:|---:|---:|
| `S` | `2.2345e-03` g/kg | `7.63e-02` | **1** | `2.99e-03` | 4 |
| `v` | `8.3330e-04` m/s | `5.93e-02` | 2 | `5.94e-02` | **1** |
| `u` | `5.9620e-04` m/s | `5.32e-02` | 3 | `5.32e-02` | 2 |
| `T` | `1.4241e-02` K | `1.21e-02` | 4 | `1.97e-03` | 5 |
| `ssh` | `4.5334e-04` m | `8.97e-03` | 5 | `8.97e-03` | 3 |

and at day 360:

| field | day 360 gap | (a) / from-rest change | (b) / field std |
|---|---:|---:|---:|
| `S` | `6.9290e-02` g/kg | `3.83e-01` | `9.76e-02` |
| `u` | `4.0919e-03` m/s | `1.46e-01` | `1.46e-01` |
| `v` | `4.0703e-03` m/s | `1.38e-01` | `1.38e-01` |
| `T` | `4.0714e-01` K | `1.59e-01` | `6.02e-02` |
| `ssh` | `7.6119e-03` m | `4.53e-02` | `4.53e-02` |

**What both normalizers agree on:** `u` and `v` are `5.3e-02` and `5.9e-02`
under BOTH at day 30, because the two normalizers coincide for a field that
starts at rest.  **Momentum is wrong at the five-per-cent level after thirty
days on both readings**, and that is the one field ranking that is not an
artifact of a choice.

**Where they disagree, and why:** normalizer (a) divides by how much NEMO
itself moved, so it flatters nothing but rewards a field that barely moves.
Salinity's own 30-day change is `2.93e-02` g/kg against a field spread of
`0.746` g/kg — it moves very little — so `S` tops (a) and falls to fourth in
(b).  Both facts are true; the sentence "salinity leads" is only true under
(a).  What survives either way is that GYRE's salinity has NO SOURCE
(`usrdef_sbc.f90:157,160`: `sfx = 0` and `emp` de-meaned), so **whatever
salinity error exists is pure transport** — and legoESM gets `7.6 %` of NEMO's
own 30-day salinity change wrong.  **PLAUSIBLE as a direction; no ablation was
run on either model (Rule 4), so it is not an attribution.**

**BY DEPTH**, temperature.  Both the SHARE and the per-cell ENRICHMENT are
given, because a share alone favours a band with more cells and the two rows
below read in OPPOSITE directions if only the share is quoted:

| band | day 30 gap [K] | share of ΣΔT² | cells | enrichment | day 360 share | day 360 enrichment |
|---|---:|---:|---:|---:|---:|---:|
| `0-100` m | `2.5079e-02` | `0.827` | 4800 | **`3.10`** | `0.396` | `1.48` |
| `100-1000` m | `1.0257e-02` | `0.173` | 6000 | `0.52` | `0.604` | `1.81` |
| `1000+` m | `1.7094e-04` | `0.0001` | 7200 | `0.00` | `0.0001` | `0.00` |
| the TOP CELL alone | `2.1393e-02` | `0.075` | 600 | **`2.26`** | `0.037` | `1.11` |

**A CORRECTION THIS RECEIPT OWES ITSELF.**  An earlier draft wrote "the gap is
in the upper ocean but NOT in the top cell", reading the top cell's `7.5 %`
share as small.  That is backwards: `7.5 %` of the difference on `3.3 %` of the
cells is an ENRICHMENT of `2.26`, the second highest in the table.  **The top
cell is strongly enriched, and nothing here argues against a surface-driven
mechanism.**  Found by an independent review; re-measured and reproduced.

What the table does support, stated correctly: the `0-100` m band as a whole is
enriched MORE (`3.10`) than its own top cell (`2.26`), and the single largest
difference at day 30 sits at `k = 3`, depth `36.5` m, not at `k = 0`.  So the
day-30 difference is **surface-intensified but subsurface-PEAKED**.  Between
day 30 and day 360 it migrates downward: the `100-1000` m band goes from
enrichment `0.52` to `1.81` and overtakes the surface band.  Below 1000 m there
is nothing in either year.

**BY REGION**, temperature, day 30, with the enrichment (share of the squared
difference over share of the cells) stated because a share alone favours a big
region:

| region | share of sum dT² | share of cells | enrichment |
|---|---:|---:|---:|
| western third | `0.620` | `0.333` | **`1.86`** |
| interior third | `0.183` | `0.333` | `0.55` |
| eastern third | `0.197` | `0.333` | `0.59` |
| `emp` south branch (`<= 37.2 N`) | `0.803` | `0.745` | `1.08` |
| `emp` north branch (`> 37.2 N`) | `0.197` | `0.255` | `0.77` |
| wind band (`15-29 N`) | `0.638` | `0.343` | **`1.86`** |

**The two enriched regions are NOT two findings — they are largely the same
cells.**  On GYRE's 45-degree-rotated grid the western third (200 surface
cells) and the `15-29 N` wind band (206) share 154; their Jaccard overlap is
`0.61`.  Measured here rather than left for a reviewer.  What separates them is
the peak: the largest single difference at day 30 is `-4.4411e-01` K at
`j=7, i=1, k=3` — depth `36.5` m, latitude `20.24 N`, and `i = 1` is ONE CELL
off the western wall, the closed-boundary ring being `i = 0`.  At day 360 the
peak is `-2.9493e+00` K at `j=6, i=4, k=7`, `91.4` m, `21.59 N`.  The western
boundary current reading is the one the peak supports; the forcing-band reading
is not separable from it by this cut, and saying so is the result.

**The `emp` split shows NO enrichment at all** (`1.08` and `0.77`), which is
the cleanest negative in this table: the freshwater flux's own branch structure
leaves no fingerprint on where the temperature difference lives.

## 5.  THREE SOURCE-LITERAL DIFFERENCES IN THE CONVECTIVE SWITCH

Read from `BLD/ppsrc/nemo/zdfevd.f90`, the compiled file:

```fortran
DO jk = 1, jpkm1 ; DO jj = ... ; DO ji = ...
   IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )      &
      &  p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)            ! :107-110
END DO ; END DO ; END DO
```

and the same block for `p_avm` under `nn_evdm == 1` (`:131-135`).

| # | NEMO | legoESM | assessment |
|---|---|---|---|
| A | `<= -1.e-12` | `N2 < _thr` with `_thr = -1e-12` (`convection/enhanced_diffusion.py:198-200`) | a transcription difference, reachable ONLY on exact float64 equality with `-1e-12`.  `rn2` is a quotient of differences of EOS polynomials; the set of inputs giving that exact bit pattern is a measure-zero set and no cell has ever been observed in it.  **Recorded as DEBT, not fixed this round: a fix is one character, but changing a threshold comparison is a change to the model's own convective switch and it goes in the ASKED table, not in a diff** |
| B | NEMO **SETS** `p_avt = rn_evd` | legoESM takes `K_v_total = maximum(K_v_total, K_conv)` (`vertical_mixing/k_profiles.py:420-425`) | the two agree wherever the other schemes' diffusivity is BELOW `rn_evd = 100` m2/s and DISAGREE wherever it is above — NEMO would reduce such a cell to 100, legoESM keeps the larger.  **This is the one worth measuring**, and the measurement is a one-line census on the traced run: how many wet interfaces ever hold a non-convective `K_v` above `100` m2/s.  Named in the open questions; UNMEASURED this round |

| C | NEMO sets `p_avt = rn_evd * wmask(ji,jj,jk)` over `jk = 1, jpkm1`, so at a FIRING interface whose `wmask` is zero — the SURFACE interface `jk=1` among them — it sets `avt` to **zero** | legoESM zeroes its own convective `K` on dry interfaces and then takes a MAXIMUM, and a maximum can never LOWER anything | **the one branch where set and max differ regardless of magnitude.**  Raised by an independent review; this round did not measure whether any firing interface on this card has `wmask = 0`, and the measurement is one mask census.  UNMEASURED |

**The card's own resolved values, printed rather than read off a namelist**:
`K_conv = nu_conv = 100.0` (= `rn_evd`), `K_bg = nu_bg = 0.0`,
`n2_threshold = -1e-12`, `two_level_trigger = True`,
`evd_n2_time_level = "nemo_now_before"`.  `K_bg = 0` makes row B's stable
branch an exact no-op under a maximum, so B binds only where the TKE closure's
own diffusivity exceeds `100` m2/s.

**A namelist fact worth stating because it changes the answer**: the shipped
`namelist_ref` has `nn_fsbc = 2` and `ln_zdfevd = .false.`; this card's
`namelist_cfg` overrides both (`nn_fsbc = 1`, `ln_zdfevd = .true.`).  Every
statement in this receipt about the forcing's time level and about the
convective switch rests on the OVERRIDES.

Neither A nor B nor C can own the day-30 gap: all three are threshold or
saturation behaviours, and section 1 shows the day-30 gap is made in step 2,
when the column is still the initial-state profile and nothing is convecting.

## 6.  THE THRESHOLD SWITCH — TRACED TO A CELL AND A STEP.  E5 HELD.

`year_owners/switch_trace_seeds01.json`.  Two members of the TIP model, seeds 0
and 1, stepped side by side from rest for 1260 steps (day 210), with the
convective trigger mask read from the MODEL'S OWN coefficient field every sixth
step.  1888 s wall.

**The first step at which the two members' enhanced-vertical-diffusion trigger
masks differ is `kt = 1213`, DAY 202** — inside the preregistered `[1080,
1260]` window.  **E5 HELD.**

| | value |
|---|---|
| step | `kt = 1213`, day `202.0` |
| cell | `j = 10`, `i = 9`, `k = 1` |
| depth | `15.096` m |
| latitude | `27.652 N` |
| cells whose mask differs | `2` of 17400 interfaces |
| interfaces firing, seed 0 / seed 1 | `834` / `832` |
| `T` rms between the two members at that step | `3.7552e-05` K |

**And the between-member temperature difference jumps at that step and nowhere
before it:**

| day | interfaces firing, seed 0 / seed 1 | mask cells differing | `T` rms between members [K] |
|---:|---:|---:|---:|
| 198 | 760 / 760 | 0 | `2.0505e-08` |
| 199 | 790 / 790 | 0 | `2.1601e-08` |
| 200 | 800 / 800 | 0 | `2.2897e-08` |
| 201 | 813 / 813 | 0 | `2.4159e-08` |
| **202** | **834 / 832** | **2** | **`3.7552e-05`** |
| 203 | 856 / 856 | 0 | `5.5345e-05` |
| 204 | 874 / 874 | 0 | `6.0400e-05` |
| 205 | 905 / 907 | 2 | `8.6260e-04` |
| 206 | 928 / 928 | 0 | `9.4389e-05` |
| 207 | 951 / 951 | 0 | `1.0329e-04` |
| 208 | 973 / 972 | 1 | `3.3573e-04` |
| 209 | 996 / 992 | 4 | `5.4552e-04` |

**`1553x` in one step, at the step the masks first disagree, after 201 days in
which they never did.**  Of 210 sampled days, exactly FOUR have a differing
mask, and the first is day 202.

**So the year receipt's section 7(d) is upgraded from PLAUSIBLE to CONFIRMED.**
It named `ln_zdfevd`'s hard branch as the only threshold of that size in the
configuration and said the timing matched, with no cell-level trace run.  The
trace is run: the branch is
`IF( MIN(rn2,rn2b) <= -1.e-12 ) p_avt = rn_evd*wmask`
(`BLD/ppsrc/nemo/zdfevd.f90:107-110`), the card's transcription fires on 813
interfaces at day 201 and 834 on day 202, two of those 834 are cells the other
member does not fire, and the between-member temperature difference multiplies
by `1553` in that step.

**NEMO does not cross it in year 1 and legoESM does, and that asymmetry is now
attached to a mechanism rather than to a correlation.**  The year receipt
measures NEMO's own four-member ensemble at `0` cells over 1 mK on every
scored day while legoESM's reaches 4014.

**What this trace cannot see, and one of these is load-bearing.**

* **It samples the mask once a DAY** (`--every 6`).  A crossing that flips and
  flips back inside one day is invisible, so day 202 is an UPPER BOUND on the
  first crossing, not a proof that none happened earlier.  What makes it
  informative anyway is the companion series: the between-member difference is
  flat at `2e-08` for 201 days and jumps `1553x` at the sampled step where the
  mask first differs.  A crossing that had happened earlier and mattered would
  have shown in that series.
* **It reads the mask from the model's own `_enhanced_diffusion_K` with the
  two-arm branch the model runs** (`before_tracers` passed explicitly, so the
  `MIN(rn2, rn2b)` branch executes rather than an argued equivalent), but
  WITHOUT the caller's `eos_fn`.  Under this card's `n2_mode = "nemo_bn2"` the
  N² that drives the trigger comes from `compute_buoyancy_frequency_nemo_bn2`,
  which takes `eos_form` and not `eos_fn`; the density `eos_fn` would build is
  consumed only by the `insitu` branch.  That is a CODE READING, not a
  measurement, and it is the trace's one unvalidated instrument assumption.
* **It is two members of legoESM**, not legoESM against NEMO.  It explains the
  FLOOR, which is what section 7(d) of the year receipt is about.  It says
  nothing about the GAP, which sections 1 and 1b own.

## 7.  WHAT THIS ROUND CANNOT SEE

* **It names a STEP, not an OPERATOR.**  The per-operator record exists at
  `kstp == nit000` only.  Naming what step 2 gets wrong needs one instrumented
  step launched from the step-1 restart.
* **No ablation on either model**, so section 4 is a direction and not an
  attribution (Rule 4).
* **The forcing gate shares one possible misreading with the literal arm** and
  cannot see a missing `nyear` term inside year 1 (section 3).
* **The per-step record stops at step 60**, so no per-step statement about day
  30 or beyond is falsifiable without a new NEMO run.
* **Days 10-29 have no NEMO record.**  The acquisition is written and not run.
* **The region cut cannot separate the western boundary from the wind band**
  on this rotated grid (`0.61` Jaccard), and its "thirds" are cuts by COLUMN
  INDEX, which on a 45-degree-rotated grid are diagonals, not meridians.
* **The field ranking depends on a choice of normalizer** and the two
  defensible choices disagree (section 4).  Only the `u`/`v` rows are
  normalizer-independent.
* **The forcing gate shares legoESM's conservative-to-potential conversion
  with its own reference arm** (section 3).
* The step walk compares STATES on the card's masks; a difference living on a
  dry face or below level 30 is invisible by construction.

## 8.  RETRACTIONS AND CORRECTIONS

1. **The year receipt's 7(c) mechanism — "the per-step error must GROW by
   orders as the flow spins up" — is CORRECTED.**  Its arithmetic stands: the
   kt=2 error compounded is `2.4e-12` K against a `1.4e-02` K day-30 gap.  Its
   mechanism does not: there is no gradual growth.  There is one step.
2. **The preregistration of the YEAR mislabels the kt=2 `T` row.**  It quotes
   `5.786374e-16` as `GYRE-zco.kt2.before.T`; that is the `S` row.  `T` is
   `6.054358e-16`.  No scored number moves.
3. **The time-level registry's citation for the step-entry dumps was wrong in
   SCOPE** — "used byte-for-byte by lanes 1 and 2" — and is corrected in
   section 1.
4. **The forcing gate's `nyear` plant was a dead variable in its first
   version** and proved nothing; it is fixed and now moves `qsr` by
   `1.4211e-13` W/m2 at kt = 181.  Found by running the plants rather than
   trusting them.
5. **"Every later step multiplies the gap by between 1.0017 and 1.045" —
   RETRACTED.**  That band is the AVERAGE over intervals of 57, 120 and 1980
   steps.  Measured per step over kt=3..60: `T` ranges `0.953` to `2.003` with
   eight steps SHRINKING the gap, `u` ranges `0.773` to `9.310`, `v` `0.368` to
   `9.180`.  The claim that survives is the one in the verdict: step 2's
   multiplier is `2.0e+11` and the largest of any later single step is `9.31`.
6. **"The gap is in the upper ocean but NOT in the top cell" — RETRACTED.**
   The top cell holds `7.5 %` of the squared difference on `3.3 %` of the
   cells, which is an ENRICHMENT of `2.26`, not a deficit.  The corrected
   statement is in section 4: surface-intensified, subsurface-peaked.
7. **"Salinity leads" — RETRACTED as a bare statement.**  It leads under the
   normalizer this round imported from DINO (gap over NEMO's own from-rest
   change) and is FOURTH under the other defensible one (gap over the field's
   own standard deviation).  Both are now in section 4.  What survives either
   normalizer is that `u` and `v` are wrong at the `5` to `6` per cent level at
   day 30 on BOTH readings, because the two normalizers coincide for a field
   that starts at rest.
8. **A finding this round did NOT make and an earlier draft implied:** step 2
   is not the only discontinuity.  **Step 3 multiplies `u` and `v` by `9.31`
   and `9.18`** — ten orders below step 2, but the largest thing in the rest of
   the year, and it is momentum-only.
9. **"The surface forcing is bit-exact" — RETRACTED in its first form and
   re-established in a narrower, stronger one.**  The first version of the
   gate re-assembled the certified gate's forcing composition instead of
   CALLING it, so it certified the `usrdef_sbc` statements and said nothing
   about the forcing the MODEL consumes.  An independent review proved the
   consequence by construction: transposing `tau_x`/`tau_y` inside the real
   `_surface_forcings` left every row bit-exact at every clock sample while the
   per-step walk's kt=2 velocity residual went from `2.2e-13` to `2.2e-02` m/s.
   The gate now calls the production function and scores what comes out of it,
   and the exoneration is re-measured on that path.
10. **"The title's claim that the whole year's difference is made in one step"
    — RETRACTED.**  In absolute rms, step 2 supplies `2.9 %` of the day-30 gap.
    What it owns is the BIRTH of the difference, not its size.
11. **The YEAR round's own `binary.sha256` does not describe the binary beside
    it.**  Its run.sh copies one provenance file into every run directory, so
    `nemo_pristine/binary.sha256` names the PATCHED binary while
    `nemo_pristine/nemo` hashes `a759e8…`, the certified one.  Measured here
    because this round's acquisition is admitted against that run; recorded as
    a defect of the year round's script, which this round does not edit.

## 8b.  REVIEWS

Two independent fresh Claude agents with no shared context: one attacked the
CLAIMS, one attacked the DIFF.  **Codex is occupied on this branch's kt=2
momentum ladder and GLM-5.2 is unavailable on this account, so neither
reviewer was codex or GLM.**  Stated because the repository's default is
codex + GLM and this round did not meet it.

Every finding was RE-MEASURED before it was acted on.  All of the ones acted on
reproduced.

### The CLAIM review

| finding | re-measured | what changed |
|---|---|---|
| the verdict splices two instruments into one curve and never shows they are one trajectory | the daily run's day-1/5/9 rows reproduce the walk's to every printed digit, its day-30 state is bit-identical to the scored member's, and NEMO's two day-30 restarts are byte-identical | section 1 now carries the three checks; the objection was correct and is closed by measurement, not by argument |
| "every later step multiplies by 1.002-1.045" is false where measured; 24 of 57 steps SHRINK `S`, `u`, `v` | reproduced: `T` `0.953`-`2.003`, `u` `0.773`-`9.310`, `v` `0.368`-`9.180` | the verdict is restated as a comparison of the largest multipliers; retraction 5 |
| a SECOND jump is hidden at step 3: `u` x`9.31`, `v` x`9.18` | reproduced | named in the verdict as a momentum-only second event |
| "salinity leads" is a normalizer artifact; under gap-over-field-std the order is `v`, `u`, `ssh`, `S`, `T` | reproduced from the same artifact, which already carried the second column | both normalizers tabulated; retraction 7 |
| the top-cell share is read BACKWARDS -- `7.5 %` on `3.3 %` of cells is `2.26x` enrichment | reproduced | retraction 6 and a corrected section 4 |
| both arms of the forcing gate use legoESM's OWN conservative-to-potential conversion, so a defect there is invisible | not re-measurable without NEMO's Fortran value; the bound that DOES exist is round 8's kt=1 `qns` row | stated as a named shared channel with its bound, in section 3 |
| NEMO SETS `avt = rn_evd*wmask`, so a firing interface with `wmask = 0` (the surface interface among them) is set to ZERO, which a maximum can never do | not measured this round | added as row C of section 5, UNMEASURED, with the census named |
| the oracle's per-step record stops at step 60, so no per-step claim past day 10 is falsifiable without a new NEMO run | read off `stprk3.F90:90` | stated in section 1 |
| `namelist_ref` carries `nn_fsbc = 2` and `ln_zdfevd = .false.`, both overridden by `namelist_cfg` | read off both files | stated in section 5 |
| the kt=1 row's exact `0.0` is a seed check, not a step check | agreed | stated in section 1 |
| the `<` vs `<=` threshold and the max-vs-set are inert on this card (`K_bg = 0`, `rn_evd = 100`, TKE `avt` is O(1)) | the card's resolved values printed | section 5 says so; both stay recorded, not fixed |

### The DIFF review

**Its first finding was that this round's central exoneration did not measure
what it claimed**, and it proved it by construction rather than by argument.

| finding | re-measured | what changed |
|---|---|---|
| **BLOCKING.** the forcing gate re-assembled the certified gate's composition instead of CALLING it, so it certified the usrdef_sbc STATEMENTS and said nothing about the forcing the MODEL CONSUMES.  Poisoning `_surface_forcings` to raise left the gate ALL BIT-EXACT with zero calls; TRANSPOSING `tau_x`/`tau_y` inside the real one left all five fields bit-exact at all thirteen samples while the per-step walk's kt=2 velocity residual went `2.2e-13` -> `2.2e-02` m/s | accepted as demonstrated; the construction is the measurement | the gate CALLS `_surface_forcings` and scores the model-facing rows (`sw_down`, the materialized `q_net`, `freshwater.evap`, `tau_i_native`, `tau_j_native`), plus two rows that invert the rotation.  Re-run: still bit-exact on all thirteen samples, and a new `forcing-stress-transpose` plant turns it red |
| **BLOCKING.** the switch trace evaluated only the NOW arm of `MIN(rn2,rn2b)` | already fixed one commit after the reviewed range: `before_tracers` is passed so the two-level branch executes | stated in section 6 |
| the trace omits the caller's `eos_fn` | under `n2_mode="nemo_bn2"` the trigger's N² takes `eos_form`, not `eos_fn`; the density `eos_fn` builds is consumed only by the `insitu` branch | recorded in section 6 as the trace's one unvalidated instrument assumption, labelled a code reading rather than a measurement |
| the acquisition stages `R41ADVSP` while `nemo_seed0/binary.sha256` names `YRPERT` | **the sha FILE is wrong, not the binary.** `nemo_pristine/nemo` hashes `a759e8…` = R41ADVSP, which is what the script stages; the year round's run.sh copies ONE `binary.sha256` into every run directory, so a file naming `578c88…` sits beside a binary that is not it | the script now compares the EXECUTABLE against the reference run's executable and refuses on a difference; the year round's provenance defect is recorded here |
| `--plant day-offset` never exits non-zero though the docstring claims every plant does | reproduced: the walks report numbers and carry no bar | the docstring says so and labels it a DIAGNOSTIC |
| `self_check`'s byte-unchanged check is a `for`/`else` with no `break`, so "OK" printed even after failures were appended | reproduced | rewritten as a list comprehension; `expect_raises` narrowed from `BaseException` |
| the run.sh test is a word-grep that passes with the wrong card, a deleted comparison and any binary | reproduced | it now pins the card name, both `cmp` invocations and the cadence |
| "one step makes the year's gap" holds on the relative column and not the absolute one — step 2 is `2.9 %` of the day-30 gap in rms | reproduced | the verdict carries both, and the title no longer says "the whole year's difference" |
| the `nyear` plant fires only because the bar is bit-exact; the term is a mathematical no-op | reproduced (`1.42e-13` W/m2) | already stated in section 3; restated as the plant's own bound |
| `glob_2Dsum` applies `smask0_i = ssmask*dom_uniq`, so both arms' de-meaning is correct — but the gate's coverage should be SAID | reproduced from `lib_fortran_generic.h90:92` and `dommsk.F90:203-205` | section 3's mask blind spot |
| `decompose` reads legoESM from the head root and NEMO from the year root with no legoESM provenance stamp | the two roots are correct (the head ensemble is the legoESM headline, the NEMO members live in the year root) but the stamp is missing | recorded as open debt; the two legoESM snapshots differ by `4.7e-08` K, which is the ENE commit the year receipt measures |
| `top_cell` overlaps the depth bands, so the depth shares sum above 1 | reproduced | section 4 lists it as a separate row with its own enrichment, not as a fourth band |
| `_load` re-executes and returns a fresh module each call | reproduced; constants only, no number moves | recorded, not fixed |
| `--snap-steps` has no CLI round-trip test | reproduced | recorded as open debt |

Its verdict was **SHIP WITH FIXES**, blocking on the first two rows.  Both are
fixed above and the gate was re-run after the fix.

## 9.  ASKED / UNASKED

| # | choice | value | ASKED? |
|---|---|---|---|
| 1 | the early-days NEMO record reuses the certified binary instead of a `makenemo` config copy | reuse | **UNASKED**, preregistered as such.  Offered for revert.  Reason: a second binary inside a one-variable comparison; the day-30 byte-identity refusal is the stronger check |
| 2 | NEMO trend diagnostics not enabled | not enabled | **UNASKED**, preregistered.  Offered for revert.  They write nothing without XIOS on this card, NEMO declares them untested under RK3, and an unclosed trend bucket may not be quoted |
| 3 | `_literal_sbc` gains `kt`/`nyear`/`qsr_pi`, defaults unchanged | additive | **UNASKED**, preregistered.  The alternative is a second copy of a transcription, which is forbidden; the defaults are shown byte-unchanged by the self-check |
| 4 | the time-level registry extends from 10 to 60 step-entry dumps | extend to the writer's own range | **UNASKED.**  Offered for revert.  It reads dumps the card already wrote and still fails closed one past them; no recorded number can move, because nothing previously READ those files |
| 5 | `run_member` gains a snapshot cadence, default unchanged | additive | **UNASKED.**  Offered for revert.  Every scored member is byte-unchanged |
| 6 | the EVD `<=` vs `<` and set-vs-max differences are RECORDED, not fixed | record | **ASKED**, pick: record.  Both are changes to the model's own convective switch and both would move every card that convects; neither can own the day-30 gap |
| 7 | the switch trace's window and cadence | 1260 steps, mask read every 6 | **ASKED**, pick: to day 210 so the year receipt's jump is inside it, daily so a crossing is dated to a day.  The cost of the cadence is stated in section 6: day 202 is an UPPER BOUND on the first crossing |
| 8 | the stress round trip is scored as a discrimination ratio, not at the bit-exact bar | ratio, bar `1e6` | **UNASKED.**  Offered for revert.  Inverting a rotation in binary64 costs `1.4e-17` Pa, so a bit-exact bar would go red on a correct model for an arithmetic reason; relaxing a bar is forbidden, so the row was rescored as what it is for.  Measured discrimination `8.6e+15` to `1.3e+16` |

## 10.  OPEN, RANKED

1. **Name the operator inside step 2, stage by stage.**  Section 1b establishes
   that step 2 makes a `4.15e-04` K difference on NEMO's own inputs, on 17999
   of 18000 cells.  It does not say WHICH of the step's operators does it.  The
   card's own stage writers (`l1_dump_stage`, `l1_dump_rhs`, `l2_dump_zdf`,
   `l1_dump_bt_frames`) fire at `kstp == nit000` only, so the measurement is:
   restart NEMO from its own kt=1 restart so that step 2 BECOMES `nit000`, then
   compare the three RK3 stages one at a time against legoESM seeded from the
   same entry state.  That is a config copy and a one-step run, and it is the
   whole campaign's next measurement: everything else in the GYRE year is
   downstream of it.
2. **Run the early-days acquisition** so days 10-29 stop being a gap in the
   record.
3. **Census the non-convective `K_v` above `100` m2/s** to decide whether
   NEMO's SET and legoESM's MAX ever disagree (section 5, row B), and the same
   census on `wmask = 0` firing interfaces for row C.
4. **Re-trace the switch at `--every 1`** across days 195-210 to turn day 202
   from an upper bound into the exact step.  Cheap: 90 steps of the pair.
5. Decide the `<=` vs `<` threshold comparison (section 5, row A) — a one-
   character change to a convective switch, so it is a question, not a diff.
6. Small debt named by the diff review and not closed: `decompose` stamps no
   provenance for its legoESM side; `--snap-steps` has no CLI round-trip test;
   `_load` re-executes a module on every call.
