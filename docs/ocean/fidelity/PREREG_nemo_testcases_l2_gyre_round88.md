# Preregistration: NEMO-testcases L2 GYRE round 88 RK3 Kaa SSH scratch bundle

Date: 2026-09-13. Frozen at legoESM `66212eace1d1` after reading the Round
85--87 receipts, Decision 39, and the executing compiled source, but before a
Round-88 scientific comparison or production edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round88/`.

## Authorized boundary and magnitude rank

Decision 39 authorizes one shared NEMO-identity state slot for NEMO's
pre-solve RK3 Kaa SSH scratch, including restart persistence and a named hard
failure when an RK3 restart does not supply it. The Round-85 landed arm remains
the immutable comparison arm: `round85/bundle_kt1_10.json` and
`round85/bundle_day_gap.json`. Its kt2 U/V maxima are
`2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S are
`1.627497246303733e-04` / `6.327735185607253e-06`, and day-30 T RMS is
`1.2397011295506804e-02 K`.

Round 87 measured the scratch mismatch at all 600 wet cells with maximum
`1.144318797451864e-02 m`. Replacing only that operand removed both
`1.9220297482797664e-09` / `1.966061294804274e-09` downstream ZAD maxima.
This is larger than the retained first cumulative LDF boundary
(`2.5292467120726215e-14` U / `3.502735092670824e-14` V), so the authorized
scratch bundle is first by magnitude. The LDF and independent
`8.470329472543003e-22` association debts remain registered.

## Executing compiled statements

The GYRE RK3 driver calls `stp_2D` before its three stages
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:187-201`). In that
call, the vector branch forms `r3t(Kaa)` from the already-carried `ssh(Kaa)`,
then WZV consumes Kbb velocity and that Kaa slot before the external solve
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-175`). The active
QCO WZV recurrence consumes `e3t*hdiv` plus the Kaa-minus-Kbb stretch and
carries bottom-up
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:277-300`).

After stage 3, NEMO rotates the completed Naa state into Nbb and writes the
now-free Naa SSH slot as `2*ssh(Nbb)-ssh(Naa)`, explicitly for the next
step's W computation
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:215-229`). A cold
start sets Kaa equal to Kbb
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:373-410`). An RK3
restart writes that extrapolated slot as `ssha` and reads it back when present
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:153-185`,
`:350-371`). Decision 39 strengthens the legoESM admission contract: an RK3
restart without `ssha` is refused by name rather than silently taking NEMO's
MLF-to-RK3 fallback at lines 369--370.

## Bundle members and local proof

The one candidate contains only these source-linked members:

1. carry a two-dimensional Kaa SSH scratch only on NEMO WS-RK3 identity
   states; initialize it equal to Kbb on a source-defined cold start;
2. persist the slot in legoESM run restarts and bridge NEMO `ssha`; refuse an
   RK3 restart whose state lacks it with a named `NEMO_RK3_KAA_SSH_REQUIRED`
   message;
3. feed that carried slot to the pre-external stage-1 velocity-form WZV/ZAD
   path instead of reconstructing it;
4. preserve `divhor`'s separate `e3t*hdiv` materialization and `sshwzv`'s
   bottom-up recurrence with the already-proven source-round identity;
5. remove the later post-external transport-W substitution that the compiled
   vector stage-1 branch does not execute; and
6. after the completed RK3 step, rotate the scratch as the compiled
   `2*new_Kbb-old_Kbb` statement.

Each member must be re-proved on the final clean candidate. Required local
rows are: cold-start Kaa, bridged `ssha`, kt2 pre-solve scratch, `e3t*hdiv`,
every W carry, final W, both post-ZAD faces, and next-scratch rotation. Given
NEMO inputs, every row must have zero unequal cells. The restart round trip
must be bit-exact and deletion of the scratch manifest/payload must raise the
named failure. Plants perturb one scratch value, one W carry value, and the
restart inventory; all must print `PLANT_FIRED` and exit nonzero.

## Frozen prediction and falsifiers

Prediction: cold-start Kaa is bit-exact at kt1; the rotated carried scratch is
bit-exact at kt2; source-rounded W and both ZAD faces are bit-exact when the
production path consumes it. Against Round 85, both kt2 U/V maxima will be
strictly smaller than `2.7377110452773967e-12` /
`3.284922138989399e-12`, and the first-over-bar remains kt2 U/V. The kt3 T/S
maxima will be strictly smaller than `1.627497246303733e-04` /
`6.327735185607253e-06`. Day-30 T RMS will be strictly smaller than
`1.2397011295506804e-02 K`.

The prediction is **CONFIRMED** only if every local row is bit-exact, all
three plants fire nonzero, no certified AT-BAR row leaves the bar, first-over-
bar is not earlier, both kt2 targets move toward the bar, and kt3 T/S plus
day-30 T all improve. It is **REFUTED** if any local row remains non-bit, the
restart can omit/cold-fill the slot, a plant exits zero, an AT-BAR row becomes
DEBT, first-over-bar moves earlier, either kt2 target fails to improve, or any
of kt3 T/S and day-30 T fails its strict direction. A refuted bundle is held
and the receipt retains every moved/worsened row.

## Rule-12 lanes

| lane | frozen Round-88 disposition |
|---|---|
| GYRE local W/ZAD/scratch | Require all registered given-input and lifecycle rows bit-exact plus three nonzero plants |
| GYRE kt=1..10 | Compare against Round 85; no AT-BAR loss, no earlier first-over-bar, both kt2 U/V improve, every moved row registered |
| GYRE days 1..30 | Run only after the ladder passes; compare all five fields on days 1--30 against Round 85; require kt3 T/S and day-30 T direction above |
| LOCK_EXCHANGE-zco | The card executes WS-RK3 scratch initialization/rotation; compare all 50 certified rows against Round 85 and reject AT-BAR loss or earlier first-over-bar |
| OVERFLOW-zps | The card executes WS-RK3 scratch initialization/rotation; compare all 50 certified rows against Round 85 and reject AT-BAR loss or earlier first-over-bar |
| DINO | **SHARED-STATEMENT RISK:** leapfrog does not consume the RK3 scratch, but it calls the shared WZV helper; numerically compare a fixed DINO state before/after and reject any changed scored row |
| ORCA2 | **UNMEASURED-WITH-SPEC:** bridge `sshn` and mandatory `ssha`, align pre-solve Kaa/Kbb SSH, WZV operands/carries, ZAD, next-scratch rotation, restarts, and T/S/U/V/SSH on native masks through kt1..10 in fp64; reject missing `ssha`, any wet-input mismatch, AT-BAR loss, or earlier first-over-bar |

No coefficient, timestep, stabilizer, freshwater pair, year harness,
reconciliation gate, #1484 guard, held manifest, NEMO source, or NEMO
executable changes. The carried-state and hard restart requirement are exactly
Decision 39; no further configuration choice is made.
