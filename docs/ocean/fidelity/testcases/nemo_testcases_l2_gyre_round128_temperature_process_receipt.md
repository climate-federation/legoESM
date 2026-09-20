# NEMO testcase L2 GYRE phase 3 — round 128 temperature-process receipt

Date: 2026-09-20

Incoming tip: `68dd5eb367dfd351fe19cacc575677830833acdf`

Status: **HELD — implicit vertical diffusion is the largest propagated
temperature-process owner of the day-180-to-239 EVD-trigger share: `50,507`
absolute / `-49,647` signed cell-equivalents.  Advection is second at
`35,618.5` absolute / `+11,744.5` signed, refuting the preregistered
lateral-diffusion second place.  The eight registered rows close exactly to
Round 127's `+608` signed / `1,097` absolute temperature endpoint, but their
`147,054` absolute total exposes `134.0510483135825`-fold process
cancellation.  No physics lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round128/`

## Outcome first

Round 127 established that temperature is the largest upstream state input to
the EVD-trigger mismatch.  Round 128 now closes the next magnitude boundary.
At every daily entry from day 180 through day 239, the admitted independent
NEMO and legoESM process traces were accumulated in the preregistered compiled
order.  Every successive temperature boundary was then run through the full
production-JIT N2 closure in all four salinity/live-depth Shapley contexts.
The scored object is the propagated threshold effect of each temperature
process, not a whole-field T norm.

The primary ranking is:

| rank | temperature process | signed trigger cell-equivalents | absolute trigger cell-equivalents | continuous-margin signed / absolute sum (s-2) | sensitive-interface gradient RMS (K) | first observed daily entry |
|---:|---|---:|---:|---:|---:|---:|
| 1 | implicit vertical diffusion | `-49647` | `50507` | `-1.1541074755753505e-2 / 2.4681473687237603e-2` | `1.3748728869318652e-1` | day 181 / step 1087 |
| 2 | advection | `+11744.5` | `35618.5` | `+1.0117595654428692e-2 / 1.2236391599017887e-2` | `5.3982079359129055e-2` | day 181 / step 1087 |
| 3 | incoming before day 180 | `+21955` | `22057` | `+9.544067180617522e-7 / 2.0795815934994997e-5` | `1.9461229831333401e-3` | day 180 / step 1081 |
| 4 | surface boundary | `+8234` | `18470` | `-5.252203325308331e-3 / 7.999075451684243e-3` | `2.9802644484845698e-2` | day 181 / step 1087 |
| 5 | lateral diffusion | `+7580` | `17784` | `+6.653519483240642e-3 / 6.7414723858528515e-3` | `4.981729027808336e-2` | day 181 / step 1087 |
| 6 | shortwave | `+741` | `2617` | `-3.761242150679501e-7 / 3.6674704045092293e-6` | `1.7569084843305707e-4` | day 181 / step 1087 |
| 7 | free-surface content geometry | `+0.5` | `0.5` | `-2.634002967504862e-10 / 8.615815722939407e-10` | `1.3305818274996922e-7` | day 209 / step 1255 |
| 8 | floating-point/temporal closure | `0` | `0` | `-1.1398035037093213e-16 / 4.454629541900945e-15` | `3.6926709877782486e-14` | none |

The signed trigger sum is exactly `608`; the propagated bit telescope has
zero residual.  The continuous-margin telescope closes to
`2.168404344971009e-19 s-2`.  The ranking is a **frozen source-order
telescope**, so individual row magnitudes are order-dependent.  The exact
endpoint and its `134.05` cancellation ratio are order-invariant.  This
receipt does not relabel the rows as independently additive causal effects.

Vertical diffusion's spatial prediction is partly confirmed and partly
refuted.  Its `50,507` absolute cell-equivalents split as `40,678 / 9,829 / 0`
over 0--100 m / 100--1000 m / below 1000 m, and `30,809 / 19,698` south/north
of 37.2 N.  The preregistered west-third prediction is **REFUTED**: the split
is `11,898.5 / 15,611.5 / 22,997` west/interior/east, so east is largest.

This names the vertical-diffusion **temperature feedback** as the next
magnitude boundary.  It does not overturn Round 126's conditional result that
the closure becomes exact on NEMO's entry state, and therefore does not name
TKE or EVD arithmetic as faulty.  No first non-bit statement is claimed in
this diagnostic round.

No production physics, card, configuration, restart schema, carried state or
stabilizer changes.  Decision-43/45 landing gates therefore do not run and the
immutable before arms do not move.

## Frozen preregistration ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round128.md`, committed
before implementation and measurement as
`7b564ec72c567492c76ccf692fc26be67664d7d4`.

- **P0 CONFIRMED.** The process, vertical and legoESM traces re-admit.  The
  Round-127 endpoint reproduces 782 full disagreements, temperature
  `+608 / 1097`, zero NEMO/model endpoint mismatches and bit-identical
  returned `rn2/rn2b`.
- **P1 CONFIRMED.** The largest pre-closure wet-temperature reconstruction
  residual is `4.263256414560601e-13 K`, below the frozen `1e-12 K` bound.
  Every exact final temperature endpoint has zero unequal wet bits.
- **P2 PARTLY CONFIRMED, PARTLY REFUTED.** Vertical diffusion is largest.
  Lateral diffusion is not second; advection is.  The upper-depth and southern
  vertical-diffusion bins are confirmed; the west-third prediction is refuted
  by the east-third maximum.  The measured ranking above wins.
- **P3 CONFIRMED.** The incoming row is present at day 180, and the first
  non-incoming process threshold effects appear at day 181.  Both plants
  print `STATUS PLANT-FIRED` and exit 1.
- **P4 CONFIRMED.** This is diagnostic-only.  Only the existing owner
  instrument, controls, tests, preregistration and receipt change.

## Compiled statements and time-level alignment

Every source statement below is from a compiled branch that produced an
admitted record.  At the whole-step entry, the Round-125 program computes
alpha/beta and `rn2b` from tracer slot `Nbb`, copies `rn2b` to `rn2`, and calls
vertical physics with both formal time levels bound to `Nbb` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168`.
The TEOS-10 branch makes alpha and beta depend on the temperature, salinity
and live depth at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1259-1308`.
The later compiled loop interpolates those coefficients and consumes the
temperature values immediately above and below each W interface at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.
Thus the receipt reports both consumed temperature levels and their
difference; it does not treat a T-field RMS as the trigger owner.

The Round-123 writer records the stage-3 `Tbb` entry, all three free-surface
ratios and the active accumulator boundaries at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`.
The compiled stage calls advection then the RK3 surface boundary condition at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`, calls
shortwave and lateral diffusion at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`, then
calls implicit vertical diffusion and records `Taa` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.

Inside the implicit routine, the recorded pre-solve tracer content and
accumulated middle-level RHS enter the forward recurrence at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560`, and the
backward solve writes the after tracer at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578`.
The admitted source trajectory then swaps completed stage-3 `Naa` into the
next step's `Nbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222`.
These statements justify accumulating each process through the step before a
scored entry; day 180 has no in-window process increment and day 181 has six.

Finally, the scored threshold is the compiled EVD condition and replacement
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`.
Every process boundary reaches that statement through the production closure;
no isolated N2 transcription supplies a scored value.

## Temperature and production-N2 controls

The authoritative report is `round128/temperature_process_budget.json`,
generated by clean instrument commit
`e13b8a29f48dd4e51ee23bea3f7f7b8194b9a497` on CPU with the fp64/libm
policy.  A prior run at `1876f7970` produced no scientific report: it raised
an `IndexError` when a two-dimensional region mask was used as a direct
three-dimensional boolean index.  Commit `e13b8a29f` explicitly broadcasts
the mask and adds a direct shape/count test.  That failed run supplied no
ranking and is not citable evidence; the authoritative rerun replaced its
log.

| fail-closed control | measured result |
|---|---:|
| inherited full-trigger daily visits | `782` |
| inherited temperature signed / absolute | `+608 / 1097` |
| NEMO stored-mask endpoint unequal | `0` |
| legoESM stored-mask endpoint unequal | `0` |
| returned `rn2` versus `rn2b` unequal | `0` |
| exact recorded legoESM temperature endpoint unequal | `0` |
| production-JIT boundary/context evaluations | `2160` |
| maximum pre-closure temperature residual | `4.263256414560601e-13 K` |
| maximum propagated bit telescope residual | `0` |
| maximum propagated margin telescope residual | `2.168404344971009e-19 s-2` |
| signed process sum minus temperature endpoint | `0` |

At each boundary, temperature's marginal uses all four salinity/live-depth
contexts with the exact Round-127 Shapley weights `2/6, 1/6, 1/6, 2/6`.
Consequently the final boundary reproduces not merely a T-only endpoint but
the admitted three-input temperature allocation.  Its continuous-margin
signed and absolute sums are exactly
`-2.1584924289917324e-5 / 4.2348751127228244e-4 s-2`.

The sensitive-interface temperature rows also show why ranking by a global T
norm would be unsafe.  Vertical diffusion's upper/lower temperature RMS is
`7.867380308035571e-2 / 8.152414213670044e-2 K`; advection's is
`3.180491672751083e-2 / 4.700347768057534e-2 K`.  The trigger ranking is
based on the propagated mask, not either pair alone.

## First observed effects and controls

The incoming row first affects 23 interfaces at day 180 and carries 16
absolute cell-equivalents there.  Its true birth remains
**UNMEASURED before day 180**.  All five non-geometry physical process rows
first affect the threshold at day 181; geometry first affects one interface
at day 209.  The first vertical-diffusion effect has 397 nonzero interfaces,
386.5 absolute cell-equivalents, and a `387 / 10 / 0` upper/middle/deep depth
census.  Step 1081 is therefore still only the first acquired incoming state,
not a physical birth.

The process-registry plant removes the real vertical-diffusion row at day 181
while retaining the frozen closure row.  It breaks 18,000 endpoint cells with
a maximum `8.217134261183645e-3 K` residual, prints
`STATUS PLANT-FIRED`, and exits 1.  Its report/log are
`round128/plant_process_registry.json` and `.log`.

The consumed-level plant changes the real wet upper temperature operand at
interface `[8,4,2]` from `24.8703777606002 C` by
`9.313225746154785e-10 C`.  It flips one production threshold bit while the
repeated untouched endpoint moves zero bits, prints `STATUS PLANT-FIRED`, and
exits 1.  Its report/log are `round128/plant_process_level.json` and `.log`.

Neither control perturbs a zero, dry cell, unconsumed reconstruction or
printed summary.

## Landing and cross-card scope

No physical statement lands.  The admitted campaign headlines remain:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The only executable change is an analysis mode in the existing year-owner
instrument.  GYRE production, generic NEMO-GYRE, DINO, LOCK_EXCHANGE and
OVERFLOW execute no changed production statement, so no certified row moves.
ORCA2 is **UNMEASURED-WITH-SPEC**: repeat its native per-step process record,
independent production trace, four-context temperature propagation, exact
endpoint controls and threshold census before transferring this owner.

No configuration or carried-state question is exposed.  `DECISION_NEEDED` is
`NONE`.

## Independent adversarial review

The required command was attempted against the complete incoming-tip diff,
authoritative report, both plant reports and compiled sources with
`codex exec --sandbox read-only`.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
codex_exit_code=1
```

Independent review unavailable in-sandbox.  It emitted neither `SHIP` nor
`DO NOT SHIP`; no verdict is fabricated.  The complete log is
`round128/codex_review.log`.

## Verification

The owner self-check prints `self-check: all checks passed`; its log is
`round128/self_check.log`.  Python byte compilation and `git diff --check`
also pass.

The clean-tree focused suite covered the literal vertical solver, TEOS-10 and
`bn2`, the compiled-intermediate observer, the year-owner gate and all three
new Round-128 synthetic controls, the receipt-citation gate, and the
worktree-stamp ratchet.  Its exact summary is:

```text
======================== 97 passed in 89.27s (0:01:29) =========================
```

The log and JUnit report are `round128/focused_tests.log` and
`round128/focused_tests.xml`.

The receipt citation gate found 11 citations, zero unmapped citations, zero
failures and zero map-audit failures in `round128/citation_gate.json`.
Shifting the compiled `bn2` assembly citation by two lines produced
`SYMBOL-NOT-AT-LINE`, printed no PASS verdict and exited 1; its report/log are
`round128/citation_gate_shifted_plant.json` and `.log`.  The final clean-tip
rerun after this verification amendment preserves those counts.

## OPEN — round 129

Stay on the magnitude program.  Do not fix TKE or EVD from this table: Round
126 already proved that their production closure becomes exact on NEMO's
entry state.

1. Preregister the first six-step vertical-diffusion feedback decomposition,
   steps 1081--1086, whose accumulated temperature first changes the trigger
   at day 181.  Reuse the admitted Round-125 vertical frames and Round-126
   production trace; no acquisition is expected.
2. At the day-181 temperature-sensitive interfaces, propagate the existing
   vertical sub-boundaries — live pre-solve column, free-surface weighting,
   TKE/EVD heat coefficient, isoneutral coefficient, matrix association and
   solve association — through the same four-context production-N2 telescope.
   The rows must close vertical diffusion's day-181 `386.5` absolute trigger
   cell-equivalents and its two consumed temperature levels.
3. Separate coefficient response from column response at each of the six
   steps.  Preserve Round 126's conditional-on-NEMO endpoint as a control;
   do not call a coefficient row an intrinsic closure defect if that control
   remains exact.
4. Report the first step and spatial bin where the winning sub-row changes a
   threshold.  The incoming pre-day-180 state remains out of scope and
   `UNMEASURED`.
5. Only a source-cited statement that survives this feedback-localized
   production proof becomes a landing candidate.  Any production landing
   still requires the Decision-43 ladder/month and Decision-45 year gates
   against a same-tip before arm, plus DINO when the statement is shared.

No NEMO acquisition is expected for Round 129.  Request one only if a required
vertical sub-boundary cannot be reconstructed or read from the admitted
records; never infer the missing row.
