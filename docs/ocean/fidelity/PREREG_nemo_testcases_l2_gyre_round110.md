# Preregistration: NEMO-testcases L2 GYRE round 110 magnitude ranking and stage-3 LDF content route

Date: 2026-09-18. Frozen at incoming lane tip
`51a4d088cf85ce1ea49b51b2ecf0ec294338e647`, before any round-110
measurement or numerical edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round110/`.

Decision 43 supersedes the Decision-41 stage-order walk and Rule 12's former
"no row may move" clause for this programme. This round first ranks the
measured month-gap owners and then tests the already-held stage-3 lateral
diffusion content route by the five Decision-43 landing conditions. It does
not resume the abandoned round-110 stage walk.

## Immutable before arm and magnitude target

The immutable before arm is
`phase3/merge_main_2026-09-17/after2/{ladder.json,day_gap.json}`. Its frozen
headline rows are:

| row | before |
|---|---:|
| kt2 T max abs | `1.4210854715202004e-14` K |
| kt2 S max abs | `2.1316282072803006e-14` psu |
| kt2 U max abs | `2.7377110452773967e-12` m/s |
| kt2 V max abs | `3.2849219221489645e-12` m/s |
| kt3 T max abs | `1.627497246303733e-4` K |
| kt3 S max abs | `6.327735185607253e-6` psu |
| day-30 T RMS | `1.2397011296352737e-2` K |

The month-gap ranking uses the committed
`nemo_testcase_l2_gyre_year_owners.py` instrument in `--self-check`,
`--step-gap`, `--day-gap`, and `--decompose` modes. The first calibration is
round 62's finding: the temperature gap is rounding-only after step 1 and is
born after step 2 at approximately `4.227e-6` K RMS; at day 30 the dominant
temperature squared-error cuts are the upper 100 m (approximately 77.19%) and
west third (approximately 50.99%), while the absolute peak is at `k=11` near
208.5 m. The peak is deliberately not described as lying in the upper-100-m
band. If the instrument fails this calibration, this round stops without a
candidate edit.

## Candidate statement, read from the compiled program

The record-producing GYRE compiled program clears tracer `Krhs` and then
accumulates advection and surface sources
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`). In
stage 3 it evaluates QSR, calls `tra_ldf`, snapshots that result, and then
calls `tra_zdf` in that order
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`). The
resolved record card selects isoneutral Laplacian diffusion. Its compiled
operator reads Kbb tracers
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`), uses
Kmm metrics to form the fluxes
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246`), and
adds the signed divergence into that same `Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305`). The
compiled vertical solve subsequently forms its content RHS from
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`).

legoESM already computes the signed GM/Redi rates `dT_gm` and `dS_gm`. Its
private diagnostic selector can route them into the stage-3 WS source tuple,
but the production path does not. The registered candidate removes that
private selector and performs the same route whenever the already-selected
WS tracer integrator executes with a configured GM/Redi operator. It adds no
configuration, coefficient, new operator, or stabilizer; `tra_zdf` keeps its
existing stage-3 half-step geometry.

The historical round-69 absolute local targets are already REFUTED by the
round-85 upstream landing. The current-tip private arm instead records T/S
stage-3 content maxima `5.743498263655056e-5` /
`7.387909136014059e-6` and kt3 T/S maxima
`8.600420500215478e-7` / `6.979443156751586e-8`. Those current-tip values,
not the stale round-69 numbers, are frozen below.

## Frozen predictions and falsifiers

**P1 — instrument calibration and immutable baseline.** `--self-check` will
pass. A 60-step walk will retain a rounding-scale step-1 temperature gap and
will first produce the approximately `4.227e-6` K RMS temperature gap after
step 2. Day-30 decomposition will retain the west-third/upper-100-m ownership
and peak `k=11` stated above. `--day-gap` over days 1--30 will reproduce the
immutable arm byte-for-byte, including day-30 T/S/u/v/ssh RMS
`1.2397011296352737e-2` / `2.1952855356523306e-3` /
`5.301687778753576e-4` / `5.632716531318026e-4` /
`4.4889711411874356e-4`. REFUTED by a different birth step, a different
dominant region/depth band or peak level, any mismatch to the immutable
day-gap JSON, or a failing self-check. A refutation stops the round before a
candidate edit.

**P2 — tip-local source-route proof.** Before the edit, the committed local
instrument must reproduce the current-tip private `ldf_only` arm. After the
production edit, both complete stage-3 content dictionaries and both complete
kt3 output dictionaries will be bitwise identical to that arm; both selected
source-injection rows will have zero unequal cells. The content T/S maxima are
predicted as `5.743498263655056e-5` / `7.387909136014059e-6`, and kt3 T/S as
`8.600420500215478e-7` / `6.979443156751586e-8`. The proof must run through
the production closure under JIT and with JIT disabled; an isolated-JIT row
may be reported but cannot substitute for either. REFUTED by one unequal
cell, by different complete dictionaries, or by a production eager/JIT
disagreement. A one-ULP production plant must print `STATUS PLANT-FIRED` and
exit nonzero.

**P3 — ladder forecast.** The full certified 954-row ladder will keep kt1
unchanged, keep first-over-bar at kt2 U/V, and keep the four kt2 headline
maxima exactly at their frozen before values. The kt3 T/S maxima will improve
to `8.600419718618468e-7` / `6.97944244620885e-8`. Based on the current-tip
held proof, exactly 53 registered rows are predicted to move; rows at kt>=2
may worsen under Decision 43. REFUTED as a numerical forecast if any frozen
headline differs or the moved-row count differs. Regardless of that forecast,
the candidate is rejected if first-over-bar becomes earlier, if any kt1
AT-BAR row leaves the bar, or if any moved row is omitted from the receipt.

**P4 — month forecast and landing direction.** Removing the known `1.63e-4`
K kt3 temperature owner is predicted to reduce day-30 T RMS from
`1.2397011296352737e-2` K to approximately `1.0e-4` K (forecast band
`1e-5`--`1e-3` K). The numerical forecast is REFUTED outside that band. The
Decision-43 landing condition is weaker but exact: day-30 T RMS must be
strictly below `1.2397011296352737e-2` K. If it is equal or larger, the
candidate stays held and receives no further fix this round.

**P5 — shared-card risk.** The production condition is specific to WS tracer
integration with an executing GM/Redi operator, so GYRE executes it. DINO's
resolved card and source must be inspected rather than inferred; if DINO also
executes it, the cheapest committed DINO gate will be run before and after and
every bit movement registered. LOCK_EXCHANGE and OVERFLOW are predicted not
to execute the statement because they have no configured GM/Redi route; this
is checked from their resolved cards and their cheapest committed tank tests.
ORCA2 remains **UNMEASURED-WITH-SPEC**: resolve its tracer integrator and
GM/Redi selector, then compare stage-3 post-QSR/LDF `Krhs`, pre/post-ZDF T/S,
and the next consumed state from identical fp64 inputs before and after.

## Decision-43 landing gate

The candidate lands only if all five conditions hold:

1. day-30 T RMS strictly decreases against the immutable before arm;
2. first-over-bar is not earlier;
3. no kt1 row that is AT-BAR leaves the bar;
4. every moved row, including every worsened row, is registered with before
   and after values; and
5. DINO is measured if source/card resolution shows that it shares the
   statement.

If the candidate passes, its complete ladder and day-gap artifacts become the
new immutable before arm and the shared implementation is committed. If it
fails condition 1, no repair is attempted this round; the receipt retains the
ranking and names the next-largest owner for round 111.

## Measurement order and controls

1. Commit this preregistration before creating round-110 evidence.
2. Run the owners self-check, 60-step gap, immutable days 1--30 day gap, and
   day-30 decomposition; emit one magnitude-ranked owner table.
3. Reproduce the private-arm local proof and plant at the incoming tip.
4. Apply only the registered production source route, re-run the local proof
   under production JIT and eager, then run the complete kt=1--10 ladder and
   days 1--30 member/score pair.
5. Resolve and run the shared-card checks, mechanically enumerate all moved
   ladder rows, and apply the five landing conditions.
6. Run a separate read-only Codex refutation pass. Run the citation gate and
   shifted-citation plant. Run focused tests and the complete
   `tests/ocean/fidelity` plus `tests/ocean/unit` trees with twelve workers;
   quote every terminal summary and diff the failing IDs against the frozen
   87-ID list.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No public configuration, default, coefficient,
timestep, carried state, restart schema, year harness, reconciliation gate,
freshwater pair, #1484 guard, or immutable before artifact is changed. No
stabilizer is introduced. No decision is inferred from a missing record.
