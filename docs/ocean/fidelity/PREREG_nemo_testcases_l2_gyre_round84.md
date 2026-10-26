# Preregistration: NEMO-testcases L2 GYRE round 84 cumulative momentum-RHS walk

Date: 2026-09-13. Frozen at legoESM `9c8cb301ab31` after reading the Round
82--83 receipts, the iteration-5 handoff, the admitted Round-64 record reader,
the existing Round-46 source-order accumulator, the Round-51 live operand
trace, and the acquired card's compiled source, but before running a new live
kt=2 comparison or reading a Round-84 scientific value.

## Inherited boundary and magnitude ranking

Decision 37 remains YES, but the Round-79 six-absolute-history candidate and
the Round-82 drag/inverse pair remain **HOLD**. Round 80 measured zero movement
in all 954 parent/candidate rows, and Round 82 left the required kt2 U/V rows
at `2.7478404751243857e-12` / `3.305560306813421e-12` while worsening later
rows by up to 7,815 row-scale ULPs. Nothing in this round reinterprets those
results.

Round 83 measured the complete kt2 three-dimensional RHS gap at all 17,400 U
and 17,100 V wet faces, maxima `1.9220276136603893e-09` and
`1.966059508480186e-09`. Replacing live geometry by NEMO's reference geometry
moved the depth mean only `1.9852334701272664e-23` U and
`2.150669592637872e-23` V. The cumulative RHS is therefore the next ranked
magnitude walk; geometry is not revisited as an owner.

## Executing compiled source and ordered rows

The admitted record was written by
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90`. The executing vector
branch calls EOS and HPG first, with HPG overwriting Krhs, then accumulates LDF
and VOR at `:141-153`. It constructs WZV and calls KEG then ZAD at `:155-175`;
the write-only `after_adv` snapshot at `:176` follows ZAD with no intervening
model statement. The final vertical average consumes that accumulator at
`:202-208`.

The resolved namelist selects `ln_hpg_sco=.true.` and
`ln_hpg_zco=.false.`. The compiled dispatcher therefore takes `hpg_sco`
(`.../dynhpg.f90:153-177`). That routine forms the surface HPG from `rhd`,
`r3t`, reference `e3w_1d`/`gdept_1d`, SSH, gravity, and the U/V reciprocal
metrics, then overwrites Krhs at `dynhpg.f90:378-409`; levels 2..jpkm1 recur
the hydrostatic gradient and overwrite Krhs at `:411-435`.

## Existing instruments reused

No NEMO acquisition is requested. The measurement consumes the admitted
Round-64 copy of `oracle_momstage_kt00000002_s1.bin` only after its existing
43-exact/20-classified/132-difference admission, producer stamp, schema, EOF,
presence flags, and zero-owned-difference projection pass. It runs the current
shared GYRE implementation through the existing production-JIT/fp64/libm
Round-51 `expose_live_stage_operands` seam at kt=2. That seam returns the
already-computed stage-1 HPG, LDF, vorticity, KEG, and ZAD arrays without
feeding them back into production.

The source-order accumulation extends the existing Round-46
`stage1_accumulators` operation: HPG; barrier after HPG+LDF; barrier after
+VOR; barrier after +KEG; barrier after +ZAD; `after_adv` identity. Each row is
scored U and V on the native three-dimensional wet mask with float64 bit
equality, differing-cell count, absolute maximum, reference maximum, and
normalized maximum. A closure row requires the reconstructed final accumulator
to equal the live `du_dt`/`dv_dt` captured by Round 83; failure names an
unregistered live contribution or association boundary and withholds operator
ownership.

For the first cumulative boundary whose absolute residual reaches at least
one tenth of the inherited final-RHS maximum on either face, the same report
scores its already-recorded/live inputs in compiled source order. If that row
is HPG, the input order is T, S, SSH/r3t, in-situ density anomaly, live/reference
vertical geometry, reciprocal U/V metric, then the HPG result. Velocity is
reported separately as a handed but non-consumed HPG-array member; it cannot
be promoted as an HPG cause.

## Frozen prediction and falsifiers

The prediction is that `after_hpg` is the first non-bit cumulative boundary
and the first magnitude-bearing boundary: at least one face has
`after_hpg.absolute_max >= 0.1 * inherited_final_rhs_max`, and no later
increment changes the residual maximum by a larger amount than HPG introduced.
The predicted first non-bit HPG input is the in-situ density anomaly derived
from the already non-bit T/S state, while reference vertical geometry and the
native masks remain bit-exact. This is an input-boundary prediction, not a
claim that the shared HPG formula is wrong.

**CONFIRMED** requires record admission, trace non-interference, exact
source-order/live-total closure, `after_hpg` first by source order, the HPG
magnitude criterion above, and the density row preceding the result being
non-bit. It is **REFUTED** if a prior registered HPG operand is first, HPG is
below the magnitude criterion, a later operator introduces the largest
residual, density is bit-exact, or the final closure fails. A refuted result is
retained and the measured first/largest row becomes the next walk; no post-hoc
reordering is allowed.

Three differential controls must print `FIRED` and exit nonzero:

1. one ULP at the maximum-residual wet oracle HPG result must change that row's
   reported maximum;
2. one ULP at the maximum-residual wet oracle density input must change its
   reported maximum and the frozen prediction status; and
3. one ULP at the maximum-residual wet live-total closure cell must make the
   closure non-exact.

A dry, zero, overwritten, aggregate-invisible, or already-green perturbation
is a failed control, not evidence.

## Conditional implementation and Rule 12

No production change is pre-authorized. If all inputs to a first large shared
statement are bit-exact and only its result is non-bit, the smallest literal
compiled-source transcription may be tested. If the first large boundary is
an imported/non-bit input or closure fails, this round lands diagnostics only
and hands the exact upstream statement to Round 85. The held Decision-37 pair
may be reconsidered only after this upstream owner is exact.

| lane | frozen Round-84 disposition |
|---|---|
| GYRE cumulative kt2 RHS | Admit the Round-64 stage-1 record; score `after_hpg`, `after_ldf`, `after_vor`, `after_keg`, `after_zad`, `after_adv`, and live-total closure in compiled order |
| GYRE kt=1..10 | Before arm remains `decision36_nemo_face_shear/after_kt1_10.json`; run only if production physics changes; register every moved row, preserve every AT-BAR row, and forbid earlier first-over-bar |
| GYRE days 1..30 | Before arm remains `decision36_nemo_face_shear/after_day_gap.json`; run only after a production candidate passes kt=1..10 |
| LOCK_EXCHANGE-zco | Same split-explicit program; if a shared RHS statement changes, measure its recorded kt rows |
| OVERFLOW-zps | Show the changed statement does not execute or measure it against its record; no neutrality inference |
| DINO | **SHARED-STATEMENT RISK:** its leapfrog card carries separate histories but executes shared HPG/LDF/VOR/KEG/ZAD implementations; 96--98% row cancellation forbids neutrality inference |
| ORCA2 | **UNMEASURED-WITH-SPEC:** independently align every cumulative RHS boundary and its operands, T/S/U/V/SSH, masks, reference/live geometry, and six histories on native staggering; require fp64 bit equality and normalized L-infinity through kt=1..10; reject AT-BAR loss or earlier first-over-bar |

No configuration/default, selector, coefficient, timestep, stabilizer,
carried-state representation, restart format, year harness, reconciliation
gate, freshwater pair, #1484 guard, held manifest, or NEMO source changes. No
scientific choice is made.

