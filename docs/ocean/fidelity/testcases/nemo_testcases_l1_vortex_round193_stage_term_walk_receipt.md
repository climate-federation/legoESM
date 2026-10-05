# NEMO testcase fidelity: round 193 / VORTEX round 8 stage-term walk

**Status: HELD.** The first non-bit live stage boundary is the stage-2 HPG
accumulator: legoESM has already associated KEG with HPG, while NEMO assigns
HPG first and calls VOR, KEG, then ZAD. The source-order candidate makes HPG
bit-exact but does not move VORTEX kt=2 U/V, so its preregistered magnitude
prediction is refuted and production is restored.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round193/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round193.md`.

## Admission and compiled statements

The operator-produced round-192 record was independently re-admitted:
`restart_byte_identical=true`, status `ADMITTED`; the corrupted-extent plant
exited 1 with status `REFUSED`.

The record's own compiled build establishes the order. Stages 2/3 call HPG,
VOR, then vector advection in
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:323-338`.
The vector call executes KEG before ZAD at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:135-141`.
On this z-coordinate build HPG overwrites `Krhs`, rather than adding to it, at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:320-334`.

## Corrected production-JIT boundary walk

Every row below is the cumulative accumulator after the named compiled
boundary, driven from NEMO's recorded stage entry through `model.step`
(`_step_jitted`), CPU fp64/libm. Counts are over the card's live face-level
cells.

| stage | boundary | U unequal / max abs | V unequal / max abs |
|---:|---|---:|---:|
| 2 | HPG | `36600 / 6.7246815857365635e-06` | `36600 / 6.7126619267201494e-06` |
| 2 | VOR | `36580 / 6.7246815857365661e-06` | `36600 / 6.7126619267201511e-06` |
| 2 | KEG | `22569 / 1.3552527156068805e-20` | `22546 / 2.7105054312137611e-20` |
| 2 | ZAD | `26782 / 1.8741000428408085e-09` | `26614 / 1.8522221618642312e-09` |
| 3 | HPG | `36600 / 6.7436631087872815e-06` | `36600 / 6.7312507731435910e-06` |
| 3 | VOR | `36580 / 6.7436631087872806e-06` | `36590 / 6.7312507731435876e-06` |
| 3 | KEG | `22445 / 2.0328790734103208e-20` | `22177 / 1.3552527156068805e-20` |
| 3 | ZAD | `36424 / 1.8905432975169711e-09` | `36368 / 1.8685471232664545e-09` |

The calibration reproduces the pre-registered completed-stage rows exactly:
stage-2 U/V `1.6701813777553198e-06` / `1.6506840992414062e-06`; stage-3 U/V
`3.3737007034684297e-06` / `3.3344424458237043e-06`.

The private source-order arm makes both stage-2 and stage-3 HPG rows BIT
(0 unequal). Its next non-bit boundary is VOR at last-bit scale; ZAD remains
the magnitude boundary at about `1.9e-09 m s-2`. The production HPG mismatch
is exactly the prematurely associated KEG magnitude (`~6.7e-06 m s-2`).

### Retraction

An initial committed probe compared isolated legoESM terms with differences
of consecutive NEMO accumulator dumps and provisionally named VOR first at
`2.03e-20 m s-2`. **RETRACTED:** subtracting two rounded accumulators cannot
recover the intervening source term bit-for-bit. The gate now compares the
cumulative boundaries directly; its earlier artifact is retained as
`stage_terms.json` and is not evidence for an operator attribution.

## Candidate and trajectory veto

The candidate made the source-order arm the shared vector WS-RK3 stage
association. Local post-HPG rows remained BIT. The VORTEX-vector ladder then
refuted the frozen prediction: kt=2 U/V were unchanged at
`3.3693297863401916e-06` / `3.337024e-06`, not at least 10% better. kt=1 was
unchanged and the first-over-bar row remained kt=2.

Twenty-four later VORTEX rows moved at last-bit scale; all are registered:

| row | before | candidate |
|---|---:|---:|
| kt3 U | `4.010084241057310e-06` | `4.010084240724243e-06` |
| kt3 V | `3.983891562253650e-06` | `3.983891562309161e-06` |
| kt4 U | `2.372738925227047e-06` | `2.372738925338069e-06` |
| kt4 V | `2.329281361790336e-06` | `2.329281361568292e-06` |
| kt4 SSH | `8.230603660037885e-06` | `8.230603659929248e-06` |
| kt5 U | `3.712188373694580e-06` | `3.712188373805603e-06` |
| kt5 V | `3.595140521395202e-06` | `3.595140521506224e-06` |
| kt5 SSH | `8.026674186059363e-06` | `8.026674186127885e-06` |
| kt6 T | `1.724755055690787e-07` | `1.724755054822483e-07` |
| kt6 U | `4.743071119972520e-06` | `4.743071119528430e-06` |
| kt6 V | `4.596828561687083e-06` | `4.596828561465038e-06` |
| kt6 SSH | `8.026496078068551e-06` | `8.026496078068117e-06` |
| kt7 V | `5.314326005700920e-06` | `5.314326005589898e-06` |
| kt7 SSH | `5.232805556677116e-06` | `5.232805556520774e-06` |
| kt8 U | `5.939296320978116e-06` | `5.939296321089138e-06` |
| kt8 V | `5.903662329220794e-06` | `5.903662329109771e-06` |
| kt8 SSH | `6.015781366473128e-06` | `6.015781366435126e-06` |
| kt9 U | `5.592987247989319e-06` | `5.592987248320715e-06` |
| kt9 V | `5.483716119818922e-06` | `5.483716120040269e-06` |
| kt9 SSH | `4.593345490368954e-06` | `4.593345490286989e-06` |
| kt10 T | `1.495331188581169e-07` | `1.495331189451039e-07` |
| kt10 U | `4.865475868018781e-06` | `4.865475868897195e-06` |
| kt10 V | `4.724706785840426e-06` | `4.724706786060600e-06` |
| kt10 SSH | `5.356564719082377e-06` | `5.356564719003013e-06` |

Because the candidate failed its preregistered magnitude falsifier, it was
removed from production. A certified GYRE ladder attempt was terminated after
making no progress for over 13 minutes and produced no artifact; no GYRE,
month, or year claim is made for the candidate. This is a candidate veto, not
a landing with missing scope.

## Retained diagnostic, plants, and shared-card check

The only retained model-file change extends an existing private WRITE-only
hook so gates can expose cumulative boundaries after the ordinary production
step. No card or public configuration can select it. The `s2.hpg.u` planted
output violation fired its own row and exited 1 with `STATUS PLANT-FIRED`.
Reader tests cover missing groups and preserve the record's declared shapes.

Because a package file changed, the mandatory private-workdir DINO month gate
was run on the restored production tree:

> `DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`

## Review and tests

The separate read-only Codex review was attempted. Verbatim result:

> `independent review unavailable in-sandbox — Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

Focused gate controls before the candidate measurement: `41 passed`. The final
focused suite, including the round-193 gate, VORTEX kt2 walk, affected operator
seams, and the complete receipt-citation test module: `87 passed in 7.85s`.
The receipt citation gate passed all three compiled citations; its shifted-line
plant exited 1 with `SYMBOL-NOT-AT-LINE`. No broad ocean battery is required for
a private diagnostic-only seam; the production candidate is not present.

## OPEN — next VORTEX round

Continue at the magnitude-owning ZAD boundary, not the harmless source-order
association. The direct cumulative ZAD residual is
`1.8741000428408085e-09 / 1.8522221618642312e-09 m s-2` at stage 2 and
`1.8905432975169711e-09 / 1.8685471232664545e-09 m s-2` at stage 3. Use the
recorded `ww`, stage velocity, and face thickness to split dyn_zad operands in
compiled order under production JIT. The remaining resolution ladder stays
blocked until the VORTEX-vector kt=2 magnitude owner is closed or proven to be
the harness floor.
