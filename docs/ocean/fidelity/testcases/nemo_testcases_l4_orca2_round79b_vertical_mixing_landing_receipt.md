# ORCA2 round 79b, second half — the wave arm measured, and what it does and does not fix

The record the first half asked for was acquired and admitted (`STATUS PASS`,
20 records, restart byte-identity against the pinned ten-step run).  This half
substitutes NEMO's own operands, names the owner, transcribes the arm and
measures the landing.

## The substitution table

Measured from the admitted record, first step, first rank, over its 242,941 wet
w-points.  The quantity every arm is scored against is the tracer diffusivity
the retired one-metre mixing length produced, which is what the round exists to
account for.

| arm of NEMO's vertical-physics chain | mean tracer diffusivity added [m2/s] | median |
|---|---|---|
| the retired 1 m mixing-length floor (what legoESM had) | 6.498e-05 | 9.962e-07 |
| **internal waves** (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:314-316`) | **4.997e-05** | 3.031e-06 |
| salt/heat split, on the momentum coefficient (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363`) | 2.756e-06 | — |
| river mouths (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:355`) | 0.000e+00 | — |
| convection (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359`) | 1.173e-01 | — |

The retired floor bound on **82.5%** of wet points, so it was active nearly
everywhere, and the wave arm is the same size as it.  **Prediction P3 is
CONFIRMED.**  Convection dominates the total but is already transcribed and
matching, and the river-mouth arm contributes nothing on this rank.

The record also settles something nobody had asked.  With the wave arm on, NEMO
does not use the namelist background coefficients at all: its wave-mixing
initialisation replaces them with molecular values and makes the equatorial
shape uniform.  Measured in the record: tracer background 1e-10, momentum
background 1.4e-6, shape 1.0 on every wet column.  The ORCA2 card had been
flooring at 1.2e-5 and 1.2e-4 with the equatorial shape — **86 times NEMO's
momentum floor and up to 120,000 times its tracer floor**.

## What was landed

One NEMO statement, "this deck runs internal-wave mixing", which is two halves
that cannot be separated in NEMO: the wave field itself and the backgrounds its
initialisation resets.  The card now states both explicitly, reads the six
wave-power and decay-scale maps from the deck's own product through the shared
loader (pass-through: the product is already on this card's grid, and the card
refuses if it is not), and takes the deck's constant-efficiency option.

Two deviations are declared rather than hidden: the deck's salt/heat
differential and its double-diffusive split both give salt a different
diffusivity from heat, and the implicit tracer solve carries one diffusivity
for both, so both are named in the card's unmeasured features.

## The ten-step ladder, and which half owns what

Given NEMO's entry, 200 rows, all three arms run on this branch at the same
protocol.  The before arm reproduces round 77's published numbers exactly, so
the baseline is confirmed rather than quoted.

| | rows toward / away / same | kt10 end-of-step T max | T rms | S max | S rms |
|---|---|---|---|---|---|
| before (round 77's after) | — | 1.11422 | 0.0106138 | 0.25508 | 0.00291914 |
| backgrounds only | 60 / 123 / 17 | 1.24107 | 0.0107669 | 0.288251 | 0.00290729 |
| backgrounds + wave field (landed) | 82 / 101 / 17 | 1.23675 | 0.0102148 | 0.287106 | 0.00280927 |

No bit-identical row left the bar in any arm, and the first non-bit statement is
unchanged: kt=1 stage-1 temperature, identical to the last digit across all
three.

**The split is the result.**  The background reset alone carries the whole
maximum degradation (+11.4% on temperature) and makes the rms slightly worse.
Adding the wave field on top moves **117 rows toward NEMO against 66 away** and
turns the rms from 1.4% worse into 3.8% better on both headline tracers.  So the
wave arm is the corrective half, exactly as the round predicted — but the pair
together still leaves the end-of-step maxima about 11% further from NEMO than
the old configuration, because the oversized background floors had been doing
part of the wave field's job and doing it in a way that flattered the extremes.

**Prediction P4 is PARTLY REFUTED as written.**  The wave arm is corrective
relative to the background reset, and it is not enough to return the maxima to
where the compensating error had them.

## What was not done, and why

* **No bit-exact gate on the wave diffusivity.**  The record carries the
  coefficients at every boundary but not the buoyancy frequency, and the
  scheme's Reynolds number is built from it, so NEMO's own wave diffusivity
  cannot be reproduced from the record alone.  Closing this needs one more
  field in the writer, not a new instrument.
* **No independent month, no DINO, no tanks, no GYRE year.**  The month is
  2.6 hours; the round's budget went to the ladder's three arms.
* The GYRE ten-step is measured, below, and is byte-identical.

## GYRE

Run at both tips at the same protocol, ten steps, trajectory only.  Every one of
the 70 residual rows, every step's row set, every barotropic state and the
first-over-bar step are IDENTICAL before and after (temperature, salinity,
velocity and sea surface first over the bar at step 3; the barotropic pair at
step 2; standing status DEBT, unchanged).  GYRE does not execute either half of
the change: its card keeps the wave arm off and the generic background floors,
which the resolved-configuration diff confirms is the only difference the new
defaulted field makes.  Per the standing reading of the shared-statement gate,
byte-identical is a PASS.

## Scoping — which cards execute the changed statements

The model diff is four files: one defaulted boolean on the wave-mixing
configuration, a refusal reachable only when that boolean is true, one defaulted
slot on the recipe container, and the ORCA2 card block.  The GYRE, OVERFLOW,
lock-exchange and VORTEX cards were built before and after: nothing changes but
the presence of the new defaulted field in the printed configuration, and all
four keep the wave arm off with no maps.  Only ORCA2's deck selects either arm.

## Review

Codex refused (quota guard 84.0%, resets 2026-10-01 21:50).  One Claude
code-reviewer: **SHIP WITH CHANGES**, one blocker and two should-fixes, all
closed in-round.

The blocker was real and reproduced against the deck: eight ORCA2 gates build a
model from the card's own configuration without threading the maps and now hit
the new refusal.  All eight, twenty-seven call sites, now pass them.  The first
should-fix was a second reader duplicating a loader this repository already has,
which in duplicating it dropped that loader's guard against a non-positive decay
scale before inverting it; the card now calls the shared loader and the loaded
maps were verified equal to a direct masked read.  The second was three stale
rows in the entry audit that said the card owed NEMO something it now matches.
The reviewer independently confirmed the six-field transcription, the two
background constants, the initialisation order that keeps the carried seeds on
the namelist values, the scoping, and that the refusal is unreachable for every
other consumer.

## OPEN

1. **DECISION.** The landing is faithful and improves the volume measure while
   worsening the extremes.  See the decision line at the end of this round's
   report.
2. The salt/heat differential and the double-diffusive split are declared,
   not built; both need a second tracer diffusivity in the implicit solve.
3. The river-mouth diffusivity is still untranscribed; it contributes nothing
   on rank 0 at step 1 and has not been measured elsewhere.
4. One more recorded field would make the wave diffusivity bit-gateable.
