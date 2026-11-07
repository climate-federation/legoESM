# NEMO testcase Lane 4 — ORCA2 card round 8 preregistration

Date: 2026-09-22

Parent: `6674da8cca15dd0d4a1e0f1e4eb5ff4c1b3bfe6e`

Status: **PREREGISTERED BEFORE ANY ROUND-8 MEASUREMENT.**

Scope is round 7's OPEN item 1 on the ocean-only `orca2_vector_een_c2` card:
the four-cell vertex thickness the energy-and-enstrophy vorticity operator
needs, on the tripolar fold row, where the production helper currently
refuses.  The six-entry sea-ice registry is frozen and stays out of scope.
Decision 52's labels are binding: every number is either **given NEMO's entry**
or **independent**, never mixed in one table.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment:

| resolved setting | value | where it is printed |
|---|---|---|
| vorticity scheme | energy and enstrophy conserving | run `ocean.output:1338-1351`; `namelist_cfg` `ln_dynvor_een = .true.` |
| vertex-thickness rule | `nn_e3f_typ = 0`, masked four-cell average divided by four | run `ocean.output:1345` |
| north-fold type | present, **T pivot** | run `ocean.output:207-208` |
| inner global domain | 180 by 148, halo width two | run `ocean.output:38,44,68-69` |
| rank layout | two ranks split in longitude only | run `ocean.output:196-202` |

No selector, default, tunable, threshold, resolution, timestep, carried state
or data source moves in this round.  The transcription is the missing half of
an operation the card already selects.

## The statement this round transcribes

NEMO does **not** give the fold row its own vertex-thickness formula.  It
evaluates the ordinary masked four-cell average over the owned domain
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:913-919`),
then completes the field with the ordinary F-point north-fold exchange
(`:935`), and only then replaces any remaining zero with the reference
thickness (`:937`).  The exchange is what defines the fold row: for a T-pivot
fold on an F-point field it rewrites the last owned row from the row below it
at the mirrored longitude with the caller's sign
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:722-746`).

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R8-P1 | The compiled fold branch for an F-point field under a T pivot rewrites the LAST OWNED row (not only halo rows) from the row immediately below it, at the mirrored longitude, with the caller's sign, and the vertex-thickness call passes sign +1. | The compiled row loop reaches the last owned row and the compiled longitude loop pairs source and destination to a constant sum; the vertex-thickness call site passes +1. | The loop stops above the last owned row, the longitude pairing is not a mirror, or the sign is not +1. |
| R8-P2 | That rule is ALREADY implemented once in legoESM and is not re-derived here: the vertex-thickness helper used by the vorticity operator is the ONLY caller that refuses it. | A repository search finds one existing shared fold helper carrying this rule and exactly one refusing call path. | A second implementation exists, or more than one path refuses. |
| R8-P3 | With the refusal replaced by that shared rule, the card's frozen vertex thickness on the fold row is bit-identical to NEMO's recorded operand at EVERY recorded F point of that row, in both fold directions — the ninety owned longitudes whose sources lie in the other rank's half, and the four recorded halo longitudes that lie in the other rank's half themselves. | Zero unequal doubles over all 94 recorded longitudes by 31 levels on the fold row. | Any unequal bit. |
| R8-P4 | The transcription is inert for GYRE by construction: GYRE's fold descriptor is inactive, so the new code is unreachable there. | Base and tip GYRE ten-step ladders give zero differing rows and array-equal residual arrays, and the thirty-day member snapshots are byte-identical. | Any differing row, unequal residual array, or differing snapshot digest. |
| R8-P5 | With the vertex thickness defined, the ORCA2 ladder advances past this stop and reaches a first non-bit ARITHMETIC statement at kt=1, which is named with a cited compiled owner and a registered kt=10 magnitude. | The ladder returns a first non-bit row inside a Runge-Kutta stage, with a kt=10 magnitude for the same field. | The ladder stops on another unbuilt statement, in which case that statement is named and cited and the kt=10 magnitude stays UNMEASURED. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- Perturbing one representable value of the recorded vertex-thickness operand
  must make the fold gate refuse and name the offending longitude and level.
- Removing the fold transcription must make the same gate refuse, so the gate
  cannot pass vacuously.
- The fold gate must also refuse if the fold row is compared with the wrong
  source row or the wrong mirror, so a self-consistent but wrong permutation
  cannot pass.
- A rigid two-line shift of a round-8 compiled citation must fail the citation
  gate.
- GYRE's trajectory is proven unchanged before anything lands, at the round's
  base tip and at its final tip, with the evaluation protocol byte-identical.
- No stabiliser, clip, damp or limiter NEMO lacks may be added.  If the fold
  row needs one to stay finite, that is the finding and the round HOLDS.
- The other half of the fold row is recorded only through the four halo
  longitudes the rank-zero writer emits.  If a full-rank record is required to
  certify the remaining eighty-six longitudes, this round writes the
  acquisition and reports ACQUISITION_NEEDED rather than asserting them.

## Labels

Every vertex-thickness number in this round is **independent**: it is built
from the card's own reference thickness ladder and masks, with no operand
loaded from the record.  Decision 52's sea-surface-height bridge is unchanged
and remains the only explicit entry replacement.
