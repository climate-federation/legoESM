# NEMO testcase Lane 4 — ORCA2 card round 30 EEN-suboperand receipt

Date: 2026-09-26

Parent: `325384a7dc`

Status: **HELD — FROZEN `e3f_0vor` ALONE OWNS THE WHOLE EEN ARM.**
Substituting only NEMO's carried frozen F thickness reproduces round 29's
all-three-operand stage-2 result bit-for-bit, reproduces every round-28 EEN
trajectory score through kt=3, and triggers the same fail-closed raw-`e3w`
refusal entering kt=4.  No model statement lands.  Every experimental
`packages/` edit is reverted, and the final `packages/` tree is identical to
the parent.

Every result is **independent with Decision-52 SSH**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple are unchanged.  Evidence
is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round30/`.

## Compiled statements and controlled boundary

The admitted build constructs the masked reference average, exchanges its
north-fold row, and replaces remaining zeros at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`.
It constructs live `r3f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`
and freezes `fe3mask` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90:258`.
The executed EEN reciprocal consumes exactly
`e3f_0vor*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.
The admitted `ocean.output` resolves `ln_dynvor_een = T`,
`nn_e3f_typ = 0`, and `ln_dynvor_msk = F`.

The gate extended round 29's production observer.  Five clean-worktree
processes ran the parent, each single carried operand, and all three carried
operands.  The source seam selected only an operand source inside the existing
production thickness builder; the EEN numerator, transports, masks, metrics,
forcing, entry, precision, backend, and score stayed fixed.

## Independent operand result

At the first production EEN call, the single frozen-`e3f_0vor` arm and the
all-three arm have identical denominator and exposed stage-2 arrays.  The
denominator changes **455,904 / 809,070 cells**, maximum
**651.2256783597969 m**.  Exposed U changes **413,554 / 803,640 cells**,
maximum **2.1873555668998308e-05 m s-2**; exposed V changes
**412,558 / 804,600 cells**, maximum
**3.213274876559519e-05 m s-2**.

| one-operand arm | denominator versus parent | first raw EEN output | exposed stage-2 accumulator | verdict |
|---|---:|---:|---:|---|
| frozen `e3f_0vor` | 455,904 unequal; max 651.2256783597969 m | 88 U / 217 V bit patterns (signed-zero-only magnitude at this first call) | 413,554 U / 412,558 V unequal | exact whole-arm owner |
| live `r3f` | 376,492 unequal; max 0.008365710955672512 m | bit-identical | 412,170 U / 411,143 V unequal, max 5.153126997217792e-09 / 2.5500881043307236e-09 | secondary later-call sensitivity; preregistered first-output prediction refuted |
| frozen `fe3mask` | bit-identical | bit-identical | bit-identical | exonerated at this boundary |

The carried and reconstructed `fe3mask` arrays are bit-identical in
**799,200 / 799,200 cells**.  The all-three exposed U/V arrays equal round
29's saved raw-F arrays bit-for-bit.  The frozen-`e3f_0vor` arrays equal the
all-three arrays bit-for-bit, so no combination claim is needed for the owner.

## Independent owner trajectory

The isolated frozen-`e3f_0vor` arm kept lateral diffusion on the parent
thickness path.  Its kt=1..3 document has **12 / 12 checkpoints exactly equal**
to round 28's all-denominator EEN-only document, including the unchanged first
NEMO mismatch at kt=1 stage-1 temperature.  Entering kt=4 it fails closed with
`raw-mesh e3w_int must contain only finite values > 0`, exactly as the whole
EEN arm did.  It therefore owns the whole measured EEN trajectory movement
and refusal, but it cannot land until the compensating statement is found.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R30-P1 | **CONFIRMED** | `e3f_0vor` is the largest arm and exactly reproduces the all-three exposed result. |
| R30-P2 | **REFUTED** | `r3f` changes the denominator and later exposed accumulator, but its preregistered first raw EEN output is bit-identical. |
| R30-P3 | **CONFIRMED** | carried/reconstructed `fe3mask`, denominator, and EEN result are bit-identical. |
| R30-P4 | **CONFIRMED** | all-three equals round 29; `e3f_0vor` alone equals it and reproduces the kt=1..3 document plus kt=4 refusal. |
| R30-P5 | **CONFIRMED** | the operand and refusal plants are rejected. |

The failed prediction is retained, not rewritten.

## Gate, review, and tests

The operand outcome gate exits 2 with `HELD`; its carried-mask plant exits 1.
The owner-trajectory gate exits 2 with `HELD`; its score and refusal plants
exit 1.  The all-three calibration is 0 unequal U/V cells against round 29.

The required separate `codex exec --sandbox read-only` review was attempted on
the committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The citation gate passes all four compiled citations with zero failures, zero
unmapped citations, and zero map-audit failures.  Its rigid +2-line EEN
reciprocal plant exits 1 with `SYMBOL-NOT-AT-LINE`; all nine citation
self-tests fire.  The preregistration's initial `domqco` header citation was
rigidly re-anchored to the executed lines 273–286 before the receipt;
no prediction or result changed.

The focused round-27/28/29/30 and citation battery reports **25 passed**.
The one required `tests/ocean/fidelity -n 12` battery collected 1,876 tests
and reached the inherited final-tail stall after **1,857 passed, 7 skipped,
5 failed, and 7 unfinished** were emitted; it was interrupted after a bounded
idle wait at 99%.  Isolated reruns reproduce the same five historical
failures as round 29: SI3 scalar-math source provenance, the stale round-51
live-trace suffix assertion, the round-129 stepping-gate stamp, three
unstamped legacy report emitters, and the missing `hires_lane_surface`
case-board row.  No round-30 test fails.

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: no model statement
lands and `git diff 325384a7dc -- packages` is empty at the final tip.

## Choices

ASKED: Decision 54 and round 29's OPEN item authorize the three-operand split.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
score, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  Split frozen `e3f_0vor` in compiled order:
   masked four-cell reference average, F-fold exchange, then zero substitution.
   Keep `r3f`, `fe3mask`, numerator, transports, forcing, and score fixed.
2. Preserve round 28's finite LDF-only artifact for the eventual whole-bundle
   retry; it is not a landing verdict.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The inherited duplicate citation-map literal keys remain open.
