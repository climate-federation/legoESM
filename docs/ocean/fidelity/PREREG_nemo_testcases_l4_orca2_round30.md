# NEMO testcase Lane 4 — ORCA2 card round 30 preregistration

Date: 2026-09-26

Parent: `325384a7dc`

Status: **PREREGISTERED BEFORE ROUND-30 SCIENTIFIC SCORING.**

Round 30 executes round 29's first OPEN item.  It splits the production EEN
live-thickness denominator into its three compiled operands: frozen
`e3f_0vor`, live `r3f`, and frozen `fe3mask`.  Each arm changes one operand
source from legoESM's card-state reconstruction to the admitted NEMO-carried
record while the other two and every EEN numerator/transport input stay at the
parent.  Results are **independent with Decision-52 SSH**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.  Evidence
root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round30/`.

## Compiled statements and controlled boundary

The admitted compiled build constructs and exchanges frozen `e3f_0vor` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`,
constructs live `r3f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:233-246`, and
freezes `fe3mask` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90:258`.
The executed EEN statement consumes exactly
`e3f_0vor*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:733-738`.
The admitted `ocean.output` resolves `ln_dynvor_een = T`,
`nn_e3f_typ = 0`, and `ln_dynvor_msk = F`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R30-P1 | Frozen `e3f_0vor` is the dominant owner because the carried and reconstructed reference thicknesses differ on the full native field. | Its one-operand arm changes the first production EEN output and accounts for the largest unequal-cell count and maximum output change of the three arms. | Its EEN output is bit-identical, or another single operand has a larger unequal-cell count or maximum change. |
| R30-P2 | Live `r3f` is a real secondary owner because its reciprocal depth uses the differing frozen F-depth. | Its one-operand arm changes the live denominator and first EEN output. | Both remain bit-identical. |
| R30-P3 | `fe3mask` is already bit-identical and its arm is inert. | Carried versus reconstructed `fe3mask`, denominator, and EEN output are each bit-identical. | Any is non-bit. |
| R30-P4 | The three-operand arm reproduces round 29's carried-denominator output. | Its exposed stage-2 U/V arrays equal round 29's raw-F arrays bit-for-bit and move 413,554 U / 412,558 V cells from parent. | Either prior array or unequal-cell count differs. |
| R30-P5 | The outcome gate binds. | A one-ULP plant in an otherwise inert operand is refused. | The plant passes. |

Failed predictions remain **REFUTED**.  The first one-operand arm whose EEN
output is non-bit owns the next compiled-order walk.  If a single arm exactly
reproduces the whole carried-denominator result, it is eligible for a ten-step
consumer-local trajectory retry; otherwise Decision 54 remains held on the
measured cancelling bundle.

## Landing and stop rules

- Commit this preregistration before implementing or running the split.
- Reuse the admitted ORCA2 record, fp64/libm policy, CPU backend, production
  JIT, forcing, Decision-52 entry, and stage-2 exposure.
- Extend the existing round-29 whole-denominator observer; do not create a
  second EEN operator or change configuration/state fields.
- Revert every experimental `packages/` edit before the receipt.
- No Decision-54 statement lands unless one operand alone reproduces the full
  arm and then passes the ORCA2 ladder plus all required shared-card gates.
- No stabilizer, clipping, NEMO-source edit, sea-ice edit, score change, or
  acquisition is authorized.

## Choices

ASKED: Decision 54 and round 29's OPEN item authorize this exact operand split.

UNASKED: none.
