# NEMO testcase Lane 4 — ORCA2 card round 28 preregistration

Date: 2026-09-26

Parent: `8237f801c6c928109ca4db049c893f4fb4e28fae`

Status: **PREREGISTERED BEFORE ROUND-28 SCIENTIFIC SCORING.**

Round 28 executes round 27's first OPEN item.  It isolates the NEMO-carried
raw F-thickness operand at its two production consumers: first EEN vorticity
with lateral diffusion held at the parent, then lateral diffusion with EEN
held at the parent.  Each arm is an independent kt=1..10 ORCA2 trajectory
using Decision-52's recorded initial SSH; no given-NEMO-entry operator result
is mixed into its score.

The six sea-ice selectors and the ORCA2 card's `unmeasured_features` tuple
remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round28/`.

## Compiled statements and executed branches

The admitted build forms lateral diffusion's F curl from live
`e3f_3d*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`.

The same build selects EEN vorticity (`ln_dynvor_een = T`,
`ln_dynvor_msk = F`).  It constructs the reference F thickness in the live
`nn_e3f_typ` branch at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:907-937`,
forms its reciprocal at `:733-738`, and consumes that reciprocal in the EEN
flux and tendency at `:782-805`.

The production path calls `nemo_qco_live_vorticity_e3f_cgrid` once from the
EEN operator and at the Kbb and stage Kmm lateral-diffusion boundaries.  The
experimental arms add no numerical method: each routes the already-carried
NEMO raw F operand to only one named consumer while the other consumer retains
the parent reconstruction.  Every experimental `packages/` edit is reverted
before the receipt.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R28-P1 | The EEN-only raw-F arm owns the round-26/27 trajectory failure. | It reproduces the shared arm's first moved row at kt=1 stage-2 U and the exact non-positive raw-`e3w` refusal while entering kt=4. | It reaches kt=10, moves first somewhere else, or does not reproduce the exact refusal. |
| R28-P2 | The LDF-only raw-F arm is not sufficient for that failure. | It reaches kt=10 without the raw-`e3w` refusal; its kt=1 stage-2 U/V rows remain at the parent values because kt=1 LDF is exactly zero. | It refuses before completing kt=10 or changes either kt=1 stage-2 U/V row. |
| R28-P3 | Consumer isolation is mechanically effective. | EEN-only changes the exposed stage-2 vorticity digest while its kt=1 LDF digest equals the parent; LDF-only keeps the vorticity digest equal to the parent and changes a non-rest direct LDF result. | Either arm changes both consumers, changes neither consumer, or fails its passive-consumer equality. |
| R28-P4 | Neither arm moves a formerly bit-identical ladder row off the bar or changes the first NEMO mismatch. | Every observable exact row stays exact and the first non-bit statement remains kt=1 stage-1 temperature. | Any formerly exact row moves or the first mismatch moves earlier. |
| R28-P5 | The outcome gate binds. | A one-representable-value plant in an exact row is refused, and a consumer-label swap is refused. | Either planted violation passes. |

Failed predictions remain **REFUTED**.  If EEN alone owns the refusal, the
next round walks the EEN numerator/denominator in compiled order.  If LDF
alone owns it, the next round resumes the Decision-54 F-curl compensation
walk.  If neither arm reproduces the shared refusal, the interaction is the
finding and Decision 54 remains held.

## Landing and stop rules

- Commit this preregistration before any arm is run.
- Reuse the admitted ORCA2 record, fp64/libm precision policy, CPU backend,
  production JIT, surface forcings, scoring rules, and Decision-52 entry.
- Compare each arm to a fresh parent ladder from this round.
- Any arm that encounters the registered raw-`e3w` refusal stops there; no
  clipping, stabilizer, or refusal relaxation is allowed.
- No Decision-54 statement lands from this attribution round.
- No NEMO acquisition, NEMO-source edit, configuration change, carried-state
  change, sea-ice edit, or scoring-rule change is authorized.

## Choices

ASKED: Decision 54 and round 27's OPEN item authorize the two consumer-local
raw-F trajectory arms.

UNASKED: none.
