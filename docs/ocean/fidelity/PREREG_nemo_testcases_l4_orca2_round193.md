# ORCA2 round 193 preregistration — corrected-entry atomic-unit growth boundary

Date: 2026-10-09. Frozen base:
`f79b81c0417779307f18ae920ea5c308cfabc787`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round193/`.

Every trajectory number is **independent hierarchy rung 0**: the private card
starts from its own corrected climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry rung-10 number is mixed into the table. The
shipped ORCA2 card, sea ice, all six ice selectors and its
`unmeasured_features` tuple remain unchanged.

## Frozen program and statistic

Reapply exactly the complete private unit refused in round 192: NEMO's
source-associated completed-RHS depth average, raw reference face depth, no
extra compact V-transport mask, the seven-array external-mode association and
separately materialised completed V transport. No partial member is scored as
a landing candidate. The unit remains private unless a later complete ladder
passes Decision 96.

The oracle is the admitted round-90 record: 80 self-describing rank shards,
two ranks, kt=1..10, entry plus stages 1--3. NEMO computes the external mode,
then stages 1, 2 and 3 in that order
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:200-233`). The gate starts
from the private card's own corrected entry, proves its five active fields
array-identical to the recorded kt=1 entry, and reports candidate-minus-NEMO
maximum and RMS for T/S/u/v/ssh after every completed stage until the candidate
refuses. It never installs NEMO's entry into the candidate.

For each boundary, the growth statistic is the largest T/S/u/v maximum divided
by the preceding boundary's largest maximum, with only the campaign's fixed
`2e-10` floor. Boundaries are ordered `(kt, stage)`. The first ratio strictly
greater than ten owns the offline replay; a later explosion cannot displace it.
Candidate non-finites are reported, never hidden by `nanmax`.

The stage exposures reuse the already validated detached round-174 path and
never feed the ordinary carried state. A same-tree stage-3 exposure must be
array-identical to the ordinary complete-arm result at every completed step.
The gate has plants for entry identity, record admission, stage passivity,
boundary order, first-growth selection and a one-ULP field change.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R193-P1 | The corrected independent entry remains exact. | Active T/S/u/v/ssh have zero unequal cells against the kt=1 entry frame. | Any active difference: invalid claim label; stop before a stage number. |
| R193-P2 | The complete round-192 unit retains its local exact prerequisites. | Its exact slow-depth arithmetic and existing raw-depth/unmasked/materialised/association controls pass before trajectory scoring. | Any local witness or dependency check fails: invalid unit; stop. |
| R193-P3 | The first greater-than-ten growth boundary is kt=1 stage 1. | The kt=1 stage-1 ratio exceeds ten and no earlier completed-stage row exists. | Ratio at or below ten, an earlier boundary, or any admission/passivity failure: REFUTED. |
| R193-P4 | The first source-ordered replay debt is downstream of the complete external-mode unit. | External endpoint/transport rows are at the floor and a stage-1 operator is the first over-floor row. | Any external row is over floor: REFUTED; name that earlier compiled statement and do not attribute a later operator. |
| R193-P5 | The admitted round-175 stage-1 record is sufficient for offline replay. | Its two ranked self-describing shards cover every required operand at the selected boundary. | First required operand absent or not rank-complete: `ACQUISITION_NEEDED`; name the missing stream and write a fail-closed launcher. |
| R193-P6 | This round is measurement-only unless a complete cited unit passes all standing gates. | Final package tree equals the frozen base, or a complete eligible unit passes both ORCA2 ladders, month boundary, GYRE, DINO and tanks. | Any partial operand lands, or any shared/card gate is skipped for a retained package change: refuse. |

## Offline replay rule

After the growth table selects its first boundary, replay the compiled stage
program offline from the passive completed input using the admitted NEMO
operands. Walk external handoff, momentum update, barotropic correction,
metric transports, tracer advection, surface source and QCO update in compiled
order (`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:137-181,
469-479,513-680`). Stop at the first over-floor row. No in-executable observer
is permitted, and no statement below a non-bit input is attributed.

ASKED choices: the complete private unit and the frozen growth/replay order
from round 192's OPEN.  
UNASKED choices: empty.

## Instrument correction after the first offline replay

The first offline replay reached substep-1 exit inverse V depth, then reported
68 northern-fold cells unequal. That row is rejected as an instrument defect:
the candidate trace reconstructed the inverse from the pre-association compact
mask, while NEMO executes the seven-array `lbc_lnk` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779` and only then
records `hvr_e` at `:791-796`. The ordinary candidate already consumed the
post-association fields, so no trajectory number moves. The corrected trace
selects the same post-association depth/inverse tuple whenever the complete
association arm is active; a synthetic raw/post pair proves the selector fires.
All frozen predictions, thresholds, field order and the growth report remain
unchanged. The refused first replay is retained in the evidence directory and
is not a statement attribution.

The corrected replay freezes one additional falsifier before rerun. If its
first debt is the substep-2 V transport accumulator, the preceding accumulator,
completed `zhV` and `wgtbtp2` must each be bit-exact. Replaying only NEMO's
unmasked `r1_e1v` factor in the compiled statement
`dynspg_ts.f90:606-608` must then close the completed accumulator bit-exact.
Any non-exact prerequisite or replay residual refutes reciprocal-metric
ownership; the receipt names no statement below it.
