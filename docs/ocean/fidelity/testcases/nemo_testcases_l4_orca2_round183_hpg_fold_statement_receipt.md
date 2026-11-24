# ORCA2 round 183 — independent HPG fold statement

Date: 2026-10-08. Base `0ac196427`; measurement commit `a7493ff7b`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round183/`.
Status: **HELD**.

Every scientific number is **independent hierarchy rung 0**. Both models start
from the corrected rung-0 climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry result is mixed into this receipt. No production
model, configuration, carried-state rule, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The first non-bit primitive HPG input is the north-fold density anomaly, not
the north-fold thickness predicted in the preregistration. The evaluated
north thickness is bit-exact on all 1,319 wet contributing levels. The
north density differs on 1,319/1,319 levels, with RMS
`1.4829794933663243e-03` and maximum `4.569568436628368e-03` at
`[j=147,i=29,k=0]`: legoESM reads `0.0`, while NEMO reads
`-0.004569568436628368`.

The first non-bit executable statement is therefore NEMO's surface
north-product contribution to `zhpj`. All 68 registered northern-fold faces
differ in that product, with RMS `1.2285268186392962e-02` and maximum
`4.569511590616214e-02`; after gravity/metric scaling the maximum is
`2.6017321153741642e-06 m s-2`. The executing assignment and its operand
order are in
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416`.

This attribution is exact at the component boundary. Replaying the five
recorded fields reproduces NEMO's `zhpj` bit-for-bit on all 1,319 levels.
Replaying legoESM's operands reproduces legoESM's literal `zhpj` bit-for-bit.
Replacing only the evaluated north thickness and density reproduces NEMO's
complete top-down `zhpj` bit-for-bit on all 1,319 levels. The fold record's
`zhpj` also equals the independently admitted round-180 component record
bit-for-bit. Thus the first statement is named without an executable observer.

## Four-operand cancelling unit

The upstream statement cannot land alone. NEMO next depth-averages the raw HPG
with `e3v`, `vmask`, and `r1_hv0`
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`). All four operands
differ on the same 68 faces: raw HPG on 1,319 wet levels, `e3v` on 27,
`vmask` on 1,319, and `r1_hv0` on 68/68. The source-ordered cumulative
substitution is:

| cumulative operands from NEMO | differing faces | RMS (m s-2) | maximum (m s-2) |
|---|---:|---:|---:|
| none (candidate all) | 68/68 | 1.0748834693208086e-06 | 2.4423127429248864e-06 |
| raw HPG | 68/68 | 1.0748834693208086e-06 | 2.4423127429248864e-06 |
| raw HPG + e3v | 68/68 | 1.0748834693208086e-06 | 2.4423127429248864e-06 |
| raw HPG + e3v + vmask | 68/68 | 1.0748834693208086e-06 | 2.4423127429248864e-06 |
| raw HPG + e3v + vmask + r1_hv0 | 0/68 | 0.0 | 0.0 |

The first three substitutions are hidden because the candidate reciprocal is
zero at every registered face; the last substitution closes the complete
unit bit-for-bit. The north-fold HPG correction alone is therefore
endpoint-inert under the current compensating depth-average operands, not an
eligible Decision-96 landing. This is a measured cancelling unit, not a claim
that any operand is dispensable.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R183-P1 record passive and eligible | **CONFIRMED**: exactly-once rank coverage and all 20 terminal restart identities pass. |
| R183-P2 exact-operand replay calibrates | **CONFIRMED**: oracle and candidate self-replays are bit-exact at their own completed boundaries. |
| R183-P3 north-fold thickness is first | **REFUTED**: thickness is bit-exact; north-fold density is first. |
| R183-P4 surface north product is first and north operands close `zhpj` | **CONFIRMED**: 68/68 surface products differ and the north-only replay closes 1,319/1,319 levels bit-for-bit. |
| R183-P5 four-operand unit remains cancelling | **CONFIRMED** by the cumulative table above; no partial unit landed. |
| R183-P6 measurement only unless the full unit passes Decision 96 | **CONFIRMED**: no production or configuration change landed. |

## Instrument correction, controls, and review

The first measurement attempt refused before producing a score because the
record includes NEMO's structural 31st level while the candidate operator has
30 physical levels. The gate was corrected to compare only physical levels,
while retaining the separate 31-level record-identity check; a regression
test now plants that mismatch. The successful run is pinned to clean commit
`a7493ff7b`.

All eleven classifier plants fire: admission, rank placement, record bit,
source order, target mask, self-replay, first input, first statement,
north-arm non-vacuity, atomic endpoint, and endpoint ULP. The focused
round-181/183 record/replay battery passes 49/49. The default citation gate
and this receipt's gate pass with no unmapped citation or audit failure;
shifting the new compiled citation makes the plant fail.

The one required `tests/ocean/fidelity -n 12` battery reported 2,851/2,869
selected cases before its xdist controller stopped emitting at 99% without a
terminal summary: 2,840 passed, seven skipped, and four failed. All four are
the registered pre-existing reds (SI3 scalar-math provenance, round-35 stamp
scope, worktree-stamp ratchet, and GYRE round-129 spread-floor record). Of the
18 unreported cases, 14 pass in isolated file groups. The remaining four are
the phase-3 stage-sweep end-to-end controls; the first remained silent for
eight minutes and was interrupted rather than launching another wide
battery. The incomplete wide log, selected-node census, and isolated-tail
evidence are retained under the evidence root. No round-183 test failed.

The separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed, so ORCA2, GYRE, DINO, tank, trajectory and
month landing gates are ineligible in this held measurement round.

## OPEN

1. Build one private atomic arm containing the exact HPG north-fold density
   association together with NEMO's `e3v`, `vmask`, and `r1_hv0`
   depth-average operands. Score both ORCA2 ladders before any landing.
2. If the complete unit is a Decision-96 net improvement, run the full GYRE,
   DINO, tank, citation and push gates and land it atomically. Otherwise keep
   it private and walk the first worsened row.
3. Only after this upstream unit closes, re-test the held
   V-transport/halo unit and the independent month's step-96 live-thickness
   refusal.

No acquisition is needed for the next atomic arm. The independent 240-step
month score remains **UNMEASURED-with-spec**.

ASKED choices: source-ordered offline split of the admitted independent HPG
fold record and atomic analysis of its four depth-average operands.  
UNASKED choices: empty.
