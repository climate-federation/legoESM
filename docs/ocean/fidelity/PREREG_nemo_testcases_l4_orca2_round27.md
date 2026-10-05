# NEMO testcase Lane 4 — ORCA2 card round 27 preregistration

Date: 2026-09-26

Parent: `aaca19714d`

Status: **PREREGISTERED BEFORE ROUND-27 SCIENTIFIC SCORING.**

Round 27 executes round 26's first two OPEN items.  It first compares the
Kbb-divergence and Kmm-divisor changes as full tendency arrays on the earliest
recorded non-rest entry.  It then separates the two production consumers of
the helper changed by round 26's nominal F-curl arm.  This separation is
required because source inspection shows that the same helper supplies both
the lateral-diffusion F thickness and the EEN potential-vorticity thickness.

Every trajectory or stage result is **independent with Decision-52 SSH**.
Every direct operator result is labelled **given NEMO's entry**.  These two
claim classes are not mixed in one score.  The six sea-ice selectors and the
card's `unmeasured_features` tuple remain frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round27/`.

## Compiled statements and executed branches

The admitted build forms lateral diffusion's F curl at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`,
its Kbb divergence at `:127-129`, and its Kmm curl divisors at `:132-140`.

The same compiled configuration selects EEN vorticity.  It constructs
`e3f_0vor` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:907-937`,
uses its live reciprocal in `vor_een` at `:733-738`, and applies the resulting
potential vorticity to the Kmm transports at `:782-805`.  The admitted
`ocean.output` resolves `ln_dynvor_een = T` and `ln_dynvor_msk = F`.

The current production path calls `nemo_qco_live_vorticity_e3f_cgrid` once for
the EEN vorticity operand and once for the lateral-diffusion operand.  Round
26's experimental F arm changed that shared helper, so the phrase “F-curl
thickness alone” is a hypothesis to be tested, not an accepted attribution.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R27-P1 | The Kbb and Kmm trajectory-score collision is not tendency equality. | On the recorded kt=2 non-rest entry, the full Kbb-only and Kmm-only tendency arrays differ bitwise; their changed intermediates are respectively grad-div and curl. | Both U and V tendency arrays are bit-identical, or both arms change the same intermediate. |
| R27-P2 | The round-26 F arm cannot move kt=1 through lateral diffusion. | The recorded kt=1 Kbb velocity is exactly zero and both the parent and raw-F lateral-diffusion arrays are exactly zero. | Either recorded face velocity or either lateral-diffusion array is non-zero. |
| R27-P3 | The first round-26 F-arm movement belongs to the shared EEN vorticity consumer, not to lateral diffusion. | With only the raw-F helper arm applied, the production stage-2 vorticity component differs bitwise from the clean parent while R27-P2 stays zero. | The stage-2 vorticity arrays remain bit-identical, or R27-P2 is non-zero. |
| R27-P4 | The direct instrument binds. | A one-representable-value plant in an otherwise exact comparison is refused, and deleting either consumer from its source-coverage inventory is refused. | Either planted violation passes. |

Failed predictions remain **REFUTED**.  If R27-P3 confirms, round 26's
“F-curl thickness alone owns the kt=4 refusal” statement is retracted: the arm
was a two-consumer experiment.  The next walk starts at EEN vorticity before
any retry of Decision 54.  If R27-P3 refutes, the F-curl attribution remains
open and the next discriminator must isolate the helper inside lateral
diffusion without changing EEN.

## Landing and stop rules

- Commit this preregistration before running any scientific score.
- Reuse the admitted kt=1/kt=2 records, fp64 CPU policy, production JIT, and
  the existing WRITE-only stage-operator exposure seam.
- Any experimental `packages/` change is one shared-helper input arm, starts
  from a clean commit, and is reverted before the receipt.
- No Decision-54 physics lands from this attribution round.
- Do not relax the raw-`e3w` refusal or add a stabilizer.
- No NEMO acquisition, configuration choice, carried-state change, sea-ice
  edit, or NEMO source edit is authorized.

## Choices

ASKED: Decision 54 and round 26's OPEN items authorize direct discrimination
of the thickness statements and the shared F-thickness consumer walk.

UNASKED: none.
