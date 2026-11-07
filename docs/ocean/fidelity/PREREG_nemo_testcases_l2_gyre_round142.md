# Preregistration — NEMO testcase L2 GYRE round 142

Date: 2026-09-21

Incoming lane tip: `a46cc1b14b7cebf18113bc54205900fc4a439267`

This document is frozen before reading any new scientific value from the
Round-140 developed three-dimensional momentum record and before running the
directed production discriminator. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round142/`.

Round 141 admitted the Round-140 record but rejected every attempted RHS
callback because even a one-face unordered callback moved 45,012 U-arm or
45,816 V-arm returned-state cells. No RHS scientific row was scored. This
round follows its OPEN item: no callback observes the three-dimensional RHS.
Instead, a private default-off test hook replaces only the owned native
`du_dt`/`dv_dt` faces after the ordinary production tendency has been computed
and before the existing stacked depth reduction. The unowned boundary face is
left exactly as production computed it.

## P0 — record and ordinary-arm controls

The compiled Round-140 program accumulates HPG, LDF, VOR, KEG, and ZAD into
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169`, writes the
completed U/V fields at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:180-193`, depth-averages
them at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, and applies
drag and wind at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260`.

Before the directed arm, the existing gate must re-admit the exact
1,486,548-byte record, its producer/stamp/manifest, literal depth and wind
replays, inherited byte identities, and every stored plant. The default-off
arm must reproduce Round 140's production-JIT incoming maxima exactly:
`4.2854247978022983e-13 m s-2` on U and
`4.433308633699682e-13 m s-2` on V. Its returned 23-leaf state and actual
external-call operands must be BIT against an independently compiled ordinary
step. Any different ordinary maximum, moved returned cell, moved external
operand, failed record control, or active public/configurable selector REFUTES
P0 and stops the scientific comparison.

## P1 — no-observer completed-RHS substitution

Extend the existing Round-83 slow-forcing walk and its Round-140 developed
bridge; do not create another stepper. Add one private test-only hook whose
default is `None` and which is absent from `LatLonCGridOceanConfig`. When
selected, it replaces only model native U faces `[:, 1:, :]` and native V
faces `[1:, :, :]` with the admitted NEMO completed `Krhs`; it does not expose,
return, callback, or materialize the model-computed RHS. The existing
passivity-proven eight-field slow-forcing callback may score the downstream
incoming/final two-dimensional boundaries, with the same override supplied to
its independently compiled plain arm.

Frozen prediction: substituting the admitted completed U/V `Krhs` makes both
incoming two-dimensional forcing rows BIT against the same-run NEMO record.
The final post-Coriolis rows may retain the already measured approximately
`1e-21` Coriolis residue, but their `1e-13` magnitude must disappear. A
non-BIT incoming row REFUTES the exact-closure prediction and remains in the
receipt.

For the conditional family verdict, the completed RHS family is CONFIRMED as
the magnitude carrier only if both incoming-face maxima decrease by at least
`1e3` and are at most `1e-18 m s-2`; otherwise it is REFUTED and no cumulative
record is acquired. The gate reports every row before/after, the directed
returned-state movement (expected because this is a substitution), and the
plain-versus-callback identity within each arm. It must not call a changed
state observer movement an instrument failure.

A default-path source-removal test and a one-ULP change to one consumed wet
NEMO RHS value must each fail closed with `STATUS PLANT-FIRED`. The ULP plant
must move a downstream incoming row, not merely the override array.

## P2 — conditional cumulative record and year sensitivity

Only if P1 confirms the completed RHS family by the registered magnitude
criterion, write and run one new passive NEMO target recording the cumulative
U/V `Krhs` after HPG, LDF, VOR, KEG, and ZAD at step 1081. The source card is
additive, syntax-proved, carries a new target name, prints a named `REFUSE`
before every nonzero exit, uses bash timing, preserves every inherited
Round-140 byte, and ships layout/stamp/truncation/ULP/terminal-closure/passive
plants. The terminal pair must be BIT against the admitted Round-140 completed
RHS before any cumulative row is used.

Each cumulative boundary is then substituted alone through the same
production boundary while later model increments remain live. No owner is
inferred by subtracting residuals. A term earns a day-240 magnitude only from
a 360-day production member using the immutable Decision-45 protocol and
registered days 30/60/90/120/180/240/300/360. Frozen ranking prediction:
HPG has the largest day-240 sensitivity. Any other ranking is a retained
REFUTATION. If the cumulative record or a source-exact per-term substitution
cannot be completed in-round, stop for that discriminator without assigning
the `1.644674193e-2 K` gap to a term.

## P3 — landing, review, citations, and scope

This directed discriminator is diagnostic and is not a landing candidate.
No production physics, configuration, default, carried state, stabilizer,
canonical NEMO source, or immutable before arm changes. The certified ladder,
month, year, DINO, LOCK_EXCHANGE, OVERFLOW, tanks, and ORCA2 are not rerun
unless P2 produces a source-exact candidate. ORCA2 remains
`UNMEASURED-WITH-SPEC` for this GYRE developed-state registry.

A separate read-only Codex pass must try to refute the no-observer claim,
native-face mapping, unowned-face preservation, default-path identity,
same-run ancestry, arm-local callback passivity, magnitude criterion, ULP
propagation, any conditional acquisition, and the distinction between a
one-step carrier and day-240 sensitivity. `DO NOT SHIP` blocks the diff.
Every compiled-source citation is mapped by the receipt citation gate, whose
shifted-citation plant must exit nonzero.

Expected status is `HELD`, or `STOPPED_FOR_RECORD` only after P1 confirms the
family and an in-round cumulative acquisition is blocked. `DECISION_NEEDED`
is `NONE`.
