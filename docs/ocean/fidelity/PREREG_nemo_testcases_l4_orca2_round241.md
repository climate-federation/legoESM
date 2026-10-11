# Preregistration — ORCA2 round 241 compact-mask boundary arm

Date: 2026-10-11. Frozen base: `cb476aa34`. Scope: measurement first;
production landing only if the frozen atomic predicates pass. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round241/`.

Every ORCA2 number is labelled either **independent** or **given NEMO's
entry**. Sea ice, its selectors, and the ORCA2 card's `unmeasured_features`
tuple are immutable in this round.

## Frozen source statement

The executing NEMO vector branch writes literal HPG into `pvv` without a
compact face mask at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`, accumulates the
remaining operators at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`, and applies its
stored `vmask` only in the later depth integral at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

legoESM instead multiplies the completed three-dimensional V tendency by its
generic compact mask at `ocean_pe_latlon_cgrid.py:5982-5983`. Round 240 proved
that the production Decision-114 geometry route makes raw HPG bit-exact on
all 1,319 wet fold-row levels, while this extra multiplication creates the
68-face completed-RHS debt.

The private arm preserves only the already-computed literal HPG contribution
through that generic boundary where NEMO's stored raw V mask is wet and the
compact mask is dry. It does not change another HPG operand, another operator,
the public card, or any configuration default. The complete round-217
fold/association/materialised-transport unit is scored atomically with it.

## Frozen predictions and falsifiers

1. **R241-P1 — boundary closure.** Under the production
   `nemo_qco_resolved_mesh_operands` route, the arm changes exactly the compact-
   masked HPG support and reduces the completed V-RHS comparison from
   `668/1,050` unequal slots to `0/1,050` under both claim labels. Any residual,
   any off-support change, or arm/passive-state disagreement **REFUTES** P1.
2. **R241-P2 — atomic-unit consequence.** With the arm plus the complete
   round-217 unit, OMT-4 kt=1 stage-1 fold-band salinity falls from about
   `3.283 PSU` to at most `0.0012 PSU`, temperature falls from about `0.177 K`
   to at most `0.0014 K`, and the run completes kt=8. Salinity or temperature
   above those limits, or the same kt=8 live-thickness refusal, **REFUTES** P2.
3. **R241-P3 — landing split.** If P1 confirms and P2 confirms, the compact-
   mask boundary statement and complete atomic unit land together only if
   Decision 96 passes on OMT-1..4, independent rung 0, rung 10, GYRE, DINO and
   tanks: majority of RMS-moved rows toward NEMO, first debt toward/unchanged,
   no exact-row loss, registered two-ULP counts, and no disallowed shared-card
   regression. If P1 confirms but P2 refutes, only the source-exact boundary
   statement is eligible to land under the same gates, and the next atomic-
   unit owner is named. Any failed predicate **HOLDS** the corresponding
   candidate.
4. **R241-P4 — controls.** A no-effect arm, altered support, incomplete claim-
   label coverage, false RHS closure, or relaxed trajectory predicate must make
   the round gate refuse. A control that stays green **INVALIDATES** the gate.
5. **R241-P5 — documentary gates.** The focused tests, independent read-only
   review, citation gate plus firing rigid-shift plant, one push-gate battery,
   and one `tests/ocean/fidelity -n 12` battery are recorded. Registered
   pre-existing reds remain labelled; any new unexplained red **HOLDS** a
   production landing.

## Stop conditions

No configuration choice, carried-state change, stabiliser, NEMO source edit,
or NEMO acquisition is authorized. A missing passive operand record yields
`ACQUISITION_NEEDED`; a deck ambiguity yields `DECISION_NEEDED`. Failed
predictions remain in the receipt as `REFUTED`.
