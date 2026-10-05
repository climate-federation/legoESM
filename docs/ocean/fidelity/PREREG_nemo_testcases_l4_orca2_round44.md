# NEMO testcase Lane 4 — ORCA2 card round 44 preregistration

Date: 2026-09-27

Parent: `399b5be650`

Status: **PREREGISTERED BEFORE ROUND-44 SCIENTIFIC SCORING.**

All scientific numbers are **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.

Round 43 made the stage-1 centered-advection accumulator bit-exact by supplying
the admitted vertical transport, then exposed 2,514 temperature and 2,418
salinity cells in the following surface/runoff-source accumulator.  This round
walks that accumulator in compiled order.  It does not infer a source from the
residual: it replays the executing statements from the record's own operands.

## Executing compiled order

At stage 1, the executing nonlinear-free-surface arm first stores
`z1_rho0_e3t = r1_rho0 / e3t(Kmm)`, then subtracts
`emp * tracer(Kbb) * z1_rho0_e3t` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:278-288`.
QNS and SFX execute only at stage 3 (`trasbc.f90:290-310`), so they are
registered as structurally inactive here.  The runoff block then forms
`zdep = 1/h_rnf` and adds `rnf_tsc * zdep` at
`trasbc.f90:314-328`.  The stored density reciprocal is initialized by
`r1_rho0 = 1/rho0` at `eosbn2.f90:2505`.

legoESM already applies EMP and runoff in that order, but its EMP operand is
currently formed as `net_freshwater_flux(..., include_runoff=False) / rho_0`.
Round 13 registered that spelling against NEMO's stored-reciprocal spelling as
Decision 57; the user approved Decision 57 on 2026-09-25.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R44-P1 | The landed round-43 instrument reproduces before any arm. | With admitted `zFw`, after-advection T/S remain 0 unequal; after-SBC remains 2,514 T and 2,418 S unequal with maxima `4.235164736271502e-22` and `1.6940658945086007e-21`. | Any count or maximum moves; stop for instrument drift. |
| R44-P2 | The admitted record contains every operand needed for the two source statements, and the scorer binds. | Exact schema/digests, finite payloads, explicit CPU/fp64/libm, and a one-ULP active-cell plant fires. | A missing/non-finite operand, unresolved time level, schema drift, or inert plant; stop for record/instrument repair. |
| R44-P3 | NEMO's literal EMP-then-runoff replay closes the recorded after-SBC accumulator bit-for-bit. | Both tracer rows are 0 unequal when the replay uses the record's after-advection/Kbb/EMP/runoff operands and NEMO's stored `r1_rho0`, reciprocal-first depth, and source order. | Either row remains non-bit; name the first operand or association and hold. |
| R44-P4 | The density-reciprocal spelling is the first non-bit statement. | Replacing stored `r1_rho0` multiplication by division by `rho_0` reproduces the production residual cell-for-cell, while a production arm changing only this spelling makes after-SBC T/S bit-exact. | The division replay does not reproduce the residual, or the one-statement arm leaves any after-SBC residual; hold and continue the operand walk. |
| R44-P5 | The source spelling plus round 43's recorded-W association closes the complete stage-1 T/S row. | Stage-1 T/S are 0 unequal on the same record-backed support, and every ORCA2/GYRE gate passes. | Any stage-1 residual remains, any AT-BAR row leaves the bar, first-over-bar moves earlier, or GYRE changes; do not land. |

The score domain is the same record-backed, support-safe rank-0 wet interior as
round 43.  The gate must print the source-order intermediate rows and distinguish
structurally inactive QNS/SFX from numerical zero.  It must refuse an unexpected
card selector, operand shape, non-finite value, or source-order plant.

## Choices

ASKED: continue the compiled-order EMP/SFX/QNS and runoff association walk; use
the already-approved Decision 57 only if the one-statement discriminator and
the full landing gates confirm it.

UNASKED: none.  No configuration, carried state, selector, sea-ice field,
stabilizer, score domain, or scientific threshold changes.
