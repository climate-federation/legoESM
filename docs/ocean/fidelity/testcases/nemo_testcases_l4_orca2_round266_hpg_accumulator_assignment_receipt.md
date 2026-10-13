# ORCA2 round 266 — stage-2 HPG accumulator assignment

Date: 2026-10-13. Base: `21943d712`. Preregistration: `606bd4218`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried state,
stabiliser, sea-ice declaration, or `unmeasured_features` entry at its final
tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, using CPU fp64/libm production JIT and the already admitted round-222
record. No NEMO acquisition was needed.

## Compiled program and record

The resolved vector-form program writes the empty stage-2 accumulator, calls
EOS, then HPG first, and records the accumulator immediately after that call
at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:372-405`.
The same compiled branch then calls VOR and ADV and records their cumulative
boundaries at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:407-421`.

Inside the executed `hpg_sco`, NEMO writes the surface U accumulator as the
literal `zhpi + zuap` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:395-406` and repeats
that assignment at every interior level at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:411-432`. The admitted
literal stream records those same operands and sums. The gate parses its own
dimensions, lengths and finite payloads; the generic stage-term reader now
accepts explicit ORCA2 dimensions rather than assuming GYRE.

## Frozen predictions and result

R266-P1 is **CONFIRMED**. Under both labels, the matched stage-2 fold-band T
and S remain `2.220446049250313e-16 K` and
`7.105427357601002e-15 PSU`, respectively. The private trace is passive for
every carried field.

R266-P2 is **CONFIRMED**. The recorded entry accumulator is array-exact. HPG
is the first cumulative boundary above the `2e-10` floor: maximum
`4.6166034627296855e-7 m/s2`, RMS `3.5305428473726975e-9 m/s2`, and
`248,848/399,600` unequal slots against NEMO. VOR and ADV are downstream;
the final ADV accumulator retains round 265's maximum
`1.706908474056055e-6 m/s2`.

R266-P3's predicted **HPG operand** owner is **REFUTED**, but the leaf walk
names a more precise first statement. On admitted active support, the
candidate-versus-NEMO operand maxima are: `rhd` `2.220446049250313e-16`,
`e3w` `1.1368683772161603e-13 m`, live depth
`9.094947017729282e-13 m`, and `r1_e1u` exactly zero. The source-ordered HPG
literal rows are all below the campaign floor: `zhpi_u` and `sum_u`
`1.0524384369634682e-19 m/s2`, `zuap_u`
`9.341929683990129e-20 m/s2`. The record's own literal replay is AT-BAR.

The first non-bit executable statement is therefore the **candidate HPG
accumulator assignment/materialisation**, after the already-exact literal
sum and before VOR. Comparing that accumulator with the candidate's own live
HPG term gives `226,231/399,600` bit differences, maximum
`4.6166034627296855e-7 m/s2`, at `[j=12,i=88,k=2]`; both labels are
identical. The following VOR addition self-replays within
`3.3881317890172014e-21 m/s2`, proving the HPG displacement is carried
forward rather than created by VOR. This boundary corresponds to NEMO's
literal assignments at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:401-402` and
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:426-427`.

R266-P4 is **NOT REACHED**. No source-exact candidate arm was scored this
round, so the complete RHS, OMT-4 trajectory, and Decision-96 vote were not
claimed. R266-P5 is **CONFIRMED**: all nine planted violations refuse,
covering both-label coverage, record identity, passivity, source order,
round-265 reproduction, first-boundary selection, recorded self-replay, leaf
sensitivity, and a one-bit control.

## Instrument corrections retained

Three failed instrument assumptions remain in the evidence and are excluded
from claims. First, the round-265 private ratio arm had been retired from the
tip, so the initial clean run refused before measurement; the exact prior
committed diagnostic arm was restored for measurement and then retired again.
Second, the first HPG gate assumed its literal was already on a 180-column
global extent, but it was correctly on the 90-column owned slab; the clean
gate refused and the parser was corrected. Third, scoring canonical land/fill
slots made the offline literal appear different by `8.451070468299159e-4`;
the existing admitted HPG predicate scores the wet T/U support, on which the
literal closes. The new cumulative self-replay then separated that exact
literal from the non-bit accumulator materialisation. The final production
package tree is byte-identical to base.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round266/`.
Principal SHA-256 values are:

- classification: `d03951524396652cffd074d739ff4bc6d9c7ea7f4cdf175687c30882bab059dc`;
- independent measurement: `899122f6752885e9c60179e09b2a33a64555e5b2990dd14c67fcdc0d007c77e3`;
- given-entry measurement: `213e9258135556cb29fb9eda57b016082445b6cde2eccb1becba02fe1cafbf6d`.

The clean focused battery passed `41/41`: round 265 compatibility, all round
266 controls, and the citation-gate unit tests.  The cumulative default
citation audit and this receipt's six-citation audit both pass with no
unmapped citations or failed map audits; the planted two-line shift refuses
with exit 1.

The required separate review was attempted with `codex exec --sandbox
read-only`.  It could not initialize its in-process app-server client because
the read-only sandbox denied the required write, so **independent review
unavailable in-sandbox**; the full diagnostic is retained as
`codex_readonly_review.log`.

The once-only `tests/ocean/fidelity -n 12` battery reached 99% before two
workers stopped producing output.  It recorded 3,416 passed, seven skipped,
and four reds, all reproduced serially and all pre-existing on this tree:
the registered GYRE-year harness-hash ratchet, thirteen legacy unscoped
allow-dirty drivers, thirteen legacy unstamped report emitters, and the known
SI3 scalar-math `MY_SRC` provenance red.  Neither Round 266 file appears in
any red.  The two unfinished nodes were the known long-running planted stage
control and the VORTEX round-205 default-hook check; each also stalled in
isolation and was terminated rather than starting a second full-tree battery.
The parallel log, serial four-red disposition, and isolated-node logs are
retained beside the measurement evidence.

## OPEN

Build one private, source-exact arm for the HPG accumulator assignment: carry
the already-exact literal `zhpi + zuap` into the stage accumulator without the
candidate materialisation that moves it. First require the HPG cumulative row
and completed stage-2 RHS to close under both labels. Then re-score the
complete atomic unit through OMT-4 kt=1..10 and the full Decision-96 gates;
land only if every predicate passes. The downstream KEG accumulator
self-replay also moves by `4.616603462729692e-7 m/s2`, so after HPG closes it
is the next registered boundary, never part of this HPG claim.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
