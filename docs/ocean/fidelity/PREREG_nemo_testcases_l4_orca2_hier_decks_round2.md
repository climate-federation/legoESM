# ORCA2 hierarchy decks round 2 preregistration — rung 10 admission and rung 9 ice-off deck

Date: 2026-10-01

Base: `432d704a7` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  Round 1's already-committed HD1-P3 through HD1-P6
predictions govern admission of the operator-produced rung-10 record.  This
round then constructs hierarchy rung 9 by removing only the one-category SI3
module.  It changes no legoESM package, card, recipe, physics, configuration,
threshold, or carried state.  Rungs 1 through 8 and main-lane rung 0 remain out
of scope.

## Frozen oracle reading

The compiled rung-10 source reads `nn_ice` in `namsbc` and selects zero as "no
ice" at `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:159-165,247-259`.
At zero it fixes ice fraction to zero and initializes only placeholder arrays
at `sbcmod.f90:274-280,376-383`; SI3's `ice_init` and every `ice_stp` are then
skipped (`sbcmod.f90:381-383,499-503`).  The only loader for
`namelist_ice_cfg` is inside the skipped `ice_init` branch
(`icestp.f90:285-305`).

Three retained ocean namelist values are ice-coupled but resolved inert at
this rung: the explicit `nn_mxlice=2` is forced to zero
(`zdftke.f90:785-787`), reference `ln_drgice_imp=.true.` is forced false
(`zdfdrg.f90:371-380`), and reference `nn_fwb_voltype=1` is forced to the
ocean-only value 2 (`sbcfwb.f90:149-156`).  They are not independently edited;
those compiled consequences belong to the one `nn_ice` module switch.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD2-P1 rung-10 admission | Round 1's operator record satisfies every previously frozen identity and inventory predicate. | The committed round-1 gate passes normally and all eight real-record plants fire against the existing record. | Any plant stays green or any record predicate fails. |
| HD2-P2 one-module deck delta | Rung 9 differs from rung 10 in exactly one parsed assignment and one non-comment line: `namsbc.nn_ice`, `2` to `0`; every other byte is inherited, including `ln_spc_dyn=.true.` and the unchanged ice namelist artifact. | Exact diff and assignment-map gates report that sole delta. | Any other assignment, non-comment line, CPP key, input, or run-protocol difference. |
| HD2-P3 ice namelist unopened | With `nn_ice=0`, NEMO never calls `ice_init`, never loads `namelist_ice_cfg`, and never emits SI3 initialization or an ice restart. | Resolved output reports `nn_ice=0`; contains no SI3/ice-namelist initialization; the run succeeds even when the staged ice namelist is replaced immediately before execution by a committed sentinel that would fail if read; no ice restart or ice month product exists. | SI3 initializes, the sentinel is parsed, an ice restart/product appears, or the run fails because of the sentinel. |
| HD2-P4 compiled consequences | The unchanged ice-coupled ocean lines resolve to no-ice values: `nn_mxlice=0`, `ln_drgice_imp=F`, `nn_fwb_voltype=2`, ice fraction zero, and no `ice_stp`. | Resolved output and compiled-path checks establish all five. | Any retained value remains ice-active or `ice_stp` executes. |
| HD2-P5 record completeness | The reused two-rank instrumented binary produces 480 finite self-describing operand frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Header-driven parser reaches physical EOF for every rank-step frame and all record/product/restart checks pass. | Missing/malformed frame, bad name/shape/payload, non-finite value, wrong step/dtype, or incomplete ledger. |
| HD2-P6 acquisition disposition | No rung-9 record exists before the operator run. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with a committed launcher. | An already-existing admissible rung-9 record is found. |

The rung-9 record is labelled **independent**: it starts NEMO from that rung's
own from-rest T/S initialization.  Failed predictions remain in the receipt as
REFUTED.

## Controls and landing predicate

The rung-9 gate must refuse at least a second deck delta, wrong CPP/build pin,
a read ice-namelist sentinel, a wrong resolved no-ice consequence, malformed or
truncated frame headers/payloads, a missing frame, non-finite terminal state,
wrong terminal step, and incomplete SHA-256 inventory.  Every plant must fire.

This round lands rung-10 admission evidence plus the rung-9 deck, manifest,
gate, launcher, tests, and receipt only if rung 10 admits and rung-9 preflight
and plants pass.  Rung 9 remains UNMEASURED until its operator-run acquisition
is admitted.
