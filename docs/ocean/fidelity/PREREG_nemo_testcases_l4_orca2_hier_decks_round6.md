# ORCA2 hierarchy decks round 6 preregistration — rung 6 admission repair and rung 5 runoff-off deck

Date: 2026-10-01

Base: `42466030b` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  The operator's rung-6 run reached step 240 and then
the round-5 admission refused because an inherited no-ice predicate required a
TKE initialization print that is absent when constant mixing replaces TKE.
This round preserves that failed prediction, repairs the predicate without
rerunning NEMO, admits the existing rung-6 record if every frozen record check
passes, and constructs rung 5 by turning runoff off as required by Decision 80.
It changes no legoESM package, card, recipe, physics, configuration, threshold,
or carried state.  Rungs 1 through 4 and main-lane rung 0 remain out of scope.

All record claims are labelled **independent**: NEMO starts from each rung's
own from-rest T/S initialization.

## Frozen oracle reading

The inherited rung-9 no-ice predicate expected the printed line
`nn_mxlice=0`.  That line and the no-ice reset both live inside
`zdf_tke_init` (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:785-798`).
The compiled vertical-physics dispatcher calls `zdf_tke_init` only when
`ln_zdftke` is true; rung 6 instead selects `ln_zdfcst=true` and
`ln_zdftke=false` (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:262-270`).
The source-exact rung-6 consequence is therefore absence of both the TKE
initializer and the `nn_mxlice` print.  The round-5 inherited-output prediction
is **REFUTED**; the admitted rung-9 through rung-7 no-ice result remains valid
on their active TKE path.

For rung 5, Decision 80 fixes the only deck delta as
`namsbc.ln_rnf=true->false`.  The selector is read and printed in the surface
manager (`sbcmod.f90:159-177`, `:182-207`).  The runoff initializer is called
unconditionally (`sbcmod.f90:359-374`), so `namsbc_rnf` remains read and must
not be replaced by an unread-group sentinel.  After reading it, the false arm
sets `ln_rnf_mouth`, `ln_rnf_tem`, `ln_rnf_sal`, and `ln_rnf_icb` false and
returns before runoff allocation or file input (`sbcrnf.f90:308-330`).  The
per-step freshwater call is guarded at `sbcmod.f90:517`; the RK3 tracer heat
and salt source at `trasbc.f90:314-328`; the continuity runoff divergence at
`divhor.f90:142` and `:199`; and the external-mode surface-height forcing at
`stp2d.f90:241-245`.

Every unnamed assignment remains byte-identical to rung 6.  In particular,
`ln_rnf_mouth=false`, its retained parameters, surface restoring, freshwater
budget control, bulk forcing, RGB shortwave, and `ln_spc_dyn=true` do not move.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD6-P1 rung-6 admission repair | The completed rung-6 record is valid; only the inherited active-TKE print predicate is wrong. | A rung-6-specific predicate requires `zdf_tke_init` and the `nn_mxlice` print both absent, admits the existing record, every prior plant still fires, and no NEMO product changes. | The print is present, another record predicate fails, a plant stays green, or a NEMO rerun is required. |
| HD6-P2 rung-6 record completeness | The existing two-rank run contains 480 finite self-describing frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Header-driven parsing reaches physical EOF for every frame and every terminal/month/inventory check passes. | Any missing or malformed frame, bad payload, non-finite value, wrong step/dtype, or incomplete ledger. |
| HD6-P3 one-module rung-5 delta | Rung 5 differs from rung 6 only on `namsbc.ln_rnf` true to false. | Complete physical-line and parsed-assignment gates report that single row; all other deck/build/input/protocol bytes are inherited. | Any additional assignment, line, CPP key, input, or protocol difference. |
| HD6-P4 runoff false branch | Rung 5 reads the retained runoff namelist group, then returns before runoff allocation/input and executes none of the runoff freshwater, tracer, continuity, or external-mode source arms. | Compiled-source census is pinned; resolved output reports runoff false and omits `sbc_rnf_init : runoff`; an execution sentinel placed in a retained numeric runoff value is read and printed, proving the group remains live. | The group is unread, the active-runoff print appears, a runoff file is opened, or any guarded source arm executes. |
| HD6-P5 retained modules | Rung 5 retains constant mixing, bulk forcing, restoring/freshwater budget, RGB shortwave, and every other hierarchy selector from rung 6. | Assignment-map equality and exact deck digest leave only `ln_rnf` changed. | Any other selector or retained parameter moves. |
| HD6-P6 rung-5 record completeness | The reused two-rank binary will produce the same record classes as upper rungs: 480 finite frames, finite month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | All header, payload, terminal, month, and inventory checks pass after the operator run. | Any product is missing, malformed, non-finite, or not pinned. |
| HD6-P7 acquisition disposition | No rung-5 record exists before this round's operator handoff. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with a committed launcher. | An already-existing admissible rung-5 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The repaired rung-6 control must fire if the inactive TKE print is injected or
if the active-TKE expectation is restored.  Every existing frame, terminal,
month, sentinel, resolved-consequence, and SHA-inventory plant must still fire.

The rung-5 gate must refuse an extra deck delta, a missing runoff selector
delta, a changed retained selector, a wrong build pin, an unread retained
runoff group, an active-runoff consequence, malformed or truncated
self-describing payloads, a missing/non-finite frame, a non-finite terminal
state, a wrong terminal step, and an incomplete SHA-256 inventory.  Every
plant must fire.

This round lands the admission repair, rung-6 evidence, and rung-5 deck,
manifest, gate, launcher, tests, and receipt only if rung 6 admits and rung-5
preflight and controls pass.  Rung 5 remains **UNMEASURED** until its operator
run is admitted.  No GYRE run is required because no model file changes.
