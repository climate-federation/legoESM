# Preregistration: GYRE kt=2 momentum stage program, round 46

Date: 2026-09-11.  Measurement status: **UNMEASURED; acquisition not run**.
The producer is the clean committed worktree that the acquisition script stamps
and the gate must match by full SHA.  This round writes an instrument and stops;
it does not change a legoESM configuration or production operator.

## Compiled source contract

The executing GYRE build calls stage 1 as `(Kbb,Kmm,Krhs,Kaa) =
(Nbb,Nbb,Nrhs,Naa)`, then swaps `Nnn/Naa`, calls stage 2 with
`(Nbb,Nnn,Nrhs,Naa)`, swaps again, and calls stage 3 with the same symbolic
tuple (`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:201-215`).
Thus Kbb is the step-before level at all stages, Kmm is respectively before,
stage-1 after, and stage-2 after, and Kaa is the stage destination.

Stage 1's compiled operator order is HPG, LDF, VOR, WZV, KEG, ZAD
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90:141-166`).
Stages 2 and 3 form velocity-form `ww` from the passed Kbb/Kmm/Kaa tuple
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333`), then
execute HPG, VOR, and `dyn_adv`
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:447-469`).  Only
stage 3 then executes LDF and ZDF
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:704-717`) before
every stage's reference-depth barotropic replacement
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:724-743`).  The
active KEG and ZAD statements are
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90:117-130` and
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90:105-137`.  The
velocity-form WZV path calls `div_hor` and integrates bottom-up
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/sshwzv.f90:277-299`);
`div_hor` uses the live Kmm face thickness in its transports
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/divhor.f90:123-154`).

## Record and admission

For kt = `nit000` and `nit000+1`, each stage record carries its exact time-level
indices; Kbb/Kmm/Kaa velocities; `ww` and the resolved-zero `wsd`; live and
reference e3t/e3u/e3v/e3w plus r3t/r3u/r3v; barotropic before levels; masks and
horizontal metrics; and named RHS/state frames before and after every executed
HPG, VOR, KEG, ZAD, LDF, ZDF, stage update, and barotropic correction.  Presence
flags distinguish a non-executed operator from a zero tendency.  Every scratch
is zeroed before assignment, streams are WRITE-only, and every payload has a
name, rank, and extents.  The widened round-40 and round-41 kt=1 records must be
byte-identical to their frozen parents; a raw difference falls through to the
consumed-field admission, except those deterministic kt=1 records, which are
hard byte gates.

## Prediction and falsifier

With the explicit candidate ZAD in place, the predicted first kt=2 momentum
statement over the bit bar is **stage 1, after ZAD**: HPG, LDF, VOR, WZV replay,
and KEG remain bit-exact, while applying ZAD using a developed before level is
the first boundary to differ.  CONFIRM requires all preceding accumulators and
the source replay to be exactly equal and at least one wet U/V cell unequal
after ZAD.  REFUTE if any earlier stage-1 boundary differs, if the WZV replay
differs, or if after-ZAD remains exact; in that event the gate reports the first
measured stage/operator and makes no replacement claim.

## Gate arms and controls

The calibration arm replays NEMO WZV, KEG, and ZAD from the record operands and
requires zero unequal cells.  The given-input arm evaluates legoESM's own
operator routes at every kt=2 stage against NEMO frames.  The trajectory arm
bridges NEMO's complete kt=1 end state, executes legoESM kt=2, and scores every
recorded boundary.  Header, truncation, calibration, given-input, trajectory,
legacy-twin, and commit-stamp plants must each exit nonzero and prove movement
on their named row.

## ASKED / UNASKED

| status | choice | disposition |
|---|---|---|
| ASKED-and-answered, 2026-09-12 | scope of literal slope association | NEMO-identity cards only; implemented by `92c00497f48f`; no wider default |
| ASKED | build the kt=2 stage record and stop for operator acquisition | this round |
| UNASKED | configuration, physics, threshold, or public API change | none |
