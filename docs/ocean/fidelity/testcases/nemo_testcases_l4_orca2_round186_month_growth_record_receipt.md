# ORCA2 round 186: independent month boundary and growth-record receipt

Date: 2026-10-08  
Status: **STOPPED_FOR_RECORD**  
Claim labels: every rung-0 number in this receipt is **independent**.  No
given-entry rung-7 number is mixed into the tables below.

## Frozen question

Round 186 localised the production rung-0 month refusal left by round 185,
requested the missing NEMO checkpoints needed for an error-growth table, and
re-tested the already-defined private halo/V-transport cancelling unit.  The
committed preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round186.md`.

No model file, configuration value, forcing, carried state, stabiliser,
public selector, sea-ice selector, or `unmeasured_features` entry changed.
The diagnostic replay reads only an ordinary completed state; it does not add
an in-executable observer.

## Production month boundary

The independent initial T, S, u, v, and SSH arrays are each bit-identical to
NEMO.  The ordinary production trajectory completes 95 finite steps.  Its
step-95 entry live thickness is valid everywhere: minimum
`0.036497629967637335 m`, zero non-finite values, and zero non-positive finite
values.  Step 96 then reaches the registered `raw-mesh e3w_int must contain
only finite values > 0` refusal at `eos.py:747`.

An offline replay of the ordinary completed step-95 state localises the first
invalid boundary to stage 1's one-third geometry, after the split-explicit
barotropic result.  That result already contains non-finite SSH in all 26,640
horizontal columns.  Consequently all 476,557 interior W spacings are
non-finite.  The deterministic first index is `[j=1, i=49, k=0]`, where the
reference bathymetry is `729.0 m`, raw `e3w_0` is
`10.00035061113249 m`, and both SSH/stretch and the resulting spacing are
NaN.  This **refutes** the preregistered `[86,159]` prediction.

The executed NEMO external-mode program first extrapolates midstep SSH and
forms live T/U/V depths at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, then forms
the face transports at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:566-569`, and updates
the after-SSH field from their divergence at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:585-595`.  Its later
live U/V depths and reciprocals are formed at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:763-766` and exchanged
at `ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`.

This is a measured executable boundary, **not** a first non-bit statement:
the corresponding NEMO states at steps 10 through 95 are absent, so neither
the first departure nor its magnitude against NEMO can yet be named.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round186/production_month_growth_classified.json`.

## Fixed growth read-out

The existing admitted round-83 NEMO month directory has only its terminal
step-240 restart.  It has no rank-complete restarts at steps 10, 20, 30, 40,
50, 60, 70, 80, 90, or 95.  Therefore every NEMO-error cell in the fixed
table is **UNMEASURED-with-spec**.  The values below are candidate-state
absolute maxima only; they are not errors and do not establish direction
toward or away from NEMO.

| step | max abs T | max abs S | max abs u | max abs v | max abs SSH | NEMO error |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 30.08449146487878 | 40.57955251445472 | 1.1000915856523426 | 3.7303310133614933 | 4.189750296636377 | UNMEASURED — missing both-rank restart |
| 20 | 30.07830498424987 | 40.57954737899503 | 2.391305220514304 | 3.2226365022635752 | 2.630771854038017 | UNMEASURED — missing both-rank restart |
| 30 | 30.073971311837827 | 40.5795388992659 | 1.2730091158981138 | 1.9302806344283199 | 2.7412515362658936 | UNMEASURED — missing both-rank restart |
| 40 | 30.070322948863275 | 40.57953547642058 | 1.1953302033525337 | 2.029778372971029 | 3.9927937420820654 | UNMEASURED — missing both-rank restart |
| 50 | 30.066566389975943 | 40.57953211246701 | 1.105634699394733 | 2.264870986079683 | 2.8856653982185736 | UNMEASURED — missing both-rank restart |
| 60 | 30.063543199792573 | 40.57952881899018 | 2.662443518200139 | 2.2742464210169295 | 4.040050520719433 | UNMEASURED — missing both-rank restart |
| 70 | 30.060131229429384 | 40.57952558105666 | 3.722848359349922 | 2.945223657312299 | 6.412446098372007 | UNMEASURED — missing both-rank restart |
| 80 | 30.0579498098994 | 40.57952224673654 | 4.680588923005454 | 5.189274199264582 | 7.392720226005224 | UNMEASURED — missing both-rank restart |
| 90 | 30.05532747844456 | 40.57950401290294 | 7.617558296350088 | 7.2412937554821815 | 14.299835622554104 | UNMEASURED — missing both-rank restart |
| 95 | 1.3183941627776004e16 | 1.5214972903422472e16 | 6.8448214687716045e22 | 9.742145311480133e22 | 1209.361763687683 | UNMEASURED — missing both-rank restart |

The candidate remains finite through step 95 but grows explosively between
steps 90 and 95.  Without matching NEMO values, that observation cannot
locate the first erroneous step or statement.

## Growth-record acquisition

The committed launcher requests a new 96-step NEMO rung-0 run with
rank-complete restarts at exactly the ten fixed checkpoints.  Its preflight
passes and names new twin targets.  It changes only run protocol, reuses the
admitted rung-0 binary and physical deck, compares T/S/u/v/SSH for both ranks
between twins, calibrates step 10 against the admitted ten-step record, and
parses the NetCDF headers rather than predicting payload sizes.  Missing-rank,
twin-ULP, calibration, and hidden-deck plants are fail-closed.

Launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round186_growth_acquisition/run.sh`.

Preflight evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round186/acquisition_preflight.log`.

## Private halo/V-transport unit

The private arm was re-run on the corrected-entry, landed-HPG tree with all
four existing controls enabled: raw reference face depth, no extra compact V
mask, the seven-array boundary association, and materialised V transport.  It
completes kt=1 through kt=7, exposes kt=8 stages 1 and 2, and reaches the same
live-thickness refusal during kt=8 stage 3.  It therefore supplies no complete
200-row Decision-96 census and remains **HELD/private**.  It does not advance
the independent trajectory past production's step-96 boundary.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round186/rung0_halo_unit.log`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R186-P1 | **CONFIRMED** | Entry is bit-exact; 95 finite steps complete; step 96 reaches the registered guard. |
| R186-P2 | **REFUTED** | First deterministic invalid index is `[1,49,0]`, not the predicted `[86,159]`; after-SSH is already non-finite globally. |
| R186-P3 | **REFUTED** | All ten required NEMO checkpoints are missing; candidate-only maxima are retained but no error is claimed. |
| R186-P4 | **CONFIRMED** | The private unit refuses during kt=8 and cannot satisfy the complete-ladder predicate. |
| R186-P5 | **CONFIRMED** | The private unit refuses at kt=8, earlier than production's step 96. |
| R186-P6 | **CONFIRMED** | Entry, boundary-step, boundary-cell, and growth-provenance plants all fire; acquisition plants are committed for admission. |

## Validation and review

The month gate's four plants each fire with a distinct refusal.  The
growth-record launcher's preflight passes.  Focused validation passes 28/28
tests.  The one required `tests/ocean/fidelity -n 12` invocation collected
2,882 tests and reached 99% before the runner disappeared without a terminal
summary; no pytest process remained.  Its durable log contains 2,851 passes,
7 skips, and 5 failures, with 19 tests left without a result.  All five are
registered pre-existing reds: SI3 scalar-math `MY_SRC` provenance, two stale
GYRE certified-record checks (year source layout and spread floor), the
allow-dirty scope ratchet, and the worktree-stamp ratchet.  Every round-186
test ran and passed.

The default receipt citation gate passes with 274 citations and zero unmapped
or failing entries.  This receipt passes with all five compiled citations
mapped; shifting the `dynspg_ts` midstep-depth citation by two lines makes the
gate fail with exactly one bad citation.

The separate `codex exec --sandbox read-only` review was attempted and
returned `failed to initialize in-process app-server client: Read-only file
system`.  Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator must run the committed growth acquisition.  Admission
   requires two rank-complete, array-identical twins and step-10 calibration
   to the existing admitted record.
2. With those checkpoints, fill the fixed table with NEMO error maxima, name
   the first step whose independent state leaves the floor, and walk that
   step's external-mode program in compiled source order.  The current
   non-finite after-SSH boundary is not yet an owner statement.
3. The halo/V-transport unit remains private and HELD.  Re-score it only after
   the upstream month/external-mode owner is corrected; no partial operand may
   land.
4. Rung 7 and sea ice are untouched.  Sea ice remains exactly the ORCA2
   card's existing `unmeasured_features` declaration.
