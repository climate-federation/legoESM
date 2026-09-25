# NEMO testcase Lane 4 — ORCA2 card round 20 exchange receipt

Date: 2026-09-25

Parent: `475d87e8c6f5ec5ba8809a8c13c71e831224a067`

Measurement tip: `fbbc4ca91059704c855d44270676c0f28f3c9a82`

Status: **STOPPED_FOR_RECORD.** Given NEMO's entry, the round-19 MPI
exchange is bit-exact. The first live non-bit statement is instead rank 1's
substep-1 final slow forcing, `zu_frc`: 64 / 64 disputed east-source cells,
maximum `7.356481146903598e-18` and maximum 383,252 ULP. Substituting only
NEMO's recorded `zu_frc` into legoESM's otherwise exact vector expression
reproduces NEMO's exit U bit-for-bit on all 64 cells. The existing
slow-forcing stream is rank-0-only, so the source-order walk stops pending a
ranked slow-forcing record. No production statement lands.

No file under `packages/` changed. The six sea-ice selectors and the card's
`unmeasured_features` tuple remain unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every solver number below is labelled **given NEMO's entry**. The whole-card
ladder is separately labelled **independent with Decision-52 SSH**.

## 1. Ranked pre-exchange admission

The admitted operator-completed run is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round19/acquisition/orca1ice_u_preexchange_ranked_np2`.
Both rank streams are exactly 228,660 bytes and carry substeps 1 and 2. Rank
0 has SHA-256
`2143fb9c619f05487d8012db630dbe206b9b92388e78c1c4093ced26ee0b5f0f`;
rank 1 has SHA-256
`cfd50c25f6d4b2d2c847ebce9e26f2ab12c6a5d4f6779f3d3f5c51e13ed059a2`.
The distinct rank headers, distinct payload digests, completion stamps,
`STOP 0`, and split-rank markers all pass.

The WRITE-only stream records `ua_e` immediately before the active exchange,
and the active variable-volume branch calls the exchange at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:766-786`.
The record is therefore paired directly with round 19's post-exchange ranked
history; no periodicity assumption supplies an oracle value.

## 2. Compiled exchange walk

This build resolves `nn_comm = 1`, so the compiled dispatcher selects the
point-to-point branch at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:623-627`.
That branch defines its two-cell send and receive offsets at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1889-1909`, packs
the east/west sources at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1960-1979`, and
assigns the received values directly at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:2055-2073`.

All four compiled east/west mappings are bit-exact across both substeps, both
halo layers, and every non-fold row: 0 / 588 differing cells for each of rank
0 west from rank 1 east, rank 0 east from rank 1 west, rank 1 west from rank 0
east, and rank 1 east from rank 0 west. On the disputed 64-cell live face,
rank 0 post-exchange U is bit-identical to rank 1's pre-exchange east source;
rank 0's overwritten pre-exchange value differs on all 64 cells with maximum
`0.00041734608020692267`.

The preregistered seam hypothesis is **REFUTED**. Both legoESM duplicate seam
faces are already bit-identical to one another. Both differ from NEMO's rank-1
source on 64 / 64 cells, maximum `1.2223159767330016e-15` and maximum 497,491
ULP. The proposed east-to-west copy therefore changes nothing: the inherited
substep-1 exit mismatch remains, and the substep-2 continuity mismatch remains
64 cells with maximum `7.705384632572532e-07`.

## 3. First non-bit statement and ownership

NEMO first copies the slow barotropic momentum forcing into `zu_frc` at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:291-294`, then
removes its two-dimensional Coriolis trend at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:325-330`.
On the 64 disputed rank-1 source cells, the candidate and record are bit-exact
through all three midpoint coefficients, `un_e`, `ub_e`, `ubb_e`, `ua_mid`,
`rDt_e`, `zu_spg`, and `zu_trd`. The first non-bit live input in compiled
source order is substep-1 `zu_frc`: 64 / 64 cells, maximum
`7.356481146903598e-18`, maximum 383,252 ULP. The recorded and candidate
midpoint and vector expressions each replay their own targets bit-for-bit.

A causal substitution closes the boundary: candidate histories, timestep,
pressure-gradient forcing, Coriolis trend, and mask plus only recorded rank-1
`zu_frc` reproduce recorded rank-1 exit U on all 64 cells, bit-for-bit. This
names the final slow-forcing boundary as owner; it does not yet name a producer
statement.

The compiled producer vertically averages the three-dimensional momentum RHS
and records the resulting forcing and reference depths at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/stp2d.f90:189-210`. It then
applies baroclinic drag and wind and records final `Ue_rhs` at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/stp2d.f90:218-233`. The
existing file `oracle_slow_forcing_kt00000001.bin` was deliberately guarded by
`lwp` and contains rank 0 only. It cannot score the rank-1 source face without
inventing a cross-rank equivalence, so the walk stops here.

## 4. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R20-P1 | **CONFIRMED** | Both distinct 228,660-byte streams admit with exact rank headers, substeps, completion stamps, and markers. |
| R20-P2 | **CONFIRMED** | The inherited round-19 boundary reproduces: 64 substep-1 post-exchange west-halo cells, maximum `1.2223159767330016e-15`; all registered expression replays remain exact. |
| R20-P3 | **CONFIRMED** | Every compiled two-halo MPI mapping is bit-exact, including the disputed rank-1 source to rank-0 destination. |
| R20-P4 | **REFUTED** | Candidate east is not NEMO's source; candidate east and west are already identical and both differ from it on all 64 cells. |
| R20-P5 | **REFUTED** | Midpoint and vector replays are exact, but rank-1 `zu_frc` is the first earlier non-bit live operand. |
| R20-P6 | **REFUTED** | East-to-west copying is a no-op and closes neither the exit nor continuity mismatch. |
| R20-P7 | **NOT REACHED** | No statement is eligible to land. The independent ladder was still rerun and remained measured. |

No prediction was rewritten after measurement. The failed predictions remain
in the committed preregistration.

## 5. Acquisition needed

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round20_slow_forcing_acquisition/run.sh`.
It creates isolated target `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK` and new run
`round20/acquisition/orca1ice_slow_forcing_ranked_np2`. Its WRITE-only patch
removes only the rank-0 guard from the existing slow-forcing instrument, adds
the MPI rank to the filename and header, and records the existing full
vertical-average, post-drag, wind, and final-forcing payload on both ranks. It
does not assign or reorder a model operand.

Each expected stream is exactly 22,814,424 bytes. Preflight passes the pinned
source, CPP, baseline binary, deck manifest, input manifest, patch digest,
exact no-fuzz dry apply, and tool checks. Admission refuses a missing or wrong
byte count, wrong rank header, byte-identical rank streams, failed completion
stamp, absent `STOP 0`, or missing split-rank marker.

The acquisition was not launched in this sandbox. The binding operator note
says PMIx socket creation is refused here; no workaround or retry was made.

## 6. Ladder, review, citation gate, and tests

The clean measurement-tip ORCA2 kt=1..10 ladder exits 0 with
`LADDER_MEASURED`, labelled `MEASURED_INDEPENDENT_WITH_DECISION52_SSH`.
Its entry at kt=1 remains bit-identical after the Decision-52 bridge, and the
independent pre-bridge initial SSH remains owned by the out-of-scope SI3
category-load selector gap. The six unmeasured features are unchanged. An
initial invocation against `instrumented_reviewfix_10step_np2` refused at
kt=1 T; digest reconciliation identified the admitted V2 arm as
`variant_icebergs_off_phase2v_tke_a_10step_np2`, after which the same gate
passed. No GYRE rerun is required because the parent-to-tip `packages/` diff
is empty; the inherited day-30 digest remains `14a7e64b4512860e` and is not
claimed as a new run.

The separate `codex exec --sandbox read-only` review was attempted at the
committed measurement tip. It failed before reading the diff because its
in-process app-server client could not initialize on a read-only filesystem.
Verdict: **independent review unavailable in-sandbox**.

The clean exchange gate exits 3 only because it deliberately stops at the
missing ranked slow-forcing record. Its planted run fires the swapped-rank,
halo-payload, midpoint, exit, wrong-neighbor, and one-ULP source controls and
exits 1 as intended. The receipt citation gate passes, and rigid two-line
shifts of every stamped round-20 citation fail.

Focused round-17 through round-20 gate and citation tests pass **31 / 31**
before this receipt's own citation tests are added. The required
`tests/ocean/fidelity -n 12` battery was launched exactly once. It recorded
1,680 passes, seven skips, and the same three pre-existing failures documented
in round 19 before its final tail stopped producing output at 99%; after a
bounded wait it was interrupted and is not represented as green. The failures
are the SI3 scalar-math provenance check, the GYRE round-129 certified-record
stamp, and the GYRE round-51 stale final-trace assertion. No file in those
paths changed in round 20.

Gate JSON, ladder output, review output, preflight output, and test logs are
under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round20/codex`.

## Choices

ASKED: admit the completed ranked pre-exchange record, walk the compiled MPI
mapping to the first non-bit statement, and land only if a single statement
passes the full bar.

UNASKED: none. Decisions 54, 57, and 58 remain pending and untouched. No
configuration choice, carried-state change, stabilizer, sea-ice change, or
production statement was made.

## OPEN

1. Run and admit the ranked slow-forcing acquisition. Walk rank 1's compiled
   vertical average, drag, and wind statements to the first non-bit producer
   of final `Ue_rhs` / `zu_frc`.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part `dyn_ldf` scoping), 57 (`r1_rho0` spelling), and
   58 (second continuity solve) remain pending and untouched.
4. The independent ORCA2 year remains dependent on the scheduled independent
   initial-state completion; this solver result is only given NEMO's entry.
5. The wide ocean-fidelity battery's hung final tail remains an operator
   action; this interrupted run is not represented as green.
