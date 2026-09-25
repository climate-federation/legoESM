# NEMO testcase Lane 4 — ORCA2 card round 19 U-history receipt

Date: 2026-09-25

Parent: `16608cd9a31edaec93a0227944d411e4d0b1b8f2`

Measurement tip: `04c5bb6c8c96b44d07f8847870a252212b2d419a`

Status: **STOPPED_FOR_RECORD.** Given NEMO's entry, the first live non-bit
statement is the substep-1 post-exchange U velocity on 64 rank-0 west-halo
faces, maximum `1.2223159767330016e-15`. Both the recorded and candidate
interior vector updates replay bit-exactly; the existing record starts again
after `lbc_lnk`, so it cannot identify the neighbor value that the exchange
supplied. No production statement lands.

No file under `packages/` changed. The six sea-ice selectors and the card's
`unmeasured_features` tuple remain unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every solver number below is labelled **given NEMO's entry**. The whole-card
ladder is separately labelled **independent with Decision-52 SSH**.

## 1. Record reconciliation and admission

The admitted existing run is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round18/acquisition/orca1ice_u_history_ranked_np2`.
Both rank streams are 2,042,100 bytes. Rank 0 has SHA-256
`3052f6b952f596127e37ca6e7ffae976fa7f3b1bde2f81b53f09db6ff4e660b8`;
rank 1 has SHA-256
`f5e0cc0092843cf85a891e67eeaede4015bece9bf08ce7acd5b84a53dd8837e6`.
Their headers name distinct ranks and substeps 1 and 2, and all completion and
split-rank marker checks pass.

The operator's prior 2,057,588-byte expectation was an instrument defect.
The record's compiled declarations put `zu_spg` and `zu_trd` on the full local
94 by 152 domain but `zu_frc` on the 90 by 148 interior at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:180-182`.
Eight full arrays plus that one interior array predict exactly 2,042,100
bytes. The acquisition check was corrected and the existing record admitted;
there was no rebuild or rerun.

The per-rank writer opens and stamps the stream at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:458-467`,
records histories and midpoint at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:501-513`,
records update operands, exchanged exit, and history rotation at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:814-822`,
and closes at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:866-870`.

## 2. Compiled statement walk

The inherited round-18 boundary reproduces exactly: substep 1 is bit-exact;
substep-2 `continuity_du` first differs at 64 / 8,794 wet T cells with maximum
`7.705384632572532e-07`; and its direct midpoint U operand differs on 64 / 128
adjacent faces with maximum `1.2223159767330016e-15`. Substitution of recorded
U still closes all 64 continuity cells.

For both substeps, recorded and candidate midpoint expressions replay their
own targets bit-exactly on all 128 scored faces. Their history rotations are
also exact: substep-1 exit equals substep-2 entry, and the recorded substep-2
entry replays its forward midpoint exactly.

NEMO's compiled vector update consumes only the interior loop at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:688-711`.
On the 64 scored live support cells, both recorded and candidate expressions
replay bit-exactly in both substeps; `zu_frc` is exact on every comparable
owned wet U cell. The first live non-bit boundary is therefore substep-1
`ua_exit`: 64 / 128 adjacent faces, all rank-0 west halo column 0 at rows
19–82, maximum `1.2223159767330016e-15` and maximum 497,491 ULP.

The exchange immediately follows the live update at
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:754-770`.
Because the U-history payload is written after that call, the record localizes
the birth to the exchange boundary but cannot attribute the neighbor source or
mapping statement.

### Retraction kept in the instrument

Before the live-domain correction, the gate named raw substep-1 halo
`zu_spg` as earlier non-bit: 64 / 128 faces, maximum
`0.00021082126041452646`; raw `zu_trd` also differed on 27 / 128 faces,
maximum `0.0010453629368898268`. That claim is **WITHDRAWN**. Those full-array
halo values are diagnostics outside the compiled vector-update loop and are
not consumed by the statement. The gate now retains them only under
`raw_full_update_arrays_on_mismatch_adjacent_faces` and excludes them from
`live_rows_in_compiled_source_order`; it no longer prints either as the first
statement.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R19-P1 | **CONFIRMED** | Both distinct 2,042,100-byte ranked streams admit with the expected headers, substeps, completion stamps, and split markers. |
| R19-P2 | **CONFIRMED** | The inherited first boundary, count, maximum, and substep-1 identity reproduce exactly. |
| R19-P3 | **CONFIRMED** | Both midpoint statements replay recorded and candidate targets bit-exactly. |
| R19-P4 | **REFUTED** | Raw full-array halo `zu_spg` and `zu_trd` are non-bit before the exit; the prediction failed as written. Compiled-loop inspection shows those cells are not live operands, so the withdrawn raw-halo claim is not promoted. |
| R19-P5 | **CONFIRMED** | Substep-1 post-exchange `ua_exit` is first live non-bit and rotates exactly into substep-2 `un_e` and midpoint. |
| R19-P6 | **CONFIRMED** | Both interior vector replays are exact and all 64 mismatches lie on the unrecorded update halo; the record stops at the exchange boundary. |
| R19-P7 | **CONFIRMED** | Swapped-rank, halo-payload, one-ULP midpoint-target, and one-ULP exchanged-exit plants all fire. |
| R19-P8 | **CONFIRMED** | No `packages/` diff; the independent ladder remains measured with the same kt=10 entry-T boundary. |

No prediction was rewritten after measurement.

## 4. Acquisition needed

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round19_preexchange_acquisition/run.sh`.
It creates isolated target `ORCA2_ORCA1ICE_OMIP_L4_R19PREX` and new run
`round19/acquisition/orca1ice_u_preexchange_ranked_np2`. Its WRITE-only patch
records per-rank U immediately before the compiled exchange for substeps 1 and
2; the admitted U-history already supplies the corresponding post-exchange
array. It assigns no model value. Preflight passes all pinned source, CPP,
deck, input, binary, patch, dry-apply, and tool checks. Admission refuses
wrong rank sizes, failed-run stamps, missing markers, or absent `STOP 0`.

The acquisition was not launched in this sandbox: the binding operator note
records that PMIx socket creation is refused here. The operator must run the
script; no workaround was attempted.

## 5. Ladder, review, and tests

The final measurement-tip ORCA2 kt=1..10 ladder exits 0 with
`LADDER_MEASURED`. Its trajectory remains
`MEASURED_INDEPENDENT_WITH_DECISION52_SSH`; kt=10 entry T remains maximum
`3.9430791763114783` on 430,552 cells. No GYRE rerun is required because
`git diff 16608cd9a..04c5bb6c8 -- packages/` is empty. The inherited day-30
GYRE digest remains `14a7e64b4512860e`; this round does not claim a rerun.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed acquisition tip. It failed before reading the diff because
the in-process app-server client could not initialize on a read-only
filesystem. Verdict: **independent review unavailable in-sandbox**.

The clean history gate exits 0 with `STOP_AT_POST_EXCHANGE_U_HALO`; its plant
run fires every control and exits 1 as intended. Gate JSON, ladder output,
review output, and test logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round19/codex`.
The focused round-18/19 gate and citation tests pass **36 / 36**.

## Choices

ASKED: admit the corrected ranked U-history record and walk its compiled
statements to the first live non-bit boundary.

UNASKED: none. Decisions 54, 57, and 58 remain pending and untouched. No
configuration choice, carried-state change, stabilizer, sea-ice change, or
production statement was made.

## OPEN

1. Run and admit the ranked pre-exchange acquisition. Map rank 0's west
   post-exchange U values to rank 1's recorded east pre-exchange source, then
   walk the compiled halo exchange to the first non-bit mapping statement.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part `dyn_ldf` scoping), 57 (`r1_rho0` spelling), and
   58 (second continuity solve) remain pending and untouched.
4. The independent ORCA2 year remains dependent on the scheduled independent
   initial-state completion; this solver result is only given NEMO's entry.
