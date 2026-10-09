# ORCA2 round 212 — OMT-1 card, ladders, and vector boundary

Date: 2026-10-09. Base: `68981c39b`. Preregistration commit:
`8b57459b7`. Measurement commits: `185cc7b6b`, `c1c5f0928`,
`df83e10b5`, `d79224be7`, and `4d80072a5`. Status: **LANDED** for the
explicit OMT-1 card/gates; the first vector statement remains **HELD** as an
atomic cancelling pair. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round212/`.

## Verdict

The Decision-109 OMT-1 edge is admitted and executable. Its only NEMO deck
changes from OMT-0 are `ln_dynadv_OFF=.true.→.false.` and
`ln_dynadv_vec=.false.→.true.`. The instantiated card restores vector-invariant
KEG/ZAD/EEN while drag, momentum LDF, tracer advection, and tracer LDF remain
OFF. The shipped rung-10 card, its sea-ice selectors, and its
`unmeasured_features` tuple are unchanged.

Round 211's record re-admits: 64 self-describing frames per twin, 320
array-identical twin field comparisons, four byte-identical terminal restarts,
and the registered NEMO `stp_ctl` boundary at kt=9 (`|V|=10.13 m/s`). Both
labelled legoESM ladders complete the full admissible kt=1..8 interval: 32
checkpoints and 160 rows each, finite throughout. **Independent OMT-1** uses
the card's own corrected climatological entry. **Given NEMO's entry OMT-1**
bridges the recorded entry. Their numerical scores are identical; only inactive
temperature signed-zero counts differ.

The first checkpoint with any bit difference is kt=1 stage 1. Contrary to
R212-P3, the gate's field-order sentinel reports temperature first, not SSH.
This is retained as **REFUTED**. In NEMO source order, the earlier internal
external-mode walk is more discriminating:

1. `ssh_frc = sshe_rhs` is the first non-bit statement, but all 8,794 wet
   differences are signed zero and its numerical error is exactly zero
   (`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:287-289`).
2. `zv_frc = Ve_rhs` is the first numerical debt: 35 northern-fold halo cells,
   maximum `1.6557659420864476e-06 m/s²`; active wet interiors are exact. The
   paired U forcing is bit-exact.
3. In the direct vector V update, entry V and `rDt_e` are bit-exact; pressure
   and trend differ only by signed zero; `zv_frc` differs on those 35 cells and
   the NEMO raw `ssvmask` differs on the same 35 cells by 1.0. Replaying the
   written expression with both recorded operands closes the raw result
   bit-exact; either half alone leaves 35 cells and a
   `2.7511187960820976e-04 m/s` maximum. NEMO's statement is
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:674-678`.

Thus the first numerical statement is not a one-variable landing. The
`Ve_rhs` fold-halo association and the vector update's raw V mask are an atomic
cancelling pair. A recorded-`Ve_rhs` one-variable replay installs that operand
bit-exact but still leaves the 35-cell V-exit debt. The admitted stream records
`va_e` only after `lbc_lnk`
(`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-753`), so the actual pre-LBC
`va_e` target remains **UNMEASURED_WITH_SPEC**. No physics change lands.

## Ladder scores

The entry rows account for the only exact rows: independent `4/160` exact and
given-entry `5/160` exact (the extra row is the already-classified dry-T zero
representation); the remaining rows are registered debt.

| label | checkpoint | T rms / max K | S rms / max PSU | u rms / max m/s | v rms / max m/s | SSH rms / max m |
|---|---|---:|---:|---:|---:|---:|
| independent OMT-1 | kt1 stage1 | `1.4249708087564076e-05 / 0.0012980888239468857` | `3.289151381491274e-04 / 0.0259273789706711` | `8.559390565263422e-04 / 0.06149743563337632` | `8.945073892070253e-04 / 0.03413076175499323` | `0.006243172975388979 / 0.13141649093684893` |
| given NEMO's entry OMT-1 | kt1 stage1 | same | same | same | same | same |
| independent OMT-1 | kt8 stage3 | `2.882408027663207e-04 / 0.03822035478318342` | `0.003873886435800206 / 0.828576467579218` | `0.003917373715014321 / 0.5200348870173395` | `0.004505846763093037 / 0.790138104782633` | `0.03182845477041683 / 0.7583316946619745` |
| given NEMO's entry OMT-1 | kt8 stage3 | same | same | same | same | same |

R212-P4 is **CONFIRMED**: stage-1 SSH returns to the vector-rung scale, maximum
`0.13141649093684893 m`, instead of OMT-0's post-landing `0.0063 m` scale.
The first numerical statement and raw cancelling pair satisfy R212-P5's
source-order requirement, but the missing pre-LBC target withholds a landing.

## Controls and blast radius

The admission, card-module, entry-bit, record-header, twin-ULP, source-order,
passivity, terminal-ULP, slow-V replay, and vector-mask replay plants all fire.
The offline trace reproduces the untraced candidate terminal SSH, U/V external
mode, and U/V accumulated transports bit-for-bit before any trace value is
read. CPU fp64/libm and production JIT are printed in every machine report.

No file under `packages/` changes in this round. Therefore GYRE, DINO, tanks,
rung 0, and shipped rung 10 cannot execute a changed statement; their
trajectories cannot move. The OMT-1 card is a committed validation identity,
not a change to an existing production card or selector.

## Validation and review

- Round-211 admission replay: PASS, including all nine record plants.
- OMT-1 ladders: PASS, 32 checkpoints / 160 rows for each label.
- OMT-1 offline vector walk: `PASS_R212_OMT1_VECTOR_WALK`; seven distinct
  replay plants fire.
- Focused OMT-1 and round-212 tests: PASS (14, then 8 tests during gate
  construction; final focused and full fidelity results recorded below).
- Citation gate and independent review results are recorded below after their
  mandatory final-diff runs.

## OPEN

1. Record per-rank substep-1 vector `va_e` immediately after
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:674-678` and before `lbc_lnk`; prove the writer additions-only
   against the admitted OMT-1 step-8 restarts.
2. Gate the complete raw-`Ve_rhs` + raw-`ssvmask` pair against that target,
   then score it atomically on OMT-1, rung 0, rung 10, GYRE, DINO, and tanks
   under Decision 96. No partial operand lands.
3. OMT-2 (+ linear implicit bottom drag) waits until this OMT-1 unit lands or
   names its next partner.

No configuration or sea-ice decision is pending. ASKED choices: Decisions 103
and 109. UNASKED choices: empty.
