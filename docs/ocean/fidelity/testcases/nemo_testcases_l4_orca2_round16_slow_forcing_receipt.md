# NEMO testcase Lane 4 — ORCA2 card round 16 slow-forcing receipt

Date: 2026-09-25

Parent: `956fd11fc257dc88bfc18c39399fe350f2e3301a`

Measurement tip: `88e4e4f78efc1edd4c00d14c1d9551618be41507`

Status: **HELD.**  Given NEMO's entry, the recorded slow momentum forcing
moves the kt=1 end-of-step sea surface by only
`2.667551013690872e-07 m`.  The complete substep-1 vector update becomes
bit-exact after that forcing is substituted.  The walk advances to substep 2,
where the first scored non-bit boundary is the U-flux difference used by
continuity: 64 cells, maximum `7.705384632572532e-07`.

No production statement lands.  No file under `packages/` changed.  The six
sea-ice selectors and the card's `unmeasured_features` tuple are unchanged:
`staged_gm_eiv`, `linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every solver number below is labelled **given NEMO's entry**.  The whole-card
ladder retains its separate **independent with Decision-52 SSH** label.  The
two populations are not mixed.

## 1. Record and executing statements

The admitted record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
The gate ran production JIT on CPU with fp64/x64/libm and the resolved
`orca2_vector_een_c2` card: 65 explicit substeps, `nemo_ab3am4`,
`nemo_literal` continuity, substep drag, and `forward_euler`.

NEMO's executing vector branch is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`:
entry velocity plus the barotropic time step times pressure gradient, combined
trend, and slow forcing, followed by the face mask.  The record writes those
operands and exits at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:755-779`.
Its next non-bit boundary is formed by the continuity differences at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`.

Rank 0's owned 90 longitude columns are scored.  No rank-1 or full-domain
claim is made.

## 2. Search and reuse

Before implementation, the repository was searched for slow-forcing readers,
substitution hooks, record slicing, masks, and vector-update replays.  The
round-15 reader, native-face slicing, entry/drag/history substitutions,
production-JIT trace and source order were reused.  The already-existing
private `barotropic_slow_forcing_override` hook was reused unchanged.  The
only new code is a validation gate and its direct tests; no model API,
operator, config field or selector was added.

## 3. Inherited result reproduced

The committed gate first reconstructs round 15's final arm.  It reproduces:

| row | result |
|---|---:|
| cold-history causal movement | exactly `0.0 m` |
| first active boundary | substep-1 `slow_u` |
| `slow_u` maximum | `5.370080135032166e-12` on 8,568 / 8,568 wet faces |
| `slow_v` maximum | `2.2928581685638914e-11` on 8,554 / 8,554 wet faces |
| inherited end-of-step SSH maximum | `0.24444708169024776 m` |

Any different value was a preregistered hard stop.  All five reproduced.

## 4. The first non-bit arithmetic statement

The arithmetic replay uses NEMO's recorded substep time
`166.15384615384616 s` and the source association shown in the compiled
statement.  As a control, NEMO's own recorded operands replay its recorded
`u_exit` and `v_exit` with zero unequal cells.  legoESM's operands similarly
replay its production-JIT exits with zero unequal cells.  The replay is
therefore the recorded quantity, not a proxy.

| substep-1 boundary | U maximum / unequal | V maximum / unequal |
|---|---:|---:|
| pressure gradient + combined trend | **bit-exact**, 0 / 8,568 | **bit-exact**, 0 / 8,554 |
| add slow forcing | `5.370080135032166e-12`, 8,568 | `2.2928581685638914e-11`, 8,554 |
| multiply by the substep time | `8.922594685670249e-10`, 8,568 | `3.809672033947605e-09`, 8,554 |
| add entry velocity | same `8.922594685670249e-10`, 8,568 | same `3.809672033947605e-09`, 8,554 |
| apply face mask / recorded exit | same `8.922594685670249e-10`, 8,568 | same `3.809672033947605e-09`, 8,554 |

Thus the first non-bit arithmetic statement is the addition of slow forcing.
The pressure-gradient-plus-trend sum is bit-exact; the later rows merely carry
the slow-forcing difference through the compiled statement.

## 5. Causal substitution

The gate retains the recorded entry velocity, recorded drag and six recorded
cold histories, then replaces only the final slow U/V arrays over rank 0.

| control or measurement | result |
|---|---:|
| own-slow trace no-op, U exit | bit-exact |
| own-slow trace no-op, V exit | bit-exact |
| own-slow no-op, end-of-step SSH | bit-exact |
| recorded slow forcing landed | U `0 / 8,568`, V `0 / 8,554` unequal |
| any earlier substep-1 boundary moved by the hook | none |
| end-of-step SSH cells moved | 8,794 / 13,320 |
| maximum end-of-step SSH movement | `2.667551013690872e-07 m` |
| movement / round-15 baseline | `1.0894940807133483e-06` |
| substituted end-of-step SSH maximum | `0.24444706952780731 m` |

The movement is below both preregistered thresholds: `1.0e-06 m` for the
direct-scale prediction and `2.4e-02 m` for ownership.  The slow forcing is
not a material owner of the 24 cm row.

After substitution, all five arithmetic boundaries in section 4 are
bit-exact on both faces, including the production-JIT `u_exit` and `v_exit`.
The complete substep-1 source walk is bit-exact.  The first later scored
boundary is substep-2 `continuity_du`, with 64 / 8,794 cells unequal and
maximum `7.705384632572532e-07`.  This receipt calls it a boundary, not an
owner: the gate has not yet localized whether its two face operands differ in
an unscored halo or dry-face region.

## 6. Non-vacuity plant

A one-ULP increase at recorded native U-face coordinate `[1, 49]` changes
exactly one scored slow-forcing cell by one ULP and changes exactly one
substep-1 `u_exit` cell by one ULP (`5.421010862427522e-20`).  The plant gate
prints `PLANT FIRED` and intentionally exits 1.  The substitution channel is
not deaf.

## 7. Frozen predictions

| ID | verdict |
|---|---|
| R16-P1 | **CONFIRMED** — all inherited round-15 values reproduce exactly. |
| R16-P2 | **CONFIRMED** — own-slow substitution is a bit-exact no-op; recorded slow forcing lands; no earlier row moves. |
| R16-P3 | **CONFIRMED** — causal movement is `2.667551013690872e-07 m`, below `1.0e-06 m` and `2.4e-02 m`. |
| R16-P4 | **CONFIRMED** — pressure gradient plus trend is bit-exact; adding slow forcing is first non-bit. |
| R16-P5 | **CONFIRMED** — both substituted vector updates and the complete substep-1 walk are bit-exact. |
| R16-P6 | **CONFIRMED** — the kt=1..10 ladder is unchanged. |
| R16-P7 | **CONFIRMED** — no file under `packages/` changed and nothing lands. |

No prediction was refuted.

## 8. Whole-card ladder and GYRE disposition

The current-tip ORCA2 ladder exits 0 with `LADDER_MEASURED`.  Its first
candidate statement remains kt=1 stage-1 temperature, unattributed.  The
kt=10 entry-temperature row remains maximum `3.9430791763114783` on 430,552
cells.  No row leaves its bar and no first-over-bar moves earlier.

No GYRE trajectory rerun is claimed or required: `git diff --name-only
956fd11fc..88e4e4f78 -- packages/` is empty.  Cards do not execute validation
scripts, and the existing private model hook was not changed.

## 9. Review, gates and tests

The required separate command was attempted at the committed measurement tip:
`codex exec --sandbox read-only`.  It failed before reading the diff because
its app-server client could not initialize on the read-only filesystem.
Verdict: **independent review unavailable in-sandbox**.

The clean gate exits 0.  The one-ULP gate prints `PLANT FIRED` and exits 1.
The ORCA2 ladder exits 0 with `LADDER_MEASURED`.  The receipt citation gate
passes all 3 rendered compiled citations with 0 unmapped and 0 failures.  Its
rigid two-line shift of the rendered vector-update citation fails with
`SYMBOL-NOT-AT-LINE`, as required.

The round-16 gate and receipt-citation tests pass **8 / 8**.  The required
`tests/ocean/fidelity -n 12` battery was launched exactly once: 1,648 tests
were collected, it reached 99%, and its final worker did not complete after a
bounded tail wait, so the run was interrupted with exit 130 and has no passing
summary.  Three failure IDs appeared before the hang and all three reproduce
in isolation:

| failure | disposition |
|---|---|
| GYRE round-129 record-backed gate | pre-existing worktree/record-certification ratchet: "the certified phase-3 stepping gate moved after the members ran" |
| SI3 scalar-math v2 gate | supplied known red: `A MY_SRC is not verbatim` |
| GYRE round-51 live-operand field-order assertion | pre-existing stale assertion against the current trace tuple |

`git diff 956fd11fc..HEAD` is empty for all six failing test/implementation
paths and for every file under `packages/`; none is caused by round 16.  The
wide battery is **not represented as green**.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round16/codex`.

## OPEN — round 17's order

1. Continue from substep-2 `continuity_du`: localize its 64 cells and compare
   both U-flux operands, including the halo/dry faces that the face-row mask
   does not score, before attributing the subtraction.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part `dyn_ldf` scoping), 57 (`r1_rho0` spelling), and
   58 (second continuity solve) remain pending and untouched.
4. The independent ORCA2 year still depends on the scheduled independent
   initial-state completion; this round is only given NEMO's entry.
5. The interrupted wide ocean-fidelity battery tail from round 15 remains an
   operator action.

## Choices

ASKED: retain the cold-history substitution, substitute recorded slow U/V,
measure causal movement, and name the first non-bit velocity-update arithmetic
statement.

UNASKED: none.  No configuration choice, carried-state change, stabilizer,
sea-ice change, or production statement was made.
