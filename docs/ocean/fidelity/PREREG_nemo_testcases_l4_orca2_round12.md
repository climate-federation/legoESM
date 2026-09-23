# NEMO testcase Lane 4 — ORCA2 card round 12 preregistration

Date: 2026-09-23

Parent: `4dfdd5a1cebc080e23eb38ff1ad31cbd57c19264`

Status: **PREREGISTERED BEFORE ANY ROUND-12 MEASUREMENT.**

Two statements, in this order, on the ocean-only `orca2_vector_een_c2` card.

**A.** Name the THIRD, unattributed 2 to 4 percent that round 11 left after
both known lateral-viscosity differences were applied to the compiled loops
(round 11 closure: L2 0.0267 on `u`, 0.0190 on `v`).  One-variable
substitutions only, on the record's own coefficients, metrics and kt=2
step-entry velocities.  **Nothing lands from statement A**: Decision 54
(mask-only versus both halves) is pending with the user, so both halves stay
gated, switchable and unlanded.

**B.** Round 11's OPEN item 3: the river-runoff tracer source
(`trasbc.f90:318-326`), a channel the production step does not carry.
Transcribe it exactly as ORCA2's namelist resolves it, gate the deposit
against the record's own `rnf_tsc` and `rnf` frames, and re-run the kt=1..10
ladder.

**C.** Decision 52's labels ("given NEMO's entry" versus "independent") stay on
every number.  The six-entry sea-ice registry is frozen and out of scope.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment.
The record is the pinned ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` (`nn_ahm_ijk_t = -30`) | run `ocean.output:1184,1197,1200-1201` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| river runoff | ON, `ln_rnf = T` | run `ocean.output:534` |
| runoff river-mouth treatment | `ln_rnf_mouth = T`, `rn_hrnf = 15 m`, `rn_avt_rnf = 1e-3` | run `ocean.output:648-650` |
| runoff multiplier | `rn_rfact = 1.0` | run `ocean.output:651` |
| runoff source file | `runoff_core_monthly`, variable `sorunoff`, monthly climatology | run `ocean.output:657-660` |
| runoff DEPTH spreading | NOT selected — `ln_rnf_depth` and `ln_rnf_depth_ini` are both absent from the printed block, so the surface arm runs | run `ocean.output:646-667` |
| runoff temperature / salinity data | NOT selected — `ln_rnf_tem` and `ln_rnf_sal` are both absent from the printed block | run `ocean.output:646-667` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

No selector default, tunable, threshold, cadence, resolution, timestep,
carried state or data source moves in this round.  GYRE has no runoff and
resolves `rn_shlat = 0`; it must stay bit-identical.

## Statement A — where the last 2 to 4 percent lives

There is **no `dynldf_lap_blp.f90`** in this build; the compiled owner of both
the laplacian and the bilaplacian level operators is `dynldf_lev.f90`, and the
resolved arm is `dyn_ldf_lev_lap`'s `np_typ_rot` branch (`dynldf.f90:85`,
called at Runge-Kutta stage three by `stprk3_stg.f90:493`).  Its four
statements are `dynldf_lev.f90:123`, `:127-129`, `:133-136` and `:137-140`.

Round 11 substituted TWO legoESM behaviours into that transcription — the
extra zero/one vertex mask on `ahmf`, and legoESM's min-rule live
`e3u`/`e3v`/`e3f` — and closed 29x/32x of the gap.  Reading the two sides
statement by statement, exactly THREE legoESM behaviours were NOT substituted,
and they are this round's candidates.

| ID | candidate statement | compiled owner | legoESM owner |
|---|---|---|---|
| A-e3t | the `zwt` OUTER DIVISOR.  NEMO divides by the live T thickness `e3t_0*(1+r3t(Kbb)*tmask)`; round 11's thickness ablation overrode only the FACE and VERTEX thicknesses, never this one, so the transcription kept NEMO's `e3t` while production divides by its own `compute_layer_thickness` output. | `dynldf_lev.f90:127` | `latlon_cgrid_operators.py:1798` |
| A-metric | the HORIZONTAL METRICS.  The transcription reads NEMO's `e1t,e2t,e1u,e2u,e1v,e2v,e1f,e2f` from the record's mesh-mask file; production reads its own stored `grid.area`, `grid.area_q`, `grid.dx_u`, `grid.dy_u`, `grid.dx_v`, `grid.dy_v`. | `domain.f90:197-205`, `:212-215` | `operators_latlon_cgrid.py:887` |
| A-slope | the SLOPE-FOOT factor the production path applies to the operator's output after it returns. | (no NEMO statement) | `ocean_pe_latlon_cgrid.py:3221-3223` |

Round 11's own OPEN item named a fourth candidate, NEMO's stage-three
`Kbb`/`Kmm` split.  It is **excluded a priori and not measured**: the gate
drives both sides with ONE recorded sea surface, so no `Kbb`-versus-`Kmm`
difference can exist inside it.  Recording that here rather than measuring it
is the one-variable rule, not an omission.

## Statement B — the river-runoff tracer source

With `ln_rnf = T` and neither depth option selected, NEMO resolves:

| statement | compiled owner | value |
|---|---|---|
| the runoff temperature content | `sbcrnf.f90:221` | `rnf_tsc(:,:,jp_tem) = MAX(sst_m, 0) * rnf * r1_rho0` — the runoff enters at the sea surface temperature, floored at 0 |
| the runoff salinity content | `sbcrnf.f90:227` | `rnf_tsc(:,:,jp_sal) = zrnf_sal * rnf * r1_rho0` with `zrnf_sal = 0` (`sbcrnf.f90:175`) — identically zero |
| how deep it goes | `sbcrnf.f90:487-488` | the surface arm: `nk_rnf = 1` and `h_rnf = e3t_0(1)*(1+r3t(Kmm)*tmask(1))`, the LIVE top-cell thickness |
| the tracer-side deposit | `trasbc.f90:318-326` | `zdep = 1/h_rnf` formed FIRST, then `pts(Krhs) += rnf_tsc * zdep` on levels 1..`nk_rnf` |
| which stages execute it | `trasbc.f90:318` | the runoff block sits OUTSIDE `SELECT CASE(kstg)` (`trasbc.f90:278`), so it runs at ALL THREE Runge-Kutta stages, unlike the EMP/QNS block |

`rnf`, `rnf_b`, `rnf_tsc` and `rnf_tsc_b` are all recorded in every surface
frame the round-5 acquisition wrote (the frame schema at
`nemo_testcase_l4_orca2_phase2b_exchange_gate.py:46,52`), so NO NEW NEMO RUN IS
NEEDED and the gate can score the deposit against NEMO's own content field
rather than a reconstruction of it.

**Rule 4 — what was searched before anything was written.**  `runoff` and
`rnf` across `packages/ocean/legoesm/ocean/`.  What exists: the runoff MASS
channel (`FreshwaterForcing.runoff` -> `runoff_mass_flux` into the horizontal
divergence, cited to `sbcrnf.F90:253-260` at
`ocean_pe_latlon_cgrid.py:1563-1565`), the NEMO depth-spreading virtual-salt
helper (`freshwater.py:485` `runoff_spread_virtual_salt_tendency_3d` and
`freshwater.py:442` `resolve_runoff_spread_arg`), and the MPAS/lat-lon callers
of both.  What does NOT exist: any channel carrying the runoff's TRACER
CONTENT (its heat).  The deposit is therefore an extension of the existing
surface-forcing channel set, not a second runoff path, and the mass side is
NOT touched this round.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R12-P1 | A-e3t is the LARGEST of the three candidates: substituting legoESM's own live cell thickness into the `zwt` divisor moves the closure residual more than either other candidate. | the one-variable A-e3t row moves the closure L2 by more than A-metric and A-slope do. | another candidate moves it more, or A-e3t moves it by less than 10 percent of the residual. |
| R12-P2 | A-slope is VACUOUS on this card: the slope-foot factor is the scalar 1.0, so its substitution changes nothing. | the A-slope row is bit-identical to the closure row. | it moves any cell, in which case a silent factor is multiplying the operator. |
| R12-P3 | With all the named substitutions applied together the residual falls BELOW the round-11 closure (0.0267 u / 0.0190 v) — the third difference is named, not merely re-described. | the full-closure L2 falls on both components. | it does not fall, which leaves a FOURTH unattributed difference and is reported as such. |
| R12-P4 | legoESM's stored horizontal metrics are NOT bitwise NEMO's mesh-mask metrics on this tripolar card, so A-metric is a real term rather than a null one. | a non-zero unequal count between the six metric pairs. | they are bitwise equal, in which case A-metric is vacuous and is reported as vacuous. |
| R12-P5 | `rnf` is non-zero on only a SMALL fraction of the 799,200 cells (river mouths), so the runoff source cannot own the 233,341-cell stage-1 temperature disagreement round 11 recorded; the ladder's hard-coded stage-1 attribution to `trasbc.f90:314-328` is therefore TOO BROAD. | the recorded `rnf` is non-zero on far fewer cells than 233,341, and the stage-1 temperature difference is non-zero on cells where `rnf` is exactly zero. | `rnf` is non-zero on essentially every wet cell. |
| R12-P6 | The runoff SALINITY content is identically zero in the record (`zrnf_sal = 0`), so the runoff explains NONE of the stage-1 salinity disagreement. | `max abs(rnf_tsc(:,:,jp_sal))` is exactly 0 in the record. | it is non-zero, which would refute the reading of `sbcrnf.f90:175,227`. |
| R12-P7 | Transcribed, the deposit reproduces NEMO's own increment bit-exactly on every cell where `rnf` is non-zero, at kt=1. | 0 unequal on the scored cells, with a gate that refuses at the base commit and whose one-representable-value plant fires. | any unequal cell. |
| R12-P8 | The ladder still runs kt=1..10 after the landing, and its first disagreement MOVES to a different statement (because the runoff cannot own a whole-field difference). | `LADDER_MEASURED` with a first non-bit statement that is not the runoff source. | the ladder refuses, or the first statement does not move. |
| R12-P9 | GYRE is bit-identical: it has no runoff (the card supplies no runoff content, so the new channel is `None` and the branch is not taken) and statement A lands nothing. | 0 differing rows, array-equal residuals, byte-identical 30-day snapshots, day-30 digest `14a7e64b4512860e`. | any movement. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- The round-11 operator gate's EXISTING rows must reproduce EXACTLY when the
  gate is re-run with the new rows added; a moved old row means the extension
  changed the instrument and the round stops.
- Every new substitution must be shown to MOVE the number; one that changes
  nothing is reported as VACUOUS, never as agreement.
- The runoff gate must REFUSE at the round's base commit (the channel does not
  exist there) and its binding row must run the PRODUCTION step, not a
  re-implementation of the deposit.
- A one-representable-value plant on the production increment must make the
  runoff gate refuse.
- Both new tests must be shown to FAIL when the landed statement is reverted,
  and the model file restored clean (`git status --porcelain`).
- GYRE's trajectory is proven unchanged at the round's base tip and at its
  final tip, with the evaluation protocol byte-identical.
- The citation gate must PASS and its plant must FIRE on a REAL citation key;
  the receipt quotes the gate's own fired line, never the exit code.
- No stabiliser, clip, damp or limiter NEMO lacks may be added, and no
  configuration value is chosen that the record does not resolve.
- The runoff MASS side (`sbcrnf.f90:263,275` into the horizontal divergence,
  and the `emp - rnf` sea-surface forcing) is NOT switched on this round.
  Turning it on would change the sea surface, which is a separate statement
  and a separate landing.

## Labels

The coefficients, metrics, masks and reference thicknesses come from input
files the card already owns, so the operator-gate numbers are **independent**.
The kt=2 velocities, the sea surfaces and every `rnf`/`rnf_tsc` frame are the
record's own, so every number computed on them is **given NEMO's entry**.  The
ladder's trajectory rows keep their existing `INDEPENDENT_WITH_DECISION52_SSH`
label.
