# NEMO testcase Lane 4 — ORCA2 card round 14 preregistration

Date: 2026-09-24

Parent: `0501c06f425aba4a5a4f3628e9e6f380d52f5470`

Status: **PREREGISTERED BEFORE ANY ROUND-14 MEASUREMENT.**

Round 13's OPEN item 1, in its own words: *split the barotropic owner.*  The
largest row on the kt=1 ladder is the end-of-step sea surface, 0.2448 m, and
round 13 left its owner PLAUSIBLE rather than named.  Round 14 runs the
discriminating substitution round 13 specified and, whichever half owns it,
walks that half.

**A.**  Substitute the record's own `oracle_slow_forcing_kt00000001.bin` into
legoESM's barotropic solve and re-measure the end-of-step sea surface.  That
separates **the solver** from **what the solver is handed**.

**B.**  Walk whichever half owns it.  If the forcing owns it, the SAME record
carries the forcing's own ordered intermediates, so the walk is a boundary
ladder over them (thickness, the 3-D right-hand side, the masks, the depth
reciprocal, the depth mean, the drag increment, the wind increment, the final
value) and the FIRST non-bit boundary names the term.  If the solver owns it,
the walk moves to the compiled `dynspg_ts.f90` sub-step order.

**C.**  Decision 52's labels ("given NEMO's entry" versus "independent") stay
on every number.  The lateral-friction operator is NOT touched (Decision 54
pending); the density-reciprocal spelling is NOT landed (Decision 57 pending);
the six-entry sea-ice registry is frozen and out of scope.

## What the record fixes, and what nothing in this round may choose

Read from the card's RESOLVED configuration and from the compiled source that
produced the record, not from a deck comment.  The record is the pinned
ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is resolved |
|---|---|---|
| barotropic solver | split-explicit sub-stepping, 65 sub-steps | card `barotropic.barotropic_solver = explicit_substep`, `n_barotropic_substeps = 65` |
| barotropic time filter | `nemo_ab3am4` | card `barotropic.barotropic_time_filter` |
| barotropic Coriolis | `een_metric`, split `live` | card `barotropic.barotropic_coriolis`, `barotropic_coriolis_split` |
| planetary Coriolis in the 3-D right-hand side | `explicit_ab2` | card `coriolis_scheme` |
| barotropic bottom-drag correction | ON | card `barotropic_drag_substep = True` |
| surface stress into the barotropic forcing | ON | card `surface_stress_implicit = True` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |
| domain decomposition of the record | 2 ranks; the slow-forcing record exists for RANK 0 ONLY | record census: no `oracle_slow_forcing_rank0001_*` file exists |

The last row is a hard constraint on statement A and is stated here, before
measuring, so it cannot be discovered afterwards and quietly absorbed: the
substitution can only be applied to the 90 longitude columns rank 0 owns, and
the remaining 90 keep legoESM's own forcing.  Sea-surface information travels
at the external gravity-wave speed, so the substituted half is contaminated
inward from both of its (periodic) edges.  The receipt therefore reports the
whole-half number AND an interior number with the margin computed from
`sqrt(g*H_max)` times the step, and says which is which.

No selector default, tunable, threshold, cadence, resolution, timestep,
carried state or data source moves in this round.  GYRE must stay
bit-identical if any model file is touched at all.

## Statement A — the substitution, as the compiled source spells the operand

Everything below is read from the build that produced the record,
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo`.

| # | what NEMO does | compiled owner |
|---|---|---|
| A1 | evaluates the 3-D right-hand side ONCE at the before level — pressure gradient, lateral viscosity, vorticity, then the kinetic-energy gradient and vertical advection | `stp2d.f90:139-147,162-167` |
| A2 | depth-averages it with the REFERENCE-thickness reciprocal: `Ue_rhs = SUM(e3u_3d*uu(Krhs)*umask) * r1_hu_0` | `stp2d.f90:196-200` |
| A3 | adds the baroclinic-residual bottom drag, and records the drag coefficient with it | `stp2d.f90:219` |
| A4 | adds the wind: `+ r1_rho0 * utauU * (r1_hu_0/(1+r3u(Kbb)))` | `stp2d.f90:228-231` |
| A5 | the value after A4 is what `dyn_spg_ts` receives and is the last thing the record writes | `stp2d.f90:232-236`, consumed at `stp2d.f90:302-303` |

The record's write statements sit between those steps — `stp2d.f90:186-191`
(the 3-D operands), `:208-211` (the depth mean and `r1_h*_0`), `:220-221`
(post-drag and the drag coefficients), `:224-227` (the wind operands) and
`:232-236` (the final value) — so the record IS the ordered ladder this round
needs, and no new NEMO run is required for any arm.

**Rule 4 — what was searched before anything was written.**  Searched
`slow_forcing`, `oracle_slow_forcing`, `barotropic_slow_forcing_override`,
`slow_forcing_incoming_override`, `post_wind`, `post_drag` and `Ue_rhs` across
`packages/`, `scripts/validate/ocean_fidelity/` and `tests/`.  What already
exists, and is therefore REUSED rather than rebuilt:

| for | what already exists |
|---|---|
| reading the record | `read_slow_forcing` in `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round16_slow_forcing.py`, which already parses the `NEMO_L2_SLOW_2` layout with explicit size and EOF checks — its only GYRE-specific part is the hard-coded grid extent, so it is EXTENDED with a dimensions argument rather than copied |
| comparing a candidate field to an oracle field | `compare` in the same module (bit-exactness, absolute maximum, ULP maximum, differing cells) |
| the substitution point itself | `_NEMOWSRK3TestHooks.slow_forcing_incoming_override`, which already lands immediately before the barotropic Coriolis subtraction — exactly where NEMO hands `Ue_rhs` over |
| legoESM's own ordered intermediates | the production step already exposes them through `expose_barotropic_substeps` (`slow_forcing_operands`) and `expose_live_stage_operands` (`slow_forcing_producer`) |
| the card, the entry state, the recorded surface operands and the stage frames | `nemo_testcase_l4_orca2_round1_ladder_gate.py` |
| the end-of-step sea-surface number to reproduce | `nemo_testcase_l4_orca2_round13_stage1_owner_probe.py` |

**So statement A owes NO new model statement and no new operator.**  It owes
one new gate script and one extension (a dimensions argument) to an existing
reader.

## Statement B — the forcing's own boundary ladder

NEMO applies the drag BEFORE the wind; legoESM applies the wind before the
drag.  The two orders are additive on top of the same depth mean, so the
INCREMENTS are compared rather than the intermediates: the drag increment is
`post_drag - depth_mean` on NEMO's side and `post_drag - post_wind` on
legoESM's, and the wind increment is `post_wind - post_drag` on NEMO's side
and `post_wind - depth_mean` on legoESM's.  Stating that here, before
measuring, is the point: a boundary compared across two different orders is a
confound, not a result.

Ladder order, first non-bit boundary owns the walk:

1. face thickness (`e3u`, `e3v`)
2. the 3-D right-hand side (`uu(Krhs)`, `vv(Krhs)`)
3. the face masks
4. the depth reciprocal (`r1_hu_0`, `r1_hv_0`)
5. the depth mean
6. the drag increment
7. the wind increment (and its operands `utauU`, `vtauV`, `r1_rho0`)
8. the final value handed to the solver

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R14-P1 | The baseline end-of-step sea-surface disagreement REPRODUCES round 13's number exactly. | the stage-3 rank-0 maximum is `2.4484e-01` m to the printed digits. | any other value, which means the instrument or the tip moved and every comparison below is void. |
| R14-P2 | **THE CONTROL.**  Feeding legoESM's OWN incoming slow forcing through the substitution hook leaves the step BITWISE unchanged, so any movement in the substituted arm is the operand and not the hook. | the control arm's stage-3 sea surface is bit-identical to the baseline on every cell. | any difference at all, which would make the substituted arm uninterpretable. |
| R14-P3 | **THE DISCRIMINATOR.**  Substituting NEMO's own final slow forcing over rank 0's 90 columns moves the end-of-step sea surface by MORE than a tenth of the baseline disagreement, i.e. the forcing is a real contributor and not inert. | the substituted arm's rank-0 maximum differs from the baseline by more than `2.4e-02` m. | it moves by less than that, which puts the owner in the solver and sends the walk to `dynspg_ts.f90`. |
| R14-P4 | legoESM's final slow forcing is NOT bit-exact against the record's `post_wind` field. | a non-zero differing-cell count on rank 0's wet faces. | bit-exact, which would name the solver immediately and make P3's movement a rounding artefact. |
| R14-P5 | The first non-bit boundary in the statement-B ladder is at or BEFORE the depth mean — i.e. it is inherited from the 3-D right-hand side or from the metric, not created by the drag or the wind. | the first non-bit boundary is one of `e3`, `Krhs`, `mask`, `r1_h0`, `depth_mean`. | the first non-bit boundary is the drag increment or the wind increment, which names a two-dimensional forcing term instead. |
| R14-P6 | The depth-mean divisor is a REAL difference on this card: NEMO divides by the REFERENCE depth `hu_0` (`stp2d.f90:198`) while legoESM divides by the LIVE depth, and the card's entry sea surface is not zero, so the two disagree on more than zero wet faces. | a non-zero count of faces where the two divisors differ. | they agree everywhere, in which case the divisor is vacuous here and is reported as vacuous. |
| R14-P7 | The substituted arm's residual sea-surface disagreement is CONCENTRATED near the two rank boundaries, because only half the domain was substituted. | the interior maximum (outside the gravity-wave margin) is smaller than the whole-half maximum. | the residual is uniform across the half, which means the contamination argument is wrong and the interior number may not be quoted. |
| R14-P8 | The ladder still runs kt=1..10 and the first non-bit statement does not move earlier. | `LADDER_MEASURED` with the same first statement. | the ladder refuses, or the first statement moves. |
| R14-P9 | GYRE is bit-identical, because this round changes no model file; if a model file IS changed the identity is proven at base and tip before anything is claimed. | 0 differing rows, array-equal residuals, byte-identical 30-day snapshots, day-30 digest `14a7e64b4512860e`. | any movement. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- The no-op substitution control (R14-P2) must EXECUTE and PRINT its own line;
  a control that did not run is not a control.
- The substitution must be shown to LAND: the production trace's own incoming
  slow forcing is read back and must equal the injected operand on the
  substituted window and legoESM's own value off it.
- A one-representable-value plant on the injected operand must make the gate's
  own comparison refuse, and the receipt quotes the plant's own fired line,
  never the exit code.
- Round 13's owner probe must reproduce its published numbers at this round's
  tip; a moved old row means the instrument changed and the round stops.
- Every boundary comparison is on rank 0's OWN 90 columns and on the record's
  own mask; no number mixes the substituted and unsubstituted halves without
  saying so.
- GYRE's trajectory is proven unchanged if and only if a file under
  `packages/` is touched; if none is, the receipt says so and names the
  `git diff --name-only` that shows it.
- The citation gate must PASS and its plant must FIRE on a REAL citation key
  that the receipt RENDERS.
- No stabiliser, clip, damp or limiter NEMO lacks may be added, and no
  configuration value is chosen that the record does not resolve.
- The lateral-viscosity operator is not touched (Decision 54 pending) and the
  density-reciprocal spelling is not landed (Decision 57 pending).

## Labels

Every sea-surface, velocity, forcing and operand number in this round is
computed from the record's own kt=1 entry state, its own surface frames and
its own slow-forcing frame, so all of them are **given NEMO's entry**.  The
masks, metrics and reference thicknesses come from input files the card
already owns, so operand rows built only from them are **independent**.  The
ladder's trajectory rows keep their existing `INDEPENDENT_WITH_DECISION52_SSH`
label.
