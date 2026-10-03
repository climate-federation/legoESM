# NEMO testcase Lane 4 — ORCA2 card round 13 preregistration

Date: 2026-09-23

Parent: `638a18efbc1843690a816db8beb0efbc0a880f1d`

Status: **PREREGISTERED BEFORE ANY ROUND-13 MEASUREMENT.**

Two statements, in this order, on the ocean-only `orca2_vector_een_c2` card.

**A.**  Round 12's OPEN item 1: pair the river runoff's **WATER** with the heat
that round 12 landed.  Transcribe how the record's ORCA2 deck applies the
runoff volume, gate it, and then re-measure the river-mouth cells — that
measurement is the discriminator for round 12's hypothesis, with round 12's
confound removed.

**B.**  Round 12's OPEN item 2: the kt=1 stage-1 disagreement is
UNATTRIBUTED.  Rank its rows by magnitude (sea surface, then the two
velocities, then the tracers) and either name an owner or state the one
discriminating measurement round 14 must run.

**C.**  Decision 52's labels ("given NEMO's entry" versus "independent") stay
on every number.  The six-entry sea-ice registry is frozen and out of scope.
Decision 54 (the three-part lateral-viscosity scoping) is pending with the
user and **nothing touches that operator this round**.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment.
The record is the pinned ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| river runoff | ON, `ln_rnf = T` | run `ocean.output:534` |
| runoff river-mouth treatment | `ln_rnf_mouth = T` | run `ocean.output:648` |
| runoff multiplier | `rn_rfact = 1.0` | run `ocean.output:651` |
| runoff DEPTH spreading | NOT selected — neither `ln_rnf_depth`'s nor `ln_rnf_depth_ini`'s banner appears | run `ocean.output:646-667` |
| closed seas | OFF, `ln_closea = F` — so no closed-sea runoff redistribution runs | run `ocean.output:118` |
| freshwater-budget control | `nn_fwb = 2`, "volume adjusted from previous year budget" | run `ocean.output:532,1652` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

`nn_fwb = 2` adds a single SCALAR correction to `emp` and leaves `rnf`
untouched, and the record's surface frames carry `emp` AFTER that correction,
so no new statement is owed for it this round.  `ln_closea = F` removes the
only compiled path that would move runoff between cells.

No selector default, tunable, threshold, cadence, resolution, timestep,
carried state or data source moves in this round.  GYRE has no runoff and must
stay bit-identical.

## Statement A — the runoff WATER, as the compiled source spells it

Everything below is read from the build that produced the record,
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo`.

| # | what NEMO does with the runoff volume | compiled owner |
|---|---|---|
| A1 | the runoff is added to the three-dimensional horizontal divergence, once per `div_hor` call | `divhor.f90:142` |
| A2 | with neither depth option selected the SURFACE arm runs: `h_rnf` becomes the live top-cell thickness and `phdivn(:,:,1)` is DECREASED by `rnf * r1_rho0 /` that same live thickness — a division, written out inline rather than through `h_rnf` | `sbcrnf.f90:279-283` |
| A3 | the barotropic sea-surface forcing is `r1_rho0 * (emp - rnf)`, with the subtraction formed BEFORE the density reciprocal multiplies | `stp2d.f90:278-281` |
| A4 | the stage-1/2 tracer concentration/dilution term reads `emp` **ALONE**; the runoff is not in it | `trasbc.f90:282-288` |
| A5 | the tracer content the runoff carries is deposited separately, at all three stages (round 12's landing) | `trasbc.f90:318-328` |

`sshwzv.f90`'s `ssh_nxt` also spells a surface freshwater term, and it is
**NOT cited as executing**: `grep` over the whole compiled branch finds no
`CALL ssh_nxt` anywhere, so that routine is dead in this RK3 build.

**Rule 4 — what was searched before anything was written.**  `runoff`, `rnf`,
`runoff_mass_flux`, `emp` and `FreshwaterForcing` across
`packages/ocean/legoesm/ocean/` and `scripts/validate/ocean_fidelity/`.
What already exists, and is therefore REUSED rather than rebuilt:

| NEMO statement | legoESM owner that already exists |
|---|---|
| A1/A2, the runoff into the divergence | `ocean_pe_latlon_cgrid.py:1577-1585`, the `runoff_mass_flux` branch of `nemo_transport_wzv_divergence_level`, already cited to `sbcrnf.F90:253-260` |
| the production wiring of that branch | `ocean_model_latlon_cgrid.py:6330-6331`, which already passes `freshwater.runoff` into the stage transport |
| A3, the sea-surface forcing | `ocean_model_latlon_cgrid.py:5725-5727`, `freshwater_eta_tendency`, whose net is `precip - evap + runoff + ice_fw (+ restoring)` — algebraically `-(emp - rnf)` on this card |
| an existing bit-exact gate for A1/A2 against NEMO's own `ww` and `pFw` | `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_wzv_gate.py` |
| the production-binding stage-source harness | `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round12_runoff_gate.py`, whose `_stage_sources` already has a `with_runoff_mass` arm |

**So statement A owes exactly ONE new model statement, and it is A4.**
legoESM's stage-1/2 dilution operand is `net_freshwater_flux(freshwater)`,
which INCLUDES the runoff (`ocean_model_latlon_cgrid.py:6413-6415`).  NEMO's
`emp` does not.  That single line is why round 12's pairing arm was
confounded: supplying the runoff there deposits a second, near-identical copy
of the same heat (`emp`'s dilution carries `rnf * T_top / rho0 / h`, and the
runoff source carries `MAX(sst,0) * rnf / rho0 / h`).  The fix is
unconditional — no new selector, no default that keeps the old behaviour.

The card-side change is one line in the ladder's surface assembly: the
recorded `rnf` frame becomes `FreshwaterForcing.runoff` instead of zeros.
That is NEMO's own recorded input, not a configuration choice.

**Nothing else is switched on.**  In particular the runoff's temperature and
salinity FILES stay unselected (the record does not select them), the depth
spreading stays unselected, and `nn_fwb` is not re-implemented.

## Statement B — what owns the kt=1 stage-1 disagreement

Round 12's ranked stage-1 rows, all "given NEMO's entry" (the kt=1 entry row
is bit-identical on all five fields):

| field | max abs | unequal / scored |
|---|---|---|
| sea surface | 8.1656e-02 m | 8,794 / 13,320 |
| zonal velocity | 6.4143e-02 m/s | 247,035 / 399,600 |
| meridional velocity | 3.3963e-02 m/s | 237,822 / 399,600 |
| salinity | 1.8538e-03 | 233,341 / 399,600 |
| temperature | 1.5287e-03 degC | 233,341 / 399,600 |

The walk is magnitude-first (Decision 43), so the sea surface owns it.  The
method is one-variable substitution using the instruments that already exist
against the round-5 record: the stage-1 transport gate, the WZV/runoff gate
and the hydrostatic-pressure-gradient gate.  The first of those that refuses
names the owner; if all of them hold, the owner is downstream of them in the
barotropic solver and round 14 gets the named discriminating measurement
rather than a guess.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R13-P1 | The runoff WATER needs **no new operator**: both of its legoESM owners (the divergence branch and the sea-surface tendency) already exist at the base tip, so supplying the recorded `rnf` on the card reaches both with no new model statement. | the Rule-4 table above resolves at the base tip, and a production step with the runoff supplied moves the stage-1 sea surface. | either owner is missing, or supplying the runoff changes nothing. |
| R13-P2 | At the BASE commit, supplying the runoff water moves the stage-1 tracer dilution rate, because legoESM's operand includes the runoff and NEMO's `emp` cannot. | the base-commit arm differs on more than zero cells; at the tip the same comparison is 0 unequal on all 799,200. | the base-commit arms already agree, which would put the double deposit somewhere else. |
| R13-P3 | **THE DISCRIMINATOR.**  With the water paired correctly the stage-1 temperature on the top cell of a runoff column falls back toward round 11's heat-free 1.4655e-04 degC — within a factor of two of it, and in any case well below round 12's heat-only 1.5287e-03. | the river-mouth maximum is below 3.0e-04 degC. | it stays at or above 1.0e-03, which refutes round 12's hypothesis and leaves the heat landing unexplained. |
| R13-P4 | The runoff does NOT own the sea-surface row: its own contribution over one stage is millimetres, three orders below 8.2 cm. | the stage-1 sea-surface maximum stays within 5 percent of 8.1656e-02 m. | it falls by an order of magnitude, which would make the runoff water the owner after all. |
| R13-P5 | legoESM forms the sea-surface freshwater tendency by DIVIDING by `rho_0` while NEMO multiplies by `r1_rho0` (`stp2d.f90:281`), and on this card that is a real, measurable difference on at least one cell. | a non-zero count of cells where `x / rho0` and `x * (1/rho0)` differ in fp64. | they agree on every cell, in which case the spelling is vacuous here and is reported as vacuous. |
| R13-P6 | GYRE is bit-identical: it resolves no runoff, so the subtraction removes exactly `+0.0` and `x - 0.0` is bitwise `x`. | 0 differing rows, array-equal residuals, byte-identical 30-day snapshots, day-30 digest `14a7e64b4512860e`. | any movement. |
| R13-P7 | The ladder still runs kt=1..10 after the landing and the first non-bit statement stays UNATTRIBUTED at kt=1 stage 1 — the runoff cannot reach the 231,291 cells that lie off every runoff column. | `LADDER_MEASURED` with the same unattributed first statement. | the ladder refuses, or the first statement moves. |
| R13-P8 | Statement B's owner is NOT the stage-1 transport, the WZV recurrence or the pressure gradient: each of those already has a gate and each is at bar, so the sea-surface row is owned downstream of them. | those gates stay at bar with the landing applied. | one of them refuses, which names the owner immediately. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- The round-12 runoff gate's EXISTING rows must reproduce EXACTLY when it is
  re-run at this round's tip; a moved old row means the change altered the
  instrument and the round stops.
- The new water gate must REFUSE at the round's base commit, and its refusal
  must be the dilution row rather than an import error.
- Every new control must be shown to EXECUTE and to FIRE — round 12's review
  found three controls that never ran, so each control prints its own line.
- A one-representable-value plant on the production rate must make the gate
  refuse, and the receipt quotes the plant's own fired line, never the exit
  code.
- Both new tests must be shown to FAIL when the landed statement is reverted,
  and the model file restored clean (`git status --porcelain`).
- GYRE's trajectory is proven unchanged at the round's base tip and at its
  final tip, with the evaluation protocol byte-identical.
- The citation gate must PASS and its plant must FIRE on a REAL citation key
  that the receipt RENDERS (round 12's recorded blind spot).
- No stabiliser, clip, damp or limiter NEMO lacks may be added, and no
  configuration value is chosen that the record does not resolve.
- The lateral-viscosity operator is not touched (Decision 54 pending).
- The runoff temperature/salinity files, the depth spreading and the
  freshwater-budget correction are NOT switched on.

## Labels

The masks, metrics and reference thicknesses come from input files the card
already owns, so operator-gate numbers built only from them are
**independent**.  Every `rnf`, `rnf_tsc`, `emp` and sea-surface frame is the
record's own, so every number computed on them is **given NEMO's entry**.  The
ladder's trajectory rows keep their existing `INDEPENDENT_WITH_DECISION52_SSH`
label.
