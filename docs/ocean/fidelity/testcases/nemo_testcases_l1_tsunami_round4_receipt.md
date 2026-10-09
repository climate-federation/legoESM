# RECEIPT — TSUNAMI lane, round 4 (build B6, B7; select the density depth)

Date 2026-10-08. Lane tip at start `c7115d9fd461`. Preregistered in
`PREREG_nemo_testcases_l1_tsunami_round4.md` (commit `e4f153afa`), before any
round-4 code or measurement. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round4/`,
every `ladder_*.json` stamped with the committed tree `3a914d3b29fe`.

Status: **LANDED.** Three card selections, each NEMO's own:

1. **Density depth (DECISION 101).** The card selects the existing
   `eos_depth = "geometric"` arm. Given NEMO's kt = 1 entry: stp_2D
   right-hand side 1.09e-19, ssh 7.63e-17 — round 3's measurement arm,
   reproduced exactly.
2. **B6, momentum advection OFF.** The card ran flux-form UP3; NEMO runs
   none. With the new card-selected `"none"` arm the right-hand side sits on
   the fixed 1.09e-19 floor at every kt = 1..10 (was 4.7e-10 .. 8.2e-09).
3. **B7, tracer advection OFF.** The card ran FCT2; NEMO runs none. T, S
   given NEMO's entry are bit-identical at kt = 1..5 and within 2 ulp
   (7.1e-15) at kt = 6..9 (were 4.6e-03, 6.9e-03).

Net: the card's own 10-step chain now ends 2.5e-16 from NEMO on ssh
(round 3: 1.05e-03). Over NEMO's 100 steps the error stays at the floor
through kt = 14, then grows from the j-seam (PLAUSIBLY B4j): 2.7e-13 at kt = 15,
6.1e-05 at kt = 20, 5e-03 .. 8e-03 from kt = 25 (ssh peak 0.1 m).

## 1. What NEMO runs (read from the compiled source and the run's log)

- `ln_dynadv_OFF` sets `n_dynadv = np_LIN_dyn` (`dynadv.f90:185`). Nothing
  in `dyn_adv`'s dispatcher handles it, so the stage calls
  (`stprk3_stg.f90:316`, `stprk3_stg.f90:334`) add nothing; stp_2D adds no
  advection and takes the plain vertical average (`stp2d.f90:176`). The
  vorticity is the planetary one only (`dynvor.f90:857-860`); the card's
  `een_planetary` arm already hands the triad f alone.
- `ln_traadv_OFF` sets `nadv = np_NO_adv` (`traadv.f90:444`); `tra_adv` adds
  nothing to the zeroed trend (`stprk3_stg.f90:467`), so stages 1-2 are the
  thickness ratio alone (`stprk3_stg.f90:503-505`).
- The reference run's `ocean.output` says so: lines 723 ("no momentum
  advection used"), 738 ("total vorticity = Coriolis"), 690 ("NO T-S
  advection"), 609 ("NO lateral diffusion").

## 2. What was built (one shared implementation each)

Searched first: `VALID_MOMENTUM_FLUX_SCHEME`, `VALID_VERTICAL_MOMENTUM_SCHEME`,
`VALID_TRACER_ADVECTION` and every reader of `momentum_flux_scheme`,
`vertical_momentum_scheme`, `tracer_advection` in `packages/ocean`. No
"none"/OFF arm existed in any of them; the flux-form momentum dispatcher has
one call site, the vertical one one, the tracer one is shared by every
tracer step. So the arms were added there, not beside them.

- **B6.** `"none"` in the flux-form horizontal and vertical momentum
  dispatchers (return the right-hand side unchanged). The two are one
  selection: a half-OFF pair, OFF without flux form, or OFF with the
  adaptive-implicit vertical advection is refused. The RK3 identity check
  admits a third complete program, `flux_form / none / none`.
- **B7.** `"none"` in the shared tracer-advection dispatcher (zero flux
  divergence), and in the RK3 tracer stage step a zero concentration
  right-hand side, so stages 1-2 take NEMO's `(1 + r3t)` ratio form rather
  than the thickness-weighted content form. Stage 3 keeps the card's
  content update (old thickness times T over new thickness), which stands in
  for NEMO's `tra_zdf` solve; it is measured bit-identical on T, S at every
  kt = 1..10 (section 3, stage-local), and the unit test covers stage 1 only.
  The RK3 identity admits `fct2` or `none`; `store_salt_flux` refuses
  `none`.
- **Card.** `build_tsunami_zco_card` selects `eos_depth="geometric"`,
  `momentum_flux_scheme="none"`, `vertical_momentum_scheme="none"`,
  `tracer_advection="none"`, each with its citation. B6 and B7 leave
  `TSUNAMI_UNMEASURED`; B4 and B4j stay.
- **Caller grep** (`momentum_flux_scheme="none"|vertical_momentum_scheme=
  "none"|tracer_advection="none"` over `packages/ src/ scripts/`): the only
  selections are the TSUNAMI card's. No config field or library default was
  added or changed (`state.py` defaults: `upwind`, `upwind_perturbation`,
  `tvd`, `insitu`).
- **Harness.** The existing ladder gains a `record100` arm (the card's own
  chain against NEMO's `f_` record at kt = 1..100) and each row's
  largest-error cell. Nothing new was written for it beyond that.

## 3. The ladder, kt = 1..10 (both labels, never mixed)

**GIVEN-NEMO-ENTRY, stp_2D right-hand side** (max abs, u; v the same):
1.09e-19 at every kt except kt = 8 (1.63e-19). Unequal cells grow with
the wave (140 at kt = 1, 12754 at kt = 10); first unequal cell (69, 38) at
kt = 1, an interior cell.

**GIVEN-NEMO-ENTRY, whole step** (max abs):

| kt | ssh | uu_b = u | vv_b = v | T | S |
|---|---|---|---|---|---|
| 1 | 7.63e-17 | 5.20e-18 | 5.20e-18 | 0 | 0 |
| 2 | 8.33e-17 | 8.67e-18 | 8.67e-18 | 0 | 0 |
| 5 | 8.67e-17 | 6.94e-18 | 7.81e-18 | 0 | 0 |
| 6 | 8.67e-17 | 5.20e-18 | 6.07e-18 | 7.11e-15 | 7.11e-15 |
| 9 | 1.37e-16 | 8.68e-18 | 8.68e-18 | 3.55e-15 | 3.55e-15 |
| 10 | 1.36e-16 | 6.94e-18 | 5.64e-18 | n/a | n/a |

**INDEPENDENT** (the card's own chain from its own initial state):

| kt | ssh | uu_b | vv_b | T | S |
|---|---|---|---|---|---|
| 1 | 7.63e-17 | 5.20e-18 | 5.20e-18 | 0 | 0 |
| 5 | 3.68e-16 | 1.21e-17 | 1.30e-17 | 0 | 0 |
| 10 | 2.49e-16 | 1.63e-17 | 1.65e-17 | n/a | n/a |

First over the bar: ssh at every kt, both labels (normalised 1e-16 .. 4e-16
is under the 1e-15 bar; "over" here means not bit-identical — the ladder's
first unequal row is ssh at every kt).

**dyn_spg_ts substeps** (179 boundaries per kt). With the card's own
forcing, the first unequal row is the entry forcing `zu_frc` at 1.1e-19
(the right-hand-side floor), 14 rows over the bar, 134 .. 170 unequal. With
NEMO's recorded forcing, the first unequal is substep 1's velocity update,
0 .. 11 rows over the bar, 126 .. 149 unequal — unchanged from round 3, as
expected: B6/B7 do not reach the loop.

**Stage-local** (NEMO's handoff and stage entry handed in, kt = 1..10):
stages 1 and 3 bit-identical on T, S at every kt; stage 2 bit-identical at
8 of 10 kt, 1 ulp (3.6e-15) at kt = 7 and 10; ssh, u, v at 1e-17 .. 1e-18
as in round 3.

**First non-bit statement after B6/B7**: the stp_2D right-hand side, at the
fixed 1.09e-19 floor, `stp2d.f90:137-138` (eos + dyn_hpg, the same
statement as round 3, now 9 orders smaller). It is interior (cell (69, 38)
at kt = 1), NOT on the j-seam. Its residual is round 3's open item 3
(exploratory 8.6e-13 static-depth piece is gone with `geometric`; the floor
itself is not attributed).

## 4. The 100-step record (task 4)

INDEPENDENT, the card's chain against NEMO's `f_` groups. Max abs / rms,
and the cell of the max (j, i; j = 0 and 200 are the j-seam rows):

| kt | ssh | uu_b (= u) | vv_b (= v) |
|---|---|---|---|
| 10 | 2.5e-16 / 1.8e-17 (125, 41) | 1.6e-17 / 2.0e-18 | 1.6e-17 / 2.0e-18 |
| 14 | 4.1e-16 / 2.6e-17 **(200, 39)** | 2.6e-17 / 2.8e-18 | 3.7e-16 / 6.6e-18 **(200, 39)** |
| 15 | 2.7e-13 / 6.4e-15 **(0, 39)** | 6.7e-15 / 1.6e-16 (200, 33) | 2.1e-13 / 3.6e-15 (200, 39) |
| 16 | 6.1e-11 / 1.5e-12 (200, 39) | 1.6e-12 / 3.9e-14 | 4.1e-11 / 7.7e-13 |
| 20 | 6.1e-05 / 2.1e-06 (200, 39) | 2.2e-06 / 7.1e-08 | 2.7e-05 / 7.7e-07 |
| 25 | 6.2e-03 / 6.8e-04 (0, 56) | 7.3e-04 / 5.1e-05 | 1.9e-03 / 2.1e-04 |
| 50 | 8.2e-03 / 2.1e-03 | 1.7e-03 / 3.4e-04 | 1.6e-03 / 6.1e-04 |
| 100 | 5.1e-03 / 1.7e-03 | 1.6e-03 / 4.6e-04 | 2.0e-03 / 5.4e-04 |

**CONFIRMED** every field leaves the bar at kt = 15 (the harness's
`first_kt_over_bar`). The ssh and vv_b maxima sit on a j-seam row at the
source column from kt = 14, every field's from kt = 15 (uu_b's kt = 14
maximum is interior, (31, 43)). **PLAUSIBLE** that this is B4j, the card's
step walling the j-seam: location only, one trajectory, no A/B against a
j-periodic step. The i-seam is crossed from kt = 5 (round 3) with no departure from
the floor through kt = 14: PLAUSIBLE evidence B4's i-seam statements are
faithful to ~1e-16 (one run; no A/B).

## 5. B4j (task 3)

The kt = 1..10 ladder's first non-bit statement is the interior
right-hand-side floor, so per the round brief B4j is **named and measured,
not built**: it PLAUSIBLY owns the 100-step record from kt = 14 (section 4),
growing by ~1e2 per step to 1e-3 by kt = 25. It is the next build.

## 6. ORCA2 pointer

ORCA2 runs vector-invariant EEN momentum and FCT2 tracers, so neither OFF
arm is an ORCA2 statement, and ORCA2 already selects `geometric`. What
transfers: with the external-mode forcing at 1e-19, TSUNAMI's whole
split-explicit program — `stp2d.f90:137-138`, the dyn_spg_ts loop, the
handoff and the per-stage barotropic correction — carries NEMO's state to
<= 4e-16 on ssh for 14 steps, across an open i-seam. That is PLAUSIBLE
evidence that ORCA2 rung 0's external-stage ssh debt is in what feeds the
loop (its forcing, its lateral boundaries), not in the loop.

## 7. Predictions scored

| id | prediction | result |
|---|---|---|
| P1 | D101: rhs 1.09e-19, ssh 7.63e-17 at kt = 1 | **CONFIRMED** exactly |
| P2 | B6 arm bit-unchanged; UP3 plant fires | **CONFIRMED** (unit tests; removing the branch turns them red) |
| P3 | rhs at the 1.09e-19 floor, kt = 2..10 | **CONFIRMED** (max 1.63e-19) |
| P4 | given-entry ssh < 1e-15, kt = 2..8; first non-bit = substep velocity update | ssh **CONFIRMED** (<= 1.4e-16, also kt 9-10). First non-bit **FALSIFIED** as worded: in NEMO's order the first unequal row is the stp_2D rhs floor; the substep update is first only given NEMO's forcing |
| P5 | B7 stage 1 = qco ratio bitwise; FCT2 plant fires | **CONFIRMED** |
| P6 | given-entry T, S <= 1e-13; stages 1, 2 bitwise | T, S **CONFIRMED** (<= 7.1e-15). Stages 1, 3 bitwise; stage 2 **FALSIFIED** at 2 of 10 kt by 1 ulp |
| P7 | independent ssh at kt = 10 < 1e-13 | **CONFIRMED** 2.49e-16 |
| P8 | 100 steps: leaves 1e-12 not before kt 13, does leave it | **CONFIRMED** (1e-12 left at kt = 16; the bar at kt = 15) |
| P9 | first non-bit within kt 1..10 not on the j-seam | **CONFIRMED** |

## 8. Review

Single review (codex), verdict **DO NOT SHIP** on `c7115d9fd461..` the
pre-fix diff (`codex_review.txt`); it found no momentum sign, index,
staggering or fall-through defect on the executed TSUNAMI path and no
changed library default. Disposition, every finding CONFIRMED and acted on:

- HIGH, the card validator did not refuse a TSUNAMI card reverted to
  `insitu`, UP3 or FCT2: fixed, the TSUNAMI branch now refuses each
  (test, shown red with the check removed).
- MEDIUM, `store_salt_flux` with `none` reached the dispatcher and raised
  late: fixed, refused at construction (test, shown red without it).
- MEDIUM, the tracer-dispatch test used zero transports, so any scheme
  passed: fixed, nonzero transports plus an upwind plant that must differ.
- MEDIUM, B4j "owns" the later record overclaimed, and uu_b's kt = 14
  maximum is interior: relabelled PLAUSIBLE, location stated per field.
- LOW, stage 3 does not take the ratio form: stated (section 2).

## 9. Choices made this round

| choice | ASKED or UNASKED |
|---|---|
| card selects `geometric` | ASKED (DECISION 101) |
| card selects momentum and tracer advection OFF | ASKED (round brief, items 1-2) |
| the momentum OFF arm is two field values (`momentum_flux_scheme`, `vertical_momentum_scheme`) enforced as one selection, rather than a new `momentum_advection` value | UNASKED (NEMO keeps `ln_dynadv_vec=F` with OFF, so the flux-form program is the right parent; offered for revert) |
| `"none"` with the adaptive-implicit vertical advection is refused | UNASKED (a refused combination NEMO never runs; offered for revert) |
| the stage-entry plant test now asserts the plant is inert, with an FCT2 card plant proving the seeding works | UNASKED (test text; the plant became physically inert) |
| `record100` arm and the largest-error cell in the harness | UNASKED (instrument addition) |
| B4j left unbuilt | per the round brief's own condition |

## 10. Gates

| gate | result |
|---|---|
| unit tests, new arms (7) | `7 passed`; removing the three "none" branches: `3 failed, 4 passed` |
| ladder + card tests | `29 passed, 1 xfailed` (round 2's strict B4j pin), plus the rewritten stage-entry test `1 passed` |
| ratchets: dispatch hardening, validate-strict coverage, private imports, constants, inline coefficients, config footguns | `5861 passed, 2 failed`: both reds are in files this round never touched (`tests/unit/test_jra55_do.py` literal 273.15; `packages/land/legoesm/land/restart.py`) |
| VORTEX/WS-RK3 suites | `170 passed, 3 failed`: the three certified-card digest tests fail identically on the round's starting commit `c7115d9fd461` (checked in a throwaway worktree) — pre-existing |
| citation gate, this receipt | `PASS`, 0 failures, 0 map entries failing audit (after re-anchoring 70 legoESM-line keys the arms shifted); planted shift of `dynvor.f90:857-860` `FAIL` (`SYMBOL-NOT-AT-LINE`), exit 1 |
| unit tests after the review fixes | `11 passed` (off arms), card `28 passed, 1 xfailed` with them |
| NOT gated (honour system) | dual review: codex only (headless brief names one review). Controlled comparison: each landing differs from its before arm in the named fields only. Non-vacuity: shown above |

Evidence sha256 prefixes: rhs `3c1f7b5e136eb11e`, given_entry
`92a1431b43bb4f35`, independent `13babe5b7e253700`, stages
`9ba79af2b2585af8`, spgts `3bd689d19b7ff2e4`, record100
`15e32ece25eab375`; D101 confirmation (dirty tree, superseded by the
stamped runs) rhs `0742d96564beb4c1`, given_entry `8766fecc7416445c`.

## 11. OPEN, for round 5

1. **B4j** — build the j-periodic wrap in the card's barotropic path
   (the card's halo exchange honouring `NEMOTestcaseCard.j_periodic`), with
   a test; it PLAUSIBLY owns the 100-step record from kt = 14 (the A/B
   against a j-periodic step is the round-5 measurement).
2. The 1.09e-19 right-hand-side floor (`stp2d.f90:137-138`), unattributed.
3. B4 (i-seam) stays declared; section 4 is the first measurement of it.

## Citations

Momentum advection OFF: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90:185`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:176`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:316`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:334`; its vorticity:
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynvor.f90:857-860`.
Tracer advection OFF: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90:444`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:467`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:503-505`.
The density depth: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90:361`.
The first non-bit statement: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:137-138`.
The one-level correction: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`.
