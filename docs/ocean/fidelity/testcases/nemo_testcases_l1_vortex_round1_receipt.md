# RECEIPT — VORTEX card, round 1 (transcription + acquisition)

Date 2026-09-27. Lane tip `c09a9e111780`. Preregistered in
`PREREG_nemo_testcases_l1_vortex_round1.md`, frozen before anything was built.

Status: **ACQUISITION_NEEDED, and the card is NOT EXECUTION-READY.**

Two things a reader must take away before anything else:

1. **The ladder survey's "UNLOCKED" verdict for VORTEX is REFUTED.** This case
   runs NEMO's energy-and-enstrophy scheme on FLUX-FORM momentum, where that
   scheme IS the Coriolis operator, and legoESM has no such arm: it binds its
   own EEN implementation to vector-invariant momentum and refuses the pair
   outright. The two tanks never exposed this because their rotation is
   structurally dead. The card declares the gap and its execution gate refuses
   it; it does not substitute legoESM's 4-point average for NEMO's triad.
2. **The acquisition is still worth running.** The initial state does not
   depend on the momentum scheme, so the bit-exactness claim stands, and the
   record is what a later round needs to measure the new arm the day it exists.

The kt=1..10 rows are empty because NEMO has not been run: MPI is refused in
the agent's sandbox, so round 1 wrote the acquisition and stopped.

## 1. What was built

| artefact | what it is |
|---|---|
| the VORTEX card | additive in the shared NEMO test-case recipe: one new whole-step identity, one new builder, one new validator branch, one new dispatch entry, and one declared capability gap that makes the execution gate refuse it |
| the card test | 11 tests, 10 run and 1 skips until the oracle record exists |
| the acquisition | builds the shipped case twice, runs both, admits by restart byte-identity |
| the record checker | parses each record's own header; predicts no size |

The card resolves the shipped namelist as tabulated in the preregistration's
section 1 and is not restated here.

## 2. The transcription

844 lines of NEMO user-defined Fortran were read; the parts that produce state
are transcribed statement by statement, in NEMO's own execution order, with
operand order preserved. The transcription itself is about 300 lines of Python
in the shared recipe module.

The one non-obvious thing, and the reason this case is not a copy of the tank
pattern: **the initial state depends on the sea surface height it also sets.**
NEMO's ordering is `rst_read_ssh` (which calls the user's ssh routine), then
the quasi-Eulerian coordinate builds the surface-height ratio from that ssh,
and only then does the initial-state routine receive the cell depth — the
LIVE, stretched one, not the one-dimensional reference ladder. Temperature and
both velocity components are functions of that stretched depth. Evaluating them
on the reference ladder instead is a 1.8e-4 relative depth error at the vortex
centre, worth 2.9e-3 degC at the deepest level, and the card test pins the
difference.

## 3. Gates

| gate | result |
|---|---|
| card test, `tests/ocean/unit/test_nemo_vortex_card.py` | 10 passed, 1 skipped |
| planted defect: reference depth ladder instead of the live stretched depth | RED, as required |
| planted defect: meridional velocity sign flipped | RED, as required |
| record checker on synthetic records | admits a good set; refuses a corrupt header, a perturbed reference restart and a missing step |
| acquisition preflight (no build, no run) | PREFLIGHT_OK; both patches apply to the shipped sources |
| certified card digests unchanged | LOCK `42d13c75ea8cbcc6`, OVERFLOW `c2bca636ac2f14ef`, GYRE `4a6f0b6b0827ac6a` — identical before and after |
| push-gate tests | see section 6 |
| citation gate | see section 6 |

## 4. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| TEOS-10 instead of the shipped S-EOS | ASKED (decision 64, operator note BF) | but see the FINDING below, which is new information the decision did not have |
| the oracle deck leaves the Courant-dependent implicit vertical advection OFF, unlike the two tanks' decks | UNASKED, and stated as a finding | it is the shipped VORTEX default and the card now matches it, so deck and card agree with no new deviation; the tanks' decks turn it on, so this diverges from their convention. One line for the operator: leave it off (my pick, no deviation) or match the tanks? |
| the AGRIF zoom is out of scope; parent grid only | ASKED (the round's own task) | both oracle builds drop the nesting key |
| ten-step run cadence for the oracle run | UNASKED, precedent-based | it is the tanks' own kt1_10 cadence, and the shipped 3000-step length is preserved on the card itself |
| mesh receipt written by the oracle run | UNASKED, precedent-based | the tanks' kt1_10 decks do the same |

**FINDING (carried from the preregistration, repeated here because it is the
one thing a reader must not miss).** VORTEX's initial temperature is defined by
inverting the simplified equation of state: it is 20 degC plus the density
anomaly divided by that equation's thermal expansion coefficient. The
coefficient is still read from the namelist under TEOS-10, so the INITIAL STATE
is unaffected and the bit-exactness prediction stands. What the deviation
changes is everything after step 1: the eddy was constructed to be in
geostrophic and hydrostatic balance under the linear equation of state, and
TEOS-10 will not reproduce that balance. The run is therefore an OMIP-style
variant of VORTEX, not the published balanced-vortex experiment. This is
acceptable for an oracle-match ladder and would not be acceptable as a physics
result. One line for the operator: keep TEOS-10 (the campaign default, my pick)
or run this one card on its shipped equation of state?

**Correction to the ladder survey.** The survey listed VORTEX as running a
horizontal Laplacian on momentum. It does not: the namelist sets the operator
OFF and only leaves the direction flag true. The card has zero lateral
viscosity, which makes it a simpler case than the survey implied.

## 5. What awaits the record

Nothing below can be filled in until the operator runs the acquisition.

| row | awaiting |
|---|---|
| initial state, cells unequal for T, S, u, v and ssh | the kt=1 step-entry record |
| the kt=1..10 ladder, five rows per step | the same run |
| the derived barotropic velocity watch row | the same run |
| citation re-anchor onto the compiled sources | the build |
| the kt=1..10 scorer | round 2; the existing trajectory gate hard-codes the tanks' header tuples and must parse each record's own header first |

ACQUISITION_NEEDED. The exact command is in section 7.

## 6. Reviews, and what they changed

Two independent adversarial reviews ran on the diff before anything was
finished. Both returned DO NOT SHIP, and both named THE SAME blocker
independently, which is the strongest signal either one could have given.

| reviewer | verdict | the blocker |
|---|---|---|
| a fresh Claude code-reviewer, given the diff and the oracle | DO NOT SHIP | the card silently inherited the Courant-dependent implicit vertical advection from a sibling case's namelist |
| codex, adversarial, read-only | DO NOT SHIP | the same switch, named as active momentum physics that no gate checked |

What each finding turned into:

| finding | disposition |
|---|---|
| BLOCKER: the card inherited implicit vertical advection ON while its own namelist leaves it OFF | FIXED. The card states it explicitly, the validator refuses a card that turns it on, the card test asserts it, and the acquisition refuses a deck that grows the switch. Both reviewers were right; this WOULD have broken the ladder from step two. |
| the oracle comparison would have crashed on shapes: NEMO's record carries one more vertical level than the card executes | FIXED. The comparison now asserts the record's level count equals the card's plus its dummy bottom record, and trims to the card's levels — the same thing the tanks' own gate does. |
| "bit-exact" was being checked with floating equality, which calls positive and negative zero the same value | FIXED. The comparison is now on raw bit patterns. |
| the record checker ignored the format version, never cross-checked the step in the filename against the step in the header, and required only the step-entry records | FIXED, and the fix immediately exposed a real defect in the checker itself: it had the barotropic record's header one integer too wide. Corrected against the writer. |
| a guard reading the pipeline status after a failing run could never be reached, and a symbol check could report clean if its own tool failed | FIXED: the run redirects instead of piping, and the symbol list is captured before it is searched. |
| the additivity digest omitted the vertical coordinate, the masks, the face Coriolis fields and the barotropic pair, so it could not support "no existing card moved" | FIXED. The digest now covers all of them, and the claim was re-measured against the lane tip with the wider digest: still identical. |
| one reviewer could not confirm that the lateral-diffusion OFF switch wins over the direction flag | NOT a defect: the card carries zero lateral viscosity either way, and the validator requires it. |
| one reviewer noted the checker's plant only exercises one refusal path | ADDRESSED by exercising four: a corrupted header, a version bump, a filename-versus-header step disagreement, and a missing record. |

**And then the review fixes found the real blocker.** Adding VORTEX to the
card constructibility tripwire — a check that builds a real model, not just a
configuration — turned it red immediately: legoESM refuses to combine the
energy-and-enstrophy vorticity scheme with flux-form momentum, because its
flux-form branch never receives the vertex Coriolis field and the combination
would drop the rotation term entirely. NEMO's own dispatch confirms the case
needs exactly that pair, and that on the flux-form arm the scheme is applied to
the planetary vorticity alone, so it IS the Coriolis operator here. The card
now declares the gap and fails its execution gate rather than running
legoESM's 4-point average under NEMO's name. This is the finding of the round,
and neither reviewer found it: only building a model did.

One thing found while fixing the first blocker, worth recording. The claim that the
two tank cards contradict their own decks is FALSE and is retracted here before
anyone builds on it: their SHIPPED namelists differ on this switch, but their
campaign decks both set it on, which is what their cards resolve. VORTEX is the
first card on this identity whose deck leaves it off, which is why the
inherited value was wrong only here.

A pre-existing observation, not this round's to fix: the reference layer
thickness the cards carry is single precision on every card, while the
thicknesses NEMO actually executes are double. It is a reference ladder, not an
executed operand, but it means a sub-1e-7 change to it would not move the
digest above.

## 6b. Test and gate output

Filled in at the end of the round; see the commit range in section 7.

## 7. How to acquire

Preflight first (reads only, writes nothing outside a temporary directory),
then the real thing:

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh --run
```

Evidence lands in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round1`,
with the un-instrumented reference run beneath it.

## 8. Citations

Every span below is checked by the citation gate against the real file.

The case's own user-defined sources. Global domain size, three integers:
`tests/VORTEX/MY_SRC/usrdef_nam.F90:96-97` for the horizontal and `:121` for
the vertical. The Cartesian origin, half a box west and south of centre:
`tests/VORTEX/MY_SRC/usrdef_hgr.F90:83-84`. The beta-plane Coriolis, with the
kilometre position scaled back to metres inside the product:
`tests/VORTEX/MY_SRC/usrdef_hgr.F90:174-177`. The uniform vertical ladder:
`tests/VORTEX/MY_SRC/usrdef_zgr.F90:126`. The flat bottom and its closed ring:
`tests/VORTEX/MY_SRC/usrdef_zgr.F90:187-193`. The analytic scalars shared by
the two initial-state routines: `tests/VORTEX/MY_SRC/usrdef_istate.F90:69-75`.
The temperature, defined by inverting the simplified equation of state:
`:83-88`. The zonal velocity and the half-sum of neighbouring cell depths it
uses: `:101-105`. The sea surface height: `:177-182`. The surface forcing,
which writes zeros: `tests/VORTEX/MY_SRC/usrdef_sbc.F90:60-68`.

The shared NEMO sources that fix the execution order. The user's ssh routine is
called from the restart module's at-rest arm: `restart.F90:461`. The
initial-state routine then receives the live cell depth:
`DOM/istate.F90:127-130`, which the coordinate substitution stretches by the
surface-height ratio: `domzgr_substitute.h90:139`. The barotropic velocities
are accumulated and divided as two separate statements:
`DOM/istate.F90:149-154`, with the face ratios built from the
area-weighted neighbour average: `domqco.F90:166-169`. The equation-of-state
namelist is read whichever equation is selected, which is why the thermal
expansion coefficient survives the deviation: `eosbn2.F90:1890-1895`. The
vorticity dispatch that makes this the first card with a live energy- and
enstrophy-conserving operator: `dynvor.F90:874`.
