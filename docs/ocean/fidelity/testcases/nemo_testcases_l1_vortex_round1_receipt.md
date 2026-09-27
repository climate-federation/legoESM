# RECEIPT — VORTEX card, round 1 (transcription + acquisition)

Date 2026-09-27. Lane tip `c09a9e111780`. Preregistered in
`PREREG_nemo_testcases_l1_vortex_round1.md`, frozen before anything was built.

Status: **ACQUISITION_NEEDED.** Everything below is built, tested and
committed. The kt=1..10 rows are empty because NEMO has not been run: MPI is
refused in the agent's sandbox, so round 1 wrote the acquisition and stopped.

## 1. What was built

| artefact | what it is |
|---|---|
| the VORTEX card | additive in the shared NEMO test-case recipe: one new whole-step identity, one new builder, one new validator branch, one new dispatch entry |
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
| certified card digests unchanged | LOCK `f248153cc366f9ea`, OVERFLOW `090acab214d20672`, GYRE `abfd869f4b1c4d66` — identical before and after |
| push-gate tests | see section 6 |
| citation gate | see section 6 |

## 4. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| TEOS-10 instead of the shipped S-EOS | ASKED (decision 64, operator note BF) | but see the FINDING below, which is new information the decision did not have |
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

## 6. Test and gate output

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
