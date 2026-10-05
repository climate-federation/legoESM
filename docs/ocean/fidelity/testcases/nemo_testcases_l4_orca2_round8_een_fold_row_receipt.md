# NEMO testcase Lane 4 — ORCA2 card round 8 transcription receipt

Date: 2026-09-22

Starting tip: `6674da8cca15681c879c7998f3bae012e4396d2d`

Preregistration: `d68556b38`  (its parent line carries a mistyped tail on that
same short hash; the tip above is the one every arm was measured against, and
no prediction depended on the string.)

Status: **HELD.**  Round 7's OPEN item 1 is discharged.  The vertex thickness
the energy-and-enstrophy vorticity operator needs is now defined on a tripolar
fold row, transcribed as NEMO states it and gated bit-exact against every
recorded F point of that row.  The ORCA2 ladder advances past it and stops on
the NEXT unbuilt statement, so the kt=10 magnitude stays **UNMEASURED** — not
zero, not extrapolated.

No configuration, selector, default, carried state, stabilizer, NEMO source or
sea-ice registry entry changed.  Sea ice remains out of scope and the
six-entry `unmeasured_features` tuple is unchanged.

Independent review **not run (codex paused)**.

## 1. What NEMO actually does on that row

It gives it no formula of its own.  Three statements, in this order:

| statement | compiled owner |
|---|---|
| masked four-cell average of the reference thickness over the OWNED domain, divided by four | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:913-919` |
| ordinary F-point north-fold exchange, sign +1 | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:935` |
| remaining zero replaced by the reference thickness | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:937` |
| the exchange itself, for an F-point field under a T pivot | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:722-746` |

The exchange is what defines the row, and the compiled loops say so in their
own comments: the row loop runs `jj = 1, ihls+1` with destination
`ipj - jj + 1`, which its comment annotates as ending at `ipj - ihls` — the
LAST OWNED row, not merely a halo row — and its source `ipj - 2*ihls + jj - 2`
ends at `ipj - ihls - 1`, the row immediately below.  The longitude loop pairs
`ii1 = ihls + ji` with `ii2 = ipi - ihls - ji + 1`, a constant sum
`ii1 + ii2 = ipi + 1`: a mirror.  The vertex-thickness call passes `1._wp`.

Nothing in this round chose anything.  The resolved run fixes every relevant
setting and each was read from the record's own output, not from a comment:
energy-and-enstrophy vorticity and `nn_e3f_typ = 0` (`ocean.output:1338-1351`,
`:1345`), a north fold of **T pivot** type (`:207-208`), inner global domain
180 by 148 with halo width two (`:38,:44,:68-69`), and two ranks split in
longitude only (`:196-202`).

## 2. What landed, and why it is not a second implementation

The rule was already written once in legoESM, for the live vorticity
thickness, and the vertex-thickness helper the vorticity operator calls was the
ONLY path that refused it.  That helper now applies the SAME shared exchange
rather than re-deriving the permutation; the shared helper's private name is
promoted so the reuse is not a private cross-module import.  The helper's
vertex array is the west-shifted layout, whose column `i + 1` is NEMO's F
column `i`, so it is shifted into NEMO's native layout and back — two exact
index moves that touch no arithmetic.

Three model lines changed, in two files, plus the rename.  No parallel ladder,
parser, launcher or second gate was created.

## 3. The measurement, and its label

Every number in this section is **independent**: the vertex thickness is built
from the card's own reference thickness ladder and masks, with no operand
loaded from the record.  Decision 52's sea-surface-height bridge is unchanged
and remains the only explicit entry replacement.

| quantity | scope | result |
|---|---|---|
| frozen vertex thickness, whole recorded block | 148 by 90 by 30 | **0 / 399,600 unequal** |
| frozen vertex thickness, the fold row alone | 90 by 30 | **0 / 2,700 unequal** |
| live vertex thickness, whole recorded block | 148 by 90 by 30 | **0 / 399,600 unequal** |
| live vertex thickness, the fold row alone | 90 by 30 | **0 / 2,700 unequal** |

The frozen operand is read out of the one shared builder by evaluating it at a
zero sea surface, which makes the live correction exactly one and leaves the
frozen field bit for bit; it is not re-derived beside it.

**The controls, each of which must leave the fold row unequal.**  A gate whose
controls pass cannot tell the rules apart, so it refuses rather than reports.

| control | unequal on the fold row | largest difference |
|---|---|---|
| no fold exchange at all | 764 / 2,700 | 248.35194650272388 m |
| the T-origin mirror instead of the F-origin one | 242 / 2,700 | 444.31758520553103 m |
| the fold row used as its own source | 801 / 2,700 | 479.1374821040031 m |

The first of these is the ablation: removing the exchange puts the ENTIRE
residual on the fold row and nowhere else, so the rule owns it rather than
shrinking it.  The second and third exist because a self-consistent but wrong
permutation would otherwise pass.

The production vertex-thickness helper — the one that used to refuse this row —
is executed on the card's tripolar grid inside the same gate, and its fold row
is checked against the compiled longitude pairing restated independently of it,
so an inverted index shift in that helper fails here.

## 4. What the record does NOT cover, made mechanical

The instrumented writer copies owned cells only, into a zero-initialised
buffer, so **every halo slot in every existing EEN record is an absence, not a
NEMO value**.  An earlier reading of this round compared all 94 recorded
longitudes and reported 17,640 unequal cells; that count is **RETRACTED** — it
was 4 columns by 30 levels in each of 147 rows, exactly the zeroed halo slots,
and the gate now asserts that emptiness so no later round can read those zeros
as the other rank's half.

So 90 of the 180 fold-row longitudes are recorded.  Their fold SOURCES all lie
in rank one's half, so the exchange across the rank boundary IS exercised — in
one direction.  Rank one's own 90 destinations are **UNRECORDED**, and the
round writes the acquisition for them rather than asserting them.

**The operand is build-invariant at the first step**, measured: the frozen
record is byte-identical across five independent builds spanning phase 2m to
phase 2v (one digest, `6239bfea69efc878`), which is what licenses the
acquisition to be built from the phase-2n lineage.

## 5. How far the ladder now reaches, and what stops it

Past the vertex thickness, into the lateral viscosity, where it stops:

> `lateral_viscosity_operator='nemo_div_curl' needs a lat-lon grid with a
> scalar dlon`

This is a bigger gap than a tripolar generalisation, and the resolved
configuration is what says so.  The record resolves `nn_ahm_ijk_t = -30`
(`ocean.output:1184`), so NEMO READS the whole three-dimensional viscosity
coefficient from `eddy_viscosity_3D.nc`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353`).  The
production arm instead builds the `nn_ahm_ijk_t = 20` formula
`½·rn_Uv·MAX(e1,e2)` from a single zonal grid spacing, which a tripolar mesh
does not have.  legoESM is therefore computing a coefficient ORCA2 never
selects, on a mesh that cannot supply its input.

The ladder gate records this the way the previous stop was recorded — named,
cited, with no magnitude registered — and exits non-zero.  Its stop classifier
is now a small registry keyed by the exact refusal text, so a refusal from any
other routine still cannot borrow a citation; anything unregistered propagates
as a defect.

## 6. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R8-P1 | **CONFIRMED** | The compiled row loop's own comment ends it at the last owned row, its source at the row below, the longitude loop pairs to a constant sum, and the call passes +1. |
| R8-P2 | **CONFIRMED** | One shared implementation of the rule and exactly one refusing path; both named in section 2. |
| R8-P3 | **CONFIRMED, with its coverage corrected** | 0 / 2,700 unequal on every recorded fold-row F point. The prediction also claimed four recorded halo longitudes in the other rank's half; there are none — those slots are empty by construction (section 4). The claim is therefore met for one fold direction, not two. |
| R8-P4 | **CONFIRMED** | GYRE base and tip agree: 70 certified rows, 0 violations, 0 worsening ULPs, all 210 residual arrays array-equal, 31 of 31 snapshots byte-identical (section 7). |
| R8-P5 | **REFUTED / kt=10 UNMEASURED** | The first non-bit statement moved later again but is still not an arithmetic row: the ladder now stops at the lateral viscosity coefficient (section 5). |

R8-P5 is the honest outcome of a ladder that gained a statement and found the
next one missing.  No magnitude is registered for a row that was never
computed.

## 7. GYRE, which shares this implementation

The changed file serves GYRE too, so GYRE was re-measured at the round's base
tip and at its tip with the eval protocol held byte-identical.

| check | result |
|---|---|
| ten-step trajectory, certified rows compared | 70 |
| rows whose status changed | 0 |
| violations | 0 |
| largest oracle-residual worsening | 0.0 ULP |
| per-cell residual arrays (210 of them) | every one array-equal |
| 30-day member, daily state snapshots | 31 files, 0 differing bytes |
| day-30 snapshot digest | `14a7e64b4512860eded79bb12a2120885b97400ecb4acb7e6a2bd3d84a245469`, the same as round 7 |
| 30-day member manifest | differs only in the recorded commit and wall time |

GYRE is inert here **by construction**, not merely by measurement: its fold
descriptor is inactive, so the new branch is unreachable on that card.  A unit
test pins that, and the day-30 digest matching round 7's is an independent
cross-round anchor.

## 8. Implementation scope, review, gates and tests

Independent review **not run (codex paused)**.  Round 7's two unreviewed gate
fixes also remain unreviewed, and they are untouched here.

New direct tests: seven on the vertex thickness at a fold row — that no rule
refuses it, that the `nemo_avg4` fold row is the row below at the mirrored
longitude, a synthetic violation proving that assertion can fail for both the
T-origin mirror and a self-sourced row, that nothing below the fold row moves
against a flat grid, and that an inactive fold takes no fold branch at all.

## 8a. Independent review

The mandated read-only independent review **was not run with codex, which the
user paused**.  A separate fresh adversarial reviewer with no part in the work
was run in its place, on the committed model diff and the oracle source.

It **confirmed the index work independently rather than on trust**: it rebuilt
a haloed array, ran the compiled Fortran row and longitude loops verbatim from
T-cell data, and compared the result to the shipped fold row — bit-identical,
largest difference 0.0.  That reconstruction starts from T data, so it does not
inherit the vertex-column convention the change assumes, which is the one thing
the round's own test could not rule out.  It also executed the pre-change
function body against the new one over all three rules, both grid kinds and
both reference-thickness settings: **byte-identical in all eleven cases, the
twelfth being the one that changes from a refusal to a value.**

It found one false statement, now fixed: the helper's parameter documentation
claimed the per-level reference thickness is consulted by only one of the three
rules, when the rule both cards select requires it outright.

It raised three residual risks, each carried into OPEN rather than argued away:

| risk | status |
|---|---|
| the dry-cell substitution is applied BEFORE the fold here and AFTER it in NEMO; the two commute only where the reference thickness is horizontally uniform per level | **MEASURED, does not bite on the recorded half**: the fold row is 0 / 2,700 unequal against NEMO's own operand. UNMEASURED on the unrecorded half. |
| this helper's rule builds its frozen sum from the uniform reference ladder, while the baroclinic lane overrides it with the literal builder that uses NEMO's own reference thickness; the helper's value now feeds the BAROTROPIC operands | **UNMEASURED.** No bit-match is claimed for it here; section 3's numbers are the literal builder's, which is what the baroclinic lane consumes. |
| the mirrored fold row derives entirely from the row below, so it is silently wrong if the card's last tracer row is not already fold-consistent | **MEASURED, does not bite on the recorded half**: same 0 / 2,700. UNMEASURED on the unrecorded half. |

A reviewer's finding is a hypothesis until measured; the first and third were
measured against NEMO's own recorded operand in this round and did not bite.
The second is genuinely open and is round 9's to settle.

## 9. OPEN

1. **The next statement in execution order is the lateral viscosity
   coefficient.**  The resolved run reads a three-dimensional field from
   `eddy_viscosity_3D.nc` (`ldfdyn.f90:348-353`); the production arm computes a
   different, formula-based coefficient from a scalar zonal spacing.  That is
   a shared-model change and will need the full GYRE identity proof again.
2. Only after that can the ladder produce a first non-bit ARITHMETIC statement
   and a kt=10 magnitude.  Neither is registered here.
3. **The other half of the fold row is unrecorded**; the acquisition is
   written and needs an operator run
   (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round8_een_rank_acquisition/run.sh`).
   Until it runs, the fold is certified in ONE direction only.
4. The independent sea-surface height still differs by up to 1.55 cm on 16,433
   of 26,640 surface cells, owned by the initial sea-ice category
   configuration, which is out of scope on this lane.
5. The recorded runoff tracer-source operands remain an explicit later
   boundary; this round did not reach them.
6. **The barotropic vertex thickness is unmeasured.** This round's helper now
   supplies it on a tripolar grid, built from the uniform reference ladder
   rather than from NEMO's own reference thickness. Nothing here claims it is
   bit-exact; measure it against the record before any such claim.
7. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Choices

ASKED: none were needed.  Every setting this round touched is fixed by the
record's resolved configuration and was read from it (section 1).
UNASKED: none.  The transcription completes an operation the card already
selects; no scheme selection, tunable, threshold, cadence, resolution,
timestep, carried state, data source or previously-tolerated condition moved,
and no stabilizer NEMO lacks was added.  The promoted helper name is a rename
with no behavioural change.

## 10. Gate and test results at the round's final tip

| check | result |
|---|---|
| EEN operand gate, including the new fold-row section | PASS; the fold row AT BAR for both operands, all three controls firing |
| ORCA2 ladder gate | exit 4, `STOP_PRODUCTION_LDF_DYN_TRIPOLAR_COEFF_GAP` at kt=1, no magnitude registered |
| receipt citation gate | PASS, 274 citations mapped, 0 failures, 0 map-audit failures, 0 unmapped |
| citation gate with a rigid two-line plant | refuses, exit 2 |
| the seven named push gates plus the new fold-row tests, at the final tip | **182 passed in 504.69 s** |
| the ocean-fidelity battery, once, with twelve workers | **3 failed, 1516 passed, 7 skipped in 2,389.81 s** |

Two of those three failures were this round's own, and they are fixed: round
7's stop tests named the refusal this round transcribed away, so they failed
against the fix they were written to guard.  The expectation was stale, not the
code.  They now exercise the stop REGISTRY instead of one hard-coded name, and
they require the retired refusal to be rejected as unregistered, so the
registry shrinking with the transcription is itself asserted; their file passes
17 of 17 and the re-run push battery above is green at the final tip.  The
third failure is the listed pre-existing sea-ice scalar-math provenance red
(`A MY_SRC is not verbatim`); no other new failing identifier appeared.

Six citations in an earlier receipt and in the citation map were rigidly
re-anchored: the transcription inserted twenty-one lines above them in the same
file, the gate reported that same uniform shift for every one, and no cited
text changed.

**The other two vertex-thickness rules are unchanged on an ACTIVE fold.**  The
control flow around the fold was restructured, and GYRE cannot catch a
regression there because its fold is inactive.  So both other rules were run on
a synthetic tripolar grid at the round's base tip and at its tip, with and
without a reference ladder: all four digests are identical
(`min` `20cb0bfc33954f7ad4b88d16`, `nemo_avg` `d5b9c3aef9be1279c5299028` and
`a054ce29531259da2a7a82ba`).  The script is kept with the round's evidence.

**Non-vacuity, measured.**  The seven new tests were run unchanged against the
round's base tip, where the helper still refuses: **4 failed, 3 passed** — the
four tripolar ones fail on the refusal itself, and the three that pass are the
non-tripolar cases, which are correctly inert.  The base clone was left clean.
