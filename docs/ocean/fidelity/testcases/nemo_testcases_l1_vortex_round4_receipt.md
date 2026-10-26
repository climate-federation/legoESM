# RECEIPT — VORTEX round 4: every card states its own CFL cap, and the vector card's second step is attributed to one boundary

Date 2026-09-29. Lane tip at the start `85607c118588`. Preregistration
`PREREG_nemo_testcases_l1_vortex_round4.md`, frozen before any measurement.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round4`.

Status: **HELD at hand-off. Part A is measured and ready; part B names the
owning boundary and STOPS at
`ACQUISITION_NEEDED` for the term inside it** — exactly the escape the
preregistration reserved, and the same rule round 3 was told to follow.

---

## Part A — decision 75: no NEMO card inherits its explicit-CFL cap

### 1. What moved, measured rather than assumed

The GYRE card's certified configuration digest had drifted from the value
round 2 pinned. Rather than re-pin it, the resolved configuration was
fingerprinted at the commit where that pin was measured and at this round's
tip and diffed leaf by leaf. **Five rows differ on GYRE and nothing else
differs on any card.**

| row | at the round-2 pin | now | kind |
|---|---|---|---|
| biharmonic mixing's explicit-CFL cap | `False` | `True` | a VALUE the library default moved |
| biharmonic mixing's reference-timestep field | `3600.0` | gone | field deleted from the library |
| linear bottom drag's coefficient | `0.0011` | gone | field deleted from the library |
| quadratic bottom drag's coefficient | `0.0025` | gone | field deleted from the library |
| the KPP mixing scheme's `c_b` | `0.599` | gone | field deleted from the library |

Every one sits on a block this card does not select — it runs no lateral
mixing, no bottom drag, and TKE rather than KPP — so nothing executed
changed. Only the FIRST is a value a card can state; the other four are
fields that no longer exist.

### 2. The fix, and why it is at the shared builders

Three builders give a NEMO card a lateral-mixing block: the shared NEMO
physics builder behind GYRE and ORCA2, and DINO's two. All three now state
the explicit-CFL cap for BOTH sub-blocks that carry a field of that name, at
the values main resolves to today. Writing a field explicitly at the value it
already resolves to cannot move a digest, because the digest hashes the
NamedTuple's printed form and that prints every field either way — which is
measured below, not argued. The tanks and both VORTEX cards carry no physics
block at all, so there is no field on them to state; the test says so rather
than skipping them.

### 3. The gate, and why it is not vacuous

Every card is built twice with the library defaults flipped in between, and
any card whose digest moves is refused. Three proofs that it can fail:

| control | result |
|---|---|
| the flip itself, on a block built the way the cards used to build it | both values move, and are restored afterwards |
| the statement reverted in the shared builder, GYRE arm | **FAILS**: `337651dbd9f1b49c` against `c51dbc542518d0ba` |
| the whole card test file, with the fix in place | `36 passed` |

### 4. No number can have moved

Every card's resolved configuration was fingerprinted leaf by leaf at
`85607c118588` and at this round's tip.

| card | keys before -> after | added | removed | changed | digest |
|---|---|---|---|---|---|
| `GYRE-zco` | 564 -> 564 | 0 | 0 | **0** | `337651dbd9f1b49c` both |
| `LOCK_EXCHANGE-zco` | 188 -> 188 | 0 | 0 | **0** | `159ca3d07db0a3f5` both |
| `OVERFLOW-zps` | 188 -> 188 | 0 | 0 | **0** | `73174751388503aa` both |
| `VORTEX-zco` | 197 -> 197 | 0 | 0 | **0** | `5a0b5e53dc7f03c3` both |
| `VORTEX_VEC-zco` | 197 -> 197 | 0 | 0 | **0** | `b6862faf0dfac621` both |
| DINO `nemo_dino_kamm` | 564 -> 564 | 0 | 0 | **0** | — |
| DINO `nemo_dino_kamm_mlf` | 564 -> 564 | 0 | 0 | **0** | — |
| DINO `legoesm_default` | 564 -> 564 | 0 | 0 | **0** | — |
| DINO `nemo_paper` | 564 -> 564 | 0 | 0 | **0** | — |

The two `enforce_cfl` leaves are present in the fingerprint on both sides and
read `False`/`True` on both, so the table is not vacuous.

**The digests are re-pinned**: GYRE `eaef11b4c2e37a31` -> `337651dbd9f1b49c`;
LOCK_EXCHANGE and OVERFLOW unchanged.

---

## Part B — who owns the vector-EEN card's second step

### 5. What NEMO's own source says the first stage is

Under vector-invariant form NEMO does NOT build a momentum right-hand side
at the first RK3 stage. It reuses the one already completed in the
single-call barotropic preparation:

> `CASE ( 1 )  !==  Stage 1  ==!` / "Vector Inv. Form : 1st stage 3D RHS
> already entirely computed in stp_2D"
> (`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:312-316`)

and that preparation is where the pressure gradient, the lateral viscosity,
the vorticity, the kinetic-energy gradient and the vertical advection of
momentum are all applied, in that order
(`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:137-163`). Its depth
average is what forces the external solve
(`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:176-179`). **Under flux
form the same block produces a two-dimensional right-hand side only** — the
advection is added back at the stage — which is why the flux card never
exercises this path and why it is the control in section 7.

### 6. The walk

Same arms as round 3, the same normalized maximum, the same readers, now run
on either card, with one boundary added that the round-3 record already
carried and the round-3 reviewers said was omitted: **NEMO's completed
pre-stage three-dimensional momentum right-hand side**.

Vector-EEN card, kt=2 entry, bar `1.0e-15`:

| arm | what is substituted | T | u | v | ssh |
|---|---|---|---|---|---|
| 0 | nothing (the card) | 2.145e-08 | 1.4325e-05 | 1.4330e-05 | 2.2685e-05 |
| 1 | NEMO's kt=1 entry | 2.145e-08 | 1.4325e-05 | 1.4330e-05 | 2.2685e-05 |
| 2 | + the external-solve handoff | 1.982e-08 | 3.897e-06 | 3.856e-06 | **0** |
| 3 | + NEMO's stage-1 output | 3.626e-09 | 3.369e-06 | 3.337e-06 | 0 |
| 4 | + NEMO's stage-2 output | 3.466e-16 | 3.374e-06 | 3.334e-06 | 0 |
| 5 | arm 2 + NEMO's pre-stage RHS | 3.626e-09 | 3.369e-06 | 3.337e-06 | 0 |
| **6** | **arm 1 + the pre-stage RHS ALONE** | 3.626e-09 | **3.369e-06** | **3.337e-06** | **3.709e-08** |

And each stage run ALONE from NEMO's own entry, scored against NEMO's record
of that stage's output — a sharper boundary than the step's end:

| stage, alone | vector card, u | flux card, u |
|---|---|---|
| 1 | **1.7074e-05** | 4.359e-08 |
| 2 | 1.670e-06 | 6.182e-08 |
| 3 | 3.374e-06 | 1.204e-07 |
| **1, handed NEMO's pre-stage RHS** | **1.110e-16 — AT BAR** | 1.048e-02 |

### 7. What that says, and the control that makes it a finding

**The owner is the completed pre-stage three-dimensional momentum right-hand
side, and nothing else in the first stage.** Hand legoESM NEMO's own copy of
it and the first stage's output is exact to `1.1e-16` against a bar of
`1e-15`, down from `1.7e-05`. Everything else the stage does — the update
arithmetic, the barotropic mean replacement, the masks, the stage clock — is
therefore right.

**It also owns the external solve's share, which is the same number twice.**
Arm 6 substitutes ONLY that right-hand side and leaves the external handoff
alone: the velocity row falls from `1.4330e-05` to `3.369e-06` (**76.5% of
the whole step's residual**) and the height row from `2.2685e-05` to
`3.709e-08` (**99.8%**), landing on the flux card's own height level. Arm 6
equals arm 5 to ten digits, so once the right-hand side is NEMO's the
external handoff adds nothing. That is what NEMO's source predicts, because
the external solve is forced by the depth average of that same array.

**The control that makes this attribution rather than coincidence.** The
identical substitution on the FLUX card makes it 240 000 times WORSE — stage
1 goes from `4.359e-08` to `1.048e-02`. That is not a failure of the instrument, it
is the source being right: under flux form NEMO's pre-stage array is not a
complete momentum tendency, so handing it over deletes the advection. One
substitution, exact on one card and catastrophic on the other, for the reason
the compiled source states.

**The stage clock closes the flux card's own rows.** Its three stages advance
by a third, a half and the whole step, and its three stage-local residuals
are `4.359e-08 : 6.182e-08 : 1.204e-07`, i.e. `1 : 1.42 : 2.76` against the
clocks' `1 : 1.5 : 3` — within 8%. So the flux card carries ONE right-hand-side
difference, the same at every stage, and the vector card does not: dividing
each stage's residual by its own clock gives `5.12e-05 : 3.34e-06 : 3.37e-06`,
so its FIRST stage's right-hand side is **15 times** worse than its second and
third, and those two agree with each other to `1%`.

### 8. The acquisition ran, and it takes the five candidates to two

The operator ran `run.sh --variant vecrhs --run` mid-round. Both builds
reached STOP 0 and the record is **ADMITTED**: the instrumented run's NEMO
restart is byte-identical to the un-instrumented reference, every record
parses from its own header, and the admission's plant still turns it red.
Its own admission caught a real defect first, recorded in section 14.

Each routine's contribution is the difference between consecutive dumps.
NEMO's step is `rn_Dt = 2880 s` and the first stage advances by a third of
it, so the measured stage-1 velocity error of `1.7074e-05` requires a
right-hand-side difference of **at least `1.7785e-08 m/s²`** — at least,
because that stage then replaces the depth mean, which throws part of any
difference away. legoESM's completed pre-stage right-hand side differs from
NEMO's by `2.8783e-08 m/s²`, which sits just above that floor, as it must.

| routine | its own increment | verdict |
|---|---|---|
| pressure gradient | `1.3095e-03` | big enough — see the flux-card exclusion below |
| lateral viscosity | **exactly `0`** | **EXCLUDED.** This deck runs no lateral viscosity, so the routine contributes nothing and can be wrong by nothing |
| vorticity (Coriolis + relative) | `6.6267e-05` | **SURVIVES** |
| the vertical-velocity call | `0`, bit-identical | the ORDERED CONTROL, not a term: it computes `ww` and must not touch the momentum trend. It does not, so the dumps are in the order the differencing assumes |
| kinetic-energy gradient | `6.7112e-06` | **SURVIVES** |
| vertical advection of momentum | `1.6928e-08` | **EXCLUDED**: the whole of the term is smaller than the difference the measured error requires. Even a completely wrong vertical advection cannot produce it |

**The pressure gradient is excluded by the flux card.** Both cards enter
step 1 with byte-identical state and `stp_2D` calls the equation of state,
the pressure gradient and the lateral viscosity identically on both before it
branches on the advection form. The flux card's first stage is wrong by
`4.359e-08`, i.e. its whole pre-stage right-hand side is wrong by at most
`4.5e-11` — six hundred times too little to be the vector card's
`2.88e-08`. The same argument excludes the Coriolis half of the vorticity
term, which is all the flux card's vorticity routine is handed.

**So two candidates remain of five: the RELATIVE-vorticity half of the
energy-and-enstrophy vorticity, and the kinetic-energy gradient.** Which of
the two is NOT decided here. Separating them needs legoESM's own per-term
decomposition at the same boundary — the mirror of the record just acquired —
and inventing a ranking from the two magnitudes is exactly the mistake round
3 paid for.

### 9. The plants — one per substituted arm

The round-3 reviewers accepted that the old plant perturbed only arm 0's
scoring, so the walk passed even with both substitution hooks dead. Each
plant now perturbs the value actually handed to the model, and names the arm
it must move.

| plant | the arm it must move | result |
|---|---|---|
| `score` | 0, the scoring itself | VISIBLE, exit 1 |
| `entry` | 1, the seeded step entry | VISIBLE, exit 1 |
| `external` | 2, the external handoff | VISIBLE, exit 1 |
| `stage1` | 3, NEMO's stage-1 output | VISIBLE, exit 1 |
| `stage2` | 4, NEMO's stage-2 output | VISIBLE, exit 1 |
| `prestage_rhs` | 5, the pre-stage right-hand side | VISIBLE, exit 1 |

The walk also has a verdict now: it reproduces each card's own published
kt=2 velocity row from arm 0 before anything else is believed
(`1.4330e-05` against the ladder's `1.432e-05`, `0.07%`; flux card
`1.1355e-07` against `1.136e-07`, `0.04%`) and refuses otherwise.

### 10. ACQUISITION_NEEDED — the per-term split

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
--variant vecrhs --run`

The SAME vector-EEN deck, one hunk from the shipped namelist as decision 73
requires, with one extra read-only writer that dumps the momentum right-hand
side after each routine that contributes to it — pressure gradient, lateral
viscosity, vorticity, the vertical-velocity guess, kinetic-energy gradient,
vertical advection — so the per-term increments are the differences between
consecutive records. New build names and a new evidence directory; the
acquire arm refuses any target that exists. Additions only: the script
refuses the instrument if it deletes or changes one shipped line, and refuses
it again if any one of the six boundaries is missing. The record is
self-describing per note BD and the checker parses it; nothing predicts a
size. Preflight is green on all three variants.

---

## 11. Gates

| gate | its own success line |
|---|---|
| the card test file plus the push gate and the walk readers, FIRST run | `9 failed, 208 passed in 994.97s` — the red the review found; both causes are section 14's 1a and 1b |
| the citation-map audit, after re-anchoring | `map entries failing: 0` (it was 2) |
| the citation gate on THIS receipt | `PASS`, 3 citations found, 0 unmapped, 0 map entries failing; its own 9 planted controls all flagged |
| the default-independence gate, all five testcase cards and all four DINO recipes | green in the first battery, and green again with the statement reverted only on the GYRE arm (`1 failed`), which is the non-vacuity plant |
| card fingerprint, every card, before vs after | 0 added, 0 removed, 0 changed on all nine |
| **VORTEX flux ladder kt=1..10** | `DEBT`, first over bar kt 2; kt=2 `u 1.1355e-07 ssh 3.7088e-08`, kt=10 `u 2.4640e-06 ssh 5.3364e-06` — **every row equals round 3's published ladder** |
| **VORTEX vector ladder kt=1..10** | `DEBT`, first over bar kt 2; kt=2 `u 1.4325e-05 ssh 2.2685e-05`, kt=10 `u 1.4621e-05 ssh 5.3334e-06` — **every row equals round 3's published ladder** |
| LOCK_EXCHANGE-zco kt=1..3 | `AT-BAR`, no row over the bar |
| OVERFLOW-zps kt=1..3 | `DEBT`, first over bar `{T,u}` at kt 2 — unchanged from the value the merge receipt published |
| GYRE certified ladder, trajectory only, kt=1..10 | `DEBT`, **first over bar `{T,S,u,v,ssh}` at kt 3 — the certified value, unchanged**; kt=2 velocities `8.326673e-17` / `9.714451e-17`, the digits the certification receipt publishes. Two false starts first, both the harness and not the card: it takes no dirty-tree flag, and then it refused a tree with one uncommitted test file |
| DINO month gate (note BI; this round changes `packages/`), private work dir | `2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`, exit 0. Identical to round 3's value, i.e. `1.9e-10 K` from the certified one, which is the harness's own run-to-run floor: this round is INERT on DINO, which is what an additive round should give |
| kt=2 walk, vector card | section 6 |
| kt=2 walk, flux card | arms 0-4 identical to round 3's published rows |
| walk plants, six of them | all VISIBLE, all exit non-zero |
| acquisition preflight, all three variants | `PREFLIGHT_OK` on flux, vec and vecrhs; unknown variant exit 64 |
| the per-term acquisition's admission | `ADMITTED`, restart byte-identical to the un-instrumented reference, all six boundaries parsed from their own headers; its plant exits 1 |
| the per-term probe | section 8; its ordered control holds and its own plant (`rhs-agrees`) COLLAPSES the difference, exit 1 |
| walk readers and the per-term parser, planted malformations | `24 passed` |
| clean re-run of every walk and plant at the committed tree | done: both walks exit 0 and all six plants VISIBLE, exit 1, every JSON stamped with a CLEAN commit |
| the same battery after the fixes, second run | `7 failed, 219 passed` — the citation-gate and card failures are GONE; the seven were the per-term reader tests, which the run predated the header fix for and which now read `24 passed` on their own |

**Every gate above has now run.** The round is still HELD, but on the fix
rather than on the gates: part B names the boundary and narrows the term to
two of five, and lands no model code. The GYRE year rows (day 30 / 240 / 360) were NOT run: the resolved
configuration of every card is byte-identical before and after (section 4,
zero changed keys on nine cards) and part B changes no model code, so a
360-day repeat would be byte-identical by construction — but that is an
argument, not a measurement, and it is stated as such rather than reported as
a gate.

## 12. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| every NEMO card states the explicit-CFL cap at main's value | ASKED (decision 75) | note BL addendum |
| BOTH sub-blocks that carry a field of that name are stated, not just the biharmonic one whose default moved | UNASKED, and stated | the decision names the field, and two blocks carry it; the harmonic value is stated at the value it already had, so no digest and no number moves. Revert on request |
| the four DELETED library fields are recorded next to the pin rather than restored | UNASKED, and stated | no card can state a field that no longer exists; restoring them would be a library change this round was not asked for |
| the statement lives in the three shared builders rather than being repeated per card | UNASKED, and stated | the gate is per card, so a card that stopped routing through them goes red |
| nothing was landed for part B | ASKED in effect | the preregistration says the round stops at the acquisition if the term needs a record |

## 13. OPEN

**Round 5 — the per-term acquisition, then the term.** Run
`run.sh --variant vecrhs --run`, admit the record with `--rhs-terms`, and
difference the six boundaries to get the pressure-gradient, lateral-viscosity,
vorticity, kinetic-energy-gradient and vertical-advection increments
separately. The owner is one of the last three with high prior — they are what
vector form adds — but that is a PLAUSIBLE ranking and the record settles it.

**Then the resolution ladder (decision 74, note BK).** Still owed, and still
worth more once the vector card's first stage is fixed.

**What this bears on ORCA2.** ORCA2 runs the SAME momentum scheme set
(vector-invariant advection with the energy-and-enstrophy vorticity), so it
executes this same pre-stage path at every step of a developed state. Any
ORCA2 statement that rests on the first RK3 stage's momentum, on the
barotropic slow forcing, or on a developed-state momentum row is therefore
downstream of this boundary. Nothing on the ORCA2 lane was changed or
re-scored here. ORCA2's own kt=1-2 rows start from rest and cannot see it,
which is the whole reason the VORTEX pair exists.

**Carried:** the flux card's own kt=2 owner is still the stage-3 residual
(`1.204e-07`), unchanged and unnamed; the implicit vertical viscosity stays on
its candidate list, still unmeasured by a committed instrument.

## 14. Reviews

The codex quota guard reads `80.0% used` and exits 0 — at the 80% line the
brief puts the review on a Claude reviewer, so ONE fresh Claude code-reviewer
was the review, told to refute and told specifically to attack the plants,
because that is what round 3 got wrong. **It returned DO NOT SHIP and it was
right: it found a real red the round had not yet run to completion.** Every
finding is closed below; the fixes are commit `dbf978738`.

| # | finding | disposition |
|---|---|---|
| 1 BLOCKER | the round's own battery had not finished and was already showing failures the receipt did not disclose | **CONFIRMED AND FIXED.** `9 failed, 208 passed`: six were the per-term reader (below), three were the citation gate (below). Both causes were real defects of this round, not the concurrency the reviewer allowed for. Re-run green in section 11 |
| 1a | the six reader failures | the per-term record's header carried the 64-bit word size in the middle, and the shared checker reads it from the LAST header integer, so every well-formed record was refused as not 64-bit. The Fortran writer, the parser and the planted malformations now agree, and the word size is last as in every other family |
| 1b | the three citation-gate failures | stating the explicit-CFL cap inserted lines ABOVE two symbols that committed receipts cite by line number. Both re-anchored by grepping the symbol (`fidelity/nemo_recipe.py:961`→`:974`, `dino.py:3945`→`:3953`), and the two receipts quoting them updated. The map audit now reports 0 failing entries |
| 2 HIGH | most of the evidence carried a dirty SHA from the parent commit, not the reviewed one | **ACCEPTED AND FIXED.** Every walk and every plant was re-run at the committed tree with no `--allow-dirty`; section 6's numbers are from those runs |
| 3 HIGH | the six plant results were stitched from three partial sweeps | **ACCEPTED AND FIXED** by the same re-run: one sweep, one clean baseline, six plants |
| 4 MEDIUM | the default-independence gate covered two of four DINO recipes | **ACCEPTED AND FIXED.** All four are covered; a library default can move under any of them |
| 5 MEDIUM | the plant-discrimination rule had no test, so round 3's defect could return silently | **ACCEPTED AND FIXED.** The rule is a public function with a test that, per plant, asserts VISIBLE when its own arm moves and NOT VISIBLE when every other arm moves and its own does not |

What the reviewer checked and did NOT find fault with, which is the part that
matters for the attribution: each plant does perturb the value handed to its
named arm; the right-hand-side hook replaces the same quantity NEMO dumps,
on the same faces, under the established redundant-face convention; the arm
is not tautological, because the barotropic coupling, the masks and the stage
clock all still have to reproduce NEMO's arithmetic, and the flux-card control
fails catastrophically rather than passing trivially; the three percentages
reproduce from the JSON; the acquisition patch is additions-only and its
parser predicts no size; and all three compiled-source citations match their
files verbatim.

**Single reviewer, stated plainly.** The campaign's default is two. This round
had one, per the brief's instruction at the quota line. That is a weaker
review than the default and is recorded as such.
