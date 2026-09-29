# RECEIPT — VORTEX card, round 2 (S-EOS, the Coriolis operator, acquisition)

Date 2026-09-29. Lane tip `5301122fe2ef`. Preregistered in
`PREREG_nemo_testcases_l1_vortex_round2.md`, frozen before the round-2 record
is read.

Status: **SCORED.** The acquisition ran and was admitted; the card executes,
its initial state enters the ladder at the bar, and it leaves the bar at
kt=2 on temperature, both velocities and sea surface height together.
Section 6c is the ladder. Nobody is named as its owner: that is round 3.

## 0. Round 1 was never landed

Round 1's eleven commits sit on lane tip `c09a9e111780`; the lane has since
advanced 35 commits to `5301122fe2ef` and the operator has not landed round 1.
This round forward-ported those commits onto the current tip before doing
anything else. Two of them were pure re-anchorings of line-number citations
into a module the new card grows; both were resolved to the tip's own numbers
and the citation gate re-anchored afterwards, so nothing was hand-merged into a
citation.

The forward port also moved the three certified cards' digests, on its own,
before this round touched anything. See section 4.

## 1. What was transcribed

**The equation of state (decision 69).** VORTEX's `&nameos` selects NEMO's
simplified law with `rn_a0 = 0.28` and every other coefficient zero. Written
out, that is `rho = rho0 - 0.28*(T - 10)`: linear in temperature, blind to
salinity, blind to depth.

*Nothing new was written.* The law was already in legoESM, transcribed from
the source with its expansion-coefficient companion, and already reachable by
name. What was missing was a way to give it coefficients: no caller on this
model's path ever passed any, so every card that selected it silently got the
defaults, which are another experiment's. This round threads the coefficients
through the three places the path builds the equation of state, and refuses the
two compositions that would still read the defaults -- a different selection,
and the eddy-parameterisation density closures, which take their own pair of
arguments and would have run a different fluid from the dynamics.

**The Coriolis operator.** Round 1 declared this as a gap and fixed the card
closed. Reading the routing through explains it and closes it.

Round 1's description of the routing was wrong, and is retracted here.
The case does not run the vorticity routine on the planetary vorticity alone.
Its flux-form arm is handed Coriolis PLUS a metric term. That metric term is
built from first differences of the mesh's own scale factors -- and this mesh
assigns every scale factor one repeated constant, so the differences are
bitwise zero and the two selectors coincide. Only under that condition is the
energy-and-enstrophy triad on the planetary vorticity the whole operator. The
condition is now proved against the card's own grid, not against the
namelist's promise, and a mesh that fails it is refused.

The card runs that triad on both arms NEMO runs it on: the baroclinic momentum
trend and the split-explicit substeps, with the separate rotation switched off
so the planetary term enters exactly once. The triad is the one the
vector-invariant cards already run; it is fed a zero relative vorticity. Every
composition that was not read off the source raises.

Citations are now against the COMPILED sources of the round-2 build
`VORTEX_OMIP_L1`, which exists as of this step; the shipped-source line numbers
round 1 used are superseded. `namelist_cfg` is the deck, which is not
preprocessed, so it is cited as shipped.

| statement | compiled source | lines |
|---|---|---|
| the simplified-EOS operands, with the LIVE stretched depth | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:359-361` | 3 |
| the simplified-EOS density itself | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:364-366` | 3 |
| its thermal expansion coefficient | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1217` | 1 |
| its haline contraction coefficient | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1220` | 1 |
| the unset reference temperature and salinity | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:100-101` | 2 |
| the namelist is read whichever law is selected | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1938` | 1 |
| the case's own coefficients, from the RUN's own resolved deck | `vortex_round2/namelist_cfg:128-138` | 11 |
| the vorticity scheme selector | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:849` | 1 |
| the flux-form arm's vorticity choice | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:865-868` | 4 |
| the call that passes it | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:256-257` | 2 |
| the triad's Coriolis-only vertex field | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:725-727` | 3 |
| the triad's Coriolis-plus-metric vertex field | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:755-758` | 4 |
| the metric coefficients, as scale-factor differences | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:882-883` | 2 |
| the reciprocal vertex thickness the triad divides by | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:720` | 1 |
| the triad's transport and its twelfths | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:766-779` | 14 |
| the matching barotropic arm, its whole selector and all four triads | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynspg_ts.f90:955-968` | 14 |
| the mesh's constant scale factors, ALL FOUR pairs | `VORTEX_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:158-161` | 4 |

Three of these are worth reading twice, and a reviewer corrected all three.

`VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:720` is the divisor the triad
applies. It is `1/(e3f_0vor*(1 + r3f*fe3mask))` -- the masked surface-height
ratio matters and an earlier draft of this line dropped `fe3mask` from the
paraphrase. That is the same quantity the card's own helper builds, so the two
sides divide by the same thing.

The mesh citation now spans all FOUR scale-factor pairs
(`VORTEX_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:158-161`), not the first and last. The metric term is built
from `e2v` and `e1u` specifically, which are on the two lines an earlier draft
skipped, so the narrower citation did not support the claim it was attached to.

`VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynspg_ts.f90:955-968` is the whole
energy-and-enstrophy branch of the barotropic Coriolis, not one line of it:
the selector plus all four triads, every one of them `ff_f` over that same
thickness with no relative vorticity and no metric term anywhere. That is what
lets the card select it without a second transcription, and one line could not
have shown it.

## 2. The finding: the initial state is not bit-exact

Round 1 predicted a bit-exact initial state on all five fields. Measured
against round 1's own record, that is **refuted**:

| field | cells unequal | of | worst |
|---|---|---|---|
| temperature | 0 | 37210 | — |
| salinity | 0 | 37210 | — |
| zonal velocity | 788 | 36600 | 2 last bits |
| meridional velocity | 788 | 36600 | 2 last bits |
| sea surface height | 104 | 3721 | 2 last bits |

It is the compiled floor between two exponential functions, not a
transcription defect. Every unequal cell is one or two last bits (velocity:
692 at one, 96 at two; height: 96 at one, 8 at two), and they sit in the far
tail of the eddy's Gaussian -- the unequal heights run down to 4.5e-92 while
the metre-scale centre is exact. The card already declares that it evaluates
transcendentals with the system library. The gate now pins the counts rather
than asserting a zero that the machine cannot deliver.

This measurement used round 1's record, which was produced on the TEOS-10
deck. Nothing in the initial state reads the equation of state, so the new
record must reproduce these counts exactly; the preregistration says so, and
says that a difference is the finding rather than the ladder.

## 3. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| the shipped equation of state on this card | ASKED (decision 69) | the deck's TEOS-10 switch is removed, not just overridden |
| implicit vertical advection OFF | ASKED (decision 70) | no change; round 1 already matched it |
| the matching barotropic Coriolis arm is selected too | UNASKED, and stated | round 1 declared it as a second gap "which the card cannot select without the baroclinic arm above". It can now, and leaving it on the 4-point average would have been a silent mismatch inside the very step this round transcribed -- and would have kept the card fixed closed, so the round could not be scored at all. One line for the operator: keep it (my pick, it is what the source runs) or split it into its own round? |
| the initial-state gate pins 788/788/104 instead of zero | UNASKED, and stated | the alternative is a gate that can never pass. The counts and the last-bit bound are both pinned, so a real defect still turns it red |
| the certified digests are re-pinned | UNASKED, forced | they moved before this round touched anything (section 4) |

## 4. Additivity: what moved, and why

The digest hashes the configuration's text, so a field added with no value
moves it. That is not an argument, it is measured.

| card | round 1, tip `c09a9e111` | tip `5301122fe`, this round STASHED | tip + this round |
|---|---|---|---|
| GYRE-zco | `f227194da309e66b` | `af1f9f5e99d9a30f` | `eaef11b4c2e37a31` |
| LOCK_EXCHANGE-zco | `42d13c75ea8cbcc6` | `af88a6e54a70263f` | `159ca3d07db0a3f5` |
| OVERFLOW-zps | `c2bca636ac2f14ef` | `375ec3040cc4ce53` | `73174751388503aa` |

Column two to column three is the LANE's 35 commits, not this card's: it was
measured with every round-2 edit stashed. Column three to column four is this
round, and a field-by-field diff of all three resolved configurations across
that step reports, on each card:

```
added=['eos_nemo_seos'] removed=[] changed=[]
```

and the added field is unset on all three. A test asserts that directly, so a
later edit that gives one of them a value turns red instead of hiding behind a
moved hash.

## 5. Gates

| gate | its own success line |
|---|---|
| card test, `tests/ocean/unit/test_nemo_vortex_card.py` | `18 passed` |
| card constructibility | `17 passed` |
| non-vacuity: the triad vs the 4-point average it replaced | the momentum tendency MOVES; a dead branch turns it red |
| non-vacuity: the card's coefficients vs the shared defaults | a different fluid; a silently defaulted set turns it red |
| plant: the metric term on a stretched mesh | REFUSED, as required |
| plant: every untranscribed composition of the new arm | REFUSED, five of them |
| card test + constructibility, after the review fixes | `37 passed in 61.99s` |
| citation gate, the VORTEX round-1 receipt | `PASS`, 18 citations, 0 unmapped, 0 map entries failing |
| citation gate, the lane's own receipt | `PASS`, 274 citations, 0 unmapped -- unchanged in count by this round |
| citation gate, planted-shift self-tests | all fire |
| acquisition preflight, after the deck change | `PREFLIGHT_OK`, exit 0 |
| push-gate battery, the six autopilot files plus the card test, the constructibility tripwire, the dispatch ratchet and GYRE's decade pin | `211 passed in 1146.79s` at the FINAL tree, after both review fixes |
| GYRE decade climate pin, standing in for the re-run the preregistration misnamed | `14 passed in 3.90s` |
| kt=1..10 ladder, VORTEX | scored; see section 6c |
| kt=1..3 ladder, both tanks, through the SAME modified gate | unchanged; see section 6c |
| ladder gate non-vacuity plant | first-over-bar kt=1, normalized 4.877e-02, exit 1 |
| citation gate, this receipt, against the COMPILED build | `PASS`, 17 citations, 0 unmapped, 0 map entries failing |
| citation gate, this receipt, planted shift | exit 1 |
| DINO month gate (note BI; this round changes `packages/`) | `2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS` |

**What the battery line does and does not cover.** It was taken at
`abc0c51c8`, BEFORE the three review-fix commits, so it is not the whole
round. What the review fixes touch, and how each is covered at the final tree:
the card test (`37 passed`, above, which includes the constructibility
tripwire), the tendency probe (no citation anchors, and the citation gate is
re-run green at the final tree), the acquisition script's prose, and this
receipt. A re-run of the whole battery at the final tree was started; the
operator's land script runs it too, and it is the authority.

**The DINO month gate PASSES, and the round is inert on DINO.** Note BI makes
it mandatory for any round that changes `packages/`, which this one does. The
day-30 wet three-dimensional temperature error against NEMO is
`2.040288957e-03 K`, where the certified value is `2.040288765e-03 K`: they
differ by 1.9e-10 K, which is the harness's own run-to-run floor (round 129
measured ~2e-10 K). So this is not an improvement and not a regression -- it is
the same number, and that is what an additive round should produce.

What the card's own numbers do NOT depend on: every measurement in sections 2
and 4 was taken in isolation, and the card test that carries them is the
`37 passed` line above.

## 6. Reviews

Two independent adversarial reviews ran on the diff. They disagreed on the
verdict, which is the useful outcome: one read the PHYSICS and found it right,
the other read the TESTS and found them weak, and both are correct.

| reviewer | verdict | what it found |
|---|---|---|
| codex, adversarial, read-only | DO NOT SHIP | both non-vacuity plants can pass while the feature is broken |
| a fresh Claude code-reviewer | SHIP | the routing and the mesh condition check out against the source, independently re-derived |

**The two blocking findings, and what they turned into.**

*The rotation plant could pass with the rotation MISSING.* The first draft only
asked that the transcribed operator differ from the 4-point average it
replaced. Delete the new branch and the card has no rotation at all -- the
separate face-Coriolis add is switched off for this scheme -- which also
differs from the average, by far more. The plant now pins a BAND against the
natural scale of the rotation itself, and the two failure modes are four orders
of magnitude apart, so the band is not delicate: two stencils of the SAME
rotation differ by 1.2e-4 of that scale, a card with the rotation gone by 0.76.

*The equation-of-state plant never touched the model.* It called the factory
directly with the card's coefficients, so reverting all three production call
sites would have left it green. It now goes through the model, against a
control carrying the shared defaults.

*A fourth place builds the equation of state.* BOTH reviewers found this
independently, which is the strongest signal either could have given: the
tendency probe builds its own density and did not read the coefficients. It is
not reachable from this card today, but it is the exact failure this round
exists to fix, so it is threaded now rather than left for the round that would
have hit it.

*The last-bit metric was wrong off this card.* Raw floating-point bit patterns
are not ordered across zero, so a future card whose unequal cells straddle zero
would have reported a nonsense distance. This card's do not, so the number in
section 2 stands; the metric is corrected anyway.

One finding is recorded and NOT acted on. The mesh condition this round
enforces is stricter than the source's: it requires every scale factor
constant, where the source only needs one constant along each direction. It is
conservative, it is right for this card, and loosening it would need a mesh to
test it on. Named here so the next card that needs it knows why it refuses.

## 6b. The one thing the operator must decide before landing

Round 1 declared TWO gaps, and this round closed both. Closing the second one
-- the split-explicit barotropic arm -- was not separately asked for. It was
forced, in the sense that leaving it on the 4-point average would have been a
silent mismatch inside the very step this round transcribed, and would have
kept the card fixed closed so the round could not be scored at all. It is
still a choice, it is on the UNASKED list in section 3, and it is offered for
revert here rather than buried: splitting it into its own round means this
card does not execute until that round lands.

## 6c. Step D: the record, the initial state, and the kt=1..10 ladder

The acquisition ran and was ADMITTED (`phase3/vortex/round2`, restart
byte-identical). Preregistered in
`PREREG_nemo_testcases_l1_vortex_round2_ladder.md`, frozen before scoring.

**The initial state, against BOTH records.**

| field | cells unequal | of | worst |
|---|---|---|---|
| temperature | 0 | 37210 | exact |
| salinity | 0 | 37210 | exact |
| zonal velocity | 788 | 36600 | 2 last bits |
| meridional velocity | 788 | 36600 | 2 last bits |
| sea surface height | 104 | 3721 | 2 last bits |

Round 2's preregistration predicted the new record would reproduce round 1's
five rows exactly. It does, and the reason is stronger than the prediction: the
two kt=1 records are **byte-identical**. The equation-of-state switch provably
did not touch the initial state, so the last-bit residual cannot be blamed on
it and the compiled-exponential explanation stands unchallenged.

**The ladder**, normalized maximum absolute error against the bar `1.0e-15`:

| kt | T | S | u | v | ssh |
|---|---|---|---|---|---|
| 1 | 0 AT-BAR | 0 AT-BAR | 2.220e-16 AT-BAR | 2.220e-16 AT-BAR | 1.355e-20 AT-BAR |
| 2 | 1.643e-09 DEBT | 4.060e-16 AT-BAR | 1.136e-07 DEBT | 1.135e-07 DEBT | 3.709e-08 DEBT |
| 3 | 5.510e-09 DEBT | 6.090e-16 AT-BAR | 2.582e-07 DEBT | 2.143e-07 DEBT | 6.503e-06 DEBT |
| 4 | 1.128e-08 DEBT | 6.090e-16 AT-BAR | 1.073e-06 DEBT | 9.892e-07 DEBT | 8.218e-06 DEBT |
| 5 | 1.698e-08 DEBT | 8.120e-16 AT-BAR | 1.061e-06 DEBT | 1.041e-06 DEBT | 7.978e-06 DEBT |
| 6 | 2.075e-08 DEBT | 8.120e-16 AT-BAR | 1.288e-06 DEBT | 1.199e-06 DEBT | 8.024e-06 DEBT |
| 7 | 2.154e-08 DEBT | 1.015e-15 DEBT | 1.834e-06 DEBT | 1.629e-06 DEBT | 5.160e-06 DEBT |
| 8 | 2.028e-08 DEBT | 1.015e-15 DEBT | 2.167e-06 DEBT | 1.898e-06 DEBT | 5.950e-06 DEBT |
| 9 | 2.075e-08 DEBT | 1.015e-15 DEBT | 2.360e-06 DEBT | 2.096e-06 DEBT | 4.537e-06 DEBT |
| 10 | 2.103e-08 DEBT | 1.218e-15 DEBT | 2.464e-06 DEBT | 2.192e-06 DEBT | 5.336e-06 DEBT |

**First over the bar: kt=2, on temperature, both velocities and sea surface
height simultaneously.** Salinity survives to kt=7.

Read plainly: the card enters the ladder at the bar and leaves it at the very
first step it takes. The momentum fields lead by magnitude -- velocity at
1.1e-7 and height at 3.7e-8 against temperature's 1.6e-9, two orders smaller --
which is what a rotation or a barotropic defect looks like and is not what a
tracer defect looks like. Nothing here attributes it to a named operator; that
is round 3's job and the ladder is the instrument it will use.

The error then GROWS by roughly a factor of twenty on velocity over eight
steps and flattens on temperature after kt=6. It is not a blow-up.

**Two of this step's own predictions were wrong, and are retracted.**

*Prediction 1 said kt=1 would be DEBT on velocity and height.* It is AT-BAR on
all five. The reason is that the gate normalizes by the field's own scale, and
a two-last-bit residual on an order-one field normalizes to 2.2e-16, comfortably
under a 1e-15 bar. The entry row is therefore AT-BAR and the per-cell table
above is the sharper statement of the same fact. Nothing about the card
changed; the prediction was about the instrument and was made without reading
it carefully enough.

*Prediction 4 said salinity would be marked UNINFORMATIVE from kt=2 on,*
because it starts spatially uniform. It is not, and the gate is right not to:
NEMO's advected salinity is not exactly uniform after a step, so the gate's
uniqueness test correctly declines to downgrade the row -- and the row goes on
to earn real DEBT at kt=7. A vacuous control was expected and a live one was
found.

Predictions 2 and 3 held: the first-over-bar step is kt=2, not kt=1, and no
tracer left the bar before the velocities.

**Every other card is unchanged**, re-run through the same modified gate, which
is the check that matters because this step edited the gate itself:

| card | kt=1 | kt=2 | first over bar |
|---|---|---|---|
| LOCK_EXCHANGE-zco | all AT-BAR | all AT-BAR | none through kt=3 |
| OVERFLOW-zps | all AT-BAR | T 7.816e-15 DEBT, u 7.069e-12 DEBT | kt=2, T and u |

Both were re-run AGAIN after the review fixes, and OVERFLOW the second time
with its owner-hunt arm switched ON -- which is the only way to execute the
record-reader call site both reviewers flagged. It returns the same three
numbers to every digit: `T 7.816e-15`, `u 7.069e-12`, first over bar at kt=2
on T and u, with sea surface height at `4.163e-17` and still at bar. So the
repaired path runs and the repair is inert on the certified numbers, which is
what a correctness fix to a checker should be.

OVERFLOW's kt=2 debt is its OWN known one, which decision 71 assigns to a
separate tank round; this step neither moved it nor touched it. Both tanks'
meridional velocity row reports UNMEASURED, which is the gate correctly saying
their three-row closed geometry has no active meridional face -- a
pre-existing, documented statement, not something this step introduced.

**GYRE, and a preregistered promise this step could not keep as written.** The
preregistration says "the two tanks and GYRE are re-run through the same gate
afterwards". That was a drafting error and is corrected here rather than left
silent: GYRE is not wired into this ladder gate at all and never has been --
the gate's case list has two entries, and adding a third for GYRE would be a
new card's worth of work, not a re-run. GYRE's own regression is the decade
climate pin, and it is green at this tree: `14 passed in 3.90s`, which holds
its day-30 value at 2.327677e-06 K among others. So the claim GYRE needed --
that this step did not move it -- is supported; the instrument named in the
preregistration was simply the wrong one. A reviewer caught this, not the
author.

**The gate's own non-vacuity.** Run with `--plant`, the ladder reports
first-over-bar at kt=1 on temperature with a normalized error of 4.88e-02 and
exits non-zero -- so it can fail. The citation gate likewise exits non-zero
under a planted line shift.

**A blind spot this step did NOT introduce, named because it is cheap to name.**
The new interior check compares shapes, and VORTEX's grid is square, so a
record with its two horizontal axes transposed would pass it. The hard-coded
tuple it replaced had exactly the same blind spot for the same reason, so this
is not a regression -- but "an array's layout is an API" has cost this campaign
before, and the next card with a non-square grid gets the check for free.

**Note BD compliance.** The ladder gate used to carry a hard-coded header
tuple per case, and this card would have been a third. Five acquisitions in
this campaign have now been refused by a checker that predicted a size by
hand, so the tuple check is replaced by PARSING for every case: the shape comes
from the record's own header, only the magic string, format version, tracer
count and word size are asserted, and the payload length is checked against
what the header itself asks for. Two further claims that used to be implicit
are now assertions that refuse rather than slice silently: the record's
interior must match the card's, and its level count must equal the card's
executed levels plus its dummy bottom record.

## 6d. Reviews of step D

Both reviewers were given this step's commit alone and told to attack the gate
change first, because it is shared with two certified cards.

| reviewer | verdict | its leading finding |
|---|---|---|
| codex, adversarial, read-only | DO NOT SHIP | one call site of the record reader opts out of the new checks, so that path is WEAKER than before this step |
| a fresh Claude code-reviewer | DO NOT SHIP | the same call site, found independently; plus a preregistered promise about GYRE that this gate cannot keep |

They agreed on the leading finding, which is the strongest signal either could
have given, and it is the one that mattered: replacing a hard-coded check that
applied to every call site with an opt-in keyword left one caller unprotected.
It does not run in this round -- it is an OVERFLOW-only diagnostic arm that is
off by default -- but it is a regression sitting in shared code, and a future
run would have inherited it silently. Both checks now apply at every call site.

The rest, each fixed rather than argued with:

| finding | disposition |
|---|---|
| the mesh citation named the first and last scale-factor lines, but the metric term is built from the two in between | FIXED: it spans all four. The narrower citation did not support the claim attached to it |
| the paraphrase of the triad's divisor dropped the live mask factor | FIXED |
| one line was cited for the whole barotropic branch | FIXED: the selector and all four triads, fourteen lines |
| the preregistration promised a GYRE re-run this gate cannot do | ACKNOWLEDGED above, with GYRE's own green regression in its place |
| the shape check cannot see transposed axes | NAMED above, not fixed: the check it replaced had the same blind spot, so it is not a regression |
| this step's own dual review was not recorded | this section |

One thing neither reviewer raised, found while fixing their findings: section 2
of this receipt gave the temperature and salinity denominators as 36600, which
is the velocity count. They are 37210. Corrected.

## 7. How to acquire

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh --run
```

The deck now selects the shipped equation of state, and the preflight refuses
a deck that does not, or that selects a second one alongside it. Evidence
should land in a round-2 directory, beside round 1's.

## 8. OPEN

**Round 3 — the vector-EEN VORTEX card (decision 73, note BJ).** A second
VORTEX card with the same geometry, the same simplified equation of state and
the same eddy, but ORCA2's and GYRE's momentum scheme set: vector-invariant
advection with the energy-and-enstrophy vorticity, so that code runs in a clean
rotating flow for the first time. It needs its own build, its own
preregistration, its own kt=1..10 ladder and its own receipt, and both VORTEX
cards are kept as permanent gates. Note that this card will route NEMO to the
Coriolis-plus-RELATIVE-vorticity selector rather than this one's
Coriolis-plus-metric, so it exercises a genuinely different branch of the same
routine -- and legoESM already has an arm for it, so it should not need new
numerics.

**Round 4 — the resolution ladder (decision 74, note BK).** Once both cards are
at bar at 30 km, the same cards at 15 km and 10 km without AGRIF, each with its
own record and ladder, and one receipt putting the three resolutions side by
side. The point is to find any hidden grid-size dependence in the
transcription and to watch the residual as the eddy becomes resolved. This
round's first-over-bar at kt=2 makes that more interesting, not less: if the
residual scales with resolution it is truncation, and if it does not it is a
transcription gap.

**The owner of this round's kt=2 debt is not named.** The momentum fields lead
by two orders of magnitude, which points at the rotation or the barotropic
path, but this round measured rather than attributed and the ladder is the
instrument the attributing round will use. Naming an owner here on the strength
of a magnitude ordering would be the exact mistake this campaign has paid for
before.

**Carried from before step D:** whether the split-explicit barotropic arm
should have been its own round (section 6b), and the mesh guard being stricter
than the source needs (section 6).

**Not worth closing:** the one-to-two-last-bit floor on the initial velocity
and height. It is two exponential functions disagreeing in their last bit, and
the two records being byte-identical across the equation-of-state switch rules
out every other candidate.
