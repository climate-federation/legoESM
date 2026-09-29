# RECEIPT — VORTEX card, round 2 (S-EOS, the Coriolis operator, acquisition)

Date 2026-09-29. Lane tip `5301122fe2ef`. Preregistered in
`PREREG_nemo_testcases_l1_vortex_round2.md`, frozen before the round-2 record
is read.

Status: **ACQUISITION_NEEDED.** The card now executes; the record it must be
scored against does not exist yet, because round 1's record was produced on a
deck that selected a different equation of state.

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

| statement | source | lines |
|---|---|---|
| simplified-EOS density | `eosbn2.F90` | 295-302 |
| its expansion coefficients | `eosbn2.F90` | 1161-1173 |
| unset reference temperature and salinity | `eosbn2.F90` | 89-90 |
| the namelist is read whichever law is selected | `eosbn2.F90` | 1890-1895 |
| the case's own coefficients | `namelist_cfg` | 130-138 |
| the vorticity scheme selector | `dynvor.F90` | 874 |
| the flux-form arm's vorticity choice | `dynvor.F90` | 891-893 |
| the triad's Coriolis-only vertex field | `dynvor.F90` | 750-752 |
| the triad's Coriolis-plus-metric vertex field | `dynvor.F90` | 780-783 |
| the metric coefficients, as scale-factor differences | `dynvor.F90` | 905-908 |
| the triad's transport weighting | `dynvor.F90` | 791-792, 804-806 |
| the matching barotropic arm | `dynspg_ts.F90` | 1326-1345 |
| the mesh's constant scale factors | `usrdef_hgr.F90` | 160-163 |

Citations are against the SHIPPED sources. The round-1 build's preprocessed
sources exist but were produced for a deck this round changes; the round-2
build's will be cited when it exists.

## 2. The finding: the initial state is not bit-exact

Round 1 predicted a bit-exact initial state on all five fields. Measured
against round 1's own record, that is **refuted**:

| field | cells unequal | of | worst |
|---|---|---|---|
| temperature | 0 | 36600 | — |
| salinity | 0 | 36600 | — |
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
| push-gate battery | PENDING |
| citation gate | PENDING |
| DINO month gate | PENDING |

## 6. Reviews

PENDING.

## 7. How to acquire

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh --run
```

The deck now selects the shipped equation of state, and the preflight refuses
a deck that does not, or that selects a second one alongside it. Evidence
should land in a round-2 directory, beside round 1's.

## 8. OPEN for round 3

* The kt=1..10 ladder, which cannot be scored until the record exists.
* Whether the barotropic arm should have been its own round (section 3).
* Whether the last-bit floor on velocity and height is worth closing at all;
  it is a property of two exponential functions, so probably not.
