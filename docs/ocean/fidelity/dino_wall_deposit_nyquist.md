# The barotropic per-step deposit at the end-wall rows: what the projection can and cannot see

**Outcome: no term is named, and the reason is a clean split.** The barotropic
loop's end-wall residual separates into a large part that is the *same field at
every state* and a small part that varies. The two-step operator annihilates
the constant part exactly, so this measurement can only see about 4.5 % of the
wall residual — and on that 4.5 % it returns a null. The other 95 % is the
loop's own fixed bias, and no per-step deposit projected onto the two-step mode
can adjudicate it, however many states it is given.

Artifact: `results/dino_1455/baro_deposit_wall_nyquist.json`.
Probe: `scripts/validate/ocean_fidelity/dino_1226/baro_deposit_wall_nyquist.py`.

Three rounds of adversarial review, five reviews, all five NO-SHIP. Every
headline this work produced before this one was wrong. All are retracted below.

## The split, which is the result

At the zonal end walls, decomposing each residual into its state-constant field
and what varies around it:

| | total residual | constant part | **varying part** | what the projector sees |
|---|---|---|---|---|
| zonal (u) | 1.67e-06 m/s | 1.66e-06 (99.90 %) | **7.29e-08 (4.4 %)** | recovers 87 % of the varying part |
| meridional (v) | 6.63e-07 m/s | 6.62e-07 (99.89 %) | **3.14e-08 (4.7 %)** | recovers 74 % |

The share is a ratio of RMS amplitudes, so 99.9 % constant leaves 4.4 % — not
0.1 % — genuinely varying. Substituting NEMO's own frozen slow forcing into
legoESM's loop (same loop, same bit-identical entry state, same weights)
removes those two parts very differently:

* the **varying** part: removed **~300×**. It is the forcing's.
* the **constant** part: removed only **3.1×** and **4.2×**. It is the loop's,
  and it survives at 5.3e-07 and 1.6e-07 m/s — **32 %** and **24 %** of the
  total wall residual — with a cross-state correlation of 1.00000.

So the honest attribution is scoped, not absent: **the slow forcing owns the
~4.5 % of the wall residual that this instrument can see; the loop owns the
~95 % that it cannot.**

## On the part it can see, the answer is a null

The comparable is **legoESM's wall amplitude over NEMO's** — the statistic the
measured excess (2.7×, CI [1.26, 5.10]) actually is, with no interior
denominator.

| component | row | lego/NEMO at the wall | its own sub-windows | operands clear their floor? |
|---|---|---|---|---|
| v (wall-normal) | j=1 (69.5 S) | **2.05** | 2.10, 2.51, 1.59 | **no** — NEMO's side is 1.47× its floor |
| v (wall-normal) | j=196 (69.5 N) | 0.94 | 1.07, 0.53, 1.21 | yes |
| u (tangential) | j=1 | 0.83 | 0.63, 0.90, 1.00 | **no** — both sides inside 1.4× |
| u (tangential) | j=197 | 1.02 | 1.24, 0.54, 1.26 | yes |

Two rows sit below one, one sits at one, and **one row is genuinely above** —
the southern wall in the wall-normal component, 2.05, and its three sub-windows
all stay above 1.5. The registered geometry criterion — the measured excess is
a *both*-end-walls object — **FAILS** in both components.

The blocker on that one row is its floor, not a trend: NEMO's own amplitude
there is only 1.47× the amount of slow-field curvature that leaks through the
two-step operator, so up to half of the denominator could be leakage rather
than a mode. That is why it is reported and not claimed.

The wall rows are also amplitude-*depleted*. Of the **difference** field they
carry 0.61× (zonal) and 0.27× (meridional) of the interior; of **legoESM's own**
field, 0.51× and 0.23× (NEMO's own: 0.30× and 0.13×).

## Retractions

**Version 1 claimed a null on the zonal velocity alone.** At a *zonal* wall the
wall-normal component is the *meridional* one, and it is the convergence of the
wall-normal transport that sets the sea surface against a closed wall. Version 1
measured the tangential component and called it an exclusion.

**Version 2 claimed the forcing was named and the machinery exonerated.**
Withdrawn as stated. The "31 000× removed" was the projector annihilating a
state-constant field, not physics — unprojected the removal is 3.1×/4.2×. And
the double ratio promoted to "the registered comparable" factorises exactly as
(wall ratio) ÷ (interior ratio); the interior factor is 0.593 (zonal) and 0.579
(meridional), a basin-wide fact with nothing to do with walls, and dividing by
it lifts every row by ~1.7×. Its numerator is the statistic the same version
retired. What survives, scoped: the forcing owns the varying 4.5 %.

**Version 3 overstated the blindness and mis-sourced its own caveat.** "The
residual is the same field at every state, so the projection cannot adjudicate
this wall error" applied the substituted term's number to the total term's
name: the projector is blind to the constant 95 %, not to everything, and it
*measured* the varying 4.5 % and returned a null. And v3's caveat on the one
excess row — "declines monotonically 5.97 → 3.71 → 2.06" — was quoted from the
double ratio, the statistic it had just called a different quantity. On the
comparable itself that row reads 2.10, 2.51, 1.59: **not monotone**. The
monotonicity claim is withdrawn; the floor caveat replaces it. (Three
overlapping three-state windows are monotone about one time in three anyway.)

**Smaller retractions.** "The two models alternate with roughly equal
amplitude" is false — legoESM alternates ~1.7× *less* basin-wide.
"Anti-correlated at the wall" was a band average over rows that disagree in
sign. "The northern wall's bare ratio is below one under every reduction" is
false: it is 1.02 in the zonal component, and the bare ratio is computed under
one reduction only. And the enrichment bar, the power argument and the
"transferred statistic" all stay retracted from earlier versions.

## What else was measured rather than quoted

* **The averaging window's attenuation**, from legoESM's own saved weights:
  uniform over 45 of 68 substeps starting at substep 24, transfer of a
  substep-alternating sequence exactly 1/45. A source born inside the loop is
  attenuated **45×** — suppressed, not annihilated, and a far weaker limit than
  the state-constancy above.
* **The interior denominator is a hot-cell statistic.** Its effective cell count
  is reported alongside every ratio, and every double ratio is computed under
  three spatial reductions; the version-2 headline moved 3.54 → 1.38 → 1.77
  across them.

## The wind-stress "dead term": the label was wrong, for a third reason

Neither genuinely unmeasurable nor the shadow of the earlier wrapper defect.

The old reason string said wind stress had "the same placement story as bottom
drag". Bottom drag's zero is genuine implicit folding. Wind stress's zero is a
**dead slot in the oracle's own dump path** — the trend-dump routine is never
invoked with the wind-stress index anywhere in the build, so the slot is
allocated, zero-initialised and never written. A physical zero would vary with
latitude; this field is exactly 0.0 over the *whole* domain.

NEMO applies the stress inside the implicit vertical solve, bundling it into the
nonzero vertical-mixing trend; legoESM's card applies it as an explicit top-cell
source.

**CORRECTED 2026-08-26** (`dino_wall_fixed_bias.md` §5, from a source trace of
both models): that is only half of NEMO's wiring. NEMO applies the wind stress
**twice, independently** — once explicitly into the *barotropic* slow forcing
(`dynspg_ts.F90:443-444`) and once as the implicit solve's top boundary
condition (`dynzdf.F90:353-363`). Since `dyn_spg` runs *before* `dyn_zdf`, the
implicit application cannot feed the same step's barotropic loop, which is
exactly why the barotropic term is re-derived explicitly. Also: the placement
term never needed reconstructing — NEMO dumps its own barotropic wind increment
(`wnd_dump_z{u,v}_frc_inc.bin`), and the "dead dump slot" finding above concerns
the per-term *trend* diagnostics, a different instrument. Both stand.

**The comparison has now been run, and the wind is EXCLUDED** — on the
structural ground that this residual is measured in the arm where NEMO's own
`zu_frc` is substituted in, so every slow-forcing assembly difference including
wind is already removed. Note one claim made in passing there and since
retracted: the meridional wind increment is exactly zero, but that is a fact
about the *forcing*, and a zonal forcing in a rotating loop still deposits
meridional velocity. See `dino_wall_fixed_bias.md` §5.
The before-level wrapper fix has no bearing on it.

The reason string is corrected in place. This is the cleanest unstarted
measurement on the card, and it now has a motive: wind stress sits inside the
forcing, which owns the varying part.

## Named next step

**Do not build the substep-resolved projection.** It was version 1's named next
step and it is the wrong instrument: it would split more finely a quantity that
is 95 % invisible to the estimator.

Cheapest first:

1. **Characterise the fixed end-wall bias without the projector.** It is
   5.3e-07 and 1.6e-07 m/s and it is already in the five saved maps. What is its
   row profile and sign structure; is it a boundary-condition difference in the
   loop? No new compute.
2. **A free run.** Whether a fixed bias can seed a two-step oscillation is an
   amplification question, and every matched-state instrument in this family is
   blind to amplification by construction — each state is re-bridged from NEMO.
3. **The wind-stress placement comparison**, offline and unstarted.

**Nine states** would help less than it sounds. Five is every consecutive NEMO
seqdump lane on disk; extending is four short single-rank lanes at kt 5765…5768
(cost estimated, not measured), using the single-file restarts that already
exist there. It would tighten the one row that exceeds unity — but that row is
limited by its operand's leakage floor, which more states do not move.

## Instrument notes

The instrument survived every round; the inference did not.

* **seven reconstructions** are pinned against fields neither the probe nor the
  instrument computed: NEMO's two barotropic sides against its own final
  fields, legoESM's **four** (production and substituted, in each component)
  against the model's own running sums, and the transport reconstruction
  against NEMO's dumped final. Across all five children they run **1.9e-16 to
  1.2e-15**. Two of those pins exist only because review found each substituted
  run constrained by nothing — the second time on the side carrying the claim.
* the projector is proved on fields with known answers before use, and the
  per-cell helper the bootstrap uses is asserted to reproduce the committed
  estimator bit-for-bit.
* **controls run per component.** They ran on the zonal grid only while the
  meridional grid carried the claim, and review mutated the meridional
  component to score with the zonal weights with the suite staying green.
* the land poison alternates over states; its production arm is labelled a
  structural invariant and the unmasked arm is the live one.
* the forcing substitution is asserted to be one variable: the meridional edge
  row must carry no wet face.
* the cross-state correlation keeps its **sign** — an anti-correlated pair is
  the two-step mode itself and must never read as state-constant.
* the leakage floor has **no window parameter**. Mutating it survived round 3
  not because it was untested but because at five states the usable series is
  one sample, so every window selects it; the parameter was removed rather than
  tested, so it cannot become live and wrong at seven states.
* the output surface is tested, at **44 tests**. Round 1 found six inverting
  mutations left the suite green; round 2 found four more; round 3 found four
  more still. All go red now, each re-verified by re-running the mutation.
* geometry is stamped from the mask. The two grids name different northern rows
  (197 vs 196) because a meridional face sits between two tracer rows — the
  staggering, verified from the masks.

## An unretracted contradiction, named not resolved

Two earlier commits on this card say the slow-forcing assembly is clean and the
gap is inside the barotropic substep dynamics; a third says the assembly matches
NEMO's dump "to roundoff" — true at the median (~1e-11), false at the maximum
(~0.09–0.12 % relative). Naming no term, this work does not contradict them, and
its scoped finding (the forcing owns the varying 4.5 %, the loop the constant
95 %) is compatible with both. But the reconciliation is still owed, and three
rows of that assembly comparison — the baroclinic split, the 2-D Coriolis
removal, and bottom drag on legoESM's side — remain unmeasured.
