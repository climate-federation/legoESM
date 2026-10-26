# PRE-REGISTRATION — the CORIOLIS PAIR, three arms (#1455 next-action 1)

**Written and committed BEFORE any arm was scored.** The instrument
(`substep_traj_compare.py`, the vertex-Coriolis substitution) and the scorer's
provenance gate are already committed at `HEAD` when this file lands; no map
directory exists yet for any of the three arms, and the baseline directory on
disk is the recorded one from commit `513339cba`.

## Why the pair, and why never one half alone

legoESM builds the Coriolis parameter at the vertex as the **average of the two
adjacent tracer rows**; NEMO evaluates it at its own f-point latitude. Measured
against NEMO's own dumped `ff_f` the gap is median **−5.48e-05**, RMS 6.15e-05,
peaking at the **equator** (−9.19e-05) and vanishing toward the walls. The
v-face zonal metric gap — the arm already run and recorded PARTIAL
(`dino_wall_fixed_bias.md` §6a) — is **+1.86e-05** median and does the reverse:
+3.35e-05 at the walls, +2.9e-09 at the equator. Their signed latitude profiles
correlate at **+1.000**, they enter the **same** EEN rotation coefficient
`e1v · f` with opposite signs, and they therefore **partially cancel**. The
completed arm removed the *smaller* of a cancelling pair.

That is the Rule-8 pattern this campaign has recorded three times (the count is
three, not four: the vertical-ladder paradox was later reversed outright by PR
#1638). The registered response to a cancelling pair is a **joint arm**, never
a revert of the half already fixed.

**One third of the Coriolis gap is not a discretisation convention at all.**
STEP 0 of this action (`coriolis_omega_routing_audit.py`, committed, artifact
`results/dino_1455/coriolis_omega_routing_audit.json`) measured the routing by
instantiation and found the run carries **three** rotation rates: every
config-side site on the card's pin `7.292116e-05`, every geometry-side site
(and therefore every Coriolis array) on legoESM's rounded `7.292e-05`, while
NEMO's own `ff_f` inverts to `7.292115083046e-05` with a row-to-row spread of
5.6e-16. Splitting the vertex-Coriolis gap by dividing each side by the rate it
was built with gives, over the 187 rows with |sin φ| ≥ 0.1:

| component | value |
|---|---|
| total, as shipped | median −5.2001e-05, RMS 1.0017e-04 |
| constant (rotation rate) | uniform −1.5908e-05 |
| placement (row average vs f at the f-point) | median −3.6094e-05, RMS 9.1434e-05 |

Substituting NEMO's own `ff_f` removes **both** halves at once, which is what
the arm below is for: it measures the pair's Coriolis side complete.

## The three arms

All three run the **same** loop, from the **same** bit-identical entry state,
with the **same** substituted (NEMO's own frozen) slow forcing, over the same
68 substeps with the same weights, at the same five consecutive states
`kt = 5760…5764`, produced by the same driver at one clean tree.

| arm | `DINO_1455_SUB_VFACE` | `DINO_1455_SUB_CORIOLIS` |
|---|---|---|
| **BASE** — plain frozen forcing | unset | unset |
| **E1V** — the completed metric arm, re-run for a common baseline | `nemo` | unset |
| **F** — the Coriolis half alone | unset | `nemo` |
| **JOINT** — the pair, both together | `nemo` | `nemo` |
| **CTRL-F** — the staggering control on the Coriolis arm | unset | `stagger` |

`stagger` feeds the **un-shifted** `ff_f`: the same 10 494 cells are
overwritten, only the row alignment differs.

## THE VERDICT IS REGISTERED ON THE JOINT ARM ONLY

The pair is the object. Scored by `baro_fixed_bias_wall_map.py`, whose scoring
mathematics is **unchanged** — the only edit this action makes to that file is
teaching its provenance gate the new arm token, without which a joint-arm map
would be scored under the e1v arm's label. The registered quantity is the
**state-constant wall-normal residual**, five-state mean, exactly as for the
completed metric arm.

- **OWNER** — the residual collapses **> 50 %**, and it must do so on **both**
  reductions: basin-wide **and** on the northern lobe (rows 185–197, which
  carried 74 % of the metric arm's whole variance reduction).
- **REFUTED** — collapse **< 10 %**.
- **PARTIAL** — in between.
- **OVERSHOOT** — a collapse past 100 % is over-correction, not ownership.

**The staggering control must FIRE**, i.e. be clearly worse: a state-constant
wall-normal residual more than **50 % above** the BASE arm's. If it does not
fire, the loop is insensitive to this array and **no verdict may be issued from
the correct arm either**.

**The one-variable gates must all pass or the arm has no measurement**: the
row-map discriminator at ≥ 100× separation, the content sweep finding exactly
one array carrying legoESM's own vertex Coriolis, and the **inertness control**
— the `f_u`/`f_v` loop kwargs scaled by 1e3 must leave the carry bit-identical,
proving the arm is not a partial perturbation.

## Also recorded, and explicitly NOT carrying the verdict

Each of these is registered so that it cannot be presented afterwards as if it
had been the target:

1. **The three-band structure's response** (rows 57–73, 121–153, 185–197),
   reported per band for all four arms.
2. **The tangential channel**, which the metric arm's registration did not
   score and which moved nearly three times as much as the wall-normal one
   (46.0 % basin-wide). Reported, not registered.
3. **Additivity.** Whether JOINT's collapse equals E1V's plus F's is the pair
   claim's own test: a cancelling pair predicts JOINT **exceeds** the sum. This
   is a *description* of the pair, not a verdict criterion.

## The 90-day gate is CONDITIONAL, and the condition is registered here

The five-metric acceptance gate and its channel band are run **only if the
joint arm clears > 50 % on the per-step measurement**. The per-step probe is
cheap; a 90-day pair of runs is not, and this campaign's compute discipline
forbids spending it on an arm the cheap measurement has already refuted. If the
joint arm lands PARTIAL or REFUTED, the gate is **not** run and the reason is
recorded.

## If the joint arm CONFIRMS ownership, this chain prediction registers first

Registered **before** any 90-day run, so it can fail:

- the state-constant wall-normal residual **falls to the floor at the wall
  rows** (rows 1 and 196), not merely basin-wide;
- and the **basin torque gap moves** — the southern-basin −0.95 Sv full-section
  deficit is the campaign's largest open term, and a term that owns the
  barotropic loop's fixed bias but leaves the basin transport untouched has not
  been shown to matter to the result anyone cares about.

A joint arm that collapses the residual and leaves the basin gap where it is
would be a **real but inconsequential** fix, and it will be reported as such.

## What is NOT being claimed

- Nothing here says the vertex Coriolis owns the **southern-basin** deficit.
  The object under test is the barotropic loop's state-constant wall-normal
  residual, which is a per-step object.
- The rotation-rate routing defect is a **harness** defect: the production
  lat-lon DINO path builds its grid from the card and carries the card's rate.
  The oracle-matching path is the one on the wrong Earth. Fixing it is a
  separate commit and is deliberately **not** landed before these arms run, so
  the BASE arm stays comparable to the recorded metric-arm baseline.
