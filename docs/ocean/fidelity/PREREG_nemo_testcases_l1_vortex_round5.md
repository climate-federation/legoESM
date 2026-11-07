# PREREGISTRATION — VORTEX round 5: which of the two surviving terms owns the vector card's second step

Frozen before any measurement of this round. Lane tip `5add31a068ad`.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round5`.

## What is already known, and is NOT re-litigated here

Round 4 proved by substitution that the whole of the vector-EEN card's
first-stage momentum error lives in the three-dimensional momentum
right-hand side NEMO completes in the barotropic preparation: handing
legoESM NEMO's own copy puts stage 1 at `1.1e-16` against a bar of
`1e-15`, from `1.7074e-05`. legoESM's own completed copy of that array
differs from NEMO's by `2.8783e-08 m/s^2` against a required floor of
`1.7785e-08 m/s^2`. NEMO's per-term record excluded the lateral viscosity
(exactly zero on this deck), the vertical advection of momentum (the whole
term is `1.6928e-08`, below the floor), and — by the flux card, which runs
the same three calls 600x closer — the pressure gradient and the planetary
half of the vorticity. **Two candidates survive: the relative-vorticity
half of the energy-and-enstrophy vorticity, and the kinetic-energy
gradient.**

## The discriminator, stated before it is run

legoESM's own per-term breakdown at the SAME boundary, term by term against
NEMO's recorded increments. The mapping is not invented: it is what
legoESM's own per-term diagnostics already contain.

| NEMO's recorded increment | legoESM's diagnostic | kind of map |
|---|---|---|
| `vor` (dyn_vor, the whole triad) | `vortcor_u` | EXACT — under `een_total` the planetary part is inside the triad on both sides |
| `zad` (dyn_zad) | `vertadv_u` | EXACT |
| `hpg` + `keg` | `KE_PGF_u` | EXACT AS A GROUP — legoESM bundles the kinetic-energy gradient with the pressure gradient in one field and cannot split it, exactly as NEMO's `adv` bundles keg with zad in the other direction |
| `ldf` | `Ah_lap_u + Bh_bilap_u + Cs_smag_u + Cl_leith_u` | EXACT; identically zero on this deck |
| — | every remaining diagnostic component | must be identically zero, or the map is incomplete and the round REFUSES |

**The owner is the term whose legoESM-minus-NEMO difference accounts for
the `2.8783e-08` while the others sit at the comparison floor.** Because
the pressure gradient is independently excluded by the flux card, a
`KE_PGF` group difference at the level of the total names the
kinetic-energy gradient; a `vortcor` difference at the level of the total
names the relative-vorticity half of the triad.

Stated in advance so the reading cannot be chosen afterwards:

* **OWNER = the vorticity term** if `|vortcor_u - vor|` is within a factor
  of two of `2.8783e-08` and `|KE_PGF_u - (hpg+keg)|` and
  `|vertadv_u - zad|` are each at least ten times smaller.
* **OWNER = the kinetic-energy gradient** if the same holds with the two
  rows exchanged.
* **NEITHER, and the round reports NO OWNER** if two rows are within a
  factor of ten of each other, or if no row reaches half the total. A
  split owner is a legitimate outcome and will be reported as one rather
  than rounded to the larger row.

## The instrument, and why it is the quantity and not a proxy

The per-term diagnostics are taken from the SAME `tendencies` call the
step already makes, with the same overrides (the stage face thicknesses,
the lateral-diffusion thickness operands, the continuity clock), through a
new WRITE-only observer next to the one round 4 used for the total. They
are NOT recomputed by a second call to a public wrapper, which would
silently drop those overrides and measure a different array.

## Controls, each fixed before the run

1. **Closure.** The diagnostic components must sum to the observed total to
   `<= 1e-18`, and the total must equal the array round 4 compared, so the
   rows are a decomposition of the number being explained and not a second
   opinion about it.
2. **Passivity.** The walk's arm-0 residual must be bit-identical with the
   observer on and off. A measurement that moves the model is refused.
3. **The ordered control.** NEMO's vertical-velocity dump must leave the
   momentum accumulator bit-identical, as in round 4, or the increments are
   not terms.
4. **Plants, one per compared row.** Each of the four compared rows is
   perturbed on its own, on the legoESM side, and the discriminator must
   flag EXACTLY that row and no other. A plant that moves a row the
   discriminator does not name, or that moves more than its own row, fails
   the round.
5. The unmapped components must be identically zero (table above).

## What happens after the owner is named

The owning routine's statements are walked against NEMO's compiled source
in `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo` to the FIRST non-bit statement,
cited `file:line` on both sides. Candidates written down now, so the walk
cannot be steered: for the vorticity, the four-triad averaging order, the
vertex thickness `e3f` (masked against unmasked, and the substitution at
land or closed faces), and the free-slip boundary at closed walls
(`rn_shlat = 0` on this deck); for the kinetic-energy gradient, the arm
`nn_dynkeg = 0` selects (C2, not the Hollingsworth correction), the
half-factor and the order of its operands.

A fix lands only under the full gate: both VORTEX ladders before and
after with every `kt` reported, the GYRE certified ladder and — if GYRE
moves at all — the full day 30/240/360 re-run against
`2.327677e-06 / 6.586172e-05 / 2.670992e-03 K`, the tanks, the DINO month
gate against `2.040288765e-03 K` in a private work directory, the cards'
own tests, the citation gate and the push gate. GYRE, DINO and ORCA2 all
run vector-invariant momentum with the energy-and-enstrophy vorticity, so
a fix here is a SHARED landing and the card census says so explicitly.

**If the owner cannot be named from this record, the round STOPS and says
so.** It does not rank the two candidates by magnitude; that is the
mistake round 3 paid for.
