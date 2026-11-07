# PREREGISTRATION — VORTEX round 4

Frozen before any measurement of this round. Lane tip `85607c118588`.
Evidence root `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round4`.

Two parts, independent: part A is decision 75 (state `enforce_cfl` on every
NEMO card and re-pin the digests it moved), part B is the one-variable walk
that names the vector-EEN card's kt=2 momentum statement.

## Part A — decision 75

### What is done

Every NEMO card's resolved configuration states `enforce_cfl` for BOTH
lateral-mixing sub-blocks that carry a field of that name (`harmonic` and
`biharmonic`), at the value main's library defaults now resolve to, so the
card's resolved value is the card's own statement and not a library default.
The three construction sites are the shared NEMO physics builder that the
GYRE, LOCK_EXCHANGE, OVERFLOW, VORTEX, VORTEX_VEC and ORCA2 cards all route
through, and the two DINO physics builders (lat-lon and MPAS).

### Predictions

* **A1.** The GYRE card's digest moved for exactly one reason. With the
  library's biharmonic `enforce_cfl` default forced back to `False` and
  nothing else changed, the GYRE card's digest is EXACTLY the round-2 pin
  `eaef11b4c2e37a31`; with it at main's `True` it is
  `337651dbd9f1b49c`. If either fails, the digest is NOT re-pinned and the
  round reports the rest of the move instead.
* **A2.** Writing a field explicitly at the value it already resolves to
  changes no digest, because the digest hashes the NamedTuple's repr, which
  prints every field either way. So after the edit every card's digest equals
  the value measured before it, and `LOCK_EXCHANGE-zco` and `OVERFLOW-zps`
  keep their round-2 pins unchanged.
* **A3.** The default-independence test is non-vacuous: with the library
  default flipped, a `LateralMixingConfig` built the way the cards used to
  build it DOES move, and the cards as they now ship do NOT.
* **A4.** No executed number moves: both VORTEX ladders, the GYRE certified
  ladder and day 30/240/360, both tanks and the DINO month gate are
  bit-unchanged from their certified values.

## Part B — the vector-EEN card's kt=2 momentum statement

### The bar and the target

Round 3 measured the vector card at kt=2: u `1.432e-05`, ssh `2.269e-05`
normalised, flat through kt=10. The flux card is at `1.136e-07` / `3.709e-08`.
Bar `1.0e-15`.

### The method

The round-3 walk is extended, not re-implemented: the same arms, the same
normalized maximum, the same record readers, run on the vector card against
the round-3 vector record, plus the two things round 3's reviewers required
and one new boundary the record already supports.

1. **A plant on EVERY substituted arm.** Round 3's plant perturbed arm 0's
   scoring only, so the walk passed even if both substitution hooks were
   inert. The new plant perturbs the SUBSTITUTED value of the named arm, so a
   dead hook makes the planted run indistinguishable from the clean one and
   the plant fails.
2. **A verdict.** The walk exits non-zero when it cannot attribute.
3. **The stage-local boundary.** Each stage is run from NEMO's own recorded
   entry and its OUTPUT is scored against NEMO's recorded output for that
   stage. That is a sharper boundary than the kt=2 entry, and the round-3
   record holds every field it needs.

### Predictions

* **B1.** Arm 0 on the vector card reproduces the round-3 ladder's kt=2 row
  (u `1.432e-05`, ssh `2.269e-05`) to the harness's own floor.
* **B2.** Arm 1 (NEMO's kt=1 entry) does not move the velocity row: the two
  VORTEX cards' kt=1 entry records are byte-identical and the flux card's
  arm 1 reproduced arm 0 bit for bit.
* **B3.** Arm 2 (the external-solve handoff) puts the height row at zero and
  leaves the velocity row where arm 1 left it, as on the flux card.
* **B4 (the pick).** The residual is flat in kt, so it is re-made every step
  by a statement all three stages share. PICK: the stage-local walk shows a
  residual in the velocity output of **stage 1** — the first stage to run the
  vector-invariant operators — of the same order as the step's, and the
  kt=2-entry arms 3 and 4 therefore do NOT collapse it (each stage re-makes
  it). Named alternative: the flux card's shape, where only stage 3's own
  contribution survives and arms 3 and 4 leave the residual at stage 3's
  share. The measurement that discriminates them is arm 4 versus the
  stage-local stage-1 row.
* **B5.** The owner is inside the three operators vector form adds — the
  kinetic-energy gradient, the vertical advection of momentum, and the EEN
  vorticity on the live relative vorticity — because the two cards share
  their initial state byte for byte and differ in nothing else. Which of the
  three is NOT predicted here.
* **B6.** If naming the operator needs NEMO's per-term momentum tendency and
  the round-3 record does not carry one, the round writes the additions-only
  acquisition and stops with `ACQUISITION_NEEDED` rather than ranking the
  three by magnitude. Round 3 paid for that mistake once.

### What holds the round

Nothing lands on part B unless the named statement's fix collapses the
vector card's ladder toward the flux card's level or below, with both VORTEX
ladders, the GYRE certified ladder and day 30/240/360, both tanks, the DINO
month gate and the cards' tests green. Part A lands on its own if part B
holds.
