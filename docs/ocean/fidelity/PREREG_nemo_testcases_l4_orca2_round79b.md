# PREREG — ORCA2 round 79b: what the 1 m TKE mixing-length floor was compensating

Frozen before any measurement.  Round 78 established that legoESM's ORCA2 card
had been running an *effective* turbulence mixing-length floor of 1.0 m (a
derivation the card's own field never reached) and that NEMO's own value for
this deck is 1.0e-3 m, forced by `ln_zdfiwm = .TRUE.` in the compiled two-arm
choice.  Moving to NEMO's value made the ten-step ladder's step-10 end-of-step
maxima 13-15% worse, and round 79a re-measured the independent month on the
corrected tree: temperature rms +7.4%, salinity maximum +79.0%.

A floor that is NEMO's own number cannot be the defect.  The 1 m floor was
therefore covering an error somewhere else, and this round names where.

## Hypothesis space

| id | hypothesis | what would own the compensation | discriminator |
|----|------------|--------------------------------|---------------|
| a | the TKE chain itself | the `zdf_mxl`-style mixing-length recurrence (`nn_mxl = 3`, two-arm up/down sweep), the surface anchor under `ln_mxl0`, or the step-entry N-squared routing that GYRE round 183 fixed | NEMO's per-step `en`, `zmxlm`, `zmxld`, `dissl` and the closure's own `avt_k`/`avm_k`, substituted one at a time |
| b | the internal-wave mixing arm | `zdf_iwm` (de Lavergne) adds a diffusivity on top of the closure's; if legoESM never adds it, legoESM's interior is under-mixed and a 1000x larger length floor partially restores it | NEMO's `avt`/`avm`/`avs` immediately before and immediately after `zdf_iwm`, i.e. the increment itself |
| c | the consumers | the final `avt`/`avm` that reach the implicit tracer and momentum solves differ for a reason downstream of the closure | NEMO's final `avt`/`avm`/`avs` at the end of `zdf_phy`, substituted into legoESM's solves |

Two further arms fall out of the same record because they sit between the
closure and the consumers in NEMO's own order: the river-mouth diffusivity
enhancement and the double-diffusive salt/heat split.

## Frozen predictions

1. **P1 (hypothesis b, checked by reading alone, before any NEMO run).**  The
   ORCA2 card does not apply internal-wave mixing.  Falsifier: the card
   resolves an enabled internal-wave arm, or the deck does not select one.
2. **P2.**  The deck's internal-wave input file is present on disk, so the
   missing arm is transcription work and not an acquisition.  Falsifier: the
   file named by the deck is absent from the pinned input set.
3. **P3.**  NEMO's internal-wave increment to the tracer diffusivity is, over
   the wet interior, at least as large as the diffusivity the retired 1 m floor
   produced.  Falsifier: the recorded increment is everywhere smaller than
   `rn_ediff * 1.0 m * sqrt(en)` evaluated on the same record.
4. **P4.**  Substituting NEMO's final end-of-`zdf_phy` diffusivities into
   legoESM moves the ten-step ladder's step-10 end-of-step temperature and
   salinity maxima back toward NEMO by more than substituting the closure's own
   `avt_k`/`avm_k` does.  Falsifier: the closure-only substitution moves them at
   least as far, which would put the owner inside the TKE chain (hypothesis a).
5. **P5.**  The mixing lengths legoESM builds agree with NEMO's recorded
   `zmxlm`/`zmxld` to better than the ladder's own residual once the floor is
   NEMO's.  Falsifier: they do not, which again puts the owner in hypothesis a.

A failed prediction is recorded as REFUTED and kept.

## Instrument

An additions-only, write-only record at the five boundaries of NEMO's compiled
`zdf_phy`, in NEMO's execution order, for the first ten steps and on every rank,
plus the closure internals the mixing-length hypothesis needs.  The record is
self-describing: a magic, fifteen header integers, then per array a name, a
rank, three extents, three origins and an eight-byte-per-value payload.  The
checker reads the names and the payload lengths out of the record and never
predicts a byte count.  Admission is restart byte-identity against the pinned
ten-step record plus the header/field/truncation/stamp plants.

## Scope

Measurement and transcription of statements the ORCA2 card does not currently
execute.  Any statement that changes a card other than ORCA2 lands only under
that card's own gate.
