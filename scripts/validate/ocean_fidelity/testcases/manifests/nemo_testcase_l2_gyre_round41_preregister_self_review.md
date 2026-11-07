# Round 41 preregistration — adversarial self-review before code

Verdict: **PASS AFTER CORRECTION**.  No instrument or reader code existed when
this review was performed.

1. **Could the prediction be unfalsifiable?**  The first draft merely ranked
   KEG.  Corrected: KEG must be over the relative `1e-15` bar and both ZAD rows
   must be at it; either contrary observation explicitly REFUTES the claim.
2. **Could `wsd` be dumped safely because its executed contribution is zero?**
   No.  With resolved `ln_wave=F`, `sbcwave` returns before allocating `wsd`.
   Corrected: the writer never references the object.  It records the live
   `ln_vortex_force=0` branch flag and a named, writer-created zero
   `wsd_effective`; the reader rejects any nonzero value.
3. **Could post-minus-pre be mistaken for the source statement?**  Yes.
   Corrected: the calibration replays the sequential accumulator updates from
   `before_keg` to `after_keg` to `after_zad` and requires bit identity first.
   Isolated contribution differences are secondary score rows only.
4. **Could stage 2 overwrite a stage-3 record?**  Corrected: `stprk3_stg`
   explicitly arms the `dynadv` writer only around the `kstg==3` call; the
   record header repeats the stage/time-level ladder and the reader requires
   `(kt,kstg,Kbb,Kmm,Krhs,Kaa)=(1,3,1,2,3,3)`.
5. **Could the writer perturb NEMO?**  The acquisition must prove additions
   only, raw identity where deterministic, and consumed-field identity on all
   parent records, with the new filename as the sole admission exception.
6. **Could a green calibration bless a reimplementation instead of the model?**
   The independent replay is only record calibration.  Scientific rows must
   call legoESM's shared KEG/ZAD paths with NEMO operands; a host-only proxy is
   inadmissible.
7. **Could an at-bar result be called exact?**  Exact bit counts and the
   relative `1e-15` classification are separate, and `DISCHARGED` is reserved
   for zero unequal cells.
8. **Could a GYRE fix evade other cards?**  No landing is authorised until the
   record arrives.  The preregistration requires resolved-arm evidence per
   card, given-input identity on every executing card, ORCA2
   UNMEASURED-WITH-SPEC, all moved trajectory rows, and no earlier first-over-
   bar.

The corrected preregistration contains no scheme/default choice, no new
stabiliser, and no post-hoc statistic presented as predicted.
