# SI3 round 24 receipt — O15/O16 stopped proposal

Date: 2026-10-28  
Round: `si3_r24`  
Parent: `87a1c8843803f0cb6263a2a84f44f8d83183ee96`  
Preregistration commit: `cac0b9a415`  
independent review: operator (Claude)

## Verdict

**STOPPED — the preregistered 0-ULP claim is refuted at the first scored
step.** No post-score physics change was made. O15's first production result
differs by one ULP; O16 also exposed two record/scorer-domain defects that must
be resolved before an O16 model verdict is admissible.

This is a model-preparation and localization proposal, not a fidelity claim.

## CONFIRMED evidence

### Executing arms and transcription

- The compiled O15 arm transfers receiver area with the verbatim statement
  `a_i_2d(ii,jl2) = a_i_2d(ii,jl2) + pdaice(ii,jl)`
  (`iceitd.f90:496-500`). The production candidate is the single implementation
  in `packages/ice/legoesm/ice/si3_finalize.py:254-311`; its receiver-area
  statement is at `:286-290`.
- The resolved deck has `nn_fsbc=1` and no `ln_drgice_imp` namelist token; the
  admitted receipt resolves that omission to `.FALSE.`. The executing source
  therefore takes `zflagi = 1._wp` (`iceupdate.f90:373-418`), not the implicit
  arm.
- Kind 15's admitted Python schema orders its final five interior fields as
  `at_i_b wfx_snw_dyn cloud_fra qsr sfx`
  (`nemo_si3_o8_o16_master_gate.py:171-180`), while the committed producer
  writes `at_i_b`, `wfx_snw_dyn`, `qsr`, `sfx`, `cloud_fra`
  (`o8_o16_master_writer.F90:446-457`). Thus the reader-labelled
  `cloud_fra` is actually `qsr`. At step 1 it ranges from 0 to
  310.9272361839405, while reader-labelled `sfx` is the actual constant cloud
  fraction 0.81. The O16 albedo mismatch is an instrument-schema mismatch, not
  an admissible model verdict.
- Kind 20 records full-halo `utau`/`vtau`, but NEMO writes only the interior in
  the final stress loop (`iceupdate.f90:410-419`). At step 1 the first halo
  contains finite uninitialised-looking payloads (`0x00007f12425fb4b0`) that
  differ between kinds 19 and 20. Full-halo stress comparison is therefore not
  an owned-output comparison; only `A2D(0)` is scoreable.

### First genuine non-bit statement

For `kt=1`, interior cell `[8,25]`, zero-based category 1 (`jl=2`):

- master O15 `a_i`: `0.005258202624687821`, bits `0x3f7589a004f0bea2`;
- production JIT `a_i`: `0.0052582026246878215`, bits
  `0x3f7589a004f0bea3`;
- difference: exactly +1 ULP.

The cell's downward transfers execute category 5→4, 4→3, then 3→2. NEMO's
stored-statement order yields
`scaled_cat2 + (cat3 + (cat4 + cat5)) = 0x...bea2`. The JIT candidate yields
the reassociated accumulation
`((scaled_cat2 + cat5) + cat4) + cat3 = 0x...bea3`. The first differing
statement is therefore the final receiver-area addition at
`iceitd.f90:497`, represented by
`packages/ice/legoesm/ice/si3_finalize.py:289-290`. The existing
`nemo_source_round` identity did not
prevent XLA reassociation across the three sequential boundary calls.

## PLANT / control disposition

The baseline fired before plants were admissible. Per the kill rule, the
one-ULP and model-mutation plants were not promoted as passing evidence. The
scorer does contain separate O15 one-ULP/rebin-mutation and O16 one-ULP/heat-
term/stress plants, but this stopped round makes no control-sensitivity claim.

## Tests run before scoring

`JAX_ENABLE_X64=1 pytest -q tests/ice/unit/test_si3_finalize.py` passed 4/4.
Those tests use an independent NumPy scalar oracle, runtime JIT arguments,
real eager-versus-JIT equality, central-difference gradient checks, zero-ice
states, and production-call-site AST checks. They do not override the master
record failure.

After localization, the compatibility gates also passed:

- receipt citation gate: 39/39;
- focused model/default/private-import gates: 13/13;
- O8–O12 duplicate-arithmetic AST guard plus plants: 1/1;
- full D5 CPU production-JIT year rescore: all 8,760 steps completed with
  `D5_PLATFORM_FLOOR`; shipped arm `r6_step1_plus_step2_snow_update` retained
  `full_year_output_sha256 =
  b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
  The generated JSON was 8.5 MiB, SHA-256
  `6d5491789c06a067b7ee90f7fafa3ee744ac04bae34fa951df374d566c14983b`.

## Proposal (not executed)

1. Start a fresh preregistered round and add a contraction/reassociation
   barrier that demonstrably preserves the nested receiver value after each
   `itd_shiftice` boundary; prove its planted removal fails before rescoring.
2. Resolve the O16 master contract explicitly: either reacquire kind 15 with
   producer/schema order aligned, or add a versioned reader translation backed
   by the producer source and a planted permutation. Do not silently relabel
   the immutable admitted bytes.
3. Restrict kind-20 stress scoring to the NEMO-owned interior, or reacquire
   initialized halos if full-halo ownership is desired.
4. Then rerun all 100 steps/categories and both plant families. No configuration
   change is proposed.

## CHOICES

### ASKED

- Production entry points only; no duplicate O15/O16 arithmetic.
- Literal source order, explicit deck resolution, 0-ULP all-step scoring, and
  stop/localize on the first non-bit result.
- Independent scalar/JIT/gradient/edge tests and operator review.

### UNASKED

- No NEMO configuration change.
- No relaxation of the 0-ULP bar.
- No correction of admitted master bytes and no post-score model tuning.
- No nested Codex review; independent review is the operator (Claude).

## CONFIRMED versus PLAUSIBLE

CONFIRMED: the O15 +1-ULP bit patterns and the exact receiver statement; the
kind-15 producer/schema permutation; kind-20's unwritten halo domain; the
resolved explicit-drag arm; the four green local tests.

PLAUSIBLE, not claimed: a stronger materialization primitive will preserve the
NEMO nested addition under JIT; a versioned kind-15 translation will make the
remaining O16 fields scoreable at 0 ULP.
