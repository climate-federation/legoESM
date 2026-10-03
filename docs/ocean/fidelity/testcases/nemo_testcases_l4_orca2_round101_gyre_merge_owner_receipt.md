# ORCA2 round 101 — GYRE merge-year owner

**Date:** 2026-10-02  
**Base:** `3c8f2095d47d05da77b3988246607b37bf3c5519`  
**Preregistration:** `20a13caf2`  
**Disposition:** **HELD — sufficient statement named; no production change**

## Claim boundary

This is a GYRE **INDEPENDENT** from-rest measurement. It diagnoses round 100's
merge gate; it does not score ORCA2 given NEMO's entry and it does not mix the
two claim classes. No model, card, deck, carried state, threshold, stabilizer,
sea-ice selector, or ORCA2 `unmeasured_features` entry changed.

The frozen endpoints are live GYRE source
`d7de69d51f6392e77085aabd879ab9e0da23ebf5` and round 100's combined tree
`23a9911ae48f80ef4c40e4639373b06d5bd42ad8`. The same CPU/fp64/libm seed-0
year harness ran every arm for 17 days with one snapshot per day. Equality
requires both value-array equality and equality of the same-width unsigned bit
views, so signed zero is part of the predicate.

## Frozen predictions

| prediction | result | verdict |
|---|---|---|
| P1 ocean-package content owns the movement | The combined control first differs at day 17; replacing all 15 differing ocean files is exact for 85/85 day-field arrays through day 17. | **CONFIRMED** |
| P2 both half-splits construct and isolate one group | Two syntactic half-arms split `_rnf_content` from its consumer and refuse before step 1. Dependency-closed arms were required. | **REFUTED**; those runs make no science claim |
| P3 one file and an executable ORCA-side statement restore the endpoint | Replacing only `ocean_model_latlon_cgrid.py`, then replacing only edit blocks 29-30, and finally reversing commit `a0b2f7a5d` each restores all 85 arrays. | **CONFIRMED as a sufficient statement, not a uniqueness claim** |
| P4 the ten-step ladder is blind | The source and statement-reversed arms have 70/70 unchanged rows and byte-identical residual archives. | **CONFIRMED** |

The synthetic bit-flip control turns an otherwise exact comparison red. The
pre-measurement signed-zero control also caught that `np.array_equal` alone is
not a bit predicate; the preregistration and committed probe were corrected
before the first model candidate ran.

## First non-bit state and sufficient owner

The unchanged combined control and the refuted `938f41892f7a0f` stage-source
order reversal have the same first difference:

| day | field | index | source | combined | day-field maximum | unequal values / unequal bytes |
|---:|---|---|---:|---:|---:|---:|
| 17 | T | `(1, 2, 10)` | `18.843907803472824` | `18.84390780347282` | `2.0250467969162855e-13 K` | 675 / 681 |

The smallest dependency-closed restoring pair is the stage-one tracer
thickness ratio import and evaluation at
`ocean_model_latlon_cgrid.py:7315-7325`. It replaces a ratio formed after SSH
interpolation with `nemo_r3t_rk3_stage1_stretch`. Reversing the complete
introducing commit `a0b2f7a5da06f416e530361355df8603b7826a3d`, including its
helper in `eos.py:975-1006`, restores every saved bit through day 17. Reversing
the nearby source-order commit `938f41892f7a0f917e09f55bd525782ec6eb22ae`
does not move the first difference by a bit, so that frozen hypothesis is
**REFUTED**.

This statement is not inferred from Python associativity. GYRE's compiled
stage program saves the after SSH and selects HYB interpolation at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:150-192`, including
the after-level ratio calculation and independent interpolation of `r3t(Kaa)`.
ORCA2's compiled program contains the same nonlinear ratio
interpolation at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
Both cards execute the statement; it is shared, not ORCA-card-scoped.

The introducing ORCA2 round-47 receipt claimed exactly this association. Its
given-entry replay closed 2/8,613 unequal ratio cells to zero, moved ORCA2
toward NEMO, left no AT-BAR row, and admitted a small GYRE year movement under
Decision 59. This round does not retract that compiled-source statement or
land a competing spelling. It shows that this already-admitted movement is
also the sufficient cause of round 100's later byte-identity failure.

Because P2's arbitrary syntactic complements did not construct, this receipt
does **not** claim mathematical uniqueness among every possible multi-hunk
combination. The complete-commit reversal is the one-variable sufficient
discriminator.

## Why ten steps missed it

The statement-reversed arm's ten-step gate exits with the expected scientific
DEBT report, then the offline movement gate passes all 70 registered rows:
zero moved cells, zero worsening ULPs, and first-over-bar retained at kt=3 for
T/S/u/v/ssh. Its residual artifact and round 100's live-source artifact are
byte-identical, SHA-256
`377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
The first daily bit separates only at day 17, so the ten-step ladder cannot
police this byte-identity predicate.

Post-hoc, and labelled as such, the existing 360-day endpoints score against
NEMO as follows:

| day | source T rms (K) | merged T rms (K) | merged - source (K) |
|---:|---:|---:|---:|
| 30 | `2.3432510206121264e-06` | `2.3432465132112266e-06` | `-4.507400899784753e-12` |
| 240 | `6.581707093530567e-05` | `6.58170609494473e-05` | `-9.985858372293065e-12` |
| 360 | `5.407735418221895e-05` | `5.4077419367442036e-05` | `+6.518522308436797e-11` |

Those magnitude changes are inside Decision 59's `2e-9 K` allowance and the
first two move toward NEMO. They do not satisfy note B33's stronger literal
byte-identity requirement, which is why this round remains HELD.

## Evidence and controls

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round101/`.
Decisive artifacts are `control_source_exact.json`,
`control_combined_different.json`, `all_source_v2.json`,
`ocean_model_only.json`, `opcodes_29_30.json`, `reverse_a0b2.json`,
`reverse_a0b2_ladder.json`, `reverse_a0b2_ladder_compare.json`, and the two
`*_year_gap_posthoc.json` files. Each overlay manifest records endpoint SHAs,
selected edit blocks or reversed commits, and candidate file hashes.

The following negative or non-scientific arms are retained:

- `control_plant.log`: the one-bit plant fires.
- `reverse_938f.json`: source-order reversal remains DIFFERENT with the exact
  registered first cell and magnitude.
- `candidate_opcodes_29_34.log` and `candidate_opcodes_29_36.log`: both refuse
  before stepping because the artificial split leaves `_rnf_content`
  undefined; neither is interpreted physically.
- `opcodes_35_40.json`: the dependency-closed later block remains DIFFERENT
  with the exact registered first cell and magnitude.

## Validation and review

The final clean-tree focused battery passes **22/22** tests. It includes the
five probe tests (bit flip, signed zero, frozen file census, reverse commit,
and 49-edit-block controls) and all 17 citation-gate tests. The default
cumulative receipt and this receipt both pass with zero failures and zero
unmapped citations. Shifting
`ocean_model_latlon_cgrid.py:7315-7325` by two lines makes the citation gate
fail, so the receipt control fires.

The one required `tests/ocean/fidelity -n 12` battery reached 99%, emitted 33
preliminary FAILED node progress lines, and then repeated the documented
xdist-controller stall. It was interrupted after bounded silent waits. Because
pytest never printed its terminal tracebacks or summary, those preliminary
lines are not classified here; the battery is **incomplete, not PASS**. The
round-101 probe and citation suites were subsequently rerun serially from the
committed tree and are the 22/22 result above.

The required separate `codex exec --sandbox read-only` review did not reach
the diff. Its terminal verdict is **independent review unavailable
in-sandbox**: `failed to initialize in-process app-server client: Read-only
file system (os error 30)`. Unavailable review is not approval.

## OPEN

1. Reconcile note B33's byte-identity requirement with the already-landed
   Decision-59 admission of the shared stage-one ratio statement. Do not
   revert or card-scope a NEMO-cited shared statement without a new gated
   authorization.
2. Round 100's rung-7 signed-zero refusal and missing committed rung-0
   given-entry gate remain open and are not repaired by this diagnosis.
3. Resume the rung-0 barotropic source-order walk only after the merge gate's
   disposition is explicit.
