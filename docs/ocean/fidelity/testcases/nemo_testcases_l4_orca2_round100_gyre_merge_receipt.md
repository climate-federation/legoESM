# ORCA2 round 100 — live GYRE lane merge

Date: 2026-10-02. ORCA2 parent: `752616c1c12d566a2d5e727434011add95ce2f0a`.
Preregistration: `10d34d3e483582e3607b6ae0ccc5e44407f30e90`.
Measurement tree: `23a9911ae48f80ef4c40e4639373b06d5bd42ad8`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round100`.

## Verdict

**HELD.** The live GYRE tip was merged completely, both parents are ancestors,
the citation registries are a semantic union, and the GYRE ten-step ladder is
array-identical to that live source tip. The combined tree nevertheless moves
the certified GYRE year starting at day 17: 344 of 360 daily snapshots differ.
That falsifies the byte-identical GYRE landing predicate.

Two ORCA2 certification requirements also remain red. The shipped rung-7 gate
refuses before the ladder because its current card differs from the V2 kt=1 T
operand at 77,662 signed-zero cells; the same refusal and count reproduce on
the pre-merge ORCA2 parent. Rung 0 has no committed kt=1..10 ladder gate: its
available independent entry/stage-1 gate passes, but all five stage-1 rows
move. No model statement, configuration choice, sea-ice selector, threshold,
or stabilizer lands in this round.

## Merge topology and live-tip rule

The preregistration observed source `413984f3bcab9c2bd287412d06d6a36dff02c868`.
While the round was running, the canonical GYRE branch advanced through its
VORTEX round 199 work to `d7de69d51f6392e77085aabd879ab9e0da23ebf5`.
Operator note B33 says to take the live tip even if it has advanced. Therefore:

1. `7b8876d32aa47b9e690385e46f127fae1cedafae` merges the observed Decision-84
   tip and contains parents `10d34d3e48` and `413984f3bc`.
2. `23a9911ae48f80ef4c40e4639373b06d5bd42ad8` advances that merge to the live
   source and contains parents `7b8876d32a` and `d7de69d51f`.

The literal source-SHA clause of R100-P1 is **REFUTED** by the live advance;
the binding live-tip clause is satisfied. Both the original ORCA2 parent and
the final source tip are ancestors of the measurement tree.

## Conflict disposition

Every conflicted hunk is listed below. “Union” means that neither lane's
registered behavior or provenance entry was discarded.

| Merge | Conflicted hunk | Resolution |
|---|---|---|
| Decision-84 | GYRE round-162 preregistration | Kept both lanes' prose and re-anchored the combined-tree references. |
| Decision-84 | Decision-36 receipt | Kept both lanes' prose and re-anchored the combined-tree references. |
| Decision-84 | GYRE phase-3 round-8 receipt | Kept the ORCA2 re-anchors and the incoming GYRE re-anchors. |
| Decision-84 | PR-1802 final receipt | Kept both provenance histories and re-anchored the combined tree. |
| Decision-84 | GYRE round-150 receipt | Kept both provenance histories and re-anchored the combined tree. |
| Decision-84 | GYRE round-162 cancellation receipt | Kept both provenance histories and re-anchored the combined tree. |
| Decision-84 | shared ocean driver | Semantic union: retained ORCA2's barotropic drag override and added the incoming substep Coriolis and PGF overrides. |
| Decision-84 | testcase recipe | Retained ORCA2's explicit first-WZV form, Decision-58 stage split, and spatial viscosity; added the source lane's shared fields. |
| Decision-84 | Decision-43 gate | Unioned the route registry and VORTEX cards while retaining ORCA2's z-coordinate route. |
| Decision-84 | citation gate | Unioned both `FILES` and `CITATION_MAP`; no key from either side was dropped. |
| Decision-84 | Decision-43 tests | Kept ORCA2 coverage and the incoming Decision-78 census. |
| Decision-84 | VORTEX card test | Kept the ORCA2 fold-in digest block and incoming Decision-78 assertions; selected the combined certified digest. |
| Decision-84 | vertical helper auto-merge | Replaced the incoming stale private fold-helper name with the public helper present on the ORCA2 tree; the focused test had exposed the stale name as a `NameError`. |
| round 199 | GYRE phase-3 round-8 receipt | Kept the new PGF provenance and anchored its three occurrences to the combined recipe. |
| round 199 | citation gate, recipe range | Unioned the incoming range and anchored it to the combined recipe range. |
| round 199 | citation gate, TKE entry | Kept the incoming TKE entry at its combined-tree location. |
| round 199 | citation gate, ZDF entry | Kept the incoming ZDF entry at its combined-tree location. |
| round 199 | citation gate, PGF entry | Kept the incoming three-location PGF entry at its combined-tree locations. |

The resolved citation registry contains 467 files and 1,432 map entries. A
semantic comparison retained all source-lane values, with re-anchoring only
where the combined file moved them. The default citation gate reports 274
checked citations, zero failures, zero unmapped citations, and zero map-audit
failures. Its real-key plant refuses as required.

The ORCA2 round-98 direct discriminator called the now-shared coefficient
builder without its mesh. A one-line gate compatibility repair passes the
card grid to that direct call. It changes no production model statement and
allows the record-backed control to exercise the same fold-aware builder as
production.

## GYRE gate — independent

The live source and combined-tree ten-step ladders each finish with the same
70 rows. Their residual archives have identical SHA-256
`377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
The committed offline comparator reports zero differing rows, zero maximum
worsening, and first-over-bar kt=3 on both sides. The short ladder therefore
confirms the source-tip comparison but does not rescue the year gate.

The 360-day combined-tree member completes. Against the certified source-lane
member, days 1 through 16 are array-identical and every day from 17 through
360 differs. Representative maximum absolute field movements are:

| Day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 17 | `2.0250e-13` | `3.5527e-14` | `8.2470e-14` | `8.2460e-14` | `5.1209e-14` |
| 30 | `7.4261e-9` | `2.8663e-9` | `3.0909e-10` | `8.0649e-10` | `5.4798e-11` |
| 240 | `2.0650e-8` | `3.5403e-9` | `1.5183e-10` | `2.6864e-10` | `3.8348e-10` |
| 360 | `2.1866e-8` | `7.4897e-9` | `1.7235e-9` | `1.1940e-10` | `8.4823e-11` |

The day-30/day-240/day-360 snapshot hashes all differ from the certified
member. R100-P2 is therefore **REFUTED** and the merge cannot land.

## ORCA2 rung-7 gate — given NEMO's entry

The certified kt=1..10 gate stops before trajectory scoring:

```text
REFUSE: current card no longer matches V2 at kt=1: T
```

The only T differences are 77,662 signed-zero cells; numerical maximum is
zero. The exact same refusal, count, and zero magnitude reproduce on parent
`752616c1c`, so this is a pre-existing gate incompatibility rather than a
merge-created result. Because the gate refuses before producing its row set,
no rung-7 moved-row or AT-BAR claim is made. R100-P3's rung-7 prediction is
**UNMEASURED**, not inferred from an ungated diagnostic.

## ORCA2 rung-0 read-out — independent

There is no committed rung-0 ten-step gate in this tree. The available
record-backed gate covers the entry and the complete first stage only. Its
entry T, S, u, v, and ssh rows are array-identical before and after the merge.
All five stage-1 rows move:

| Row | RMS before -> after | Maximum before -> after | Unequal cells before -> after | Direction by RMS |
|---|---:|---:|---:|---|
| T | `1.3963531e-5 -> 1.1801029e-5` | `1.4541891e-3 -> 1.3726779e-3` | `582469 -> 582469` | toward |
| S | `1.0106188e-5 -> 1.0414643e-5` | `1.1482595e-3 -> 1.1734531e-3` | `430552 -> 430551` | away |
| u | `1.2410437e-3 -> 8.5854379e-4` | `6.4217063e-2 -> 6.1715202e-2` | `444098 -> 443423` | toward |
| v | `1.1930849e-3 -> 8.9322101e-4` | `3.2637398e-2 -> 3.4021125e-2` | `440917 -> 440386` | toward RMS, away maximum |
| ssh | `6.5360018e-3 -> 6.2433846e-3` | `1.2284385e-1 -> 1.3145860e-1` | `16433 -> 16433` | toward RMS, away maximum |

Both before and after gates report `PASS_RUNG0_CARD_STAGE1`; that status means
record and card admission, not bit identity. The first non-bit boundary remains
the complete stage-1 T row. A kt=1..10 bar/first-over-bar verdict remains
**UNMEASURED_NO_COMMITTED_RUNG0_TEN_STEP_GATE**. Thus R100-P3 is only partially
confirmed and cannot satisfy its landing predicate.

## EEN discriminator after the merge — independent

The record-backed discriminator covers both rank slabs. The source-associated
coefficient application still has 398 U and 397 V signed-zero differences at
substep 1. Using the oracle coefficients makes both rows bit-exact. Substep-2
U still differs at 68 cells with maximum `2.9617669311254642e-8` at
`(j=147,i=134)`; substep-2 V remains bit-exact.

The eight coefficient read-out is unchanged from round 99: `ffu_ne/nw/se/sw`
have 3,514/3,515/3,574/3,577 signed-zero-only differences; `ffv_se/sw` have
3,585/3,584 signed-zero-only differences; `ffv_ne/nw` add 67/66 northern-fold
magnitude differences to 3,505/3,504 signed-zero differences. Therefore:

- R100-P4 is **REFUTED**: round 99's arm had already exercised this shared
  builder, so its complete counts were not stale.
- R100-P5 is **CONFIRMED**: the coefficient zero signs remain the first
  non-bit boundary; the fold magnitude debt and 68-cell substep-2 U residual
  survive as separate findings.

Both the coefficient-bit and application-bit plants refuse.

## Tests and review

- Focused conflict, card, shared-helper, and Decision-43 tests: 92 passed.
- The one `tests/ocean/fidelity -n 12` battery reached 99% and exposed the
  listed pre-existing worktree-stamp ratchet in the round-129 spread-floor
  test. The isolated ID reproduces
  `the certified year harness moved after the registered members ran`.
  The broad process then remained silent in a final stage sweep and was
  interrupted after bounded waits. It is **incomplete**, not PASS; no other
  failure is claimed absent a terminal summary.
- R100-P6 is **REFUTED** by the scientific gates and incomplete broad battery,
  although the citation gate itself passes and its plant fires.

The required separate read-only Codex review did not reach the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Bisect the combined ORCA-side shared statements that make the GYRE year
   first move on day 17 while the live-source ten-step ladder remains exact.
   Re-run the full landing gate after the owner is named; do not waive byte
   identity.
2. Repair the rung-7 signed-zero entry contract by a source-derived statement
   or gate-input correction. Do not relax exact equality.
3. Commit a rung-0 kt=1..10 gate over the admitted hierarchy frames and
   register every moved row, AT-BAR transition, and first-over-bar step.
4. Only after items 1-3 pass may the merge land and round 101 resume the
   source-ordered barotropic walk.

## UNVERIFIED

- The first combined-tree statement responsible for the day-17 GYRE movement
  is not yet named.
- Rung-7 trajectory movement is unmeasured because its entry gate refuses.
- Rung-0 kt=2..10 movement and bar classifications are unmeasured because no
  committed gate exists.
- No NEMO acquisition was requested or run.
