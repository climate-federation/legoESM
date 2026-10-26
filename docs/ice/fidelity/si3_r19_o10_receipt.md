# SI3 Ladder Round 19 (`si3_r19`) — O10 model-preparation receipt

Date: 2026-10-26. Parent: `9abb6be028cb`. Scope: production salinity-4
VC19/RJW gravity drainage and flushing, `jpl=5`, 10 ice + 5 snow layers,
without acquisition. Preregistration commit: `0c9bc8164c`.

**Verdict: MODEL PREP COMPLETE; O10 remains UNMEASURED against NEMO.** No
master-record array was read and no acquisition tool was built. The next legal
step is the O10 projection from DECISION 173's one shared 100-step production
record; the KILL RULE still applies to any differing normal binary64 bit.

Independent review: Argon HOLD and Sonnet HOLD. Every confirmed HOLD item is
resolved below; neither review claimed master-record confirmation.

## Production identity and transcription

The record build is
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF`.
Its flags are pinned at `ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`. The
production order is TEMP1, `ice_thd_sal`, TEMP2 at
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205`.

The executing salinity-4 source is
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`:

- initialization and VC19 brine salinity are `:317-353`, including the cubic at
  `:345`;
- drainage is `:326-560`, harmonic permeability is `:377-390`, and RJW's
  product and selector are `:397-398` and `:403-422`;
- bounded drainage and its layer salt/flux updates are `:447-548` and
  `:541-544`;
- flushing is `:562-669`, with count, brine-volume gate, and salt updates at
  `:607-608`, `:616-622`, and `:637-651`;
- final bounds and residual salt flux are `:671-683` and `:677-679`.

The selectable model is split into source-order helpers and one core:

- VC19 rounding barriers are at
  `packages/ice/legoesm/ice/bitz_lipscomb.py:1634-1643`;
- the initial/repeated harmonic forms are at `:1646-1671`, and the RJW chain is
  at `:1674-1728`, including true constant divides at `:1690-1692`;
- the line-644 layer-specific object roles are isolated at `:2915-2927`;
- drainage, flushing, and final bounds are `:1739-1844`, `:1847-1917`, and
  `:1921-1937`;
- the fail-closed public entry is `:1949-1986`, and the real O9 -> SAL -> TEMP2
  category composition is `:1997-2020`.

The corrected destination/source roles match the record object at source lines
535, 538, 641, 644, and 677. In particular, line 644 has `MIN(0,zds)` roles
`dst:src = 1:3` across its unrolled copies and outer `MAX` roles `2:2`; the
model preserves those per-layer splits. Salinity-2/D5 code is untouched.

## Compiled MIN/MAX/SIGN audit

The audit contains only real ppsrc variables, covers every executing floating
site, and records one explicit verdict at
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py:12-76`.
It pins ppsrc SHA-256
`db71f3961ffd52fdce8092f9930d69a4abd2247b5b9cc744e9a00f5f9135c216`
and record-object SHA-256
`a99e605bbf4b454393a2d2b499d6779c25dfd2037ddd7e78fc22a757f8a0f72b`.
The signature, dead-arm, real-variable census, operand-count, and five mutant
checks are at
`tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py:21-169`.

The tie conclusions are now deliberately narrow:

- source line 334 uses nonzero `epsi10`, not `epsi06`;
- lines 350, 397, and 482 are `MASKED` only because checked downstream
  comparisons cannot observe the surviving zero sign;
- lines 535 and 641 are `MASKED` only because `zs_min` is read solely by the
  deck-disabled `ln_sal_chk` diagnostics;
- lines 538 and 644 are `FOUND`: signed-zero operand swaps change the flux bit,
  and the per-layer line-644 object roles are pinned;
- line 619 is `NO_TIE` only under the production TEMP1 bound proved below.

## Independent controls and review mutants

The scalar oracle is a literal transcription and shares no numerical model
helper. Its 13-column fixture hits the drain floor, flush floor,
`MIN(0,zds)`, initial harmonic early exit, a zero `zmsk` layer, both 50/5 caps,
and repeated loop bodies. The exact census is pinned at
`tests/ice/unit/test_si3_o10_salinity.py:461-498`.

Both eager and compiled calls take arrays as arguments; no constant-input JIT
closure remains. Scalar == eager == JIT is checked bit-for-bit at `:414-424`.
Four independently planted missing-barrier controls prove that unbarriered
eager and JIT differ, while the protected version agrees, at `:427-447`.
Their pinned changed-column counts are 1022 (VC19), 181 (flux increment), 3
(Rayleigh chain), and 506 (loop body). The full-output digest is pinned at
`:450-458`.

Signed-zero survivor plants for the drainage clamp and both line-644 clamps
change output flux bits at `:502-526`. The three `MASKED` downstream arguments
are separately checked at `:529-545`.

Real `jpl=5` O9 -> SAL -> TEMP2 composition runs all five categories against
the scalar oracle at `:548-603`. The separate wiring plant gives every array a
distinct identity token and checks each keyword, SAL ordering, and TEMP2 use at
`:658-724`. Required-input and selector refusal checks are `:627-655`. The
production gradient/finite-difference check is `:606-624`.

## Edge deviations and reachability

Two forward implementations intentionally avoid invalid intermediate values:

1. At NEMO source line 618, `zmsk*salinity/zs_br` can form `0/0 = NaN`; the
   compiled `MINSD` can then drop that NaN and allow flushing. The model's
   `safe_brine` instead yields zero and blocks flushing when a layer temperature
   is exactly `T0`. The relevant NEMO loop is
   `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:593-644`, and the model
   code is `packages/ice/legoesm/ice/bitz_lipscomb.py:1876-1884`.
2. After the first drainage iteration, NEMO's repeated harmonic mean has no
   early exit; a zero-mask layer can therefore produce NaN gradients. The NEMO
   repeated form is `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:455-474`
   and the model's guarded arithmetic is
   `packages/ice/legoesm/ice/bitz_lipscomb.py:1646-1671`.

Both deviations are unreachable for the checked production composition. For
every active state across all five resolved-deck categories, the composition
test proves `sz_i >= rn_simin = 0.1`; TEMP1 therefore proves
`t_i <= T0 - rTmlt*rn_simin = T0 - 0.0054 K`, strictly below the `epsi06`
zero-mask threshold. The check is part of
`tests/ice/unit/test_si3_o10_salinity.py:548-603`. The exact final-bound kink
and synthetic zero-mask columns exist only to exercise forward branches and
are explicitly excluded from the smooth production gradient claim at
`tests/ice/unit/test_si3_o10_salinity.py:606-624`. If a future admitted
record violates this bound, these deviations become reachable and model prep
must stop for an operator decision.

## Citation-gate repair

The former helper could overwrite a live citation-map key after AST duplicate
inspection. Historical O10 line shifts now use a single delete-and-reinsert
operation and reject unapplied replacement anchors at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py:10233-10270`.
The gate now also rejects `CITATION_MAP.update` hidden in any helper and counts
all top-level writes at
`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py:183-234`.

## Preregistered projection and choices

The frozen F1/F2/F3 master projection remains
`docs/ice/fidelity/si3_r19_o10_preregister.md:34-68`. No observer or acquisition
tool was added. P1-P3 remain untested; P4, independent scalar/model equality,
is confirmed.

All changes in this resume were ASKED review fixes: true JIT arguments,
barrier mutants, complete branch census, distinct wiring tokens, real five-
category composition, object-role correction, bit-changing signed-zero plants,
real-variable audit census, checked `MASKED` arguments, both edge-deviation
reachability proofs, and the citation-gate bypass repair. No configuration
choice was made.

## CONFIRMED vs PLAUSIBLE

CONFIRMED:

- Sonnet independently transcribed source lines 311-683 and found model
  arithmetic bit-exact on 6,500 random/stress columns; eager == JIT and the
  object contains no FMA, square root, or reciprocal instruction;
- every requested planted mutant now turns its local test red, including true
  JIT barrier removal, uncovered event sites, input swaps, MIN/MAX role swaps,
  signed-zero survivors, real composition, and the citation bypass;
- the record-object census and hashes regenerate, and line 644's four unrolled
  copies have the pinned operand-role split;
- the two edge deviations are unreachable under the checked five-category
  production bound; D5 remains untouched;
- no physical-input default, acquisition tool, or configuration change exists.

PLAUSIBLE:

- bit identity to the future O10 shared-master projection. It is deliberately
  not called confirmed before acquisition and admission.

## Gates

The complete CPU/x64 O10 suite passed `40 passed in 429.89s`. O9 and O10
object-audit regressions plus every non-stamping citation/default check passed
`58 passed, 2 deselected in 18.44s`; the regenerated object audit reports
`O10_MINMAX_AUDIT MATCH`. The full-namespace private-import ratchet passed
`6 passed in 5.07s`. On clean commit `a7d5dded14`, the complete citation/default
suite passed `35 passed in 6.91s`; the explicit O10 gate passed with 45
citations, no unmapped citations, no failures, and no full-map audit failures.
Shifting the VC19 citation by two lines produced `SYMBOL-NOT-AT-LINE`, gate
status `FAIL`, and exit status 1.
