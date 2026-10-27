# SI3 Ladder Round 19 (`si3_r19`) — O10 model-preparation receipt

Date: 2026-10-26. Parent: `9abb6be028cb`. Scope: production salinity-4
VC19/RJW gravity drainage and flushing, `jpl=5`, 10 ice + 5 snow layers,
without acquisition. Preregistration commit: `0c9bc8164c`.

**Verdict: MODEL PREP COMPLETE; O10 remains UNMEASURED against NEMO.** No
master-record array was read and no acquisition tool was built. The next legal
step is the O10 projection from DECISION 173's one shared 100-step production
record; the KILL RULE still applies to any differing normal binary64 bit.

The supplied Argon and Sonnet re-reviews both held. This resume resolves their
confirmed model-preparation items below; neither review claimed master-record
confirmation.

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
- the initial/repeated harmonic forms are at `:1646-1672`, and the RJW chain is
  at `:1675-1729`, including the zero-volume divisor guard at `:1687-1689`;
- drainage, flushing, and final bounds are `:1740-1843`, `:1846-1917`, and
  `:1921-1937`; line 641's role is at `:1891`, and line 644's per-layer roles
  remain at the executing call site at `:1895-1897`;
- the fail-closed public entry is `:1949-1986`, and the real O9 -> SAL -> TEMP2
  category composition is `:1997-2020`.

The corrected destination/source roles match the record object at source lines
535, 538, 641, 644, and 677. In particular, line 644 has `MIN(0,zds)` roles
`dst:src = 1:3` across its unrolled copies and outer `MAX` roles `2:2`; the
model preserves those per-layer splits. Salinity-2/D5 code is untouched.

## Compiled MIN/MAX/SIGN audit

The audit contains only real ppsrc variables, covers every executing floating
site, and records one explicit verdict at
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py:12-98`.
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
  deck-disabled `ln_sal_chk` diagnostics; their distinct object roles are still
  pinned at the real model call sites;
- lines 538 and 644 are `FOUND`: signed-zero operand swaps change the flux bit,
  and the per-layer line-644 object roles are pinned;
- line 619 is `NO_TIE` only under the production TEMP1 bound proved below.

## Independent controls and review mutants

The scalar oracle is a literal transcription and shares no numerical model
helper. Its 14-column fixture hits the drain floor, flush floor,
`MIN(0,zds)`, initial harmonic early exit, a zero `zmsk` layer, both 50/5 caps,
repeated loop bodies, and an explicit `h_i = 0`, `t_i = T0` column. The exact
census is pinned at `tests/ice/unit/test_si3_o10_salinity.py:562-599`.

Both eager and compiled calls take arrays as arguments; no constant-input JIT
closure remains. Scalar == eager == JIT is checked bit-for-bit at `:442-452`.
Four missing-barrier controls now strip `nemo_source_round` from the real VC19,
flux, RJW, and salinity-core functions, and each changes output bits while the
protected function agrees with eager, at `:455-512`. The full-output digest is
pinned at `:551-559`.

Source-compiled plants flip the real drainage, line-641, and per-layer line-644
call sites in `_si3_ice_thd_sal4_core`; the dedicated fixture reaches the
line-641/644 signed-zero ties with `sfx_bri = -0`, and every flip changes output
bits at `:603-677`. The three `MASKED` downstream arguments are separately
checked at `:680-696`.

Real `jpl=5` O9 -> SAL -> TEMP2 composition runs all five categories against
the scalar oracle and pins the reachability census at `:699-772`. The separate
wiring plant gives every array a distinct identity token and checks each
keyword, SAL ordering, and TEMP2 use at `:846-912`. Required-input and selector
refusal checks are `:815-843`. The smooth production gradient/finite-difference
check is `:775-793`, and the melted `h_i = 0` gradient check is `:796-812`.

## Edge deviations and reachability

The seed-19 five-category production composition resolves the reviewer
disagreement rather than averaging it. It produces 27 gathered columns:

- 19/27 are salinity-active (`h_i >= rn_sal_himin`), and 0/19 has any
  pre-salinity layer below `rn_simin = 0.1`;
- 3/27 are melted-out numerical-residue columns with `sz_i = 0` and every
  `t_i = T0`; thus the global lower-bound claim is false even though its
  salinity-active form is true;
- 3/27 are flush-eligible, but 0/3 of those has a `T0` layer.

These counts are computed and pinned in the real composition test at
`tests/ice/unit/test_si3_o10_salinity.py:699-772`.

Two forward implementations avoid invalid intermediate values:

1. At NEMO source line 618, `zmsk*salinity/zs_br` can form `0/0 = NaN`; the
   compiled `MINSD` can then drop that NaN and allow flushing. The model's
   `safe_brine` instead yields zero and blocks flushing when a layer temperature
   is exactly `T0`. The relevant NEMO loop is
   `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:593-644`, and the model
   code is `packages/ice/legoesm/ice/bitz_lipscomb.py:1874-1882`. This deviation
   remains unreachable in the measured composition because the overlap count
   between flush eligibility and `T0` layers is exactly 0.
2. After the first drainage iteration, NEMO's repeated harmonic mean has no
   early exit; a zero volume therefore has a forward `1/0 -> infinity -> +0`
   path whose transpose produced NaN. This path is reachable in the 3 melted
   columns above. The model now forms exact zero volume without `0/0` and stops
   the repeated harmonic mean at the same absorbing +0 without evaluating
   `1/0`, at `packages/ice/legoesm/ice/bitz_lipscomb.py:1653-1672` and
   `:1682-1689`.

The guard changes zero forward bits against the pre-fix arithmetic on the full
14-column fixture and deterministic 12,000-column fuzz for both harmonic forms,
as proved at `tests/ice/unit/test_si3_o10_salinity.py:516-548`. The full fixture
digest is also pinned. The explicit `h_i = 0`, all-layer `t_i = T0` column has
finite gradients, with temperature gradient exactly zero and salinity gradient
exactly one, at `:796-812`.

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

All changes in this resume were ASKED review fixes: corrected audit labels,
real-model barrier plants, real-call-site operand-role plants, the measured
five-category reachability census, a forward-neutral permeability gradient
guard, and the explicit melted-column gradient fixture. No configuration
choice was made.

## CONFIRMED vs PLAUSIBLE

CONFIRMED:

- Sonnet independently transcribed source lines 311-683 and found model
  arithmetic bit-exact on 6,500 random/stress columns; eager == JIT and the
  object contains no FMA, square root, or reciprocal instruction;
- every requested barrier and operand-role mutant now turns its local test red
  in the real salinity functions/call sites, including the line-641/644 ties;
- the record-object census and hashes regenerate, and line 644's four unrolled
  copies have the pinned operand-role split;
- the flush `safe_brine` deviation is unreachable in the measured production
  composition (0/3 flush-eligible columns have a `T0` layer); the repeated
  permeability gradient defect is reachable in 3/27 melted columns and is now
  guarded with zero forward-bit changes on the full fixture plus fuzz;
- no physical-input default, acquisition tool, or configuration change exists.

PLAUSIBLE:

- bit identity to the future O10 shared-master projection. It is deliberately
  not called confirmed before acquisition and admission.

## Gates

The complete CPU/x64 O10 suite passed `46 passed in 673.71s`. The O9/O10
object-audit regressions passed `20 passed in 15.10s`, and the regenerated
object audit reports `O10_MINMAX_AUDIT MATCH`. The full-namespace
private-import ratchet passed `6 passed in 5.19s`. The non-stamping full-map and
all-SI3-receipt citation checks passed `21 passed, 14 deselected in 3.49s`.
