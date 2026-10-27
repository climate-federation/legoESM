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
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py:12-100`.
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
- line 535 is `MASKED` because `zs_min` is read solely by the deck-disabled
  `ln_sal_chk` diagnostic;
- lines 538 and 641 are `NO_TIE`: the former line-538 plant depended on an
  unordered NaN, and the former line-641 plant depended on impossible
  negative-zero layer salinity; both fixtures are removed;
- line 644 is `FOUND`: finite, subfreezing columns under the unchanged
  production namelist reach a real `-0/+0` clamp tie and flip the brine-flux
  zero bit when either reachable role is swapped;
- line 619 is `NO_TIE` under the analytic production TEMP1 bound proved below.

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

Source-compiled plants flip the two reachable line-644 roles in
`_si3_ice_thd_sal4_core`. The two-column fixture uses `h_i = 1 m`,
`t_i = T0 - 0.1 K`, `sz_i = rn_simin = 0.1`, melting of `-0.001 m`, the
unchanged production namelist, and initial `sfx_bri = {-0,+0}`. Both columns
flush once; the protected result is `{+0,+0}`, while either role mutant gives
`{-0,+0}`. The former NaN and zero-salinity role fixtures no longer exist.
The real-call test is at `tests/ice/unit/test_si3_o10_salinity.py:603-668`.

The three `MASKED` zero paths are now driven through `si3_ice_thd_sal`, not
only through scalar argument algebra. Two melted boundary columns carry `+0`
and `-0` salinity into the permeability numerator and `-0` VC19 brine into the
`zs_brmax` accumulation; their complete model outputs agree bit-for-bit at
`tests/ice/unit/test_si3_o10_salinity.py:671-704`.

Real `jpl=5` O9 -> SAL -> TEMP2 composition runs all five categories against
the scalar oracle and pins the reachability census at `:740-824`. The separate
wiring plant gives every array a distinct identity token and checks each
keyword, SAL ordering, and TEMP2 use at `:898-964`. Required-input and selector
refusal checks are `:867-895`. The smooth production gradient/finite-difference
check is `:827-845`, and the melted `h_i = 0` gradient check is `:848-864`.

## Edge deviations and analytic reachability

The production exclusion of `t_i = T0` at flushing is analytic, not inferred
from the former `0/3` synthetic overlap. NEMO first bounds layer salt volume in
`ice_cor(1)` at
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:113-121`. For positive ice
volume, gathering divides that bounded salt volume by the positive layer
volume, hence `sz_i >= rn_simin = 0.1`. The production SAS `sosaline` field has
minimum positive value `4.484964847564697 g/kg`, so its upper bound
`rn_sinew*sss` is always above `rn_simin`. The enthalpy construction and
conservative remap at
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:952-960` and `:1040-1046`
preserve positive salt and physical nonnegative enthalpy.

TEMP1 then follows
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:279-291`. For `h_i > 0` and
`sz_i > 0`, write `ztmelts = -m` with `m > 0`. Its discriminant is
`zbbb**2 + 4*rcpi*rLfus*m`, strictly greater than `zbbb**2`; therefore
`zccc > abs(zbbb)`, `zbbb + zccc > 0`, and the returned temperature is strictly
below `rt0`. Exact `rt0` can occur only on the explicit `h_i <= 0` arm or when
`sz_i == 0` permits exact quadratic cancellation. Flushing requires
`h_i >= rn_sal_himin = 0.1` at
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:569-576`, so neither case
is flush-eligible.

The committed fp64 scan calls TEMP1 on 2,000,000 states: 1,000 energies from
`0` through `1e9 J/m3` crossed with 2,000 salinities from `0.1` through
`35 g/kg`. It finds no exact `T0` and minimum `T0 - t_i =
5.399650853860294e-3 K` (5.40e-3 K). It also pins the two legal `T0` boundary
examples. The test is
`tests/ice/unit/test_si3_o10_salinity.py:707-737`.

Argon's
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:1045` observation identifies
why that line alone is not a lower-bound proof: it has a `MAX(0,...)`. The
lower bound comes from the upstream `ice_cor(1)` floor plus conservative
overlap, not from the `MAX` token. In the seed-19 composition, all 19
salinity-active states have
`sz_i >= 0.1`. Of the eight states below the salinity threshold, five are
thin positive columns (`h_i = 0.0348..0.0380 m`) with positive salt and are
excluded from flushing by thickness; the other three are gone-ice numerical
residues (`h_i = 2.17e-19..2.39e-18 m`, `sz_i = 0`, `t_i = T0`) with no melt
and are excluded by both thickness and melt. Exactly three other columns are
flush-eligible. These state classes are pinned at
`tests/ice/unit/test_si3_o10_salinity.py:740-824`.

Two forward implementations avoid invalid intermediate values:

1. At NEMO source line 618, `zmsk*salinity/zs_br` can form `0/0 = NaN`; the
   compiled `MINSD` can then drop that NaN and allow flushing. The model's
   `safe_brine` instead yields zero and blocks flushing when a layer temperature
   is exactly `T0`. The relevant NEMO loop is
   `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:593-644`, and the model
   code is `packages/ice/legoesm/ice/bitz_lipscomb.py:1874-1882`. This deviation
   is unreachable by the analytic TEMP1 and salinity-floor argument above;
   the composition census is only a regression example, not its proof.
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
exactly one, at `:848-864`.

## Citation-gate repair

The former helper could overwrite a live citation-map key after AST duplicate
inspection. Historical O10 line shifts now use a single delete-and-reinsert
operation and reject unapplied replacement anchors at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py:10234-10271`.
The gate now also rejects `CITATION_MAP.update` hidden in any helper and counts
all top-level writes at
`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py:183-234`.

## Preregistered projection and choices

The frozen F1/F2/F3 master projection remains
`docs/ice/fidelity/si3_r19_o10_preregister.md:34-68`. No observer or acquisition
tool was added. P1-P3 remain untested; P4, independent scalar/model equality,
is confirmed.

All changes in this resume were ASKED review fixes: reachable line-644 role
plants, removal of the NaN/zero-salinity pseudo-ties, the analytic TEMP1 proof
and two-million-state scan, a model-driven masked-zero test, and an exact
classification of the eight thin/residual composition states. No model or
configuration code changed.

## CONFIRMED vs PLAUSIBLE

CONFIRMED:

- Sonnet independently transcribed source lines 311-683 and found model
  arithmetic bit-exact on 6,500 random/stress columns; eager == JIT and the
  object contains no FMA, square root, or reciprocal instruction;
- both reachable line-644 role mutants turn the real salinity call red on the
  finite production-NML columns; line 538's NaN and line 641's impossible
  negative-zero pseudo-ties are explicitly no longer claimed;
- the record-object census and hashes regenerate, and line 644's four unrolled
  copies have the pinned operand-role split;
- the flush `safe_brine` deviation is unreachable by the analytic TEMP1 proof
  and the two-million-state scan; the repeated
  permeability gradient defect is reachable in 3/27 melted columns and is now
  guarded with zero forward-bit changes on the full fixture plus fuzz;
- no physical-input default, acquisition tool, or configuration change exists.

PLAUSIBLE:

- bit identity to the future O10 shared-master projection. It is deliberately
  not called confirmed before acquisition and admission.

## Gates

The complete CPU/x64 O10 suite passed `44 passed in 617.62s`; the focused
reachable-role, masked-zero, two-million-state, and composition selection
passed `5 passed in 281.52s`. The regenerated object audit reports
`O10_MINMAX_AUDIT MATCH`. On clean implementation commit `8bceba2dab`, the
O9/O10 object-audit regressions, full-namespace private-import ratchet, default
citation gate, and all-SI3-receipt checks passed `61 passed in 26.80s`. The
explicit O10 gate passed with 56 citations, no unmapped citations, no failures,
and no full-map audit failures. Shifting the new TEMP1 scan citation by two
lines produced `SYMBOL-NOT-AT-LINE`, gate status `FAIL`, and exit status 1.
