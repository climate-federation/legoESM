# SI3 Ladder Round 19 (`si3_r19`) — O10 model-preparation receipt

Date: 2026-10-26. Parent: `9abb6be028cb`. Scope: production salinity-4
VC19/RJW gravity drainage and flushing, `jpl=5`, 10 ice + 5 snow layers,
without acquisition. Preregistration commit: `0c9bc8164c`.

**Verdict: MODEL PREP COMPLETE; O10 remains UNMEASURED against NEMO.** No
master-record array was read and no acquisition tool was built. The next legal
step is the O10 projection from DECISION 173's one shared 100-step production
record; the KILL RULE still applies to any differing normal binary64 bit.

independent review: operator (Claude)

## Search and production identity

I searched `bitz_lipscomb.py` for `ice_thd_sal`, `nn_icesal`, `drain`,
`flush`, `VC19`, `RJW`, `zv_br`, `zperm`, `sfx_bri`, and `sfx_res`. The shared
D5 column already had the option-2 bulk-timescale update; O9 already carried
salinity-4 layer state through ZDF/DH/TEMP, but there was no salinity-4
`ice_thd_sal` transcription. O10 is therefore in that shared module and
composes the public O9 exit; it does not duplicate the D5 path.

The record build is
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF`.
Its exact flags are pinned at
`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`. The production call sequence says
verbatim “`CALL ice_thd_temp`”, “`CALL ice_thd_sal`”, then
“`CALL ice_thd_temp`” at
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205`.

Every executing statement is covered by the compiled ppsrc range
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`:

- `:311` is verbatim “`CASE( 4 )`”. Initialization and VC19 brine salinity are
  `:317-353`; the cubic assignment is verbatim
  “`zs_br(jk) = -18.7_wp * zt1 - 0.519_wp * zt2 - 0.00535_wp * zt3`” at
  `:345`.
- Drainage is `:326-560`. Harmonic effective permeability is `:377-390`, the
  RJW Rayleigh product is `:397-398`, the RJW selector/velocity is `:403-422`,
  bounded subcycling is `:447-548`, and the layer salt/flux assignments are
  verbatim “`sz_i_1d(ji,jk) = sz_i_1d(ji,jk) + zds(jk)`” and
  “`sfx_bri_1d(ji) = sfx_bri_1d(ji) - rhoi * ...`” at `:541-544`.
- Flushing is `:562-669`: the meltwater velocity is `:601-602`, the bounded
  count is `:607-608`, the brine-volume gate is `:616-622`, and the salt
  clamp/update/flux is `:637-651`.
- Final salinity bounds and residual salt flux are `:671-683`; the assignments
  are verbatim “`sz_i_1d(ji,jk) = MIN( MAX( rn_simin, ...`” and
  “`sfx_res_1d(ji) = sfx_res_1d(ji) - rhoi * ...`” at `:677-679`.

The dead option-2 statements are `:277-292`; the dead GN/CW selector arms are
`:424-435` and `:509-521`. They are inventoried by the object audit but are not
transcribed as selectable production behavior.

## Model transcription

The selectable implementation is
`packages/ice/legoesm/ice/bitz_lipscomb.py:1634-1939`. It includes:

- VC19 source-order cubic evaluation at `:1634-1643`;
- the initial early-exit and repeated harmonic-mean permeability forms at
  `:1646-1671`;
- RJW Rayleigh and velocity evaluation at `:1674-1728`, with true constant
  divides at `:1690-1692`;
- drainage, its 50-bound conditional subcycles, layer salt and salt flux at
  `:1739-1845`;
- flushing, its 5-bound subcycles and brine-volume gate at `:1847-1919`;
- final salinity bounds and residual salt flux at `:1921-1939`.

Every multiply that feeds an add/subtract has its own `nemo_source_round`; the
constant reciprocals use `_div_const`. The executing production arm contains
no `EXP`, `LOG`, `POW`, `SQRT`, reciprocal-root, or `SIGN`, so DECISION 162's
glibc arm is inert here.

The public entry requires every physical input and refuses every nonproduction
selector and non-10-layer shape at
`packages/ice/legoesm/ice/bitz_lipscomb.py:1950-1995`. The category composition
calls O9, O10, and the second temperature reconversion in that order at
`:1998-2021`. No configuration was changed.

## Compiled MIN/MAX/SIGN audit

The O9 object machinery was generalized, not copied. O10 selects only
`ice_thd_sal` and records one tie verdict per executing floating site at
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py:12-70`.
The four integer count sites (`447`, `448`, `607`, `608`) are explicitly
censused at `:37-39`.

The committed report pins:

- ppsrc SHA-256
  `db71f3961ffd52fdce8092f9930d69a4abd2247b5b9cc744e9a00f5f9135c216`;
- record object SHA-256
  `a99e605bbf4b454393a2d2b499d6779c25dfd2037ddd7e78fc22a757f8a0f72b`;
- all 25 floating source sites, all four integer sites, no `SIGN`, and no
  instruction without a source token.

The gate pins every site's full statement/instruction/operand-role signature,
the exact dead-arm set, and five planted violations at
`tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py:21-123`. The production
floating sites are `334,350,367,397,412,444,482,497,534,535,538,569,605,619,
637,641,644,677`; every one has `NO_TIE` or `MASKED` evidence. NEMO's own
source/destination survivor roles, including `-0`, come from the record
object—not from Python `min`/`max` assumptions.

## Tests and controls

The independent scalar oracle is literal Python scalar arithmetic and shares
no numerical model helper. The committed tests pin:

- real `jax.disable_jit()` and production JIT, both bit-for-bit against every
  scalar-oracle output at `tests/ice/unit/test_si3_o10_salinity.py:266-277`;
- inexact values and output digest
  `60b8eb225a171059d4e25f39b99f000b3c41b2b0127f1d57af304da67fbc68c3`
  at `:280-288`;
- real drainage/flushing/inactive and multi-subcycle coverage at `:291-299`;
- finite nonzero reverse gradient and central difference at `:302-315`;
- all 20 required inputs and all production-selector refusals at `:318-346`;
- a call-removal-sensitive O9 -> SAL -> TEMP2 production composition plant at
  `:349-387`.

Foreground results:

| Control | Result |
|---|---|
| O10 scalar/model/JIT/AD/composition | `33 passed in 205.52s` |
| O10 record-object audit | `8 passed in 1.69s` |
| generalized O9 audit regression | `10 passed in 12.84s` |
| D5 C1D year, parent vs current, production CPU JIT | 18/18 arm hashes identical; shipped arm `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` on both; `D5_PLATFORM_FLOOR` on both |
| all `tests/ice/unit` | `586 passed, 1 skipped, 1 xfailed`; one shared `sum_first_category` plant failed identically on current and parent when rerun alone |

The last row is a pre-existing control defect, not hidden as green: both exact
trees produce `mismatches == []` for
`test_si3_surface.py::test_production_path_plants_fire[sum_first_category]`.
O10 does not touch that path. The stronger full-year D5 control is byte
identical on every arm.

## Preregistered master projection

The committed preregistration freezes F1 at TEMP1/O10 entry, F2 at salinity
exit, and F3 at TEMP2 exit for all 100 steps and five categories at
`docs/ice/fidelity/si3_r19_o10_preregister.md:34-68`. This round creates no
observer or acquisition tool. P1-P3 remain untested until the shared master
record is admitted; P4 (independent scalar/model equality) is confirmed.

## CHOICES (ASKED/UNASKED)

- ASKED: production salinity-4, VC19/RJW, drainage+flushing, 50/5 bounded
  subcycles, layer salt, `sfx_bri`, `sfx_res`, TEMP2 reconversion, jpl=5 and
  10+5 layers.
- ASKED: one shared 100-step master-record projection, preregistered without
  acquisition; exact-input claim only.
- ASKED: compiled object audit of every executing MIN/MAX/SIGN site, no-FMA
  and constant-division barriers, independent scalar, eager/JIT, AD/FD,
  plants, no defaults, parent/current caller controls.
- UNASKED, mechanical: keep salinity-2 D5 code untouched and add salinity-4 as
  a sibling selectable stage in the already shared module; generalize the O9
  audit engine; expose bounded-subcycle diagnostics for testing. Offered for
  revert.
- No configuration choice was made.

## CONFIRMED vs PLAUSIBLE

CONFIRMED:

- every executing salinity-4 statement is transcribed in NEMO source order;
- independent scalar == eager == JIT for every tested output bit, including
  real drainage/flushing and bounded subcycles;
- gradients are finite/nonzero and agree with the central difference;
- the compiled record object/ppsrc hashes and every MIN/MAX/SIGN site are
  pinned; dead selector arms are explicit; all plants fire;
- D5 C1D's 18 year hashes, including the shipped `b7a7329b...` arm, are
  parent/current identical;
- no physical-input defaults, acquisition tool, or configuration change.

PLAUSIBLE:

- the hand-written IEEE arguments behind `NO_TIE`/`MASKED`; their object roles,
  site coverage, source statements and plants are machine checked;
- bit identity to the future O10 shared-master projection. It is deliberately
  not called confirmed before acquisition/admission.

## Gates

The mandatory full SI3 citation map, default receipt gate and private-import
ratchet are run after this receipt is mapped and committed. Every citation
shifted by the O10 insertion is re-anchored in both the historical receipts and
the live full-map gate; its planted shift audit remains mandatory.

