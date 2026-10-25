# ORCA2 round 335 — standalone design and S1 model preparation

Date: 2026-10-25.  Base `072e4aa1b1c8`; initial preregistration/design commit
`9194a71e6`; initial model commit `12c0e9ac6`; exact-resume implementation
commit `34c520d79`.  Final disposition: **LANDED; the offline S1 boundary is
certified at the exact bar**.  No configuration changed and no NEMO executable
ran.

## Result

The literal NEMO arm is selectable in the existing shared bulk module while
all public/default callers retain the parent arithmetic.  The executing
`blk_oce_1` and `blk_oce_2` statements remain sourced from
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:791-948` and
`:1019-1045`; the NCAR five-iteration loop and helpers remain sourced from
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:125-238`
and `:241-381`.
The pressure, Exner, density, latent heat, Goff saturation, Obukhov, bulk
assembly, roughness, and neutral-wind operations remain sourced from
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:250-409`,
`:443-506`, `:548-581`, `:645-766`, `:898-940`, `:1019-1042`,
`:1062-1080`, and `:1198-1213`.

The final XLA debt was not fixed by one barrier per assignment.  Operand
barriers now also prevent multiply-add contraction, constant-division
reciprocal multiplication, `x/sqrt(x)` regrouping, and cross-helper fusion.
The first remaining mismatch localized to `One_on_L`; after materializing its
virtual-temperature numerator and denominator, every truncated NCAR iteration
0--5 produced eager/JIT-identical `Cd,Ch,Ce,Ub`.  A second localization found
the Goff `5.028*log10(T/T0)` contraction.  The complete result is now
eager/JIT-identical for every returned field.

The record build's own `SIGN` uses `b >= 0`, so `-0` selects positive
magnitude.  The shared literal helper and a dedicated `-0` regression test
now pin that behavior.  Calm-wind magnitude and masked zero-temperature inputs
use safe inactive operands; both eager and JIT reverse-mode gradients are
finite without changing any primal value.

The duplicate NCAR drag/stability implementations were removed in favor of
selectable exact arms in the shared core kernels.  The two host/scan copies of
`blk_oce_2` are one backend-selectable implementation.  `NemoBlkOceResult`
now carries `Cd`, `Ch`, `Ce`, and `rhoa` as the design requires.

## Exact offline score

The registered parent replay reproduced 48,051 unequal words.  The rejected
first-round literal replay reproduced 21,652 unequal registered words and
34,769 unequal words across the expanded 18-field census, with latent heat's
maximum residual reduced from `11.997116616015276` to
`2.0463630789890885e-12 W m-2`.

The final production CPU JIT replay is **0 / 158,292 unequal** across all 18
defined wet bulk fields: every field is individually at bar.  The upstream
O1 mapping boundary is independently **0 / 119,880 unequal** across its nine
fields.  Every bulk and mapping output retains a one-ULP binding plant through
the real scorer.  Execution is CPU, binary64, JIT enabled, with the scalar
glibc transcendental policy and immutable record digest
`751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`.

## Non-NEMO compatibility control

The rejected implementation changed the NCAR public output everywhere.  The
review's full-field control measured latent-heat changes up to about
`59 W m-2` globally and `12 W m-2` on 72 negative-humidity ORCA2 wet cells,
with the latent-heat slope reversal producing roughly `1e-5` relative changes
elsewhere.  An independent replay on the immutable O1 frame reproduced the
`11.997116616015276 W m-2` wet-cell maximum; away from the humidity-floor
tail, the median relative change was `1.31e-5`.

That change is fully removed from non-NEMO callers.  A real checkout of base
`072e4aa1b1c8` and the final tree consumed identical CORE O1 arrays.  NCAR,
legacy `ly09_2coeff`, potential temperature, pressure, moist density,
saturation humidity, shared drag, and both stability functions were
bit-identical: **0 / 226,440 unequal binary64 words across 17 outputs**.  This
control covers the two OMIP driver/coupler paths and the shared defaults used
by AMIP, land, and SI3 surface code.  The shared-caller suite passes 142/142.
The earlier failure in
`test_aerobulk_oracle.py::test_aerobulk_convention_L_matches_explicit_injection`
is fixed; the focused source/oracle suite passes 36/36.

## Corrected standalone design

The design is now five rungs.  New S0-F owns `fld_read` temporal record
selection/interpolation for ice cover, runoff, chlorophyll, and other
time-varying inputs; the prior O1 result certifies only spatial interpolation
and rotation with temporal interpolation disabled.  New S0-M owns the
stateful two-step `sbc_ssm` mean on every step for `nn_fsbc=2`, before S1.
The source execution order remains anchored by
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:428-568`.

The ice input is corrected from monthly to a yearly mean: `sn_ice%freqh=-12`
means one yearly record.  S3 now records the `sbcice_if` mutation of
`ts(:,:,:,Kmm)` on the full `A2D(nn_hls)` halo as well as its owned-interior
flux outputs.  The design also no longer assumes that dropping `key_si3`
leaves the ocean unchanged.  Compile-time SI3 branches alter TKE/GLS surface
mixing, implicit top drag, bulk/ABL allocation, and coupling paths even when
the selected runtime ice model is `nn_ice=1`.

## Gates and review

- O1 production JIT numerical gate: PASS, mapping and all 18 bulk fields at
  exact bar; all plants PASS_NONZERO.
- Focused literal/aerobulk tests: 36/36 PASS, including signed-zero and finite
  calm-wind/masked-fill gradients.
- OMIP/coupler/legacy/AMIP/land/SI3 shared-caller tests: 142/142 PASS.
- Parent/current real-CORE control: 0/226,440 unequal.
- Changed-path default gates: 36/36 focused, 142/142 shared callers, and 16/16
  in-tree SI3 PASS.  The full ratchet sweep has 14,585 PASS, 2,359 SKIP, and
  23 base-tree failures outside this diff (package-root discovery and existing
  AMIP/land/ocean/config findings); no failure names a changed file.  The
  round citation gate and its plant controls are run from the final clean
  documentation commit.

Independent review: Claude HOLD + Argon HOLD findings addressed by this
resume; operator confirmation remains external to this receipt.

## Choices and open work

- ASKED and complete: exact JIT S1, parent-compatible shared defaults, custom
  record `SIGN`, finite gradients, removal of duplicate drag/stability and
  `blk_oce_2`, result coefficients/density, design corrections, and gates.
- UNASKED and unchanged: configure/build/run NEMO, select an ice-cover file,
  alter the shipped comparison/training card, or relax any exact bar.
- DECISION_NEEDED: ice-cover provenance remains the three-option choice in the
  design.  The pick is a pinned native-grid yearly mean derived from the NEMO
  ORCA2_ICE reference; no option is selected.
- DECISION_NEEDED: reference build identity remains open.  The pick is
  `key_si3` compiled with `nn_ice=1`, preserving the comparison card's ocean
  branches at the cost of SI3 linkage/initialization and a mandatory boot plus
  ten-step smoke gate.  Removing `key_si3` requires a separately named build,
  zero-step comparison, and full ten-step recertification; carrying both builds
  doubles acquisition and long-run cost.  No option is selected.
