# ORCA2 round 335 — standalone design and S1 model preparation

Date: 2026-10-25.  Base `072e4aa1b1c8`; initial preregistration/design commit
`9194a71e6`; initial model commit `12c0e9ac6`; exact-resume implementation
commit `34c520d79`; re-review repair `44ca47e21`.  Final disposition:
**LANDED; the offline S1 boundary and non-NEMO compatibility boundary are
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

The duplicate NCAR drag/stability implementations were removed where the
operation is measured bit-identical and made selectable in shared core
kernels.  The adjacent legacy/NEMO copies of `_one_on_obukhov`, `_ch_n10`,
and `_ce_n10` deliberately remain separate: the exact arm has per-statement
barriers plus Fortran `SIGN`/`MIN`/`MAX` semantics, while the legacy arm must
reproduce the parent expression graph.  They are physically equivalent but
not word-identical, so unifying either direction breaks one of the two exact
bars.  `_virt_temp` remains shared because its arms are word-identical.  The
two host/scan copies of `blk_oce_2` are one backend-selectable implementation,
and `NemoBlkOceResult` carries `Cd`, `Ch`, `Ce`, and `rhoa`.

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

The first control was incomplete: its 17-output, 226,440-word score omitted
`q_net`.  Re-review found that the shared legacy `blk_oce_2` arm had flattened
the parent's precipitation/evaporation sub-sum.  Each affected `q_net` array
then differed on 4,142 / 13,320 ORCA2-tile words and 26,269 / 72,192
global-CORE words, with maximum absolute difference about
`4.6e-13 W m-2`, in both host and scan consumers.

The legacy arm now restores the parent's exact grouping: build
`q_precip_evap` first, then evaluate `sensible + latent + qlwn +
q_precip_evap`.  The expanded parent/current control scores every returned
output of every non-NEMO caller: all eager/JIT `air_sea_fluxes` arms; public
helpers; AMIP, land, coupler, SI3 and other core consumers; host absolute and
feedback surface forcing; both host freshwater paths; and scan surface
forcing.  Its 214-array census is **0 / 10,231,076 unequal binary64 words**.
The surface inventory explicitly includes `tau_x,tau_y,q_net,sw_down` on all
six ORCA2/global host/scan paths; freshwater includes
`precip,evap,runoff,ice_fw,restoring`.  A one-ULP plant in
`A/app_jax/q_net` exits nonzero.  The shared-caller suite passes 142/142.  The
earlier failure in
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
flux outputs.

Decision 172's premise is now explicitly **REFUTED**.  In a key-SI3 build,
`nn_ice=1` enters neither the no-ice allocator nor `ice_init`
(`sbcmod.F90:331-340`).  NEMO forcibly resolves implicit ice drag false
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:380`), TKE's case 2 reads `hm_i`
(`zdftke.F90:622`), and initialization rejects `nn_mxlice>1` for ice-if
(`zdftke.F90:794-795`).  Active icebergs are also unsafe with the key: their
copy/interpolation reads uninitialized SI3 snow/ice mass, concentration,
thickness, and velocity (`icbutl.F90:89-93`, `:174-181`).

The design records the complete viable configuration families and costs.
Their shared safety envelope keeps coupling off because `sbcice_if` otherwise
writes `a_i` (`sbcice_if.F90:105`), keeps embedded ice off because the ocean
step reads `snwice_mass*` (`stp2d.F90:213-220`), excludes the `sshice`
diagnostic (`diaar5.F90:197-200`), and accepts that freshwater budgeting
automatically switches to ocean-only volume (`sbcfwb.F90:119`).  It also
enumerates the off assimilation, observation, section, boundary, bulk/ABL,
coupler, restart and ice-writer consumers found by the full `src/OCE` grep.

## Gates and review

- O1 production JIT numerical gate: PASS, mapping and all 18 bulk fields at
  exact bar; all plants PASS_NONZERO.
- Focused literal/aerobulk tests: 36/36 PASS, including signed-zero and finite
  calm-wind/masked-fill gradients.
- OMIP/coupler/legacy/AMIP/land/SI3 shared-caller tests: 142/142 PASS.
- Complete parent/current non-NEMO control: 0/10,231,076 unequal across all
  214 arrays; one-ULP `q_net` plant PASS_NONZERO.
- Resume-2 focused bulk/applicator/scan/aerobulk tests: 72/72 PASS in 95.31 s.
  The authoritative `land.sh` ORCA2 default inventory passes 152/152 in
  1,292.61 s, foreground.  The per-receipt citation gate passes all 23
  citations with zero failures or unmapped entries; its shifted new-source
  plant exits 1.
- The dedicated CI ratchet inventory completed foreground: 12,277 PASS, 4
  SKIP, 8 FAIL in 134.45 s.  All eight are inherited audit findings in files
  outside this commit (slab-land dispatch; JRA55 test literal; land restart;
  geothermal/MLE coefficients; MLE/shortwave specs); no failure names a
  changed file.

Independent review: Claude HOLD + Argon HOLD findings addressed by this
resume; operator confirmation remains external to this receipt.

## Choices and open work

- ASKED and complete: exact JIT S1, parent-compatible shared defaults, custom
  record `SIGN`, finite gradients, unification of only bit-identical helpers
  and `blk_oce_2`, documented non-unifiable helper pairs, result
  coefficients/density, design corrections, and gates.
- UNASKED and unchanged: configure/build/run NEMO, select an ice-cover file,
  alter the shipped comparison/training card, or relax any exact bar.
- DECISION_NEEDED: ice-cover provenance remains the three-option choice in the
  design.  The pick is a pinned native-grid yearly mean derived from the NEMO
  ORCA2_ICE reference; no option is selected.
- DECISION_NEEDED: Decision 172 is refuted; reference configuration remains
  open.  The design lists five viable families.  The pick is **K0**:
  `key_si3`, `nn_ice=1`, `ln_drgice_imp=F`, `ln_mxl0=T`, `nn_mxlice=0`, and
  `ln_icebergs=F`, with the full off-consumer safety envelope.  Its three
  changed branches are independently certified by the round-324 pre-D150
  control, round-326 pre-D154/round-325 control, and accepted Phase-2d
  icebergs-off VARIANT, but their ice-if composition still requires a new boot
  and full ten-step acquisition.  K1 (`nn_mxlice=1`), K2 (`ln_mxl0=F`), N0
  (no SI3, icebergs off), and N1 (no SI3, icebergs on with the zero-ice
  fallback) all need new branch or compiled-path certification.  No option is
  selected or applied.
