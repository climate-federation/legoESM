# Round 60 pre-code self-review — kt=2 TKE closure walk

Date: 2026-09-11. Frozen model parent `1a695951be1396abdd4d7a85f57b31de9c0917df`;
governing preregistration is round 57. The R59 record passes the full
calibration before this review: `en` 0/17,400, `zmxlm` 0/18,000, `zmxld`
0/18,000, Prandtl 0/17,400, and `avm`/`avt`/`dissl` each 0/18,000 unequal
consumed wet interfaces. The gate output is
`phase3/round60/round60_tke_calibration.json`.

## Independent round-56 findings, re-audited

| finding | current evidence | disposition before round-60 code |
|---|---|---|
| z-star cards raised without `is_active` | The compiled surface anchor multiplies `taum` by `tmask(:,:,1)` at `GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:612-619`; the production caller now obtains that mask from `state.land_mask.data` when the coordinate has no active mask. | CONFIRMED finding; FIXED in round 57 and covered by the full recipe-step test. |
| legacy `nemo_v1` floor move omitted | Its choice 3 takes the shared derived floor, while the prior configured value was `1e-8`; NEMO derives `rmxl_min` from `rn_ediff` and `rn_emin` in the compiled initializer. | CONFIRMED; REGISTERED as non-ULP UNMEASURED-with-spec debt in the governing preregistration. |
| all focused tests reported green | The named RK3 pairing test is red at `3a8d94ea1985` and remains a known pre-existing red, not a round-56 result. | CONFIRMED; receipt corrected in round 57. |
| shape plant did not reach the new guard | Its old stamped-extent mutation reached the fixed GYRE-shape check first. | CONFIRMED; FIXED to alter a decoded array shape under unchanged extents. |
| receipt parent/stamp were stale | Git history gives round 56 parent `03098e6fe91c`; the record stamp names the committed producer. | CONFIRMED; FIXED in round 57. |
| writer citation/check and diagnostic floor reads | Compiled R59 declares `zmxlm,zmxld` at `zdftke.f90:584`; round 57 corrected the source comment to MY_SRC line 571 and removed the reviewed tautological `SIZE(zdiag,...)` check. The remaining explicit-shape `zmxlm` guard was not the reviewed check. Both DINO diagnostics now call `_mixing_length_floor`. | CONFIRMED; FIXED in round 57. |

The review therefore leaves no confirmed round-56 blocker to repair before the
walk. The R59 writer's corrected RHS dummy is compiled as the same no-halo
domain as `en`, and the independent reconstruction of the solve is now exact.

## Before-code operator audit

The instantiated GYRE card selects `tke_mxl_choice=3`, binary64, and
`tke_mxl_raw_evaluation="factored"`. The compiled NEMO statement instead
evaluates `zrn2=MAX(rn2,rsmall)` followed by
`SQRT(2._wp*en/zrn2)` at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:627-630`.
This is a **candidate**, not a conclusion: only the frozen cumulative
substitution gate may name it, and no configuration or physics line changes
before that score.

The implementation work will extend the existing R54/R59 operand reader, not
create a second parser. It will invoke the production mixing-length and
diffusivity functions with the GYRE card's instantiated TKE config, score only
the 17,400 interfaces copied by `zdfphy` (`zdfphy.f90:348-354`), require exact
uint64 equality, and carry a one-wet-operand plant that exits nonzero.

ASKED: this source audit and pre-code self-review. UNASKED: none. No NEMO
source/data, model physics/configuration, trajectory, or gate was changed.

## First discriminator result and frozen continuation

The committed first pass rejected the raw-length candidate: even after the
recorded NEMO mixing length is substituted, the production path remains
unequal; the first exact cumulative row is `prandtl_factor`. Source inspection
locates the remaining association: NEMO stores `pdlr` and later multiplies
`pdlr*p_avt` (`zdftke.f90:412,699-701`), while legoESM computes
`Pr=1/pdlr` and then divides `K_M/Pr`. The continuation is frozen before its
admissible score: expect `prandtl_factor` to be first exact, and land a direct
inverse-Prandtl multiplier shared by both production callers. A different
first exact row refutes this continuation.

The committed post-landing walk did refute a complete raw-stage match: after
activating the literal raw expression and direct `pdlr` multiplication, the
first exact row moved only to the recorded mixing-length substitution. Reading
the compiled bound at `zdftke.f90:674-676` shows why: NEMO starts with the
untouched `zmxlm(jpk)=rmxl_min` but its first upward iteration **updates
`jk=jpkm1`** using `rmxl_min+e3t(jpk)`. The literal legoESM branch returned the
seed itself at `jpkm1`. Before rescoring, the continuation is frozen: restore
that first `jpkm1` update in the single shared scan; expect the exact row to
move at least to `matrix_rhs_sweep_en_floor`. Failure to do so refutes it.

That committed score is 744 unequal cells in the unsusbstituted production
row and 0/17,400 immediately after the recorded matrix/RHS/sweep result is
substituted. The carried shear, buoyancy, dissipation, matrix, and solver
selectors are already literal; the GYRE card alone still selects the
vectorized Langmuir association. Compiled R56TKE evaluates the PE sum, reverse
`imlc` choice, `zus3`, and source update in loop order at
`zdftke.f90:318-380`. The next discriminator is frozen: selecting the existing
shared `nemo_literal` Langmuir path, while threading `bottom_level` for its
compiled `mbkt+1` fallback, must take the unsusbstituted production row to
0/17,400. Any remaining unequal cell refutes sole ownership and stops further
physics landing.

The literal walk reduced that row from 744 to 645 unequal cells but did not
make it exact: the preceding prediction is **REFUTED**. The remaining active
line-378 operand is `SIN(rpi*depth/zhlc)`. The same linked glibc-2.34 vector
sine transcription already present in the shared TKE module is the final
frozen discriminator. Applying it inside the literal Langmuir implementation,
together with the preceding literal walk, must make the full row 0/17,400.
Any unequal cell refutes this continuation and stops the Langmuir landing.
