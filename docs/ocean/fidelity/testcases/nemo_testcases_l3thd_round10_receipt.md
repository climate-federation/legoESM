# Lane 3b round-10 receipt — SI3 bulk bit closure

Date: 2026-09-04
Tracker: `climate-federation/legoESM#1699`
Starting commit: `4e29046372739009e473f8ebe053844ee63819a9`
Recovery git: `/tmp/codex-si3thd-localgit`

## Outcome

The 16,494 Round-9 ordinary-arithmetic non-bit rows were **CONFIRMED source
association debt and FIXED**.  The accepted Python 3.13.0, JAX/jaxlib 0.10.0,
NumPy 2.4.4 CPU/fp64 run now has 271,542/271,560 bit-identical rows and exactly
18 non-bit rows: six each in `POST_BLK_ICE_2.albedo`, `qsr_ice`, and `qsr_tot`.
All 271,560 rows remain at the pointwise `1e-15` normalized bar.  The residual
18 are owned by JAX `exp` in `icealb.F90:167-169` and remain
**AWAITING_LIBM_POLICY**; this lane did not create a math policy.

Rung 3.6 remains design-only and was not implemented.

## Search and preregistration

The preregistration is commit `dc1029f5fa8`.  The pre-implementation search
covered core precision/bulk machinery, ocean EOS, thermo, the ice package,
column card, and existing gates/tests.  The current tree had no shared source
rounder; imported GYRE history had one private
`ocean.eos._nemo_source_round`.  Its identity/custom-JVP mechanism was promoted
once to `core.precision.nemo_source_round`; no private cross-package import and
no second bulk, albedo, saturation, or rounding implementation was added.

## Source association ownership

The pre-edit kt6236 NumPy probe reproduced NEMO's statement order exactly:
oracle and replay were `5.270331800527903e-10`, bits `3e021bd537764c00`;
the unguarded legoESM tree was `5.270331800526845e-10`, bits
`3e021bd537764800`.  This **CONFIRMS** the preregistered operation-order owner.

Each NEMO assignment below is represented by the same association and a
source-statement rounding boundary.  The private `_source_round=False` arm
changes only that boundary and restores every baseline count.

| formula/output | NEMO statement | legoESM executing lines | before non-bit | after non-bit |
|---|---|---|---:|---:|
| `blk_ice_1` wind/stress, `utau_ice` | `sbcblk.F90:1090,1145-1147` | `core/bulk_flux.py:1564-1579` | 2,821 | 0 |
| `blk_ice_1` `vtau_ice` | `sbcblk.F90:1145-1147` | `core/bulk_flux.py:1571-1579` | 2,797 | 0 |
| Goff ice saturation/derivative | `sbc_phy.F90:665-711,727-790` | `thermo.py:373-418` | 0 | 0 |
| no-pond `ice_alb` | `icealb.F90:124-185` | `ice/sea_ice.py:498-607` | 6 `exp` rows per affected output | unchanged; 18 total `exp` rows |
| `blk_ice_2` heat/moisture | `sbcblk.F90:1231-1273` | `ice/sea_ice.py:610-695,804-830` | 0 | 0 |
| `evap_ice` | `sbcblk.F90:1279,1288` | `ice/sea_ice.py:697-703` | 3,879 | 0 |
| `devap_ice` | `sbcblk.F90:1279,1289` | `ice/sea_ice.py:697-703` | 3,852 | 0 |
| `emp_ice` | `sbcblk.F90:1298` | `ice/sea_ice.py:707-738` | 1,900 | 0 |
| `emp_tot` | `sbcblk.F90:1299` | `ice/sea_ice.py:723-738` | 1,232 | 0 |
| `ice_flx_other` friction/basal `fhld` | `icesbc.F90:328-340,358-366,391-405` | `ice/sea_ice.py:833-977` | 13 (`fhld`) | 0 |

A codex-internal source review found that the first receipt/gate census stopped
at Stage-1 field 44.  The corrected census also scores fields 45--49
(`qsat_ice`, `theta_ice`, `qlw_ice`, `qsb_ice`, `dqlw_ice`) on all 8,760 steps:
43,800 additional comparisons, all bit-identical.  The same review found that
the friction diagnostic had the wrong association and omitted its mask; a
non-vacuous `mask=0.25` replay now checks the literal
`zdrag * (0.5 * (u-square + v-square)) * tmask` association from
`icesbc.F90:328-340` bit-for-bit.  Friction is not an exchange-frame output, so
this is a source-identity probe rather than an added oracle comparison row.

The output/assembly statements also retain the NEMO multiply-by-reciprocal
form where NEMO uses `r1_rdtice`; the dumped `inverse_dt` operand is consumed
rather than recomputing division.

## Compiler-folded constants

A WRITE-only config-local instrument was applied only in two new source copies
and wrote four active-path operands once.  Both builds use
`arch-conda-scalarmath.fcm` SHA-256
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`,
completed 8,760 CPU-only steps without `mpirun`, and have zero `_ZGV*` dynamic
symbols.  The dumps are byte-identical, 32 bytes, SHA-256
`2e7fe35234848a0978fcd3ded8327c1f3478191a7f787e36f4bf79f1451fe4ad`.

| pinned name | value | IEEE-754 hex | NEMO source |
|---|---:|---|---|
| `epsilon_air_nemo` | 0.6220002383557786 | `3fe3e76d0b3af3e8` | `sbc_phy.F90:33-35` |
| `poisson_dry_air_nemo` | 0.2856285160421812 | `3fd247bcd3cd320c` | `sbc_phy.F90:39-43` |
| `goff_ice_D_nemo` | 0.7858350313586662 | `3fe9258f81f79246` | `sbc_phy.F90:74-79` |
| `ln10_nemo` | 2.302585092994046 | `40026bb1bbb55516` | `sbc_phy.F90:709-711` |

JAX 0.10.0 runtime evaluation happened to produce all four same bits, so the
preregistered folded-value numerical owner is **REFUTED**: pinning did not move
the 18-row census.  Pinning is retained because gfortran defines these as
compile-time operands and the selectable NEMO identity must not evaluate them
with a runtime transcendental.

Copy-only provenance:

- source copies: `nemo502_si3bulk_folded_a_src` and
  `nemo502_si3bulk_folded_b_src` under the lane-3 data root;
- run roots: `c1d_omip_l3_sasice_folded_a` and `_b` under that root;
- executables: SHA-256 `b5ebd6c2a4ff98801d3962a6d0515f8399c80ef8a003a8c2357b5418d5a34cbf`
  and `911859a1018389838670b15f74df36137d9c51a7a10adda773d959c6135426ad`;
- committed instrumentation replay patch (blank context canonicalized without
  changing the resulting Fortran): SHA-256
  `bd2fe0b51e7516ac1df7736e374e793eed47e1ccc04541a0b0a3b598dd66635e`.

The shipped NEMO tree and every retained oracle root remained unmodified.

## Metrics and run configuration

Every gate row now emits absolute and normalized error, relative error over a
nonzero oracle, bit counts, and row-scale ULP error with
`ulp=spacing(max(max(abs(oracle)),1))`.  The residual group has maximum
relative error `8.4104602317484995e-16` and maximum row-scale ULP error `1.0`.
The largest normalized row is albedo at kt6320:
`2.220446049250313e-16`, relative `3.405906152206416e-16`, one row-scale ULP.

ORCA1/C1D selections are now `SeaIceConfig` values with `__param_spec__`
coverage, not physical constants.  The card explicitly selects ORCA1
`Cd_ice=Ce_ice=Ch_ice=1e-3` (`namelist_cfg:139-142`) and the C1D resolved
`drag_ocean=5e-3`, snow-blow exponent 0.66, four albedos, and 1 m pivot
(`namelist_ice_ref:136,141,305-311`; `ocean.output:733,735,873-879`).  `Ce_ice`
now has its missing parameter specification.  The validator rejects mixed
combinations.

Runtime versions are always stamped.  **Bit-exactness claims are valid only
under Python 3.13.0, JAX/jaxlib 0.10.0, NumPy 2.4.4.**  Other stacks can run the
normalized gate and receive `WITHHELD_RUNTIME`; only
`--require-bit-identity` fails closed.

## Evidence, controls, and tests

The final external gate artifact is
`/data/abyssal/dbalwada/nemo-testcases-l3/round10_si3_bulk/bulk_round10_final_gate.json`,
SHA-256 `a05b4b90b0b2346995430183cf5162319d94b72fe53346d551b445235f03d4d4`.
All twelve plants exit nonzero: `blk_ice_1`, `ice_alb`, `blk_ice_2`,
`ice_flx_other`, stream hash, coverage, selector, bit-owner registry, explicit
runtime claim, source rounding, folded constant, and friction association.

- Final bulk gate tests: `13 passed in 23.24s`.
- Historical external-artifact/year-gate tests: `17 passed in 41.91s`.
- Final combined bulk/year/parameter-spec run: `100 passed in 67.33s`.
- `tests/test_no_hardcoded_constants.py` selected for every touched bulk/ice
  implementation, gate, and test file: `19 passed, 3372 deselected in 1.02s`.
- Repository-wide constants ratchet: `3384 passed, 2 skipped, 5 failed`; all
  five failures are untouched pre-existing FV3/grid/DINO sites.  No touched
  file failed.

## Repository hygiene

Round 9's 783,517-line runtime year JSON was copied byte-for-byte to
`/data/abyssal/dbalwada/nemo-testcases-l3/round9_scalarmath_v2_artifacts/`
before removal from git.  Its SHA-256 remains
`91956787dcabc1080e352b0a2920c13f33a0c8794dabb9cba3261d9e2f85e5e1`;
the external-artifact test rechecks the hash.  No data artifact was deleted.

Six earlier multi-megabyte year-gate artifacts were also copied byte-for-byte
to
`/data/abyssal/dbalwada/nemo-testcases-l3/historical_si3_year_gate_artifacts/`
and removed from git.  Their retained hashes are:

| artifact | SHA-256 |
|---|---|
| phase 2b year | `6c21d14f3c85d0fa99be7e31770a4a4dcad548d98f857ea78455bd6344622160` |
| phase 3 baseline operator | `d642f532ed92910b49f29919dd5e4ae3fe84381ecd1536b7886a2c1f4b36f95f` |
| phase 3 year | `77c78a816b254484afe06632d77378336ddac1153ef9190d17c555f41c45a6e1` |
| phase 4 year | `f81fe9312fb9ade51ae87a8d73970c5281a7ed036d5f7bcaa5c39644ec75c925` |
| phase 5 year | `d5cd3dc274687b56370075ea51b9df82cc01bf912727e25f2078b6c26fe95cc7` |
| phase 6 year | `9571996d72875a3c312fb5b84170d5383bedc7d41fff8ebd9a75f38d3b2a0f9f` |

Historical tests and the year-gate tool bind those external paths to the same
hashes.  Only summary tables and hashes remain in the repository.

Commits through the five requested fixes are:

- `ea5117d1537` source-statement association;
- `31d33ce4fca` folded constants;
- `f0cb842eea4` config separation and ULP metrics;
- `a27eb13955c` runtime-output removal from git; and
- `892cbed5015` conditional runtime bit claims;
- `9e2f4df878f` corrected Stage-1 census and friction association;
- `56be30820f9` externalized six historical year artifacts; and
- `2b7d0b108ba` withheld the friction probe's bit verdict off the registered
  runtime.

## Codex-internal review record

Two codex-internal reviews of `6e30ff8d993` returned **HOLD**, not SHIP.  A
source-fidelity re-review and a config/hygiene re-review of `56be30820f9` again
returned **HOLD** on the remaining unconditional friction-probe bit label.
Commit `2b7d0b108ba` fixes that last shared finding.  The review artifacts name
the reviewer identity, exact commit reviewed, verdict, and unverified work;
the final re-review artifacts are the only basis for any final review verdict
stated for this round.  These are codex-internal reviews, not independent
external Claude/Opus reviews.

The final codex-internal source-fidelity and config/hygiene re-reviews of
`b57abb598e9f9f6486172541a7505bfc5d715ce6` both returned **SHIP**.  Their
verbatim records are
`nemo_testcases_l3thd_round10_source_review.md` and
`nemo_testcases_l3thd_round10_hygiene_review.md`.

## Decisions

| choice | status | disposition |
|---|---|---|
| fix all 16,494 ordinary association rows | ASKED | fixed; private arm restores exact baseline counts |
| pin compiler-folded active-path operands | ASKED | two-copy dump reproducible; four exact bits pinned |
| move namelist selections into config and add `Ce_ice` spec | ASKED | complete; card explicit and mixtures rejected |
| emit relative/ULP/bit metrics | ASKED | complete; definition and measured maxima above |
| remove large Round-9 runtime output from git | ASKED | externally retained and hash-bound |
| let CI run on another numeric stack while withholding bit claims | ASKED | complete; explicit claim still fails closed |
| implement rung 3.6 | UNASKED | not done |
| build a lane-local exp/tanh policy | UNASKED | not done; shared GYRE policy awaited |
| change forcing, timestep, physical scope, or oracle V2 | UNASKED | not done |
| modify/delete shipped NEMO or retained data | UNASKED | not done |
| push | UNASKED | not done |

## Flagged for future deletion

Nothing was deleted.  The two 61-GiB copy-only folded-constant source/build
trees are now stale after producing the pinned dump and are **FLAGGED FOR
FUTURE DELETION**, subject to user approval.  All intermediate Round-10 gate
JSONs under the data root are retained and likewise flagged; the final artifact
named above is the accepted evidence.
