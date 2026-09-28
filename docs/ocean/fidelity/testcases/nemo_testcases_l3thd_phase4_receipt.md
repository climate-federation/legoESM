# NEMO testcase fidelity receipt — lane 3b, SI3 Phase 4

Tracker: `climate-federation/legoESM#1699`

Physics commit: `a2d867908fd`; complete operator artifact/gate commit:
`4194d092153`.  Recovery git is `/tmp/codex-si3thd-localgit`.

## Verdict

**DEBT, with the kt4239/kt5285 owner confirmed and repaired.**  NEMO does not
carry an unknowable snow temperature across the `ice_thd` entry.  It
reconstructs or resets `t_s` in `ice_var_glo2eqv(2)` before every thermodynamic
call.  legoESM had the same enthalpy inverse but omitted NEMO's unconditional
`[rt0-100,rt0]` bounds.  At tiny positive snow depth and zero registered
enthalpy this produced `432.1427505127 K` instead of NEMO's `273.15 K`.

The one-variable repair removes the kt4239 injection and the kt5285
cold/fixed-melting branch split.  Exact-entry over-bar rows fall from 37,659 to
36,852.  The first material injection above `1e-12` moves from kt4239 to kt4242.
The continuous trajectory and all six phenomenology rows are unchanged from
Phase 3, because that trajectory does not visit the repaired tiny-positive-
snow/zero-enthalpy state.  The earliest strict bar crossing remains the known
kt5 `POST_DO.e_s` roundoff row at `1.011825579870701e-15`; no later-first-frame
claim is made.

## Rule 0: NEMO's active tiny/zero-snow state

The active call order is explicit: `ice_var_glo2eqv(2)` precedes
`ice_sbc_flx` and `ice_thd` (`icestp.F90:182-206`).  In that conversion NEMO
tests `IF (v_s > epsi20)`, reconstructs snow temperature as
`rt0 + MAX(-100, MIN(...,0))`, and otherwise sets `t_s=rt0`
(`icevar.F90:404-416`).  `epsi10=1e-10` and `epsi20=1e-20` are fixed in
`par_ice.F90:123-126`.

The correction paths do not establish a different active rule.  `zapsmall`
tests `MIN(a_i,v_i,h_i) < epsi10`; only then does it zero snow enthalpy and set
`t_s=rt0`, and its volume zap uses the same ice predicate
(`icevar.F90:601-706`).  `zapneg` zeros snow energy/volume for negative or
ice-absent states but does not receive `t_s` (`icevar.F90:726-816`).

Inside thermodynamics, `ice_thd_1d2d` copies `t_s` and `e_s` independently at
`icethd.F90:343-355`; it subsequently sets volumetric `e_s_1d=0` when
`h_s*a_i <= epsi20` at `:418-435`.  BL99 treats every `h_s_1d>0` as
snow-present, uses a minimum conductive layer thickness, and saves
`ztsold=t_s_1d` (`icethd_zdf_bl99.F90:159-198`).  For exactly zero snow it
selects the no-snow matrix range (`:462-513`) and skips snow-temperature solves
(`:541-550`); the preceding global conversion has already reset `t_s` to
`rt0`.  Solved snow and surface temperatures are bounded again at `:553-575`.

legoesm does not retain a separate prognostic `t_s` carrier.  It reconstructs
this diagnostic from the column enthalpy at each ZDF call.  The executing
inverse now applies NEMO's bounds at `bitz_lipscomb.py:299-312`; ZDF consumes it
at `:398-441`.  The old unbounded expression survives only behind the private
gate hook `_nemo_snow_temperature_bounds=False` (`:409-411,428-441`), and the
public SI3 identity always enables the NEMO rule (`:907-923`).  NEMO exposes no
namelist switch for this conversion, so production is unbranched.

## Rule-1d operand registry and copy-only oracle run

The write-only version-2 ZDF-entry record appends the three-layer `t_s_1d`
operand (`nemo502_si3thd_MY_SRC/icethd.F90:277-295`).  Its registered time level
is **current ice step, after `ice_var_glo2eqv(2)` and
`ice_thd_1d2d`, immediately before `ice_thd_zdf`**, sourced to
`icestp.F90:182-206`, `icethd.F90:343-355,418-435`, and
`icethd_zdf_bl99.F90:189-199`.  The registry is emitted in the Phase-4 JSON.

The source copy is
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3thd_phase4_src`; it was
copied from the prior Phase-2b source copy, then only its `C1D_OMIP_L3/MY_SRC`
was updated from the committed write-only source.  It was rebuilt with
`makenemo -n C1D_OMIP_L3 -d 'OCE SAS ICE' -m conda -j 8`.  The new additive
run root is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase4_operands`.
It completed all 8,760 one-hour CPU/no-MPI steps with `STOP 0`.  The original
thermodynamics frames, restarts, and `ocean.output` remain byte-identical; only
the expanded operand stream changes.  No shipped NEMO source/configuration or
earlier run root was modified or deleted.

## Kt4239 ownership, arm, and scaling

At kt4239 the global ENTRY has `v_s=7.318364664277155e-20 m`, three zero snow-
enthalpy layers, and `h_s=8.131516293641283e-20 m`.  Because `v_s>epsi20`, the
source replay takes the reconstruction arm and its upper bound yields exactly
`[273.15,273.15,273.15] K`, bit-for-bit equal to the new dump.  The pre-fix
unbounded operand is `[432.1427505127]*3 K`, a `158.9927505127 K` operand
difference.

The first material exact-entry divergence is POST_ZDF.  Both sides are
snow-present and fixed-melting at the surface, so this row is an operand
difference within the same branch, not a branch split.

| POST_ZDF row | disabled/pre-fix absolute | enabled absolute | improvement |
|---|---:|---:|---:|
| `e_i` | `788.4015902281 J m-3` | `0` | `3.55e18` |
| `e_s` | `7453.811608702 J m-3` | `1.1920928955e-7 J m-3` (`1.08356e-15` normalised) | `6.25e10` |

The single `t_s` operand arm therefore explains the observed `e_i` row by far
more than the preregistered 100-fold discriminator.  The residual `e_s` value
is a near-bar arithmetic row; it is not described as zero or as whole-step
agreement.

## Kt5285 branch split

Kt5285 has `v_s=3.8116482626443516e-22 m <= epsi20`, so NEMO takes the reset
arm and again dumps `[273.15]*3 K`.  Snow depth remains positive
(`4.235164736271502e-22 m`), so both executions enter BL99's snow-present row
range.  The branch condition is then `t_su_1d < rt0`: NEMO remains cold
(`true`), while the pre-fix unbounded operand drives legoESM to `rt0`
(`false`) and hence the fixed-melting surface equations
(`icethd_zdf_bl99.F90:433-460`).  After the bound, legoESM is also `true`; the
branch split is removed.

The private arm changes kt5285 POST_ZDF `t_su` error from
`0.7936539212 K` to `1.1823431123e-11 K` (improvement `6.71e10`), `e_i` from
`103075.5750 J m-3` to zero, and `e_s` from `547515.2093 J m-3` to
`8.2850456238e-6 J m-3`.  The nonzero `t_su` and `e_s` residuals are reported
with their normalised values in the JSON; only `e_i` is exactly zero.

## Full-year exact-entry sweep

The committed JSON contains all 8,760 `oracle_entry_operator_sweep.per_step`
records, not a running maximum.  Every record gives the step maximum, owning
sub-call/field/branch, numerator, denominator, and that step's over-bar count.

| measure | Phase 3 before | Phase 4 after |
|---|---:|---:|
| first over-bar | kt5 `POST_DO.e_s`, `1.01182558e-15` | unchanged |
| over-bar field rows | 37,659 | 36,852 |
| first injection `>1e-12` | kt4239 `POST_ZDF.e_i`, `2.66636e-6` | kt4242 `POST_DH.h_i`, `2.83155e-11` |
| first injection `>1e-3` | kt5285 `POST_ZDF.t_su`, `2.91403e-3` | kt5734 `POST_DH.e_s`, `1.0` |

The Phase-4 largest exact-entry outlier is kt5734 `POST_DH.e_s`: absolute
numerator `777833.3635432672 J m-3`, denominator
`777833.3635432672 J m-3`, quotient `1.0`.  This is the
snow-sublimation/basal-thermodynamic-remap row; it remains debt.

The earlier `1.1e8` row is also now explicitly attributed.  In the retained
pre-Phase-3 census it is kt5406 `POST_DH.e_s`, numerator
`111080922.31885993 J m-3`, denominator `1`, quotient
`111080922.31885993`.  NEMO's three-layer oracle value is exactly zero because
the DH snow-first surface-melt path removed all snow; the then-missing legoESM
path retained snow enthalpy.  That owner was repaired in Phase 3
(`icethd_dh.F90:107-120,204-315`).

## Continuous growth table

Values are Phase 3 before → Phase 4 after normalised L-infinity errors.  They
are identical at every requested sample; this is a measured null trajectory
effect, not a fidelity claim.

| step | `t_su` | `e_i` | `h_i` | `h_s` |
|---:|---:|---:|---:|---:|
| 1 | `4.4501e-16 → 4.4501e-16` | `3.5558e-16 → 3.5558e-16` | `0 → 0` | `5.5511e-17 → 5.5511e-17` |
| 10 | `2.2611e-16 → 2.2611e-16` | `7.6074e-15 → 7.6074e-15` | `6.6513e-16 → 6.6513e-16` | `3.8858e-16 → 3.8858e-16` |
| 100 | `1.8449e-15 → 1.8449e-15` | `3.9660e-14 → 3.9660e-14` | `3.5127e-15 → 3.5127e-15` | `2.8588e-15 → 2.8588e-15` |
| 1000 | `1.6713e-14 → 1.6713e-14` | `1.4516e-13 → 1.4516e-13` | `1.2740e-13 → 1.2740e-13` | `3.0309e-14 → 3.0309e-14` |
| 3000 | `1.8244e-14 → 1.8244e-14` | `3.0119e-13 → 3.0119e-13` | `2.7116e-13 → 2.7116e-13` | `1.1408e-13 → 1.1408e-13` |
| 5000 | `0 → 0` | `9.6606e-7 → 9.6606e-7` | `6.6909e-7 → 6.6909e-7` | `0 → 0` |
| 8760 | `6.6334e-6 → 6.6334e-6` | `1.6342e-4 → 1.6342e-4` | `1.4899e-4 → 1.4899e-4` | `6.3172e-14 → 6.3172e-14` |

## Six phenomenology rows

Phase 3 before and Phase 4 after are identical in every row.  The status is
the established legoESM-fp32-versus-fp64 floor classification, not a statement
that the full state trajectory reaches the oracle bar.

| quantity | NEMO | before | after | after distance | fp32-fp64 floor | status |
|---|---|---|---|---:|---:|---|
| minimum thickness | `0.5550952377 m` | `0.5551788291 m` | `0.5551788291 m` | `8.3591e-5 m` | `0.1106972 m` | AT-FLOOR |
| maximum thickness | `2.4456886226 m` | `2.4456886226 m` | `2.4456886226 m` | `6.7280e-13 m` | `5.1158e-5 m` | AT-FLOOR |
| minimum date | 2018-09-09 00Z | 2018-09-09 00Z | 2018-09-09 00Z | `0 h` | `2304 h` | AT-FLOOR |
| maximum date | 2018-05-13 00Z | 2018-05-13 00Z | 2018-05-13 00Z | `0 h` | `0 h` | AT-FLOOR |
| melt onset | 2018-05-14 | 2018-05-14 | 2018-05-14 | `0 d` | `0 d` | AT-FLOOR |
| growth onset | 2018-09-09 | 2018-09-09 | 2018-09-09 | `0 d` | unavailable (fp32 has no onset) | UNMEASURED |

## Provenance and hashes

The forcing remains the official NEMO `sette_inputs` r5.0.0 archive/member;
no analytic or synthetic substitute is used.

| artifact | SHA-256 |
|---|---|
| official archive (MD5 `9456e6a0a84d40630ad1804fd4061caf`) | `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| ERA5 North Greenland forcing | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| Phase-4 `nemo.exe` | `2619c0b0a2d0d38f86aefb68c1e55a96e40c44200ee29bab0b5664121b510b3d` |
| `namelist_cfg` | `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| `namelist_ice_cfg` | `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |
| 70,080 thermodynamics frames | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| 8,760 version-2 ZDF inputs + `t_s` | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` |
| kt1+kt3 ZDF operands | `aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442` |
| kt3 reassociation operands | `4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589` |
| final ice restart | `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` |
| final ocean restart | `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265` |
| `ocean.output` | `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` |
| committed copy-only `MY_SRC/icethd.F90` | `3afb13b7612f09b5cdb1322a55186a53f77891ff5016d3859ec9e2da58cc5490` |
| Phase-4 year-gate JSON | `f81fe9312fb9ade51ae87a8d73970c5281a7ed036d5f7bcaa5c39644ec75c925` |

## Controls and tests

The snow-temperature owner plant exits nonzero with
`snow-temperature bounds arm failed 100-fold discriminator`.  The per-step
trajectory plant withholds record 8760 and exits nonzero with
`oracle-entry per-step count`.  The outlier-attribution plant doubles only the
reported denominator and exits nonzero with
`oracle-entry largest-outlier attribution`.

The direct Phase-2/Phase-2b/Phase-4 suite plus the three exact touched-file
constant-ratchet nodes reports:

```text
============================= 26 passed in 43.53s ==============================
```

The exact ratchet line is `tests/test_no_hardcoded_constants.py ... [100%]`.
The repository-wide ratchet invocation still exposes six unrelated literal
failures plus its partial-checkout discovery guard; this receipt does not call
those pre-existing failures successes.  Direct `banned_hits` checks on both
touched validation scripts print `[]`.

## Coupled-rung debt

| component | status | reason |
|---|---|---|
| NEMO `sbcblk` bulk fluxes | **DEBT / NOT CERTIFIED** | The isolated column consumes NEMO-written `qns_ice`/`dqns_ice`; it does not recompute `sbcblk.F90:1273,1480-1491`. |
| remaining exact-entry DH outlier | **DEBT** | Kt5734 `POST_DH.e_s` has quotient `1.0`; it is outside this dispatch's confirmed snow-temperature owner. |
| continuous year-end state | **DEBT** | At kt8760, `e_i=1.6342e-4`, `h_i=1.4899e-4`, and `t_su=6.6334e-6` normalised; the null Phase-4 trajectory movement does not certify them. |

## End-of-task choice register

- ASKED — source-first the tiny/zero-snow `t_s` reset/carry rules and register
  the post-conversion, pre-ZDF operand time level.
- ASKED — first-divergence kt4239, scale a private one-variable arm, and repair
  the unconditional NEMO identity without adding a public switch.
- ASKED — name kt5285's snow-present cold/fixed-melting branch predicates on
  both sides and verify the split is removed.
- ASKED — rerun 8,760 steps; report first over-bar, count, seven-step/four-field
  growth table, and six phenomenology rows before/after.
- ASKED — commit every oracle-entry per-step record and attribute both the
  current maximum and the historical `1.1e8` row by numerator/denominator.
- ASKED — CPU/fp64, copy-only NEMO work, planted nonzero controls,
  explicit-path local-git commits/bundle, no push.
- UNASKED — alternate SI3 identities, public physics switches, new thresholds,
  forcing changes, coupled bulk-flux certification, GPU/MPI work, or deletion
  of any source/run root.

## FLAGGED FOR FUTURE DELETION

Nothing was deleted.  All earlier source copies and run roots remain on disk;
no new item is flagged for deletion by this dispatch.
