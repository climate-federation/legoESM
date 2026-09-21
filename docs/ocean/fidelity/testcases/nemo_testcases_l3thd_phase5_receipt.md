# NEMO testcase fidelity receipt — lane 3b, SI3 Phase 5 scope audit

Tracker: `climate-federation/legoESM#1699`

Recovery git: `/tmp/codex-si3thd-localgit`

Implementation/evidence commits through this receipt:

- `7954ceae00a7` — add the NEMO written-order mode to the one shared Thomas
  solver and prove bit agreement while the private copy still exists.
- `e7fba49e758e` — delete the private Thomas implementation and route SI3
  through `timestepping.tridiagonal.thomas_solve`.
- `0c1a456d448b` — preregister the four bundled-change ablations.
- `219edbba84f` — isolate the basal-layer-loop transcription behind a private
  gate hook.
- `d1bd76a2af0` — isolate the snow-ice salinity term behind a private gate hook.
- `51fb9fdde7f` — isolate the three EOS operation orders behind a private gate
  hook.
- `ba3d17eabab` — run and retain all exact-entry/continuous arms, the complete
  remaining-debt histogram, jump attribution, and plant.
- `ee2cc77a734` — retract the incomplete Round-4 scope account and correct its
  clean-suite count.

## Verdict

**DEBT; scope account corrected.**  Commit `80b07f39afa` did not contain only
the two disclosed and independently confirmed owners.  It contained four
active physics corrections (negative-evaporation deposition, snow-first
surface melt, ZDF no-snow/melting row ranges, and the bottom-up basal layer
loop), one active arithmetic-order transcription (three EOS conversions), and
one measured-inert term (snow-ice salinity).  The corrected Round-4 receipt now
states this plainly.

The current exact-entry sweep still has **36,852 over-bar field rows** among
621,960 comparisons.  None is in the requested denominator-one/NEMO-exactly-
zero class; all 36,852 are in the genuine-relative partition.  The continuous
trajectory has a mixed diagnosis: kt4239 and kt5045 amplify same-step
`<=2e-15`-class differences, but later exact-entry injections are materially
larger.  Therefore this receipt does not label the remaining year trajectory
as mere summation noise and does not claim whole-column fidelity.

## Shared Thomas solver: one implementation

Before deleting the duplicate, `test_nemo_operation_order_shared_solver_is_bit_exact`
compared the private SI3 implementation and shared solver with the same seven-
row matrices and required `np.testing.assert_array_equal`; the focused run
reported `10 passed`.  Commit `e7fba49e758e` then removed
`_nemo_thomas_solve`.  The surviving test compares the shared implementation
against a scalar replay of NEMO's written forward/back substitutions and the
post-deletion focused run reported `11 passed`.

The source operations are NEMO `icethd_zdf_bl99.F90:516-558`.  The executing
shared mode is `packages/core/legoesm/timestepping/tridiagonal.py:112-147`, and
SI3 calls it at `packages/ice/legoesm/ice/bitz_lipscomb.py:567`.  The ordinary
shared users retain the pre-existing normalized mode; there is no second
Thomas implementation.

## Bundled-change preregistered ablations

The preregistration is
`nemo_testcases_l3thd_phase5_scope_preregister.md`.  Each arm changes one
private underscore hook while retaining the ORCA1-resolved identity everywhere
else.  The gate compares every arm against the same 8,760 exact NEMO ENTRY
states at all eight registered boundaries, then runs each arm continuously for
the full year.

| change | NEMO source | exact-entry arm | continuous arm | disposition |
|---|---|---|---|---|
| ZDF no-snow/melting row ranges | `icethd_zdf_bl99.F90:433-513` | 39,505 rows change; disabled/enabled over-bar counts `54,551/36,852`; first change kt3836 `POST_ZDF.e_i`, disabled `2.9695e-8`, enabled exactly zero | disabled minimum `0.3828710148 m`, growth day 252; enabled `0.5551788291 m`, day 251 | **CONFIRMED active/source-aligned**; this was silently bundled and never ablated in Round 4 |
| bottom-up basal layer loop | `icethd_dh.F90:309-315,366-424` | 11,190 rows change; disabled/enabled over-bar counts `36,944/36,852`; first bit change kt2776 | legacy-loop minimum differs from enabled by `6.16e-13 m`; dates unchanged | **ACTIVE real fix**, although this particular continuous arm has no material phenology effect |
| snow-ice salinity contribution | `icethd_dh.F90:441-485,509-519` | 0 of 621,960 rows change | 0 of 61,320 field-step rows change; all six phenology values bit-identical | **INERT in this C1D year**, not an owner; retained because the selected NEMO identity is unconditional |
| three EOS operation orders | `icevar.F90:938-946`; `icethd.F90:233-243`; `icethd_dh.F90:498-503` with bounds from `icevar.F90:404-416` | 24,293 rows change; disabled/enabled over-bar counts `38,233/36,852`; 9,185 rows favor enabled, 7,471 favor disabled, zero surface-branch splits | arithmetic differences accumulate, but minimum differs by `7.70e-13 m` and all dates agree | **ACTIVE FLOAT RE-ASSOCIATION**, not a branch or phenology owner |

The no-basal-mechanism arm is a separate scaling discriminator, not the
one-variable legacy-loop arm: disabling basal melt moves melt onset from day
133 to 176 and the minimum from `0.5551788291 m` to `1.5898552841 m`.  This
confirms that basal melt is dynamically important.  The legacy-loop arm above
specifically measures the previously hidden loop rewrite and corrects the old
“basal melt was already present” wording: only a simplified bottom-layer
calculation was present.

The executing locations are `bitz_lipscomb.py:507-567` for row selection,
`:792-868` for basal growth/melt, `:882-924` for flooding and bulk salinity,
and `:177-255,320-339` for EOS conversions.  Private hooks are passed only by
the validation gate; none is exposed in `IceConfig` or the column card.

## Remaining 36,852 exact-entry rows

The committed JSON retains the full `by_step` mapping for all 3,292 affected
steps.  Compact histograms are:

| sub-call | rows |
|---|---:|
| POST_ZDF | 2,100 |
| POST_DH | 4,819 |
| POST_TEMP1 | 4,819 |
| POST_SAL | 4,829 |
| POST_TEMP2 | 4,829 |
| POST_DO | 7,728 |
| EXIT | 7,728 |

| variable | rows |
|---|---:|
| `e_i` | 13,996 |
| `e_s` | 6,942 |
| `h_i` | 4,836 |
| `h_s` | 160 |
| `t_su` | 3,542 |
| `v_i` | 2,418 |
| `v_s` | 72 |
| `sv_i` | 2,436 |
| `sz_i` | 20 |
| `szv_i` | 2,430 |

The rows-per-affected-step frequency is retained as
`rows_per_step_frequency`; its most common bins are 16 rows on 1,137 steps, 7
rows on 527 steps, 2 rows on 422 steps, and 6 rows on 340 steps.  The full bin
map and the exact per-step counts are in the JSON so this prose does not hide
the long tail.

| normalization class | count | largest row |
|---|---:|---|
| denominator `1` and every NEMO operand exactly zero | **0** | none; the historical kt5406 row belonged to the earlier baseline, not this current census |
| genuine relative rows | **36,852** | kt5734 `POST_DH.e_s`, absolute and denominator `777833.3635432672 J m-3`, quotient `1.0`; NEMO is nonzero |

The first exact-entry injection over `1e-12` is kt4242 `POST_DH.h_i`, absolute
`6.2567284687e-11 m`, normalized `2.8315499259e-11`.  The first over `1e-3`
is the kt5734 `POST_DH.e_s` row above.  Both remain **DEBT**; no new owner is
assigned without the operand-level DH discriminator.

## Continuous growth and jump attribution

All entries below are normalized L-infinity errors from the current production
fp64 trajectory.  These values are unchanged from Round 4 because this round
audited already-landed operations rather than changing the selected identity.

| step | `t_su` | `e_i` | `h_i` | `h_s` |
|---:|---:|---:|---:|---:|
| 1 | `4.450125e-16` | `3.555755e-16` | `0` | `5.551115e-17` |
| 10 | `2.261134e-16` | `7.607426e-15` | `6.651279e-16` | `3.885781e-16` |
| 100 | `1.844910e-15` | `3.965980e-14` | `3.512692e-15` | `2.858824e-15` |
| 1000 | `1.671332e-14` | `1.451553e-13` | `1.273959e-13` | `3.030909e-14` |
| 3000 | `1.824396e-14` | `3.011895e-13` | `2.711632e-13` | `1.140754e-13` |
| 5000 | `0` | `9.660627e-7` | `6.690896e-7` | `0` |
| 8760 | `6.633380e-6` | `1.634242e-4` | `1.489868e-4` | `6.317169e-14` |

The conspicuous continuous jumps and same-step exact-entry maxima are:

| step/field | previous → current continuous error | exact-entry step maximum | classification |
|---|---:|---:|---|
| kt4239 `e_i` | `4.774e-13 → 7.371e-7` | `1.084e-15` | threshold amplification from 2e-15-class input/operator noise |
| kt4239 `h_i` | `4.615e-13 → 3.875e-7` | `1.084e-15` | same event |
| kt4943 `h_s` | `0 → 2.664e-10` | `2.593e-8`; same-field POST_DH is `2.782e-12` | genuine exact-entry injection is already present |
| kt5045 `t_su` | `0 → 4.363e-10` | `4.663e-16` | threshold amplification from 2e-15-class noise |
| kt5238 `t_su` | `4.526e-9 → 2.929e-6` | `4.469e-14`; same-field `2.581e-14` | above the reassociation class used for the initial jumps |

Thus the terminal classification is **MIXED DEBT**.  Bit-exact reproduction of
NEMO's summation order across the whole step is needed to remove and test the
threshold-amplified component.  Operand-level DH frames at kt4242 and kt5734
are needed before assigning the later genuine operator debt.  A blanket
“threshold amplification only” statement is refuted by the current exact-entry
trajectory.

## Six phenomenology rows

| quantity | NEMO | legoESM fp64 | distance | fp32-fp64 floor | status |
|---|---|---|---:|---:|---|
| minimum thickness | `0.5550952377 m` | `0.5551788291 m` | `8.3591e-5 m` | `0.1106972 m` | **AT-FLOOR** |
| maximum thickness | `2.4456886226 m` | `2.4456886226 m` | `6.7280e-13 m` | `5.1158e-5 m` | **AT-FLOOR** |
| minimum date | 2018-09-09 00Z | 2018-09-09 00Z | `0 h` | `2304 h` | **AT-FLOOR** |
| maximum date | 2018-05-13 00Z | 2018-05-13 00Z | `0 h` | `0 h` | **AT-FLOOR** |
| melt onset | 2018-05-14 (day 133) | 2018-05-14 (day 133) | `0 d` | `0 d` | **AT-FLOOR** |
| growth onset | 2018-09-09 (day 251) | 2018-09-09 (day 251) | `0 d` | unavailable: fp32 has no onset | **UNMEASURED** |

`AT-FLOOR` applies only to these measured phenomenology rows.  It does not
override the exact-entry or continuous-state DEBT verdicts.

## Execution, controls, and provenance

The complete gate ran with `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and
`PrecisionPolicy.fp64()`.  Its artifact prints backend `cpu`, oracle and
legoesm production dtype `float64`, floor-run dtype `float32`, one-hour step,
and 8,760 steps.  No GPU and no MPI launcher were used.  No shipped NEMO file,
configuration, or prior run root was modified or deleted.

The focused SI3/year/shared-solver suite plus exact constant-ratchet nodes
reports:

```text
collected 33 items
tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py .............
tests/ocean/fidelity/test_nemo_si3thd_phase2b_year_gate.py .............
tests/unit/test_tridiagonal.py ....
tests/test_no_hardcoded_constants.py ... [100%]
============================= 33 passed in 48.49s ==============================
```

The new accounting plant subtracts one changed row and exits nonzero with
`scope-arm row accounting`.  The same unit run executes the established
arithmetic, deposition, surface-melt, snow-temperature, branch-census,
trajectory-truncation, and largest-outlier plants.  A repository-wide constants
ratchet run reports five failures in unrelated FV3/DINO files; this receipt does
not call that broad run green.  The exact ratchet nodes for every touched
production/test file report the quoted line
`tests/test_no_hardcoded_constants.py ... [100%]`.

| artifact | SHA-256 |
|---|---|
| Phase-5 year gate JSON | `d5cd3dc274687b56370075ea51b9df82cc01bf912727e25f2078b6c26fe95cc7` |
| year-gate script | `3b522d9c7c6825c1f0a031101472cc57bfa41ab5bf177eae5dbe7ae3b06585fa` |
| SI3 implementation | `d7caca803fe9f1eb1525fc1010f67755d8be3138b03dfc52d942655605d2e947` |
| shared tridiagonal solver | `a6be7f03d5adefa1395112ff837f368b7eecc518ef195fe75c7372189ca0ab77` |
| year-gate unit tests | `22d8d6e95bd2aa69fc6788d66e7436e5845b97fd453cbe03fac1037688e93e75` |
| Phase-5 preregistration | `929eaf8af4ae88daa14e83fb3df03a8404bb5ee92eb07f27324226987418c136` |
| ERA5 forcing | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| 70,080 thermodynamics frames | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| 8,760 exact ZDF-input frames | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` |

The evidence commit holding the JSON is `ba3d17eabab`; the corrected Round-4
receipt commit is `ee2cc77a734`.  This receipt itself is the following
documentation commit, so it does not make a circular self-SHA claim.

## Coupled-rung debt

NEMO's `sbcblk` bulk-flux computation remains **NOT CERTIFIED**.  This isolated
column consumes NEMO-written `qns_ice`/`dqns_ice` and does not recompute
`sbcblk.F90:1273,1480-1491` feeding `icestp.F90:201,206`.

## End-of-task choice register

- ASKED — remove the duplicate Thomas solver after proving bit agreement and
  keep one shared implementation.
- ASKED — preregister and measure separate one-variable arms for the ZDF row
  ranges, basal loop, snow-ice salinity, and EOS operation orders.
- ASKED — rewrite the Round-4 receipt with the complete change count/table and
  correct `18 passed` to the clean-checkout `19 passed` result.
- ASKED — histogram all 36,852 remaining rows by sub-call, variable, and step;
  separate denominator-one/NEMO-zero rows from genuine relative errors.
- ASKED — retain the seven requested continuous growth rows, six phenomenology
  rows, and same-step exact-entry evidence at trajectory jumps.
- ASKED — CPU/fp64 execution, copy-only oracle use, explicit-path recovery-git
  commits, bundle, hashes, session ID, and no push.
- UNASKED — alternate SI3 identities, public physics switches, threshold/bar
  changes, coupled bulk-flux certification, GPU/MPI execution, or modification
  and deletion of shipped NEMO/run-root files.

## FLAGGED FOR FUTURE DELETION

Nothing was deleted.  Every prior C1D oracle/run root remains on disk and is
only flagged for a future owner decision.
