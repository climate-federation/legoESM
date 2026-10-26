# OVERFLOW-zps final water-mass census owner hunt

## Verdict

**The census is not owned.**  The registered bottom-process search is
exhausted and the root owner remains **UNMEASURED**.  The source-exact NEMO
bottom-boundary-layer geometry and raw W-level metric are real transcription
repairs, but causal full-duration arms **REFUTE** both as owners of the
12--18 C slope-water deficit.  They are retained under Rule 8 because the
oracle runs them and exposes no switches for the old operands.

At the incoming revision `e4263111f`, the sole OUTSIDE row was
`0.009437984098628803`.  With NEMO's BBL reference geometry it becomes
`0.010032411356701660` (wrong-sign movement `+5.94427258072857e-4`).  Changing
only the Aimp W metric is bit-identical in the final census to 17 decimal
places.  Changing only the active momentum-ZDF W divisor produces a further
wrong-sign `3.415845e-9` movement.  The combined NEMO package ends at
`0.010032414772546482`; no interaction is present.  NEMO FCT2/FCT4 scheme
spread is `0.003362090947063974`, and the newly rerun legoESM fp32/fp64 floor
is `0.0007434240252439661`, so the verdict remains exactly **OUTSIDE**.

The result is deliberately negative: no damping, clipping, alternate limiter,
or oracle-absent diffusion was added.

## Catch-up and pre-implementation search

Before touching this branch I read the landed selector/transport/SSH/remainder/
phantom/census chain in the requested order: the stage-3-baroclinic and
stage-1-transport preregistrations, the SSH-walk preregistration and receipt,
the stage-3-remainder preregistration and receipt, the phantom-velocity
preregistration and receipt, and the census preregistration and receipt.  I
also audited `git log --oneline 5104de943784..e4263111f` (80-entry window).

That search found the following already-certified facts and prevented this
round from rebuilding them:

* the stage tracer selectors, Kbb RHS seed, Kmm transport, qco face thickness,
  three-dimensional `umask`, reference depth-mean weights, and stage HPG/UP3
  ordering already live inside the single NEMO WS-RK3 identity;
* `nemo_testcase_census_map_probe.py` is the sole implementation of the
  census mask, live-volume weights, bins, NEMO/legoESM mapping, and planted
  controls, so this round extended it rather than cloning the reduction;
* `_NEMOWSRK3TestHooks` is the established private A/B surface; no arm added a
  public selector or a constructible Frankenstein configuration;
* `ocean.physics.bbl_adv` and
  `nemo_wicker_aimp_partition_transport` are the canonical shared kernels.

## Rule-0 resolved physics census

The resolved OVERFLOW-zps namelist at
`/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps/namelist_cfg` selects:

| family | executed resolution | disposition |
|---|---|---|
| tracer advection | FCT2, `nn_fct_h=2`, `nn_fct_v=2`, `nn_fct_imp=1` (`namelist_cfg:67-72`) | VERIFIED active |
| adaptive vertical advection | `ln_zad_Aimp=T` | VERIFIED active |
| tracer lateral diffusion | `ln_traldf_OFF=T` (`:75-79`) | VERIFIED off |
| momentum advection | flux-form UP3 (`:81-87`) | VERIFIED active |
| momentum lateral diffusion | `ln_dynldf_OFF=T` (`:106-110`) | VERIFIED off |
| vertical closure | constant, `rn_avm0=1e-4`, `rn_avt0=0` (`:123-131`) | VERIFIED active; momentum only |
| convection/TKE/EVD | EVD, TKE, GLS, OSM, MFC, NPC off (`ocean.output:634-698`) | VERIFIED off |
| BBL | `nn_bbl_ldf=0`, `nn_bbl_adv=2`, `rn_gambbl=20` (`namelist_cfg:134-140`) | VERIFIED active |

NEMO executes this as one ordered package.  Stage 3 builds BBL coefficients
and transports (`src/OCE/stprk3_stg.F90:463-468`), calls FCT
(`:508-519`), applies BBL (`:583-589`), and performs the implicit Aimp/ZDF
solve (`:596-599`).  Enabling EVD, lateral diffusion, or another convection
scheme would be the stabilizer-the-oracle-lacks class forbidden by Rule 9.

The partial-cell source is material.  OVERFLOW leaves `gdept_0` and `e3w_0`
on the 20 m one-dimensional reference ladder and thins only T/U/V/F bottom
cells (`tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:157-180`).

## Budget: when the gap appears, and what it can say

The committed budget advances one faithful fp64 state and evaluates every
private counterfactual from that same entering state at 60-step cadence.  The
legoESM mixed-water fraction first becomes nonzero at the registered step 180,
rises from `0.0032379913` at step 600 through `0.0391371771` at 2400, peaks
near `0.07675` around step 3900, and falls to `0.0504290773` at 6120.

Only completed steps 0, 3059, and 6120 have matching certified NEMO states.
The common-frame census difference is zero at 0, `0.004512016197481275` at
3059, and `0.009437984098628803` at 6120.  Therefore the honest first matched
statement is: **the deficit is already measurable by step 3059**.  No claim is
made about an earlier NEMO crossing between unavailable oracle samples.

The one-step budget proves that all three active transport families have local
leverage at the slope, but it does not assign the NEMO difference:

* BBL omission changes local T by up to O(`1e-2`) K during the descent, while
  the reference-vs-legacy BBL-geometry correction is also O(`1e-2`) K;
* the FCT two-step predictor diagnostic changes local T by O(`1e-4`--`1e-2`)
  K; it is not a reference configuration and is never used for a verdict;
* deleting vertical transport produces a gross excursion, so the probe records
  `GROSS-EXCURSION` and refuses to manufacture a census number from it.

The census is a discrete bin crossing, so most same-entry one-step effects are
zero even when their temperature tendencies are nonzero.  NEMO internal term
dumps do not exist at this cadence.  Consequently **which term first creates
the inter-model deficit remains UNMEASURED**; the budget is a scale/ranking
instrument, not ownership evidence.

## Arm 1: BBL reference geometry — REFUTED

NEMO builds `mbku_d=max(mbkt(i),mbkt(i+1))`, signs the slope from reference
bottom T depths, and takes `e3u_bbl_0` from U-face reference thicknesses
(`src/OCE/TRA/trabbl.F90:507-533`).  At run time option 2 reads Kbb T/S and
Kmm geometric depth (`:342-353`) and applies the Campin--Goosse three-leg
exchange (`:207-290,415-454`).

The incoming legoESM operand used continuous partial-cell centroids and the
minimum adjacent bottom-T thickness.  On the certified mesh this creates 141
active faces instead of NEMO's 29; 112 are same-bottom-index faces where
NEMO's mask is exactly zero.  Shared-face thickness differs by as much as
`9.960723613572725 m`.  The scaling probe finds maximum induced one-step T
tendency changes `6.128389e-4 K s-1` at the midpoint and
`8.161556e-4 K s-1` at the final state: large enough to test.

The full arm nevertheless moves the census in the wrong direction,
`0.0094379841 -> 0.0100324114`.  Classification: **REFUTED as census owner**.
Because the new operands transcribe `trabbl.F90` and NEMO exposes no geometry
switch, they remain the production WS-RK3 option-2 behavior; the prior path is
available only as `legacy_bbl_partial_geometry` in private test hooks.

## Arm 2: raw W metric — REFUTED, with a retraction

NEMO divides the Aimp Courant transport by `e3w(Kmm)`
(`src/OCE/DYN/sshwzv.F90:776-843`) and uses the same raw W-grid reference
object in momentum ZDF (`src/OCE/DYN/dynzdf.F90:180-205`;
`src/OCE/DOM/domzgr_substitute.h90:126-133`).  legoESM used the midpoint of
adjacent live T thicknesses.  At partial bottoms that differs from 20 m by up
to `4.980361806786362 m`, a W-metric ratio as large as `1.33159`.

**Retraction.**  The first implementation wired that raw field into both Aimp
and ZDF while calling the run “Aimp only.”  It was stopped before 612 steps
and wrote no completed artifact.  Commit `8ace6513aa` records the correction;
the committed addendum `1b72ea5660` then preregisters three genuinely
discriminating runs:

| arm | one variable | final census distance | label |
|---|---|---:|---|
| Aimp-only | raw W in partition, legacy ZDF midpoint | `0.010032411356701666` | **REFUTED**, no movement |
| ZDF-only | raw W in ZDF, legacy Aimp midpoint | `0.010032414772546489` | **REFUTED**, wrong sign and only `3.4e-9` |
| combined reference package | raw W in both NEMO consumers | `0.010032414772546482` | **REFUTED**, no interaction |

The Aimp scaling rows also show the implicit fraction and implicit transport
are exactly zero at initial, midpoint, and final registered states.  The ZDF
scaling arm is momentum-only directly (`rn_avt0=0`): its one-step U effect is
`5.3814e-7` at the midpoint and `6.1956e-7` at the endpoint, so it is
scale-compatible but causally the wrong size/sign for the census.

The source-exact raw `e3w_0*(1+r3t/r3u/r3v)` ladder now feeds both consumers
inside the NEMO WS-RK3 identity.  Legacy midpoint restores are private hooks
only.  This is the reference package NEMO actually runs, not a public
composition of micro-selectors.

## FCT bottom boundary and stopping rule

The remaining registered question was reread rather than guessed.  NEMO's
vertical FCT flux arrays are initialized to zero, interior interfaces are
filled, and the bottom interface stays zero before both low-order predictor
passes (`src/OCE/TRA/traadv_fct.F90:470-629`).  legoESM likewise pads the
vertical transport with zero top and bottom fluxes, applies the wet mask, and
executes the already-certified two-step `nn_fct_imp=1` predictor.  No
source-level bottom-BC discrepancy was found to arm.  Structural disposition:
**VERIFIED**.  Numerical NEMO term parity remains **UNMEASURED** because the
oracle has no matching internal flux dump.

This satisfies the frozen stopping rule: BBL and both raw-W consumers are
refuted, and FCT exposes no source mismatch.  The register is exhausted; no
post-hoc fourth arm is invented.

## Regression gates

All runs use the common NEMO halo strip, certified wet-mask intersection, and
native staggering.  The kt trajectory rows are Nbb/before instantaneous fields;
U is a three-dimensional instantaneous U-face field, not `un_adv` time-mean
transport.

| kt | T normalized L-inf | U normalized L-inf | SSH normalized L-inf |
|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` |
| 2 | `7.81597009336e-15` | `7.06425196118e-12` | `1.05054853705e-14` |
| 3 | `5.72573100044e-12` | `4.18144414926e-9` | `1.50946095900e-13` |
| 4 | `4.11080058882e-11` | `2.33231629981e-8` | `2.89701387879e-10` |
| 5 | `1.18363807644e-10` | `4.62438738622e-8` | `1.17943627920e-8` |
| 6 | `2.37059349928e-10` | `3.18436057283e-7` | `1.05324352295e-7` |
| 7 | `3.72533559556e-10` | `1.26263117927e-6` | `3.89638123377e-7` |
| 8 | `5.48413847667e-10` | `2.83786255263e-6` | `7.51826953360e-7` |
| 9 | `9.63236157503e-10` | `4.34409136913e-6` | `8.26113347008e-7` |
| 10 | `1.52259440611e-9` | `5.42269569600e-6` | `6.26501828027e-7` |
| 60 | `1.36521296312e-7` | `3.22638371373e-5` | `1.23344556174e-5` |

The kt=2 certified headline remains T `7.8e-15`, U `7.1e-12`, SSH
`1.05e-14`; classifications are unchanged.  The 19-frame `--compare-to`
gate passes at kt=1.  At kt=2--4 it passes every
`reseeded_from_oracle_entry` row; inherited-entry rows are intentionally
excluded because this source correction changes the upstream trajectory and
the trajectory gate reports that movement directly.  The filtered row set is
stamped in each comparison log.  The stage-sweep comparison likewise passes
all `.faithful.` rows under the two-ULP gate; private causal arms are excluded
because their definitions are the deliberate subject of this round.  The
unfiltered trajectory `--compare-to` result is **FAIL**, as it must be: the
source-faithful raw-W package changes T/S/U/SSH reductions above two ULPs from
kt=3 onward.  The table above is the regenerated post-change trajectory, and
its AT-BAR/DEBT classifications—not bit identity with the old wrong operand—
are the frozen regression guard that remains unchanged.

## Full-duration statistical verdict

Fresh production fp64 and fp32 runs each complete all 6120 CPU steps with
finite state.  Their metadata print and validate every state and geometry
dtype; the fp64 state NPZ is bit-identical to the preregistered combined arm.

| metric | candidate distance | precision floor | NEMO scheme spread | verdict |
|---|---:|---:|---:|---|
| final temperature histogram TV | `0.03400880367` | `0.00692327669` | `0.04423095060` | WITHIN-SCHEME-SPREAD |
| final water-mass census | `0.01003241477` | `0.00074342403` | `0.00336209095` | **OUTSIDE** |
| instantaneous U L-inf | `0.66075722687` | `0.07487340307` | `0.67269430107` | WITHIN-SCHEME-SPREAD |
| plume descent | `1.422447776 m` | `0.000661150 m` | `1499.623281 m` | WITHIN-SCHEME-SPREAD |
| plume front | `3.117200197 km` | `0.031029903 km` | `121.930713 km` | WITHIN-SCHEME-SPREAD |
| temperature L-inf | `0.34010634999` | `0.05090695688` | `0.35674371774` | WITHIN-SCHEME-SPREAD |

Final status: `5 WITHIN-SCHEME-SPREAD / 1 OUTSIDE / 0
INDISTINGUISHABLE-AT-FLOOR`.

## Isomorphism and tests

The production changes extend the branch-isomorphism registry rather than
creating a second scheme: S-20 binds the Aimp W metric and S-34 binds the
literal ZDF divisor to the same raw NEMO mesh field.  Private legacy hooks are
instrumentation only.  `test_no_scheme_duplication.py` remains green.

The final test count is reported by file, not averaged with earlier rounds:

* 6 census probe controls;
* 13 OVERFLOW 19-frame gate controls;
* 13 BBL kernel/geometry tests;
* 17 NEMO testcase recipe/geometry tests;
* 17 implicit W-slot tests;
* 35 scheme-duplication/isomorphism tripwires.

Total: **101 tests**.  The external JUnit/log hashes and all scientific
artifact hashes are pinned in
`nemo_testcases_l1_census_owner_artifacts.sha256`.

No NEMO rerun was needed.  All oracle fields came from the certified phase-1
and later lane-1 artifacts; all new integrations were legoESM CPU runs.
