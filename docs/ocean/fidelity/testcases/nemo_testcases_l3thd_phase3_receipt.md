# NEMO testcase fidelity receipt — lane 3b, SI3 Phase 3

Tracker: `climate-federation/legoESM#1699`

Implementation commit: `80b07f39afa` (recovery git at
`/tmp/codex-si3thd-localgit`).

## Verdict

**DEBT, with four active physics corrections and one active arithmetic-order
transcription in the implementation commit.**  The original version of this
receipt disclosed only two: the step-74 negative-evaporation snow-deposition
sequence (`icethd_dh.F90:179-202`) and the `qml_ice` snow-first surface-melt
sequence (`icethd_dh.F90:107-120,204-315`).  Review found that the same commit
also carried the ZDF no-snow/melting-surface row ranges
(`icethd_zdf_bl99.F90:433-513`), the bottom-up basal-melt layer loop
(`icethd_dh.F90:309-315,366-424`), a snow-ice salinity term
(`icethd_dh.F90:441-485,509-519`), and three NEMO-order EOS conversions
(`icevar.F90:938-946`; `icethd.F90:233-243`; `icethd_dh.F90:498-503`).
The later preregistered Phase-5 ablations show that the first two extra physics
changes and the EOS ordering are active; the snow-ice salinity term is inert
in this one-year column.  Thus “two confirmed owners” was an incomplete scope
claim and is retracted.  All disabled arms are private gate hooks; no public
selector can construct the non-NEMO combinations.

The first row above the fixed `1e-15` bar moves from kt3 `POST_ZDF.e_i`
(`2.559660701514903e-15`) to kt5 `POST_DO.e_s`
(`1.011825579870701e-15`).  The final exact-entry census is still DEBT:
37,659 of 621,960 field rows are over-bar and its maximum normalised error is
1.0.  The fixes therefore support the measured improvements below, not a
whole-year trajectory claim.

## Source-first and pre-implementation search

The active call order is `ice_thd_frazil`, 1-D conversion, `ice_thd_zdf`,
`ice_thd_dh`, `ice_thd_temp`, `ice_thd_sal`, a second `ice_thd_temp`, 2-D
conversion, `ice_thd_do`, then `ice_cor` (`icethd.F90:109-190`).  HFN
redistribution, ponds, virtual-ITD melt, and lateral melt remain inactive in
the resolved `jpl=1`, `ln_pnd=.false.`, `ln_virtual_itd=.false.`,
`ln_icedA=.false.` deck (`icethd.F90:158-181`).

Before implementation, the search covered `packages/ice/legoesm/ice/sea_ice.py`,
the thermodynamic modules, `brine/`, `snow/`, and the promoted
`bitz_lipscomb.py` code originating in `_future/bitz_lipscomb.py`.  The change
extends that existing SI3 path.  It reuses `snow.py::snow_ice_flooding`; it
does not add a second ice model or duplicate that freeboard formula.  The
shared bulk snow-consumption helpers cannot preserve NEMO's four ordered
enthalpy segments, so the already-existing SI3 segment carrier is retained
for `snw_ent`-order remapping (`icethd_dh.F90:494-507,535-613`).

## Corrected per-change scope table

This table is the post-review disposition of every change bundled in
`80b07f39afa`.  Counts come from the committed Phase-5 artifact's 621,960
exact-entry field rows and 61,320 continuous field-step rows.  “Active” means
the private one-variable arm changed measured outputs; it does not imply that
all remaining column debt is owned by that change.

| bundled change | NEMO transcription | one-variable result | disposition |
|---|---|---|---|
| negative-evaporation deposition | `icethd_dh.F90:166-202` | kt74 `POST_DH.h_s` improves by `8.18e9`; planted arm exits red | **CONFIRMED owner** |
| snow-first surface melt | `icethd_dh.F90:107-120,204-315` | no-surface arm moves minimum `0.55518→1.78596 m` and growth day `251→261` | **CONFIRMED owner** |
| ZDF no-snow/melting row ranges | `icethd_zdf_bl99.F90:433-513` | 39,505 exact rows change; disabled has 54,551 over-bar rows versus 36,852 enabled; minimum becomes `0.382871 m`, growth day 252 | **ACTIVE correction; omission from the original scope account retracted** |
| bottom-up basal layer loop | `icethd_dh.F90:309-315,366-424` | 11,190 exact rows change; disabled has 36,944 over-bar rows versus 36,852 enabled; the legacy-loop continuous minimum differs by `6.16e-13 m` | **ACTIVE correction**; basal melt existed only as a simplified bottom-layer calculation, so “already present” was incomplete |
| snow-ice contribution to bulk salinity | `icethd_dh.F90:441-485,509-519` | 0/621,960 exact rows and 0/61,320 continuous field-step rows change | **INERT for this C1D year**; retained because it is unconditional in the selected source identity |
| three EOS operation orders | `icevar.F90:938-946`; `icethd.F90:233-243`; `icethd_dh.F90:498-503` | 24,293 exact rows change; enabled/disabled over-bar counts are 36,852/38,233; zero ZDF surface-branch splits | **ACTIVE arithmetic-order transcription**, not a phenomenology owner |

## Step 74: first branch owner

At kt73, evaporation is `+1.400605059070903e-7 kg m-2 s-1`.  At kt74 it
changes sign to `-1.665210755849663e-7`, while snow precipitation is already
active at `4.320384505263064e-6 kg m-2 s-1`; this is not first snowfall.
NEMO computes precipitation first (`icethd_dh.F90:166-177`), then
`zdeltah=max(-evap/rhos*dt,-h_s)` and deposits snow when it is positive
(`:179-202`).  The pre-fix legoESM path had no corresponding branch, so its
effective deposition condition was absent/false on the same entry.

ENTRY and POST_ZDF remain in the arithmetic class.  The first material state
split is POST_DH:

| row | pre-fix absolute error | enabled absolute error | improvement |
|---|---:|---:|---:|
| `h_s` | `1.816593551834389e-6 m` | `0` | `8.181210043e9` |
| `e_s` | `58.0184666514 J m-3` | `1.6391277313e-7 J m-3` (`1.3133e-15` normalised) | `3.539593989e8` |

The thickness increment is exactly `-evap*3600/rhos`.  The executing repair
is `bitz_lipscomb.py:673-708`: it preserves NEMO's precipitation-before-
deposition order and uses the surface-temperature enthalpy only when NEMO's
`ze_s(0)` remains zero (`icethd_dh.F90:186-188`).

The alternatives are refuted at the first boundary: `t_su=249.4340162584 K`,
so surface melt is inactive; the thick-ice freeboard makes flooding zero;
salinity occurs later; concentration stays 0.9; HFN is inactive at `jpl=1`;
and correction/zapsmall occurs only after POST_DO.  The retained machine
states for both models at all eight registered boundaries of kt73–76 are in
the two census JSONs.  Each frame includes its Rule-1d time level and source
from the frame registry (`nemo502_si3thd_MY_SRC/icethd.F90:116,158-170,196,231`).

Before repair, kt74's largest normalised row goes from `1.232e-15` at POST_ZDF
to `1.817e-6` at POST_DH and `9.301e-6` at EXIT; after repair, the respective
maxima are `4.776e-16`, `1.075e-15`, and `1.089e-15`.  Kt75–76 similarly
return from `1e-6..1e-5` material errors to the `1e-15` class.

## Exact-entry census and remaining branch debt

The retained pre-fix artifact enumerates all 116,274 over-bar rows by step,
sub-call, and variable.  Its first one-step injection over `1e-12` is kt74
`POST_DH.h_s`, `1.816593551834389e-6`, in the negative-evaporation deposition
branch.  Its first over `1e-3` is kt3837 `POST_ZDF.e_s`, normalised
`0.003146445537810156`, in the fixed-melting-surface branch.  The maximum is
`1.1108092231885993e8`.

The post-fix artifact enumerates all 37,659 remaining rows.  Its first
one-step injection over `1e-12` moves to kt4239 `POST_ZDF.e_i`, absolute
`788.4015902281 J m-3`, normalised `2.6663649474e-6`.  Both models take the
snow-present/fixed-melting branch.  The registered ENTRY has positive but
tiny snow volume and zero snow enthalpy; NEMO carries `t_s` independently
through `ice_thd_1d2d` (`icethd.F90:350,463`), but the ENTRY frame does not
register `t_s` (`nemo502_si3thd_MY_SRC/icethd.F90:269-274`).  This row is an
operand-registration debt, not evidence of a different implemented formula.

The first post-fix row over `1e-3` is kt5285 `POST_ZDF.t_su`, absolute
`0.7936539212 K`, normalised `0.002914027643`.  Here that missing carried snow
temperature changes the iterative branch: NEMO remains below `rt0`, while
legoesm reaches `rt0` and takes the fixed-melting surface arm.  The JSON prints
both Boolean predicates.  NEMO's branch and row ranges are
`icethd_zdf_bl99.F90:433-513`; the executing legoESM implementation is
`bitz_lipscomb.py:497-561`.  This disclosed input-state debt is why the final
census remains DEBT.

## Melt-season owner and scaling

NEMO forms `qml_ice=qns+qsr-qtr-qcn_top` only at a melting surface, consumes
snow segments first, then surface ice (`icethd_dh.F90:107-120,204-315`).
The executing implementation is `bitz_lipscomb.py:710-779`.  The first exact-
entry snow removal is kt3836 POST_DH; enabling the path reduces `h_s` error
from `1.2869717243e-4 m` to `5.5511151231e-17 m`.  The first ice removal is
kt4238; `h_i` error falls from `6.2086120068e-4 m` to
`2.2204460493e-15 m`, and `e_i` from `2804.4601199` to
`5.9604644775e-8 J m-3`.  Each improvement exceeds `1e8`.

All scales below are sums of independent exact-entry POST_DH arm differences;
they identify dimensional relevance, not additive attribution in a nonlinear
trajectory.

| candidate | first active | exact-entry scale | ratio to old `1.2273381762 m` minimum gap | baseline status |
|---|---:|---:|---:|---|
| `qml` surface ice melt | kt4238 | `0.8635115099 m` | `0.70356` | missing |
| snow-melt-first shield | kt3836 | `0.5500534667 m` snow = `0.1979472672 m` ice-equivalent | `0.16128` | missing with `qml` path |
| basal-melt sensitivity | kt2776 | `1.0371414870 m` | `0.84503` | a simplified bottom-layer path existed, but the source bottom-up layer loop was also rewritten in this commit |
| `qtr` transmission sensitivity | kt4242 | signed `0.4777193222 m` ice | `0.38923` | already supplied by pinned NEMO input |

The decisive trajectory arm changes only `_surface_melt`: disabled gives a
minimum `1.7859553657 m` and growth day 261, reproducing the old seasonal
deficit; enabled gives `0.5551788291 m` and day 251.  The NEMO comparison is
`0.5550952377 m` and day 251.  Thus the missing surface path, not the already-
present basal or shortwave paths, owns the reported deficit.

## Full-year growth table

Entries are pre-fix → post-fix normalised L-infinity errors.  The committed
JSON retains every hourly row.

| step | `t_su` | `e_i` | `e_s` | `h_i` | `h_s` | `a_i` | `sv_i` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.563e-16 → 4.450e-16` | `2.489e-15 → 3.556e-16` | `6.657e-16 → 5.325e-16` | `0 → 0` | `5.551e-17 → 5.551e-17` | `0 → 0` | `0 → 0` |
| 10 | `4.522e-16 → 2.261e-16` | `5.308e-15 → 7.607e-15` | `1.253e-15 → 2.381e-15` | `2.217e-16 → 6.651e-16` | `3.886e-16 → 3.886e-16` | `0 → 0` | `1.564e-16 → 6.254e-16` |
| 100 | `2.923e-5 → 1.845e-15` | `7.569e-7 → 3.966e-14` | `6.661e-4 → 1.593e-14` | `8.795e-13 → 3.513e-15` | `1.305e-4 → 2.859e-15` | `0 → 0` | `3.579e-12 → 7.338e-15` |
| 1000 | `8.369e-4 → 1.671e-14` | `6.925e-5 → 1.452e-13` | `1.882e-2 → 1.251e-13` | `2.106e-6 → 1.274e-13` | `5.221e-3 → 3.031e-14` | `0 → 0` | `9.460e-6 → 2.322e-13` |
| 8760 | `1.061e-1 → 6.633e-6` | `9.098e-1 → 1.634e-4` | `1.735 → 9.141e-6` | `7.932e-1 → 1.490e-4` | `4.396e-1 → 6.317e-14` | `0 → 0` | `5.310e-1 → 1.044e-4` |

## Six phenomenology rows and precision floor

The floor is legoESM fp32-vs-fp64, not an unavailable NEMO scheme spread.
Dates use the preregistered hourly/daily definition.

| quantity | NEMO | legoESM fp64 before | legoESM fp64 after | fp64-NEMO distance after | fp32-fp64 floor | status |
|---|---|---|---|---:|---:|---|
| minimum thickness | `0.5550952377 m` | `1.7824334139 m` | `0.5551788291 m` | `8.3591e-5 m` | `0.1106972 m` | **AT-FLOOR** |
| maximum thickness | `2.4456886226 m` | `2.4457929329 m` | `2.4456886226 m` | `6.7280e-13 m` | `5.1158e-5 m` | **AT-FLOOR** |
| minimum date | 2018-09-09 00Z | 2018-09-19 00Z | 2018-09-09 00Z | `0 h` | `2304 h` | **AT-FLOOR** |
| maximum date | 2018-05-13 00Z | 2018-05-13 00Z | 2018-05-13 00Z | `0 h` | `0 h` | **AT-FLOOR** |
| melt onset | 2018-05-14 | 2018-05-14 | 2018-05-14 | `0 d` | `0 d` | **AT-FLOOR** |
| growth onset | 2018-09-09 | 2018-09-19 | 2018-09-09 | `0 d` | unavailable (fp32 has no onset) | **UNMEASURED** |

`AT-FLOOR` is used only for the measured precision-floor classification.  The
remaining operator and year-end state errors stay explicit above.

## Provenance, controls, and hashes

The column consumes the unchanged CPU/fp64 replay root
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase2b_replay`.
No shipped NEMO source/configuration or prior run root was modified or deleted.
The forcing remains the official ERA5 member; no synthetic input is involved.

| artifact | SHA-256 |
|---|---|
| ERA5 forcing | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| 70,080 thermodynamics frames | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| 8,760 exact ZDF input frames | `5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27` |
| pre-fix retained operator census | `d642f532ed92910b49f29919dd5e4ae3fe84381ecd1536b7886a2c1f4b36f95f` |
| Phase-2b baseline year JSON | `6c21d14f3c85d0fa99be7e31770a4a4dcad548d98f857ea78455bd6344622160` |
| post-fix Phase-3 year JSON | `77c78a816b254484afe06632d77378336ddac1153ef9190d17c555f41c45a6e1` |

The owner, branch-census, arithmetic, and cursor plants all exit nonzero.  A
clean checkout of the originally reported direct suite collects and reports
`19 passed` (the prior `18 passed` count omitted one collected test); with the
three exact touched-file constant
ratchets it reports:

```text
============================= 23 passed ========================================
```

That ratchet run includes the collected line
`tests/test_no_hardcoded_constants.py ... [100%]` for the package/test files.
Because its discovery excludes validation scripts, its own `banned_hits`
function was also run directly on all five touched Python files; every printed
`[]`.  A broad invocation exposed unrelated existing failures outside this
lane; they were not described as untouched successes.

## Coupled-rung debt

| component | status | reason |
|---|---|---|
| NEMO `sbcblk` bulk fluxes | **DEBT / NOT CERTIFIED** | The isolated column consumes NEMO-written `qns_ice`/`dqns_ice`; it does not recompute `sbcblk.F90:1273,1480-1491` feeding `icestp.F90:201,206`. |
| tiny-snow ENTRY temperature | **DEBT / NOT REGISTERED** | NEMO carries `t_s`; the exact-entry frame has only enthalpy, which can be zero at tiny positive snow volume.  This owns the post-fix kt4239/kt5285 operator rows described above. |

## End-of-task choice register

- ASKED — identify the first kt74 branch disagreement from existing frames and
  active NEMO source.
- ASKED — retain every pre/post-fix over-bar operator row and report the first
  injections over `1e-12` and `1e-3` with branch predicates.
- ASKED — scale surface, bottom, shortwave, and snow-first melt candidates
  against the observed minimum-thickness gap.
- ASKED — preregister private one-variable arms, repair confirmed owners inside
  the sole ORCA1-resolved SI3 identity, and keep NEMO's unconditional paths
  unbranched in production.
- ASKED — rerun all 8,760 steps/70,080 frames; report the seven-field growth
  table and six phenomenology rows before/after.
- ASKED — CPU/fp64 oracle execution, copy-only NEMO work, explicit-path local
  commits/bundle, no push.
- UNASKED — alternate SI3 identities, public physics switches, threshold or
  forcing changes, coupled bulk-flux certification, GPU/MPI execution, or
  deletion of old run roots.

## FLAGGED FOR FUTURE DELETION

Nothing was deleted.  All earlier C1D exploratory/run roots remain on disk and
are only flagged for a future owner decision.
