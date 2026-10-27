# ORCA2 round 340 — exact standalone S4 acquisition tool and composed card

Date: 2026-10-27. Base `d5d97cfc3121`. Disposition:
**TWINS_ADMITTED (`ORCA2_ROUND340_S4_TWINS_ADMITTED`) / COMPOSED_CARD_SCORED (`FAIL_S4_SURFACE`)**.
Both operator-run NEMO arms A and B are admitted (`0` unequal words across all
surface and rung-10 state records); no agent ran NEMO.

## Source order and delivered scope

The admitted K0 ppsrc calls S0-M at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:428`,
the bulk selector at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:440-449`,
the observed-ice interface at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:499-503`,
then runoff/restoring/FWB at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:517-521`
and stress rotation at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570`.
The observer brackets those executing boundaries outside NEMO loops and now
retains the runoff, restoring, freshwater-budget, and rotation subframes.
The two bulk routines additionally emit the S1 turbulent state and its
precipitation/longwave flux state, so `theta_air` through `qlwn` are not
inferred from the outer frames.
Inside S3 it records immediately after the real field read
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:149`),
freezing-point evaluation
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:158`),
temperature clamp
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:166`),
and the ice-cover flux arm
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:177`).

Step entry is immediately before the real surface call at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:128`.
Each stage checkpoint is after its boundary work and before scratch release at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:582-585`.
The record reader derives its exact time-level sequence from the initialized
`Nbb/Nnn/Naa = 1/2/3` state
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90:368`)
and the three executing swaps/calls
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:185-197`).

The operator tool is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_s4_acquisition/run.sh`.
Its default preflight patches the admitted real K0 ppsrc, syntax-checks the
writers, compiles all seven changed production units at the production flags,
obtains nonzero and equal FP census counts (355, 30, 1,175, 59, 50,
569/294/107, and 1,161), and fires an injected FP-operation plant. It
bounds-compiles those seven real patched sources and executes a two-rank
`-fcheck=bounds` writer fixture with rank-distinct and negative-zero data.
The mixed full/interior allocations are taken from the real K0 ppsrc
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbc_oce.f90:195-216`).
It verifies both round-339 manifests and exercises real-ppsrc and whole-build
same-program plants, placement, namelist schema, checksum, generated-driver
syntax, refusal, retention, retained admission, and success controls. It reports
`ORCA2_ROUND340_S4_PREFLIGHT_READY`.

The legoESM card wraps the existing certified rung-10 ocean rather than
copying its dynamics. `NemoStandaloneK0Config` has no defaults and accepts exactly
`nn_ice=1`, drag off, `ln_mxl0=true`, `nn_mxlice=0`, and icebergs off. The
call accepts physical CORE/ice operands but no NEMO exchange record, executes
its own S0-F interpolation, then S0-M -> S1 -> S3 -> S2, and hands those
forcings to `LatLonCGridOceanModel.step`. The ten-step driver carries the
resulting three-dimensional state and materializes entry plus all three RK3
stage exits at each step (200 T/S/u/v/ssh rows). The gate reads the
digest-pinned Decision-171 binary64 native-grid cover and exposes a direct
NetCDF CORE reader without fallback or regridding.

The record reader unpacks every field, validates exact trailing length and
finiteness, removes the two-cell NEMO halos and dummy bottom level from state
records, slices full surface fields to the writer-declared owned cells, and
joins both MPI slabs. The composed scorer first enforces 0/n for every
registered surface field, then imports round 332's existing normalized
per-row rung-10 bar (`<= 1e-15`) for all 200 rows. A one-ULP surface plant and
an over-bar rung-10 plant refuse, while the established at-bar one-ULP state
case remains admitted. The focused test runs the whole S0->S1->S3->S2
production call, proves the ocean driver is invoked for a three-dimensional
state, produces all ten ladder steps, and confirms finite N=1 forward and
reverse gradients, with reverse equal to forward.

Resume 2 addressed Argon's original HOLD groups. The composed driver carries
S0/S2 state and the preceding surface result across all ten calls, enforces
the K0 `nn_fsbc=2` cadence, updates the live SST slot after the ice-interface
clamp, and derives S0 means from the live ocean time levels, thickness stretch,
and step number. Chlorophyll, geothermal heat flux, and ice concentration are
mandatory: the first two come from the same digest-pinned deck files as K0.
Resume 3 corrects the TKE input to the interior binary `fr_i` written by the
ice-interface arm; observed `sf_ice(1)%fnow` remains confined to that arm's
heat-flux calculation. A post-fix operator smoke ran the real
rung-10 `LatLonCGridOceanModel` through all ten `run_composed_card` steps on
CPU/fp64 in 766.84 seconds and found every final state leaf finite.

Surface and ice-interface records now use boundary-specific inventories: the
pre-initialization frame is header-only, later frames add only fields already
defined at that source boundary, and every full-grid value is sliced to the
owned `qsr` bounds before writing. The S0-F v3 writer likewise emits only the
owned `fnow` and associated `fdta` slabs, while its decoder retains v1/v2
compatibility. Scoring uses the exact time-qualified keys
emitted by the loader, including S0-F and ice-interface labels.

The bounds executable links all seven patched bounds objects, executes every
one through its patched-module probe, and runs every surface kid 0--11 and
ice-interface kid 0--3. Its A2D(0) extents are generated
from the admitted `sbc_oce` allocation after the deck's `nn_hls=2` value is
verified, giving `3:jpi-2,3:jpj-2`; a `NOT USED` FLD entry retains untouched
pointers and emits no frame. The generated run driver
captures patched sources, parent manifest, and a producer manifest before
launch. Its refusal trap remains armed through admission and restart/twin
comparison. Preflight routes retained evidence through the actual
`--admit-existing` entry point and proves that either log containing
`E R R O R` is quoted and quarantined. A failed retained-run admission does
not move or rename already-quarantined evidence.

Resume 3 also replaces the frozen CORE record with the records NEMO selects at
its own `nn_fsbc=2` clock. Atmospheric `isecsbc` is `kt*10800`: six-hourly
wind, air temperature, humidity, and pressure advance through records 1--5 at
steps 1,3,5,7,9; daily radiation advances to record 2 at step 9; monthly
precipitation remains on record 1. Runoff and SSS use the same surface clock.
The cold-start shortwave fraction is 1.0 and each next S0-M call consumes the
fraction recomputed from the preceding real stage-3 Kmm thickness. The
ten-step composed driver returns the materialized S0/S1/S3/S2 boundary
dictionary used by the surface-first scorer. Its 99-field S0-F inventory also
materializes all three `kt=1` initialization calls, all odd-step surface calls,
and every per-step chlorophyll and 31-level T/S call with two-rank metadata.

## Checklist and choices

The acquisition checklist is encoded in the preflight or run admission. Array writes use
declared full or interior bounds; observer calls are outside NEMO's loops; the
build uses `-O3 -funroll-all-loops -fno-tree-vectorize`; same-program proof
strips exactly seven patches and two writer units; manifests use relative paths and
pin producer commit and executable; exact readers reject duplicate, missing,
truncated, trailing, wrong-rank, wrong-level, or nonfinite frames; refused
evidence stays quarantined until build-source capture, manifest creation,
clean stdout `STOP 0`, both-log error scans, restart comparison, and admission complete; the
README gives retained/in-place admission and move-aside retry instructions. The
operator's first `--run` completed arm A, but the wrapper silently exited
because it incorrectly required `STOP 0` in `ocean.output`; NEMO emits that
marker in launcher stdout. The retained route finalized the interrupted
manifest, admitted the complete record inventory, and moved arm A to its
canonical location. Admission now scans both logs for errors, every unexpected
shell failure names its step in a `REFUSE` line, and `--run` resumes with arm B
only when it finds admitted arm A and the retained executable.

- ASKED: exact K0 acquisition, independent twin, source boundaries, existing
  ten-step state ladder, explicit K0 model card, exact surface-first scoring,
  finite N=1 gradients, and no author-run NEMO.
- UNASKED: none.
- independent review: operator

## Resume 6 — Argon re-review fixes and ten-step composed-card score

Resume 6 resolves all three items from the Argon re-review without modifying
`packages/ocean/legoesm/ocean/`:

1. **North-fold T-pivot stress LBC and `vmask` on row `-1`**:
   `_build_stress_lbc_and_masks` folds and negates only the right half
   `91..179` (`nx // 2 + 1:`) of the T-pivot row `center_index[-1]` /
   `center_sign[-1]` per
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/lbcnfd.f90:619-628`
   (preserving `+0.0` at `i = 0, 90`), folds `north_index[-1]` /
   `north_sign[-1]` from row `-2` per
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/lbcnfd.f90:588-606`,
   and folds `vmask[-1]` from `vmask[-2, perm_v]` per
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dommsk.f90:232`.
   Replaying arm-A `kt=1` `S3_EXIT` (`utau`, `vtau`) through
   `nemo_t_stress_lbc_neighbors` and `nemo_t_stress_to_uv` gives `0 / 26,640`
   unequal words on `S2_LBC/utau`, `S2_LBC/vtau`, `S2_EXIT/utauU`, and
   `S2_EXIT/vtauV` (`test_item1_north_fold_t_pivot_stress_lbc_and_vtauv`).
2. **Even-`kt` surface boundary retention (`nn_fsbc = 2`)**:
   On even `kt`, NEMO skips the `MOD( kt - 1, nn_fsbc ) == 0` updates per
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521`,
   retaining `kt - 1` `S2_EXIT` (`tail`) values across outer `kid = 3..10` and
   entering/exiting `sbc_ice_if` (`S3_COVER_IIF`, `S3_EXIT_IIF`) with the
   retained `tail.emp`, `tail.utau`, `tail.vtau`, `post.qsr` (`tail.qsr`), and
   `tail.qns`. `_surface_boundaries_for_step` now emits those retained values on
   even `kt` (`test_item2_even_kt_surface_boundaries_retain_prior_s2_exits`).
3. **Admission order and shared NEMO-`ALLOCATE` extent gate**:
   `run.sh` now invokes `check_same_program_build` (`same_program_observers`)
   at the entry of `admit_pair` (before `admit_one`) and in
   `admit_existing_route` (`--admit-existing`), and replaces the inline
   `qns`-only regex with `verify_r340_allocated_extents` backed by the shared
   Fortran-`ALLOCATE` parser in `nemo_testcase_l4_orca2_round339_common.py`
   across `sbc_oce`, `sbcrnf`, `sbcssr`, `dom_oce`, and `oce`
   (`test_item3_shared_allocate_extent_gate_and_admit_order`).

### Reproducibility control (admitted arm B vs admitted arm A)

The committed scoring driver
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_score_composed_card.py`
first scores admitted arm B (`orca2_s4_exact_b_np2`) against admitted arm A
(`orca2_s4_exact_a_np2`) through `score_composed_card`:

- Status: `PASS_REPRODUCIBILITY_EXACT`
- Surface boundaries (`259` time-qualified boundaries, `1,899` field rows):
  `0 / 123,824,304` unequal words (`max |diff| = 0.0`)
- Rung-10 state ladder (`40` checkpoints, `200` field rows):
  `0 / 128,937,600` unequal words (`200 / 200 BIT_EXACT`, `max |diff| = 0.0`)

### Real legoESM composed-card score (`kt = 1..10` vs arm A)

Running the production ORCA2 K0 composed card (`NemoStandaloneK0Config(nn_ice=1,
ln_drgice_imp=False, ln_mxl0=True, nn_mxlice=0, ln_icebergs=False)` explicit)
on CPU (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`)
for `kt = 1..10` and scoring against arm A via `score_composed_card`:

- Overall status: `FAIL_S4_SURFACE` (`surface exact bar failed: 58062489 / 123824304`)
- First non-bit surface row: `kt01/S0_EXIT/sst_m` (`4,668 / 26,640` unequal
  dry-cell words `-0.0` vs `+0.0`, `max |diff| = 0.0`)
- First non-zero-`max |diff|` surface row: `kt01/S0_EXIT/ssh_m`
  (`16,433 / 26,640` unequal wet-cell words, `max |diff| = 0.093435058726849693`)
- First non-bit rung-10 row: `kt01/entry/T` (`151,917 / 799,200` unequal
  dry-cell words `-0.0` vs `+0.0`, `max |diff| = 0.0`, `AT_BAR_NOT_EXACT`)
- First `OVER_BAR` rung-10 row: `kt01/entry/ssh` (`16,433 / 26,640` unequal
  wet-cell words, `max |diff| = 0.093435058726849693`,
  `normalized_max_abs = 0.093435058726849693 > 1e-15`)

#### Surface boundaries by step (`kt = 1..10`)

| Step | Unequal / Total | Max `\|diff\|` (all) | Max `\|diff\|` (excl. `S0_F` `tradmp` slot metadata) | First non-bit row |
| --- | --- | --- | --- | --- |
| `kt01` | `3,813,813 / 18,035,664` | `2,678,400.0` | `126.98717158149199` | `kt01/S0_EXIT/sst_m` |
| `kt02` | `4,816,044 / 10,842,528` | `2,678,400.0` | `0.18087092394585402` | `kt02/S0_EXIT/sst_m` |
| `kt03` | `6,517,647 / 12,894,000` | `2,678,400.0` | `127.08469959657452` | `kt03/S0_EXIT/sst_m` |
| `kt04` | `5,840,139 / 10,842,528` | `2,678,400.0` | `37.454735975171342` | `kt04/S0_EXIT/sst_m` |
| `kt05` | `6,516,488 / 12,894,000` | `2,678,400.0` | `127.18207189455814` | `kt05/S0_EXIT/sst_m` |
| `kt06` | `5,839,307 / 10,842,528` | `2,678,400.0` | `37.456757650794302` | `kt06/S0_EXIT/sst_m` |
| `kt07` | `6,517,860 / 12,894,000` | `2,678,400.0` | `127.27955704224556` | `kt07/S0_EXIT/sst_m` |
| `kt08` | `5,840,350 / 10,842,528` | `2,678,400.0` | `37.45901721853463` | `kt08/S0_EXIT/sst_m` |
| `kt09` | `6,519,419 / 12,894,000` | `2,678,400.0` | `127.37701245735659` | `kt09/S0_EXIT/sst_m` |
| `kt10` | `5,841,422 / 10,842,528` | `2,678,400.0` | `37.587321078771261` | `kt10/S0_EXIT/sst_m` |
| **Total** | **`58,062,489 / 123,824,304`** | **`2,678,400.0`** | **`127.37701245735659`** | **`kt01/S0_EXIT/sst_m`** |

#### Surface boundaries by boundary family (`kt = 1..10` aggregate plus `kt01` and `kt02`)

| Boundary family | `kt01` unequal / total (`max \|diff\|`) | `kt02` unequal / total (`max \|diff\|`) | `kt = 1..10` unequal / total | `kt = 1..10` max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- | --- |
| `S0_ENTRY` | `0 / 0` (`0.0`) | `0 / 0` (`0.0`) | `0 / 0` | `0.0` | `None` |
| `S0_EXIT` | `37,534 / 186,480` (`0.093435058726849693`) | `124,168 / 186,480` (`0.18087092394585402`) | `1,155,046 / 1,864,800` | `0.18087092394585402` | `kt01/S0_EXIT/sst_m` |
| `S0_F` | `3,267,384 / 10,550,208` (`2,678,400.0`) | `3,267,384 / 3,863,328` (`2,678,400.0`) | `32,673,840 / 57,890,304` | `2,678,400.0` | `kt01/S0_F/c009/f001/votemper/fdta` |
| `S1_ENTRY` | `37,534 / 186,480` (`0.093435058726849693`) | `124,168 / 186,480` (`0.18087092394585402`) | `1,155,046 / 1,864,800` | `0.18087092394585402` | `kt01/S1_ENTRY/sst_m` |
| `S1_TURB` | `77,539 / 506,160` (`126.98717158149199`) | absent on even `kt` | `1,281,689 / 2,530,800` | `127.37701245735659` | `kt01/S1_TURB/theta_air` |
| `S1_FLUX` | `0 / 159,840` (`0.0`) | absent on even `kt` | `197,196 / 799,200` | `8.7492443823707049` | `kt03/S1_FLUX/qlwn` |
| `S1_EXIT` | `37,534 / 399,600` (`0.093435058726849693`) | `128,836 / 399,600` (`0.18087092394585402`) | `1,811,244 / 3,996,000` | `37.587321078771261` | `kt01/S1_EXIT/sst_m` |
| `S3_ENTRY` | `37,534 / 399,600` (`0.093435058726849693`) | `128,836 / 399,600` (`0.18087092394585402`) | `1,811,244 / 3,996,000` | `37.587321078771261` | `kt01/S3_ENTRY/sst_m` |
| `S3_COVER_IIF` | `14,004 / 426,240` (`0.0`) | `70,400 / 426,240` (`0.0015752427908566347`) | `1,261,790 / 4,262,400` | `37.587321078771261` | `kt01/S3_COVER_IIF/sst_m` |
| `S3_FREEZE_IIF` | `14,004 / 452,880` (`0.0`) | absent on even `kt` | `668,429 / 2,264,400` | `8.7492443823707049` | `kt01/S3_FREEZE_IIF/sst_m` |
| `S3_CLAMP_IIF` | `4,668 / 452,880` (`0.0`) | absent on even `kt` | `659,093 / 2,264,400` | `8.7492443823707049` | `kt01/S3_CLAMP_IIF/sst_m` |
| `S3_EXIT_IIF` | `4,668 / 452,880` (`0.0`) | `70,400 / 452,880` (`0.0015752427908566347`) | `1,252,472 / 4,528,800` | `37.58738259116825` | `kt01/S3_EXIT_IIF/sst_m` |
| `S3_EXIT` | `37,534 / 426,240` (`0.093435058726849693`) | `128,836 / 426,240` (`0.18087092394585402`) | `1,811,262 / 4,262,400` | `37.58738259116825` | `kt01/S3_EXIT/sst_m` |
| `S2_ENTRY` | `37,534 / 426,240` (`0.093435058726849693`) | `128,836 / 426,240` (`0.18087092394585402`) | `1,811,262 / 4,262,400` | `37.58738259116825` | `kt01/S2_ENTRY/sst_m` |
| `S2_RNF` | `37,534 / 506,160` (`0.093435058726849693`) | `128,836 / 506,160` (`0.18087092394585402`) | `1,838,322 / 5,061,600` | `37.58738259116825` | `kt01/S2_RNF/sst_m` |
| `S2_SSR` | `42,202 / 586,080` (`0.093435058726849693`) | `128,836 / 586,080` (`0.18087092394585402`) | `2,101,266 / 5,860,800` | `37.587321078771261` | `kt01/S2_SSR/sst_m` |
| `S2_FWB` | `42,202 / 586,080` (`0.093435058726849693`) | `128,836 / 586,080` (`0.18087092394585402`) | `2,101,266 / 5,860,800` | `37.587321078771261` | `kt01/S2_FWB/sst_m` |
| `S2_LBC` | `42,202 / 586,080` (`0.093435058726849693`) | `128,836 / 586,080` (`0.18087092394585402`) | `2,101,266 / 5,860,800` | `37.587321078771261` | `kt01/S2_LBC/sst_m` |
| `S2_EXIT` | `42,202 / 639,360` (`0.093435058726849693`) | `128,836 / 639,360` (`0.18087092394585402`) | `2,370,756 / 6,393,600` | `37.587321078771261` | `kt01/S2_EXIT/sst_m` |

#### Rung-10 200-row state ladder by step (`kt = 1..10`)

| Step | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Unequal / Total | Max `\|diff\|` | Max normalized `\|diff\|` | First non-bit row | First `OVER_BAR` row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kt01` | `3` | `1` | `16` | `5,380,027 / 12,893,760` | `0.12232800948594585` | `0.10788153410639778` | `kt01/entry/T` | `kt01/entry/ssh` |
| `kt02` | `0` | `0` | `20` | `6,910,788 / 12,893,760` | `0.15316280730765836` | `0.11087151431396823` | `kt02/entry/T` | `kt02/entry/T` |
| `kt03` | `0` | `0` | `20` | `6,927,019 / 12,893,760` | `0.1562993752167991` | `0.098746931654029202` | `kt03/entry/T` | `kt03/entry/T` |
| `kt04` | `0` | `0` | `20` | `6,904,727 / 12,893,760` | `0.1562993752167991` | `0.096155534456881714` | `kt04/entry/T` | `kt04/entry/T` |
| `kt05` | `0` | `0` | `20` | `6,926,697 / 12,893,760` | `0.417157541668007` | `0.082750909843826118` | `kt05/entry/T` | `kt05/entry/T` |
| `kt06` | `0` | `0` | `20` | `6,906,414 / 12,893,760` | `0.4177226476560002` | `0.078136519387538614` | `kt06/entry/T` | `kt06/entry/T` |
| `kt07` | `0` | `0` | `20` | `6,923,906 / 12,893,760` | `0.15460514354559504` | `0.107563584651355` | `kt07/entry/T` | `kt07/entry/T` |
| `kt08` | `0` | `0` | `20` | `6,905,100 / 12,893,760` | `0.16287418728619046` | `0.10226793729789506` | `kt08/entry/T` | `kt08/entry/T` |
| `kt09` | `0` | `0` | `20` | `6,929,976 / 12,893,760` | `0.16851252191320415` | `0.10226793729789506` | `kt09/entry/T` | `kt09/entry/T` |
| `kt10` | `0` | `0` | `20` | `6,906,096 / 12,893,760` | `0.169272389257619` | `0.11725721808307056` | `kt10/entry/T` | `kt10/entry/T` |
| **Total** | **`3`** | **`1`** | **`196`** | **`67,620,750 / 128,937,600`** | **`0.4177226476560002`** | **`0.11725721808307056`** | **`kt01/entry/T`** | **`kt01/entry/ssh`** |

### Root-cause attribution (`CONFIRMED` vs `PLAUSIBLE`)

At `kt = 1`, every active wet-cell surface flux and stress field (`utau`,
`vtau`, `utauU`, `vtauV`, `qsr`, `qns`, `emp`, `sfx`, `taum`, `wndm`, `fr_i`,
`rnf`, `rnf_tsc_T`, `rnf_tsc_S`, `erp`, `qrp`, `coefice`, and every field of
`kt01/S1_FLUX`) is bit-identical (`0 / 26,640` unequal). Every non-bit row at
`kt = 1` is **CONFIRMED**:

1. **CONFIRMED — SI3 (`nn_ice = 2`) levitating-ice `eta` offset in `card.ocean.recipe.initial_state`**:
   `build_nemo_orca2_k0_ocean_card` wraps `build_orca2_zps_card`, whose
   `initial_state` sets `eta = -0.093435058726849693 m` on all `16,433` wet
   cells (the SI3 `iceistate` sea-surface depression), whereas standalone K0
   (`nn_ice = 1`) does not execute `iceistate` and starts from `ssh = 0.0`
   everywhere. This directly causes `kt01/S0_EXIT/ssh_m` (`16,433 / 26,640`,
   `max |diff| = 0.093435058726849693`), `kt01/S0_EXIT/e3t_m` (`16,433 / 26,640`,
   `max |diff| = 0.031141729619529102`), and `kt01/entry/ssh` (`OVER_BAR`), and
   drives the subsequent barotropic/baroclinic state divergence across
   `kt01/stage1`..`kt10/stage3` (`196 / 200 OVER_BAR` rows) and the `kt >= 2`
   surface mean/flux divergence (`max |diff| = 0.18087092394585402` on `ssh_m` at
   `kt02` and `37.58738259116825` on `qns` at `kt09`).
2. **CONFIRMED — Dry-cell signed zero (`-0.0` vs `+0.0`) on initial `T`**:
   In NEMO `istate`, dry cells of `ts(:,:,:,jp_tem,:)` retain `-0.0` (`151,917 /
   799,200` 3-D words and `4,668 / 26,640` surface words where raw NetCDF
   temperature is negative before `tmask` multiplication), whereas
   `build_orca2_zps_card` sanitizes dry `T` cells to `+0.0`. This produces the
   first non-bit row on both surface (`kt01/S0_EXIT/sst_m`: `4,668 / 26,640`,
   `max |diff| = 0.0`) and rung-10 (`kt01/entry/T`: `151,917 / 799,200`,
   `max |diff| = 0.0`, `AT_BAR_NOT_EXACT`), and propagates `-0.0` vs `+0.0`
   onto dry cells of `ts_kbb`/`ts_kmm` in `S3_COVER_IIF`/`S3_FREEZE_IIF` and
   `qns` in `S2_SSR`..`S2_EXIT` (`4,668 / 26,640`, `max |diff| = 0.0` at `kt=1`).
3. **CONFIRMED — Dry-cell masking of `S1_TURB` diagnostics in `_nemo_ncar_turbulent`**:
   NEMO's `blk_oce_1` computes `theta_air_zt`, `ssq`, `Cd_atm`, `Ch_atm`,
   `Ce_atm`, and `rhoa` unmasked over `A2D(0)` (`10,207` dry cells; `9,544`
   nonzero dry cells for `Cd`/`Ch`/`Ce`), whereas `_nemo_ncar_turbulent` masks
   those intermediate diagnostics by `wet` (`mask`) and yields `+0.0` vs `-0.0`
   on dry `evap`, `latent` (`4,738` cells) and `sensible` (`4,142` cells). On
   all `16,433` wet cells at `kt=1`, `S1_TURB` is bit-identical (`0` unequal).
4. **CONFIRMED — `S0_F` `tradmp` (`damping_call`) `fdta`/`nrec`/`nbb`/`naa` slot order**:
   Across all `99` `S0_F` frames, every interpolated `fnow` array is
   bit-identical (`0` unequal). On the `10` per-step `tradmp` calls
   (`votemper`, `vosaline`), NEMO's `sf_tsd` descriptor keeps its unswapped
   year-seam slots (`nrec = [12, -1339200, 1, 1339200]`, `nbb = 1`, `naa = 2`,
   `fdta[..., 0] = Dec`, `fdta[..., 1] = Jan`), whereas
   `build_real_composed_inputs` populated `_s0f_actual` on `damping_call` with
   `istate`'s swapped slot order (`nrec = [1, 1339200, 12, -1339200]`,
   `nbb = 2`, `naa = 1`, `fdta = stack((after, before), axis=-1)`), accounting
   for all `32,673,840` unequal `S0_F` words (`3,267,384` per step).
5. **PLAUSIBLE — Secondary `kt >= 2` ocean/surface coupling differences**:
   Any additional `kt >= 2` state or surface differences beyond the downstream
   propagation of the `kt = 1` initial `eta = -0.093435058726849693 m` offset
   are masked until `card.ocean.recipe.initial_state` uses the standalone K0
   `eta = 0.0` initial condition.

## Resume 7 — four confirmed S4 root-cause fixes and ten-step re-score

Resume 7 resolves all four **CONFIRMED** `kt = 1` surface and entry root causes
at their literal source locations and re-scores the 10-step standalone K0
composed card against admitted arm A (with the arm-B reproducibility control
repeated):

1. **K0 initial state (`ssh = 0.0`, no `ice_istate` sea-surface depression)**:
   In standalone K0 (`nn_ice = 1`), `sbc_init` only invokes `ice_init` (and its
   `ice_istate` levitating-ice sea-surface depression) under `IF( nn_ice == 2 )`
   at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:376-384`,
   so `ssh` remains at its `0._wp` cold-start allocation from
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:398,405,408`.
   `NemoStandaloneK0Config` now requires the explicit no-default field
   `initial_state="nemo_k0_istate"` (`validate_nemo_standalone_k0` refuses any
   other value, and `_validate_k0_ocean_card` refuses non-zero or `-0.0` initial
   `eta`), and `build_nemo_orca2_k0_ocean_card` passes
   `initial_state=k0.initial_state` to `build_orca2_zps_card` via
   `build_orca2_istate_for_mode(..., initial_state="nemo_k0_istate")`. The SI3
   card (`build_orca2_zps_card(deck_root)` with `initial_state="si3_iceistate"`)
   is 100% bit-identical (`ORCA2-zps` fingerprint unchanged at
   `d699476d15b0cf24`).
2. **Dry-cell signed zero (`-0.0` vs `+0.0`) on initial `T` and `tra_dmp` target**:
   In NEMO `istate_init`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90:118`) and
   `dta_tsd`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dtatsd.f90:308-309`),
   initial and damping tracers are masked by `REAL(wp)` multiplication
   `ptsd(:,:,:,jp_tem) = sf_tsd(jp_tem)%fnow(:,:,:) * tmask(:,:,:)` and
   `ptsd(:,:,:,jp_sal) = sf_tsd(jp_sal)%fnow(:,:,:) * tmask(:,:,:)`, which
   preserves IEEE-754 `-0.0` on the `151,917 / 799,200` 3-D dry cells (`4,668 /
   26,640` surface dry cells) where the raw NetCDF temperature is negative.
   `build_orca2_ts_at_step`, `build_orca2_initial_ts`, and
   `build_orca2_damping_for_step` now support
   `mask_evaluation="nemo_tmask_multiply"`, selected by
   `initial_state="nemo_k0_istate"` and the K0 composed-card damping builder,
   while `mask_evaluation="where_zero"` keeps the SI3 card and all other cards
   bit-identical.
3. **Unmasked `S1_TURB` intermediate diagnostics over `A2D(0)` in `_nemo_ncar_turbulent`**:
   In `blk_oce_1`, NEMO initializes `tsk` on `A2D(0)` as
   `tsk(ji,jj) = ( sst_m(ji,jj) + rt0 ) * tmask(ji,jj,1)` at
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:656-657`,
   executes `turb_ncar` and the intermediate turbulent diagnostics (`theta_air`,
   `ssq`, `Cd`, `Ch`, `Ce`, `rhoa`, `zsen`, `zlat`, `evap`) unmasked over
   `A2D(0)` at
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:740-876`, and
   masks only the exit fluxes (`sen`, `lat`, `evap`, `utau`, `vtau`, `taum`,
   `wndm`) by `tmask(ji,jj,1)` (with `tsk` converted back to Celsius via
   `( tsk(ji,jj) - rt0 ) * tmask(ji,jj,1)` at
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:925`).
   Removing the non-NEMO `jnp.where(active, ...)` dry-cell operand substitution
   from `_nemo_ncar_turbulent` makes all `19` fields of `kt01/S1_TURB`
   bit-identical (`0 / 506,160` unequal words across both wet and dry cells),
   while leaving all non-NEMO bulk callers (`air_sea_fluxes`,
   `ncar_transfer_coefficients`, `blk_oce_2(nemo_source_exact=False)`)
   bit-identical.
4. **`S0_F` `tradmp` (`damping_call`) `sf_tsd` year-seam slot order**:
   Because `istate_init`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90:118`) already
   called `dta_tsd( nit000, ts(:,:,:,:,Kbb) )` at `kt = 1`, swapping `sf_tsd` to
   `(nbb = 2, naa = 1)`, the subsequent `tra_dmp` call `dta_tsd( kt, zts_dta )`
   at `kt = 1`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/tradmp.f90:181`) enters
   `fld_read` with `ll_firstcall = ( kt == nit000 )` true a second time
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/fldread.f90:192,209-215`)
   and runs `fld_init` + `fld_update` again on `sf_tsd`, swapping its year-seam
   slots back to `nbb = 1, naa = 2, nrec = (12, -1339200, 1, 1339200)`, and
   `fdta = (before, after)` for `kt = 1..10`. Updating `damping_call` in
   `build_real_composed_inputs` to emit `sf_tsd`'s unswapped slots makes all
   `99` `S0_F` frames across `kt = 1..10` 100% bit-exact (`0 / 57,890,304`
   unequal words, `max |diff| = 0.0`).

### M6 verification: non-NEMO callers and SI3 card bit-identical

- `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_card_fingerprint.py`:
  `PASS` — `ORCA2-zps` (the SI3 card) fingerprint is unchanged at
  `d699476d15b0cf24`, and `GYRE-zco`, `LOCK_EXCHANGE-zco`, `OVERFLOW-zps`,
  `VORTEX-zco`, and `VORTEX_VEC-zco` are all unchanged (`0` leaf changes).
- Non-NEMO bulk callers (`air_sea_fluxes(algo="ncar")`,
  `air_sea_fluxes(algo="ly09_2coeff")`, `ncar_transfer_coefficients`, and
  `blk_oce_2(nemo_source_exact=False)`): `0` unequal words across mixed wet/dry
  inputs (`test_non_nemo_callers_bit_identical_for_item3`).

### Repeated reproducibility control (admitted arm B vs admitted arm A)

Re-running `score_reproducibility_control` in
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_score_composed_card.py`:

- Status: `PASS_REPRODUCIBILITY_EXACT`
- Surface boundaries (`259` time-qualified boundaries, `1,899` field rows):
  `0 / 123,824,304` unequal words (`max |diff| = 0.0`)
- Rung-10 state ladder (`40` checkpoints, `200` field rows):
  `0 / 128,937,600` unequal words (`200 / 200 BIT_EXACT`, `max |diff| = 0.0`)

### Re-scored real legoESM composed card (`kt = 1..10` vs arm A)

Running the production ORCA2 K0 composed card (`NemoStandaloneK0Config(nn_ice=1,
ln_drgice_imp=False, ln_mxl0=True, nn_mxlice=0, ln_icebergs=False,
initial_state="nemo_k0_istate")` explicit) on CPU (`JAX_PLATFORMS=cpu
JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`) for
`kt = 1..10` and scoring against arm A via `score_composed_card`:

- Overall status: `FAIL_S4_SURFACE` (`surface exact bar failed: 24496893 / 123824304`,
  reduced from `58,062,489 / 123,824,304`; `max |diff| = 37.58738257120578`,
  reduced from `2,678,400.0` / `127.37701245735659`)
- **`kt = 1` surface boundaries (`18` outer/S1/S3/S2 boundaries + `18` `S0_F` frames)**:
  **`0 / 18,035,664` unequal words (`max |diff| = 0.0`, 100% bit-exact across all
  wet and dry cells)**
- **All `99` `S0_F` boundaries across `kt = 1..10`**:
  **`0 / 57,890,304` unequal words (`max |diff| = 0.0`, 100% bit-exact)**
- **`kt = 1` `entry` state rows (`kt01/entry/T`, `kt01/entry/S`, `kt01/entry/u`, `kt01/entry/v`, `kt01/entry/ssh`)**:
  **`5 / 5 BIT_EXACT` (`0 / 3,223,440` unequal words, `max |diff| = 0.0`)**
- First non-bit surface row (and first non-zero-`max |diff|` surface row):
  `kt02/S0_EXIT/sst_m` (`16,433 / 26,640` unequal wet-cell words,
  `max |diff| = 0.0006906256512557718`)
- First non-bit rung-10 row (and first `OVER_BAR` rung-10 row):
  `kt01/stage1/T` (`587,799 / 799,200` unequal words,
  `max |diff| = 0.0004687813029160548`,
  `normalized_max_abs = 1.558028817559533e-05 > 1e-15`)

#### Re-scored surface boundaries by step (`kt = 1..10`)

| Step | Unequal / Total | Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- |
| `kt01` | `0 / 18,035,664` | `0.0` | `None` |
| `kt02` | `1,497,312 / 10,842,528` | `0.18087092394585402` | `kt02/S0_EXIT/sst_m` |
| `kt03` | `3,176,738 / 12,894,000` | `5.904199766100569` | `kt03/S0_EXIT/sst_m` |
| `kt04` | `2,572,755 / 10,842,528` | `4.500545930355912` | `kt04/S0_EXIT/sst_m` |
| `kt05` | `3,175,632 / 12,894,000` | `37.55767153423536` | `kt05/S0_EXIT/sst_m` |
| `kt06` | `2,571,923 / 10,842,528` | `37.55767153423536` | `kt06/S0_EXIT/sst_m` |
| `kt07` | `3,177,037 / 12,894,000` | `37.58738257120578` | `kt07/S0_EXIT/sst_m` |
| `kt08` | `2,572,966 / 10,842,528` | `37.58732115683465` | `kt08/S0_EXIT/sst_m` |
| `kt09` | `3,178,492 / 12,894,000` | `37.55243048136965` | `kt09/S0_EXIT/sst_m` |
| `kt10` | `2,574,038 / 10,842,528` | `37.552424399404195` | `kt10/S0_EXIT/sst_m` |
| **Total** | **`24,496,893 / 123,824,304`** | **`37.58738257120578`** | **`kt02/S0_EXIT/sst_m`** |

#### Re-scored surface boundaries by boundary family (`kt = 1..10` aggregate plus `kt01` and `kt02`)

| Boundary family | `kt01` unequal / total (`max \|diff\|`) | `kt02` unequal / total (`max \|diff\|`) | `kt = 1..10` unequal / total | `kt = 1..10` max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- | --- |
| `S0_ENTRY` | `0 / 0` (`0.0`) | `0 / 0` (`0.0`) | `0 / 0` | `0.0` | `None` |
| `S0_EXIT` | `0 / 186,480` (`0.0`) | `124,168 / 186,480` (`0.18087092394585402`) | `1,117,512 / 1,864,800` | `0.18087092394585402` | `kt02/S0_EXIT/sst_m` |
| `S0_F` | `0 / 10,656,384` (`0.0`) | `0 / 5,035,008` (`0.0`) | `0 / 57,890,304` | `0.0` | `None` |
| `S1_ENTRY` | `0 / 186,480` (`0.0`) | `124,168 / 186,480` (`0.18087092394585402`) | `1,117,512 / 1,864,800` | `0.18087092394585402` | `kt02/S1_ENTRY/sst_m` |
| `S1_TURB` | `0 / 506,160` (`0.0`) | absent on even `kt` | `910,171 / 2,530,800` | `8.992867801479244` | `kt03/S1_TURB/sst` |
| `S1_FLUX` | `0 / 159,840` (`0.0`) | absent on even `kt` | `197,196 / 799,200` | `8.749036513328946` | `kt03/S1_FLUX/qlwn` |
| `S1_EXIT` | `0 / 399,600` (`0.0`) | `124,168 / 399,600` (`0.18087092394585402`) | `1,769,042 / 3,996,000` | `37.58732115683465` | `kt02/S1_EXIT/sst_m` |
| `S3_ENTRY` | `0 / 399,600` (`0.0`) | `124,168 / 399,600` (`0.18087092394585402`) | `1,769,042 / 3,996,000` | `37.58732115683465` | `kt02/S3_ENTRY/sst_m` |
| `S3_COVER_IIF` | `0 / 426,240` (`0.0`) | `65,732 / 426,240` (`0.001455833838534204`) | `1,243,118 / 4,262,400` | `37.58732115683465` | `kt02/S3_COVER_IIF/sss_m` |
| `S3_FREEZE_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `654,425 / 2,264,400` | `8.749036513328946` | `kt03/S3_FREEZE_IIF/sss_m` |
| `S3_CLAMP_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `654,425 / 2,264,400` | `8.749036513328946` | `kt03/S3_CLAMP_IIF/sss_m` |
| `S3_EXIT_IIF` | `0 / 452,880` (`0.0`) | `65,732 / 452,880` (`0.001455833838534204`) | `1,243,136 / 4,528,800` | `37.58738257120578` | `kt02/S3_EXIT_IIF/sss_m` |
| `S3_EXIT` | `0 / 426,240` (`0.0`) | `124,168 / 426,240` (`0.18087092394585402`) | `1,769,060 / 4,262,400` | `37.58738257120578` | `kt02/S3_EXIT/sst_m` |
| `S2_ENTRY` | `0 / 426,240` (`0.0`) | `124,168 / 426,240` (`0.18087092394585402`) | `1,769,060 / 4,262,400` | `37.58738257120578` | `kt02/S2_ENTRY/sst_m` |
| `S2_RNF` | `0 / 506,160` (`0.0`) | `124,168 / 506,160` (`0.18087092394585402`) | `1,796,120 / 5,061,600` | `37.58738257120578` | `kt02/S2_RNF/sst_m` |
| `S2_SSR` | `0 / 586,080` (`0.0`) | `124,168 / 586,080` (`0.18087092394585402`) | `2,054,396 / 5,860,800` | `37.58732115683465` | `kt02/S2_SSR/sst_m` |
| `S2_FWB` | `0 / 586,080` (`0.0`) | `124,168 / 586,080` (`0.18087092394585402`) | `2,054,396 / 5,860,800` | `37.58732115683465` | `kt02/S2_FWB/sst_m` |
| `S2_LBC` | `0 / 586,080` (`0.0`) | `124,168 / 586,080` (`0.18087092394585402`) | `2,054,396 / 5,860,800` | `37.58732115683465` | `kt02/S2_LBC/sst_m` |
| `S2_EXIT` | `0 / 639,360` (`0.0`) | `124,168 / 639,360` (`0.18087092394585402`) | `2,323,886 / 6,393,600` | `37.58732115683465` | `kt02/S2_EXIT/sst_m` |

#### Re-scored rung-10 200-row state ladder by step (`kt = 1..10`)

| Step | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Unequal / Total | Max `\|diff\|` | Max normalized `\|diff\|` | First non-bit row | First `OVER_BAR` row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kt01` | `5` | `0` | `15` | `5,506,511 / 12,893,760` | `0.08960306993076475` | `0.08960306993076475` | `kt01/stage1/T` | `kt01/stage1/T` |
| `kt02` | `0` | `0` | `20` | `6,912,223 / 12,893,760` | `0.12147086390205597` | `0.0953385078356453` | `kt02/entry/T` | `kt02/entry/T` |
| `kt03` | `0` | `0` | `20` | `6,927,385 / 12,893,760` | `0.12437813393687264` | `0.08705966238910777` | `kt03/entry/T` | `kt03/entry/T` |
| `kt04` | `0` | `0` | `20` | `6,904,848 / 12,893,760` | `0.13166476376835412` | `0.0598007900115033` | `kt04/entry/T` | `kt04/entry/T` |
| `kt05` | `0` | `0` | `20` | `6,926,791 / 12,893,760` | `0.41715841853835123` | `0.05424006434964804` | `kt05/entry/T` | `kt05/entry/T` |
| `kt06` | `0` | `0` | `20` | `6,906,448 / 12,893,760` | `0.41772315226799306` | `0.06127472183902331` | `kt06/entry/T` | `kt06/entry/T` |
| `kt07` | `0` | `0` | `20` | `6,923,911 / 12,893,760` | `0.13581575230804432` | `0.06127472183902331` | `kt07/entry/T` | `kt07/entry/T` |
| `kt08` | `0` | `0` | `20` | `6,905,061 / 12,893,760` | `0.13648423773422635` | `0.10226624263910054` | `kt08/entry/T` | `kt08/entry/T` |
| `kt09` | `0` | `0` | `20` | `6,930,006 / 12,893,760` | `0.13176780483340167` | `0.10226624263910054` | `kt09/entry/T` | `kt09/entry/T` |
| `kt10` | `0` | `0` | `20` | `6,906,109 / 12,893,760` | `0.14902761351543603` | `0.11725492112954827` | `kt10/entry/T` | `kt10/entry/T` |
| **Total** | **`5`** | **`0`** | **`195`** | **`67,749,293 / 128,937,600`** | **`0.41772315226799306`** | **`0.11725492112954827`** | **`kt01/stage1/T`** | **`kt01/stage1/T`** |

- With all four `kt = 1` surface and entry root causes resolved, the first
  departure in the composed 10-step run occurs inside the `kt = 1` ocean step at
  `kt01/stage1` (`kt01/stage1/T`: `587,799 / 799,200` unequal words, `max |diff|
  = 0.0004687813029160548`; `kt01/stage1/S`: `430,552 / 799,200`, `max |diff| =
  0.00034781990987120537`; `kt01/stage1/u`: `438,720 / 799,200`, `max |diff| =
  0.02714041350475488`; `kt01/stage1/v`: `421,889 / 799,200`, `max |diff| =
  0.01631687050598546`; `kt01/stage1/ssh`: `16,433 / 26,640`, `max |diff| =
  0.009738004150344978`), which then feeds `kt01/stage3` and becomes the first
  non-bit surface row at `kt02/S0_EXIT/sst_m` (`16,433 / 26,640` unequal wet-cell
  words, `max |diff| = 0.0006906256512557718`).
- independent review: operator

## Resume 8 — offline stage-1 sub-step discrimination under K0

Offline diagnosis on the clean `34ecef485fcd` tree (no model code changed)
discriminating why `kt01/stage1` diverges when `kt01` surface boundaries (`0 /
18,035,664`) and `kt01/entry/*` (`5 / 5 BIT_EXACT`, `0 / 3,223,440`) are
bit-identical against the admitted arm-A record (`orca2_s4_exact_a_np2`).

### (1) Stage-1 surface and sea-ice inputs consumed by the ocean step at `kt = 1`

In NEMO `stp_RK3` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:128`),
after `CALL sbc( kstp, Nbb, Nbb )` completes, the `kt = 1` pre-stage-1 and
stage-1 ocean sequence runs `eos_rab`, `bn2`, and `zdf_phy`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:146-155`),
`CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:176`), and
`CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Naa, Nrhs )`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:185-197`).
Every input from the surface/ice side is inventoried and compared below against
admitted arm A (recorded) or reconstructed from NEMO's K0 code path:

| Input field / K0 branch | Provenance (recorded vs reconstructed) | Stage-1 ocean consumer (`NEMO` / `legoESM`) | Surface-card output vs NEMO (`unequal / total`, `max \|diff\|`) | Ocean Stage-1 operand consumed vs NEMO (`unequal / total`, `max \|diff\|`) |
| --- | --- | --- | --- | --- |
| Clamped surface `T` (`ts(:,:,1,jp_tem,Nbb)`) | **Recorded** in `kt01/S3_EXIT_IIF/ts_kbb` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:166`) | `eos_rab`/`bn2`/`zdf_phy`, `stp_2D` `eos(ts,Kbb)`, `stp_RK3_stg(1)` `tra_adv`/`tra_sbc_RK3` (`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1044-1048`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| `emp` (`E - P` after `sbc_ice_if`, `sbc_ssr`, `sbc_fwb`) | **Recorded** in `kt01/S2_EXIT/emp` | `stp_2D` `sshe_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:242-267`) and Stage-1 `tra_sbc_RK3` dilution (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| `rnf`, `rnf_tsc_T`, `rnf_tsc_S` (`sbc_rnf`) | **Recorded** in `kt01/S2_EXIT/{rnf,rnf_tsc_T,rnf_tsc_S}` | `rnf` in `stp_2D` `sshe_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:242-267`) and Stage-1 `div_hor` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:270-313`); `rnf_tsc` in Stage-1 `tra_sbc_RK3` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| `qns` (non-solar heat flux) | **Recorded** in `kt01/S2_EXIT/qns` | Stage 3 only (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328`); not consumed in Stage 1 (`Kmm == Kbb`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) (Stage 3 only) |
| `qsr` (penetrative solar heat flux) | **Recorded** in `kt01/S2_EXIT/qsr` | Stage 3 only (`tra_qsr` at `kstg == 3`); not consumed in Stage 1 | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) (Stage 3 only) |
| `sfx` (salt flux from ice) | **Recorded** in `kt01/S2_EXIT/sfx` | Identically zero under K0 (`nn_ice = 1`) and consumed at Stage 3 only (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) (Stage 3 only) |
| `erp`, `qrp`, `coefice` (`sbc_ssr`) | **Recorded** in `kt01/S2_EXIT/{erp,qrp,coefice}` | Already folded into `emp` and `qns` during `S2_SSR`; not read separately by ocean | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| `taum`, `wndm` | **Recorded** in `kt01/S2_EXIT/{taum,wndm}` | `taum` consumed in `zdf_tke` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:260-275` and `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:608-638`), whose outputs `avt, avs, avm` are consumed at Stage 3 only (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:405,538`) | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| `fr_i` (binary `{0, 1}` ice cover from `sbc_ice_if`) | **Recorded** in `kt01/S2_EXIT/fr_i` | Consumed in `zdf_tke` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:260-275` and `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:365,501`); `zdf_tke` outputs are consumed at Stage 3 only | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) |
| **Wind stress `utau, vtau` (T-points) and `utauU, vtauV` (U/V-points)** | **Recorded** in `kt01/S2_EXIT/{utau,vtau,utauU,vtauV}` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570`) | `stp_2D` adds `utauU, vtauV` to `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199`); in legoESM, `nemo_s2_to_ocean_interfaces` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-412`) passes `result.utau_u, result.vtau_v` into `tau_i_native, tau_j_native`, which `surface_stress_faces` (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4631-4663`, called at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149`) interpolates T->U/V and coastal-unmasks a **second time** | `utau`: `0 / 26,640` (`0.0`)<br>`vtau`: `0 / 26,640` (`0.0`)<br>`utauU`: `0 / 26,640` (`0.0`)<br>`vtauV`: `0 / 26,640` (`0.0`) | **MISMATCH**:<br>`_wind_tau_i_u`: `16,977 / 26,640` (`16,522` non-zero, `455` signed-zero, `max \|diff\| = 0.89328994835474895 N/m^2`)<br>`_wind_tau_j_v`: `16,865 / 26,640` (`16,505` non-zero, `360` signed-zero, `max \|diff\| = 0.70509415925258379 N/m^2`) |
| Top drag / `rCdU_top` (`ln_drgice_imp = F`, `ln_isfcav = F`) | **Reconstructed** from NEMO K0 code path (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfdrg.f90:414-416`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfphy.f90:327-337`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90:1246-1247,1277-1300`) | Under `ln_isfcav = F` and `ln_drgice_imp = F`, NEMO does not allocate `rCdU_top`, skips `zdf_drg('TOP')`, sets `pCdU_u/v` from `rCdU_bot` only, and skips the top-drag residual in `dyn_drg_init`; legoESM `implicit_ice_ocean_top_drag = False` skips all top-drag terms | `0` (not allocated / off) | `0 / 26,640` (`0.0`) |
| TKE ice-mixing branch (`nn_ice = 1`, `ln_mxl0 = T`, `nn_mxlice = 0`) | **Reconstructed** from NEMO K0 code path (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:260-275`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:365,501`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:608-638`) | `nn_eice = 2` sets `zice_fra = fr_i`, `en(1) = MAX(rn_emin0, zbbrau * taum)`, attenuates Langmuir and `etau` by `1 - zice_fra`, and `CASE(0)` (`nn_mxlice = 0`) sets `zmxlm(1) = MAX(rn_mxl0, zraug * taum * tmask)` with no sea-ice mixing-length override; `avt/avs/avm` are consumed at Stage 3 only (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:405,538`) | `0 / 26,640` (`0.0`) | `0 / 799,200` (`0.0`) (and Stage 3 only) |
| Icebergs (`ln_icebergs = F`) | **Reconstructed** from NEMO K0 code path (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:482-495` and `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:507`) | Skips `utau_icb/vtau_icb` and `CALL icb_stp( kt, Kmm )` in both NEMO and legoESM | `0` (off) | `0` (off) |

### (2) Stage-1 sub-steps in NEMO order at `kt = 1`: first differing sub-step, counts, and maxima

Running the `kt = 1` legoESM stage-1 sub-steps in NEMO execution order on
production (`tau_i_native = result.utau_u, tau_j_native = result.vtau_v`) vs
passing T-point stresses (`tau_i_native = result.utau, tau_j_native =
result.vtau`) and comparing against the admitted arm-A checkpoints:

| Order | Stage-1 sub-step (`NEMO` source -> `legoESM` function) | Comparison target | Production (`tau_i_native = utau_u`) `unequal / total` (`max \|diff\|`) | T-point stress (`tau_i_native = utau`) `unequal / total` (`max \|diff\|`) | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `sbc_ice_if` surface `T` clamp on `ts(:,:,1,jp_tem,Nbb)` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:166` -> `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1044-1048`) | Recorded `kt01/S3_EXIT_IIF/ts_kbb` | `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) | `BIT_EXACT` |
| 2 | `eos_rab`, `bn2`, `zdf_phy`: `zdf_drg('BOTTOM')` and `zdf_tke` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:146-155`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfphy.f90:327-337`) | Reconstructed K0 path (`16` `tke_statement_trace` leaves; consumed at Stage 3 only) | `0 / 799,200` (`0.0`) | `0 / 799,200` (`0.0`) | `BIT_EXACT` |
| 3 | `stp_2D` 3-D momentum RHS `uu(Krhs), vv(Krhs)` on `Nbb` and pre-wind depth-average `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:137-200`) | Reconstructed K0 path (`du_dt(Nbb)`, `dv_dt(Nbb)`, `depth_avg_u/v`) | `du_dt`: `0 / 799,200` (`0.0`)<br>`dv_dt`: `0 / 799,200` (`0.0`) | `0 / 799,200` (`0.0`) | `BIT_EXACT` |
| 4 | `stp_2D` `dyn_drg_init` baroclinic drag residual on `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90:1246-1247,1277-1300`) | Reconstructed K0 path (`post_drag_u/v`; `uu(Nbb) = 0` at `kt = 1`, top drag off) | `post_drag_u`: `0 / 26,640` (`0.0`)<br>`post_drag_v`: `0 / 26,640` (`0.0`) | `0 / 26,640` (`0.0`) | `BIT_EXACT` |
| **5** | **FIRST DIFFERING SUB-STEP**: `stp_2D` surface wind stress addition to `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199` -> `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4631-4663`, `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149`, `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7208-7220`) | Recorded `kt01/S2_EXIT/{utauU,vtauV}` and resulting `Ue_rhs, Ve_rhs` (`post_wind_u/v`) | `utauU`: **`16,977 / 26,640`** (`0.89328994835474895 N/m^2`)<br>`vtauV`: **`16,865 / 26,640`** (`0.70509415925258379 N/m^2`)<br>`Ue_rhs`: **`15,853 / 26,640`** (`2.3875891177624881e-6 m/s^2`)<br>`Ve_rhs`: **`15,875 / 26,640`** (`2.5992867799449502e-6 m/s^2`) | `utauU`: **`0 / 26,640` (`0.0`)**<br>`vtauV`: **`0 / 26,640` (`0.0`)**<br>`Ue_rhs`: **`0 / 26,640` (`0.0`)**<br>`Ve_rhs`: **`0 / 26,640` (`0.0`)** | **FIRST DIFF** |
| 6 | `stp_2D` `sshe_rhs` (`emp`, `rnf`) and barotropic subcycling `dyn_spg_ts` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:242-267`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90:650-652`) | Reconstructed `sshe_rhs`, `ssha`, `uu_b(Kaa)`, `vv_b(Kaa)` | `sshe_rhs`: `0 / 26,640` (`0.0`)<br>`ssha`: `16,433 / 26,640` (`0.029214012451034943 m`)<br>`uu_b(Kaa)`: `15,853 / 26,640` (`0.027140413504754879 m/s`)<br>`vv_b(Kaa)`: `15,943 / 26,640` (`0.016316870505985459 m/s`) | `0 / 26,640` (`0.0`) on all | Downstream of Sub-step 5 |
| 7 | `stp_RK3_stg(1)` stage-1 `ssh(Kaa)`, `r3*(Kaa)`, horizontal transports `zFu, zFv`, `div_hor`, and `wzv` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:131-175`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:270-313`) | Recorded `kt01/stage1/ssh` + reconstructed `zFu, zFv, wzv` | `kt01/stage1/ssh`: **`16,433 / 26,640`** (`0.0097380041503449775 m`, `OVER_BAR`)<br>`zFu`: `414,784 / 803,640` (`3.3311550054648425e4 m^3/s`)<br>`zFv`: `415,175 / 804,600` (`2.6342172065184452e4 m^3/s`)<br>`wzv`: `430,314 / 825,840` (`4.7567815064641905e-6 m/s`) | `kt01/stage1/ssh`: **`0 / 26,640` (`0.0`, `BIT_EXACT`)**<br>`zFu, zFv, wzv`: `0` (`0.0`) | Downstream of Sub-step 5 |
| 8 | `stp_RK3_stg(1)` stage-1 3-D velocity update `u1_raw, v1_raw` and barotropic replacement `uu(Kaa), vv(Kaa)` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:366-421`) | Recorded `kt01/stage1/u` and `kt01/stage1/v` | `u1_raw, v1_raw`: `0 / 799,200` (`0.0`)<br>`kt01/stage1/u`: **`438,720 / 799,200`** (`413,030` non-zero, `25,690` dry `-0.0`, `max \|diff\| = 0.027140413504754879 m/s`, `OVER_BAR`)<br>`kt01/stage1/v`: **`421,889 / 799,200`** (`415,175` non-zero, `6,714` dry `-0.0`, `max \|diff\| = 0.016316870505985459 m/s`, `OVER_BAR`) | `kt01/stage1/u`: **`25,690 / 799,200` (`0` non-zero, `max \|diff\| = 0.0`, `AT_BAR_NOT_EXACT`)**<br>`kt01/stage1/v`: **`6,529 / 799,200` (`0` non-zero, `max \|diff\| = 0.0`, `AT_BAR_NOT_EXACT`)** | Non-zero diffs downstream of Sub-step 5 |
| 9 | `stp_RK3_stg(1)` stage-1 `tra_adv_cen`, `tra_sbc_RK3`, and `ts(Kaa)` update (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:431,474-507,572-573`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:503-505`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328` -> `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2881-2885`) | Recorded `kt01/stage1/T` and `kt01/stage1/S` | `tra_sbc_RK3` RHS: `0 / 799,200` (`0.0`)<br>`kt01/stage1/T`: **`587,799 / 799,200`** (`430,343` non-zero wet-cell diffs, `157,456` dry-cell `-0.0` vs `+0.0`, `max \|diff\| = 4.6878130291605480e-4 degC`, `OVER_BAR`)<br>`kt01/stage1/S`: **`430,552 / 799,200`** (`430,552` non-zero wet-cell diffs, `max \|diff\| = 3.4781990987120537e-4 PSU`, `OVER_BAR`) | `kt01/stage1/T`: **`157,456 / 799,200` (`0` non-zero wet-cell diffs, `157,456` dry-cell `-0.0` vs `+0.0`, `max \|diff\| = 0.0`, `AT_BAR_NOT_EXACT`)**<br>`kt01/stage1/S`: **`0 / 799,200` (`max \|diff\| = 0.0`, `BIT_EXACT`)** | Wet diffs downstream of Sub-step 5; dry `-0.0` from `:2881-2885` |

#### Effect of the one-variable `tau_i_native = result.utau, tau_j_native = result.vtau` ablation across all `kt = 1` checkpoints (`kt01/entry`..`kt01/stage3`)

| Checkpoint / Row | Production `unequal / total` | Production `max \|diff\|` | Production status | With `tau_i/j_native = result.utau/vtau` `unequal / total` (non-zero) | With `tau_i/j_native = result.utau/vtau` `max \|diff\|` | Ablation status |
| --- | --- | --- | --- | --- | --- | --- |
| `kt01/entry/{T,S,u,v,ssh}` (`5` rows) | `0 / 3,223,440` | `0.0` | `5 BIT_EXACT` | `0 / 3,223,440` (`0`) | `0.0` | `5 BIT_EXACT` |
| `kt01/stage1/T` | `587,799 / 799,200` | `4.6878130291605480e-4` | `OVER_BAR` | `157,456 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage1/S` | `430,552 / 799,200` | `3.4781990987120537e-4` | `OVER_BAR` | `0 / 799,200` (`0`) | `0.0` | `BIT_EXACT` |
| `kt01/stage1/u` | `438,720 / 799,200` | `0.027140413504754879` | `OVER_BAR` | `25,690 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage1/v` | `421,889 / 799,200` | `0.016316870505985459` | `OVER_BAR` | `6,529 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage1/ssh` | `16,433 / 26,640` | `0.0097380041503449775` | `OVER_BAR` | `0 / 26,640` (`0`) | `0.0` | `BIT_EXACT` |
| `kt01/stage2/T` | `430,453 / 799,200` | `0.0010531521942188246` | `OVER_BAR` | `157,456 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage2/S` | `430,552 / 799,200` | `5.2131918432110443e-4` | `OVER_BAR` | `0 / 799,200` (`0`) | `0.0` | `BIT_EXACT` |
| `kt01/stage2/u` | `438,720 / 799,200` | `0.027141522911035032` | `OVER_BAR` | `25,690 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage2/v` | `421,889 / 799,200` | `0.016317726569221469` | `OVER_BAR` | `6,529 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage2/ssh` | `16,433 / 26,640` | `0.014607006225517469` | `OVER_BAR` | `0 / 26,640` (`0`) | `0.0` | `BIT_EXACT` |
| `kt01/stage3/T` | `587,899 / 799,200` | `0.0020118983392072909` | `OVER_BAR` | `157,456 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage3/S` | `430,552 / 799,200` | `0.0011648528991944914` | `OVER_BAR` | `0 / 799,200` (`0`) | `0.0` | `BIT_EXACT` |
| `kt01/stage3/u` | `426,563 / 799,200` | `0.089603069930764751` | `OVER_BAR` | `25,436 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage3/v` | `411,624 / 799,200` | `0.068089226629258877` | `OVER_BAR` | `6,349 / 799,200` (`0` non-zero; dry `-0.0` vs `+0.0`) | `0.0` | `AT_BAR_NOT_EXACT` |
| `kt01/stage3/ssh` | `16,433 / 26,640` | `0.029214012451034943` | `OVER_BAR` | `0 / 26,640` (`0`) | `0.0` | `BIT_EXACT` |
| **All `20` `kt01` rows** | **`5,506,511 / 12,893,760`** | **`0.089603069930764751`** | **`5 BIT_EXACT`, `15 OVER_BAR`** | **`568,591 / 12,893,760` (`0` non-zero)** | **`0.0`** | **`11 BIT_EXACT`, `9 AT_BAR_NOT_EXACT`, `0 OVER_BAR`** |

### (3) Classification (`CONFIRMED` vs `PLAUSIBLE`) and (4) Exact `file:line` quotes

1. **CONFIRMED — Double T->U/V interpolation and coastal unmasking of surface wind stress in `nemo_s2_to_ocean_interfaces` -> `surface_stress_faces` (100% of non-zero differences at `kt = 1` across `stage1`, `stage2`, and `stage3`)**:
   - In NEMO `sbc` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570`), T-point stresses `utau, vtau` are interpolated once onto U/V points with coastal unmasking:
     ```fortran
     DO jj = ntsj-( nn_hls-1), ntej+(  nn_hls-2 ) ; DO ji = ntsi-( nn_hls-1), ntei+(  nn_hls-2)
        utauU(ji,jj) = 0.5_wp * ( utau(ji,jj) + utau(ji+1,jj) ) * MAX(tmask(ji,jj,1), tmask(ji+1,jj,1)) * ( 2. - umask(ji,jj,1) )
        vtauV(ji,jj) = 0.5_wp * ( vtau(ji,jj) + vtau(ji,jj+1) ) * MAX(tmask(ji,jj,1), tmask(ji,jj+1,1)) * ( 2. - vmask(ji,jj,1) )
     END DO   ;   END DO
     ```
     and `stp_2D` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199`) consumes `utauU, vtauV` directly:
     ```fortran
     Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj) * (r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kbb)))
     Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj) * (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kbb)))
     ```
   - In legoESM, `nemo_s2_step` already computes `utau_u, vtau_v = nemo_t_stress_to_uv(...)` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:348-358`, matching `kt01/S2_EXIT/{utauU,vtauV}` with `0 / 26,640` unequal words), and `nemo_s2_to_ocean_interfaces` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-412`) then binds those U/V-point arrays into `OceanSurfaceForcing`:
     ```python
     tau_i_native=result.utau_u,
     tau_j_native=result.vtau_v,
     ```
     However, `surface_stress_faces` (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4631-4663`, called at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149` and consumed at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7208-7220`) treats `tau_i_native, tau_j_native` as **T-point** stresses `tau_i_T, tau_j_T` and applies T->U/V averaging, `(2 - u_wet) * jnp.maximum(t_w, t_e)` coastal unmasking, and north-fold `top` overwrite a **second time**:
     ```python
     tau_i_T = jnp.asarray(_native_i, dtype=u_dtype)
     tau_j_T = jnp.asarray(_native_j, dtype=u_dtype)
     ...
     u_avg = jnp.float64(0.5) * (tau_i_T + tau_i_E)
     u_coast = jnp.maximum(t_w, t_e) * (jnp.float64(2.0) - u_wet)
     tau_i_int = u_avg * u_coast
     ```
   - Replacing `tau_i_native=result.utau_u, tau_j_native=result.vtau_v` with `tau_i_native=result.utau, tau_j_native=result.vtau` eliminates `100%` of the non-zero differences across all `20` rung-10 state rows at `kt = 1` (`0 / 20 OVER_BAR`, `max |diff| = 0.0` on every field of `kt01/entry`, `kt01/stage1`, `kt01/stage2`, and `kt01/stage3`), and also eliminates `100%` of the non-zero differences on `kt02/S0_EXIT/{sst_m,sss_m,ssu_m,ssv_m,ssh_m,e3t_m}` and `kt02/{entry,stage1,stage2}/*`.
2. **CONFIRMED — Dry-cell signed-zero (`-0.0` vs `+0.0`) preservation on `kt01/stage1/T` (`157,456 / 799,200` dry cells, `max |diff| = 0.0`, `AT_BAR_NOT_EXACT`)**:
   - In NEMO `stp_RK3_stg` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:503-505`), the stage tracer update runs unmasked over all cells:
     ```fortran
     ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )                      &
        &                   + rDt * ( 1._wp + r3t(ji,jj,Kmm) )*ts(ji,jj,jk,jn,Krhs) * tmask(ji,jj,jk) )  &
        &                /          ( 1._wp + r3t(ji,jj,Kaa) )
     ```
     On the `151,917` interior dry cells where `istate_init` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90:118`) left `ts(Kbb) = -0.0`, IEEE-754 addition `(-0.0) + (+0.0) = +0.0` clears `-0.0` to `+0.0` (and `lbc_lnk` at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:431,474-507,572-573` writes `+0.0` on the `5,539` border dry cells).
   - In legoESM `_nemo_ws_rk3_tracer_pair_step` (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2881-2885`), dry cells bypass the arithmetic and copy `base` (`ts(Kbb)`):
     ```python
     out = (
         ((one + r3t_kbb_b) * base + dt_rk * (one + r3t_kmm_b) * rhs * active_3d)
         / denom
     )
     return jnp.where(active_3d > 0.5, out, base)
     ```
     preserving `-0.0` on `151,917` interior + `5,539` folded border = `157,456` dry cells (`max |diff| = 0.0`, `AT_BAR_NOT_EXACT` under the rung-10 bar).

### Missing-boundary check

**No additional NEMO record is needed (`ACQUISITION_NEEDED: no`)**: the admitted
arm-A record (`orca2_s4_exact_a_np2`) already contains `kt01/S2_EXIT/{utau,vtau,utauU,vtauV,...}`,
`kt01/S3_EXIT_IIF/ts_kbb`, and `kt01/{entry,stage1,stage2,stage3}/{T,S,u,v,ssh}`,
which completely localize 100% of the `kt = 1` differences.

- independent review: operator

## Verification

- Round-340 focused fast tests (`test_orca2_round340_standalone.py`,
  `test_nemo_card_opt_in_defaults.py`, `test_ncar_bulk.py`, and
  `test_dispatch_hardening.py`): all pass (`37 passed, 2 skipped` in
  `test_orca2_round340_standalone.py` + `test_nemo_card_opt_in_defaults.py`;
  `72 passed` in `test_ncar_bulk.py`; `24 passed` in
  `test_dispatch_hardening.py`).
- M6 card-fingerprint gate (`nemo_testcase_l4_orca2_card_fingerprint.py`):
  `PASS` (`ORCA2-zps` SI3 card `d699476d15b0cf24` and all five phase-2 cards
  100% bit-identical).
- Admitted arm-B vs arm-A reproducibility control: `PASS_REPRODUCIBILITY_EXACT`
  (`0 / 123,824,304` surface words and `0 / 128,937,600` rung-10 words unequal).
- Real 10-step CPU composed-card re-score (`nemo_testcase_l4_orca2_round340_score_composed_card.py`):
  executed on CPU (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`); full re-scored tables
  committed above (`0 / 18,035,664` unequal surface words at `kt = 1`,
  `0 / 57,890,304` unequal `S0_F` words across `kt = 1..10`, `5 / 5 BIT_EXACT`
  on `kt01/entry/*`).
- Resume-8 offline stage-1 sub-step discrimination: executed on CPU
  (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`)
  against admitted arm A (`orca2_s4_exact_a_np2`), confirming `0` model-code
  edits and localizing `100%` of `kt = 1` non-zero errors to `tau_i_native =
  result.utau_u, tau_j_native = result.vtau_v` in `nemo_s2_to_ocean_interfaces`.
- Per-receipt citation gate: PASS, 55 citations, zero failures, zero unmapped
  citations, zero map-audit failures, and all nine self-tests fired.
- Required receipt-citation pytest (`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`
  with `JAX_ENABLE_X64=1`): 17/17 PASS on the clean committed tree.
