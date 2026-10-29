# ORCA2 round 340 — exact standalone S4 acquisition tool and composed card

Date: 2026-10-27. Base `a91d673b6fcf` (rebased from `d5d97cfc3121`). Disposition:
**TWINS_ADMITTED (`ORCA2_ROUND340_S4_TWINS_ADMITTED`) / COMPOSED_CARD_SCORED (`PASS_S4_EXACT`)**.
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
| **Wind stress `utau, vtau` (T-points) and `utauU, vtauV` (U/V-points)** | **Recorded** in `kt01/S2_EXIT/{utau,vtau,utauU,vtauV}` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570`) | `stp_2D` adds `utauU, vtauV` to `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199`); in legoESM, `nemo_s2_to_ocean_interfaces` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-414`) passes `result.utau_u, result.vtau_v` into `tau_i_native, tau_j_native`, which `surface_stress_faces` (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4653-4685`, called at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149`) interpolates T->U/V and coastal-unmasks a **second time** | `utau`: `0 / 26,640` (`0.0`)<br>`vtau`: `0 / 26,640` (`0.0`)<br>`utauU`: `0 / 26,640` (`0.0`)<br>`vtauV`: `0 / 26,640` (`0.0`) | **MISMATCH**:<br>`_wind_tau_i_u`: `16,977 / 26,640` (`16,522` non-zero, `455` signed-zero, `max \|diff\| = 0.89328994835474895 N/m^2`)<br>`_wind_tau_j_v`: `16,865 / 26,640` (`16,505` non-zero, `360` signed-zero, `max \|diff\| = 0.70509415925258379 N/m^2`) |
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
| **5** | **FIRST DIFFERING SUB-STEP**: `stp_2D` surface wind stress addition to `Ue_rhs, Ve_rhs` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199` -> `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4653-4685`, `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149`, `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7208-7220`) | Recorded `kt01/S2_EXIT/{utauU,vtauV}` and resulting `Ue_rhs, Ve_rhs` (`post_wind_u/v`) | `utauU`: **`16,977 / 26,640`** (`0.89328994835474895 N/m^2`)<br>`vtauV`: **`16,865 / 26,640`** (`0.70509415925258379 N/m^2`)<br>`Ue_rhs`: **`15,853 / 26,640`** (`2.3875891177624881e-6 m/s^2`)<br>`Ve_rhs`: **`15,875 / 26,640`** (`2.5992867799449502e-6 m/s^2`) | `utauU`: **`0 / 26,640` (`0.0`)**<br>`vtauV`: **`0 / 26,640` (`0.0`)**<br>`Ue_rhs`: **`0 / 26,640` (`0.0`)**<br>`Ve_rhs`: **`0 / 26,640` (`0.0`)** | **FIRST DIFF** |
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
   - In legoESM, `nemo_s2_step` already computes `utau_u, vtau_v = nemo_t_stress_to_uv(...)` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:348-358`, matching `kt01/S2_EXIT/{utauU,vtauV}` with `0 / 26,640` unequal words), and on parent commit `1c8bdb666599` `nemo_s2_to_ocean_interfaces` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-414`) then bound those U/V-point arrays into `OceanSurfaceForcing`:
     ```python
     tau_i_native=result.utau_u,
     tau_j_native=result.vtau_v,
     ```
     However, `surface_stress_faces` (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4653-4685`, called at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149` and consumed at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7208-7220`) treats `tau_i_native, tau_j_native` as **T-point** stresses `tau_i_T, tau_j_T` and applies T->U/V averaging, `(2 - u_wet) * jnp.maximum(t_w, t_e)` coastal unmasking, and north-fold `top` overwrite a **second time**:
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

## Resume 9 — literal single-interpolation face wind-stress handoff and 10-step re-score

### (1) Literal fix and M6 bit-equality / caller-identity proofs

1. **Explicit face-stress handoff (`utauU/vtauV`, interpolated once)**:
   - In NEMO `sbc` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570`), `utauU` and `vtauV` are computed once from `utau` and `vtau` with coastal unmasking (`MAX(tmask) * (2. - umask)` / `MAX(tmask) * (2. - vmask)`), and `stp_2D` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199`) consumes `utauU, vtauV` directly with no second interpolation.
   - `OceanSurfaceForcing` (`packages/ocean/legoesm/ocean/state.py:378-379`) adds optional explicit face-stress fields `utau_u_native: object = None` and `vtau_v_native: object = None`.
   - `nemo_s2_to_ocean_interfaces` (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-414`) passes `result.utau_u` and `result.vtau_v` (already computed once by `nemo_t_stress_to_uv` at `packages/ocean/legoesm/ocean/forcing/nemo_surface.py:348-358`) through `utau_u_native=result.utau_u, vtau_v_native=result.vtau_v` with `tau_i_native=None, tau_j_native=None`.
   - `surface_stress_faces` (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4626-4651`, called at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7132-7149` and consumed at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7208-7220`) validates that `utau_u_native` and `vtau_v_native` are supplied as a pair, refuses simultaneous `utau_u_native/vtau_v_native` and `tau_i_native/tau_j_native`, checks that their shapes match `dz_0_T.shape`, and when present with `use_native=True` embeds `utau_u` (`tau_i_u = jnp.concatenate([utau_u[:, -1:], utau_u], axis=1)`) and `vtau_v` (`tau_j_v = jnp.concatenate([jnp.zeros_like(vtau_v[:1, :]), vtau_v], axis=0)`) into the C-grid face arrays with **no second T->U/V interpolation or coastal unmasking**.
2. **Bit-equality with NEMO's recorded `utauU/vtauV` and M6 parent-vs-current bit-identity counts**:
   - **Standalone S4 `utauU/vtauV` bit-equality vs admitted arm A (`orca2_s4_exact_a_np2`, `kt = 1..10`)**:
     - `tu[:, 1:]` vs recorded `kt01..kt10/S2_EXIT/utauU`: **`0 / 266,400` unequal words** (`max |diff| = 0.0`, down from `16,977 / 26,640` at `kt = 1` on parent `1c8bdb666599`).
     - `tv[1:, :]` vs recorded `kt01..kt10/S2_EXIT/vtauV`: **`0 / 266,400` unequal words** (`max |diff| = 0.0`, down from `16,865 / 26,640` at `kt = 1` on parent `1c8bdb666599`).
     - Combined across `kt = 1..10`: **`0 / 532,800` unequal words (`100%` bit-exact)**.
   - **Certified rung-10 SI3 card (`ORCA2-zps`, `nemo_testcase_l4_orca2_round1_ladder_gate.py:_surface_forcings`) parent (`1c8bdb666599`) vs current (`7b5451497`)**:
     - `_surface_forcings` supplies `tau_i_native=fields["utau"], tau_j_native=fields["vtau"]` and leaves `utau_u_native=None, vtau_v_native=None`, taking the unchanged `_native_i` branch (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4653-4685`).
     - `surface_stress_faces` outputs `(tau_i_u, tau_j_v, dz_0_u, dz_0_v)` across `kt = 1..10` on the admitted round-5 SI3 record: **`0 / 1,072,160` unequal words** (`0 / 267,880` `tau_i_u`, `0 / 268,200` `tau_j_v`, `0 / 267,880` `dz_0_u`, `0 / 268,200` `dz_0_v`; `max |diff| = 0.0`).
     - `_surface_forcings` fields across `kt = 1..10`: **`0 / 4,262,400` unequal words** (`max |diff| = 0.0`).
   - **Non-NEMO callers of `surface_stress_faces` (`tau_x, tau_y` geographic and `tau_i_native, tau_j_native` T-point callers) and all shipped cards (parent vs current)**:
     - Geographic caller `surface_stress_faces(OceanSurfaceForcing(tau_x=..., tau_y=...))`: **`0 / 107,216` unequal words** (`max |diff| = 0.0`).
     - Native T-point caller `surface_stress_faces(OceanSurfaceForcing(tau_i_native=..., tau_j_native=...))` across `kt = 1..10`: **`0 / 1,072,160` unequal words** (`max |diff| = 0.0`).
     - M6 card-fingerprint gate (`nemo_testcase_l4_orca2_card_fingerprint.py`): **`0` changes across all `6` cards** (`GYRE-zco: fcad441865c685c4`, `LOCK_EXCHANGE-zco: ce229cd64fd544d9`, `ORCA2-zps: d699476d15b0cf24`, `OVERFLOW-zps: a4f02a29ba0dee53`, `VORTEX-zco: 13d7c997a4d8e13f`, `VORTEX_VEC-zco: 515b238ef4bb290a`).

### (2) Production-call-site test, revert plant, `eager == jit`, and finite JVP == VJP

- `test_resume9_production_call_site_single_interpolation_and_revert_plant`
  (`tests/ocean/fidelity/test_orca2_round340_standalone.py`):
  - Calls the production chain `nemo_orca2_k0_surface_card -> nemo_s2_to_ocean_interfaces -> OceanSurfaceForcing -> surface_stress_faces` on a coastal wet/dry mask where a second interpolation visibly perturbs the face stresses.
  - Verifies true `eager == jit` (`with jax.disable_jit():` vs `jax.jit` with dynamic atmospheric arguments `u_w, v_w, t_a, q_a, slp, qsr_d, qlw_d, pr, sn`): `0` unequal `uint64` words across all `6` output leaves (`s2.utau_u, s2.vtau_v, tu, tv, dzu, dzv`), and `tu[:, 1:] == s2.utau_u`, `tv[1:, :] == s2.vtau_v` bit-for-bit.
  - Plants the parent (`1c8bdb666599`) binding (`tau_i_native=s2.utau_u, tau_j_native=s2.vtau_v, utau_u_native=None, vtau_v_native=None`) and proves the bit-equality assertion fails on both `tu[:, 1:]` and `tv[1:, :]` (and at `kt = 1` on the full ORCA2 grid in `test_resume9_orca2_recorded_utauu_vtauv_and_m6_caller_bit_identity`, where the parent revert produces `16,977 / 26,640` and `16,865 / 26,640` unequal words).
  - Verifies finite forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) derivatives through `nemo_orca2_k0_ocean_step -> surface_stress_faces` into the ocean velocity state (`rtol=2e-14, atol=2e-14`), and `test_real_composed_card_n1_reverse_equals_forward` (`ORCA2_R340_REAL_GRADIENT=1`) passes on the full 3-D ORCA2 composed card (`N = 1` step, `rtol=2e-10, atol=2e-12`).
- `test_native_face_stress_pair_direct_handoff_and_fail_closed`
  (`tests/ocean/unit/test_surface_forcing_sign_convention.py`): verifies half-pair refusal, mutual-exclusion refusal with `tau_i_native/tau_j_native`, shape-mismatch refusal, and bit-exact handoff.

### (3) Repeated arm-B control and 10-step re-score of the real composed card

Command executed on CPU (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`):

```bash
python3 scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_score_composed_card.py \
  --report-only \
  --output-json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r340_resume9/score_composed_card.json
```

#### (a) Admitted arm-B vs arm-A reproducibility control (repeated)

- Status: `PASS_REPRODUCIBILITY_EXACT`
- Surface boundaries (`171` outer/S1/S3/S2 boundaries + `99` `S0_F` boundaries = `270` boundaries): **`0 / 123,824,304` unequal float64 words (`max |diff| = 0.0`)**
- Rung-10 state ladder (`40` checkpoints x `5` fields = `200` rows): **`0 / 128,937,600` unequal float64 words (`max |diff| = 0.0`, `200 / 200 BIT_EXACT`)**

#### (b) Real 10-step composed-card re-score against admitted arm A (`kt = 1..10`)

- Overall status: `FAIL_S4_SURFACE` (`surface exact bar failed: 18843285 / 123824304`,
  reduced from `24,496,893 / 123,824,304` in Resume 7; `max |diff| = 0.18087092394585402`,
  reduced from `37.58738257120578` in Resume 7)
- **`kt = 1` surface boundaries (`18` outer/S1/S3/S2 boundaries + `18` `S0_F` frames)**:
  **`0 / 18,035,664` unequal words (`max |diff| = 0.0`, 100% bit-exact)**
- **All `99` `S0_F` boundaries across `kt = 1..10`**:
  **`0 / 57,890,304` unequal words (`max |diff| = 0.0`, 100% bit-exact)**
- **`kt = 1` rung-10 state rows (`20` rows across `entry, stage1, stage2, stage3` x `T, S, u, v, ssh`)**:
  **`0 / 20 OVER_BAR` (`11 BIT_EXACT`, `9 AT_BAR_NOT_EXACT` dry-cell signed-zero `-0.0` vs `+0.0` rows, `max |diff| = 0.0`, `max_normalized_max_abs = 0.0` across all `12,893,760` words)**
- **`kt = 2` surface boundaries**:
  - `kt02/S0_EXIT/{sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m}`: **`0 / 159,840` unequal words (`max |diff| = 0.0`, 100% bit-exact)**
  - `kt02/S3_COVER_IIF` (`0 / 426,240`) and `kt02/S3_EXIT_IIF` (`0 / 452,880`): **`0` unequal words (`max |diff| = 0.0`, 100% bit-exact)**
  - Only `frq_m` (`26,640 / 26,640` words per boundary across the `11` boundaries that carry `frq_m` = `293,040 / 10,842,528` words, `max |diff| = 0.18087092394585402`) differs at `kt = 2`.
- **`kt = 2` `entry`, `stage1`, and `stage2` rung-10 state rows (`15` rows)**:
  **`0 / 15 OVER_BAR` (`6 BIT_EXACT`, `9 AT_BAR_NOT_EXACT` dry-cell signed-zero rows, `max |diff| = 0.0` on every field)**, plus `kt02/stage3/ssh` is `BIT_EXACT` (`0 / 26,640` unequal words, `max |diff| = 0.0`).
- **New first non-bit surface row (and first non-zero-`max |diff|` surface row)**:
  `kt02/S0_EXIT/frq_m` (`26,640 / 26,640` unequal words, `max |diff| = 0.18087092394585402`, moved from `kt02/S0_EXIT/sst_m`)
- **First non-bit rung-10 row**:
  `kt01/stage1/T` (`157,456 / 799,200` dry-cell signed-zero `-0.0` vs `+0.0` words, `max |diff| = 0.0`, `normalized_max_abs = 0.0`, `AT_BAR_NOT_EXACT`)
- **New first `OVER_BAR` rung-10 row**:
  `kt02/stage3/T` (`26,525 / 799,200` unequal words, `max |diff| = 0.00033734852885558553`, `normalized_max_abs = 1.1208924249237574e-05 > 1e-15`, moved from `kt01/stage1/T`)

#### Re-scored surface boundaries by step (`kt = 1..10`)

| Step | Unequal / Total | Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- |
| `kt01` | `0 / 18,035,664` | `0.0` | `None` |
| `kt02` | `293,040 / 10,842,528` | `0.18087092394585402` | `kt02/S0_EXIT/frq_m` |
| `kt03` | `381,847 / 12,894,000` | `0.18087092394585402` | `kt03/S0_EXIT/sst_m` |
| `kt04` | `1,469,485 / 10,842,528` | `0.18087092394585402` | `kt04/S0_EXIT/sst_m` |
| `kt05` | `2,979,282 / 12,894,000` | `0.18087092394585402` | `kt05/S0_EXIT/sst_m` |
| `kt06` | `2,465,317 / 10,842,528` | `0.18087092394585402` | `kt06/S0_EXIT/sst_m` |
| `kt07` | `3,083,337 / 12,894,000` | `0.18087092394585402` | `kt07/S0_EXIT/sst_m` |
| `kt08` | `2,521,780 / 10,842,528` | `0.18087092394585402` | `kt08/S0_EXIT/sst_m` |
| `kt09` | `3,111,992 / 12,894,000` | `0.18087092394585402` | `kt09/S0_EXIT/sst_m` |
| `kt10` | `2,537,205 / 10,842,528` | `0.18087092394585402` | `kt10/S0_EXIT/sst_m` |
| **Total** | **`18,843,285 / 123,824,304`** | **`0.18087092394585402`** | **`kt02/S0_EXIT/frq_m`** |

#### Re-scored surface boundaries by boundary family (`kt = 1..10` aggregate plus `kt01` and `kt02`)

| Boundary family | `kt01` unequal / total (`max \|diff\|`) | `kt02` unequal / total (`max \|diff\|`) | `kt = 1..10` unequal / total | `kt = 1..10` max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- | --- |
| `S0_ENTRY` | `0 / 0` (`0.0`) | `0 / 0` (`0.0`) | `0 / 0` | `0.0` | `None` |
| `S0_EXIT` | `0 / 186,480` (`0.0`) | `26,640 / 186,480` (`0.18087092394585402`) | `910,937 / 1,864,800` | `0.18087092394585402` | `kt02/S0_EXIT/frq_m` |
| `S0_F` | `0 / 10,656,384` (`0.0`) | `0 / 5,035,008` (`0.0`) | `0 / 57,890,304` | `0.0` | `None` |
| `S1_ENTRY` | `0 / 186,480` (`0.0`) | `26,640 / 186,480` (`0.18087092394585402`) | `910,937 / 1,864,800` | `0.18087092394585402` | `kt02/S1_ENTRY/frq_m` |
| `S1_TURB` | `0 / 506,160` (`0.0`) | absent on even `kt` | `650,362 / 2,530,800` | `0.10348143166507384` | `kt03/S1_TURB/sst` |
| `S1_FLUX` | `0 / 159,840` (`0.0`) | absent on even `kt` | `138,350 / 799,200` | `0.15084745942783684` | `kt03/S1_FLUX/qlwn` |
| `S1_EXIT` | `0 / 399,600` (`0.0`) | `26,640 / 399,600` (`0.18087092394585402`) | `1,375,137 / 3,996,000` | `0.18087092394585402` | `kt02/S1_EXIT/frq_m` |
| `S3_ENTRY` | `0 / 399,600` (`0.0`) | `26,640 / 399,600` (`0.18087092394585402`) | `1,375,137 / 3,996,000` | `0.18087092394585402` | `kt02/S3_ENTRY/frq_m` |
| `S3_COVER_IIF` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `914,429 / 4,262,400` | `0.1584001668307451` | `kt03/S3_COVER_IIF/sss_m` |
| `S3_FREEZE_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `472,550 / 2,264,400` | `0.15084745942783684` | `kt03/S3_FREEZE_IIF/sss_m` |
| `S3_CLAMP_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `472,212 / 2,264,400` | `0.15084745942783684` | `kt03/S3_CLAMP_IIF/sss_m` |
| `S3_EXIT_IIF` | `0 / 452,880` (`0.0`) | `0 / 452,880` (`0.0`) | `915,655 / 4,528,800` | `0.1584001668307451` | `kt03/S3_EXIT_IIF/sss_m` |
| `S3_EXIT` | `0 / 426,240` (`0.0`) | `26,640 / 426,240` (`0.18087092394585402`) | `1,376,701 / 4,262,400` | `0.18087092394585402` | `kt02/S3_EXIT/frq_m` |
| `S2_ENTRY` | `0 / 426,240` (`0.0`) | `26,640 / 426,240` (`0.18087092394585402`) | `1,376,701 / 4,262,400` | `0.18087092394585402` | `kt02/S2_ENTRY/frq_m` |
| `S2_RNF` | `0 / 506,160` (`0.0`) | `26,640 / 506,160` (`0.18087092394585402`) | `1,397,001 / 5,061,600` | `0.18087092394585402` | `kt02/S2_RNF/frq_m` |
| `S2_SSR` | `0 / 586,080` (`0.0`) | `26,640 / 586,080` (`0.18087092394585402`) | `1,590,723 / 5,860,800` | `0.18087092394585402` | `kt02/S2_SSR/frq_m` |
| `S2_FWB` | `0 / 586,080` (`0.0`) | `26,640 / 586,080` (`0.18087092394585402`) | `1,590,723 / 5,860,800` | `0.18087092394585402` | `kt02/S2_FWB/frq_m` |
| `S2_LBC` | `0 / 586,080` (`0.0`) | `26,640 / 586,080` (`0.18087092394585402`) | `1,590,723 / 5,860,800` | `0.18087092394585402` | `kt02/S2_LBC/frq_m` |
| `S2_EXIT` | `0 / 639,360` (`0.0`) | `26,640 / 639,360` (`0.18087092394585402`) | `1,785,007 / 6,393,600` | `0.18087092394585402` | `kt02/S2_EXIT/frq_m` |

#### Re-scored rung-10 200-row state ladder by step (`kt = 1..10`)

| Step | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Unequal / Total | Max `\|diff\|` | Max normalized `\|diff\|` | First non-bit row | First `OVER_BAR` row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kt01` | `11` | `9` | `0` | `389,384 / 12,893,760` | `0.0` | `0.0` | `kt01/stage1/T` | `None` |
| `kt02` | `7` | `9` | `4` | `115,990 / 12,893,760` | `0.00033734852885558553` | `1.1208924249237574e-05` | `kt02/entry/T` | `kt02/stage3/T` |
| `kt03` | `1` | `0` | `19` | `4,750,221 / 12,893,760` | `0.00033768906680009536` | `1.122027118600728e-05` | `kt03/entry/T` | `kt03/entry/T` |
| `kt04` | `0` | `0` | `20` | `6,603,874 / 12,893,760` | `0.00032915546303158294` | `5.55517683449017e-05` | `kt04/entry/T` | `kt04/entry/T` |
| `kt05` | `0` | `0` | `20` | `6,767,313 / 12,893,760` | `0.00032134585836729457` | `5.55517683449017e-05` | `kt05/entry/T` | `kt05/entry/T` |
| `kt06` | `0` | `0` | `20` | `6,826,723 / 12,893,760` | `0.0009521877223015451` | `3.158367658993312e-05` | `kt06/entry/T` | `kt06/entry/T` |
| `kt07` | `0` | `0` | `20` | `6,879,815 / 12,893,760` | `0.00995590568548943` | `0.004510917554737096` | `kt07/entry/T` | `kt07/entry/T` |
| `kt08` | `0` | `0` | `20` | `6,878,083 / 12,893,760` | `0.00995590568548943` | `0.007208576105557409` | `kt08/entry/T` | `kt08/entry/T` |
| `kt09` | `0` | `0` | `20` | `6,912,052 / 12,893,760` | `0.005711342197502384` | `0.005068461616668585` | `kt09/entry/T` | `kt09/entry/T` |
| `kt10` | `0` | `0` | `20` | `6,893,881 / 12,893,760` | `0.01817450117457753` | `0.012533582463925592` | `kt10/entry/T` | `kt10/entry/T` |
| **Total** | **`19`** | **`18`** | **`163`** | **`53,017,336 / 128,937,600`** | **`0.01817450117457753`** | **`0.012533582463925592`** | **`kt01/stage1/T`** | **`kt02/stage3/T`** |

- independent review: operator

## Resume 10 — Instrument audit of `S0_EXIT` field pairing and offline localization of `kt = 2` differences

### (1) Instrument audit: `S0_EXIT` field order/names, why `0.18087092394585402` was previously cited on `ssh_m`, field-swap plant, and whether `frq_m` genuinely differs

1. **Writer stream order vs decoder order vs `NemoSurfaceMeans` card order**:
   - In the NEMO surface writer (`round340_surface_writer.F90:23-25`), `r340_surface_frame` writes the 7 `sbc_ssm` fields for `kid >= 1` (`S0_EXIT`..`S2_EXIT`) in fixed unformatted stream order:
     `sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m, frq_m`.
   - In the standalone gate (`nemo_testcase_l4_orca2_round340_standalone_gate.py:57-68` and `nemo_testcase_l4_orca2_round340_standalone_gate.py:223-225`), `read_exact_record` decodes `kid = 1` (`S0_EXIT`) and `kid = 2` (`S1_ENTRY`) in the identical tuple order `OUTER_FULL_FIELDS[:7] = ("sst_m", "sss_m", "ssu_m", "ssv_m", "ssh_m", "e3t_m", "frq_m")`.
   - In the card (`packages/ocean/legoesm/ocean/forcing/nemo_surface.py:27-34`), `NemoSurfaceMeans` declares its `NamedTuple` slots in `sbc_ssm` accumulation order `(ssu, ssv, sst, sss, ssh, e3t, frq)`. However, `_surface_boundaries_for_step` (`nemo_testcase_l4_orca2_round340_standalone_gate.py:935-940,1009-1011`) binds each field by explicit attribute name (`"sst_m": means.sst, "sss_m": means.sss, "ssu_m": means.ssu, "ssv_m": means.ssv, "ssh_m": means.ssh, "e3t_m": means.e3t, "frq_m": means.frq`) and materializes `kt02/S0_EXIT` in `OUTER_FIELDS_BY_KID[1]` order; `score_surface_boundaries` (`nemo_testcase_l4_orca2_round340_standalone_gate.py:1176-1185`) enforces `tuple(actual[boundary]) == tuple(expected[boundary])` and scores `actual[boundary][name]` against `expected[boundary][name]`.
   - At `kt = 2` in Resume 9, all six other `S0_EXIT` fields match NEMO **bit-for-bit** (`0 / 26,640` unequal words, `max |diff| = 0.0` each) across mutually disjoint physical ranges:
     - `kt02/S0_EXIT/sst_m`: `0 / 26,640` unequal (`range [-1.9168, 30.0923]` degC)
     - `kt02/S0_EXIT/sss_m`: `0 / 26,640` unequal (`range [0.0, 40.3241]` PSU)
     - `kt02/S0_EXIT/ssu_m`: `0 / 26,640` unequal (`range [-0.3289, 0.3368]` m/s)
     - `kt02/S0_EXIT/ssv_m`: `0 / 26,640` unequal (`range [-0.5055, 0.2948]` m/s)
     - `kt02/S0_EXIT/ssh_m`: `0 / 26,640` unequal (`range [-1.2268, 1.3251]` m)
     - `kt02/S0_EXIT/e3t_m`: `0 / 26,640` unequal (`range [9.7952, 10.0750]` m)
2. **Why `0.18087092394585402` was previously cited on `kt02/S0_EXIT/ssh_m` in Resume 6**:
   - In Resume 6 (`by_boundary["kt02/S0_EXIT"]`) and Resume 7 (`by_boundary["kt02/S0_EXIT"]`), `0.18087092394585402` was the boundary-aggregate `max_abs` over all 7 fields in `kt02/S0_EXIT`.
   - That value (`0.18087092394585402 = |0.819129076054146 - 1.0|`) always came from `kt02/S0_EXIT/frq_m` on dry cell `(j = 0, i = 0)` (where NEMO `fraqsr_1lev` is `1.0` because `qsr == 0`, while `top_layer_absorbed_fraction` with unit `sw = 1.0`, `stage3_stretch = 1.0`, `e3t_0 = 10.0`, and `chl[0, 0] = 0.06423438986142477` returns `0.819129076054146` identically in Resume 6, Resume 7, and Resume 9).
   - In Resume 6, `kt02/S0_EXIT/ssh_m` had `max |diff| = 0.09343506101532833` (from the SI3 initial SSH depression), which was smaller than `frq_m`'s `0.18087092394585402`; Resume 6's prose parenthetical misattributed the `kt02/S0_EXIT` boundary maximum `0.18087092394585402` to `ssh_m`.
3. **Field-swap plant (`test_resume10_s0_exit_field_pairing_plant_and_kt2_offline_diagnosis` in `tests/ocean/fidelity/test_orca2_round340_standalone.py`)**:
   - **Plant 1 (`frq_m <-> ssh_m` value swap in `kt02/S0_EXIT`)**: `score_composed_card` raises `GateError("surface exact bar failed: 12 / 42")` with `kt02/S0_EXIT/frq_m` and `kt02/S0_EXIT/ssh_m` flagged (`6 / 6` unequal words each) and `e3t_m` at `0 / 6`.
   - **Plant 2 (`ssh_m <-> e3t_m` value swap in `kt02/S0_EXIT`)**: `score_composed_card` raises `GateError("surface exact bar failed: 12 / 42")` with `kt02/S0_EXIT/ssh_m` and `kt02/S0_EXIT/e3t_m` flagged (`6 / 6` unequal words each) and `frq_m` at `0 / 6`.
   - **Plant 3 (`ssh_m <-> frq_m` key-order swap in `kt02/S0_EXIT`)**: `score_surface_boundaries` raises `GateError("kt02/S0_EXIT field inventory moved")`.
   - Against admitted arm A (`orca2_s4_exact_a_np2`), scoring the reconstructed `kt = 2` `fraqsr_1lev` against recorded `kt02/S0_EXIT/frq_m` gives `0 / 26,640` unequal words (`max |diff| = 0.0`), whereas swapping `frq_m` with `ssh_m` or `e3t_m` fails on `26,640 / 26,640` words (`max |diff| = 2.2268` and `9.2869`).
4. **Does `frq_m` genuinely differ?**:
   - **Yes**: `kt02/S0_EXIT/frq_m` genuinely differs on **`26,640 / 26,640` cells** (`16,433` wet cells with `max |diff| = 0.18087092394585402` on polar-night wet cells where `qsr == 0` and NEMO sets `fraqsr_1lev = 1.0`, plus `10,207` dry cells with `max |diff| = 0.18087092394585402` at `[0, 0]`; on sunlit wet cells where `qsr > 0`, `14,007` cells differ with `max |diff| = 0.0020165529064691036`).
   - Under K0 (`nn_ice = 1`), `frq_m` is a diagnostic S0 mean field: it is never read by `sbc_blk`, `sbc_ice_if`, `sbc_rnf`, `sbc_ssr`, `sbc_fwb`, or any ocean routine (`frq_m` is only consumed by SI3 `icesbc.F90` when `nn_ice = 2`).

### (2) Localization of the first genuine `kt = 2` differences and proposed literal fixes (not landed)

#### (a) First non-bit surface row: `kt02/S0_EXIT/frq_m` (`26,640 / 26,640`, `max |diff| = 0.18087092394585402`) — `CONFIRMED`

1. **S0 two-step means at the `nn_fsbc = 2` boundary (`sbc_ssm` time levels at `kt = 2` vs the card)**:
   - In NEMO (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:109-176`, specifically `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:133-162`), `sbc` is called at step entry with `Kbb = Nbb, Kmm = Nbb` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:128`).
   - At `kt = 1` (`kt == nit000 .AND. .NOT. l_ssm_mean`), `sbc_ssm` initializes `ssu_m..frq_m` to `(nn_fsbc - 1) * value = 1 * value`, adds `1 * value`, and divides by `REAL(nn_fsbc, wp) = 2._wp`, where `fraqsr_1lev` is the cold-start `1._wp` initialized by `tra_qsr_init` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:1289-1293`).
   - At `kt = 2`, `MOD(kt - 2, nn_fsbc) == 0` (`MOD(0, 2) == 0`) triggers the `ELSEIF` branch (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:133-162`), which resets `ssu_m, ssv_m, sst_m, sss_m, ssh_m, e3t_m, frq_m` to `0._wp` and accumulates the `kt = 2` step-entry `Nbb` state (`kt = 1` Stage-3 exit `uu, vv, ts, ssh, r3t` plus `fraqsr_1lev` written during `kt = 1` Stage 3) without dividing by `nn_fsbc` (since `MOD(kt - 1, nn_fsbc) == 0` is false at `kt = 2`, and the division happens at `kt = 3` after adding the `kt = 3` step-entry state).
   - Both NEMO (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:133-162`) and the card (`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1006-1030` -> `packages/ocean/legoesm/ocean/forcing/nemo_surface.py:143-187`) reset at `kt = 2` and accumulate the identical `kt = 2` step-entry (`kt = 1` Stage-3 exit) state for `ssu_m, ssv_m, sst_m, sss_m, ssh_m, e3t_m` (`0 / 26,640` unequal words on all six).
2. **First differing statement (`CONFIRMED`) for `kt02/S0_EXIT/frq_m`**:
   - **NEMO**: During `kt = 1` Stage 3 (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:185-197`, `CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )`), `stp_RK3_stg` calls `IF( ln_traqsr ) CALL tra_qsr( kstp, Kmm, ts, Krhs )` with `Kmm = Nnn`, the **Stage-2 exit** `r3t` (`r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )` at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:215,525`), and `qsr_RGBc` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402`) updates `fraqsr_1lev` via:
     - `zpsi = zc(ji,jj,3) * ((gdepw_1d(jk+1)) *(1._wp+r3t(ji,jj,Kmm)))` at `jk = 1` (`gdepw_1d(2) = 10.000015058326067` m, not `e3t_0(1) = 10.0` m),
     - `itab(ji,jj,jk) = NINT( 41 + 20.*LOG10(zchl) + 1.e-15 )` with source-rounded intermediate operations,
     - `ze0 = zz0 * qsr ; zeR = zz1 * qsr ; zeG = zz1 * qsr ; zeB = zz1 * qsr` and `zzeT = ( zze0 + zzeB + zzeG + zzeR ) * wmask(ji,jj,jk+1)` at `jk = 1` (`ze3t = (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))`),
     - `WHERE( qsr(:,:) /= 0._wp ) fraqsr_1lev(:,:) = 1._wp - zeT(:,:) / qsr(:,:) ELSEWHERE fraqsr_1lev(:,:) = 1._wp END WHERE`.
   - **legoESM**: `nemo_orca2_k0_ocean_step` (`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1076-1107`) and `top_layer_absorbed_fraction` (`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:1178-1201`) instead:
     1. Read `stage3_eta = stepped.stage_outputs[-1][4]` / `state_after.eta.data` (Stage-3 `Kaa` `ssh` at `t = N + 1`) instead of Stage-3 `Kmm = Nnn` (`0.5 * (r3t(Kbb) + r3ta)` at `t = N + 1/2`),
     2. Call `shortwave_penetration_rgb_tendency` with default `ShortwavePenetrationConfig(scheme="rgb_chl")`, which uses `gdepw_bottom = dz_live[..., 0] = e3t_0(1) * stretch` instead of `gdepw_1d(2) * stretch` and omits `+ 1.e-15` and `nemo_source_round` (whereas the ocean step's own `tra_qsr` already uses `_nemo_morel_berthon_chl_column` and `_nemo_rgb_class_row` at `packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:404-451`),
     3. Pass `wet = jnp.ones(shape + (2,))` instead of `wmask(ji,jj,2)` (`z_coord.is_active[..., 1]`, which is `0` on `11` single-level wet cells where `mbathy == 1`),
     4. Evaluate with unit `sw = jnp.ones(shape)` and return `frac_per_m[..., 0] * dz_live[..., 0]` instead of `WHERE(qsr /= 0._wp) 1._wp - zzeT / qsr ELSEWHERE 1._wp` (causing `10,207` dry cells and `2,426` polar-night wet cells where `qsr == 0` to get `~0.819` instead of `1.0`, plus `2,718` sunlit wet cells to differ by `1-2` ULPs from `qsr` scaling before division).
   - **Offline ablation (`CONFIRMED`)**: Replacing `next_shortwave_fraction` at `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1076-1107` with the literal `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402` evaluation on `stretch_kmm = nemo_r3t_rk3_stage1_stretch(zc, state.eta.data, state_after.eta.data, state.H_bathy.data, half=True)` (`packages/ocean/legoesm/ocean/eos.py:1004-1035`) reduces `kt02/S0_EXIT/frq_m` (and all `11` `kt = 2` surface boundaries carrying `frq_m`) from `26,640 / 26,640` (`max |diff| = 0.18087092394585402`) to **`0 / 26,640` unequal words (`max |diff| = 0.0`, 100% bit-exact)**.

#### (b) First `OVER_BAR` state row: `kt02/stage3/T` (`26,525 / 799,200`, `max |diff| = 3.3734852885558553e-4`) — `CONFIRMED`

1. **Non-update step retention at `kt = 2` (`MOD(kt - 1, nn_fsbc) /= 0`)**:
   - In NEMO, `sbc` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521`) gates `sbc_blk`, `sbc_ice_if`, `sbc_rnf`, `sbc_ssr`, and `sbc_fwb` on `MOD(kt - 1, nn_fsbc) == 0` (`nn_fsbc = 2`), so at `kt = 2` those surface fluxes are retained unchanged from `kt = 1`.
   - However, chlorophyll `sf_chl` is NOT a surface-module `nn_fsbc` field: `tra_qsr` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:310`) executes `CALL fld_read( kt, 1, sf_chl )` with `kn_fsbc = 1` at **every** ocean time step `kt` during Stage 3 (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:525`). Between `kt = 1` and `kt = 2`, `sf_chl(1)%fnow` changes on `14,344 / 26,640` wet cells (`max |diff| = 0.007459677455405878`), shifting the discrete RGB lookup index `itab` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402`) by `1` table class on `1,303` sunlit wet cells across levels `1..22`.
2. **First differing statement (`CONFIRMED`) for `kt02/stage3/T`**:
   - **NEMO**: `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:310` (`CALL fld_read( kt, 1, sf_chl )` inside `tra_qsr`, called at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:525`).
   - **legoESM**: `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1279-1351` (`if not update: ... return prior_surface._replace(means=means)`). Although `nemo_orca2_k0_ocean_step` receives the `kt = 2` `chl` array and passes `chl=chl` into `nemo_orca2_k0_surface_card`, `nemo_blk_oce_ncar` returns `prior_surface._replace(means=means)` on `not update` steps without updating `prior_surface.surface_forcing.chl` (or `geothermal_heat_flux_mwm2`). As a result, `model.step` at `kt = 2` receives `kt = 1`'s stale `chl` inside `surface.surface_forcing.chl`.
   - **Offline ablation (`CONFIRMED`)**: Updating `surface_forcing` on the `not update` return path at `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1279-1351` so `surface_forcing.chl` carries the current step's `chl` reduces `kt02/stage3/T` by **nearly 7 orders of magnitude**, from `26,525 / 799,200` (`max |diff| = 3.3734852885558553e-4`, `normalized_max_abs = 1.1208924249237574e-5`) to **`216 / 799,200` (`max |diff| = 4.622280336263884e-11`, `normalized_max_abs = 1.5358238197493165e-12`, of which `179` are dry `-0.0` words and only `37` are wet words)**.
3. **Residual ULP-scale `kt = 2` Stage-3 differences after the `chl` fix (`kt02/stage3/{T, S, u, v}`)**:
   - With `surface_forcing.chl` updated at `kt = 2`, the `kt = 2` Stage-3 rows become:
     - `kt02/stage3/T`: `216 / 799,200` (`37` wet words, `max |diff| = 4.622280336263884e-11`, `normalized_max_abs = 1.5358238197493165e-12`)
     - `kt02/stage3/S`: `29,446 / 799,200` (`144` wet words, `max |diff| = 5.802682775196445e-10`, `normalized_max_abs = 1.4383963487311444e-11`)
     - `kt02/stage3/u`: `29,638 / 799,200` (`130` wet words, `max |diff| = 5.3600008365184126e-11`, `normalized_max_abs = 1.1869583804731195e-10`)
     - `kt02/stage3/v`: `29,517 / 799,200` (`138` wet words, `max |diff| = 1.3563434252512031e-11`, `normalized_max_abs = 2.6606467527401134e-11`)
     - `kt02/stage3/ssh`: `0 / 26,640` (`BIT_EXACT`, `max |diff| = 0.0`)
   - This residual is the same `kt = 2` Stage-3 implicit vertical solver boundary (`CALL dyn_zdf` / `CALL tra_zdf` at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:405,538` and `zdf_tke` `CASE(0)` surface mixing-length anchor at `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:608-638` vs `tke.py:285-289`) already localized on the certified SI3 rung-10 card in Rounds 332–333 (`CALL tra_zdf` / `CALL dyn_zdf` `CONFIRMED` at the call boundary; internal solver recurrence primitive `PLAUSIBLE`).

#### (c) Proposed literal fixes (do not land in Resume 10)

1. **Proposed literal fix 1 (`kt02/S0_EXIT/frq_m` -> `0 / 26,640`, bit-exact at `kt = 2`)**:
   - In `packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:1178-1201`, add a literal helper `nemo_fraqsr_1lev(qsr, chl_surface, e3t_1_kmm, gdepw_2_kmm, wmask_2, config=...)` transcribing `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402` using `_nemo_morel_berthon_chl_column` and `_nemo_rgb_class_row` (`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:404-451`) on `gdepw_2_kmm = sr(gdepw_1d(2) * stretch_kmm)`, `ze3t = sr(e3t_0(1) * stretch_kmm)`, `zzeT = sr(total0 * wmask_2)`, and `jnp.where(qsr != 0.0, sr(1.0 - sr(zzeT / safe_qsr)), 1.0)`.
   - In `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1076-1107`, compute Stage-3's `Kmm` stretch via `stretch_kmm = nemo_r3t_rk3_stage1_stretch(card.ocean.recipe.z_coord, state.eta.data, state_after.eta.data, state.H_bathy.data, half=True)` (`packages/ocean/legoesm/ocean/eos.py:1004-1035`, matching `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:215,525` without depending on `stage_outputs`) and call `nemo_fraqsr_1lev` with `surface.s2.penetrative_qsr` and `z_coord.is_active[..., 1]`.
2. **Proposed literal fix 2 (`kt02/stage3/T` -> `3.37e-4` reduced to `4.62e-11`, `0 / 10,842,528` unequal surface words at `kt = 2` when combined with Fix 1)**:
   - In `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1279-1351`, on `not update` steps (`MOD(kt - 1, nn_fsbc) /= 0`), update the carried `OceanSurfaceForcing` with the current step's `kn_fsbc = 1` ocean fields (`chl` and `geothermal_heat_flux_mwm2`) while retaining all `nn_fsbc = 2` surface fluxes (`q_net, sw_down, emp, utau_u_native, vtau_v_native, ice_concentration, runoff_tracer_content`) unchanged:
     ```python
     if not update:
         if prior_surface is None:
             raise ValueError(
                 "nn_fsbc=2 preservation step requires prior_surface carry")
         sf = prior_surface.surface_forcing
         if sf is not None:
             sf = sf._replace(
                 chl=_crop_to_surface_interior(chl, interior_shape),
                 geothermal_heat_flux_mwm2=_crop_to_surface_interior(
                     geothermal_heat_flux_mwm2, interior_shape),
             )
         return prior_surface._replace(means=means, surface_forcing=sf)
     ```

- independent review: operator

## Resume 11 — Literal `fraqsr_1lev` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402`), `kt = 2` non-update step retention (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521`), and 10-step re-score

### (1) Literal implementation of the two Resume-10 `CONFIRMED` fixes in the standalone K0 path (`d67e5e22f9c7`)

1. **Fix (a): `fraqsr_1lev` / `frq_m` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402`)**:
   - **NEMO statement**: During Stage 3 (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:215,525`), `tra_qsr` is called with `Kmm = Nnn` (`r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )` from Stage 2), and `qsr_RGBc` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402`) evaluates `zpsi = zc(ji,jj,3) * ((gdepw_1d(jk+1)) *(1._wp+r3t(ji,jj,Kmm)))` at `jk = 1` (`gdepw_1d(2)`), `itab(ji,jj,jk) = NINT( 41 + 20.*LOG10(zchl) + 1.e-15 )`, `ze0 = zz0 * qsr ; zeR = zz1 * qsr ; zeG = zz1 * qsr ; zeB = zz1 * qsr`, `ze3t = (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))`, `zzeT = ( zze0 + zzeB + zzeG + zzeR ) * wmask(ji,jj,2)`, and `WHERE( qsr(:,:) /= 0._wp ) fraqsr_1lev(:,:) = 1._wp - zeT(:,:) / qsr(:,:) ELSEWHERE fraqsr_1lev(:,:) = 1._wp END WHERE`.
   - **legoESM transcription**:
     - In `packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:454-548`, extracted shared helpers `_nemo_qsr_rgb_attenuation_coeffs`, `_nemo_qsr_rgb_surface_bands`, and `_nemo_qsr_rgb_attenuate_level` (reusing `_nemo_morel_berthon_chl_column` and `_nemo_rgb_class_row` at `packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:404-451`) so `_nemo_qsr_rgb_tendency` and `nemo_fraqsr_1lev(sw_down, chl_surface, e3t_1_live, gdepw_2_live, wmask_2, config)` share a single literal RGB/chl implementation. `top_layer_absorbed_fraction` (`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:1178-1201`) is untouched so all non-K0 callers (`run_omip_core2.py`, `test_lead_freeze_nemo_qlead.py`) stay bit-identical.
     - In `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1076-1107`, `nemo_orca2_k0_ocean_step` computes `stage3_kmm_stretch = nemo_r3t_rk3_stage1_stretch(z_coord, state.eta.data, state_after.eta.data, state.H_bathy.data, half=True)` (`packages/ocean/legoesm/ocean/eos.py:1004-1035`), `e3t_1_kmm = R(e3t_reference * R(1.0 + R(r3t_kmm * tmask_1)))`, `gdepw_2_kmm = R(gdepw_1d_2 * stage3_kmm_stretch)`, `wmask_2 = is_active[..., 1] * is_active[..., 0]`, and calls `nemo_fraqsr_1lev`.
2. **Fix (b): `kt = 2` non-update step retention (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521` and `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:310`)**:
   - **NEMO statement**: On `MOD(kt - 1, nn_fsbc) /= 0` steps (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521`), `sbc_blk`, `sbc_ice_if`, `sbc_rnf`, `sbc_ssr`, and `sbc_fwb` do not execute, so the module arrays `utau, vtau, taum, wndm, qsr, qns, emp, sfx, fr_i, rnf, rnf_tsc, erp, qrp, coefice` retain their `kt - 1` post-S2 exit values across `kid = 3..10`, while `sbc_ssm` updates `sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m, frq_m` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:133-162`), `ts(:,:,1,:,Kbb/Kmm)` in `S3_COVER_IIF` / `S3_EXIT_IIF` holds the current step's unclamped surface temperature, and `tra_qsr` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:310`, `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:525`) reads `sf_chl` with `kn_fsbc = 1` at every ocean step `kt`.
   - **legoESM transcription**:
     - In `packages/ocean/legoesm/ocean/bulk_flux_omip.py:1279-1351`, on `not update` steps (`MOD(kt - 1, nn_fsbc) /= 0`), `nemo_blk_oce_ncar` constructs `retained_pre`, `retained_iif`, `retained_bulk`, `retained_s2`, and `retained_forcing` from `prior_surface`: `bulk_before_ice`, `ice_if`, `bulk`, and `s2` retain `prior_surface.s2`'s post-S2 fluxes (`qsr, qns, emp, utau, vtau, taum`), `sst/ssu/ssv` reflect the current `means`, `ice_if.ts_surface` and `bulk.ts_surface` hold the current step's unclamped `ice_if.ts_surface`, and `surface_forcing` retains all `nn_fsbc = 2` surface fluxes while updating `chl` and `geothermal_heat_flux_mwm2` to the current step's `kn_fsbc = 1` values.
     - In `nemo_testcase_l4_orca2_round340_standalone_gate.py:935-940,1009-1011`, removed the gate-side `if kt % 2:` flux-substitution workarounds in `_surface_boundaries_for_step` so `outer_values[kid]` (`kid = 3..11`) and `S3_COVER_IIF` / `S3_EXIT_IIF` read `surface.bulk_before_ice`, `surface.ice_if`, `surface.bulk`, and `surface.s2` directly on every `kt`.

### (2) Checklist M1–M6 verification and parent-vs-current bit-identity proofs

- **M1 (literal transcription)** & **M2 (no default arguments)**: `nemo_fraqsr_1lev` has 6 required parameters and 0 default values (`test_literal_physical_inputs_have_no_defaults` in `tests/ocean/fidelity/test_nemo_testcase_l4_orca2_round337_surface.py`).
- **M3 (true `eager == jit` with dynamic `jit` arguments)**: Verified bit-for-bit (`0` unequal `uint64` words across all pytree leaves) between `with jax.disable_jit():` and `jax.jit(...)` with dynamic wind, air temperature, humidity, SLP, radiative, precipitation, snow, chlorophyll, geothermal, and SSH arguments in `test_resume11_fraqsr_1lev_production_call_site_revert_plant_jit_and_jvp` and `test_resume11_kt2_non_update_retention_production_call_site_revert_plant_jit_and_jvp` (`tests/ocean/fidelity/test_orca2_round340_standalone.py`).
- **M4 (differentiability)**: Finite, non-zero forward JVP == reverse VJP (`rtol = 2e-14, atol = 2e-14`) verified through the 2-step composed `nemo_orca2_k0_ocean_step` card wrt `(chl, eta0)` in `test_resume11_fraqsr_1lev_production_call_site_revert_plant_jit_and_jvp` and wrt `(u_wind, qsr_down, chl_kt2, ggh_kt2)` in `test_resume11_kt2_non_update_retention_production_call_site_revert_plant_jit_and_jvp`, plus 1-step real ORCA2 `31 x 148 x 180` JVP == VJP in `test_real_composed_card_n1_reverse_equals_forward` (`ORCA2_R340_REAL_GRADIENT=1`, `rtol = 2e-12, atol = 2e-12`).
- **M5 (production-call-site revert plants)**:
  - `test_resume11_fraqsr_1lev_production_call_site_revert_plant_jit_and_jvp`: reverting to parent `top_layer_absorbed_fraction(e3t_0 * stretch_kaa, chl_surface=chl)` fails on `6 / 6` cells of the `2 x 3` grid and `26,640 / 26,640` words against admitted arm A `kt02/S0_EXIT/frq_m` (whereas production `nemo_fraqsr_1lev` gives `0 / 26,640` unequal words, `max |diff| = 0.0`).
  - `test_resume11_kt2_non_update_retention_production_call_site_revert_plant_jit_and_jvp`: reverting `not update` retention to parent `prior_surface._replace(means=means)` fails on `6 / 6` cells of `bulk_before_ice.qns`, `6 / 6` cells of `bulk_before_ice.emp`, `6 / 6` cells of `ice_if.qns`, `6 / 6` cells of `bulk.qns`, `6 / 6` cells of `bulk.emp`, `6 / 6` cells of `s2.qns_after_ssr`, `6 / 6` cells of `s2.emp_after_ssr`, `6 / 6` cells of `surface_forcing.chl`, `6 / 6` cells of `surface_forcing.geothermal_heat_flux_mwm2`, and `12 / 12` cells of `r2.ocean.T.data`.
- **M6 (non-K0 callers and other cards bit-identical, parent `5f0c96a4194e` vs current `d67e5e22f9c7`)**:
  - Card-fingerprint gate (`nemo_testcase_l4_orca2_card_fingerprint.py`): `0` changed arrays across all 6 cards (`GYRE-zco`: `fcad441865c685c4`, `LOCK_EXCHANGE-zco`: `ce229cd64fd544d9`, `ORCA2-zps`: `d699476d15b0cf24`, `OVERFLOW-zps`: `a4f02a29ba0dee53`, `VORTEX-zco`: `13d7c997a4d8e13f`, `VORTEX_VEC-zco`: `515b238ef4bb290a`).
  - Non-K0 caller array counts (`test_resume11_kt2_non_update_retention_production_call_site_revert_plant_jit_and_jvp` and parent-vs-current comparison):
    - `_nemo_qsr_rgb_tendency` on full ORCA2 `(148, 180, 30)`: `1 / 1` array (`799,200 / 799,200` `float64` words bit-identical, `0` unequal words vs parent `5f0c96a4194e`).
    - `top_layer_absorbed_fraction` (RGB-chl and Jerlov modes on ORCA2 `(148, 180)`): `2 / 2` arrays (`53,280 / 53,280` `float64` words bit-identical, `0` unequal words vs parent `5f0c96a4194e`).
    - `nemo_blk_oce_ncar` non-K0 (`s0=None, s2=None`, `_NO_ICE` and observed-ice modes on `(2, 3)`): `46 / 46` non-`None` arrays (`276 / 276` `float64` words bit-identical, `0` unequal words vs parent `5f0c96a4194e`).

### (3) Re-scored 10-step real ORCA2 K0 composed card (+ arm-B control)

Executed `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_score_composed_card.py` on commit `d67e5e22f9c7` (full JSON saved at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r340_resume11/score_composed_card.json`):

- **Arm-B vs arm-A reproducibility control**: `PASS_REPRODUCIBILITY_EXACT` (`0 / 123,824,304` surface words unequal, `0 / 128,937,600` rung-10 state words unequal across `200 / 200` rows).
- **Overall surface score (`kt = 1..10`)**: `FAIL_S4_SURFACE`, `914,406 / 123,824,304` unequal words (`0.7385%`, down from `11,769,984 / 123,824,304`), `max |diff| = 3.1096305974642746e-08` (down from `0.18087092394585402`).
  - **Both `kt = 1` (`0 / 18,035,664`) and `kt = 2` (`0 / 10,842,528`) are 100% bit-exact (`max |diff| = 0.0`) across every surface boundary**.
  - **First non-bit (and first non-zero `max |diff|`) surface row**: `kt03/S0_EXIT/sst_m` (`4 / 26,640` unequal words, `max |diff| = 8.992806499463768e-15`, `rms = 7.794270656231612e-17`, `first_unequal_index = [147, 29]`), inherited from the `37` wet unequal words on `kt02/stage3/T` via `sbc_ssm` at `kt = 3` (`kt03/S0_EXIT/ssh_m`, `kt03/S0_EXIT/e3t_m`, and `kt03/S0_EXIT/frq_m` are all `0 / 26,640` bit-exact).
- **Overall rung-10 200-row state ladder (`kt = 1..10`)**: `19 BIT_EXACT`, `18 AT_BAR_NOT_EXACT`, `163 OVER_BAR`, `4,916,620 / 128,937,600` unequal words (down from `53,017,336 / 128,937,600`), `max |diff| = 2.3088674083737715e-08` (down from `0.01817450117457753`, `~787,000x` smaller), `max_normalized_max_abs = 2.0148460472540064e-09` (down from `0.012533582463925592`).
  - **First non-bit rung-10 row**: `kt01/stage1/T` (`AT_BAR_NOT_EXACT`, `157,456 / 799,200` dry `-0.0` vs `+0.0` words, `0` wet words unequal, `max |diff| = 0.0`, `normalized_max_abs = 0.0`).
  - **First `OVER_BAR` rung-10 row**: `kt02/stage3/T` (`OVER_BAR`, `216 / 799,200` unequal words — `179` dry `-0.0` words and `37` wet words — `max |diff| = 4.622280336263884e-11`, `normalized_max_abs = 1.535823805833218e-12`, `rms = 1.1306128932932049e-13`, `first_unequal_index = [2, 43, 20]`).

#### Re-scored surface boundaries by step (`kt = 1..10`)

| Step | Unequal words | Total words | Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- |
| `kt01` | `0` | `18,035,664` | `0.0` | `None` |
| `kt02` | `0` | `10,842,528` | `0.0` | `None` |
| `kt03` | `874` | `12,894,000` | `2.0566187641790634e-12` | `kt03/S0_EXIT/sst_m` |
| `kt04` | `31,704` | `10,842,528` | `1.4932322739413806e-11` | `kt04/S0_EXIT/sst_m` |
| `kt05` | `62,403` | `12,894,000` | `3.872457909892546e-11` | `kt05/S0_EXIT/sst_m` |
| `kt06` | `80,528` | `10,842,528` | `1.9749601953833462e-10` | `kt06/S0_EXIT/sst_m` |
| `kt07` | `128,260` | `12,894,000` | `2.1825144358444959e-08` | `kt07/S0_EXIT/sst_m` |
| `kt08` | `142,839` | `10,842,528` | `2.1803089111926965e-08` | `kt08/S0_EXIT/sst_m` |
| `kt09` | `224,299` | `12,894,000` | `3.1096305974642746e-08` | `kt09/S0_EXIT/sst_m` |
| `kt10` | `243,499` | `10,842,528` | `2.8152214781584917e-08` | `kt10/S0_EXIT/sst_m` |
| **Total** | **`914,406`** | **`123,824,304`** | **`3.1096305974642746e-08`** | **`kt03/S0_EXIT/sst_m`** |

#### Re-scored surface boundaries by boundary family (`kt = 1..10`)

| Boundary family | `kt01` (`unequal / count`, `max \|diff\|`) | `kt02` (`unequal / count`, `max \|diff\|`) | `kt = 1..10` (`unequal / count`) | `kt = 1..10` Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- | --- |
| `S0_ENTRY` | `0 / 0` (`0.0`) | `0 / 0` (`0.0`) | `0 / 0` | `0.0` | `None` |
| `S0_EXIT` | `0 / 186,480` (`0.0`) | `0 / 186,480` (`0.0`) | `54,354 / 1,864,800` | `1.1644181174830237e-09` | `kt03/S0_EXIT/sst_m` |
| `S0_F` | `0 / 10,656,384` (`0.0`) | `0 / 5,035,008` (`0.0`) | `0 / 57,890,304` | `0.0` | `None` |
| `S1_ENTRY` | `0 / 186,480` (`0.0`) | `0 / 186,480` (`0.0`) | `54,354 / 1,864,800` | `1.1644181174830237e-09` | `kt03/S1_ENTRY/sst_m` |
| `S1_TURB` | `0 / 506,160` (`0.0`) | absent on even `kt` | `21,389 / 2,530,800` | `1.704711394268088e-08` | `kt03/S1_TURB/sst` |
| `S1_FLUX` | `0 / 159,840` (`0.0`) | absent on even `kt` | `2,505 / 799,200` | `3.1096305974642746e-08` | `kt05/S1_FLUX/qlwn` |
| `S1_EXIT` | `0 / 399,600` (`0.0`) | `0 / 399,600` (`0.0`) | `65,460 / 3,996,000` | `3.1096305974642746e-08` | `kt03/S1_EXIT/sst_m` |
| `S3_ENTRY` | `0 / 399,600` (`0.0`) | `0 / 399,600` (`0.0`) | `65,460 / 3,996,000` | `3.1096305974642746e-08` | `kt03/S3_ENTRY/sst_m` |
| `S3_COVER_IIF` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `42,845 / 4,262,400` | `3.1096305974642746e-08` | `kt03/S3_COVER_IIF/sss_m` |
| `S3_FREEZE_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `19,611 / 2,264,400` | `3.1096305974642746e-08` | `kt03/S3_FREEZE_IIF/sss_m` |
| `S3_CLAMP_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `19,199 / 2,264,400` | `3.1096305974642746e-08` | `kt03/S3_CLAMP_IIF/sss_m` |
| `S3_EXIT_IIF` | `0 / 452,880` (`0.0`) | `0 / 452,880` (`0.0`) | `43,625 / 4,528,800` | `2.815312427628669e-08` | `kt03/S3_EXIT_IIF/sss_m` |
| `S3_EXIT` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `66,652 / 4,262,400` | `2.815312427628669e-08` | `kt03/S3_EXIT/sst_m` |
| `S2_ENTRY` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `66,652 / 4,262,400` | `2.815312427628669e-08` | `kt03/S2_ENTRY/sst_m` |
| `S2_RNF` | `0 / 506,160` (`0.0`) | `0 / 506,160` (`0.0`) | `67,594 / 5,061,600` | `2.815312427628669e-08` | `kt03/S2_RNF/sst_m` |
| `S2_SSR` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `80,170 / 5,860,800` | `2.8152214781584917e-08` | `kt03/S2_SSR/sst_m` |
| `S2_FWB` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `80,170 / 5,860,800` | `2.8152214781584917e-08` | `kt03/S2_FWB/sst_m` |
| `S2_LBC` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `80,170 / 5,860,800` | `2.8152214781584917e-08` | `kt03/S2_LBC/sst_m` |
| `S2_EXIT` | `0 / 639,360` (`0.0`) | `0 / 639,360` (`0.0`) | `84,196 / 6,393,600` | `2.8152214781584917e-08` | `kt03/S2_EXIT/sst_m` |

#### Re-scored rung-10 200-row state ladder by step (`kt = 1..10`)

| Step | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Unequal / Total | Max `\|diff\|` | Max normalized `\|diff\|` | First non-bit row | First `OVER_BAR` row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kt01` | `11` | `9` | `0` | `389,384 / 12,893,760` | `0.0` | `0.0` | `kt01/stage1/T` | `None` |
| `kt02` | `7` | `9` | `4` | `89,681 / 12,893,760` | `5.7983129408967216e-10` | `5.3579557457439364e-11` | `kt02/entry/T` | `kt02/stage3/T` |
| `kt03` | `1` | `0` | `19` | `202,604 / 12,893,760` | `2.7709248229257355e-09` | `2.0361313947731371e-10` | `kt03/entry/T` | `kt03/entry/T` |
| `kt04` | `0` | `0` | `20` | `269,589 / 12,893,760` | `4.1856225152514526e-09` | `2.0525656202541726e-10` | `kt04/entry/T` | `kt04/entry/T` |
| `kt05` | `0` | `0` | `20` | `366,313 / 12,893,760` | `4.7461057306463772e-09` | `3.4699908870270795e-10` | `kt05/entry/T` | `kt05/entry/T` |
| `kt06` | `0` | `0` | `20` | `439,156 / 12,893,760` | `4.8810733233040082e-09` | `6.8480669424619982e-10` | `kt06/entry/T` | `kt06/entry/T` |
| `kt07` | `0` | `0` | `20` | `561,336 / 12,893,760` | `7.3165864478141884e-09` | `6.8480669424619982e-10` | `kt07/entry/T` | `kt07/entry/T` |
| `kt08` | `0` | `0` | `20` | `675,167 / 12,893,760` | `1.1674536892769538e-08` | `1.6076197448443067e-09` | `kt08/entry/T` | `kt08/entry/T` |
| `kt09` | `0` | `0` | `20` | `878,221 / 12,893,760` | `1.6137548186634376e-08` | `2.0103533253419035e-09` | `kt09/entry/T` | `kt09/entry/T` |
| `kt10` | `0` | `0` | `20` | `1,045,169 / 12,893,760` | `2.3088674083737715e-08` | `2.0148460472540064e-09` | `kt10/entry/T` | `kt10/entry/T` |
| **Total** | **`19`** | **`18`** | **`163`** | **`4,916,620 / 128,937,600`** | **`2.3088674083737715e-08`** | **`2.0148460472540064e-09`** | **`kt01/stage1/T`** | **`kt02/stage3/T`** |

- independent review: operator

## Resume 12 — Rebase onto ORCA2 lane tip (`a91d673b6fcf`), `nemo_sh2_fold_top_v` verification, 10-step re-score (`PASS_S4_EXACT`), and landing battery

### (1) Rebase onto ORCA2 lane tip (`a91d673b6fcf`) and mechanical delta / gate accounting

Round 340 (`23` commits based on `d5d97cfc31218e12a59950e84f09bc9382255f37`,
pre-rebase tip preserved on backup branch `backup/r340_pre_rebase` at
`c81be0d554508b0dcffa927c300cff8766ec9498`) was rebased onto the ORCA2 lane tip
`a91d673b6fcff38b097f7c0fc8ddb009c05d7a32` (`98` commits ahead from Round 336,
landing the north-fold V-face mask row in `_tke_bottom_dirichlet` at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12466-12495`),
producing rebased tip `424890bbbc18d3e718d214d89f689e39b620a144` and re-anchored
commit `8a74ac5dd1697d36ecfaa264e081a751d77adcfd`. No model code, configuration
value, or test file was changed during or after the rebase.

Amendment (2026-10-29 provenance audit): the backup value above was a pre-rebase commit no longer reachable; the post-rebase landing commit is `424890bbbc18d3e718d214d89f689e39b620a144`.

1. **Conflict resolution and citation re-anchor (`1` conflict hunk, `1` shifted citation key)**:
   - **Conflict (`1` hunk)**: At commit `eb4909b171` (rebased as `b9dc379088`,
     step `1/23`), `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`
     conflicted at the opening of `CITATION_MAP = {`, where Round 336 and
     Round 340 both inserted their round entries. Resolved by keeping both
     sides' blocks verbatim in full.
   - **Re-anchored citation (`1` entry)**: Because Round 340 inserted
     `build_orca2_istate_for_mode` (`+77` lines) earlier in
     `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`, Round 336's
     citation of `nemo_sh2_fold_top_v=True,` shifted from line `2063` to
     `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2140`:

```text
packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2063
  -> packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2140 ('nemo_sh2_fold_top_v=True,')
```
2. **Mechanical non-gate delta proof**:
   - At the rebased tip `424890bbbc18` (immediately after `git rebase FETCH_HEAD`),
     comparing `git diff d5d97cfc3121..backup/r340_pre_rebase` against
     `git diff a91d673b6fcf..424890bbbc18`:
     - Across all `33` non-gate, non-shared-receipt files (`packages/`, `scripts/`,
       `tests/`, `PREREG_nemo_testcases_l4_orca2_round340_s4.md`, and Round 340's
       receipts), the raw `git diff` output (including `index` headers) is
       **100% byte-identical** (`0` differing bytes across `7,451` lines and
       `414,599` bytes; SHA-256
       `d4be84a1ba18a514095f527402057040180de83c7f05a7df2e28be6608ab86ea`).
     - Across all `35` non-gate files (including the two historical receipts
       `nemo_testcases_l2_gyre_phase3_round8_receipt.md` and
       `nemo_testcases_l4_orca2_round332_rescore_acquisition_receipt.md` where
       the lane and Round 340 re-anchored disjoint citations), the patch content
       (excluding only the two git blob `index` header lines on those two shared
       receipts) is **100% byte-identical** (`0` differing bytes across `7,460`
       lines and `416,833` bytes; SHA-256
       `928cd9f8365e23554e15438b38f9974a238a1447952ec3b8d870d3aaa29a0e93`).
     - Symmetrically, `git diff backup/r340_pre_rebase..424890bbbc18` vs
       `git diff d5d97cfc3121..a91d673b6fcf` outside the citation gate is
       **100% byte-identical** (`12,198` lines, `606,425` bytes, SHA-256
       `1cea014b48839ea17a590a202c287789b41f45e65a2df5b492a862f77188cac0`;
       raw `48`-file diff SHA-256
       `3f795b35dbc9523ba818d5d900e223916cb4dd66f2d68a4eea933a709b69228c`).
3. **Citation-gate line and key accounting**:
   - Across `base = d5d97cfc3121`, `lane = a91d673b6fcf`,
     `backup = c81be0d55450`, `rebased = 424890bbbc18`, and `HEAD = 8a74ac5dd169`:
     - `FILES` entries: `736` unique (`740` raw in `base`) + `34` added by lane +
       `31` added by Round 340 (`1` shared key) = `800` unique (`805` raw) in
       `HEAD` (`0` lane entries missing, `0` Round-340 entries missing).
     - `CITATION_MAP` keys: `2,371` (`base`) + `85` net new added by lane + `79`
       net new added by Round 340 = `2,535` keys in `HEAD` (`0` duplicate keys;
       `84 / 85` lane additions present verbatim and `1 / 85` re-anchored from
       line `2063` to line `2140` as shown above; `79 / 79` Round-340 additions present
       verbatim; all `25` lane re-anchors of base `ocean_model_latlon_cgrid.py` /
       `round331` keys and all `21` Round-340 re-anchors of base
       `nemo_testcase_recipe.py` / `ocean_pe_latlon_cgrid.py` /
       `shortwave_penetration.py` / `state.py` keys preserved).
     - Added diff lines (`git diff -U0`): at `424890bbbc18`, `378 / 378`
       lane-added lines and `358 / 358` Round-340-added lines in
       `nemo_testcase_receipt_citation_gate.py` are present verbatim (`377 / 378`
       and `358 / 358` at `8a74ac5dd169` after the single line `2063 -> 2140`
       re-anchor line).

### (2) Confirmation of `nemo_sh2_fold_top_v=True` on the standalone K0 card

Instantiating both `build_orca2_zps_card(deck_root)` and
`build_nemo_orca2_k0_ocean_card(deck_root)` on
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0` (Rule 10):

- `build_nemo_orca2_k0_ocean_card` in `packages/ocean/legoesm/ocean/bulk_flux_omip.py`
  constructs `ocean=build_orca2_zps_card(deck_root, initial_state=k0.initial_state)`,
  which sets `nemo_sh2_fold_top_v=True` at
  `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2140`.
- Verified on the instantiated `k0` card:
  - `k0.ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_sh2_fold_top_v == True`
  - `k0.ocean.recipe.model_config.physics.vertical_mixing.tke.bottom_tke_bc == True`
  - `k0.ocean.recipe.grid.fold.is_active == True`
  - `k0.ocean.recipe.z_coord.nemo_een_barotropic is not None == True`
  - `r10.recipe.model_config.physics.vertical_mixing.tke == k0.ocean.recipe.model_config.physics.vertical_mixing.tke == True`
- Thus the standalone K0 card already enables `nemo_sh2_fold_top_v=True` and
  executes Round 336's north-fold V-face mask restoration in
  `_tke_bottom_dirichlet`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12466-12495`,
  matching `src/OCE/DOM/dommsk.F90:146-172`,
  `ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:303-309`, and
  `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12881-12898`)
  with zero configuration changes.

### (3) Re-scored 10-step real ORCA2 K0 composed card (+ arm-B control) on `8a74ac5dd169`

Executed `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round340_score_composed_card.py`
on CPU (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`;
artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r340_resume12/score_composed_card.json`,
SHA-256 `fd278bc2ffd096ea0920c7f4bbde0f071629edb1a38a5521dfc3d0521d77376d`;
log `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r340_resume12/score_composed_card.log`,
SHA-256 `4d5582565b477158d171659776cc778239d7ed125a4ce7f8a23b6c8f06f1e6b6`):

- **Arm-B vs arm-A reproducibility control**: `PASS_REPRODUCIBILITY_EXACT`
  (`0 / 123,824,304` surface words unequal, `0 / 128,937,600` rung-10 state
  words unequal across `200 / 200` rows).
- **Overall composed-card status**: **`PASS_S4_EXACT`** (`failure_message = None`).
- **Overall surface score (`kt = 1..10`)**: **`PASS_S4_EXACT`**,
  **`0 / 123,824,304` unequal words (`0.0%`, down from `914,406 / 123,824,304` in
  Resume 11), `max |diff| = 0.0` (down from `3.1096305974642746e-08` in Resume 11)**.
  - **Every single step `kt = 1..10` and every single boundary family (`S0_ENTRY`,
    `S0_EXIT`, `S0_F`, `S1_ENTRY`, `S1_TURB`, `S1_FLUX`, `S1_EXIT`, `S3_ENTRY`,
    `S3_COVER_IIF`, `S3_FREEZE_IIF`, `S3_CLAMP_IIF`, `S3_EXIT_IIF`, `S3_EXIT`,
    `S2_ENTRY`, `S2_RNF`, `S2_SSR`, `S2_FWB`, `S2_LBC`, `S2_EXIT`) is 100%
    bit-exact (`0` unequal words across all wet and dry cells)**.
  - **First non-bit surface row**: `None`.
  - **First non-zero `max |diff|` surface row**: `None`.
- **Overall rung-10 200-row state ladder (`kt = 1..10`)**:
  **`88 BIT_EXACT`, `112 AT_BAR_NOT_EXACT`, `0 OVER_BAR`** (down from `163 OVER_BAR`
  in Resume 11), `1,221,696 / 128,937,600` unequal words (down from
  `4,916,620 / 128,937,600`), **`max |diff| = 1.1102230246251565e-16`** (down from
  `2.3088674083737715e-08`), **`max_normalized_max_abs = 3.677867469028952e-18 <= 1e-15`**
  (down from `2.0148460472540064e-09`).
  - **Old first `OVER_BAR` row (`kt02/stage3/T`)**: resolved from `OVER_BAR`
    (`216 / 799,200` words, `37` wet words, `max |diff| = 4.622280336263884e-11`)
    to **`AT_BAR_NOT_EXACT` with `0` unequal wet words (`140 / 799,200` dry
    `-0.0` vs `+0.0` words, `max |diff| = 0.0`, `normalized_max_abs = 0.0`)**;
    `kt02/stage3/S` is now **`BIT_EXACT` (`0 / 799,200`, `max |diff| = 0.0`)**,
    and `kt02/stage3/{u, v}` have `0` unequal wet words (`max |diff| = 0.0`).
  - **First non-bit rung-10 row**: `kt01/stage1/T` (`AT_BAR_NOT_EXACT`,
    `157,456 / 799,200` dry `-0.0` vs `+0.0` words from
    `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2881-2885`,
    `0` wet words unequal, `max |diff| = 0.0`, `normalized_max_abs = 0.0`).
  - **First non-zero `max |diff|` rung-10 row**: `kt07/stage2/T`
    (`AT_BAR_NOT_EXACT`, `10,213 / 799,200` unequal words, `max |diff| =
    5.551115123125783e-17`, `normalized_max_abs = 1.8412868400289552e-18 <= 1e-15`),
    followed only by `kt09/stage2/T` (`AT_BAR_NOT_EXACT`, `10,210 / 799,200`
    unequal words, `max |diff| = 1.1102230246251565e-16`, `normalized_max_abs =
    3.677867469028952e-18 <= 1e-15`); all other `198 / 200` rows have
    `max |diff| = 0.0`.
  - **First `OVER_BAR` rung-10 row**: **`None` (`0 / 200 OVER_BAR`)**.

#### Re-scored surface boundaries by step (`kt = 1..10`)

| Step | Unequal words | Total words | Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- |
| `kt01` | `0` | `18,035,664` | `0.0` | `None` |
| `kt02` | `0` | `10,842,528` | `0.0` | `None` |
| `kt03` | `0` | `12,894,000` | `0.0` | `None` |
| `kt04` | `0` | `10,842,528` | `0.0` | `None` |
| `kt05` | `0` | `12,894,000` | `0.0` | `None` |
| `kt06` | `0` | `10,842,528` | `0.0` | `None` |
| `kt07` | `0` | `12,894,000` | `0.0` | `None` |
| `kt08` | `0` | `10,842,528` | `0.0` | `None` |
| `kt09` | `0` | `12,894,000` | `0.0` | `None` |
| `kt10` | `0` | `10,842,528` | `0.0` | `None` |
| **Total** | **`0`** | **`123,824,304`** | **`0.0`** | **`None`** |

#### Re-scored surface boundaries by boundary family (`kt = 1..10`)

| Boundary family | `kt01` (`unequal / count`, `max \|diff\|`) | `kt02` (`unequal / count`, `max \|diff\|`) | `kt = 1..10` (`unequal / count`) | `kt = 1..10` Max `\|diff\|` | First non-bit row |
| --- | --- | --- | --- | --- | --- |
| `S0_ENTRY` | `0 / 0` (`0.0`) | `0 / 0` (`0.0`) | `0 / 0` | `0.0` | `None` |
| `S0_EXIT` | `0 / 186,480` (`0.0`) | `0 / 186,480` (`0.0`) | `0 / 1,864,800` | `0.0` | `None` |
| `S0_F` | `0 / 10,656,384` (`0.0`) | `0 / 5,035,008` (`0.0`) | `0 / 57,890,304` | `0.0` | `None` |
| `S1_ENTRY` | `0 / 186,480` (`0.0`) | `0 / 186,480` (`0.0`) | `0 / 1,864,800` | `0.0` | `None` |
| `S1_TURB` | `0 / 506,160` (`0.0`) | absent on even `kt` | `0 / 2,530,800` | `0.0` | `None` |
| `S1_FLUX` | `0 / 159,840` (`0.0`) | absent on even `kt` | `0 / 799,200` | `0.0` | `None` |
| `S1_EXIT` | `0 / 399,600` (`0.0`) | `0 / 399,600` (`0.0`) | `0 / 3,996,000` | `0.0` | `None` |
| `S3_ENTRY` | `0 / 399,600` (`0.0`) | `0 / 399,600` (`0.0`) | `0 / 3,996,000` | `0.0` | `None` |
| `S3_COVER_IIF` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `0 / 4,262,400` | `0.0` | `None` |
| `S3_FREEZE_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `0 / 2,264,400` | `0.0` | `None` |
| `S3_CLAMP_IIF` | `0 / 452,880` (`0.0`) | absent on even `kt` | `0 / 2,264,400` | `0.0` | `None` |
| `S3_EXIT_IIF` | `0 / 452,880` (`0.0`) | `0 / 452,880` (`0.0`) | `0 / 4,528,800` | `0.0` | `None` |
| `S3_EXIT` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `0 / 4,262,400` | `0.0` | `None` |
| `S2_ENTRY` | `0 / 426,240` (`0.0`) | `0 / 426,240` (`0.0`) | `0 / 4,262,400` | `0.0` | `None` |
| `S2_RNF` | `0 / 506,160` (`0.0`) | `0 / 506,160` (`0.0`) | `0 / 5,061,600` | `0.0` | `None` |
| `S2_SSR` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `0 / 5,860,800` | `0.0` | `None` |
| `S2_FWB` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `0 / 5,860,800` | `0.0` | `None` |
| `S2_LBC` | `0 / 586,080` (`0.0`) | `0 / 586,080` (`0.0`) | `0 / 5,860,800` | `0.0` | `None` |
| `S2_EXIT` | `0 / 639,360` (`0.0`) | `0 / 639,360` (`0.0`) | `0 / 6,393,600` | `0.0` | `None` |

#### Re-scored rung-10 200-row state ladder by step (`kt = 1..10`)

| Step | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Unequal / Total | Max `\|diff\|` | Max normalized `\|diff\|` | First non-bit row | First `OVER_BAR` row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kt01` | `11` | `9` | `0` | `389,384 / 12,893,760` | `0.0` | `0.0` | `kt01/stage1/T` | `None` |
| `kt02` | `8` | `12` | `0` | `89,009 / 12,893,760` | `0.0` | `0.0` | `kt02/entry/T` | `None` |
| `kt03` | `8` | `12` | `0` | `103,775 / 12,893,760` | `0.0` | `0.0` | `kt03/entry/T` | `None` |
| `kt04` | `8` | `12` | `0` | `81,297 / 12,893,760` | `0.0` | `0.0` | `kt04/entry/T` | `None` |
| `kt05` | `8` | `12` | `0` | `103,113 / 12,893,760` | `0.0` | `0.0` | `kt05/entry/T` | `None` |
| `kt06` | `8` | `12` | `0` | `83,299 / 12,893,760` | `0.0` | `0.0` | `kt06/entry/T` | `None` |
| `kt07` | `8` | `12` | `0` | `100,705 / 12,893,760` | `5.551115123125783e-17` | `1.8412868400289552e-18` | `kt07/entry/T` | `None` |
| `kt08` | `8` | `12` | `0` | `81,670 / 12,893,760` | `0.0` | `0.0` | `kt08/entry/T` | `None` |
| `kt09` | `9` | `11` | `0` | `106,701 / 12,893,760` | `1.1102230246251565e-16` | `3.677867469028952e-18` | `kt09/entry/T` | `None` |
| `kt10` | `12` | `8` | `0` | `82,743 / 12,893,760` | `0.0` | `0.0` | `kt10/entry/u` | `None` |
| **Total** | **`88`** | **`112`** | **`0`** | **`1,221,696 / 128,937,600`** | **`1.1102230246251565e-16`** | **`3.677867469028952e-18`** | **`kt01/stage1/T`** | **`None`** |

### (4) Full `land.sh` landing battery (`T4` pytest set + 30-day GPU DINO month gate)

Executed in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r340_resume12/`:

- **ORCA2 `land.sh` `T4` pytest battery** (`gate.log`, SHA-256
  `0a51a806b45072237a805b6cece18eb91b479b9a25fe8a2101c9a4ebf0d82c10`):
  `======================= 152 passed in 1079.48s (0:17:59) =======================`
- **30-day GPU DINO from-rest month gate** (`dino_month_gate.log`, SHA-256
  `6e121e0aa74be8e28f806cd0949fb6246de91e6f88bc9f6f2695fb0796c9a51f`;
  snapshot SHA-256
  `40a63d3a43a80600c740067fd3446bf428cd20e362a60987908c1a079a6e6bb5`;
  score JSON SHA-256
  `0e8c9a347c74db77971e8146474d11ba486acda33cd0b0985c6da0dda01113d5`):
  `DINO month gate: day-30 wet 3-D T rms vs NEMO kt=960: 2.056821683e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`
  (raw log pass line:
  `DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.056821683e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`).
- independent review: operator

## Verification

- Round-340 focused fast tests (`test_orca2_round340_standalone.py`,
  `test_surface_forcing_sign_convention.py`,
  `test_nemo_testcase_l4_orca2_round337_surface.py`,
  `test_nemo_card_opt_in_defaults.py`, and
  `test_lead_freeze_nemo_qlead.py`): all pass (`20 passed, 2 skipped` in
  `test_orca2_round340_standalone.py` on the rebased tree; `21 passed, 1 skipped`
  with `ORCA2_R340_REAL_GRADIENT=1`; `7 passed` in
  `test_nemo_testcase_l4_orca2_round337_surface.py`; `22 passed` in
  `test_nemo_card_opt_in_defaults.py`; `7 passed` in
  `test_lead_freeze_nemo_qlead.py`).
- M6 card-fingerprint gate (`nemo_testcase_l4_orca2_card_fingerprint.py`):
  `PASS` (`ORCA2-zps` SI3 card and all five phase-2 cards 100% bit-identical).
- Admitted arm-B vs arm-A reproducibility control: `PASS_REPRODUCIBILITY_EXACT`
  (`0 / 123,824,304` surface words and `0 / 128,937,600` rung-10 words unequal).
- Real 10-step CPU composed-card re-score (`nemo_testcase_l4_orca2_round340_score_composed_card.py`):
  `PASS_S4_EXACT` (`0 / 123,824,304` unequal surface words across all `10` steps
  and all `259` time-qualified boundaries, `max |diff| = 0.0`, first non-bit
  surface row `None`; `88 BIT_EXACT`, `112 AT_BAR_NOT_EXACT`, `0 OVER_BAR` across
  all `200` rung-10 state rows, `max |diff| = 1.1102230246251565e-16`,
  `max_normalized_max_abs = 3.677867469028952e-18 <= 1e-15`, first non-bit state
  row `kt01/stage1/T`, first `OVER_BAR` state row `None`).
- Full `land.sh` landing battery: `152 passed in 1079.48s (0:17:59)` on the
  ORCA2 `T4` pytest set and `PASS` on the 30-day GPU DINO from-rest month gate
  (`2.056821683e-03 K against bar 2.244317642e-03 K`).
- Per-receipt citation gate: PASS, 84 citations, zero failures, zero unmapped
  citations, zero map-audit failures, and all nine self-tests fired.
- Required receipt-citation pytest (`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`
  with `JAX_ENABLE_X64=1`): 17/17 PASS on the clean committed tree.
