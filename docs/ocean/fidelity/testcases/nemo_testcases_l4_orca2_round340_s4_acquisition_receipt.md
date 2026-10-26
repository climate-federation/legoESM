# ORCA2 round 340 — exact standalone S4 acquisition tool and composed card

Date: 2026-10-26. Base `d5d97cfc3121`. Disposition:
**ACQUISITION_NEEDED**. No NEMO executable was run in this round.

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

Resume 2 closes Argon's four HOLD groups. The composed driver now carries
S0/S2 state and the preceding surface result across all ten calls, enforces
the K0 `nn_fsbc=2` cadence, updates the live SST slot after the ice-interface
clamp, and derives S0 means from the live ocean time levels, thickness stretch,
and step number. Chlorophyll, geothermal heat flux, and ice concentration are
mandatory: the first two come from the same digest-pinned deck files as K0,
and ice concentration is the same interpolated `sf_ice(1)%fnow` cover consumed
by the ice-interface heat-flux arm. A post-fix operator smoke ran the real
rung-10 `LatLonCGridOceanModel` through all ten `run_composed_card` steps on
CPU/fp64 in 785.47 seconds and found every final state leaf finite.

Surface and ice-interface records now use boundary-specific inventories: the
pre-initialization frame is header-only, later frames add only fields already
defined at that source boundary, and every full-grid value is sliced to the
owned `qsr` bounds before writing. The S0-F v3 writer likewise emits only the
owned `fnow` and associated `fdta` slabs, while its decoder retains v1/v2
compatibility. Scoring uses the exact time-qualified keys
emitted by the loader, including S0-F and ice-interface labels.

The bounds executable now links and calls a probe compiled into patched
`sbcmod` and executes both S1 sub-frame writers. The generated run driver
captures patched sources, parent manifest, and a producer manifest before
launch. Its refusal trap remains armed through admission and restart/twin
comparison. Preflight routes retained evidence through the actual
`--admit-existing` entry point and proves that either log containing
`E R R O R` is quarantined.

## Checklist and choices

The acquisition checklist is encoded in the preflight or run admission. Array writes use
declared full or interior bounds; observer calls are outside NEMO's loops; the
build uses `-O3 -funroll-all-loops -fno-tree-vectorize`; same-program proof
strips exactly seven patches and two writer units; manifests use relative paths and
pin producer commit and executable; exact readers reject duplicate, missing,
truncated, trailing, wrong-rank, wrong-level, or nonfinite frames; refused
evidence stays quarantined until build-source capture, manifest creation,
clean `STOP 0` log checks, restart comparison, and admission complete; the
README gives retained/in-place admission and move-aside retry instructions. The
actual NEMO arms and run admission remain unmeasured until the operator invokes
`--run`.

- ASKED: exact K0 acquisition, independent twin, source boundaries, existing
  ten-step state ladder, explicit K0 model card, exact surface-first scoring,
  finite N=1 gradients, and no author-run NEMO.
- UNASKED: none.
- independent review: Argon HOLD; items 1--4 repaired and re-preflighted in
  Resume 2; item 5 is parent-identical and intentionally out of scope

## Verification

- Round-340 focused tests: 8/8 pass; round-337 production-surface regression:
  7/7 pass (15/15 combined).
- Real rung-10 CPU smoke after the final ice-cover source correction: 1/1 pass
  in 785.47 seconds; ten actual model steps, 40 ladder checkpoints, finite
  final tree.
- Acquisition preflight: READY; same-program, placement, FP-census, bounds,
  namelist, checksum, generated-flow, `ocean.output` error quarantine,
  actual-route retained admission, and generated-syntax controls pass or fire
  as intended.
- Per-receipt citation gate: PASS, 14 citations, zero failures, zero unmapped
  citations, zero map-audit failures, and all nine self-tests fired.
- Required receipt-citation pytest: 17/17 PASS on the clean committed tree.
- Resume-2 pre-final default ORCA2 landing inventory: all 150 non-stamp tests
  pass; the two receipt-stamp tests correctly refuse the dirty pre-commit tree.
  The clean-tree inventory is rerun after this commit.
- Param-spec ratchet: 73 PASS, 1 inherited FAIL. The sole failure is
  `shortwave_penetration.py`'s unchanged non-float parameter-spec fields; it
  fails identically on parent `d5d97cfc31` and is deliberately not fixed in
  this round.

NEMO acquisition is deliberately absent.
