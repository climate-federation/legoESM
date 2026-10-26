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
The observer brackets those executing boundaries outside NEMO loops.
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
writers, compiles all five observed units at the production flags, obtains
nonzero and equal FP census counts (355, 30, 1,175, 59, and 50), and fires an
injected FP-operation plant. It executes a two-rank `-fcheck=bounds` fixture
through the actual patched `sbcmod`, including rank-distinct and negative-zero
data. The mixed full/interior allocations are taken from the real K0 ppsrc
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbc_oce.f90:195-216`).
It verifies both round-339 manifests and exercises real-ppsrc and whole-build
same-program plants, placement, namelist schema, checksum, generated-driver
syntax, refusal, retention, retained admission, and success controls. It reports
`ORCA2_ROUND340_S4_PREFLIGHT_READY`.

The legoESM card adds only an explicit K0 binding around the shared production
surface path. `NemoStandaloneK0Config` has no defaults and accepts exactly
`nn_ice=1`, drag off, `ln_mxl0=true`, `nn_mxlice=0`, and icebergs off. The
call accepts physical CORE/ice operands but no NEMO exchange record. The gate
reads the digest-pinned Decision-171 binary64 native-grid cover and exposes a
direct NetCDF CORE reader without fallback or regridding.

The composed scorer first enforces 0/n for every registered surface field,
then imports the unchanged round-1 rung-10 word scorer. One-ULP surface and
rung-10 plants both refuse. The focused test runs the whole S0->S1->S3->S2
production call and confirms finite N=1 forward and reverse gradients, with
reverse equal to forward.

## Checklist and choices

Checklist 1--15 is encoded in the preflight or run admission. Array writes use
declared full or interior bounds; observer calls are outside NEMO's loops; the
build uses `-O3 -funroll-all-loops -fno-tree-vectorize`; same-program proof
strips only five patches and two writer units; manifests use relative paths and
pin producer commit and executable; exact readers reject duplicate, missing,
truncated, trailing, wrong-rank, wrong-level, or nonfinite frames; refused
evidence is retained; the README gives move-aside retry instructions. The
actual NEMO arms and run admission remain unmeasured until the operator invokes
`--run`.

- ASKED: exact K0 acquisition, independent twin, source boundaries, existing
  ten-step state ladder, explicit K0 model card, exact surface-first scoring,
  finite N=1 gradients, and no author-run NEMO.
- UNASKED: none.
- independent review: operator (Claude)

## Verification

- Round-340 focused tests: 6/6 pass; round-337 production-surface regression:
  7/7 pass (13/13 combined).
- Acquisition preflight: READY; same-program, placement, FP-census, bounds,
  namelist, checksum, generated-flow, retained-admission, and generated-syntax
  controls pass or fire as intended.
- Per-receipt citation gate and required default gates: recorded after final
  clean-tree execution below.

NEMO acquisition is deliberately absent.
