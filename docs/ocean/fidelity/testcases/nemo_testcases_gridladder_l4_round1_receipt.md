# Grid ladder rung L4 — round 1 receipt (gridladder_r53, Codex)

Date: 2026-10-20

Starting parent: `7a31b67ef96e3954b64b204671942c84119fa923`

Status: **STOPPED_FOR_RECORD.**  L4 is implemented as certified G5 plus one
variable geothermal-bottom-heating switch.  The committed seven-item preflight
passes while the round-53 evidence parent is absent.  NEMO was not built or run
in this session, and no L4 fidelity score or verdict is reported.

- codex review: operator (Claude)
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- preregistration commit: `50ea9550d0`
- implementation/acquisition commit: `763a8625f3`

## 1. Survey verification

The executed ORCA2_OMIP_L4 deck states, individually, `ln_trabbc=.true.`,
`nn_geoflx=2`, `cn_dir='./'`, and the `sn_qgh` descriptor
`('geothermal_heating.nc', -12., 'heatflow', .false., .true., 'yearly', '',
'', '')` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:256,257,259,263`.  No tracer-advection
switch is needed for Decision 140: L4 is based on G5 and retains
`ln_traadv_OFF=.true.`.

The compiled constant initialization first forms `rho0_rcp=rho0*rcp` and then
its reciprocal at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:2504-2507`.  `tra_bbc_init` is the
initialization entry point
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:177`); `nn_geoflx=2` takes the
variable-file arm (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:228`), fills and
reads `sn_qgh`, and evaluates
`(r1_rho0_rcp * heatflow) * 1.e-3` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:239-243`.  At every call, NEMO adds
that `qgh_trd0` divided by the live Kmm bottom-cell thickness to temperature
Krhs (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:156-160`).  Stage 3 calls the
BBC after tracer LDF and before BBL and damping at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`; all three neighbors
remain off on the G5-based L4 card.

Both certified G5 binaries contain `__trabbc_MOD_tra_bbc` and
`__trabbc_MOD_tra_bbc_init`.  Their SHA-256 values are
`e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7`
(writer-free) and
`41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275`
(record).  The G5 and official ORCA2 compiled `trabbc.f90` files are bytewise
identical, SHA-256
`65e0d2b9302248caece34141d05e78b1982e4ac954771bb16370437de61ace3b`.
Thus the writer-free G5 binary can serve L4 without rebuild; only the new
WRITE-only bottom-trend record needs an instrumented binary.

The official local `ORCA2_ICE_v5.0.0.tar.gz` has SHA-256
`5d47eab63a54c0e3a41dc11aac823e447bc3a687126789863d8f86bab7e263ec`
and contains `ORCA2_ICE_v5.0.0/geothermal_heating.nc`.  Streaming that member
and hashing the extracted file both give
`e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb`
and 116,499 bytes.  The acquisition copies those bytes and rechecks both
properties.  No fetch or vendoring is needed.

## 2. Existing legoESM code and the one L4 extension

Before this round, the shared operator already selected one deepest wet cell
and evaluated `Q/(rho*c_sw*h)` in
`geothermal.py:126-173`.  The only state-level caller was a separate
forward-Euler post-step wrapper (`geothermal_apply.py:79-87`).  That was
physically equivalent at a whole-step level, but it did not reproduce NEMO's
stage-3 statement association or its Kmm thickness time level.

This round extends that module rather than adding another geothermal
implementation.  The new source-literal helper accepts the traced field in
the file's native mW/m2 units and materializes, with `nemo_source_round`, the
three verified NEMO statements: `rho0*rcp` then reciprocal, `*heatflow*1.e-3`,
then division by Kmm bottom thickness.  L4 adds one optional forcing member
(`geothermal_heat_flux_mwm2`) and one appended default-false config selector
(`rk3_stage3_geothermal_nemo_written`).  This is forcing, not new prognostic
model state.  The selected path refuses a missing forcing field, missing
stage-3 written-heat identity, or missing stored Kmm thickness.

`ORCA2L4-bbc` is built through the grid-ladder reader.  Its resolved model
configuration equals `ORCA2G5-ice` after exactly
`_replace(rk3_stage3_geothermal_nemo_written=True)`.  Its deck differs from
G5 only in the four `nambbc` assignments in section 1.  Exact x64 tests cover
eager and JIT execution, traced heat-flow input, gradients, one-hot bottom
placement, land zeros, the written result's bits, and a wiring-removal plant.

The selector is false on G1--G9 and on the default shared configuration; the
new forcing defaults to `None`, so it contributes no dynamic pytree leaf.
The branch is Python-static and is not present in flag-off JAXPRs.  Card
construction checks cover G1/G1b selector tables and G2--G9 resolved cards;
the unchanged G5 production-step controls also pass.  Therefore the existing
rungs retain their two-step values exactly; L4 alone selects the new branch.

## 3. Acquisition tool and preflight

The committed operator tool is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2bbc/run.sh`.
Default invocation is read-only; `--run` is the acquisition; and
`--admit-existing` validates an already produced record.  `--run` performs,
in order:

1. a two-step instrumented smoke;
2. a writer-free boundary run using the pinned G5 reference binary, to NEMO's
   own stop or 100 steps; and
3. three separately staged instrumented twins, capped at ten completed steps
   and rounded down to the `nn_fsbc=2` cadence.

The record is rank-zero, stream-unformatted, and one frame per completed
step.  Its 16-byte magic is `NEMO_L4_BBC__1` padded with two spaces; its eight
native int32 header values are version, kt, Kmm, Krhs, jpi, jpj, real bits,
and field count.  Three binary64 `(jpi,jpj)` fields follow: canonical wet
`qgh_trd0`, live Kmm bottom thickness, and bottom-cell temperature trend.
The validator derives the exact EOF, checks all values finite, demands
positive acting heat, requires dry slots to be canonical zero, and reconstructs
the trend exactly as `qgh_trd0/e3t`.  Admission requires exact inventory and
raw equality across all three twins.  Log validation parses `nn_geoflx` from
both deck and `ocean.output` as floats and compares the numeric values exactly.

The clean-tree default preflight at commit `763a8625f3` printed:

```text
L4_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=true
L4_PREFLIGHT_OK commit=763a8625f3331bef2c9018c8f31d868ec50c1b3e reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L4_BBC_R53
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round53/oracle_l4_r53
```

Its seven checks are: patched-Fortran syntax and continuation placement;
registered `makenemo -r ORCA2_ICE_PISCES`; rendered stop/stock divisibility;
writer/reader byte-layout round trip; complete frame inventory; viable output
path with absent evidence parent; and absent target/run directories.  Each
check also runs a planted violation through the same validator and proves a
nonzero rejection.

## 4. Tests, gates, and disposition

The focused x64 CPU suite passed 43 / 43: the geothermal unit tests, the L4
card/deck/record/acquisition tests, and the G5 card plus production top-drag
controls.  After the required line re-anchor, the direct gate on this receipt
found 9 citations, 0 unmapped citations, 0 citation failures, and 0 failing
map entries; all 9 non-vacuity controls fired.  Its unit suite passed 17 / 17,
and the private-import ratchet passed 6 / 6.

The prescribed default landing gate passed 142 / 142 in CPU x64: 17 citation,
6 private-import, 68 TKE-term, 25 recipe, 16 freshwater, 2 stage-face-mask,
and 8 prognostic-barotropic-state tests.  The combined process was split at
the documented per-process compiler limit; the remaining files passed in
fresh processes, including the 2 stage-face-mask tests in 614.65 seconds and
the 8 prognostic-state tests in 71.70 seconds.  The added L4 and geothermal
suites also passed separately, 10 / 10 and 14 / 14.  No NEMO model execution
occurred.

ASKED and completed: verify the oracle deck, executed sources, input tar
member, and compiled G5 route; preregister before implementation; build L4 as
G5 plus one switch; preserve every other selector; extend the existing
geothermal module; commit a smoke/boundary/three-twin acquisition with exact
records and binding plants; run the clean seven-item preflight; stop for the
operator record.

UNASKED and not done: fetch or vendor data; add prognostic state; merge lane B;
select G9; execute NEMO; score L4; relax any bar; publish or push.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r53-27712084
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r53-27712084/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2bbc/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — preflight 7/7 and plants 7/7 pass; operator must run the committed acquisition before any score
