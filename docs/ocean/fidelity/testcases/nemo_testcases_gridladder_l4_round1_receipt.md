# Grid ladder rung L4 — round 1 receipt (gridladder_r53, Codex)

Date: 2026-10-20

Starting parent: `7a31b67ef96e3954b64b204671942c84119fa923`

Status: **L4 CERTIFIED.**  L4 is certified G5 plus one variable
geothermal-bottom-heating switch.  The operator-produced ten-step record was
admitted, then every available INDEPENDENT, GIVEN-NEMO-ENTRY, full-depth,
stage, wet-cell and dry-cell row was scored.  Every row is bit-identical; no
statement owns an over-bar row and no corrective physics change was needed.

- codex review: operator (Claude)
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- preregistration commit: `50ea9550d0`
- implementation/acquisition commit: `763a8625f3`
- operator pipefail correction commit: `1bd536e885`

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

## 5. Admitted operator record

The operator ran the committed acquisition after the `nm | grep -q` pipefail
preflight defect was corrected at `1bd536e885`; this scoring session did not
build or run NEMO.  Read-only admission printed:

```text
L4_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=false
L4_GEOTHERMAL_RECORDS_PASS steps=1..10 sha256=0b184a56aca2c3c7c871cc545a6df057f7318c09d2695f8c412ac34712c93ab4
L4_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round53/oracle_l4_r53
```

The manifest's producer-list SHA-256 is
`d666a29a4accc44ec6acf6582d2e15fadd81cd8b6f18e27a771c33039e69394d`.
All three geothermal streams are byte-identical; their admitted payload hash
is the hash printed above.  The ordinary surface, stage, barotropic and restart
payloads passed the acquisition's pre-existing G5 identity checks.

## 6. Both-label score, full depth, wet and dry

The shared multilevel scorer was extended to register `ORCA2L4-bbc` and to
attach the native `heatflow` field to the existing G5 surface-forcing object.
It was not forked.  Restarts exist after kt 2--10, not kt 1; therefore full
depth is scored wherever the record permits it, while the kt 1 stage record
supplies all-level stage coverage.  GIVEN-NEMO-ENTRY has no synthetic kt 2
entry: its kt 2 result is the two-step output from kt 1, and direct restart
entries cover kt 3--10.

| label / record | coverage | exact rows | at-bar-not-exact | over bar |
|---|---:|---:|---:|---:|
| INDEPENDENT surface (ssh, uu_b, vv_b, u, v, T, S) | kt 1--10 | 70 / 70 | 0 | 0 |
| INDEPENDENT full/partial/dry depth summaries (u, v, T, S) | restarts kt 2--10 | 36 / 36 | 0 | 0 |
| GIVEN-NEMO-ENTRY direct + two-step surface | outputs kt 1--10 | 119 / 119 | 0 | 0 |
| GIVEN-NEMO-ENTRY full/partial/dry depth summaries | every available direct/two-step restart row | 64 / 64 | 0 | 0 |
| kt 1 stages 1--3 surface | 21 rows | 21 / 21 | 0 | 0 |
| kt 1 stages 1--3 full/partial/dry depth | 12 summaries | 12 / 12 | 0 | 0 |

Every maximum absolute difference is exactly `0.0`; every unequal-cell count,
including every dry-cell count, is zero.  There are no flushed-subnormal
exceptions.  Thus there is no first over-bar row and no NEMO statement to fix.
The score JSON SHA-256 values are:

- INDEPENDENT: `b3eb42c06149309fc5f6cfd6228a9194a373b5242f4bb4da03e0f6f095b4c587`;
- GIVEN-NEMO-ENTRY: `993abdbb8dea04a862c2d0d1c5438b6e4996f283d8fe442fcd6e0bdf8e0c56ee`;
- kt 1 stages: `cd454ab5899569ebd8202a13f2fae0e92a8db7a99f6a540bfd2c068332216abd`.

## 7. Geothermal trend, bit for bit

At every kt 1--10, a JIT-compiled fp64 call to the production
bottom-heating operator receives traced `e3t_Kmm` and `heatflow_mwm2` inputs.
It is compared directly with NEMO's recorded result of the cited
`qgh_trd0/e3t(Kmm)` statement at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:156-160`.

Each step has 16,433 wet bottom columns: 16,428 carry nonzero heating and five
have zero input heat flow.  For each of the ten steps, the result is 0 unequal
of 16,433 wet cells, maximum difference exactly `0.0`, with 0 noncanonical dry
cells.  The ten-row JSON SHA-256 is
`06389175a92a3cdfd0d1544c78f3b765df798a26e54a55824d79e7c315375fdf`.

Two non-vacuity controls fired.  Removing the selector from the direct trend
comparison produced 16,428 unequal wet cells and exited 1.  Removing it from
the full production step made kt 1 temperature DEBT: 249 unequal cells,
maximum `2.769e-11` K, and exit 1; that plant JSON SHA-256 is
`6919bd98985099d127dd9e7d8c6b3eaedde86de545cb1221d9943e4f957efce1`.

## 8. Unchanged-card controls and round-53 code

The implementation commit and this scoring run have the same numerical
production package files; the base `packages` tree was
`7a1a6409ea80b0320a56a338176fae65605e1bf3`.  The only package edit in this
resume is a selector-census assertion.  It now pins G9's
`rk3_stage_transport_stored_r3uv` flag, closing the standing round-52 nit; it
does not enter any model JAXPR.  Consequently the previously gated two-step
values of G1--G9 and the flag-off JAXPRs of full ORCA2, DINO, GYRE, SMT,
VORTEX, OMT and rung 0 are byte-for-byte unchanged.  L4 is the only card whose
forcing reader attaches `geothermal_heat_flux_mwm2`, and the selected field is
checked exactly against the official netCDF variable.

The focused L4/G9 x64 CPU suite passed 32 / 32.  The exact record test is JIT
compiled with both numeric inputs traced, and its wiring-removal plant fires.
After the scoring commit, the receipt-specific citation gate found 9 citations,
0 unmapped, 0 failures and 0 failing map entries; all nine self-test plants
fired.  The default x64 CPU gate passed 142 / 142 when split at the documented
compiler limit: citation 17, private imports 6, TKE/recipe/freshwater 109,
stage-face-mask 2, and prognostic barotropic state 8.  The separately required
private-import ratchet therefore passed 6 / 6.
The required review disposition remains `codex review: operator`; no local
Codex review was run.

ASKED and completed: verify and admit the operator record; score both labels at
every recorded output, full depth wherever a restart exists, the kt 1 stage
record, and wet/dry cells; compare the geothermal trend bit for bit; prove the
score fails without its wiring; preserve every other card; pin G9's open
stored-r3u/r3v selector census; and certify L4 because every row is exact.

UNASKED and not done: fetch or vendor data; add prognostic state; merge lane B;
select G9 for L4; execute NEMO; relax any bar; publish or push.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r53/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L4 CERTIFIED: every admitted trajectory, full-depth, stage, wet/dry and geothermal-trend row is bit-identical
