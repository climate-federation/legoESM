# ORCA2 round 243 — substep-6 V-transport operand split

Date: 2026-10-11. Frozen base: `92358803b`. Preregistration commit:
`51281d087`. Instrument commits: `d07c8bbcc`, `ffadf80e3`. Status:
**HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round243/`.

This round changes no package/model file, deck, selector, carried state,
stabiliser, sea-ice setting, or `unmeasured_features` entry. It measures the
round-242 OPEN item only. Results are separately labelled **independent** and
**given NEMO's entry**; the two labels produce identical numbers.

## Compiled statement and admitted record

OMT-4 reuses the `ORCA2_OMIP_L4_R210OMT1_P3` record binary and changes only
namelist switches. The executing compiled source builds the midpoint V
transport in this order:

1. AB3 extrapolation produces `va_e` from `vn_e`, `vb_e`, and `vbb_e` at
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:460-484`;
2. the live V depth `zhvp2_e` is built from `hv_0`, the midpoint SSH,
   horizontal areas, and `ssvmask` at
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519`;
3. `zhV = e1v * va_e * zhvp2_e` at
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:530-535`.

The admitted `NEMO_L2_BTADV_2` record contains `va_e`, `zhvp2_e`, and `zhV`
for all 65 external substeps. Its static `e1v` comes from the admitted
source-complete ordered stream. Both record twins are exact on every defined
payload. The candidate product replays
`barotropic_latlon_cgrid.py:789-809` and reproduces the production trace
bit-for-bit before any oracle comparison. The observer leaves every completed
state array bit-identical to the unobserved run. Execution is production JIT,
CPU, fp64, x64 enabled, and libm transcendental policy; every instantiated
operand prints `float64` and shape `(148, 90)`.

## Retraction: canonical fill is not a metric defect

The first measurement compared raw candidate `e1v` on all 13,320 rank-zero
slots to the recorder's canonical array and falsely reported 4,731 unequal
metrics with maximum 222,395.72857949696 m. This is **RETRACTED**. The run's
`mesh_mask_0000.nc` and production `grid.dx_v` carry positive `e1v` in those
slots, while the record writer canonicalises V fields at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:574-580` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:755-779`; its 4,731
zeros are fill outside the V-defined support. On the
self-described 8,589-cell nonzero support, production `e1v` is bit-exact:
0/8,589 unequal, maximum 0.0.

The corrected gate and its test are committed. The false artifacts remain
under `*_pre_support_RETRACTED.{json,log}`; they were not overwritten or used
for the attribution below.

## Frozen predictions

| prediction | result |
|---|---|
| P1: admitted, passive fp64 measurement agrees under both labels | **CONFIRMED** |
| P2: `e1v` is bit-exact | **CONFIRMED**, 0/8,589 unequal |
| P3: `va_e` is the first non-bit operand | **CONFIRMED**, 129/8,589 unequal |
| P4: the 29-cell interior product debt closes after substituting `va_e` | **REFUTED**, 19 non-bit / 17 over-floor interior cells remain |
| P4: all three recorded operands replay `zhV` bit-exactly | **CONFIRMED**, 0/8,589 unequal |
| P5: a single statement is eligible to land | **REFUTED**, `va_e` and `zhvp2_e` are a cancelling pair |
| P6: every control fires | **CONFIRMED** |

## Operand table — both labels

All counts below use the record's 8,589-cell V-defined support. “Over” means
strictly greater than the fixed 2e-10 floor.

| operand / product | unequal | over | max abs | argmax |
|---|---:|---:|---:|---|
| `e1v` | 0 | 0 | 0.0 m | — |
| `va_e` | 129 | 27 | 1.5051524978638986e-9 m/s | `[147,43]` |
| `zhvp2_e` | 120 | 73 | 1.9759179072480038e-2 m | `[147,49]` |
| candidate `zhV` | 129 | 124 | 2.690599228371866 m3/s | `[147,43]` |

The candidate product has 29 non-bit interior cells, 24 over the floor, with
interior maximum 8.189330110326409e-4 m3/s at `[144,45]`. This refines round
242's 29-cell census: 29 is the exact-bit count; 24 is the fixed-floor count.

One-variable substitutions prove neither upstream operand is independently a
closer:

| substitution | product unequal | product over | max abs |
|---|---:|---:|---:|
| recorded `e1v` only | 129 | 124 | 2.690599228371866 |
| recorded `va_e` only | 119 | 105 | 2.164499265840277 |
| recorded `zhvp2_e` only | 129 | 124 | 0.5260976927820593 |
| recorded `va_e` then `zhvp2_e` | 0 | 0 | 0.0 |

Thus the compiled multiplication itself is **DISCHARGED** given NEMO's own
operands: zero cells unequal. The first upstream non-bit statement is the AB3
midpoint-V producer at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:460-484`, but its
depth partner at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519` must be
walked with it. Landing either alone would split a measured cancelling pair
and is forbidden.

## Controls and tests

The label-coverage, source-order, one-ULP operand, replay-closure, and
passivity plants all exit red. Focused round-235/243 tests pass 9/9.

The required once-only `tests/ocean/fidelity -n 12` run reached 99% and then
lost its final worker without a terminal summary: 3,184 printed passes, seven
skips, four printed failures, and one unfinished node. The four failures
reproduce serially with their registered pre-existing messages:

1. GYRE round-129 spread-floor record: certified year harness moved after the
   registered members ran;
2. SI3 scalar-math provenance: `A MY_SRC is not verbatim`;
3. allow-dirty scope ratchet: the same 13 historical drivers;
4. worktree-stamp ratchet: the same 13 historical emitters.

The unfinished
`test_nemo_testcase_phase3_stage_sweep_gate.py::test_prediction_plant_is_fail_closed`
was run alone; its process again exited without a pytest verdict after
collection. It passed alone in round 242 at the same model tree. Round 243
does not claim a green full battery, and it does not rerun the full suite a
second time.

## Independent review

`codex exec --sandbox read-only` was attempted against `92358803b..HEAD` and
failed before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. **Independent review unavailable
in-sandbox.** The exact output is retained as `independent_review.log`; no
verdict is invented.

## OPEN

Walk the measured pair together at OMT-4 substep 6:

1. split the AB3 midpoint-V statement (`za1*vn_e + za2*vb_e + za3*vbb_e`) in
   compiled source order, using the current production D114 operand route;
2. split the midpoint V-depth statement across `hv_0`, midpoint SSH,
   `e1e2t/r1_e1e2v`, and `ssvmask`;
3. re-score the pair atomically against `zhV` and then `vn_adv`; no partial
   operand lands.

Both entry labels remain mandatory. No acquisition or configuration decision
is presently needed.
