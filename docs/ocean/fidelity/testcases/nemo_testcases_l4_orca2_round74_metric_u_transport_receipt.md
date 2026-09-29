# ORCA2 round 74 receipt — independent metric-U transport walk

Date: 2026-09-29

Base: `083cefe542b2794f313470e93e44f26471e9eefc`

Disposition: **HELD; the first non-bit arithmetic is the hybrid barotropic
velocity correction, with two already non-bit operands**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. No NEMO entry field, external endpoint,
stage transport, or carried state is substituted into the production arm.

Sea ice remains out of scope. The card's six-entry `unmeasured_features`
tuple, selectors, 10,800 s step, thresholds, and carried state are unchanged.

## Answer

All four built cards resolve the shared RK3-WS momentum and tracer stage
program, so ORCA2, GYRE, OVERFLOW, and LOCK execute this stage-transport
construction.

The compiled selector fixes `n_baro_upd=np_HYB` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:45-49`.
The active branch first computes

`zub = un_adv * inverse_depth - uu_b(Kmm)`

and then

`zFu = e2u * e3u(Kmm) * (uu(Kmm) + zub*umask)`

at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284`.

On NEMO's actual rank-0 3-D active-U support, the source walk is:

| source-order row | unequal / count | maximum absolute difference | verdict |
|---|---:|---:|---|
| `un_adv` | 8,568 / 8,568 | 13.240378093773609 m3 s-1 | first non-bit primitive |
| inverse depth | 8,568 / 8,568 | 1.731223240825086e-05 m-1 | non-bit, same statement |
| `uu_b(Kmm)` | 0 / 8,568 | 0.0 m s-1 | bit-exact |
| `zub` | 8,568 / 8,568 | 0.01834375357001168 m s-1 | first non-bit arithmetic |
| `uu(Kmm)` | 0 / 226,236 | 0.0 m s-1 | bit-exact |
| `umask` | 0 / 399,600 | 0.0 | bit-exact, full owned array |
| corrected velocity | 226,236 / 226,236 | 0.01834375357001168 m s-1 | non-bit |
| `e2u` | 0 / 8,568 | 0.0 m | bit-exact |
| `e3u(Kmm)` | 226,236 / 226,236 | 0.0051604827037348144 m | non-bit |
| `e2u*e3u(Kmm)` | 226,236 / 226,236 | 1067.9735324520152 m2 | non-bit |
| `zFu` | 226,236 / 226,236 | 321212.60790659266 m3 s-1 | non-bit |

Thus `un_adv` is the first non-bit primitive in written order, and the
hybrid `zub` assignment is the first non-bit arithmetic statement. It has two
non-bit inputs, `un_adv` and inverse depth. No single input is eligible for a
fix until their cancelling-pair substitution is measured.

The source replay from NEMO's own operands is raw-bit exact on 226,236 U and
226,637 V active cells. Reconstructing live thickness, corrected velocity,
and final `zFu` through the production helpers also reproduces their exposed
production arrays on all 251,670 round-73 score-support cells. The instrument
therefore separates operand debt from transcription error.

## Round-73 support-label retraction

Round 73 called its 251,670-cell U comparison support "active U faces." That
label is **RETRACTED**. Its support builder broadcasts a 2-D U wet-column mask
through all 30 levels and then applies the rank-safe window; it is not NEMO's
3-D `umask`.

The exact census is:

| support | cells |
|---|---:|
| round-73 broadcast-column score support | 251,670 |
| NEMO 3-D active U support | 226,236 |
| round-73 support but not NEMO-active | 30,030 |
| NEMO-active but outside round-73 support | 4,596 |

The model's actual 3-D `umask` is nevertheless bit-exact to NEMO on all
399,600 owned values. Round 73's historical `zFu` tuple still reproduces
exactly on its original support: 221,640 / 251,670 unequal, maximum
`321212.60790659266 m3 s-1`. Its first-non-bit conclusion remains true, but
the word "active" and the physical cell count do not. This receipt uses the
recorded 3-D mask for every physical headline.

## Frozen prediction ledger

| prediction | verdict | deciding evidence |
|---|---|---|
| all four cards execute the shared stage transport | **CONFIRMED** | all resolve momentum/tracer `rk3_ws` |
| recorded-operand replay is exact | **CONFIRMED** | U 0/226,236 and V 0/226,637 unequal |
| `un_adv` is the first non-bit primitive | **CONFIRMED** | 8,568/8,568 unequal; first row |
| entry `uu(Kmm)` is exact | **CONFIRMED** | 0/226,236 unequal |
| entry-SSH consequence makes `e3u(Kmm)` non-bit | **CONFIRMED** | 226,236/226,236 unequal |
| round-73 `zFu` tuple reproduces | **CONFIRMED** | exact historical count and maximum on historical support |
| disposition is HELD | **CONFIRMED** | no model or configuration statement changed |
| initial `card-active` support label in the addendum | **REFUTED** | the builder broadcasts a 2-D mask; corrected above |

Failed support terminology remains explicitly **REFUTED**. Primitive input,
derived arithmetic, mask, record-replay, and historical-reproduction
predicates are separate.

## Validation

- metric-U gate: `PASS_METRIC_U_TRANSPORT_WALK`; production CPU/JIT/fp64/libm;
- card-scope, record-replay, exposure-calibration, first-classification, and
  round-73-boundary plants: all five FIRED;
- focused round-71/72/73/74 and citation tests: `50 passed`;
- default citation gate: PASS, 274 citations, zero failures, zero unmapped,
  and zero map-audit failures; this receipt's gate: PASS, two citations,
  zero failures, zero unmapped, and zero map-audit failures; the shifted
  hybrid-transport citation plant FIRES with `SYMBOL-NOT-AT-LINE`;
- shared-card battery: `160 passed` with nine dtype warnings;
- tank battery: `10 passed`;
- wide `tests/ocean/fidelity -n 12`: 2,065 collected; reached 99% with 2,041
  observed passes and five registered failures before reproducing the known
  no-summary wrapper stall. All eight round-74 tests passed. The five failing
  IDs were rerun serially and reproduced their standing signatures: SI3
  `MY_SRC` provenance, round-129 stale phase-3 certification, round-51 private
  trace registry, four worktree-stamp emitters, and missing
  `hires_lane_surface` case-board row;
- separate read-only Codex review: **independent review unavailable
  in-sandbox**; `codex exec --sandbox read-only` exited 1 with
  `failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed. GYRE, DINO, tank, and ORCA2 trajectories cannot
move; trajectory landing gates do not apply to this diagnostic round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round74/`.

## Scope ledger

ASKED: walk the independent stage-1 metric-U transport upstream of the first
non-bit CEN2 flux, name the first non-bit primitive and arithmetic statement,
and preserve exact mechanical gates.

UNASKED and unchanged: model arithmetic, configuration, selectors, thresholds,
stabilisers, carried state, sea ice, and the held shared tracer QCO/RK change.

No unasked choice was made. The physical score support is NEMO's recorded
3-D `umask`; round 73's legacy support is retained only for its explicitly
labelled reproduction row.

## OPEN

1. Preregister and measure the two-input cancelling pair at `zub`: substitute
   NEMO `un_adv`, NEMO inverse depth, and both together into the recorded
   statement. Score `zub`, corrected velocity, and `zFu` on the 3-D active
   support before proposing any landing.
2. If `un_adv` has the first non-cancelling ownership, walk its external-mode
   producer in compiled order. If inverse depth owns it, stop at the already
   recorded independent-entry SSH selector gap; do not change the six frozen
   sea-ice selectors.
3. Replace the phrase "active faces" with "broadcast-column score support"
   anywhere round 73's 251,670-cell tuple is carried forward. New physical
   scores use the recorded 3-D mask.
4. The independent kt=1 SSH selector gap remains `STOP_SELECTOR_GAP`; the held
   tracer QCO/RK statement remains downstream and unchanged.
