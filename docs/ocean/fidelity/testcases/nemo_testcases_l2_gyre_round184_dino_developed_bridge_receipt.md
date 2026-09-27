# Round 184 receipt — DINO developed-state closed-face bridge repair

**Status: LANDED.**  The first non-finite statement in the unchanged DINO
developed-state control was not the step-2 `e3w` guard.  It was the
meridional QCO face-ratio evaluation in the first barotropic solve: 104 closed
V faces combined a zero weighted-height numerator with a non-finite storage
metric, producing `NaN` before the inverse depth was consumed.  Applying
NEMO's closed-boundary result to that ratio repairs both DINO cards without a
clip, floor, replacement constant, configuration choice, or state-schema
change.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round184.md`, commit
`3705d3cf3`.  Production repair: `93cae0fae`.  Final DINO measurement commit:
`d079793f9`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round184/`.

## Compiled-source statement and repair

The compiled DINO QCO routine computes the U/V ratios only on its declared
interior-plus-halo-minus-one range and spells the V expression as the two
area-weighted SSH terms times `r1_hv_0` and `r1_e1e2v` at
`DINO/BLD/ppsrc/nemo/domqco.f90:211-215`.  Its caller then applies NEMO's
boundary-link operation to both V arrays at
`DINO/BLD/ppsrc/nemo/domqco.f90:177-181`.

For this closed north/south boundary, the compiled boundary implementation
sets its default land value to zero at
`DINO/BLD/ppsrc/nemo/lbclnk.f90:1816-1820`, chooses that constant fill when
there is neither an MPI neighbour nor self-periodicity at
`DINO/BLD/ppsrc/nemo/lbclnk.f90:1866-1871`, and writes the zero into the
south/north halo at `DINO/BLD/ppsrc/nemo/lbclnk.f90:2130-2135`.  The first
consumer constructs live face depths and their reciprocals from these ratios
at `DINO/BLD/ppsrc/nemo/dynspg_ts.f90:484-487`.

legoESM already closed the corresponding stored V-face height before this
round, but then re-evaluated the ratio on the full padded array.  On DINO the
direct operands were:

| operand / output | count |
|---|---:|
| non-finite `r1_e1e2v` storage cells | 104 |
| zero weighted-SSH half sums | 428 |
| overlap, `0 * non-finite` | 104 |
| non-finite `r1_u_entry` | 0 |
| non-finite `r1_v_entry` before repair | 104 |

The repair applies the existing meridional closed-face helper to `r3_v`
immediately after the source-ordered ratio expression.  The arithmetic on
every NEMO-computed face is unchanged; only the storage boundary NEMO fills
after `dom_qco_r3c` is restored.  The after row has the same 104 non-finite
metric operands and 104 zero-times-nonfinite opportunities but zero
non-finite values in either entry inverse.

## First-failure walk and non-vacuity

The bridged day-180 entry is exact on wet T/S/u/v/SSH, and analytic surface
forcing preserves finite state.  On the unmodified implementation the first
barotropic entry reports the operand counts above; the returned first step
then contains non-finite values in all active cells: 9,920 SSH, 342,134 T,
342,134 S, 337,531 U, and 340,271 V.  The later `e3w` refusal therefore
observes already-destroyed state and is not the first statement.

The focused eager/JIT regression failed before the repair because all 16
closed V-face test cells were `NaN` and passes after the repair.  The
production plant restores one non-finite closed V-face after the repaired
statement.  It recreates the all-family active-state destruction, prints
`STATUS PLANT-FIRED`, and exits 1.  The admission-gate plant independently
changes one registered active T count from zero to one, prints the same fired
marker, and exits 1.  Neither plant can reach a success marker.

## DINO developed-state measurement

Both the landed `carried_step_entry` arm and the historical `recompute` arm
were measured from the same exact NEMO day-180 entry.  The gate registers all
five active-family finite counts and maxima at each of the first two steps;
every non-finite count is zero for both cards and both arms.  The step-2
maximum magnitudes are:

| card / arm | max SSH (m) | max T | max S | max U (m/s) | max V (m/s) |
|---|---:|---:|---:|---:|---:|
| `nemo_dino_kamm`, carried | 8.260484155594140e-1 | 2.597446902246825e1 | 3.685768317404258e1 | 7.517635241753557e-1 | 8.605875643536963e-1 |
| `nemo_dino_kamm`, recompute | 8.260484155594088e-1 | 2.597446902247861e1 | 3.685768317404536e1 | 7.517635241753577e-1 | 8.605875643529618e-1 |
| `nemo_dino_kamm_mlf`, carried | 8.263917501699938e-1 | 2.597413631757519e1 | 3.685764351587721e1 | 7.514047725522055e-1 | 8.594809786299208e-1 |
| `nemo_dino_kamm_mlf`, recompute | 8.263917502085033e-1 | 2.597413631804545e1 | 3.685764351581607e1 | 7.514047728787721e-1 | 8.594809787483791e-1 |

Both MLF arms also complete all 32 steps of day one with finite active state.
Every day-one surface row moved by the carried N2 selector is registered:

| MLF field | unequal cells | maximum absolute carried-minus-recompute move |
|---|---:|---:|
| SSH | 8,479 / 10,348 | 1.8164515495300293e-5 m |
| SST | 2,534 / 10,348 | 2.105712890625e-3 K |
| surface U | 8,177 / 10,348 | 5.127839744091034e-3 m/s |
| surface V | 8,160 / 10,348 | 7.992575410753489e-4 m/s |

Those rows measure the selector's execution and stability, not DINO accuracy:
this record has no matched NEMO one/two-step endpoint for an accuracy score.
The Euler-style `nemo_dino_kamm` card has a separate later instability in
both arms: its first four steps are finite, its fifth step enters the implicit
tracer solve with a positive `1.2861315034513562e-3 m` minimum live layer and
then returns 199 non-finite T/S, 400 U, and 364 V active cells.  This is after
the preregistered two-step bridge window, is shared by control and candidate,
and is not hidden as a successful day-one result.  The MLF card is the
complete day-one developed trajectory.

## GYRE and blast radius

The repair touches shared production code, so the base commit `7abb7a196` was
rerun with the current 954-row ladder instrument rather than compared to an
older report schema.  Before and after have zero moved cells on all 954 rows,
maximum worsening 0 ULP, no status changes, and first-over-bar kt=3
T/S/U/V/SSH.  Thus kt2 T/S/U/V remain respectively
`1.4210854715202004e-14`, `2.1316282072803006e-14`,
`8.326672684688674e-17`, and `9.71445146547012e-17`; kt3 T/S remain
`4.9403105251144552e-07` and `4.0085410546453204e-08`.

The from-rest month and year rows are byte-identical to Round 183:

| day | T3D RMS (K) |
|---:|---:|
| 30 | 2.3276772050683987e-06 |
| 60 | 1.4785764394709149e-05 |
| 90 | 1.6288518106294924e-05 |
| 120 | 1.0963874497720542e-04 |
| 180 | 6.113303379494912e-05 |
| 240 | 6.586171881479517e-05 |
| 300 | 5.474857690595152e-05 |
| 360 | 2.670992385329469e-03 |

The generic NEMO-GYRE three-step card has 0 moved rows.  The recipe-derived
execution census is:

| card | executes `nemo_ssh_avg` ratio | disposition |
|---|---|---|
| GYRE-zco | yes | 954-row ladder plus month/year, no move |
| DINO `nemo_dino_kamm` | yes | both arms finite through registered steps; later shared Euler instability disclosed |
| DINO `nemo_dino_kamm_mlf` | yes | both arms stable through day one |
| LOCK_EXCHANGE-zco | yes | focused tank and barotropic gates pass |
| OVERFLOW-zps | yes | focused tank and barotropic gates pass |
| ORCA2-zps | yes | UNMEASURED-with-spec; closed-face statement is shared, while the pending ORCA2 identity selector decisions remain outside this round |
| generic NEMO-GYRE | no | independent three-step gate passes with 0 moves |

## Frozen predictions, tests, citations, and review

Predictions 1-4 are confirmed.  Prediction 5 is confirmed by both plants.
Prediction 6 is confirmed on the registered finite/stability rows; it is not
inflated into an unavailable endpoint-accuracy claim.  Prediction 7 is
confirmed for GYRE, DINO from-rest, generic GYRE, LOCK_EXCHANGE, and OVERFLOW;
ORCA2 remains explicitly UNMEASURED-with-spec.  No configuration or
carried-state decision was made.

The focused battery reports **238 passed, 9 warnings in 682.62s**.  It covers
the repaired eager/JIT row, all DINO experiment tests, generic GYRE, the
barotropic restart/state gates, both tanks, and the Decision-43/45 controls.

The receipt citation gate and its shifted-citation plant are run at the final
receipt commit.  The separate read-only Codex review verdict is quoted below
after that pass.

| artifact | SHA-256 |
|---|---|
| `dino_bridge_gate.json` | `9a7c676ad83bdca8de413b668ff5dc985bf8ad83cf5cd73227b01f022d9ef3b2` |
| `dino_bridge_gate_plant.log` | `e303afe73a45397da31ae8acc27ce2bda745e877536263c2d1894135e0ba71c4` |
| `dino_closed_v_plant.log` | `e513daac70729b1fee35376562ffb7667668cd265c7ae55a71e0a1687ee34386` |
| `ladder_comparison_same_tip.json` | `197891522c1f91cb085faf3ad2732c72ae1735d0e1ff1fc4af3d615f32471bce` |
| `after_day_gap.json` | `19a4e244a3d3461fa8d8fa8108286c8f9480716418fc87be5b01cf112d8d6a0d` |
| `after_year_gap.json` | `50bbaabbcd243ba32f79b17fbea5ff249b2fbda7bd3e9a33836c08f563cf9647` |
| `generic_comparison.json` | `c6ed37544facd4afb3f26c464e09724de10d4fb641a6d6d229054d3f56e98195` |
| `focused_tests.log` | `33a24ce181248e857ab595393c550427dabc42709e58673bbee69fa87a91a869` |

## OPEN

The Decision-61 DINO-developed-state debt is closed for the two-step bridge
window and for a full MLF day.  The next magnitude round first re-ranks the
remaining day-240 owners on the Round-183 carried-N2 baseline (the old
vertical-diffusion ranking predates its 249.74x day-240 improvement), then
walks the largest surviving owner.  Separately registered but not promoted as
that target: diagnose the `nemo_dino_kamm` Euler card's shared fifth-step
implicit-solve instability if a multi-day developed Euler claim is needed.
