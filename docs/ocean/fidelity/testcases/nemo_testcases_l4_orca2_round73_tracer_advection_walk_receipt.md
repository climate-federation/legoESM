# ORCA2 round 73 receipt — independent tracer-advection walk

Date: 2026-09-29

Base: `7f7f91cbe1d123b5afec7e6e1972109b4ce98431`

Disposition: **HELD; the first non-bit arithmetic is the stage-1 centered U
face flux, driven by an already non-bit metric transport**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and consumes the admitted exact kt=1 surface operands. No NEMO entry
field or transport is substituted into the production arm.

Sea ice remains out of scope. The ORCA2 card, selectors, 10,800 s time step,
carried state, and exact six-entry `unmeasured_features` tuple are unchanged.

## Answer

All four built cards resolve `tracer_advection="fct2"` and
`tracer_time_integrator="rk3_ws"`. The compiled dispatcher nevertheless runs
CEN2 at RK3 stages 1 and 2 and reserves FCT2 for stage 3: it clears
`ll_dofct` whenever `kstg /= 3`, then routes the FCT selection to
`tra_adv_cen` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv.f90:491-535`.
The statement is therefore executed by ORCA2, GYRE, OVERFLOW, and LOCK.

Independent kt=1 temperature is exact at entry on all 228,641 scored wet
rank-0 cells. The first non-bit operand inside the active tracer path is the
metric U transport `pU`: 221,640 / 251,670 active U faces differ, maximum
absolute transport difference `321212.60790659266 m3 s-1`. The V and W metric
transports are also non-bit (222,048 / 252,330 and 219,977 / 220,028).

The first non-bit arithmetic statement is the U face-flux assignment
`0.5 * pU * (T + T_east)` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:149-155`:
221,640 / 251,670 active faces differ, maximum absolute flux difference
`1230550.1692212257 degC m3 s-1`. The following V assignment differs on
222,048 / 252,330 faces, maximum `1111281.8978551854 degC m3 s-1`.

The full production boundary reproduces round 72 exactly: immediately after
advection, 228,641 / 228,641 wet temperature cells differ, maximum
`3.0869700763080185e-07 K`. The next source walk therefore belongs upstream
of line 153, in the construction of the stage-1 metric U transport; no tracer
arithmetic change is eligible to land from this measurement.

## Calibration and retained refutations

The first committed control predicted that recorded Kmm T plus recorded metric
transports would reproduce the after-advection accumulator. It refused:
228,641 / 228,641 cells remained non-bit, maximum
`1.4948368769042145e-09 K`. The prediction is **REFUTED**. The centered
horizontal and vertical divergence statements divide by live `e3t(Kmm)` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:157-161`
and `:202-228`; independent entry SSH makes that thickness non-bit.

Adding only the recorded external endpoint did not change that residual,
because it supplies `Kaa`, not the step-entry `Kmm` SSH. That second incomplete
control is also **REFUTED**. Bridging the admitted entry SSH as well as the
external endpoint and all three transports closes the after-advection
accumulator at 0 / 228,641 unequal. This complete-operand control validates
the source replay while leaving the reported production arm independent.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R73-P1 card execution census | **CONFIRMED** | All four cards resolve FCT2 plus RK3-WS; compiled stages 1-2 execute CEN2. |
| R73-P2 entry calibration | **CONFIRMED** | 0 / 228,641 wet T cells unequal. |
| R73-P3 first operand is pU | **CONFIRMED** | pU is the first scored non-bit operand, 221,640 / 251,670. |
| R73-P4 first arithmetic is U face flux | **CONFIRMED** | The line-153 row is first non-bit, 221,640 / 251,670. |
| R73-P5 recorded T plus transports close | **REFUTED** | Missing Kmm thickness leaves every wet accumulator cell non-bit at `1.4948368769042145e-09 K` maximum. |
| R73-P6 round-72 boundary reproduces | **CONFIRMED** | Count and maximum reproduce exactly. |
| R73-P7 disposition HELD | **CONFIRMED** | No model or configuration statement changed. |

Failed predictions remain **REFUTED**. Operand, arithmetic, calibration, and
boundary predicates are separate.

## Validation

- tracer-advection gate: `PASS_TRACER_ADVECTION_WALK`; production CPU/JIT/fp64/libm;
- card-scope, entry-ULP, recorded-replay-ULP, and round-72-boundary plants: all FIRED;
- focused round-71/72/73 and citation tests: `42 passed`;
- default citation gate: PASS, 274 citations, zero failures, zero unmapped,
  and zero map-audit failures; this receipt's gate: PASS, four citations,
  zero failures, zero unmapped, and zero map-audit failures; the shifted
  `traadv_cen` citation plant FIRES with `SYMBOL-NOT-AT-LINE`;
- shared-card battery: `160 passed` with nine dtype warnings; tank battery:
  `10 passed`;
- wide `tests/ocean/fidelity -n 12`: 2,057 collected and reached 98% with
  five registered failures and seven skips before reproducing round 72's
  no-summary wrapper stall. The five failing IDs were rerun serially and
  reproduced the standing signatures: round-51 private trace registry, SI3
  `MY_SRC` provenance, four worktree-stamp emitters, missing
  `hires_lane_surface` case-board row, and round-129 stale phase-3
  certification. No round-73 failure appeared in the wide run;
- separate read-only Codex review: **independent review unavailable
  in-sandbox**; `codex exec --sandbox read-only` exited 1 with
  `failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed. GYRE, DINO, and tank trajectories cannot move;
trajectory landing gates do not apply to this diagnostic round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round73/`.

## Scope ledger

ASKED: walk inside independent ORCA2's selected tracer-advection call, print
the cards that execute it, and name the first non-bit compiled arithmetic
statement.

UNASKED and unchanged: model arithmetic, configuration, selectors, thresholds,
stabilisers, carried state, sea ice, and the held shared tracer QCO/RK change.

## OPEN

1. Walk the stage-1 metric U transport construction upstream of CEN2, starting
   with the source-ordered corrected velocity and `e2u * e3u(Kmm)` product;
   preserve the independent label and score all four executing cards under
   their own ladder predicates before any shared landing.
2. Preserve the three calibration arms: transport-only, endpoint plus
   transport, and complete Kmm-entry plus endpoint plus transport.
3. The independent kt=1 SSH selector gap remains at `STOP_SELECTOR_GAP`; all
   six sea-ice selectors stay frozen.
4. The source-ordered tracer QCO/RK statement remains downstream and held by
   round 48's OVERFLOW rows.
