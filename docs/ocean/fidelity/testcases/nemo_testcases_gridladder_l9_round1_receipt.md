# Grid ladder rung L9 — round 1 build receipt (gridladder_r57, Codex)

Date: 2026-10-23

Starting commit: `bd83f3fd1eeb`

Preregistration commit: `a174af33e1`

Implementation/acquisition commit: `e4f204272d`

Receipt/citation commits: `a27eba2567`, `6afb3af28d`

Status: **STOPPED FOR RECORD.** L9 is designed, implemented, guarded, tested,
and has a clean passing acquisition preflight. NEMO was not built or run, no
record was admitted, no trajectory was scored, and L9 is not certified.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not performed
- scoring: not performed; no L9 record exists

## 1. Running NEMO order

ORCA2 computes the standard-isoneutral slope and then refreshes its
time-varying eddy coefficient before the Runge--Kutta stages
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:184-194`). Stage 3 next forms
the ordinary U/V/W tracer transport and calls `tra_adv_trp`
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:568-582`).

Inside `tra_adv_trp`, after the bottom transport slots are zeroed, stage 3
alone calls `ldf_eiv_trp`
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:248-265`). The RK3 EIV arm
builds its U/V streamfunctions and adds their vertical differences to the
ordinary transports (`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:938-955`).
`tra_adv_trp` then recomputes WZV and W transport from that augmented pair
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:267-312`).

Only then does the stage program clear `Krhs`, call stage-3 FCT, and add SBC
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:600-645`). The remaining
calls are QSR, isoneutral LDF, geothermal BBC, BBL, damping, and implicit ZDF
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:700-749`). The frozen L9
order is therefore:

```text
ordinary U/V transport -> EIV U/V addition -> recomputed W transport ->
zero Krhs -> FCT advection -> SBC -> QSR -> isoneutral LDF -> geothermal BBC
-> diffusive BBL -> interior damping -> implicit ZDF
```

ORCA2 selects diffusive BBL option 1, advective BBL option 0,
`rn_ahtbbl=1000`, and `rn_gambbl=10`
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:265-272`). The running BBL branch
therefore calls the diffusive arm, skips the advective arm, and adds its
bottom-cell result to the accumulated `Krhs`
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:182-202`,
`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:249-264`). It cannot modify or
reorder the earlier EIV-augmented transport.

## 2. Candidate, route, and refusals

`ORCA2L9-eiv-krhs` is the L8 deck plus exactly the five ORCA2 L3 EIV keys:
`ln_ldfeiv=.true.`, `nn_aei_ijk_t=21`, `rn_Ue=0.03`, `rn_Le=200000`, and
`ln_eke_equ=.false.`. The recipe assigns L9 all L8 feature groups and the L3
EIV group, while every prior card keeps the default-false L9 selector
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:4078-4087`,
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:4961-4994`).
The selector itself is appended with default `False`, preserving positional
callers and all existing defaults
(`packages/ocean/legoesm/ocean/state.py:3422-3429`).

The former L8 refusal is relaxed only for L9's complete L3 selection:
`kappa_GM=3000`, Treguier enabled with `aei0=3000` and `kappa_min=0`,
bolus-through-FCT, face averaging, `nemo_left`, and `nemo_forward_exact`.
Partial EIV, missing L8, missing Krhs routing, and any other L8+EIV composition
remain refused
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4130-4218`).

Decision 147 is preserved independently of L9. Construction validation,
every eager `step()` call, and the live inner EIV path all refuse staged EIV
without the stage-3 Krhs route
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4040-4054`,
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10527-10551`,
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:15020-15028`).

On L9, the stage-3 transport helper performs only the two NEMO-ordered U/V
additions. The existing W recomputation then consumes those exact augmented
operands before the existing L8 Krhs walker executes
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:212-215`,
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10541-10572`).
No division, square root, new forward branch, or gradient-only guard was
introduced.

## 3. Bit-position and production controls

The x64 JIT position control uses INEXACT binary64 operands. EIV maps
`0x1.64d6a9b31f28fp+31 + -0x1.921fb54442d18p+29` to
`0x1.004ebc620e749p+31`, and
`-0x1.3bd3cc9be45dep+28 + 0x1.0c6f7a0b5ed8dp+30` to
`0x1.7af50dc8cb82bp+29`. The seven Krhs operand/accumulated pairs are:

| position | operand | accumulated word |
|---|---|---|
| advection | `0x1.8484bbe636a7cp+25` | `0x1.8484bbe636a7cp+25` |
| SBC | `0x1.80c4d885778a5p+25` | `0x1.82a4ca35d7190p+26` |
| QSR | `-0x1.e4576fad0145bp+34` | `-0x1.e2d4cae2cb6e9p+34` |
| LDF | `-0x1.fbbb660e98d04p+8` | `-0x1.e2d4cb61ba481p+34` |
| BBC | `0x1.99a22105eba7ep+34` | `-0x1.24caa96f3a80cp+32` |
| BBL | `0x1.792a95b49c117p+27` | `-0x1.190154c195a03p+32` |
| DMP | `-0x1.0452f7986b12bp+33` | `-0x1.90d3a1f935e2cp+33` |

Temperature and salinity are pinned bit for bit at every boundary. Each of
the six adjacent-call swaps changes a displaced boundary and fails. Four
selector plants fail, including a partial face-averaging selection and the
Decision 147 route removal. The same test file proves ordinary and
gradient-transformed forward bits identical and the gradient finite
(`tests/ocean/fidelity/test_nemo_testcase_l9_orca2eiv_krhs.py:91-189`).

The full one-step ORCA2 production test observes the selected live path. Its
call list is exactly `["eiv", "krhs"]`; produced temperature and salinity are
finite. It then mutates the already-constructed model's Krhs selector and
proves the eager Decision 147 refusal still fires before cached execution
(`tests/ocean/fidelity/test_nemo_testcase_l9_orca2eiv_krhs.py:192-245`).

Final focused results under CPU, x64, and the landing `PYTHONPATH` were 6
passed / 1 deselected in 7.50 seconds for the fast section and 1 passed / 6
deselected in 204.47 seconds for the live section. Before the final guard was
added, the same live route passed in 189.91 seconds. The focused inherited
recipe/L8/L3 regression passed 32 / 32 selected tests in 24.68 seconds; one
unrelated existing GYRE message-regex test was deselected after the identical
failure was reproduced at the starting commit. No production result or
selector was changed to hide that pre-existing mismatch.

## 4. Acquisition and preflight

The committed launcher defaults to read-only `--preflight-only`; `--run`
requires `NEMO_OPERATOR_ACK=L9_R57_OPERATOR`. Its schedule is a 2-step acting
smoke, a writer-free 100-step reference boundary, and three independently
staged twins ending at one even step between 2 and 10. `check_acting` is its
own statement on the smoke only
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l9_orca2eiv_krhs/run.sh:310-325`).

The combined reader requires every complete L8 stage boundary for all smoke
steps and L3's EIV transport witness at kt 1. Both observers carry full
`jpi*jpj*jpk` arrays, canonicalize cells not defined by NEMO, and are installed
outside XIOS-false branches. The preflight compiles a real gfortran writer
covering both exact streams and proves exact EOF rejection.

The seven-part preflight checks all 4 patched Fortran sources, a continued-line
instrumentation plant, a live moved-call plant, the listed reference, every
possible clock deck, both binary layouts, writer completeness, whitespace-
tolerant selections against one real committed `ocean.output` excerpt and one
stamped L9 deck, smoke-only acting scope, output viability, and fresh target
and run paths. Every control has a planted failure. Scratch is created inside
the repository and removed by an `EXIT` trap
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l9_orca2eiv_krhs/run.sh:115-239`).

The first preflight found and refused a cleanup lifetime defect after printing
the seven check results: function-local `scratch` was unbound at process exit.
The path was moved to script scope, the abandoned scratch was removed, and the
complete clean-tree preflight was rerun. It exited 0 and left no scratch:

```text
L9_RECORD_SELFTEST_OK combined_layout=true truncation_refused=true
L9_PREFLIGHT_7_OF_7_PASS plants=all_refused compiled_gfortran_layout=true patched_syntax=4 real_log=true stamped_deck=true
WOULD_RUN smoke -> writer-free boundary -> three twins into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round57/oracle_l9_r57
PREFLIGHT_EXIT=0
```

No NEMO target, run directory, record, or score was created.

## 5. Landing gates

The receipt-specific citation gate found all 21 L9 citations, with 0 unmapped
citations, 0 endpoint failures, and 0 global map-audit failures; all 9
self-test controls fired. Shifting the stage-order citation by 2 lines
produced `SYMBOL-NOT-AT-LINE` and exited 1. The mandatory citation-gate pytest
file passed 17 / 17 in 5.58 seconds.

The exact ORCA2 default gate from `claude_rounds/land.sh` passed 145 / 145 in
996.24 seconds (16 minutes 36 seconds) under CPU, x64, and the prescribed
landing package path. That total includes the private-import ratchet at 6 / 6
and both ORCA2 parallel-receipt checks. The private-import ratchet was also
rerun alone and passed 6 / 6 in 4.97 seconds. No gate was relaxed, skipped, or
given a longer timeout.

## 6. ASKED / UNASKED

ASKED and completed: read the executed Oracle path with L8 and L3 selected;
state and cite the exact order; build L9 as L8 plus ORCA2's complete L3 EIV;
relax only that guarded composition while retaining Decision 147; pin every
ordered position with INEXACT one-step controls and moved-call plants; preserve
forward/gradient behavior; commit the ladder-standard acquisition; run its
read-only preflight; and stop at the acquisition boundary.

UNASKED and not done: build or run NEMO; admit or score a record; certify L9;
change any other configuration, fidelity bar, existing card, or two-step
value; push; post to GitHub; or perform an author review.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r57-37096084
ACQUISITION_NEEDED: NEMO_OPERATOR_ACK=L9_R57_OPERATOR /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r57-37096084/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l9_orca2eiv_krhs/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L9 is built and preflighted; operator acquisition is required before any score or certification
