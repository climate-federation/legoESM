# Grid ladder rung L8 — round 1 design/build receipt (gridladder_r56, Codex)

Date: 2026-10-22

Starting commit: `f32c3be7ac9d`

Preregistration commit: `12483a8c59`

Implementation/acquisition commits: `ce9636bc1f`, `28fefb1eaa`,
`bf01715c57`, `d710262489`, `98feff346e`

Receipt/citation re-anchor commit: `e111ccce8a`

Status: **ACQUISITION_NEEDED.**  This round designed and built L8 and ran its
committed preflight only.  It did not build or execute NEMO and did not score
or certify L8.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not run
- scoring: not run

## 1. Oracle statement order and selected BBL arm

The running stage-3 tracer path clears `Krhs`, calls advection, and then calls
the surface boundary condition (`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:600-645`).
After the stage-3 branch begins, it calls QSR, LDF, BBC, BBL, and DMP in that
order before ZDF (`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:700-749`).
The exact L8 write order is therefore:

```text
zero Krhs -> tra_adv -> tra_sbc_RK3 -> tra_qsr -> tra_ldf ->
tra_bbc -> tra_bbl -> tra_dmp -> tra_zdf
```

ORCA2 selects FCT order 2 horizontally and vertically, standard isoneutral
Laplacian diffusion with MSC, the variable geothermal source, and full-column
damping (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:260-307`,
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:325-329`).  Its BBL is diffusive option
1 with advective option 0, `rn_ahtbbl=1000`, and `rn_gambbl=10`
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:265-273`).

The selected `tra_bbl` arm calls only `tra_bbl_dif`; the separate advective arm
is guarded by `nn_bbl_adv /= 0`
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:176-202`).  The diffusive
routine adds its bottom-cell tendency directly to `Krhs`
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:242-268`).  Consequently L8's
BBL has no effect on the `zFu/zFv/zFw` operands already passed to `tra_adv`;
its effect is the later BBL `Krhs` write between BBC and DMP.

## 2. Built card and guarded production path

`ORCA2L8-krhs` inherits every G9 selector and adds the already-certified L1
diffusive BBL, L2 standard isoneutral diffusion, L4b geothermal source, and L6
interior damping.  The recipe carries L8 through G9's table values while the
five physical feature sets are explicit
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:4952-4986`,
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:5105-5134`).

The new default-false selector is accepted only with FCT, the stored stage-3
thickness/source selectors, the exact diffusive BBL constants and guards, the
certified L2 operator, geothermal-after-LDF, and damping-after-geothermal.
Every non-L8 combination retains the old BBL refusal
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4088-4163`).

At stage 3, L8 separates G9's SBC and QSR groups, obtains isolated LDF, BBC,
and DMP increments from the existing shared kernels, and leaves every non-L8
term tuple on its prior route.  The BBL kernel receives a zero accumulator for
L8, so the final walker alone owns the BBL position
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2738-2767`).
The final production call materializes all seven boundaries and then forms the
existing NEMO-written ZDF content
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2868-2904`).

The ordered walker uses only rounded additions, with no new division, square
root, or guard.  Its JIT gradient test is finite.  A full ORCA2 one-step L8
model run also completed with finite temperature and salinity.

## 3. Bit-position controls

The preregistered INEXACT binary64 operands and expected boundary words are:

| boundary | operand | expected accumulated word |
|---|---|---|
| advection | `0x1.8484bbe636a7cp+25` | `0x1.8484bbe636a7cp+25` |
| SBC | `0x1.80c4d885778a5p+25` | `0x1.82a4ca35d7190p+26` |
| QSR | `-0x1.e4576fad0145bp+34` | `-0x1.e2d4cae2cb6e9p+34` |
| LDF | `-0x1.fbbb660e98d04p+8` | `-0x1.e2d4cb61ba481p+34` |
| BBC | `0x1.99a22105eba7ep+34` | `-0x1.24caa96f3a80cp+32` |
| BBL | `0x1.792a95b49c117p+27` | `-0x1.190154c195a03p+32` |
| DMP | `-0x1.0452f7986b12bp+33` | `-0x1.90d3a1f935e2cp+33` |

The test checks every temperature and salinity boundary bit for bit under JIT,
not merely the final sum.  Its planted LDF/BBC move and BBL/DMP move both fire
(`tests/ocean/fidelity/test_nemo_testcase_l8_orca2krhs.py:80-114`).  Card
plants separately remove four required selectors and are rejected; the
ordinary one-step test exercises the complete selected route
(`tests/ocean/fidelity/test_nemo_testcase_l8_orca2krhs.py:151-200`).

The final focused L8 run passed 6 / 6 tests in 114.99 seconds under CPU and
x64.  The earlier 40-test combined G9/L4b/L6/L8 regression attempt completed
the L4b and L6 sections, found one expected-set assertion that needed L8 added
as a G9 composition, and later hit the documented per-process compiler limit.
After that test correction, the two isolated heavy G9 controls passed in
96.01 and 167.07 seconds.  No inherited production selector was changed.

## 4. Acquisition boundary and preflight result

The committed acquisition entry point defaults to `--preflight-only` and has
separate `--run` and `--admit-existing` modes.  `--run` requires
`NEMO_OPERATOR_ACK=L8_R56_OPERATOR`.  Its schedule is a 2-step acting smoke,
a writer-free reference boundary, and three independently staged twins at an
even endpoint no greater than 10 steps.  `check_acting` is a standalone call
on the smoke only (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l8_orca2krhs/run.sh:201-215`).

The stage observer writes full `jpi x jpj x jpk` canonical T arrays after
zero, advection, SBC, QSR, LDF, BBC, BBL, and DMP.  The inherited canonicalizer
zeros every unowned, halo, dry, and dummy cell.  The reader requires the exact
16-byte magic, 11-integer header, 22 full 3-D arrays, 3 full 2-D arrays, finite
payload, exact EOF, and exact inventory
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l8_orca2krhs/l8_records.py:14-74`).
At kt 1 its acting check requires damping to be bitwise inert; every other
boundary must act.  At later smoke steps damping must act too.

The preflight applies the observer patch to the actual instrumented source,
executes a moved-call plant, checks both a committed real `ocean.output`
excerpt and a stamped L8 deck with whitespace-tolerant parsing, compiles and
runs the committed gfortran layout writer against the Python reader, fires an
EOF plant, checks the schedule/acting scope, checks output viability, and
requires fresh target and run paths.  Scratch is removed by an EXIT trap
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l8_orca2krhs/run.sh:80-145`).

From clean commit `98feff346e`, the required preflight ended:

```text
L8_PREFLIGHT_7_OF_7_PASS plants=3/3 compiled_gfortran_layout=true real_log=true stamped_deck=true
WOULD_RUN smoke -> writer-free boundary -> three twins into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round56/oracle_l8_r56
```

## 5. Landing gates

After re-anchoring 73 citation-map lines and 18 audited-receipt lines, the
explicit citation gate passed 17 / 17 tests in 5.40 seconds.  The exact ORCA2
default gate from `claude_rounds/land.sh` then passed 145 / 145 tests in
993.72 seconds under CPU, x64, and the landing package path.  That total
includes the private-import ratchet at 6 / 6 and both ORCA2 parallel-receipt
checks.  No bar was changed and no test was skipped to obtain these results.

## 6. ASKED / UNASKED

ASKED and completed: trace the running Oracle and selected BBL branch; state
the exact order; build the G9+L1+L2+L4b+L6 L8 card; make only that exact
composition legal; preserve all old routes; add bit-position call-order
controls, a finite-gradient control, a production one-step control, and the
committed ladder acquisition; execute its clean preflight.

UNASKED and not done: build, run, or alter NEMO; acquire a record; score L8;
relax a fidelity bar; push; post to GitHub; or perform an author review.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r56-672930488
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r56-672930488/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l8_orca2krhs/run.sh --run (operator-only; NEMO_OPERATOR_ACK=L8_R56_OPERATOR)
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L8 design/build and clean seven-item preflight complete; no NEMO run and no score
