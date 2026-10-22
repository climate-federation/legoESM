# Grid ladder rung L8 — round 1 certification receipt (gridladder_r56, Codex)

Date: 2026-10-22

Starting commit: `f32c3be7ac9d`

Preregistration commit: `12483a8c59`

Implementation/acquisition commits: `ce9636bc1f`, `28fefb1eaa`,
`bf01715c57`, `d710262489`, `98feff346e`

Receipt/citation re-anchor commit: `e111ccce8a`

Review-gap and L8 citation-map commits: `738c136797`, `0131fd4439`

Status: **L8 CERTIFIED.**  The operator acquired and admitted the complete
ten-step record.  From clean commit `738c136797`, both trajectory labels,
every available full-depth row, every stage-3 record boundary, and every
selected term record are exact or at the fixed bar.  The trajectory rows are
bit-exact; a later live-term audit corrects the term-boundary classification
from exact to AT-BAR-NOT-EXACT without changing any model value.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; admitted record supplied by the operator
- scoring: complete from a clean tree

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
The original six L8 tests did not prove that call-site ordering: their move
plants supplied reordered operands directly to the walker helper.  The review
correctly identified that moving the live BBL group after DMP still passed all
six.  The added one-step production-route test spies on the live geothermal,
BBL, damping, and walker calls.  It requires group 3 to be the exact geothermal
rate object with empty salinity, group 4 to be the exact BBL result with
nonzero T and S only at bottom cells, and group 5 to equal the two live
`gamma * (target - Kbb)` rates bit for bit.

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

The original test checks every temperature and salinity boundary bit for bit
under JIT, not merely the final sum.  Its planted LDF/BBC and BBL/DMP moves are
helper-only controls (`tests/ocean/fidelity/test_nemo_testcase_l8_orca2krhs.py:80-114`),
so the earlier receipt's statement that they pinned the live call position is
retracted.  Card plants separately remove four required selectors and are
rejected; the ordinary one-step test exercises the complete selected route
(`tests/ocean/fidelity/test_nemo_testcase_l8_orca2krhs.py:151-200`).  The new
production-route spy passed on the selected call site in 205.66 seconds.  The
exact call-site plant moving BBL after DMP then failed at the group-4 BBL
identity assertion in 205.96 seconds; restoring the call made the full focused
battery pass 10 / 10 in 309.65 seconds.

The final focused L8 run passed 6 / 6 tests in 114.99 seconds under CPU and
x64.  The earlier 40-test combined G9/L4b/L6/L8 regression attempt completed
the L4b and L6 sections, found one expected-set assertion that needed L8 added
as a G9 composition, and later hit the documented per-process compiler limit.
After that test correction, the two isolated heavy G9 controls passed in
96.01 and 167.07 seconds.  No inherited production selector was changed.

After the live-term correction, the complete L8 focused file passed 9 / 9 in
470.67 seconds.  Its new test requires the live arm to be exact through QSR,
first differ at `ldf_T`, remain inside the fixed bar, and report no dry writes.

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

The operator then ran that exact acquisition.  Its admitted final line was:

```text
L8_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round56/oracle_l8_r56
```

No NEMO build or run was performed by this round agent.

## 5. Landing gates

From clean commit `0131fd4439`, the receipt-specific citation gate found all
15 recognized L8 citations, with zero unmapped citations, zero endpoint
failures, and zero global map-audit failures; all 9 self-test plants fired.
Shifting the L8 stage-3 citation by two lines produced
`SYMBOL-NOT-AT-LINE` and the required nonzero exit.

The exact ORCA2 default gate from `claude_rounds/land.sh` then passed
145 / 145 tests in 1025.26 seconds under CPU, x64, and the prescribed landing
package path.  That total includes the private-import ratchet at 6 / 6 and
both ORCA2 parallel-receipt checks.  No bar was changed and no test was
skipped to obtain these results.

After the correction, the same 145-test default gate was rerun in three
synchronous chunks and passed 25 / 25, 94 / 94, and 26 / 26.  The first chunk
includes the private-import ratchet at 6 / 6 and both ORCA2 parallel-receipt
checks.  The receipt-specific gate again found all 15 citations with zero
failures and zero global map-audit failures; its shifted-citation plant exited
nonzero.

## 6. Clean certification score and live-term correction

The score ran from clean commit `738c13679724536b70975ac80110050076a774cd`;
its packages tree is `db06c1e1514fade4fb59ee756eb7eedde101fdef`, identical
to the acquisition's package code.  The independent label is bit-exact at
kt 1--10 for SSH, barotropic U/V, three-dimensional U/V/T/S, every available
full and partial depth, and every dry cell.  The given-NEMO-entry label at
entries 1, 3, 5, 7, and 9 and its two-step results at kt 2, 4, 6, 8, and 10
are also bit-exact at full depth.  All stage-1, stage-2, and stage-3 kt-1
surface and full-depth rows are bit-exact.

The live-term correction ran from clean commit
`9e3590e2d45eaf860afac77758ce96cd093b572c`; its packages tree is still
`db06c1e1514fade4fb59ee756eb7eedde101fdef`, so no model value changed.

The original JIT/traced/x64 `l8_krhs` arm starts from NEMO's recorded LDF
boundary and feeds NEMO's recorded BBC, BBL, and DMP increments to the ordered
walker.  Its 80 zero-unequal rows and its JIT/traced/x64 removal controls prove
the walker wiring only; they do **not** test the model's own isoneutral, BBL,
or damping terms.  Calling those rows exact model-term comparisons is
retracted.  The geothermal record separately remains exact at all 16,433 wet
columns per step, and the isolated damping record remains exact over 430,552
wet cells per tracer per step, with zero dry writes.

The added `l8_live_krhs` arm advances the actual L8 card, passively observes
the seven compiled stage-3 boundaries, carries the state from a separately
compiled unobserved ordinary step, and compares those model-produced boundaries
directly with NEMO's boundary record.  All 140 T/S rows at kt 1--10 are
**AT-BAR-NOT-EXACT** with zero dry writes.  The first scattered differences
appear at the isoneutral-diffusion boundary and carry through BBC, BBL, and DMP:
temperature has 3--25 unequal wet words per step and salinity 0--7.  At kt 10,
the advection boundary itself has 3 unequal T words and 6 unequal S words,
which are carried forward before the additional LDF differences.  The maximum
is 32 ulps except for 1024 ulps in one near-zero salinity cell; maximum
normalized error is `6.372895144965704e-17`, safely inside the unchanged
`1e-15` bar.  Reassociation in the inherited isoneutral and FCT operators is a
**PLAUSIBLE**, not established, cause of these last-bit differences.

The two bit-exact trajectory labels do not discriminate a BBL/damping call
swap at kt 1--2: damping is inert at kt 1, and the completed trajectory sees
the final accumulator rather than the intermediate boundary positions.  Only
the NEMO term-boundary record discriminates that swap.

Score SHA-256 hashes:

```text
f60cba0ea7885ca48f1e8040d86b555e4d99dc9e318225a89b2beea183dc89f6  l8_independent_kt1_10.json
d2081f712888dda66d4320043893e0c5c0aca4a4968ffa8e00e9c7e14f30235a  l8_given_entry_kt1_10.json
0440eec583415f8667bb1b5f66a3bdf62e1cebb12886a1238dda06b9f0fcb4d0  l8_stages_kt1.json
2cccb926070fcb471abf537c5f998e48439fa9542f9e77305d983c9a4f85b816  l8_krhs_kt1_10.json
c4c22458648ab3b0b3eec5298e7a175411f8af3254da473cf6d4b4139a7d1017  l8_geothermal_kt1_10.json
80143e2ae669753957df22ea4df95525c7bbef7e043613a13621f16fec0ba80a  l8_damping_kt1_10.json
22b944170099f453b04855fc339ea42afdec0afb13ee37146033da534efe04b6  l8_live_krhs_kt1_10.json
```

Every trajectory row is exact and every live term-boundary row is at the
preregistered bar.  L8 therefore remains **CERTIFIED: the stage-3 Krhs route
carries diffusive BBL, standard isoneutral diffusion, geothermal heating, and
interior damping in NEMO's order.**

## 7. Inherited-card preservation

The extended merge-identity probe compares every certified card in both lanes:
G1--G9, L4, L4b, L6, V1--V5, and L1--L3.  Baseline snapshots use clean
pre-L8 commit `f32c3be7ac9d`; candidate snapshots use clean commit
`738c136797`.  Across 20 cards and 772 state leaves after each of two
production steps, `different_count=0`.  All 20 address-normalized one-step
JAX programs have identical SHA-256 hashes and lengths.  A planted one-bit L2
leaf digest was refused.  Evidence is under
`gridladder_rounds/round56/merge_identity`.

## 8. ASKED / UNASKED

ASKED and completed: trace the running Oracle and selected BBL branch; state
the exact order; build the G9+L1+L2+L4b+L6 L8 card; make only that exact
composition legal; preserve all old routes; add bit-position call-order
controls, a finite-gradient control, a production one-step control, and the
committed ladder acquisition; execute its clean preflight; close the
production-route and inherited-card review gaps; admit the operator record;
score both labels through kt 10 at full depth; score all three kt-1 stages and
the BBL, geothermal, and damping records; add a live-model term arm and its
direct test; correct the term rows from exact to AT-BAR-NOT-EXACT; and certify
only because every row is exact or at the fixed bar.

UNASKED and not done: build, run, or alter NEMO; relax a fidelity bar; push;
post to GitHub; or perform an author review.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r56/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L8 CERTIFIED; both trajectory labels are exact and all live model-term boundaries are at bar
