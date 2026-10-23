# Grid ladder rung L9 — round 1 build receipt (gridladder_r57, Codex)

Date: 2026-10-23

Starting commit: `bd83f3fd1eeb`

Preregistration commit: `a174af33e1`

Implementation/acquisition commit: `e4f204272d`

Receipt/citation commits: `a27eba2567`, `6afb3af28d`

HOLD repair commits: `f644ffd530`, `760335a39d`, `287994dbe4`,
`bde87fb35c`, `fd362e9caa`

Scoring commit: `234a5b62467496181a186ea4e96f4ecd315b6e89`

Status: **L9 CERTIFIED.** The operator acquired and admitted the complete
ten-step, three-twin record. Both preregistered trajectory labels are bit-exact
through kt 10 at full depth, every kt-1 stage row is bit-exact, and all 146
live model-term rows are exact or inside the unchanged normalized `1e-15` bar.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; completed and admitted by the operator
- scoring: completed from the clean scoring commit above

## 1. Running NEMO order

ORCA2 computes the standard-isoneutral slope and then refreshes its
time-varying eddy coefficient before the Runge--Kutta stages
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:184-194`). Stage 3 next forms
the ordinary U/V/W tracer transport in the live non-Shuman branch: it calls
`tra_adv_trp` at line 555, then computes selected BBL coefficients at line 560
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:555`,
`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:560`). The former
range 568--582 was the disabled `ln_shuman` branch and is retracted.

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
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10527-10554`,
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:15020-15028`).

L9 now runs the already-certified L3 production statements directly; the
redundant L9-only helper and branch were deleted. The two U/V additions feed
the existing W recomputation, which consumes those exact augmented arrays
before the existing L8 Krhs walker executes
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10541-10572`).
No arithmetic or configuration changed: the deleted L9 arm was textually and
arithmetically identical to these L3 statements.

## 3. Bit-position, production controls, and mutation proof

The x64 JIT position control retains the seven L8 Krhs
operand/accumulated pairs:

| position | operand | accumulated word |
|---|---|---|
| advection | `0x1.8484bbe636a7cp+25` | `0x1.8484bbe636a7cp+25` |
| SBC | `0x1.80c4d885778a5p+25` | `0x1.82a4ca35d7190p+26` |
| QSR | `-0x1.e4576fad0145bp+34` | `-0x1.e2d4cae2cb6e9p+34` |
| LDF | `-0x1.fbbb660e98d04p+8` | `-0x1.e2d4cb61ba481p+34` |
| BBC | `0x1.99a22105eba7ep+34` | `-0x1.24caa96f3a80cp+32` |
| BBL | `0x1.792a95b49c117p+27` | `-0x1.190154c195a03p+32` |
| DMP | `-0x1.0452f7986b12bp+33` | `-0x1.90d3a1f935e2cp+33` |

Temperature and salinity remain pinned bit for bit at every L8 boundary. Each
of the six adjacent-call swaps changes a displaced boundary and fails. Four
selector plants fail, including partial face averaging and removal of the
Decision 147 route; JIT/gradient forward bits remain identical and gradients
finite (`tests/ocean/fidelity/test_nemo_testcase_l9_orca2eiv_krhs.py:99-181`).

The full one-step ORCA2 production test now observes the exact live arrays at
the production W-recomputation boundary. It requires nonzero EIV U/V deltas,
pins the W input arrays bit for bit to the post-EIV arrays, and proves this
boundary is immediately before the L8 Krhs walk; finite T/S and the eager
Decision 147 refusal remain checked
(`tests/ocean/fidelity/test_nemo_testcase_l9_orca2eiv_krhs.py:184-255`).
The unmutated test passed in 214.13 seconds. In separate scratch package copies,
discarding both production additions failed at the nonzero-delta assertion in
203.97 seconds, and moving both additions after W recomputation failed with
410,769 U-array bit mismatches in 204.48 seconds. These were source mutations,
not helper substitutions; both exited 1. The fast section passed 6 / 6 selected
tests in 7.60 seconds.

## 4. Acquisition and preflight

The committed launcher defaults to read-only `--preflight-only`; `--run`
requires `NEMO_OPERATOR_ACK=L9_R57_OPERATOR`. Its schedule is a 2-step acting
smoke, a writer-free 100-step reference boundary, and three independently
staged twins ending at one even step between 2 and 10. `check_acting` is its
own statement on the smoke only
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l9_orca2eiv_krhs/run.sh:335-353`).

The combined reader requires every complete L8 stage boundary for all smoke
steps and L3's EIV transport witness at kt 1. Both observers carry full
`jpi*jpj*jpk` arrays, canonicalize cells not defined by NEMO, and are installed
outside XIOS-false branches. The preflight compiles a real gfortran writer
covering both exact streams and proves exact EOF rejection.

The seven-part preflight checks all 4 patched Fortran sources, a continued-line
instrumentation plant, a real scratch-source mutation moving `tra_bbl` before
`tra_bbc`, the listed reference, every possible clock deck, both binary
layouts, writer completeness, smoke-only acting scope, output viability, and
fresh target/run paths. The log checker now checks `nn_aei_ijk_t` once, removes
the duplicate `nn_aht_ijk_t`, and does not require the impossible false
`ln_eke_equ` echo; it instead refuses an acting GEOMETRIC line when the stamped
deck selects false. Preflight runs that checker against the stamped L9 deck and
a committed excerpt copied verbatim from the real L3 round-20 smoke output
(source SHA-256 `d0d5fdd5d142e2352a77d142f0025df219dcd30d14b31e29c3710ccbc50e4bda`),
then proves a mutated real line is refused. No invented log line is appended.
The `EXIT` trap removes only the target and run paths created by a failed
`--run`, so a failed operator attempt is retryable without manual deletion
outside this acquisition's output
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l9_orca2eiv_krhs/run.sh:163-262`).

The repaired complete clean-tree preflight exited 0 and left no scratch. The
standalone `stage order` line is the expected rejection from the real-source
mutation plant:

```text
L9_RECORD_SELFTEST_OK combined_layout=true truncation_refused=true
L9_PREFLIGHT_7_OF_7_PASS plants=all_refused compiled_gfortran_layout=true patched_syntax=4 real_log=true stamped_deck=true
WOULD_RUN smoke -> writer-free boundary -> three twins into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round57/oracle_l9_r57
```

That sentence described the build boundary. The operator subsequently ran the
unchanged acquisition at commit `76e54db84fa5a4fb491805a65475ab9f707375fd`.
The admitted log reports `changed_transport_values=823803` in both acting
checks, exact defined-twin counts of 413,030 U cells and 415,175 V cells for
each before/after/delta array, L8-record SHA-256
`32dc2ba1c42a93a120b1d9d2aa2818227dcb4fba2c208bac388e4d3229e49ab3`,
steps 1--10, and three twins. Its final line is:

```text
L9_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round57/oracle_l9_r57
```

## 5. Landing gates

The receipt-specific citation gate found all 21 L9 citations, with 0 unmapped
citations, 0 endpoint failures, and 0 global map-audit failures; all 9
self-test controls fired. Shifting the corrected live `tra_adv_trp` citation
by 2 lines exited 1. The citation-gate pytest file passed 17 / 17 inside the
default gate.

The exact ORCA2 default gate from `claude_rounds/land.sh` passed 145 / 145 in
1121.61 seconds (18 minutes 41 seconds) under CPU, x64, and the prescribed
landing package path. That total includes the private-import ratchet at 6 / 6
and both ORCA2 parallel-receipt checks. The private-import ratchet was also
rerun alone and passed 6 / 6 in 11.30 seconds. The full clean-tree preflight
was rerun after the final code/comment commit and again printed
`L9_PREFLIGHT_7_OF_7_PASS`. No gate was relaxed or skipped.

The final certification-gate results are recorded below after the clean score.

## 6. Clean certification score

All four admitted score artifacts were produced from clean commit
`234a5b62467496181a186ea4e96f4ecd315b6e89`. Their package tree is
`8496180ea7d363cbfa3c32e1752462a18712fe38`, exactly the package tree of the
operator's acquisition commit. The scoring commit changes only the shared
scorer and scorer tests: no package, model, existing card, two-step program,
configuration value, or fidelity bar changed.

The score uses kt 1's three stage records and, from kt 2 onward, the independently
staged restarts. INDEPENDENT is bit-exact at kt 1--10: 70 / 70 surface and
2-D rows (1,864,800 values), 40 / 40 full-depth field summaries (16,279,300
full cells plus 613,790 partial cells), and 6 / 6 barotropic-history rows
(159,840 values), with 0 unequal and 0 flushed-subnormal values in every
class. GIVEN-NEMO-ENTRY is bit-exact at entries 1, 3, 5, 7, and 9 and at their
two-step successors 2, 4, 6, 8, and 10: again 70 / 70 surface and 2-D rows
(1,864,800 values) and 40 / 40 full-depth field summaries (16,279,300 full
plus 613,790 partial cells), with 0 unequal and 0 flushed-subnormal values.

At kt 1, stage 1, stage 2, and stage 3 are bit-exact: 21 / 21 surface and 2-D
rows (559,440 values) and 12 / 12 full-depth field summaries (4,883,790 full
plus 184,137 partial cells), with 0 unequal and 0 flushed-subnormal values.
Every full-depth dry-cell comparison in both trajectory labels and all three
stages has 0 unequal and 0 flushed-subnormal values.

The direct L9 score test runs the actual card, requires all 20 kt-1 EIV/Krhs
rows at the fixed bar, requires all six EIV rows exact, and refuses dry writes
(`tests/ocean/fidelity/test_nemo_testcase_l9_orca2eiv_krhs.py:262-280`). The
clean ten-step live score has 146 / 146 rows at bar: 83 bit-exact and 63
AT-BAR-NOT-EXACT, 404 unequal wet words in total, 0 flushed subnormals, and 0
card or NEMO dry writes. Each of the 140 Krhs rows compares 430,552 wet cells.
The six EIV rows are bit-exact: before/delta/after U each compare 413,030 wet
cells and before/delta/after V each compare 415,175, all with 0 unequal,
0 max-absolute error, 0 normalized error, 0 ulps, and 0 dry writes.

For the complete live Krhs table below, each T/S tuple is
`unequal / max-absolute / normalized / maximum-ulps`; `E` means
`0 / 0 / 0 / 0`. Boundary groups on one row have identical tuples.

| kt | boundary or boundaries | T tuple | S tuple |
|---:|---|---|---|
| 1 | ADV, SBC, QSR | E | E |
| 1 | LDF, BBC, BBL, DMP | 7 / 1.32348898008484428e-23 / 2.96141064078980926e-19 / 2 | 1 / 2.11758236813575085e-22 / 3.17348422538388520e-18 / 1 |
| 2 | ADV | 1 / 1.05879118406787542e-22 / 9.09119680325445228e-19 / 2 | 5 / 7.62329652528870305e-21 / 4.60076511710721600e-17 / 73 |
| 2 | SBC | 1 / 1.05879118406787542e-22 / 9.09119680325445228e-19 / 2 | 5 / 7.62329652528870305e-21 / 4.64361114523413000e-17 / 73 |
| 2 | QSR | 1 / 1.05879118406787542e-22 / 9.09119599886241466e-19 / 2 | 5 / 7.62329652528870305e-21 / 4.64361114523413000e-17 / 73 |
| 2 | LDF, BBC, BBL, DMP | 1 / 1.05879118406787542e-22 / 9.11223020745474825e-19 / 2 | 5 / 7.62329652528870305e-21 / 4.64349720463274263e-17 / 73 |
| 3 | ADV, SBC, QSR | E | E |
| 3 | LDF, BBC | 5 / 2.64697796016968856e-23 / 1.71762826154187503e-19 / 2 | 5 / 4.23516473627150170e-22 / 2.35034908079448919e-18 / 4 |
| 3 | BBL, DMP | 5 / 2.64697796016968856e-23 / 1.71762826154187503e-19 / 2 | 5 / 4.23516473627150170e-22 / 2.35057742535413530e-18 / 4 |
| 4 | ADV, SBC, QSR | E | E |
| 4 | LDF, BBC, BBL, DMP | 10 / 5.29395592033937712e-23 / 2.90655728436045978e-19 / 16 | E |
| 5 | all seven boundaries | E | E |
| 6 | ADV, SBC | 8 / 2.37169225231204095e-20 / 1.41822619722475999e-16 / 28 | E |
| 6 | QSR | 8 / 2.37169225231204095e-20 / 1.41806995717161662e-16 / 28 | E |
| 6 | LDF, BBC, BBL, DMP | 9 / 2.37169225231204095e-20 / 1.40052146604316911e-16 / 28 | 1 / 6.61744490042422140e-24 / 2.28588816860137173e-20 / 1 |
| 7 | ADV | 1 / 2.64697796016968856e-23 / 1.09717523372792592e-19 / 1 | E |
| 7 | SBC | 1 / 2.64697796016968856e-23 / 1.07039870912264957e-19 / 1 | E |
| 7 | QSR | E | E |
| 7 | LDF, BBC | 3 / 2.11758236813575085e-22 / 8.77314062438461449e-19 / 2 | E |
| 7 | BBL, DMP | 3 / 2.11758236813575085e-22 / 8.77314062438461449e-19 / 1 | E |
| 8 | ADV, SBC, QSR | E | E |
| 8 | LDF, BBC, BBL, DMP | 3 / 5.29395592033937712e-23 / 2.66603518595952324e-19 / 2 | E |
| 9 | ADV, SBC, QSR | E | E |
| 9 | LDF, BBC, BBL, DMP | 18 / 1.05879118406787542e-22 / 2.70989175085999629e-19 / 4 | E |
| 10 | ADV, SBC, QSR | E | E |
| 10 | LDF, BBC, BBL, DMP | 22 / 1.05879118406787542e-22 / 2.48874897580083398e-19 / 16 | E |

The largest live error is kt-6 ADV temperature: maximum absolute
`2.37169225231204095e-20`, normalized `1.41822619722475999e-16`, and 28 ulps,
safely within `1e-15`. The largest maximum-ulp count is 73 in kt-2 salinity;
its largest normalized error is `4.64361114523413000e-17`. No statement fails.

Score SHA-256 hashes and exact byte counts:

```text
8e856529195737cb83dde31f99f6e630b7ff9fbe4f98bbee158cda712c05bb10  70831   l9_live_terms_kt1_10.json
43daa0fdfd370fe88f324ff591bfe1da4260f6b4b206e601769b21bc236d8a8f  288582  l9_stages_kt1.json
5c9a3412a78b560c6e26d55a4f9c682087324578e2f2cfa1432129f39aa317c7  980819  l9_independent_kt1_10.json
7793630eaa3a6472bdef9e2564be300e8954fecda42f9030023818ee6268d859  964393  l9_given_entry_kt1_10.json
```

The finite-gradient control still gives identical ordinary/JIT forward bits
and a finite gradient. L9 therefore introduces no new NaN-gradient source;
the known inherited dry-gradient behavior is neither changed nor claimed
fixed.

## 7. Known residual and certification

The L9 production test alone does **not** catch FCT reading transports without
the eddy-induced part. The L3 record test catches that defect on the same
shared production path. This residual is retained explicitly and is not
papered over.

Every other card's two-step state and step program remain the acquisition's
zero-difference merge-identity result. The identical package-tree stamps above
prove that scoring did not alter that result. No configuration change was
made or needed.

Both preregistered trajectory labels are bit-exact, the full-depth kt-1 stage
record and kt-2-onward restarts are bit-exact, and every live EIV/Krhs boundary
is exact or inside the preregistered bar. L9 is therefore **CERTIFIED: L8 plus
the complete L3 eddy-induced transport enters FCT before the seven ordered
stage-3 Krhs boundaries.**

## 8. Final certification gates

All final gates used `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and the exact
`PYTHONPATH` from `claude_rounds/land.sh`.

- The receipt-specific gate found 22 citations, 0 unmapped citations, 0
  endpoint failures, and 0 global map-audit failures; all 9 self-test controls
  fired. Shifting the new direct-score citation by 2 lines exited 1.
- The focused L9 file passed 8 / 8 in 545.38 seconds, including the six fast
  source-order/selector/finite-gradient controls, the full production route,
  and the admitted-record direct score.
- The exact unsplit ORCA2 default gate from `claude_rounds/land.sh` passed
  145 / 145 in 1118.50 seconds (18 minutes 38 seconds). Its citation-gate
  pytest passed 17 / 17, its private-import ratchet passed 6 / 6, and both
  ORCA2 parallel-receipt tests passed 2 / 2.
- The private-import ratchet rerun alone passed 6 / 6 in 5.29 seconds. The
  D127 comparison-mask regression passed 2 / 2 selected tests in 0.42 seconds
  (19 deselected).

No gate was relaxed or skipped. No NEMO command was run during scoring or
certification.

## 9. ASKED / UNASKED

ASKED and completed: read the executed Oracle path with L8 and L3 selected;
state and cite the exact order; build L9 as L8 plus ORCA2's complete L3 EIV;
relax only that guarded composition while retaining Decision 147; pin every
ordered position with INEXACT one-step controls and moved-call plants; preserve
forward/gradient behavior; commit the ladder-standard acquisition; run its
read-only preflight; admit the operator record without rerunning NEMO; score
both labels through kt 10 at full depth, kt 1 against stage records and kt 2
onward against restarts; score all live model-term boundaries; retain the
known production-test residual; and certify only because every row meets the
fixed bar.

UNASKED and not done: rerun or alter NEMO; change any configuration, fidelity
bar, existing card, or two-step value; push; post to GitHub; or perform an
author review.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r57/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L9 CERTIFIED; both trajectory labels are bit-exact and all live model-term boundaries are exact or at bar
