# NEMO-testcases L2 GYRE round 105 receipt: shear/solver free-surface routing split

Date: 2026-09-17. Branch `fidelity/nemo-testcases-l2-gyre-codex2`; incoming
tip `c9f6742f37d8ad566fdc2fdabb8dfb5a6efee05a`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round105/`.

## Verdict

**HELD. No production physics, configuration, carried state, restart schema,
or selector landed.** The requested routing split is source-required and it
makes the shear face metrics and `p_sh2` bit-identical to NEMO given NEMO's
recorded entry in both the full eager closure and production-step JIT.
Nevertheless it fails both conditions that control a landing:

1. the next stage boundary, the TKE `en` right-hand side, is exact eager but
   remains non-bit in **11,024 of 20,416** cells at
   `5.551115123125783e-17` through the production JIT; and
2. the complete Rule-12 comparison is **FAIL**: **82 of 954** rows move,
   **59 rows / 203,983 cells** worsen beyond the two-ULP cellwise bar, and
   the worst worsening is `5,816,172.3125` row-scale ULP.

No row changes AT-BAR/DEBT class and first-over-bar stays kt2 U/V, but those
necessary conditions do not erase the cellwise violations. Aggregate kt3
T/S and day-30 T improve; they are reported below and do not override the
gate. Candidate commit `55fb9165399dcd9e350ae09654a3988d9ac1966e` was
removed from production at `f0d037c16ac8` and preserved only as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round105_held_shear_routing_split.patch`.

The retained diff is measurement infrastructure, controls, the held patch,
the tank exclusion test, citation re-anchors, this receipt, and no numerical
candidate. No NEMO source was modified; neither `makenemo` nor `mpirun` ran.
No acquisition and no user configuration decision are needed.

## Registration and immutable arm

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round105.md`, commit
`52c3bbbb75aa`. It precedes the new diagnostic and every measurement. The
immutable trajectory arm is
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`, documented
bit-identical to the round-96 full arm used by the 954-row comparator.

Incoming headline values were:

| row | immutable before |
|---|---:|
| kt2 T | `1.4210854715202004e-14` K |
| kt2 S | `2.1316282072803006e-14` |
| kt2 U | `2.7377110452773967e-12` m/s |
| kt2 V | `3.2849219221489645e-12` m/s |
| kt3 T | `1.627497246303733e-4` K |
| kt3 S | `6.327735185607253e-6` |
| day-30 T RMS | `1.2397011296352737e-2` K |

## Compiled-source basis

Every numerical statement below is from the compiled branch of the record
that supplies its operands.

GYRE's step program contains a commented call with `Nnn`, followed by the
executed call with **both** formal time levels bound to `Nbb` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:167-168`.
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:319-320`
forwards those two formals to `zdf_sh2`. The executed no-Stokes shear block is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:97-109`;
its U divisor is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:102` and its
V twin is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:107`.
Therefore both divisor slots consume the step-entry free-surface ratio on
this card.

The tracer solver is deliberately not changed by that conclusion. Stage 3
declares `Kmm=N+1/2` and writes its half-step ratio at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3_stg.f90:240-256`;
the implicit tracer matrix consumes `r3t(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/trazdf.f90:461-477`.
The rejected candidate consequently threaded a second argument only to the
shear calculation and left the existing half-step solver input untouched.
It added no card field and changed no default.

The statement-boundary record was emitted by the R101 compiled target. Its
three matrix writes and following `en` right-hand side are contiguous at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-442`.
After the shear operand becomes BIT, the first non-bit statement is the
right-hand-side assignment on lines 439--442, not any of the three matrix
writes on lines 434--436.

DINO is a different compiled program. It calls vertical physics with distinct
`Nbb,Nnn` slots at `DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193`, and
`DINO/BLD/ppsrc/nemo/zdfphy.f90:316-319` forwards them to the shared shear
routine. Its tracer solver independently consumes `Kmm` at
`DINO/BLD/ppsrc/nemo/trazdf.f90:219-235`. That is why DINO must keep its
existing NOW x BEFORE route and why the GYRE-only call argument was omitted
from every non-WS caller.

## Diagnostic extension and fail-closed scope

The existing round-46/103/104 stage tool was extended; no second harness was
created. The production TKE trace now returns the exact four face metrics
consumed by the shear helper. The gate reconstructs their NEMO references
from recorded `r3u/r3v` and `e3w_1d`, scores the direct captures, and can run
the entire production closure in either of two explicitly labelled modes:

- `production step JIT`: the normal compiled `model.step` path;
- `production closure eager`: `_step_jitted.__wrapped__` under
  `jax.disable_jit`, after the same production carry seeding and cache prime.

This is not an isolated-closure shortcut. A NumPy-to-JAX leaf adapter was
needed only for the eager arm; the first attempt failed loudly on NumPy's
missing `.at` and is retained as `local_before_eager.log`. The corrected run
is `local_before_eager_retry.json`.

The DINO routing audit is a bounded helper execution from a developed NEMO
restart. It labels eager and isolated-JIT results separately and labels the
full production step **UNMEASURED** because the current committed DINO twin
refuses upstream in GM/Redi raw-mesh N2 before reaching `zdf_sh2`. No helper
hash is promoted to a full-step claim.

## Frozen predictions and dispositions

| prediction | disposition |
|---|---|
| P1: baseline `p_sh2` reproduces 17,400 unequal at `3.811744924985501e-14`, and a captured face metric is non-bit | **CONFIRMED** in eager and production JIT |
| P2: the split makes all four face metrics and production `p_sh2` BIT in eager and production JIT | **CONFIRMED** |
| P3: the production-JIT route plant changes exactly one `p_sh2` cell, prints `STATUS PLANT-FIRED`, exits nonzero | **CONFIRMED**, exit 1 |
| P4: kt2 rows retained and kt3 T/S improve without a Rule-12 violation | kt2 retention and aggregate improvement **CONFIRMED**; cellwise Rule 12 **REFUTED**, so candidate HELD |
| P5: day-30 T RMS decreases | **CONFIRMED** by `5.8615519e-11` K; cannot override P4 |
| P6: DINO's measured route is bit-identical before/after | **CONFIRMED for eager and isolated-JIT helper arms**; production step remains UNMEASURED upstream |
| P7: LOCK_EXCHANGE and OVERFLOW do not select prognostic TKE shear; ORCA2 remains unmeasured | **CONFIRMED** for both tanks by instantiated config and test; ORCA2 is **UNMEASURED-WITH-SPEC** |
| P8: land only if every gate passes | **CONFIRMED as a veto**; production restored and patch held |

The preregistration's plausible expectation that making `p_sh2` exact would
also make the full `en` RHS exact is **REFUTED under production JIT**. It is
true only eager. This failed prediction is the main new result of the round.

## Direct before/after statement tables

All shear-output rows score 20,416 owned cells; 17,400 are wet. Metric rows
have their own U/V face domains. Before values were identical in eager and
production JIT.

| direct row | before unequal | before max abs | candidate eager | candidate production JIT |
|---|---:|---:|---:|---:|
| `e3u_now` | 16,820 / 21,054 | `1.0986094088138998e-4` | 0 / `0.0` | 0 / `0.0` |
| `e3u_before` | 16,820 / 21,054 | `1.0986094088138998e-4` | 0 / `0.0` | 0 / `0.0` |
| `e3v_now` | 16,530 / 21,344 | `1.0950258047159878e-4` | 0 / `0.0` | 0 / `0.0` |
| `e3v_before` | 16,530 / 21,344 | `1.0950258047159878e-4` | 0 / `0.0` | 0 / `0.0` |
| production `p_sh2` | 17,400 / 20,416 | `3.811744924985501e-14` | 0 / `0.0` | 0 / `0.0` |

The all-recorded reference replay is BIT before and after. Thus the change in
the production row is attributable to the routed time level rather than to a
broken mirror.

The next statement boundary distinguishes eager from production:

| matrix/RHS write | before eager | before JIT | candidate eager | candidate JIT |
|---|---:|---:|---:|---:|
| `zd_up` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` |
| `zd_lw` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` |
| `zdiag` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` |
| `en_rhs` | 2,663 / `5.488912657725109e-10` | 11,993 / `5.488912518947231e-10` | **0 / `0.0`** | **11,024 / `5.551115123125783e-17`** |

The production-JIT residue is roughly seven orders below the former maximum,
but the proof unit is BIT, not magnitude. Because eager is exact and the
production step is not, this is positive evidence for a full-step XLA
association/fusion boundary. It does not yet name which materialization of
the four-term expression is required; that is the next discriminating walk.

The `stage-shear-operand-ulp` control advanced one wet reference value at
index `[1,1,0]`. Exactly one candidate `p_sh2` cell changed by
`3.308722450212111e-24`; the gate printed `STATUS PLANT-FIRED` and exited 1.
The first attempt was interrupted before it emitted a report and is retained
as an empty `local_after_plant.log`; only `local_after_plant_retry.json` is
canonical.

## Restored-production stage-twin tables

After holding the patch, the consolidated stage twin was rerun at clean commit
`aa7e78ba3f81ac89bfb36b1e13099cacae7aa877`. Cells below are counts of output
rows in the order **BIT / non-BIT AT-BAR / DEBT**.

Given NEMO's recorded entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `13 / 4 / 0` | `11 / 0 / 0` | `14 / 2 / 1` | `11 / 1 / 5` |
| 2 | `10 / 4 / 3` | `0 / 0 / 11` | `11 / 3 / 3` | `10 / 2 / 5` |

Model-chained entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `8 / 4 / 5` | `10 / 1 / 0` | `5 / 5 / 7` | `3 / 5 / 9` |
| 2 | `3 / 0 / 14` | `0 / 0 / 11` | `0 / 0 / 17` | `0 / 0 / 17` |

The table's global first owned non-bit output remains kt1 stage-1 U:
7,620 unequal cells at `5.421010862427522e-20` (AT-BAR); V is 8,460 at the
same maximum. Round 105 does not claim that downstream stages are now
eligible, and it lands no downstream statement. Within the explicitly ordered
TKE sub-walk authorized for this round, the first remaining non-bit boundary
after the exact shear operand is the production-JIT `en_rhs` above.

## Rule 12: complete 954-row result

The short ten-step trajectory comparison first refused the candidate across
70 rows. The canonical result is the complete report
`ladder_full_rule12.json`, comparing candidate `ladder_full.json` with the
unchanged round-96 full arm through the shared offline comparator and its
per-cell residual sidecars.

| property | result |
|---|---:|
| certified rows | 954 |
| moved rows | 82 |
| violating rows | 59 |
| violating cells | 203,983 |
| status-class changes | 0 |
| first-over-bar | kt2 U/V -> kt2 U/V |
| worst oracle-relative worsening | `5,816,172.3125` row-scale ULP |
| verdict | **FAIL** |

The full moved-row table is the `field_moves` array in the artifact; no moved
row is omitted from this receipt by implication. A baseline self-comparison
passes all 954 rows with zero movement. Its `worsen-3ulp` plant changes one
cell of kt1 `uu_b`, reports exactly 3.0 ULP against the 2-ULP bar, and exits 1.

Headline maxima show why aggregate-only scoring would have been misleading:

| row | immutable before | candidate | movement |
|---|---:|---:|---:|
| kt2 T | `1.4210854715202004e-14` | same | none; AT-BAR retained |
| kt2 S | `2.1316282072803006e-14` | same | none; AT-BAR retained |
| kt2 U | `2.7377110452773967e-12` | same | none; first-over-bar retained |
| kt2 V | `3.2849219221489645e-12` | same | none; first-over-bar retained |
| kt3 T | `1.627497246303733e-4` | `1.6274801378912684e-4` | improves `1.71084124646e-9` K |
| kt3 S | `6.327735185607253e-6` | `6.327662326555128e-6` | improves `7.2859052125e-11` |

The candidate improves some cells and worsens others; its smaller maximum is
not evidence that every moved cell approached NEMO. Rule 12 correctly catches
that cancellation.

## Days 1--30

The fresh candidate member completed all 180 steps and emitted every daily
snapshot. The fixed scorer read days 1 through 30 against
`phase3/year_owners/nemo_seed0` and exited zero.

| statistic | immutable before | candidate | movement |
|---|---:|---:|---:|
| day-30 T RMS | `1.2397011296352737e-2` K | `1.2397011237737218e-2` K | improves `5.8615519e-11` K |
| day-30 S RMS | `2.1952855356523306e-3` | `2.1952855294158956e-3` | improves |
| day-30 U RMS | `5.301687778753576e-4` m/s | `5.301687775467233e-4` m/s | improves |
| day-30 V RMS | `5.632716531318026e-4` m/s | `5.632716504029849e-4` m/s | improves |
| day-30 SSH RMS | `4.4889711411874356e-4` m | `4.48897114373766e-4` m | worsens `2.550224e-13` m |

This confirms P5 for T but does not rehabilitate the candidate after the
954-row veto.

## DINO and the other cards

DINO's clean before and after audit hashes are identical within each
execution arm:

| arm | before `p_sh2` SHA-256 | after | disposition |
|---|---|---|---|
| isolated shared helper eager | `95e795e70f6054a177aeef255beaa9c6cd14a4ce8427476b973a39043b5144cf` | same | unchanged |
| isolated shared helper JIT | `464655da339eab9d8086deee6b5961106912ae871a5af7569a6f292d728d7df1` | same | unchanged |

All four face-metric hashes also reproduce within their respective arms. The
eager and isolated-JIT hashes differ from one another; they are never called
the same execution. The after plant needed 1,024 ULP advances of eta at
`[47,1]` to reach the first observable change; it moved 28 V-NOW metric cells
and 12 `p_sh2` cells, printed `STATUS PLANT-FIRED`, and exited 1. The DINO card
suite reports `128 passed, 9 warnings in 104.43s`; the identical before suite
reported `128 passed, 9 warnings in 106.67s`.

The full DINO production-step result is **UNMEASURED**, not PASS, for the
upstream GM/Redi reason above. This candidate was held independently, so no
DINO number enters production.

LOCK_EXCHANGE-zco and OVERFLOW-zps instantiate `model_config.physics=None`.
The production predicate returns false before a prognostic TKE shear call in
that state; the committed assertion now covers both cards and the full recipe
file reports `24 passed in 15.25s`. The statement is therefore not executed
on either tank. ORCA2 remains **UNMEASURED-WITH-SPEC**: instantiate its card,
prove the resolved vertical-mixing program, execute one production step with
the optional/default shear input, and compare its full state bitwise before
using this routing there.

## Independent review

The required command was run as `codex exec --sandbox read-only` against the
final diff and all round-105 evidence. It emitted no scientific verdict
because its own app-server could not initialize in the read-only sandbox.
The exact disposition required by the standing operator note is:

> independent review unavailable in-sandbox

The terminal diagnostic was: `Error: failed to initialize in-process
app-server client: Read-only file system (os error 30)`. The complete output
is `codex_review.log`. There is no hidden SHIP verdict and no `DO NOT SHIP`
verdict to override.

## Tests and controls

Every suite summary is quoted rather than inferred from shell status:

| suite/control | result |
|---|---|
| retained TKE/stage focused tests | `81 passed in 29.20s` |
| DINO card before arm | `128 passed, 9 warnings in 106.67s` |
| DINO card after arm | `128 passed, 9 warnings in 104.43s` |
| tank recipe assertions | `24 passed in 15.25s` |
| citation suite before rigid re-anchor | `4 failed, 12 passed in 1.81s`; all four failures were the expected rigid line shifts from the diagnostic edit |
| citation suite after rigid re-anchor | `16 passed in 1.82s` |
| final citation suite | `16 passed in 1.77s` |
| stage shear plant | `STATUS PLANT-FIRED`, exit 1, exactly one target cell |
| DINO routing plant | `STATUS PLANT-FIRED`, exit 1, 12 `p_sh2` cells |
| Rule-12 self-control | PASS, 954 rows, zero movement |
| Rule-12 `worsen-3ulp` plant | FAIL, exit 1, one 3-ULP violation |
| full `tests/ocean/fidelity` + `tests/ocean/unit` | **NO TERMINAL SUMMARY**: 8,136 items collected, progress exceeded 95%, ten xdist workers aborted in JAX compilation, then the final replacement stalled; controller interrupted with exit 130 |
| documented-red nodes, current clone | `1 failed, 1 passed, 1 warning in 87.43s` |
| same documented-red nodes, incoming tip | `1 failed, 1 passed, 1 warning in 87.87s` |
| receipt citation gate | PASS, 11 citations, zero failures/unmapped citations; the final clean-tree stamp is recorded in `citation_gate_final.json` |
| shifted-citation plant | FAIL as required, exit 1, `SYMBOL-NOT-AT-LINE` at the planted R101 first endpoint |

The combined tree run never produced a pytest summary and therefore is not
represented as a pass or as a meaningful failure count. Its complete log
contains ten independent `Fatal Python error: Aborted` stacks in JAX
`backend_compile_and_load`, across unrelated tests, followed by a stalled
replacement after more than 95 percent progress. This is the large-suite
per-process compiler-limit failure mode documented in the repository
orientation. It was not hidden or retried as another monolithic run.

The two operator-named red nodes were instead run together in a fresh process
at both trees. Both trees fail only
`test_every_report_emitter_stamps_the_worktree`; their nine-entry offender
lines are byte-identical (`cmp` exit 0, SHA-256
`07475992194326f75221d06f968b2c25ec6629e9646813dae625acda07e2ebaf`).
`test_rk3_ws_differs_from_rk3_and_is_finite`, although documented as an
incoming red, passed in both fresh runs on this machine. The round claims only
what was observed: it does not relabel that intermittent test as repaired.

## Canonical evidence hashes

| artifact | SHA-256 |
|---|---|
| `local_before_jit.json` | `9dd0474ae8418ea1f41735761ff065f274cdb5ac860d9281bd397f7a38383256` |
| `local_before_eager_retry.json` | `a1171aa16057f65410abe4a17ad52562a4524b6dfb864a707318bd086a5bd856` |
| `local_after_jit.json` | `80def0c96e9d997e4434069056c6098e3f20d9b125db9e41f0d695a785ed2921` |
| `local_after_eager.json` | `c19fb18b0e9e3a16654ae4defd34d57d5e82be15919dddf334fbe7790b4f6827` |
| `local_after_plant_retry.json` | `5071598cd4069a2d9d481b1d2052dab7818178ce7f4ad2b2a9b8a38529b90210` |
| `dino_routing_before_v2.json` | `5f68b0120e012ff55911548dec43b508833cb833ac69cbb46715f407262dfda3` |
| `dino_routing_after.json` | `4a9721ab7188218e6e5734327578abf4a40ac4bf55acac35a3de4cdd79534c4e` |
| `dino_routing_plant_after.json` | `e31b76aa0637fb5be2f959035345259aa62eb68ea531eb5298f113795a22ed66` |
| `ladder_full.json` | `2b1cccd765cb470c17bc7467fd1caa2b7e6b3afdd5165081bbee6b072ea40431` |
| `ladder_full_rule12.json` | `43ace522660d2c22f81438562ee8228dc1a529be28203e17b9a84f26a05cbe7` |
| `day_gap.json` | `7188e59c073580e4914ab5235794cb3d8bfa3a9b6ee7fb03935eaac5bf773839` |
| `stage_twin_restored.json` | `9fbbfaa039eda1101e879f26b4d2b106e9616107b69987e30e0709e887c725b8` |
| `rule12_self_control.json` | `0a571b55fe618c7353fb03bd2ef2ef4d132c247efc29bdbe264c59ecaf54b78c` |
| `rule12_self_plant.json` | `2bd439518594583366140b4f6cba7d2067d27c8cf68abbb4a5c812229094f5c2` |

## OPEN - round 106

1. **Stay at the same production-JIT TKE stage boundary.** Reapply the held
   round-105 routing patch only in a preregistered measurement arm so `p_sh2`
   is BIT, reproduce eager `en_rhs` BIT and production-JIT 11,024 unequal at
   `5.551115123125783e-17`, then walk the compiled right-hand-side expression
   on R101 lines 439--442 in source order through the **full production step**.
   Distinguish operand association/materialization from XLA fusion with one
   variable per arm. An isolated JIT closure is not production evidence.
2. **Do not re-promote the routing split on its shear proof.** It remains held
   until the production-JIT RHS boundary is BIT, the complete stage output
   condition closes, and the 954-row Rule-12 gate passes. Its current
   aggregate improvements are not a landing argument.
3. **Respect the global stage order.** The restored stage twin still names
   kt1 stage-1 U as the first owned output (7,620 cells at
   `5.421010862427522e-20`). No later stage may land while that row remains
   owned and non-bit. The RHS discriminator is permitted only as continuation
   of the already-authorized same vertical-physics sub-walk, not as permission
   to land downstream physics.
4. The round-104 inline shear-ratio divide versus shared
   multiply-by-reciprocal difference remains measured inert on this record;
   do not bundle it. `p_pdlr` remains UNMEASURED-WITH-SPEC.
5. DINO's full production shear result remains blocked upstream in GM/Redi;
   helper hashes are not a substitute. ORCA2 retains the one-step spec above.

No NEMO acquisition is requested. No configuration or carried-state decision
is requested.
