# Round 248 — Redi routing review disposition receipt

**Status: LANDED.** Base `0575754c0`; preregistration commit `a1d6f5560`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round248/`. This round changes
no NEMO source, numerical statement, card parameter, carried state, record, or
acceptance threshold. It makes the already-landed isoneutral tracer-diffusion
routing explicit and mechanically reviewable.

## 1. Compiled statement and review findings

The compiled SMT-3 program closes the W-point coefficient below the deepest
wet tracer cell at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
Its ordinary and deepest-level divergence statements divide by the live
`e3t(Kmm)` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310` and
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.

The draft-PR review found:

1. `closed_bottom_wmask=True` was an undocumented default with only a
   low-level override test.
2. The live T-point divisor was selected indirectly by the existing
   horizontal face-thickness selector.
3. Reverting either the production hook or dispatcher construction did not
   make that low-level test fail.

All three are closed. The low-level default remains `True`, as Decision 104
requires, and its docstring now names the compiled statement. An AST test pins
the complete package caller census: two production-model calls, one box-budget
call, one tendency-probe call, and one shared-wrapper call. A new
`redi_divisor_thickness_evaluation` field independently selects
`"reference_jacobian"` or `"nemo_qco_live"`; its library default is the former,
which is the pre-round-237 behavior. Unknown values fail loudly.

The production RK3 test observes a non-null stage-3 divisor SSH at the model
hook. The dispatcher test proves that live divisor thickness can be built while
the face construction remains the reference/Jacobian arm. It also pins the
pre-split precedence: an explicit divisor SSH wins; otherwise an already-routed
face SSH wins; only then may the primary dispatcher SSH be used.

## 2. Explicit execution census

| card family | executes isoneutral Redi | explicit divisor selection |
|---|---:|---|
| certified GYRE-zco and generic NEMO-GYRE recipe | yes | `reference_jacobian` |
| VORTEX_SMT3_VEC-zps, VORTEX_SMT4_VEC-zps | yes | `nemo_qco_live` |
| `nemo_dino_kamm`, `nemo_dino_kamm_mlf` | yes | `nemo_qco_live` |
| all other DINO recipes | card-dependent, non-NEMO program | explicit/default `reference_jacobian` |
| six flat VORTEX cards and SMT-0/1/2 | no (`gm_redi is None`) | not executed |

No executing NEMO card inherits a hidden numerical choice. The DINO field is
validated and threaded through every DINO GM/Redi construction; its two NEMO
recipes select the live statement explicitly. The VORTEX card validator and
the generic GYRE recipe test pin their selections.

## 3. Refuted intermediate result

Preregistered P4 predicted that merely separating the selectors would preserve
DINO's registered month value. The first clean committed arm (`9fd4c9222`)
produced `2.053801170e-03 K`, so that prediction is **REFUTED** and retained in
`dino_month.log`. The new selector had fallen back to the dispatcher's primary
SSH when DINO supplied no distinct divisor SSH; before the split, the divisor
had inherited DINO's independently routed face SSH. This was a routing
regression, not a new NEMO choice.

Commit `2fb92e4b3` restores that pre-split operand precedence and strengthens
the dispatcher test so the two inputs differ. The clean final DINO replay
reports:

> `DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.056821682e-03 K against bar 2.244317642e-03 K ... -- PASS`

The registered `+0.147%` DINO value is therefore reproduced exactly.

## 4. Fail-closed controls

Both required reverse plants run against the final implementation:

| removed behavior | decisive failure | process result |
|---|---|---:|
| production RK3 `redi_divisor_eta` hook | `assert seen[0] is not None` | `ROUND248_PRODUCTION_HOOK_PLANT_FIRED exit=1` |
| dispatcher live-divisor construction | `assert live[2] is not None` | `ROUND248_DISPATCHER_PLANT_FIRED exit=1` |

The logs are `production_hook_plant_final.log` and
`dispatcher_plant_final.log`. The production files were restored byte for
byte after each plant; `git diff --check` is clean.

## 5. Certified trajectories

The final clean-tip GYRE ladder compares all 954 rows with the round-247 arm:
zero rows move, maximum worsening is zero ULP, and first-over-bar remains kt=3.
The clean-tip 360-day member reproduces every registered score and snapshot;
the headline rows are:

| day | round 247 T3D RMS [K] | round 248 T3D RMS [K] | move |
|---:|---:|---:|---:|
| 30 | `2.3432419318363155e-06` | `2.3432419318363155e-06` | 0 |
| 240 | `6.5816987106668941e-05` | `6.5816987106668941e-05` | 0 |
| 360 | `5.4077212586815052e-05` | `5.4077212586815052e-05` | 0 |

SMT-1 through SMT-4 reproduce 3,200/3,200 daily score scalars and every
50-row registry on the final clean tip:

| card | day-100 T3D RMS [K] | registry |
|---|---:|---|
| SMT-1 | `4.3321114781972461e-05` | reproduced |
| SMT-2 | `8.1037591477894766e-06` | reproduced |
| SMT-3 | `1.7729713625071864e-04` | reproduced |
| SMT-4 | `2.5527080520554426e-04` | reproduced |

LOCK_EXCHANGE-zco and OVERFLOW-zps each reproduce 50/50 rows with zero cell
movement and unchanged first-over-bar (kt=8 U and kt=2 T/U respectively). The
tank runs were made at `9fd4c9222`; the only later production change restores
DINO's divisor-SSH fallback and cannot execute on either no-Redi tank. The
generic NEMO-GYRE recipe is exercised again in the focused and push batteries.

There are no moved certified rows to register. The full reports are
`gyre_ladder_final.json`, `gyre_compare_full_final.json`,
`gyre_day_gap_final.json`, `smt_final_exact_compare.log`, `dino_month_final.log`,
and `tanks/*` under the evidence root.

## 6. Citations, tests, and independent review

The default citation gate and this receipt's gate both pass with zero unmapped
citations, map-audit failures, or claim failures. The shifted-citation plant
exits nonzero. The final focused suite reports:

> `7 passed in 38.26s`

The prescribed push-gate battery reports:

> `PENDING`

The separate read-only Codex review was attempted on clean committed tree
`2fb92e4b3`. Independent review is unavailable in-sandbox; its complete output
is quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
> Reading additional input from stdin...
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

There is no `SHIP` or `DO NOT SHIP` verdict. The standing operator instruction
permits the mechanically gated round to continue when the review cannot start
inside the sandbox.

## 7. Verdict and OPEN

Decision 104 is implemented without changing any certified trajectory. The
closed-bottom default is documented and census-controlled; the live Kmm
divisor has an independent fail-closed selector; every executing NEMO card
states its choice; and both production-routing reverse plants fire.

**OPEN:** the operator's Codex re-review of draft PR #1910. No fidelity walk,
configuration decision, NEMO acquisition, or numerical debt is opened by this
round.

`ACQUISITION_NEEDED: NONE`

`DECISION_NEEDED: ready for the operator's codex re-review`
