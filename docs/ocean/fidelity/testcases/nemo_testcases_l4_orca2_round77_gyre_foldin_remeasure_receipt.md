# ORCA2 round 77 — folding the GYRE lane in, and re-measuring ORCA2 under it

Status: **SHIP the fold-in.** The four conflicts resolve from what the merged
tree actually holds, no map entry from either lane is dropped, and the citation
gate is green with its plants firing. The ORCA2 card now states the after-SSH
form on the card rather than inheriting it, and the three other fields the
order named were already stated and are checked rather than re-written.
ORCA2's ladders move, and the field that moves them is named and arrives with
the merge, not from any choice taken here.

## 1. Scope and provenance

Merge commit `ff08f47a9`, parents the ORCA2 lane `1142d4182` and the GYRE lane
`ea12107cb` (GitHub `github/fidelity/nemo-testcases-l2-gyre-codex2`). Merge
base `d3631f884`: the ORCA2 lane carries 643 commits on it, the GYRE lane 455.
Ordinary merge, two parents, no squash and no rebase.

Commit range `1142d4182..<tip>`: the merge, the stated field with the
re-pinned certified digests, the preregistration, and this receipt.

Preregistration
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round77_gyre_foldin.md`,
committed while the ten-step ladders were still running and before any ladder
number of this round existed. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round77_foldin/`.

## 2. The four conflicts

| # | file | what each side held | resolution, and what decides it |
|---|---|---|---|
| 1 | the Decision-43 admission gate | three hunks, each the same list with one lane's own source route in it: ORCA2's stage-1 r3t ratio route, the GYRE lane's carried step-entry mixed-layer route | **UNION.** The two routes name different statements and neither list is a choice about physics; keeping one would delete a lane's admission path. Both are selectable, in the route set, the dispatch chain and the command-line choices alike. |
| 2 | the citation gate, hunks 1 and 20 | blocks of map entries only one lane ever wrote: ORCA2's OVERFLOW and ORCA1ICE compiled spans, the GYRE lane's round-184 DINO fold spans | **UNION**, both blocks kept. Hunk 20's ORCA2 side already contains the GYRE side's three entries verbatim plus its own round-61 and round-63 observers. |
| 3 | the citation gate, the other 22 hunks | the SAME entries with each lane's own line numbers, and in two places with each lane's own pinned extent | **Neither side.** The merged file is longer than either parent, so both sides' numbers are wrong in it. Every entry is re-anchored BY SYMBOL against the merged tree: each pinned anchor is resolved in the merged file, the key's endpoints are rewritten from where the anchors actually are, and the pinned extent is recomputed from the resolved endpoints. 74 entries moved; the map's own audit then reports nothing failing. |
| 4 | the two receipts (`..._phase3_round8_receipt.md`, `..._round162_cancellation_t_ratio_receipt.md`) | fifteen and two hunks, in every case the same sentence with a different line number | **Neither side**, same method: the prose citations move with the map, driven by the resolved anchors. Six further receipts that the re-anchor touched moved in the same commit, which is what the lane's re-anchor rule requires. |

**Nothing was dropped, and that is structural rather than asserted.** Both
parents' maps were parsed and reduced to anchor signatures — the file plus the
pinned anchor texts, with the line numbers removed, because the line numbers
are exactly what the merge invalidates. Every signature held by either parent
is present in the merged map: `ours-not-in-merged: 0`,
`theirs-not-in-merged: 0`. The merged map holds **1295** entries against
**1146** on the ORCA2 lane and **1099** on the GYRE lane.

### The citation gate after the re-anchor

| run | result |
|---|---|
| `audit_map()` on the merged tree | **0 entries failing**, from 74 before the re-anchor |
| the gate's nine self-test plants | all nine fired |
| the default cumulative receipt | `PASS`, 274 citations, 0 unmapped, 0 failures |
| the default receipt with a planted two-line shift | exits 1 |
| this lane's newest receipt (round 77's external transport pair) | `PASS`, 4 citations |
| the same with a planted shift | `FAIL`, `SYMBOL-NOT-AT-LINE` |
| the GYRE lane's newest receipt (VORTEX round 5) | `FAIL` on one unmapped citation, `stp2d.f90:153` — and it fails **identically at the GYRE lane's own tip**, measured there. Pre-existing, not caused by the fold-in; that receipt's NEMO citations are written without a build prefix, which no map entry carries. |

## 3. The fields this card now states

| field | what the card holds | how it got there |
|---|---|---|
| `nemo_first_wzv_after_ssh` | `rk3_extrapolated` | **WRITTEN on the ORCA2 card this round.** It resolved to the same value before, by inheriting the shared GYRE identity the ORCA2 identity specialises; Decision 76 asks for it on the card. The resolved value does not change, so no number moves on this account. |
| `nemo_derived_mxl_min` | `False`, with `mxl_min = 1e-3` | already stated on the card (Decision D1's "keep and re-certify"). NEMO forces `rmxl_min = 1e-3` on this deck because it resolves `ln_zdfiwm = .TRUE.`, and the derived expression is never evaluated there. |
| `enforce_cfl` | harmonic `False`, biharmonic `True`, lateral mixing `scheme = "none"` | already stated, and now GATED: the card is built twice with every `enforce_cfl` default it could inherit flipped in between, and its resolved lateral-mixing block does not move. The flip harness is shown to move something by its own existing non-vacuity test. |
| `nemo_stage_momentum_wzv_split` | `True` | confirmed unchanged from round 23 (Decision 58). It is `False` on the GYRE lane's copy of this card, which is exactly the difference Decision 58 made; the fold-in keeps this lane's value. |

**ORCA2 does run the RK3 vector-invariant program, read from its own build and
not inferred.** Its resolved configuration reports `momentum_time_integrator =
rk3_ws` and `momentum_advection = vector_invariant`, and the compiled source of
the build the ladder scores against agrees: `stp2d.f90:145-149` takes the
"Vector Inv. Form" Coriolis arm and the "only KEG + ZAD in Vector Inv. Form"
advection, `stprk3.f90:239-241` leaves `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) -
ssh(:,:,Naa)` in the after slot, `stp2d.f90:152` turns that slot into
`r3t(:,:,Kaa) = ssh(:,:,Kaa) * r1_ht_0`, and `stp2d.f90:156` is the first
`CALL wzv( ..., np_velocity )`. So `rk3_extrapolated` is the value this build
implies as well as the one the card resolves.

**One limit, stated rather than implied.** The new test gates the card's
resolved VALUE, which is what the run executes. It cannot fail if the explicit
line is deleted again, because the shared identity supplies the same value —
that placement is carried by this commit and the comment beside it, not by a
gate.

## 4. Configuration fingerprints

Each number is the certified-card digest: the printed model configuration
together with the initial state, the grid's Coriolis fields, the vertical
coordinate, the land mask and the card's own deck fields.

| card | pre-merge ORCA2 tip `1142d4182` | merged tip |
|---|---|---|
| GYRE-zco | `55116336c6fe349b` | `c051113f8fe759ec` |
| LOCK_EXCHANGE-zco | `8e00d6f102896f44` | `4ec82201f0fae7bf` |
| OVERFLOW-zps | `30d7c10b621e4d1d` | `7f1ec89b3e6578f5` |
| **ORCA2-zps** | `d4038559fbf44bbd` | `b4184d71c4b96e4b` |
| VORTEX-zco | card does not exist | `1391b9a601f48110` |
| VORTEX_VEC-zco | card does not exist | `29865c1001698398` |

All four move, as the order expected. WHICH fields move them was measured, not
assumed: every card's resolved configuration was flattened to one leaf per
dotted path and diffed.

**ORCA2-zps, pre-merge tip → merged tip: two values changed, four fields
removed, seven added.**

| what | field | before → after |
|---|---|---|
| CHANGED | `physics.vertical_mixing.tke.mxl_min` | `1e-08` → `0.001` |
| CHANGED | `physics.lateral_mixing.biharmonic.enforce_cfl` | `False` → `True` |
| REMOVED | `physics.bottom_drag.linear.r`, `physics.bottom_drag.quadratic.C_d`, `physics.lateral_mixing.biharmonic.cfl_dt_estimate`, `physics.vertical_mixing.kpp.c_b` | fields `main` deleted; ORCA2 selects none of those blocks |
| ADDED | `nemo_first_wzv_after_ssh='rk3_extrapolated'`, `physics.vertical_mixing.tke.nemo_derived_mxl_min=False`, `.nemo_mxl0_rmxl_min_overwrite=True`, `.nemo_mxl0_surface_tmask=True`, `.mxl0_min_m=0.04`, `barotropic.nemo_prognostic_barotropic_state=True`, `eos_nemo_seos=None` | all from the GYRE lane |

**The one that moves ORCA2's numbers is the turbulence mixing length**: the
floor goes from `1e-8 m` to NEMO's forced `1e-3 m`, a factor of 100 000, and
three further mixing-length statements arrive with it. The CFL cap sits on a
block this card does not select (`lateral_mixing.scheme = "none"`), and the
four removals are fields that no longer exist anywhere.

**GYRE, both tanks and both VORTEX cards, GYRE lane tip → merged tip: ONE
field added, none removed, none changed.** The field is
`lateral_viscosity_coefficient_source`, which ORCA2 needs because it resolves
`nn_ahm_ijk_t = -30` and reads that coefficient whole from a file; every other
card resolves it to `nemo_ldf_c2d`, the metric formula they already ran. The
only other row in that comparison is ORCA2's own `nemo_stage_momentum_wzv_split`
`False → True`, which is Decision 58 and belongs to this lane.

That is why the three certified digests are re-pinned here: a digest prints
every field, so one additive field moves it while nothing executed changes.
The added field's value is now asserted on each certified card beside its
digest, so a future edit that gives one of them ORCA2's file-read arm turns
that test red instead of moving a number quietly.

## 9. Choices made this round

| choice | status |
|---|---|
| resolve the Decision-43 route conflict as a union rather than by preferring a lane | NOT A CHOICE about physics: dropping either route deletes a lane's admission path, and neither route selects anything by itself |
| resolve the 22 same-entry citation hunks by re-anchoring rather than by taking a side | NOT A CHOICE: both sides' line numbers are wrong in the merged file, and the gate's own audit decides what is right |
| write `nemo_first_wzv_after_ssh` on the ORCA2 card | ASKED — the order for this round, Decision 76. The resolved value is unchanged |
| re-pin the three certified card digests | ASKED in effect by the same order ("fingerprint before/after for every card"), and taken only with the field-by-field proof that ONE additive field moved them |
| keep `nemo_derived_mxl_min = False` with the `1e-3` floor | ASKED — Decision D1 |
| keep `nemo_stage_momentum_wzv_split = True` | ASKED — Decision 58, round 23 |

UNASKED: none.

UNCHANGED and not touched: the six sea-ice selectors, the carried state, every
threshold and cadence, the forcing, the NEMO sources, and the first-non-bit
statement walk.

## 5. ORCA2, before and after — the ladders

Both ladders were run on BOTH trees in this round, with identical inputs, from
clean committed trees: the pre-merge ORCA2 tip `1142d4182` and the merged tip.
Nothing here is carried over from an older round's record.

### The ladder "given NEMO's entry" (the Decision-52 bridge)

| row | before | after | direction |
|---|---|---|---|
| rows at bar / debt | 5 / 195 | 5 / 195 | unchanged |
| first non-bit statement | kt 1, stage 1, `T` | kt 1, stage 1, `T` | **unchanged** |
| kt 1 entry, all five fields | bit-identical | bit-identical | unchanged — the entry bridge is untouched (P1 CONFIRMED) |
| kt 1 stage 1 `T` max | `0.0014770192519700243` | `0.0014770192519700243` | unchanged |
| kt 1 stage 2 `u` max | `0.06470386625377142` | `0.06470386625377142` | unchanged |
| kt 10 entry `T` max | `1.2745209751477056` | `1.1020016057695017` | toward, −13.5% |
| kt 10 entry `T` rms | `0.0101522815010971` | `0.01009415351510639` | toward, −0.57% |
| kt 10 stage 3 `u` max | `0.3384137672623481` | `0.33986170691596` | **AWAY, +0.43%** |
| kt 10 stage 3 `v` max | `0.5353621181458693` | `0.5327649846470696` | toward, −0.49% |
| kt 10 stage 3 `T` max | `0.9838385161101275` | `1.114220520669864` | **AWAY, +13.3%** |
| kt 10 stage 3 `S` max | `0.22099092586135072` | `0.25508044621046366` | **AWAY, +15.4%** |

### The INDEPENDENT ladder (the card's own initial state)

| row | before | after | direction |
|---|---|---|---|
| rows at bar / debt | 4 / 196 | 4 / 196 | unchanged |
| first non-bit statement | kt 1 entry `ssh`, `iceistate.f90:440-465` | the same | **unchanged** |
| kt 1 entry `ssh` | 16,433 / 26,640, max `0.015479333813968585` m | **identical to the last digit** | unchanged (P2 CONFIRMED; it also reproduces round 66) |
| kt 1 stage 1 `T` max | `0.0014771067613864597` | identical | unchanged |
| kt 10 entry `T` unequal | 430,552 / 799,200 | 430,552 / 799,200 | unchanged |
| kt 10 entry `T` max | `1.2654113454705964` | `1.1020348928205674` | toward, −12.9% |
| kt 10 entry `T` rms | `0.010149097763919606` | `0.010093213342619738` | toward, −0.55% |
| kt 10 stage 3 `T` max | `0.984131770365277` | `1.1143079975665042` | **AWAY, +13.2%** |
| kt 10 stage 3 `S` max | `0.22106498534098762` | `0.25510075611161653` | **AWAY, +15.4%** |

### Every row, both ladders

Both ladders give the SAME census, which is itself a check that the two
initial states do not change the attribution: **200 rows, 24 unchanged, 77
toward NEMO, 99 away.** **No bit-identical row left the bar and no row reached
it**, and the first non-bit statement is the same one in both arms before and
after. The largest single move is kt 7 stage 1 `v`, `0.4381` → `0.5646`
(+28.9%); the `S` and `T` rows from kt 7 on move away by 10 to 15 per cent.

**So ORCA2 does worsen on headline rows, and the round says so rather than
reporting only the two that improved.**

### Which field owns it — one measured refutation, one named candidate

The preregistration predicted the turbulence **mixing length** would own the
movement: it is the only CHANGED value on a block ORCA2 selects, and it moves
by a factor of 100 000. **That prediction is REFUTED, by a one-variable control
rather than by argument.** A committed arm (`r77-mxl-arm`, commit `91daa26f5`,
the merged tip with `mxl_min` back at `1e-8` and both `nemo_mxl0_*` statements
off) was run through the same independent ladder:

| comparison | result |
|---|---|
| the arm against the merged tip | 145 of 200 rows differ, **maximum relative difference `2.0e-04`** |
| the arm against the pre-merge tip | the same 200-row census: 24 unchanged, 77 toward, 99 away — i.e. the arm reproduces the FOLD-IN, not the pre-merge ladder |

A group that moves the ladder by two parts in ten thousand cannot own moves of
ten to thirty per cent. The mixing-length floor is NEMO's own forced value on
this deck and it stays; Decision D1 is re-certified in the sense that asked
for — it was measured on this card and it is very nearly inert here.

**The second candidate is Decision 76's first-`wzv` after-SSH statement**,
which the GYRE lane landed in VORTEX round 5 and which ORCA2 executes: before
the fold-in this card built the first `wzv` call's scale-factor term from a
continuity prediction, and after it the term comes from the previous step's
linear extrapolation. That is the statement the operator's note says blocks
this lane's ladder claims until re-measurement. **It is REFUTED too**, by its
own committed arm (`r77-afterssh-arm`, `8d32f0564`, the merged tip with the
card's after-SSH form put back to `leapfrog_continuity`):

| row | pre-merge | merged tip | after-SSH arm |
|---|---|---|---|
| kt 10 stage 3 `T` max | `0.984131770365277` | `1.1143079975665042` | `1.1142389733330837` |
| kt 10 stage 3 `S` max | `0.22106498534098762` | `0.25510075611161653` | `0.25508515293330447` |
| kt 10 entry `T` max | `1.2654113454705964` | `1.1020348928205674` | `1.1020081449939436` |

The after-SSH form is NOT inert — it moves 175 of 200 rows against the merged
tip and one of them by 66 per cent — but it does not take the ladder back:
176 of 200 rows still differ from the pre-merge arm, and the headline kt=10
rows move by less than one part in ten thousand. **So neither named candidate
owns the fold-in's ten-to-thirty-per-cent moves**, and the round says so
rather than picking the more convenient of the two.

**What is left, stated as the search space rather than as a guess.** The
merge brings 455 GYRE-lane commits onto this card's shared code. The two
config-visible candidates are measured and refuted; the owner is therefore a
shared MODEL statement that leaves no trace in the resolved configuration.
Ranking that list is the next round's work, and the two arms above are the
method: one variable, on the record, with the ladder as the read-out.

## 6. GYRE, the tanks and DINO — measured on this branch

Every row below was run from this branch's own clean committed tree, not
quoted from the GYRE lane's receipt.

| card | this branch | the GYRE lane's certified value | verdict |
|---|---|---|---|
| LOCK_EXCHANGE-zco, ten steps | `DEBT`, first over bar `{u}` at kt 4, residuals sha256 `a28d1f8ac5482b0f…` | `DEBT`, `{u}` kt 4, `a28d1f8ac5482b0f` | **byte-identical** |
| OVERFLOW-zps, ten steps | `DEBT`, first over bar `{T,u}` at kt 2, residuals sha256 `43e297c25b34b950…` | `DEBT`, `{T,u}` kt 2, `43e297c25b34b950` | **byte-identical** |
| DINO from-rest month | `2.040288957e-03 K` against bar `2.244317642e-03 K`, exit 0, private work directory | certified `2.040288765e-03 K` | **PASS**, `1.9e-10 K` apart, which is the harness's own run-to-run floor — inert |
| GYRE ten-step trajectory | `DEBT`, first over bar `{T,S,u,v,ssh}` at kt 3 | — | ran clean; not yet compared residual-for-residual against the GYRE lane's own arm (OPEN 4) |
| **GYRE from-rest year, day 30** | `2.3440e-06` K | `2.3440e-06` K | **equal** |
| **GYRE from-rest year, day 240** | `6.5826e-05` K | `6.5826e-05` K | **equal** |
| **GYRE from-rest year, day 360** | `5.4085e-05` K | `5.4085e-05` K | **equal** |

The two tank residual files are a stronger claim than a metric: the whole
`.npz` hashes to the same bytes the GYRE lane certified, so those cards do not
merely agree to a tolerance, they agree exactly. **P4 holds for both tanks and
for DINO.**

## 7. Gates

| gate | result |
|---|---|
| citation gate, default cumulative receipt | `PASS`, 274 citations, 0 unmapped, 0 failures, 0 map entries failing audit, nine self-test plants fired |
| the same with a planted shift | exits 1 |
| citation gate, this lane's newest receipt | `PASS`; with a planted shift, `FAIL` with `SYMBOL-NOT-AT-LINE` |
| citation gate, the GYRE lane's newest receipt | `FAIL` on one unmapped citation, identically at the GYRE lane's own tip — pre-existing |
| `tests/ocean/unit/test_nemo_vortex_card.py` and `tests/ocean/unit/test_nemo_card_opt_in_defaults.py` | `70 passed`, including the re-pinned certified digests and the new ORCA2 decision-75/76 test |
| both lanes' push-gate file sets, run together as ONE battery | **`206 passed`**, with only the two citation-gate tests red on the clean-tree stamp because the receipt was still being written; re-run after the receipt commit, see below |
| the ORCA2 lane's five files | `test_nemo_testcase_receipt_citation_gate.py`, `test_tke_nemo_terms.py`, `test_nemo_recipe.py`, `test_real_freshwater_closure.py`, `test_nemo_testcase_l2_orca2_parallel_receipt_gate.py` |
| the GYRE lane's files | the same three shared ones plus `test_nemo_ws_stage_face_mask_rank.py` and `test_nemo_prognostic_barotropic_state.py` |
| the two card files | `test_nemo_vortex_card.py`, `test_nemo_card_opt_in_defaults.py` |

### 7b. The citation gate on a committed tree

Re-run after this receipt was committed: `17 passed` (the new duplicate-key
test is the seventeenth), `map entries failing audit: 0`, every plant firing.

## 8. Review

`codex exec --sandbox read-only` on the merge resolution and on the post-merge
commits. Verdict, quoted:

> **HOLD — citation integrity is broken; the configuration claims otherwise
> survive static review.**

Three findings, all real, all closed in this round:

1. **Seven citation-map keys were written twice**, each with a different anchor
   from the one that survived, so the earlier anchor was silently discarded by
   the dict literal. It predates the fold-in — BOTH parents carry the same
   seven — and it had been invisible because the audit walks the BUILT
   dictionary. The seven dead lines are removed (a no-op, since Python already
   ignored them) and a test now reads the source and refuses a repeated key.
   **The "union, nothing dropped" claim is re-stated more carefully**: measured
   at the LITERAL level, including the shadowed entries, every anchor
   signature held by either parent is present in the merged file (0 and 0).
   What the duplicates broke was not the union, it was the audit's coverage.
2. **Stating the after-SSH form left six map entries stale**, because the
   comment block inserted thirteen lines. Re-anchored by symbol, with the
   receipts' prose, in the same commit.
3. **One ORCA2 receipt kept three pre-fold-in spans**
   (`2763-2787`, `9336-9343`, `9495-9499`). Rewritten to the lines their own
   symbols identify: `2806-2830`, `9506-9513`, `9665-9669`.

The reviewer confirmed the rest: both Decision-43 routes survive, no extent is
wrong, nothing in the resolution changes an executed number, the digest
attribution and the fingerprint script check out, and the CFL half of the new
test is non-vacuous. It also handed over a cheap way to make the
stated-versus-inherited half non-vacuous — poison the shared identity builder
and require the card to resolve its own value — which is now in the test.

## OPEN

1. **The ORCA2 ten-step ladder is NOT re-certified.** 99 of 200 rows moved
   away from NEMO at this fold-in, in both the given-entry and the independent
   ladder. No bit-identical row left the bar and the first non-bit statement is
   unchanged, so nothing regressed at the bar — but the ladder's magnitudes are
   a different ladder from round 77's and every magnitude claim on this lane
   older than this round is superseded.
2. **Name the owner.** BOTH named candidates are refuted by their own
   one-variable arms: the mixing-length group moves the ladder by two parts in
   ten thousand, and the first-`wzv` after-SSH form moves many rows but does
   not take the headline ones back. The owner is a shared MODEL statement that
   the resolved configuration does not show, somewhere in the 455 GYRE-lane
   commits this merge brings across. Next round: rank those by what they touch
   on the ORCA2 card's executed path, and walk them with the same method.
3. **The independent month is NOT re-measured.** Round 71's ranking (terminal
   `T` max `21.637832697714707` K, rms `0.17564842520962015` K at 240 steps)
   was measured at `46fe3a2e`, and `git diff` reports NO change to `packages/`
   or `src/` between that commit and the pre-merge tip, so it is a valid BEFORE
   for this fold-in. The AFTER costs about 2.6 hours of CPU (round 71's own
   `wall_seconds` is 9297) and did not fit this round.
4. **The GYRE ten-step ladder's residual digest is not yet compared
   like-for-like** against the GYRE lane's own arm at `ea12107cb`; the tanks
   and DINO are, and they are byte-identical.
5. The certified-card digest for GYRE-zco depends on global precision-policy
   state left behind by an earlier test in its module: measured standalone it
   is `c051113f8fe759ec`, measured under `pytest` it is `db95b2a4d1989f93`.
   The pin follows the value the test itself measures. The two tanks do not
   show this. Worth one round to make that digest order-independent.
6. Everything round 77's own receipt left open — the external transport pair
   walk — is untouched here.
