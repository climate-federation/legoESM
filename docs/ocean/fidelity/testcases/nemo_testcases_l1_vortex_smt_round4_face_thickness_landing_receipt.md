# Round 214 / VORTEX_SMT round 4 — the reference U/V face thickness LANDS

**VERDICT: the statement is landed.** legoESM's one shared vertical-mesh
operand builder aliased the reference U/V face thicknesses to the T thickness
for every card carrying NEMO's raw mesh. NEMO's reference face thickness is
the shallower neighbour's. The repair reads the card's own NEMO-verified
arrays — no re-derivation, which is what the round-213 reviewer refused — and
the seamount card's resolved faces now equal NEMO's recorded ones bit for bit,
northern row intact.

**Decision 88 (user), operator note CC addendum 4.** Predictions were frozen
before any edit:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round4/predictions.md`.

---

## 1. PRE-IMPL SEARCH (RULE 4) AND THE PATH TABLE

Searched, and what was found:

```
grep -rn "nemo_qco_resolved_mesh_operands" packages/ src/ scripts/ tests/
grep -rn "e3u_0=e3t\|e3v_0=e3t\|e3f_0=e3t\|e3w_0=e3t" packages/ src/
grep -rn "nemo_een_barotropic\|nemo_e3u_0\|nemo_e3v_0\|nemo_e3f_0\|nemo_e3w_0" packages/ src/
```

One builder, `nemo_qco_resolved_mesh_operands`
(`packages/ocean/legoesm/ocean/vertical.py:673`), reached from one production
call site (`ocean_pe_latlon_cgrid.py:1695`, inside `nemo_qco_wzv_operands`)
and one diagnostic one (`:1519`). Exactly one alias in the package tree
(`vertical.py:701`) plus one in the DINO analytic mesh
(`nemo_dino_mesh.py:397`), which is a different and correct statement (§1.2).
`e3f_0` is NOT in this builder's operand set and `nemo_e3w_0` is a separate
raw field, so the statement is `e3u_0`/`e3v_0` only; both were checked and
are reported below.

**WHICH CARD TAKES WHICH PATH — read off the code, with line numbers, not
assumed.** The branch is chosen by whether the card carries all eight raw qco
operands (`nemo_e3t_0`, `nemo_hu_0`, `nemo_hv_0`, `nemo_e1e2t`, `nemo_e1e2u`,
`nemo_e1e2v`, `nemo_e2u`, `nemo_e1v`); a card carrying some but not all is a
hard error, and a card carrying none rebuilds them from its own grid.

| card | attaches the raw set at | takes | carries NEMO's own `e3u_0`/`e3v_0` at | changed by this round? |
|---|---|---|---|---|
| `GYRE-zco` | `nemo_testcase_recipe.py:1119-1128` | raw | bundle, `:1128` | reaches it, **bitwise inert** (§3) |
| `ORCA2-zps` | `:1603-1614` | raw | bundle, `:1579-1581` (real partial faces from `domain_cfg`) | **yes — operand moves** (§5) |
| `VORTEX{,_VEC}-zco` and the 15/10 km rungs | `:2276-2285` | raw | bundle, `:2285` | reaches it, **bitwise inert** |
| `VORTEX_SMT{,_VEC}-zps` | `:2516-2522` | raw | bundle, `:2522` | **yes — the statement** |
| DINO (bridge, both entries) | `nemo_state_bridge.py:259-268`, `:888-897` | raw | bundle via `_nemo_een_barotropic_operands` `:71-90` | reaches it, **inert** (§1.2) |
| `LOCK_EXCHANGE-zco` | — (`:934`) | rebuild | n/a | **cannot reach the lines** |
| `OVERFLOW-zps` | — (`:983`) | rebuild | n/a | **cannot reach the lines** |

### 1.2 DINO, proved rather than assumed

DINO's analytic mesh sets `e3u_0 = e3v_0 = e3t_0` at
`nemo_dino_mesh.py:397` — the SAME object, not a coincidence — and says why
one line above (`zgr_lib.F90:206-264`; on a horizontally uniform ladder the
neighbour average, and equally the neighbour minimum, IS `e3t_0`). The
file-read entry (`nemo_state_bridge.py:259`) takes `grid.e3u_0` straight out
of `mesh_mask.nc`, and the same bridge refuses a non-flat bathymetry outright
(`nemo_state_bridge.py:240-247`), so the mesh it serves is full-step and the
minimum again IS `e3t_0`. DINO therefore cannot move; the month gate was run
anyway (§6).

---

## 2. THE REPAIR, AND WHAT IT CITES

One variable, in the one shared builder. NEMO builds the reference face
thickness once, in the domain builder:

```
pe3u(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji+1,jj,jk) )
pe3v(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji,jj+1,jk) )
CALL lbc_lnk( 'usrdef_zgr', pe3u,'U', 1._wp, pe3v,'V', 1._wp, kfillmode=jpfillcopy )
```

The statements that EXECUTE on these builds are the seamount hook's own, at
`usrdef_zgr.F90:225` and `:228` with the exchange at `:231`
(`tests/VORTEX_SMT_R3_OMIP_L1_P3/MY_SRC/` and
`tests/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/MY_SRC/`, identical in both). They
transcribe `tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1166` and `:1167` with the
exchange at `:1177-1178` — an OFFLINE TOOL, which is where the rule was
transcribed FROM and not something these builds run (round 213's minor 6).
`E3u_0` resolves to the 3-D `e3u_3d` under `key_vco_1d3d` by
`src/OCE/DOM/domzgr_substitute.h90:94-95`. `e3f_0` follows from `e3v_0`, not
from `e3u_0` (hook `:235`, `zgr_zps:1194`).

**CITATION RE-ANCHOR.** Note CC addendum 4 named hook lines `:213`/`:216`.
Re-read in the current file they are `:225`/`:228`; the comment and the error
message carry the re-anchored numbers, not the inherited ones.

**The faces are read, never re-derived.** The round-213 attempt re-derived
them with legoESM's own `min_cell_to_uface`/`min_cell_to_vface`, which zeroed
the whole northern row of `e3v_0` (630 cells at 0.0 where NEMO carries 500.0)
and wrapped a closed east column. Every card on this branch already carries
the arrays NEMO itself built — round 212 proved them equal to
`mesh_mask.nc` at zero ULP — on the EEN barotropic operand bundle. The repair
reads them, exactly as the neighbouring `nemo_ldf_reference_e3f`
(`vertical.py:358-375`) reads `e3f_0` from that same bundle, and FAILS CLOSED
rather than falling back to the alias. No new mask rule, no new halo rule, no
new north-fold rule: the arrays arrive with NEMO's conventions already in
them.

---

## 3. THE IDENTITY PROOF

`nemo_testcase_l1_vortex_smt_round2_geometry_gate.py` is round 212's
instrument, extended with the rows that matter here: the arrays the SOLVER
RESOLVES, not merely the ones the card carries.

| row | `VORTEX_SMT-zps` | `VORTEX_SMT_VEC-zps` |
|---|---|---|
| RESOLVED `e3u_0` vs NEMO `mesh_mask.nc` | **EXACT, 0 differing** | **EXACT, 0 differing** |
| RESOLVED `e3v_0` vs NEMO `mesh_mask.nc` | **EXACT, 0 differing** | **EXACT, 0 differing** |
| northern row `e3v_0` wrongly zeroed | **0 cells** | **0 cells** |
| east column `e3u_0` differing | **0 cells** | **0 cells** |
| non-vacuity: resolved `e3u_0` != `e3t_0` | 1 164 cells | 1 164 cells |

`GEOMETRY IDENTICAL` on all 17 rows of both cards (the twelve round-212 rows
are unchanged). The last row is the control: if the alias were still in place
the two identity rows would be vacuous.

**Inertness proved by construction as well as by ladder.** A probe over every
card on the branch (`round4/alias_probe.txt`) asks whether the card's own
reference face thickness is BITWISE equal to its T thickness — where it is,
the repair cannot change a single operation:

| card | `e3u_0 == e3t_0` | `e3v_0 == e3t_0` | cells differing (u / v) |
|---|---|---|---|
| `VORTEX-zco`, `VORTEX_VEC-zco`, both 15 km, both 10 km | yes | yes | 0 / 0 |
| `GYRE-zco` | yes | yes | 0 / 0 |
| `VORTEX_SMT{,_VEC}-zps` | **no** | **no** | 1 164 / 1 322 |
| `ORCA2-zps` | **no** | **no** | **18 803 / 18 300** (max 917 m / 949 m) |
| `LOCK_EXCHANGE-zco`, `OVERFLOW-zps` | n/a | n/a | rebuild branch, never reach the lines |

This is why a flat box is inert and is the proof note CC addendum 4 asked for:
with one uniform reference ladder, the minimum of the two neighbours IS the T
thickness, as the same numbers.

---

## 4. THE REGISTRIES, BEFORE -> AFTER

Bar 1e-15, 50 rows per card, one variable, same records (the R3 admitted set).

### `VORTEX_SMT_VEC-zps` — 37/50 moved, 35 toward NEMO

| kt | row | before | after | |
|---:|---|---|---|---|
| 2 | T | 1.733725e-09 | **4.539846e-11** | 38x toward |
| 2 | u | 1.748385e-07 | **6.004125e-08** | 2.9x toward |
| 2 | v | 1.607001e-07 | **4.037530e-08** | 4.0x toward |
| 2 | ssh | 3.664757e-07 | 3.664757e-07 | unmoved |
| 3 | T | 1.131661e-08 | 1.913613e-10 | 59x toward |
| 5 | u | 2.428854e-06 | 2.034024e-07 | 12x toward |
| 10 | T | 1.786773e-07 | 1.464035e-08 | 12x toward |
| 10 | u | 8.666645e-06 | 1.201966e-06 | 7.2x toward |
| 10 | v | 5.743456e-06 | 4.034260e-07 | 14x toward |
| 10 | ssh | 2.100024e-06 | 2.681764e-07 | 7.8x toward |

Every moved row is in `round4/after_VORTEX_SMT_VEC-zps.json`. `first_over_bar`
is unchanged at kt=2 T/u/v/ssh. The predicted values in note CC addendum 4
(~4.5e-11, ~6.0e-08, ~4.0e-08) are reproduced to every digit they stated.

**THE TWO ROWS THAT MOVE AWAY, AND THIS LANDING'S ONE EXACT-ROW LOSS.** Both
are the uniform-salinity row, and both move by EXACTLY one quantum of the
diagnostic. The card's S residual is quantized at 2.0301e-16: kt=3 S goes
6.090366e-16 -> 8.120488e-16 (3 -> 4 quanta) and kt=5 S goes 8.120488e-16 ->
1.015061e-15 (4 -> 5 quanta). The second crosses the 1e-15 bar, so AT-BAR
falls **9 -> 8**. Registered, not waved past: it is a one-quantum move of a
constant field, the only exact-row loss on any card this round, and it is the
price of the 35 rows above. No committed ratchet pins these rows (the SMT
registries are receipt-registered; `tests/ocean/unit/test_nemo_vortex_smt_card.py`
pins geometry, not ladder rows), so nothing went red — but it is a loss and
it is named.

### `VORTEX_SMT-zps` (flux) — 0/50, AT-BAR 10 -> 10

Exactly unmoved, as predicted. This is NOT a control: legoESM materialises
this operand only for the `zad` arm, which the flux card does not run, where
NEMO's `CALL wzv` (`stp2d.f90:153`) is unconditional. That conditionality is
round 213's open finding and is carried forward unchanged, not closed here.

---

## 5. EVERY OTHER CARD — MEASURED

| card | rows | moved | AT-BAR before -> after | `first_over_bar` |
|---|---:|---:|---|---|
| `VORTEX-zco` | 50 | **0** | 11 -> 11 | kt=2 T/u/v/ssh, unchanged |
| `VORTEX_VEC-zco` | 50 | **0** | 15 -> 15 | kt=2 u/v/ssh, unchanged |
| `VORTEX-15km-zco` | 50 | **0** | 9 -> 9 | unchanged |
| `VORTEX_VEC-15km-zco` | 50 | **0** | 12 -> 12 | unchanged |
| `VORTEX-10km-zco` | 50 | **0** | 8 -> 8 | unchanged |
| `VORTEX_VEC-10km-zco` | 50 | **0** | 12 -> 12 | unchanged |
| `LOCK_EXCHANGE-zco` | 50 | **0** | 34 -> 34 | kt=8 u, unchanged |
| `OVERFLOW-zps` | 50 | **0** | 13 -> 13 | kt=2 T/u, unchanged |

The two 30 km VORTEX rows and both tanks are before/after on the SAME base
tree (round 213's `inert/` arm, tip `b1e7d3a03`); the four refined rungs are
against round 208's registered ladders. OVERFLOW is the other partial-cell
card and it is 0/50 for a reason that is structural, not lucky: it carries no
raw qco set, so it takes the rebuild branch, which already applied the
min-of-neighbours rule.

---

## 6. ORCA2 — THE POINTER, AND A RETRACTION I OWE THE READER

**RETRACTED, in this same receipt, before anyone built on it: an earlier
draft of this section said ORCA2's rung-0 ladder moved 0 of 200 rows.** It
moves **185 of 200**. I read the comparator's printed tail, which shows the
rung-7 block, and quoted it as the rung-0 result. The independent reviewer
caught it (§8). The corrected numbers are below; nothing else in this receipt
rested on the wrong one.

**Which path ORCA2's card takes, proved from the code.** ORCA2 attaches the
whole eight-operand raw set (`nemo_testcase_recipe.py:1603-1614`), so it takes
the raw branch — it does not read `e3u_0`/`e3v_0` by some other route. Its
real partial-cell faces come from its own `domain_cfg` and are attached on the
EEN bundle at `:1579-1581`; its `hu_0`/`hv_0` are summed from those very
arrays at `:1570-1571`. It selects the changed arm: the ORCA2 identity
specialises GYRE's (`:284-292`), which sets
`zad_qco_evaluation="nemo_literal"` (`:482`), and that selector routes
`wzv`/`div_hor` through this builder (`ocean_pe_latlon_cgrid.py:1652,1695`).

**The operand changes by a lot:** 18 803 U cells and 18 300 V cells now carry
the shallower neighbour's reference thickness instead of the T thickness,
differing by up to 916.98 m and 949.12 m
(`round4/orca2_alias_probe.txt`, which also confirms the resolved array equals
the bundle array and no longer equals `e3t_0`). Before this round ORCA2's
`wzv`/`div_hor` ran with `e3u_0 = e3t_0` on every stepped face in the global
ocean.

**ORCA2's certified rung-0 ladder, measured on this tree:**

| | value |
|---|---|
| status | `PASS_RUNG0_TEN_STEP_LADDER`, 200 rows |
| comparator vs round 207's registered ladder | `PASS_R111_ORCA2_LADDER_COMPARE` |
| rows moved | **185 / 200** |
| bit-identical losses | **0** |
| first non-bit statement | unchanged (`kt=1 stage1 T`, `UNATTRIBUTED`) |
| direction, by rms | **151 toward NEMO**, 34 away |
| direction, by max abs | 111 toward, 73 away, 1 equal |
| largest movers | kt=10 stage3 `v` max abs **1.555881 -> 0.514415**; kt=10 stage2 `v` 1.406992 -> 0.653191; kt=10 stage1 `v` 1.303560 -> 0.658426; kt=9 stage3 `v` 1.096846 -> 0.687284 |

So this is the statement landing on ORCA2, and it lands hard: the four worst
rows in the whole ladder are cut by a factor of two to three, every field is
touched (u 38, v 38, S 37, T 37, ssh 35 rows), and nothing that was exact
stops being exact. **This is the most ORCA2-relevant result VORTEX has
produced**, which is what the round order predicted it would be.

**ORCA2's rung-7 certified ladder** (`--initial-mode decision52-bridge`) is
reported in §8.1 with the rest of the gate lines.

**What the ORCA2 lane should do next:** re-pin its registered rung-0 ladder to
these numbers with the 185 rows registered, then look at the 73 max-abs rows
that move away — on a change this size a mixed max-abs direction with a
uniformly improving rms is what a redistribution looks like, and the lane's
own per-cell instruments can say whether those 73 are the same cells.

## 7. CHOICES MADE THIS ROUND

| choice | ASKED? |
|---|---|
| read the faces from `nemo_een_barotropic`, not from a new raw `z_coord` field | ASKED — note CC addendum 4 names "the card's verified arrays"; the bundle is where every card already carries them, and `nemo_ldf_reference_e3f` is the existing precedent for reading `e3f_0` from it |
| raise instead of falling back to the alias when a raw-mesh card carries no face thicknesses | **UNASKED, and offered for revert.** It makes a previously-tolerated configuration a hard error. It is UNREACHABLE by every card in the tree (§1 table: all six raw-mesh card families attach the bundle), the `nemo_ldf_reference_e3f` sibling already fails closed on the same bundle, and the alternative default is the defect itself (RULE 3). Say the word and it becomes a documented fallback instead |
| the geometry gate gains four rows (resolved faces, northern row, east column, non-vacuity) | ASKED — note CC addendum 4 names the identity proof and says to extend the round-2 instrument |
| the test file's two pre-existing reds repaired in this diff | not a choice of behaviour: the fixture lacked a now-required argument and could not run at all; proven pre-existing by re-running on the stashed clean tree |

**UNASKED list: one item, named above, offered for revert in this message.**

**COMPLIANCE, stated because no gate checks it (RULE 2):**
* **DUAL review: ONE fresh adversarial reviewer ran, not two.** The second is
  codex, which is paused on this account. That is a GAP, not an exemption.
* **Controlled comparison:** every before/after pair in §4 and §5 differs in
  this one commit; the two 30 km VORTEX cards and both tanks are on the same
  base tree, the four refined rungs against round 208's registered ladders
  (noted as the one place the baseline tree is older).
* **Instrument validated before its numbers were quoted:** the geometry gate
  carries its own non-vacuity row (§3) and pins the fp64 policy; the unit
  test's alias arm fails if the repair is reverted.
* **Non-vacuity:** the unit test's new arm rebuilds the OLD alias explicitly
  and asserts the vertical velocity differs — restoring the alias turns the
  two identity assertions red.
* **Pre-impl search:** §1, with the greps pasted.

---

## 8. THE ADVERSARIAL REVIEW

One fresh reviewer (not the author, read-only, given the diff, the compiled
NEMO sources and the evidence directory) returned **DO NOT SHIP** on the
snapshot it was handed, with two MAJOR findings. Both are taken; one of them
corrected a real error in this receipt.

| # | finding | taken |
|---|---|---|
| MAJOR 1 | the NEMO line numbers are wrong in four places: `usrdef_zgr.F90:213` is `ik = k_bot(ji,jj)`, `:216` is the below-bottom copy and `:219` is a bare comment; the real statements are `:225`, `:228`, `:231-232`. Also `zgr_zps:1191-1197` is the F-point loop, not U/V, so a span written `1162-1197` over-claims | **ALREADY FIXED, and the reviewer's independent reading confirms the fix exactly.** The reviewer was handed `round4/round214.diff`, a snapshot saved BEFORE the citation re-anchor; the committed code has carried `:225`, `:228`, `:231` and `zgr_zps:1166-1167` / `:1177-1178` since `ae5115683`. That two independent readings of the hook produced the same four line numbers is the re-anchor's own control. The over-claiming span never reached the commit |
| MAJOR 2 | ORCA2 moves and its certified gate did not run: the rung-7 log held only XLA warnings, the compare died on a missing JSON, and `orca2_rung0_compare.json` carries **185 moved rows**, not the 0 this receipt claimed | **ACCEPTED, decisive, and it caught a real error of mine.** The 0 was the comparator's rung-7 block, which I had fed its own base as candidate; §6 is rewritten with the retraction and the corrected 185/200. The rung-7 refusal was the dirty-tree guard doing its job (the first run was launched before the commit); re-run on the clean tree it completes — §8.1 |
| minor | "1 084 wet U faces" counts the jpk=11 mesh array; the operand the solver receives is truncated to nlev=10, where it is 686 | **TAKEN**: the comment now gives both numbers and says which array each governs |
| minor | the non-vacuity row prints `"bit_identical": true` beside `"n_differing": 1164`, which reads backwards in a certification artifact | **TAKEN**: the row is renamed to say it is a counter and carries `row_is_a_counter` |
| minor | no shape check that the bundle's faces are on NEMO's native extent; a redundant-layout bundle would broadcast silently | **TAKEN**: the builder now refuses a face array whose shape is not `e3t_0`'s |
| minor | the three appended probe rows (north, east, non-vacuity) are implied by the full-array row above them | **noted, kept.** They cost nothing and they are the rows a reader of round 213 will look for first |
| minor | `nemo_dino_mesh.py:397` still aliases `e3u_0 = e3t_0` | **noted, correct there** (§1.2) and left alone: on that uniform ladder it is NEMO's own value, and changing it would be an unasked edit to a certified card |
| minor | these operands are stored float32 (from `mesh_mask.nc`), so `hu_0 != sum(e3u_0*umask)` by 4.88e-4 m | **pre-existing, named once here**, not introduced by this round and not fixed in it |
| PASS | northern row / halo / layout: the new code does no horizontal indexing at all, so the round-213 zeroing and east wrap are structurally unreachable; north row `e3v_0` min = 500.0 measured | — |
| PASS | provenance: single-source now, and the change REMOVES a split — before it, `hu_0` was min-rule-derived while `e3u_0` was the T alias | — |
| PASS | the fail-closed branch is unreachable by every real card; the test is non-vacuous; the `after_ssh_form` addition is a genuine stale-fixture repair | — |

**COMPLIANCE (RULE 2): ONE review, not two.** The second reviewer is codex,
paused on this account. That is a gap, not an exemption.

### 8.1 THE GATE LINES

| gate | result |
|---|---|
| geometry identity, both seamount cards | `GEOMETRY IDENTICAL`, 17/17 rows exact |
| `VORTEX_SMT_VEC-zps` registry | 37/50 moved, 35 toward, AT-BAR 9 -> 8 (one loss, §4) |
| `VORTEX_SMT-zps` registry | 0/50, AT-BAR 10 -> 10 |
| six flat VORTEX cards | 0/50 each |
| `LOCK_EXCHANGE-zco`, `OVERFLOW-zps` | 0/50 each |
| GYRE certified ladder | `OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0 first_over_bar={'fields': ['T','S','u','v','ssh'], 'kt': 3}->same` |
| GYRE from-rest year, 360 days | day 360 wet rms T **5.4077419367442036e-05 K** = note BZ's certified value to all 17 digits; snapshot digests `day030 4e36c106403b495e…`, `day240 8b9cd60475626373…`, `day360 e3e0a068346c7866…` = note BZ's. **Byte-identical** |
| ORCA2 rung-0 + rung-7 | `PASS_R111_ORCA2_LADDER_COMPARE`; 185/200 moved on each, 0 bit-identical losses, both first non-bit statements unchanged, majority toward NEMO (§6) |
| DINO from-rest month gate | run inside `land.sh` (model code changed), §9 |
| push battery | §9 |

Two gate invocations copied from round 208's script were stale and were
repaired rather than worked around: the day-gap tool's `--output` is now
`--json`, and its `--lego-root` must name `year_fromrest` for a from-rest
member. Neither is a model change.

---

## 9. LANDING, AND WHAT IS OPEN

OPEN, carried and not closed here:

1. **The kt=2 `ssh` owner is still the shared free-surface / barotropic
   path.** This statement does not move `ssh` on either seamount card, so
   round 2's reading survives untouched. The next measurement is the
   round-196 per-substep instrument on an SMT build — preregistered as round
   5's first item (it was not reached inside this round's budget).
2. **The flux card does not materialise this operand** where NEMO's `wzv` is
   unconditional. Round 213's finding, unchanged.
3. **S2, the depth average of the slow forcing**, stays held at
   `min_rule_live` with round 213's measurement; nothing here changes that
   decision.
4. **ORCA2's lane should re-pin its rung-0/rung-7 ladders** to the 185-row
   numbers in §6 and look at the max-abs rows that move away.
5. The stage-terms writer's uninitialised-halo dump; the stage-one stretch
   helper's missing floor; the untranscribed `r3u`/`r3v` stage-one ratios.
6. These operands are float32 (`mesh_mask.nc` precision), so `hu_0` and
   `sum(e3u_0*umask)` differ by 4.88e-4 m. Pre-existing; named once.
7. **Round 5** = the 100-day comparison on both seamount cards plus the
   movie, with round 3's predictions R4-P1..P4 still standing as written.

## 10. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round4/`:
`predictions.md`, `geometry_gate.json`, `alias_probe.txt`,
`orca2_alias_probe.txt`, `after_VORTEX_SMT{,_VEC}-zps.{json,log,residuals.npz}`,
`inert/`, `gyre_ladder_r214.json`, `cmp_gyre_ladder.json`,
`gyre_year_r214.log`, `gyre_day_gap_r214.json`,
`orca2_rung{0,7}_ladder.{json,log}`, `orca2_ladder_compare.json`,
`round214.diff`, `ledger_line.draft.txt`.
