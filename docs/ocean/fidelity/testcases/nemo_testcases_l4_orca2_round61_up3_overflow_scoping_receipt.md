# ORCA2 round 61 receipt — the UP3 statement is not ORCA2's

Date: 2026-09-28
Base: `1bbf37814553`
Preregistration: `27712fbfb`
Disposition: **HELD; no model statement lands; the ORCA2 framing is retracted**
ORCA2 claim label: **by construction**
OVERFLOW trajectory claim label: **independent** (read from the committed
round-47 reference report and round-59 comparison)
NEMO statement claim label: **compiled source**

## Answer

Rounds 57 to 60 held four times on a question whose premise is false. The
source-ordered UP3 U T-face flux is not an ORCA2 statement and cannot become
one: the ORCA2 deck selects `ln_dynadv_vec = .true.` with
`ln_dynvor_een = .true.`, and the OVERFLOW deck selects
`ln_dynadv_up3 = .true.` with `ln_dynvor_ens = .true.`. Built from their own
builders the four cards resolve exactly that way, and the held candidate's
only model hunk sits behind the single `flux_form` guard that the two
vector-invariant cards never take. Round 59 measured the consequence without
naming it: ORCA2 completed 40 checkpoints with **zero** moved rows.

The `282 / 17,000 -> 0 / 17,000` improvement is therefore an **OVERFLOW**
number, not an ORCA2 one. Round 57's own receipt says so in its header
("OVERFLOW claim label: given NEMO's recorded operands", "no ORCA2 score is
reported here"); the ORCA2 reading entered the round orders afterwards. The
landing predicate the last four rounds tried to satisfy — "the ORCA2 ladder
improves with the direct flux at 0" — is unsatisfiable by this statement, for
any implementation of it.

The second premise is also false. The OVERFLOW card is **not** certified
bit-exact. Its own reference report carries top-level `status` `DEBT` and
`first_over_bar` at `kt = 2` in `T` and `u`. Only `kt = 1` is exact. Of the 50
certified rows, 13 are `AT-BAR`, 26 are `DEBT`, 10 `UNMEASURED` and 1
`UNINFORMATIVE`. Every one of the 20 rows the candidate pushed over the bar is
a row already registered `DEBT`, and on each of them the worsening is a
fraction of the residual that row already carries.

## Hypothesis table

| hypothesis | verdict | the number that decides it |
|---|---|---|
| (A) the two cards select different momentum-advection switches | **HELD** | ORCA2 `ln_dynadv_vec = .true.`; OVERFLOW `ln_dynadv_up3 = .true.`. Constructed cards: ORCA2-zps and GYRE-zco `vector_invariant`; OVERFLOW-zps and LOCK-zco `flux_form` + `nemo_up3`. ORCA2 ladder under the candidate: 40 checkpoints, 0 moved rows. |
| (B) the candidate is wrong on the tank's partial-step / closed-wall cells | **not the blocker; stays open on the OVERFLOW card** | 0 of the 20 violating rows is `AT-BAR`; all 20 are `DEBT`. Largest worsening / reference-residual ratio 0.3101, median 0.0054, all below 1. |
| (C) the tank's certified record came from a different NEMO build | **REFUTED, on binary provenance** | The two runs use BYTE-IDENTICAL namelists. Their executables differ: the certified reference run's own binary hashes `08d83236dd7f5b92bb4f194e414e2ef5`, equal to the uninstrumented `OVERFLOW_OMIP_L1` build, whose `MY_SRC` carries NO `dynadv_up3` override at all, so the certified record ran the SHIPPED UP3; the walk record's binary hashes `59da10f439b2316b72778440d8e11ff2`, the instrumented build, whose `dynadv_up3.F90` differs from the shipped file by additions only (one `USE` and eleven writer `CALL`s) at the same `-O3`. Round 57's source-order replay of the recorded operands reproduced the recorded flux bit-exactly, which is not what a re-associated statement looks like. |
| (D) the candidate changes something upstream that the tank runs | **REFUTED** | The edited routine has exactly one production call site, `ocean_pe_latlon_cgrid.py:5247`, inside the guard at `ocean_pe_latlon_cgrid.py:5229`; the sibling branch is the vector-invariant ENE/EEN operator. The other references are unit tests and probes. |

## Compiled statements and the card that runs them

NEMO forms the masked horizontal curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects it
by the sign of the advected-velocity pair at `:182-192`, and forms the T-face
flux at `:194-195`. The deck that reaches those lines selects
`OVERFLOW_OMIP_L1/EXP00/namelist_cfg:83` and `:89`, and the certified
reference run's own namelist makes the same two selections at
`overflow_kt1_10/namelist_cfg:86` and `:92`. The deck that does not reach them
selects `ORCA2_OMIP_L4/EXP00/namelist_cfg:346` and `:352`. legoESM pins the
same split on the card: the condition at `nemo_testcase_recipe.py:1596`
refuses, at `:1599`, to build ORCA2-zps or GYRE-zco with anything but the
vector-invariant selection.

## Measured results

Card construction (this round, from the four public builders):

| card | momentum_advection | momentum_flux_scheme | vorticity_scheme |
|---|---|---|---|
| ORCA2-zps | `vector_invariant` | `upwind` (inert) | `een_total` |
| GYRE-zco | `vector_invariant` | `upwind` (inert) | `ene_total` |
| OVERFLOW-zps | `flux_form` | `nemo_up3` | `al81` |
| LOCK-zco | `flux_form` | `nemo_up3` | `al81` |

`momentum_flux_scheme` is read only when `momentum_advection == "flux_form"`,
so its value on the two vector-invariant cards selects nothing.

OVERFLOW reference report (round 47), read through the committed
`legoesm.ocean.fidelity.ulp_move_gate.certified_rows`: top-level `status`
`DEBT`, `bar` `1e-15`, `first_over_bar` `{"fields": ["T", "u"], "kt": 2}`; the
four measured `kt = 1` rows are exact at `normalized_max_abs` `0.0`.

The 20 violating rows. Both sides of every ratio below are the SAME
quantity, the absolute distance from NEMO, recomputed through the committed
probe; an earlier version of this table divided an absolute worsening by a
NORMALIZED residual and so overstated every `T` row by the factor 20, that
row family's oracle scale. The `ssh` and `u` rows were unaffected because
their oracle scale is below 1.

| row | reference residual | worsening | row ratio | cellwise ratio at the worst cell | cells moving by more than their own residual |
|---|---:|---:|---:|---:|---:|
| kt4 ssh | `2.897004e-10` | `9.091877e-13` | 0.0031 | 0.0727 | 0 |
| kt4 u | `2.332317e-08` | `4.050436e-14` | 0.0000 | 0.0843 | 26 |
| kt5 T | `2.367280e-09` | `1.065814e-14` | 0.0000 | 1.00 | 18 |
| kt5 ssh | `1.179436e-08` | `1.269041e-10` | 0.0108 | 0.2332 | 0 |
| kt5 u | `4.624389e-08` | `5.826245e-12` | 0.0001 | 0.2527 | 0 |
| kt6 T | `4.741183e-09` | `2.842171e-14` | 0.0000 | **8.00** | 71 |
| kt6 ssh | `1.053242e-07` | `2.834538e-09` | 0.0269 | 0.3849 | 0 |
| kt6 u | `3.184359e-07` | `1.474591e-10` | 0.0005 | 0.4289 | 73 |
| kt7 T | `7.450669e-09` | `2.273737e-13` | 0.0000 | 0.0001 | 34 |
| kt7 ssh | `3.896376e-07` | `2.189887e-08` | 0.0562 | 0.4670 | 0 |
| kt7 u | `1.262631e-06` | `1.392228e-09` | 0.0011 | 0.5573 | 0 |
| kt8 T | `1.096826e-08` | `3.431921e-12` | 0.0003 | 0.0096 | 77 |
| kt8 ssh | `7.518260e-07` | `8.072998e-08` | 0.1074 | 0.4957 | 0 |
| kt8 u | `2.837862e-06` | `6.541136e-09` | 0.0023 | 0.6030 | 0 |
| kt9 T | `1.926472e-08` | `3.015188e-11` | 0.0016 | 0.0252 | 85 |
| kt9 ssh | `8.261121e-07` | `1.640471e-07` | 0.1986 | 0.4778 | 0 |
| kt9 u | `4.344091e-06` | `1.754177e-08` | 0.0040 | 0.5996 | 1 |
| kt10 T | `3.045184e-08` | `1.666756e-10` | 0.0055 | 0.3209 | 93 |
| kt10 ssh | `6.265025e-07` | `1.942932e-07` | 0.3101 | 0.4202 | 0 |
| kt10 u | `5.422695e-06` | `2.902065e-08` | 0.0054 | 0.5756 | 103 |

Seven `ssh` rows, seven `u` rows and six `T` rows. As a ROW statistic every
worsening is a fraction of the residual its row already carries: maximum
0.3101 at kt10 `ssh`, median 0.0027.

**The cellwise form of that sentence is weaker, and on some rows it is
false.** A row's largest worsening and its largest residual are in general at
DIFFERENT cells. Taken at the same cell: no `ssh` cell anywhere moves by more
than it was already wrong by, but 13 of the 20 rows contain cells that do —
up to 103 of 16,900 on kt10 `u`, and with a worst cellwise ratio of 8.00 on
kt6 `T`. Those cells are the smallest ones in absolute terms: kt6 `T`'s worst
cell is a worsening of `2.84e-14` K against a prior error of `3.6e-15` K, i.e.
8 row-scale ULP against 1. So the bar is plainly non-discriminating on the
sea-level rows and is a real, if few-ULP, signal on parts of the `T` and `u`
rows.

Reproducer for both statistics, and for the card table above:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_up3_card_scope.py`
(`card_scope` / `row_move_ratios`), tested in
`tests/ocean/fidelity/test_nemo_testcase_up3_card_scope.py`, whose fourth test
is a synthetic case built so the row ratio reads 0.4 while the cellwise ratio
at the same cell is 4.0 — the failure mode this table had.

Of the 31 rows that move at all, 7 are `AT-BAR` salinity rows; their largest
move is `1.421085e-14`, which is two row-scale ULP at a salinity row scale,
i.e. at the bar and not over it. The gate's own `AT-BAR -> DEBT` status rule
never fires.

## Controls

- Status classifier plant: relabelling one violating row `AT-BAR` in a copy of
  the reference report flips "all violating rows are `DEBT`" from true to
  false and "any `AT-BAR` row among violators" from false to true. The claim
  is not vacuous.
- Ratio instrument: the earlier "plant" merely set a derived ratio to `1.5`,
  which could never detect a wrong ratio computation and is recorded as
  VACUOUS, not as evidence. It is replaced by a synthetic case in the
  committed test, built so the row ratio reads 0.4 while the cellwise ratio at
  the same cell is 4.0; that case is what exposed the unit error in the first
  table.
- Cellwise re-derivation of the ratio claim through the committed probe,
  directly from the two residual arrays (`round47/overflow_tip.residuals.npz`
  and `round59/iterm_1/overflow_trajectory_tip.residuals.npz`), after checking
  that the stored `residual` array is bit-identical to `|candidate - oracle|`
  so both sides of every ratio are the same quantity.
- Two parsed rows re-read by eye in the raw reference report:
  `kt2.before.T` (`DEBT`, `7.815970093361103e-15`) and `kt10.before.ssh`
  (`DEBT`, `6.265025088159071e-07`).
- Call-site census printed rather than asserted: two references in
  `packages/`, both in the same file, one definition and one call.
- Citation gate on this receipt (10 citations, PASS) and on the default
  receipt (274 citations, PASS), its nine built-in plants firing on both.
- Shifted-line plants on two of this round's own citations,
  `ocean_pe_latlon_cgrid.py:5229` and `ORCA2_OMIP_L4/EXP00/namelist_cfg:346`:
  both make the gate FAIL with `SYMBOL-NOT-AT-LINE`. Recorded because a first
  attempt passed `--plant 3`, which names no citation and was therefore a
  no-op that proved nothing; that vacuous run is not evidence.

## Retractions

- **"ORCA2's direct flux error is 282 / 17,000."** Retracted. That count is
  OVERFLOW's, measured against OVERFLOW's recorded operands. ORCA2 has no
  direct flux error on this statement because it does not evaluate it.
- **"The OVERFLOW card is certified bit-exact against NEMO's own OVERFLOW
  record."** Retracted. The card's own report is `DEBT` from `kt = 2` onward;
  only `kt = 1` is exact.
- **"On each violating row the worsening is smaller than the row's own
  residual."** Retracted as a statement about cells. It holds for row maxima
  and it is what R61-P4 froze, but the maxima are at different cells; taken
  cell by cell it is false on 13 of the 20 rows.
- **The first version of this receipt's ratio table.** Retracted: it divided
  an absolute worsening by a normalized residual, overstating every `T` row by
  the factor 20. The table above puts the same quantity on both sides.
- An intermediate reading of this round's own evidence treated
  `largest_previous_legoesm_field_move_in_row_scale_oracle_ulps`
  (`24,736,344,919`) as a previously accepted trajectory move. It is not: the
  gate defines it as the size of the candidate's own change to the model
  field. No claim rests on it.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R61-P1 | **CONFIRMED** | The four constructed cards carry the predicted selections; each matches its deck's switch. |
| R61-P2 | **CONFIRMED** | One production call site, under the `flux_form` guard, with the vector-invariant operator in the sibling branch. |
| R61-P3 | **CONFIRMED** | 20 violating rows, all `DEBT`; no `AT-BAR` row among them; the plant fires. |
| R61-P4 | **CONFIRMED as preregistered, REFUTED cellwise** | As frozen (row maxima, consistent units): maximum ratio 0.3101, median 0.0027, all below 1. Cellwise the same sentence is FALSE on 13 of the 20 rows, worst cellwise ratio 8.00 at kt6 `T`. The prediction's row-maximum wording is recorded as too strong, and its first numeric table was unit-inconsistent; both are retracted above. |
| R61-P5 | **CONFIRMED** | The `packages/` tree is byte-identical to base `1bbf37814553`; this round's commits touch documentation and the citation map only. |

## Reviews

Two independent adversarial reviews, both run on the committed round.

**codex** (read-only, quota guard 68% used, exit 0). One BLOCKING: hypothesis
C's "additions only" proves source preservation, not binary equivalence.
ACCEPTED and fixed — the round then hashed the executables and found the
certified record was produced by the uninstrumented build, which is stronger
evidence than the diff and is what the table now cites. Three SHOULD-FIX:
wrong row cardinalities (six/five/five for what is seven/seven/six) ACCEPTED
and fixed; results presented without a committed reproducer, and a vacuous
ratio plant, ACCEPTED — a probe and its direct test are now committed and the
vacuous plant is labelled as such; the retraction section omitting the
row-to-cell inference ACCEPTED and added. One NIT on citing the validator's
condition rather than its message text, ACCEPTED and added.

**Claude code-reviewer** (independent, fresh context). Agreed with codex's
blocking finding and went further, supplying the binary-hash provenance this
receipt now uses. Its own BLOCKING finding — that the row ratio is a
reduction-versus-reduction comparison across different cells, with two rows
locally at or above 1.0 — was already being corrected when the review landed
and is now the cellwise table above. It independently re-derived and confirmed
R61-P1, R61-P2, the single guarded call site, the 40-checkpoint ORCA2 result,
the moved-row census, the citation gate and its plants, and the untouched
`packages/` tree. Its NIT on "38 diff lines" being raw diff output rather than
added physics lines is accepted: that count is `diff` output lines; the added
content is one `USE` and eleven writer `CALL`s.

Both reviewers state that the round's headline conclusion — the UP3 statement
is unreachable from ORCA2 — survives their checks.

## Scope ledger

**ASKED.** Settle which of the four named hypotheses explains the two facts,
fix at the root, and record the finding.

**UNASKED and unchanged.** No configuration field, default, scheme selection,
threshold, forcing, resolution, timestep, carried state, ORCA2 entry, or
sea-ice field changed. No model file is edited. The citation map gained seven
entries and two deck paths, and the round adds one probe under
`scripts/validate/` with its direct test; all three are additions that change
no model behaviour.

## OPEN

1. **Stop walking the tank's UP3 statement on the ORCA2 lane.** It cannot
   move ORCA2. Whatever remains of it belongs to the OVERFLOW card's own
   ladder.
2. **The tank's earliest unowned divergence is `kt = 2` in `T` and `u`, one
   step upstream of the `kt = 3` statement rounds 57 to 63 walked.** That is
   the likely reason rounds 60, 61 and 63 could name no strict compensating
   owner: a toward/away classification taken at `kt >= 3` is measured on top
   of an unowned upstream error. The next OVERFLOW round should start at the
   `kt = 2` `T` and `u` owner.
3. **DECISION_NEEDED (threshold, operator's to answer).** The frozen strict
   2-ULP bar is an absolute no-worsening ratchet — `4.440892e-16` m of sea
   level — and it is being applied to rows whose own residual is up to
   `5.42e-06`. On the sea-level rows it therefore refuses a change that no
   cell's own error is smaller than; on parts of the `T` and `u` rows the
   change is a real few-ULP signal, so the question is not purely about a
   non-discriminating instrument. One line, three options:
   - (1) keep the ratchet exactly as it is on every row, and accept that no
     UP3 statement lands on OVERFLOW until the `kt = 2` owner is closed;
   - (2) read the ratchet on `DEBT` rows as a registered ratio against the
     row's own residual. If this is chosen, it must be the SAME-CELL ratio,
     not the row-maximum one: on the row statistic this candidate looks
     uniformly small (worst 0.31), while cell by cell 13 of its 20 rows
     contain cells moving by more than their own residual, two of them at or
     above 1.0. A row-maximum admission rule would let those through unseen;
   - (3) close the `kt = 2` `T`/`u` owner first, re-certify the tank, then
     apply the unchanged ratchet to a record it can actually discriminate.

   My pick is **(3)**: it changes no threshold, it follows the lane's own
   first-non-bit-statement discipline, and it makes the bar meaningful again
   instead of loosening it. Option (2) is a threshold change and is not taken
   here; both reviewers independently flagged that the row-maximum ratio is
   too weak to carry it, which is a second reason not to pick it today.
