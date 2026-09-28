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
| (C) the tank's certified record came from a different NEMO build | **REFUTED** | Instrumented `MY_SRC/dynadv_up3.F90` differs from the shipped file by 38 diff lines, every one an addition (one `USE` and eleven writer `CALL`s). The two decks' `namelist_cfg` and the two `cpp_*.fcm` are byte-identical. |
| (D) the candidate changes something upstream that the tank runs | **REFUTED** | The edited routine has exactly one production call site, `ocean_pe_latlon_cgrid.py:5247`, inside the guard at `ocean_pe_latlon_cgrid.py:5229`; the sibling branch is the vector-invariant ENE/EEN operator. The other references are unit tests and probes. |

## Compiled statements and the card that runs them

NEMO forms the masked horizontal curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects it
by the sign of the advected-velocity pair at `:182-192`, and forms the T-face
flux at `:194-195`. The deck that reaches those lines selects
`OVERFLOW_OMIP_L1/EXP00/namelist_cfg:83` and `:89`. The deck that does not
selects `ORCA2_OMIP_L4/EXP00/namelist_cfg:346` and `:352`. legoESM pins the
same split on the card: `nemo_testcase_recipe.py:1599` refuses to build
ORCA2-zps or GYRE-zco with anything but the vector-invariant selection.

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

The 20 violating rows, worsening against the residual each row already carries:

| row | reference status | reference residual | worsening | ratio |
|---|---|---:|---:|---:|
| kt4 ssh | `DEBT` | `2.897004e-10` | `9.091877e-13` | 0.0031 |
| kt4 u | `DEBT` | `2.332317e-08` | `4.050436e-14` | 0.0000 |
| kt5 ssh | `DEBT` | `1.179436e-08` | `1.269041e-10` | 0.0108 |
| kt6 ssh | `DEBT` | `1.053242e-07` | `2.834538e-09` | 0.0269 |
| kt7 ssh | `DEBT` | `3.896376e-07` | `2.189887e-08` | 0.0562 |
| kt8 ssh | `DEBT` | `7.518260e-07` | `8.072998e-08` | 0.1074 |
| kt9 ssh | `DEBT` | `8.261121e-07` | `1.640471e-07` | 0.1986 |
| kt10 ssh | `DEBT` | `6.265025e-07` | `1.942932e-07` | 0.3101 |
| kt10 u | `DEBT` | `5.422695e-06` | `2.902065e-08` | 0.0054 |
| kt10 T | `DEBT` | `1.522592e-09` | `1.666756e-10` | 0.1095 |

The remaining ten violating rows (kt5-kt9 `T` and `u`) have ratios between
`0.0001` and `0.0313`. Round 59's headline `875,018,892` ULP is the kt10 `ssh`
row: `1.942932e-07` m of sea level expressed in units of
`numpy.spacing(1.0)` = `2.220446e-16`. The row it worsens is already
`6.265025e-07` m from the oracle.

**That ratio is a row statistic, and the cellwise form of it is weaker.** The
row maximum of the worsening and the row maximum of the residual are in
general at DIFFERENT cells, so the table above does not say that every cell
moves by less than its own error. Re-derived per cell, at the cell where each
row's worsening is largest:

| row family | cellwise ratio at the worst cell | cells where the worsening exceeds that cell's own residual |
|---|---|---|
| six `ssh` rows (kt4-kt10) | `0.0727` to `0.4957` | **0** on every one |
| five `u` rows (kt4-kt10) | `0.0843` to `0.6030` | 0 to 103 of 16,900 |
| five `T` rows (kt5-kt10) | `6.3e-05` to **`8.0`** | 18 to 93 of 17,000 |

So on the two large-amplitude fields the candidate never dominates a cell's
existing error, but on 10 of the 20 rows some cells do move by more than they
were already wrong by. Those cells are the smallest ones in absolute terms:
the worst of them, kt6 `T`, is a worsening of 8 row-scale ULP at a cell whose
prior residual was 1 ULP. The honest summary is that the bar is plainly
non-discriminating on the `ssh` rows and is a real, if tiny, signal on parts
of the `T` and `u` rows.

Of the 31 rows that move at all, 7 are `AT-BAR` salinity rows; their largest
move is `1.421085e-14`, which is two row-scale ULP at a salinity row scale,
i.e. at the bar and not over it. The gate's own `AT-BAR -> DEBT` status rule
never fires.

## Controls

- Status classifier plant: relabelling one violating row `AT-BAR` in a copy of
  the reference report flips "all violating rows are `DEBT`" from true to
  false and "any `AT-BAR` row among violators" from false to true. The claim
  is not vacuous.
- Ratio plant: setting one ratio to `1.5` flips "all ratios below 1" from true
  to false.
- Cellwise re-derivation of the ratio claim directly from the two residual
  arrays (`round47/overflow_tip.residuals.npz` and
  `round59/iterm_1/overflow_trajectory_tip.residuals.npz`), which is what
  exposed that the row-maximum ratio is not the cellwise ratio.
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
| R61-P4 | **CONFIRMED as preregistered, REFINED against itself** | As frozen (row maxima): maximum ratio 0.3101, median 0.0054, all below 1; the plant fires. The cellwise form of the same sentence is FALSE on 10 of the 20 rows, worst cellwise ratio 8.0 at kt6 `T`. The refinement was found by this round, not by a reviewer, and the row-maximum wording of the prediction is recorded as too strong. |
| R61-P5 | **CONFIRMED** | The `packages/` tree is byte-identical to base `1bbf37814553`; this round's commits touch documentation and the citation map only. |

## Scope ledger

**ASKED.** Settle which of the four named hypotheses explains the two facts,
fix at the root, and record the finding.

**UNASKED and unchanged.** No configuration field, default, scheme selection,
threshold, forcing, resolution, timestep, carried state, ORCA2 entry, or
sea-ice field changed. No model file is edited. The citation map gained seven
entries and two deck paths, which add citations and change no behaviour.

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
     row's own residual, admitting a change whose effect is a stated fraction
     of that residual;
   - (3) close the `kt = 2` `T`/`u` owner first, re-certify the tank, then
     apply the unchanged ratchet to a record it can actually discriminate.

   My pick is **(3)**: it changes no threshold, it follows the lane's own
   first-non-bit-statement discipline, and it makes the bar meaningful again
   instead of loosening it. Option (2) is a threshold change and is not taken
   here.
