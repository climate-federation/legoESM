# Grid ladder rung G1b — round 5 receipt (gridladder_r29)

Status: **LANDED**.  The round-4 contraction reading is confirmed and the
cited KEG accumulator is now bit-identical.  G1b is **NOT certified**:
GIVEN-NEMO-ENTRY remains at the bar over the record, while INDEPENDENT first
exceeds the `1e-15` bar at kt 81 (sea level) and reaches a worst `1.44e-14`.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round5.md`
(`8dbe666b4`, before measurement).  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round29/`.
The full-label artifacts were produced from clean commit `e4ed232c0`; the
final RHS probe from clean `7559db437` differs only by retaining the four-face
baseline census in its JSON.

## Result: the four faces are an XLA contraction

At kt 2, U faces `(ji,jj) = (43,68), (45,68), (54,73), (43,78)` are the
complete prior production residual.  On all four:

- the unguarded traced-input JIT equals an exact-rational fused multiply-add
  replay and differs from the one-operation-at-a-time binary64 value;
- the prior production value equals that unguarded JIT;
- the traced JIT using `nemo_source_round` equals the one-operation-at-a-time
  value; and
- V has zero unequal faces in every comparison.

The U maximum is `1.6543612251060553e-24`.  Counts are: production versus
written order `4 -> 0`; unguarded traced JIT versus written order `4`; traced
JIT versus fused replay `0`; guarded traced JIT versus written order `0`.
Thus R5-M1 and R5-M2 are **CONFIRMED**, not merely plausible.

The source statement is NEMO's metric-scaled KE-gradient product inside the
in-place Krhs subtraction, in both the record build and the Decision-122 full
ORCA2 build: `BLD/ppsrc/nemo/dynkeg.f90:128-130`.  It is one expression, not a
stored gradient temporary.  legoESM now calls the existing shared
source-operation materializer, **`nemo_source_round`**, around that completed
product only.  The existing `rk3_vector_rhs_nemo_order` route selects it; G1b
is the ladder card selecting that route, the library default remains false,
and no selector was added.

## Statement walk, kt 2 GIVEN-NEMO-ENTRY

Counts are U/V unequal wet faces.  The only changed statement is KEG; the
earlier boundaries are held fixed and the following statement is measured on
the completed RHS.

| compiled boundary | before | after | after maximum |
|---|---:|---:|---:|
| after HPG | 0 / 0 | 0 / 0 | 0 |
| after HPG + VOR | 0 / 0 | 0 / 0 | 0 |
| after HPG + VOR + KEG | 4 / 0 | 0 / 0 | 0 |
| after ZAD (zero on one level) | 4 / 0 | 0 / 0 | 0 |
| stage-1 velocity multiply/add/mask on the card RHS | 0 / 0 | 0 / 0 | 0 |
| stage-1 velocity after barotropic correction | 0 / 0 | 0 / 0 | 0 |

The separate diagnostic `HPG + KEG` addition remains 5 U / 0 V unequal at
`8.27e-25`; it is not NEMO's production accumulator order and is retained as
an instrument control, not a production debt.  The complete stage-1 RHS is
now 0/0 and every kt-2 stage field is bit-identical.  R5-S1 is **CONFIRMED**.

The next compiled statement is the vector velocity update at the record
build's `stprk3_stg.f90:366-370`, the same statement at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-471`; it was already exact
and remains exact.

## Decision 122 ORCA2 source pointer

The enclosing routine was opened rather than inferred from a line number.
In `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90`, line 648 is the `np_CRV` arm of
`vor_ens`, whose subroutine ends at :689.  The EEN routine starts at :692 and
its executed `np_CRV` statement is **:762-766**.  That is the same full-ORCA2
line cited for the G1b relative-vorticity path; :648 is not used as an EEN
citation.

## Both labels, before and after

The GIVEN entry table excludes only kt-1 T/S land cells, the standing
NEMO-held-land convention.  No wet-cell or scored-field row is excluded.

| label / scope | before (`fe88ed406`) | after (`e4ed232c0`) |
|---|---:|---:|
| GIVEN entry frames 1..50, first over bar | none | none |
| GIVEN entry exact rows / unequal cells | 250 / 2,320 | 261 / 2,267 |
| GIVEN restart steps 3..100, first over bar | none | none |
| GIVEN restart exact rows / unequal cells | 274 / 5,181 | 305 / 5,034 |
| INDEPENDENT first over bar, ssh / u / v | 27 / 63 / 62 | **81 / none / 99** |
| INDEPENDENT first over bar, T / S | 37 / none | **none / none** |
| INDEPENDENT worst ssh / u / v | 1.7032e-14 / 2.0919e-15 / 2.2807e-15 | **1.4407e-14 / 9.9028e-16 / 1.1367e-15** |
| INDEPENDENT worst T / S | 1.0648e-15 / 3.5495e-16 | **0 / 0** |
| INDEPENDENT exact rows / total unequal cells | 46 / 5,508,585 | **170 / 3,216,622** |

R5-L1 and R5-L2 are **CONFIRMED**.  This is a strict trajectory improvement,
not evidence from a later first step alone: the cited statement becomes exact,
GIVEN loses no exact row, INDEPENDENT gains 124 exact rows, and its unequal
cell count falls by 2,291,963.  G1b remains open because sea level and V still
cross the bar.

## Controls, tests, provenance, and choices

- R5-C1 is **REFUTED for its outer-round half**.  Removing only the inner
  product round makes the traced-input test fail and restores the kt-2 stage-1
  RHS census to 4 U / 0 V; removing only the outer accumulator round leaves
  that test and the RHS at 0/0.  The production helper therefore retains only
  the independently necessary inner materialization.  Its portable control
  requires at least one unguarded difference; the guarded result must equal
  every expected element.
- Full-record removal control: excluding only the stamped git SHA, the
  no-outer-round `independent100.json`, `given_entry50.json`, and
  `restart_entry100.json` objects are exactly equal to their two-round
  counterparts.  Therefore every GIVEN and INDEPENDENT number in the table
  above is unchanged.
- `rk3_vector_rhs_nemo_order=True` now raises on flux-form momentum configs,
  where there is no separate `dyn_keg` Krhs accumulation to order.
- Focused CPU tests, synchronously with `JAX_ENABLE_X64=1`: G1b card plus
  traced contraction, record-backed split, and stage replay **18 passed**;
  other-card tests **169 passed, 3 deselected**; ratchets **6083 passed,
  2 skipped, 3 pre-existing failures**; citation gate **17 passed**.
- The three deselected other-card checks are stale certified-card digest pins
  (GYRE, LOCK_EXCHANGE, OVERFLOW), and the three ratchet failures are in
  unrelated JRA55/land files.  Each set was run separately and reproduced
  unchanged at the pre-round-5 commit `e1062dce9`; no baseline was silently
  re-pinned or widened.
- Search-before-build found and reused `nemo_source_round`, the existing
  `rk3_vector_rhs_nemo_order` route, the production accumulator exposures, the
  stage replay, and `fused_multiply_add`.  One probe function extends the
  existing ladder; there is no parallel implementation or new config field.
- independent review: operator (Claude).  No nested Codex review was run, per
  the round instruction; the operator reviews this diff before landing.
- Controlled comparison: one source-statement materialization changed; deck,
  card, record, precision, masks, score, timestep, and labels are identical.
- Choices made: ASKED — apply the existing materializer if contraction was
  confirmed, select through G1b's existing route, walk velocity, rescore both
  labels.  UNASKED: none.

## Independent-review collateral measurements

The private `nemo_stage_rhs_accumulation_order_arm` reaches the GYRE round-11
and round-83 gates, ORCA2-L4 round-41/42 gates, and VORTEX round-193 and SMT
round-238 gates.  OVERFLOW round 51 passes the older private hook directly,
but uses flux-form momentum; `_nemo_vector_order` is false there, so the KEG
branch cannot be reached.

The initially broad guard changed VORTEX last-bit censuses.  For stage 2,
KEG U/V moved from `21797/21995` to `21799/21995`, and ZAD from
`22066/21995` to `22068/22004`; for stage 3, KEG moved from `21945/21534` to
`21950/21534`, and ZAD from `21945/21534` to `21950/21539`.  Maxima stayed
at `1.3553e-20` U and `2.7105e-20` V, with outputs at `1.1102e-16` (stage 2)
and `2.2204e-16` (stage 3).  The guard is therefore gated to the public G1b
selector; private-hook arms retain their pre-round-5 addition expression.

The other measured arms did not move: GYRE round 11 remained U/V
`11469/10677`, maximum `8.131516293641283e-20`; ORCA2-L4 round 41 retained
first non-bit HPG V at rank 0 `668/226637`, maximum
`2.362041079622794e-06`, and rank 1 `651/188538`, maximum
`3.1384570529614414e-06`, while all 64 disputed rank-1 U rows remained exact.
The final VORTEX, GYRE, and ORCA2 artifacts were produced at `ded95e469`.

Independent-review evidence is under `round29/review_fix/`.  Final SHA-256
(first 16): `independent100.json` `cdbbeb5fa65e7400`;
`given_entry50.json` `034dc7a265d69914`; `restart_entry100.json`
`c46bd539973aeb01`; `gyre_nemo_order.json` `966aa602c45d31a5`;
`orca2_rhs_family.json` `6f83682a9fc81404`; `vortex_source_order.json`
`a57a7817f5679c50`.

UNVERIFIED:
- the full ORCA2 card executes the cited lines but was read, not run;
- the cause of the remaining INDEPENDENT kt-81 sea-level debt is not yet
  localized.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r29/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — the KEG accumulator is exact; G1b remains uncertified on the INDEPENDENT label.
