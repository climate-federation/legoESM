# Grid ladder rung G6 — round 1 receipt (gridladder_r36)

Status: **STOPPED_FOR_RECORD.**  Deck, card, acquisition script and
preregistration are committed; NEMO has not run.  The operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g6_orca2vec/run.sh --run`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g6_round1.md`
(`8e74844e5`, before any G6 record).  Card/deck/acquisition: `7a554c247`; review fixes: `5b5046102`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round36/`;
the record will land in `round36/oracle_g6_r36/`.

## Take-aways

1. **G6 = G5's deck with ORCA2's `&namdyn_adv`, nothing else.**  Assignment
   diff G5 -> G6 (committed test): `ln_dynadv_OFF` T -> F, `ln_dynadv_vec`
   F -> T, `nn_dynkeg` unstated -> 0, each equal to ORCA2's
   (`ORCA2_OMIP_L4/EXP00/namelist_cfg:346-347`).  The ice overlay is G5's,
   byte-identical.  No DECISION_NEEDED.
2. **What NEMO derives from the switch** is tabled in the preregistration,
   section 1, every row cited in the reused writer-free build: np_VEC_c2,
   np_CRV total vorticity, KEG + ZAD in stp_2D and stages 2-3,
   velocity-form `wzv`, velocity-form stage, `dyn_zdf` and external updates,
   velocity time sums, no post-loop depth recompute, and `ll_Fw` for
   tracers (consumer OFF).  Readers that do not execute are listed too.
3. **No new cpp key and no build.**  G5's round-31 binaries are reused,
   pinned by SHA-256, keys `key_RK3 key_qco key_si3 key_vco_1d3d`.  Their
   record writer set came from the ORCA2 lane, whose deck is itself vector
   form.
4. **Card = G5's path plus G1b's vector statements, now selectable on G6:**
   np_CRV vorticity (e1f*e2f area, U fold halo on the top F row),
   `rk3_vector_rhs_nemo_order` (HPG, VOR, KEG order and the KEG product
   rounding), the vector-form fold unit, and the vector momentum selectors;
   flux-form-only options deselected exactly as on G1b.  G5's ice top drag
   and stage-2/3 tracer statements stay selected.  Fail-closed:
   `g6_record_not_admitted`, `nemo_post_si3_surface_records`,
   `vector_form_multilevel_step`.
5. **Prediction (the brief's, stated as such): the first non-bit statement
   is the fold-unit vector-form V update** of the external mode, the
   statement the ORCA2 lane is walking, at kt 1.  PLAUSIBLE only: the G
   builder reads `hv_0` and masks raw, and G1b's identical fold unit is at
   the bar on GIVEN.  Fallback: `dyn_zad` or velocity-form `wzv` at kt 1
   stage 2, never run on more than one level on any rung.
6. **Expected NEMO boundary:** stop while beginning kt 11, as G5 (|V| 10.25
   m/s), so 10 admitted steps.  The script follows whatever NEMO does.

## 1. Survey (searched before building)

- Searched: the G1 builder and its rung tables (`_ORCA2G_*`), G1b's vector
  block and selectors, G5's acquisition and admission
  (`run.sh`, `g5_records.py`), G1b's acquisition, the re-anchor tool.  Found
  all of it; nothing new was written for physics.  G6 adds one case key, one
  shared set of the two record-fed rungs (`_ORCA2G_ICE`), one builder alias,
  one unmeasured tuple.
- G1b statements by round, now selectable on G6: round 2 (np_CRV F area;
  fold unit outside the vector/flux branch), round 3 (top F row U fold halo;
  Krhs order HPG/VOR/KEG), round 4 (vector stage update in written order —
  already on for every rung), round 5 (KEG product source-rounding inside
  the same Krhs-order selector).  Round 6 removed a workaround; nothing to
  carry.
- New against G1b: 30 wet levels, so `dyn_zad`'s interior loop, `wzv` on
  velocities over partial cells, KEG/np_CRV on partial-cell faces, the
  vector `dyn_zdf` update over many levels, and the vector external update
  with real depths and the ice top drag.

## 2. Acquisition

`nemo_testcase_g6_orca2vec/run.sh`.  Smoke (record binary, 2 steps) first;
writer-free 100-step reference discovers NEMO's boundary; twins A/B/C to
N = min(10, completed) rounded down to `nn_fsbc` = 2; restarts A at N, B at
2,4,..,N, C at 3,5,..,N = every step 2..N; step 1 is never requested.
Each run must log ORCA2's vector form and total vorticity.  Admission:
surface payload schema and twin bytes, then G5's admission with the record
length (`g5_records.py admit --steps N`): passivity (every restart variable
of the record twins byte-identical to the writer-free reference at common
steps), per-step frame twin bytes, stage-3 = next entry, restarts = stage-3
frames.  The admission previously assumed 10 steps; it now takes N
(default 10, G5 unchanged).

- Preflight at `7a554c247`: `G6_PREFLIGHT_OK ... expected_boundary=kt11_after_10_completed`
  (`preflight.log`).
- Plant: one-byte deck change committed in a scratch clone -> `REFUSE: G6
  acquisition: committed G6 deck changed`, exit 66
  (`preflight_plant_deck.log`).

## 3. Tests and gates (CPU, fp64, synchronous)

- New `tests/ocean/fidelity/test_nemo_testcase_g6_orca2vec.py`: 5 passed.
  Plant: G6's vector selection forced off -> the card test fails
  (`ValueError: ORCA2G6-vec disagrees with its deck: ['ln_dynadv_vec']`,
  exit 1, `plant_g6_vec_off.log`).
- New admission test on G5's real record: the record-length argument,
  twin C reported as "no restart step in common" (it was reported identical
  after comparing nothing), a length with no twin-A restart refused by name,
  passivity plant fires.  G5 + G6 files: 19 passed.
- Regression: G3/G4/G5 50 passed + 1 red pin (the G4 test listing which
  rungs carry G4's wind/stage statements; now lists G6); G4 + G5 top-drag +
  G2 44 passed; G2 ladder had the same rung-table pins (4), updated.
- Required gates, clean tree at `5b5046102`: receipt-citation gate +
  no-private-cross-imports + G2 ladder + G1b card: **47 passed**
  (`gates_final.log`).  One citation-map range re-anchored with the
  repository tool (the stage-order table moved): now
  `nemo_testcase_recipe.py:4090-4095`.
- Review, single review (codex): first pass timed out before a verdict
  (`codex_review.log`); second pass **DO NOT SHIP**, three CONFIRMED
  findings, all fixed in `5b5046102`: G6 had dropped G5's
  `nemo_post_si3_surface_records` unmeasured marker; the admission reported
  twin C's passivity as success with zero compared steps (inherited from
  G5's admission); the run-script test only grepped text
  (`codex_review2.log`).  Third pass on the fix: **"No CONFIRMED defects.
  ... VERDICT: SHIP"** (`codex_review3.log`).

## OPEN (next round)

1. Operator runs the acquisition; next round admits it (passivity first).
2. Wire `ORCA2G6-vec` into the G2-family ladder (`CASES`/`FORCED` and the
   five G5-only `case ==` sites in `nemo_testcase_g2_orca2bathy_ladder.py`)
   — not done this round; the card is fail-closed meanwhile.
3. Card smoke on G6 (finite steps fed with records) not run: no G6 record
   yet, and the ladder is not wired.
4. Measure P1-P5; walk the first non-bit statement.

## Choices

- ASKED (brief): G6 definition, reuse of G5's record build, smoke first,
  restarts at every writable step, G1b statements selectable on G6, the
  brief's predicted first statement.
- UNASKED: (a) twins capped at 10 steps, as G5 — if NEMO runs past kt 11 the
  record still stops at 10; (b) `g5_records.admit` takes the record length
  (default unchanged).  Both offered for revert.

UNVERIFIED:
- NEMO's actual G6 boundary and passivity under the vector form;
- the card's vector arm together with the ice top drag has not stepped;
- the predicted first non-bit statement.
