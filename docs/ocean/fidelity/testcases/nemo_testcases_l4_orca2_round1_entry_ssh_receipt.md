# NEMO testcase Lane 4 — ORCA2 card round 1 entry/record receipt

Date: 2026-09-21

Starting tip: `dda3f3257dad5a7f86e1177f934afc531e6567d9`

Preregistration: `ba735dd27`

Status: **STOPPED_FOR_RECORD.**  The pinned ORCA1-ice root is admitted and
the first ocean non-bit field is kt=1 `ssh`, before the first time-step
statement.  The accepted records contain only the kt=1 post-`sbc` ocean
surface-input frame, so a legoESM kt=1..10 trajectory is not mechanically
claimable.  No model file changed, no NEMO/MPI process was launched, and no
SI3 selector or card `unmeasured_features` entry changed.

## 1. Complete receipt/gate inventory at entry

This is the requested single inventory table.  “Admits” is deliberately
narrow: a record can be admitted without admitting its producer or a later
model statement.  Superseded and rejected records remain listed because they
must never be silently mixed into the active root.

| receipt or gate / record | mechanically admitted state | UNMEASURED-with-spec, rejection, or next boundary |
|---|---|---|
| Phase 1 / 90-stream `SHIPPED_DECK_RECORD` | Frozen schemas, exact EOF, scalar-math provenance, ordinary-output passivity; round-1 rerun PASS with 10/10 plants. | No legoESM stage arithmetic was admitted.  This root has icebergs enabled and is not the active comparison root. |
| Phase 2 entry / ORCA2 card | External deck, tripolar geometry, masks, 3-D partial cells, initial T/S/u/v/ssh against the then-selected five-category root. | Full SI3/iceberg carried state and every later consumer were explicit `UNMEASURED_STOP`. |
| Phase 2b / post-`sbc` surface stream | Self-describing 35-field ocean surface-input schema at kt=1. | Shipped-deck arm superseded by the icebergs-off decision; producing SI3 and iceberg operators were not admitted. |
| Phase 2c / icebergs-off variant | One resolved option changed: `ln_icebergs=F`; three operator-run recipes prepared. | No MPI result in this receipt; no ocean arithmetic entered. |
| Phase 2d / first variant runs | Icebergs-off ordinary outputs and kt=1 card entry; O1 acquisition specified. | O1 could not separate `fld_read` from NCAR bulk using only the final surface stream. |
| Phase 2e / first O1 record | Successful run retained. | Rejected: malformed second frame and only 84/91 inherited raw identity; no O1 score. |
| Phase 2f / schema-fixed O1 | Two complete O1 frames and canonical defined cells; replacement twins prepared. | Twin reproducibility still pending in that receipt. |
| Phase 2g / 91-stream V2 witness | Nine `fld_read` CORE fields exact at 0/13,320; NCAR bulk handed to Lane 3b. | Two writer-owned O1 fields needed canonicalization; SI3 producer remained unmeasured. |
| Phase 2h / 92-stream systematic candidate | O1 map remained exact and the RGB boundary was localized. | Rejected as a complete oracle: 91/92 twin identity due unowned halo values. |
| Phase 2i / RHS candidate | RGB source identity and continued ocean walk. | Rejected as a complete oracle: concurrent rank writes made the RHS stream non-reproducible; replacement prepared. |
| Phase 2j / 92-stream `VARIANT_ORACLE_V2` | Full twins exact; RGB, EOS-80, SCO HPG and source-adjacent transports exact. | First debt was shared external-mode EEN; WZV/runoff frame not yet acquired; SI3 producer unmeasured. |
| Phase 2k / 93-stream WZV extension | WZV divergence, runoff decrement, QCO recurrence and final product exact on the resolved stage-1 clock. | The mixed state/clock factor-three conclusion is retracted in the tool; receipt is superseded for ownership. |
| Phase 2l / compiled-production W | Tripolar U-face layout repair, full-endpoint W adjudication, and exact oracle-supplied transport continuation. | Stops at shared stage-transport source association/tracer advection; BBL and ZDF/TKE remained unentered. |
| Phase 2m / external-EEN acquisition | Four self-describing EEN operand streams and twins prepared. | Operator execution was pending in that receipt. |
| Phase 2n / EEN extension | Phase-2m twin record admitted; EEN discriminator assigns partial-cell ownership. | Diffusive-BBL record/twins were the next acquisition. |
| Phase 2o / diffusive BBL | BBL twins and source-ordered ordinary division certified. | ZDF entry stopped at `UNMEASURED_NEEDS_WRITE_ONLY_FRAMES`; IWM and SI3 producer unmeasured. |
| Phase 2p / partial-cell EEN | Frozen `fe3mask`/live partial-cell EEN repair and ZDF acquisition prepared. | ZDF twins not yet admitted in this receipt. |
| Phase 2q / first ZDF stream | 95/95 twins and inherited passivity established. | Rejected: false payload count in the new ZDF header; no SH2 score. |
| Phase 2r / corrected ZDF extension | Corrected 95-stream root, carried `avm/avt/en` exact, cold-start SH2 0/n on reconstructible cells. | Zero-gradient result is explicitly vacuous; wrong ORCA2 SH2 selector tuple was the next boundary. |
| Phase 2s / ZDF+EEN acquisition | Face-native Nbb/Nbb SH2 selector restored; kt=1/2 ZDF and EEN records/twins prepared. | kt=2 nonzero SH2 and all later TKE statements were pending operator records. |
| Phase 2t / 100-stream V2 root | Twins/schema/passivity admitted; non-vacuous SH2 and bottom-TKE boundary exact. | First TKE debt was wrong `nn_eice=1` selection; later TKE/EVD/IWM unmeasured. |
| Phase 2u / first TKE walk record | `nn_eice=1` restored and exact; TKE schema prepared. | Produced record rejected later: undefined level-31 workspace; `zWlc2` and later rows not admitted from it. |
| Phase 2v / replacement TKE record | Corrected canonical writer and replacement twins prepared. | MPI pending in this receipt; no numerical claim yet. |
| Phase 2w / 101-stream `VARIANT_ORACLE_V2` | Replacement twins admitted 101/101; ordered kt=2 walk exact through `zWlc2`. | First non-bit TKE statement is `zpelc`, 39,290/242,135, max 4 row ULP; shared-GYRE owner.  ORCA1-ice twins were staged. |
| Phase 2x / ORCA1-ice rebuild | Phase-2w pond-selector init failure retained; complete 91-row ORCA1 ice override and tailored SI3 schemas certified; replacement twins staged. | No oracle admitted in this receipt. |
| Phase 2y / `VARIANT_ORACLE_ORCA1ICE` | 116/116 twin-raw-exact streams, all schemas/passivity, 26 plants; A pinned, B witness.  kt=1 T/S/u/v exact versus V2. | kt=1 SSH differs because category-resolved analytic snow/ice load differs.  Resolved ice namelists differ only at the registered initial-file selector. |
| Phase 2z / two later SI3 thermodynamic frames | Isolated config, write-only schemas, scalar-math binary and twins staged. | Not executed or admitted; sea-ice integration remains outside this lane. |
| Active V2 record | Phase-2v A is the single 101-stream five-category root; B is witness. | Historical root differential only; not the user-selected root for this card round. |
| Active ORCA1-ice record | Phase-2x A is the single 116-stream pinned root; B is witness. | Only one post-`sbc` surface frame exists; kt=2..10 inputs are absent. |
| Parallel native TKE-boundary acquisition | Current-source passive build admitted 94 comparable inherited streams plus native `en_after_boundaries`; all binding passivity controls fire. | Seven historical writer streams are named absent; stage-1 `zFw` is excluded because the reference writer dumps uninitialized storage.  This is a boundary record, not an exact Phase-2v source clone. |
| Merged-tree gate 1: Phase-1 schema | PASS; 90 streams and 10/10 plants. | None for schema; producer physics not implied. |
| Merged-tree gate 2: Phase-2v TKE walk | PASS admission; historical first debt remains kt=2 `zpelc`. | TKE walk stops at first debt. |
| Merged-tree gate 3: Phase-2y ORCA1-ice admission | PASS; 116/116 and 26/26 plants. | Does not make the one-category SI3 producer a legoESM implementation. |
| Merged-tree gate 4: resolved ice namelist | PASS; 185 fields, one registered initial-file difference. | Selector equality is not an exact-input SI3 physics score. |
| Merged-tree gate 5: Round-20 Phase-1 schema | Expected FAIL for five extra streams, reproduced. | The frozen 90-stream parser cannot admit the larger Round-20 inventory. |
| Merged-tree gate 6: Round-20 SI3 headers | VALID; counts 5/140/10/25/15. | Schema only; no SI3 physics identity. |
| Merged-tree gate 7: Round-20 exact-input admission | `STOP_SELECTOR_GAP`, reproduced after SI3 merge. | Six exact selectors remain unsupported: 5 categories, 10 ice layers, 5 snow layers, salinity scheme 4, ponds, lateral melt.  This round never changes that verdict. |

The live card registry is unchanged, including order:
`staged_gm_eiv`, `linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

## 2. Frozen-gate reruns

- Phase-1 gate: PASS, 90 records, six exact restart shards, eight history
  payloads under the timestamp rule, and all ten plants `PASS_NONZERO`.
  Artifact SHA-256:
  `cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f`.
- Pinned ORCA1-ice admission: PASS, 116/116 twin records, four restart
  shards, eight history payloads, and 26/26 plants.  The result is
  byte-identical to the Phase-2y artifact, SHA-256
  `d3c60bf16a84f8739f8106541bea2bfcf4941e75b17e7a4f1bd0d5f7cd371297`.
- Historical V2 TKE walk: admission PASS at 101 twin and 100 inherited
  streams; first debt remains `zpelc`, 39,290/242,135, maximum absolute
  `1.7763568394002505e-15`, maximum four row ULP, first zero-based cell
  `[1,49,2]`.  JSON SHA-256:
  `c7ef6ed73efe06c1f8a894c9e76f9e6b8f4051a579df420ed54a8cd62ef1f947`.

The first attempt to launch the Phase-2y admission by filename failed before
the gate imported because that historical script does not add the repository
root to `sys.path`; the unchanged gate was then run as a module and passed.
The first round-1 ladder invocation also stopped before reading a state frame
because the new probe treated its constructor label as a runtime config
attribute.  Commit `bf1e10784` replaced that stamp with assertions on the
resolved WS-RK3, total-EEN and vectorized-Langmuir selectors.  No failed
invocation produced a numerical artifact.

## 3. First ocean statement and magnitude ranking

The round-1 gate uses the rank-0-owned 148-by-90 domain after stripping two
halo cells and drops the dummy level before every 3-D comparison.  Identity
means `np.array_equal`, without tolerance.

At kt=1 the current card is bit-identical to the five-category V2 entry for
all five fields.  Against the pinned ORCA1-ice root it gives:

| ordered field | unequal / count | max absolute | verdict |
|---|---:|---:|---|
| T | 0 / 399,600 | 0 | AT-BAR |
| S | 0 / 399,600 | 0 | AT-BAR |
| u | 0 / 399,600 | 0 | AT-BAR |
| v | 0 / 399,600 | 0 | AT-BAR |
| ssh | 8,794 / 13,320 | 0.015479333813968585 m | FIRST NON-BIT |

Every differing owned SSH cell has the same absolute delta to displayed
precision (mean over unequal cells
`0.015479333813968583 m`).  The first differing index is `[1,49]`.
The gate's one-representable-step T plant exits 1 with
`REFUSE: planted kt1 T identity control fired`.

The committed gate also compares V2 and ORCA1-ice NEMO entry/stage frames
through kt=10, but labels every row
`NEMO_ROOT_DIFFERENTIAL_NOT_LEGOESM_TRAJECTORY`.  This is a carry-magnitude
diagnostic only.  At the kt=10 entry, the magnitude ranking is:

| rank | field | max absolute | unequal / count |
|---:|---|---:|---:|
| 1 | T | 0.9213698244489771 K | 233,341 / 399,600 |
| 2 | S | 0.348005900030401 | 233,341 / 399,600 |
| 3 | ssh | 0.11244887791514496 m | 8,794 / 13,320 |
| 4 | u | 0.0690388788854221 m/s | 226,236 / 399,600 |
| 5 | v | 0.05184263533089248 m/s | 226,637 / 399,600 |

Preregistered R1-P4 predicted kt=10 SSH in `[0.014,0.017] m`; it is
**REFUTED and retained**.  The result does not assign any of the compounded
kt>1 differences to legoESM because no legoESM trajectory ran.  The full
round-1 JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round1/round1_ladder.json`,
SHA-256 `e8f10afa89b61d64e233834a65f528a5ac9e85287c8e7cdc5ae831de15b1c66b`.
It stamps clean producer commit `88f759bcba02` and records SHA-256 for both
NEMO inputs at every entry/stage checkpoint.

## 4. Exact record stop and acquisition

The pinned root contains and schema-validates only
`oracle_ocean_surface_input_kt00000001.bin` (4,101,508 bytes, SHA-256
`34fa2926f7a0fd0e91eec326973b070fb5f9b1193cb1133de60b10b020fca344`).
The exact missing streams are:

```text
oracle_ocean_surface_input_kt00000002.bin
oracle_ocean_surface_input_kt00000003.bin
oracle_ocean_surface_input_kt00000004.bin
oracle_ocean_surface_input_kt00000005.bin
oracle_ocean_surface_input_kt00000006.bin
oracle_ocean_surface_input_kt00000007.bin
oracle_ocean_surface_input_kt00000008.bin
oracle_ocean_surface_input_kt00000009.bin
oracle_ocean_surface_input_kt00000010.bin
```

No carry-forward or inferred frame is allowed.  The operator-only acquisition
is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round1_surface_acquisition/run.sh`.
Its `--preflight-only` mode passes without invoking NEMO.  Its `--run` mode is
fail-closed, uses Bash `time`, creates a new isolated config, restores the
frozen Phase-2x one-category ice writers, changes only the post-`sbc` surface
writer to one `STATUS='NEW'` frame per step, runs A/B twins, and requires:

- all ten dynamic-kt surface schemas and exact EOF;
- A/B raw identity;
- 109 comparable inherited streams passive to the pinned root, with the seven
  historically overwritten writers named individually;
- the existing stage-1 transport exclusion narrowed to only uninitialized
  `zFw`, while header, length, `zFu`, and `zFv` remain binding; and
- ordinary-output identity to the pinned root.

The script was syntax-checked, its patch dry-run succeeds, and its preflight
pins the binary, deck/input manifests, source card, compiler card, frozen
Phase-2x sources, and patch hashes.  Each staged twin also retains the
producer's compiled `iceistate.f90` and its SHA-256 so the executing branch
cannot be lost as it was in the older record.  Its final admission additionally
stamps the clean Git worktree and requires twin equality of both the binary and
compiled producer source before reporting their digests.  It was not run by
this agent.

## 5. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R1-P1 | CONFIRMED | 116/116 ORCA1-ice admission, ordinary identity, 26 plants. |
| R1-P2 | CONFIRMED | T/S/u/v exact; SSH first non-bit at 0.015479333813968585 m. |
| R1-P3 | CONFIRMED WITH PROVENANCE LIMITATION | Card is exact to V2; only SSH differs to ORCA1-ice; compiled category-mass/global-adjustment statement is live.  The retained Phase-2x producer's preprocessed directory has been culled, so the cited compiled file is the current ORCA2 card's byte-pinned branch; the acquisition must retain its own compiled producer branch. |
| R1-P4 | **REFUTED** | kt=10 NEMO-root-differential SSH is 0.11244887791514496 m, not `[0.014,0.017]`. |
| R1-P5 | CONFIRMED | kt=2..10 post-`sbc` frames are absent; status `STOP_RECORD_GAP`. |

## 6. Citation, tests, and independent review

The receipt citation gate passed from `## Mechanically cited finding` with
one mapped compiled-source citation, no failures, no unmapped citations, no
map-audit failures, and all nine gate self-controls firing.  Its JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round1/citation_gate.json`
(SHA-256
`7ad7c29217c7b7b422ccf448417d8aba9c3034d2767fa0841d5cd71c4b50f06b`).
Shifting the ORCA2 citation by two lines made the first anchor fail at line
444 instead of 442 and exited 1 as required; planted JSON SHA-256 is
`47a949ce44e7efcebf1dcbba5f5bdb7c1a66c782b8f7ba2c7a60d344dbde166d`.

The final focused round-1 ladder plus citation tests passed 23/23 in 2.26
seconds, including the clean-worktree stamp and acquisition-producer binding
controls (log SHA-256
`ff3a4c86341117bac6277b432a80b8ce7cd8c201c8aa7257b895325636d85bbb`).
The required single `tests/ocean/fidelity -n 12` invocation collected 1,514
tests, reached beyond 95% with one displayed failure and seven skips, then
made no progress for more than the five-minute deadlock threshold used by the
preceding merge receipts.  It was interrupted and has no pytest summary, so
it is **INCOMPLETE**, not a pass.  No second broad battery was run.  The
preserved partial log SHA-256 is
`ee9c676e895d87fbb9cc8857ca8c4c103b341d5ed0f920b85fd16ef99945aae1`.
The separately isolated listed worktree-stamp ratchet passed 1/1.  The listed
pre-existing SI3 scalar-math provenance red reproduced 1/1 with
`GateError: A MY_SRC is not verbatim`.  Because xdist did not flush the broad
run's failing node ID, this receipt does not claim those two failures are the
same observation.

The required separate `codex exec --sandbox read-only` review was attempted
twice, including ephemeral mode, but both processes stopped before reading
the diff with `failed to initialize in-process app-server client: Read-only
file system (os error 30)`.  Verdict: **independent review unavailable
in-sandbox**.

## 7. No landing and GYRE disposition

No file below `packages/` changed.  Therefore the shared-model GYRE base/tip
trajectory requirement was not triggered; claiming a GYRE physics landing
would be misleading.  The first non-bit field is owned by the initial
category-load configuration, which is sea-ice-adjacent and explicitly outside
the autonomous ocean-only fix scope.  A one-statement landing was not
attempted.

## 8. OPEN / round-2 plan

1. The operator must run the committed acquisition unchanged and return its
   two admitted roots; until then, the kt=1..10 candidate ladder is
   `UNMEASURED_RECORD_GAP`.
2. User decision is required on the initial SSH target: the selected oracle
   root is one-category ORCA1-ice, while the frozen card computes a
   five-category load and its SI3 registry must not change.  This round's pick
   is to adopt the pinned root's global SSH load as an explicit ocean entry
   operand, without claiming or changing SI3 physics, but no implementation is
   authorized yet.
3. After both conditions close, round 2 must rerun the kt=1 entry gate, then
   enter stage 1 in NEMO statement order with exact per-step surface operands;
   the first non-bit statement owns the next walk.  It may not use the
   V2-vs-ORCA1-ice differential as a candidate score.
4. The historical kt=2 `zpelc` debt stays assigned to the shared GYRE lane and
   is later than the current kt=1 stop.
5. GitHub issue 1455 could not be read or updated because this clone has only
   a local filesystem remote and no GitHub connector.  The operator must post
   this verdict and acquisition handoff.

## Mechanically cited finding

The current ORCA2 compiled branch sums snow, ice, and pond mass over the live
category dimension, forms the domain-global levitating-ice sea-level
adjustment, and subtracts it from both ocean time levels at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459`.  This is the first
source statement capable of carrying the observed kt=1 SSH-only difference;
it precedes the ocean step loop.
