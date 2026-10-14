# ORCA2 round 275 — certified G1/G2 source-unit bisect

**Status: HELD.**  The round-274 thirteen-selector activation has been split
into source units on the independent rung-0 ladder.  Five units are
individually net-toward there, but their cumulative production selection is
majority-away on OMT-1, OMT-2, and OMT-3.  Decision 96 therefore forbids the
landing.  All production selectors are left OFF; the G2 implementation and
the fail-closed private measurement gate remain selector-gated.

## Frozen protocol and records

The frozen predictions and falsifiers are in
`PREREG_nemo_testcases_l4_orca2_round275.md`, committed before the first
measurement.  Every number below is **independent** (the corrected legoESM
initial state), never the NEMO-entry twin label unless explicitly stated.
The common control is round 274's admitted `rung0_control.json`; the oracle is
the admitted round-90 rank-complete entry/stage record.  Execution was CPU,
JIT enabled, fp64/libm.

NEMO's executing statements are in the compiled ORCA2 source:

- `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-424`
  — stage RHS and velocity/update program;
- `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770`
  and `:784-785` — the ORCA2 `np_CRV` EEN circulation and trend;
- `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:191-205`
  and `:360-375` — the distinct live face and 1-D interface thickness
  divisors of the U/V implicit momentum matrices.

The ORCA2 deck resolves vector-form momentum advection and `np_CRV`; therefore
the one-level G1 `np_COR`/flux-form claims are not silently treated as proofs
of their ORCA2 branches.  The G2 divisor statement is directly executed:
ORCA2 has 31 levels and partial cells.

## Rung-0 source-unit census

Counts are RMS directions over bit-moved ladder rows, shown as
`toward / away / score-equal`.  Every constructible arm retained the first
non-bit row at kt=1 stage-1 T and lost zero exact rows.

| source unit | toward / away / equal | verdict |
|---|---:|---|
| EEN ratio-fold + fe3mask-pivot atomic pair | 114 / 61 / 0 | qualifies on rung 0 |
| EEN ratio written order | 59 / 70 / 32 | REFUTED, majority away |
| EEN vorticity written order | 57 / 123 / 14 | REFUTED; G1's planetary-only proof does not close ORCA2 `np_CRV` |
| stored T-ratio product | 73 / 67 / 29 | qualifies on rung 0 |
| stage T-ratio literal order | 68 / 66 / 34 | qualifies on rung 0 |
| stage velocity written order | 89 / 82 / 18 | qualifies on rung 0 |
| stage F-ratio literal order | 39 / 73 / 34 | REFUTED, majority away |
| stage U/V-ratio literal order | 59 / 74 / 15 | REFUTED, majority away |
| ZDF barotropic residual order | 77 / 93 / 7 | REFUTED, majority away |
| completed-stage top-V fold exchange | 23 / 165 / 0 | REFUTED; explosive, kt=10 SSH max 0.384033466829 -> 8.38845402852 m |
| stage-1 RHS literal handoff | 0 / 0 / 0 | inert on ORCA2's executed branch |
| EOS rounded-depth product | 0 / 0 / 0 | inert at this boundary |
| G2 momentum thickness divisors | 132 / 47 / 0 | qualifies on rung 0; kt=10 SSH max 0.384033466829 -> 0.383539664889 m |

The gate refuses either half of the EEN fold coefficient pair by itself:
both planted attempts returned `barotropic_een_ratio_fold_exchange and
barotropic_een_fe3mask_pivot_row are one fold unit: select both or neither`.
The registered `extra-selector` plant also fired (exit 2), proving that an arm
cannot inherit an unregistered selector.

Preregistration R275-P1 predicted that the RK3 stage-update unit would own the
round-274 growth.  It is **REFUTED**: the combined stage-1-RHS + velocity-order
arm is 89 / 82 / 18 and has no explosive growth.  R275-P4 is **CONFIRMED on
rung 0**: the G2 divisor arm is 132 / 47 / 0.  Neither result is promoted past
the cross-rung gate below.

## Atomic promotion attempt

The five qualifying source units were selected in separate commits, then
measured together.  On rung 0 the cumulative set passes Decision 96:
135 / 50 / 4, zero exact-row losses, first debt unchanged.  At kt=10 stage 3,
the RMS changes are T 0.00593198529410 -> 0.00589623560204 K, S
0.00145644015616 -> 0.00144237065435 PSU, u 0.00440170650728 ->
0.00438495192825 m/s, v 0.00431446337727 -> 0.00429552072188 m/s, and ssh
0.0268325914693 -> 0.0268376308700 m.  SSH maximum improves
0.384033466829 -> 0.383539664887 m.

The required OMT cross-rung gate vetoes that selection:

| card/label | toward / away / equal | exact losses | terminal SSH max before -> after |
|---|---:|---:|---:|
| OMT-1 independent, kt=1..8 | 66 / 83 / 6 | 0 | 0.746627180790 -> 0.746717960307 m |
| OMT-2 independent, kt=1..10 | 78 / 104 / 13 | 0 | 0.346486937521 -> 0.346451366223 m |
| OMT-3 independent, kt=1..10 | 79 / 110 / 4 | 0 | 0.363532594273 -> 0.363550379935 m |
| OMT-4 independent, kt=1..10 | 166 / 27 / 2 | 0 | 16.3373754582 -> 0.393531689285 m |
| OMT-4 given NEMO's entry, kt=1..10 | 166 / 27 / 2 | 0 | 16.3373754582 -> 0.393531689285 m |

Both OMT-4 labels are identical because the admitted card entry is bit-exact.
OMT-1 through OMT-3 also report bit-exact entry identity, so their independent
label is the discriminating trajectory.  The majority-away rows on three
rungs are hard Decision-96 reds even though rung 0 and OMT-4 improve.  The
production selections were therefore reverted in commit form before the test
and citation gates.  No configuration choice, stabiliser, floor, or clip was
introduced.

Evidence JSON SHA256:

- rung-0 cumulative: `3923f39b15f7170076729ee526eb7d39bd60bd8080258609bdcc7c4f713632bc`;
- OMT-1: `e0abf3d640195c03d51f2488a788b0894bdb011fd298b34ca64fd1bef7f8166a`;
- OMT-2: `d729d6141cb9208583f089f5be27462df9150001049d9c66b1387a97e2eed279`;
- OMT-3: `873886b4b3026a38e36709ff58d56c78410a4a49b80f7e927401e7823f4d418a`;
- OMT-4: `8197b5523943d29fd7e45cbb0a1afc92e769a06d2401b1764b8617a7e0d905be`.

## Gate disposition

- The final focused recipe/selector suite passes **30/30**.  The three G1
  config-layout tests that used brittle negative tuple offsets were updated
  to assert named-field order after the G2 selector was inserted; their
  isolated rerun passes 3/3.
- The imported G2 implementation remains false on every production card.
  GYRE's tip ladder is byte-identical to the round-274 baseline: 70/70 rows
  unchanged, zero ULP worsening, first-over-bar step 3 unchanged, and
  residual archive SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5`.
  The required 30-day member is byte-identical at all 30 daily snapshots;
  day 30 remains
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
  Only the manifest differs, as it stamps the new producer commit/root.
- The required `tests/ocean/fidelity -n 12` battery was run once.  It
  collected 3,582 tests, reached 99%, lost two G1 workers, and then stalled
  without a terminal summary; it was terminated and not rerun.  Both crashed
  node IDs pass serially (1/1 each).  Three selector-layout failures caused by
  this round pass after the named-field repair.  Eight isolated reds reproduce
  and are pre-existing at base `b7bf36941`: the registered GYRE round-129
  harness stamp and SI3 `MY_SRC` provenance reds; the worktree-stamp and
  allow-dirty-scope ratchets; OMT-1's stale "only vector module" assertion;
  round 228's stale hard-coded citation; and two stale GYRE round-33 mutation
  anchors.  The relevant test files and recipe are byte-identical to base (or,
  for the citation, base already maps `693-747` while its receipt says
  `688-742`).
- Default and round-receipt citation gates pass with no unmapped citations;
  the shifted compiled `dynzdf` plant fires.  Independent review is
  **independent review unavailable in-sandbox**: `codex exec --sandbox
  read-only` failed before review because its app-server client could not
  initialize on the read-only filesystem.

## OPEN

Round 276 should bisect the five rung-0 qualifiers **one at a time on OMT-1**,
then only the survivors on OMT-2 and OMT-3.  The first unit that changes the
vote from rung-0 net-toward to OMT majority-away names the module-dependent
interaction.  The completed-stage top-V-row exchange stays OFF and registered
as the isolated explosive singleton.  The next physics walk resumes only
after a source-valid unit passes every OMT rung independently.
