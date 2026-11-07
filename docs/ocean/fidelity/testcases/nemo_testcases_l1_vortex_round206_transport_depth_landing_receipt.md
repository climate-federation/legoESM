# VORTEX round 20 (lane round 206) — landing the transport divisor

**ROUND_STATUS: LANDED** (Decision 86).  No new investigation; the statement
was proven in round 205 and approved by the user.  No production default,
scheme, deck value, bound or carried state changed — the transcription is
the only change.

## The statement, and where NEMO puts it

NEMO's RK3 stage builds the barotropic correction that enters the advective
transport as

```
zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj)/(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)
```

so the depth it divides the barotropic transport by is `hu_0*(1+r3u(Kmm))`,
the surface-height-ratio column depth.  legoESM divided the same transport
by the sum of its own MIN-RULE face thicknesses, which is first order wrong
in the sea-surface height difference ACROSS the face.

**The branch proof, which is why this is NOT card-scoped.**  The statement
sits in `CASE ( np_LIN, np_HYB )`, and `n_baro_upd = np_HYB` is the module
default assigned nowhere else in the compiled source.  The SAME line with
the SAME default is compiled into every card's `stprk3_stg.f90`:

| compiled build | `n_baro_upd = np_HYB` | the `zub` statement |
|---|---:|---:|
| `tests/VORTEX_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:270` |
| `tests/VORTEX_VEC_R8_OMIP_L1_P3` | `stprk3_stg.f90:50` | `stprk3_stg.f90:272` |
| `cfgs/GYRE_OMIP_L2_P3_SM` | `stprk3_stg.f90:49` | `stprk3_stg.f90:276` |
| `tests/LOCK_EXCHANGE_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:273` |
| `tests/OVERFLOW_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:274` |

(The line numbers differ only because the builds preprocess to different
lengths; the statements are character-identical.  Re-anchored by symbol, per
the citation rule, not carried over from the VORTEX build.)

So the fix lands in the SHARED RK3 path in
`ocean_model_latlon_cgrid.py` — the one `transport_target_u/v` every card's
WS-RK3 stage reconcile reads (`momentum_transport_reconcile` defaults
`True`) — and follows NEMO's branch for every card, rather than being scoped
to the card that exposed it.  **GYRE and the vector card are not asserted
inert; they are MEASURED inert, below.**

**SCOPE, stated rather than implied.**  The target is built ONCE per step
from the step-entry (Kbb) sea surface, while NEMO re-evaluates `:270` inside
every stage with that stage's own `r3u(Kmm)`, and legoESM's own
`_nemo_ws_stage_transport` does use the per-stage ssh.  That placement is
inherited from the structure this target already had (`H_u_pre` and
`Hu_avg` are both step-entry quantities); round 206 changed the depth RULE,
not the time level.  The time level is OPEN item 2.

## What moved — every card, every moved row registered

Measured at commit `0ab3f9999`, `--max-step 10 --continue-after-first`,
50 certified rows per card.  Evidence under `phase3/round206/`
(`traj_<case>_after.json`, `ulp_<case>.json`, `registry_summary.txt`).

### `VORTEX-zco` (the flux card) — the new reference

**40 of 50 rows moved: 39 TOWARD NEMO, 1 away, 10 unchanged.**  This
reproduces round 205's measured candidate to every printed digit.

| row | before | after | |
|---|---:|---:|---|
| `kt2.before.u` | `1.135534e-07` | `1.216831e-08` | TOWARD, 9.3x |
| `kt2.before.v` | `1.135465e-07` | `1.239432e-08` | TOWARD, 9.2x |
| `kt2.before.T` | `1.642800e-09` | `2.823263e-10` | TOWARD |
| `kt10.before.u` | `4.888403e-07` | `9.953976e-09` | TOWARD, 49x |
| `kt10.before.v` | `4.859896e-07` | `8.874215e-09` | TOWARD, 55x |
| `kt10.before.ssh` | `1.899224e-07` | `1.226045e-08` | TOWARD |
| `kt6.before.S` | `1.015061e-15` | `8.120488e-16` | TOWARD, **DEBT -> AT-BAR** |
| `kt5.before.S` | `6.090366e-16` | `8.120488e-16` | **AWAY**, AT-BAR both sides |

`first_over_bar` unchanged (`{T,u,v,ssh}` at `kt=2` on both sides); the only
row-status change is the DEBT -> AT-BAR above; **no AT-BAR row became DEBT.**

**The ratchet's own numbers, printed because nothing is hidden:**
`largest_oracle_residual_worsening_ulps 25133237.0` against a 2-ulp bar,
`largest_previous_legoesm_field_move_in_row_scale_oracle_ulps
2250763106.5`, **941,565 cells improved and 124,605 worsened** — the
worsening is 5.5 per cent of the move by ulp and 11.7 per cent by cell
count.  Status `FAIL`.  **Decision 86 authorises landing over this red
ratchet**; it is registered here, not waived silently.

### `VORTEX_VEC-zco` — inert, as predicted

**0 of 50 rows moved.**  `max_worsening_ulps 0`, 0 cells improved, 0
worsened, `first_over_bar` unchanged (`{u,v,ssh}` `kt=2`), status **PASS**.

### GYRE — byte-identical, so the shared landing does not disturb it

* Certified `kt=1..10` ladder: **954 rows, 0 moved, 0 ulp**, no status
  change, `first_over_bar` unchanged (`{T,S,u,v,ssh}` `kt=3`), **PASS**.
* 360-day from-rest year (note BW): **360 of 360 snapshots byte-identical**
  to the certified arm `lego_seed0_r203a`, 0 differing, 0 missing.

```
certified arm files: 360 | compared: 360 | differing: 0 | missing: 0
day030.npz certified 4e36c106403b495e round206 4e36c106403b495e
day240.npz certified a63befc30bf03b44 round206 a63befc30bf03b44
day360.npz certified dcb7bc46c8bc75bd round206 dcb7bc46c8bc75bd
```

So note BW's three certified day numbers stand to every digit: day 30
`2.3432465132112266e-06`, day 240 `6.58170624837412e-05`, day 360
`5.407736527246344e-05` K, carried by byte-identity of the snapshots they
are computed from.  The day-gap scorer was not re-run, and that is said here
rather than implied.

**The MECHANISM of GYRE's inertness is NOT established and is not claimed.**
The citation shows GYRE's own NEMO runs the identical statement, so the
inertness is on legoESM's side of the comparison, and an unexplained zero is
a finding, not a reassurance.  OPEN item 3.

### The tanks — note BY expected `0/50` each; **MEASUREMENT REFUTES THAT**

Both tanks move.  This is the round's one surprise and it is reported as a
finding, in the direction of NEMO on net.

| | `LOCK_EXCHANGE-zco` | `OVERFLOW-zps` |
|---|---|---|
| rows moved | 16 of 50 | 25 of 50 |
| toward NEMO / away | 14 / 2 | 20 / 5 |
| row-status changes | **8 DEBT -> AT-BAR** | none |
| `first_over_bar` | `{u}` kt=4 -> **kt=8** (LATER) | `{T,u}` kt=2, unchanged |
| ratchet worsening (ulp) | `3.0` | `404099481.78` |
| cells improved / worsened | 960 / 462 | 4501 / 5817 |

`LOCK_EXCHANGE` headline: `kt10.before.u` `1.137147e-11` ->
`9.291179e-15`; the eight rows leaving DEBT are `kt4/5/6/7.before.u` and
`kt7/8/9/10.before.T`.  Its two AWAY rows are `kt9/kt10.before.ssh`, both
AT-BAR on both sides, moving `1.354660e-17 -> 1.389981e-17` and
`3.285302e-17 -> 3.420150e-17`.

`OVERFLOW` headline: `kt10.before.u` `5.422695e-06` -> `5.414931e-06`,
`kt10.before.ssh` `6.265025e-07` -> `6.253960e-07`.  Its five AWAY rows are
`kt4/kt5/kt6.before.ssh` (DEBT both sides, moving away by 0.01 to 0.06 per
cent) and `kt6/kt9.before.S` (`6.090366e-16 -> 8.120488e-16`, AT-BAR both
sides).  **OVERFLOW is the one card where more cells worsen than improve
(5817 vs 4501) while the rows move toward NEMO** — on a partial-cell (zps)
card the min-rule and the qco depth differ most, so this is where the
statement bites hardest, and it deserves the operator's eye.  No AT-BAR row
became DEBT and `first_over_bar` did not move earlier on either tank, so
Rule 12 is satisfied on both; the cellwise ratchet is red on both.

### ORCA2 — NOT measured here, and the exemption is stated

`orca2_vector_een_c2` inherits `momentum_time_integrator="rk3_ws"` from the
shared testcase base, so it reaches the edited lines too.  It is NOT in
this lane's registry set and was not measured this round: ORCA2 is scored on
its own lane with its own rungs.  The landing reaches it at the note-BX
merge (round 207), and **ORCA2 must be measured there, not assumed inert** —
it is a z-partial-cell global card, i.e. the same geometry class as
`OVERFLOW-zps`, which is the card this statement moved most.

### DINO

The DINO month gate runs inside `land.sh` against reference
`2.053801168e-03` K with bar `2.244317642e-03`.  Its decisive line is
recorded with the push in the ledger.  DINO reaches the edited lines (its
recipe selects the WS-RK3 integrator), so a move there is possible and is to
be registered, not assumed away.

## Gates

| gate | result |
|---|---|
| GYRE certified `kt=1..10` ladder vs round 205 | **PASS**, 954 rows, 0 moved, `max_worsening_ulps 0` |
| GYRE 360-day year vs the certified arm | **byte-identical**, 360/360 |
| `VORTEX_VEC-zco` registry vs round 205 | **PASS**, 50 rows, 0 moved |
| `VORTEX-zco` registry vs round 205 | 40/50 moved, 39 toward; ratchet **RED** at `2.5133237e+07` ulp — landed under Decision 86 |
| `LOCK_EXCHANGE-zco` registry vs round 203 | 16/50 moved, 14 toward, 8 DEBT->AT-BAR, first-over-bar later; ratchet red at 3 ulp |
| `OVERFLOW-zps` registry vs round 203 | 25/50 moved, 20 toward; ratchet red at `4.0409948e+08` ulp |
| ratchet plants on a CLEAN pair (the vector card) | `worsen-3ulp` exit 1, `at-bar-to-debt` exit 1, `improve` exit **0** — the controls fire and the benign plant stays green |
| citation gate, `DEFAULT_RECEIPT` | **PASS** exit 0, `unmapped_citations []`, `map_entries_failing_audit []` |
| citation gate, shifted-citation plant | exit 2 (nonzero), as required |
| focused battery (serialized) | **105 passed, 1 failed** in 586.67s; the one failure is an EXPECTATION this round invalidated, fixed and re-run **10 passed** (`phase3/round206/focused_pytest.log`, `..._round205_module.log`) |
| generic NEMO-GYRE recipe gate | run by `land.sh` with the push gate |
| DINO month gate | run by `land.sh`; line quoted in the ledger |

## Non-vacuity

* **The pin fails when the statement is reverted.**  With the landed
  divisor put back to `H_u_pre` in the working tree,
  `test_the_legacy_min_rule_depth_arm_is_live_and_finite` FAILS with
  `assert 0.0 > 0.0` — the control arm becomes an exact no-op
  (`phase3/round206/nonvacuity_test_reverted.log`).  This is only true
  because the arm was rebuilt this round to TAKE the pre-round-206 target
  whole instead of re-multiplying the landed one; in its round-205 shape the
  test would have passed with either divisor, and the first reviewer caught
  exactly that.
* **The retired arm name raises.**  `"qco_depth"` is no longer a legal value
  of `momentum_transport_stage1_operand` and is added to the
  unknown-string-raises parametrisation.
* **The ratchet's plants are exercised on a pair it calls CLEAN** (the
  vector card's, which moved 0 of 50): the two violating plants go red and
  the benign one stays green.  Running them on the flux card's pair would
  prove nothing this round, because that pair is already red by design.
* **The arm is an exact control of the RHS path ONLY.**
  `_nemo_ws_velocity_stages` reads `transport_target_u/v` directly and the
  arm does not invert that consumer, so "bitwise revert" means the stage
  right-hand side, not every reader of the target.  Said rather than
  widened.
* **The registry numbers still cover the tip.**  They were measured at
  `0ab3f9999`; `git diff 0ab3f9999..HEAD` on the ocean module is
  COMMENT-ONLY (no non-comment line changed) across the four later commits,
  so nothing after the measurement can have moved a row.
* **The landed graph is the measured graph.**  The control arm's
  pre-round-206 target is built by a Python helper called only inside the
  arm, so production traces nothing extra and the registry above was
  measured on the code that ships.

## The one test this landing broke, and why the EXPECTATION was wrong

`test_the_prognostic_mean_arm_really_reads_uu_b` asserted that on the
card's unperturbed initial state the refuted `prognostic_mean` arm is an
EXACT no-op (`== 0.0`).  It now returns `3.552713678800501e-15` inside the
module's own battery — and exactly `0.0` when run alone on an idle machine,
which is why it took a serialized re-run to see it reproduce.

**The expectation was wrong, not the code.**  The two operands (the
prognostic `uu_b` and the re-reduced depth mean) are still equal on that
state; what changed is compilation.  With the arm on, the re-reduced mean
becomes dead code, and before this round the production target shared its
denominator `H_u_pre` with that mean, so the two programs fused the same
way.  The landed target divides by the qco depth instead, so they no longer
do.  Exact equality was a property of the previous build, never of the
statement.

The control's content is a SEPARATION, and it is intact: perturbing
`uu_b`/`vv_b` moves the arm away from production by
`1.9073486328125e-06`, 5.4e8 times the unperturbed residue.  The test now
asserts `unperturbed < 1e-12`, `perturbed > 1e-9` and
`perturbed > 1e6 * unperturbed`, which still fails if the arm stops reading
the operand.  Module re-run: **10 passed**.

## One cost owned

The round-204 addendum in the ledger already records that the card
trajectory gate needs BOTH `--max-step 10` AND `--continue-after-first` to
produce the certified 50-row registry, because `--max-step` defaults to 3.
I read the round-203 command list instead of that addendum and ran all four
cards twice before reaching the right span — two wasted card sweeps, same
mistake the previous round paid for.  The working command, for the next
round:

```
nemo_testcase_phase3_trajectory_gate.py --case <CASE> --max-step 10 \
    --continue-after-first --output <out.json>
```

## Independent review

One fresh `code-reviewer` subagent on the diff, told to refute it.

**First verdict: DO NOT SHIP**, four blockers, six minors.

1. **"The landed expression is not NEMO's statement, and is not bitwise."**
   NEMO multiplies by a STORED RECIPROCAL (`r1_hu_0/(1+r3u)`); the landing
   divides by `sum_k(e3u_0*(1+r3u))`.  The literal operand already exists one
   keyword away — `_nemo_ws_qco_stage_faces(..., include_reciprocals=True)`
   returns `r1_hu = r1_hu0/one_plus_r3u` (`vertical.py:253`), which
   `_nemo_stage_corrected_velocity` already consumes through
   `nemo_source_round`.  **ACCEPTED AS CORRECT; I verified the operand
   myself and did NOT take it this round.**  Decision 86 approved one
   measured candidate (the divide form, whose `1.216831e-08` is the number
   in the approval); switching association would change that number and put
   two statements in one landing.  It is OPEN item 1, with the citation.
2. **Per-stage scope wrong at stages 2 and 3, and a comment asserted the
   opposite.**  TAKEN: the false claim is deleted and the scope is stated.
   The placement is unchanged and is OPEN item 2.
3. **The renamed test was a smoke test, not a pin.**  TAKEN, and the
   reviewer's reasoning was exactly right: the old arm multiplied, so it
   stayed live after a revert.  The arm now takes the legacy target whole;
   the measured failure above is the proof.
4. **Shared-path change, one card measured.**  TAKEN by MEASUREMENT: GYRE
   ladder and year, both VORTEX cards and both tanks are all above.

Minors taken: the arm is now a bitwise, not merely algebraic, revert;
the round-205 receipt's stale `qco_depth` / `_held.patch` references are
amended; the dropped `jnp.maximum(..., 1e-10)` floor is preserved because
the dry-column fallback is `H_u_pre`, which carries that floor (said here
rather than left silent).  Minor NOT taken, and recorded instead: the
reviewer spot-checked seven re-anchored spans as correct but found two
citations in the gated receipt that were ALREADY wrong before this round and
that difflib faithfully relocated while still wrong — the fenced grep block
naming `:12478`/`:12864` for `tracer_combine` (real lines 13970/14384) and
`:6069-6089` for a WZV call.  Both are invisible to the gate and
pre-existing; fixing them is OPEN item 5, not this diff's business.  So
"re-anchored" here means mechanically relocated with zero unmapped
endpoints and a passing audit — not that every pre-existing citation was
re-verified.


**The re-anchor was got wrong TWICE before it was got right, and the gate's
own `audit_map()` caught it both times.**  A difflib map built against a
tree that then GROWS is stale, and this round's source grew twice after the
first pass (the scope note, then the form note).  Thirty, then twenty-eight
map entries ended up one block short of their anchor symbol;
`audit_map()` went 0 (lane tip) -> 30 -> 28 -> 0, the last only after both
files were restored to the lane tip and re-anchored in ONE pass against the
FINAL source (15295 -> 15336 lines, 40 spans in the map, 18 in the receipt,
zero unmapped).  Rule for the next round: re-anchor LAST, once, after the
final source edit, and check it with `audit_map()` — NOT with the gate's own
run, which audits only the spans the receipt happens to cite and passed
green while thirty map entries were broken.
## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Land the statement in the shared RK3 path rather than card-scoping it | ASKED — note BY requires NEMO's branch structure, and the branch proof above shows every card compiles the same default |
| Transcribe the held patch by hand onto the moved tip instead of applying it | forced; the round-205 review had already relocated the arm validation and the qco build, so the patch no longer applied |
| Keep the DIVIDE association rather than NEMO's stored reciprocal | DEFERRED, not chosen silently — reviewer blocker 1, registered as OPEN item 1 |
| Keep the once-per-step time level | forced; changing it is a second statement, registered as OPEN item 2 |
| Rebuild the control arm as a whole-array take | forced by reviewer blocker 3; it is what makes the pin non-vacuous |
| Run the ratchet plants on the vector pair rather than the flux pair | forced; the flux pair is red by design this round, so it cannot host a control |

**UNASKED list: empty.**

## OPEN, in order

1. **The divisor's ASSOCIATION.**  NEMO computes
   `un_adv * (r1_hu_0/(1+r3u))` — a multiply by a stored reciprocal that the
   repo already builds (`vertical.py:253`, reachable as
   `_nemo_ws_qco_stage_faces(..., include_reciprocals=True)`) — while the
   landed code divides by the summed depth.  Algebraically equal, not
   bitwise.  This is the next walk item on the flux card and it is cited
   and ready.
2. **The time level of the transport target**: once per step from the
   step-entry ssh here, per stage with `r3u(Kmm)` in NEMO.
3. **Why GYRE is byte-identical** when its own NEMO runs the same statement:
   measured, unexplained.  A zero that is not understood can hide a path
   that never executes.
4. **The flux card's remaining `kt=2` residual, `1.216831e-08`** (u;
   `1.239432e-08` v) and its later-step rows (`kt10` u `9.953976e-09`,
   v `8.874215e-09`, ssh `1.226045e-08`) — the next walk items under the
   agreed stopping rule, after item 1.
5. The two pre-existing wrong citations in the gated receipt named above.
6. **Note BX**: merge the ORCA2 lane to adopt the carried `fe3mask` in the
   frozen EEN coefficient; both VORTEX cards are at `rn_shlat=0` and must be
   PROVEN inert (0/50 each), not assumed.  **This is round 207.**
7. Decision 74's 30/15/10-km VORTEX ladder.  **Round 208+.**
