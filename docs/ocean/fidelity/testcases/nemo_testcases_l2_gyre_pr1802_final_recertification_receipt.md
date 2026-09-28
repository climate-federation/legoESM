# PR #1802 final pre-merge round — decisions 66 and 67, and the re-certification

**Status: D66 LANDED.  D67 IMPLEMENTED, MEASURED, AND THEN HELD — ITS PREMISE
IS REFUTED.**

Decision 66 is done and proven.  Decision 67 was implemented as asked, every
NEMO card was re-measured against the same tip, and then the second
adversarial review refuted the NEMO reading the decision rests on.  The refutation
is a grep and it is decisive, so the code is reverted to the before-arm form and
the question goes back to the user.  Nothing in the certified set moved either
way: GYRE's 70-row ladder and all 360 of its daily year snapshots, and both tank
ladders, are byte-identical with the change and without it.

* Before arm: the lane tip `e413aee94`.
* After arm: `a43397691`, this round's third commit.
* Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/pr1802_final/`.
* Run-to-run floor quoted throughout: `2e-10 K` (round 129).

---

## Decision 66 — the ORCA1 OMIP card keeps its unmasked surface anchor

**What changed.**  `scripts/run/run_omip_core2.py`, the ORCA1 TKE card, sets
`nemo_mxl0_surface_tmask=False` where the review round had set it True.  False
is also the library default, so the line is now a record of the choice rather
than a behaviour change; the flag itself, and NEMO's masked statement behind
it, stay exactly where the review round put them, selected by the NEMO-literal
cards only.

**The NEMO statement, cited.**  The compiled GYRE branch of the mixing-length
routine multiplies the surface stress by the surface land mask at
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:606`, then floors the result at
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:610`.  The mask is one on every
WET column, so masked and unmasked differ on LAND columns only, and the user's
instruction is that this card is not ours to switch.

**The proof, by value.**  The card's resolved TKE config and a one-step
mixing-length / diffusivity column were fingerprinted on this tip and on
GitHub `main` (`d7109d9b0`), by the review round's own method: every config
field by value, every array by SHA-256 of its bytes.  Probe:
`phase3/pr1802_final/orca1_tke_fingerprint.py`; outputs `orca1_fp_repo.json`
and `orca1_fp_legoESM.json` in the same directory.  The column is deliberately
chosen in the regime where the anchor BINDS (strong turbulence over weak
stratification, so the thickness sweeps carry the surface value down the
column); the probe's `--plant` arm doubles the anchor and moves every reported
hash on both trees, which is what makes a zero difference readable.

| anchor column, both `ln_zdfiwm` arms | GitHub `main` | this tip, D66 applied |
|---|---|---|
| wet column, stress 0.10 Pa | `0.7951003609964353` m | `0.7951003609964353` m |
| LAND column, stress 0.07 Pa | `0.5565702526975047` m | `0.5565702526975047` m |
| calm column, stress 0 Pa | `0.04` m | `0.001` m (`ln_zdfiwm`) / `1e-08` m |

The land-column value — the quantity decision 66 names — is restored exactly.
The remaining row is a different, older item, and it is reported rather than
silently carried: see "Behaviour changes for the PR body" below.

**Fails when reverted: YES.**  `tests/ocean/unit/test_nemo_card_opt_in_defaults.py::test_orca1_card_keeps_the_unmasked_ln_mxl0_anchor`
resolves the card from its own builder and asserts both the card's selection
and the library default.  Flipping the card back to the masked arm gives
`2 failed, 1 passed, 12 deselected`.

---

## Decision 67 — HELD, because the statement it transcribes does not execute

**What was asked.**  Give the replacement depth-mean that the barotropic
correction subtracts from the 3-D velocity the card's own
`barotropic_seed_face_depth` / `barotropic_seed_evaluation` pair, instead of the
generic minimum-rule reduction.

**What it was justified by, and why that is wrong.**  The reading was that NEMO
forms its barotropic velocity by dividing the accumulated transport by the
e1e2-weighted sea-surface-height-averaged face depth
(`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:835-842`).  That statement sits
inside

> `IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) THEN`

and **every card in this campaign sets `ln_dynadv_vec = .TRUE.`** — GYRE's own
resolved log says `Vector form: 2nd order centered scheme  ln_dynadv_vec = T`
and DINO's says the same.  So the cited branch never runs.  What NEMO actually
does on these cards is sum the substep VELOCITIES and divide by the weight sum,

> `IF( ln_dynadv_vec .OR. lk_linssh ) THEN    ! Sum velocities`
> `puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) + za1 * ua_e  (:,:)`

(`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:767-769`) and then
`puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) / r1_wgt1s`
(`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:802`) — **with no face depth in
it at all.**  Threading the ssh-average convention through would therefore
transcribe a convention NEMO does not use on any card under test.  This is the
campaign's own attribution rule: prove the path executes before citing the line.
I did not, and the second reviewer caught it.

**So the code is reverted** to the before-arm form at
`barotropic_latlon_cgrid.py:2648-2649`, with the refutation recorded at the call
site so the next reader does not repeat it.  The decision is not dropped — it is
returned to the user in "Behaviour changes", item 4, with the real statement
named.

**What the held candidate was measured to cost, so the user can decide with a
number.**  It was fully implemented and fully re-certified before being
reverted, and every one of those measurements stands:

* the threading is LIVE — one production GYRE step makes two calls to the shared
  depth-mean helper, and the second, the replacement reference, received the
  card's pair (recorded by `phase3/pr1802_final/d67_live_probe.py`);
* it is INERT on every card whose momentum integrator is `rk3_ws` — GYRE and
  both tanks are byte-identical with and without it, and multiplying the
  reference by `1 + 1e-6` inside the production module moves **0 of 70 ladder
  rows and 0 of 210 residual arrays**, because
  `rk3_stage_barotropic_correction` in `ocean_model_latlon_cgrid.py` replaces
  the velocity downstream against the carried `uu_b`/`vv_b`;
* it moves exactly one card — DINO, which does not run `rk3_ws` — by `7e-06 Sv`
  of 90-day ACC, one part in ten million (the table below).

**RETRACTION, kept loud.**  The previous round left a comment at this call site
saying that adopting the card's convention here "would change every certified
NEMO trajectory".  That is false: it changes none of them.  The comment is
replaced.

**Which cards select the pair** (read off the RESOLVED config, probe
`phase3/pr1802_final/d67_card_census.py`, output `d67_card_census.json`):

| card | seed pair | momentum integrator | consumes the correction? |
|---|---|---|---|
| GYRE-zco, LOCK_EXCHANGE-zco, OVERFLOW-zps | `nemo_ssh_avg` / `nemo_literal` | `rk3_ws` | no |
| DINO `nemo_dino_kamm`, `nemo_dino_kamm_mlf` | `nemo_ssh_avg` / `nemo_literal` | not `rk3_ws` | yes |
| every other DINO recipe | `min_rule` / `generic` | not `rk3_ws` | yes, but the pair is the default |

---

## The certified numbers, before and after

Every row was re-measured on BOTH arms with identical commands.  Nothing is
quoted from an older receipt.

### GYRE — the certified 70-row ladder

`nemo_testcase_l2_gyre_phase3_gate.py --trajectory-only --max-step 10`.

| row | before (`e413aee94`) | after (`a43397691`) |
|---|---|---|
| kt2 T | `1.4210854715202004e-14` | unchanged |
| kt2 S | `2.1316282072803006e-14` | unchanged |
| kt2 u | `8.326672684688674e-17` | unchanged |
| kt2 v | `9.71445146547012e-17` | unchanged |
| kt3 T | `4.940310525114455e-07` | unchanged |
| kt3 S | `4.0085410546453204e-08` | unchanged |
| first over bar | `{T,S,u,v,ssh}` at kt 3 | unchanged |
| barotropic first over bar | `{uu_b,vv_b}` at kt 2 | unchanged |
| all 14 report content keys | — | identical |
| 210 residual arrays | — | **0 unequal** |
| report digest, provenance stripped | `7ba15556de2de841` | `7ba15556de2de841` |

### GYRE — the from-rest year

`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag year`, scored by `nemo_testcase_l2_gyre_year_owners.py --day-gap` on the
eight certified checkpoints.

| day | before, T rms [K] | after, T rms [K] |
|---|---|---|
| 30 | `2.3276772050683987e-06` | `2.3276772050683987e-06` |
| 240 | `6.586171881479517e-05` | `6.586171881479517e-05` |
| 360 | `0.002670992385329469` | `0.002670992385329469` |

All **360 daily snapshots are byte-identical** by whole-file comparison, and
every non-provenance field of the two day-gap reports is equal.  This is a
stronger statement than the 2e-10 K floor: the two arms agree to the byte, so
the floor is not what is carrying the claim.

### The tanks

`nemo_testcase_phase3_trajectory_gate.py --case <case> --max-step 10
--continue-after-first`, both arms.

| case | before | after |
|---|---|---|
| LOCK_EXCHANGE-zco | DEBT, first over bar `{u}` at kt 4 | identical; 150 residual arrays, **0 unequal** |
| OVERFLOW-zps | DEBT, first over bar `{T,u}` at kt 2 | identical; 150 residual arrays, **0 unequal** |

LOCK_EXCHANGE is bit-unchanged, which was the condition on this round.
OVERFLOW's kt>=2 rows are its known debt (note BG, decision 71): they are
reported, not gated, and no row that was at the bar left it — the two reports
differ in no field at all.

NOTE, because it would otherwise read as a regression: the committed reference
documents `nemo_testcases_l1_phase3_ref_lock_trajectory_kt10.json` and
`..._overflow_trajectory_kt10.json` are OLDER than this tip (LOCK's kt2 T is
`1.6277349838370963e-13` there and `0.0` now, and its first-over-bar sits at
kt 2 rather than kt 4).  The tip is better than those files on 33 of 50 rows.
They are left alone here — refreshing them is a separate, deliberate act — and
the comparison above is tip-before against tip-after, which is the controlled
one.

### DINO — BLOCKED, and the blocker is not this round's

The DINO year certification is the from-rest annual screen
(`dino_year_screen_fullframe.py <recipe> <out.npz>`, 11,520 steps at 2,700 s,
scored by `compare_fullframe.py` on ACC, the sea-surface-height small-scale
ratio and the sea-surface-temperature correlation and bias — the table in
`docs/ocean/fidelity/dino_handoff_2026_07.md`, which reports the year-5 form of
the same protocol).  Reproduced here, it refuses on the NEMO DINO card with

> `raw-mesh e3w_int must contain only finite values > 0`  (`eos.py:742`)

**measured in the BEFORE worktree**, so the refusal is pre-existing and not
decision 66's or 67's.  It is the same class of DINO geometry refusal round 182
recorded and stopped on.  Two real defects were found and closed on the way
there, and the second is the reason this is reported rather than worked around:

1. The screen bridged NEMO's mesh without asking for the native degree
   latitudes, so a card selecting NEMO's literal latitude-dependent surface
   profile refused at its first step.  Fixed in commit `968dd5afc` with the same
   predicate the 90-day DINO twin already uses; cards that do not select the
   literal profile are byte-unchanged.
2. The remaining refusal is the raw-mesh thickness guard above, which is a
   model-side geometry question and outside this round.

**What DOES run on DINO at this tip**, and is therefore the DINO measurement
this round can make: the developed-state 90-day twin acceptance gate
(`acceptance_gate_90d.py --run-recipe nemo_dino_kamm_mlf`, #1492 item 2.2).
It was run as a matched pair, before arm from a detached worktree at the
before commit, after arm from this tip, identical commands.  **This is the only
card in the campaign that CONSUMES the statement decision 67 corrects**, and it
is the one card where a number moves:

| metric, 90-day twin from NEMO's day-180 state | before | after | NEMO day 90 | 5x threshold |
|---|---|---|---|---|
| ACC [Sv] | `65.390274` | `65.390267` | `65.369204` | `4.550e-01` |
| upper contrast < 1400 m [kg/m3] | `-0.288146` | `-0.288146` | `-0.288182` | `5.500e-04` |
| deep contrast > 1400 m [kg/m3] | `-0.011261` | `-0.011261` | `-0.011258` | `2.250e-04` |
| southern-band surface sigma MAX [kg/m3] | `0.909348` | `0.909348` | `0.909343` | `4.750e-04` |
| southern-band surface sigma MEAN, \|diff\| from NEMO | `1.023e-06` | `1.026e-06` | — | `4.750e-04` |
| verdict | `PASS 5 / FAIL 0` | `PASS 5 / FAIL 0` | — | level 5x |

The whole movement is `7e-06 Sv` of ACC, one part in ten million, four orders
inside the gate's own threshold and inside the ensemble noise floor the
threshold is derived from.  Both arms pass all five metrics at the 5x level,
and the gate's two instrument self-checks passed on both
(`NEMO y10 ACC through THIS harness: 121.07 Sv vs recorded 121.07 Sv`).
So: the statement lands, the card that runs it moves at its last bits, and
nothing crosses a bar.

SELF-CAUGHT ERROR, recorded rather than quietly fixed: the FIRST after-arm twin
integrated all 2,880 steps and stayed stable, then correctly REFUSED to save —
`producing checkout changed during integration` — because I committed to the
repository while it was running.  That is the harness doing exactly its job.
The arm was re-run on a frozen tree, and the refused run's numbers were
discarded, not reported.

---

## Gate results

One battery at a time on this host, every log under
`phase3/pr1802_final/logs/`.

| gate | result |
|---|---|
| card gates: both DINO recipes, both tanks, the tank zero-diffusion removal, the DINO mesh / from-rest / step-1 / rank-dump gates | `221 passed, 9 warnings in 503.11s` |
| the six-file push gate, on the committed tip, clean tree | `135 passed in 1298.87s (0:21:38)` |
| the four CI ratchets | `1 failed, 10114 passed, 4 skipped in 109.21s` |
| receipt citation gate, cumulative default receipt | `PASS`, 274 citations, 0 failures, 0 unmapped, 0 map-audit failures, all 9 self-tests fired |
| the same gate with a planted shift | `FAIL` and exit 1 — it can fail |
| receipt citation gate, THIS receipt | `PASS`, 7 citations, 0 failures, 0 unmapped |
| DINO 90-day twin acceptance gate, both arms | `PASS 5 / FAIL 0` each, at level 5x |
| the same gate with a planted shift on this receipt's own NEMO citation | `FAIL` and exit 1 |

The single ratchet failure is `scripts/validate/cg_helmholtz_mixed_precision_1675.py`
(an Earth-radius literal).  It is not this round's: no commit here touches that
file.  The review round's OTHER known ratchet failure, the reference-salinity
literal in `fesom_integration.py`, is GREEN now.

The citation work this round is larger than the two new statements, because the
gate found three real defects:

* the surface-anchor statement was cited at the line range of the FLOOR that
  follows it in every one of seven places; re-anchored on the masked stress
  itself (commit `8345d7533`);
* that one-line comment fix then shifted three unrelated citations, caught by
  the gate rather than by arithmetic, and re-anchored in the receipt and in the
  map (`git log` of this round);
* this receipt's own two legoESM-side citations are pinned by symbol AND
  occurrence, so a second copy of the same text cannot satisfy them.

---

## Review

Both reviewers ran on this diff, adversarially, before it was declared done.

**Claude `code-reviewer`, independent fresh context: SHIP.**  No blocker, no
high.  It independently CONFIRMED the inertness reading from the code rather
than from my control — it traced that on the GYRE/tank path `target_u` is
sourced from `state_new.uu_b.data` and the reference thickness, never from
`state_new.u`, which is the output decision 67's threading touches.  It
verified the calm-column attribution (no commit here touches the floor
machinery), re-ran the new test and measured that it fails on revert, checked
`_depth_average_to_faces`'s defaults against the old implicit call, found no
mixed time level in `eta_dyn=_eta_corr`, and spot-checked all five NEMO
citations against the real oracle build.

Its one substantive ask — that the call site itself say the result is currently
unconsumed on the RK3 cards, not just the receipt — is closed by commit
`2968c0bb9`.  Its second note, that the latent-but-correct consumption on the
DINO recipes was PLAUSIBLE rather than CONFIRMED for it, is now CONFIRMED by
measurement: the DINO 90-day pair above moves, and it is the only card that
does.

**codex `exec --sandbox read-only`: DO NOT SHIP**, one blocking finding, quoted
verbatim:

> D67 cites an inactive GYRE branch as justification for the cards where the
> change is live. DINO sets `ln_dynadv_vec=T`; its compiled NEMO path accumulates
> velocities directly, without the cited SSH-depth division. The RK3 cards
> discard D67's result, leaving DINO as the affected but unsupported path.

**The two reviewers disagreed, so the discriminating check was run rather than
averaged.**  It is a grep, and codex is right — and it is worse than it said:
GYRE sets `ln_dynadv_vec = .TRUE.` as well, so the cited branch is inactive on
EVERY card, not only DINO.  Decision 67 is held on that measurement.  Codex's
second point, that the receipt claimed a re-certification it had not finished
for DINO, was also fair at the time it read the file; the DINO 90-day pair has
since completed on both arms and is in the table above.

## Known-red list

Carried forward, each checked to reproduce without this round's commits:

* `scripts/validate/cg_helmholtz_mixed_precision_1675.py` — an Earth-radius
  literal the constants ratchet rejects.  New on this branch, not this round's.
* `packages/ocean/legoesm/ocean/physics/vertical_mixing/fesom_integration.py` —
  a reference-salinity literal, same ratchet, same provenance.
* `tests/ocean/unit/test_tke_carried_coefficients.py::test_step_entry_n2_bundle_*`
  (two) — reproduce on the lane's pre-fix tip.
* The DINO from-rest year screen's raw-mesh thickness refusal, above.

## Behaviour changes for the PR body

1. **The ORCA1 OMIP card keeps its unmasked surface mixing-length anchor.**
   NEMO's masked statement stays available to the NEMO-literal cards only.
   Land columns on that card behave as they did before the branch.
2. **No change to the barotropic correction ships.**  Decision 67 was built,
   measured and then held; the code is back to the before-arm form, so no
   certified trajectory moves.
3. **A separate, older ORCA1 difference is still open and is NOT decision 66's.**
   The surface anchor's FLOOR moved from the card's `rn_mxl0` field (0.04 m on
   `main`) to the card's mixing-length floor (1e-3 m under `ln_zdfiwm`, 1e-8 m
   otherwise).  Two earlier lane items own it: rounds 55/56, which made the
   anchor read the derived floor, and PR #1749, which set the two `ln_zdfiwm`
   floors from NEMO's own namelist.  NEMO agrees with the lane here — under
   `ln_mxl0` it overwrites `rn_mxl0` with the derived floor at
   `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:831` — so this is very likely a
   fix rather than a regression, but it changes Pierre's card on calm and land
   columns and it was never asked.  **DECISION NEEDED:** keep it (NEMO's value)
   or restore 0.04 m on the ORCA1 card only.  Recommendation: keep it, and say
   so in the PR body.
4. **DECISION NEEDED, decision 67.**  Its premise is refuted: the NEMO statement
   it transcribes sits in a branch no card runs.  What NEMO does run on these
   cards has NO face depth in the barotropic velocity at all — it is a
   weight-averaged mean of the substep velocities.  Options, with my pick:
   (a) **leave it held**, which is what ships today and is what I recommend
   until someone walks the real statement; (b) re-aim it at
   `dynspg_ts.f90:767-769,802`, which is a different change to a different
   operator and needs its own round; (c) land the held version anyway, which
   costs `7e-06 Sv` of DINO 90-day ACC and buys a convention NEMO does not use.
5. **The DINO year certification is not re-run in this round** — the from-rest
   screen refuses on a pre-existing geometry guard.  The 90-day developed twin
   ran instead, on both arms, and is above.

## OPEN

1. Decision 67's real statement: walk `dynspg_ts.f90:767-769,802` — the
   velocity-sum branch that actually runs — and decide what legoESM's
   replacement reference should be under it.  The held patch is in this round's
   history if it is ever wanted.
2. The DINO from-rest year screen's raw-mesh thickness refusal — the last thing
   standing between this lane and a re-certified DINO year.
3. The ORCA1 anchor-floor decision in "Behaviour changes", item 3.
4. Refresh the two committed tank trajectory reference documents, which are
   older than the tip they are compared against.
