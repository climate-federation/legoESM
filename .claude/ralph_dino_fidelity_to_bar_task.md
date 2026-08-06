# Ralph task: DINO/NEMO fidelity sweep → AT BAR 45 | DEBT 0 | UNMEASURED 0

## Goal (completion ONLY when TRUE)
```
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
```
exits **0** and prints `AT BAR 45 | DEBT 0 | UNMEASURED 0`, with **zero** rows listed under
"AT BAR on CANCELLING statistics only". Bar: `corr >= 1-1e-9`, `|ratio-1| <= 1e-6`,
per-element `<= 1e-9`.

`--self-test` must still pass. **NEVER relax `BAR_CORR`, `BAR_RATIO_EPS`, `BAR_PER_ELEM_EPS`,
delete a `PER_ELEMENT` entry, or flip a `BINARY_GATES` verdict to clear a row.** Those constants
are the task. Weakening them is failure disguised as success, and the gate exists precisely
because that happened repeatedly before it did.

## THE RULE (human directive 2026-07-30 — on any conflict, this wins)
**NEVER GUESS. Every fix is a TRANSCRIPTION from NEMO source** — quote the
`file:line` in the commit message. If matching NEMO's code exactly does not
clear the row — or clears the row but not the behaviour — **PAUSE THAT ROW AND
ESCALATE to the human.** Do not invent stabilizers, plausible corrections, or
"equivalent" formulations (skill Rules 0 and 9). An escalation after an exact
match that failed is a SUCCESS of this process, not a failure.

## TOKEN ECONOMY (human directive 2026-07-30 — MEASURED, not guessed)
Subagent lanes are **~95% of this campaign's spend** (~30 lanes x 100-400k tokens on
2026-07-30; single lanes hit 220k/364k/400k). My context is a cached prefix and costs
single digits by comparison. So:
- **ONE Sonnet lane per iteration.** If a lane is still running, do gate + housekeeping and
  STOP — never stack a second.
- **CAP THE SCOPE IN THE BRIEF: "do steps 1-2, report, STOP."** Open-ended
  walk+A/B+decompose+generalise briefs ran 80-240 tool calls, long after the answer was
  visible. This is FREE savings — it loses no findings.
- **Briefs point at `docs/ocean/fidelity/dino_1226_state.md`** instead of re-typing the
  refuted-hypothesis lists, preconditions and trap catalogue inline.
- `Agent` calls set `model:"sonnet"` EXPLICITLY (omitting it inherits the expensive model).
- Loop cadence 3-hourly is a FLOOR; the human fires it manually for bursts.
**PROTECT: the adversarial reviews** (they caught a stale test encoding a bug as correct, a
wrong `Kmm` instruction, a missing mandatory `mask=`, a vacuous fp32 control) **and numeric
precision in any compression** — reconciling a new measurement against a recorded one is
what repeatedly saved this campaign, and that needs old numbers exact and findable.

## The METHOD that works (use it; it produced every win on 2026-07-28)
1. **Read the oracle's source first** for the term (`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/`,
   DINO overrides in `cfgs/DINO/MY_SRC/`). Write out the exact formula and its time levels.
2. **Walk the oracle's dumped INTERMEDIATES in execution order**, not the endpoints. Find the
   first stage whose err_norm jumps (>~100x its predecessor). That stage owns the defect.
   Endpoint comparison cannot see composition — a perfect KEG masked a completely wrong ZAD for a day.
3. **A/B the single factor that stage depends on.** One variable, everything else byte-identical.
4. Fix, re-measure, record in the gate WITH provenance, commit.

**If the needed intermediates are not dumped, ADD THEM.** The DINO build is on-box and rebuildable;
`cfgs/DINO/MY_SRC/ldfslp.F90` shows the pattern (units 8840-8848, guarded by a
`l_1226_raw_dump_done` first-call flag). This is the single highest-leverage action for the
big rows: `ldf_slp` closed in hours *because* its whole chain was dumped, while the 3e-6 band
resisted for weeks with endpoints only. Register every new dump in
`legoesm.ocean.fidelity.time_levels` with its `file:line`.

## Established 2026-07-28 (do NOT re-litigate)
**Two mechanical preconditions, both fail-closed — use them in every probe:**
- `ocean.fidelity.precision_gate.require_fp64(z_coord, T, S, context=...)`. NEMO is fp64; the
  legoESM policy defaults to **fp32** and `JAX_ENABLE_X64=1` does NOT change it. An f32 depth
  ladder silently corrupted every measurement in this campaign for weeks.
- `ocean.fidelity.time_levels.time_level_for_dump(...)`. NEMO runs `eos_rab(ts(Nbb), rab_b, Nnn)`
  — T/S BEFORE, geometry NOW. Feeding NOW T/S produces a *beautifully depth-structured* phantom
  signal (it is `|T_now - T_before|`, largest in the thermocline). This mistake was made twice in
  one day and its numbers were committed.

**Closed at machine roundoff:** `eos_rab alpha` (0.0), `eos_rab beta` (0.0, but trivially — beta is
CONSTANT for DINO's coefficients, so it carries no evidence about S-dependent terms), `bn2`
(1.41e-17), `zdf_mxl nmln` (0/9920 columns), `zdf_drg_nonlin` (0.0).

**Two production defects fixed, SAME root cause, two call sites** — the geometric EOS was fed the
STATIC depth ladder where NEMO uses the LIVE `gdept_0*(1+r3t)`:
`gm_redi_density_and_jacobian` (commit 9156aa13e) and `_nemo_wpoint_e3w_wmask_n2`'s
`slope_n2="nemo_bn2"` branch (commit 1b37ea059). **Suspect this defect at every other call site
that touches a depth ladder.** Both are gated: bit-identical when `t_depth_ref is None`, and
`nemo_bn2_live_ladders` honours `linear_free_surface` (key_linssh: r3t == 0).

**`ldf_slp` is TRANSCRIPTION-CLEAN.** Every dumped line verified: `zcj` mask count exact,
`zbj` three-way MIN branch (3/332214), `zfk` integer-division step EXACT (0/332214), and the
`zwslpj_hml` recurrence IS carried (`ldfslp.F90:209` is `DO jk = jpkm1,2,-1`, bottom-to-top, so a
vectorised gather is equivalent). Do not re-audit these.

**FALSIFIED — do not retry:**
- The 3e-6..5e-5 band is NOT float32. All four band probes re-run under fp64: not one moved.
- The vertical ladder does NOT explain the ACC deficit: the CORRECT ladder is 1.5 Sv *further*
  from NEMO (67.7 -> 66.2 vs NEMO 91.1).
- kappa_GM +-25% moves ACC by 0.1 Sv from rest (slopes are shallow during spin-up, so GM has no
  grip in that window — the test says nothing about equilibrium).
- The 16-point Shapiro smoother is NOT the ldf_slp defect (90%/87% of the error is present
  pre-smoother; >=100% in the bottom-3 levels, where it slightly REDUCES the error).

**metric_convention — DECIDED by the human 2026-07-28: GO FOR THE NEMO MATCH.**
NEMO's DINO `usrdef_hgr.F90:111-119` sets `pe2t = pe1t = ra*rad*COS(phi)*rn_e1_deg` — the
meridional scale factor IS the zonal one (Mercator conformality imposed analytically). legoESM's
default `"exact"` computes the true finite-difference `dy`; they differ at O(dphi^2), measured
median 2.798e-05 / max 8.241e-03. Reproducing NEMO's approximation is REQUIRED for fidelity.
- `zaj`/`wslpj` need `e2t` (T-face): already covered by the shipped option. MEASURED
  1.180e-06 -> 5.232e-11.
- `vslp` needs `e2v` = `dy_v` (v-point MERIDIONAL spacing). **DONE 2026-07-28**: `nemo_isotropic`
  now also sets `dy_v = R*dlon*cos(lat_v)`.
- **The "#516 adjointness invariant" blocker was a FALSE DILEMMA — do not re-raise it.** That
  invariant (strain/stress adjoint pair + divergence/advection mass consistency) is a property of
  `dx_v`, the ZONAL v-face length, via `vface_zonal_cos_lat`; it holds because operators SHARE the
  metric, not because of its value. `strain_rate_cgrid` takes a `LatLonGrid`, which has no `dy_v`
  field at all. All 9 #516 invariant tests pass unchanged with the new `dy_v`.
- The `test_vface_metric_invariant_under_metric_convention` guard was NARROWED deliberately
  (dx_v/area_q/cos_alpha_v/sin_alpha_v still asserted invariant) and the new `dy_v` behaviour is
  PINNED by two assertions. Do not widen it back, and do not weaken the remaining four.
REMAINING: wire the DINO bridge to select `nemo_isotropic` (it defaults to `"exact"` regardless of
the recipe card — a harness gap) and run ONE controlled pass recording before/after for every
affected row (`vslp`, `wslpj`, `zaj`, `ssh_nxt`, and anything else that moves). Then adversarial
review: this touches barotropic v-point areas (`1/(dx_v*dy_v)`), the PGF, and `latlon_cgrid_operators`.

## Guardrails
- **Reconcile before recording (skill Rule 1e).** When a new measurement disagrees with a recorded
  one, ONE OF THEM HAS A BUG and you do not know which. Do not explain away the old one. This cost
  two wrong commits on 2026-07-28; both times the NEW probe was the broken one (a missing
  `eos_depth="geometric"` kwarg; a wrong time level). Diff the CONFIGS field by field, not just the
  code — a silent default is the trap.
- **Controlled comparison, one variable.** Never attribute a test failure to your change without
  running the SAME command with the change stashed. A stashed-standalone vs suite-with-change
  comparison is two variables and nearly caused a correct fix to be reverted.
- **Conditioning: use the robust metric on sign-changing fields.** N^2, slopes and density
  anomalies pass through zero, so pointwise `|lego-nemo|/|nemo|` is meaningless there (its max
  balloons to 1e5-1e6 from near-zero cells). Use `err_norm = |lego-nemo| / RMS(nemo)` and report
  the near-zero fraction. Two findings were retracted as exactly this artifact.
- **Every production change: adversarial review** via a fresh `physics-validator` or
  `code-reviewer` subagent (no codex on this account) BEFORE declaring done. The review of the
  live-depth fix found the physics correct but the GATING wrong in two ways — both real defects.
- Every new `.py` gets a direct test. New gates ship a synthetic-violation self-test proving
  they are non-vacuous.
- Probes go in `scripts/validate/ocean_fidelity/dino_1226/`; throwaway in `scripts/tmp/`.
  Stage with explicit pathspecs, NEVER `git add .`/`-A`.
- Do NOT stage `docs/ocean/fidelity/mitgcm_oracle_status.md` or `pyproject.toml` (pre-existing edits).
- GPUs: pin with `CUDA_VISIBLE_DEVICES`, check `nvidia-smi` owners before killing anything.

## Per-row STOPPING RULE (prevents infinite churn)
A row may be marked `CONDITIONING-LIMITED` and escalated — **not** silently accepted — only when ALL
of these are measured and written down:
1. every input stage to it is at roundoff (err_norm <= 1e-9), AND
2. the transcription is verified line-by-line against the oracle source, AND
3. the amplification is DEMONSTRATED (e.g. `zbj` p99 7.07e-11 -> `zww_raw` p99 4.44e-04 through
   division by a vanishing denominator).
Anything less is an unfinished row, not a limit. If you cannot show all three, keep working. This
is the one place the bar can be missed, so it must be EARNED with evidence and reported to the
human — never asserted because a number is stubborn.

## HUMAN DECISIONS (2026-07-30) — the three prior escalate items are DECIDED
- `dyn_hpg dv` → **NEMO-match, recipe-gated.** Extend `nemo_isotropic` to the
  dynamics v-face metric (`e2v = ra*rad*COS(phi_v)*rn_e1_deg`, usrdef_hgr.F90:117)
  on NEMO cards ONLY. Credit the row ONLY after every #516 adjointness +
  conservation test passes UNDER the NEMO convention, machine-gated, same PR.
  All other recipes byte-identical (assert it).
- `zdf_mxl_turb` → **verify-unconsumed, then WAIVE.** First grep the DINO-active
  NEMO source proving nothing consumes the turbocline depth (the eiv taper uses
  the DENSITY-criterion hmlp, already matched 0/9920 — VERIFY, do not assume).
  Then record an explicit HUMAN-WAIVED row in the gate with the reason string +
  the grep evidence; the row stays visible, never deleted. If a consumer IS
  found, the waiver is void — escalate.
- `STABILITY on NEMO true grid` (restart-start blow-up, GM-implicated) →
  **fold into the eiv row.** Drive `eiv transport u/v` to bar by pure NEMO
  transcription; that row's EXIT CRITERION includes re-testing restart
  stability on `LEGOESM_NEMO_E3T=both` (20 d from the y5 restart: max|u| must
  hold ~0.6 m/s, not grow toward 3). If EXACT eiv still blows up → PAUSE +
  ESCALATE (that is THE RULE firing as designed, and it is the informative
  outcome).

## STILL ESCALATE (undecided)
- Any change that would alter non-fidelity recipes' numerics.
- Any row where the exact NEMO transcription is in place and the row still
  misses the bar (THE RULE).

### ESCALATION 1 (2026-07-30) — `zdftke` sh2 face-native: EXACT TRANSCRIPTION, ROW STILL MISSES
Commit `ca60df2ba`. THE RULE has fired. PAUSED pending human decision; loop moved on.
- Transcribed `zdfsh2.F90:78-94` (no-Stokes branch live: namelist_ref:597 `ln_stshear=.false.`,
  :212 `ln_wave=.false.`): u/v-FACE-NATIVE `zsh2u/zsh2v`, now(Kmm)xbefore(Kbb) cross term
  `(avm(i+1)+avm(i))*(uu(jk-1,Kmm)-uu(jk,Kmm))*(uu(jk-1,Kbb)-uu(jk,Kbb))/(e3uw(Kmm)*e3uw(Kbb))*wumask`,
  T-point combine `0.25*[(zsh2u(i-1)+zsh2u(i))*(2-umask(i-1)*umask(i)) + v-analogue]`;
  `wumask/wvmask` per dommsk.F90:176-182.
- **Faithfulness CONFIRMED**: independent NumPy loop-port matches 0.0 diff; review pass 1
  (physics-validator) found one stale test pin -> fixed; pass 2 (code-reviewer) CLEAN.
- **RESULT: restricted-to-signal corr 0.963 -> 0.983, ratio -> 0.9036. NOT ~1.0.**
  (Unrestricted 0.983/0.974 — but 95.2% of wet cells are below the |sh2|<1e-9 noise floor at this
  state, so ONLY the restricted number is meaningful. Do NOT quote the unrestricted one.)
- Dispatch: `nemo_face_native` SUPERSEDES `nemo_burchard` when selected (NEMO's live branch is
  always now x before; no face-native-but-squared-only variant exists) and composes the Burchard
  cross term internally. Default `squared_centered` BIT-IDENTICAL (`assert_array_equal`).
- **DECISION NEEDED**: (a) deeper walk of sh2's remaining inputs — `avm_in` is the outstanding
  suspect at cell ratios 1.04-1.10 and NEMO feeds the PREVIOUS step's avm — or (b) park as a
  known-limit row and move to higher-magnitude targets. **The ACC acceptance run (88m) measured
  this row's family as CLIMATE-INERT, which argues for (b).**

### ESCALATION 2 (2026-07-30) — a gate row's recorded pipeline DOES NOT EXIST IN THE REPO
`zdftke pdlr`'s provenance names `probe_zdftke_prandtl_e3tboth.py` (+ `_scan_pdlr.py`). **Neither
is in the repo tree** — they exist only in a session scratchpad. That row's numbers CANNOT be
reproduced from a clean checkout, and the "re-measure through the row's OWN pipeline" requirement
is unsatisfiable. The agent correctly refused to fabricate a substitute.
- PROVENANCE-CLASS defect, not a one-off: any row whose pipeline is a scratchpad script is
  unreproducible and its recorded tuple unverifiable.
- **DECISION NEEDED**: adopt the rule that a gate row's pipeline MUST be a committed script under
  `scripts/validate/ocean_fidelity/dino_1226/`, and audit which rows violate it? (The
  artifact-risk audit already in flight covers adjacent ground and could absorb this.)

### ESCALATION 3 (2026-07-30) — `ldf_slp wslpi/wslpj/uslp/vslp`: STOPPING RULE independently
RE-VERIFIED, all four qualify as CONDITIONING-LIMITED. ONE cause, four rows — consolidated here,
not written four times. Ratios today: wslpi 0.999962, wslpj 0.999957, uslp 1.000017, vslp
1.000021 (all still DEBT under `BAR_RATIO_EPS=1e-6` — classification NOT changed).
- **Condition 1 (roundoff of every input stage), RE-MEASURED TODAY, not relayed**: re-ran
  `ldf_slp_per_element.py` fresh (fp64, `LEGOESM_NEMO_E3T=both`, `RUN_GDB` restart
  `DINO_00057600_restart.nc`, kt=57601). `prd` err_norm = **1.804e-11** (note claims
  2.559e-6→1.804e-11 after the live-gdept fix — CONFIRMED, exact digit match). `zbw` err_norm =
  **9.369e-16** (note claims 3.467e-07→9.369e-16 after the slope-N² live-ladder fix — CONFIRMED,
  exact digit match). Both are far below `BAR_PER_ELEM_EPS=1e-9`. `zfk` integer-division stage:
  0/332214 cells differ (EXACT). CONDITION 1 HOLDS.
- **Condition 2 (transcription verified line-by-line), RE-READ TODAY against
  `cfgs/DINO/MY_SRC/ldfslp.F90` at current HEAD** (not the note's cached line numbers):
  `zcj = MAX(vmask sum, zeps) * e2t` at lines **307-308** (confirmed exact match); `zbj = MIN(zbw,
  -100*|zaj|, -7e3/e3w*|zaj|)` at line **317** (confirmed exact match); `zfk = REAL(1 -
  1/(1+jk/(nmln+1)))` Fortran-integer-division step at line **320** (confirmed exact match); the
  level loop `DO jk = jpkm1, 2, -1` at line **209** (confirmed exact match — bottom-to-top, so
  legoESM's vectorised `take_along_axis` gather for `zwslpj_hml` is the correct equivalent, not a
  missing recurrence). All four cited line numbers are UNCHANGED in the checked-out source.
  CONDITION 2 HOLDS.
- **Condition 3 (amplification demonstrated), RE-MEASURED TODAY via the same script's
  j-direction chain walk**: `zbj` p99 **7.067e-11** → `zww_raw` p99 **4.437e-04** (note's cited
  numbers — CONFIRMED, exact digit match), a 52358x jump through `zaj/(zbj-eps)` with `zbj`
  approaching the seafloor's near-zero-slope regime (deepest wet levels, columns j~185-191).
  Every intervening stage (`zaj`, `zbw`, `zfk`) sits at/below roundoff; only the division by a
  vanishing `zbj` amplifies it. CONDITION 3 HOLDS.
- **All three STOPPING RULE conditions hold for all four rows** (they share the single `zbj`
  divisor mechanism inside the same w-point slope kernel). DEBT classification is UNCHANGED (no
  `BAR_*` constant touched, no `PER_ELEMENT` value deleted/relaxed). Recorded in
  `fidelity_bar_gate.py`'s `MEASUREMENTS` notes for all four rows.
- **DECISION NEEDED**: these four rows are conditioning-limited on `zbj→0` at the seafloor, not a
  transcription defect — is there a NEMO-faithful way to raise this floor (e.g. does NEMO itself
  see the same `zbj` near-zero cells and get bailed out by something legoESM is missing upstream,
  or is this genuinely irreducible IEEE cancellation inherited from `prd`'s own residual)? If no
  such lever exists, these four rows are a permanent gate-debt residual and the campaign should
  stop chasing them and move to the ranked higher-magnitude rows (`dyn_spg_ts`, `eiv transport`,
  per the reordered WORK ORDER above).

## Current state (2026-07-30, loop restarted)
`AT BAR 11 | DEBT 31 | UNMEASURED 3`, 6 of the 11 still passing on cancelling statistics only.
Ranked worst-first: `dyn_spg_ts puu_b` 1.3e-2, `eiv transport v` 9.5e-3, `dyn_spg_ts un_adv`
8.4e-3, `dyn_adv ZAD` 4.9e-3, `ATF filter u` 4.4e-3, `eiv transport u` 3.9e-3, `dyn_ldf u` 3.9e-3,
`zdftke composite` 3.8e-3, `zdftke pdlr` 3.2e-3, then ~1e-3 and below.

**WHY THIS ORDER (context): the 2026-07-30 attribution excursion proved the free-run ACC deficit
is 100% density / deep / southern / thermal, that matched-state budget probes at sem precision see
"agreement" the roundoff gate refutes, and that the DEBT rows at 1e-3..1e-2 sit exactly in the
stratification chain. The exactness sweep IS the ACC fix (user doctrine + Phase-G precedent).**

**WORK ORDER — REORDERED 2026-07-30 BY MEASURED MAGNITUDE. The old order (zdftke first) is
SUPERSEDED; the reason is EVIDENCE, not preference:**
**THE ACC ACCEPTANCE RUN (addendum 88j) MEASURED THE FIRST THREE LANDED FIXES AS CLIMATE-INERT.**
Years 1-2, protocol byte-identical to the frozen baseline, harness self-validated first:
upper contrast 0.9082 -> 0.9082 (+0.0000, BIT-FLAT); dACC -0.0027 then -0.0051 Sv, accumulating
monotonically but **FOUR ORDERS below the 25.6 Sv deficit**. The fixes DO change the state
(max|dT| 0.26 degC) but change it in the WRONG PLACE — upper ocean, NORTH of the channel band,
while 80-98% of the missing thermal wind is sourced BELOW 1000 m IN THE SOUTHERN CHANNEL.
**⇒ ROWS IN THE 1e-6..1e-5 BAND DO NOT MOVE THE ACC. PRIORITY FOLLOWS RESIDUAL MAGNITUDE.**
(This does NOT refute the exactness doctrine — the tested fixes were ~1e-6; the big rows are
1e-2, four orders larger, and remain untested. It bounds optimism, it does not close the question.)
1. **`dyn_spg_ts puu_b` (1.3e-2) + `un_adv` (8.4e-3) — LARGEST RESIDUALS IN THE CAMPAIGN.**
   The advanced thread is `zu_frc` (8.03e-3) and the `zu_trd` ABSENCE (`zu_frc = zu_frc -
   zu_trd*ssumask`, dynspg_ts.F90:304, :367) — a pure transcription. BEFORE adding it, check
   whether legoESM's formulation makes the removal unnecessary by construction. NOTE the
   momentum-Jacobian suspect is CLOSED (r=+0.11, addendum 88h) — do not re-chase it.
2. **`eiv transport v` (9.5e-3) / `u` (3.9e-3)** — the deep-contrast lever, and the deep is
   where the deficit lives. Exit criterion includes the true-ladder restart-stability re-test.
3. **`dyn_adv ZAD` (4.9e-3), `ATF filter u` (4.4e-3), `dyn_ldf u` (3.9e-3).**
4. **`zdftke composite` (3.8e-3) / `pdlr` (3.2e-3)** — DEMOTED from #1. The formula is already
   exact (4.4e-16); the row's residual is in its production INPUTS, with `sh2` measured as the
   bad one (corr 0.9344) and a face-native transcription in flight. Finish that, then reassess.
5. Everything below ~1e-3, the metric-convention pass, `dyn_hpg dv`, and the cheap interleaves
   (cancelling rows, `vslp` i/j asymmetry, `lbc_lnk`-class one-offs) — LAST. They are gate
   debt, not ACC levers.
**NEW LANE, promoted out of the eiv exit criterion: the TRUE-LADDER RESTART INSTABILITY is now
BLOCKING, not merely open.** The only clean acceptance test must run on NEMO's true 3-D ladder,
but from a restart legoESM blows up (max|u| 0.66 -> 3 m/s over 20 d) and the baseline therefore
runs `LEGOESM_NEMO_E3T=off` — the known-wrong 1-D ladder, 12.9% off below k=25, i.e. WRONG
EXACTLY WHERE THE DEEP FIXES ARE SUPPOSED TO ACT. Deep-ocean fixes cannot be honestly evaluated
until this is fixed. Treat as a first-class row.

**ACCEPTANCE METRICS per subsystem (after a row reaches bar; do NOT wait on the ACC itself — it
scatters +-40%/yr in NEMO's own run):** re-run the from-rest twin and track (a) channel z(maxN2):
166.6 m now vs NEMO 110-127 m; (b) year-1 upper N-S density-contrast ratio: 0.908 now, target
1.0; (c) deep-contrast growth y1->y5. These are the ACC's leading indicators.

**COVERAGE (skill Rule 1, extended to the call graph): enumerate every CALL in stpmlf.F90's step
(descending one level through dispatch wrappers to the concrete routine DINO runs) and force a
disposition per routine — VERIFIED (number + where) / WAIVED (written reason) / UNVERIFIED
(listed loudly, ranked by climate leverage). Until that table exists, the 31 DEBT rows are a
checklist, not coverage; a term off the list is invisible by construction.**

## 2026-07-30 durable outputs (do NOT re-derive; detail in memory addenda 87-87j)
- **Instrument FIXED (commits 63fd21b04 + 63ecfc702): `box_heat_budget.py` dropped K33 AND
  `eos_depth` — its per-term split was NEVER valid for oracle comparison; only its sums were.
  Valid pairings now: lego `iso_redi + k33` <-> NEMO `ttrd_ldf + (ttrd_zdf - ttrd_zdfp)`;
  lego `vertmix` <-> NEMO `ttrd_zdfp`. Realized-increment buckets REQUIRE fp64 storage (see the
  module's PRECISION block — a small fp32 bucket is QUANTIZATION, not physical absence).**
- **Addenda 20/21 RETIRED.** The "vertical mixing 8x too weak" chain was the broken split; its
  downstream 0.46x/0.34x numbers must be RE-MEASURED (with the fixed instrument) before reuse.
- Exonerated WITH BOUNDS — but all matched-state, hence blind to through-state error (carry the
  caveat): momentum/form stress 0.02%; realized post-EVD K 0.9994 (CORE) / 0.9846 (SEDGE —
  note: by THIS gate's bar 0.9846 is itself DEBT); N2 form no-op at fp64; deep-band tracer
  TOTALS 0.7-1.2 sem; GM bolus ablation 0.013 W/m2.
- New traps beyond Rules 1c-1e: NEMO `avt_k`/`avm_k` are CLOSURE-ONLY (EVD applied to the
  copies avt/avm, never written back; no `iom_put('avt')` exists — zdfphy.F90:313-323); an MLF
  dump of `ts(Naa)` after a stage carries the WHOLE STEP'S accumulated RHS, not that stage's
  increment; `set_policy(fp32())` does not reach numpy-f64 leaves under x64 (any precision
  control must ASSERT the dtypes differ); constructors read `get_policy().storage` at BUILD
  time (`init_latlon_cgrid.py:97,172,224,284`), so fp64 states must be built under the policy.
- **TRUST SUMS, DISTRUST SPLITS: every wrong per-term conclusion of 2026-07-30 survived because
  the sums closed. A closing budget validates the OPERATOR, not the ATTRIBUTION. Before
  believing a split, prove bucket membership on BOTH sides with file:line.**

Report the gate line after every iteration. Stop when it prints `DEBT 0 | UNMEASURED 0` — or when
every remaining row is escalated per THE RULE with its evidence.
