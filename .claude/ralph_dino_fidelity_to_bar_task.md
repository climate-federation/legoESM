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

**Known and measured, needs its own controlled pass:** `metric_convention="nemo_isotropic"` OWNS
`zaj` (1.180e-06 -> 5.232e-11) because NEMO's DINO builds the grid isotropically (`pe1t = pe2t`).
The bridge defaults to `"exact"` regardless of the recipe card — a harness gap. Adopting it also
moves `ssh_nxt`, so it needs a single controlled change, not a smuggled one.

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

## ESCALATE to the human (do NOT decide these alone)
- `dyn_hpg dv`: blocked on the deferred v-face metric. Genuine design tension between the #516
  adjointness/mass-consistency invariants and oracle fidelity. Someone must choose.
- `zdf_mxl_turb`: legoESM has NO equivalent. Implement, or waive explicitly (it is diagnostic-only
  in NEMO and never feeds dynamics). A waiver is a human decision.
- Adopting `metric_convention="nemo_isotropic"` as the DINO default (moves several rows at once).
- `STABILITY on NEMO true grid`: from rest it now runs 5 years stable, but the RESTART-start
  blow-up (max|u| 0.66 -> 3 m/s over 20 d) is real and unfixed. Open-ended bug hunt.
- Any change that would alter non-fidelity recipes' numerics.

## Current state (2026-07-28)
`AT BAR 11 | DEBT 31 | UNMEASURED 3`, 6 of the 11 still passing on cancelling statistics only.
Ranked worst-first: `dyn_spg_ts puu_b` 1.3e-2, `eiv transport v` 9.5e-3, `dyn_spg_ts un_adv`
8.4e-3, `dyn_adv ZAD` 4.9e-3, `ATF filter u` 4.4e-3, `eiv transport u` 3.9e-3, `dyn_ldf u` 3.9e-3,
`zdftke composite` 3.8e-3, `zdftke pdlr` 3.2e-3, then ~1e-3 and below.

**Suggested order:** (a) the 6 cancelling-only rows — cheapest, they only need a per-element
measurement, and that gap is exactly what let `bn2` sit falsely AT BAR; (b) instrument NEMO for
`dyn_spg_ts`/`zdftke`/`dyn_vor`/`traadv_fct`, then walk them; (c) the metric-convention controlled
pass; (d) `vslp`'s i/j asymmetry (uslp 3.6e-10 vs vslp 1.6e-6 from the SAME NEMO block —
suspect `r1_e2v`, the smoother's axis, or `ikv`'s neighbour shift).

Report the gate line after every iteration. Stop when it prints `DEBT 0 | UNMEASURED 0` — or when
every remaining row is escalated above with its evidence.
