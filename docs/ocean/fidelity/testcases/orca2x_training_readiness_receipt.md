# ORCA2X round 2 — training readiness: multi-step rung-10 gradients

Date: 2026-10-24 (Claude round; codex review: operator). Revised after
review (Argon LAND; codex HOLD on claim wording): claims only, no new runs.
Status: **STOPPED_FOR_DECISION**: measurement only, no model or
configuration change. The horizon and remat choice are the user's call (below).
Base `484b60f5b7da`. Preregistration `afaf87cf02` (frozen before any
measurement). Probe `nemo_testcase_l4_orca2x_r2_multistep_grad.py`
(jobs ran at `29c084162b6d`/`8a65299cb38d`, probe unchanged between them
except the summary mode). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round2/`
(`jobs/*.json`, `*.log`, `*.done`, `*.free`). CPU, fp64, x64, libm policy.

## Verdict

- Multi-step rung-10 gradients are usable at every horizon the record
  allows (N = 1, 2, 5, 10), for both losses. No non-finite value anywhere
  it was looked for: reverse mode checked every initial-state float leaf and
  rn_Ud; forward mode checked only the two directions it ran, T_dir0 and
  rn_Ud.
- Reverse vs forward, worst over N = 1..10, per loss and direction:
  T loss along T_dir0 <= 3.3e-14; SSH loss along T_dir0 <= 8.3e-13;
  T loss on rn_Ud <= 1.8e-13; SSH loss on rn_Ud <= 2.6e-11 (at N = 10).
  Central differences on rn_Ud agree to <= 1.4e-5.
- Memory: compile residue dominates. After compile the process already
  holds ~73 GiB, so with per-step remat N = 2, 5, 10 all peak at
  94-97 GiB (101-104 GB decimal). The run itself adds a flat ~24 GiB.
- Time grows with N: 134 / 338 / 528 s per fwd+bwd per loss at
  N = 2 / 5 / 10 (remat), and each horizon's compile takes ~41-43 min.
- Without remat, N = 2 is the limit (91 GiB); N = 5 and N = 10 do not
  compile.
- Training blind spot (Argon, pre-existing, no change now): MLE stops the
  gradient on the mixed-layer depth and the mixed-layer membership mask
  (`mle_mld_and_buoyancy` returns both through `jax.lax.stop_gradient`;
  the mixed-layer buoyancy itself stays differentiable). No gradient flows
  through mixed-layer depth changes on any card that runs MLE. Whether
  rung 10 runs MLE was not re-checked this round.
- No gradient-only guard was needed, so item 3 has nothing to localise.

## Setup (as asked)

- Card: rung 10 as the NEMO comparison runs it (the ladder's comparison
  card, DECISIONS 150/154/157), stepped from its own initial state. This is
  the same card and state that round 330 used at N = 1.
- Path: the model's `integrate_scan` with per-step inputs (surface forcing,
  freshwater, damping target) and `nemo_kt0 = 1`. Step 1 runs outside the
  scan (cold-start barotropic history) and steps 2..N run inside it.
- Trainable: rn_Ud alone, collected by the rung-10 trainer entry
  (DECISION 149) with start value 0.018. It is spliced into the config
  inside the loss, so its leaf is traced. The step body is the model's own
  jit body on a model copy that carries the traced value.
- Losses: round 315's area/volume-weighted T^2 and SSH^2 of the final
  state.
- Reverse mode is one fused `jit(vjp)` with respect to every float leaf of
  the initial state and the raw rn_Ud leaf, run once per loss. Forward mode
  is a separate `jit(jvp)`. Central differences use the raw leaf at
  eps = 1e-2, 1e-4, 1e-6.
- Remat = `jax.checkpoint` around every step, step 1 included.
- One N per process. Each process was stopped at 100 GiB resident or below
  150 GiB machine-available (round 317's guard, by PID). Six processes ran
  at once, each under its cap; the lowest available memory seen was
  458 GiB. Machine load was 14-27 on 80 cores, so wall times are upper
  bounds.

## Instrument check

N = 1 reproduces round 330's T-loss numbers digit for digit
(|dL/dT0| = 0.01931092431777193, dL/dln(rn_Ud) = -4.511704487033998e-05);
the SSH reverse along T_dir0 equals round 330's to 1e-14 relative. Program A's primal loss equals
production `model.step` bit for bit at N = 1 and N = 2 (T). At N >= 2 it
differs by <= 1.4e-14 relative, as P6 allowed.

## Results (remat = per-step checkpoint)

| N | mode | non-finite (T / SSH) | rev vs fwd, T_dir0 (T / SSH) | rev vs fwd, rn_Ud (T / SSH) | FD best, rn_Ud (T / SSH) | \|dL_T/dT0\| | \|dL_SSH/dT0\| | dL_T/dln rn_Ud | peak RSS, process | run growth | fwd+bwd per loss | compile (reverse) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | remat | 0 / 0 | 7.7e-15 / 1.1e-14 | 1.4e-15 / 0 | 2.9e-6 / exact 0 | 1.9311e-2 | 6.857e-4 | -4.512e-5 | 48 GiB | 12.5 GiB | 55 s | 951 s |
| 2 | remat | 0 / 0 | 6.4e-15 / 1.9e-13 | 2.0e-14 / 3.8e-14 | 1.4e-5 / 7.6e-6 | 1.9291e-2 | 1.506e-3 | -8.952e-5 | 94 GiB | 25.0 GiB | 134 s | 2,467 s |
| 5 | remat | 0 / 0 | 2.5e-14 / 2.3e-15 | 1.8e-14 / 3.4e-13 | 4.2e-6 / 1.2e-5 | 1.9255e-2 | 3.48e-3 | -2.251e-4 | 97 GiB | 24.0 GiB | 338 s | 2,539 s |
| 10 | remat | 0 / 0 | 3.3e-14 / 8.3e-13 | 1.8e-13 / 2.6e-11 | 3.3e-6 / 4.1e-6 | 1.9234e-2 | 5.5e-3 | -4.451e-4 | 97 GiB | 24.2 GiB | 528 s | 2,590 s |
| 1 | none | 0 / 0 | 7.3e-15 / 1.1e-14 | 1.4e-15 / 0 | 2.9e-6 / exact 0 | 1.9311e-2 | 6.857e-4 | -4.512e-5 | 44 GiB | 12.3 GiB | 35 s | 855 s |
| 2 | none | 0 / 0 | — | — | — | 1.9291e-2 | 1.506e-3 | -8.952e-5 | 91 GiB | 24.6 GiB | 82 s | 2,517 s |
| 5 | none | compile failed | | | | | | | 77 GiB at stop | | | > 7,000 s |
| 10 | none | compile failed | | | | | | | 80 GiB at stop | | | > 7,000 s |

- FD best = the smallest |AD - FD| / max over the three eps (mostly at
  1e-4). At N = 1 the SSH loss does not depend on rn_Ud (AD = FD = 0
  exactly, as in round 330).
- No-remat versus remat gradient of T0: max relative difference 3.2e-15
  (N = 1, T), 4.4e-15 (N = 2, T), 2.3e-13 (N = 2, SSH).
- Peak RSS, process = the sampled peak during the run, including what the
  process already held. After compile the process holds ~73 GiB (compiler
  residue plus the production executable). The run then adds the "run
  growth" column, and XLA's own buffer estimate agrees with it
  (12.2-24.6 GiB). With remat, run memory is flat from N = 2 to N = 10.
- Units are GiB, as `free -g` reports them. 97 GiB is 104 GB decimal; see
  the decision below.
- No-remat N = 5 and N = 10: exit 134 after ~2 h of compiling, with "LLVM
  ERROR: Unable to allocate section memory" at 77-80 GiB resident while the
  machine had ~900 GiB available. PLAUSIBLE cause: the per-process
  memory-map limit (vm.max_map_count = 65,530). That is the same failure
  rounds 315-317 traced to the map count, but I did not count maps in these
  runs. Round 317's no-remat N = 5/10 retries on the older card reached
  152-224 GiB, which is over the cap either way.

## Gradient norm growth with N

- T loss: |dL/dT0| is flat (1.9311e-2 -> 1.9234e-2, -0.4 % over 10 steps).
  Over ten 3-hour steps the T^2 loss is dominated by the initial T.
- SSH loss: |dL/dT0| grows 8.0x by N = 10 (6.86e-4 -> 5.5e-3), matching
  round 317's 8-11x on other cards. The SSH loss's full-state cotangent
  norm grows 23x (0.278 -> 6.52).
- rn_Ud: dL_T/dln(rn_Ud) grows linearly with N (-4.5e-5 per step,
  -4.45e-4 at N = 10). The SSH sensitivity appears from N = 2 (-3.9e-7)
  and is -8.9e-6 at N = 10.

## Predictions (PREREG_orca2x_training_readiness.md)

- P1 no non-finite: CONFIRMED for what was measured (every N, both losses;
  reverse mode on every leaf, forward mode on T_dir0 and rn_Ud only). P1
  said "both modes" for every leaf; forward mode never covered that.
- P2 rev vs fwd along T_dir0: CONFIRMED and better than predicted:
  T loss <= 3.3e-14, SSH loss <= 8.3e-13 at every N. Round 317's 2.6e-5 gap on the pre-L8 card does not
  appear here; why is not located (different card and composition,
  different instrument).
- P3 rn_Ud rev vs fwd <= 1e-12: CONFIRMED for T. For SSH at N = 10,
  REFUTED (2.6e-11 > 1e-12); this is still far below any training
  tolerance. FD at N = 1 <= 1e-5: CONFIRMED (2.9e-6).
- P4 T norm flat within 0.1 %: REFUTED, 0.4 % over 10 steps (still flat for
  practical purposes). SSH grows 5-12x: CONFIRMED (8.0x).
- P5 remat N = 10 < 100 GiB: CONFIRMED (97 GiB). No remat: N = 2 fits
  (CONFIRMED, 91 GiB), N >= 5 does not (CONFIRMED, though by compile
  failure, not by the cap).
- P6 primal = production at N = 1: CONFIRMED. Last-bit differences at
  N >= 2: CONFIRMED (<= 1.4e-14 relative).

## Not done

- sqrt(N) remat schedule: not run. `integrate_scan` has no hook for
  segment checkpoints without a model change, and per-step remat already
  makes run memory flat in N (24 GiB from N = 2 to 10). The binding cost is
  the ~73 GiB held after compile, which sqrt(N) does not reduce.
- N > 10: the record has 10 forced steps. Longer horizons need more
  recorded forcing (an acquisition question; not asked).
- Item 3: nothing to localise (no non-finite value at any N).

## DECISION_NEEDED: training horizon and remat strategy

Memory barely depends on N with remat (~73 GiB compile residue + ~24 GiB
run), so the choice is time per gradient against dynamics in the window.

1. **Pick:** N = 10 with per-step remat: 97 GiB (104 GB decimal),
   528 s per fwd+bwd per loss, compile ~43 min. Longest window the record
   allows (30 h).
2. N = 5 with per-step remat: 97 GiB (104 GB decimal), 338 s per
   fwd+bwd per loss, compile ~42 min. Same memory as N = 10, ~0.64x the
   time, half the window (15 h).
3. N = 2 without remat: 91 GiB (98 GB decimal), 82 s per fwd+bwd. The only
   multi-step option under 100 GB decimal; a 6-hour window gives the
   gradient little dynamics. (N = 2 with remat: 94 GiB, 134 s — no gain.)
4. N = 1: 44 GiB, 35 s. The existing round-330 gradient, no multi-step
   dynamics.

Deciding factor: whether the 100 GB cap is GiB (`free -g`) or decimal GB;
if GiB, 1 vs 2 is time per step against window length.

## Gates (x64, CPU; logs `orca2x_rounds/round2/gates/`)

- Default gate (land.sh list, split for the 10-minute limit):
  citation-gate test + private-import ratchet + TKE terms + ORCA2 parallel
  receipt 97 passed; recipe + freshwater + prognostic barotropic 49 passed;
  stage face mask 2 passed (12 min, detached and read back). Total 148
  passed, 0 failed.
- Probe tests: `tests/ocean/fidelity/test_orca2x_r2_multistep_grad.py`
  (static/dynamic input split, CLI refusal of N outside 1..10, summary
  rows and the no-remat-vs-remat comparison).
- Receipt citation gate on this receipt: PASS, 0 citations, 0 failures,
  0 unmapped (`gates/citation_gate_receipt.log`).
- Plants (`gates/plants.log`): input split tracing the step numbers -> 1
  test fails; the summary comparison unnormalised -> 1 fails; restored, 6 pass.

- Revision (receipt text only, no code change): citation gate on this
  receipt PASS, 0 citations, 0 failures (`gates/citation_gate_receipt_rev.log`);
  citation-gate test + private-import ratchet + TKE terms + probe tests +
  ORCA2 parallel receipt 103 passed; recipe + freshwater + prognostic
  barotropic 49 passed. Stage face mask not re-run (12 min, over the
  10-minute limit; no code changed since its 2 passed).

## Choices

- ASKED (round prompt): card, losses, rn_Ud via the rung-10 entry,
  N = 1, 2, 5, 10, integrate_scan path, remat measurement, 100 GB bound.
- UNASKED: the 100 GB bound read as 100 GiB (`free -g` units, the memory
  rule's units); peaks are reported in both units.
- UNASKED: FD on the raw (unconstrained) rn_Ud leaf at eps 1e-2/1e-4/1e-6.
- UNASKED: remat wraps step 1 too (it runs outside the scan).
- UNASKED: six capped processes ran concurrently (wall times inflated).
- UNASKED: the round's one-line result is not appended to the ORCA2
  campaign ledger (round 1 of this lane did not either); the operator can
  add it.

codex review: operator
