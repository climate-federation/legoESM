# ORCA2 round 206 — OMT-0 atomic fold-unit census

Date: 2026-10-09. Base: `0f069e01c`. Measurement commits:
`a46f165f0` (frozen census), `94acef28f` (production-path verifier), and
`2f7ed65fc` (held-tip retraction). Status: **HELD**. All OMT-0 numbers below
are reported separately as **independent OMT-0** and **given NEMO's entry**;
their censuses are identical. The shipped rung-10 card, sea ice, its six
selectors, and its `unmeasured_features` tuple are unchanged.

## CORRECTION (round 207, 2026-10-09)

The statements below that both DINO processes were externally killed before
producing a score are **RETRACTED**. Codex's foreground tool call timed out;
the detached integrations continued and both logs end in the complete gate
result `2.056821682e-03 K <= 2.244317642e-03 K -- PASS`. Their SHA-256 values
are `4026dcc50417660a2251902e76f045ff74bdaa34f5fd62566bc2bd6e52c3b63e`
and `ba51562599d41738c5b4727a833f9f4948ee73c42506400456430e61f7293c7d`.
Round 207 therefore resumes from the already-qualified census and promotes
the exact candidate. The original text is retained below as the audit trail,
but its resource-termination and DINO-unmeasured conclusions are false.

## Result

The complete four-statement candidate is Decision-96 eligible on both OMT-0
claim labels. It touches 195/200 ladder rows and every RMS-moved row moves
toward NEMO: 195 toward, zero away, zero score-equal, five unchanged. The
first over-bar row moves toward, no bit-identical row is lost, and both
candidate trajectories remain finite through kt=10.

The kt=1 stage-1 SSH RMS/maximum improve from
`0.006243153742634906 / 0.13136125371061705 m` to
`0.00023211739258069189 / 0.006278809746666877 m`. The kt=10 stage-3 SSH
RMS/maximum improve from `0.0288100436468264 / 0.5086605459790218 m` to
`0.0036726314904968077 / 0.07685333444272535 m`.

The candidate is the indivisible unit named in round 205:

- raw `hu_0/hv_0` reference face depths at
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:512-519`;
- the unmasked `zhV = e1v * va_e * zhvp2_e` product at
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:532,535`;
- the separately materialised binary64 V product; and
- the seven-array post-update association at
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741`, including
  V sign `-1` and depth/reciprocal sign `+1` at `:721-730`.

The round-205 recorded-slow-V control remains score-null once the association
is enabled: all 65 SSH/U/V substep state summaries are identical with and
without those 35 recorded fold values. Thus the census does not infer or
fabricate an unavailable forcing stream. Missing-arm, exact-row-loss,
false-majority, and score-equal-as-vote plants all refuse.

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round206/omt0_atomic_fold_census.json`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round206/measurement_rerun.log`

## Production proof and retraction

The unit was temporarily promoted as one unbranched production identity. A
tri-state validation control then proved the ordinary production path equal
to the frozen private candidate on every row: zero differences over 200
independent rows and zero differences over 200 given-entry rows. The first
monolithic verifier was externally killed after finishing the independent
label, so the same committed calculation was split by claim label; both fresh
processes passed.

That promotion is **not landed**. The required DINO month could not complete
on this CPU sandbox: the ordinary attempt and an isolated
`MALLOC_ARENA_MAX=2`, non-preallocating retry were both externally killed
without a Python traceback, before any score, and each left only snapshot 0.
A missing DINO score is not a PASS. Commit `2f7ed65fc` removes the production
promotion and restores both model files byte-for-byte to base `0f069e01c`.
The held tip therefore contains no `packages/` diff.

Temporary-production artifacts retained as evidence, not as a landing:

- `omt0_production_independent_after.json` and
  `omt0_production_given_after.json` (400/400 rows equal to the candidate);
- `dino_month.log` and `dino_month_retry.log` (both resource terminations,
  neither contains a scientific score).

## Shared gates measured before retraction

- GYRE ten-step: 70-row comparison PASS, maximum worsening zero ULP, first
  over-bar unchanged at kt=3.
- GYRE year: all twelve fresh snapshots are byte-identical to the certified
  member. Day-30/day-240/day-360 SHA-256 values are
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
  `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
  and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.
  Therefore its certified day-30/day-240/day-360 T RMS values remain
  `2.3432419318363155e-06`, `6.5816987106668941e-05`, and
  `5.4077212586815052e-05 K`.
- LOCK_EXCHANGE: 50-row comparison PASS, zero worsening ULP, first over-bar
  unchanged at kt=8 U.
- OVERFLOW: 50-row comparison PASS, maximum worsening 1.5 ULP (inside the
  two-ULP ratchet), first over-bar unchanged at kt=2 T/U.
- DINO: **UNMEASURED_WITH_SPEC**. Required number: day-30 wet-3D T RMS from
  the committed CPU month protocol, required to be at most
  `2.244317642e-03 K`. Both executions were externally killed before that
  number existed.

## Frozen predictions

| ID | disposition |
|---|---|
| R206-P1 | **CONFIRMED**: exactly the four registered controls form the candidate. |
| R206-P2 | **CONFIRMED**: both labels emit 40 checkpoints and 200 finite rows. |
| R206-P3 | **CONFIRMED** on OMT-0: 195/195 RMS-moved rows toward, no exact loss. Landing predicate remains unmet because DINO is unmeasured. |
| R206-P4 | **CONFIRMED**: both frozen SSH headline comparisons improve. |
| R206-P5 | **CONFIRMED**: both labels have identical row-direction maps and exact-loss sets. |
| R206-P6 | Not reached: the candidate is eligible, so no next-partner walk is claimed. |
| R206-P7 | **CONFIRMED**: all four census plants fire. |

## Validation and review

Focused round-206 tests pass 6/6 on the final held tip. The one permitted
`tests/ocean/fidelity -n 12` invocation collected 2,982 tests and reached 98%
before its process disappeared without a summary. Its log records 2,938
passes, seven skips, zero errors, and exactly four failures, all registered
pre-existing reds: the GYRE round-129 spread-floor stamp, allow-dirty scope,
worktree-stamp ratchet, and SI3 scalar-math provenance gate. No round-206 test
fails.

The cumulative citation gate passes with 274 citations, no failures and no
unmapped citations; its rigid line-shift plant refuses. Independent review was
attempted with `codex exec --sandbox read-only` and exited 1 before reading the
diff: `failed to initialize in-process app-server client: Read-only file
system (os error 30)`. Independent review is unavailable in-sandbox; this is
not represented as a PASS.

## OPEN

1. Re-run the committed DINO CPU month gate in an environment with sufficient
   compiler memory and obtain the required day-30 wet-3D T RMS. Do not reuse a
   snapshot-0-only directory and do not infer a score from GYRE or the tanks.
2. If DINO passes its fixed bar, reapply the already-qualified atomic unit and
   rerun the ordinary OMT-0 production equality plus the complete shared gate
   set on that exact tip. Only then may the four statements land together.
3. OMT-1 (+ linear implicit bottom drag) remains next only after this OMT-0
   unit is either landed or scientifically refuted.

No NEMO acquisition is needed. No configuration or sea-ice decision is
pending.
