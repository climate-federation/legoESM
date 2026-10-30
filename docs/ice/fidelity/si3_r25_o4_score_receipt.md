# SI3 Ladder Round 25 (`si3_r25`) — O4 production-score receipt

Date: 2026-10-29. Verdict: **PASS_O4_EXACT**. The landed five-category O4
Prather path is 0 ULP on every scored NEMO-written wet scalar at advection,
zapsmall, LBC, and end-of-step exits for both sweep orders in true eager and
production JIT fp64/x64. Wet seam and north-fold regions are exact separately.

## Required lane rebase

The round began at `896c2acf024274db71e4b39b4cd8eaf8bd0620fd` and was rebased
first onto the requested current SI3 lane tip
`0d339a87e5c1d718a0b9aa4bb49f8fb8788562be`. Git replayed all seven round
commits cleanly and produced `0deec1acfcb0406fa1b2f690a4fbcfadc23d5e81`.
There were **no conflict hunks**, so no conflict-side selection or manual hunk
resolution occurred. The score preregistration was then committed as
`f761e959dfb60640a6b57ff2a804dc8b122600dd`, before any score command, and the
extended committed scorer as `557425cdd1a2ea339ed29b0658c5114920565af6`.

## Acquisition and admission

The admitted evidence root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather`.
Its `o4_admission.json` records producer
`896c2acf024274db71e4b39b4cd8eaf8bd0620fd`, status
`PASS_O4_ADMISSION`, 18 record files, and byte-identical ocean and ice restart
pairs. Its `manifest/run_manifest.json` pins the reviewed producer, source,
deck, input, and both retained binary digests. The manifest declares two steps,
one rank, Prather on, UMx off, prescribed `rn_uice=rn_vice=0.5`, and all other
dynamics and thermodynamics off.

`manifest/artifacts.sha256` contains 122 rows. A fresh
`sha256sum -c manifest/artifacts.sha256` returned `OK` for all 122 artifacts,
including every one of the 18 records, both binaries, both restart pairs,
inputs, deck, generated preprocessed sources, same-program proof, and FP
census. `manifest/same_program.txt` is exactly
`PASS_REAL_PPSRC_WHOLE_UNITS_SAME_PROGRAM`. `manifest/fp_census.json` reports
`pass=true` over every emitted function and unmerged compiler clone, including
zero-FP functions; the four recording/reference FP totals agree at Prather
3851/3851, advection caller 279/279, dynamics 310/310, and step 455/455.

Every digest below was freshly computed with `sha256sum` from the named file:

| Evidence file | Computed SHA-256 |
|---|---|
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/o4_admission.json` | `79a550c16a7b0ad24a39b8ad84f340960e744c438e15afb3f1e0599fac1bcee2` |
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/manifest/run_manifest.json` | `32f0e2b1cd684bd88c2e3429348fb35d79c96acf9ccee508517052bc84f3ed73` |
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/manifest/artifacts.sha256` | `fbfbe97615a18037328e36b30fabe551fbf37b6d8966591ac6a7bd42e5241aa0` |
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/manifest/same_program.txt` | `7a945d6445ccdd8f745a6423e5bc10b38acc379daed115fa44f144dc77e45ac6` |
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/manifest/fp_census.json` | `2990f7abc14a78ab6e3bd928757b4f4c9119715a844661e2a2b758d7132db9ee` |

## Preregistered production score

The frozen contract is
`docs/ice/fidelity/si3_r25_o4_score_preregister.md:1-84`. Its exact bar is 0
ULP on every NEMO-written wet cell for every recorded state/residual exit at
advection, zapsmall, LBC, and step, with seam and fold reported separately.
Signed-zero differences count as unequal. PASS requires exact eager and JIT
rows; anything else is O4_DEBT and no model fix is allowed in this round.

The scored M4 entry point is the landed production `si3_o4_step`, not a test
copy (`packages/ice/legoesm/ice/si3_orca2_state.py:3807-3910`). Its call chain
enters `si3_dyn_orca2` and the landed `si3_dyn_adv_pra_orca2`. The committed
scorer runs the whole eager arm inside `jax.disable_jit()`, wraps the same
production calls with `jax.jit` in the JIT arm, and emits strict signed-zero
cell counts and first differences
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_gate.py:248-506`).
The direct fixture proves eager/JIT identity and fires every stage, sweep,
fold-sign, and subcycle-LBC plant
(`tests/ocean/fidelity/test_nemo_si3_o4_prather_acquisition.py:43-75`).

Kinds 1--3 are the fail-closed grid/state/moment input contract. Kind 4 is the
exact CFL/subcycle/sweep control. Kinds 5--9 are the scored advection state,
advection moments, zapsmall state, post-LBC state, and end-of-step state. The
table reports both modes as `eager / JIT`; their summaries are identical.
Regional rows intentionally overlap the full row and are reported separately,
not added as unique physical cells.

| Step/sweep | Kind | Stage | Region | Wet scalar cells | Unequal eager/JIT | Max ULP eager/JIT | First differing cell eager/JIT |
|---|---:|---|---|---:|---:|---:|---|
| 1 x-y | 4 | controls | full | 4 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 5 | advection | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 5 | advection | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 5 | advection | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 6 | advection moments | full | 13,900,425 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 6 | advection moments | seam | 463,650 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 6 | advection moments | fold | 330,825 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 7 | zapsmall | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 7 | zapsmall | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 7 | zapsmall | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 8 | LBC | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 8 | LBC | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 8 | LBC | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 9 | step | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 9 | step | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 1 x-y | 9 | step | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 4 | controls | full | 4 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 5 | advection | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 5 | advection | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 5 | advection | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 6 | advection moments | full | 13,900,425 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 6 | advection moments | seam | 463,650 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 6 | advection moments | fold | 330,825 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 7 | zapsmall | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 7 | zapsmall | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 7 | zapsmall | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 8 | LBC | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 8 | LBC | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 8 | LBC | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 9 | step | full | 5,555,594 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 9 | step | seam | 182,369 | 0 / 0 | 0 / 0 | NONE / NONE |
| 2 y-x | 9 | step | fold | 130,845 | 0 / 0 | 0 / 0 | NONE / NONE |

Each step has 38,170,136 reported scalar positions when the separately
required full and regional rows are counted together. Both modes report zero
unequal and maximum 0 ULP over every row. After replacing only
`"execution_mode": "eager"` versus `"jit"` with a common label, `diff -u`
reported no difference between the two JSON files.

| Score artifact | Computed SHA-256 |
|---|---|
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/o4_score_eager.json` | `7f38d2a0cdb6e0605398cfd2422a7a6eec2f2cc72d1c420fe3d366148b37bc6a` |
| `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather/o4_score_jit.json` | `b015d0201978f049ba9178bfa53a1e405150ac5786ac2bb9d923682d7806680c` |

## Attribution, choices, and status

There is no first differing cell and therefore no first differing statement to
attribute as CONFIRMED or PLAUSIBLE. No model statement was changed after the
score and no fix was attempted.

ASKED choices: rebase onto the named lane tip; admit the named acquisition;
freeze the exact O4 bar; score the landed production entry point in eager and
JIT x64; report every kind/stage and seam/fold region; classify exact versus
debt. UNASKED choices: none. No configuration, state, threshold, tolerance,
record, production model code, NEMO source, or default changed.

Final classification: `PASS_O4_EXACT`. Acquisition is complete, no further
NEMO run is needed, and O4 carries no local score debt on this admitted record.
