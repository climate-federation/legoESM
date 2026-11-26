# ORCA2 round 262 — stage-1 V-ratio fold integration

Date: 2026-10-12. Base: `5e7ea0504`. Preregistration commit:
`0d0f6e202`. Measurement commits: `3e40ef906`, `8f143a552`, `fce2a12d3`,
`c48a836ee`, and `ea737ac21`. Private-arm removal commit: `6ef43a2f7`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, on CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled statement

NEMO computes the stage-after V-point free-surface ratio in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217`. The compiled
stage program then calls `dom_qco_r3c_RK3` and associates `r3va` with the
V-point north-fold rule in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170` before
storing the stage ratio. The private candidate placed that association before
both shared live-ratio consumers, so the stage-2 `e3v` and `r1_hv`/`zvb`
members moved together. The production operand route was D114's resolved raw
NEMO mesh route throughout.

## Frozen predictions and verdicts

R262-P1 is **REFUTED at the bit level and registered under Decision 96**.
Under both labels the valued rows close: live `e3v` is 0/1,977 unequal,
`r1_hv` is 0/100, and `zvb` is 0/100. Stage-2 `zFv` has maximum absolute
error 0.0 but 181/8,100 signed-zero-only differences. Candidate and oracle
source replays each reproduce their own completed statement, and the passive
trace is array-identical for every carried field. The two-label classifier is
`PASS_R262_V_RATIO_FOLD_D96_RATCHET`; all nine source controls refuse.

R262-P2 is **REFUTED**. The complete atomic unit's first valued trajectory
boundary remains kt=1 stage 1 V in the fold band under both labels: maximum
`0.027227136128767482` m/s and RMS `0.002202566269216334` m/s. At kt=1
stage 2, fold-band T/S are not at the campaign floor: T maximum is
`0.368232078772583` K and S maximum is `6.817829255334281` PSU. By stage 3
those maxima are `0.7220339611990294` K and `13.350001820956706` PSU. The
candidate therefore closes the round-261 source replay but does not close the
earlier stage-1 V writer.

R262-P3 is **REFUTED**. The full independent kt=1..10 run completes kt=1 and
then refuses before kt=2 at the existing live-thickness guard:
`raw-mesh e3w_int must contain only finite values > 0`. The bounded prefix
gate reproduces the same first boundary under both labels and all four prefix
plants refuse. Because P2 and P3 fail, R262-P4 and R262-P5 are **NOT
REACHED**: no Decision-96 row vote, OMT/rung ladders, GYRE year, DINO month,
or tank landing gate can authorize this unit. R262-P6 is **CONFIRMED** for
the controls that were reached.

The initial diagnostic placement, which associated only a private `_qv_13`
copy, left the round-261 counts unchanged (`668`, `35`, `35`, `849`) and is
retained as a refuted instrument attempt. The corrected instrument couples
the actual shared consumers. No number from the first attempt supports the
verdict.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round262/`.
Principal SHA-256 values are:

- independent source replay:
  `34797661c625e40d9bdf8cbd2e54529559d21cf9d66b4cbbca671d868fa274b1`;
- given-NEMO-entry source replay:
  `f9e73f63396a3ea717270e327b11e97d9d37609d38853ea348675c4a01392793`;
- source classifier:
  `a9a6f70f5b5275323a3a436fc53a7b58356595a8411592b81963a6e6b65d5017`;
- independent / given-entry atomic prefixes:
  `0a92840dbaf9004809a35edb4ecce33071707dfe9c713f6c7dff389530e4577c` /
  `b0c534c69f15aa4c41a7530dba7d7414d83b366627e31d9401b555ad4d9d84f6`;
- prefix classifier:
  `fe7564519a465987e6c220b39c9bdaa3b8ddc8971280d22933d02ae100f1fa0a`;
- full independent refusal log:
  `7a0b0c38de14cb4036916667c916d2491e206e878ed94edd85bcd0c112e8525e`.

The package candidate and its temporary cumulative-receipt citation
re-anchor were removed. The final shared model files and cumulative citation
files are byte-identical to `5e7ea0504`; GYRE, DINO, tank, rung-0, rung-10,
and OMT trajectories therefore cannot move on the final tree.

The requested separate `codex exec --sandbox read-only` review failed before
review because its in-process app-server could not initialize on the
read-only filesystem. **Independent review unavailable in-sandbox.** Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused round-261/262 and citation-gate battery passes 44/44; its log
SHA-256 is
`6e3e53fabd6094785ca3791aaf2106ee2c69bf0d66688ef9eb37122d25de6a03`.
The once-only `tests/ocean/fidelity -n 12` battery reached 99% and then
stopped emitting output for 12 minutes in the known late-worker hang class,
while running
`test_nemo_testcase_phase3_stage_sweep_gate.py::test_prediction_plant_is_fail_closed`;
it was terminated and not re-run. It emitted the same four failures as round
261, all of which reproduce serially in isolation:

- the GYRE round-129 record gate refuses because its certified-year harness
  pin moved;
- the round-35 stamp-scope ratchet lists 13 pre-existing VORTEX drivers;
- the known worktree-stamp ratchet lists 13 pre-existing report emitters;
- the known SI3 scalar-math provenance gate refuses its non-verbatim
  `MY_SRC`.

None of the offender lists names the round-262 driver. The interrupted suite
log SHA-256 is
`cd76867ae45f3408699ff16c2c9864cf0072eb3ba29939c0bc80df22d08ee415`;
the four-test serial disposition log SHA-256 is
`0bb79557a09a200e738157b9e01c8e15f4a567b5f727ffed380cfb1de9b757f0`.

The default cumulative citation gate and this receipt's citation gate both
pass with zero unmapped citations. Their JSON SHA-256 values are
`ce7216036b37288c23c25ab3885154f4b6b53a00455bea665fe22e5547c5ee9f`
and
`5b50b4e360794b43dc1b6632b4a7999da15d9e40da646c9ba5f21f171d46ad7d`.
The planted two-line shift of the compiled V-ratio citation turns the receipt
gate red; its JSON SHA-256 is
`de58fb0d96eb20812f922a97bd8de1df7a96f700d4cbd533cfc0d91e7feefb0a`.

## OPEN

The r3va fold source is no longer the first valued owner. Starting from the
exact stage-1 ratio/transport replay, walk the kt=1 stage-1 V update at the
fold in compiled call-site order. The first candidate is the held post-stage
V association/placement that must populate the pivot row before the next
consumer. Keep the full unit private; Decision 116 remains pending only for
an association-alone landing, and is not re-emitted here.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
