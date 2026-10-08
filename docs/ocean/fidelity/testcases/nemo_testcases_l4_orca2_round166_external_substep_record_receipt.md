# ORCA2 round 166 — kt=8 external-substep record request

Date: 2026-10-07. Base `a3b92b9d2`; preregistration `6f343ed1e`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round166/`.
Verdict: **STOPPED_FOR_RECORD**. The complete private V-transport arm first
becomes non-finite in external substep 2 at the U-face exit inverse depth. The
existing records do not contain a rank-complete kt=8 external-substep frame,
so this round localizes but does not assign the producing operand or land
physics.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple are
unchanged.

## Source order and passive boundary

The compiled rung-0 program extrapolates barotropic velocity and sea surface,
then constructs the mid-step U/V face depth at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:502-559`. It forms
the depth-weighted transports at `dynspg_ts.f90:564-570` and advances sea
surface from their divergence at `dynspg_ts.f90:580-595`. After pressure,
Coriolis and drag, the flux-form velocity update is
`dynspg_ts.f90:704-757`. NEMO then reconstructs the exit face depths and
their masked reciprocals at `dynspg_ts.f90:761-767` and exchanges the seven
arrays together at `dynspg_ts.f90:770-779`.

The round reuses the admitted 65-substep trace registry and the same complete
four-statement private arm as rounds 164-165: raw reference face depth,
seven-array external-mode association, no extra compact V-transport mask and
materialised completed `zhV`. The observer reduces the already materialised
trace on the host and returns the model state unchanged.

The unobserved and observed executions both complete kt=7, expose kt=8 stages
1 and 2, do not return stage 3, and stop on exactly `raw-mesh e3w_int must
contain only finite values > 0`. For every kt=1..7 checkpoint,
`T`, `S`, `u`, `v` and `ssh` are byte-identical between the two executions;
all 35 SHA-256 digests match. R166-P1 is therefore **CONFIRMED**.

At kt=8 the first source-ordered non-finite trace is:

| external substep | boundary | trace | value | invalid U faces | first index | flat index |
|---:|---|---|---:|---:|---|---:|
| 2 | exit inverse U depth | `r1_face_depth_u_exit` | `+Infinity` | 42 | `[j=7, i=114]` | 1,381 |

Entry, midpoint, midpoint face depth, transports, divergence, after-SSH,
pressure/trends, exit velocity and exit face depth are source-ordered before
that row. Thus R166-P2 is **REFUTED**: after-SSH is not the first non-finite
boundary. The first invalid quantity is the reciprocal corresponding to
NEMO's masked division at `dynspg_ts.f90:761-767`; localization does not say
whether the face-depth operand, its boundary association, or a preceding
substep value is the first non-bit statement relative to NEMO.

## Record census and acquisition

The inventory finds many legacy root-only `oracle_bt_frames_kt00000008.bin`
files, including rung-0 records, but no self-describing, rank-complete kt=8
external-substep record. Those terminal frames do not carry the named
substep-2 operands. R166-P3 is **CONFIRMED**.

The emitted launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round166_spg8_acquisition/run.sh`.
It content-pins the admitted round-96 deck, binary, compiled call layout,
writer and checker; creates a new `ORCA2_OMIP_L4_R166SPG8` target; writes one
self-describing `NEMO_L4_R166SPG` file per rank at kt=8; parses field names,
ranks, shapes and payload lengths from the headers; and requires both ranks'
twenty kt=1..10 terminal restarts to be byte-identical to the admitted
round-96 additions-only baseline. Its preflight reports
`ORCA2_ROUND166_SPG8_PREFLIGHT_READY`; independent layout and content plants
both refuse with exit 69. Per the sandbox PMIx rule, NEMO was not launched in
this round.

## Prediction ledger, controls and validation

| ID | Result | Mechanical disposition |
|---|---|---|
| R166-P1 | **CONFIRMED** | Same terminal class and 35/35 completed field checkpoints byte-identical. |
| R166-P2 | **REFUTED** | Substep-2 exit inverse U depth, not after-SSH, is first non-finite. |
| R166-P3 | **CONFIRMED** | No rank-complete kt=8 external-substep frame; acquisition emitted. |
| R166-P4 | **CONFIRMED** | No `packages/` or card/configuration change; the atomic unit stays private and HELD. |

The four measurement plants separately perturb terminal passivity, checkpoint
passivity, trace source order and record support; each exits nonzero with
`PLANT-FIRED`. Final focused coverage passes 34/34 across the round-166 gate
and generalized round-95 record checker. Evidence artifacts and control logs
are frozen in `SHA256SUMS`.

Two instrument findings are retained rather than silently discarded. A
combined observed/unobserved process exhausted memory after retaining both JIT
executables, so the unchanged control was isolated in its own process and its
field digests became the fail-closed comparison input. A kt=8 effect-only
callback did not run before the downstream Equinox invariant stopped the
trace; the final instrument instead reads the existing early-return private
trace and proves passivity on every completed kt=1..7 state plus the identical
kt=8 terminal class. A NumPy integer in the first JSON index also required an
explicit scalar conversion; the regression test covers it. None changes a
scientific value or the disposition.

ASKED choices: continue round 165's compiled-source barotropic walk. UNASKED
choices: empty. No configuration, forcing, carried-state policy, stabiliser,
sea-ice selector or production model statement changed.

## OPEN

1. The operator runs the round-166 acquisition. Admit it only if the
   self-describing checker, rank census, seven plants and additions-only
   restart comparison all pass.
2. At kt=8 external substep 2, compare NEMO and legoESM in source order through
   exit face depth, the masked reciprocal and the seven-array exchange. Name
   the first non-bit operand and its magnitude before proposing a statement.
3. Keep the complete V-transport/halo unit private and **HELD**. A landing
   still requires the full rung-0/rung-7, independent-month, GYRE, DINO and
   tank gates under Decision 96.
