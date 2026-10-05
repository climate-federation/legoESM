# ORCA2 round 151 — raw reference-depth causal arm

Date: 2026-10-05. Base `01feb01bd9`; measurement tip `267cbba64`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round151.md`
at `2c4c63f46`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The private/default-off causal arm supplies only NEMO's raw reference
`hu_0/hv_0` to the existing SSH-average face-depth builder. With the hook
unset, the complete control is bit-identical. With the hook set, the substep-2
V midpoint depth closes from **30 unequal cells, maximum 899 m**, to
**0 unequal cells**. This confirms the round-150 local owner at NEMO's
reference-depth plus SSH-average statement, compiled at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`.

The causal chain predicted in round 150 is **REFUTED**. The arm leaves each
downstream row numerically unchanged:

| substep-2 row | control unequal / maximum | raw-depth arm unequal / maximum |
|---|---:|---:|
| `transport_v` | 68 / `155776.5627856178` | 68 / `155776.5627856178` |
| `continuity_dv` | 68 / `155776.5627856178` | 68 / `155776.5627856178` |
| `after_ssh` | 68 / `0.003203816535399729` m | 68 / `0.003203816535399729` m |

All three maxima remain at `(j,i)=(147,135)`. The 30-cell midpoint-depth debt
and the 68-cell transport chain are independent, not ancestor and descendant.
The first remaining non-bit source boundary is therefore NEMO's V metric
transport statement
`zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)`, followed by the
continuity and SSH update at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591`.
At this boundary the arm's recorded `va_e` and `zhvp2_e` operands are already
bit-exact. The next split is the recorded `e1v`, the written two-product order,
and legoESM's compact V-mask/loop-domain association; this receipt does not
assign ownership among them.

The final evidence is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round151/reference_depth_arm_final.json`,
SHA-256 `dec3ee8571361eddfecf4045cbd8f6b14cca7163adc23c38a9c599fc3ec2f268`.
It is stamped to `267cbba641f0187f878019f22a2997b08f8d454e`, CPU,
production JIT, fp64/libm, and x64.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R151-P1 prerequisites and registered control reproduce | **CONFIRMED**: midpoint depth 30/899 m and transport/continuity/SSH 68/68/68 reproduce with their registered maxima. |
| R151-P2 unset hook is passive and changes one variable when armed | **CONFIRMED**: unset is bit-identical; the arm replaces only the frozen reference face-depth pair. |
| R151-P3 raw reference depth closes midpoint V depth | **CONFIRMED**: `mid_depth_v` closes 30→0 unequal. |
| R151-P4 raw reference depth closes transport, continuity, and SSH | **REFUTED**: all three rows remain bit-for-bit equal to their controls at 68 unequal cells. |
| R151-P5 shared GYRE trajectory is unchanged | **CONFIRMED**: certified ten-step ladder and 30-day snapshots are byte-identical base versus tip. |

The first attempted arm measurement refused before reporting evidence because
the native-record U adapter padded compact periodic column zero with zero; the
live compact mask consequently exposed 64 zero denominators. The adapter was
corrected to copy the native periodic closure into column zero, matching the
existing compact association, and a non-vacuity test was added. The refused
attempt is retained in the round evidence and supplies no numerical claim.

## Controls, shared-path gate, review, and verification

Both registered controls fire. The malformed reference-depth shape exits with
`STATUS PLANT-FIRED reference-depth-shape`; the exact-cell bit plant exits with
`STATUS PLANT-FIRED reference-depth-arm-bit`.

Because this round adds a private hook in `packages/`, the full required GYRE
comparison was run at base and tip. The certified ten-step comparison passes:
70 rows, zero moved rows, zero status changes, first-over-bar step unchanged at
kt=3, and all 210 residual arrays `np.array_equal`. The base and tip 30-day
members each write 30 snapshots; all files are byte-identical and both trees
have SHA-256
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.
Thus the hook's default is genuinely passive on GYRE.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No second independent review service is
available in this sandbox, so the arm remains explicitly UNREVIEWED despite
its mechanical controls.

The one required `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**, not
PASS: it reached 99% with 2,571 observed passes, 7 skips, and the four
registered pre-existing reds (SI3 scalar-math provenance, round-35 escape
scope, worktree-stamp scope, and GYRE round-129 spread-record stamp). All
pytest worker processes then disappeared without a terminal summary while the
execution pipe remained open; the stale session was closed. No round-151 test
failed. Focused round and citation checks are recorded after this receipt is
committed.

No ORCA2 ladder was run: frozen R151-P4 failed, and the registered
approximately 31 PSU salinity exposure remains a hard veto. No production
physics or default changed.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Split substep-2 `zhV` over the same 68 northern-fold-row cells in compiled
   order: `e1v`, `va_e`, `zhvp2_e`, the written product association, and the
   compact V-mask/loop domain. Measure whether the extra compact mask is causal;
   do not infer that from the source spelling alone.
2. Only if transport, continuity, and SSH close, run both ORCA2 ladders and the
   registered salinity/tracer compensation gate before any production landing.
3. After rung 0 is mechanically safe, score its independent month and resume
   the hierarchy merge/climb program.
