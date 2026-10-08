# ORCA2 round 168 — kt=8 exit-depth operand hold

Date: 2026-10-07. Base `4adbb02ce`; preregistration `1c37bf32f`;
measurement instrument `6ff7e6989`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round168/`.
Verdict: **HELD**. The carried raw `hu_0` is exonerated by an independent sum
replay. The candidate U-face SSH owns all 41 zero exit depths, but the
source-ordered walk shows that kt=8 external substep 2 already enters with a
non-bit U velocity from substep 1. No physics or configuration changed.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple are
unchanged.

## Compiled source order

The admitted executable forms the transport divergence and advances
`ssha_e` at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:585-595`, constructs
the area-weighted U-face sea surface `zsshu_a` at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:633-634`, and forms the
exit depth and masked reciprocal at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:763-766`. After the
seven-array exchange, NEMO carries substep 1's completed `ua_e` into
substep 2's `un_e` at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:836-844`.

The round-166 record re-admits with two rank slabs, exactly-once coverage, 65
external substeps, 2,106 self-described groups per rank and 20 terminal
restarts byte-identical to the additions-only baseline. The measurement uses
the unchanged complete private arm from rounds 164-167 and again completes
independent kt=1..7 before exposing kt=8.

## Operand split

At the fixed 41 unique native U faces from round 167, the carried raw depth is
finite and spans 36 to 5,392 m. This is an observation, not a candidate-versus-
itself equality claim. Its independent causal check is the sum replay:

| registered replay or operand | unequal / 41 | candidate range | oracle range | maximum absolute difference |
|---|---:|---:|---:|---:|
| raw depth + recorded `j002_sshu_a` vs recorded `j002_hu_e` | 0 | 36.60196376160646..5392.468449038692 m | same | 0 m |
| candidate face SSH vs recorded `j002_sshu_a` | 41 | -5392..-36 m | -1.4082999125879239..0.927040999777258 m | 5392.468449038692 m |
| candidate raw + candidate face SSH vs candidate exit depth | 0 | 0..0 m | same | 0 m |
| candidate exit depth vs recorded `j002_hu_e` | 41 | 0..0 m | 36.60196376160646..5392.468449038692 m | 5392.468449038692 m |

Thus R168-P1 and R168-P2 are **CONFIRMED**: recorded face SSH alone closes
all 41 depths bit-for-bit, while the unchanged candidate face SSH cancels the
entire raw water column at all 41. The round-167 reciprocal overflow is a
downstream consequence.

R168-P3 is **REFUTED and retained**. Replaying recorded `j002_ssha_e` through
the literal face-average helper leaves 1/41 registered faces unequal, maximum
0.14741707684196548 m. Therefore this round does not claim that the
face-average arithmetic is bit-exact from that recorded boundary. All 41
candidate two-cell after-SSH stencils are non-bit, so the input is certainly
already wrong, but the one replay residual must be separated from the input
timing/boundary association before the face-average statement itself can be
cleared.

## First upstream boundary

The source-ordered prefix from the previous substep's carried state through
the current after-SSH contains no candidate non-finite before the exit
reciprocal. Its first non-bit row is already the substep-2 entry U velocity,
which is substep 1's carried `ua_e`:

| row | active unequal | candidate non-finite | maximum absolute difference |
|---|---:|---:|---:|
| entry U | 15,789 | 0 | 2.528343863624811e53 m s^-1 |
| entry V | 15,875 | 0 | 2.7085791531051274e53 m s^-1 |
| entry SSH | 16,433 | 0 | 8.697372772619884 m |

R168-P4 is **CONFIRMED**, but the named result is a carried boundary, not yet
an owning arithmetic statement. The next walk must start at substep 1's exit
U update and exchange, not continue backward from substep 2's face average.
R168-P5 is **CONFIRMED**: the atomic raw-depth/no-extra-V-mask/seven-array/
materialised-`zhV` unit remains private and HELD.

## Controls, review and validation

The first instrument revision called round 129's finite-only scorer and was
rejected on the already-known non-finite path before emitting JSON. The second
revision incorrectly made the frozen face-SSH replay prediction an admission
invariant; the observed refutation therefore stopped classification. The
final gate separates protocol invariants from predictions and includes a
regression proving that a failed prediction is retained. A candidate-versus-
itself raw-depth row was also removed before citation; only its range and the
independent sum replay remain.

All four plants fire by corrupting the native-face census, source order,
depth-replay coverage and face-SSH-replay coverage. Focused coverage passes
44/44. The cumulative citation gate passes 274 citations with zero failures
or unmapped spans; the round receipt passes all four citations, and shifting
the U-face-average span by two lines makes the plant fail.

The one required `tests/ocean/fidelity -n 12` invocation reached 99%, then
stopped emitting for more than ten minutes and was interrupted without a
second full battery. Its retained log has 2,707 terminal outcomes: 2,696
passed, 7 skipped and 4 failed. The four failures are the registered
pre-existing reds from rounds 166-167: the moved GYRE spread record, unscoped
allow-dirty drivers, unstamped report emitters and SI3 scalar-math provenance.
No round-168 test failed; the log contains no symbol-materialisation error,
`MemoryError`, worker crash or pytest error.

The separate `codex exec --sandbox read-only` review attempt returned
**independent review unavailable in-sandbox** before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.

ASKED choices: continue the compiled-source independent rung-0 walk. UNASKED
choices: empty. No selector, stabiliser, forcing, carried-state policy,
configuration, package implementation or sea-ice feature changed.

## OPEN

1. At kt=8 external substep 1, walk the completed U velocity in compiled
   source order through the velocity update, implicit drag and seven-array
   exchange. The admitted record already carries every required `j001` row;
   no acquisition is currently needed.
2. Separately localise the 1/41 recorded-after-SSH face-average replay residual
   to its native face and decide whether it is pre/post-association timing or
   arithmetic before clearing the cited U-face average.
3. Keep the complete V-transport/halo unit private and HELD until its upstream
   compensating statement is exact and the Decision-96 shared gates pass.
