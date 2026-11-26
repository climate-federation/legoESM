# ORCA2 round 174 — passive completed-stage growth boundary

**Status:** `HELD`  
**Claim label:** every scientific number is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into this receipt.  
**Evidence:**
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round174/`.

## Verdict

The kt=8 HPG explosion is downstream, not the origin of the complete private
arm's growth. The first registered increase greater than 10x occurs at kt=1
stage 1: the four-field maximum rises from the frozen `2e-10` floor to
`3.2847473521544472 PSU`, a ratio of `16423736760.772236`. Salinity owns that
maximum; its argmax is `[86,159,3]`. R174-P3 is **CONFIRMED**.

The next greater-than-10x boundary is kt=7 stage 1 (`32.20926675776068x`),
followed by kt=7 stages 2 and 3 (`225.25225159192706x` and
`187915674.96454084x`). At kt=8, completed stages 1 and 2 are already wholly
non-finite in u/v and non-finite on 430,552 T/S cells; the ordinary stage-3
call retains the registered `e3w_int` refusal. Thus the round-172 local HPG
owner remains a true kt=8 RHS boundary but cannot own the upstream trajectory
growth. No HPG operand or statement is claimed here.

## Compiled order and instrument

The compiled rung-0 program computes the external mode and then calls stage 1
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:148-217`), followed by
stages 2 and 3, writing each completed stage immediately after its call
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:221-233`). The gate
re-admits all 80 self-describing frame shards: two ranks, ten steps, entry plus
three stage boundaries, with exactly-once global coverage.

The candidate uses the unchanged complete private arm: raw NEMO reference
face depth, no extra compact V mask, the seven-array external-mode association,
and materialised V transport. Existing detached stage-exposure models return
stages 1 and 2; their values never feed the ordinary model whose stage-3 result
advances the trajectory. For kt=1..7 the ordinary arm reproduces all 35
round-166 T/S/u/v/ssh digests exactly. The kt=8 ordinary call refuses at the
same registered `raw-mesh e3w_int` check. R174-P1, P2 and P4 are
**CONFIRMED**.

## Independent growth table

Each field is maximum absolute candidate-minus-NEMO error over the full
recorded field. `nf` is the candidate non-finite count; omitted means zero.
Growth is the current four-field maximum divided by the previous boundary's
maximum, floored only at the preregistered `2e-10`.

| kt | stage | T max [K] | S max [PSU] | u max [m/s] | v max [m/s] | growth |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.17740943620440253 | 3.2847473521544472 | 0.009592972317513049 | 0.027227136128767482 | 16423736760.772236 |
| 1 | 2 | 0.2723098244573918 | 5.040692521805088 | 0.009911672050702425 | 0.029019283591256526 | 1.534575412169501 |
| 1 | 3 | 0.5192523322242211 | 9.588461781459554 | 0.07242563381768057 | 0.07410933414583976 | 1.902211202127817 |
| 2 | 1 | 0.5841187435120525 | 10.80526801868621 | 0.06585695498476885 | 0.10411789412708727 | 1.126903174352689 |
| 2 | 2 | 0.6191613980781432 | 11.454031376607272 | 0.07593025470134163 | 0.10185190703576064 | 1.0600413943272038 |
| 2 | 3 | 0.656507336134716 | 12.177080638966174 | 0.17819254000739826 | 0.2721011595858127 | 1.063126181392832 |
| 3 | 1 | 0.6118347768422834 | 12.02007608926814 | 0.17703856555033826 | 0.2775085826301098 | 0.9871065525183741 |
| 3 | 2 | 0.6804205865610744 | 13.370243160591706 | 0.17735374036532825 | 0.29202910668901016 | 1.1123260003760735 |
| 3 | 3 | 1.0738664070052812 | 16.606036769677182 | 0.4449313673118699 | 0.7091369493035571 | 1.242014566991785 |
| 4 | 1 | 1.0735364476507145 | 16.74949581048776 | 0.42568614841341956 | 0.7139590706524628 | 1.0086389692375326 |
| 4 | 2 | 1.07329933231648 | 16.814390091367517 | 0.4388463900957804 | 0.7418847186516393 | 1.0038744020485157 |
| 4 | 3 | 1.5053588596521372 | 27.88199114798191 | 0.6273324341053012 | 1.0075365944945296 | 1.658221975134054 |
| 5 | 1 | 1.5232963025135335 | 28.318832371093315 | 0.5985223693317832 | 1.0111102095034967 | 1.0156675045477526 |
| 5 | 2 | 1.2940449112208774 | 23.976421512826395 | 0.6134622281440854 | 1.0457350167158839 | 0.8466599610688937 |
| 5 | 3 | 2.709266677030779 | 50.04827182938212 | 0.5110550935698221 | 0.9496048088901552 | 2.087395393954363 |
| 6 | 1 | 3.294749219372913 | 61.29145592327886 | 0.4824284312110756 | 0.9510511758917382 | 1.2246467996382673 |
| 6 | 2 | 9.503531644963774 | 177.97337593223784 | 2.2181341390785083 | 1.994102762576276 | 2.903722439796743 |
| 6 | 3 | 17.397469733047654 | 321.54624806544115 | 6.056877191279427 | 6.72230656465624 | 1.8067098316316015 |
| 7 | 1 | 560.1463437509537 | 10356.768878896883 | 25.985401302186975 | 28.753544001186764 | 32.20926675776068 |
| 7 | 2 | 26702.329150240894 | 498112.16028013075 | 2189575.448312433 | 2332885.509188721 | 225.25225159192706 |
| 7 | 3 | 156719139981.1695 | 3824151314223.051 | 294860536606946.0 | 438385755074195.06 | 187915674.96454084 |
| 8 | 1 | 0.0 (`nf=430552`) | 0.0 (`nf=430552`) | no finite pair (`nf=799200`) | no finite pair (`nf=799200`) | Infinity |
| 8 | 2 | 0.0 (`nf=430552`) | 0.0 (`nf=430552`) | no finite pair (`nf=799200`) | no finite pair (`nf=799200`) | 1.0 |

The machine-readable result is `stage_growth.json`, SHA-256
`e918fda0933efb764711e388237d2c0a5bc8af322bbdfa7b17c6b68d89aad8ce`.
The final serialized-report reclassification passes. Admission, passivity,
boundary-order and first-growth plants all refuse with exit 2 at their intended
predicates. The one-ULP control moves exactly one cell.

## Instrument corrections retained

Four pre-result refusals remain recorded in the preregistration: mismatched
digest serialization; an over-strict candidate-finiteness check; an undefined
`Infinity/Infinity` ratio; and NumPy argmax indices that were not JSON
serializable. A fifth check found sorted JSON mapping order differs from the
explicit T/S/u/v registry. Each defect was corrected mechanically before this
result; no failed run emitted a scientific JSON artifact.

## Scope, review and verification

This round changes only a preregistration, measurement gate, tests, citation
registry and receipt. No `packages/`, card, configuration, carried state,
stabiliser, threshold, sea-ice selector or production halo-unit path changed;
R174-P5 is **CONFIRMED**. Consequently GYRE, DINO, tanks and both production
ORCA2 ladders cannot move by construction.

The requested separate review verdict is **independent review unavailable
in-sandbox**. `codex exec --sandbox read-only` failed before reading the diff
with `failed to initialize in-process app-server client: Read-only file
system`; review log SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Focused round-174/166/103 tests report 23 passed; log SHA-256
`8c55f25f748ce9162af69782a345e0b969556458fd9a7107016a7f909f7be342`.
The required full `tests/ocean/fidelity -n 12` battery reports 2,751 passed,
7 skipped and 21 failed in 2,508.74 s; log SHA-256
`d24614eab6fbbd9fc28ab18b320b4d3c21f8a8cc4d71476ce70e7360a817cfde`.
Seventeen failures were fail-closed dirty-tree refusals because the citation
map was committed while those parallel workers were already running. All 17
pass together on the clean committed tip in 30.28 s; isolation log SHA-256
`18532e734a892e5530d66337642a0302592ff5b8c016133721aa5bdfaa23c64d`.
The four residual failures are the registered pre-existing reds: the GYRE
round-129 certified-year harness pin, allow-dirty scope ratchet, report
worktree-stamp ratchet and SI3 scalar-math `MY_SRC` provenance gate.

## OPEN

1. Round 175 starts at independent kt=1 stage 1, not kt=8 HPG.
2. Replay the stage-1 program offline from the exact kt=1 entry in compiled
   order, using the existing round-90/92 momentum, transport and tracer operand
   records. Select the first operator whose completed output leaves the floor;
   compare its operands one variable at a time.
3. If an operand boundary is absent or not rank-complete, write a new
   self-describing per-rank acquisition. No in-executable observer is allowed.
4. The complete halo/transport unit remains private and HELD. No configuration
   decision or NEMO acquisition is requested by this round.

## Choices

ASKED choices: continue the independent rung-0 passive-stage walk.  
UNASKED choices: empty.
