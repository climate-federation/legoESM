# ORCA2 round 156 — external-mode association growth owner

Date: 2026-10-05. Base `b0f6f56c1`; preregistration `90e23ae30`;
measurement commit `974c11ed4`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
initial state, forcing, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The corrected complete association arm first departs from unchanged production
at **kt=1 stage 1**. The first scored field in the frozen T/S/u/v/ssh/uu_b/vv_b
order is T: 23,866 cells differ and the maximum absolute difference is
`0.17740749781639398 K` at `[147,49,0]`. At the same boundary its maxima are
`3.284732984680897` for S, `0.0614312493956707 m s-1` for u,
`0.033962181653721245 m s-1` for v, and `0.13111400818677185 m` for ssh.
The arm then reproduces round 155's first non-finite boundary at **kt=8 stage-1
T** (430,552 non-finite T cells; first `[1,49,0]`).

The compiled source unit remains NEMO's one seven-field `lbc_lnk` call at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`. NEMO builds
the face depths earlier at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, materialises
the V transport at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, and consumes
the completed U/V transports in the separate continuity loop at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591`.

At the first finite departure, seven source-ordered one-output measurement
arms give the following result against unchanged production:

| associated output | boundary bit-exact? | T unequal | T maximum absolute |
|---|---:|---:|---:|
| U velocity | no | 10,956 | `0.0016328912725630529 K` |
| V velocity | no | 23,505 | `0.1774972822409795 K` |
| U depth | yes | 0 | 0 |
| V depth | yes | 0 | 0 |
| U inverse depth | yes | 0 | 0 |
| V inverse depth | yes | 0 | 0 |
| sea surface | yes | 0 | 0 |

Thus only the call's first two outputs carry the kt=1 stage-1 movement. In
compiled order the **U association argument is first**, with V a second
non-vacuous carrier. This names the first non-bit source statement as the
seven-field association call and narrows its active outputs to U/V. It does
not authorize a U-only or V-only landing: those are not NEMO program units,
and the complete cited statement remains non-finite by kt=8.

The causal artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round156/association_growth.json`,
SHA-256 `b521afca3ef9d94b0aa56c2f5283bf06be84722115f95d62be6c65682cc484c2`.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R156-P1 production is passive | **CONFIRMED**: 0/200 rung-0 rows moved against round 155, first debt remains kt=1 stage-1 T, and kt=10 stage-3 S remains `0.4156673855360964`. |
| R156-P2 finite departure precedes refusal | **CONFIRMED** at kt=1 stage 1. |
| R156-P3 a returned barotropic field precedes T/S | **REFUTED**: T is first in the frozen scored-field order at the first moved stage boundary. |
| R156-P4 a one-field split is non-vacuous | **CONFIRMED**: U and V move; the other five outputs stay bit-exact. |
| R156-P5 terminal boundary reproduces round 155 | **CONFIRMED** at kt=8 stage-1 T. |

## Retained instrument and shared path

The retained change is private measurement plumbing. The empty selector is the
only default and changes no card or normal production call. Unknown fields
refuse, a complete and partial arm cannot be selected together, and the split
gate rejects a missing/out-of-order seven-report set.

The required GYRE default path is exact. Its ten-step gate has 0/70 moved rows,
all 210 saved residual arrays are `np.array_equal`, and the residual archive
SHA-256 remains
`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`.
All 30 day-1..30 snapshot archives are byte-identical to round 155; their
snapshot-set digest remains
`9ca5110d40c5d512eb2754b6519974807964e9445c6190b66fbc04084c3677aa`.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No independent verdict is claimed.

No DINO, VORTEX, tank, generic-card, rung-7 candidate, or production landing
gate was run. The complete source statement is already terminal on rung 0, so
those downstream landing predicates cannot change this round's HELD verdict.
ASKED choices: none. UNASKED choices: empty.

## Controls and verification

The one-ULP comparison, signed-zero, non-finite, and unknown-field-selector
plants all fire nonzero. Focused round-156, round-146, and rung-103 tests pass
**38/38**. The canonical citation gate reports zero unmapped citations, zero
failures, and zero failing map entries after both changed model files were
mechanically re-anchored with `difflib.SequenceMatcher`.

The one required `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**, not
PASS. It reached 99% and showed exactly the four registered pre-existing
failures from round 155: SI3 scalar-math provenance, the GYRE round-129 spread
record stamp, round-35 escape scope, and worktree stamping. After several
minutes without log growth, the invocation was interrupted once and was not
rerun. No round-156 test failed.

## OPEN

1. Keep production unchanged. At kt=1, walk the U then V branches of the
   cited association through the external substeps, recording the first
   differing cell immediately after each association and after the first
   transport/continuity consumer.
2. Split cyclic exchange from the north-fold pivot inside the compiled U and
   V association branches; a partial output remains a measurement arm only.
   Name the first geometry operation that reproduces the finite stage-1
   growth before considering a complete production statement.
3. Any complete statement candidate must first finish rung 0 and retain all
   200 rows before rung 7, month, or shared-card landing gates are relevant.
