# ORCA2 round 150 — midpoint V reference-depth owner

Date: 2026-10-05. Base `203d5bc225`; measurement tip `1f0da11f8`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round150.md`
at `50fd66604`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The first non-bit midpoint V-depth operand is NEMO's frozen reference depth
`hv_0`: **30/26,640 cells unequal, maximum 899 m**, first `(j,i)=(147,39)`.
Every other operand in the compiled statement is bit-exact over its complete
recorded domain: midpoint SSH, T-cell area, local and associated-north
area-times-SSH products, reciprocal V-face area, and `ssvmask` are each
0 unequal. NEMO's executed order is the midpoint extrapolation followed by
`hv_0 + 0.5*r1_e1e2v*(local+north)*ssvmask` at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`.

The read-only one-variable replay replaces only the builder's reference V
depth with the card's raw NEMO `hv_0`, preserving the already-measured dynamic
SSH average. It makes `mid_depth_v` **bit-exact: 0/26,640 unequal**, from the
30-cell/899-m control. The replay itself reproduces the production trace
bit-for-bit before substitution. The shared builder whose operands were split
is `barotropic_latlon_cgrid.py:570-649`.

This does not yet land physics. The substitution was an offline statement
replay, not a production arm, so its descendants remain unmeasured. The frozen
control still has 68 unequal `transport_v` cells (maximum
`155776.5627856178` transport units), 68 unequal `continuity_dv` cells at the
same maximum, and 68 unequal `after_ssh` cells (maximum
`0.003203816535399729` m). Their compiled order is
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591`.

The final evidence is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round150/midpoint_v_split_final2.json`,
SHA-256 `284117a9f838ac319b998b659d1bfe440479595775269035687813675b4e8493`.
It is stamped to `1f0da11f8365b5fd7f3207ebff2e2a552c3ad962`, CPU,
production JIT, fp64/libm, and x64. Its numerical subtree is exactly equal to
the prior final-stamped rerun; only the worktree stamp and the corrected
fail-closed P4 wording differ.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R150-P1 prerequisites and 30/68/68/68 control reproduce | **CONFIRMED**: observer, seven post-association arrays, and eight EEN coefficients remain exact; the registered control census and maxima reproduce. |
| R150-P2 midpoint SSH and both area-SSH products are exact | **CONFIRMED**: midpoint SSH, T area, local product, and north product are each 0 unequal. |
| R150-P3 first non-bit operand is `hv_0` | **CONFIRMED**: `hv_0` is 30 cells/899 m; reciprocal area and `ssvmask` are exact. |
| R150-P4 raw `hv_0` closes depth and descendants | **UNMEASURED_DESCENDANTS_DEPTH_CONFIRMED**: depth closes 30→0, but transport/continuity/SSH were not run through a production causal arm. |
| R150-P5 production unchanged | **CONFIRMED**: only a validator, tests, preregistration, and receipt changed. |

## Controls, review, and verification

The operand-registry and exact-cell one-ULP plants each exit 2 with their named
`STATUS PLANT-FIRED` line. The new instrument initially refused before any
number because native masks were passed to a compact-face helper; the adapter
was corrected to use the existing certified native-to-model converters, then
the clean measurement was repeated. The failed attempt is retained in
`midpoint_v_split.log` and supplies no evidence.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No second independent review service is
available in this sandbox, so the measurement harness remains explicitly
UNREVIEWED despite its mechanical controls.

Focused round-146/150 and citation-gate tests pass 30/30. The one required
`tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**, not PASS: it reached
99%, with 2,578 passed, 7 skipped, and the four registered pre-existing reds
(SI3 scalar-math provenance, round-35 escape scope, worktree-stamp scope, and
the GYRE round-129 spread-record stamp), then repeated the known xdist tail
stall and was interrupted. No round-150 test failed.

No ORCA2 ladder was run: R150-P4 remains unmeasured for the descendants and
the registered approximately 31 PSU salinity exposure remains a hard veto.
No `packages/` file changed, so the GYRE trajectory gate is not triggered.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Add one private/default-off production causal arm that supplies the card's
   raw NEMO `hu_0/hv_0` to the existing SSH-average prep, without changing its
   dynamic products, reciprocal areas, masks, configuration, or carry.
2. Re-run the substep-2 `mid_depth_v`, `transport_v`, `continuity_dv`, and
   `after_ssh` chain. The arm is eligible for a production fix only if all four
   become bit-exact and every earlier exact row stays exact.
3. If the chain closes, run both ORCA2 ladders and the registered compensating
   salinity/tracer gate before any landing; then run the full shared-path GYRE,
   DINO, tank, citation, and push gates required for a `packages/` change.
