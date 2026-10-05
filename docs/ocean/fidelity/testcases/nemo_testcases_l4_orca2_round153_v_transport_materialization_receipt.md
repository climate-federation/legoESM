# ORCA2 round 153 — V transport causal arm and continuity materialization

Date: 2026-10-05. Base `e1b0390b1`; preregistration `a866704d5`;
corrected measurement tip `b0778fa25`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The private/default-off production-JIT arm confirms half of round 152's
offline replay and refutes the full causal-chain prediction. With NEMO's raw
reference V depth held exact in both control and candidate, omitting only the
extra compact V mask closes the substep-2 metric transport from **68 unequal
cells, maximum 155776.5627856178**, to **0 unequal cells** over the complete
26,640-cell record. This is the compiled statement NEMO evaluates separately
from continuity at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591`.

The next separate statement is not bit-exact in the live JIT. The
north-minus-south V-transport difference has **8,786 unequal cells**, maximum
`2.3283064365386963e-10`; the complete divergence has **6,761**, maximum
`1.3552527156068805e-20`; and the resulting sea surface has **6,499**, maximum
`3.469446951953614e-18 m`. The transport operands themselves are exact, so
the first remaining source boundary is materialization/source rounding
between NEMO's completed `zhV` statement and its later subtraction, not the V
mask and not a configuration choice. The current JAX helper permits the
transport expression to fuse into the subtraction when the mask disappears.
Round 154 must test one source-rounding barrier at that written boundary
before either ORCA2 ladder.

The final evidence is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round153/causal_arm_final.json`,
SHA-256 `57881f88dc65e31d8dce99151f85594209a880b9a5d552467a35f90f6364ff4b`.
It is stamped clean at `b0778fa25cafa219439d8a508a485a5222b91411`, CPU,
production JIT, fp64/libm, and x64.

## Retraction

The earlier `causal_arm.json` artifact is **RETRACTED** as support for the
round-153 attribution. Its candidate omitted the mask without carrying the
round-151 raw-reference-depth prerequisite, mixing two variables and leaving
28 known midpoint-depth descendants in the V transport. It is retained for
provenance. The gate now supplies the same raw depth to control and candidate;
the corrected artifact above is the only cited causal result.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R153-P1 default-off hook is passive | **CONFIRMED**: ordinary state and exposed-state control remain bit-identical; GYRE base/tip checks below are also exact. |
| R153-P2 unmasking V closes transport, continuity, and SSH | **REFUTED**: transport closes 68→0, but continuity/SSH retain 8,786/6,499 unequal cells at the source-materialization boundary. |
| R153-P3 both ORCA2 ladders retain exact rows and first debt | **UNMEASURED_PREREQUISITE_R153-P2**: the causal statement chain is not bit-exact, so no ladder was authorised. |
| R153-P4 kt=10 stage-3 salinity remains safe | **UNMEASURED_PREREQUISITE_R153-P2**: the registered approximately 31 PSU veto was not approached with a non-exact causal unit. |
| R153-P5 every shared executing card passes if production changes | **NOT_APPLICABLE_NO_PRODUCTION_CHANGE**. |

## Controls, shared path, and review

All three current controls fire through the real gate. Disabling the corrected
causal arm while retaining the raw-depth prerequisite exits 2 with
`STATUS PLANT-FIRED transport-v-causal`. The inherited operand-registry and
one-ULP exact-cell controls exit 2 with `transport-v-registry` and
`transport-v-bit`, respectively.

The retained hook changes files under `packages/` but is private and false by
default. The required GYRE base/tip ten-step comparison is byte-identical:
70 certified rows, zero moved cells, zero status changes, first debt kt=3,
and both residual archives have SHA-256
`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`.
The 30-day members each produced 30 snapshots; every snapshot file is
byte-identical and the normalized hash manifest is
`20e05cb57c3e29d38cae784551c0b0ceb6170215e14f83f381ec118a2d3cb34e`.
Only each run's manifest provenance and wall time differ.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No independent review verdict is claimed.

No ORCA2 ladder, DINO, tank, or production-landing gate was run because
R153-P2 failed before the landing predicate. ASKED choices: none. UNASKED
choices: empty.

## OPEN

1. Preregister one source-materialization arm between the completed unmasked
   V metric transport and the north-minus-south subtraction. Test NEMO's two
   written statements with a rounding barrier, preserving the raw-depth and
   boundary-association prerequisites.
2. Only if transport, continuity, divergence, and SSH are all bit-exact may
   both ORCA2 ladders and the approximately 31 PSU salinity veto run.
3. A production landing still requires the complete shared GYRE, DINO, tank,
   generic-card, citation, and push gates. The default path remains unchanged.

## Verification

Focused unit, gate, and canonical citation tests pass **47/47**. The canonical
citation gate and this receipt's gate both pass with zero unmapped citations,
zero failures, and zero failing map entries. Shifting the receipt's compiled
source span by two lines makes the gate fail as `SYMBOL-NOT-AT-LINE`.

The one required `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**, not
PASS. It collected 2,606 items and reached 99% with 2,584 observed passes,
7 skips, and exactly the four registered pre-existing failures: the SI3
scalar-math provenance gate, GYRE round-129 spread-record stamp, round-35
escape scope, and worktree-stamp scope. The log then stopped advancing with
11 items unaccounted and no terminal summary; the stale controller was
interrupted. No round-153 test failed, and the corrected process census found
no remaining pytest process.
