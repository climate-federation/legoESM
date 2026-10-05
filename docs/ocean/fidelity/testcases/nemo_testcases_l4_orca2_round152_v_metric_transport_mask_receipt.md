# ORCA2 round 152 — V metric-transport mask owner

Date: 2026-10-05. Base `c00b179bc1`; measurement tip `c6f0b7693`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round152.md`
at `0145bae51`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The substep-2 V metric transport's three compiled operands are individually
bit-exact over the complete 26,640-cell record: `e1v`, `va_e`, and `zhvp2_e`
are each **0 unequal cells**. Their source-written unmasked two-product is also
**0 unequal cells**. NEMO executes
`zhV = e1v * va_e * zhvp2_e`, with no face-mask multiplication, then uses its
north-minus-south difference in continuity and the sea-surface update at
compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591`.

legoESM's extra compact V-face mask is the first non-bit statement. It differs
from NEMO's recorded V mask at exactly **68 cells**, all on native row
`j=147`, first `(j,i)=(147,29)`, and is zero at every one. Multiplying the
otherwise-exact transport by that mask reproduces the production trace
bit-for-bit and creates exactly **68 unequal transport cells**, maximum
`155776.5627856178` transport units. Omitting only that uncompiled multiplier
makes the V transport, its north-minus-south difference, and the resulting sea
surface each bit-exact: **0 unequal cells** in the statement replay. The
control sea-surface replay also reproduces the production trace bit-for-bit at
the instantiated fast timestep `166.15384615384616 s`.

This does not land physics. The closing result is an offline statement replay,
not a production causal arm, and the registered approximately 31 PSU salinity
exposure remains a hard veto before either ORCA2 ladder may certify a change.

Final evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round152/transport_v_split_final.json`,
SHA-256 `b0fda1ce4c945d2d4cba581997202cd78d7139787ed89846073a85c2ca7733d6`.
It is stamped to `c6f0b7693d0e8bef85ecb3704a2dbeb9200f92bb`, CPU,
production JIT, fp64/libm, and x64.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R152-P1 prerequisites and round-151 control reproduce | **CONFIRMED**: all prerequisite rows remain exact and the raw-depth arm stays 0/68/68/68. |
| R152-P2 `e1v`, `va_e`, and `zhvp2_e` are exact | **CONFIRMED**: each is 0/26,640 unequal. |
| R152-P3 the extra compact mask creates the 68-cell debt | **CONFIRMED**: unmasked product 0 unequal; masked production 68 unequal; masked replay equals production exactly. |
| R152-P4 the unmasked transport closes V difference and SSH | **CONFIRMED**: both replay rows are 0 unequal. |
| R152-P5 production remains unchanged | **CONFIRMED**: only gate, test, preregistration, and receipt files changed. |

The first measurement artifact is **RETRACTED** as support for mask ownership.
It compared the production mismatch locations with NEMO's mask rather than the
live compact model mask and therefore reported zero overlap. The corrected
gate compares both masks, requires the masked replay to reproduce production,
and was committed before the final clean measurement. The invalid artifact is
retained as `transport_v_split.json`.

## Controls, review, and verification

Both preregistered controls fire through the real gate. The reordered operand
registry exits 2 with `STATUS PLANT-FIRED transport-v-registry`; the one-ULP
perturbation of an exact measured transport cell exits 2 with
`STATUS PLANT-FIRED transport-v-bit`.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No second independent review service is
available in this sandbox, so the measurement harness remains explicitly
UNREVIEWED despite its controls.

The final focused gate and citation battery passes 36/36. The one required
`tests/ocean/fidelity -n 12`
invocation is **INCOMPLETE**, not PASS: it reached 99% with 2,577 observed
passes, 7 skips, and the four registered pre-existing reds (SI3 scalar-math
provenance, GYRE round-129 spread-record stamp, round-35 escape scope, and
worktree-stamp scope). All pytest worker processes then disappeared without a
terminal summary while the execution pipe remained open; the stale session was
closed. No round-152 test failed.

The round receipt citation gate passes with 1/1 citation mapped and no
failure. The canonical receipt gate passes with 274 citations and no failure.
The round citation's shifted-line plant fires as `SYMBOL-NOT-AT-LINE`.

No ORCA2 ladder was run because the result is not yet a production causal arm.
No `packages/` file changed relative to the round base, so the GYRE trajectory
gate is not triggered. ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Add one private/default-off causal arm at the literal V metric-transport
   statement that omits only the extra compact `v_mask`, with the default
   proven bit-identical.
2. Re-run the substep-2 transport, continuity, and sea-surface chain through
   the production JIT. If it closes, run both ORCA2 ladders and the registered
   salinity/tracer compensation gate before any production landing.
3. A landing then requires the full shared-path GYRE, DINO, tank, citation,
   and push gates. Rung 0 remains HELD until that causal arm is safe.
