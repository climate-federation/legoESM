# SI3 Ladder Round 25 (`si3_r25`) — O4 acquisition-tool receipt

Date: 2026-10-28. Parent: `2351cfdda9f5`. Scope: O4 acquisition tooling only;
NEMO was not run. The measurement and confirm/refute decisions were frozen in
`docs/ice/fidelity/si3_r25_o4_acquisition_preregister.md:1-131` and committed as
`700a161af7` before the observer, reader, or scorer was exercised.

**Verdict: `PREFLIGHT_PASS`; ACQUISITION NEEDED.** The committed default route
is read-only. Its complete stub flow passed at
`/tmp/si3_o4_preflight.bHeJl7`; it rendered the exact two-step overlay, applied
the observer patch with fuzz zero, resolved all writer extents from real NEMO
allocations, derived the replay read-set from preprocessed NEMO, generated only
the files that the real writer can generate, replayed both sweep orders through
the production legoESM Prather path at 0 ULP, fired every plant, and completed
the real-flag FP census. No `nemo.exe` was invoked.

independent review: operator (Claude)

## Acquisition contract

The tool implements the O4 design row (`docs/ice/fidelity/si3_ladder_design.md:224`)
and Decision 170's prescribed-velocity overlay. It records, at the wrapper around
NEMO's production Prather routine, the horizontal metrics and surface masks,
constant ice velocities, the full five-category state, all 55 moment arrays,
the current and one-call-delayed CFL values, `icycle`, the six accumulators, and
the full exit. Steps 1 and 2 prove x-then-y and y-then-x ordering respectively.
The writer uses declared full/interior/ice-layer/snow-layer shapes; the shared
allocation gate resolved 89 calls at `(jpi,jpj,jpl,nlay_i,nlay_s) =
(184,152,5,10,5)` (`scripts/validate/ocean_fidelity/orca2_l4/nemo_acquisition_extents.py:309-350`).

The fail-closed binary reader rejects bad magic/version, truncation, extra bytes,
undeclared kinds, inconsistent extents, wrong step parity, and cross-record
metadata drift. The scorer constructs the explicit O4 configuration, feeds each
recorded entry through `si3_dyn_adv_pra_orca2`, and requires strict signed-zero
0 ULP equality for every state, diagnostic and moment exit, including nonempty
wet seam and north-fold regions (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_gate.py:68-239`).

## Observer and same-program controls

Observation is outside the physics body in a wrapper. Only three marked scalar
capture blocks remain inside the preserved body for current/delayed CFL and
`icycle`. Under NEMO's real `-O3` flags the observer-free and instrumented
physics functions each contain 1,225 FP instructions with no mnemonic-count
difference. The planted arithmetic statement changes the census. The shared
observer-strip implementation also proves the complete normalized physics
source body is identical after removing exactly three marked blocks
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_passivity.py:26-98`).

The O4 registration imports the admitted shared read-set scanner and follows
the production root plus locally resolvable callees. Dropping `sxice` fires the
control. The shared extent gate's missing-allocation, writer-mode,
dummy/actual, and generated-fixture plants all fire
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_readset.py:21-71`).

## Execution, retention and tests

The runner defaults to `--preflight-only`; only explicit `--run` can invoke
MPI/NEMO. The real route refuses insufficient RAM/disk, dirty tracked input,
existing evidence paths, absent independent executables, bad logs, changed ice
restarts, checksum drift, nonzero-ULP scoring, oversized evidence, or timeout.
It writes the source/binary/producer manifest before execution, computes all
artifact hashes afterward, and moves failed evidence to the retained refusal
area (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh:1-156`).

The five focused tests cover schema plants, exact production replay and four
physics plants, the shared real-allocation gate, the shared source-derived
read-set gate, fuzz-zero patching, and shell syntax
(`tests/ocean/fidelity/test_nemo_si3_o4_prather_acquisition.py:1-67`). They pass
in `114.90 s`. The complete default preflight also passed. The acquisition
itself remains deliberately unrun.

ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh
