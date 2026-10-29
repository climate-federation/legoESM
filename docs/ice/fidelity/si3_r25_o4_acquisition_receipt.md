# SI3 Ladder Round 25 (`si3_r25`) — O4 acquisition-tool receipt

Date: 2026-10-29. Parent: `fb6a6b736bf6`. Scope: O4 acquisition tooling only;
NEMO was not run. The measurement and confirm/refute decisions were frozen in
`docs/ice/fidelity/si3_r25_o4_acquisition_preregister.md:1-132` and committed as
`700a161af7` before the observer, reader, or scorer was exercised.

**Verdict: `PREFLIGHT_PASS`; ACQUISITION NEEDED.** The default route is
read-only. The resumed preflight rendered the exact two-step overlay, compiled
and ran the patched NEMO source under `-fcheck=bounds`, completed the real-flag
1,225 = 1,225 FP census, replayed both sweep orders at 0 ULP, fired every plant,
and exercised generated-driver success, refusal, retention, move-aside, rerun,
and `--admit-existing`. No real `nemo.exe` was invoked.

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
wet seam and north-fold regions (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_gate.py:71-242`).

## Observer and same-program controls

Observation is outside the physics body in a wrapper. Only three marked scalar
capture blocks remain inside the preserved body for current/delayed CFL and
`icycle`. Every executable capture has a real-ppsrc line-order plant. Under
NEMO's real `-O3 -funroll-all-loops -fno-tree-vectorize` flags the observer-free and instrumented
physics functions each contain 1,225 FP instructions with no mnemonic-count
difference. The planted arithmetic statement changes the census. Admission
repeats that census on the actual `icedyn_adv_pra.o` objects linked into the
retained REF/REC binaries. The normalized same-program proof compares the
marked nonblank insertions against real CPP output and rejects marked-content
or unmarked-statement plants
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_passivity.py:26-109`).

The O4 registration imports the admitted shared read-set scanner and follows
the production root plus locally resolvable callees. Dropping `sxice` fires the
control. The shared extent gate's missing-allocation, writer-mode,
dummy/actual, and generated-fixture plants all fire
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_readset.py:21-71`).

## Execution, retention and tests

The runner defaults to `--preflight-only`; only explicit `--run` builds or runs
NEMO. It reuses the admitted ORCA2/SAS bundle staging and `makenemo` REF/REC
recipe, uses separate user stdout/stderr logs, compares the real
`ORCA2_SAS_00000002_restart*.nc` sets byte-for-byte, and sets
`nn_stock=nn_itend=2`. The pre-run manifest pins repository-relative sources,
producer commit, resolved deck, inputs and retained binary digests. Admission
re-hashes them and every artifact before scoring. Failed evidence and both
builds remain under `refused/`; the documented rerun moves builds aside and
never deletes them (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh:286-456`).

The stub writes only real-build output names. Its twelve records come from an
independent analytic zero-transport construction; production code is used only
by the scorer. A separate nontrivial fixture makes the sweep, fold-sign and
subcycle plants red. Six focused tests cover both fixtures, schema/physics
plants, shared extent/read-set gates, fuzz-zero patching and shell syntax
(`tests/ocean/fidelity/test_nemo_si3_o4_prather_acquisition.py:1-85`). They pass
in `103.72 s`; the complete default preflight also passes. The acquisition
itself remains deliberately unrun.

ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh
