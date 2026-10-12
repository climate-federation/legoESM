# NEMO SWG level-1 round 8 — shipped multi-year rung

Date: 2026-10-11. Base: `d250ce95a`. Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round8`.

## Preregistration

This section is frozen before implementing the round-8 scorer or running any
long legoESM integration. Round 7b has content-pinned the admitted ten-step and
thirty-day records; its clean citation gate passes and its shifted-citation
plant exits nonzero.

The oracle is the operator-acquired shipped reference run at
`round0_probe/year10`: 172800 one-hour steps, `nn_write=480`, 360 stored
twenty-day means, and the terminal step-172800 restart. The compiled engine
reads the run controls at
`SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:332-333`, copies `nn_itend` into the
executed limit at `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:374-377`, and executes
through that limit at `SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:147-160`.

Frozen predictions and falsifiers:

1. **Immutable oracle.** Before reading values, the scorer will require the
   recorded SHA-256 of the manifest, T/U/V/W files, final restart, mesh,
   initial-state output, and resolved `ocean.output`. It will also require the
   manifest's binary and namelist hashes. Any missing or changed byte
   **REFUTES** admission; a one-byte file plant must refuse.
2. **Independent shipped run.** Starting only from the SWG card's shipped rest
   state and analytic shipped wind, normal production `step()` will execute
   all 172800 one-hour steps on CPU under the explicit fp64/libm certification
   policy. No oracle state will enter the model. The run will use 480-step
   blocks and retain only the block diagnostics needed for scoring and
   statistics. Feeding an oracle state, changing the deck, or using a replay
   path **REFUTES** independence.
3. **Exact output semantics.** NEMO passes the just-computed stage-3 state to
   diagnostics after the time-level swap at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321`. The compiled diagnostic is
   called every step and documents the cadence at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:123-126`; it declares averaged output
   at `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:156-160` and writes SSH and
   native U/V from that state at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:457` and
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:536-538`. The scorer will reproduce
   IOIPSL's source-associated running mean, including its first-sample branch,
   from compiled support source `ext/IOIPSL/src/mathelp.f90:3086-3119`, then
   cast only at the stored output boundary. NEMO declares these three outputs
   as 32-bit records at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:258-259` and
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:363-374`. An instantaneous snapshot,
   a differently associated average, or an fp64-versus-fp32 comparison
   **REFUTES** the instrument.
4. **Bit prediction.** All 1080 stored output rows (360 times times SSH/U/V) are
   predicted bit-identical on their native wet/staggered masks after the
   compiled output cast. All ten fp64 terminal restart fields are predicted
   bit-identical. Therefore the predicted first over-bar output and restart
   rows are both `None`, and every maximum absolute difference is zero. Any
   unequal stored bit is **DEBT**, even if its relative score is at the 1e-15
   numerical bar.
5. **Frontier rule.** The scorer will emit every twenty-day row and the growth
   curve. If a row differs, it will report the first output index, ending step,
   field, unequal-cell count, first unequal wet cell, and maximum absolute
   difference. A twenty-day mean cannot identify an arithmetic statement, so
   no source attribution will be made. Instead, a new daily-cadence acquisition
   around that first window becomes `ACQUISITION_NEEDED`. Attribution without
   that narrower record **REFUTES** the receipt.
6. **Final restart.** NEMO schedules the terminal restart through the compiled
   `nn_stock`/`nitend` logic at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90:103-129`, after the just-computed
   time level is selected at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321`. The existing ten-field
   restart bridge, staggered masks, fp64 checks, and bit-exact gate will be
   reused unchanged. Omitting a restart field **REFUTES** coverage.
7. **Statistics after exactness only.** Only if the full rung is bit-identical,
   statistics will be computed from both matched stored series with shared
   diagnostics already in the repository: temporal-mean SSH, the shared
   barotropic transport streamfunction, shared cell-centred volume-integrated
   KE and zonal-anomaly EKE, and the shared isotropic KE spectrum on the 80 by
   80 wet basin. The two sides must be bit-identical inputs and yield identical
   reported statistics. New or duplicated diagnostic numerics, alternate
   weighting, or statistics before fidelity closure **REFUTE** this arm.
8. **Performance.** Wall time will exclude the initial JIT warm-up and include
   all 172800 production steps; peak resident memory will come from the process
   high-water mark. Throughput will be reported per 360-day SWG model year.
   **Correction to the operator shorthand:** the deck's `nn_leapy=30` selects a
   360-day calendar at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:416-426`, so the shipped run is exactly
   20 model years (19.71 365-day-year equivalents), not 19.7 model years under
   its own calendar.
9. **Controls and Rule 12.** The running-mean unit test will use nonzero values
   for which changing the source association changes a stored bit. The
   existing recorded-input one-ULP production control must remain DEBT. Given,
   chained, independent, carry, and month rows may not leave the bar, and all
   certified card hashes remain unchanged. The shifted-citation plant must
   exit nonzero. A vacuous plant, earlier frontier, changed certified hash, or
   dirty measurement tree **REFUTES** landing.

No physical, configuration, cadence, weighting, or carried-state choice is
introduced. The shipped deck and requested statistics are fixed by the user;
the statistical implementations reuse the shared diagnostics rather than
creating lane-specific definitions.

## Independent review

Independent review run by the operator.

## Measurement

Measured from clean commit `9135f587fc736f34d536ef28d389fdf8327631af` on
CPU with JAX x64 enabled and the explicit fp64/libm policy.  The legoESM run
started from the card's rest state, generated the shipped analytic wind, and
called ordinary production `step()` 172800 times.  No NEMO state entered the
model.  The initial three-stage compilation was discarded before the clock and
before resetting to rest.

### Verdict: the shipped rung is CLOSED bit-for-bit

All **1080/1080** stored rows are bit-identical: 360 twenty-day means times
SSH, native U, and native V on their T/U/V wet masks.  All **10/10** terminal
restart rows are also bit-identical.  There are zero unequal wet cells, every
row's maximum absolute difference is zero, and `first_output_over_bar`,
`first_restart_over_bar`, and `first_kt_over_bar` are all `None`.  Thus the
preregistered bit prediction is **CONFIRMED**.  The complete JSON artifact is
`shipped_ladder.json`, SHA-256
`0909e4dae34d9cca368a824d5434bf680fb991ebcc9dd69dc55cd50f21c45768`.

The 360 output times are exactly the compiled 480-step averages, from model day
10 through model day 7190 at twenty-day cadence.  The scorer reproduced the
compiled `ave(x)` declaration at
`SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:156-160`, the IOIPSL update association
at `ext/IOIPSL/src/mathelp.f90:3086-3119`, and the compiled fp32 storage
boundary at `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:258-259` and
`SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:363-374`.  The candidate remained fp64
through accumulation and was cast once at that boundary.  A direct test uses
nonzero values for which replacing the compiled association changes the stored
fp32 bit.

### First non-bit statement

There is none in the observed shipped record.  The diagnostic receives the
just-computed state after the compiled time-level swap at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321`; all stored output means and the
terminal restart are exact.  A twenty-day mean is not a statement-level
instrument, so this result attributes no unrecorded intermediate arithmetic.
The preregistered daily-window acquisition is therefore not triggered.

### Performance

The 172800 timed production steps took **2088.485138501972 s**, excluding JIT
warm-up: **82.7393965196927 steps/s** and **104.4242569250986 s per 360-day SWG
model year**.  Process high-water resident memory was **1140.3984375 MiB**;
the high-water mark at timing start was 982.24609375 MiB.  With the compiled
`nn_leapy=30` calendar selection at
`SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:416-426`, this is exactly twenty model
years.

### Matched statistics

Statistics were computed separately from the masked NEMO and legoESM stored
series only after the bit gate closed.  Both calls used the repository's shared
cell-centre interpolation, volume-integrated KE, zonal-anomaly EKE, barotropic
streamfunction, and isotropic spectrum implementations.  Every statistics
array is bit-identical between the two calls.  The compressed arrays are in
`shipped_statistics.npz`, SHA-256
`de435291718e02ecc21a56d310f724fe8ca0d3f34ffe347122a6c068b66d8f61`.

The matched summary is:

- temporal-mean SSH over wet cells: mean
  `-1.012340362649411e-09 m`, minimum `-163.39653205341762 m`, maximum
  `302.3985734624995 m`; field SHA-256
  `963575d1425ded100f7363d49b8caac5394a4a74752273324ad818658922f713`;
- mean-state barotropic transport streamfunction: minimum
  `-0.30541917652108147 Sv`, maximum `49.510267281504845 Sv`; field SHA-256
  `e7bae929effe94a02f899fa8e7c360a3c9b92eba598fbb1819756f074566996d`;
- volume-integrated KE across the stored series: temporal mean
  `66867299270393.21 m5/s2`, minimum `829892428692.0115`, maximum
  `78089765965358.61`; series SHA-256
  `95968dbd655f858831c000a3ddf5f55036ee6a13b17ea62f3d2ec66f5332a177`;
- volume-integrated zonal-anomaly EKE: temporal mean
  `24976339491516.715 m5/s2`, minimum `417412323284.8407`, maximum
  `29291282292237.133`; series SHA-256
  `6bee0892ccd0929eccfeeb9565d04a8765cff5ba861f57052073dd7978dec251`;
- the 80 by 80 wet-basin KE spectrum has 40 bins, spectrum SHA-256
  `f01bcf8a88984d318ac15384232d149148614dc1ff20f404bed9be3c14d37001`
  and wavenumber SHA-256
  `f43c88496355b8a9c965fe97fffd5db7fc8f82ec9a57d5ae7645035a902a7159`.

The units above follow the shared diagnostics' existing one-layer
volume-integral contract; no SWG-specific statistic was introduced.

### Instrument erratum and controls

The first full-arm launch correctly **REFUSED** before scoring its first row
because the shared numerical scorer accepts fp64 arrays while the shipped
records are intentionally fp32.  The repaired adapter now requires both
stored operands to be native fp32, promotes both only for the 1e-15 numerical
score, and keeps the native equality as the binding bit verdict.  A one-fp32-
ULP stored-record plant goes `DEBT`; it cannot be hidden by the numerical bar.
No result from the refused launch is used.

The manifest, binary, namelists, T/U/V/W files, final restart, mesh,
initial-state file, and resolved output echo all passed their fixed content
hashes.  The direct one-byte file mutation refuses admission.  The running-
mean reassociation changes a stored bit.  The established recorded-input
stage and month one-fp64-ULP controls both bind, go `DEBT`, and exit 1.

## Rule 12 disposition

Fresh executions at the measured commit preserve every established row:
given-entry **338/338**, chained **338/338**, independent ten-step **50/50**,
carried-state **29/29**, and thirty-day **690/690** are bit-identical.  Their
artifact SHA-256 values are, respectively,
`5dad6a10412f760a1d6eaea3bd2e1ffb1ad667b45937a3b08de4ff6c007016d6`,
`de8d80acf70520fbfb423b1c32f09fb5eac1ab36a913edd3387174397791a602`,
`8b3c17f1fcf68e6543df5a8897f9f77d6c5321be7e13977de8f09f34be787b0a`,
`9614b4c33a332bc539eb2c5252cb33c094bc1800b85c71bc64164fc63d0c0977`,
and `f54d2c1734b75ccf54ad7b8f40365b837e9cdd62fa10b2735df6046d176d5b41`.
Geometry remains 34/34 exact; its signed-y Coriolis, wet-cell, and layer-
thickness plants each exit 1.

The base-to-tip production diff under `packages/` and `src/` is empty.  The
round changes only the shared SWG validation script, its direct tests, citation
map, and receipts.  TSUNAMI, VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE,
OVERFLOW, DINO, and ORCA2 therefore execute no changed model, operator, card,
or configuration statement; their certified inputs remain byte-identical.

## Gates

The focused SWG ladder, card, and stage battery passes **186/186**; log SHA-256
`27079032740e692d47e70d7f6753f287b9f70c53320d8490a9ef076debf15bf7`.
The preregistration citation gate maps all thirteen compiled-source citations
with zero failures, unmapped citations, or map-audit failures.  Its shifted
stage-swap citation exits 1.  The final receipt gate is rerun after this receipt
is committed.

## Choices this round

No unasked physical, configuration, output cadence, weighting, or carried-
state choice was made.  The output cadence, run length, fp32 storage boundary,
calendar, and diagnostic fields all come from the shipped deck and compiled
consumers.  Offline statistics use existing shared implementations exactly as
preregistered.

## OPEN

1. The operator runs the independent review of round 8.
2. The SWG fidelity ladder itself is closed; the learned-closure phase may use
   the pinned statistics artifact after that review ships.
