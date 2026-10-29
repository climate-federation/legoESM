# Preregistration: ORCA2 round 345 — Decision-188 year-comparison gate

Date: 2026-10-29. Base: `d3a0a329c692`. Imported reviewed round-342 tool:
`c983dc345a3520cc6f45be900496bb4d935c5058`.

## Frozen work

1. Extend the observer-free round-342 K0 year tool by one arm only. The arm
   stages the same deck and binary, replaces the staged
   `data_1m_potential_temperature_nomask.nc` symlink with a regular copy, and
   advances every finite wet `votemper[:, 0, :, :]` float32 value by exactly
   one representable value toward positive infinity. All 12 records use the
   same uniform wet-surface pattern. Land, deeper levels, metadata, and all
   other input files remain unchanged. The file is read by NEMO both for cold
   start and for the configured T/S damping target; that dual use is recorded,
   not hidden.
2. Record the resolved source path, source and perturbed SHA-256 digests,
   variable/dtype/indexing, wet mask source, changed-value count, and exact ULP
   audit in a manifest. Run the same observer-free build, real-flag census,
   allocation, fixture, namelist, forcing, completion, retention, and stub
   checks inherited from round 342. Never run NEMO in this round.
3. Add an automatic month-by-month Decision-188 gate over outputs of the
   Decision-160 metric code. Fixed physical bars are SST and T-band RMS
   `0.01 K`, SSS and S-band RMS `0.002 psu`, SSH RMS `0.001 m`, PSI RMS
   `0.1 Sv` plus pattern correlation `>= 0.999`, Drake absolute monthly-mean
   difference `0.5 Sv`, and MLD RMS `1 m`. Surface KE passes when its relative
   RMS is at most `max(0.01, 3 * envelope_relative_rms)` for the matching
   month. The gate supports both the 12-month round-342 report and the
   Decision-160 ORCA2X 30-day pilot report.

## Frozen confirmations and falsifiers

- Perturbation **CONFIRMS** only if at least one value changes, every selected
  value is exactly one float32 ULP toward positive infinity, every unselected
  array element is bit-identical, and source/perturbed digests differ.
  Otherwise it **REFUTES** the acquisition tool.
- A fixed-bar plant with SST RMS `0.010000000000000002 K` must fail while an
  otherwise identical report at `0.01 K` passes. Failure to separate them
  **REFUTES** the fixed-bar gate.
- A KE plant at relative RMS `0.02` with envelope relative RMS `0.007` must
  pass (`0.02 <= 0.021`). Failure **REFUTES** the 3x envelope rule.
- A KE plant at relative RMS `0.011` with zero envelope must fail because the
  floor is `0.01`; a candidate at `0.01` must pass. Failure to separate them
  **REFUTES** the floor.
- The ORCA2X pilot-shaped 30-day report must produce exactly one evaluated
  partial-month verdict and no invented pending-month verdicts. Failure
  **REFUTES** pilot compatibility.

No model configuration, physics default, numerical bar, or NEMO source is
changed by this round.
