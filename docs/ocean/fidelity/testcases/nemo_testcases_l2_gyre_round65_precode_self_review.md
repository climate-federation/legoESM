# Round-65 pre-code self-review

Date: 2026-09-12

## Scope audit

- The admitted round-64 record is immutable and is consumed only through its
  stamped parser/gate.
- The candidate is a shared FCT arithmetic transcription, not a configuration
  selection or carried-state change.
- The candidate does not touch the year harness, reconciliation gate,
  freshwater pair, #1484 guard, or held round-60/round-62 patches.
- No NEMO source will be modified and no NEMO build or integration will run.

## Source/branch audit

- Active compiled GYRE call site: `stprk3_stg.f90:860` calls `tra_adv` after
  zeroing at `827-829` and after staged transports are formed at `278-297`.
- Active FCT call: `traadv_fct.f90:172`; upstream writer: `503-510`; first-guess
  writer: `602-611`; centred/anti-diffusive writers: `197-201,264-269`; active
  memory-optimised nonosc: `745-944`; final correction: `318-329`.
- The candidate therefore lies on the compiled, executed branch, not a dead
  source arm.

## Failure audit

- Calibration inequality stops before the walk.
- Candidate failure is a recorded REFUTED result, not a broadened patch.
- Plants are required to exit nonzero.
- No configuration choice is authorised; any unavoidable choice is put in the
  ASKED/UNASKED table and stops implementation.

