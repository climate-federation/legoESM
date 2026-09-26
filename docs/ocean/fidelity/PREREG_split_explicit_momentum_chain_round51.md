# Preregistration: free-surface filter chain, round 51

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 50 (`9d17606b1dfe4190456df58cb1a0e4aaac6e6456c86849fb93dbfc39f3b7dc80`)
releases the free-surface filter at NEMO `stpmlf.F90:458-459`. This round uses
only retained day-180 full-halo streams and scores four ordered operands:

1. `ssh_atf`, `sshwzv.F90:518-528`: plain Robert-Asselin filtering of Kmm SSH
   from restart Kbb, retained Kmm, and the final split-explicit Kaa SSH;
2. filtered `r3t_f`, `domqco.F90:159-161`;
3. filtered metric-weighted `r3u_f`, `domqco.F90:165-166`;
4. filtered metric-weighted `r3v_f`, `domqco.F90:167-168`.

The DINO freshwater-removal term at `sshwzv.F90:522-528` is exactly zero:
`emp_b=emp=0`, `ln_rnf=F`, and `ln_isf=F`. Kaa is
`spg_dump_pssh_final.bin`, not the older pre-split-explicit
`sshnxt_dump_ssh_after.bin`; the latter is a planted wrong-time-level arm and
must fail.

Each ordered row retains the POINTWISE `1e-15` bar on normalized RMS and
maximum error over NEMO RMS. Identity must pass, wet-point and roll plants
must fail, the stale-Kaa arm must fail, masks/populations and all full-halo
shapes must match, and all inputs/sources/receipts are hash-bound. The first
row outside either bar stops the chain. Only all four rows AT BAR releases the
momentum-RHS tail; no tracer-tail row is measured out of order.
