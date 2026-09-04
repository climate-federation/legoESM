# NEMO testcase Lane 4 ORCA2 — MPI-writer correction preregistration

Date frozen: 2026-09-04 (America/New_York)  
Scope: one WRITE-only writer correction, scalar-math rebuild, and replacement
ten-step record run. The frozen oracle inventory and all physics remain
unchanged. No legoESM work is authorized.

## Failure discovered before acceptance

The completed `instrumented_full_10step_np2` run reached step 10 and emitted
the frozen 90-file inventory, but a sequential schema walk refuted four SI3
streams. The append writers in `icesbc.F90`, `icestp.F90`, and `icethd.F90`
were called on both MPI ranks and opened one shared filename. For example,
`oracle_si3_thd_frames.bin` contains valid-looking headers from both local
domains: its first category has `npti=1851` and an interleaved header has
`npti=1779`. It contains 158 magic strings where rank 0 alone would emit 35
frames. `oracle_si3_zdf_inputs.bin` contains 42 magic strings rather than 25,
and `oracle_si3_reassoc_operands.bin` contains six rather than five. The bulk
and exchange streams have the expected structural counts only because both
ranks write equal-sized frames; their payload ownership is still racy and is
therefore invalid.

This is a record-I/O defect, not a model-arithmetic result. The run and all of
its outputs remain retained and hashable, but it is **retracted as the oracle
record set**. No record from it may be mixed into the replacement set.

## Frozen correction

The selected layout has always been NEMO's `lwp` rank-0 convention. Add only
`IF( .NOT.lwp ) RETURN` at the entry of these config-local WRITE-only helper
routines:

- `icesbc.F90`: `l3bulk_dump_tau`, `l3bulk_dump_flux`;
- `icestp.F90`: `l3xchg_dump`;
- `icethd.F90`: `l3thd_dump_global`, `l3thd_dump_1d`, `l3zin_dump`,
  `l3rea_dump`.

These guards suppress non-writer ranks before any file open or write. They do
not assign a model array, change arithmetic, or change a selector. All other
MY_SRC bytes, the CPP keys, scalar-math arch, input symlinks, namelists, and
`jpni=2,jpnj=1` layout remain fixed.

The corrected binary must be rebuilt with `conda-scalarmath`, have zero
dynamic `_ZGV*` symbols, and be copied to the Lane-4 data root with a new
content hash. A fresh directory named
`instrumented_rank0_10step_np2` must contain the exact ten-step deck and input
links plus a guarded Bash-time launcher executing only:

```text
mpirun -np 2 --oversubscribe ./nemo
```

## Frozen acceptance

**Pre-run erratum:** the first committed version incorrectly wrote 35 for the
thermodynamic-frame count by collapsing each category's five internal stages
to one. `icethd.F90:116,138-174,196,231` gives, on each of the five odd ice
steps, one entry + five stages for each of five categories + one post-remap +
one exit = 28 frames, hence **140**. This correction is frozen before the
replacement run and does not use a measured replacement result.

The replacement run is accepted only if it reaches step 10 with launcher
status zero; emits exactly the same 90 filenames; every stream parses
sequentially against its writer schema with no trailing bytes and finite
defined-domain payloads; the SI3 counts are rank-0 counts (bulk 15 frames,
exchange 10, thermodynamics 140, ZDF inputs 25, reassociation 5); and every
ordinary model output passes the preregistered identity comparison against the
uninstrumented ten-step control. The planted-control set is unchanged.

A failure remains **REFUTED**. This correction does not authorize a revised
inventory, a per-family splice, another precision, or any Phase-2 work.

## ASKED / UNASKED

| Item | Disposition |
|---|---|
| rank-0 writers and documented local domain | ASKED in the original brief; implementation correction |
| preserve the malformed run | standing no-delete rule; retracted and retained |
| seven early-return guards | UNASKED enabling correction; WRITE-only I/O ownership only |
| rebuild and replacement MPI run | UNASKED enabling correction; required to meet the frozen gate |
| user-shell MPI execution | ASKED standing rule |
| physics, deck, inputs, layout, inventory | unchanged; no new choice |
| legoESM code or numerics | forbidden |
