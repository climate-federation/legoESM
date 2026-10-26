# NEMO testcase Lane 4 ORCA2 — SI3 payload-count correction preregistration

Date frozen: 2026-09-04 (America/New_York)  
Scope: one WRITE-only header correction, scalar-math rebuild, and replacement
ten-step record run. No physics, inventory, or legoESM work is authorized.

## Failure found by the full schema walk

The rank-owned `instrumented_rank0_10step_np2` run proves that the concurrent
writer defect is fixed: bulk, exchange, thermodynamic, ZDF-input, and
reassociation streams contain respectively 15, 10, 140, 25, and 5 monotone
rank-0 frames. All 90 files parse to EOF and their payload values are finite
when the reassociation payload is read from its source write list.

One header field is nevertheless false. Config-local
`MY_SRC/icethd.F90:l3rea_dump` declares `18*npti` values, then writes:

```text
t_su(1) + t_i(10) + t_s(5) + sz_i(10) + e_i(10) + e_s(5)
+ qns_ice(1) + dqns_ice(1) = 43 values per active point.
```

Thus the retained file's five frames are structurally unambiguous but not
header-valid. It is retained and **retracted as the final oracle record set**.
The already-passing ordinary-output identity comparison does not waive a
malformed record header.

## Frozen correction and acceptance

Change only the declared integer expression in `l3rea_dump` from
`18 * npti` to `43 * npti`. The payload write list, model arrays, arithmetic,
CPP keys, namelists, inputs, rank ownership, and frozen 90-file inventory are
unchanged. Rebuild with the same `conda-scalarmath` arch and require zero
dynamic `_ZGV*` symbols.

Prepare one fresh two-rank directory named
`instrumented_rank0_schema_10step_np2`, using the same 19 copied deck files,
40 input symlinks, guarded Bash-time launcher, and exact command:

```text
mpirun -np 2 --oversubscribe ./nemo
```

Acceptance requires step 10 and launcher status zero; exactly 90 records; all
headers, declared sizes, payload sizes, kt/category/substep sequences,
binary64 finiteness, and EOF checks passing without a compatibility exception;
all planted controls exiting nonzero; and the same per-file ordinary-output
identity result against `uninstrumented_10step_np2`. No record may be spliced
from either retracted run.

## ASKED / UNASKED

| Item | Disposition |
|---|---|
| truthful binary record schema | ASKED by schema validation requirement |
| change `18*npti` to `43*npti` | UNASKED enabling correction; WRITE-only metadata only |
| retain the rank-owned but header-invalid run | standing no-delete rule; retracted and hash-pinned |
| rebuild and one user-shell run | UNASKED enabling correction; frozen before execution |
| all scientific and layout choices | unchanged |
| legoESM work | forbidden |
