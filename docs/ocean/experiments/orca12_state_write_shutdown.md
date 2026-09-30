# ORCA12 state-write shutdown fix

## Evidence and decision

The user supplied a `Shutdown` barrier error with 15 of 16 ranks present,
rank 0 absent, and incomplete restart/snapshot temporary archives. This is
evidence of shutdown skew during rank-0 output. It does **not** establish that
compression starved heartbeats. No new ocean integration was run for this fix.

The shared checkout's installed JAX source is
`/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/distributed.py`:
`State.initialize` defaults to heartbeat 100 seconds and shutdown 300 seconds.
The worktree has no local `.venv`. The driver's import-time bootstrap calls
`init_multicontroller_distributed`; both its explicit-coordinator branch and
the fallback initialization leave these timeout defaults intact. The cited
`early_init.py:497` is specifically the PALS/mpi4py fallback branch.

`run_omip._save_restart` already compresses in a background thread. Its final
join only holds rank 0, and the final snapshot was also rank-0-only. Peers
could finish the driver and enter JAX shutdown while root was still writing.

| Option | Benefit | Cost / decision |
|---|---|---|
| Increase JAX timeouts | Allows more rank exit skew | Delays dead-rank detection or shutdown failure; unchanged |
| Uncompressed NPZ | Removes compression CPU work | Larger files and filesystem traffic; cannot guarantee a write finishes before shutdown timeout; unchanged |
| Per-rank shards | Removes global-state gather and root serialization bottleneck | Requires new write/load/probe contracts; deferred |
| Coordinate background serialization | Keeps all hosts alive until root finishes | Selected; pauses model advancement at large checkpoints and exchanges a scalar status about once per second |

The new path applies only to multiprocess tripole states whose **global**
temperature array occupies at least 256 MiB. This is an I/O-size threshold,
not a resolution or physics choice; ORCA12 qualifies. Smaller states, other
grids, and single-process execution retain their save paths. Compression,
atomic publication, archive contents, and pressure-gradient defaults are
unchanged. No new launch flag is required.

All ranks finish gathering before root starts a worker. The main host threads
broadcast root's completion/error status until it finishes. Periodic and
wallclock checkpoints join the async restart writer inside this coordinated
operation. Final restart joining and final snapshot writing are coordinated
too. Restart errors raise on all ranks; optional snapshot errors retain the
existing warning behavior.

This is **not a filesystem-stall watchdog**. Dead-process heartbeat detection
is unchanged, but a live worker indefinitely blocked in filesystem I/O still
needs scheduler walltime enforcement or cancellation. Gathering and global
state memory costs also remain. The 16-rank, multi-GB filesystem rerun is
still required to validate the production outcome.

## Column probe review

`scripts/tmp/probe_tripole_runaway_column.py` was read and left unchanged.

- Centered speed nominally matches the driver's scalar, but averaging can
  hide opposite-sign face extremes. Report native U and V extrema and their
  vertical indices alongside centered speed. The driver's gathered diagnostic
  also masks the final V face, whereas the probe uses its stored value.
- `z_center_ref < H_bathy` can discard a wet partial bottom cell when its
  reference center lies below the actual bottom. Use actual wet thicknesses
  and cell centers from the reconstructed state/geometry. Mask each neighbor
  at each level, not just by the two-dimensional land mask.
- Neighbor indices are computational directions, not geographic north/east
  on a tripole. Longitude wraps and the north-fold mapping are absent.
- The probe prints T/S for seven levels and only a 3-by-3 density min/max at
  the selected level. It does not print the promised density column profiles.
  `wright_eos` takes potential temperature in degrees Celsius and pressure in
  Pa; `depth * 1e4` is an approximate pressure. Exact model density requires
  the run's EOS selection, hydrostatic pressure, geometry and precision policy.
- A vertical location cannot uniquely identify forcing, pressure gradient,
  or advection as the cause. It selects the column for a tendency budget.

Prefer supplying the completed `restart_day*.npz` together with the snapshot
and exact launch configuration: the diagnostic snapshot lacks the full
partial-cell geometry needed by this attribution.

## Validation and review

Unit coverage includes the size/process/grid gate, delayed compressed atomic
publication with two simulated ranks, a planted disk-full failure, and
execution of the production cadence/final routing blocks. Existing restart
archive tests are included: **12 focused tests passed**. Bypassing coordination
in memory made the delayed-write test fail, as required. The broader module
run was interrupted during the ocean integration tests after prolonged lack
of progress; it is not reported as passed. An archive-label test exposed an
async read race and now joins the writer before reading.

A real two-process test uses a write longer than
its deliberately short shutdown timeout, plus an uncoordinated negative arm.
The latter test cannot run in this sandbox: socket creation raises
`PermissionError: Operation not permitted` before either process launches.

Two independent agent reviews found no blocking production defect. They
identified and prompted correction of a test race between archive publication
and completion notification. These were available agent reviews, not the
house rule's specifically named Claude/GLM reviewer pair.

Authorized choices: coordinated large-tripole I/O; retain compression and JAX
timeouts. No physics choices changed. No production throughput or ocean
stability claim is made by these software tests.
