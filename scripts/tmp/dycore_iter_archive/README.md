# Archived ralph-loop dycore iteration tests (pending review → deletion)

These 419 files are the **non-curated remainder** of the 476-file
FV3/cubed-sphere dycore iteration corpus that accumulated at `tests/` root. The
57 that lock distinct numerical invariants were promoted to
`tests/atmosphere/dycore/regression/` (see its `MANIFEST.md`); these are the
redundant siblings — earlier/exploratory iterations, per-flag sweeps, and
near-duplicate PE/NH variants subsumed by a kept representative.

They are parked here (NOT in `tests/`, so pytest does not collect them) as a
**recovery pool**: if a curated test rots or an invariant turns out to be
under-guarded, pull the relevant sibling back. This whole directory is intended
for eventual deletion once the curated suite proves sufficient over a few release
cycles.

**Self-contained import note:** three archived files import each other —
`test_guard_sweep_no_duplicates_iter415.py` → `test_fv3_fidelity_guard_sweep_iter383.py`,
and `test_fv3_nh_toolkit_iter172.py` / `test_fv3_pe_toolkit_iter188.py` →
`_iter187_marker.py`. All four+marker live here together, so those edges resolve
if the files are run from this directory. No **kept** regression test imports any
archived file.

Do not wire these into CI. To resurrect one, `git mv` it back to
`tests/atmosphere/dycore/regression/` and confirm it passes under
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`.
