# NEMO testcase Lane 4 — ORCA2 Phase-2x preregistration

Date: 2026-09-06

Parent: `b7ce08cc8afa5cf377922abf198cf1794fab8a73`

Status: **PREREGISTERED BEFORE THE REPLACEMENT VARIANT IS STAGED OR RUN.**

## P2X-1 — reject the Phase-2w launch, resolve the complete ice deck

The Phase-2w A/B runs are `REJECTED_INIT_FAILURE`: both stopped with code 123
at `ocean.output:800-804`, before stepping, because `ln_pnd=F` was combined
with inherited `ln_pnd_LEV=T`.  They and their logs remain immutable evidence.

Resolve every assignment made by
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg` against the
ORCA2 variant's `namelist_ice_cfg` over `SHARED/namelist_ice_ref`.  The
replacement must match every ORCA1-set row, block by block.  A row may remain
different only when its ORCA1 value requires an ORCA1-grid input unavailable
to this ORCA2 oracle, and each such row must be named.  The registered case is
`nn_iceini_file`: keep ORCA2's source-supported analytic initialization (`0`)
rather than read ORCA1's grid-specific `Ice_initialization` file (`1`); copy
the ORCA1 scalar initialization values and file descriptors even though the
file descriptors are inactive.  Ocean geometry, forcing, icebergs-off, run
length, MPI layout, binary, and instruments remain fixed.

## P2X-2 — source-artifact hygiene

Replace the three committed full NEMO files with unified patches against the
exact shipped NEMO 5.0.2 base files.  Store the full applied sources under
`/data/abyssal/dbalwada/nemo-testcases-l4/build/phase2x_orca1ice/MY_SRC/`.
The receipt and manifest must pin base, patch, and applied-file SHA-256 values.
Applying each patch to its pinned base must reproduce the external full file
byte-for-byte.  Nothing in the shipped checkout is modified.

## P2X-3 — ORCA1 CORE2 driver audit

Diff `scripts/run/run_omip_core2.py` across the Decision-12 commit.  Instantiate
or intercept its resolved TKE configuration before and after.  Retain the
change only if it merely admits/document mode 2 and the production ORCA1
selection remains exactly mode 3 with no other resolved-config difference.

## P2X-4 — SI3 acquisition coverage

Audit writers from their WRITE lists.  The dynamics acquisition must cover
`ht/hu/hv(Kmm)`, iceberg and landfast T/U/V masks, per-subcycle basal stresses,
`tau_icebfr`, `rn_lf_tensile`, entry velocities, stresses, strength, and
category state.  Thermodynamics must cover the per-category column state and
surface fluxes in `icethd.F90` order.  Missing cheap operands are added as
WRITE-only, canonical, self-describing fields before any replacement build.
Schema plants must exit nonzero.

## Admission boundary

The new Phase-2x twins are admissible only if both finish ten steps, their new
records decode to exact EOF from their headers, all record streams are
byte-identical A/B, inherited records satisfy the established identity rule,
ordinary outputs are instrument-inert, and all plants bind.  MPI execution is
reserved for the user shell; this turn stops after handing off fresh guarded
launchers.

## ASKED / UNASKED

| action | classification | preregistered disposition |
|---|---|---|
| retain Phase-2w failures | ASKED | label `REJECTED_INIT_FAILURE`; delete nothing |
| resolve all ORCA1-set ice rows | ASKED | exact match except named ORCA1-grid input rows |
| keep `nn_iceini_file=0` | ASKED by input-compatibility constraint | only deliberate ice-namelist difference; ORCA1 scalar initial values still copied |
| edit shipped NEMO | forbidden | never |
| replace full sources with patches | ASKED, repository hygiene | base+patch+applied hashes and reproduction gate |
| alter ORCA1 CORE2 behavior | decision-gated | revert if the audited diff changes mode 3 or any resolved value |
| extend missing SI3 operands | ASKED | WRITE-only, zero-first, header-derived schema only |
| execute MPI/NEMO in sandbox | forbidden | stage two new run directories for user shell |
| delete, push, or commit large records | forbidden | none |
