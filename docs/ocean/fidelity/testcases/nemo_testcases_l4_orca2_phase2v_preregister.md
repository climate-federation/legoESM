# NEMO testcase Lane 4 — ORCA2 Phase-2v preregistration

Date: 2026-09-06

Starting parent: `74964cadc9de5644f1e7adaf91ad978722654d9b`

Status: **PREREGISTERED BEFORE REPLACEMENT MEASUREMENT.**  The Phase-2u twins
are rejected because their sole new record is not reproducible.  All 100
inherited records and ordinary model output remain the WRITE-only control.

## P2V-1 — reconcile and repair the TKE record

The observed 54,889 twin-differing bytes must first be mapped through the
self-describing header.  The initial allocation hypothesis confirms only if
they lie in slots the writer never canonically assigned; any differing byte in
a source-defined/consumed slot voids the WRITE-only acquisition and stops.

The source artifact already zeroes every persistent `l4_tke_*` allocation at
`l4_tke_begin`.  Therefore distinguish an allocation omission from a later
copy of an undefined local workspace.  NEMO defines the tridiagonal workspaces
at the surface and on `jk=2:jpkm1` (`zdftke.F90:264-268,405-420,451-469`).
Their writer copies must never include `jk=jpk`.  Preserve the existing
`tmask`/`mbkt` wet-column guard and zero-first storage; restrict each workspace
copy to the exact source-defined active range.  No model array may be assigned.

Rebuild with `makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath`; admission
requires zero `_ZGV*` dynamic symbols.  Stage independent A/B `(2,1)` twins.
The replacement confirms only if:

- both launchers finish at step 10 with zero MPI status;
- all 100 inherited records are raw-identical to the Phase-2s V2 root;
- the new record is raw-identical between twins, giving 101 / 101 total;
- restart shards/history payloads pass the existing identity gate;
- the header-derived schema reaches exact EOF and every canonical slot is zero;
- magic, extent, count, truncation, trailing, canonical, and twin plants all
  exit nonzero.

The Phase-2u roots remain retained and are labelled
`REJECTED_NONREPRODUCIBLE_TKE_RECORD`; they cannot be used as an oracle target.

## P2V-2 — large per-run result relocation

Move only the two Lane-4 Phase-2t per-run JSON outputs exceeding 20 KiB to
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2t/`, preserving their bytes and
SHA-256.  Update receipt and manifest paths.  Receipts and the small cited
summary JSONs stay in git.  No unrelated large historical lane output is in
scope.

## P2V-3 — one-pass ordered TKE scorer

Prepare one production-JIT CPU/fp64/scalar-libm gate which reads the admitted
kt=2 ZDF inputs and replacement TKE targets, then walks source order:

1. no-Stokes `zWlc2=zcsd*taum` (`zdftke.F90:332`);
2. `zpelc`, `imlc`, `zhlc`, `zus3`, and the Langmuir en increment (`:339-367`);
3. Prandtl `p_pdlr`, tridiagonal coefficients and RHS (`:381-420`);
4. forward/RHS/back substitution, floor, and `nn_etau=1` (`:451-500`);
5. raw and bounded `nn_mxl=3` `mxlm/mxld` (`:645-704`);
6. `avm`, `avt`, `dissl`, and `nn_pdl=1` assembly (`:711-724`).

Each row compares the production path and a source-literal statement arm to
the oracle target on defined rank-zero cells, reports `unequal / n` plus the
first cell and row-scale ULP, and has a target-bit plant that exits nonzero.
Stop at the first non-bit statement.  Shared arithmetic is
`GYRE_OWNER_SHARED_TKE`; an ORCA2 selector or forcing operand is Lane 4.  The
gate may be written and statically tested now, but no numerical result is
admissible from the rejected Phase-2u target.

## ASKED / UNASKED

| action | classification | preregistered disposition |
|---|---|---|
| decode the rejected twin difference | ASKED | map every differing range before editing |
| repair and rebuild the writer | ASKED | zero-first plus source-defined copy bounds only |
| execute replacement MPI/NEMO | forbidden in sandbox | user-shell twins only |
| relocate two >20 KiB Phase-2t per-run JSONs | ASKED review carry | preserve bytes and hashes under data root |
| prepare full TKE scorer | ASKED | one gate, ordered rows, binding plants |
| score rejected Phase-2u TKE target | forbidden by twin rule | never used for a physics verdict |
| change shared TKE arithmetic | Lane-4-forbidden | register and hand to GYRE after valid acquisition |
| enter EVD/IWM or SI3 | downstream/unasked | not entered before TKE closes |
| delete, edit shipped NEMO, add multi-MB git data, or push | forbidden | none planned |
