# NEMO testcase Lane 4 — ORCA2 card round 40 RHS-family acquisition receipt

Date: 2026-09-26

Parent: `f07500248a8d326e785a6b78044dfdb9c78ceed3`

Status: **STOPPED_FOR_RECORD — THE COMPILED MOMENTUM-OPERATOR BOUNDARIES
ARE NOT IN THE ADMITTED RECORD.**  A fail-closed acquisition is ready.  No
production model file changed.

All scientific results below are **given NEMO's entry**.  No independent
initial-state result is mixed into the table.  The six sea-ice selectors and
the card's `unmeasured_features` tuple remain unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round40/`.

## Reproduction and first missing boundary

Round 39 reproduces exactly at this parent, so R40-P1 is **CONFIRMED**:

| compiled boundary | unequal / scored | maximum |
|---|---:|---:|
| per-level `(e3u_3d*uu(Krhs))*umask` | 1,754 / 1,920 | `5.321462756514503e-08` |
| first partial sum, level 0 | 64 / 64 | `9.242193862245139e-11` |
| completed 30-level sum | 64 / 64 | `2.721729894586411e-07` |
| completed sum times production reciprocal | 64 / 64 | `7.356587026022005e-18` m/s2 |

The two reciprocal arms also reproduce: candidate sum with NEMO's reciprocal
differs by `9.420110520449829e-11` m/s2, and NEMO's sum with the candidate
reciprocal differs by `9.423577944554755e-11` m/s2.

The executing stage-1 order is pressure gradient, lateral diffusion,
vorticity, kinetic-energy gradient, then vertical advection at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:139-166`.
The admitted Round-20 ranked stream contains only the completed U/V RHS, not
the accumulator after those five calls.  Therefore R40-P2 is **UNMEASURED**:
this round cannot mechanically name HPG, LDF, VOR, KEG, or ZAD as the first
non-bit operator.  R40-P3 is also **UNMEASURED** until the new stream exists.
R40-P4 is likewise **UNMEASURED**; without an admitted isolated-operator
boundary, no single-statement landing is eligible in this round.

Exactly these per-rank files are missing:

| file | required bytes | registered contents |
|---|---:|---|
| `oracle_rhs_families_ranked_kt00000001_r0000.bin` | 35,434,332 | U/V after HPG, LDF, VOR, KEG, ZAD |
| `oracle_rhs_families_ranked_kt00000001_r0001.bin` | 35,434,332 | U/V after HPG, LDF, VOR, KEG, ZAD |

## Acquisition and controls

The acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round40_rhs_family_acquisition/run.sh`.
It uses the new target `ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM`, one file per rank,
and adds WRITE-only instrumentation to the cited compiled sequence.  It pins
the source build, binary, deck and inputs; requires exact inherited ranked
streams, four exact ocean/ice restart files, same-run post-ZAD closure, exact
headers and byte counts, unique completion markers, and a clean producer
commit.  Its header, truncation, swapped-rank, final-ULP, parent-byte, and
restart-byte plants must all fire before admission.

The clean committed preflight passes, the patched Fortran syntax-compiles,
and the writer-layout plant fires with exit 69.  No NEMO acquisition was run
in the sandbox, per the PMIx restriction.  The final focused
Round-38/39/40, receipt-citation, and citation-gate battery passes 27/27.  The
receipt citation gate reports zero failures, zero unmapped citations, and zero
map-audit failures; its rigid two-line plant fires.

The required `tests/ocean/fidelity -n 12` battery was launched once.  It
reached 99% and the inherited final-tail stall, then was interrupted after a
bounded wait and is not represented as green.  It emitted the same five known
failures as Rounds 38 and 39: SI3 scalar-math provenance, stale GYRE record
stamp, the round-51 trace suffix, unstamped legacy report emitters, and the
missing case-board row.  The new Round-40 tests passed.

The required separate `codex exec --sandbox read-only` review failed before
reading the diff because its app-server client could not initialize on the
read-only filesystem: **independent review unavailable in-sandbox**.

## Choices

ASKED: split the completed three-dimensional RHS by the compiled momentum
operators before any fix.

UNASKED: none.  No model, configuration, carried state, stabilizer, NEMO
source, sea-ice selector, score, or scientific threshold changed.

## OPEN

1. The operator runs the exact Round-40 acquisition, then its admission gate
   scores HPG, LDF, VOR, KEG, and ZAD in compiled order.  Failed preregistered
   predictions remain REFUTED.
2. The slow-forcing thickness/RHS/reference-depth compensation remains held;
   no operand changes before the missing operator boundary is admitted.
3. The actual whole-card first non-bit row remains kt=1 stage-1 temperature;
   the barotropic/transport handoff remains its upstream owner candidate.
4. The northern-fold mask/wind debt and Decision-52 independent year remain
   open.
