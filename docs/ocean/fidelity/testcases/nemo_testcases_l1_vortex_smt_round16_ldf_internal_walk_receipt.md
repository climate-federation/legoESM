# Receipt — VORTEX_SMT round 16 (lane round 228): internal LDF walk

**Status: STOPPED_FOR_RECORD.**  No physics, card, carried state, trajectory,
or certified number changed.  The admitted Round-227 record closes the whole
native-slope producer under the production JIT and names the first non-bit
output as A33, but A33 carries only `4.017621229876821e-14` of the measured
LDF magnitude.  The record then becomes non-citable at six horizontal-tensor
rows because its writer copied scalar temporaries after their producing loop.
A corrected, new-target acquisition is committed and requested.

Base: `cffa2ab79afacfb3c70361b40df7766eeedf7a3a` (round 227).
Preregistration: `2a97d4649`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round228/`.

## 1. Admission and aggregate calibration

The Round-227 payload is passive and self-consistent: the instrumented and
plain step-10 restarts are byte-identical; all 16 slope and 50 ISO groups parse
from their self-described ranks/extents to EOF; `rhs_after-rhs_before` rebuilds
`rhs_increment` with zero unequal cells; the header, field-name and truncation
plants all exit nonzero; and the two SHA-256/commit/name stamps pass.  The
operator-run log ends in `ROUND227_LDF_INTERNAL_RECORD_READY`.

The production-JIT walk reproduces the frozen Round-226 boundaries exactly:

| boundary | max absolute T difference |
|---|---:|
| pre-LDF | `7.418332614861356e-11 K` |
| post-LDF | `2.0915088416728622e-07 K` |
| additional across LDF | `2.090767008411376e-07 K` |

This confirms R16-P1 and R16-P2.  The compiled program constructs EOS, N2 and
slopes before its RK stages at
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177`, and the
resolved dispatcher calls `traldf_iso_lap` at
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110`.

## 2. Production-JIT slope walk

The executed slope loop and its four filtered outputs are the statements at
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:226-356`.
The intermediate rows exclude the unexecuted surface plane because NEMO's
backward loop starts at Fortran level 2; the final arrays retain their defined
zero surface.  Every scored array is fp64 on a `63×63×10` card interior.

| row family | cells unequal | max absolute |
|---|---:|---:|
| `prd`, `pn2` | 0 / 0 | 0 / 0 |
| live `e3u`, `e3v` | 0 / 0 | 0 / 0 |
| raw `zau`, `zav` | 0 / 0 | 0 / 0 |
| bounded `zbu`, `zbv` | 0 / 0 | 0 / 0 |
| mixed-layer raw U/V | 0 / 0 | 0 / 0 |
| final `uslp`, `vslp`, `wslpi`, `wslpj` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

R16-P3 predicted a non-bit final slope.  It is **REFUTED**: the complete
recorded slope chain is bit-exact under the production JIT.

## 3. First non-bit statement and magnitude falsifier

The first genuine non-bit output after those slopes is NEMO's A33 assignment,
`pah_wslp2 = (zahu_w*wslpi)*wslpi + (zahv_w*wslpj)*wslpj`, at
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:822-837`:
910 wet cells differ, with maximum `4.336808689942018e-19`.  `akz` and all
three following tracer-gradient rows are bit-exact.

This is a compiled-rounding residue, not the magnitude owner.  A post-hoc
one-variable reconstruction substitutes only NEMO's A33 array into the
already-materialized production operands and rebuilds the exact downstream
A33 flux/divergence algebra.  Its maximum tendency change is
`2.916635388791526e-24 K s-1`; even the deliberately conservative `dt`-scaled
bound is `8.399909919719594e-21 K`, only
`4.017621229876821e-14` of the additional LDF maximum.  R16-P4 required at
least 90%; it is **REFUTED**.  No rounding candidate is built or landed.

The one-ULP plant on `iso.ah_wslp2` exits 1 and prints `STATUS PLANT-FIRED`.
Thus the claimed production row is controlled, while the reconstruction is
correctly labelled post-hoc rather than promoted to a production measurement.

## 4. Record retraction and corrected acquisition

The next compiled horizontal-flux block is
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:240-267`.
Lines 240-260 produce scalar `zA11/zA22/zA13/zA23/zmsku/zmskv` inside one
cell loop.  The Round-227 writer then opens a second loop and copies the final
scalar values across every cell (lines 262-267).  Therefore the six recorded
groups `A11`, `A22`, `A13`, `A23`, `hmsku`, and `hmskv` are **RETRACTED as
cellwise evidence**.  The committed walk excludes them from its first-non-bit
selection; it does not print a false operator verdict.

The vertical loop records its scalars inside the producing loop at
`VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:282-333`, but
the first horizontal-flux statement cannot be isolated without its valid
factor rows.  A new target pair `VORTEX_SMT3_VEC_R16_OMIP_L1{,_P3}` moves only
those six assignments inside the existing producing loop.  The source card
otherwise reuses the admitted R15 files and remains additions-only.  Its dry
preflight reports both helper and patched-NEMO gfortran syntax passes and
`ROUND228_LDF_INTERNAL_PREFLIGHT_PASS`.  It retains restart passivity,
self-describing parsing, three nonzero plants and commit stamps.

Acquisition requested:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round16_ldf_internal/run.sh`.

## 5. Landing and blast radius

R16-P5 is not reached: the first non-bit statement fails the magnitude
falsifier, and the downstream discriminator needs a corrected record.  No
candidate enters production, so the SMT/flat-VORTEX/tank, GYRE ladder/year,
generic recipe and ORCA2 numerical registries have no moved row to register.
The only `packages/` changes are private write-only diagnostics: an LDF-only
trace no longer requires the unrelated GYRE shortwave observer, the returned
prognostic state still comes from an independent ordinary compiled call, and
both final W slopes plus the unshifted A33 arrays are exposed.  Ordinary calls
with the hook unset take the pre-existing branch.

**UNASKED list: EMPTY.**  No physical/configuration/default choice was made.

## 6. Tests, review, and citations

The clean production walk passes, while its one-ULP `iso.ah_wslp2` plant exits
1 and reports `STATUS PLANT-FIRED`.  The private-work-directory DINO month
gate exits 0 with:

> DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS

The focused pytest invocation reports:

> 56 passed in 100.85s (0:01:40)

This covers the Round-227 record/admission controls, the Round-224 SMT-3
record controls, the receipt-citation gate, and the SMT card unit tests.  The
clean citation gate passes all six mapped compiled-source citations with zero
failures, audit failures, or unmapped citations.  Its shifted A33 citation
plant exits 1 with `SYMBOL-NOT-AT-LINE` after moving the cited start to line
824.  The default campaign receipt also remains fully mapped after the
model-file citation re-anchor.

The required separate read-only Codex review could not initialize inside the
sandbox.  Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict.  The round nevertheless lands no physics candidate,
and the corrected acquisition is fail-closed behind admission and calibration.

## 7. OPEN

1. Operator: run the committed Round-228 acquisition.  It must report NEMO
   `STOP 0`, byte-identical step-10 restarts, exact payload calibration, all
   three plants, commit-stamp pass and `ROUND228_LDF_INTERNAL_RECORD_READY`.
2. Next round: admit the R16 record, reproduce this round's exact slope/A33
   and aggregate rows, then resume at the six corrected horizontal tensor
   factors in compiled order.  Name the first operand/association whose
   one-variable production substitution carries the `2.090767008411376e-07 K`
   LDF increment.
3. Only a magnitude-closing statement proceeds to the full SMT/flat-VORTEX,
   tank, GYRE ladder/year, DINO-month, citation, plant and review gates.  Then
   continue Decision 93 with SMT-4.
