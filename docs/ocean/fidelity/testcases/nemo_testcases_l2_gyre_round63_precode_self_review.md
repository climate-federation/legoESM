# Round 63 pre-code self-review

Date: 2026-09-12

## Scope and evidence audit

- Read the compiled R46 stage driver at all requested ranges and followed every
  active call into its compiled implementation.
- Checked the resolved ocean.output; optional inactive calls will not be
  represented as measured terms.
- Read the exact content association in compiled trazdf.f90:545--548.
- Read the explicit and substituted array declarations before choosing record
  bounds: do_loop_substitute.h90:72--89, traadv_fct.f90:94--111,
  zdf_oce.f90:85--87, dom_oce.f90:170,180,197, and
  MY_SRC/zdftke.F90:194--233.
- Read and will extend the existing round-54 tracer-decomposition parser.
- Read the round-56/59 writer and runner patterns and the round-59
  assumed-shape rebase failure.
- Preserved the two round-62 production candidates without applying either.

## Controls frozen before code

- Writer is WRITE-only and activates only at kt=2, stage 3.
- Reduced-domain dummy arguments will have explicit NEMO bounds; no
  assumed-shape dummy may receive an inner-domain-indexed array.
- Tracer calibration must reconstruct the recorded content using the exact
  NEMO association and require zero unequal values before analysis.
- One-ULP content plant must exit nonzero.
- Runner must enforce clean/stamped source identity, twin admission,
  resolved-configuration checks, per-file stamps, and nonzero plants.
- Build validation is preprocessing with the card keys and include paths,
  followed by gfortran -fsyntax-only for the writer and every dry-patched
  source. NEMO build and execution remain operator-only.

## Pre-implementation repository search

The closest reusable artifacts are:

- nemo_testcase_l2_gyre_round54_tracer_decomposition.py
- nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90
- the round-56 and round-59 runners and receipts
- R46's existing l2_r46_stage.F90/trazdf.F90 instruments

No second binary parser will be introduced.

## ASKED / UNASKED

| Kind | Item | Disposition |
|---|---|---|
| ASKED | Record active Krhs boundaries, content operands/assembly, and TKE RHS products | Implement |
| ASKED | Compile-check only, then stop | Enforce |
| UNASKED | Production physics or carried-state change | Not performed |
| UNASKED | NEMO build/run or configuration choice | Not performed |

