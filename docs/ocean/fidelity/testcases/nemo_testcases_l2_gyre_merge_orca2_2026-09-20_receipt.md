# GYRE NEMO-fidelity ORCA2 merge receipt — 2026-09-20

Status: **PREREGISTERED BEFORE MERGE.**  The model-hunk inventory below was
written on clean GYRE lane tip `4cac617cd` before merging ORCA2 tip
`4092639c3d32`.  Conflict resolution, trajectory comparisons, ORCA2 gate
reproduction, citation re-anchoring, tests, and the final verdict are pending.

## Scope and provenance

The merge target is branch `fidelity/orca2-on-lane`.  ORCA2 tip
`4092639c3d32` is 170 commits ahead of and 1,241 commits behind the target;
their merge base is `03c6e8d96ff7`.  Evidence is written under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_merge/`.

The lane already admits the native post-boundary/pre-Langmuir TKE frame, but
does not contain the certified ORCA2 assembly.  The ORCA2 branch's Phase-2y
handoff admits `VARIANT_ORACLE_ORCA1ICE`; its Phase-2w handoff pins the
icebergs-off `VARIANT_ORACLE_V2` and stops the ordered TKE walk at the first
non-bit statement, `zpelc` at kt=2.  Phase 2z staged two later SI3 frames but
did not execute or admit them.

## Pre-merge model-hunk inventory

This table records every non-merge ORCA2-side commit that changes `packages/`
before conflict resolution.  The synchronization merge `5bbb9ee56` imported
the common GYRE parent `03c6e8d96ff7`; it has no unique package delta relative
to that second parent and is not an ORCA2-authored model hunk.

| ORCA2 commit | Package hunk and purpose |
|---|---|
| `2c60066a9` | Adds the fail-closed, external-deck `ORCA2-zps` card; reads native 3-D partial cells, masks, metrics, initial T/S and analytic ice-load SSH; selects EOS-80/EEN/RGB/RK3; admits EEN in the shared WS-RK3 composition; reads literal `ff_t/ff_f`. |
| `af54d11ef` | Separates native NEMO F-point `ff_f` from generic V-face Coriolis, threads it through geometry padding/slicing, and routes only EEN/ENE vorticity consumers through the native F-point map. |
| `3a3f12e03` | Makes iceberg state explicit card metadata and pins the comparison card to `icebergs_enabled=False` with no iceberg inputs. |
| `ee1bbfcc2` | Adds the shared pure-JAX NEMO `fld_read` bilinear/bicubic map, source-index decoder, scalar-libm grid rotation, and east/north-to-i/j rotation for ORCA cards. |
| `5260fa8ae` | Selects `nemo_fld_read` explicitly on the ORCA2 card and removes that arm from its unresolved-feature list. |
| `c650babc6` | Adds policy-controlled scalar-libm `log/log10` and the source-statement NEMO RGB shortwave/chlorophyll/extinction path, wired with live/reference depth operands. |
| `ec576f51a` | Registers the ORCA2 stage-1 WZV diagnostic record's NEMO time levels; diagnostic metadata only. |
| `020a5043b` | Threads surface runoff into the shared WZV transport divergence and factors the literal horizontal-divergence and bottom-up W recurrences into shared helpers. |
| `d061ce395` | Adds private WRITE-only ORCA2 stage-1 W exposure/external-endpoint test hooks and strengthens vector/C2 card validation; no public card can select the hooks. |
| `f31af67d4` | Corrects the ORCA2 card validator's kinetic-energy-gradient selector spelling from `centered` to the actual `c2`. |
| `e30e7a93b` | Temporarily allows the private W diagnostic to construct around the then-uncertified EOS-80 geometric-depth guard. |
| `7aa560bc5` | Narrows that EOS bypass to construction of the exact private diagnostic combination, leaving ordinary execution fail-closed. |
| `1b7ea8be3` | Extends the same private construction-only bypass to the stage-1 tracer-boundary diagnostic. |
| `4b7eb88ee` | Maps native NEMO U metrics and rotation to redundant legoESM U faces with native U(i) at index `i+1` and the periodic last face at index 0. |
| `311654c63` | Applies NEMO's F-origin north-fold overwrite to live QCO vorticity thickness and its live stretch before EEN consumption. |
| `b51b3d519` | Adds shared NEMO diffusive-BBL geometry, EOS-80 alpha/beta coefficients, coefficient gate and bottom-tracer tendency; wires independent advective/diffusive BBL selectors and selects `nn_bbl_ldf=1` on ORCA2. |
| `d3e3aafc4` | Carries the frozen free-slip `fe3mask` separately from later slip/strait `fmask` edits and uses it in partial-cell live EEN thickness. |
| `01dcc1149` | Admits the measured EOS-80 geometric-depth combination in the model validation allow-list. |
| `9f11b5784` | Replaces the provisional adjacent-ULP BBL quotient search with the source-ordered, materialized ordinary IEEE division. |
| `fde040e91` | Seeds ORCA2's carried pre-closure TKE, viscosity/diffusivity, surface viscosity and dissipation fields from the resolved NEMO initialization, including the latitude-band tracer-diffusivity factor. |
| `076be0932` | Selects and validates ORCA2's RK3 `zdf_sh2` identity: face-native Nbb×Nbb velocity, face AVM weighting, and live-QCO face metric. |
| `a20407e3b` | Selects the executed ORCA2 bathymetry-relative bottom TKE Dirichlet boundary and corrects its config documentation. |
| `50c2b643c` | Renames the RK3 shear selector from ambiguous `now2` to the NEMO time-level name `nbb2` across validation and vertical-mixing dispatch. |
| `2efc0ce15` | Restores NEMO `nn_eice=1` as `tanh(10*fr_i)` through one shared dispatcher and selects it on ORCA2; default remains mode 0. |
| `ab223d8cd` | Removes remaining ambiguous “now” wording for Nbb; comments/docstrings only. |
| `bb1c60beb` | Adds NEMO `nn_eice=2` as raw ice fraction to the same shared dispatcher and validators; no default or existing card selection changes. |

## Textual conflicts

PENDING.

## Model-hunk execution audit on the GYRE card

PENDING.  Final classifications will be `yes`, `no`, or `inert`, with the
clean committed before/after trajectory proof as the binding check.

## GYRE trajectory proof

PENDING.

## ORCA2 gate reproduction

PENDING.

## Citation gate and tests

PENDING.

## Independent review

PENDING.  In-sandbox `codex exec` review is expected to be unavailable; the
operator will run the required Claude review.

## Choices

- No configuration default, scheme selection, threshold, cadence, state, or
  data source has been changed in this preregistration commit.

## OPEN

PENDING.  The ORCA2 handoff's current first non-bit statement is `zpelc` at
kt=2; the final receipt will name the exact next round from that boundary.
