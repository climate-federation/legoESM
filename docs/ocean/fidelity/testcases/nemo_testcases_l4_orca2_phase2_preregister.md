# NEMO testcase Lane 4 — ORCA2 Phase-2 preregistration

Date: 2026-09-04  
Parent: `5bbb9ee56ad885e35abd8b244831697194e0e0c7`  
Oracle receipt: `nemo_testcases_l4_orca2_phase1_receipt.md`  
Scope: legoESM ORCA2 card and ordered kt=1 fidelity walk only

## Claim and stopping rule

This round builds an ORCA2 card by selecting or extending the shared canonical
NEMO WS-RK3 implementation.  It does not add card-local numerical formulas.
Every new formula must be the shared NEMO identity with a NEMO 5.0.2
`file:line` citation.  The card sets fp64 and the shared scalar-libm policy
explicitly.

Measurements proceed in NEMO execution order and stop at the first boundary
that is over its registered bar or needs a modelling choice.  Results beyond
that boundary are **UNMEASURED**.  A missing input, state component, or selected
scheme is also UNMEASURED, never treated as zero-equivalent without an oracle
source branch proving that zero.

The detailed oracle files were written by rank 0 on its local `94 x 152`
domain.  Each comparison strips two NEMO halo points on all four sides and
therefore scores the owned `90 x 148` western subdomain against the same slice
of the legoESM full `180 x 148` owned domain.  No rank-1 payload is inferred.

## Frozen inputs and interpretation

The input archive is immutable:

- `/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz`
- size `1,365,673,011` bytes
- SHA-256 `5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4`

The unpacked deck root is
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0`.
The accepted oracle root is
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/instrumented_reviewfix_10step_np2`.
Every consumed external file is hash-pinned in the Phase-2 manifest.

`ORCA_R2_zps_domcfg.nc` has `x=180`, `y=148`, `z=31`.  The last vertical
record is NEMO's dummy `jpk` record; the prognostic domain is `180 x 148 x 30`.
The Phase-1 receipt's `184 x 152 x 31` description includes four MPI halo
points horizontally and the dummy vertical record and is not the legoESM
owned-array shape.

The geometry loader will reproduce the source order, not reconstruct geometry
from nominal resolution:

1. `domhgr` reads T/U/V/F longitude, latitude, `e1`, `e2`, `ff_t`, and `ff_f`.
2. `domzgr` reads `e3t_1d`, `e3w_1d`, 3-D `e3t/e3u/e3v/e3f`, and
   `top_level/bottom_level`; the exact `e3_to_depth` recurrence supplies the
   reference depths.
3. masks and T/U/V/F water-column depths are derived using NEMO's indexing and
   north-fold/cyclic neighbour rules.  The derived fields must be checked
   against NEMO's emitted mesh fields before they can be labelled VERIFIED.

The no-restart initial ocean state follows `istate`: temperature and salinity
are the `fld_read` interpolation of the two bracketing monthly records at
kt=1; velocity and sea-surface height are initialized by the selected source
branch.  The interpolation operation order is frozen to NEMO's
`before_weight * before + after_weight * after`, with float32 file values
promoted to fp64 before arithmetic.  Alternative reassociations are plants.

## Ordered boundaries and bars

| ID | boundary | registered output and bar | disposition before measurement |
|---|---|---|---|
| B1 | deck/card construction | exact source-file hashes; exact raw coordinate/metric/thickness values; exact masks and grid topology; every resolved feature VERIFIED or explicitly UNMEASURED | UNMEASURED |
| B2-O | kt=1 ocean step entry | `T,S,u,v,ssh`: byte identity on every rank-0 owned payload element; `0 / n` unequal | UNMEASURED |
| B2-I | kt=1 SI3/iceberg step entry | all Phase-1 registered SI3 `jpl=5`, layered/moment, and iceberg state represented and byte-identical; `0 / n` unequal | UNMEASURED; current shared main state is not assumed sufficient |
| B3a | stage-1 surface forcing | field-by-field rank-0 comparison in source order: NCAR/CORE `sbcblk`, runoff, freshwater carry, geothermal, tidal mixing, RGB chlorophyll/QSR, and ice-ocean exchange; oracle-relative pointwise bar from the shared gate | UNMEASURED |
| B3b | EOS/BN2 and SCO HPG | reuse the shared Lane-2 schemas and oracle-relative pointwise gate unchanged | UNMEASURED |
| B3c | 65-substep external mode | reuse the shared Lane-1/2 split-explicit state/RHS/drag/mean gates unchanged | UNMEASURED |
| B3d | momentum and volume transports | shared WS-RK3/QCO EEN program, including census-round bottom-boundary-layer geometry and raw-W ladder; shared gate unchanged | UNMEASURED |
| B3e | tracer stages | shared FCT, RGB QSR, vertical diffusion/IWM and GM/EIV program; shared gate unchanged | UNMEASURED |
| B4 | SI3 dynamics and thermodynamics | Lane-3 certified implementations and gates, unchanged; entered only after B3e clears | NOT ENTERED |

For an exact-copy/input boundary the bar is bit identity.  For an arithmetic
boundary the shared oracle-relative cellwise gate and its frozen per-stream
bar apply; an aggregate norm cannot override a failed cell.  Every gate must
include a binding planted violation that traverses the production validator
and exits nonzero.  Reports use exact numerator and denominator (`0 / n`), not
rounded percentages.

## Resolved-program coverage registry (pre-measurement)

| component | required ORCA2 selection | starting disposition |
|---|---|---|
| horizontal/vertical grid | ORCA2 tripole, cyclic I, north T-fold, z-partial steps, raw 3-D `e3` | UNMEASURED |
| ocean time levels | WS-RK3/QCO `Kbb/Kmm/Krhs/Kaa` registry from Phase 1 | UNMEASURED |
| equation of state | EOS-80 | UNMEASURED |
| hydrostatic pressure gradient | SCO | UNMEASURED |
| momentum | vector form, EEN vorticity; 65 split-explicit substeps | UNMEASURED |
| tracer transport | FCT with selected GM/EIV path | UNMEASURED |
| vertical physics | TKE, EVD, double diffusion and internal-wave mixing | UNMEASURED |
| shortwave | RGB bands with file chlorophyll and analytical depth profile | UNMEASURED |
| atmospheric forcing | NCAR/CORE bulk fields with NEMO `fld_read` time/space interpolation | UNMEASURED |
| freshwater/bottom forcing | runoff/river-mouth, freshwater carry, geothermal and tides | UNMEASURED |
| SI3 | enabled, `jpl=5`, 10 ice layers, 5 snow layers, Prather moments | UNMEASURED |
| icebergs | enabled with calving input present; `nn_test_icebergs=10`, `ln_use_calving=F` | UNMEASURED |
| PISCES/TOP | excluded exactly as Phase 1; no physics feedback selected | WAIVED by oracle definition |
| XIOS | excluded exactly as Phase 1; WRITE-only binary oracle records replace diagnostics | WAIVED by oracle definition |

The detailed state coverage and time-level tables will enumerate every card
field, including state that is not yet representable.  A broad component row
cannot hide an omitted field.

## Controls

The committed validation command will run at least these plants through the
same production code paths:

1. corrupt one raw grid/metric value;
2. use the wrong north-fold partner;
3. retain the `jpk` dummy level as a prognostic level;
4. reassociate the monthly T/S interpolation;
5. corrupt one T payload value;
6. corrupt one S payload value;
7. corrupt one zero-initialized velocity/SSH value;
8. score the local record without stripping halos;
9. omit one resolved input/coverage item.

Each plant must exit nonzero.  Later boundary gates retain their own existing
plants unchanged.

## ASKED / UNASKED

| choice or fact | status | preregistered handling |
|---|---|---|
| begin Phase 2 from merged commit `5bbb9ee56ad8` | ASKED | exact parent |
| new ORCA2 card on shared NEMO WS-RK3 identity; no ORCA-local numerics | ASKED | enforced by source layout and provenance audit |
| full global card, compare only rank-0 owned subdomain | ASKED | `[:, :90]` owned-I slice after halo stripping |
| CPU, production JIT, fp64 plus scalar-libm | ASKED | stamped by validator |
| ordered walk and stop at first decision/over-bar boundary | ASKED | B1 through B4 above |
| defer SI3 dynamics/thermodynamics until ocean kt=1 chain is walked | ASKED | B4 NOT ENTERED initially |
| exact monthly interpolation weights and operation order | UNASKED fact | derived from resolved kt=1 clock and `fld_read`; gate against oracle |
| 30 prognostic levels rather than file dimension 31 | UNASKED fact | derived from NEMO `jpkm1` and bottom-level maximum; planted control |
| any unsupported selected mechanism | UNASKED implementation state | report UNMEASURED and stop before its first consumer; do not invent a substitute |

No NEMO run is preregistered for this round.  If a missing oracle field makes
one necessary, preparation of a self-contained run directory and launcher is
the stopping receipt; execution remains in the user's shell.
