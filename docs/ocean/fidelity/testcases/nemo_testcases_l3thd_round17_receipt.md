# SI3 lane 3b round 17 receipt — second-step momentum closure

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `4c3550e411456d6d348b372250bf1bed8eae7020`

Preregister commit: `a68f5d57941`

Physics and gate commit: `3d5ebfedebc8b8958dfe8e6851fd8902b4430a0c`

Status at Round 17: **SUPERSEDED BY ROUND 19**.  The former Kmm-seed arm
re-derived NEMO's prognostic `uu_b/vv_b` state and is not a valid source
identity.  After the required removal, Round 19 measures the first debt at
kt2 PRE_SSM.u (`1.1172865415493005e-7`).  The historical measurements below
are retained as an arm record, not as a current fidelity claim.

## Outcome

The first non-bit-exact operand in the kt2-to-kt3 momentum chain was the
split-explicit integration seed.  NEMO's live `ln_bt_fw` branch copies
`pssh/puu_b/pvv_b(Kmm)` into the external-mode carry
(`dynspg_ts.F90:484-493`).  The shared legoESM path instead let the ordinary
post-stage workspace seed the barotropic solver.  That u/v workspace was
`4.467046238296415e-3` / `-2.491851118505196e-3`, while the NEMO u/v seed was
`5.128977951775159e-3` / `7.477198513071789e-5 m s-1` (the v Kmm replay is
within `5.42e-20`, not bit-identical).  The correction threads the already
shared `eta_init/u_init/v_init` operands; it adds no new solver or slab arm.

A second, independent construction defect remained after that fix.  The slab
card represented the resolved nonlinear bottom drag by its zero-speed linear
value, `5e-5 m s-1`.  The actual oracle selects `ln_non_lin=T`, `rn_Cd0=1e-3`,
and `rn_ke0=2.5e-3` (`ocean.output:1122,1128,1131`), so it executes
`-Cd0*SQRT(.25*(zut*zut+zvt*zvt)+ke0)`
(`zdfdrg.F90:183-190`).  These are the ORCA1 deck values at
`namelist_cfg:258,264-268`.  The card now selects the existing shared
`nemo_quadratic` implementation.

Together the source-identity corrections make kt3 PRE_SSM u, v, T, S, SSH,
and e3t **6/6 bit-identical**.  The 1e-15 normalized walk then covers every
remaining transition through kt8760.  Its `first_over_bar` is null: all
52,554 rows are AT_BAR.  This is deliberately not called a bit-exact year;
34,526/52,554 rows are bit-identical and the below-bar counts are reported
below.

## Preregistered bisection and causal arms

The fresh statement stream registers kt1 and kt2, 946 split-explicit cycles
each.  Replaying NEMO's source-literal velocity statement through the shared
helper gives 1,892/1,892 u and 1,892/1,892 v bits.  The first mismatch was
therefore its entry operand, before the first kt2 substep, confirming ranked
hypothesis 1; the ENE, ZDF, SSM, and OFF-advection/LDF hypotheses were refuted
as first owners.

| arm / state | kt3 u absolute error | kt3 v absolute error | result |
|---|---:|---:|---|
| pre-round-17 combined construction | `1.7912434032934096e-3` | `1.8350774187287796e-3` | DEBT |
| correct quadratic drag, old seed (`weight=0`) | `1.7918228959916214e-3` | `1.8345519569816266e-3` | DEBT |
| seed `weight=0.5` | `8.959114479958237e-4` | `9.172759784908135e-4` | DEBT; linear scaling |
| correct Kmm seed, linearized-drag ablation | `7.912963721230099e-7` | `3.2662598687185906e-7` | DEBT |
| production Kmm seed + quadratic drag | `0` (1/1 bit) | `0` (1/1 bit) | AT_BAR |

Both controls live only in `_NEMOWSRK3TestHooks`; neither is a public selector.
NEMO has no switch for starting the forward RK3 external mode from a different
time level, so the production Kmm route is unbranched.  Its signed u/v operand
gap was `-6.619317134787441e-4` / `-2.566623103635914e-3`.  The drag arm recreates
the superseded construction consistently at all shared consumers.

The row plant adds `1e-8` to kt2 cycle-1 u immediately after the registered
barotropic source statement.  It makes exactly one of 1,892 u rows non-bit,
reports `9.999999999940612e-9`, labels the planted row, and exits 1.  The normal
gate exits 0.

## Continuous-year measurement

Per-field totals and maxima over kt2..kt8760:

| PRE_SSM field | bit rows | non-bit rows | max absolute error | max normalized error | status |
|---|---:|---:|---:|---:|---|
| u | 458/8,759 | 8,301 | `4.683753385137379e-17` | same | AT_BAR |
| v | 796/8,759 | 7,963 | `4.553649124439119e-17` | same | AT_BAR |
| temperature | 8,758/8,759 | 1 | `2.220446049250313e-16` | `1.477207281063869e-16` | AT_BAR |
| salinity | 8,759/8,759 | 0 | `0` | `0` | bit-identical |
| SSH | 8,759/8,759 | 0 | `0` | `0` | bit-identical |
| e3t | 6,996/8,759 | 1,763 | `1.776356839400251e-15` | `2.220223823994237e-16` | AT_BAR |

The sampled absolute-error growth table is:

| kt | u | v | T | S | SSH | e3t |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | `1.735e-18` | 0 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100 | `1.640e-18` | `1.019e-17` | 0 | 0 | 0 | 0 |
| 1,000 | `5.294e-22` | `1.059e-22` | 0 | 0 | 0 | 0 |
| 3,000 | `6.300e-21` | `1.789e-20` | 0 | 0 | 0 | `8.882e-16` |
| 5,000 | `1.204e-21` | `2.118e-22` | 0 | 0 | 0 | 0 |
| 8,760 | `1.165e-21` | `1.429e-20` | 0 | 0 | 0 | 0 |

There is consequently no later owner requiring a decision in this dispatch.
The measured terminal classification is AT_BAR with sub-bar roundoff, not
trajectory bit identity.

## Drag-association carry-over

The shared face helper already carries the requested literal association:
NEMO halves the combined bottom-plus-top neighbour sum once at
`dynspg_ts.F90:1611-1612`; the implicit ZDF top term is
`zDt_2*(top_east+top)/e3u` at `dynzdf.F90:302` (v at `:475-476`).  Those replace
the stale `:1607-1610` and `:303-306` citations.

The round-17 replay measures the first nonzero ice-top drag at kt5 and the
literal combined u coefficient there at 1/1 bit.  Comparing the superseded
separate-half association with the literal association moves **0/8,760**
steps on this input.  The correction is therefore source-literal but inert for
this column; it is not credited as an owner.

## Oracle provenance, instruments, and immutability

The accepted full-year CPU run is:

`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r17_oracle_b`

It is a new config-local copy, `C1D_OMIP_L3_COUPLED10M_R17_SM`, under the stable
source tree.  Nothing in the shipped NEMO tree or any retained oracle root was
modified.  It ran 8,760 one-hour steps without `mpirun`; `nm -D nemo.exe`
contains zero `_ZGV*` symbols.

| item | SHA-256 |
|---|---|
| `arch-conda-scalarmath.fcm` (`-fno-tree-vectorize`) | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` |
| R17 `nemo.exe` | `50cef4f60fa138f4aedd921233f3180aab32731f2a9badc89e4b72d5026ac052` |
| config-local `MY_SRC/dynspg_ts.F90` | `8aab1ad7a372c027ee88a50674933550d839c845130eab9baeb6cc552e86bfff` |
| config-local `MY_SRC/dynzdf.F90` | `d868a71306036d9898a26174a370ba18c75f82db0c08dde66e25dc141bd73073` |
| config-local `MY_SRC/stprk3_stg.F90` | `798b789085cda9b183ef437d22d6fd858934a01e67ebca9e0f996c7b763d12af` |
| `ocean.output` | `53bc133cb27f3391cf8813b5b39ec5534ab258b3e02a7d0f878e480d8fb27018` |
| ocean restart | `45aecc2778fdf786d082363fcc869d63a99f810a801939c7c47903ce72b7b16c` |
| ice restart | `c756f3e2117ad84ba73fa17a8b00470aba1b6824f504f86f8159a23777288b79` |

All inherited physical streams and both restarts are byte-identical to R16.
Selected stream hashes are exchange
`b97b85ef8381a06e6ac548084337684875a8a634486415075ab35d239945188a`,
bulk `bcce3ffc423e958a8d995f1b890db98ccba2f5863da811e3e7d33d98f05c800a`,
thermodynamics
`8e6ea3881d2f08b646f3d0e407749b781d5fa8c50f36821f0a480983d355fb54`,
and SI3 ZDF inputs
`88a5f24ad9c823963b99f1ad76c7cdff1e73bd231ef83efdfe5804268b2abf5b`.
No forcing, weights, or interpolation were generated in this round; the input
ledger remains the hash-pinned round-13 ledger.

The fresh WRITE-only instruments merely extend the previously accepted writer
conditions through kt2.  Their headers retain counts derived from their write
arrays.  The repository stream-header gate reports VALID on every registered
full-year stream, and the focused schema plant rejects a false derived count.
The earlier assumed-shape lower-bound instrument correction is retained; its
448,950 exchange rows were remeasured below, so no result rests on the
superseded indexing.

| artifact | registry | SHA-256 |
|---|---|---|
| kt1-2 SPG statements | `(kt,Kbb,Kmm,Krhs,Kaa,jn,nn_e,nvalue,bits)`; 1,892 records | `7837148debe4ebc95c7c8daed78486c0787f673d2f0c70f3d079b2ad4d87cb32` |
| dyn-ZDF owner operands | explicit kt/stage/time levels; derived values | `2387c7f2b08bfe9214b46807929d16229bd23690fbfc56b0b2d55de912a6d947` |
| dyn-ZDF RHS operands | explicit kt/stage/time levels; derived values | `5d25a2e87789aaf98315224d61423fb9b0d2782457c713c887a13190ff8f382a` |
| RK3 stage-end operands | explicit kt/stage/time levels; derived values | `d311ad2326ee8cf5ecc785a65016e59690b469956737c5ccc1fc8bf0ad435e10` |
| normal round-17 gate | source statements + compact all-year summary | `7795bfdaf37a45ebd6545f8c275ddaa0115f24c2c3c550387662c0a7de733f41` |
| planted gate | named red row, exit 1 | `7b6c6b40f74688452cccd29f7c4d6d4d2c066b366c576a414a99b486411a4e88` |

Runtime streams and JSON remain under `/data`; none is committed.

## Repeated carry-overs and Rule 12

The round-13 prefix is now stated plainly in its own receipt and remeasured
again here.  On both accepted R13 `_i` and R17 roots, the exchange gate gives
**448,950/448,950 bit-identical**, including POST_FZP.t_bo **2,190/2,190**;
both reports have SHA-256
`aef5fb7dda64e37981904ac232ca92b5ed32d877b6b2aac23e2e6a17f2e99ca7`.
The missing-stream unit control raises the named `GateError`, not
`UnboundLocalError`.

| shared change | boundary | this-card measurement | Rule-12 disposition |
|---|---|---|---|
| forward RK3 Kmm external-mode seed | dynspg entry / SSH substeps | kt3 exact; year AT_BAR | shared change from this lane; other cards require merge verification |
| literal combined top+bottom drag | PRE_DYN_SPG_TS / PRE_DYN_ZDF_SOLVE | 0/8,760 association moves; kt5 1/1 bit | already shared; citation corrected |
| round-15 ZDF input ordering | PRE_DYN_ZDF_SOLVE | inherited kt2 closure retained | **GYRE lane is owner-of-record**; merge verification required |

The OVERFLOW/LOCK/GYRE registers are **UNMEASURED on this branch**, for a
specific fail-closed reason.  Shared `_model_config` selects horizontal
`momentum_advection='flux_form'` with `momentum_flux_scheme='upwind3'` but
`vertical_momentum_scheme='off'`.  The constructor at
`ocean_model_latlon_cgrid.py:2447-2455` correctly rejects this half-OFF
`ln_dynadv` program.  The focused cross-card file gives 12 passes and three
failures: two assert the canonical `nemo_up3` selector that this branch lacks,
and construction raises the named selector-pair error.  This is a branch
merge-base contradiction introduced before round 17, not evidence that a
gate ran and passed.  Validation was not weakened and the selector was not
silently changed; GYRE/OVERFLOW/LOCK merge verification must run the round-15
ZDF ordering and this Kmm-seed shared change.

Focused rung-3.6 tests: **34 passed**.  The hard-coded-constant ratchet line
for all four touched Python files is exactly **4 passed, 3,397 deselected**.
The final log SHA-256 values are respectively
`b26d063f91070dc86ed5b133d87ed77d8d155b045bc348fe5e3178968407886c`
and `624d6cb032f71aaca2c1a2ace1b695090d72c486873dc50fc87be837a03722ba`.
The expected-blocker log is
`996908035ef2718fd397c6ca0fec1f690525d7807ce3a606c1685802013b251c`.

## FLAGGED FOR FUTURE DELETION (nothing deleted)

- The eight unpinned roots
  `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}`; `_i` remains
  the accepted R13 root.
- Failed `c1d_omip_l3_coupled10m_r17_oracle_a`, whose first attempt omitted
  the required `SAS/` forcing link.  It exited 123 and is not an oracle.
- All earlier iteration roots already flagged by rounds 13-16.  Nothing was
  deleted or overwritten.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh kt2 config-local WRITE-only run and ranked owner walk | ASKED | completed; accepted root `_b` |
| private scaling/ablation arms and row-level plant | ASKED | completed; all bind |
| source-identity fix in the shared path | ASKED | Kmm seed threaded through the existing solver; no fork |
| resolve the card's actual quadratic drag | ASKED | completed from resolved output and ORCA1 deck |
| confirm kt3 and continue through the year | ASKED | kt3 6/6 bits; no later over-bar row through kt8760 |
| promote 448,950/448,950, GateError, stale citations, and nonzero drag row | ASKED | all remeasured and explicit above |
| weaken selector validation or silently repair other lanes' recipes | UNASKED | not done; merge verification registered |
| modify shipped NEMO, delete roots, commit runtime output, GPU, `mpirun`, push | UNASKED | not done |

## Terminal classification

**CONFIRMED:** the scalar-math oracle provenance and zero `_ZGV*`; truthful
headers; first differing seed operand; both causal arms; kt3 6/6 bit identity;
3,784/3,784 source-statement bits; 52,554/52,554 continuous rows AT_BAR through
kt8760; binding row plant; 448,950/448,950 exchange rows on both roots; and the
exact cross-card validation blocker.

**PLAUSIBLE but not promoted:** none of the unexecuted ranked operators owns
the former kt3 error; their refutation is boundary-based, not an independent
operator proof on another geometry.  Cross-card non-regression remains
UNMEASURED pending merge verification.  No external review artifact for round
17 exists on this branch; all review/check activity here is Codex-internal.
