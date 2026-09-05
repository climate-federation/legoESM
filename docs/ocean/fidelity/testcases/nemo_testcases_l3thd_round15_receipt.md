# SI3 lane 3b round 15 receipt — coupled-slab ZDF owner

Date: 2026-09-04

Tracker: `climate-federation/legoESM#1699`

Round-14 parent: `dad9dd304872649d66044d1db45e3e68b1e63f45`

Physics/gate commit: `792e2a1bc3a9bfad0b2d90507976fb5e5e177acd`

Status: **MOMENTUM/SSH PREFIX CONFIRMED; STOPPED AT TRACER DEBT**

## Outcome first

The former first continuous debt, `kt=2 PRE_SSM.u =
1.0844585561836217e-4`, is closed: `u`, `v`, `ssh`, and `e3t` are now each
**1/1 bit-identical** after one complete coupled ocean step.  The ordered walk
then stops, as required, at `kt=2 PRE_SSM.temperature`: absolute error
`2.267231727692831e-5 K`, normalized error `1.308615514721144e-5`, 0/1 bits.
Salinity is also red behind that first row (`7.624427225110253e-4`, normalized
`2.2424454879980685e-5`).  No later trajectory claim is made.

The ice/exchange prefix was independently re-scored by the committed gate:
**448,950/448,950 bit-identical**, including `POST_FZP.t_bo` 2,190/2,190.
The exchange gate's standalone `scope_complete=false` field still names the
ocean boundaries which its own module does not score; the round-14 ocean gate
and this receipt's round-15 continuation are the separate evidence for those
boundaries.

## Oracle provenance and immutability

The accepted new oracle is the full 8,760-step scalar-math run at:

`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r15c_oracle_a`

It was built from the config-local copy
`C1D_OMIP_L3_COUPLED10M_R15C_SM`; no shipped NEMO source or shipped config was
edited.  `ocean.output` reaches time step 8760 and the executable contains
zero `_ZGV*` symbols.

| item | SHA-256 |
|---|---|
| `arch/arch-conda-scalarmath.fcm` | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` |
| R15C `nemo.exe` | `6951e517d29fe22ce83768ee2b06f94e1aeb87466e590ec0a95daa58b2e5e299` |
| run `namelist_cfg` | `12abc86c56a1d5222ede1a8213ccbae1b783b0c7f512e4fa795e00502c726132` |
| run `namelist_ice_cfg` | `d4da9b215b1d762b7a776324ee7ce2ea6a2c0ee18452b434bc9cec65cc8fbce6` |
| `cpp_C1D_OMIP_L3_COUPLED10M_R15C_SM.fcm` | `d7c0c4659bf16f500b4658907f65e21bb03eb88df0b5e875e9a6727cbcc62384` |
| generated C1D chlorophyll weights | `715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc` |
| derived initial state | `0ded4378da5d454d1658a889190428cd19404e2e9fd8f27a6781d1c635e838fa` |
| `ocean.output` | `53bc133cb27f3391cf8813b5b39ec5534ab258b3e02a7d0f878e480d8fb27018` |
| exchange stream | `b97b85ef8381a06e6ac548084337684875a8a634486415075ab35d239945188a` |

Every pre-existing physical stream (drag, ZDF, FWB, QSR, SPG, SPG-statement,
SSM, stagger, STP2D, TRA-SBC, update, bulk operands, exchange, thermodynamics,
ZDF inputs and ZDF operands) and both final restarts compare byte-identical
between accepted R14F and R15C.  The new writers therefore did not perturb the
oracle trajectory.

## Rule 0, time levels, and the measured owner

The executing NEMO order is source-bound:

1. `dynzdf.F90:119-140` forms the QCO stage-3 RHS.  The registered
   `(Kbb,Kmm,Krhs,Kaa)=(1,2,3,3)` values show `u(Kbb)=0`,
   `u(Krhs)=1.0845180127162992e-8`, `v(Kbb)=0`, and
   `v(Krhs)=-7.439242071479682e-7`.  Replaying `:127-132` in literal order is
   1/1 bit-identical for both components.  This **CONFIRMS H15-C**: legoESM's
   former early depth-mean subtraction erased the entire nonzero one-layer
   `Krhs`.
2. `dynzdf.F90:148-171` strips `uu_b/vv_b(Kaa)` immediately before the
   implicit solve and adds the barotropic bottom/top drag terms.  The top
   coefficient is signed zero at kt=1, so top drag cannot own this event.
   Surface stress is added at `dynzdf.F90:326-345,500-519` after the strip.
3. `stprk3_stg.F90:430,433-445` returns from `dyn_zdf`, computes the column
   correction, and unconditionally replaces the 3-D mean by
   `uu_b/vv_b(Kaa)`.  NEMO exposes no switch for this RK3 operation.  Literal
   replay of the dumped correction and post-LBC values is 6/6 bit-identical.

legoESM now keeps the full WS-RK3 RHS through its shared stages
(`ocean_model_latlon_cgrid.py:4210-4279`), gives the un-reconciled stage-3
value to the shared implicit ZDF path (`:4699-4713,6138-6211`), performs the
NEMO strip/stress order (`:7990-8330`), and applies the one shared literal
mean-replacement kernel after the solve (`:6224-6297`).  No slab-specific
solver or second Thomas implementation was added.

The large owner is the missing mandatory post-ZDF mean imposition, with the
missing full `Krhs` lifetime as its upstream operand owner.  The private hook
`_NEMOWSRK3TestHooks.post_zdf_mean_scale` is not a public model selector and
does not construct a NEMO combination that does not exist.  Its measured
scaling is:

| correction scale | kt2 u absolute error | kt2 v absolute error |
|---:|---:|---:|
| 0 | `5.277866797459287e-3` | `1.2662969863419457e-3` |
| 0.5 | `2.638933398729643e-3` | `6.331484931709729e-4` |
| 1 | `0` (1/1 bits) | `0` (1/1 bits) |

H15-A is **REFUTED as the large owner**: faithful implicit surface-stress
placement is required by `dynzdf`, but the kt=1 top-drag coefficient is zero
and selecting stress placement alone did not close the carried momentum.

Three smaller source-association debts were also closed inside the same NEMO
identity: raw rather than pre-normalized `wgtbtp1` accumulation
(`dynspg_ts.F90:820-847`; the former sum first split at subcycle 316), the
metric-weighted SSH face-depth statements (`:751-758,884-891`; former frozen
v-face Kmm depth +1 ULP), and separately guarded post-loop divisions.  These
move last bits only; none is relabelled as the owner of the original
`1.084e-4` row.

### Frame registry (Rule 1d)

| stream | header/time level | values | SHA-256 |
|---|---|---:|---|
| `oracle_rung36_dynzdf_owner_frames.bin` | kt=1, Kbb=1, Kmm=2, Krhs=3, Kaa=3, stage=3 | 43 fp64 | `5b1ce184583976cf1db3512222bf767b12e269f03eaf4337e44b42afac9c39bf` |
| `oracle_rung36_dynzdf_rhs_frames.bin` | same | 13 fp64 | `ac64fa8c498f9adc49cbce58ae797f537c15da69aee1f713a8a7253853ac8c5a` |
| `oracle_rung36_stage_end_frames.bin` | same, after `stprk3_stg` velocity LBC | 18 fp64 | `35da6c38f8e310da252b664c7be1d03afd2aac6d28216bed4e7fc11e3fe09e78` |

All three headers are truthful and gate-validated: the writers emit
`SIZE(z)` and `STORAGE_SIZE(z(1))`, and the reader requires respectively
43/13/18 values and 64 bits.  The inherited assumed-shape correction is also
retained: work arrays use their exact `jpi,jpj` declarations and forcing
arrays use `A2D(0)`, so the one-column element mapping does not silently shift.
That correction affects the exploratory SPG operand register only; accepted
R14F and R15C use the corrected declarations, and the complete physical
stream comparison above is byte-identical.

## Gates, controls, and retained artifacts

Large runtime JSON remains outside git under
`/data/abyssal/dbalwada/nemo-testcases-l3/round15_results/`.

| artifact | result | SHA-256 |
|---|---|---|
| `exchange_gate.json` | 448,950/448,950 bits | `aef5fb7dda64e37981904ac232ca92b5ed32d877b6b2aac23e2e6a17f2e99ca7` |
| `owner_gate.json` | 6/6 owner rows bit-identical; normal exit 0 | `4d742049117c0726ff82657b794e445df4b5cdb38d6f1617a48b84fd3fadc76a` |
| `ocean_gate.json` | momentum/SSH 4/4 bits; STOP at temperature | `a1e2001a77feb30c5da4ed34ea96ccaf236bba7cf6527730f9a47fcb08e59905` |
| `owner_gate_plant.json` | `POST_ZDF_MEAN.u +1e-8`, red, exit 1 | `8bbdea4fbdb3cd56ee6a70bdc42b79f1f62008498e5ef3ee569c48f253a24141` |
| `ocean_gate_trajectory_plant.json` | kt2 u `+1e-8`, red, exit 1 | `063d3f2de57b3e631c120059041d587f326ecdf6137ff752fc7e17e5988ea0d0` |

The exchange gate now fails closed with named `GateError` before reading when
any required stream is missing; its focused unit control passes.

Tests: 20/20 focused rung-3.6 tests pass; 51/51 hardcoded-constant ratchets
selected for every touched file pass.  A broader relevant run had 113 passes
and two pre-existing `test_nemo_ab3am4_filter.py` construction failures
(`pgf_quadrature=nemo_trapezoid` paired with non-`nemo_sco`); they are not
touched or claimed fixed here.  The full constants ratchet likewise retains
five unrelated pre-existing failures outside this lane; the receipt does not
claim a clean full suite.

### FLAGGED FOR FUTURE DELETION (nothing deleted)

- The eight exploratory round-13 roots
  `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}`.
- `c1d_omip_l3_coupled10m_r15b_oracle_a` (failed before SAS input staging).
- `c1d_omip_l3_coupled10m_r15b_oracle_b` (invalid RHS diagnostic: `Kaa`
  aliased `Krhs` and the writer sampled after overwrite); superseded by R15C's
  pre-overwrite snapshot.
- Copied stale `cpp_*R15*.fcm` names in the R15B/R15C config directories.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh config-local operand writers and full-year scalar-math run | ASKED | completed; immutable R15C run pinned above |
| identify/fix kt2 momentum owner using shared canonical blocks | ASKED | completed; momentum/SSH prefix bit-identical |
| one-variable private arm, scaling, and row-level plants | ASKED | completed; table and exit codes above |
| continue to the next first-over-bar boundary | ASKED | stopped at kt2 temperature |
| promote the round-13 exchange prefix and fix missing-stream failure | ASKED | 448,950/448,950; named `GateError` |
| flag eight exploratory roots without deleting them | ASKED | listed above |
| implement or diagnose the new tracer debt in this dispatch | UNASKED | stopped for the required next decision |
| edit shipped NEMO, delete retained artifacts, add slab numerics, GPU, `mpirun`, push | UNASKED | not done |

## Terminal classification

**CONFIRMED:** the R15C oracle is a successful CPU-only 8,760-step
scalar-math run; all inherited physical streams and restarts are unchanged;
the QCO RHS and post-ZDF correction replays are bit-exact; the private scaling
binds; the repaired kt2 momentum/SSH rows are bit-identical; and the next
first-over-bar row is temperature.

**PLAUSIBLE, NOT OWNED:** the new temperature/salinity debt lies in the
coupled tracer RK3 / implicit-ZDF interval after the already certified
exchange operators and before kt2 SSM.  It needs a separately preregistered
tracer operand walk.  This round makes no owner or trajectory claim beyond
that measured stop.
