# SI3 lane 3b round 16 receipt — coupled tracer end step

Date: 2026-09-04

Tracker: `climate-federation/legoESM#1699`

Parent: `5cd6203aaad2c65eba82ec2de8b6ed86bb3e0991`

Preregister commit: `57899611a4e51ffc26b0def839bcce35c4bd8af2`

Carry-over commit: `d1950dc06a786813246359972847eb21ac1506eb`

Physics/gate commit: `5a298c65768af1e3bed8cbb83e7f865013128b00`

Header-gate commit: `c3488e7af8c9e8e0de1c9415e7d578f85fd5b9e3`

Status: **TRACER PREFIX CONFIRMED; STOPPED AT THE NEXT MOMENTUM DEBT**

## Outcome

The carried `kt=2 PRE_SSM` tracer defect is closed.  Temperature and salinity,
together with the already-closed velocity, SSH, and thickness rows, are now
**6/6 bit-identical** after one complete coupled step.  All ten registered NEMO
source statements inside the stage-1/2/3 surface-source and one-layer
`tra_zdf` chain are also bit-identical.

The ordered trajectory walk then stops at the next red boundary:
`kt=3 PRE_SSM.u`, absolute and normalized error
`1.7912434032934096e-3`, 0/1 bits.  The following `v` row is also red at
`1.8350774187287796e-3`; T, S, SSH, and e3t remain bit-identical there.  The
gate labels only the measured interval, `STEP2_MOMENTUM_END_TO_PRE_SSM`.
It does **not** claim a causal owner for that new discrepancy; resolving it
needs the next decision/operand walk.

## Oracle provenance and immutability

The new full-year CPU oracle is retained at:

`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r16_oracle_a`

It was built in the stable source copy as the new config
`C1D_OMIP_L3_COUPLED10M_R16_SM`.  The shipped NEMO tree and shipped configs
were not edited.  `time.step` is 8760; `nemo.exe` has zero dynamic `_ZGV*`
symbols.  The build uses `arch-conda-scalarmath.fcm`, whose Fortran flags add
`-fno-tree-vectorize` to the established conda architecture.

| item | SHA-256 |
|---|---|
| scalar-math arch | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` |
| R16 `nemo.exe` | `265fe0291eb3d0f94f8f88177352f2636ab5565d16d2730fa48f40df225afeba` |
| run `namelist_cfg` | `12abc86c56a1d5222ede1a8213ccbae1b783b0c7f512e4fa795e00502c726132` |
| run `namelist_ice_cfg` | `d4da9b215b1d762b7a776324ee7ce2ea6a2c0ee18452b434bc9cec65cc8fbce6` |
| config cpp keys | `d7c0c4659bf16f500b4658907f65e21bb03eb88df0b5e875e9a6727cbcc62384` |
| generated C1D chlorophyll weights | `715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc` |
| derived initial state | `0ded4378da5d454d1658a889190428cd19404e2e9fd8f27a6781d1c635e838fa` |
| `ocean.output` | `53bc133cb27f3391cf8813b5b39ec5534ab258b3e02a7d0f878e480d8fb27018` |
| ocean restart | `45aecc2778fdf786d082363fcc869d63a99f810a801939c7c47903ce72b7b16c` |
| ice restart | `c756f3e2117ad84ba73fa17a8b00470aba1b6824f504f86f8159a23777288b79` |

The input-file ledger is unchanged from the round-13 receipt: the sanctioned
NEMO-WEIGHTS bilinear file and derived initial state above are byte-identical,
and all other inputs remain the hash-pinned, read-only ORCA1/C1D files there.
No forcing or interpolation was generated in this round.

Nineteen inherited physical streams compare byte-for-byte with the accepted
R15C oracle: drag, both prior dyn-ZDF owner streams, FWB, QSR, SPG,
SPG-statement, SSM, stage-end, stagger, STP2D, TRA-SBC, update, bulk operands,
exchange, ice thermodynamics, SI3 ZDF input/operand, and coupled ZDF.  Selected
stable hashes are:

| stream | SHA-256 |
|---|---|
| exchange | `b97b85ef8381a06e6ac548084337684875a8a634486415075ab35d239945188a` |
| ice thermodynamics | `8e6ea3881d2f08b646f3d0e407749b781d5fa8c50f36821f0a480983d355fb54` |
| SI3 ZDF inputs | `88a5f24ad9c823963b99f1ad76c7cdff1e73bd231ef83efdfe5804268b2abf5b` |
| bulk operands | `bcce3ffc423e958a8d995f1b890db98ccba2f5863da811e3e7d33d98f05c800a` |

This byte comparison re-measures the rows affected by the inherited
assumed-shape lower-bound instrument correction.  It remains an instrument
correction, not a physics change.

## Rule 0 and first divergence

The resolved oracle prints `ln_traadv_OFF=T`, `ln_traqsr=T`,
`ln_trabbc=F`, and `ln_drgice_imp=T` (`ocean.output:1264,556,1242,1125`).
NEMO clears and fills each tracer `Krhs` by calling `tra_adv` and then
`tra_sbc_RK3` (`stprk3_stg.F90:519-521`).  In stages 1 and 2, the executing
surface statements form `r1_rho0/e3t(Kmm)` and subtract
`emp*T_or_S(Kbb)*scale` (`trasbc.F90:282-290`).  Stage 3 adds non-solar heat
and PSS salt in two separate statements (`trasbc.F90:294-315`), then the
selected RGB-with-chlorophyll solar operator runs before LDF and ZDF
(`stprk3_stg.F90:568-598`; `traqsr.F90:172-176`).

The source-level register shows:

| boundary | bit rows | result |
|---|---:|---|
| stage 1/2 post-TRA-SBC T/S RHS | 4/4 | exact |
| stage 3 post-QSR T/S RHS | 2/2 | exact |
| `tra_zdf` content RHS and terminal solve T/S | 4/4 | exact |

The one-layer ZDF identity is still nontrivial: NEMO forms
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at
`trazdf.F90:271-278`, then divides the terminal row by its after-level
diagonal at `:281-286`.  The QCO stage statements likewise distinguish the
old, RHS, and after time levels (`stprk3_stg.F90:540-559`).

Two source-identity defects owned the carried row:

The first pre-fix non-bit operand in NEMO order was the stage-1 Kbb layer
thickness: NEMO's `e3t0*(1+r3t)` produced
`8.333333333333332` (`0x4020aaaaaaaaaaaa`), while generic `H+ssh` produced
`8.333333333333334` (`0x4020aaaaaaaaaaab`).  This is the scaling-before-owner
discriminator: the split exists before any tracer source is accumulated.

1. the shared stage helper also multiplied the RHS by Kaa thickness; NEMO uses
   Kmm for that product and Kaa only as the final divisor;
2. its Kbb/Kaa thicknesses came from generic `H+ssh`.  The executing QCO
   source instead constructs `r3t=ssh*r1_ht_0` (`domqco.F90:159-160`) and
   uses `e3t0*(1+r3t)` in the stage content expression.  The slab now calls
   the already-shared `compute_nemo_qco_layer_thickness` for both levels.

The implementation preserves raw `emp`, `qns`, `qsr`, and PSS `sfx` in the
existing exchange forcing object, routes them through one promoted shared
`nemo_tra_sbc_rk3_source`, and consumes them inside the shared WS-RK3 tracer
program.  It does not add a slab solver.  Conventional heat/freshwater/salt
sources are withheld only when that raw NEMO bundle is present, preventing a
duplicate source.  The C1D card also now names its actual resolved OFF tracer
advection and real-volume freshwater arms rather than inheriting unrelated
GYRE defaults.

The private `tracer_surface_source_scale=0` arm moves kt2 temperature by
`4.247252062285334e-2 K` and salinity by
`2.3868025802187276e-4`; u, v, SSH, and e3t do not move.  This confirms the
source path is causal and the control is not vacuous.  The hook is absent
from all public configuration records.

## Drag source association

The carried association caveat is transcribed literally in the shared path.
NEMO constructs the barotropic coefficient as
`r1_2*((bot_east+bot)+(top_east+top))` and its v analogue in single source
statements (`dynspg_ts.F90:1611-1612`); the implicit top diagonal uses the
unhalved neighbour sum with `zDt_2` (`dynzdf.F90:302,478`).  The shared helper
now preserves those groupings with `nemo_source_round` per statement.

This correction is measured inert for this one-column year: top drag is zero
at kt1, first becomes nonzero at kt5, and the former separate-halves versus
literal-combined barotropic expression differs at **0/8,760** steps.  It is
therefore not relabelled as a tracer or momentum owner.

## Frame registry, gates, and controls

| stream/artifact | registered level/result | SHA-256 |
|---|---|---|
| `oracle_rung36_tracer_owner_frames.bin` | kt1; stages 1/2/3; explicit Kbb/Kmm/Krhs/Kaa; 18 derived fp64 values/record | `97630ac5b61761353914d3a78d56af0e27062cf2e254138967bf7f9348e8069e` |
| `oracle_rung36_trazdf_owner_frames.bin` | kt1 stage 3; Kbb=1, Kmm=2, Krhs=3, Kaa=3; 14 derived fp64 values/record | `c34151d211f151e66b1a5463aba2de4cdb0c053442df2ad3c553695449eca495` |
| instrumented `stprk3_stg.F90` | config-local WRITE-only source | `67a8f49d2b52ac79054d791807e62416479a24fa81e224a8c80212fa10f593ce` |
| instrumented `trazdf.F90` | config-local WRITE-only source | `4540379b18a7313d4766ac06dcb9ec16f3aa26dd1cf61468bf765397a885c048` |
| normal gate | 10/10 source bits; kt2 6/6 bits; stops at kt3 u; exit 1 | `d43e4cbf504253f88226e9686cbf5e4686829d761d0ce0a74e2c8d1572456005` |
| `+1e-8` source-row plant | `kt1.stage1.Krhs_T` red; exit 1 | `a2bc1d060223c3ce5759c478f72d6bb6a4ed1c7b3c787da4f5a11890b9e4dc34` |
| zero-source private arm | kt2 T red by `4.247e-2`; exit 1 | `6471049d29c8a311590660952ac69b7d1264a76240ccaca05e488ba68355938d` |

All binary counts are derived from the write arrays with `SIZE`; the readers
require 64-bit values and reject a false count.  The repository-wide stream
header gate returns `VALID` on the new root, and the round-16 unit plant also
rejects an untrue derived count.  Runtime JSON and binaries stay under the
hash-bound `/data` root; none is committed.

The round-13 exchange prefix was re-scored on both the accepted R13 `_i` root
and R16: **448,950/448,950 bit-identical** on each, with JSON SHA-256
`aef5fb7dda64e37981904ac232ca92b5ed32d877b6b2aac23e2e6a17f2e99ca7`.
The exchange reader raises named `GateError` for a missing stream; its focused
unit test passes.

Focused tests: **25 passed** (rounds 13, 15, and 16).  The hard-coded-constant
ratchet line for every touched Python file is **55 passed, 3,345 deselected**.
The full ratchet was also run: 3,418 passed, 2 skipped, with five known
unrelated failures in FV3 coupling and older DINO tests; no clean full-suite
claim is made.

The requested OVERFLOW/LOCK/GYRE cross-card register claim remains
**UNMEASURED**, not silently passed.  The retained OVERFLOW/LOCK trajectory
entry points fail during pre-measurement construction on their existing
`momentum_advection='off'` plus `vertical_momentum_scheme='off'` validation;
the GYRE recipe run reaches 22 passes but its remaining five constructions
fail the existing `nemo_trapezoid`/non-`nemo_sco` pairing check.  Those checks
and configurations are unchanged by this round.  The phase-3 EOS controls
pass 3/3.  Because the shared QCO-thickness path cannot be exercised by those
blocked cards, this receipt does not claim Rule-12 non-regression for them.

## FLAGGED FOR FUTURE DELETION (nothing deleted)

- The eight unpinned round-13 roots
  `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}`; `_i` is the
  accepted retained root.
- The unused stable-source root-level `cfgs/BLD`, `cfgs/MY_SRC`, and
  `cfgs/EXP00` directories created by the first malformed `makenemo` command.
- Two failed R16 build attempts: one lacked the intended build environment;
  the second proved that `kt` is not in `trazdf` scope.  The accepted writer
  obtains the step from its config-local diagnostic module.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh config-local WRITE-only tracer operands and full-year CPU oracle | ASKED | completed and pinned |
| source-order bisection, private one-variable arm, shared identity fix | ASKED | completed; kt2 T/S exact |
| literal drag grouping and later nonzero-step measurement | ASKED | completed; inert at 0/8,760 changed steps |
| continue the ordered year walk | ASKED | stopped at the next red boundary, kt3 u |
| promote round-13 448,950/448,950 and named missing-stream error | ASKED | completed in the carry-over commit and remeasured here |
| flag unpinned roots; delete nothing | ASKED | listed above |
| diagnose or fix the new kt3 momentum interval | UNASKED | requires the next decision; not attempted |
| modify shipped NEMO, add a slab-specific solver, GPU, `mpirun`, push | UNASKED | not done |

## Terminal classification

**CONFIRMED:** successful 8,760-step scalar-math CPU oracle; zero `_ZGV*`;
unchanged physical streams and restarts; truthful headers; all ten internal
source rows and all six kt2 PRE_SSM rows bit-identical; causal private arm and
red plant; 448,950/448,950 exchange rows on both pinned roots; and the next
first-over-bar row at kt3 u.

**PLAUSIBLE, NOT OWNED:** the new kt3 u/v discrepancy is created somewhere in
the nonzero-state step-2 momentum/end-step interval.  This round makes no
claim about which operand owns it and no trajectory claim past that measured
stop.  No external review artifact for round 16 exists on this branch; any
checks performed here are author/Codex-internal and are not represented as
independent review.
