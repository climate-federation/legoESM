# NEMO testcase Lane 4 — ORCA2 Phase-2j handoff receipt

Date: 2026-09-06

Parent: `110f7521ca2cbaf74fe18ba85d74dbaa4a583008`

Status: **STOP FOR USER-SHELL WZV ACQUISITION.**  The icebergs-off oracle is
now pinned as reproducible `VARIANT_ORACLE_V2`; RGB, EOS-80, SCO HPG and the
source-adjacent stage-1 horizontal transports are exact.  The first over-bar
boundary remains the frozen barotropic EEN coefficient program,
`GYRE_OWNER_SHARED_EXTERNAL_MODE`.  Lane 4 did not change that shared
operator.  SI3 remains `UNMEASURED_PENDING_ICE_MERGE` and no SI3 or iceberg
operator was entered.

## 1. Reproducible VARIANT V2 pin

The accepted root is:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2i_rhsrank0_a_10step_np2`

Its independent twin is the adjacent `_b_` root.  Both ran the same
scalar-math binary and icebergs-off deck to step 10 with zero MPI status.  The
campaign rule is now explicit: **an instrumented record set is oracle evidence
only after two independent executions produce the complete frozen inventory
byte-for-byte**.  The streaming gate reports 92 / 92 exact; the complete
record-manifest digest is
`2d1a69e430d8cf3fca0c6f642578c7f32ee6dd0c6f6823fb1d9abeced3d11f8e`.
Its one-byte `oracle_rhs` plant goes through the same twin validator, exits 1,
and yields 91 / 92.

The accepted acquisition gate schema-walks the inherited 90 streams plus
`oracle_ocean_surface_input` and the two-frame O1 stream.  Ordinary outputs
remain identical to the uninstrumented icebergs-off control: four restart
shards and eight history payloads are counted dynamically.  The inherited
Phase-1 gate on the 90-stream view passes all ten plants; the O1 gate passes
its three plants.  The pinned terminal artifacts are:

| artifact | SHA-256 |
|---|---|
| ocean restart rank 0 | `28fbf31286d90f31e6f72083b942a322a4d2cbfb18db6bf4983d76bae82c7280` |
| ocean restart rank 1 | `4f7873be26a96f9694470e5e7b67b93e7956180045effe948e2d3080eca4a326` |
| SI3 restart rank 0 | `4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2` |
| SI3 restart rank 1 | `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229` |
| `ocean.output` | `7590d8ff2afc830e73baa3238e98a87b7be44235179ce3f0c008ebab0aa7dea6` |

Every earlier variant, pre-canonical, and canonical-candidate root remains on
disk and is labelled superseded in the V2 pin receipt.  Twin B is retained as
the reproducibility witness, not a competing oracle.

## 2. V2 rescore

All gates ran production JIT on CPU with explicit fp64 and scalar-libm.  They
score raw binary64 equality on the rank-zero two-halo-stripped, source-defined
wet cells; every listed plant changes one live oracle value by one ULP and
exits 1.

| boundary | V2 result | record SHA-256 | plant stdout SHA-256 |
|---|---:|---|---|
| RGB `tra_qsr` increment | 0 / 233,341 | `oracle_rgb_chl`: `ddb14cf01c9483b107544e239784ef7bca25e29d7b27ffdc51cdb18e36f6da59` | `e5cffa2a0c66073742c9bbc0024ad0b106103a70d1e1b513b410ef3309df59bd` |
| EOS-80 source walk | 13 fields, each 0 / 233,341 | `oracle_rkstage2_eos_operands`: `ac18a72fcd77ba224cab3320120642bf127f03582de6c0dd4b6025ab93d28a72` | `dd0aeb9bcfe793b509dde016b9cecd7803b923b246a36df39b03bba467e9d3ab` |
| SCO HPG source walk | 3 U fields each 0 / 221,640; 3 V fields each 0 / 222,048 | literal `cf7189a7773c86d058a8bd9ecfa7abf6cd820b45ad4a1ccdfd7df9e146a7d3ac`; operands `2e2891e04ed3d6edd035aba55759434d2f713d1f9b3f5dd2bce23ace0c0b1c4f` | `b9f033acb338fbe6fe0799b8e4c3da0c42318af8393d3352420f014a6a288db0` |
| stage-1 `zFu` | 0 / 226,236 | transport operands `365299bc7ce66ee0ea8b3436e16fd0c136bdbc3d1d02a07ee56c29921106de45` | `c7e576675f6f78a0bdd4029f9dadd94d657c7cef6c3f7574bdabcee281ecc113` |
| stage-1 `zFv` | 0 / 226,637 | same | same binding gate |

The horizontal-transport test is an operand substitution, not a whole-step
claim.  `un_adv`, `vn_adv`, `r1_hu/r1_hv(Kmm)` and `uu_b/vv_b(Kmm)` come
straight from the V2 record and are labelled
`ORACLE_SUPPLIED_EXTERNAL_MODE`.  It executes the shared production
`_nemo_metric_stage_transport` helper; no ORCA2 transport fork was added.

## 3. Precise shared EEN handoff

### NEMO source program and record semantics

ORCA2 resolves `ln_dynvor_een=T`, `ln_dynvor_ene=F` and therefore
`nvor_scheme=np_EEN=3`.  At stage entry `dynspg_ts.F90:294-306` installs the
slow forcing and calls `dyn_cor_2D_init(Kmm)`.  The selected program:

1. zeros all eight frozen coefficient fields at `:1514-1518`;
2. forms the four three-vertex `ff_f/e3f_vor` sums for U at `:1520-1533`,
   accumulates the source-associated `e3u*e3v*mask*zpvo` products at
   `:1535-1538`, and applies `r1_12*r1_e1u*r1_hu*e1v` at `:1540-1543`;
3. repeats the distinct V-point neighbor program at `:1547-1565` and applies
   its four final scale statements at `:1567-1570`; and
4. consumes the frozen fields in the paired product/addition order at
   `dyn_cor_2D:1685-1694`.

The V2 evidence is
`oracle_bt_ene_coeff_kt00000001.bin`, SHA-256
`e7282ddc105300f8ee101d00ba9859eeed487a9c480947ae55fd5aac9f2c9363`.
Its validated header is magic `NEMO_L2_ENECO_1`, version 1, `kt=1`, `Kmm=1`,
`nvor_scheme=3`, `(jpi,jpj)=(94,152)`, binary64.  The payload is eight
A2D(0) rank-zero owned arrays, in order `ffu_nw,ffu_ne,ffu_sw,ffu_se,
ffv_nw,ffv_ne,ffv_sw,ffv_se`, emitted immediately after
`dyn_cor_2D_init` and before the first consumer.

Every field is over bar: U fields differ in 8,304--8,331 of 8,568 live cells;
V fields differ in 8,285--8,320 of 8,589.  Maximum absolute differences span
`6.983696920301593e-05` to `8.320420480597231e-05 s-1`.  The binding plant
starts from oracle/oracle equality, changes one live `ffu_nw` value, exits 1
at 1 / 8,568, and has stdout SHA-256
`e84dbaf2745f5ffb8a75661eb6ffd646731d0e67d9ad11e30f0ef8e6c8168ebe`.

### Shared legoESM boundary

The differing statements are in shared
`barotropic_latlon_cgrid.py:_nemo_literal_een_coefficients`:

- `:884-921` builds live U/V/F thickness and `q=ff_f/e3f`;
- `:923-937` shifts, forms the EEN triads, accumulates vertically and scales;
- `:960-972` selects the EEN U/V triads; and
- `:1001-1035` maps neighbors and returns the eight corners.

The consumer at `:1146-1184` already mirrors the four product/pair sums; this
measurement stops before attributing any recurrence error to it.  The ORCA2
card bridge currently carries domain-file `e3f_0` directly at
`nemo_testcase_recipe.py:1001-1017,1039-1058`, while NEMO's EEN divisor is
`e3f_vor`: `dynvor.F90:918-950` derives frozen `e3f_0vor` from four masked T
thicknesses, performs the F-point halo/fold exchange, and only then falls back
to `e3f_0` where zero.  A separate shared helper already names this distinction
at `vertical.py:197-257`, but its north operator is explicitly the closed-box
GYRE form.  **PLAUSIBLE, not confirmed:** feeding raw `e3f_0` instead of the
tripolar `e3f_0vor` program is the leading owner.  The required GYRE-lane
one-variable arm is to replace only the EEN divisor construction while holding
`ff_f`, U/V live thicknesses, masks, metrics and source association fixed, then
rescore all eight fields against the V2 record.

This is `GYRE_OWNER_SHARED_EXTERNAL_MODE`.  The GYRE oracle selected ENE, and
its certified shared walk exercised the `scheme="ene"` branch; it never walked
ORCA2's 12-point `scheme="een"` branch.  Lane 4 made no edit to shared
external-mode code and makes no north-fold claim from this pre-recurrence
record.

## 4. Continued ladder and new acquisition

With external results supplied from the oracle, the source-adjacent horizontal
products pass exactly.  The next executed boundary is vector-form
`tra_adv_trp`: `traadv.F90:138` activates W transport, `:201-228` establishes
the bottom condition and calls `wzv(np_transport)`, and `:241-243` forms
`pFw=e1e2t*ww`.  The V2 canonical streams deliberately zero halo bands, so
they omit the one-cell `pFu/pFv` support needed to recompute rank-zero
`div_hor`.  Inferring WZV or FCT from them would be a Frankenstein comparison.

One configuration-copy WRITE-only stream is therefore prepared around the
stage-1 WZV call.  It records the exact source-defined transport stencil,
Kmm/QCO thickness and stretch operands, surface runoff, `ww`, and `pFw` under
an `lwp` guard.  The block assigns no model field.  A first compile failed
because a substituted whole-array `e3t` expression is not rank conformable;
that failed log is retained.  The writer now fills a zero-first temporary by
the scalar `e3t(ji,jj,jk,Kmm)` accessor before writing and compiles cleanly.

The scalar-math executable is 55,536,112 bytes, SHA-256
`78d0a06c2f21cd174579d60d65083e050fa396ec321ad33a6a69a8cbae89357a`;
dynamic-symbol inspection finds zero `_ZGV*`.  Two independent directories
are prepared with copied namelists/XML, symlinked immutable inputs, copied
manifest files, the exact binary symlink, CPU/one-thread environment,
`jpni=2,jpnj=1`, two MPI ranks, bash timing and hash-guarded self-contained
launchers:

| arm | prepared manifest SHA-256 | launcher SHA-256 |
|---|---|---|
| `variant_icebergs_off_phase2j_wzv_a_10step_np2` | `dba319c412657fe20e0d32551c8916bb06a25f4b644b5a2b6e4af341819089ff` | `6f6676ff4611678f1590b956e3f6f7bc07f756eb56172ab4ba24f7e901c1abed` |
| `variant_icebergs_off_phase2j_wzv_b_10step_np2` | `e088058506ed857f29af2b8f6b0994d1506a0983c5a878a9e218a78630627187` | `449266f6019d5607c91cccaf5eae817dce5051b093848cf26016aca2885ff505` |

After execution, both must finish ten steps, produce the original 92 streams
plus the single WZV stream, preserve ordinary-output identity against the
uninstrumented control, and produce a byte-identical new stream.  Only then
will its schema and source boundaries be scored.  FCT, its tripolar north-fold
exchange, BBL, and TKE/EVD/IWM remain not entered.

The 34-entry artifact manifest is
`nemo_testcases_l4_orca2_phase2j_artifacts.sha256`, SHA-256
`33ffce86fad37de6b8013c54a61c96f8fc48f213c21caa59ccb19a3b96b66426`.
Large binaries, records and outputs remain under the Lane-4 data root.

## 5. Coverage and time levels at this stop

| family | disposition / level |
|---|---|
| deck/card entry T/S/u/v/ssh | VERIFIED exact at `Kbb=1` |
| O1 nine CORE inputs | VERIFIED exact; `Kmm` forcing frame |
| NCAR bulk outputs | `ORACLE_SUPPLIED`, `LANE3B_OWNER` operator debt |
| SI3 exchange | `ORACLE_SUPPLIED`; producer `UNMEASURED_PENDING_ICE_MERGE` |
| RGB | VERIFIED exact on stage-3 `Krhs`, using `Kmm` depth/chlorophyll |
| EOS/HPG | VERIFIED exact at stage 2, density/live geometry `Kmm=3`, accumulating momentum `Krhs=2` |
| frozen EEN | CONFIRMED DEBT at stage-1 `Kmm=1`; GYRE owner |
| external mean/transports | `ORACLE_SUPPLIED_EXTERNAL_MODE`, stage-1 `Kmm=1` |
| `zFu/zFv` | VERIFIED exact at stage-1 `Kmm=1` |
| `ww/zFw`, runoff-in-divergence | ACQUISITION PREPARED; unmeasured |
| tripolar FCT fold and FCT tendency | NOT ENTERED |
| BBL | NOT ENTERED; ORCA2 owner uses shared census identity |
| TKE/EVD/shared ZDF | NOT ENTERED; shared debt goes to GYRE |
| IWM/geothermal/runoff file semantics | REGISTERED ORCA2 owner; only runoff WZV operands acquired here |
| SI3 dynamics/thermodynamics | `UNMEASURED_PENDING_ICE_MERGE` |

## 6. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| pin V2 only at 92 / 92 twin identity | ASKED | complete; earlier roots preserved |
| rerun identity and all plants | ASKED | Phase-1 ten, O1 three, acquisition and twin plants green |
| rescore RGB/EOS/HPG against V2 | ASKED | exact results above |
| precise EEN handoff, no shared repair | ASKED | handed to GYRE with source, record and arm |
| continue using oracle external output | ASKED | `zFu/zFv` exact; no external certification inferred |
| acquire missing WZV stencil | ASKED process | WRITE-only record; twins prepared |
| run MPI from this agent | UNASKED and prohibited | not done |
| certify north fold from scrubbed halos | UNASKED and invalid | not claimed |
| fix shared WZV/FCT/ZDF/TKE debt in Lane 4 | UNASKED and forbidden | owner rule retained |
| enter SI3 or iceberg numerics | UNASKED and forbidden | not done |
| remove failed/superseded evidence | UNASKED and forbidden | nothing deleted; failed compile and missing-venv attempt retained |

## 7. Exact user-shell handoff

Run these directories **one at a time**, using their unchanged `run.sh`:

1. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2j_wzv_a_10step_np2`
2. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2j_wzv_b_10step_np2`

On resume, require ten steps and 93 streams in each arm, ordinary-output
identity, and raw equality of the new stream between twins.  Then schema-walk
and score divergence, surface runoff, QCO WZV and `pFw`; continue to FCT only
if those boundaries are exact.  No legoESM card or SI3 work is authorized by
this handoff.
