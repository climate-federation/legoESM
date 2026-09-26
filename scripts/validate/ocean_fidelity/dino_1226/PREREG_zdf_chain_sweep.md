# Preregistration: day-180 ZDF execution-chain sweep

Date: 2026-08-28.  Status at commit: **measurement not yet run**.

This lane returns the DINO campaign to an ordered source-execution sweep.  It
starts from the matched day-180 state, compares one live NEMO operation at a
time, and stops at the first numeric divergence.  Pattern statistics from the
MLD audit select columns; they do not decide fidelity.

## Fixed oracle and state

- legoESM source is this preregistration's parent commit.  The measurement
  artifact will record its full SHA.
- NEMO is `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`; the measurement
  artifact will record SHA256 for every quoted source and every consumed dump.
- The oracle lane is the existing `dump_lane.py` selector `d180`:
  `cfgs/DINO/RUN_SEQDUMP_D180_1R`, restart
  `DINO_00005760_restart.nc`, `kt=5761`, `rn_Dt=2700 s`, one rank,
  `nn_hls=2`, fp64.  No historical year-5 result may satisfy a row here.
- The active build has `key_qco key_vco_3d`; the resolved run reports
  `l_zco=T`, `ln_isfcav=F`, `ln_zdftke=T`, `ln_zdfevd=T`, `nn_evdm=1`,
  `rn_evd=100`, `ln_zdfddm=F`, `ln_zdfswm=F`, `ln_zdfiwm=F`,
  `ln_zdfmfc=F`, `ln_zdfnpc=F`, `ln_zad_Aimp=F`, `nn_pdl=1`,
  `nn_mxl=3`, `ln_lc=T`, `nn_etau=1`, and `nn_htau=1`
  (`RUN_SEQDUMP_D180_1R/ocean.output:424-427,727,735-747,758-792`).

## Registered columns

The targeting source is the committed MLD-audit artifact
`docs/ocean/fidelity/dino_mld_climate_audit_artifact.json` at audit commit
`49fbb5ddf52`; its `mld_maps.npz` has SHA256
`9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0`.
The focus rule is fixed before the ZDF measurements: take every wet day-90
southern-basin column (audit rows 0 through 13) whose legoESM and NEMO native
MLD base indices differ.  In zero-based, halo-stripped `(j,i)` coordinates the
registered set is exactly

```text
(11, 1), (12, 1), (13, 1), (13, 23)
```

The probe must independently reproduce this set from the audit maps and fail
if it changes.  Every numeric row is scored both for these four columns and
for a census of every wet whole-domain column.  Focus columns receive the same
bar as all other columns.

## Frozen metrics and bars

The campaign gate constants are not changed: aggregate correlation must be at
least `1 - 1e-9`, and the aggregate RMS ratio must be within `1e-6` of one.
Aggregate values are diagnostics only: **every wet column must also pass**.

For a continuous field row, define one score per horizontal column without
regional or layer pooling:

```text
column_error(j,i) = max_k |lego(k,j,i) - NEMO(k,j,i)|
                    / RMS_wet(NEMO)
```

The maximum is over the row's live wet levels and `RMS_wet(NEMO)` is one fixed
whole-domain scale for that row.  If that scale is exactly zero, equality must
be bit-exact.  A POINTWISE row passes only if every column error is at most
`1e-15`; a vertically accumulating or tridiagonal-solve row passes only if
every column error is at most `1e-12`.  Integer indices and branch masks must
be exactly equal.  Ratio/correlation are reported only where mathematically
defined and never rescue a failed column.

Each measured row receives exactly one disposition:

- `VERIFIED`: all applicable exact checks, column bars, and aggregate bars pass;
- `DIVERGED`: the first failed row, followed by operand substitutions until
  the first failing operand is named; subsequent live rows remain unmeasured;
- `WAIVED`: only a source branch proven inactive by the resolved DINO run, with
  the reason and source/config evidence written in the result.

## Ordered live call table

Rows are in execution order, not diagnostic priority.  `P` means POINTWISE,
`A` means vertically ACCUMULATING/solve, and `E` means exact integer/branch
equality.  Existing dump names are listed where present; missing slots must be
added using the existing write-only `MY_SRC` stream-dump pattern and bracketed
by a no-dump versus dump-enabled CPU one-step restart comparison before they
can be cited.

| Row | Operation and active operands | NEMO execution evidence | Class | Existing checkpoint |
|---:|---|---|:---:|---|
| 1 | `eos_rab(Nbb)` alpha/beta inputs to `rn2b` | `cfgs/DINO/MY_SRC/stpmlf.F90:204` | P | restart + existing EOS probe |
| 2 | `bn2(Nbb)`: interpolation weight, alpha, beta, T/S numerator, **live `e3w(Kmm)` divisor**, mask, then `rn2b` | `stpmlf.F90:206`; `src/OCE/TRA/eosbn2.F90:1458-1467` | P | `tke_dump_rn2b.bin`; existing bn2 loaders |
| 3 | `eos_rab(Nnn)` and `bn2(Nnn)` producing `rn2`, with the same live divisor | `stpmlf.F90:205,207`; `eosbn2.F90:1458-1467` | P | `tke_dump_rn2.bin` |
| 4 | shear production `sh2`: NOW×BEFORE velocity differences, live NOW×BEFORE `e3uw/e3vw`, wet-only coastal interpolation | `cfgs/DINO/WORK/zdfphy.F90:264-269`; `cfgs/DINO/WORK/zdfsh2.F90:68-99` | P | `tke_dump_sh2.bin` |
| 5 | bottom-drag coefficient update used by TKE bottom boundary | `zdfphy.F90:277`; `cfgs/DINO/WORK/zdfdrg.F90:103-109` dispatch, nonlinear bottom operands `:171-190` | P | restart operands; add slot if reached |
| 6 | native MLD `nmln`: positive-`rn2b` integral with live `e3w(Kmm)` and density threshold | `zdfphy.F90:280`; `cfgs/DINO/WORK/zdfmxl.F90:90-100` | A/E | restart + `tke_dump_rn2b.bin`; add `nmln` slot if reached |
| 7 | native MLD depth `hmlp=gdepw(nmln,Kmm)` | `zdfmxl.F90:101-105` | P | add `hmlp` slot if reached |
| 8 | TKE surface Dirichlet boundary `max(rn_emin0,rn_ebb/rho0*taum)` | `cfgs/DINO/MY_SRC/zdftke.F90:334-365` | P | restart operands; add slot if reached |
| 9 | TKE bottom boundary: wet-only U/V average and bottom drag | `zdftke.F90:368-384` | P | restart operands; add slot if reached |
| 10 | Langmuir velocity scale, integrated `rn2b` potential energy, diagnosed depth and TKE source | `zdftke.F90:401-468` (`ln_lc=T`, stress-derived arm `:422-431`) | A/E/P | add stage slots if reached |
| 11 | Richardson number and inverse Prandtl number from `rn2b`, incoming `avm`, and `sh2` | `zdftke.F90:477-496` | P/E | `tke_dump_{zri,pdlr,avm_in,sh2,rn2b}.bin` |
| 12 | TKE diffusion matrix lower/upper/diagonal coefficients | `zdftke.F90:499-510` | P | add coefficient slots if reached |
| 13 | TKE RHS shear operand | `zdftke.F90:513` | P | `tke_dump_sh2.bin`; add stage slot if reached |
| 14 | TKE RHS stratification operand `-avt*rn2` | `zdftke.F90:514` | P | `tke_dump_rn2.bin`; add stage slot if reached |
| 15 | TKE RHS semi-implicit dissipation operand | `zdftke.F90:510,515`; carried `dissl` | P | `tke_dump_dissl.bin`; add stage slot if reached |
| 16 | wave-coupled surface boundary | `zdftke.F90:524-545` | E | WAIVE if resolved `ln_wave=F` continues to prove the condition false |
| 17 | TKE tridiagonal forward/back solve and floor/mask | `zdftke.F90:547-566` | A/E | add pre-penetration `en` slot if reached |
| 18 | `nn_etau=1` exponentially penetrating TKE addition using live `gdepw(Kmm)` and `htau` | `zdftke.F90:588-592` | P | `tke_dump_en.bin` is post-stage; add pre-stage slot if reached |
| 19 | buoyancy mixing length from final `en` and `rn2` | `zdftke.F90:693-760` | P | `tke_dump_{en,rn2}.bin` |
| 20 | `nn_mxl=3` downward/upward limiting and combined mixing/dissipation lengths | `zdftke.F90:799-813` | A | `tke_dump_{zmxlm,zmxld}.bin` |
| 21 | base `avm`, `avt`, and new `dissl` assembly | `zdftke.F90:832-838` | P | `tke_dump_{avm_final,avt_final}.bin`; next-step dump for new `dissl` if needed |
| 22 | inverse-Prandtl correction to `avt` | `zdftke.F90:841-844` | P | `tke_dump_{pdlr,avt_final}.bin` |
| 23 | closure `avt_k/avm_k` copied to composed `avt/avm` | `zdfphy.F90:311-315` | E | closure final dumps |
| 24 | river-mouth enhancement | `zdfphy.F90:317-321` | E | WAIVE only if resolved `ln_rnf_mouth=F` |
| 25 | EVD tracer branch mask `min(rn2,rn2b)<=-1e-12`, overwrite `avt=100*wmask` | `zdfphy.F90:323`; `cfgs/DINO/WORK/zdfevd.F90:79-95` | E/P | add pre/post EVD slots if reached |
| 26 | EVD momentum overwrite under the same mask (`nn_evdm=1`) | `zdfevd.F90:105-121` | E/P | add pre/post EVD slots if reached |
| 27 | `avs=avt`; DDM, surface-wave and internal-wave enhancements | `zdfphy.F90:326-336` | E | equality is live; WAIVE the three disabled enhancement branches with resolved flags |
| 28 | turbocline scan on composed `avt`, then `hmld=gdepw(imld,Kmm)` | `zdfphy.F90:338`; `cfgs/DINO/WORK/zdfmxl.F90:123-153` | E/P | add composed-`avt`/index/depth slots if reached |
| 29 | `avm` lateral boundary update | `zdfphy.F90:344` | E | compare interior unchanged; halo is outside the column census |
| 30 | standard `ldf_slp` inputs and slopes, including live-`e3w(Kmm)` limiter | `stpmlf.F90:214-234`; `cfgs/DINO/MY_SRC/ldfslp.F90:202-365` (notably `:310-327`) | P | `seq_dump_rhd_bbb.bin`; existing ldf-slop probes/dumps where available |
| 31 | momentum vertical-diffusion matrices and U/V implicit solves using composed `avm` | `stpmlf.F90:396`; `cfgs/DINO/MY_SRC/dynzdf.F90:199-213,337-380,391-406,531-566` | A | existing state and surface-stress brackets; add matrix slots if reached |
| 32 | tracer `zwt=avt/avs+ah_wslp2`, matrices with live `e3w(Kmm)`, and T/S implicit solves | `stpmlf.F90:551`; `cfgs/DINO/WORK/trazdf.F90:157-222,237-275` | A | existing before/after tracer state; add matrix slots if reached |

The top-drag/cavity arm (`zdfphy.F90:278`) is preregistered `WAIVED` because
`ln_isfcav=F` and `ln_drgice_imp=F`.  The TKE cavity boundary
(`zdftke.F90:385-395`) is likewise inactive.  OSM/RIC/GLS/CST alternatives,
DDM, NPC, MFC, surface-wave mixing, internal-wave mixing, and adaptive
implicit vertical advection are not substitute candidates; their disabled
arms will be reported as explicit waivers where execution reaches them.

## Controls and stop rule

Before accepting any numeric disposition, the committed probe must:

1. fail closed unless `DINO_1226_LANE=d180`, fp64 is active, array shapes and
   time-level registry entries match, and every input SHA matches the artifact;
2. reproduce the four focus columns from the MLD maps;
3. prove halo stripping/axis order with the existing dump loader conventions;
4. plant a nonzero perturbation into one compared wet value and demonstrate
   that its column changes from passing to `DIVERGED`;
5. plant a one-cell horizontal roll and demonstrate that the census fails;
6. if a new NEMO write-only slot is needed, compare a dump-disabled and a
   dump-enabled one-step CPU run from the same restart and require all normal
   output/restart bytes to be identical.

Measurement stops at the first failed row.  Operand substitution proceeds
within that row only, in the expression's NEMO evaluation order, until the
first failing operand is localized.  Later rows are reported `UNMEASURED`, not
waived and not inferred from downstream composites.
