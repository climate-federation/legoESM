# NEMO testcase lane 2 — GYRE phase-3 whole-step RK3 receipt

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Preregistration: `e0e1b89639874c7d1e8c777028d30ffa9f943557`

Committed gate: `99a1ab9e6`

Latest authoritative follow-up: the causal and within-substep result is in
`nemo_testcases_l2_gyre_phase3_barotropic_receipt.md`.  Historical numbers
below are retained as the audit trail for the card state that produced them.

## Verdict

**DEBT at the first completed whole step (`kt=2`); continued under the
preregistered owner-exhausted arm through `kt=10`.**  The `kt=1` entry remains
bit-exact in native fp64.  At `kt=2`, T, S, u, v, and SSH are all over the
immutable normalized max-absolute bar `1e-15`.  The fail-closed report retains
that first boundary; later samples never restore an exact-prefix claim.

This is the first ladder case whose NEMO oracle executes the complete
three-stage RK3 composition: 3-D momentum, tracers, analytic seasonal SBC,
and the primary/advecting barotropic frames on the rotated GYRE grid.  The
oracle inventory and all Kaa/Kmm headers are VERIFIED.  legoESM's exposed
momentum states are measured at every stage and are DEBT beginning at stage 1.
Internal legoESM tracer-stage and barotropic-frame numerics remain honestly
**UNMEASURED**; the completed-step T/S/u/v/SSH rows are measured.

| frame | T | S | u | v | SSH |
|---|---:|---:|---:|---:|---:|
| `kt=1` Nbb entry | exact | exact | exact, UNINFORMATIVE | exact, UNINFORMATIVE | exact, UNINFORMATIVE |
| `kt=2` normalized max error | `3.3845756796730703e-3` | `1.1181970871158484e-4` | `5.9613298992648146e-2` | `1.1394000503432558e-1` | `1.6868983837431638e-3` |
| `kt=10` normalized max error | `7.854782002100048e-2` | `5.293769180801024e-4` | `5.503377940554033e-2` | `1.6267913746004683` | `9.94681112869481e-3` |

The at-rest `kt=1` u/v/SSH rows are explicitly UNINFORMATIVE for the same
reason as lane 1: exact zero can check storage and alignment but cannot
exercise momentum or barotropic evolution.

## Oracle build, run, and stage evidence

The oracle is NEMO 5.0.2 `cfgs/GYRE_PISCES`, physics-only with the shipped
`ln_p4z=.false.` toggle, rebuilt as `GYRE_OMIP_L2_P3` with exactly
`key_qco key_vco_1d3d key_RK3`.  It retains the phase-1 resolved OMIP-style
TEOS-10 deviation and changes only the end/restart/output step to 10 plus
read-only instrumentation.  The serial CPU executable completed ten steps
with exit 0 and wrote the `kt=10` restart.  The first attempted executable,
which was accidentally rebuilt without `MY_SRC`, is retained under
`uninstrumented_attempt/` and excluded from every scored artifact.

The instrumented executable uses lane 1's byte-identical `stprk3.F90` and
`stprk3_stg.F90` dump machinery.  NEMO's executed call order is source-pinned:

- `src/OCE/stprk3.F90:127-186` evaluates the clock/SBC, EOS and closures, then
  composes `stp_2D` once;
- `:192-230` runs all three stages, writes their Kaa states, and performs the
  alternating index swaps;
- `src/OCE/stprk3_stg.F90:257-304` reconciles stage Kmm transports;
- `:309-446` forms the momentum RHS, completes stages 1--3, and applies each
  stage's barotropic mean correction;
- `:456-521,568-599` executes tracer transport/FCT/SBC at every stage and the
  stage-3 solar, lateral, and vertical tracer closures.

The observed registry is stage 1 `(Kaa=3,Kmm=1)`, stage 2 `(2,3)`, and stage
3 `(3,2)`.  Whole-step entry `Nbb` alternates `1,3,...`; the barotropic `Naa`
frame alternates `3,1,...`.  Every binary reader calls the repository-wide
`time_level_for_dump` registry before parsing and also validates the numeric
header.  The first committed run caught an incorrect hard-coded RHS/alternating
level expectation and stopped before a model step; commit `422b9d3a3` replaces
that assumption with the observed, source-consistent registry.

The momentum-stage normalized max errors are:

| stage | u | v |
|---:|---:|---:|
| 1 | `1.465032210595928e-4` | `5.8579299575484675e-2` |
| 2 | `1.4735166781177294e-4` | `8.802447761163189e-2` |
| 3 | `5.9613298992648146e-2` | `1.1394000503432558e-1` |

Thus the first measurable stage divergence is stage 1, not merely the
committed `kt=2` entry.  These rows demonstrate a complete NEMO stage
execution and a measured legoESM momentum-stage mismatch; they do not claim
numerical tracer-stage or barotropic-frame parity.

## Seasonal forcing composition

The GYRE card evaluates the already certified rotated grid and seasonal
analytic fields at each NEMO clock.  `usrdef_sbc.F90:109-120` defines qsr and
the Nbb-SST Haney `qns`; `:122-140` fills EMP over all 704 cells, divides its
unmasked numerator by the 600-cell wet denominator, then subtracts the mean
only on wet cells; `:138-145` supplies EMP heat content with `sfx=0`; and
`:161-184` supplies grid-aligned ocean stress.  The gate therefore passes
stress through legoESM's atmosphere-sign surface channel, heat as the
Nbb-evaluated total plus penetrative shortwave, and EMP as a real-freshwater
volume flux.  The legoESM configuration reports
`freshwater_closure=real_freshwater` with its required source-inclusive eta
projection.

This faithful transcription does not imply that every closure placement is
already identical.  In particular, the card's collapsed flux-UP3 identity,
legacy shortwave split, and missing GYRE closure selections remain candidate
differences.  The trajectory result measures their composed consequence; it
does not silently call any one of them the owner.

## Scaling-first one-variable arms

Four literal one-input arms were run at `kt=2`.  The surface and freshwater
forcing controls are separate kwargs; neither arm changes both.

| sole changed operand | movement / faithful worst residual | disposition |
|---|---:|---|
| `stage_barotropic_correction` omitted | `2.8731014633985343` | PLAUSIBLE_CONTRIBUTOR_NOT_OWNER |
| `surface_forcing` omitted | `1.5231060545470452` | PLAUSIBLE_CONTRIBUTOR_NOT_OWNER |
| `freshwater` omitted | `6.021626711589655e-4` | REFUTED_AS_PRIMARY_OWNER |
| `momentum_transport_reconcile` omitted | `2.4539303606152927e-6` | REFUTED_AS_PRIMARY_OWNER |

The faithful worst normalized residual is `0.11394000503432558`.  No arm
clears all fields or supplies two-sided source-isolated closure, so the owner
verdict is **UNMEASURED_AFTER_REGISTERED_ARMS**.  The two material arms are
not proposed fixes: both remove required pieces of NEMO's executed program,
and improvement/movement at the right scale is only contributor evidence.

## kt=2--10 growth

The continuation reuses lane 1's committed `characterize_growth` function.
For T, the tail log-log fit is preferred (`p=1.3505617315357719`,
`R2=0.9996568072930037`) and successive error ratios decrease.  SSH is also
classified polynomial-preferred (`p=2.610959375196431`,
`R2=0.9698283181904321`), but its step ratios are not monotone.  U's final
four samples decay, producing `BOUNDED_OR_DECAYING_NO_AMPLIFYING_MODE` with
tail power `-1.8540726911292147`.  The lane-1 instrument does not fit S or V;
their growth characterization is therefore UNMEASURED even though their
pointwise trajectory rows are present through `kt=10`.

These are short, nine-sample diagnostic fits, not seasonal-gyre phenomenology
or an owner assignment.

## Controls, tests, and artifacts

The committed unit controls pass: exact fp64 equality, a planted wet-cell
state violation, all three at-rest UNINFORMATIVE classifications, the
one-variable manifest, the central registry, and lane-1 growth-instrument
reuse.  The combined live planted invocation (`--plant-state
--plant-registry --plant-arm --max-step 2`) exits 2 and names both injected
failures: `GYRE.kt1.stage1.Kaa_Kmm_registry` and
`arm_manifest.omit_stage_barotropic_correction`.  The planted state row also
turns DEBT.  The focused suite passes 30 tests before the final arm split and
6/6 tests after it; Ruff passes on the new gate/tests and `git diff --check`
is clean.

| artifact | SHA256 |
|---|---|
| full phase-3 gate report | `692534082071d7ff18e4a0ad85fa5c931505e599c294afcc70b5f30eda6ad95d` |
| instrumented `nemo.exe` | `66acfda7aee933aa93813e0afb98d4b7733bcd8ec81c36169091ffa0b597148d` |
| `kt=10` restart | `3271da17716957c2c043f16af62b77d3aad3213bffd83ddce433d65edb6794ba` |
| committed/run namelist | `85d0f2b0655d7d89dbe676e1d1a0dd1b73ab387b76111dfabc1627d4ab81271d` |
| resolved output namelist | `66b532caccfcef7b799a2669fcbdb47fbabe319331ae7a978f8998696feee4c7` |
| `mesh_mask.nc` | `3bf5d10e36dc52336b9797b13eb1efb4f02d25e0fb3a0c450ac6ce69e65471df` |
| `stprk3.F90` | `5fe6e2d341579f85e3b37e400aae0ee7f571a70fa46cc35d66da46961e1c9df3` |
| `stprk3_stg.F90` | `f3e19ce0e37df219040ddf6ccd50493e2bd5e045aa6eaaa2eb539efdaafb59f1` |
| `kt=1` / `kt=2` entry | `9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0` / `887f3bbb51e047ee7c072ae74b77a8e5b461525341b1ab228ae7d800fa757777` |
| stage 1 / 2 / 3 | `35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402` / `55e780b8d56eb249e5387123b20aeae8f735325200714c02a816ef719d6b3351` / `3703a9f2e369f8f05cc439564729e36801227b54eb8afb7df2552a5120a45f2e` |
| transport 1 / 2 / 3 | `fb18372778bd345092cda706b60a4caa0d06bb112803f73413772f4c44a17d90` / `eb0c44fa404ba40b1d6342c314f4904ddb3e88bb6febb2c71850857e4b000aaa` / `b5feddae7255b3645ce97d12c951c13104810453f414fef318d5fa324605cdcb` |
| entering RHS / first BT frame | `a09426f638de0ce384b739621d08cf7ae27d41339cadcc2f16f8bbd3de215c45` / `b8f474af46b665773152bd2f152d20f29f658b51a052422dcb35dc406fdf2fa9` |

All remaining per-file hashes are embedded in the full JSON gate report.
Claude's Phase 2 review required the four corrections now closed.  Claude's
Phase 3 review and its redirect appear below; GLM review remains outstanding,
and no dual-review claim is made.

## Claude phase-3 redirect closure

Claude's phase-3 verdict was **SHIP as an honest debt round**, with a major
coverage finding: the trajectory card did not arm the resolved GYRE operator
program.  The historical measurements above remain valid for that incomplete
card.  They are not evidence against the corrected card.  Coverage was frozen
before implementation in `ee6b5aed0`; the corrected run stops again at `kt=2`.

### Full resolved-program coverage

The inventory is driven by a runtime parse of every `namdyn*`, `namzdf*`, and
`namtra*` block in the resolved `output.namelist.dyn`, then joined against the
card's static scientific-disposition checklist.  A future unmatched block is
DEBT; this is block discovery, not runtime parsing of every selector value.

| block | resolved NEMO selection | corrected collapsed card | gate |
|---|---|---|---|
| `namdyn_adv` | vector invariant, C2 KE | vector invariant + `c2` + `nemo_advective` vertical path | VERIFIED |
| `namdyn_vor` | ENE total vorticity | `ene_total` at F points | VERIFIED |
| `namdyn_hpg` | SCO | `nemo_sco`, NEMO trapezoid | VERIFIED |
| `namdyn_spg` | TS, auto 50, filter 3/alpha .07 | 50, `nemo_ab3am4`, .07 | VERIFIED |
| `namdyn_ldf` | level Laplacian, `Uv=2`, `Lv=100000` | `nemo_div_curl`, `nemo_e3`, `Ah=100000` | VERIFIED |
| `namtra_adv` | FCT 2/2 | `fct2` in WS-RK3 | VERIFIED |
| `namtra_ldf` | standard isoneutral Laplacian, `Ud=.02`, `Ld=100000` | `nemo_iso_lap`, Redi 1000, slope cap .01 | VERIFIED |
| `namtra_eiv` | off | GM/bolus coefficient zero | VERIFIED-INACTIVE |
| `namtra_qsr` | two band, `.58/.35/23` | Jerlov type I exact literals | VERIFIED |
| `namtra_dmp` | off | no tracer damping | VERIFIED-INACTIVE |
| `namtra_mle` | off | no MLE | VERIFIED-INACTIVE |
| `namzdf` | TKE+EVD; `evd=100`; `avm0=1.2e-4`, `avt0=1.2e-5`; adaptive ZAD off | prognostic TKE, momentum+tracer EVD 100, matching floors, adaptive off | VERIFIED |
| `namzdf_tke` | `.1/.7/67.83`, `mxl=3`, `mxl0`, Ri-Prandtl, LC, surface/bottom BCs | canonical NEMO prognostic closure with TEOS-10 BN2 | VERIFIED |

The card has no unaccounted requested block.  The three live omissions beyond
Claude's six were tracer isoneutral diffusion, exact two-band penetration, and
`ln_zad_aimp=.false.`.  Inactive CST/RIC/GLS/OSM/MFC/NPC/DDM/SWM/IWM selectors
remain written waivers.  The fail-closed gate emits all 13 rows and its planted
`namdyn_vor` violation exits 2.

### Pre-implementation search and reuse

Search found, and the implementation reused, the DINO/NEMO canonical options:
vector-invariant momentum, ENE-total (`dynvor.F90:406-536`), C2 KE
(`dynkeg.F90:104-123`), `nemo_advective` ZAD, `nemo_div_curl` level viscosity,
`nemo_iso_lap` Redi, prognostic TKE, EVD, and Jerlov two-band penetration.
No new physical recurrence was added.  The WS-RK3 validator was widened from
one complete flux-UP3 identity to exactly two complete identities; hybrids and
nonzero staged GM bolus still raise.  GYRE's independently generated MI96
`gdept/e3w` pair has one four-ULP recurrence miss at level 23, so both raw
oracle operands are retained under a four-ULP construction gate with a larger
planted mismatch test.

### Review redirect evidence

| Claude row | incomplete card | corrected selection |
|---|---|---|
| momentum advection | flux-form UP3 | vector invariant + NEMO advective ZAD |
| F-point vorticity | bypassed AL81 declaration | ENE total `(f+zeta)/e3f` |
| KE gradient | centered | C2 mean-of-squares |
| momentum LDF | `Ah=0` | level Laplacian, `Ah=100000 m2/s` |
| vertical momentum closure | constant `1e-4`, no TKE/EVD | prognostic TKE + EVD, floor `1.2e-4` |
| vertical tracer closure | `Kv=0` | TKE tracer floor `1.2e-5` + EVD |

The incomplete stage-1 residuals were u `1.465032210595928e-4` and v
`5.8579299575484675e-2`: v/u = `399.85`.  After coverage completion they are
u `1.305741086612574e-4` and v `1.3169038482681593e-4`.  The 400x asymmetry is
gone, consistent with the review's vorticity redirect.  This is structural
evidence, not a sole-owner claim.  Stage-3 u remains
`5.959892337101344e-2`, so the barotropic-composition boundary remains open.

### Corrected kt=1, kt=2, and scaling-first sweep

`kt=1` remains bit-exact in fp64; at-rest u/v/SSH remain UNINFORMATIVE.  The
corrected `kt=2` row remains DEBT:

| field | normalized max error |
|---|---:|
| T | `2.2011146495261743e-3` |
| S | `1.095339477691381e-4` |
| u | `5.959892337101344e-2` |
| v | `3.7669958034661236e-2` |
| SSH | `1.4793035345182532e-3` |

Each arm changes one registered scheme identity.  The table reports magnitude
before any label; every owner remains `UNMEASURED_SCALING_ONLY`.

| armed term removed | field | corrected residual | term magnitude | residual / term |
|---|---|---:|---:|---:|
| vector/ENE/C2 identity | v | `3.766996e-2` | `7.401938e-9` | `5.089202e6` |
| level momentum Laplacian | u | `5.959892e-2` | `9.179315e-9` | `6.492742e6` |
| isoneutral tracer Laplacian | T | `2.201115e-3` | `5.547859e-7` | `3.967503e3` |
| TKE+EVD+background identity | T | `2.201115e-3` | `3.604915e-3` | `0.610587` |
| TKE+EVD+background identity | v | `3.766996e-2` | `8.914170e-2` | `0.422585` |
| two-band shortwave | T | `2.201115e-3` | `1.799818e-3` | `1.222965` |

Thus the remaining T and v residuals are at the scale of the TKE/EVD and
shortwave arms; u is millions of times larger than either newly armed
momentum term at the completed-step frame.  Scale agreement is not ownership,
and no post-hoc owner label is made.

The two momentum arms are therefore **NEAR-NULL AT kt=2** and have no
discriminating power; that label is not exoneration.  The fail-closed gate and
runtime-namelist regression test carry this annotation explicitly.

Corrected gate artifact:
`legoesm_phase3_redirect_gate.json`, SHA256
`d7fac51e3094681514fd6d0c510595388fab5c0b0972a8832f1ba598e0cdf859`.
Backend is CPU; state, geometry, and oracle arrays report fp64.  The focused
coverage/card/gate suite passes 48 tests.  GLM review remains outstanding; this
receipt makes no dual-review claim.
