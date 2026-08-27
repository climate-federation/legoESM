# DINO end-wall 2dt source lane: wind operands measured, placement open

Branch: `fidelity/dino-basin-rectification-codex`. Oracle repository revision:
`dcc7fb8`; legoESM base:
`782b0d7887277c88bcaa9c1be24eedad5447d9ae`. Because DINO `MY_SRC` content is
not fully pinned by that oracle Git revision, the committed probe prints
SHA-256 hashes for every cited source and every binary operand.

The accepted run receipt below names the clean probe commit, registered
preregistration commit and content hashes. The full source/operand hash map is
printed before any measurement output.

Accepted run: clean probe commit
`18107779e9c8af083b15addd1c444679e07af3d8`, preregistration commit
`4869209d918843c86bbc1368abdd16db209d7831`, current preregistration SHA-256
`e2217a90d40ab64f4c82b300f8623ef5dfeea46d838edc78f2bd028204f41ee3`,
and probe SHA-256
`0b6cdc0a93947a66b4881c5632cf89675afb38d30ef0cfe457db0bd935bfd9e9`.
The run used CPU fp64, `LEGOESM_NEMO_E3T=both`, the d180 lane and kt=5761.

## Outcome

The preserved DINO `jpdyn_tau` slot is not evidence that wind is zero. It is
emitted to the step-5764 restart but was never populated. The original offline
probe proves that donor slot is exact zero, copies its allocated 3-D arrays,
wires only their top level from NEMO's pre/post-stress brackets, proves every
lower level remains zero, and routes the source score through that copy. The
oracle artifact itself is read-only and unchanged.

The literal source gap is now closed in an isolated CPU one-step run. A
committed NEMO patch writes the exact active `trddyn` named diagnostic into
`jpdyn_tau` without adding tau to the registered nine-term interval
accumulator. The patched restart and independent stream reference contain the
same 10,082 nonzero zonal top cells; their maximum and normalized storage
errors are exact zero, as are all lower levels and meridional controls.
Classification: **CONFIRMED_NAMED_TAU_SLOT_WIRED**.

That named diagnostic is not the stress increment applied inside `dyn_zdf`.
Measured against the independent pre/post-stress bracket divided by `rDt`, its
RMS ratio is 2.0966766111, normalized difference 1.1815467878, and correlation
0.9459345292. The preregistered source distinction is therefore
**CONFIRMED_NAMED_AND_APPLIED_DIFFER**. This closes the previously unverified
premise about what the dump slot represents; it does not measure the
tridiagonal placement response or eta ownership.

The wired top-cell source operand and the independently captured
barotropic `F_slow` entry agree in RMS amplitude to within 4.1% on southern
row j=1. The frozen source-operand verdict is nevertheless
**PLAUSIBLE/UNRESOLVED**: the direct-deposit row correlation is 0.418, below
the registered 0.999 equivalence bar, while both row normalized errors are
well below the 0.25 material-DIFF bar. The `F_slow` normalized error is 0.0403
at j=1, so the fourth wind premise is **CONFIRMED**: the supplied wind slow
forcing reaches the split-explicit loop in tendency units without another
face-depth division.

This is not a measurement of the explicit-versus-implicit *response* through
the tridiagonal vertical solve and is not an eta ownership test. Candidate B
therefore remains **UNRESOLVED** pending the registered five-day placement
A/B. Candidate A, the split-explicit machinery, also remains **UNRESOLVED**:
source alignment leaves several statement-level differences and does not yet
identify the first one that changes a consuming wall-row stencil.

## Search and reuse audit

Before writing a loader, this lane searched the campaign tree for
`RUN_SEQDUMP_D180_1R`, `RUN_SEQDUMP_KT`, `wnd_dump`, `zdf_dump`, `jpdyn_tau`,
`utrd_tau`, `build_replay_ic`, `_load`, `eta_flicker_decay`, `wall_direction`,
`barotropic_substep`, `spg_substep_chain`, and `substep_traj_compare`. It
reused:

- `multistep_replay.build_replay_ic` for the kt=5760 bridge, geometry, masks,
  u-face offset, and day-180 lane conventions;
- `post_tendency_stage_birth._load` for NEMO binary shape, halo, and registered
  time-level conventions;
- `dump_lane` for the fail-closed D180/KT selector and banner;
- `zdf_dump_{u,v}1_{pre,post}stress.bin` and
  `wnd_dump_z{u,v}_frc_inc.bin` from `RUN_SEQDUMP_D180_1R`;
- `spg_substep_chain.py` and `substep_traj_compare.py` as the registered
  convention for the next first-divergence discriminator;
- `eta_flicker_decay.py` as the registered per-cell-first scorer for GPU
  handoff.

No new bridge convention or eta reducer was invented. GitHub issue #1455 was
required by the orientation, but neither an authenticated GitHub CLI/API path
nor web access was available in this sandbox. This is the local evidence
record awaiting coordinator posting.

## Active oracle route

DINO selects `ln_dynspg_exp=.false.`, `ln_dynspg_ts=.true.`,
`ln_bt_fw=.false.`, and `nn_bt_flt=2`
(`RUN_SEQDUMP_D180_1R/namelist_cfg:351-358`). The run resolves automatic
`nn_e=23` and a 68-iteration centred window
(`RUN_SEQDUMP_D180_1R/ocean.output:1043-1050,1191-1193`). DINO does not define
`key_RK3`, so the MLF arms below are live. `stpmlf` calls `sbc` before dynamics,
`dyn_spg` before `dyn_zdf`, and `dyn_zdf` before after-level reconciliation
(`cfgs/DINO/MY_SRC/stpmlf.F90:190,269-270,309-332,396-409`; reconciliation at
`:752-790`).

## Alignment A — barotropic wall-row update chain

Rows follow NEMO control-flow order. `MATCH` means the active formula and
time-level role align; it is not a causal verdict. `DIFF` names an executable
implementation difference whose wall-row consequence remains to be measured.

| Order | NEMO active statement | legoESM symbol | Alignment |
|---:|---|---|---|
| 1 | Centred wind enters frozen `zu_frc/zv_frc` as `0.5*(tau_b+tau_now)/(rho0*H)` (`cfgs/DINO/MY_SRC/dynspg_ts.F90:423-445`). | `_step_impl` centres previous/current stress and forms `F_slow`; the implicit alternative restores the same barotropic term (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3489-3498,3616-3640`). | **MATCH** in ordering, centring and units; operand score below. |
| 2 | The centred loop seeds ssh and barotropic velocity from Kbb (`dynspg_ts.F90:561-580`). | `_leapfrog_step` supplies before-level eta/u/v to `barotropic_substeps_latlon_cgrid` (`ocean_model_latlon_cgrid.py:8415-8438`; `barotropic_latlon_cgrid.py:1231-1243,1301-1308`). | **MATCH**. |
| 3 | `jn=1..icycle`, with `nn_e=23` and `icycle=68` (`dynspg_ts.F90:600-615`; `ocean.output:1046,1191`). | MLF uses `substep_scale=2` while retaining the unscaled `nn_e` boxcar width (`ocean_model_latlon_cgrid.py:4165-4186,8415-8418`). | **MATCH** active window. |
| 4 | Velocity prediction uses the initialized forward ramp then AB3 coefficients 1.781105/−1.06221/0.281105; ssh uses half-step-back interpolation (`dynspg_ts.F90:627-649,766-780`). | `nemo_boxcar_ab3` builds the ramped AB3/AM4 arrays without cross-window history (`barotropic_latlon_cgrid.py:1546-1560`). | **MATCH**. |
| 5 | Area-weighted ssh builds u/v face depths, which multiply extrapolated velocity flux (`dynspg_ts.F90:651-688,698-704`). | DINO selects `barotropic_face_depth="nemo_ssh_avg"` and the same seed convention (`packages/ocean/legoesm/ocean/experiments/dino.py:1378-1414`). | **MATCH**. |
| 6 | Flux divergence advances masked ssh (`dynspg_ts.F90:718-725`). | `_run_substep_loop` advances eta from the same face transports and mask (`barotropic_latlon_cgrid.py:851-868`). | **MATCH** equation/operator. |
| 7 | Half-back pressure gradient, live `dyn_cor_2D`, drag, frozen forcing, then masking update velocity (`dynspg_ts.F90:766-784,818-850`). | `_run_substep_loop` builds pressure, live EEN Coriolis, drag, adds `F_slow`, then masks (`barotropic_latlon_cgrid.py:903-961`). | **MATCH** order. `F_slow` is not depth-divided again. |
| 8 | Every substep commits ssh and velocity halos with `lbc_lnk`, including vector signs (`dynspg_ts.F90:899-916`). | Updated serial arrays are multiplied by local wet masks; there is no statement-level halo commit (`barotropic_latlon_cgrid.py:932-961`; rationale at `ocean_model_latlon_cgrid.py:8744-8759`). | **DIFF (boundary representation)**; equivalence at the wall Nyquist mode is unmeasured. |
| 9 | Primary velocity/ssh sums use `wgtbtp1`; transport uses the secondary tail; all normalize after the loop. Filter 2 constructs a `2*nn_e` boxcar and tail (`dynspg_ts.F90:733-738,974-1003,1242-1292`). | Weight construction, accumulators and normalized velocity/transport outputs are explicit (`barotropic_latlon_cgrid.py:1131-1140,1577-1616`). | **MATCH**. |
| 10 | NEMO commits a second LBC on normalized `un_adv/vn_adv` after the loop (`dynspg_ts.F90:999-1011`). | legoESM returns its separately accumulated transport output without a corresponding statement-level halo commit (`barotropic_latlon_cgrid.py:1577-1616`). | **DIFF (transport boundary representation)**, but scoped out of eta ownership: this output feeds tracer transport, not the completed eta. It remains a control unless a next-step eta consumer is demonstrated. |
| 11 | Immediately before `dyn_zdf`, NEMO temporarily installs the transport-mean correction in Kmm (`dynspg_ts.F90:1170-1174`). | legoESM retains the primary velocity-average momentum state and routes Hu/Hv separately to tracer transport (`barotropic_latlon_cgrid.py:1627-1651`; `packages/ocean/legoesm/ocean/experiments/dino.py:1618-1650,1760`). | **DIFF (bookkeeping only; CLOSED as momentum candidate)**. Active vector `dyn_zdf` builds Kaa from Kbb and Krhs (`cfgs/DINO/MY_SRC/dynzdf.F90:133-159`), not Kmm velocity, and later reconciliation restores the momentum mean. |
| 12 | No NEMO statement projects the completed split-explicit eta by a spatially uniform correction after the solver. | `_step_impl` applies `fix_eta_drift` after the barotropic solve (`ocean_model_latlon_cgrid.py:4216-4260`); DINO keeps it on while documenting no NEMO analogue (`packages/ocean/legoesm/ocean/experiments/dino.py:1561-1595`). | **DIFF (uniform projection)**. Its contribution to the end-wall 2dt mode is unmeasured; uniformity alone is not an exoneration. |
| 13 | `ssh_atf` applies the plain Robert-Asselin update to Nnn ssh; its variable-volume freshwater correction is exactly zero for DINO (`cfgs/DINO/MY_SRC/stpmlf.F90:455-472`; `sshwzv.F90:518-531`; `usrdef_sbc.F90:251-259`). | `_leapfrog_step` applies `now + gamma*(before-2*now+after)` to eta and stores it as the next before level (`ocean_model_latlon_cgrid.py:8644-8665,8683-8689`). | **MATCH** formula/time levels. The existing forward bracket reports correlation 1 and ratio 0.99999995; its tiny residual is not a source verdict. |
| 14 | After tracer completion, `mlf_baro_corr` installs the after barotropic mean (`cfgs/DINO/MY_SRC/stpmlf.F90:578`; reconcile body `:752-790`). | DINO selects `barotropic_after_reconcile="nemo_mlf_baro_corr"` at the post-vmix site (`packages/ocean/legoesm/ocean/experiments/dino.py:1761`; `ocean_model_latlon_cgrid.py:8081-8128,8761-8770`). | **MATCH** for the paired shipped configuration. |
| 15 | `finalize_lbc` commits U/V after-level vector signs and halos before momentum filtering (`cfgs/DINO/MY_SRC/stpmlf.F90:579-613,820-836`). | The serial domain continuously masks/wraps after-level arrays and has no discrete post-solver halo commit (`ocean_model_latlon_cgrid.py:8524-8530,8744-8759`). | **DIFF (final boundary representation)**. Existing sign/idempotence controls pass, but its next-step wall-Nyquist consequence is not exonerated; it joins the registered same-input boundary counterfactual. |
| 16 | Active vector `dyn_atf_qco` applies the plain velocity Robert-Asselin filter to Nnn after `finalize_lbc` (`cfgs/DINO/MY_SRC/stpmlf.F90:612-613`; `dynatf_qco.F90:150-167`). | `_leapfrog_step` uses the same formula and stores filtered u/v as the next before level (`ocean_model_latlon_cgrid.py:8644-8660,8683-8689`). | **MATCH** formula/order; the committed forward bracket is exact for u and v. |
| 17 | The index swap makes filtered Nnn the next Nbb and after Naa the next Nnn (`cfgs/DINO/MY_SRC/stpmlf.F90:620-624`). | The returned `naa` carries after fields as current and the filtered fields in `*_before` (`ocean_model_latlon_cgrid.py:8683-8700`). | **MATCH** return contract. |

**Candidate-A verdict: UNRESOLVED.** There is no justified single-variable GPU
arm yet. Cross-model first divergence cannot assign a mechanism while other
state/operator residuals coexist. The registered STOP now requires two
same-input counterfactuals: literal NEMO per-substep plus final post-solver LBC
versus shipped boundary representation at their first next-step consumers,
and fixer correction `c` versus zero at the next-step consumers. Each predicts
an independently dumped residual under frozen error/correlation/explained-
fraction bars. No free run is authorized until exactly one counterfactual
confirms.

## Alignment B — wind-stress entry and placement

| Order | NEMO active statement | legoESM symbol | Alignment |
|---:|---|---|---|
| 1 | DINO constructs zonal stress from the analytic latitude spline and sets meridional stress exactly zero (`cfgs/DINO/MY_SRC/usrdef_sbc.F90:157-163,200-201,219-223`). | DINO provides the same forcing class; `surface_stress_faces` owns sign/interpolation/rotation (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3607-3643`). | **MATCH** forcing class; established field identity is not re-derived here. |
| 2 | Step entry swaps current face stress into `utau_b/vtau_b`; restart/cold start reads or seeds it (`src/OCE/SBC/sbcmod.F90:378-393,549-575`). | `tau_x_prev/tau_y_prev` carry the prior step and are averaged with current stress (`ocean_model_latlon_cgrid.py:3475-3498`). | **MATCH** time levels and 0.5 centring. |
| 3 | `sbc` constructs U/V stress with neighbour averages and coastal masks (`sbcmod.F90:529-547`). | `surface_stress_faces` interpolates T stress to faces and returns top-cell face thickness (`ocean_pe_latlon_cgrid.py:3619-3643`). | **DIFF (representation)**. Direct-source normalized error is 0.0278 domain / 0.0357 j=1: close, not bit-identical. |
| 4 | `dynspg_ts` adds centred wind divided by full face-column depth to frozen slow forcing (`cfgs/DINO/MY_SRC/dynspg_ts.F90:423-445`). | The explicit route's depth mean enters `F_slow`; the implicit route restores `tau/(rho0*H)` explicitly (`ocean_model_latlon_cgrid.py:3607-3640`), and the substep consumes it directly (`barotropic_latlon_cgrid.py:932-961`). | **MATCH** source algebra. The offline `F_slow` score confirms no second face-depth division. |
| 5 | `dyn_zdf` forms `Kbb+rDt*Krhs`, removes the after barotropic mean, then applies its vertical boundary condition (`cfgs/DINO/MY_SRC/dynzdf.F90:133-170`). | Shipped `surface_stress_implicit=False` keeps stress in the explicit tendency before the barotropic solve; `withhold_stress` selects the alternative route (`ocean_pe_latlon_cgrid.py:3684-3701,4495-4501`). | **DIFF (shipped placement)**. |
| 6 | MLF adds `zDt_2*(tau_b+tau_now)/(rho0*e3_face(Kaa))` between tridiagonal recurrences (`dynzdf.F90:340-373,535-566`). | The alternative arm adds `dt_mom*tau/(rho0*dz0)` to the solve input before the same vertical diffusion solve (`ocean_model_latlon_cgrid.py:6681-6702,7003-7009`). | **MATCH** source formula in the alternative arm. Its dynamic response is unmeasured offline. |
| 7 | Active DINO `trddyn` computes the named diagnostic as `(tau_b+tau_now)/(rho0*e3_face(Kmm))` (`cfgs/DINO/MY_SRC/trddyn.F90:159-170`). The original dump live-slot list excludes tau (`cfgs/DINO/MY_SRC/trddump.F90:97-105`) but emits its unpopulated slot (`:348-350,408-410`); `jpdyn_tau=11` (`src/OCE/TRD/trd_oce.F90:74`). `dyn_zdf` instead applies the MLF half-sum with after-level thickness (`cfgs/DINO/MY_SRC/dynzdf.F90:535-566`; u-side `:340-373`). | The offline source probe scores the actually applied pre/post bracket. The committed oracle patch independently wires the exact active named `trddyn` array to the dormant slot. | **DIFF (named diagnostic versus applied term), measured**: named/applied RMS ratio 2.09668, normalized difference 1.18155. This is not a placement-response or eta-ownership measurement. |
| 8 | `mlf_baro_corr` follows `dyn_zdf` (`stpmlf.F90:396-409,752-790`). | The active shipped `leapfrog`/split-explicit path accepts either stress placement and retains the configured post-vmix reconciliation. `kamm_twin_90d.py` now exposes and stamps the existing Boolean selector. | **MATCH (arm availability)** for the registered A/B. The separate `outer_integrator="nemo_mlf"` path still has its own guard and is not this arm. |

## Offline wind source-operand measurement

The shared bridge is kt=5760 and NEMO operands are kt=5761; `rDt=5400 s`.
Statistics are pointwise on wet faces before spatial aggregation.

| Statistic | Whole wet u domain | Southern wall j=1 |
|---|---:|---:|
| direct top-cell increment normalized error | 0.0278114 | 0.0357160 |
| direct top-cell increment correlation | 0.999551 | 0.417740 |
| direct top-cell RMS ratio, lego/NEMO | 1.003474 | 1.005723 |
| direct source delta RMS (m/s) | 1.29368e-3 | 4.31475e-6 |
| NEMO direct increment RMS (m/s) | 4.65163e-2 | 1.20807e-4 |
| slow-forcing normalized error | 0.0386709 | 0.0403375 |
| slow-forcing correlation (reported, not gated) | 0.999151 | 0.860905 |
| slow-forcing RMS ratio, lego/NEMO | 1.006705 | 1.007294 |
| slow-forcing delta RMS (m/s2) | 9.34962e-10 | 4.11262e-12 |
| NEMO slow-forcing RMS (m/s2) | 2.41774e-8 | 1.01955e-10 |
| wet u faces | 9,758 | 49 |

Controls pass: both NEMO meridional operands are exact zero; legoESM's direct
operand is zero and `F_slow` maximum is `3.388e-21 m/s2`, below `1e-12`.
The preserved emitted `utrd_tau/vtrd_tau` slots are exact zero; the controlled
offline applied-term copy preserves exact-zero lower levels and reconstructs
the brackets with maximum absolute
error `1.388e-17` against a registered `1.844e-16` rounding bar. Planted
emitted-slot, lower-level and top-reconstruction violations all fire.
The helper gives exact 0x and exact doubling at 2x for both components; a
one-ULP mutation makes each equality fail. The corrected probe additionally
requires exact 1x entry identity, exact non-stress fields across arms, and the
active leapfrog event sequence `stress(Nnn), barotropic(Nnn), stress(Nbb),
momentum-vmix`. The first stress call feeds the live barotropic pass; the
second comes from the Nbb dissipative pass whose barotropic result is
discarded. A planted event swap must make the order gate fail.

The downstream B1 response is diagnostic only: zonal wind produces a
0.129398 m/s u change and `6.34021e-4 m/s` v change by the pre-vmix hook because
the barotropic loop has already rotated momentum. Comparing that v response to
NEMO's direct zero meridional deposit is retracted.

**Candidate-B source verdict: PLAUSIBLE/UNRESOLVED.** Source amplitudes are
close, but the direct j=1 pattern misses the frozen equivalence correlation
bar and neither j=1 normalized error reaches the material-DIFF bar.

## Live named `jpdyn_tau` wiring receipt

The source patch, preregistration and verifier are committed as
`nemo_endwall_tau_slot.patch`, `PREREG_nemo_endwall_tau_slot_live.md`, and
`verify_nemo_endwall_tau_slot.py`. Their SHA-256 values at the accepted
execution were `7da0400d...c6cc6d`, `4740130c...01aa8`, and
`72d4fd10...d5b46`. The temporary clone retained oracle revision `dcc7fb8`
and copied the campaign's untracked DINO `MY_SRC`, work-configuration registry
and `arch-conda.fcm` before applying the patch.

The accepted model execution used the direct single-rank binary after the
documented setup refusals and exited zero. Patched source hashes were
`9df0e792...19e9db` (`trddump.F90`), `534f00d7...bbc15`
(`trddyn.F90`), and `cb5b27ac...b08aa` (unchanged `dynzdf.F90`); executable
hash `6cfd6161...19e7`; input restart hash `0cc00f99...ff3e`; output restart
hash `33c0c1a2...a115c`. The named reference hashes were
`34c4beb3...0d111` (u) and `72ecf1fe...b7d4a` (v). The four bracket hashes
are identical to the accepted preserved-run operands, proving the hook changed
diagnostic storage, not the applied wind term.

The preregistered verifier at Git commit `b43bf5746` printed:

- zonal slot/reference nonzero cells: 10,082 / 10,082;
- lower-level maximum: exact zero;
- meridional slot and named-reference maxima: exact zero;
- storage maximum and normalized error: exact zero versus the
  `1.7763568394002505e-15` bar;
- named/applied RMS ratio `2.0966766111118194`, normalized difference
  `1.1815467878135875`, correlation `0.9459345292292405`;
- both planted controls: fired;
- `SOURCE_DISTINCTION=CONFIRMED_NAMED_AND_APPLIED_DIFFER`,
  `CLASSIFICATION=CONFIRMED_NAMED_TAU_SLOT_WIRED`, and
  `PLACEMENT_OWNERSHIP=UNRESOLVED`.

The preceding corrected run is a retained failed-instrument control. Its
explicit `(jpi,jpj)` dummy did not conform to the active 52-by-199 `T2D(0)`
actual bounds inside the 56-by-203 haloed domain, so the verifier classified
`REFUTED_WRONG_SOURCE`. The preregistration records that failure and the
native-bound correction before the accepted rerun. The earlier claim that an
applied-increment hook was the named `utrd_tau` diagnostic is retracted.

**Candidate-B placement/ownership verdict: UNRESOLVED.** The registered GPU
discriminator runs the same bridged state for five days with only
`surface_stress_implicit` changed. The committed preregistration contains the
exact implicit/control commands, per-step fp64 capture contract, certified
NEMO artifact and scorer hashes, and coherent first-eight whole-domain plus
aggregate-wall gates: `1.25/0.17` confirmation versus `2.30/0.38` refutation.
The previous last-half/zonal-wall labels and first-sample 0.85 gate are
retracted because they mixed different observables.

## Review disposition and retractions

Two independent read-only reviewers iterated over the evidence and mechanism
claims before these findings were finalized.

- Evidence review: the probe measured a wired source operand, not the
  implicit-placement response; the `jpdyn_tau` slot is emitted but not
  populated; provenance did not bind untracked `MY_SRC` content; and the
  correlation gate exceeded the preregistration. Disposition: scope narrowed,
  wording corrected, source/operand content hashes and entry controls added,
  and the implementation gate now exactly matches the registered direct-only
  correlation rule.
- Mechanism review: the wind command did not select an arm; the barotropic
  command selected no mechanism; the pre-vmix reconciliation was temporarily
  different; `fix_eta_drift` was omitted; and an LBC comparison must inspect
  the first consuming stencil. Disposition: an executable stamped wind
  selector was added, the nominal barotropic GPU arm was removed, the two
  missing DIFF rows were added, and the next discriminator is registered at
  the first consumer.
- Second evidence review: the accepted log named a stale prereg commit, the
  temporary Kmm state was incorrectly called a live `dyn_zdf` input, and the
  diagnostic slot was bypassed. Disposition: the probe now binds the final
  prereg commit, Kmm is closed as momentum bookkeeping, and source scoring
  runs through a controlled offline-wired copy of the emitted 3-D slot.
- Second mechanism review: the GPU commands produced daily fp32 eta, the Kmm
  bookkeeping DIFF was not live, cross-model divergence could not assign an
  LBC mechanism, and normalized `un_adv/vn_adv` had a second LBC. Disposition:
  the twin now has a fail-closed per-step fp64 eta contract with exact
  NEMO/scorer invocations; Kmm is closed; candidate A requires same-input
  counterfactuals; and the second LBC is listed and scoped to tracer transport.
- Final evidence review: the offline copy still did not satisfy the literal
  request to wire NEMO's emitted `jpdyn_tau` store. Disposition: a committed
  source patch produced an isolated NEMO CPU restart. The later mechanism
  review corrected its source identity; the named-diagnostic result is the
  independently verified one reported below.
- Final mechanism review: the GPU preregistration mixed whole-domain
  first-eight amplitude, first-sample aggregate-wall share, and a last-half
  zonal-wall label; its extractor pointed at a directory without retained
  restart tiles; and alignment A stopped before the Asselin/final-LBC/swap
  feedback chain. Disposition: one coherent first-eight domain/locus pair is
  frozen, the certified NEMO eta artifact and scorer are SHA-bound, rows 13-17
  complete the active chain, and final `finalize_lbc` joins the same-input
  boundary counterfactual.
- Named-slot mechanism review: the first live hook populated the slot from the
  applied `dynzdf` bracket, but active `trddyn.F90:164-170` names a different
  no-half, Kmm-thickness diagnostic. Disposition: the old classification is
  retracted in the tool and preregistration; the hook now copies the exact
  active `trddyn` arrays, an independent stream proves exact storage, and the
  named-versus-applied distinction is reported separately.

## Verification receipts

- Clean offline probe at the accepted commit: PASS, frozen science tuple
  unchanged, all entry/helper/order/slot/structural-zero controls PASS.
- NEMO source patch: `git apply --check` PASS; isolated DINO fp64 build PASS;
  direct single-rank one-step CPU execution PASS (exit 0).
- Live-slot verifier: `CONFIRMED_NAMED_TAU_SLOT_WIRED`; 10,082 nonzero
  top-level zonal values in both slot/reference, exact-zero storage error and
  lower/meridional controls, and both planted violations fire. Independent
  source distinction: `CONFIRMED_NAMED_AND_APPLIED_DIFFER`.
- Python compile and ruff on the new probe: PASS; E501 check on the legacy twin
  harness: PASS.
- Twin selector/config smoke and implicit-arm construction on CPU: PASS;
  resolved active path is `leapfrog` + `explicit_substep` +
  `surface_stress_implicit=True`.
- Zero-day artifact-contract smoke: PASS; primary eta is fp64 per-step shape,
  relative `t_seconds`, `capture_every_steps=1`, daily eta is preserved, and
  the resolved placement stamp is true.
- `test_nemo_mlf_rejects_surface_stress_implicit` and
  `test_kamm_mlf_ships_the_faithful_pair`: PASS.
- Existing `test_surface_stress_implicit_wiring` remains RED at the unchanged
  base and current branch because it expects the old non-split solver guard;
  the production split-explicit exemption predates this lane. No failure was
  hidden or threshold relaxed.

Standing retractions:

- `utrd_tau == 0` does not mean the physical wind term is zero.
- “Every emitted `utrd_tau` artifact is unwired” is retracted: the preserved
  campaign artifact is unwired, while the isolated patched one-step artifact
  is source-wired and independently verified.
- The former `CONFIRMED_SLOT_WIRED` result is retracted as a statement about
  NEMO's named `utrd_tau`: that hook stored the applied increment divided by
  `rDt`. The corrected named slot is separately verified above.
- NEMO wind is not implicit-only; it enters `dynspg_ts` and `dyn_zdf`.
- Whole-B1 stress linearity is not a valid exact helper control.
- Expecting one stress-helper call was wrong: active leapfrog evaluates the
  helper in both its live Nnn advective pass and its Nbb dissipative pass.
- Scaling current stress without its previous centred carry is a confounded
  arm.
- The raw lane mask is already aligned; stripping another halo is wrong.
- NEMO's direct deposit and legoESM's post-barotropic B1 state span different
  control flow.
- The offline source score measures neither implicit-solve response nor eta
  ownership. No causal `CONFIRMED` label is issued for either candidate.
