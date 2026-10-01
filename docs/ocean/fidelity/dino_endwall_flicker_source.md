# DINO end-wall 2dt source lane: wind operands measured, placement open

## LANE STATUS — finished unit, 2026-08-28

- **CONFIRMED:** the dominant wall-flicker source was a one-cell coastal-donor
  double interpolation in bridge/mimicry glue.  NEMO's U-point prior stress
  was stored as a T-point carry and interpolated to U again.  The faithful
  fix reconstructs the prior analytic T-point stress, is now the harness
  default, and stamps `T`, reconstruction time, and content hash.  Corrected
  explicit GPU scoring moved flicker ratio `2.89x -> 1.16x` and wall share
  `0.486 -> 0.119`.
- **NULL at the measured resolution:** corrected T carry does not resolve a
  change in the 90-day southern-basin gap.  Its measured
  `Delta90 = +2.939395701062608e-05 Sv` is below
  `F90_current = 1.5266693430725714e-04 Sv`; the frozen verdict is
  **UNRESOLVED/FLOOR**.
- **RECONCILED:** the historical/current basin-baseline epoch is owned by the
  combined EEN-plus-bridge-Omega change, with measured non-additive interaction
  `I = -0.0840808093571308 Sv`.  `Glegacy90` is re-registered as
  `-0.43908550999203477 Sv` and its floor was re-measured at the current epoch.
- **OPEN for a later branch:** the barotropic boundary/eta-fixer
  counterfactuals are designed but not built; explicit-versus-implicit
  placement ownership remains unresolved; and the approximately `200x`
  corrected-carry implicit shock remains undiagnosed.

**Standing human rule, DECIDED 2026-08-28:** when a faithful fix exists, the
default selects it.  A known-wrong path is opt-in only.  Accordingly the
reconstructed T-point carry is the default and historical U-as-T reproduction
requires `--bridge-before-stress-legacy-u-as-t`.

Branch: `fidelity/dino-basin-rectification-codex`. Oracle repository revision:
`dcc7fb8`; legoESM base:
`782b0d7887277c88bcaa9c1be24eedad5447d9ae`. Because DINO `MY_SRC` content is
not fully pinned by that oracle Git revision, the committed probe prints
SHA-256 hashes for every cited source and every binary operand.

The accepted run receipt below names the clean probe commit, registered
preregistration commit and content hashes. The full source/operand hash map is
printed before any measurement output.

Accepted donor-aware round-2 run: clean probe commit
`6572f6e7469fd13be3e54cc57ccd73bf36d4671b`, donor preregistration commit
`25a5afef6dd3f4e50c8b9d7febdbaf69b3b87095`, probe SHA-256
`2a5a27543e981d4df6b3f253a89921daa0bf79955f5d0e3391fa1dd52f1be252`,
and retained log SHA-256
`825c22478ca4e19a422a46fdbe5e9c5cfd2ba5753f86d3a2623459e1663c8dff`.
The run used CPU fp64, `LEGOESM_NEMO_E3T=both`, the d180 lane and kt=5761.

## Outcome

The preserved DINO `jpdyn_tau` slot is not evidence that wind is zero. It is
emitted to the step-5764 restart but was never populated. The original offline
probe proves that donor slot is exact zero, copies its allocated 3-D arrays,
wires only their top level from NEMO's pre/post-stress brackets, proves every
lower level remains zero, and routes the source score through that copy. The
oracle artifact itself is read-only and unchanged.

The isolated CPU patch writes the active `trddyn` named array into
`jpdyn_tau`. Its restart slot and stream have the same 10,082 nonzero top
values and exact-zero storage residual, but both are output paths from the
same in-memory array. The result is therefore **CHECKED-CLEAN write-only
plumbing**, not independent verification. The patched restart has about
10,082 nonzero `utrd_tau` entries where base `trddump.F90` asserts the slot is
zero, so downstream budget-summing tools would differ.

**RETRACTED loudly:** `CONFIRMED_NAMED_AND_APPLIED_DIFFER` and 2.0966766111.
Active `dynzdf` uses `zDt_2=rDt/2`; on wet U faces the no-half named diagnostic
is exactly twice the applied increment. The old union mask admitted 324 dry
coastal faces, and its normalized-difference REFUTE state was unreachable.
The replacement claim is only the wet arithmetic identity, verified to
`1e-8`; it is not a placement or ownership finding.

The round-2 one-cell probe **CONFIRMS a localized source difference**. On
southern row `j=1`, both direct `dynzdf` and `F_slow` put the discrepancy at
the single face `(j=1,i=49)` out of 49: direct delta
`3.0203216696068663e-5 m/s`, row-scale maximum `0.2500117364`, and
peak-equivalent face count `1.0000000231`. The `F_slow` row-scale maximum is
`0.2826356698`; the direct and `F_slow` one-percent supports have Jaccard 1.0.
Across 9,758 wet U faces, their peak-equivalent counts are 24.27148675 and
24.27150701. All 162 faces exceeding one percent lie at `i=49`; “about 24
faces” is effective squared-error support, not a literal outlier count.

The earlier receiving-face and receiving-support coastal verdicts are
**RETRACTED**. They classified the wet receiver, which cannot itself have
`umask=0`, rather than the upstream donor. The donor-aware one-cell trace
**CONFIRMS_COASTAL_DONOR_DOUBLE_INTERPOLATION**. NEMO restart `utau_b` is
already a U-point field, but the bridge stores it in legoESM's T-point
`tau_x_prev`; the first step interpolates it to U again. At receiver `(1,49)`,
the east donor `(1,50)` is dry coastal-unmasked with multiplier 2 and carries
exactly twice the receiver stress (`4.6526513163130625e-4` versus
`2.3263256581565312e-4 Pa`). The resulting production stress exceeds the
stagger-correct value by exactly the predicted donor contribution.

The predicted direct increment is `3.020321760467329e-5 m/s` versus measured
`3.0203216696068663e-5 m/s` (relative error `3.0083e-8`). Predicted/direct and
predicted/`F_slow` material-support Jaccards are both 1.0; predicted domain
support is 24.27023460 peak-equivalent faces versus measured 24.27148675.
Thus the fourth wind premise is closed at the source: the same one-column
donor error enters both direct and barotropic paths before either consumer.

Candidate B placement ownership remains **UNRESOLVED** after the five-day GPU
A/B: implicit placement collapses the first-eight wall share to 0.05596, near
NEMO's locus, but explodes amplitude to 201.41 through a diffuse startup
transient. Candidate A also remains **UNRESOLVED** pending the registered
same-input boundary/fixer consumers.

## Search and reuse audit

Before writing a loader, this lane searched the campaign tree for
`RUN_SEQDUMP_D180_1R`, `RUN_SEQDUMP_KT`, `wnd_dump`, `zdf_dump`, `jpdyn_tau`,
`utrd_tau`, `build_replay_ic`, `_load`, `eta_flicker_decay`, `wall_direction`,
`barotropic_substep`, `spg_substep_chain`, `substep_traj_compare`, `utau_b`,
`tau_x_prev`, `surface_stress_faces`, and `interp_cell_to_uface`. That search
found the existing restart loader and bridge carry in `nemo_io.py` and
`nemo_state_bridge.py`, plus the shared production interpolator; it reused:

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
| 1 | Centred U/V-face wind enters frozen `zu_frc/zv_frc` as `0.5*(tau_b+tau_now)/(rho0*H)` (`cfgs/DINO/MY_SRC/dynspg_ts.F90:423-445`). | `_step_impl` centres previous/current stress and forms `F_slow`; the implicit alternative restores the same barotropic term (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3594-3603,3616-3640`). | **DIFF on the active bridged first entry** because the prior U/V-face restart stress is carried as T-point and interpolated again; structural ordering, centring and units **MATCH only after a representation-consistent carry**. |
| 2 | The centred loop seeds ssh and barotropic velocity from Kbb (`dynspg_ts.F90:561-580`). | `_leapfrog_step` supplies before-level eta/u/v to `barotropic_substeps_latlon_cgrid` (`ocean_model_latlon_cgrid.py:8556-8579`; `barotropic_latlon_cgrid.py:1231-1243,1301-1308`). | **MATCH**. |
| 3 | `jn=1..icycle`, with `nn_e=23` and `icycle=68` (`dynspg_ts.F90:600-615`; `ocean.output:1046,1191`). | MLF uses `substep_scale=2` while retaining the unscaled `nn_e` boxcar width (`ocean_model_latlon_cgrid.py:4271-4292,8415-8418`). | **MATCH** active window. |
| 4 | Velocity prediction uses the initialized forward ramp then AB3 coefficients 1.781105/−1.06221/0.281105; ssh uses half-step-back interpolation (`dynspg_ts.F90:627-649,766-780`). | `nemo_boxcar_ab3` builds the ramped AB3/AM4 arrays without cross-window history (`barotropic_latlon_cgrid.py:1546-1560`). | **MATCH**. |
| 5 | Area-weighted ssh builds u/v face depths, which multiply extrapolated velocity flux (`dynspg_ts.F90:651-688,698-704`). | DINO selects `barotropic_face_depth="nemo_ssh_avg"` and the same seed convention (`packages/ocean/legoesm/ocean/experiments/dino.py:1378-1414`). | **MATCH**. |
| 6 | Flux divergence advances masked ssh (`dynspg_ts.F90:718-725`). | `_run_substep_loop` advances eta from the same face transports and mask (`barotropic_latlon_cgrid.py:851-868`). | **MATCH** equation/operator. |
| 7 | Half-back pressure gradient, live `dyn_cor_2D`, drag, frozen forcing, then masking update velocity (`dynspg_ts.F90:766-784,818-850`). | `_run_substep_loop` builds pressure, live EEN Coriolis, drag, adds `F_slow`, then masks (`barotropic_latlon_cgrid.py:903-961`). | **MATCH** order. `F_slow` is not depth-divided again. |
| 8 | Every substep commits ssh and velocity halos with `lbc_lnk`, including vector signs (`dynspg_ts.F90:899-916`). | Updated serial arrays are multiplied by local wet masks; there is no statement-level halo commit (`barotropic_latlon_cgrid.py:932-961`; rationale at `ocean_model_latlon_cgrid.py:8885-8900`). | **DIFF (boundary representation)**; equivalence at the wall Nyquist mode is unmeasured. |
| 9 | Primary velocity/ssh sums use `wgtbtp1`; transport uses the secondary tail; all normalize after the loop. Filter 2 constructs a `2*nn_e` boxcar and tail (`dynspg_ts.F90:733-738,974-1003,1242-1292`). | Weight construction, accumulators and normalized velocity/transport outputs are explicit (`barotropic_latlon_cgrid.py:1131-1140,1577-1616`). | **MATCH**. |
| 10 | NEMO commits a second LBC on normalized `un_adv/vn_adv` after the loop (`dynspg_ts.F90:999-1011`). | legoESM returns its separately accumulated transport output without a corresponding statement-level halo commit (`barotropic_latlon_cgrid.py:1577-1616`). | **DIFF (transport boundary representation)**, but scoped out of eta ownership: this output feeds tracer transport, not the completed eta. It remains a control unless a next-step eta consumer is demonstrated. |
| 11 | Immediately before `dyn_zdf`, NEMO temporarily installs the transport-mean correction in Kmm (`dynspg_ts.F90:1170-1174`). | legoESM retains the primary velocity-average momentum state and routes Hu/Hv separately to tracer transport (`barotropic_latlon_cgrid.py:1627-1651`; `packages/ocean/legoesm/ocean/experiments/dino.py:1618-1650,1760`). | **DIFF (bookkeeping only; CLOSED as momentum candidate)**. Active vector `dyn_zdf` builds Kaa from Kbb and Krhs (`cfgs/DINO/MY_SRC/dynzdf.F90:133-159`), not Kmm velocity, and later reconciliation restores the momentum mean. |
| 12 | After `dyn_spg` returns, active MLF proceeds through `dyn_zdf` and the normal ssh-after/filter path without a spatially uniform post-solver eta projection (`cfgs/DINO/MY_SRC/stpmlf.F90:396-455`). | `_step_impl` applies `fix_eta_drift` after the barotropic solve (`ocean_model_latlon_cgrid.py:4322-4366`); DINO keeps it on while documenting no NEMO analogue (`packages/ocean/legoesm/ocean/experiments/dino.py:1561-1595`). | **DIFF (uniform projection)**. Its contribution to the end-wall 2dt mode is unmeasured; uniformity alone is not an exoneration. |
| 13 | `ssh_atf` applies the plain Robert-Asselin update to Nnn ssh; its variable-volume freshwater correction is exactly zero for DINO (`cfgs/DINO/MY_SRC/stpmlf.F90:455-472`; stock `src/OCE/DYN/sshwzv.F90:390-443`, RA at `:419-425`; active instrumented override mirrors it at `cfgs/DINO/MY_SRC/sshwzv.F90:480-553`, RA at `:513-522`; `usrdef_sbc.F90:251-259`). | `_leapfrog_step` applies `now + gamma*(before-2*now+after)` to eta and stores it as the next before level (`ocean_model_latlon_cgrid.py:8785-8806,8683-8689`). | **MATCH** formula/time levels. The existing forward bracket reports correlation 1 and ratio 0.99999995; its tiny residual is not a source verdict. |
| 14 | After tracer completion, `mlf_baro_corr` installs the after barotropic mean (`cfgs/DINO/MY_SRC/stpmlf.F90:578`; reconcile body `:752-790`). | DINO selects `barotropic_after_reconcile="nemo_mlf_baro_corr"` at the post-vmix site (`packages/ocean/legoesm/ocean/experiments/dino.py:1761`; `ocean_model_latlon_cgrid.py:8222-8269,8761-8770`). | **MATCH** for the paired shipped configuration. |
| 15 | `finalize_lbc` commits U/V after-level vector signs and halos before momentum filtering (`cfgs/DINO/MY_SRC/stpmlf.F90:579-613,820-836`). | The serial domain continuously masks/wraps after-level arrays and has no discrete post-solver halo commit (`ocean_model_latlon_cgrid.py:8665-8671,8744-8759`). | **DIFF (final boundary representation)**. Existing sign/idempotence controls pass, but its next-step wall-Nyquist consequence is not exonerated; it joins the registered same-input boundary counterfactual. |
| 16 | Active vector `dyn_atf_qco` applies the plain velocity Robert-Asselin filter to Nnn after `finalize_lbc` (`cfgs/DINO/MY_SRC/stpmlf.F90:612-613`; `dynatf_qco.F90:150-167`). | `_leapfrog_step` uses the same formula and stores filtered u/v as the next before level (`ocean_model_latlon_cgrid.py:8785-8801,8683-8689`). | **MATCH** formula/order; the committed forward bracket is exact for u and v. |
| 17 | The index swap makes filtered Nnn the next Nbb and after Naa the next Nnn (`cfgs/DINO/MY_SRC/stpmlf.F90:620-624`). | The returned `naa` carries after fields as current and the filtered fields in `*_before` (`ocean_model_latlon_cgrid.py:8824-8841`). | **MATCH** return contract. |

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
| 1 | At restarted `kt=nit000`, the normal prior-field swap is skipped (`src/OCE/SBC/sbcmod.F90:378-393`), then DINO constructs current zonal T-point stress from the analytic latitude spline and sets meridional stress exactly zero (`sbcmod.F90:415-424`; `cfgs/DINO/MY_SRC/usrdef_sbc.F90:157-163,200-201,219-223`). | DINO provides the same forcing class; `surface_stress_faces` owns sign/interpolation/rotation (`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3607-3643`). | **MATCH** current forcing class; established field identity is not re-derived here. |
| 2 | `sbc` constructs current U/V stress with neighbour averages and the explicit coastal multiplier `(2-umask)*MAX(tmask_i,tmask_i+1)` (`src/OCE/SBC/sbcmod.F90:539-547`, U formula `:543-544`). | Current legoESM T-point forcing is converted by plain `interp_cell_to_uface` (`ocean_pe_latlon_cgrid.py:3657-3693`; `operators_latlon_cgrid.py:332-352`). | **MATCH at the measured wet receiver for current stress**; NEMO's coastal representation also creates the doubled dry U donor retained in its restart carry. |
| 3 | Still at `kt=nit000`, NEMO reads prior `utau_b/vtau_b` with `cd_type='U'/'V'` (`sbcmod.F90:549-558`); the arrays are declared U/V-point (`src/OCE/SBC/sbc_oce.F90:106-110`). | `read_nemo_restart_before` labels/loads them as if T-point (`packages/ocean/legoesm/ocean/fidelity/nemo_io.py:225-265`), and `bridge_before_state_topo` stores them directly in T-point `tau_x_prev/tau_y_prev` (`packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py:802-833`) before centring and another interpolation (`ocean_model_latlon_cgrid.py:3595-3604`). | **DIFF (bridged first-step stagger), CONFIRMED coastal-donor source**. Receiver `(1,49)` is wet, but its east prior-U donor `(1,50)` is dry coastal-unmasked, multiplier 2, and exactly twice the receiver stress. Later legoESM steps replace the carry from native current forcing (`ocean_model_latlon_cgrid.py:8831-8840`), so this is not a perpetual carry-type claim. |
| 4 | `dynspg_ts` adds centred U-face wind divided by full face-column depth to frozen slow forcing (`cfgs/DINO/MY_SRC/dynspg_ts.F90:423-445`). | The explicit route's depth mean enters `F_slow`; the implicit route restores `tau/(rho0*H)` explicitly (`ocean_model_latlon_cgrid.py:3713-3746`), and the substep consumes it directly (`barotropic_latlon_cgrid.py:932-961`). | **DIFF on the bridged first entry; MATCH after a representation-consistent carry**. `F_slow` reproduces the donor prediction with support Jaccard 1.0 and no second face-depth division. |
| 5 | `dyn_zdf` forms `Kbb+rDt*Krhs`, removes the after barotropic mean, then applies its vertical boundary condition (`cfgs/DINO/MY_SRC/dynzdf.F90:133-170`). | Shipped `surface_stress_implicit=False` keeps stress in the explicit tendency before the barotropic solve; `withhold_stress` selects the alternative route (`ocean_pe_latlon_cgrid.py:3734-3751,4495-4501`). | **DIFF (shipped placement)**. |
| 6 | MLF adds `zDt_2*(tau_b+tau_now)/(rho0*e3_face(Kaa))` between tridiagonal recurrences (`dynzdf.F90:340-373,535-566`). | The alternative arm adds `dt_mom*tau/(rho0*dz0)` to the solve input before the same vertical diffusion solve (`ocean_model_latlon_cgrid.py:6794-6815,7003-7009`). | **MATCH** source formula in the alternative arm. Its dynamic response is unmeasured offline. |
| 7 | Active DINO `trddyn` computes the named diagnostic as `(tau_b+tau_now)/(rho0*e3_face(Kmm))` (`cfgs/DINO/MY_SRC/trddyn.F90:159-170`). The original dump live-slot list excludes tau (`cfgs/DINO/MY_SRC/trddump.F90:97-105`) but emits its unpopulated slot (`:329-330`) and waived accumulated tau (`:389-390`); `jpdyn_tau=11` (`src/OCE/TRD/trd_oce.F90:74`). `dyn_zdf` applies the MLF half-sum with `zDt_2=rDt/2` (`cfgs/DINO/MY_SRC/dynzdf.F90:115,340-373,535-566`). | The offline source probe scores the applied pre/post bracket. The committed patch copies the named array to the dormant slot and a stream through two write paths. | **MATCH (wet arithmetic), CHECKED-CLEAN plumbing**: named is twice applied on wet faces to `1e-8`. The old 2.09668 source-DIFF claim is retracted as half-step identity plus dry-union-mask artifact. |
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

The aggregates conceal the registered one-cell result:

| Localizer | Direct `dynzdf` | `F_slow` |
|---|---:|---:|
| `j=1` argmax | `(1,49)` | `(1,49)` |
| argmax delta | `3.02032167e-5 m/s` | `2.87883172e-11 m/s2` |
| max abs / absolute NEMO row mean | 0.2500117364 | 0.2826356698 |
| peak-equivalent faces, `j=1` / domain | 1.0000000231 / 24.27148675 | 1.0000000231 / 24.27150701 |
| wet faces with pointwise ratio error >1%, `j=1` / domain | 1 / 162 | 1 / 162 |
| receiving argmax context (descriptive, not source verdict) | wet-wet, multiplier 1 | wet-wet, multiplier 1 |
| receiving-support coastal fraction (descriptive) | 0.5134 | 0.0 |
| current one-cell label | `CONFIRMED_ONE_CELL_SIGNATURE` | `CONFIRMED_ONE_CELL_SIGNATURE` |

The one-percent supports are identical (Jaccard 1.0) and occupy column `i=49`
across 162 rows. Peak-equivalent faces is
`sum(delta**2)/max(delta**2)`; the first execution's inverse-participation name
is retracted and retained only as a descriptive value. The corrected
participation, second-face and support-Jaccard controls all fire. The former
receiving-support coastal labels are retracted because they did not trace the
donor.

The donor-aware centerpiece measures:

| Quantity | Value |
|---|---:|
| receiver / east donor | `(1,49)` / `(1,50)` |
| receiver / donor prior ocean-sign stress | `2.3263256581565312e-4` / `4.6526513163130625e-4 Pa` |
| donor/receiver ratio | `2.0` |
| receiver / donor `umask`, coastal multiplier | `1, 1.0` / `0, 2.0` |
| production / stagger-correct receiver stress | `2.907907072695664e-4` / `2.326325658156531e-4 Pa` |
| predicted / measured direct delta | `3.020321760467329e-5` / `3.0203216696068663e-5 m/s` |
| prediction relative error | `3.0083042e-8` |
| predicted/direct; predicted/`F_slow` support Jaccard | `1.0`; `1.0` |
| predicted / measured domain peak-equivalent faces | `24.27023460` / `24.27148675` |
| verdict | `CONFIRMED_COASTAL_DONOR_DOUBLE_INTERPOLATION` |

The shifted-donor control lands on noncoastal `(1,51)`, replacing the donor
with receiver stress annihilates the exact re-interpolation anomaly, and
synthetic donor support moves the overlap gate from 1.0/CONFIRM to
0.0/REFUTE. All three controls fire.

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

**Candidate-B source verdict:
CONFIRMED_COASTAL_DONOR_DOUBLE_INTERPOLATION for the bridged first step.** The
old `j=1` Pearson gate is retracted because NEMO's row is effectively constant.
The measured receiver, donor ratio, production reconstruction, direct-delta
prediction and both support maps clear every frozen donor gate. This assigns
the source discrepancy, not free-run flicker ownership. The discriminating
next measurement is the preregistered one-step same-state counterfactual with
a reconstructed T-point previous-stress carry, followed—only if that gate
passes—by the representation-consistent explicit/implicit GPU pair.

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
are identical to the accepted preserved-run operands, showing the applied-wind
brackets were unchanged. This is a write-path plumbing check.

The strengthened verifier at clean probe commit `6572f6e74` (script SHA-256
`c45989d6fc9427c87239cc2804ee4a9dc168bb115be742e514f2c0fa0d753fc9`,
retained log SHA-256
`8160cc3f2ef7195aa7094888cdc06a8b9f9bb6c6969561cde66eeb0e87d74fb8`)
printed:

- zonal slot/reference nonzero cells: 10,082 / 10,082;
- lower-level maximum: exact zero;
- meridional slot and named-reference maxima: exact zero;
- storage maximum and normalized error: exact zero versus the
  `1.7763568394002505e-15` bar;
- wet pointwise ratio mean `2.0000000004056662`, absolute mean error
  `4.0567e-10`, standard deviation `9.9839e-9`, and descriptive maximum
  error `1.6380e-7`;
- lower/top storage, single-face ratio-spread, and uniform `+1` ratio-mean
  controls: fired;
- round-2 classification: **CHECKED-CLEAN write-only**. The slot and stream
  are the same in-memory array through two output paths, and the offline
  `rDt*((post-pre)/rDt)` reconstruction is a float64 round trip, not an
  independent validation.

The preceding corrected run is a retained failed-instrument control. Its
explicit `(jpi,jpj)` dummy did not conform to the active 52-by-199 `T2D(0)`
actual bounds inside the 56-by-203 haloed domain, so the verifier classified
`REFUTED_WRONG_SOURCE`. The preregistration records that failure and the
native-bound correction before the accepted rerun. The earlier claim that an
applied-increment hook was the named `utrd_tau` diagnostic is retracted.

**Candidate-B placement/ownership verdict: UNRESOLVED.** The clean explicit
control at Git `4d00f81bb78caa29dd07ab6e3a4081e6d57a4bd2` measured first-eight
ratio 2.8824000013 and wall share 0.4855027388, inside its validity band. Its
NPZ is pinned by SHA-256
`2845a5c7166baad483f89ae91a1fb1c9b03971a6a63c4e41506615b2a632df5f`.
The implicit arm measured ratio 201.41187345 and wall share 0.0559638894.
Mechanical bars are UNRESOLVED: amplitude fails CONFIRM while locus fails
REFUTE. The implicit field is diffuse (2,234 cells for half variance versus 74
in explicit) and its approximately eight-step decay is consistent with, but
does not prove, a placement-swap startup shock against an explicit-consistent
bridged restart.

The discard-32 proposal is **retracted as a decisive ownership test** because
it only hides the now-measured U-as-T bridge shock. Round 3 implements the
fail-closed `--bridge-before-stress-tpoint`, reconstructs prior T-point analytic
stress at the restart's own `15,552,000 s` seasonal time, and stamps stagger,
time and array-content hash. It does not invert `utau_b/vtau_b`, edit a
prognostic, or alter the later-step carry path.

The retained same-state CPU gate at clean Git
`fd0e83c40a28a8cb34be8c6db0582b0e474bdfde` passed:

| Registered CPU quantity | Legacy U-as-T | Reconstructed T carry |
|---|---:|---:|
| direct `(1,49)` increment | `1.510160880233664e-4` | `1.2081287041869309e-4 m/s` |
| direct excess versus NEMO | `3.0203216696068663e-5` | `-9.086046395519534e-13 m/s` |
| `F_slow` `(1,49)` | `1.439415861637953e-10` | `1.1515326893097271e-10 m/s2` |
| `F_slow` excess versus NEMO | `2.8788317232756413e-11` | `-6.617444900424221e-23 m/s2` |

The direct excess removal is `0.9999999699169579`
(`99.9999969917%`), above the frozen `99.9999%` bar. Replanting the legacy
stagger restores the entire excess with relative error `0`, below `1e-6`.
The donor prediction, direct correction and `F_slow` correction each have 154
wet material faces; both prediction-support Jaccards are `1.0`. All 15 other
state leaves, geometry, ladders, config, forcing and external tendency are
bit-identical; a planted eta edit makes that identity check fail.

The retained CPU-gate log SHA-256 is
`4cb84d20b7b0236b842932f5683a13e7be82c7bca748e40bd824d694bdbc030c`.
The reconstructed T carry has SHA-256
`b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae`.
A zero-step artifact smoke serialized `bridge_before_stress_stagger="T"`,
reconstruction time `15552000.0`, the same content hash, and explicit placement;
artifact/log SHA-256 are `f8a55cd59aa4f5dd6474786acc412d7202d1277b98a9bcfce23412eaced936f8`
and `a68998694ce091b44d0bf60c38e900f80433e9bb4d10a570fae48c8f39b0822e`.

This defect is **bridge/mimicry glue, not model configuration**. The
oracle-recipe doctrine's §3 rule B puts oracle-only time-level/stagger handling
in the bridge/harness, using the test “would a user pursuing a different goal
ever select this?”; here the answer is no. Rule C likewise reserves convention
re-encoding for I/O boundaries while keeping one canonical model
representation (`docs/ocean/fidelity/oracle_recipe_strategy.md:141-161`;
`CLAUDE.md`, “Oracle-Recipe Fidelity”). Therefore no model-config field or
production solver branch was added.

The reuse audit searched for `_restart_elapsed_seconds`,
`dino_lat_lon_surface_forcing_arrays`, `dino_step_surface_forcing`,
`tau_x_prev`, and `_seed_centred_forcing_carry`. It found and reused the
restart `adatrj/kt` cross-check and the existing analytic forcing/sign loaders
in `kamm_twin_90d.py` and `dino.py`; later-step carry remains the existing
`ocean_model_latlon_cgrid.py:8831-8840` path.

## Round-4 GPU disposition

**Registered verdict: STOP/control-invalid.** The corrected explicit control
measured first-eight ratio `1.1607251697830108` and aggregate-wall share
`0.11941867158491266`, both outside the frozen `[2.60,3.18]` and
`[0.38,0.59]` control bands. The preregistered STOP therefore fired and the
corrected implicit arm is **not classified**. Its ratio `200.41106914842686`,
wall share `0.05585390128091391`, last-half ratio `10.545129752846789`, and
last-half bootstrap CI90 `[8.67094529695706,12.53280691395405]` are
descriptive shock evidence only.

**DESCRIPTIVE pending re-frozen bars:** against the round-1 uncorrected
explicit baseline (`2.882400001277192`, `0.4855027387838016`), the T-carry
bridge correction removed `91.46168881885039%` of the excess-over-one flicker
ratio and `87.81675304445774%` of the excess wall share above NEMO
(`0.0686300413685866`). Thus the bridge representation defect, not wind
placement, owned most of the baseline wall flicker. This is not a registered
placement verdict. The round-1 numbers remain true of the uncorrected bridge;
nothing is retracted.

Both five-day arms were clean at Git `9ca58a379afe65f4ee485615622c541c06d5d7c7`,
fp64, stable for 160 samples, and stamped T carry, reconstruction time
`15552000.0 s`, and content hash
`b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae`.
The explicit/implicit NPZ SHA-256 values are
`8b290ff1a2bd06bb757366d5e08d51a81f8bc4259975d26e376241a42ab97cc7`
and `ffd77c118772d3d6cc859995c2c75a791eceeb556caae86f8d7678d27f86f58d`;
their clean run-log hashes are
`ecff84341c4ab0960f3b08276d69666d83de0a144dca47863693b49e054ccb17`
and `aa9c91492361a836a850a370ab2edfbe64c2143c1bbe005f797e563edbcbe0aa`;
their scorer JSON hashes are
`5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0`
and `b2727b02c402b0746eb2ac8ca44440b47eaece84cdf855d271597c1130c23060`.

The first implicit-shock counterfactual proposed after Round 4 is withdrawn
before execution: its depth-uniform `du_dt_pert` change is removed by the
barotropic reconstruction/reconciliation, so it could not remove eta while
`F_slow` stayed fixed. The replacement preregistration names the post-`dyn_zdf`,
post-reconcile zonal-velocity leapfrog pair (`u`, filtered `u_before`) as the
candidate inconsistent carry. A first-divergence trace must first prove equal
step-one `F_slow`/substep eta and locate the first persistent difference there;
then a one-step step-two substitution, planted restoration, and explicitly
mapped 1%-material T-grid support decide the mechanism. No GPU rerun is
authorized before that CPU gate.

**DECIDED 2026-08-28:** reconstructed T carry is the default.  The known-wrong
legacy U-as-T carry remains reachable only through
`--bridge-before-stress-legacy-u-as-t`.  Standing rule: when a faithful fix
exists the default selects it; historical defect paths are opt-in only.

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
  runs through a controlled applied-term copy using the emitted 3-D shape.
- Second mechanism review: the GPU commands produced daily fp32 eta, the Kmm
  bookkeeping DIFF was not live, cross-model divergence could not assign an
  LBC mechanism, and normalized `un_adv/vn_adv` had a second LBC. Disposition:
  the twin now has a fail-closed per-step fp64 eta contract with exact
  NEMO/scorer invocations; Kmm is closed; candidate A requires same-input
  counterfactuals; and the second LBC is listed and scoped to tracer transport.
- Final evidence review: the offline copy still did not satisfy the literal
  request to wire NEMO's emitted `jpdyn_tau` store. Disposition: a committed
  source patch produced an isolated NEMO CPU restart. The later mechanism
  review corrected its source identity; round-2 review further limits the
  named-diagnostic result to checked-clean write-path plumbing.
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
  active `trddyn` arrays. Round-2 review downgrades this to checked-clean
  write-path plumbing because slot and stream share one in-memory array.
- Round-2 cross-family review: 2.0966766111 was a half-step identity plus 324
  dry faces admitted by a union mask, the `j=1` Pearson bar was unreachable,
  and aggregates concealed a one-face row discrepancy. Disposition: both
  claims are retracted in the tools; wet-only arithmetic and row-scale gates
  replace them; the one-cell/coastal probe was preregistered and run clean.
- Donor-aware mechanism review: the receiving-face classifier was unreachable,
  and source inspection showed NEMO `utau_b` is U-point while the bridge stores
  it as a T-point carry. Disposition: receiving-support verdicts are retracted,
  the exact donor interpolation is reconstructed under reachable controls, and
  the GPU follow-up now requires a representation-consistent bridge.

## Verification receipts

- Clean offline probe at the accepted commit: PASS, frozen science tuple
  unchanged, all entry/helper/order/slot/structural-zero controls PASS.
- NEMO source patch: `git apply --check` PASS; isolated DINO fp64 build PASS;
  direct single-rank one-step CPU execution PASS (exit 0).
- Live-slot verifier: `CHECKED-CLEAN write-only`; 10,082 nonzero
  top-level zonal values in both slot/reference, exact-zero storage error and
  lower/meridional controls. The old independent source distinction is
  retracted; wet arithmetic is named=2*applied to `1e-8`.
- Donor-aware one-cell probe at clean commit `6572f6e74`: PASS; direct and
  `F_slow` one-cell, second-face, identical/disjoint Jaccard, shifted/equalized
  donor, donor-overlap, entry/helper/order/slot, and structural-zero controls
  all fire/pass. Independent mechanism-review rerun also exited zero (log
  SHA-256 `c91d054ac58103d5dc27743ba287b8dd4ae9fd7dfb436dda51ab4ff2b78b595f`).
- Round-3 T-point bridge CPU gate at clean commit `fd0e83c40`: PASS; direct
  excess removal `99.9999969917%`, planted-stagger restoration relative error
  `0`, and both donor-support Jaccards `1.0`. The removal and restoration
  scoring paths each have planted reachable REFUTE controls.
- Independent Round-3 evidence review: PASS; frozen bars, direct/`F_slow`
  conventions, clean input hashes, other-leaf identity, and all planted
  decision/support/order controls were checked with no hold.
- Independent Round-3 mechanism review: PASS; NEMO U/V restart staggering,
  existing DINO loader/sign chain, restart clock, fail-closed selector, later
  carry path, serialized receipts, oracle-recipe classification, and exact GPU
  handoff were checked with no hold.
- Python compile and ruff on the new probe: PASS; E501 check on the legacy twin
  harness: PASS.
- Twin selector/config smoke and implicit-arm construction on CPU: PASS. The
  shipped card resolves `leapfrog` + `explicit_substep` +
  `surface_stress_implicit=False`; `--surface-stress-implicit` is the treatment
  and `--no-surface-stress-implicit` is the control.
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
  is source-wired and checked clean through two non-independent output paths.
- The former `CONFIRMED_SLOT_WIRED` result is retracted as a statement about
  NEMO's named `utrd_tau`: that hook stored the applied increment divided by
  `rDt`. The corrected named slot is separately verified above.
- `CONFIRMED_NAMED_AND_APPLIED_DIFFER` and 2.0966766111 are retracted. Wet
  faces obey named=2*applied; 324 dry coastal faces and an unreachable gate
  produced the former headline.
- The `j=1` Pearson equivalence gate is retracted. NEMO's row is effectively
  constant; max-absolute delta divided by absolute row mean replaces it.
- The first round-2 domain “participation” label is retracted: it printed
  inverse participation. The corrected review metric is peak-equivalent
  faces, 24.27149 in both source paths.
- Receiving-face and receiving-support coastal classifications are retracted:
  they tested the wet recipient rather than its dry coastal-unmasked east
  donor. The donor-aware result supersedes them.
- Discarding 32 startup steps is retracted as a decisive placement test; a
  representation-consistent prior-stress carry is required first.
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
