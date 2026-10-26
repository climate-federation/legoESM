# PREREGISTRATION — end-wall wind source and placement response

Frozen before the first execution of `endwall_wind_placement.py` on this
branch. The source alignment was read first; no array in the KT lane was
numerically reduced before version 1 was committed.

## Audit trail: four refused runs

No result from these runs is reused.

1. Version 1 put the exact 0x/1x/2x linearity control on the downstream B1
   state. It fired because the intervening barotropic recurrence is
   state-dependent; the premise was invalid.
2. Version 2 moved the control to `surface_stress_faces`, but scaled current
   stress without the previous centred-stress carry. The exact-zero and
   exact-doubling controls correctly failed.
3. Version 3 scaled both stress time levels, then refused on a shape mismatch:
   it stripped a halo from an already aligned `199x52` mask.
4. Version 4 passed setup controls, then refused because it compared unequal
   control-flow spans: NEMO's direct `dyn_zdf` deposit against legoESM's B1
   state after the rotating barotropic loop. The valid lego operand is the
   direct production helper result `rDt*tau/(rho0*dz0)`; B1 is diagnostic.

Version 5 made that operand correction without changing the science bars.
Independent review then found that its prose still called a reconstructed
source operand a measured *placement response*. Version 6 restricts the
offline claim to source-operand agreement. Neither the alternative implicit
arm nor its tridiagonal response runs in the offline probe. The dynamic
placement response and eta ownership require the free-run A/B below.

Version 6 also proves exact 1x entry identity and that 0x/2x change stress
fields only; counts every stress, barotropic and momentum-vmix hook; stamps all
lane-selector environment values; and SHA-256 hashes the probe, current
preregistration, six operands, and every cited NEMO source. This is required
because the DINO `MY_SRC` files are not pinned by the oracle Git revision
alone. No science threshold changes.

Version 6's first rerun produced **no valid measurement**: the strengthened
counter found two production helper calls, not the preregistered one. Source
tracing shows that active leapfrog evaluates `_step_impl` first on Nnn for the
advective/barotropic pass and again on Nbb for the dissipative pass whose
barotropic result is discarded (`ocean_model_latlon_cgrid.py:8396-8455` and
the following Nbb pass). Version 7 registers the live order as
`stress(Nnn), barotropic(Nnn), stress(Nbb), final momentum-vmix`. The source
operand is captured from the first call, whose explicit tendency feeds the
live barotropic pass. Swapping the first two event labels is the planted
violation and must fail the order equality. No score from the refused run is
reused and no science threshold changes.

Evidence re-review found that version 7 did not exercise the allocated restart
shape. Version 8 registered a controlled offline copy:
load `utrd_tau/vtrd_tau` from `DINO_00005764_restart.nc`, require their emitted
3-D arrays to be exact zero, copy them, populate only top level `k=1` with
`(poststress-prestress)/rDt`, require every lower level to remain exact zero,
and feed all source scoring through `rDt*wired_slot[k=1]`. Controls plant a
nonzero in the emitted slot, a nonzero at a lower level, and a material top-
level reconstruction error; each must fire. The restart file joins the hashed
inputs. This changes the data route, not the registered quantity or bars.

Final evidence review found that version 8's words “actual offline wiring” and
`jpdyn_tau wiring` incorrectly identified that controlled copy as NEMO's named
diagnostic. **Those labels are retracted.** Active DINO `trddyn` defines the
named diagnostic without the MLF half factor and with Kmm thickness; the
offline copy contains only the independently dumped applied `dynzdf` bracket
divided by `rDt`. Version 9 renames the routine/output to “applied-term
slot-shape copy” and prints the retraction. No operand, number, bar, or prior
classification changes; named-diagnostic wiring is measured only by
`verify_nemo_endwall_tau_slot.py` under its separate preregistration.

## Round-2 correction and one-cell preregistration

Cross-family review **RETRACTS** the live-slot headline
`CONFIRMED_NAMED_AND_APPLIED_DIFFER` and its value 2.0966766111. On wet U
faces, `zDt_2=rDt/2` makes the named `trddyn` diagnostic exactly twice the
applied `dynzdf` increment; the prior union mask admitted 324 dry coastal
faces and the normalized-difference REFUTE band was unreachable. The only
retained claim is plumbing: the patched named slot is `CHECKED-CLEAN`, and the
named diagnostic is twice the applied increment on every wet face to `1e-8`.

Before the round-2 execution, the existing probe is extended with a one-cell
localizer. It reuses the same kt=5761 arrays, bridge, `umask`, T masks, and
NEMO-to-lego U offset; it does not define a second loader. For both direct
`dynzdf` increment and captured `F_slow`, it prints:

1. the wet `j=1` argmax `(j,i)`, signed delta, NEMO value, lego value, and
   `max(abs(delta))/abs(mean(NEMO[j=1,wet]))`;
2. the participation number
   `P=(sum(delta**2)**2)/sum(delta**4)` on the 49 wet `j=1` faces and on all
   9,758 wet U faces;
3. the count and coordinates for which
   `abs(lego/NEMO - 1) > 0.01`, restricted to finite, nonzero NEMO wet faces;
4. the Jaccard overlap of the direct and `F_slow` one-percent outlier sets;
5. for every outlier coordinate, NEMO `umask`, its two adjacent surface
   `tmask` values, the exact `sbcmod.F90:543-544` multiplier
   `(2-umask)*max(tmask_w,tmask_e)`, and whether the point is a coastal-unmask
   point (`umask=0`, multiplier=2); and
6. independently on the full 199x52 U plane, the overlap between non-roundoff
   source residuals and that coastal-unmask set. This full-plane result is
   never merged into the wet-face score.

The expected buried signature is `P=1.0000` and one one-percent outlier over
49 wet `j=1` faces, a direct argmax delta about `3.02e-5 m/s` and normalized
row maximum about 0.25, approximately 24 one-percent outliers over 9,758 wet
domain faces, and identical direct/`F_slow` support. This expectation is
context, not a gate.

The one-cell signature is **CONFIRMED** separately for each operand iff
`P_j1 <= 1.01`, exactly one of 49 wet row faces exceeds one percent, and its
row-normalized maximum is `>=0.20`. It is **REFUTED** iff `P_j1 >= 2.0`, at
least two row faces exceed one percent, or the normalized maximum is
`<=0.01`; otherwise it is **UNRESOLVED**. Common upstream support is
**CONFIRMED** iff direct/`F_slow` outlier Jaccard is `>=0.90`, **REFUTED** iff
it is `<=0.10`, otherwise **UNRESOLVED**.

The coastal-unmask hypothesis is **CONFIRMED** iff at least 90% of the
full-plane non-roundoff residual support is in the exact coastal-unmask set
and the `j=1` argmax itself is in that set. It is **REFUTED** iff at most 10%
overlaps or the `j=1` argmax is not a coastal-unmask point; otherwise it is
**UNRESOLVED**. `non-roundoff` means
`abs(delta) > 128*eps*max(1, max(abs(NEMO)), max(abs(lego)))`, fixed before
the run. The wet-only classification remains separate even if this source
formula acts outside `umask`.

Controls must plant a second argmax-sized residual on a previously
non-outlying wet row face (forcing `P` above 1.9 and the one-percent count to
increase), flip the coastal flag at the measured argmax (forcing the coastal
classifier to reject), and perturb a copy of one support set at a known index
(forcing Jaccard below its original value). Any control failure invalidates
the localization.

## Candidate and inputs

Candidate: NEMO places the centred surface-stress increment inside `dyn_zdf`,
whereas the shipped `nemo_dino_kamm_mlf` card places the same increment in the
explicit momentum RHS before the implicit solve. NEMO independently adds the
same centred stress to barotropic `zu_frc`; this registration treats that as a
separate entry point.

The primary offline state is kt=5761. NEMO inputs are
`RUN_SEQDUMP_D180_1R/zdf_dump_{u,v}1_{pre,post}stress.bin` and
`wnd_dump_z{u,v}_frc_inc.bin`. The legoESM side reuses
`multistep_replay.build_replay_ic`, `post_tendency_stage_birth._load`, and the
production `surface_stress_faces` and barotropic call. It runs the same bridged
state with 0x, 1x and 2x centred stress; every non-stress input must be exact.

The oracle `utrd_store(:,:,:,jpdyn_tau)` /
`vtrd_store(:,:,:,jpdyn_tau)` slot is emitted to the step-5764 restart but
never populated by the DINO dump path. The offline probe preserves that donor,
copies its 3-D shape, and places the applied `dynzdf` term in that controlled
copy. It does not call this NEMO's named diagnostic.

## Exact offline numbers

For the whole wet u-face domain and separately for southern row `j=1`, print:

1. `zdf_increment_err_norm`: RMS of
   `lego rDt*tau/(rho0*dz0) - (NEMO poststress-NEMO prestress)`, divided by
   NEMO RMS, in m/s.
2. Direct-increment correlation and lego/NEMO RMS ratio.
3. `fslow_err_norm`: RMS of
   `(lego F_slow[wind]-lego F_slow[zero]) - NEMO wnd_dump_zu_frc_inc`, divided
   by NEMO RMS, in m/s2. This directly tests premise 4: `F_slow` reaches the
   loop in tendency units without another face-depth scaling.
4. The numerator RMS for items 1 and 3 in their native units.

The exact 0x/1x/2x control is on `surface_stress_faces`: `tau(0x)=0` and
`tau(2x)=2*tau(1x)` must be bit-exact, and a one-ULP planted mutation must make
the equality fail. Meridional forcing must be exactly zero on the oracle side
and no larger than `1e-12` on the legoESM side. Each arm must show the exact
event sequence `stress, barotropic, stress, momentum-vmix`; a different count
or order, non-finite data, or failed entry identity invalidates every score.

## Frozen interpretation

The domain source test **CONFIRMS algebraic source-operand equivalence** when
direct-increment normalized error is `<=0.05`, domain direct correlation is
`>=0.999`, and `F_slow` normalized error is `<=0.05`. On `j=1`, Pearson
correlation is **RETRACTED as a gate** because the NEMO row is effectively
constant. Its replacement is the registered row-scale statistic above:
direct and `F_slow` each match at max-absolute delta divided by absolute NEMO
row mean `<=0.01`, and materially differ at `>=0.20`; values between are
`UNRESOLVED`. The aggregate normalized errors remain descriptive only.

These bars classify source operands only. They do not classify the response
through the implicit solve or sea-surface ownership, and no post-hoc transfer
coefficient will be invented.

## Free-run placement/ownership discriminator — GPU handoff

Review found that the previous registration incorrectly mixed three different
observables: the whole-domain first-eight ratio (2.89), the aggregate-wall
first-sample share (0.852), and a last-half zonal-wall label. It is retracted.
The exact primary number is now one coherent observable from
`eta_flicker_decay.py`: `regions.all.ratio_lego_over_nemo.first8`, the ratio of
the mean of the first eight per-sample area-weighted wet-domain RMS amplitudes
of the **2dt-alternating eta component**. The companion is the same-window aggregate-wall locus,
`wall_share.legoESM.first8.wall`. The certified baseline values are
2.886305221717904 and 0.48541937969116244; NEMO's same-window wall share is
0.0686300413685866. The often-cited 0.8524656 is the first sample only and is
context, not a gate.

Round 2 pins the load-bearing clean explicit control to Git
`4d00f81bb78caa29dd07ab6e3a4081e6d57a4bd2` and these committed artifacts:
`wind_place_explicit.npz` SHA-256
`2845a5c7166baad483f89ae91a1fb1c9b03971a6a63c4e41506615b2a632df5f`,
log SHA-256
`fc4841aea5a8107f8935ecdd7170957f31feda061f8d2f465f8360180de9046d`,
and scorer JSON SHA-256
`452c29a77b36b7da121be48b5b430c039945bcfdcb20e0f714a5dd27b583c9f6`.
The log stamps `dirty_tracked_files=0`, fp64, bridged before-level, seasonal
clock, and `surface_stress_implicit=False`. Its measured control ratio
2.882400001277192 and wall share 0.4855027387838016 pass the frozen band.

- **Control validity:** explicit-placement first-eight whole-domain ratio in
  `[2.60, 3.18]` and first-eight aggregate-wall share in `[0.38, 0.59]`.
- **CONFIRMS ownership:** implicit-placement ratio `<=1.25` and first-eight
  aggregate-wall share `<=0.17` while the control is valid.
- **REFUTES ownership:** implicit-placement ratio `>=2.30` and first-eight
  aggregate-wall share `>=0.38` while the control is valid.
- Otherwise: **UNRESOLVED**.

The active shipped outer integrator is `leapfrog` with the split-explicit
solver, which accepts either placement. The twin harness exposes the existing
config field as a one-variable selector and stamps the resolved model value.
Exact implicit arm:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_implicit.npz \
  --days 5 --bridge-before --save-step-eta --surface-stress-implicit
```

Exact explicit control:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_explicit.npz \
  --days 5 --bridge-before --save-step-eta --no-surface-stress-implicit
```

The artifacts must stamp `surface_stress_implicit=True` and `False`,
respectively. `--save-step-eta` makes the primary `eta` payload exactly 160
per-step fp64 samples, writes relative `t_seconds=2700..432000`, stamps
`capture_every_steps=1`, and refuses a non-fp64 materialized state. Daily eta
is retained separately as `eta_daily`.

Use the already certified fp64 160-sample NEMO comparator. The earlier
extractor command is retracted because `/tmp/dino_eta_waves/nemo_5d` no longer
contains its per-step restart tiles. Refuse unless the retained artifact has
this exact SHA-256:

```sh
test "$(sha256sum /tmp/dino_eta_waves/nemo_5d_eta.npz | cut -d' ' -f1)" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
```

Score each arm against that same comparator with the exact registered tool:

```sh
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py \
  --nemo /tmp/dino_eta_waves/nemo_5d_eta.npz \
  --lego results/dino_1455/wind_place_implicit.npz \
  --out results/dino_1455/wind_place_implicit_flicker.json

.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py \
  --nemo /tmp/dino_eta_waves/nemo_5d_eta.npz \
  --lego results/dino_1455/wind_place_explicit.npz \
  --out results/dino_1455/wind_place_explicit_flicker.json
```

The scorer SHA-256 must remain
`13916d43f75586358eb4bedec603d023433c23bf61eaf77c320e012f4570568c`.
Any missing or changed artifact is a STOP, not permission to re-extract from
the incomplete directory or substitute daily output. No GPU arm runs here.

## Round-2 GPU result and representation-consistent discriminator — preregistered STOP

The clean implicit artifact at the same Git SHA measured ratio
201.41187344917114 and wall share 0.055963889374377046. The original mechanical
bars therefore return **UNRESOLVED**: amplitude fails CONFIRM while locus fails
REFUTE. Its approximately eight-step decaying, diffuse launch anomaly is
consistent with an explicit-consistent bridged state shocked by the implicit
placement, but that interpretation is not yet a finding.

Mechanism review identified a concrete startup representation error, so the
earlier discard-32 follow-up is **RETRACTED as a decisive ownership test**.
It may be run only as a post-hoc/descriptive rescore of the retained artifacts.
NEMO restart `utau_b/vtau_b` are U/V-point fields, but the bridge puts them in
legoESM's T-point `tau_x_prev/tau_y_prev` carry. The first legoESM step centres
that carry with current T-point forcing and interpolates it again. Discarding
the resulting shock does not test the source faithfully.

Before another live arm, add a fail-closed harness selector named
`--bridge-before-stress-tpoint`. It must reconstruct the previous T-point DINO
analytic stress at the restart's own prior seasonal time through the existing
forcing loader; it must not invert `utau_b/vtau_b`, edit prognostic fields, or
change later-step carry behavior. The artifact must stamp
`bridge_before_stress_stagger="T"`, the reconstruction time, and content hash.
First run the committed one-step CPU counterfactual from the identical bridge:
legacy U-as-T carry versus reconstructed T carry, holding all other leaves
bit-identical. It must print the direct and `F_slow` deltas at `(1,49)`, their
wet material-support Jaccards against the donor prediction, and a planted
stagger swap. The representation correction is valid only if it removes at
least 99.9999% of the measured `(1,49)` direct excess and the planted legacy
swap restores that excess within `1e-6` relative error. Otherwise STOP; no GPU
arm is authorized.

Once that CPU gate passes, the coordinator's exact representation-consistent
GPU invocations are:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_tcarry_explicit.npz \
  --days 5 --bridge-before --bridge-before-stress-tpoint --save-step-eta \
  --no-surface-stress-implicit

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_tcarry_implicit.npz \
  --days 5 --bridge-before --bridge-before-stress-tpoint --save-step-eta \
  --surface-stress-implicit
```

Score both complete 160-sample artifacts against the same certified NEMO
comparator with the exact scorer commands above, changing only output names.
The primary remains samples 1--8. Control validity is corrected-explicit ratio
in `[2.60,3.18]` and wall share in `[0.38,0.59]`.
**CONFIRMS placement ownership** iff corrected-implicit ratio is `<=1.25` and
wall share `<=0.17`; **REFUTES** iff ratio is `>=2.30` and wall share `>=0.38`;
otherwise **UNRESOLVED**. Both conditions require amplitude and locus. If the
corrected explicit control is outside its band, STOP without classifying the
implicit arm. Samples 33--40 and 33--160 may be printed as descriptive
companions but cannot change the primary verdict. The selector does not exist
yet at registration. Round 3 implemented it and the committed CPU prerequisite
passed without changing these frozen bars. The two GPU commands are now
authorized for the coordinator, but neither was run in this CPU-only sandbox.

## Round-4 result: frozen STOP, descriptive ownership, and re-freeze

The coordinator ran the two exact commands above. Both artifacts are clean at
Git `9ca58a379afe65f4ee485615622c541c06d5d7c7`, fp64, stable for 160 samples,
and stamp `bridge_before_stress_stagger="T"`, reconstruction time
`15552000.0`, content hash
`b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae`,
and the requested explicit/implicit selector. Comparator and scorer hashes
match the registration.

The corrected explicit arm measured ratio `1.1607251697830108` and wall share
`0.11941867158491266`. Both miss the frozen control bands, so the registered
result is **STOP/control-invalid** and the implicit arm is **not classified**.
The implicit numbers (`200.41106914842686`, `0.05585390128091391`; last-half
ratio `10.545129752846789`, CI90 `[8.67094529695706,12.53280691395405]`) are
descriptive only.

**DESCRIPTIVE pending re-frozen bars:** compared with the round-1 uncorrected
explicit baseline (`2.882400001277192`, `0.4855027387838016`), the corrected
T carry removes `91.46168881885039%` of the excess-over-one amplitude and
`87.81675304445774%` of the excess wall share above NEMO
(`0.0686300413685866`). The bridge representation defect, not wind placement,
owned most of the baseline wall flicker. The old measurements are not
retracted; they describe the old bridge.

### Corrected-explicit control band

The new control band uses the same asymmetric fractional half-widths as the
old band, applied to the measured corrected-explicit baseline. No rounding is
used in the decision:

| quantity | old lower/upper fractional half-width | corrected control band |
|---|---:|---:|
| first-eight ratio | `0.0979739110297185` / `0.10324729343288278` | `[1.0470043852687352,1.2805669019825299]` |
| first-eight wall share | `0.21730616607454992` / `0.21523516320004085` | `[0.0934682579050795,0.14512176885262343]` |

The repaired same-HEAD explicit arm must fall inside both intervals or STOP.

The old CONFIRM/REFUTE bars are transferred by preserving their fraction of
the old explicit-to-NEMO excess. For amplitude the reference is ratio `1`; for
locus it is NEMO wall share `0.0686300413685866`. This gives:

- **CONFIRMS placement ownership** only if repaired implicit ratio is
  `<=1.021345777952874` and wall share is `<=0.08098019376738272`.
- **REFUTES placement ownership** only if repaired implicit ratio is
  `>=1.1109980453549448` and wall share is `>=0.10656501237402145`.
- Otherwise **UNRESOLVED**. Both amplitude and locus are mandatory.

### Mandatory one-step implicit-shock diagnosis

No replacement implicit GPU arm is authorized until a committed CPU probe
identifies and tests the first state inconsistency introduced by the placement
swap. The earlier proposed depth-uniform `du_dt_pert` subtraction is
**WITHDRAWN BEFORE PROBE**: the shipped leapfrog reconstruction and
`mlf_baro_corr` remove/re-pin that column mean, so its eta REFUTE state is not
reachable while `F_slow` is unchanged.

The candidate persistent inconsistent state is the post-`dyn_zdf`,
post-`mlf_baro_corr` zonal-velocity leapfrog carry: the pair `u` (after/NOW on
the next call) and `u_before` (Asselin-filtered BEFORE). The first corrected-T
step is traced for explicit and implicit placement at the centred face stress,
`F_slow` entering the barotropic solver, every substep eta, pre-`dyn_zdf`
`u_naa`, post-`dyn_zdf` u, post-reconcile u, and filtered `u_before`. It must
show equal `F_slow` and substep eta through the first barotropic solve and name
post-`dyn_zdf` u as the first differing persistent state. If an upstream item
differs instead, STOP: this registered carry counterfactual is not run and the
first differing item becomes the next preregistration target.

The discriminating one-step CPU counterfactual is the *next* model step. Its
reference arms are corrected-T explicit and corrected-T implicit. The
counterfactual starts from the implicit arm after step one and replaces only
the zonal-velocity leapfrog pair (`u`, `u_before`) with the explicit arm's
pair; T, S, eta, v, every forcing/carry/config leaf, coefficients, and the
later stress carry remain bit-identical to implicit. A planted arm restores
the original implicit velocity pair byte-for-byte.

The primary is `E = RMS_wet_T(eta_implicit_step2-eta_explicit_step2)` and the
counterfactual removal is exactly `1-E_counterfactual/E`. Material support is
defined on the common T grid: thickness-average each 3-D U field using the
production `min_cell_to_uface` thickness, take the maximum absolute difference
across the NOW/BEFORE pair, then map U faces to T cells by the maximum of the
west/east adjacent face magnitudes. The eta target is
`abs(eta_implicit_step2-eta_explicit_step2)`. Each support is cells at or above
`1%` of its own wet-domain maximum; dry T cells are excluded. Jaccard is
intersection/union of those two Boolean supports and is a STOP if either
support is empty. The retained first-step max-absolute controls are
`0.0012013470296322257 m` implicit versus NEMO and
`7.03693132994565e-06 m` explicit versus NEMO.

- **CONFIRMS this shock source** iff the carry substitution removes at least
  `99%` of `E`, the planted implicit-pair restoration reproduces `E` within
  `1e-6` relative error, and support Jaccard is `>=0.99`.
- **REFUTES** iff the carry substitution removes `<=10%` of `E` while the
  planted restoration passes.
- Otherwise **UNRESOLVED**. All other state leaves and build inputs must be
  bit-identical, with planted state/config and decision-bar failures.

Only a passing CPU diagnosis may authorize a same-HEAD explicit/implicit GPU
rerun and application of the re-frozen placement bars above.
