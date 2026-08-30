# DINO transfer question — T3 certified, T2 scoped, T1 reducer repaired

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

T3 is **TRANSFER_CONFIRMED**. T2 is
**PERPETUAL-FE-OUT-OF-MODEL-CLAIM-BY-USER** and has no science verdict. The
forward-Euler crash is localized to the live continuity/surface-pressure fast
pair under a plain temporal boxcar. Repeating NEMO's first-Euler nn_bt_flt=2
AB3/AM4 temporal composition passes the frozen 40-step discriminator and the
64-step clean-card gate, and remains the correct default for NEMO's actual one
Euler bootstrap. No further perpetual-Euler stabilization is in scope. T1's
first wave reached 360 cold-start days cleanly before a reducer-shape bug
aborted its first capture; that reducer path is now repaired and admitted at
day 0. No GPU, NEMO, MPI process, push, or remote action was launched here.

## T3 — catalog construction is certified

The public `nemo_dino_kamm_mlf_v1` recipe and the oracle
`nemo_dino_kamm_mlf` card resolve identically. The config-identity artifact is
`/tmp/dino-transfer-01a053d4/t3/config_identity.json`, SHA-256
`cae74e17dd2e3c60f592c203b398a9d1d15790532be1387090ad9dc35fdd27fe`.
It reports `verdict=PASS`, zero identity-difference rows, a fired one-ULP
`config.asselin_gamma` plant, a fired `outer_integrator` ownership-collision
plant, and a nonempty `nemo_dino_v1` negative control.

`OWNERSHIP_COLLISION=FIRED` is the planted negative control, not a live
collision. The JSON has no live `ownership_collisions` row and records
`ownership_collision_plant={field: outer_integrator, fired: true}`. A live
collision would have made `verdict=PASS` impossible.

The five-day behavioral artifact is
`/tmp/dino-transfer-01a053d4/t3/behavior_identity.json`, SHA-256
`f52990b76afe44cc5a045fd3d2d8e7c10c311a79838bc83296fd6b873cf28169`.
It reports `verdict=PASS`, `bit_differences=[]`, no missing or extra keys,
equal normalized run configs, all required construction receipts present, and
49 compared keys. The oracle artifact SHA-256 is
`d94fd2370413c151723a9986df046d9a5d1051202802b34de2f21b7f9e725f52`;
the catalog artifact SHA-256 is
`fa778f3ed225955cf0f2ae9ac523d48b07b5763840d05fb7d7387726b65c8983`.

Therefore the catalog-built model is bit-identical to the oracle-card model
under the registered bridge and five-day behavioral check. This transfers
construction-path identity for the certified MLF card. It does not extend the
round-94 bridge claim to a standalone start or another card.

## T2 r2 — historical exact crash and owner

Both corrected `nemo_dino_kamm` climate arms stopped before writing an
artifact. Their logs are
`/tmp/dino-transfer-01a053d4-t2r/logs/fe_climate_r2_a.log`, SHA-256
`e07de38ab808ce86d46ba813b488b3c4661c9bed9b5627eeb5479685971db664`,
and `fe_climate_r2_b.log`, SHA-256
`637b769cba5c81dd6f3080d22ad3322e64d29bb65be6bc986258a23a80ce4452`.
Each stamps the intended FE admission bundle: `outer_integrator=forward_euler`,
`barotropic_diffusion_alpha=0.01`, and the generic Nbb/Kaa selectors. Each
reports day 1 with `|eta|max=525.7945 m` before the asynchronous Equinox check
surfaces `raw-mesh e3w_int must contain only finite values > 0`.

The committed CPU reproducer is
`scripts/validate/ocean_fidelity/dino_1226/fe_stability_repro.py`, SHA-256
`d8d8be9e0d2b057260dcebdd9a3c77c370d16005a26cf3b0f26864ee68289fae`.
It ran from clean producer `fb46fe02f0e0a9ce7414966b57c375ae9402c673`
with CPU, fp64, `JAX_DISABLE_JIT=1`, the same restart, ladder, bridges,
seasonal forcing, and no selector override. Its JSON receipt is
`/tmp/t2-fe-r2-cpu-repro.json`, SHA-256
`3adc8ba20f26dec0631b7118c5f881abd3f630391c808067ccf7725435a5281e`;
the complete log SHA-256 is
`24d0c63f2208a476d70e011587a341de00e6964a57b638a84d332042b0d0489b`.

The replay completes steps 1--10, so the former five-step control did not
exercise the failing window. `|eta|max` is `0.825653 m` at step 1,
`0.834720 m` at step 10, `10.389009 m` at step 22, `684.484519 m` at step 29,
and `525.794486 m` at step 32, exactly matching the GPU day-1 SSH maximum.
Velocity then reaches `|u|max=214.312787 m s-1` at step 35. Step 36 raises the
unmasked exact exception:

```text
equinox._errors.EquinoxTracetimeError:
raw-mesh e3w_int must contain only finite values > 0
```

The immediate raise owner is
`compute_buoyancy_frequency_nemo_bn2` (`eos.py`) called by
`compute_nemo_native_slopes` in the GM/Redi live-QCO geometry path. That check
is behaving correctly and is downstream: the permanent-FE trajectory has
already developed hundreds of metres of SSH and order-100 m/s velocity. The
upstream owner is therefore the forward-Euler/split-explicit integration of
the admitted Kamm sibling bundle. These data do not isolate which remaining
FE/barotropic composition choice causes the unstable mode, so another selector
patch is not licensed.

The earlier statement that alpha `0.01` "keeps FE stable" is **retracted**.
It only keeps five steps bounded and delays the same failure. The MLF-only
Nbb/Kaa guards and the FE alpha restoration remain valid admission/scope
fences, but they are not an FE stability fix.

## T2 historical stop disposition (superseded by the authorized continuation)

At that checkpoint T2 was **BLOCKED-NEEDS-FE-STABILIZATION**. It was not
`CONFIRMED` or `REFUTED` by
the registered climate statistics, or science-scored: no climate, wall, or
score artifact exists. The 360-day and five-day GPU blocks are withdrawn from
the handoff and must not be relaunched from this producer. A future FE-specific
round must diagnose and preregister the unstable free-surface/barotropic
composition, demonstrate stability beyond this step-36 horizon with a control
that can fail, and only then reissue the frozen twin battery.

This is already a real transfer-scope finding: the certified MLF configuration
does not transfer to a runnable forward-Euler sibling under the currently
admitted faithful defaults. T3 remains complete, and T1's standalone build
round can proceed independently.

## Block-0 admission correction

The first rerun stopped in
`TestSurfaceTendencyPlacement::test_retention_synthetic_violation_both_directions`
before either registered retention assertion. The production surface-placement
path had not regressed, and none of the T2 card fences caused the exception.
The synthetic ten-column fixture inherits the faithful MLF card but has no raw
bridged NEMO `e3t_0`; commit `d1cf21b425` later promoted
`gm_redi_flux_face_thickness_evaluation="nemo_qco_live"` without adding the
matching generic fixture override. The resulting exception was
`redi_flux_face_thickness_evaluation='nemo_qco_live' requires raw NEMO e3t_0`.

The test setup now pins
`gm_redi_flux_face_thickness_evaluation="tpoint_jacobian"`, paired with its
existing synthetic-grid slope and QCO overrides. Production code is unchanged.
The isolated test passes and still executes both original plants:
`applied_now` must retain below `0.6`, while `leapfrog_rhs` must retain above
`0.6`. The complete Block-0 admission suite passes `178/178` after this
fixture repair. That admission result never licensed a long-horizon FE
stability claim.

## #1696 selector bisection — regression premise refuted

The committed `PREREG_dino_fe_default_scope_bisect.md` froze a paired causal
gate before the new CPU arms. Restoring all sweep-promoted barotropic,
TKE/ZDF, Redi, wind and V-metric selectors to their pre-sweep evaluation paths
did not rescue the current FE card. It reached `|eta|max=3999.5 m`,
`|v|max=9.54e127 m s-1` at step 36 and raised the same raw-`e3w_int` error at
step 37. Receipt SHA-256 is
`6392216940040b02184f1c2d4a3da99adab1dfda55a7f833f7bc3a0318cbe4d1`;
log SHA-256 is
`a70c489bae288b657113401ee974928d66a6fd78e7bfaf0c320b07e5c9fa6fed`.

The clean pre-#1696 target `b794c0618e287ebf1d364a8713c3c504ac2eb01c`
was then run through its original committed twin harness with the same
developed state, before bridge, TKE bridge, fp64 and CPU/no-JIT policy. It was
already fully nonfinite at the first day checkpoint, step 32. Its artifact and
log SHA-256 values are
`9f97a8ba82800672116a7f60c1cb1043dfc1aab7b51ab230311bced1284f2839`
and
`4f5f128c7ea83ebfe3d81a41da4b29146b4d744e0862a573b3c9f04605218464`.

Thus no promoted selector passes the preregistered activation/restoration
causality gate, and no new FE default was withdrawn. The proposed #1696
regression mechanism is refuted; the actual finding is pre-existing permanent-
FE developed-state instability. The already-bound MLF-only fences for alpha
and the Nbb/Kaa QCO pair remain valid. No green post-step-32 FE regression test
exists to add until FE stabilization itself is implemented.

## T2 authorized continuation — causal owner and admitted stabilization

The executed DINO source establishes the scope boundary. `RUN_TRAJ/namelist_cfg`
sets `nn_it000=1` at line 94 and `ln_rstart=.false.` at line 101. The from-rest
branch sets `l_1st_euler=.true.` (`src/OCE/DOM/istate.F90:107-110`), initializes
velocity at rest (`:121-123`), and copies Kbb into Kmm (`:135-137`). The built
`MY_SRC/stpmlf.F90` selects `rDt=rn_Dt` at lines 134-137, still calls
`mlf_baro_corr` at line 578, and only at lines 685-688 restores `2*rn_Dt` and
clears the Euler flag. NEMO therefore takes exactly one Euler outer step and
then MLF; it has no perpetual-Euler oracle trajectory.

That one Euler step is not a plain forward/backward barotropic loop.
`MY_SRC/dynspg_ts.F90:202-208` initializes the boxcar/temporal histories;
lines 241-251 choose the first-Euler forward window; lines 546-553 reset the
history; and lines 766-780 run `ts_bck_interp` before each surface pressure
gradient. Lines 718-725 are the live continuity update, while lines 1123-1127
return the barotropic increment to the 3-D momentum RHS. Repeating this
first-Euler composition is a source-grounded stabilization, not an
oracle-faithful trajectory.

The committed CPU probe wrapped the production kernels with diagnostic-only
callbacks over outer steps 25-35. The plain `nemo_boxcar_centred` control
reproduced the step-36 failure. At step 25 it recorded:

- `|eta|max=29.4297 m`;
- live surface-PGF tendency `0.90269 m s-2`;
- continuity divergence `22.1698 m s-1`;
- frozen slow tendency `3.3084e-5 m s-2`;
- EEN Coriolis tendency `2.308e-4 m s-2`.

By step 29, SSH is `684.48 m` and surface PGF is `3.478 m s-2`. Thus the
continuity/surface-PGF gravity-wave pair injects the growth; the slow tendency
and Coriolis are subleading through onset. Both the column-mean-deposit plant
and the fast-term callback plant fire.

The preregistered corrector hypothesis is refuted. Repeating
`mlf_baro_corr` reduces the after-solve column-mean deposit but still raises the
same raw-`e3w_int` check at step 37. Its experimental FE wiring was removed.

The next frozen arm changes exactly one resolved field:
`barotropic_time_filter=nemo_boxcar_centred` to `nemo_boxcar_ab3`. It completes
40 steps finite. The shipped `nemo_dino_kamm` card now pins that value and,
without any probe override, completes the registered 64-step gate. At step 64,
`|eta|max=0.852169 m`, `|u|max=0.898959 m s-1`, and
`|v|max=1.092863 m s-1`. The old plain-boxcar receipt is the non-vacuous red
control; a five-step synthetic test would not reach the registered failure and
is intentionally not substituted for it.

Evidence SHA-256 values:

- plain-filter fast-term JSON/log:
  `c2623575caf21283f0abe820873f94611f5d401cd3f32e611fd2cad967dcf26c` /
  `15936968c8cb9197b6ec8568d369c64bbb8bf4d004c425c05096292274790a38`;
- refuted corrector JSON/log:
  `5ec1765659fc68d3f56f3cd0255b027b162734cc9e66eeedde82f90b8f84a91f` /
  `e6a8394dbcc69547e27f78c001b19b38ed156cf29610a86bc1aec3d371be7223`;
- one-field AB3/AM4 40-step JSON/log:
  `eb7a378a8eb5c85294e0591db17927bfc1f52fb12c6e7fbf3700aebef794c92e` /
  `7c91c5c77732b0f979a05de477f3ff744aaecab03226187c9a8e0902c90c7a2d`;
- promoted-default 64-step JSON/log:
  `1d2da9994bc18246a05de5ac20ad5d333ee1e42b83447e2cbac18afdb735ef38` /
  `1d5649ffe67344ace9125be863e5baf2f8447b21bb866d8c562a9fe2cbd5e724`.

The 64-step result remains an engineering receipt, not the target claim. The
user's final scope decision is that only NEMO-covered integration matters:
one Euler bootstrap followed by leapfrog. Perpetual-Euler longevity is
explicitly out of scope. The AB3/AM4 fix stands because it matches NEMO's
actual Euler treatment; no further FE stabilization or perpetual-Euler
fidelity verdict will be pursued.

T2 is disposed as `PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER`. One five-day FE
probe may be run as the cheap admission check. A 360-day duplicate may run only
after that probe writes `stable=True`, and would be descriptive diagnostics
only. The offline T2 fidelity scorer is withdrawn because NEMO has no
perpetual-Euler trajectory against which it could issue an oracle-backed model
claim.

## T1 build round — cold-start row closed; legoESM arms ready

`standalone_20y.py` now constructs the public `nemo_dino_kamm_mlf` card from
legoESM's own analytic NEMO-grid DINO grid, vertical coordinate and from-rest
state. It has no restart or bridge selector, owns the six-member temperature-
only perturbation contract, streams the exact 75 fp64 dates, imports the live
campaign reducer, and hashes configs, initialization, snapshots and reductions.
`standalone_20y_score.py` implements the six-family, six-member, 20,000-draw,
seed-1455 horizon-matched floor/classifier with quantization and simultaneous-
maximum plants.

The first CPU smoke initially exposed missing raw literal-kernel operands. The
runner now constructs those vertical, face, metric and EEN operands from the
public analytic DINO grid rather than a NEMO file. A real one-step CPU/no-JIT
smoke completes with every prognostic field finite and stamps
`claim_admissible=false` as required for a truncated run.

The first human GPU attempt then completed its initial 360 from-rest days with
a clean, finite model trajectory, but aborted at the first scheduled capture:
`live reduction failed at day 360: shapes (199,52,36) vs (195,48,1)`. This is
a user-reported run receipt; the abort occurred before the runner wrote a
day-360 snapshot or manifest, so there is no citable artifact hash. It proves
only 360-day cold-start runnability, not a T1 statistic or verdict.

The failure was diagnostic geometry, not model state. The recorded reducer
owns NEMO's `(199,52,36)` native T frame; the standalone model owns the
physical `(195,48,35)` core, with U `(195,49,35)` and V `(196,48,35)`.
`standalone_20y.py` now preserves the recorded reducer verbatim and supplies a
stamped `dino_standalone_reducer_frame_v1` adapter: model scalars map to
`[2:-2,2:-2,:-1]`, two periodic zonal halo rings are restored, NEMO's dry
terminal `jpk` level is padded, and U's `[:,1:]` east/native faces map to the
reducer's selected U core.

Before step 1, the runner now asserts the exact shapes of T, S, eta, U, V, and
land mask, asserts every imported mesh-weight shape, fires a planted land-mask
mismatch, and executes the real live reducer. The clean CPU receipt from
producer `8209506a38c91aff9e293192eb6c18ff37118ff4` reports
`REDUCER_DAY0=PASS`, all 12 reduction keys, and a finite one-step state.
The final focused CPU suite passes `196/196`.
SHA-256 values are:

- reducer convention receipt:
  `afcbaf82948e50747f2cf66911d758bfdcfba23cf60062e4380356542d4adcfc`;
- one-step manifest:
  `917a9020e8b008244dea8b371da941fded9ceff8a18ec6b638a9bbec4cfd7ecf`;
- one-step log:
  `7839ef464a7b96d734efb9fdd8c9f73e2c2788b64aa2ebf2a9a678f7eedb9992`.

The remaining start-path row is now closed against the executed oracle source.
For a non-restart start, `src/OCE/DOM/istate.F90:107-110` enters the from-rest
branch and sets `l_1st_euler=.true.`; lines 121--123 initialize velocity at
rest, and lines 135--137 copy `Kbb` to `Kmm`. In the built DINO override,
`cfgs/DINO/MY_SRC/stpmlf.F90:134-137` consequently selects the one-`rn_Dt`
Euler step. The corrector call at line 578 is guarded only by
`ln_dynspg_ts`, not by `.NOT.l_1st_euler`:

```fortran
IF( ln_dynspg_ts ) CALL mlf_baro_corr ( kstp, Nnn, Naa, uu, vv )
```

`l_1st_euler` is cleared later, at `stpmlf.F90:685-688`. Thus the call is live
on `nn_it000`; the reduction/replacement itself is at lines 752--765.

legoESM now exposes
`barotropic_cold_start_after_reconcile={off,nemo_mlf_baro_corr}`. The DINO MLF
oracle card and `nemo_dino_kamm_mlf_v1` catalog card both select the faithful
value. Unknown values, non-MLF use, and a regular/cold selector mismatch fail
at model construction. On the no-history bootstrap, `_step_impl` captures the
pre-mixing barotropic target and raw pre-projection Kaa SSH, applies implicit
mixing, then invokes the existing shared `_apply_after_level_reconcile` kernel
before the conservation fixer. The retired skip warning no longer exists.

The pre-change claim-admission test failed on the registered refusal. It now
passes by resolving both selectors from the shipped card; the off plant moves
the committed U and V means on both leapfrog-family paths, and ordering tests
observe exactly `implicit -> reconcile`. Clean-producer config identity remains
`PASS` with zero diff rows (artifact SHA-256
`c8f885a669734ed03f9bfd769eca5e367488bb34dcc21a1b52f07f676de73fe1`).
A CPU/fp64/no-JIT standalone step from producer
`28be310c6ab487c3fa685063af6554abea5938f2` is finite and stamps the faithful
cold-start selector, empty bridge/restart paths, and zero admission blockers.
Its manifest SHA-256 is
`41c044ad8e13e1ce9745a94b2d0724451b32d113f4ed1c74f74434d28217ebc1`;
`claim_admissible=false` means only that this receipt intentionally ran one
step rather than all 230,400.

The six legoESM 20-year arms must be rerun from fresh output directories with
the repaired runner. T1 remains **UNMEASURED** until those arms, the fresh
from-rest NEMO ensemble, normalized six-family statistics, and the science
score all exist. The scorer cannot be invoked until both roots contain the
registered `dino_standalone_20y_statistics_v1` products.
