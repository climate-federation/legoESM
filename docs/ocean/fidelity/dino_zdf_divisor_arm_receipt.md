# Receipt — NEMO's `e3w(Kmm)` is now the DINO twin's implicit-mixing divisor

Branch `fidelity/dino-zdf-divisor-scaling`. Preregistration:
`dino_zdf_divisor_arm_preregister.md`, committed at `5c6ee4ebb` **before** the
change was written. Change: `d80df3ed7`. Probe update: `c24b42d72`.
Measurement: `dino_zdf_divisor_scaling.md` (the study this arm ships).

Everything below is CPU, fp64 (`PrecisionPolicy.fp64()`; the probe printed
`precision policy control dtype = jax.numpy.float64` and every geometry array
as `float64`), on the twin's production ladder `LEGOESM_NEMO_E3T=both`.

## Verdict

The certified DINO twin now divides by NEMO's own `e3w(Kmm)` **exactly** —
`nemo_e3w_kmm / (e3w_0·(1+r3t)) − 1` is `0.0000e+00` at max over 347,200 wet
entries, fp64. The defect removed was `+0.30%` in the median and `+12.7%` at
the deepest wet level. The 5-day twin moved by `6.8e-3 K` at the surface,
inside the preregistered CONFIRM band and within 16% of the scaling study's own
`8.13e-3 K`. Every contamination control is exactly `0.0`, the resolved run
config is byte-identical, and the LOCK_EXCHANGE RK3 card is bit-identical.

## 1. What changed, and which cards it reaches

One canonical symbol,
`packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py::nemo_e3w_kmm`,
produces `e3w(:,:,jk,Kmm)`; the tracer solve and the momentum solve both call
it (`e3uw_0 == e3w_0` on the zco branch, `zgr_lib.F90:111-112`), so there is one
object where there used to be a branch per solve.

Oracle lines, read not inferred:

```
TRA/trazdf.F90:219-221              zwi = -p2dt*zwt(jk  )/e3w(ji,jj,jk  ,Kmm)
                                    zws = -p2dt*zwt(jk+1)/e3w(ji,jj,jk+1,Kmm)
                                    zwd = e3t(ji,jj,jk,Kaa) - (zwi+zws)
cfgs/DINO/MY_SRC/dynzdf.F90:200-203 zzwi = -zDt_2*(avm(ji+1,jj,jk)+avm(ji,jj,jk))
                                         / (e3u(...,Kaa)*e3uw(ji,jj,jk,Kmm))*wumask
cfgs/DINO/MY_SRC/zgr_lib.F90:111-112   pe3uw(:,:,:) = pe3w(:,:,:)   ! zco
DOM/domzgr_substitute.h90:131          e3w(i,j,k,t)  = (E3w_0(i,j,k) Time(r3t,i,j,t))
DOM/domzgr_substitute.h90:132          e3uw(i,j,k,t) = (E3uw_0(i,j,k) Time(r3u,i,j,t))
DOM/domzgr_substitute.h90:108          E3w_0(i,j,k) = e3w_3d(i,j,k)      [key_vco_3d]
DOM/domzgr_substitute.h90:49           Time(r3,i,j,t) = *(1._wp+r3(i,j,t)) [key_qco]
DOM/domqco.F90:160                     pr3t = pssh * r1_ht_0
```

Time levels, read at the call sites (Rule 1d):

```
MLF  cfgs/DINO/MY_SRC/stpmlf.F90:551   CALL tra_zdf( kstp, Nbb, Nnn, Nrhs, ts, Naa )
MLF  cfgs/DINO/MY_SRC/stpmlf.F90:396   CALL dyn_zdf( kstp, Nbb, Nnn, Nrhs, uu, vv, Naa )
RK3  src/OCE/stprk3_stg.F90:598        CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts, Kaa )      [kstg==3]
RK3  src/OCE/stprk3_stg.F90:430        CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa )  [kstg==3]
RK3  src/OCE/stprk3.F90:207            CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )
```

Both identities take `Kmm` = NOW for the divisor and `Kaa` = AFTER for the
diagonal, so RK3 needs the same object, not a different one — which is why the
divisor is bound to the routine's identity rather than to an integrator.

### Reach table

| card | selects the NEMO identity? | divisor BEFORE | divisor AFTER | moves? |
|---|---|---|---|---|
| DINO `nemo_dino_kamm_mlf` (certified twin) | yes (`zdf_implicit_solver_evaluation="nemo_literal"`) | midpoint · (1+eta_AFTER/H) | `e3w_0·(1+r3t(Nnn))` | **YES**, disclosed in §3 |
| DINO `nemo_paper`, `nemo_dino_kamm` | no (`shared_thomas`) | midpoint | midpoint | no |
| DINO `legoesm_default` | no | midpoint | midpoint | no |
| LOCK_EXCHANGE-zco (RK3) | yes | `build_dz_half(e3t_now)` | same expression (uniform ladder, no NEMO mesh `e3w_0`) | **no — bit-identical, measured** |
| OVERFLOW-zps (RK3) | yes | `build_dz_half(e3t_now)` | same expression | **no — bit-identical, measured** |
| NEMO GYRE (RK3) | yes | `build_dz_half(e3t_now)` | `e3w_0·(1+r3t)` (it carries a NEMO mesh on a stretched ladder) | **YES, but UNMEASURED — the card does not construct** |
| ORCA1, every catalog recipe | no (`shared_thomas`) | midpoint default | midpoint default | no |
| `veros_faithful_v1` | no (Veros `implicit_vmix_dzw_slot`) | Veros `dzw` | Veros `dzw` | no |

### Config fields

| field | before | after |
|---|---|---|
| `implicit_vmix_e3t_now_divisor` | `bool = False`; opt-in NEMO-ish arm that fixed only the TIME LEVEL and kept the midpoint SLOT; set by the NEMO test-case cards, never by DINO | **REMOVED.** NEMO has no such switch; the divisor belongs to the routine's identity |
| `zdf_implicit_solver_evaluation` | selected the literal `dyn_zdf`/`tra_zdf` solver recurrence only | **MEANING WIDENED**: `"nemo_literal"` now also carries NEMO's `e3w(Kmm)` divisor. Every card already on it gets the divisor unbranched |
| `implicit_vmix_dzw_slot` | Veros `dzw` slot | **UNCHANGED** — referenced arm, selected by `veros_faithful_v1` (`recipes.py:99`), cited to Veros `thermodynamics.py:267`. Not deleted |
| (none) | — | **no new public field** |

Two construction guards moved with the field. `outer_integrator="nemo_mlf"`
now hard-requires `zdf_implicit_solver_evaluation="nemo_literal"` instead of the
deleted flag (same requirement, expressed on the surviving selector). The old
"requires `implicit_vertical_mixing=True`" raise went with the field and was NOT
re-added, and the old mutual-exclusion raise was NOT retargeted — see §6.

## 2. Is the shipped divisor NEMO's? (the decisive row)

`scripts/validate/ocean_fidelity/dino_1226/dino_zdf_divisor_scaling.py`,
day-180 bridged NEMO restart, fp64. The reference side is built INDEPENDENTLY
from `RUN_TRAJ/mesh_mask.nc`'s own `e3w_0` and the state's `eta`, not from the
coordinate the model uses:

```
RESOLVED CARD   zdf_implicit_solver_evaluation = 'nemo_literal'
                implicit_vmix_dzw_slot         = False
                implicit_vertical_mixing       = True
                outer_integrator               = 'leapfrog'

SHIPPED DIVISOR  nemo_e3w_kmm / (e3w_0*(1+r3t)) - 1 :
                 max|0.0000e+00|  median|0.0000e+00|  n=347200  dtype=float64
TRUE NEMO GAP    midpoint/e3w_0 - 1 : max|1.1280e-01|  median|3.0343e-03|
```

so the divisor the card now runs is NEMO's array, bit for bit, and the error it
replaced was 0.30% in the median and 11.3% at worst. Per level, the deepest wet
level is where it lived: NEMO `e3w_0 = 592.638 m` at k=34 against a midpoint of
`525.789 m`.

On the stricter both-neighbours-wet interface mask the scaling study used, the
removed error is `max 8.965e-03 / median 2.935e-03 / mean +2.395e-03` over
332,214 interfaces — the same numbers that study recorded, unchanged (it is a
pure geometry ratio).

**Independent cross-check.** The probe's C1b control now compares the shipped
card against a midpoint-divisor arm and reports a one-step `max|dT| =
1.129e-04 K`. The scaling study's one-step arm difference, measured before this
change existed, was `1.129e-04 K`. The shipped code reproduces the measured arm
to every printed digit.

## 3. The 5-day DINO twin, BEFORE code vs AFTER code

`kamm_twin_90d.py nemo_dino_kamm_mlf --days 5 --bridge-before`, byte-identical
invocation on both sides, one variable: the commit. 160 leapfrog steps from
NEMO's day-180 restart; 196 s and 159 s wall.

| field | day-5 max abs | day-5 rms | preregistered CONFIRM band | verdict |
|---|---|---|---|---|
| sst (T) | **6.815e-03 K** | 1.143e-04 | `[1e-3, 2e-2] K` | **INSIDE** |
| u | **1.236e-02 m/s** | 3.007e-04 | `[3e-3, 6e-2]` | **INSIDE** |
| v | **3.069e-03 m/s** | 7.661e-05 | `[1e-3, 3e-2]` | **INSIDE** |
| eta | 3.216e-05 m | 1.070e-06 | not preregistered | — |
| S | **not stored** by the twin archive | — | `[1e-4, 3e-3]` | **UNMEASURED** |

Per-day surface `max|dT|`: `7.57e-04, 3.19e-03, 3.05e-03, 1.26e-02, 6.82e-03` K.

Neither REFUTE condition fired (`< 3e-4 K` would have said the change does not
reach the executing path; `> 5e-2 K` would have said something else moved).

**Reduction caveat, stated because the two numbers look comparable and are
not.** The scaling study's `8.13e-3 K` is a 3-D fp64 maximum over the whole
domain. The `6.8e-3 K` here is a **float32 SURFACE slice**, which is what the
twin archive stores (`storage_dtypes: sst/eta/u/v = float32`, no 3-D block).
Different staggering-free reduction, same order, 16% apart — a consistency
check between two instruments, not the same measurement twice.

```
sha256 fa8793dd024de35a3093fd0804554099f1a146d574daef78e6163351b36d77a2  twin_BEFORE_d5.npz
sha256 75e5b245666c192e86f812eaae938104cfbd895d1703cab0694681aaf05a38ba  twin_AFTER_d5.npz
sha256 2336e897bc6bc482b21ca6f3fbe0ab8a2948c8a4b14d16832f515365bad0e214  sweep_BEFORE_LOCK.json
sha256 8ae251f080db569751d3316498cc4a34da228f01227dc197fe0fc9282b46b42d  sweep_AFTER_LOCK.json
sha256 4b81acb0ff29667a82d524df49645aa75d9063d87f6253c448db15c298f60bc9  sweep_BEFORE_OVERFLOW.json
sha256 0fd98b21807e31b9ff2366ef752f9b2551215c7548ababfd800e84ab63d94db6  sweep_AFTER_OVERFLOW.json
sha256 62e789de3d272282c7c996a6519b263627147ea16133f7158d781e97b12d832c  divisor_shipped.log
producer_git_sha  BEFORE 5c6ee4ebb8fb5bedefe1387d4e248ac285f80b62
producer_git_sha  AFTER  c24b42d725d580f9e6fd82384f9e57943f750485
```

Artifacts live in `/tmp/wt-dino-zdf-arms/` (runtime outputs are not tracked).
Commands, verbatim, from the repository root of the worktree:

```
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:\
packages/ice:packages/land:packages/ml:packages/tools:src \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf <out>.npz --days 5 --bridge-before

... JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_stage_sweep_gate.py \
  {LOCK_EXCHANGE-zco,OVERFLOW-zps} --output <out>.json

... LEGOESM_NEMO_E3T=both .venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/dino_zdf_divisor_scaling.py --states restart
```

### Contamination controls — every one at exactly 0.0

| control | required | measured |
|---|---|---|
| `initial_state_sha256` (day-0 bridged state) | identical | **identical** |
| `vertical_ladder_sha256` | identical | **identical** |
| resolved `run_config` string (2357 chars) | identical | **byte-identical** |
| `land_mask` | `0.0` | **0.0** |
| `bridge_before_stress_sha256`, `bridge_f_T/u/v_sha256`, `bridge_omega_rad_s`, `seasonal_t0_seconds`, `rn_Uv` | identical | **identical** |
| probe C3 / C4 contamination controls | `0.0` | **0.0 / 0.0** |

The byte-identical `run_config` is worth naming: the change is invisible in the
run's own configuration record, because it removed a field rather than adding
one. That is the "no new public field" property, and it also means a
config-diff between an old run and a new one will NOT show this — the commit is
the record.

## 4. Gates

| gate | BEFORE | AFTER | verdict |
|---|---|---|---|
| LOCK_EXCHANGE-zco stage sweep, kt=1..10 | status DEBT | status DEBT | **BIT-IDENTICAL** — every one of the 20 JSON sections equal after stripping `legoesm_git_sha` (which differs: `5c6ee4ebb` vs `d80df3ed7`) |
| OVERFLOW-zps stage sweep, kt=1..10 | status DEBT | status DEBT | **BIT-IDENTICAL** — every JSON section equal after stripping `legoesm_git_sha` (`5c6ee4ebb` vs `c24b42d72`) |
| `fidelity_bar_gate.py` | exit 1 | exit 1 | **output byte-identical** (`diff` empty). A recorded-measurement bookkeeping gate; it re-measures nothing, and its standing DEBT list is unchanged |
| `nemo_geometry_gate.py` | exit 1 | exit 1 | **UNRUNNABLE AT BOTH COMMITS**, identical `ValueError` from `create_z_star_from_thicknesses`: "nemo_gdept_0_m differences must exactly equal interior nemo_e3w_0_m where mesh_reference identity is claimed". PRE-EXISTING, not caused here, and it means this change has NO passing geometry control — stated rather than implied |
| `test_fidelity_card_constructibility.py` | — | 6 passed, 1 xfail | the xfail is the pre-existing GYRE `pgf_quadrature` allow-list finding |

There is no `--compare-to` flag on the stage-sweep gate (checked: the string
appears nowhere in `scripts/` or `tests/`). The comparison was done by running
the gate at both commits and diffing the emitted JSON artifacts section by
section, which is the same thing.

## 5. Tests

```
tests/ocean/unit/test_implicit_vmix_dzw_slot.py            17 passed
tests/ocean/unit/test_zdf_implicit_literal.py               9 passed
tests/ocean/unit/test_no_scheme_duplication.py             35 passed
tests/ocean/unit/test_nemo_mlf_step_transcription.py       22 passed
tests/ocean/unit/test_barotropic_after_reconcile.py        37 passed
tests/ocean/fidelity/test_fidelity_card_constructibility.py 6 passed, 1 xfailed
tests/test_dispatch_hardening.py                           all passed
```

The synthetic divisor test the preregistration asked for is
`test_nemo_e3w_kmm_reproduces_the_oracle_divisor_on_a_stretched_column`: a
6-level stretched column with off-midpoint T points and a 3 m ssh, asserting the
survivor equals `diff(gdept_0)·(1+ssh/H)` to `< 1e-15` relative. Its
**non-vacuity leg is part of the test**: it also asserts the midpoint expression
it replaced differs by `> 1e-2` on that same column, so the test cannot pass
against the reverted code. Three siblings pin the rest: the face map is the
same object as the tracer divisor, the no-NEMO-mesh arm is bit-identical to
`build_dz_half`, and a stretched `t_depth_ref` with no mesh RAISES.

**Pre-existing reds, verified identical at the pre-change commit `5c6ee4ebb` in
a temporary worktree** (not caused here, not fixed here):
`tests/test_validate_strict_coverage.py::test_known_unvalidated_is_shrink_only_and_real`
and 10 failures in `tests/ocean/test_recipe_option_threading.py` — 11 failures
on both sides, same names.

## 6. Isomorphism tripwire

`nemo_branch_isomorphism_map.md` row **S-34** is now **SHARED**: one symbol
(`nemo_e3w_kmm`), reached by every card that runs the routine, no selector. The
rank-4 collapse item the map itself prescribed is marked DONE with what actually
landed. The registry row in `_nemo_branch_isomorphism_baseline.py` is the
COMBINED `S-33_34`; it keeps `disposition="OTHER_RECIPE"` because it also covers
**S-33**, whose `shared_thomas`/`nemo_literal` solver-evaluation fork is a
genuine cross-recipe fork and is unchanged — writing `SHARED` on the combined
row would state something false about S-33. The row's note records S-34 as
SHARED explicitly. There was no baseline entry for `S-33_34` to remove (the
2026-09-02 reclassification had already dropped it), so "baseline entry removed"
was a no-op; `test_no_scheme_duplication.py` is green.

## 7. What was NOT run, and what it would show

**NOT RUN — GPU, the user's decision, explicitly out of scope:** the year-1
climate battery and the 20-year ensemble member. Nothing here speaks to the
water-mass census, the MLD seasonal cycle, the basin transports or the
variability families.

If the scaling study's hypothesis holds, those runs would show: the abyssal
`S_mean` and `T_mean` census statistics (`north of band.abyss_ge1400m`) moving
toward NEMO by order their present gaps (`1.09e-5 g/kg`, `2.28e-4 K`), since
only `0.2-0.3%` of the now-removed divergence rate had to rectify to produce
them; a smaller move in the May mixed-layer depth north of the band (`0.032 m`
on `86 m`), where the divisor error was 3-4x weaker; and no defensible
prediction at all for the transport and variability families, which have no
linear translation from a mean-tendency bias. **A move in the predicted
direction would still not be an attribution** — Rule 4 needs the term removed
from BOTH models, and the twin has no such arm now that the midpoint is
unreachable from a NEMO card.

**Also disclosed, not fixed:** NEMO stretches the momentum divisor by `r3u` —
the area-weighted ssh average over the two T cells divided by `hu_0`
(`domqco.F90:164-167`) — while legoESM's face map averages the already-stretched
T-point field. On DINO's zco mesh `e3w_0` is horizontally uniform, so this is a
pure ssh-level difference, orders below the slot error just removed. It is an
open row, named here rather than silently carried.

**And:** the NEMO GYRE card is the one card whose divisor moves without a gate
covering it, because `test_nemo_gyre_card_constructs` is `xfail(strict=True)` on
a pre-existing `pgf_quadrature` allow-list finding. Unmeasured reach, recorded.

## 8. Choices

| choice | status |
|---|---|
| collapse NEMO's divisor onto the NEMO identity; delete `implicit_vmix_e3t_now_divisor`; keep the Veros `dzw` arm | **ASKED** (task brief; also the isomorphism map's own rank-4 prescription) |
| bind the divisor to `zdf_implicit_solver_evaluation="nemo_literal"` rather than adding a selector | **ASKED** in shape ("no new public field") |
| `outer_integrator="nemo_mlf"` now hard-requires the NEMO identity instead of the deleted flag | forced — the guard's subject was removed; the requirement is unchanged |
| do NOT retarget the old mutual-exclusion raise onto the identity | **UNASKED**, offered for revert. Retargeting would make a previously-tolerated combination a hard error that no card in the tree hits, and would break the committed divisor probe. Precedence (Veros `dzw` wins) is documented at the branch and pinned by a test with its own non-vacuity leg |
| do NOT re-add the "divisor requires `implicit_vertical_mixing=True`" raise on the identity | **UNASKED**, offered for revert. Same reason: it would be a new hard error, and `nemo_literal` was already inert under explicit vertical mixing before this change |
| `DINO_NEMO_KMM_DIVISOR` env-var A/B hooks deleted (they now raise with a pointer) | **UNASKED**, offered for revert. They cannot survive the field's removal; the before/after comparison was done across commits instead |
| do NOT change the momentum face stretch to NEMO's `r3u` | **UNASKED**, deliberately out of scope, recorded in §7 as an open row rather than silently fixed |
| corrected the geometry gate's stale `e3uw_0` waiver reason | **UNASKED**, offered for revert. Its reason string ("legoESM builds from h_u directly") became false with this change, and a waiver reason is a claim |

**Own error, one line:** the first AFTER twin run was discarded — it refused to
save because the working tree changed mid-integration (I edited the probe while
it ran). Re-run on a clean tree; the numbers above are from that re-run.
