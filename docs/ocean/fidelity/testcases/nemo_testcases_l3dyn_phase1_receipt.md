# NEMO testcase lane 3 dynamics — phase-1 receipt (oracle only)

Scope: NEMO 5.0.2 SI3 dynamics rungs 3.1–3.3, oracle side only.
**No legoESM comparison is made or claimed anywhere in this document.**

Pre-registration: `nemo_testcases_l3dyn_phase1_preregister.md` (written before
build or run).  Spec: `si3_lane3_scoping_dossier.md` §1–3, §6, §8, Appendix A.
Tracker: climate-federation/legoESM #1699.

Oracle tree: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`, commit
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`.
Repo commit stamped into every gate/manifest JSON: `git_sha` field — it is the
commit the gate was *run* at, i.e. the parent of the commit that carries the
manifest, since a manifest cannot contain its own commit hash.
Every run is a single-process CPU execution of `nemo.exe` (no `mpirun`, no GPU),
built with `arch-conda.fcm` (`-fdefault-real-8`); the instrument additionally
aborts unless `STORAGE_SIZE(1._wp) == 64`, and every emitted frame header
carries `storage_bits = 64`.

## 1. Run verification

| rung | case copy | run root | exit | frames | final ice restart | mesh |
|---|---|---|---:|---:|---|---|
| 3.1 | `tests/ICE_ADV1D_OMIP_L3` | `…/nemo-testcases-l3/ice_adv1d/final` | 0 | 40 / 40 | `ICE_ADV1D_OMIP_L3_00000040_restart_ice.nc` | present |
| 3.2 | `tests/ICE_ADV2D_OMIP_L3` | `…/nemo-testcases-l3/ice_adv2d/final` | 0 | 485 / 485 | `ICE_ADV2D_OMIP_L3_00000485_restart_ice.nc` | present |
| 3.3 | `tests/ICE_ADV2D_RHG_OMIP_L3` | `…/nemo-testcases-l3/ice_adv2d_rhg/final` | 0 | 485 / 485 | `ICE_ADV2D_RHG_OMIP_L3_00000485_restart_ice.nc` | present |

Rungs 3.1 and 3.2 were completed by the predecessor session (`FINAL_EXIT=0`,
`FRAME_COUNT=40` / `485`).  Rung 3.3 was found **incomplete** — killed at step
299 of 485 with an empty `run.stdout` and no restart — and was **re-run to
completion in this session** in the same run root (`FINAL_EXIT=0`,
`FRAME_COUNT=485`).  Every frame `oracle_ice_step_entry_kt*.bin` is re-read by
the gate, which checks `kt` against its ordinal, fixes the header across the
whole run, and refuses any trailing byte outside the registry.

The shipped cases are byte-unchanged: `git status --porcelain --
tests/ICE_ADV1D tests/ICE_ADV2D` is empty, and `tests/ICE_ADV1D/EXPREF/
make_initice.py` still carries its original Python-2 `print` statements at
lines 21, 30–32.  The Python-3 translation of that bootstrap script exists only
in the run root (`…/ice_adv1d/bootstrap/make_initice.py`); its formulas and
output schema are untouched.  `tests/ICE_ADV1D/MY_SRC/` contains only
`usrdef_{hgr,nam,sbc,zgr}.F90` — the lane-3 `icestp.F90` instrument was never
written into a shipped case.

## 2. Compiled cpp keys

All three copies resolve to exactly `key_si3 key_linssh key_vco_1d`, i.e. the
shipped selectors (`tests/ICE_ADV1D/cpp_ICE_ADV1D.fcm:1`,
`tests/ICE_ADV2D/cpp_ICE_ADV2D.fcm:1`) minus `key_xios`.

| rung | `BLD/cpp.fcm` sha256 |
|---|---|
| 3.1 | `6ab1b3b7a7d9306c58c2047c737213aacdddfba5e48d840434110376f4dc6101` |
| 3.2 | `00d9bdbb4debde6d6100f0666d5e5688d62219a86152b377ca0b7f45a2e28451` |
| 3.3 | `00d9bdbb4debde6d6100f0666d5e5688d62219a86152b377ca0b7f45a2e28451` |

3.2 and 3.3 share a `cpp.fcm` because they differ only in namelist selectors.

All three copied cases use the **identical** committed instrument
`scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/si3_l3/icestp.F90`
(verified by `diff -q` against each `tests/*_OMIP_L3/MY_SRC/icestp.F90`).

## 3. Resolved configuration, read from `output.namelist.{dyn,ice}`

These are the values NEMO itself wrote back, not deck comments.  Rows are the
resolved value; the "source" column names where the value comes from —
shipped case cfg, shared ref, or the ORCA1 overlay at
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg`.

| selector | 3.1 | 3.2 | 3.3 | source |
|---|---|---|---|---|
| `cn_exp` | `ICE_ADV1D_OMIP_L3` | `ICE_ADV2D_OMIP_L3` | `ICE_ADV2D_RHG_OMIP_L3` | copied-rung namespace |
| `nn_itend` / `nn_stock` | 40 / 40 | 485 / 485 | 485 / 485 | shipped `nn_itend`; `nn_stock` moved off `0` for the required final restart |
| `rn_dt` [s] | 2.0 | 1200.0 | 1200.0 | shipped case cfg |
| `nn_fsbc` / `nn_ice` | 1 / 2 | 1 / 2 | 1 / 2 | shipped case cfg |
| `ln_meshmask` / `l_sasread` | T / F | T / F | T / F | shipped case cfg |
| `jpl` | 1 | 1 | 1 | ORCA1 `namelist_ice_cfg:24`; also the only value `ln_dynADV2D=T` supports (`tests/ICE_ADV2D/EXPREF/README:55`) |
| `nlay_i` / `nlay_s` | 3 / 3 | 3 / 3 | 3 / 3 | ORCA1 `namelist_ice_cfg:25-26` |
| `ln_icedyn` / `ln_icethd` | T / F | T / F | T / F | shipped `ICE_ADV1D/EXPREF/namelist_ice_cfg:26-27`, `ICE_ADV2D/…:24-25` |
| `ln_dynALL` | F | F | F | shipped case cfg |
| `ln_dynRHGADV` | F | F | **T** | dossier rung 3.3 (shipped `ICE_ADV2D/EXPREF/namelist_ice_cfg:35`) |
| `ln_dynADV1D` | **T** | F | F | shipped `ICE_ADV1D/EXPREF/namelist_ice_cfg:38` |
| `ln_dynADV2D` | F | **T** | F | shipped `ICE_ADV2D/EXPREF/namelist_ice_cfg:37` |
| `ln_adv_Pra` / `ln_adv_UMx` | T / F | T / F | T / F | ORCA1 `namelist_ice_cfg:75-76` (shipped cases ship `F/T`) |
| `ln_icediachk` | T | T | T | shared `namelist_ice_ref:316` is `F`; turned on to read SI3's own ledger |
| `rn_icechk_glo` | 1.0e-4 | 1.0e-4 | 1.0e-4 | shared `namelist_ice_ref:319` |
| `rn_ishlat` | 2.0 | 2.0 | 2.0 | shared `namelist_ice_ref:57` (ORCA1 does not override) |
| `ln_landfast_L16` | F | F | F | shared/shipped `F` **retained**; ORCA1 `namelist_ice_cfg:46` is `.true.` |
| `ln_str_H79` | T | T | T | ORCA1 `namelist_ice_cfg:52` |
| `rn_pstar` [N/m²] | 2.0e4 | 2.0e4 | 2.0e4 | ORCA1 `namelist_ice_cfg:53` |
| `rn_crhg` | 20.0 | 20.0 | 20.0 | ORCA1 `namelist_ice_cfg:54` |
| `ln_str_smooth` | T | T | **F** | ORCA1 `namelist_ice_cfg:55`; made explicit for 3.3 only |
| `ln_rhg_EVP` / `ln_rhg_EAP` | T / F | T / F | T / F | ORCA1 `namelist_ice_cfg:70`; EAP from shared ref `:109` |
| `ln_aEVP` | T | T | T | shared `namelist_ice_ref:110` |
| `rn_creepl` [1/s] | 2.0e-9 | 2.0e-9 | 2.0e-9 | shared `namelist_ice_ref:111` |
| `rn_ecc` | 2.0 | 2.0 | 2.0 | shared `namelist_ice_ref:112` |
| `nn_nevp` | 100 | 100 | 100 | shared `namelist_ice_ref:113` |
| `rn_relast` | 0.333 | 0.333 | 0.333 | shared `namelist_ice_ref:114`; **inert under aEVP** (`icedyn_rhg_evp.F90:236-247`) |
| `nn_rhg_chkcvg` | 0 | 0 | 0 | shared `namelist_ice_ref:116` |
| `nn_icesal` | 4 | 4 | 4 | shared `namelist_ice_ref:199` — **not** replaced by ORCA1's `2` (`namelist_ice_cfg:108`) |
| `ln_pnd` | T | T | T | shared `namelist_ice_ref:241` — **not** replaced by ORCA1's `.false.` (`namelist_ice_cfg:151`) |
| `rn_uice` / `rn_vice` [m/s] | 1.0 / 0.0 (inert under `ln_dynADV1D`) | 0.5 / 0.5 | 0.5 / 0.5 | shipped case cfg |
| `rn_amax_n` | 0.997 | 0.997 | 0.997 | shared ref |

`nn_icesal=4` and `ln_pnd=T` are the resolved testcase/ref values.  They are
deliberately kept, so the restart gate must (and does) VERIFY-load layer
salinity and pond moments.

Only rungs 3.1/3.2 keep `ln_str_smooth=T`; 3.3 alone carries the ORCA1 `F`.
That asymmetry is the preregistered scope (`ln_str_smooth` is a rheology-strength
knob and 3.1/3.2 never call the rheology), and it is recorded here rather than
silently normalised.

## 4. Documented phenomenology, per rung

The comparison sample is fixed by the pre-registration: the kt=1 step-entry
frame versus the completed-run final ice restart, reduced over the surface
wet window (the `mesh_mask.tmask` bounding box).

### 3.1 ICE_ADV1D — verdict: **phenomenology CONFIRM, conservation REFUTE**

Quoted expectation, `tests/README.rst:191-199`:

> "This experiment is the classical SCHAR1996 test case, which has been used in
> LIPSCOMB2004, and in which very specific shapes of ice concentration,
> thickness and volume converge toward the center of a basin. Convergence is
> unidirectional (in x) while fields are homogeneous in y. … especially the
> constitency between concentration, thickness and volume, and the preservation
> of initial shapes."

| number | value | verdict |
|---|---|---|
| max y-spread of final `a_i`, `v_i`, `h_i` | `0.0` exactly | CONFIRM (homogeneous in y) |
| `a_i` centroid distance from basin centre, initial → final | 4.1836 → 1.8350 | CONFIRM (converges) |
| `v_i` centroid distance from basin centre, initial → final | 7.8802 → 3.1575 | CONFIRM (converges) |
| `h_i = v_i/a_i` centroid distance, initial | **exactly 0.0** | predicate **VOID-ILL-POSED**, see below |
| final range of `a_i` / `v_i` / `h_i` | all > 0 | CONFIRM (shapes not collapsed) |
| total-`a_i` relative drift (UNCLASSIFIED) | 1.792e-07 | reported, not scored |
| total-`v_i` relative drift (UNCLASSIFIED) | 1.157e-10 | reported, not scored |

**Instrument corrections made in this session, and which side was wrong.**
Two of the predecessor session's checks disagreed with the case, and in both
cases the *check* was wrong, not the model:

1. *Whole-array reductions counted the land rim.* ICE_ADV1D closes its basin
   with a one-cell land rim (`mesh_mask.tmask` rows/columns 0 and 58 are 0) and
   NEMO zeroes the ice fields there.  The y-spread over the full 59×59 array was
   therefore 2.69 for `a_i` — the rim, not the physics.  Restricted to the
   surface-tmask window the final field has **exactly two distinct y-rows**: the
   land rim, and 57 byte-identical wet rows.  Fixed by `wet_window()`.
2. *The `h_i` convergence predicate is unsatisfiable.* The shipped IC lays a
   symmetric thickness notch (`hti=0.2` on x ∈ [15,43] of the 59-cell basin,
   `tests/ICE_ADV1D/EXPREF/make_initice.py:97-102`) on a uniform `hti=1` field,
   so the `h_i` weighted centroid *starts* on the basin centre: its initial
   distance is `0.0` exactly (printed as `repr` to confirm it is not a rounded
   zero).  Nothing can strictly decrease from zero.  The predicate is recorded
   **VOID-ILL-POSED** with the measured baseline emitted, and the gate asserts
   that baseline is still exactly zero, so the void is itself checkable.  No
   replacement predicate was invented.

The `AssertionError: Regex pattern did not match` the predecessor session left
behind is **not** either of these: rerunning its committed test file as handed
over gives `8 passed`.  That failure had already been fixed before the session
died.

**Conservation: REFUTE.** SI3's own online ledger printed **19**
`iceupdate : violation heat cons. [J]` lines, magnitudes 3.818 … 31.292 J
(`icectl.F90:236-237`, threshold `rchk_t · rn_icechk_glo · ice_area · rDt_ice`).
Per the pre-registration this is REFUTE and is **not** relabelled.  What is
measured about it, so the next lane does not re-derive it:

* **RETRACTED, and the retraction matters more than the original claim.** An
  earlier version of this receipt said the largest event was "115× below SI3's
  own per-gridcell stop threshold … which is why the run did not abort", citing
  `icectl.F90:51,323`.  That is wrong twice over.  (a) The aborting per-cell
  check lives in `ice_cons2D`, and `ice_cons2D` is **never called on a
  pure-advection path**: `icedyn.F90:144-168` runs only `ice_dyn_adv` +
  `ice_var_zapsmall` for `np_dynADV1D`/`np_dynADV2D`, and every `CALL
  ice_cons2D` in `src/ICE/` sits in `icedyn_rhg`, `icedyn_rdgrft`, `icecor`,
  `iceitd`, `icethd`, `icethd_do` or `icethd_pnd`.  Rungs 3.1 and 3.2 therefore
  run with **no aborting conservation check at all**; only rung 3.3 has one, via
  `ice_dyn_rhg` (`icedyn_rhg.F90:64,102`), and it did not fire.  (b) The
  comparison was also mean-against-max: 6.5e-4 W/m² is a global area-weighted
  mean out of a `glob_2Dsum` (`icectl.F90:229`), while `rn_icechk_cel` gates a
  per-gridcell `MAXVAL` (`icectl.F90:323`).  The receipt credited a guard that
  never runs.
* Largest event is 31.29 J, which over the ~24 200 m² of ice in a 2 s step is a
  **global-mean** heat-flux imbalance of 6.5e-4 W/m².  Relative to the basin's
  total ice+snow enthalpy (2.687e12 J) it is 1.16e-11.  The line is printed by
  `ice_cons_final` (`icectl.F90:197-240`), which is **print-only**: it contains
  no `ctl_stop`, and the only conservation `ctl_stop`s in the file are
  `icectl.F90:358-360`, inside `ice_cons2D`.
* **Rungs 3.2/3.3 being silent is not evidence that they conserve better.** The
  same threshold is a *flux* threshold multiplied by the ice timestep, so in
  absolute joules it is 600× looser at `rn_dt=1200 s` than at `2 s`; measured
  threshold-to-enthalpy ratios are 1.35e-13 (3.1) versus 3.25e-10 (3.2).  A
  3.1-sized relative imbalance would not have printed in 3.2.
* Mechanism: **RESOLVED — CONFIRMED, and it is a ledger defect, not lost heat.**
  The residual is ~5e4× above the fp64 rounding floor of the sum (5.97e-4 J), so
  it is not roundoff.  Three measurements, all emitted by the gate:

  1. **Every printed violation equals the ice+snow enthalpy that left the
     frames during that same step.**  The gate stamps each ice step entry, so
     each violation is attributed to its step and joined against the frame
     series (`heat_residual_attribution_unclassified`): 18 of the 19 events have
     a successor frame to difference, and all 18 ratios lie in
     **[0.999978, 1.000078]**.  The largest absolute residual is 9.2e-4 J
     against an fp64 differencing floor of 6.0e-4 J on a 2.687e12 J sum — the
     agreement is *at* the roundoff floor.  The 19th event is at `kt = 40`,
     which has no successor frame.  Closure: 310.935 J of printed violations
     minus that last 20.443 J event = 290.493 J, versus a measured total frame
     enthalpy loss of **290.492 J**.
  2. **The sink is a zap routine booking into `hfx_res`.**  `ice_var_zapsmall`
     removes small-but-positive layer enthalpy into `hfx_res`
     (`icevar.F90:657,671`) and runs on every ADV1D step (`icedyn.F90:157`);
     `ice_var_zapneg` does the same for negative values inside Prather
     (`icevar.F90:791,798,831`, called at `icedyn_adv_pra.F90:421`).
  3. **The discriminating pair: the bracket that ACCOUNTS `hfx_res` is silent,
     the ledger that DROPS it fires.**  `ice_cons_hsm`'s heat term includes
     `− hfx_res` (`icectl.F90:110-111`) and is called for `icedyn_adv`
     (`icedyn_adv.F90:113`) — **zero** `icedyn_adv` violation lines were
     printed; all 19 are `iceupdate`.  With `ln_icethd = F`, `iceupdate.F90:118-119`
     **overwrites** `qt_oce_ai` without any `hfx_*` term, so the `hfx_res` the
     zap booked never reaches `ice_cons_final`'s ledger (`icectl.F90:221`),
     while `diag_heat` still sees the enthalpy go.  SI3's end-of-step heat
     ledger is structurally unclosed against zapping whenever `ln_icethd = F`.

  **Also retracted:** the earlier candidate "convergent piling makes `1 − at_i_b`
  negative in `iceupdate.F90:105-119`" is refuted by its own premise.  Under
  `ln_icethd = F`, `qt_oce_ai − qt_atm_oi = −(1 − at_i_b) · qsr_oce`
  (`iceupdate.F90:117-119`), and the case sets `qsr_oce = qns_oce = emp_oce = 0`
  with `sprecip = 0`, hence `qemp_oce = 0`
  (`tests/ICE_ADV1D/MY_SRC/usrdef_sbc.F90:119-140`).  That difference is
  **identically zero whatever the sign of `1 − at_i_b`**.
* Still PLAUSIBLE, not confirmed: which of `zapsmall` and `zapneg` dominates.
  Every measured step is a net enthalpy *decrease*, which matches `zapsmall`
  removing positive `e_i`; `zapneg` zaps *negative* enthalpy and would raise the
  total (`icevar.F90:791`).  That is evidence, not proof that `zapneg` never
  fires within a step.

### 3.2 ICE_ADV2D (prescribed velocity) — verdict: **phenomenology REFUTE, conservation CONFIRM**

Quoted expectations, `tests/README.rst:181-186` and
`tests/ICE_ADV2D/EXPREF/README:64-65`:

> "The purpose of this configuration is to test the advection schemes available
> in the sea-ice code … especially the occurence of overshoots in ice thickness"
>
> "Prather conserves the max values but also creates side lobes / UM does not
> conserve the max but does not create side lobes"

| number | value | verdict |
|---|---|---|
| `max a_i`, initial → final | 0.8999999761581422 → 0.8999999761570061 (Δ = −1.14e-12, rel 1.3e-12) | **UNMEASURED documentation conformance**; endpoint values reported, no unsourced tolerance |
| worst step-boundary `max a_i` excursion | +0.02519473924661253 at entry kt=8; 0.027994155460050227 relative | **MEASURED-UNCLASSIFIED** |
| `max h_i`, initial → final | 2.0 → 1.2395552072741867 | — |
| `h_i` overshoot (final − initial max) | **−0.7604** | **REFUTE** (predicate demanded strictly positive) |
| upper side-lobe cell count (`h_i` > initial max) | **0** | **REFUTE** (predicate demanded > 0) |
| total-`a_i` relative drift (UNCLASSIFIED) | 0.4928 | reported, not scored — see note |
| total-`v_i` relative drift (UNCLASSIFIED) | 1.021e-08 | reported, not scored |
| SI3 native violations | 0 | CONFIRM |

Additional measurements taken to say *why* it is refuted, none of which change
the verdict. All of them are emitted by the committed gate under
`trajectory`, not by a throwaway probe:

* **RETRACTED: the exact endpoint maximum-concentration predicate.** The phase-1
  pre-registration compared entry kt=1 to the final restart, but that is not a
  valid quantitative operationalisation of the shipped qualitative,
  Prather-versus-UM remark. The README supplies no tolerance, and the selected
  source itself says its advected fields are "not perfectly bounded" before
  `ice_var_zapneg` (`icedyn_adv_pra.F90:418-421`). The endpoint difference no
  longer contributes a REFUTE. Documentation conformance is loudly
  **UNMEASURED** until a source-backed band or comparative UM experiment is
  pre-registered.
* **The replacement is diagnostic, not a new post-hoc predicate.** The committed
  gate scans all 485 step-entry frames plus the post-step-485 restart: initial,
  post-steps 1--484, and post-step 485. It records the worst absolute relative
  excursion, its sign, step and phase. `max a_i` peaks at
  **0.9251947154047547 at entry kt=8**, +0.02519473924661253 or
  **2.7994155460050227 %** from the initial 0.8999999761581422. This is
  **MEASURED-UNCLASSIFIED** because it may be part of the documented
  side-lobe/overshoot behaviour. Within-step x/y split states are not sampled
  and remain UNMEASURED.
* **The thickness overshoot is absent from the whole trajectory, not just the
  endpoint.** Over all 485 frames `max h_i` is largest at kt = 1 (2.0) and never
  exceeds it.  So this REFUTE is not an artefact of sampling only the final
  restart.
* **What that REFUTE does and does not say.** The predicate is a *global* max of
  the *ratio* `v_i / a_i` against its initial value.  A side lobe is a local
  oscillation flanking the patch, which such a global maximum cannot see, and
  the shipped note is *comparative* (Prather versus UM), which a single-scheme
  run cannot test at all.  So the correct reading is narrow: **this run never
  produces a thickness larger than it started with** — not "the shipped
  documentation is wrong about Prather".
* **The 0.49 concentration "drift" is a first-step effect, not a leak.**
  `sum a_i` is 2924.10 at kt = 1 and 1689.30 at kt = 2, then decays slowly to
  1483.02 at kt = 485, while `sum v_i` is 141.371666 at kt = 1 and 141.371665 at
  kt = 485 (9e-9 relative).  The shipped IC paints `ati = 0.9` over the whole
  57×57 box while `hti` is a Gaussian that is numerically zero near the box
  edge (`tests/ICE_ADV2D/EXPREF/make_INITICE.py:108-114`), so the first ice step
  removes the zero-volume concentration ring.  Volume, the conserved quantity,
  is untouched.

### 3.3 ICE_ADV2D_RHG (ORCA1 aEVP rheology overlay) — verdict: **CONFIRM**

Quoted expectation, `tests/ICE_ADV2D/EXPREF/README:52-53`: the alternative to a
constant velocity is "a constant ice-atm. stress, thus velocity is calculated by
rheology (ln_dynRHGADV=T)", driven by the shipped
`utau_ice = 1.3 N/m² ("<=> 0.5 m/s")` (`tests/ICE_ADV2D/MY_SRC/usrdef_sbc.F90:93`).

| number | value | verdict |
|---|---|---|
| final `max u_ice` [m/s] | 0.5033997478575521 | CONFIRM (positive) |
| free-drift identity `sqrt(utau_ice / (rho0 · rn_Cd_io))` [m/s] | 0.5033997477580665 | CONFIRM — **gated**, relative miss **1.98e-10** against a 1e-2 band |
| max abs change in `v_i` (entry kt=1 → final) | 1.6469 | CONFIRM (state evolved) |
| SI3 native violations | 0 | CONFIRM |
| total-`a_i` relative drift (UNCLASSIFIED) | 0.5036 | reported, not scored (same first-step zapping as 3.2) |
| total-`v_i` relative drift (UNCLASSIFIED) | 5.580e-09 | reported, not scored |

The free-drift row is the rung's substantive bar, and it is checked by the gate
rather than admired in prose.  The documented claim is that the rheology
*calculates* the velocity a constant stress implies, so the bar is the identity
that stress balance fixes.  Ice-ocean drag is quadratic and both stress terms
carry the same U-point ice fraction `zaU`, which therefore cancels
(`icedyn_rhg_evp.F90:310,580-590`):

> `utau_ice = rho0 · rn_Cd_io · |u_ice|²`

Nothing in that is defaulted: `rn_Cd_io = 5.0e-3` is read from the run's
resolved `output.namelist.ice`, `rho0 = 1026.0` from its own `ocean.output:162`,
and `utau_ice = 1.3 N/m²` is the case constant at
`tests/ICE_ADV2D/MY_SRC/usrdef_sbc.F90:93`.  Predicted 0.5033997477580665 m/s
against measured 0.5033997478575521 m/s: the aEVP steady state **is** free drift
to ten significant figures.  The shipped `"<=> 0.5 m/s"` annotation is that
identity, rounded.  The gate's band is 1e-2 relative, which the run clears by a
factor of 5e7; the band exists to catch a broken stress balance, drag law or
aEVP solve, all of which move the steady speed by tens of percent.

Rung 3.3 is also the **only** rung with an aborting per-cell conservation check:
`ice_dyn_rhg` calls `ice_cons2D` (`icedyn_rhg.F90:64,102`), which `ctl_stop`s on
a per-gridcell violation (`icectl.F90:323,355-360`).  It did not fire.

Total concentration is deliberately not classified: `Hpiling`
(`icedyn.F90:209-232`) may rescale it.

Cross-check that the two ADV2D rungs really start from the same state: their
kt=1 frame sha256 is identical (`4039ea6647cc12fb…`) and their `mesh_mask.nc`
sha256 is identical, while their final-frame hashes differ.

## 5. Ice oracle gate — coverage

Gate: `scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py`.
Manifests (the committed coverage ledgers, each stamped with `git_sha`):
`scripts/validate/ocean_fidelity/testcases/manifests/ice_adv{1d,2d,2d_rhg}_l3.json`.

Every variable discovered in `mesh_mask`, the final SI3 restart, and both
resolved namelists must appear exactly once with a `VERIFIED` or `WAIVED`
disposition and a non-empty, source-based reason.  `UNMEASURED` is rejected
outright in the mesh and restart namespaces.  A discovered-but-unlisted name or
a listed-but-absent name is fatal, so **UNACCOUNTED is structurally 0**.

| namespace | 3.1 | 3.2 | 3.3 |
|---|---|---|---|
| mesh | 35 (31 V / 4 W) | 35 (31 V / 4 W) | 35 (31 V / 4 W) |
| restart | 109 (105 V / 4 W) | 109 (105 V / 4 W) | 112 (108 V / 4 W) |
| `output.namelist.dyn` | 325 (8 V / 317 W) | 325 (8 V / 317 W) | 325 (8 V / 317 W) |
| `output.namelist.ice` | 281 (12 V / 269 W) | 281 (12 V / 269 W) | 281 (26 V / 255 W) |
| **UNACCOUNTED** | **0** | **0** | **0** |
| Appendix-A restart contract | 117 (104 V / 13 W) | 117 (104 V / 13 W) | 117 (107 V / 10 W) |

* Every `VERIFIED` mesh/restart row is actually opened, is numeric, is finite,
  and — for floating restart fields — is asserted `float64`.
* **The split itself is regenerated, not trusted.**  Every disposition and
  reason is recomputed from the run and required to equal the manifest, the same
  ratchet the Appendix-A contract already had.  Without that, a `VERIFIED` row
  could be quietly re-labelled `WAIVED` with any non-empty reason string and
  would then *skip* its numeric/finite/fp64 check — measured on the shipped
  gate before the fix: downgrading 18 of the 31 VERIFIED mesh rows plus
  `DELAY__r8_cflice` still exited 0 with `status: VERIFIED`.
* Each manifest's `git_sha` is required to be a 40-hex commit rather than any
  provenance string a hand edit might leave.
* The four waived rows in each file are NEMO's own metadata variables.  The
  waiver set is no longer hand-written: it is NEMO's `meta(1:11)` list copied
  from `src/OCE/IOM/iom.F90:365-375`, which is what caught `numcat` (the SI3
  category axis, `src/OCE/IOM/iom_nf90.F90:139`) being float32.
* `DELAY__r8_cflice` is VERIFIED-loaded with its own reason: it is iom's
  delayed-global-reduction restart buffer (`icerst.F90:140` via `iom.F90:1939-1955`) for the ice-CFL
  `mpp_max` (`icedyn_adv_pra.F90:122`, `icedyn_adv_umx.F90:185`), not an
  Appendix-A prognostic.
* Waived namelist rows are resolved NEMO defaults outside the rung's claim; the
  full resolved-namelist file hash is pinned regardless, so a default cannot
  move silently.

### Appendix-A restart contract (117 rows, regenerated and required to match exactly)

`nn_fsbc, kt_ice, v_i, v_s, a_i, t_su, u_ice, v_ice, oa_i, a_ip, v_ip, v_il,
sv_i, snwice_mass, snwice_mass_b` VERIFIED; `e_s(1..3)`, `e_i(1..3)`,
`szv_i(1..3)` VERIFIED; all mandatory Prather moments `sx/sy/sxx/syy/sxy` ×
{`ice, sn, a, age`}, × `c0(1..3)`, × `e(1..3)`, plus the salinity-mode moments
`si(1..3)` (active because `nn_icesal=4`) and the pond moments `ap, vp, vl`
(active because `ln_pnd=T`) VERIFIED.

Waived, each with a reason: `t_s(1..3)` **WAIVED-NOT-RESTARTED** (Appendix A's
`t_s(l)` is reconstructed; `icerst.F90:153-159` writes `e_s(l)`);
`sxsal, sysal, sxxsal, syysal, sxysal` WAIVED-INACTIVE (`nn_icesal=4` selects
the layer moments); `cnd_ice, t1_ice` WAIVED-INACTIVE (`ln_cpl=F` in all three
resolved `output.namelist.dyn`; `icerst.F90:178-181`); `stress1_i, stress2_i,
stress12_i` WAIVED-INACTIVE on 3.1/3.2 (advection-only rungs never call
`ice_dyn_rhg`) and VERIFIED on 3.3.

The gate regenerates this contract from the resolved `nlay_i`, `nlay_s`,
`nn_icesal`, `ln_pnd` and demands exact set equality, so deleting a row is
fatal — proven by a test that pops `t_s_l01` and requires the gate to go red.

### Step-entry frame registry

The instrument writes one frame per ice step immediately before `store_fields`
(`nemo502_MY_SRC/si3_l3/icestp.F90:154-171`; the upstream `store_fields` call it
precedes is `src/ICE/icestp.F90:151`).  Header: 16-byte magic `NEMO_L3_ICE_1`,
then nine native 32-bit integers (version, `kt`, `jpi`, `jpj`, `jpl`, `nlay_i`,
`nlay_s`, real storage bits, registry count = 19), then native fp64 payload in
Fortran order.  Registry order was verified against the three `WRITE` statements
at `icestp.F90:165-167`.

| # | field | shape | time level | source |
|--:|---|---|---|---|
| 1–8 | `v_i, v_s, a_i, t_su, oa_i, a_ip, v_ip, v_il` | `(jpi,jpj,jpl)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:143-152` |
| 9 | `sv_i` | `(jpi,jpj,jpl)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:168` |
| 10–11 | `u_ice, v_ice` | `(jpi,jpj)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:147-148` |
| 12–14 | `stress1_i, stress2_i, stress12_i` | `(jpi,jpj)` | CARRIED_PREVIOUS_STEP | `icestp.F90:154-171`; `icedyn_rhg_evp.F90:1110-1112` |
| 15 | `snwice_mass` | `(jpi,jpj)` | CARRIED_PREVIOUS_STEP | `icestp.F90:154-171`; `iceupdate.F90:479` |
| 16 | `snwice_mass_b` | `(jpi,jpj)` | CARRIED_PREVIOUS_BEFORE_LEVEL | `icestp.F90:154-171`; `iceupdate.F90:480`, pair advanced in sequence at `iceupdate.F90:188-192` |
| 17 | `e_s` | `(jpi,jpj,nlay_s,jpl)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:153-159` |
| 18 | `e_i` | `(jpi,jpj,nlay_i,jpl)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:160-166` |
| 19 | `szv_i` | `(jpi,jpj,nlay_i,jpl)` | STEP_ENTRY_CURRENT | `icestp.F90:154-171`; `icerst.F90:170-175` |

Resolved frame headers: 3.1 `jpi=jpj=63, jpl=1, nlay_i=nlay_s=3, storage=64`;
3.2 and 3.3 `jpi=jpj=103, jpl=1, nlay_i=nlay_s=3, storage=64`.  A frame carrying
an unknown field count, a wrong `kt`, a changed shape, a non-64-bit storage word,
or one trailing byte is fatal.

### Geometry

`nz` in the gate is the **resolved** `nav_lev`, not the case's `jpkglo`:
ICE_ADV1D asks for `kpk = 1` (`tests/ICE_ADV1D/MY_SRC/usrdef_nam.F90:76`) and
NEMO raises it via `jpk = MAX( 2, jpkglo )` (`src/OCE/LBC/mppini.F90:80,375`),
which the run's own `ocean.output` confirms (`jpk : 2   jpkglo : 1`).  The
predecessor gate had `nz=1` and would have failed every 3.1 run.

3.1: 59×59×2, all eight horizontal scale factors uniform at 4.0 m, `ff_t=ff_f=0`,
masks binary.  3.2/3.3: 99×99×2, scale factors uniform at 3000.0 m, same.

## 6. Planted-violation controls

Both preregistered controls are run **end to end through the shipped CLI**
against each of the three real run roots, and all six exit nonzero:

| control | 3.1 | 3.2 | 3.3 | message asserted |
|---|---|---|---|---|
| `--plant-unaccounted` (adds `PLANTED_UNACCOUNTED_FILE_ARRAY` to the discovered mesh inventory) | rc=1 | rc=1 | rc=1 | `mesh coverage mismatch: missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']` |
| `--plant-field` (adds 1 m to one in-memory `e1t` value) | rc=1 | rc=1 | rc=1 | `e1t metric` |

The message is asserted, not just the exit code.  Rungs 3.1 and 3.2 already exit
1 unplanted, so a bare `returncode != 0` assertion would have passed on four of
these six arms even with both plants disabled — measured: with the unaccounted
plant made a no-op, all three of its arms go red on the message and only the 3.3
arm would have gone red on the exit code.

Further controls, each shown to fail when its subject is reverted:

| control | what it pins | measured on revert |
|---|---|---|
| wet-window regression on rung 3.1 | the land-rim fix that this rung's phenomenology verdict rests on | neuter `wet_window()` → that test alone goes red (`CONFIRM` → `REFUTE`); the other 18 stay green |
| registry order versus the committed instrument | the `WRITE(itraj)` order that NAMES every frame array | swap two `FRAME_REGISTRY` rows → red at index 0 |
| manifest status downgrade | a VERIFIED row silently re-labelled WAIVED (which would skip its numeric/finite/fp64 check) | `pytest.raises` on the regenerated-template mismatch |
| manifest `git_sha` | a hand-written provenance string | `pytest.raises` on a non-40-hex value |
| `ocean.output` hash | the file that carries the conservation verdict | `pytest.raises` on an edited copy |
| committed input decks | run inputs drifting from `configs/` | `input_namelists(3.1 root, "3.2")` raises |
| free-drift identity | rung 3.3's quantitative bar | a 0.6 m/s prediction gives REFUTE |

Non-vacuity: the unplanted arms are **not** all red — rung 3.3 exits 0 with
`status: VERIFIED`, while 3.1 and 3.2 exit 1 with `status: DEBT` for the
substantive reasons in §4.  That asymmetry is itself asserted by a test.

Tests: `tests/ice/fidelity/test_nemo_si3_oracle_gate.py` — **21 passed**
(`21 passed in 11.92s`). They cover the frame round trip in fp64 Fortran order,
registry completeness/uniqueness/sourcing and its ORDER against the committed
instrument, the Appendix-A contract dispositions, the contract-omission,
status-downgrade, bad-`git_sha` and `ocean.output`-hash red paths, the
phenomenology REFUTE path, the maximum-trajectory diagnostic's equal,
intermediate-only and restart-only controls, the free-drift REFUTE path, the
UNMEASURED aggregate fail-closed path, the native-conservation REFUTE path, the
input-deck binding, the rung-3.1 wet-window pin, and the six
CLI control invocations above (skipped, not silently passed, if the run roots
are absent).

Repo-wide: `tests/ocean/fidelity/` reports
`2 failed, 674 passed, 7 skipped, 18 deselected in 161.96s`.
Both failures — `test_recipe_case_board.py::test_every_oracle_comparison_has_a_row`
(missing rows for `advection_nemo`, `grids_tripole_mpas`, `tendencies_nemo`,
`three_way_nemo`) and `test_recipe_comparison.py::test_committed_markdown_is_fresh`
(an MPAS recipe-field count) — are **pre-existing on this branch** and touch no
file in this receipt.

## 7. Gate exit semantics

The gate is fail-closed: it exits nonzero whenever the native conservation
ledger or the phenomenology verdict is anything but CONFIRM, and whenever any
coverage, contract, geometry, namelist, cpp, or frame check fails.  Coverage and
structural checks still abort immediately.  The two *scientific* verdicts
(conservation, phenomenology) are reported into the JSON before the nonzero
exit, so a refuted rung still leaves its measured numbers on the record rather
than only an error string.  Current exits: 3.1 = 1 (DEBT), 3.2 = 1 (DEBT),
3.3 = 0 (VERIFIED).

## 8. Artifact sha256

| rung | file | sha256 |
|---|---|---|
| 3.1 | `mesh_mask.nc` | `ba0e884eab64dd4ef659b21e6369d5999bda20e1e243c5c541ae5b93008148ad` |
| 3.1 | `ICE_ADV1D_OMIP_L3_00000040_restart_ice.nc` | `bc49d8dd9633210a1c8f759b60c27919d48a789637dffae7ef7ee66682488dca` |
| 3.1 | `output.namelist.dyn` | `c23fbfd78caf8ceb68cb75f4896bdb0969cedde0949e89d57c19912509b5bdf7` |
| 3.1 | `output.namelist.ice` | `a1d997a06f2fba512e11ea1bc9ca062f9b234428afafd7ba01264740b19e6ac1` |
| 3.1 | `ocean.output` | `ff28b78808388209c873dd5bd72b593268f5ac7b437a71d88f96e8186e8519d1` |
| 3.1 | ordered frame-hash aggregate (40 frames) | `7bf091ff458dad1a473a42ec2441a036e93148701e26019c80f7578f14e1c8f1` |
| 3.2 | `mesh_mask.nc` | `a74a6cd55b11084715b0e4e964f820ea9c2e222a73bd02f960edf73b58970503` |
| 3.2 | `ICE_ADV2D_OMIP_L3_00000485_restart_ice.nc` | `bbbecb7eae63ebc014f4b868cf11bde319b6b3716e5e38a92d5ff88da71f33cf` |
| 3.2 | `output.namelist.dyn` | `54d6721465269330f434743551e61091198a6195292bc4c4504e25332a7e5990` |
| 3.2 | `output.namelist.ice` | `70755ff89192cef49dbcda9c24f4263c43527fefb488971fdd9a49e6c5454f3c` |
| 3.2 | `ocean.output` | `cd1bf063da6801ffe97febb7bfaacb22e61e813ff17fb7bf4c6a753004e77fb6` |
| 3.2 | ordered frame-hash aggregate (485 frames) | `8f6fa12be7ca329a3520b31317b97dea60d7cdf4fed728e47f574b20f9480122` |
| 3.3 | `mesh_mask.nc` | `a74a6cd55b11084715b0e4e964f820ea9c2e222a73bd02f960edf73b58970503` |
| 3.3 | `ICE_ADV2D_RHG_OMIP_L3_00000485_restart_ice.nc` | `863ee8ea8204618baa57b0d3d1aeec08acff4c7e6f55a47437630a82bbd748b0` |
| 3.3 | `output.namelist.dyn` | `fa4c442c361b44fc14291efe19a98086ce53a9bf89d874467ce160196724b460` |
| 3.3 | `output.namelist.ice` | `3990bbc8b8a71499f0c3958338eea393697308768a8010b426b16ea8a20873b5` |
| 3.3 | `ocean.output` | `8512581c3022c6be20e07bb2db90570a5d683c13e050fac38143b76485026f0b` |
| 3.3 | ordered frame-hash aggregate (485 frames) | `fcfd975bffe378bbbbe145a005fa29285d0e95413f586610ac1e658be8abd79b` |

Every hash above — **including `ocean.output`**, which alone carries the
conservation verdict — is embedded in the committed manifest for its rung, and
the gate refuses to run against a file whose hash has moved.  The two input
decks each run consumed (`namelist_cfg`, `namelist_ice_cfg`) are additionally
required to be byte-identical to the committed copies under
`scripts/validate/ocean_fidelity/testcases/configs/`, so those copies are bound
to the runs rather than being documentation nothing checks.

## 9. UNMEASURED / UNVERIFIED — loud

* **legoESM alignment: UNMEASURED.** Out of phase-1 scope; nothing in this
  receipt is a statement about legoESM.
* **Post-dynamics SI3 stage arrays: UNMEASURED.** Only the step-entry state and
  the final restart are sampled; no intra-step stage was instrumented.
* **Landfast (`ln_landfast_L16`): UNVERIFIED-deferred to lane 4.** ORCA1 sets it
  `.true.` (`namelist_ice_cfg:46`); all three rungs run it `F`, as preregistered.
  No landfast behaviour is measured or claimed.
* **`rn_relast = 0.333` is recorded but inert** under `ln_aEVP=T`
  (`icedyn_rhg_evp.F90:236-247`): its value is pinned, its effect is UNMEASURED.
* **Cause of the rung-3.1 heat-conservation violations: RESOLVED, CONFIRMED**
  (§4) — SI3's end-of-step heat ledger drops `hfx_res` whenever
  `ln_icethd = F`, so enthalpy removed by zapping is invisible to it.  Every
  printed violation matches that step's frame enthalpy loss to the fp64
  roundoff floor.  What remains UNMEASURED is only the split between
  `zapsmall` and `zapneg`.  The earlier "negative open-water fraction"
  candidate is RETRACTED as refuted, and the earlier per-gridcell-threshold
  claim is RETRACTED as citing a guard that never runs on this path.
* **Rung-3.2 maximum-concentration documentation conformance: UNMEASURED.** The
  previous exact endpoint predicate is retracted above. The step-boundary peak
  of 0.925 at kt=8 remains measured but unclassified; no claim is made about
  whether it is the "side lobes" the shipped README describes. No ICE_ADV2D
  notebook exists in the shipped tree; the available documentation is the case
  README and `tests/README.rst`.
* **Tracker #1699 live issue contents: UNVERIFIED** — `api.github.com` was not
  reachable from this session either.
* **Rungs 3.2/3.3 conservation silence is weak evidence**, quantified in §4:
  their conservation test is ~2400× less sensitive in relative terms than
  rung 3.1's, and rungs 3.1/3.2 have no aborting per-cell check at all.
* **Prather side lobes: still UNMEASURED.** §4 now says precisely what rung
  3.2's overshoot predicate does and does not measure; a field-shape diagnostic
  able to see a *local* lobe was not preregistered and was not built.
* **`rn_relast` is not the only pinned-but-inert row**; see §3.

## 10. Pre-registration amendments

Recorded rather than silently applied.  **One of these amendments changes a
verdict, and that is stated plainly here rather than left for a reader to
discover:** with the preregistered whole-array reduction, rung 3.1's
phenomenology is REFUTE (`y_homogeneity_max_abs` = 2.6873480199492166, refuting
`a_i` 2.6873 / `v_i` 2.3410 / `h_i` 1.0); over the wet window it is CONFIRM with
`y_homogeneity_max_abs` exactly 0.0.  The wet-window amendment is the **sole**
reason rung 3.1 reads CONFIRM.  It is defended on NEMO's own masking rather than
on the outcome — `dommsk` zeroes the one-cell land rim and `mesh_mask.tmask`
rows and columns 0 and 58 are 0 — and it is now pinned by a test that goes red
if it is reverted (§6).  An earlier version of this section asserted "no verdict
was relabelled"; that sentence was wrong and is retracted.

| preregistered text | amendment | reason |
|---|---|---|
| reductions over the field arrays | reductions over the **surface-tmask wet window** | ICE_ADV1D's land rim is identically zero by `dommsk`, and dominated every whole-array spread; ICE_ADV2D is fully wet so this is a no-op there. **Flips rung 3.1 phenomenology REFUTE → CONFIRM** |
| "all three centroid distances from basin centre strictly decrease" | `h_i` centroid convergence recorded **VOID-ILL-POSED**; `a_i` and `v_i` unchanged and both CONFIRM | `h_i`'s initial distance is exactly `0.0`, so the predicate cannot be satisfied by any run; a control that perturbs a zero is not a control |
| `nz = 1` for rung 3.1 | `nz = 2` | NEMO's own `jpk = MAX(2, jpkglo)` (`mppini.F90:80`), confirmed in `ocean.output` |
| `cn_exp` compared verbatim | compared after stripping quotes **and** blank padding | NEMO writes it back as a blank-padded `CHARACTER(lc)` field |
| conservation/phenomenology raise on failure | verdict reported into the JSON, **exit still nonzero** | a refuted rung must still leave its measured numbers on the record |
| rung 3.3 scored on "the ice moved" | additionally scored on the **free-drift identity** `sqrt(utau_ice / (rho0 · rn_Cd_io))`, band 1e-2 relative | the only green rung carried the whole non-vacuity argument on two predicates that no moving run could fail; the quantitative agreement was in the prose and gated nowhere.  Measured miss 1.98e-10 |
| — | the gate additionally emits a **full-trajectory scan** and a per-step **heat-residual attribution** | numbers this receipt cites must come from the committed instrument, not a throwaway probe |
| entry-kt=1 versus final-restart exact `max(a_i)` predicate | **RETRACTED**; documentation conformance is UNMEASURED, with all step boundaries reported MEASURED-UNCLASSIFIED | the shipped qualitative Prather-versus-UM remark gives no float tolerance; an endpoint round trip is not trajectory evidence, and the source says the scheme is not perfectly bounded |

The gate aggregates component status with `REFUTE > UNMEASURED > CONFIRM`:
if the independent thickness predicate is made to pass in the planted test,
rung-3.2 phenomenology remains UNMEASURED and the top-level gate remains red.
Both maximum-concentration documentation conformance and unsampled within-step
Prather split states also appear in the root `unmeasured` inventory.

## 11. FLAGGED FOR FUTURE DELETION

The phase-1 rungs 3.1--3.3 proposed no deletion. Rung 3.4 adds two shipped-case
files to this list; neither was modified or deleted:

* `tests/ICE_RHEO/MY_SRC/icedyn_rhg_evp.F90`, SHA-256
  `7efffd18b5a452d403e1a3e4e813f389a9c18a1a4f5a08e280b92272a2ed22ee`:
  it shadows the released 5.0.2 solver but does not compile with the 5.0.2
  module API. The production copy excludes it; the mandatory shipped control
  retains it and fails to build.
* `tests/ICE_RHEO/MY_SRC/icedyn_rhg_eap.F90`, SHA-256
  `56bcddf22a533b3ce688a66051de91eab78397a260948cdd0d9de7417cbc1bf0`:
  it is selector-dead after the requested EAP-to-aEVP overlay, but Fortran
  compilation still sees it and fails against the 5.0.2 module API. Both final
  copies therefore exclude it. The original failed copies are preserved under
  names ending `_FAILED_STALE_EAP` in the writable clone.

## 12. Rung 3.4 ICE_RHEO oracle boundary (2026-09-03)

Status: **ORACLE COMPLETE; COVERAGE VERIFIED; README PHENOMENOLOGY
MEASURED-UNCLASSIFIED**.  This is not a legoESM trajectory verdict.

The experiment was preregistered in
`nemo_testcases_l3dyn_phase2_rung34_preregister.md` at commit `b49ede50860`
before either build. A shared local clone of the unmodified shipped commit was
created at
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3dyn_rung34_src`.
The two required final copied-case paths are `tests/ICE_RHEO_OMIP_L3` and
`tests/ICE_RHEO_OMIP_L3_SHIPPED`; the latter retains the shipped EVP override.

The first build demonstrated that the shipped EAP override also cannot compile
against 5.0.2, despite `ln_rhg_EAP=F`. That forced post-preregistration amendment
is disclosed, not hidden: the failed copies are preserved, then both arms were
recopied with EAP excluded. This leaves the EVP override as the sole source
difference. The production arm, using `src/ICE` EVP and EAP, builds successfully
with `key_si3 key_linssh key_vco_1d`. The initial default-flag executable's
SHA-256 is `0f4f97d58467581326c59c130368e4883489f5bd51ceb4d33f91ded590703173`;
its runtime disposition and the successful production binary are distinguished
below.

The required shipped-EVP control does not build. The repeatable build exits 1;
`control_build.stderr:6-106` reports 21 compiler errors, including undefined
`epsi06`, `ln_aEVP`, `nn_nevp`, `rDt_ice`, and `rn_ishlat`, plus the `ht`
rank mismatch. This is direct evidence that the shipped override predates the
released module interfaces, not evidence about its numerical trajectory.
The user subsequently classified this as an **UPSTREAM NEMO DEFECT** and
explicitly selected the override-excluded copy as the only buildable form of
the shipped case and therefore the oracle.  The control remains
**UNBUILDABLE**, not silently skipped; no source repair or analytic substitute
was invented, and neither shipped override was modified or deleted.

The source-identical default O3 plus function-loop-unroll executable built but
segfaulted in `usr_def_hgr` before the trajectory.  That binary is preserved as
`ice_rheo/nemo.o3_funroll.segfront` (SHA-256 below).  An O3 build without the
function-loop-unroll flag completed the one-step bootstrap.  The production
oracle uses the same NEMO source at fp64 with the conservative CPU-only
O2/`-fno-tree-vectorize` architecture; it was invoked directly, with
`OMP_NUM_THREADS=1`, no `mpirun`, and no GPU.  This compiler provenance is a
disclosed harness choice, not a change to the NEMO case or physics source.

The oracle ran all documented 720 30-second steps and exited 0.  It wrote 720
fp64 step-entry frames, the final ocean and ice restarts, and the mesh under
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final`.  The ordered frame
hash aggregate is
`3e347b438776c077a75c8c02ab3e43ac9158037d164ef48a2e5017dd152ddd27`.
The frame registry remains the 19-array `icestp.F90:154-171` registry; every
frame has `(jpi,jpj,jpl,nlay_i,nlay_s)=(1004,1004,1,10,5)` and storage width
64 bits.

The regenerated manifest accounts for all 35 mesh arrays, 208 restart
variables, 325 resolved ocean-namelist entries, and 281 resolved ice-namelist
entries.  Its Appendix-A contract has 203 VERIFIED-loaded active fields,
including all three EVP stresses and all 160 active Prather moments.  The 32
WAIVED rows are individually reasoned: reconstructed/inactive Appendix-A state
plus 20 named ephemeral ridging work/diagnostic arrays from
`icedyn_rdgrft.F90:46-75`; the carried redistribution lives in the VERIFIED
area/volume/energy/salt/pond prognostics at `:779-888`.  Both full planted
controls bind: an unaccounted mesh array exits 1 with a coverage mismatch and
an `e1t` perturbation exits 1 on the geometry row.

The resolved gate confirms `jpl=1`, `nlay_i/nlay_s=10/5`, `ln_dynALL=T`,
Prather, H79 `rn_pstar=2e4` / `rn_crhg=20`, exponential participation and
redistribution, ridging plus rafting, aEVP with 100 subcycles, landfast off,
and the exact 1000 by 1000, 2-km, zero-Coriolis geometry.  The ORCA1-deck
retention inputs remain `0.5`, while SI3's executed thermodynamics-off branch
forces porosity to zero and all snow/pond retention factors to one at
`icedyn_rdgrft.F90:1244-1247`.

The non-XIOS run wrote only `ssv_m`, not `sishea`, in its final six-hour file.
Round 8 therefore evaluates NEMO's own diagnostic directly from every one of
the 720 registered entry-frame `u_ice`/`v_ice` arrays and `mesh_mask.nc`: the
F-point shear is `icedyn_rhg_evp.F90:793-796`, T-point tension is `:802-806`,
four-F-point weighting is `:810-813`, and `sishea` is
`SQRT(zdt**2+zds**2)*zmsk` at `:815-816`, with `zmsk` from `:190-191`.
**CONFIRMED measurement:** frame 720 has maximum
`1.778565218457925e-4 s^-1`, p95 `6.658604946931389e-7 s^-1`, and p99
`1.6653117537818702e-5 s^-1`.  The README claim remains
**MEASURED-UNCLASSIFIED**, not passed, because it supplies no numeric sharpness
threshold and this aEVP run supplies no EAP comparator.  The longer paired
EVP/EAP intersection-angle claim is OUT-OF-SCOPE.  The nonzero final maximum
ice speed remains `0.3576987760534231 m s-1`, and maximum ice-volume change is
`1.2818995636818311`; neither is substituted for shear.  SI3's native
conservation check resolves off; zero printed violations are WAIVED-INACTIVE.
Reported, unclassified endpoint drifts are `1.8631458995434149e-3` for
concentration sum and `3.199507647832436e-14` for ice-volume sum.

Focused gate controls are **5 passed, 22 deselected**; the final combined
CPU/fp64 ridging, card, and rung-3.4 oracle selection is **21 passed,
15 deselected in 107.98 s**.  Ruff is clean on the gate and its tests.
Round 8's expanded oracle-gate file is **28 passed in 11.76 s** and includes a
binding velocity/mask shear plant plus fail-closed selective-frame reads.  The
full round-8 gate exits nonzero by design because the README result is
MEASURED-UNCLASSIFIED, not because coverage or geometry failed.

| artifact | SHA-256 |
|---|---|
| production safe build stdout | `58ca109387789e6de60aaff83cfcde8bf88be203c54836dc43165b83df9fe5d6` |
| production safe build stderr | `9dca52b86c211bcc0ec1e8d19af9d296ccbb2da21516e65c6d38858a7a7a183d` |
| shipped-control build stdout | `6080c0bff2f916ca99530cae57a01d937e2010156f27626755f81ab80a8b5346` |
| shipped-control build stderr | `baef2fede631b7868aef20a5fe88605bd073812f46f634e9559d9089634a41cc` |
| failed O3+funroll executable | `0f4f97d58467581326c59c130368e4883489f5bd51ceb4d33f91ded590703173` |
| production safe executable | `84d32b9c78da328af19340fb394923e80847005bcf17424bc383c169029284eb` |
| production safe architecture | `311a9ab58bb264a294dc1c70fdfb17352cba0121596b11d00f93c63c2746f4d7` |
| resolved cpp deck, both arms | `58c1f115e87c6177f282da0affbd67657a2d1a5ce4d6276dc752d4a19c4d29de` |
| used `namelist_cfg` | `b79b0da851a617b7a09fafd670e0bcab3d738d6175904423e68f12a5d9da8a93` |
| used `namelist_ice_cfg` | `6e66ccd0eb895c35f6423bf7c09e8f6d293494f531f81ae78fd2b01dc56f7d98` |
| `mesh_mask.nc` | `cf4c4f41cba49d790a9be7aebc7eae68c79493442ed9b0020022f268b3b27146` |
| `output.init_ice.nc` | `5feba84566f0d847cbd75e89b1ad2652e60b2054fe10a6a9b78420d7f7631ca6` |
| final ice restart | `e5161e64a5c8e4d9180420dca58f99a43088952d0e9826c802d94ac1693b4bf6` |
| first / last entry frames | `1097a76b1ddff7fcb54ae6b77537e9d061f250de1257ca6e5ca946da355a75a6` / `2e957065bfa3d7b28f38679b6a3a31128c3c571aaa4c2c234bc2cb7ab2d845b5` |
| ordered 720-frame SHA-256 aggregate | `3e347b438776c077a75c8c02ab3e43ac9158037d164ef48a2e5017dd152ddd27` |
| coverage manifest | `2e004e1a662931f488f3a7669c98310f896ac63a551f178e97f64023497befa7` |
| round-7 oracle gate JSON | `5e70bf27ef2bb1d570a1e32fc82e199d0dbb2f175a8d893d882331a2df347bd6` |
| round-8 full oracle gate with `sishea` | `34ac7cabe31d7ea742409b3552c14ec101437880cea4090bf2f59c2953e358da` |
| round-8 720-frame `sishea` measurement | `3c741bf041896d2c6be354ce54338a0b050e7f36a9949bae3a390a766d1acf87` |
| final oracle gate source | `01f226c15d2a17efaa91d698112e022d860a534f188546d4a0fefde6fc073226` |
| oracle gate tests | `9f8124f291bf369b7f89be9fabe01b12b52b725e3d55c852f0aa778a06924184` |
| unaccounted-array plant stderr | `1700153282cd779d281edf0ca7365579fd72c3053621ae31c4f738566d413904` |
| perturbed-geometry plant stderr | `868cbcbf82d4de702c4f7b8f9abaf65feff3e3604723627349880dae2c7fc258` |

The stale-source trajectory difference is **UNMEASURED-IMPOSSIBLE FROM THIS
CONTROL** because the retained override is unbuildable.  Landfast remains OFF
and **UNVERIFIED-deferred**.  The legoESM result and its louder unmeasured tail
are recorded separately in the phase-2 receipt.
