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
(`icectl.F90:232-236`, threshold `rchk_t · rn_icechk_glo · ice_area · rDt_ice`).
Per the pre-registration this is REFUTE and is **not** relabelled.  What is
measured about it, so the next lane does not re-derive it:

* Largest event is 31.29 J on ~24 200 m² of ice over a 2 s step, i.e. a surface
  heat-flux imbalance of **6.5e-4 W/m²**.  That is 115× *below* SI3's own
  per-gridcell stop threshold (`rchk_t · rn_icechk_cel = 7.5e-2 W/m²`,
  `icectl.F90:51,323`; `namelist_ice_ref:318`), which is why the run did not
  abort.  It trips only the global warning, which `rn_icechk_glo = 1e-4`
  tightens by four orders of magnitude.
* Relative to the basin's total ice+snow enthalpy (2.687e12 J) the largest event
  is 1.16e-11.
* **Rungs 3.2/3.3 being silent is not evidence that they conserve better.** The
  same threshold is a *flux* threshold multiplied by the ice timestep, so in
  absolute joules it is 600× looser at `rn_dt=1200 s` than at `2 s`; measured
  threshold-to-enthalpy ratios are 1.35e-13 (3.1) versus 3.25e-10 (3.2).  A
  3.1-sized relative imbalance would not have printed in 3.2.
* Mechanism: **UNRESOLVED**.  All surface fluxes are zero in this case
  (`tests/ICE_ADV1D/MY_SRC/usrdef_sbc.F90:119-128`), so the residual is inside
  `qt_oce_ai − qt_atm_oi + diag_heat − diag_adv_heat` (`icectl.F90:221`).  It is
  ~5e4× above the fp64 rounding floor of that sum (5.97e-4 J), so it is
  **not** roundoff.  Candidate causes (all PLAUSIBLE, none discriminated):
  convergent-flow concentration piling to `a_i` = 2.687 making `1 − at_i_b`
  negative in `iceupdate.F90:105-119`; a `diag_heat`/`diag_adv_heat` staging
  mismatch under `ln_icethd=F`.  Discriminating this is lane-4 work.

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
| `max a_i`, initial → final | 0.8999999761581422 → 0.8999999761570061 (Δ = −1.14e-12, rel 1.3e-12) | **REFUTE** as written (predicate demanded exact preservation) |
| `max h_i`, initial → final | 2.0 → 1.2395552072741867 | — |
| `h_i` overshoot (final − initial max) | **−0.7604** | **REFUTE** (predicate demanded strictly positive) |
| upper side-lobe cell count (`h_i` > initial max) | **0** | **REFUTE** (predicate demanded > 0) |
| total-`a_i` relative drift (UNCLASSIFIED) | 0.4928 | reported, not scored — see note |
| total-`v_i` relative drift (UNCLASSIFIED) | 1.021e-08 | reported, not scored |
| SI3 native violations | 0 | CONFIRM |

Additional measurements taken to say *why* it is refuted, none of which change
the verdict:

* **Concentration maximum is not exactly preserved but is preserved to 1.3e-12**
  at the endpoint — and it is genuinely *not* conserved mid-run: scanning all
  485 frames, `max a_i` peaks at **0.9251947 at kt = 8**, 2.8 % above the
  initial 0.9.  So the shipped note "Prather conserves the max values" is a
  scheme description, not a bit-identity, and the endpoint agreement is a
  round trip (the patch traverses ~97 of the 99 doubly-periodic cells in 485
  steps at 0.5 m/s).
* **The thickness overshoot is absent from the whole trajectory, not just the
  endpoint.** Scanning all 485 frames, `max h_i` is largest at kt = 1 (2.0) and
  never exceeds it.  So this REFUTE is not an artefact of sampling only the
  final restart.
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
| final `max u_ice` [m/s] | 0.5033997478575521 | CONFIRM (positive; and within 0.7 % of the shipped 0.5 m/s annotation) |
| max abs change in `v_i` (entry kt=1 → final) | 1.6469 | CONFIRM (state evolved) |
| SI3 native violations | 0 | CONFIRM |
| total-`a_i` relative drift (UNCLASSIFIED) | 0.5036 | reported, not scored (same first-step zapping as 3.2) |
| total-`v_i` relative drift (UNCLASSIFIED) | 5.580e-09 | reported, not scored |

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

| control | 3.1 | 3.2 | 3.3 | message |
|---|---|---|---|---|
| `--plant-unaccounted` (adds `PLANTED_UNACCOUNTED_FILE_ARRAY` to the discovered mesh inventory) | rc=1 | rc=1 | rc=1 | `mesh coverage mismatch: missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']` |
| `--plant-field` (adds 1 m to one in-memory `e1t` value) | rc=1 | rc=1 | rc=1 | `e1t metric` |

Non-vacuity: the unplanted arms are **not** all red — rung 3.3 exits 0 with
`status: VERIFIED`, while 3.1 and 3.2 exit 1 with `status: DEBT` for the
substantive reasons in §4.  That asymmetry is itself asserted by a test.

Tests: `tests/ocean/fidelity/test_nemo_si3_oracle_gate.py` — **15 passed**
(`15 passed in 10.98s`).  They cover the frame round trip in fp64 Fortran order,
registry completeness/uniqueness/sourcing, the Appendix-A contract dispositions,
the contract-omission and `UNMEASURED`-in-mesh red paths, the phenomenology
REFUTE path, the native-conservation REFUTE path, and the six CLI control
invocations above (skipped, not silently passed, if the run roots are absent).

Repo-wide: `tests/ocean/fidelity/` reports `2 failed, 670 passed, 7 skipped`.
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

Every mesh/restart/namelist hash above is also embedded in the committed
manifest for its rung, and the gate refuses to run against a file whose hash has
moved.

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
* **Cause of the rung-3.1 heat-conservation violations: UNRESOLVED.** Magnitude
  and thresholds are measured (§4); the mechanism is not, and the two candidate
  causes named there are PLAUSIBLE only.
* **Rung-3.2 `max a_i` mid-run overshoot to 0.925 at kt=8**: measured, but no
  claim is made about whether it is the "side lobes" the shipped README
  describes — that would need the field-shape diagnostic this phase did not
  preregister.
* **Tracker #1699 live issue contents: UNVERIFIED** — `api.github.com` was not
  reachable from this session either.
* **Rungs 3.2/3.3 conservation silence is weak evidence**, quantified in §4:
  their conservation test is ~2400× less sensitive in relative terms than
  rung 3.1's.

## 10. Pre-registration amendments

Recorded rather than silently applied.  No verdict was relabelled after seeing
output; the two amendments below are instrument corrections and one ill-posed
predicate.

| preregistered text | amendment | reason |
|---|---|---|
| reductions over the field arrays | reductions over the **surface-tmask wet window** | ICE_ADV1D's land rim is identically zero by `dommsk`, and dominated every whole-array spread; ICE_ADV2D is fully wet so this is a no-op there |
| "all three centroid distances from basin centre strictly decrease" | `h_i` centroid convergence recorded **VOID-ILL-POSED**; `a_i` and `v_i` unchanged and both CONFIRM | `h_i`'s initial distance is exactly `0.0`, so the predicate cannot be satisfied by any run; a control that perturbs a zero is not a control |
| `nz = 1` for rung 3.1 | `nz = 2` | NEMO's own `jpk = MAX(2, jpkglo)` (`mppini.F90:80`), confirmed in `ocean.output` |
| `cn_exp` compared verbatim | compared after stripping quotes **and** blank padding | NEMO writes it back as a blank-padded `CHARACTER(lc)` field |
| conservation/phenomenology raise on failure | verdict reported into the JSON, **exit still nonzero** | a refuted rung must still leave its measured numbers on the record |

## 11. FLAGGED FOR FUTURE DELETION

*(empty — nothing in the repo or the shipped NEMO tree is proposed for deletion
by this phase.)*
