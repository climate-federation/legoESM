# Preregistration — NEMO-testcases grid ladder G7, round 1 (gridladder_r41)

**Frozen:** 2026-10-16, before any G7 card change and before any G7 record
exists.  Nothing is measured against NEMO in this round; every statement
below is read off the compiled source (READ) or is a prediction (PREDICTED).

Builds cited: `G5` = `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo` (the
record binary G7 reuses; its writer-free twin `ORCA2G5_OMIP_L4_R31REF_FIX1`);
Decision-122 pointer `ORCA2` = `ORCA2_OMIP_L4/BLD/ppsrc/nemo`.  `zdfdrg.f90`,
`dynzdf.f90` and `dynspg_ts.f90` drag lines are identical in both (`diff`
clean on `zdfdrg.f90`; line numbers below hold in both).

## 1. Survey (search before build)

| item | value | where |
|---|---|---|
| ORCA2_OMIP_L4 `&namdrg` | one line, `ln_lin = .true.` | `ORCA2_OMIP_L4/EXP00/namelist_cfg:249-252` |
| ORCA2 `&namdrg_bot` / `&namdrg_top` | not stated -> `namelist_ref` | — |
| resolved companions | `ln_drg_OFF=F` (:812), `ln_non_lin=F` (:814), `ln_loglayer=F` (:815), `ln_drgimp=T` (:817), `ln_drgice_imp=T` (:818); bottom `rn_Cd0=1.e-3` (:834), `rn_Uc0=0.4` (:835), `ln_boost=F` (:839) | `ORCA2_OMIP_L4/EXP00/namelist_ref` (= the run input `namelist_ref`, SHA-256 `b04f2ce4…1cfd`) |
| top drag | `ln_isfcav=F`, so `rCdU_top = rCdU_ice` (SI3) only | `zdfdrg.f90:172-191` (`zdf_drg` :150) |
| G5 deck (and its record) | `ln_drg_OFF = .true.` (TSUNAMI) — **no bottom drag** | `nemo_testcase_g5_orca2ice/namelist_cfg:193`; record `round31/oracle_g5_r31/twin_a/namelist_cfg:193` |
| G5 card | `bottom_drag_scheme="legacy"`, r = 0, matrix/substep drag OFF; ice top drag ON (round 33) | `build_orca2g1_flat_card` |
| existing linear-drag transcription | VORTEX SMT-2 card (certified, lane round 223): `nemo_linear` + `zdf_drag_in_matrix` + `zdf_baroclinic_only` + `barotropic_drag_substep`; ORCA2 lane OMT-2 uses the same bundle | `nemo_testcase_recipe.py` SMT-2 block; `ocean_tendency_common.nemo_drag_r_from_speed_sq` |
| full ORCA2 card | lists `linear_implicit_bottom_drag` as UNMEASURED | `build_orca2_zps_card` |
| combination top + bottom | **refused** by the model ("ORCA2 G5 no-bottom-drag composition only") | `ocean_model_latlon_cgrid.py` construction guard |

So G5 does NOT run ORCA2's bottom drag; G7 is real work.

**Premise correction.** The round brief says G7 keeps "G5's flux-form
momentum advection".  G5 runs momentum advection OFF (`ln_dynadv_OFF=.true.`,
`np_LIN_dyn`, `dynadv.f90:185`).  G7 keeps G5's setting, which is what
"G5 + bottom drag, nothing else" requires; G6's vector form (and its pending
Decision-132 question) is not inherited.

## 2. Claim and exact target

G7 = the certified G5 deck plus ORCA2's `&namdrg` selection and nothing else.
Deck `scripts/validate/ocean_fidelity/testcases/nemo_testcase_g7_orca2drg/namelist_cfg`
(SHA-256 `48a8ea9d…cc28`); `namelist_ice_cfg` is G5's, byte-identical.

| key | G5 | G7 = ORCA2 |
|---|---|---|
| `namdrg.ln_drg_OFF` | `.true.` | `.false.` |
| `namdrg.ln_lin` | `.false.` | `.true.` (ORCA2 `namelist_cfg:251`) |

`ln_non_lin`, `ln_loglayer` stay `.false.`; nothing else moves (a committed
test pins the assignment diff).

### What NEMO derives (READ)

| item | G7 | cite |
|---|---|---|
| `ndrg = np_lin`, `l_zdfdrg = .FALSE.` (set once) | | `zdfdrg.f90:540-547` (`drg_init` :427) |
| `rCd0_bot = rn_Cd0 * ssmask` (boost off) | | `zdfdrg.f90:546`, mask `:525` |
| `rCdU_bot = - rCd0_bot * rn_Uc0`, halo included | | `zdfdrg.f90:276` (`zdf_drg_lin` :258) |
| barotropic `pCdU = r1_2*((bot pair) + (top pair))` — ONE statement | | `dynspg_ts.f90:1431-1432` (`dyn_drg_init` :1404) |
| bottom residual, then top residual, into `pu_RHSi` | | `dynspg_ts.f90:1462-1463`, `:1487-1488` |
| `dyn_zdf` RHS: bottom (`mbku`) then top (`miku`) | | `dynzdf.f90:166-169`, `:175-178` |
| `dyn_zdf` diagonal: bottom then top | | `dynzdf.f90:306`, `:312`; V `:473-474`, `:479-480` |
| explicit `zdf_drg_exp` | not run (`ln_drgimp=T`) | `dynzdf.f90:120` |
| TKE/GLS/OSM readers of `rCdU_bot` | not run (`ln_zdfcst`) | — |

## 3. Card change (to be built after this commit)

- New case `ORCA2G7-drg` = G5's rung tables + the SMT-2 linear-drag bundle
  (reused, not re-written): `nemo_linear`, `rn_Cd0`/`rn_Uc0` from the SMT-2
  constants (same `namelist_ref:834-835` values), `zdf_drag_in_matrix`,
  `zdf_baroclinic_only`, `barotropic_drag_substep`.  G5's statements (ice top
  drag, stage-2 stored thickness, stage-3 salt) stay selected.
- The ONE missing statement is the combined barotropic coefficient
  (`dynspg_ts.f90:1431-1432`): the existing top-drag face helper takes NEMO's
  bottom T-field `rCdU_bot` in place of its explicit zero pair.  The model
  guard is relaxed only for the full bundle (bottom drag in matrix + substeps
  + baroclinic-only); any partial combination still raises.
- Residual and `dyn_zdf` consumers already run bottom then top, NEMO's order;
  unchanged.
- Every other card unchanged (G5 passes no bottom field: its helper result is
  the same expression).  Card is fail-closed: `ORCA2G7_UNMEASURED =
  (nemo_post_si3_surface_records, g7_record_not_admitted,
  linear_bottom_drag_multilevel_step)`.

## 4. Predictions (frozen; scored when the record exists)

- **P1 (READ+PREDICTED).** NEMO's G7 log prints
  `linear BOTTOM friction (constant coef = Cd0*Uc0 =` and
  `implicit friction … ln_drgimp = T` (strings as printed by the ORCA2 lane's own run, `oracle_phase2v_tke_boundary_np2/ocean.output:1097,1109`); falsified if either is absent.
- **P2 (PREDICTED).** The record is admitted with passivity PASS (writer set
  unchanged from G5).
- **P3 (PREDICTED).** G7 is NOT bit-identical at kt 1 on the first scoring.
  The first non-bit statement is one of, in this order of likelihood:
  (a) the pivot / north-fold V-row bottom rate — the shared bottom face
  helper has no fold image (north V row padded from the interior, bottom
  level 0) while NEMO computes `rCdU_bot` in the folded halo;
  (b) the bottom barotropic residual's arithmetic form — the shared path
  divides by the live column depth and rebuilds the depth mean, NEMO
  multiplies by the stored `r1_hu_0/(1+r3u(Kmm))` and reads the carried
  `puu_b(Kmm)`;
  (c) the `dyn_zdf` diagonal association where top and bottom share one cell
  (one-wet-level columns: card adds bottom+top then the diagonal, NEMO
  subtracts bottom then top).
  Falsified if kt 1 is bit-identical on every field under both labels.
- **P4 (PREDICTED).** Magnitude: the drag rate is 4e-4 m/s, so the bottom
  damping per step on a 10 m cell is `dt*r/e3 = 10800*4e-4/10 ≈ 0.43`;
  a missing or misplaced drag on a face shows as O(10%) of that face's bottom
  velocity, a wrong association as last-place (≤1e-15 relative).

## 5. Acquisition (operator runs it)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g7_orca2drg/run.sh`,
the G6 pattern: no build (drag is a namelist switch; `zdfdrg` is compiled in
every build), G5's two round-31 binaries pinned by SHA-256, G4's inputs,
smoke 2 steps -> writer-free 100-step boundary -> three record twins ->
passivity + frame admission with `g5_records.py`.  Evidence root
`gridladder_rounds/round41/oracle_g7_r41`.
