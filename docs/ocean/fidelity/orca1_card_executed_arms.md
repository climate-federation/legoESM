# ORCA1 executed-arm table (which of the 51 isomorphism-map rows the production card actually runs)

Worktree `/tmp/wt-branch-iso`, branch `fidelity/nemo-branch-isomorphism-audit`,
HEAD `7587ba4f7` (clean). MEASURED, not read off a comment: the config below
was built by calling the real code, in fp64
(`JAX_ENABLE_X64=1`, `legoesm.core.precision.PrecisionPolicy.fp64()` set
explicitly — the oracle-fidelity Rule 1c gate).

## Assembly path (traced, not assumed)

`scripts/cluster/omip_nemo/run_standard_faithful_1deg.sbatch` (no
`--barotropic-solver`, no `--momentum-advection`, no `--convection`, no
`--gm-treguier`, no `--gm-slope-scheme` flags — all silently keep whatever the
base recipe or `_create_setup` sets)
&rarr; `scripts/run/run_omip_core2.py --grid tripole ...` (`main()`, arg parser
`_build_arg_parser` at `run_omip_core2.py:4375`)
&rarr; `build_tripole(...)` (`run_omip_core2.py:1048`), called at
`run_omip_core2.py:5924` with `args.*`
&rarr; `run_omip._create_setup("tripole", ...)` (`scripts/run/run_omip.py:1224`,
tripole branch at `:1590`) builds the BASE config via
&rarr; `nemo_match_tripole_model_config(physics=...)`
(`packages/ocean/legoesm/ocean/fidelity/nemo_match_recipe.py:394`, cfg class
`NEMOMatchTripoleRecipeConfig` at `:178`)
&rarr; back in `build_tripole`, a CLI-driven `_ovr` dict of only the
**non-`None`** flags (`run_omip_core2.py:1139-1365`) is applied via
`config.replace_flat(**_ovr)` (`:1375`) — this is the ONLY place CLI flags can
change a scheme selector; anything not in `_ovr` keeps the recipe's own
default.

**What was mocked**: nothing about any scheme/selector field. The CONFIG
object needs no mesh file — only the spatial grid (`create_tripole_grid`,
needs `data/grids/eORCA1.2_mesh_mask.nc`, absent on this machine) and the
IC/forcing loaders do. The reproduction script
(`/tmp/claude-10257/.../scratchpad/build_orca1_config.py`, not committed —
scratch) calls `nemo_match_tripole_model_config` and
`build_tripole_vmix_config`/`orca1_zdftke_config` directly with the sbatch's
literal CLI values, and applies the identical `_ovr`/`replace_flat` sequence
`build_tripole` uses, skipping only `create_tripole_grid` /
`read_mesh_mask_bathy` / `compute_woa_3d` / `_init_rest_state` (grid, mask,
bathymetry, initial state — none of these feed back into a scheme selector).
One real IWMConfig was hand-built matching `--iwm`'s own defaults instead of
loading `load_iwm_forcing`'s NetCDF file — inert for this table, since `iwm`
only changes the `A_v`/`K_v` background floor values already captured by the
`_ovr["A_v"]`/`_ovr["K_v"]` line every run takes when IWM or a vmix closure is
on.

## Resolved config (MEASURED, fp64)

```
outer_integrator = forward_euler          momentum_time_integrator = rk3 (SSP Shu-Osher, NOT rk3_ws)
tracer_time_integrator = euler            eos = wright                  eos_depth = insitu
tracer_advection = superbee               momentum_advection = vector_invariant
vertical_momentum_scheme = upwind_perturbation      vorticity_scheme = al81
coriolis_scheme = matsuno_split            pgf_scheme = smc03             pgf_quadrature = cell_integral
lateral_viscosity_operator = vector_laplacian       lateral_viscosity_e3_weighting = off
adaptive_implicit_vertadv = True           zdf_implicit_solver_evaluation = shared_thomas
implicit_vmix_e3t_now_divisor = False      implicit_vmix_dzw_slot = False
zad_qco_evaluation = generic               wzv_call2_evaluation = generic
barotropic_solver = implicit_cn            barotropic_time_filter = cosine
barotropic_coriolis = avg (inert, see S-17)   barotropic_face_depth = min_rule (inert, see S-15/16)
physics.vertical_mixing.scheme = tke       physics.convection.scheme = none
gm_redi.slope_scheme = centered            gm_redi.treguier.enabled = False   gm_redi.msc_stabilize = False
kappa_GM = kappa_Redi = 600.0
tke.{tke_matrix,tke_solver,tke_mxl_raw}_evaluation = factored/shared_thomas/factored (class defaults, NOT dino.py's nemo_literal set)
tke.n2_mode = nemo_bn2   tke.tke_n2_evaluation_stage = implicit_solve_state (NOT "step_entry")
```

**Real NEMO ORCA1 namelist, read directly** (`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg`,
`namelist_cfg` overrides `namelist_ref` — CONFIRMED, not inferred):
`ln_teos10=T`, `ln_dynadv_vec=T`, `ln_dynvor_een=T`, `ln_hpg_sco=T`,
`ln_dynspg_ts=T`, `ln_zad_Aimp=T`, `ln_zdftke=T`, `ln_zdfevd=T`
(`nn_evdm=0`, `rn_evd=100`), `ln_traadv_fct=T` (`nn_fct_h=2`, `nn_fct_v=2`),
`ln_traldf_lap=T`, `ln_traldf_iso=T`, `ln_traldf_msc=T`, `nn_aht_ijk_t=21`,
`ln_ldfeiv=T`, `nn_aei_ijk_t=21`, `ln_trabbl=T` (`nn_bbl_adv=2`),
`ln_qsr_rgb=T`, `ln_dynldf_lap=T`, `ln_dynldf_lev=T`.

## The 51-row table

Columns: **exec** = does ORCA1's resolved config actually run this row's code
(enclosing function + branch condition traced), not just "does a field hold a
value". **verdict**: YES (matches the real NEMO arm, cited) / NO (a real NEMO
arm exists — often already used by DINO/GYRE/LOCK/OVERFLOW — but ORCA1 doesn't
select it) / NO-COUNTERPART (legoESM runs something NEMO has no equivalent of)
/ NOT_EXECUTED (the code never runs on ORCA1 regardless of what the field
says). CONFIRMED = checked against the real namelist above; PLAUSIBLE =
standard OMIP practice, namelist line not checked for this specific item.

| # | routine | exec? | running impl / why not | verdict | evidence |
|---|---|---|---|---|---|
| S-01 sbc | YES | `bulk_flux_omip.air_sea_fluxes`, driver loop, every step | YES (PLAUSIBLE) | CORE-II bulk forcing is standard OMIP practice; namelist sbc block not read |
| S-02 eos_rab | YES | `wright_eos` (paper arm) | **NO-COUNTERPART** | `ln_teos10=T` (CONFIRMED); real arm = `nemo_roquet_alpha_beta`, already used by D/G |
| S-03 bn2 | YES | `compute_buoyancy_frequency_nemo_bn2` (`n2_mode="nemo_bn2"`) | YES (CONFIRMED) | `ln_teos10=T`; matches `eosbn2.F90`; re-confirms the registry's own prior CONFIRMED finding |
| S-04 bn2 live-e3w geometry | YES (default arm) | gate needs `n2_mode=="nemo_bn2"` AND `tke_n2_evaluation_stage=="step_entry"`; ORCA1 has the first, not the second (`implicit_solve_state`) | **NO** | falls to unreferenced `nemo_bn2_depth_ladders`, not DINO/GYRE's nemo arm |
| S-05 zdf_cst | NOT_EXECUTED | ORCA1 vmix scheme = `tke`, not const/none; not this row's card | n/a | — |
| S-06 zdf_tke | YES | `physics.vertical_mixing.scheme="tke"` (`--tripole-vmix tke`) | YES (CONFIRMED) | `ln_zdftke=T` |
| S-07 zdf_tke 10 sub-selectors | YES (default arm, NEW finding) | all 10 `tke_*_evaluation` fields = TKEConfig class defaults; `orca1_zdftke_config` never sets them, unlike `dino.py:1247-1256`'s explicit `nemo_literal`/`step_entry`/`nemo_qco_live_face`/`carried_previous_step` overrides | **NO** | production TKE, despite being built as "NEMO ORCA1 &namzdf_tke mapped", silently runs the generic/Veros sub-arm on all 10 selectors |
| S-08 zdf_evd | NOT_EXECUTED | `physics.convection.scheme="none"` — `--convection` never passed | **NO** (missing physics) | `ln_zdfevd=T`, `nn_evdm=0`, `rn_evd=100` (CONFIRMED); the matching impl `enhanced_diffusion_convection` exists, unused |
| S-09 ldf_slp | YES | `gm_redi.slope_scheme="centered"` (recipe's own choice, not CLI) | **NO** | `ln_traldf_iso=T`, `ln_traldf_msc=T` (CONFIRMED); real arm = `nemo_iso_lap` + `msc_stabilize=True`, used by D/G |
| S-10 ldf_tra/ldf_eiv | YES | `static_kappa_redi_override` (constant `kappa_GM=kappa_Redi=600`) | **NO** | `nn_aht_ijk_t=21`, `ln_ldfeiv=T`, `nn_aei_ijk_t=21` (CONFIRMED, flow-dependent Treguier); `--gm-treguier` exists, unused |
| S-11 ldf_dyn (scalar A_h) | YES | config constant `A_h=1e5` | YES | SHARED, no branch |
| S-12 stp_2D pre-step | NOT_EXECUTED | requires `momentum_time_integrator=="rk3_ws"`; ORCA1 = `"rk3"` (SSP, a third lane) | n/a | registry: "no ORCA1 ... a third distinct branch" |
| S-13 dyn_spg_ts substep loop | NOT_EXECUTED (registered impl) | `barotropic_solver="implicit_cn"` &rarr; dispatches to `barotropic_implicit_latlon_cgrid.py`, which reads NONE of the explicit-substep selectors (grep-confirmed) | **NO-COUNTERPART** (biggest finding) | `ln_dynspg_ts=T` — real NEMO ORCA1 is split-explicit, same family D/L/O/G already transcribe; legoESM's ORCA1 runs an unrelated implicit Crank-Nicolson solver instead |
| S-14 external velocity update gate | NOT_EXECUTED | inside the bypassed module; gate also needs `rk3_ws` | n/a | bypassed twice over |
| S-15 backward face depth (zhu_bck) | NOT_EXECUTED | inside the bypassed module; `barotropic_face_depth="min_rule"` is an inert value | n/a | — |
| S-16 continuity/transport/spg | NOT_EXECUTED | inside the bypassed module | n/a | re-confirms the registry's own S-16 retraction |
| S-17 dyn_cor_2D barotropic Coriolis | NOT_EXECUTED (registered impl) | registered impls live in the bypassed module. **NEW**: `barotropic_implicit_latlon_cgrid.py` runs its OWN uncited forward-backward "avg"-style Coriolis unconditionally (active since `coriolis_scheme != "explicit_ab2"`) — a third, unregistered implementation | n/a (UNVERIFIED third impl) | not in the registry; flagged, not added (registry not touched per task) |
| S-18 dom_qco_r3c face thickness | NOT_EXECUTED (either impl) | both known callers (`_nemo_ws_qco_stage_faces`: rk3_ws-only; "MLF tracer transport": leapfrog-only) require a lane ORCA1 doesn't run | n/a | **corrects** the map's "NEEDS_ORCA1_OPERANDS" framing — see corrections §ORCA1-1 |
| S-19 wzv | YES | `diagnose_w_from_flux_div` (generic), confirmed to run unconditionally (same indent level as, not nested in, the rk3_ws-gated block at `omlc:4798`) | **NO** | NEMO_DUPLICATE; blocked from `nemo_literal` by missing NEMO mesh operands on `z_coord` (registry's own finding, still valid) |
| S-20 wAimp | YES | post-program site, gated `momentum_time_integrator != "rk3_ws"` (true) | YES (CONFIRMED) | `ln_zad_Aimp=T` |
| S-21 stage transport triplet | NOT_EXECUTED | `_nemo_ws_stage_transport`, cards L/O/G only | n/a | — |
| S-22 dyn_adv dispatch | YES | `latlon_cgrid_ocean_baroclinic_tendencies`, `momentum_advection="vector_invariant"`, called every stage via `self.tendencies()` regardless of outer lane | YES (CONFIRMED) | `ln_dynadv_vec=T` |
| S-23 dyn_adv_up3 vertical flux | NOT_EXECUTED | requires `vertical_momentum_scheme=="nemo_up3"`; ORCA1 = `upwind_perturbation` | n/a | — |
| S-24 dyn_zad | NOT_EXECUTED | requires `vertical_momentum_scheme=="nemo_advective"`; not selected | n/a | this IS the real NEMO arm (see S-25) |
| S-25 vertical momentum advection (non-NEMO) | YES | `flux_form_vertical_momentum_advection` (`upwind_perturbation`) | **NO** | since ORCA1 is vector-form (`ln_dynadv_vec=T`), real arm = `nemo_advective` (S-24), used by D/G, unselected |
| S-26 dyn_vor | YES | `pv_flux_al81_partial_cell` (`al81`) | **NO-COUNTERPART** | `ln_dynvor_een=T` (CONFIRMED); real arm = `een_total`, used by DINO |
| S-27 Coriolis time placement | YES | `_forward_backward_coriolis_3d` (`matsuno_split`) | **NO-COUNTERPART** | NEMO always folds Coriolis into `dyn_vor`'s RHS = legoESM's `explicit_ab2`, used by D/G; map's own ranked item 9 (HIGH stability risk to change) |
| S-28 dyn_hpg&rarr;hpg_sco dispatch | NOT_EXECUTED | cards D/L/O/G only, not this row for A | n/a | — |
| S-29 dyn_hpg on ORCA1 | YES | `compute_pressure_at_target_smc03` (`pgf_scheme="smc03"`, paper arm) | **NO** | `ln_hpg_sco=T` (CONFIRMED) — **corrects** the map's own "PLAUSIBLE ln_hpg_zps" guess; the real arm is `nemo_sco` (S-28's `_bc_ke_and_pressure_gradients`), already implemented and used by D/L/O/G, but `--pgf-scheme choices=[adcroft,smc03]` structurally can't select it |
| S-30 WS ladder written twice | NOT_EXECUTED | `momentum_time_integrator=="rk3_ws"` only | n/a | registry already says "no ORCA1" |
| S-31 stage 2/3 eos/hpg Kmm operands | NOT_EXECUTED | cards L/O/G only | n/a | — |
| S-32 dyn_ldf&rarr;ldf_lap operator | YES | `vector_laplacian_dissipation_cgrid` | **NO** | `ln_dynldf_lap=T`, `ln_dynldf_lev=T` (CONFIRMED); real arm = `nemo_div_curl`, used by D/G |
| S-33 dyn_zdf/tra_zdf solver | YES | `_apply_implicit_vertical_mixing`, `zdf_implicit_solver_evaluation="shared_thomas"` | **NO** | real arm = `nemo_literal`, used by D/L/O/G |
| S-34 e3w(Kmm) divisor | YES | legacy midpoint (`implicit_vmix_e3t_now_divisor=False`, `implicit_vmix_dzw_slot=False`) | **NO** | structural NEMO divisor (trazdf.F90:219-220) unused, same gap DINO also has |
| S-35 stage-3 zub correction | NOT_EXECUTED | rk3_ws-only sites | n/a | — |
| S-36 tra_adv_trp reuse | NOT_EXECUTED | cards L/O/G only | n/a | — |
| S-37 tra_adv dispatch | YES | `_veros_superbee_face_flux` (`tracer_advection="superbee"`); exact call site for the rk3(SSP)+euler-tracer lane not traced (NOT `_nemo_ws_rk3_tracer_pair_step`, which is rk3_ws/MLF-keyed) | **NO** | `ln_traadv_fct=T`, `nn_fct_h=2`, `nn_fct_v=2` (CONFIRMED); real arm = `fct2`, used by D/L/O/G |
| S-38 fct implicit-w | NOT_EXECUTED | ORCA1 doesn't run fct2 at all | n/a | moot until S-37 fixed |
| S-39 tracer stage weighting | NOT_EXECUTED | cards L/O/G only | n/a | — |
| S-40 tra_sbc placement | NOT_EXECUTED (registered impl) | `_check_surface_tendency_placement` is DINO-scoped; ORCA1's own site is hard-wired elsewhere, untraced here | n/a (UNVERIFIED) | — |
| S-41 tra_qsr | YES | `apply_shortwave_penetration` (`--sw-rgb-chl` &rarr; `rgb_chl`) | YES (CONFIRMED) | `ln_qsr_rgb=T` |
| S-42 bbl+tra_bbl | YES | `apply_bbl_adv_step` (driver-only, `--bbl-adv`); `bbl_adv_option=0` measured &rarr; in-model dispatch OFF | **NO** | `ln_trabbl=T`, `nn_bbl_adv=2` (CONFIRMED); faithful arm = `bbl_adv_option=2`/`apply_bbl_adv_tendency`, used by OVERFLOW, never selected; driver path adds an extra cap NEMO lacks |
| S-43 tra_ldf | YES | same as S-09 | **NO** | same evidence as S-09 |
| S-44 tra_zdf | YES | same as S-33/34 | **NO** | same evidence |
| S-45 tra_npc | NOT_EXECUTED | ABSENT, no legoESM impl for anyone | n/a | not ORCA1-specific |
| M-01 stp_MLF composition | NOT_EXECUTED | `outer_integrator="forward_euler"` (measured); MLF-only row | n/a | — |
| M-02 ssh_nxt/div_hor | NOT_EXECUTED | MLF-only | n/a | — |
| M-03 ssh_atf (Asselin) | NOT_EXECUTED | MLF-only | n/a | — |
| M-04 tracer_combine | NOT_EXECUTED | MLF-only | n/a | — |
| M-05 mlf_baro_corr | NOT_EXECUTED | MLF-only | n/a | — |
| M-06 dyn_ldf/tra_ldf at Kbb | NOT_EXECUTED | MLF-only | n/a | — |

## Summary counts

| verdict | count | rows |
|---|---|---|
| YES | 7 | S-01, S-03, S-06, S-11, S-20, S-22, S-41 |
| NO | 15 | S-04, S-07, S-08, S-09, S-10, S-19, S-25, S-29, S-32, S-33, S-34, S-37, S-42, S-43, S-44 |
| NO-COUNTERPART | 4 | S-02, S-13, S-26, S-27 |
| NOT_EXECUTED | 25 | S-05, S-12, S-14, S-15, S-16, S-17, S-18, S-21, S-23, S-24, S-28, S-30, S-31, S-35, S-36, S-38, S-39, S-40, S-45, M-01..M-06 |

51 total. 19 rows are real work items (NO + NO-COUNTERPART).

## Ranked NO / NO-COUNTERPART work items (highest climate leverage first)

1. **S-13 (+ the whole S-14..S-19 barotropic family)** — `barotropic_solver="implicit_cn"` has no NEMO counterpart; real NEMO ORCA1 (`ln_dynspg_ts=T`) is split-explicit, the SAME family D/L/O/G already transcribe (`explicit_substep`). Biggest single divergence in the table.
2. **S-37** — tracer advection runs Veros `superbee`; real NEMO ORCA1 (`ln_traadv_fct=T`, `nn_fct_h/v=2`) is FCT, already implemented and used by every other NEMO card.
3. **S-29** — PGF runs `smc03`; real NEMO ORCA1 (`ln_hpg_sco=T`) wants `nemo_sco`, already implemented (D/L/O/G) but structurally unreachable from `--pgf-scheme`.
4. **S-10** — GM/Redi kappa is constant 600; real NEMO ORCA1 (`nn_aei_ijk_t=21`) wants the flow-dependent Treguier form — `--gm-treguier` exists, unused in production.
5. **S-32** — lateral viscosity is `vector_laplacian`; real NEMO (`ln_dynldf_lap+lev=T`) wants `nemo_div_curl`, used by D/G.
6. **S-26** — vorticity scheme `al81` has no NEMO arm; real NEMO ORCA1 (`ln_dynvor_een=T`) wants `een_total`, used by DINO.
7. **S-27** — Coriolis placement `matsuno_split` has no NEMO arm; real placement is `explicit_ab2`. HIGH stability risk to flip (map's own ranked item 9).
8. **S-02** — EOS is `wright`; real NEMO ORCA1 (`ln_teos10=T`) wants `nemo_roquet_alpha_beta`, used by D/G.
9. **S-42** — BBL runs the driver-only path with an extra cap; real NEMO (`nn_bbl_adv=2`) wants the in-model `bbl_adv_option=2`, used by OVERFLOW.
10. **S-33 / S-44** — implicit ZDF solver is `shared_thomas`; real arm is `nemo_literal`, used by D/L/O/G.
11. **S-34** — e3w(Kmm) divisor is the legacy midpoint; real NEMO divisor unused (same gap as DINO).
12. **S-08** — EVD convection is entirely OFF; real NEMO (`ln_zdfevd=T`) runs it. Missing physics, not a wrong arm.
13. **S-07** — TKE's 10 structural sub-selectors are all on the generic/Veros default, not DINO's `nemo_literal` set, even though the TKE closure itself is correctly on.
14. **S-09 / S-43** — GM/Redi slope form is `centered`; real NEMO (`ln_traldf_iso=T`, `ln_traldf_msc=T`) wants `nemo_iso_lap` + `msc_stabilize=True`.
15. **S-04** — bn2 live-e3w geometry misses the nemo arm on one of its two AND-conditions (`tke_n2_evaluation_stage`).
16. **S-19** — wzv on the generic arm, blocked by a missing-mesh-operand gap already on file.
17. **S-25** — vertical momentum advection `upwind_perturbation` instead of the already-implemented, vector-form-matching `nemo_advective`.

## 2026-09-02 ORCA1 executed-arm corrections (appended, history not rewritten)

- **S-16, S-18, S-19** (map's own prior corrections): re-verified independently
  in this pass — CONFIRMED still correct. S-16/S-18 are NOT_EXECUTED for
  ORCA1 (barotropic bypass / lane mismatch, not an operand gap for S-18's own
  quantity); S-19 IS EXECUTED (generic arm), confirmed to run unconditionally
  regardless of `momentum_time_integrator`.
- **S-18 refinement**: the map's "NEEDS_ORCA1_OPERANDS" framing for ORCA1
  implied the missing NEMO mesh operand is the sole blocker (i.e. supplying it
  would let ORCA1 reach the NEMO arm). MEASURED here: this is incomplete —
  ORCA1's `momentum_time_integrator="rk3"` (SSP) and `outer_integrator=
  "forward_euler"` mean it never calls EITHER known caller of this row's
  routine (`_nemo_ws_qco_stage_faces` is rk3_ws-only; "the MLF tracer
  transport" is leapfrog-only) — the lane mismatch is the primary blocker, the
  operand gap only would matter if a caller in ORCA1's own lane existed. What
  ORCA1 actually uses for its own r3u/r3v-equivalent face thickness inside the
  rk3(SSP) lane was not located in this pass (UNVERIFIED).
- **S-29** (`dyn_hpg` on ORCA1): the map labelled "the claim that ORCA1
  resolves `ln_hpg_zps` is inference ... labelled PLAUSIBLE". MEASURED here,
  reading `namelist_cfg` directly: NEMO ORCA1 actually sets `ln_hpg_sco=.true.`
  (the ONLY `ln_hpg_*` line present), not `ln_hpg_zps`. This is a real
  correction, not a confirmation — the row's already-implemented `nemo_sco`
  arm (used by D/L/O/G) is exactly the arm ORCA1 should run, structurally
  blocked only by the driver's `--pgf-scheme` choices.
- **S-35, S-42, M-01**: re-verified independently — CONFIRMED, no correction
  needed to their existing statements. S-35 and M-01 are rk3_ws-only /
  leapfrog-only and structurally do not apply to ORCA1 at all (`NOT_EXECUTED`,
  a stronger statement than the map made about them, since the map's S-35/M-01
  text does not discuss ORCA1 specifically). S-42's existing ORCA1 text
  (driver-only `apply_bbl_adv_step`, `bbl_adv_option` stays 0) is CONFIRMED
  correct and is now additionally cross-checked against the real namelist
  (`ln_trabbl=T`, `nn_bbl_adv=2`).
- **New finding, not in the map at all**: `barotropic_implicit_latlon_cgrid.py`
  (the module ORCA1 actually dispatches to) runs its OWN uncited
  forward-backward "avg"-style barotropic Coriolis unconditionally (S-17's
  registered impls never run for ORCA1). This is a third, unregistered
  implementation of the same physical quantity S-17 names. Flagged here;
  NOT added to the registry or the map's row table per this task's scope (no
  registry/map edits authorized).
- **New finding, not in the map at all**: ORCA1's production TKE closure
  (`--tripole-vmix tke`) resolves all 10 of S-07's structural sub-selectors to
  the TKEConfig class default (the generic/Veros arm), not DINO's
  `nemo_literal` set — even though `orca1_zdftke_config` is explicitly built
  as a literal `&namzdf_tke` transcription for the namelist-VALUE fields. The
  map's own S-07 entry only names Veros recipes as reaching the default arm;
  ORCA1 reaching the same default arm (despite running TKE at all, contrary to
  what the map's §0 "physics.vertical_mixing: kpp (run-dependent)" framing
  would suggest for the untouched recipe) was not previously stated.

## UNVERIFIED

- S-01's real NEMO ORCA1 sbc block (bulk-formula variant, e.g. CORE / DFS5.2 /
  JRA55-do specifics) was not read from the namelist; PLAUSIBLE only.
- S-11's `nn_ahm_ijk_t` value was not read from the namelist (SHARED, no
  branch risk regardless).
- S-17's third, unregistered implicit-barotropic Coriolis was found by
  grep/read but not numerically compared against S-17's registered `avg` arm
  to confirm they are bit-identical in form (they look algebraically
  equivalent — same 4-pt V-at-u average — but this was not measured).
- S-18's actual r3u/r3v-equivalent mechanism for the rk3(SSP)/euler-tracer
  lane was not located (see the correction above).
- S-37's exact ORCA1 tracer-advection call site (not
  `_nemo_ws_rk3_tracer_pair_step`) was not traced; the SCHEME (`superbee`) is
  confirmed via the resolved config, the call site is not.
- S-40's ORCA1-side `tra_sbc` placement mechanism was not traced (the
  DINO-scoped selector is confirmed not to apply; what runs instead for ORCA1
  is unknown here).
- No claim in this table was checked against a running simulation — every
  verdict is source-traced + one resolved-config instantiation, matching the
  audit's own stated method throughout `nemo_branch_isomorphism_map.md`.
