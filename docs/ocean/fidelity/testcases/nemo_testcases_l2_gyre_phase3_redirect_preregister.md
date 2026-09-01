# NEMO testcase lane 2 — GYRE phase-3 coverage redirect preregistration

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Trigger: Claude's phase-3 review found that the trajectory card did not cover
the resolved GYRE `namdyn`, `namzdf`, and `namtra` program.  This file is
committed before changing the card or rerunning legoESM.  The prior trajectory
is retained as an honest debt round, not reinterpreted as an operator match.

## Coverage-first resolved-program inventory

Oracle: phase-3 `output.namelist.dyn`, SHA256
`66b532caccfcef7b799a2669fcbdb47fbabe319331ae7a978f8998696feee4c7`.
The inventory is driven by every resolved block whose name begins `namdyn`,
`namzdf`, or `namtra`.  `LIVE-DIFF` is a hard card-completion requirement;
`MATCH` identifies a selected equivalent; `WAIVED-INACTIVE` names the resolved
off-switch that makes all subordinate values dead.

| resolved block | oracle selection and live values | current GYRE card | disposition before correction |
|---|---|---|---|
| `namdyn_adv` | vector invariant; `nn_dynkeg=0` | flux-form UP3; centered KE | **LIVE-DIFF**: horizontal vector advection and C2 KE absent |
| `namdyn_vor` | ENE total vorticity at F points; `nn_e3f_typ=0`; masking off | AL81 declaration, bypassed by flux form | **LIVE-DIFF**: ENE carries planetary plus relative vorticity on this rotated-f grid |
| `namdyn_hpg` | `ln_hpg_sco=T`; all other HPG selectors off | `nemo_sco` plus NEMO trapezoid | **MATCH** |
| `namdyn_spg` | split-explicit TS; auto resolves 50 substeps; Demange filter 3, alpha 0.07 | explicit substep, 50, `nemo_ab3am4`, 0.07 | **MATCH** |
| `namdyn_ldf` | Laplacian, level, `rn_Uv=2`, `rn_Lv=100000`, constant builder | `A_h=0` | **LIVE-DIFF**: level momentum viscosity absent; oracle coefficient is `0.5*rn_Uv*rn_Lv=100000 m2/s` |
| `namtra_adv` | FCT, horizontal/vertical order 2, implicit option 1 | `fct2`, NEMO RK3/FCT identity | **MATCH** |
| `namtra_ldf` | Laplacian isoneutral, `rn_Ud=.02`, `rn_Ld=100000`, `rn_slpmax=.01`; triad/MSC off | `K_h=0`, no Redi object | **LIVE-DIFF**: standard rotated isoneutral Laplacian absent; coefficient `1000 m2/s` |
| `namtra_eiv` | `ln_ldfeiv=F`, EKE equilibrium off | no GM/bolus | **MATCH / WAIVED-INACTIVE** |
| `namtra_qsr` | two-band penetration; `rn_abs=.58`, `rn_si0=.35`, `rn_si1=23`; RGB/5-band/bio off | no physics card; legacy step forcing | **LIVE-DIFF**: exact two-band depth recurrence is not selected |
| `namtra_dmp` | `ln_tradmp=F` | no tracer damping | **WAIVED-INACTIVE** |
| `namtra_mle` | `ln_mle=F` | no MLE | **WAIVED-INACTIVE** |
| `namzdf` | prognostic TKE and EVD on; `nn_evdm=1`, `rn_evd=100`; backgrounds `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`; `ln_zad_aimp=F`; all other closure selectors off | no physics card; constant `A_v=1e-4`, `K_v=0`; adaptive vertical advection on | **LIVE-DIFF**: TKE, EVD, both backgrounds, and the adaptive-advection off-selection all disagree |
| `namzdf_tke` | `ediff=.1`, `ediss=.7`, `ebb=67.83`, `emin=1e-6`, `emin0=1e-4`, `nn_mxl=3`, `ln_mxl0=T`, `mxl0=.04`, `nn_pdl=1`, `ln_lc=T`, `rn_lc=.15`, `nn_etau=0`, `rn_efr=.05`, surface/bottom BC 1; no-ice options resolve inactive | no prognostic TKE state | **LIVE-DIFF**: complete prognostic closure and its parameters absent |

This table adds three live differences to the six rows called out by review:
isoneutral tracer diffusion, exact two-band penetration, and
`ln_zad_aimp=.false.`.  Inactive selectors within `namzdf` are explicitly
waived: CST/RIC/GLS/OSM/MFC/NPC/DDM/SWM/IWM are false, so their subordinate
parameters do not execute.  No namelist block in the requested namespace is
unaccounted.

## Source-grounded execution

The vector branch calls KE, vorticity, and vertical momentum advection rather
than the flux-form UP3 routine (`src/OCE/DYN/dynadv.F90:58-94`).  C2 KE is the
four-face mean of squares followed by the T-to-U/V gradient
(`src/OCE/DYN/dynkeg.F90:104-123`).  ENE constructs total `(f+zeta)/e3f` and
uses the Sadourny four-flux recurrence (`src/OCE/DYN/dynvor.F90:406-536`).
The constant level viscosity coefficient is built from
`0.5*rn_Uv*rn_Lv` (`src/OCE/LDF/ldfdyn.F90:251-254`), and the standard
isoneutral coefficient analogously from `0.5*rn_Ud*rn_Ld`
(`src/OCE/LDF/ldftra.F90:290-293`).  TKE is prognostic and updates both
viscosities/diffusivities (`src/OCE/ZDF/zdftke.F90:120-190`); EVD replaces both
with `rn_evd` where the two-level N2 test is unstable
(`src/OCE/ZDF/zdfevd.F90:41-133`).

## Preregistered completion and measurements

The corrected card must select one collapsed GYRE identity containing, as a
unit: vector-invariant momentum, ENE total vorticity, C2 KE, NEMO WS-RK3,
`nemo_sco`, level Laplacian viscosity, FCT2, standard isoneutral Laplacian,
two-band shortwave, prognostic NEMO TKE, EVD, the two background coefficients,
and adaptive implicit vertical advection off.  Existing canonical options are
reused where source search proves equivalence; a missing operator may be added
only with a literal-recurrence test and planted synthetic violation.  No
independent gate-side physics implementation may substitute for a card option.

After card completion, the phase-2 geometry and `kt=1` entry gate is rerun in
native fp64 and must remain exact with dtype receipts.  Then only `kt=2` is
rerun initially.  Before any owner label, the gate reports a scaling table for
each newly armed term: the oracle term magnitude, the old residual, the new
residual, and `residual/term` on the same wet mask and normalization.  Exact
closure may establish a match; same-scale movement without closure remains a
candidate; wrong-scale movement refutes primary ownership.  The immutable bar
is `1e-15`, first-over-bar remains fail-closed, and all ownership is
**UNMEASURED** until those rows exist.

## Redirect evidence retained

Claude's six review rows are retained verbatim in substance: vector momentum
advection, ENE vorticity, C2 KE, level Laplacian viscosity, TKE+EVD with the
20% momentum-background value mismatch, and the nonzero tracer background.
The old stage-1 normalized errors were `1.465032210595928e-4` for u and
`5.8579299575484675e-2` for v, a ratio of `399.85`.  On the 45-degree rotated
grid that directional asymmetry is structural evidence consistent with the
missing F-point vorticity path, not ownership proof.  The old u residual rises
to `5.9613298992648146e-2` at stage 3, where the barotropic correction lands.
