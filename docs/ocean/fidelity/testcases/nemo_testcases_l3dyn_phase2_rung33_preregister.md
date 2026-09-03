# SI3 lane 3 — phase 2 rung 3.3 design preregistration

Issue: climate-federation/legoESM #1699

Branch: `fidelity/nemo-testcases-l3-si3dyn-codex`

Oracle root:
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final`

This design is committed before any rung-3.3 legoESM implementation or
candidate/oracle trajectory score.  The immutable oracle consists of 485
fp64 `ice_stp` entry frames and the post-step-485 ice restart.  This boundary
therefore makes **no rung-3.3 trajectory claim**.

## Resolved card: one permitted composition

The card is the shipped `ICE_ADV2D` case with the phase-1 overlay already run
as `ICE_ADV2D_RHG_OMIP_L3`.  Its resolved files, rather than defaults, are the
configuration authority:

| choice | resolved value | provenance and disposition |
|---|---:|---|
| dynamics branch | `ln_dynRHGADV=T`; all other `ln_dyn*` false | `output.namelist.ice:219-222`; selects rheology, advection, `Hpiling`, and `zapsmall` in `icedyn.F90:137-142` |
| rheology | EVP + adaptive EVP | `output.namelist.ice:280-288`: `ln_rhg_EVP=T`, `ln_aEVP=T`, EAP/VP false, `rn_creepl=2e-9`, `rn_ecc=2`, `nn_nevp=100`, convergence check 0 |
| strength | H79, unsmoothed | `output.namelist.ice:252-259`: `rn_pstar=20000`, `rn_crhg=20`, other strength arms false; ORCA1 source rows are `ORCA1-omip/EXPREF/namelist_ice_cfg:52-55` |
| advection | Prather | `output.namelist.ice:293-297`; ORCA1 source rows `:75-76` |
| grid and clock | 99 x 99 physical T cells, 3000 m square, 1200 s, 485 steps, two-cell halo, one rank | `ocean.output:64-65,178-180,194,468`; `usrdef_hgr.F90:91-137`; `output.namelist.dyn:25-33,67` |
| topology | periodic in x and y | `ocean.output:182-183`; no fold or wall arm |
| Coriolis | zero | `output.namelist.dyn:27`; the case writes zero `ff_t/ff_f` in `usrdef_hgr.F90:145-156` |
| lateral ice boundary | `rn_ishlat=2` | `output.namelist.ice:225`; `fimask` follows `icedyn_rhg_evp.F90:208-225`; all physical top-level masks are one in this periodic ocean, so the no-slip repair has no physical boundary on which to act |
| landfast | OFF | user choice and `output.namelist.ice:226`; the source sets every basal stress to zero at `icedyn_rhg_evp.F90:364-373`.  **UNVERIFIED-deferred**, not implemented or inferred from this rung |
| open boundary | OFF | `output.namelist.dyn:302`; calls at `icedyn_rhg_evp.F90:751-752` are inactive |
| ocean forcing | resting U/V, zero SSH; non-embedded ice | `output.init.nc` has zero `vozocrtx`, `vomecrty`, and `sossheig`; `output.namelist.dyn:126,137`; `ice_var_sshdyn` returns bare SSH when non-embedded (`icevar.F90:1060-1077`) |
| atmosphere forcing | `utau_ice=1.3 N m-2`, `vtau_ice=0` | shipped `ICE_ADV2D/MY_SRC/usrdef_sbc.F90:83-96`; independently present as constant fp64 arrays in `output.init_ice.nc` |
| ocean drag | `rho0=1026`, `rn_Cd_io=0.005` | `ocean.output:162`; `output.namelist.ice:94`; copied-case `icestp.F90:332` sets `drag_io` to that value |
| state scope | `jpl=1`, three ice and three snow layers, thermodynamics off | `output.namelist.ice:2-7`; no multi-category or thermodynamic arm |

The ORCA1 sources are not one undifferentiated configuration.  The reference
deck declares `jpl=5`, `nlay_i=10`, and `nlay_s=5`
(`ORCA1-omip/EXPREF/namelist_ice_ref:24-26`), while the checked EXPREF overlay
explicitly replaces those with `1/3/3`
(`ORCA1-omip/EXPREF/namelist_ice_cfg:24-26`).  The overlay also selects
`ln_dynALL=T` and `ln_landfast_L16=T` (`namelist_ice_cfg:44-46`).  Rung 3.3
instead uses the shipped ICE_ADV2D case's `ln_dynRHGADV=T`, landfast OFF,
`jpl=1`, and `3/3`; these are explicit out-of-rung deviations from the ORCA1
dynamics composition (and from the reference deck's category/layer defaults),
not silent ORCA1 inheritance.  The only ORCA1-resolved choices imported into
this rung are `rn_ishlat=2` (`namelist_ice_ref:57`), H79 with
`rn_pstar=20000`, `rn_crhg=20`, and smoothing disabled
(`namelist_ice_cfg:52-55`), adaptive EVP with `rn_creepl=2e-9`, `rn_ecc=2`,
`nn_nevp=100`, and convergence output disabled
(`namelist_ice_ref:108-116`), and `rn_Cd_io=0.005`
(`namelist_ice_ref:136`).

`rn_uice` and `rn_vice` remain 0.5 in the resolved namelist, but they are
dead inputs under `ln_dynRHGADV`; only the prescribed-stress routine supplies
the momentum forcing.  `rn_relast=0.333` is likewise resolved but the aEVP arm
uses the adaptive coefficients instead (`icedyn_rhg_evp.F90:235-247`).
The card validator will reject any different composition, including nonzero
Coriolis/SSH/ocean current, landfast, BDY, a second category, a different
strength law, a different subcycle count, or a non-Prather transport selector.
There is no fallback to a nearby supported arm.

Immutable-input SHA256s at preregistration:

| artifact | SHA256 |
|---|---|
| `mesh_mask.nc` | `a74a6cd55b11084715b0e4e964f820ea9c2e222a73bd02f960edf73b58970503` |
| `output.init.nc` | `fc16ec3862793bc0d35ab98b2bc2dc1b7f70acc6d3f65139f6e8d0f82ac65451` |
| `output.init_ice.nc` | `bf73e0b3eb3e5d229cc3c25afa7ba26de147d4435454915bce67db423e1c19b9` |
| resolved ice namelist | `3990bbc8b8a71499f0c3958338eea393697308768a8010b426b16ea8a20873b5` |
| resolved ocean/SAS namelist | `fa4c442c361b44fc14291efe19a98086ce53a9bf89d874467ce160196724b460` |
| ordered 485-frame aggregate | `9190cf89c8131fdd1087d80903b6168bc6c55c3ba43d9326a83072052f402004` |
| post-step-485 ice restart | `863ee8ea8204618baa57b0d3d1aeec08acff4c7e6f55a47437630a82bbd748b0` |

## One outer ice step: coverage call graph

The frame is written at `ice_stp` entry before `store_fields`
(`nemo502_MY_SRC/si3_l3/icestp.F90:154-171`).  `store_fields` then makes the
fixed outer-step copies, including `u_ice_b/v_ice_b`
(`icestp.F90:388-426`).  The active call graph and planned disposition are:

| order | active oracle call | rung-3.3 disposition |
|---:|---|---|
| 1 | `store_fields` | IMPLEMENT: retain immutable `u_b/v_b` and tracer before-state for the entire outer step |
| 2 | `ice_sbc_tau` -> shipped `usrdef_sbc_ice_tau` | IMPLEMENT: load the documented constant T-point stress, not wind-derived bulk drag |
| 3 | `diag_set0`, `ice_rst_opn` | WAIVE for dynamics numerics; restart schema is separately gated |
| 4 | `ice_dyn` -> `ice_dyn_rhg` -> `ice_dyn_rhg_evp` | IMPLEMENT below; the concrete dispatch is proved by `icedyn_rhg.F90:75-90` |
| 5 | `ice_dyn_adv` -> `ice_dyn_adv_pra` | REUSE the existing selectable SI3 Prather arm; rung 3.2 is DEBT, so it is not inherited as an AT-BAR claim |
| 6 | `Hpiling` | IMPLEMENT/reuse the rung-3.2 source correction in the same order; it caps concentration after transport (`icedyn.F90:209-232`) |
| 7 | `ice_var_zapsmall` | IMPLEMENT/reuse after `Hpiling`; no reordering with Prather's own `zapneg` |
| 8 | final `lbc_lnk` calls | IMPLEMENT periodic T/U/V/F halo updates exactly where the concrete routines call them |
| 9 | thermodynamics | INACTIVE: `ln_icethd=F`, verified from the resolved namelist |

Inside `ice_dyn_rhg`, the pre/post `ice_cons_hsm` and `ice_cons2D` calls are
oracle diagnostics (`icedyn_rhg.F90:61-64,98-103`).  They fired no phase-1
violation, but are not candidate physics and do not replace state comparison.
The EVP restart write at `icedyn_rhg.F90:92-95` is a final-step I/O side effect;
the three carried arrays are covered explicitly below.

## Grid-point and operand registry

The implementation uses SI3's same-index, two-halo storage for the exact card:
every T/U/V/F array is `(103,103)`, while its semantic stagger is mandatory.
The physical comparison crop is `(99,99)`.  A single adapter may expose these
arrays to legoESM's existing `LatLonCGridGeometry` (whose canonical T/U/V/Q
shapes and metrics are declared at `grids/latlon.py:1174-1303`); it may not
contain a second set of physics formulas.  This preserves the Fortran index
and halo-write order needed by the oracle card.  A later production adapter,
if needed, must call this same kernel.

| quantity | point | source definition |
|---|---|---|
| `a_i,v_i,v_s,v_ip,v_il` and their totals | T | mass inputs named at `icedyn_rhg_evp.F90:88-90`; aggregation is fixed at `:270-288` |
| `m`, `m*f`, `dt/max(m,1)`, `strength`, `P/(delta+creepl)`, `delta`, `beta` | T | `:270-274`, H79 `icedyn_rdgrft.F90:1048-1056`, and EVP `:401-470` |
| `stress1_i`, `stress2_i` (`zs1,zs2`) | T | loaded from carry at `:249-252`, updated at `:457-461`, stored at `:838-843` |
| `stress12_i` (`zs12`) and shear `zds` | F | shear `:392-399`, stress `:472-491`; restart declares F at `:1082-1089` |
| `u_ice`, ice fraction/mass, air/ocean/basal/SSH/internal forces | U | declarations `:140-158`; interpolation and forcing `:276-329`; internal-force divergence `:493-503` |
| `v_ice`, ice fraction/mass, air/ocean/basal/SSH/internal forces | V | same declarations and setup; internal-force divergence `:504-510` |
| cross `v_iceU/v_oceU` | U | bracketed four-V-point interpolation `:290-293,512-514` |
| cross `u_iceV/u_oceV` | V | bracketed four-U-point interpolation `:290-293,512-514` |
| `utau_ice/vtau_ice`, `ssh_m`, `at_i`, `ff_t` | T inputs | T-to-face stress `:299-305`; slope `:315-317`; area/mass and Coriolis `:270-288` |
| `umask/vmask` | U/V | all face quantities and final velocity switches `:276-329,575-579,626-630` |
| `fimask` | F | four surrounding T masks plus `rn_ishlat` repair, then F halo, `:208-225` |

The stored `stress1/2/12` variables are SI3's transformed stress coordinates,
not legoESM's current A-grid `sigma_11/22/12`; SI3 explicitly warns against
that identification at `icedyn_rhg_evp.F90:866-877`.  The C-grid arm will name
them `stress1_t`, `stress2_t`, and `stress12_f` internally and in its restart.

## Exact setup before the 100 iterations

The card will transcribe the statement order in `icedyn_rhg_evp.F90:187-377`:

1. Form the T ice-presence masks at `at_i >= 1e-10`, and form `fimask` once.
2. Set `ecc2=rn_ecc*rn_ecc`; because aEVP is selected, set the pseudo-step
   `zdtevp=rDt_ice=1200 s`, not `rDt_ice/nn_nevp` (`:231-247`).
3. Copy all three restart stresses into working arrays and compute H79
   `P = rn_pstar * sum(v_i) * exp(-rn_crhg*(1-at_i))` where ice exists
   (`icedyn_rdgrft.F90:1048-1056`).
4. Set landfast tensile strength and every basal-stress array to zero.  No
   Lemieux formula is present behind this selector.  The resolved reference
   value `rn_lf_relax=1e-5` (`namelist_ice_ref:62`) remains registered, but its
   static-friction uses below are dead for this card.
5. Set T mass in the source order
   `330*v_s + 917*v_i + 1000*(v_ip+v_il)`, then `m*f` and
   `zdtevp/max(m,1)` (`phycst.F90:57-59`; `icedyn_rhg_evp.F90:270-274`).
6. Area-weight T concentration and mass to U and V, form the bracketed cross
   ocean currents, `m/dt`, face air stress, `rho0*A*Cd`, and `-m*g*grad(ssh)`
   exactly as `:276-317` spells them.  Uniform geometry does not authorize
   reassociation of these expressions.
7. Form the no-mass and low-mass/low-concentration switches at 1 kg m-2 and
   0.001 (`:319-329`).  These switches are active near the Gaussian tail and
   are not a waiver.

The card pins `rho_snow=330`, `rho_ice=917`, `rho_pond=1000`, `rho0=1026`,
`g=9.80665`, and SI3's exact pi through existing constants/config plumbing;
none will be an inline empirical literal.  The exact source parameters
`zepsi=1e-20`, `zmmin=1`, `zamin=0.001`, and alpha floor 50 will be named
module constants with these line citations (`icedyn_rhg_evp.F90:164-166,446`).

## One aEVP subcycle, and the exact 100-subcycle order

For every `jter=1..100`, in the single `jax.lax.fori_loop`/`scan` body:

1. Compute F shear in the written metric order (`:391-399`).
2. At T, compute the area-weighted four-F shear square, divergence, tension,
   `delta`, and `P/(delta+rn_creepl)`; then apply the T periodic halo
   (`:401-427`).
3. Recompute divergence and tension rather than reusing the earlier
   temporaries (`:430-440`).
4. At T compute
   `alpha=beta=max(50, pi*sqrt(0.5*(P/delta)*1/area*(dt/max(m,1))))`, then
   update `stress1_t` and `stress2_t` with denominator `alpha+1`
   (`:442-470`).  With landfast off, `zkt=0` in these expressions.  Preserve
   the source asymmetry: `zmsk` multiplies both T stresses at `:458,460`.
5. At F take alpha as the maximum of the four surrounding T betas, take
   `P/delta` with the source's bracketed pair-of-pairs average, and update
   `stress12_f` (`:472-491`).  Unlike the two T stresses, `stress12_f` has no
   `zmsk` multiplier at `:488-489`; the implementation must reproduce that
   asymmetry.
6. Diverge the transformed stresses to U and V in the exact metric and
   parenthesis order (`:493-510`), then compute the bracketed cross-component
   ice velocities once from the velocities at the start of this paired update
   (`:512-516`).
7. Update the velocity components sequentially, never simultaneously:

   * odd `jter` (1,3,...,99): U first at `:638-687`, then V at `:690-741`;
   * even `jter` (2,4,...,100): V first at `:532-580`, then U at `:583-634`.

   For each component, compute the quadratic ocean-drag magnitude without an
   added square-root floor, assemble internal + prescribed-air + Coriolis +
   SSH + ocean forces, choose beta as the maximum of the two adjacent T
   values, and apply the aEVP quotient in `:561-562,612-613,667-668,719-720`.
   The second component's energy-conserving Coriolis stencil sees the newly
   updated first component, while its precomputed cross velocity does not.
   That distinction is part of the program even though this card's `ff_t=0`.
   The aEVP static-friction alternatives at `:556-559`, `:607-610`,
   `:662-665`, and `:714-717`, including `rn_lf_relax=1e-5`, are
   **WAIVED-dead for this card**:
   landfast OFF makes `ztaux_base=ztauy_base=0` at `:365-369`, reducing their
   predicate to `zRHS < 0 .AND. zRHS >= 0`, which is unsatisfiable.  The normal
   quotient arm is therefore the only active update; a later landfast card
   must implement and test the static-friction alternative.
8. Apply the no-mass/low-mass switches and fast-mask factor in source order,
   then perform the paired U/V periodic halo update (`:575-579,626-634,
   681-685,733-741`).  The fast-mask provenance is the file-field read in
   `icedyn.F90:113-118`, independent of the landfast selector; the masks resolve
   to zero for this card, but the card must verify that input rather than infer
   zero from `ln_landfast_L16=F` or silently omit the factor.
9. Skip AGRIF, BDY, convergence output, and within-subcycle strength advection:
   the build has no AGRIF, `ln_bdy=F`, `nn_rhg_chkcvg=0`, and the source-local
   `ll_advups=.FALSE.` at `:180-182,745-780`.

All 100 iterations execute; there is no early convergence branch.  After
iteration 100, recompute F shear and T tension/shear/divergence/delta from the
final velocity in `:788-836`, halo T/T/F stresses, and carry the three stress
arrays to the next outer step (`:838-843`).

### Loop-extent and halo contract

For this one-rank card, `nn_hls=2`, `ntsi=ntsj=3`, and `ntei=ntej=101`.
`DO_2D(L,R,B,T)` expands to `i=ntsi-L:ntei+R` and
`j=ntsj-B:ntej+T` (`do_loop_substitute.h90:53-57,117-125`).  The table gives
inclusive Fortran cells followed by Python half-open slices in the planned
same-index `(103,103)` storage.  “Periodic halo” means the same stagger-aware
one-rank `lbc_lnk` mapping as NEMO; it is not a generic `roll` over cells the
source did not write.

| kernel/source | NEMO write cells | same-index Python write | edge/halo source |
|---|---|---|---|
| F shear, `:392-399` | `i,j=1:102` | `[0:102,0:102]` | reads the already full-halo U/V velocity; outer high row/column is deliberately not written |
| first T `delta`/`Pdelta`, `:401-426` | `i,j=3:101` | `[2:101,2:101]` | `lbc_lnk` at `:427` fills both T halo widths periodically before any consumer |
| duplicated T div/tension + alpha + `stress1/2`, `:430-463` | `i,j=2:103` | `[1:103,1:103]` | deliberately reaches `jpi,jpj`; low outer halo remains from the prior carry/update, and no stress halo exchange occurs here, exactly as the source comment says |
| T beta, `:467-469` | `i,j=1:103` | `[0:103,0:103]` | every allocated cell is overwritten from halo-complete `Pdelta`, so no beta exchange is needed |
| F alpha/`Pdelta` + `stress12`, `:472-491` | `i,j=1:102` | `[0:102,0:102]` | surrounding T beta is already full; high outer F row/column remains from prior carry/update |
| stress divergence + cross U/V, `:495-516` | `i,j=2:102` | `[1:102,1:102]` | consumes the deliberately wide T/F stress writes; no halo exchange of these temporaries |
| first component on either parity, `:532-580` or `:638-687` | `i,j=2:102` | `[1:102,1:102]` | halo-1 region is updated from registered forces/stresses |
| second component on either parity, `:583-632` or `:690-739` | `i,j=3:101` | `[2:101,2:101]` | physical interior only; paired full U/V `lbc_lnk(...,ldfull=.true.)` at `:634`/`:741` then reconstructs both halo widths periodically |
| final F shear, `:791-798` | `i,j=1:102` | `[0:102,0:102]` | reads final full-halo velocities; same deliberate high-edge retention as subcycle shear |
| final T diagnostics, `:800-836` | `i,j=3:101` | `[2:101,2:101]` | diagnostics are physical-interior values; final T/T/F stress exchange at `:838` reconstructs both stress halos before restart carry |

## A-grid versus C-grid implementation map

The new solver belongs beside the A-grid twin `evp_solver` in the existing
`packages/ice/legoesm/ice/dynamics.py:377`; it is not a second ice model and
does not belong in `rheology.py`.  The selector design is a default-preserving
`rheology_staggering="a_grid" | "si3_c_grid"` plus a scheme selector.  Existing
`evp`/`mevp` remain valid only with `a_grid`; SI3 adaptive EVP is valid only as
`scheme="si3_aevp", staggering="si3_c_grid"`.  Validation raises on all other
cross-products, preventing Frankenstein combinations.

| existing legoESM piece | disposition for SI3 C-grid aEVP | evidence |
|---|---|---|
| `ice_strength` | REUSE the formula with SI3 effective thickness `sum(v_i)`; put the T `at_i > epsi10` guard in the C-grid caller, outside this shared helper | algebra at `rheology.py:178-204` equals H79 `icedyn_rdgrft.F90:1048-1056`, whose guard at `:1048` surrounds the formula at `:1050`; moving that guard inside the shared helper would change its A-grid contract |
| `LatLonCGridGeometry` and beta-plane factory | REUSE geometry/metrics; exact-card adapter supplies NEMO same-index halos | stagger contract `grids/latlon.py:1174-1303`; factory metrics `:2034-2157` |
| current A-grid `strain_rates` | KEEP for current users; NOT reusable in C arm | centered co-located derivatives at `rheology.py:233-336` versus SI3 T/F operators `icedyn_rhg_evp.F90:391-440` |
| `delta_deformation`/`vp_stress` | KEEP; NOT reusable as a composed helper | current tensor components and generic powers at `rheology.py:343-429`; SI3 uses transformed stresses, area-averaged F shear, creep only in `P/delta`, and fixed statement order `:401-491` |
| `evp_stress_update`/`mevp_stress_update` | KEEP; NOT reusable | scalar relaxation at `rheology.py:436-526,533-628`; SI3 has cell-varying alpha/beta, T/T/F placement, and distinct formulas `:442-491` |
| A-grid `stress_divergence` | KEEP; add a C-grid variant in the same arm | co-located centered form `dynamics.py:111-230` differs from SI3 transformed T/T/F divergence `:493-510` |
| A-grid `evp_solver`/`mevp_solver` | KEEP and leave defaults unchanged; NOT reusable | simultaneous co-located stress/velocity system `dynamics.py:377-592,599+` versus parity-dependent U/V sequential aEVP `:381-782` |
| existing air/ocean stress helpers | NOT reusable in exact arm | they derive air stress from wind and add a square-root floor (`dynamics.py:303-370`); this case prescribes T stress and SI3's C-grid ocean magnitude has no floor |
| `lax.fori_loop`/`lax.scan` pattern | REUSE control-flow pattern only | it contains no physical formula; both paths must produce identical fp64 values and finite gradients |
| SI3 Prather arm and rung-3.2 corrections | REUSE, with no copied kernel | `transport.py` selectable implementation and `nemo_adv2d_testcase_recipe.py`; its existing DEBT remains visible |

No constitutive, divergence, drag, or velocity-update helper is physically
identical between the A-grid and SI3 paths, so factoring one would merge
different formulas.  `ice_strength` is the one identical physics function and
must be shared.

## Prognostic and restart contract

The C-grid carry for this card is:

* the existing 16 transported contents and all five Prather moments for each
  content (80 moment arrays);
* `u_ice` at U and `v_ice` at V;
* `stress1_i` and `stress2_i` at T and `stress12_i` at F;
* surface temperature and the carried bulk-salt diagnostic already present in
  the rung-3.2 card.

The stress carry is mandatory, not a diagnostic cache.  SI3 reads the three
arrays or initializes them to zero at cold start
(`icedyn_rhg_evp.F90:1078-1103`), evolves them through all outer steps, and
writes them at `:1105-1112`.  A split-run test will reject a missing, extra,
retyped, restaggered, or perturbed stress or moment leaf and require the next
step after restore to be bitwise identical to uninterrupted execution.

## Preregistered gates

Every scored array is fp64 on CPU after
`set_policy(PrecisionPolicy.fp64())`; the gate prints candidate state and
geometry dtypes.  The immutable pointwise normalized bar is `1e-15`.

1. **Discovery and geometry.** Re-run the phase-1 discovery contract: every
   array in `mesh_mask.nc`, the resolved namelists, and the 112-variable final
   restart must remain VERIFIED or WAIVED with its committed reason.  Compare
   the candidate's T/U/V/F coordinates, all `e1/e2`, areas/inverses, zero
   Coriolis, masks, topology, and 2-cell halo layout against the oracle.  An
   unaccounted array is fatal.
2. **Cold entry.** Compare all 19 registered frame arrays at entry `kt=1`.
   U/V and all three stresses must be exact zero; candidate initial tracers are
   constructed from the shipped initializer, not loaded from the frame.
3. **First completed step (`kt=1`).** Run rheology first, then Prather,
   `Hpiling`, and `zapsmall`.  Against entry frame `kt=2`, score U, V, and the
   three carried stresses first, then all transported tracers.  Thus
   “velocity, stresses, then advected tracers” is an ordering of registered
   results, not an assertion that the existing frames expose a hidden
   post-rheology intermediate.
4. **Trajectory/first divergence.** Sweep the 485 available entry frames in
   increasing `kt`, then the post-step-485 restart.  Within each boundary use
   U, V, `stress1_i`, `stress2_i`, `stress12_i`, then the registered tracer
   order.  Stop and report at the first numeric row above `1e-15`; never call a
   DEBT row matched.  Final restart comparison additionally covers all 80
   moments and all three stresses.
5. **JIT/gradient.** JIT the complete 100-subcycle dynamics + Prather outer
   step and require finite nonzero reverse-mode gradients through active ice.
   The nondifferentiable mask/threshold surfaces are tested away from equality;
   no stop-gradient or alternate differentiable physics arm is allowed.

Counts are discovery-driven and will be stamped by the implemented gate;
this design does not freeze a hand-written count that could omit a new oracle
array.

Planted violations must independently turn the gate red for: unaccounted mesh
and restart arrays; perturbed U, V, each stress stagger, and a Prather moment;
swapped T/F stress registration; one missing stress at restart; a perturbed
coefficient/metric weight in the `:495-510` stress-divergence stencil; 99 or
101 subcycles; a nonzero Coriolis/SSH/ocean-current input accepted by this
narrow card; changed `Pstar`, `Crhg`, creep, eccentricity, drag, stress, clock,
dtype, or topology.  The stress-divergence-weight control is the ordering
control that **must bind on this exact card**: its nonzero initial concentration
gradient creates nonzero H79 stresses and internal force.

Two source-order mutations are instead preregistered **EXPECTED-INERT on this
card**, and must not be advertised as planted controls.  Reversing odd/even
component order or making the U/V pair simultaneous cannot change the card:
`ln_corio=F` makes `pff_t=0` (`usrdef_hgr.F90:154-155`) and hence `zmf=0`
(`icedyn_rhg_evp.F90:272`), so `zCorU=zCorV=0`; cross velocities are computed
once before either update at `:512-514`; and stresses are fixed before the
pair.  U and V therefore share no updated operand inside a subcycle.  The only
remaining loop-extent distinction is erased by the full two-halo exchange at
`:634/:741` on this one-rank doubly periodic domain.  A future nonzero-Coriolis
variant card is required before those mutations become binding controls.
Every touched Python file must pass both coefficient ratchets.

## Scaling before ownership

At the first over-bar boundary, the gate will record maximum absolute error,
normalized maximum, ULP distance, nonzero-cell count, oracle magnitude, and
the maximizing coordinate for all five dynamics carries and all tracer
families.  A source-ordered replay from the identical preceding entry state
will checkpoint, in order, setup mass/strength/forcing; each subcycle's
strain; `delta/Pdelta`; alpha/beta; T/T/F stresses; U/V internal force; first
component update; second component update; and halo.  Error growth will be
reported against both operand magnitude and subcycle/outer-step index before
an owner is named.

If candidate and source-ordered replay differ, the first differing operand is
the implementation owner.  If they are byte-identical but both differ from
the next immutable oracle frame, ownership remains **UNMEASURED** pending an
oracle-side internal dump from a new copy of the shipped case; an explanation
or magnitude trend cannot clear it.  No shipped tree or shipped case will be
edited to obtain that dump.

## End-of-task ASKED / UNASKED choice list

**ASKED choices:** design before code; one ice model; SI3 C-grid aEVP as a
selectable arm beside the existing solver in the dynamics module; preserve the current
A-grid paths and every default; ORCA1 H79 `Pstar/Crhg`, adaptive alpha/beta,
100 subcycles, jpl=1, case-resolved zero Coriolis/resting ocean/prescribed
stress; landfast off; reuse Prather; fp64 CPU; geometry, first-step,
first-divergence, restart, JIT/gradient, and planted-violation gates; scaling
before ownership; no shipped-NEMO edits; no push.

**UNASKED choices:** no general nonzero-Coriolis, nonzero-SSH, moving-ocean,
landfast, BDY, AGRIF, tripole, multi-category, VP/EAP, thermodynamics,
ridging/rafting, or coupled-ocean certification; no production default switch;
no second ice model or duplicate Prather/strength implementation; no tolerance
relaxation; no trajectory or fidelity claim at this design-only boundary.
