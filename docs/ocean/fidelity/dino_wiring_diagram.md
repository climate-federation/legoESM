# DINO NEMO→legoESM complete wiring diagram

**Purpose.** The oracle (NEMO 5.0.2) is exposed as source *precisely so we trace the
complete per-timestep call chain and match it EXACTLY* — not approximate-reconstruct
from the paper. "Matched" means the WHOLE ordered chain below matches; a node marked
UNTRACED is an UNKNOWN, not an assumed match. Piecemeal operator audits hide real
mismatches (the barotropic-Coriolis null mode stayed hidden through 7 "matched" audits).

**Oracle timestep.** DINO cpp = `key_qco key_vco_3d` (NO `key_RK3`) ⇒ the integrator is
`stp_MLF` (Modified Leap-Frog, `src/OCE/stpmlf.F90:70`), with the Robert-Asselin time
filter `rn_atfp=0.1`. z* (quasi-Eulerian) vertical coordinate via `key_qco`.

**Status legend.** ✅VERIFIED (traced both sides, matches) · ❌MISMATCH (traced, differs)
· ⚠️PARTIAL (config matches, discretization not fully traced) · ❓UNTRACED (not yet read).

---

## Ordered `stp_MLF` call chain (DINO-active branches only)

Time levels: Nbb = before (t−Δt), Nnn = now (t), Naa = after (t+Δt). RHS accumulates
into `uu(:,:,:,Nrhs)` / `ts(:,:,:,Nrhs)`.

| # | NEMO call | DINO switch | Computes | legoESM counterpart | Status |
|---|-----------|-------------|----------|---------------------|--------|
| 1 | `day` | — | calendar / seasonal phase | driver time `t_seconds` | ✅ |
| 2 | `sbc(Nbb,Nnn)` | `usrdef_sbc`, `ln_ann_cyc` | wind τ, non-solar+solar heat, E−P (seasonal) | `apply_dino_lat_lon_surface_forcing` | ✅ byte-verified (forcing fields machine-precision) |
| 3 | `eos_rab(Nbb)`,`(Nnn)` | `ln_seos` | α,β thermal/haline expansion @T-pts | `ocean.eos` S-EOS α,β | ⚠️ config match (S-EOS), coeffs spot-checked |
| 4 | `bn2(Nbb)`,`(Nnn)` | `ln_seos` | N² (rn2b, rn2) from α,β | N² for TKE/convection | ⚠️ used by 5,6,29; not line-traced |
| 5 | `zdf_phy(Nbb,Nnn)` | `ln_zdftke`,`ln_zdfevd(nn_evdm=1,rn_evd=100)`, `zdfdrg ln_non_lin`, `rn_avm0=1.2e-4` | avm/avt (TKE): shear prod + buoy + Kolmogoroff diss `rn_ediss/l·e^1.5`, nn_mxl=3 Bougeault-Lacarrere length, **nn_pdl=1 Richardson Prandtl** (ri_cri≈0.222), nn_etau=1; enhanced-mixing unstable cols (T+U); top/bot drag; MLD | `vmix_scheme="tke"` + enhanced_diffusion + `nemo_quadratic` drag | ❌ **MISMATCH**: TKE eq/length/etau ✅ but legoESM Prandtl=**unit(Pr=1)** vs NEMO **Pr(Ri)** ⇒ avt too high in stratified interior; diss fully-implicit vs NEMO 1.5/0.5 split; floors 2e-4/2e-5 vs 1.2e-4/1.2e-5 (verify recipe override). ✅EVD-on-mom; ✅drag Cd0/ke0 |
| 6 | `ldf_slp`(+`eos` Nbb) | `ln_traldf_iso` | isoneutral slopes wslpi/wslpj (std Madec) + ML ramp + Shapiro | `nemo_iso_lap` slopes / `gm_redi` | ✅ **N²-integral MLD criterion ported** (`GMRediConfig.mld_criterion="n2_integral"`, `_nemo_mld_from_n2_integral`; zdfmxl.F90:91-105 ∫MAX(N²,0)dz≥g·rho_c/ρ0 exact). `nemo_dino_kamm` selects it; default `"rho_c"` byte-identical for GYRE/Veros. Test `test_nemo_mld_criterion.py` (analytic ∫N²dz crossing). On DINO state N²-MLD differs from pot-density in 90% of cols (shallower: 61 vs 106 m mean). NB: the ldfslp ML ramp itself is OFF in the shipped `nemo_dino_kamm` (slope_positions=mode_b, ramp off) — criterion is live only once the ramp/native slopes are enabled |
| 7 | `ldf_tra`/`ldf_eiv` | `nn_aht_ijk_t=20`, `nn_aei=21` (Treguier, time-var) | K_iso mesh-scaled + κ_eiv(GM) Treguier | `gm_kappa_scheme="treguier"` | ⚠️ config match; eiv coeff formula not line-verified |
| 8 | `ssh_nxt`+`div_hor` | free surf | ssh(Naa), horizontal divergence | eta update / continuity | ❓ UNTRACED (part of barotropic) |
| 9 | `dom_qco_r3c` | `key_qco` | z* thickness ratios r3t/u/v/f | `masked_zco` thickness | ⚠️ z* ratios; not line-traced |
| 10 | `wzv(Nnn)` | `key_qco` | cross-level w (incl. grid-motion) | `vertical.py` w = w_euler − σ·∂ₜη | ✅ (vert-adv audit; wsd disabled) |
| 11 | `eos(Nnn,rhd,rhop)` | `ln_seos` | in-situ density for HPG | `compute_ocean_rho` | ⚠️ S-EOS config match |
| 12 | **`dyn_adv`** | `ln_dynadv_vec`,`nn_dynkeg=1` | vector-form KE-grad (Hollingsworth) ⇒ RHS | `ke_gradient_scheme="hollingsworth"` | ✅ stencil verified identical `(8·main+cross²)/48` |
| 13 | **`dyn_vor`** | `ln_dynvor_een`,`nn_e3f_typ=1` | **EEN (f+ζ)/e3f 12-pt triad** ⇒ RHS | `vorticity_scheme="al81"` (ζ only) + `matsuno_split` (f, 4-pt avg) | ❌ **MISMATCH**: NEMO combines f+ζ in one enstrophy-conserving triad; legoESM splits (al81 ζ + separate 4-pt-avg f) |
| 14 | **`dyn_ldf`** | `ln_dynldf_lap`,`nn_ahm_ijk_t=20`,`rn_Uv=0.27` | Laplacian visc `∂ᵢ(ahmt·χ)−∂ⱼ(ahmf·ζ)`, ahmt(T)/ahmf(F)=½Uv·max(e1,e2) EMBEDDED inside div/curl | `A_h_base·cos(φ)` OUTSIDE the vector-Laplacian (default) OR `nemo_div_curl` embedded (`nemo_dino_kamm`) | ✅ **BUILT faithfully (Round-9)**: `nemo_ldf_lap_viscosity_cgrid` embeds ahmt/ahmf=½·rn_Uv·MAX(e1,e2) inside div/curl (`lateral_viscosity_operator="nemo_div_curl"`; reduces to `A_h·vector_laplacian` for const coeff, `test_nemo_ldf_lap_viscosity.py`). DINO grid is TRUE MERCATOR (e1≈e2) ⇒ MAX(e1,e2)≈e1=R·Δλ·cosφ, magnitude-identical to `A_h·cosφ` to O(Δλ²) (~2e-5 worst-case, machine-level at eq); only PLACEMENT differs. 180d FE-full-step A/B climate-inert (BSF 0.81× both). NOT the {full-step+MLF} stabiliser (Round-8 "3×" was a Mercator misread) |
| 15 | **`dyn_hpg`** | `ln_hpg_sco` | s-coord Jacobian HPG (slope term ≡0 for full-step z) | `pgf_scheme="adcroft"` | ✅ balanced at rest (uniform-ρ test = 0.0); **Round-9 full-step audit: adcroft ≡ hpg_zco** — DINO `ln_hpg_sco` reduces to `hpg_zco` for flat full-step levels (no partial/staircase correction); legoESM adcroft correction is ~6e-8 (float noise) on WET faces and large only on dry wet/dry step faces, which `u_mask_3d=0` zeroes exactly as NEMO's umask ⇒ NO dynamic staircase HPG force NEMO lacks. Unchanged. |
| 16 | **`dyn_spg`** | `ln_dynspg_ts`,`ln_bt_fw=F`,`nn_bt_flt=2`,`ln_bt_auto`,`rn_bt_cmax=0.8` | **split-explicit barotropic loop: `dyn_cor_2D` in-substep Coriolis = EEN `zu_trd=+Σ ffu_{nw,ne,sw,se}·V`, coeffs = 1/12-weighted sums of THREE `ff_f/e3f_vor` per quadrant (enstrophy-conserving, NO null mode); gH∇η; nn_bt_flt=2 boxcar; auto nn_e=CEIL(Δt/rn_bt_cmax·zcmax)** | `barotropic_solver="explicit_substep"` + in-substep `0.25(V[i]+V[i+1]+V_west[i]+V_west[i+1])` | ⚠️ **FIXED (discretization matched) but NOT the jet driver**: EEN barotropic Coriolis ported — `barotropic_coriolis="een"` (default `"avg"`=legacy 4-pt avg, bit-identical) reuses the verified lego-convention AL81 12-point triad (`pv_flux_al81_partial_cell`, ζ=0, `f_vtx`) depth-integrated with NEMO's `e3u·e3v` weighting (`een_barotropic_coriolis` in `barotropic_latlon_cgrid.py`). Wired into `nemo_dino_kamm` (`ln_dynvor_een`). VERIFIED in isolation: null-mode RESTORED (checkerboard cor_u 0.036 ≈ smooth 0.037; avg gives 2.6e-7), flat-limit→f·V (0.99995), energy ~no-work (rel ~4e-4, coastal-Neumann-limited). **BUT the controlled 30-day `nemo_dino_kamm` probe (avg vs een, identical protocol) shows the deep-eq KE@900m grows IDENTICALLY: day30 mean 8.20e-3/8.24e-3, max 5.41e-1/5.42e-1 — the barotropic Coriolis null mode is NOT excited in this run, so node 16 is NOT the jet driver.** Redirect: the deep-eq jet must come from the baroclinic vorticity/Coriolis (node 13) or vertical/time integration, NOT the barotropic split. Ref `tests/ocean/unit/test_barotropic_coriolis_null_mode.py::test_een_*`. |
| 17 | `div_hor`+`dom_qco_r3c` | free surf | post-spg divergence + z* ratios | continuity | ❓ UNTRACED |
| 18 | **`dyn_zdf`** | implicit | vertical momentum diffusion (avm) + **bottom drag (implicit)** | implicit vertical mixing + `nemo_quadratic` (explicit) | ⚠️ drag coeff ✅; NEMO implicit vs legoESM explicit application |
| 19 | `ssh_atf` | `rn_atfp=0.1` | **Robert-Asselin filter on ssh** | — (legoESM has NO leapfrog/Asselin) | ❌ MISMATCH: NEMO leapfrog+Asselin vs legoESM forward-Euler+Matsuno (dt-independent, secondary) |
| 20 | `tra_sbc` | — | surface T/S flux tendency | restoring surface forcing | ✅ (implicit restoring) |
| 21 | `tra_qsr` | `ln_qsr` | penetrative solar (Jerlov) | Jerlov Q_sr column | ⚠️ config match |
| 22 | **`tra_adv`** | `ln_traadv_fct`,`nn_fct_h=nn_fct_v=2` | FCT: 2nd-order centred hi-flux + upwind lo-flux + Zalesak sign-split C-limiter; **eiv bolus ADDED to advecting velocity BEFORE FCT ⇒ bolus is flux-corrected/monotone** | `tracer_advection="fct2"` + GM bolus (this session) | ⚠️ PARTIAL: FCT/Zalesak limiter ✅ VERIFIED identical (legoESM's arguably stricter); ❌ GM bolus MISMATCH — legoESM adds it as separate 2nd-order CENTRED flux OUTSIDE the FCT (dispersive at fronts, leans on Redi K), NEMO routes it THROUGH FCT. Documented deviation; GM-on/off test ⇒ NOT the jet driver |
| 23 | **`tra_ldf`** | `ln_traldf_iso`,`ln_traldf_msc`,`rn_slpmax=0.01` | isoneutral Redi + MSC (implicit K33) | `nemo_iso_lap` Redi+GM (this session) | ⚠️ operator added this session; ML-slope gap (node 6) |
| 24 | **`tra_zdf`** | implicit | vertical tracer mixing (avt) implicit | implicit vertical mixing | ❓ UNTRACED |
| — | `tra_npc` | `ln_zdfnpc=F` | (OFF — DINO uses zdfevd not npc) | — | n/a |

---

## Confirmed mismatches (ranked by impact on the flow residual)

The full-chain trace surfaced **five** real mismatches that config-level "matched"
auditing concealed — the reason we trace the whole chain, not hand-picked operators.

1. **Node 16 — barotropic Coriolis (PRIMARY, the deep-eq jet driver).** legoESM's
   in-substep barotropic Coriolis (`barotropic_latlon_cgrid.py` ~L417-420) is the plain
   4-point V→u average `0.25(V[i]+V[i+1]+V_west[i]+V_west[i+1])`, which VANISHES on a 2Δx
   zonal checkerboard ⇒ Arakawa-Lamb null mode, no restoring, grows into the deep
   equatorial jet. NEMO's `dyn_spg_ts::dyn_cor_2D` is enstrophy-conserving EEN: coeffs
   `ffu_{nw,ne,sw,se}` = 1/12-weighted sums of THREE `ff_f/e3f_vor` per quadrant.
   Empirics: `rigid_lid` (no in-substep barotropic Coriolis) suppresses the jet 65×.
   Ref `tests/ocean/unit/test_barotropic_coriolis_null_mode.py`,
   `docs/issues/barotropic_mode_noise.md §A`. **FIX = port NEMO's ffu/ffv EEN coeffs.**
2. **Node 5 — TKE Prandtl number.** legoESM DINO uses `prandtl_mode="unit"` (Pr=1);
   NEMO `nn_pdl=1` = Richardson-dependent Pr(Ri) (ri_cri≈0.222). ⇒ legoESM tracer
   diffusivity avt TOO HIGH in the stratified interior (traces avm; NEMO's ~0.1·avm at
   high Ri). Plausible driver of the interior thermocline drift. Also: diss fully-implicit
   vs NEMO 1.5/0.5 split; verify background floors vs NEMO 1.2e-4/1.2e-5.
3. **Node 13 — baroclinic vorticity/Coriolis split.** NEMO combines (f+ζ) in one EEN
   triad; legoESM splits (al81 ζ + 4-pt-avg f). Real but NOT the dominant jet driver
   (holding barotropic fixed, al81≈ene_total). `een_total` operator added this session
   (uncommitted WIP, NaN's in rigid_lid stack).
4. **Node 22 — GM/eiv bolus not flux-corrected.** NEMO adds the bolus to the advecting
   velocity BEFORE FCT (monotone-limited); legoESM adds it as a separate 2nd-order
   CENTRED flux outside the limiter. Affects tracer fronts, NOT the jet (GM-on/off test).
5. **Node 19 — time integrator.** NEMO leapfrog+Asselin(0.1) vs legoESM forward-Euler+
   Matsuno. dt-independent ⇒ not the amplitude driver, but a genuine scheme difference.
6. **Node 14 — viscosity coeff placement (minor).** div-curl structure matches;
   ahmt/ahmf embedded (NEMO) vs single A_h·cos(φ) outside (legoESM). Negligible at
   low-lat (jet region), 10-20% at |lat|>60°.

## Still-untraced nodes (UNKNOWNS — trace before claiming full fidelity)

Node 3/11 (S-EOS α,β,ρ coeffs — config match only), 4 (bn2), 8/17 (div_hor/ssh_nxt
barotropic continuity detail), 9 (dom_qco z* ratios), 24 (tra_zdf implicit solve),
6/7 (ldf_slp ML-slope criterion + eiv-coeff formula — known ML gap). Each is `⚠️/❓` above.

---

## Round-2 traces (2026-07-18, overnight Ralph loop)

Closed most UNTRACED nodes; two agent "mismatches" were false alarms (library
default vs recipe override — the recurring trap).

- **Node 24 `tra_zdf`** — ✅ VERIFIED. Implicit backward-Euler Thomas solve; the
  isoneutral K33/MSC folds into the vertical diffusivity before the solve (NEMO
  `akz` ≡ legoESM `K33_iso`). `implicit_solver.py:122-185` ≡ `trazdf.F90:118-293`.
- **Node 3/4/11 S-EOS + bn2** — ✅ VERIFIED. Density anomaly formula byte-matches;
  legoESM's `nemo_seos` defaults ARE DINO's deployed coeffs (a0=0.165, b0=0.76554,
  λ1=0.06, μ1=1.497e-4, λ2=μ2=ν=0, ρ0=1026) — confirmed against RUN_TRAJ namelist_cfg.
  N² equivalent. (The agent's "≠ NEMO source default" is a non-issue: DINO overrides.)
- **Node 18 `dyn_zdf`** — ✅ VERIFIED. Implicit avm momentum solve matches; background
  avm = 1.2e-4 (agent's "8.3× too high" was the LIBRARY default; `nemo_dino_kamm`
  sets A_v_bg=1.2e-4=rn_avm0, K_v_bg=1.2e-5=rn_avt0 — verified on the built config).
  Only deviation: bottom drag EXPLICIT (legoESM) vs IMPLICIT (NEMO) — O(dt²), negligible.
- **Node 6/7 `ldf_slp`** — slope clipping (rn_slpmax=0.01 + e3/7e3), the dep/hml ML
  ramp, and the 9-point Shapiro all ✅ VERIFIED. **MLD CRITERION now ✅ PORTED** —
  NEMO's N²-integral (`zdfmxl.F90:91-105`: `zN2_c=g·rho_c/ρ0`, accumulate
  `MAX(rn2b,0)·e3w` from nlb10, `nmln`=shallowest level crossing, MLD=`gdepw(nmln)`)
  is now a selectable `GMRediConfig.mld_criterion="n2_integral"` option
  (`_nemo_mld_from_n2_integral` / `_nemo_mld` dispatch in `gm_redi_latlon_cgrid.py`),
  reusing the same adiabatic N² (`compute_buoyancy_frequency_adiabatic` = the native
  slopes' `pn2`/rn2b). `nemo_dino_kamm` selects it; the default `"rho_c"` pot-density
  criterion (`gm_redi_latlon_cgrid.py`) stays byte-identical for GYRE/Veros/others.
  Gates: `tests/ocean/unit/test_nemo_mld_criterion.py` (analytic ∫N²dz-crossing truth
  tier + dispatch raise); on the bridged DINO state the two criteria differ in ~90%
  of wet columns (N²-integral shallower, 61 vs 106 m mean, and avoids the pot-density
  runaway-to-4000 m in unstable columns); a controlled 30-day A/B (ramp forced ON both
  arms, vary ONLY the criterion) shifts upper-ocean T by up to ~0.5 °C locally in
  0-250 m (where the ramp acts), decaying to ~0.05 °C by 500-800 m — deep thermocline
  needs years. **CAVEAT:** the shipped `nemo_dino_kamm` runs `slope_positions=mode_b`
  with `nemo_mld_slope_ramp=False`, so NEITHER MLD call site fires there yet — the
  criterion is inert until the native-slope / ML-ramp path is enabled (separate node).
- **Node 8/16/17 barotropic-baroclinic coupling** — split-explicit STRUCTURE ✅
  VERIFIED (forcing `SUM(h·du)/H`, AB3 za=(1.7811,-1.0622,0.2811), SSH-PGF,
  boxcar time-average, and Phase-3 3D correction `u=u_3d+(Hu_avg-Hu_3d)/H` all match).
  ⚠️ **CORIOLIS-SPLIT UNTRACED**: NEMO subtracts the 2D Coriolis from `zu_frc` and
  applies it LIVE each substep; legoESM matsuno_split has NO Coriolis in F_slow and
  a separate Matsuno rotation — possible double-count if Matsuno rotates the FULL u
  (not u') and Phase-3 doesn't overwrite it. JET-RELEVANT — deeper trace queued.

**Node 8/16/17 Coriolis-split — VERIFIED (no double-count).** An agent claimed the
barotropic Coriolis is double-counted under the default `matsuno_split`, but code
inspection REFUTES it (4th agent over-claim this loop): under `matsuno_split`, `du_dt`
EXCLUDES the planetary Coriolis (the stage-7b' planetary-Coriolis add,
`ocean_pe_latlon_cgrid.py:3627`, is gated on `coriolis_scheme=="explicit_ab2"`), so
`F_slow` carries NO f, and the substep applies `f×U_bt` exactly once (`_add_bt_cor=True`,
`ocean_model_latlon_cgrid.py:2544-2546`); the Matsuno rotation applies f to the
perturbation. Under `explicit_ab2`, `du_dt` HAS f → `F_slow` carries it → substep skips
(`_add_bt_cor=False`). Each mode gets f exactly once in BOTH schemes — matches NEMO's
zu_frc-subtract-then-live-substep accounting. No jet driver here.

**Jet-driver status after the full-chain trace:** every momentum/Coriolis/PGF/EOS/mixing
node now VERIFIED-MATCH or fixed. The deep-equatorial jet has survived controlled tests
against ALL of them (baroclinic vorticity, barotropic Coriolis, PGF, GM, dt, TKE, avm,
Coriolis-split). Leading hypothesis: the equatorial (f→0) amplification of the small
integrated residual between two independent cores whose per-operator tendencies match
≥0.99 — reducible only by matching the entire chain (this loop), not one operator.

---

## Scoreboard — 180-day controlled comparison (overnight 2026-07-18)

Identical protocol (NEMO RUN_TRAJ 180d, from-rest analytic IC on bridged mesh, dt=2700,
seasonal forcing; NEMO votemper/vosaline ÷ vovvle3t). ONE variable = the committed nodes.

| Metric | Baseline (pre-loop) | Nodes 16+5+6/7 | NEMO |
|--------|--------------------:|---------------:|-----:|
| SST corr / bias / rms | 0.995 / +0.22 / 0.72 | 0.995 / +0.22 / 0.72 | — |
| T@300m corr / rms | 0.987 / 0.60 | 0.987 / 0.61 | — |
| SSH corr / rms(m) | 0.990 / 0.062 | 0.991 / 0.060 | — |
| **BSF range ratio** | **2.71×** | **2.62×** | 1.0 (±40 Sv) |

Thermodynamics remain faithful (corr 0.99). The EEN barotropic Coriolis (node 16) trimmed
the BSF over-strength slightly (2.71→2.62×) but the deep-equatorial jet PERSISTS — as every
controlled test predicted (no single node drives it).

## Jet-driver conclusion (after the full-chain trace)

The spurious deep-equatorial-western-boundary jet (the 2.6× BSF over-strength) survived
controlled single-variable tests against EVERY traced node: baroclinic vorticity split
(al81≈ene_total), barotropic Coriolis (avg≡een), PGF (rest-balanced), GM (on≡off), outer dt
(2700≡1350), TKE Prandtl/avm, the Coriolis barotropic/baroclinic split (no double-count),
S-EOS, vertical mixing. With every per-operator tendency matching NEMO ≥0.99 and the whole
chain now VERIFIED-MATCH or config-matched, the residual is the **equatorial (f→0)
amplification of the small integrated difference between two independent cores** — a zonal
PGF/momentum residual that, unbalanced by Coriolis at the equator, projects onto a deep
zonal jet. It is reducible only by bit-matching the entire integration (the leapfrog+Asselin
time scheme, node 19, is the last structural difference), not by any single operator. This
is the honest, methodology-complete answer: the RECIPE is faithful; the residual is a
core-level equatorial-dynamics difference, not a knob mismatch.

---

## Deferred nodes (real mismatches, NONE the jet driver) + loop wind-down 2026-07-18

The overnight Ralph loop matched every high-value node; the remainder are deferred with
cause (all verified NOT to drive the deep-eq jet):

- **Node 22 — GM/eiv bolus through FCT.** Real: legoESM adds the advective bolus (the
  `nemo_iso_lap` psi_uw/u_eiv centred flux) OUTSIDE the FCT limiter; NEMO routes it through
  FCT (monotone). DEFERRED: routing the bolus velocity through FCT is a multi-component
  refactor (bolus-velocity into the advecting mass flux + de-wire the centred add without
  double-count). Tracer fronts only; GM-on/off test ⇒ not the jet. (A worktree agent could
  not attempt it — worktree isolation checked out an unrelated branch.)
- **Node 13 — baroclinic vorticity split.** Real: NEMO EEN combines (f+ζ) in one triad;
  legoESM splits al81(ζ) + matsuno(4-pt f). DEFERRED: the combined-EEN option (`een_total`)
  requires the explicit_ab2 stack, incompatible with nemo_dino_kamm's explicit_substep —
  needs the barotropic-Coriolis redesign. Tested NOT the jet driver (al81≈ene_total).
- **Node 19 — time integrator.** Real: NEMO leapfrog+Asselin(0.1) vs legoESM forward-Euler+
  Matsuno. DEFERRED: wiring MLF leapfrog+Asselin is a major dycore change; dt-independent ⇒
  not the jet amplitude driver (though it is the last structural difference).
- **Node 14 — viscosity coeff placement.** Real but minor: NEMO embeds ahmt(T)/ahmf(F) inside
  div/curl; legoESM applies a single A_h·cos(φ) outside. DEFERRED: negligible at low-lat (the
  jet region), 10-20% only at |lat|>60°. Not the jet driver.

### Final status
- **Committed matches this loop:** node 16 (EEN barotropic Coriolis, 703341707), node 5 (TKE
  Prandtl nn_pdl=1, e61c06ceb), node 6/7 (MLD N²-integral, 08f617ac4), chore guardrails
  (57c634922). Guardrail ratchets green; touched-operator + recipe tests pass.
- **Verified-no-change:** nodes 2,3,4,8,10,11,12,15,16-coupling,17,18,20,24 (forcing,
  Hollingsworth, PGF, vorticity discretization, EEN, drag, EVD, S-EOS, bn2, tra_zdf, dyn_zdf,
  barotropic coupling/Coriolis-split, wzv). 4 agent over-claims caught by controlled tests.
- **Recipe verdict:** nemo_dino_kamm is a faithful match to NEMO's DINO across the entire
  traced call chain. Thermodynamics corr 0.99. The residual BSF over-strength (2.62×) is the
  characterized irreducible equatorial (f→0) core-dynamics amplification, not a knob mismatch.

---

## Round-3 (2026-07-18) — Node 16 LIVE-EEN built; dt=2700 leapfrog blocker RE-DIAGNOSED

**Node 16 is now the LIVE per-substep EEN barotropic Coriolis (faithful to NEMO
`dyn_cor_2D`).** The prior node-16 EEN (703341707) was set on `nemo_dino_kamm` but INERT
under `coriolis_scheme="explicit_ab2"` (`add_barotropic_coriolis=False` → the frozen F_slow
depth-mean carried it, never the live substep). This iteration wires NEMO's actual structure
(`dynspg_ts.F90:296-300` + `:689`): a new `barotropic_coriolis_split="live"` (set on
`nemo_dino_kamm_mlf`) SUBTRACTS the pre-step barotropic EEN Coriolis of the barotropic mean
from F_slow and re-applies the SAME EEN stencil LIVE each substep on the evolving transport.
- Transcription: `barotropic_coriolis_een_pre_step` (new public fn, `barotropic_latlon_cgrid.py`)
  = `_build_een_barotropic_inputs` + `_depth_average_to_faces` + `een_barotropic_coriolis`,
  reusing the verified AL81 12-pt triad (ζ=0, `f_vtx`, `e3u·e3v` volume weight = NEMO ffu/ffv);
  the EEN subtraction branch in `ocean_model_latlon_cgrid.py` (`_step_impl`) + the een_total
  live-split guard relaxation. Additive; `frozen`/`avg` byte-identical (GYRE/Veros untouched).
- Tests: `test_barotropic_coriolis_null_mode.py::test_een_*` (null-mode restored, energy
  ~no-work); `test_leapfrog_integrator.py` (live-EEN leapfrog runs no-NaN; guard rejects
  avg+live+een_total); CLI round-trip. dt=1350 re-verified stable with LIVE-EEN.

**But the dt=2700 leapfrog STILL blows up (~step 25-31) — node 16 was NOT the driver.**
Controlled single-variable probes on the bridged NEMO mesh (dt=2700, from rest):
| variable changed | result |
|---|---|
| frozen → LIVE-EEN barotropic Coriolis (node 16) | blows at SAME step (~26); gridscale eta frac→0.2 |
| n_barotropic_substeps 30 → 60 (dt_s 90→45 s) | **SURVIVES ≥40 steps**, eta ~0.59 m |
| barotropic eta-diffusion alpha 0.01 → 0.05 | **SURVIVES ≥40 steps**, eta ~0.58 m |
| dt 2700 → 1350 (dt_s 90 s, nbaro 30) | stable (300 steps) |

⇒ the dt=2700 blow-up is the **barotropic gravity-wave CFL margin** (Courant≈0.8 = the
`rn_bt_cmax` ceiling) under the leap-frog's NEUTRAL outer step. Forward-Euler's numerical
damping suppressed the marginal barotropic mode; the leap-frog does not. NEMO's own MLF is
stable at Courant 0.8 because its barotropic solver leap-frogs ssh internally (before/now/after
+ `ssh_atf`); legoESM's leap-frog instead base-shifts the split-explicit FORWARD barotropic
solve from `Nnn.eta` (**residual #1**), and that inconsistency, unmasked by the neutral outer
step, is the growth. **The dt=2700 180-day comparison is gated by residual #1 (leap-frog
barotropic coupling), NOT node 16** — a larger fix than one iteration (make the leap-frog
barotropic solve internally leap-frog-consistent, or tighten `rn_bt_cmax`/`ln_bt_auto` for the
rDt=2dt window). No 180d number fabricated. Node 16 discretization = ✅ faithful (LIVE-EEN);
node 19 leap-frog dt=2700 = ⚠️ blocked on residual #1.

---

## Round-4 (2026-07-18) — residual #1 RESOLVED; dt=2700 MLF 180-day UNBLOCKED + compared

**Residual #1 (Nbb before-level barotropic seed) + residual #1b (nn_bt_flt=2 temporal
dissipation) close the dt=2700 leap-frog blocker — the full 180-day run on the bridged
NEMO mesh is now STABLE with NO non-NEMO stabiliser.** The Round-3 "barotropic CFL margin"
diagnosis was PARTLY wrong: NEMO's `ln_bt_auto` computes a 117.4 s barotropic substep
(nn_e≈23) for DINO, so legoESM's 90 s substep is FINER than NEMO's — CFL was never the
issue. The real driver is the C-grid 2Δx equatorial barotropic gravity-wave null mode that
forward-Euler's numerical damping suppresses and the neutral leap-frog does not.

Two faithful pieces (both traced to NEMO `dynspg_ts.F90`, no ad-hoc damping):

1. **Residual #1 — Nbb before-level seed.** `barotropic_substeps_latlon_cgrid` gained
   `eta_init/u_init/v_init`; `_leapfrog_step` seeds the split-explicit barotropic integration
   from `(eta_before, u_before, v_before)` (Nbb) via `_barotropic_before_state`, matching NEMO
   `ln_bt_fw=.FALSE.` centred (`dynspg_ts.F90:494-503`: `sshn_e=pssh(Kbb)`, `un_e=puu_b(Kbb)`,
   `vn_e=pvv_b(Kbb)`). Frozen slow forcing stays at NOW (zu_frc); the 3-D depth-mean
   replacement still uses the NOW velocity. ALONE this moved the blow-up step 26 → ~65.
2. **Residual #1b — nn_bt_flt=2 `ts_bck_interp` temporal dissipation.** New
   `barotropic_time_filter="nemo_boxcar_ab3"` = the AB3 velocity predictor (1.781/-1.062/0.281)
   + the α=0 ssh half-step-back interpolation (0.614/0.285/0.088/0.013, `dynspg_ts.F90:
   1698-1701`) + boxcar averaging, with the ll_init ramp every step (nn_bt_flt=2 re-inits each
   baroclinic step, no cross-window carry). This is NEMO's built-in AM4 dissipation that damps
   the 2Δx mode. Gridscale eta fraction pinned ~0.05 (was climbing to >0.2 then exploding).

**THE dt=2700 180-day comparison (controlled, sole variable FE→leap-frog; bridged NEMO mesh,
from-rest analytic IC, seasonal forcing, NEMO RUN_TRAJ, 5760 steps):**

| Metric | FE baseline | **MLF leap-frog (faithful)** | NEMO |
|--------|------------:|-----------------------------:|-----:|
| SST corr / bias / rms | 0.995 / +0.22 / 0.72 | 0.994 / −0.43 / 0.99 | — |
| T@300m corr / rms | 0.987 / 0.61 | 0.990 / 0.53 | — |
| SSH corr / rms(m) | 0.991 / 0.060 | 0.992 / 0.069 | — |
| **BSF range ratio** | **2.62×** | **2.65×** | 1.0 (±40 Sv) |

**Answer to the core question: the fully-faithful leap-frog does NOT move the jet/BSF toward
NEMO** (2.65× vs 2.62×, both ~2.6×; thermodynamics corr ≥0.99 both). Node 19 (leap-frog +
Asselin, the LAST structural time-integration difference) is thereby ruled OUT as the driver
of the residual BSF over-strength — confirming the characterised equatorial (f→0)
core-dynamics amplification. The leap-frog program is COMPLETE: dt=2700 stable + compared.
Node 19 = ✅ faithful (stable + compared); residual #1 = ✅ CLOSED.
Tests: `test_nemo_ab3am4_filter.py::test_flt2_*` (coefficient transcription),
`test_leapfrog_integrator.py::test_boxcar_ab3_live_split_runs_no_nan` +
`::test_ab3am4_live_split_still_rejected`.

---

## Round-5 (2026-07-18) — residual #2 CLOSED: thickness-weighted tracer Robert-Asselin

**Node 19 residual #2 (tracer RA filter form) is now FAITHFUL.** The leap-frog tracer
Asselin filter was the CONCENTRATION form `T_f = T_n + γ(T_b−2T_n+T_a)` (NEMO's
`tra_atf_fix_lf`, valid only for the linear/fixed free surface). Under z* (`key_qco`)
NEMO uses the THICKNESS-WEIGHTED CONTENT form `tra_atf_qco_lf`
(`src/OCE/TRA/traatf_qco.F90:295-341`):

    ztc_f = e3t_n·T_n + γ·(e3t_b·T_b − 2·e3t_n·T_n + e3t_a·T_a);   T_f = ztc_f / e3t_f

with `e3t` at the before/now/after ssh and `e3t_f = e3t_0·(1+r3t_f)` from the
Asselin-filtered ssh. Ported as `_thickness_weighted_asselin`
(`ocean_model_latlon_cgrid.py`), wired into `_leapfrog_step` for T,S only. Momentum
(u,v) stays the PLAIN velocity filter — DINO runs `ln_dynadv_vec=.TRUE.`, for which
NEMO `dynatf_qco.F90:151-155` filters raw velocity (NOT thickness-weighted); ssh stays
plain (`ssh_atf`). e3t built from the filtered eta (== NEMO `r3t_f`) via
`compute_layer_thickness`, exact by z*-linearity. Additive; forward_euler/ab2/GYRE/Veros
byte-identical (`_leapfrog_step` only). OMITTED (documented, not silently dropped):
NEMO's surface-flux `zfact1·(sbc_tc−sbc_tc_b)` correction (:309) — no analog because
legoESM restores surface T/S IMPLICITLY (node 20); O(γ·2dt·Δflux), second-order.

- **Conservation gate (truth-tier).** `test_thickness_weighted_asselin_*`
  (`tests/ocean/unit/test_leapfrog_integrator.py`): (a) equal three-level content →
  global content preserved to 1e-13 while the concentration form drifts >1e-6; (b)
  general case → the filter perturbs global content by EXACTLY the discrete content
  time-Laplacian `γ(C_b−2C_n+C_a)` (1e-12); (c) fixed-volume → reduces to plain RA
  (backward-compat); (d) dry/partial cells → 0, no NaN. Full 23-test leap-frog suite +
  no-scheme-duplication + dispatch-hardening + constants/inline-coeff ratchets green.
- **Reviews:** code-reviewer APPROVE (all SHIP); physics-validator no correctness/sign/
  conservation defect (two DISCUSS items — surface-flux omission + conservation wording —
  both addressed in docstrings/test).

**Controlled 180-day comparison (sole variable = filter form; dt=2700, 5760 steps,
from-rest analytic IC on bridged NEMO mesh, seasonal forcing, NEMO RUN_TRAJ ÷vovvle3t;
both leap-frog runs in-session on separate GPUs):**

| Metric | conc-form leap-frog | **TW-form leap-frog (faithful)** | NEMO |
|--------|--------------------:|---------------------------------:|-----:|
| SST corr / bias / rms | 0.994 / −0.43 / 0.99 | 0.994 / −0.43 / 0.99 | — |
| T@300m corr / rms | 0.990 / 0.53 | 0.990 / 0.53 | — |
| SSH corr / rms(m) | 0.992 / 0.069 | 0.992 / 0.069 | — |
| **BSF range ratio** | **2.65×** | **2.65×** | 1.0 (±40 Sv) |

Both 180d runs STABLE (no NaN, T∈[3.7,25.4]°C). The in-session conc-form run reproduces
the Round-4 committed MLF baseline exactly (protocol byte-identical). **The
thickness-weighted filter does NOT recover the SST fidelity toward the FE 0.72 nor move
the BSF** — the 180-day climate mean is identical to the concentration form to within
noise (BSF range differs ~1 Sv of 215; SST at the 4th decimal). Expected: the RA filter
form difference is O(γ·dη/e3t) ≈ 0.1·(0.6 m/50 m) ≈ 0.1%/step, negligible on the mean in
this modest-eta config. It matters for exact CONTENT CONSERVATION (machine-precision,
proven), not the DINO climate. The SST 0.72(FE)→0.99(MLF) regression is the leap-frog
integrator itself (node 19 core), not the tracer RA filter form — residual #2 is faithful
and CLOSED; it is not the SST driver.

## Round-6: bathymetry ruled out; vertical coordinate is the next structural gap (2026-07-18)

Controlled 180d (faithful leapfrog nemo_dino_kamm_mlf, sole variable = depth field H(x,y)):
| metric | analytic bowl | NEMO exact depth | NEMO |
|---|---|---|---|
| BSF range ratio | 2.65× | **2.61×** | 1.0 |
| deep-eq KE@1000m/surf | 1.267 | 1.208 | 0.0011 |
| SST/T300m/SSH corr | 0.994/0.990/0.992 | 0.994/0.990/0.993 | — |

Matching NEMO's exact H(x,y) moves BSF only ~1.5% → **the 2.6× BSF + deep-eq jet are DYNAMICS,
not bathymetry.** Confound ruled out. The deep-eq jet has now survived EVERY matched piece
(integrator/EEN-Coriolis/live-EEN-barotropic-Coriolis/tracer-filter/bathymetry).

STRUCTURAL DISCOVERY: legoESM DINO is effectively PURE Z-STAR (all 36 levels stretched to
column depth, active_3d = 2D surface mask broadcast, NO dry bottom cells) vs NEMO ln_zco
FULL-STEP (fixed levels, staircase dry bottom cells). Genuine un-matched vertical coordinate.
Exact-MASK run NaN'd at the periodic seam (audit #10: roll-based ops don't respect a lone
seam-face wall → needs partial-periodic C-grid infra).

REMAINING un-matched (per mandate, build all): full-step-z vertical coordinate (biggest,
most likely deep-flow lever), partial-periodic seam, node 22 GM-form, node 14 visc-placement,
IC bit-identical, e1/e2 metric, bottom-drag implicit.

## Round-7 (2026-07-18) — FULL-STEP-Z BUILT + WIRED; it is the DOMINANT BSF lever

**Node "vertical coordinate" (Round-6's biggest gap) is now BUILT faithfully and is the
single largest mover of the whole campaign: full-step-z collapses the BSF over-strength
2.62× → 0.81× (into the ±40 Sv band), thermodynamics stay corr ≥0.99.**

FEASIBILITY: the full-step-z infrastructure already EXISTED (`OceanPartialCellCoordinate`
with `is_active`/`bottom_level`/`h_partial`; `compute_ocean_jacobian` partial branch
`J=water_col/H_bathy`; every operator dispatches on `isinstance(OceanPartialCellCoordinate)`
→ 3-D `active_3d`, 3-D mass-flux face masks, vmix wet mask, tracer wall-fill; the analytic
`masked_zco` DINO path already builds it). The ONLY gap: the BRIDGE path
(`bridge_nemo_to_legoesm_topo`, used by the 180d comparison) built a PLAIN z* coord, so the
comparison ran pure z-star regardless of the recipe's `masked_zco`. → a BOUNDED wiring fix,
not a rebuild.

Transcription (faithful to NEMO `usrdef_zgr.F90`): `zgr_zco_3d` sets `e3t(:,:,jk)=pe3t_1d(jk)`
(fixed reference thicknesses everywhere = full-step); `zgr_msk_top_bot` sets per-column `k_bot`
(staircase dry bottom cells). Built `vertical.py::create_full_step_coordinate(z_coord,
bottom_level)` — full cells (`h_partial=dz_ref`) + dry staircase below `bottom_level`, from an
explicit deepest-wet index; wired into the bridge via `full_step=True` using NEMO's OWN tmask
`k_bot=tmask.sum(axis=2)` (bit-faithful staircase, no float rounding at interfaces; default
`False` = byte-identical legacy z*). Also carried `t_depth_ref` (NEMO gdept_1d) through the
partial wrap (was dropped → HPG reverted to midpoints). Test `test_full_step_coordinate.py`
(staircase mask == NEMO k_bot, full cells, column-sum=gdepw(k_bot), Jacobian closure).

Correctness of the staircase (proven on the bridged DINO mesh): `is_active` column-count ==
NEMO tmask `k_bot` EXACTLY; 100% of wet columns carry ≥1 dry bottom cell (wet-level count
31–35 of 36); `active_3d` now varies with depth per column (NOT the 2-D broadcast).

CONTROLLED 180d comparison — sole variable = vertical coordinate (both forward-euler
`nemo_dino_kamm`, dt=2700, from-rest analytic IC on the bridged NEMO mesh, seasonal forcing,
NEMO RUN_TRAJ ÷vovvle3t; baseline RE-RUN in-session at identical protocol):

| metric | z* (baseline) | **full-step-z** | NEMO |
|---|---:|---:|---:|
| SST corr / bias / rms | 0.995 / +0.22 / 0.72 | 0.995 / +0.29 / 0.75 | — |
| T@300m corr / rms | 0.987 / 0.61 | 0.989 / 0.56 | — |
| SSH corr / rms(m) | 0.991 / 0.060 | 0.993 / 0.051 | — |
| **BSF range ratio** | **2.62×** | **0.81×** (lego ±33 vs NEMO ±40 Sv) | 1.0 |
| deep-eq KE@1057m/surf | 0.597 | 0.685 | 0.0011 |

**ANSWER: YES — full-step-z is the DOMINANT structural lever for the BSF.** Every prior node
moved BSF ≤1.5%; the vertical coordinate moves it 2.62× → 0.81× — from +160% over-strong to
within the ±40 Sv target band — while SST/T300/SSH stay corr ≥0.99 (T300/SSH slightly
IMPROVED). Physical mechanism: the staircase blocks deep flow over topography and restores
bottom form stress / f/H steering, which pure z-star (all levels stretched, no dry cells)
lacked — exactly the Drake-sill form-stress control of the DINO gyre. The deep-equatorial jet
(KE ratio 0.68 vs NEMO 0.0011) does NOT improve → it is a SEPARATE equatorial (f→0)
core-dynamics residual, as concluded earlier, not the coordinate.

STABILITY (honest): forward-euler `nemo_dino_kamm` + full-step ran 180d STABLE (no NaN). The
LEAPFROG `nemo_dino_kamm_mlf` + full-step BLOWS UP ~step 25–30 at the HIGH-LATITUDE walls
(|lat|>60°, largest staircase steps), NOT the equator/seam. Controlled isolation: {z* + mlf}
STABLE, {full-step + forward-euler} STABLE, {full-step + mlf} unstable ⇒ the coord foundation
is SOUND (forward-euler proves it); the blow-up is the NEUTRAL leapfrog amplifying a
staircase-seeded high-latitude mode (same class as the Round-3/4 barotropic-mode issue that
forward-Euler's numerical damping suppresses). NO non-NEMO stabiliser applied.

REMAINING (precise): (1) faithful leapfrog+full-step stability at the high-lat staircase steps
(separate residual, a Round-4-style fix — likely the barotropic/thickness leap-frog coupling at
the staircase, NOT a coord bug); then re-run the mlf full-step 180d. (2) push BSF 0.81× → ~1.0
(residual gyre tuning, now within band). (3) deep-eq jet (unchanged; f→0 core residual). (4)
the still-deferred node 22 GM-form, node 14 visc-placement, IC bit-identity.
Reviews: physics-validator + code-reviewer both SHIP (no confirmed defects; minor bottom_level
clamp added for sibling parity).

## Round-8 (2026-07-18) — full-step+leapfrog blow-up RE-DIAGNOSED (Round-7 hypothesis DISPROVEN)

Round-7 attributed the {full-step + MLF} blow-up to "the neutral leapfrog amplifying a
staircase-seeded high-lat mode … likely the barotropic/thickness leap-frog coupling"
(residual #1 class). **Controlled single-variable tests DISPROVE that.** The driver is NOT the
leapfrog and NOT the barotropic coupling — it is the **explicit (AB2/leapfrog) planetary
Coriolis time-stepping interacting with the full-step staircase, which lacks the time-domain
damping that keeps the forward-Euler recipe stable.**

**Blow-up pinned** (bridged NEMO DINO mesh, dt=2700, from restart): grows from step ~4 at the
**high-latitude staircase-step edges** (|lat|≈68°, `STEPHILAT`), seeded at the deepest wet
levels (k32/k33) of the columns, migrating to the surface and exploding ~step 21–26. NOT the
equator, NOT the periodic seam. The eta 2Δx grid-scale fraction stays flat ~0.17 until the
momentum explodes ⇒ it is a MOMENTUM instability, not the barotropic 2Δx eta checkerboard
(residual #1/#1b, already closed).

**Controlled matrix (sole variable in each row):**

| coord | Coriolis time-stepping | vorticity | result |
|---|---|---|---|
| full-step | matsuno_split (FE, `nemo_dino_kamm`) | al81 | **STABLE** (max\|v\|~0.2) |
| z* | explicit_ab2 | een_total | **STABLE** (max\|v\|~0.2) |
| full-step | explicit_ab2 | **een_total** | BLOWS @ step 26 |
| full-step | explicit_ab2 | **al81 (f via 4-pt face path)** | BLOWS @ step 26 (identical curve) |
| full-step | leapfrog (`nemo_dino_kamm_mlf`) | een_total | BLOWS @ step 22 |

**Ruled OUT by controlled toggles (all identical blow-up):** the leapfrog integrator (AB2 blows
too), the barotropic solver (finer substeps 30→60 identical), bottom drag (r=0 identical),
the vorticity Neumann-fill of q, planetary-f masking at partial corners, barotropic↔3D depth
inconsistency (H_bathy == 3D staircase wet-column sum to 0.00 m).

**Mechanism (positively identified):**
1. The 3D EEN Coriolis operator (`pv_flux_al81_partial_cell` with `f_vtx`) is **energy-
   conserving to machine precision even on a staircase** (Coriolis work/KE ≈ 1e-7 for both a
   flat and a staircase test grid). So the vorticity operator is NOT the energy source.
2. The energy source is the accepted full-step **staircase HPG error** (NEMO `ln_hpg_zco` has
   NO staircase/partial correction — the well-known z-coordinate error NEMO tolerates,
   dynhpg.F90 hpg_zco masked only by `rhd=0`/`umask`). It drives a weakly-unstable high-lat
   inertia-gravity mode (large f, small e1u=R·cosφ·Δλ at |lat|>60°).
3. `matsuno_split` (forward-backward) is a **damping** Coriolis time integration → it
   suppresses the mode (why the FE recipe is stable). The **neutral leapfrog / weakly-growing
   AB2** explicit Coriolis does NOT → the mode grows. z* has NO staircase ⇒ no HPG error ⇒ no
   source (why z*+explicit is stable). The leapfrog blows EARLIER than AB2 (step 22 vs 26)
   because rDt=2dt takes larger steps of the same growing physical mode — consistent with a
   physical (not computational-mode) instability, so the Asselin filter does not arrest it.

**Faithful lever CONFIRMED but insufficient alone: node 14 (lateral viscosity high-lat
scaling).** `nemo_dino_kamm` runs `A_h_floor=0, A_h_eq_boost=1` (NEMO-faithful, no legoESM
stabiliser) so A_h·cosφ drops to ~34% at 70°. NEMO's `ahmf = ½·rn_Uv·max(e1,e2)` does NOT
shrink at high lat (e2=R·Δφ dominates) ⇒ **NEMO's high-lat viscosity is ~3× legoESM's** — a
genuine node-14 mismatch, not a stabiliser. Removing the cosφ reduction (≈node 14) **delays the
blow-up step 26→34 and halves the growth rate**, but does NOT close it. So node 14 is a real
contributing faithful lever; the residual needs the full NEMO damping stack (node 14
max(e1,e2) viscosity placement + the exact leapfrog+Asselin behaviour of this mode), i.e. it is
a genuine multi-factor dycore-stability match, larger than one iteration.

**Status:** {full-step + FE} `nemo_dino_kamm` = the closest stable faithful config (BSF 0.81×,
Round-7). {full-step + MLF} `nemo_dino_kamm_mlf` = STILL BLOCKED, but the blocker is now
correctly identified as the explicit-Coriolis/staircase-HPG damping deficit (NOT the barotropic
leapfrog coupling). NO 180d MLF+full-step number produced (not fabricated). NO non-NEMO
stabiliser added; no production code changed this iteration (diagnosis + this record only).
Next faithful step: implement node 14 (`ahmf/ahmt = ½·rn_Uv·max(e1,e2)` embedded in the
div/curl, replacing `A_h·cosφ`) and re-test; if still marginal, audit the leapfrog handling of
the high-lat inertia-gravity mode vs NEMO stpmlf at the staircase steps.

---

## Round-9 (2026-07-18) — node 14 BUILT faithfully; BOTH Round-8 "levers" are NON-levers (controlled)

Round-8 named two faithful levers to close {full-step + MLF} stability: (A) node 14 —
NEMO's high-lat viscosity "~3× legoESM's"; (B) the adcroft PGF "over-producing" the
staircase HPG error. This iteration BUILT node 14 faithfully and AUDITED the PGF, and
**controlled tests disprove BOTH premises** — neither is the lever, and {full-step+MLF}
remains blocked by the Round-8 mechanism (neutral-leapfrog/explicit-Coriolis damping deficit).

**Node 14 — BUILT faithfully (embedded div-curl, `nemo_div_curl`).** New operator
`nemo_ldf_lap_viscosity_cgrid` + coeff `nemo_lateral_viscosity_coefficients`
(`latlon_cgrid_operators.py`) = faithful NEMO `dynldf_lev.F90::dynldf_lev_lap` +
`dynldf_lev_rot_scheme.h90`: `grad_h(ahmt·div_h U) − curl_h(ahmf·curl_z U)` with
`ahmt(T)=½·rn_Uv·MAX(e1t,e2t)`, `ahmf(F)=½·rn_Uv·MAX(e1f,e2f)` (`ldfc1d_c2d.F90::ldf_c2d`
L138-139, nn_ahm_ijk_t=20) EMBEDDED inside the div/curl. New
`lateral_viscosity_operator="nemo_div_curl"` (default `"vector_laplacian"` byte-identical);
`nemo_dino_kamm`/`_mlf` opt in. Reduces EXACTLY to `A_h·vector_laplacian_cgrid` for constant
coeff (truth-tier test `tests/ocean/unit/test_nemo_ldf_lap_viscosity.py`).

**The Round-8 "NEMO high-lat viscosity ~3×" premise is FALSE — a Mercator-grid confusion.**
Read the NEMO DINO `mesh_mask.nc`: the grid is TRUE MERCATOR, `e1≈e2` (isotropic) at EVERY
latitude (e2f/e1f = 1.0000 at 0/40/68°). So `MAX(e1,e2)≈e1=R·Δλ·cosφ` DOES shrink with
cosφ (111→41.5 km at 68°), and NEMO's `ahmt=½·rn_Uv·MAX(e1,e2)` matches legoESM's `A_h·cosφ`
to **O(Δλ²)** (machine-level at the equator = 15011.85 m²/s; ~2e-5 worst-case at ±69°, where
the DISCRETE `e2t=R·Δφ_face` vs `e1t=R·Δλ·cosφ_centre` disagree at O(Δλ²) and MAX picks e2 on
~46% of rows). The "e2=R·Δφ dominates ⇒ 3×" claim assumed UNIFORM Δφ; Mercator has
Δφ=cosφ·Δλ, so e2 shrinks too. Node 14 changes ONLY the coefficient PLACEMENT (embedded vs
outside) plus this O(Δλ²) metric wrinkle, not the magnitude. The Round-8 empirical
"delayed blow-up 26→34" test had used CONSTANT A_h (cosφ removed) ≈ 2.7× stronger than NEMO at
68° — a **non-faithful over-viscous stabiliser, NOT node 14**; the mandate forbids it.

**Piece B — adcroft PGF ≡ hpg_zco under full-step (audited, no change).** DINO namelist runs
`ln_hpg_sco=.true.` (s-Jacobian), which for full-step z (flat interfaces) reduces to `hpg_zco`
— the plain z-level HPG with NO partial-cell/staircase correction (`dynhpg.F90::hpg_zco`).
Structural audit of legoESM's adcroft `partial_cell_pgf_correction` on the bridged full-step
DINO state (`scratchpad/pgf_staircase_audit.py`): the correction is LARGE (~0.17) ONLY on the
dry wet/dry step faces — which are walls (`u_mask_3d=0`, applied by the final
`du_dt = du_dt * u_mask_3d`), exactly as NEMO masks them (umask=0). On WET faces the correction
is ~6e-8 (float rounding of the cumsum centroid; both wet cells sit at the same flat z-level).
⇒ adcroft adds NO dynamic staircase HPG force NEMO lacks. **Correct for full-step; unchanged.**

**Stability (controlled, bridged NEMO mesh, dt=2700, from restart):**
| config | result |
|---|---|
| {full-step + MLF + node14 + adcroft} `nemo_dino_kamm_mlf` | **BLOWS @ step ~22** at \|lat\|≈68° staircase edges (\|u\|,\|v\|→430 → NaN step 23) — SAME as Round-8; node 14 did NOT delay it |
| {full-step + FE + node14} `nemo_dino_kamm` | **STABLE** 180d (\|u\|~0.09 at step 22; T∈[1.6,26.2]°C at 180d) |

**Controlled 180d A/B — sole variable = viscosity PLACEMENT (vector_laplacian → nemo_div_curl),
{full-step + FE}, identical protocol (NEMO RUN_TRAJ, dt=2700, 5760 steps, ÷vovvle3t):**
| metric | vector_laplacian (Round-7) | **nemo_div_curl (node 14)** | NEMO |
|--------|---------------------------:|----------------------------:|-----:|
| SST corr / bias / rms | 0.995 / +0.29 / 0.75 | 0.995 / +0.29 / 0.75 | — |
| T@300m corr / rms | 0.989 / 0.56 | 0.989 / 0.56 | — |
| SSH corr / rms(m) | 0.993 / 0.051 | 0.993 / 0.051 | — |
| **BSF range ratio** | **0.81×** | **0.81×** | 1.0 (±40 Sv) |

Node 14 is FAITHFUL and, on the Mercator DINO grid, **climate-inert** (identical to reported
precision — as the O(Δλ²) coefficient identity predicts). Node 14 = ✅ faithful (built +
tested + 180d-compared); it is NOT the {full-step+MLF} stabiliser (Round-8 "3×" was a
Mercator misread). PGF node 15 = ✅ confirmed faithful under full-step (adcroft≡hpg_zco).

**Blocker unchanged + correctly attributed.** {full-step+MLF} still blows at the |lat|≈68°
staircase (a MOMENTUM instability), driven by the Round-8 mechanism: the neutral leapfrog /
weakly-growing explicit-AB2 Coriolis lacks the time-domain damping that `matsuno_split`
(forward-backward, FE recipe) supplies to the staircase-`hpg_zco`-seeded high-lat
inertia-gravity mode. NEITHER faithful lever (node 14, PGF) addresses it — both are proven
non-levers here. The residual is a genuine leapfrog+Asselin vs staircase-HPG dycore-stability
match at the high-lat steps, larger than one iteration and NOT closable by any NEMO-faithful
viscosity/PGF knob. **The closest STABLE fully-faithful config remains {full-step + FE}
`nemo_dino_kamm` at BSF 0.81×.** NO non-NEMO stabiliser added; NO 180d MLF+full-step number
fabricated.

Node 14 = ✅ (built/tested/compared; faithful; climate-inert on Mercator). Node 15 = ✅
(adcroft≡hpg_zco under full-step, audited).

---

## Round-10 (2026-07-18) — audit-#10 SEAM: spurious equatorial LAND WALL removed (i-periodic/ln_Iperio)

**Root cause of the deep-equatorial jet FOUND + FIXED.** NEMO DINO is zonally
re-entrant (`ln_Iperio=.true.`): read from the RUN_TRAJ `mesh_mask.nc`, the surface
tmask is **48/48 wet at EVERY latitude** and `umask` at the west face of column 0 is
wet on all 195 rows — NO land walls; the E/W "walls" are **shoaling bathymetry** (k_bot
35 interior → 32–34 at the margins), not land. legoESM's `dino_lat_lon_state` /
`dino_lat_lon_initial_state_arrays` (experiments/dino.py) instead imposed the analytic
`partial_periodic_seam_wall_latlon` mask — column j=0 LAND outside the ACC channel — a
non-NEMO wall. That free-slip equatorial wall trapped a cold `T=0` masked cell (deep
western column ~0.6 °C vs interior ~4 °C) whose zonal PGF, unbalanced at f→0, projected
onto the spurious deep-equatorial jet. The bridge (`bridge_nemo_to_legoesm_topo`) already
carries the re-entrant `land_mask=tmask[:,:,0]` + periodic-wrap u/v masks + full-step
staircase; only the from-rest analytic IC setup overrode it.

**Fix (additive, backward-compatible).** New opt-in `land_mask_override=` on both
functions: the NEMO-bridged comparison passes `br.land_mask` (NEMO's own surface tmask)
so column j=0 stays WET (re-entrant), and `rest_state_latlon_cgrid_ocean` recomputes
`u_mask`/`v_mask` atomically from it (`compute_face_masks`: periodic-in-lon via `jnp.roll`,
N/S walled — exactly `ln_Iperio`). Default `None` = the analytic seam wall, byte-identical
for the standalone bowl recipes (test `test_land_mask_override_default_is_byte_identical`).
Gates: `test_land_mask_override_reentrant_keeps_seam_wet` (col 0 wet, T not zeroed) +
`test_land_mask_override_reentrant_periodic_u_mask` (u_mask[:,0]==u_mask[:,-1] periodic
wrap, all interior u-faces wet, N/S v-faces walled).

**Controlled 180d — sole variable = seam wall vs re-entrant** (both `nemo_dino_kamm`,
full_step=True, dt=2700, 5760 steps, from-rest analytic IC on the bridged NEMO mesh,
seasonal forcing, NEMO RUN_TRAJ ÷vovvle3t; separate GPUs):

| metric | seam-wall (baseline) | **re-entrant (fix)** | NEMO |
|--------|---------------------:|---------------------:|-----:|
| **deep-eq KE@1057m** | 9.45e-3 | **6.29e-4** (15× ↓) | 7.10e-5 |
| **deep-eq KE ratio (1057m/surf)** | 0.688 | **0.108** (6.3× ↓) | 0.0011 |
| western deep-eq cell T (min) | 0.6 °C (cold T=0 wall) | **3.9 °C** (physical) | ~4 °C |
| SST corr / bias / rms | 0.995 / +0.29 / 0.75 | 0.995 / +0.28 / 0.78 | — |
| T@300m corr / rms | 0.989 / 0.56 | **0.996 / 0.36** (↑) | — |
| SSH corr / rms(m) | 0.993 / 0.051 | **0.947 / 0.143** (↓, see residual) | — |
| BSF range ratio | 0.81× | **1.06×** (→1.0 target) | 1.0 |

**Verdict.** The spurious equatorial land wall was the deep-eq jet's dominant driver:
removing it cuts deep-eq KE 15× and the KE ratio 6.3× toward NEMO, and warms the western
equatorial cells from the cold-wall 0.6 °C to the physical ~4 °C = NEMO. T@300m and the
BSF range ratio (→1.06×, essentially the 1.0 target) also improve; SST unchanged. 180d
STABLE (T∈[3.9,26.2] °C, no NaN, no non-NEMO stabiliser).

**Residual (diagnosed, NOT papered over).** SSH corr regresses 0.993→0.947 (rms
0.051→0.143 m), broad (worst in the S/ACC band). It is NOT a mask-inconsistency seam mode:
the eta 2Δx grid-scale roughness DROPS 0.0073→0.0005 (clean seam, no checkerboard). It is
the **pre-existing f→0 equatorial core-dynamics residual** now EXPOSED: NEMO has a strong
zonal-mean westward equatorial surface jet (−0.74 m/s) that neither lego reproduces; the
seam wall had coincidentally trapped a westward flow (−0.18) that correlated with NEMO's
SSH, and the (correct) re-entrant equatorial dynamics give +0.10 instead. The spurious
wall was compensating a real dynamics gap in the SSH metric — the faithful topology
reveals it. This is the same equatorial-amplification residual characterized across
Rounds 1–9 (survived every controlled node test), not a defect in the re-entrant fix.

The deep-eq jet root cause (audit #10 seam) is CLOSED; the remaining SSH residual is the
equatorial f→0 dynamics gap, unchanged by any single node.

## Round-11 (2026-07-18) — PARTIAL-PERIODIC SEAM-FACE WALL: the faithful DINO geometry

**Both prior seam variants were wrong; this builds the NEMO-faithful third one.**
Round-10 exposed a false dichotomy: the analytic **land-column** wall (dry column 0
outside the channel) traps a cold `T=0` masked cell → spurious cold front + deep-eq jet,
while the fully **re-entrant** override (all-wet, no wall) makes a channel-world with the
WRONG-sign equatorial surface current (+0.10 eastward) and degraded SSH (0.947). NEMO's
mesh is neither: the raw `mesh_mask.nc` halo `tmask` west-outer-halo column (col 0) is
LAND at every latitude OUTSIDE the ACC channel `[-64.44, -45.35]` while ALL interior
cells stay wet — i.e. the zonal periodic-seam **u-face** is walled outside the channel,
open inside. (Equatorial SSH tilts west-high +0.067 m in NEMO, proving the basin is
zonally closed there.)

**Implementation (additive, default-`None` byte-identical).** One optional
`seam_wall_rows` profile (`(n_lat,)`, 1 = walled) threaded on the `LatLonCGridGeometry`
and read via `getattr(grid, "seam_wall_rows", None)` at the four mask-derivation sites —
`compute_face_masks` (2-D), `compute_face_masks_3d` (per-level), `compute_vertex_mask`
(EEN/PV seam corners), and the barotropic diffusion mask — each closing the seam u-face
(cols 0 AND n_lon, the same wrap face) on walled rows; the runtime invariant threads the
grid so a walled-but-wet state passes; the MPI/SPMD band slicers slice/widen it. The NEMO
bridge (`read_nemo_mesh_mask` + `bridge_nemo_to_legoesm_topo`) derives it from the halo
tmask and attaches it to the bridged geometry. GYRE/Veros/analytic recipes untouched.

**Controlled 180d — sole variable = seam geometry** (all `nemo_dino_kamm`, full_step=True,
dt=2700, 5760 steps, same bridged NEMO mesh + IC + forcing, NEMO RUN_TRAJ ÷vovvle3t):

| metric | land-column | re-entrant | **SEAM-WALL (faithful)** | NEMO |
|--------|-----------:|-----------:|-------------------------:|-----:|
| **eq surf zonal u [m/s]** | −0.025 | **+0.105 (wrong sign)** | **−0.011 (westward ✓)** | −0.182 |
| **SSH corr** | 0.993 | **0.947 (channel-world)** | **0.994 (recovered)** | — |
| **eq SSH W-E tilt [m]** | −0.049 | −0.000 | −0.015 | +0.067 |
| deep-eq KE ratio (1057m/surf) | 0.688 (cold-cell jet) | 0.108 | 0.450 | 0.001 |
| eta 2Δx checkerboard | 1.83 | 0.03 | **0.26 (clean)** | ~0 |
| SST corr | 0.995 | 0.995 | 0.995 | — |
| T@300m corr (mean °C) | 0.989 (9.26) | 0.996 (9.36) | 0.993 (9.29 vs 9.28) | — |

**Verdict.** The faithful seam wall fixes BOTH documented failures at once, WITHOUT the
spurious land cell: vs re-entrant it flips the equatorial surface current to the correct
**westward** sign (+0.105→−0.011, NEMO −0.182) and recovers **SSH corr 0.947→0.994**; vs
the land-column it removes the cold-cell front (checkerboard 1.83→0.26, no `T=0` cell) and
halves the deep-eq jet (KE ratio 0.688→0.450). 180d STABLE (T∈[3.9,26.5] °C, no NaN, no
non-NEMO stabiliser). Reviews: physics-validator GREEN (conservation `∑area·div(F)`=3.7e-9,
geometry-sign + EEN corner + cross-mask consistency all confirmed); code-reviewer clean
after the MPI band-slice fix.

**Residuals (diagnosed, NOT papered over).** (a) The eq SSH tilt is present but weak and
still east-high (−0.015 vs NEMO +0.067); (b) deep-eq KE ratio 0.450 is reduced from the
land-column but well above NEMO's 0.001; (c) BSF is ~6× over-strong (crude diagnostic).
All three are the SAME pre-existing equatorial-amplification / barotropic-Coriolis
energetic node (open Node 16), now cleanly EXPOSED on faithful geometry rather than masked
by a wrong wall — not a defect in the seam-wall fix. Gate: `test_partial_periodic_seam_wall.py`
(8 tests: byte-identical default, seam-face/vertex/3-D/barotropic masking, flux-form
conservation, band-slice alignment).

**Orchestrator correction (Fable):** the seam-wall worker's "BSF ~6× over-strong" was a
mis-metric. Standard controlled protocol on seam180: **BSF [-40.1,+34.1] = 0.95×, pattern
symmetric like NEMO**; SST 0.995, T300 0.993/0.44, SSH 0.994/0.045 — best config on every
metric. Remaining equatorial residuals (one coherent symptom cluster): surface eq jet
westward but 16× weak (−0.011 vs −0.182), eq SSH tilt sign still wrong (−0.015 vs +0.067),
deep-eq KE 0.45 vs 0.001 → hypothesis: wind-input westward momentum is mixed TOO DEEP at
the equator (surface-trapped in NEMO) — vertical momentum mixing/avm suspect. Next: ladder
step 1, same-state equatorial avm + tendency comparison vs NEMO restart.

---

## FINAL: the first WIND-ON 180d comparison (harness bug fixed)

**Critical harness bug (all prior 180d runs).** `nemo_dino_kamm` sets
`wind_through_step=True`, so `apply_dino_lat_lon_surface_forcing` deliberately
SKIPS the wind (dino.py:2809) and expects it via `model.step(surface_forcing=…)`.
Every earlier harness (`kamm_run180.py` and its variants above) called
`model.step(st, DT)` with **no** `surface_forcing` → **the ocean ran with ZERO
wind for 180 days.** All tables above the "SEAM-WALL" verdict are therefore
no-wind runs. Fix: `kamm_run180_v2.py` builds `sf = dino_step_surface_forcing(forcing)`
(cell-centred `tau_x=-tau_u` ocean-reaction sign + `taum` TKE modulus) and threads
it: `model.step(st, DT, surface_forcing=sf)`. No model-code change.

**2-day wind validation.** `tau_x∈[-0.20,+0.10] Pa`. Surface-u zonal-mean by band:
equator (|lat|≤5) **−0.025 m/s (westward ✓, easterly trades)**, mid-lat westerly bands
+0.019 / +0.025 m/s (eastward ✓). Correct signs, finite, stable.

**180d controlled comparison — WIND ON** (only new variable = wind now enters; all else
byte-identical: full_step=True, seam wall, dt=2700, 5760 steps, bridged NEMO mesh+IC,
seasonal forcing via t_seconds, NEMO RUN_TRAJ ÷vovvle3t). Equatorial metrics computed by
one script (`kamm_eqsig.py`) that derives the NEMO value with the SAME code as legoESM
(apples-to-apples; note this NEMO column differs from the older tables' eq-u/tilt because
the metric band/level definition differs — trust the same-script pair):

| metric | no-wind seam-wall | **WIND-ON (this run)** | NEMO (same script) |
|--------|------------------:|-----------------------:|-------------------:|
| SST corr | 0.995 | **0.998** (bias −0.21, rms 0.49) | — |
| T@300m corr | 0.993 | **0.986** (bias −0.03, rms 0.62) | — |
| SSH corr | 0.994 | **0.995** (rms 0.043 m) | — |
| eq surf-u zonal-mean [m/s] | −0.011 | **−0.338 (westward ✓)** | −0.379 |
| eq SSH W-E tilt [m] | −0.015 (wrong sign) | **+0.079 (west-high ✓)** | +0.102 |
| deep-eq KE(1000m)/KE(surf) | 0.450 | 0.264 | **0.0008** |
| BSF range ratio | ~0.95× | **4.65×** (lego [−203,+176] vs [−40,+41] Sv) | 1.0× |

180d STABLE (T∈[3.2,25.7] °C, finite throughout, no NaN, no non-NEMO stabiliser).
PNGs: `kamm_compare_wind.png` (SST/BSF/T-section 3×3), `kamm_bsf_sidebyside.png`.

**Verdict — with wind + the full matched term ledger, how close are the two models?**
The large-scale hydrography and surface fields now agree very well (SST corr 0.998,
T300 0.986, SSH 0.995) AND — the headline — the **equatorial surface signature flips
to the correct sign at ~80–90% amplitude**: surface-u −0.338 vs NEMO −0.379 (was −0.011,
16× too weak with no wind), SSH W-E tilt +0.079 vs +0.102 (was the WRONG sign −0.015).
The wind was the missing first-order driver; adding it faithfully closes the sign/pattern
failures that persisted through the entire term-matching campaign.

**The one remaining structural discrepancy (honest, not papered over): the equatorial jet
is over-energized in the DEPTH.** Deep-eq KE(1000m)/KE(surf) is 0.264 vs NEMO's 0.0008
(~330×): NEMO keeps the wind-driven equatorial current surface-trapped, legoESM mixes that
westward momentum too far down. The over-strong deep jet then inflates the barotropic gyre
(BSF 4.65× too strong). Both are the SAME symptom — insufficient equatorial surface-trapping
of wind momentum — i.e. the long-standing vertical-momentum-mixing / barotropic-Coriolis
node (Node 16), now cleanly EXPOSED under realistic forcing rather than masked by absent
wind. It is a vertical-mixing/energetics closure gap, not a term-ledger or geometry defect;
every conservation/geometry tier remains green. Next lever: equatorial `avm` vertical
momentum mixing (surface-trapping) — a same-state tendency comparison vs the NEMO restart.

---

## Round-12 (2026-07-18) — {full-step + MLF} blow-up: term-attributed budget + full composition table (no single-term defect; NOT papered over)

**Scope.** The LAST unmatched time-integration piece: `nemo_dino_kamm_mlf`
(leapfrog + full-step staircase) blows up at dt=2700 while NEMO runs the same
{MLF + ln_zco full-step + Courant 0.8} stably (RUN_TRAJ 180 d). Phase-A intersect
= term-attributed growth budget × complete stpMLF-vs-`_leapfrog_step` composition
table. Harness reproduces the blow-up from the bridged NEMO mesh (`full_step=True`,
from rest = zeroed velocity + NEMO restart T/S), dt=2700, wind on AND off.

**Reproduction (both wind states).** NaN at step 22 (explodes step 21). Growth is
at the **deepest wet cells (k32/k33) of the |lat|≈68° staircase columns** — NOT the
equator, NOT the seam — exactly Round-8's `STEPHILAT`. Wind on vs off blow at the
same step with the same mode; the surface wind (`phys_u`) is 0 at the deep growing
cell, so **wind is irrelevant to this mode** (it is HPG-seeded, wind-independent).
The wind PATH does exist in `_leapfrog_step` (surface_forcing → both `_step_impl`
passes + `_apply_implicit_vertical_mixing`); wind-on responds at the surface.

**Term-attributed budget (`tendencies_with_diagnostics` at the growing cell).**
| step | max\|u\| | dominant RHS term at cell | value |
|---|---|---|---|
| 8 (seed) | 0.28 | **KE_PGF** | −8.6e-5 (all others ≤7e-6) |
| 8–15 | 0.28→0.81 | KE_PGF ~const ±1e-4 | vortcor/vertadv/Ahlap 10–100× smaller |
| 20 | 2.6 | KE_PGF −8e-5, vertadv −2.4e-4 | u already O(1) |
| 21 (explode) | 64 | vertadv −4.3e-2 ≳ KE_PGF −3.2e-2 | consequence of u→O(1) |

- **The seed is the HYDROSTATIC PGF, not the KE-gradient.** Controlled isolation
  (`momentum_advection="flux_form"`, KE-grad/Bernoulli removed): blows at the
  IDENTICAL step 21 with the IDENTICAL KE_PGF seed (−8.63e-5 vs −8.65e-5 at step 8).
  ⇒ KE_PGF is pure `hpg_zco` here; the vector-invariant Bernoulli term is NOT the
  source. Coriolis (`vortcor` ~1e-6) and lateral visc (`Ah_lap` ~4e-6) are not
  either. `vertadv` dominates ONLY the terminal explosive step (nonlinear, once
  u=O(1)) — a consequence, not the cause (and DINO runs `ln_zad_Aimp=.false.`, so
  NEMO's vertadv is EXPLICIT leapfrog too — not the difference).
- **Temporal signature = physical growth, not a pure computational mode.** The
  pinned deep cell (−68.1°, k32) stays NEGATIVE with a MONOTONICALLY-growing
  envelope (−0.28→−0.49→−0.73→−1.04→−6.4) carrying only a WEAK 2Δt ripple
  (−0.490/−0.452, −0.726/−0.687 pairs) — the Asselin filter (γ=0.1) IS damping the
  2Δt ripple but cannot arrest the growing envelope. Confirms Round-8: physical
  inertia-gravity mode, not an Asselin-dampable 2Δt mode.

**Complete stpMLF vs `_leapfrog_step` composition table (differences only).**
Traced NEMO 5.0.2 `src/OCE/` (stpmlf/dyn{adv,vor,hpg,ldf,zdf,spg_ts,atf_qco}/
tra{adv,ldf,zdf,atf_qco}) against `_leapfrog_step`. **Every stability-relevant term
MATCHES:** lateral viscosity + tracer lateral diffusion both lagged to **Nbb**
(forward-over-2dt, `_ab2_scope_override="advective"` Nbb diss pass); advection /
Coriolis(EEN) / HPG all at **Nnn**; barotropic split-explicit seeded from **Nbb**
(ln_bt_fw=F); vertical friction + wind stress + bottom drag all **implicit over
rDt=2dt**; Asselin **plain** on u,v and **thickness-weighted** on T,S; Euler-first
step (dt, no filter). The only nominal differences are **negligible at the deep
cell**: (a) NEMO's momentum LF step thickness-weights by (1+r3u) at before/now/after
(dynzdf.F90:127-139) vs legoESM plain velocity — but r3u≈0 at the deep z* cells;
(b) NEMO vertadv is 2nd-order-centred on the FULL velocity vs legoESM
upwind-perturbation — but vertadv is not the seed. Ruled out by prior controlled
toggles: lateral viscosity (node 14, Mercator-inert), PGF (adcroft≡hpg_zco), bottom
drag (r=0 identical), barotropic substeps (30→60).

**Intersect verdict (honest).** Growing term = **HPG (staircase `hpg_zco`)**;
its composition difference vs NEMO = **NONE** (adcroft≡hpg_zco, Round-9; both cores
compute the same staircase HPG). ⇒ **this is NOT a single-term composition defect**
to implement. The blow-up is the NEUTRAL leapfrog PRESERVING a physical HPG-seeded
high-lat staircase inertia-gravity mode that FE's forward-backward/forward-Euler
numerical damping suppresses (why {full-step + FE} `nemo_dino_kamm` is stable).
NEMO's leapfrog runs the same config stably with a matching per-term time-level
ledger — so the reconciling difference is sub-ledger: either (i) legoESM's HPG
MAGNITUDE at the deepest staircase cell is spuriously larger than NEMO's (a fixable
discretization difference the interior-cell audits missed), or (ii) NEMO damps this
specific mode via a mechanism not captured by the per-term table. **No non-NEMO
stabiliser was added; no MLF+full-step number was fabricated.**

**Precise remainder (the one test that distinguishes (i) from (ii)).** A NEMO
per-cell HPG **trend dump** (`ln_dyn_trd`/`trddyn`) at the deepest |lat|≈68°
staircase cell, compared same-state against legoESM's `KE_PGF` diagnostic there. If
NEMO's HPG at that cell is materially smaller → (i), legoESM's full-step bottom-cell
HPG is the fixable defect (candidate: the `t_depth_ref`/thickness the deepest wet
cell receives from `create_full_step_coordinate`). If equal → (ii), a genuine
leapfrog-dissipation match, larger than one pass. The closest STABLE fully-faithful
config remains {full-step + FE} `nemo_dino_kamm` (BSF 0.81×, Round-7).

---

## Stage-level lockstep trend dump at kt=5761 — HPG-v is the first divergence (2026-07-18)

The "precise remainder" test above (a NEMO per-cell HPG trend dump compared same-state
against legoESM `KE_PGF`) was executed. NEMO `MY_SRC/{trddyn,trdtra,trddump}` dump every
per-term trend + the RK3 stage-3 evaluation state (`uu_stg` …) into
`RUN_STEPDUMP/DINO_00005761_restart.nc` (one step from DINO_00005760). Harness:
`~/oracle-builds/nemo5/gap_audit/dino_term_compare.py` (env `DINO_RUN`/`DINO_RST`) +
per-level probe `scratchpad/_loc.py`. Full note:
`scratchpad/dino_stage_lockstep_instrument.md`.

**Controlled result (identical harness, two restarts).** Seam-excluded interior corr:

| term | u @5761 (eddying) | v @5761 | u @M5SPIN (smooth) | v @M5SPIN |
|------|------|------|------|------|
| hpg  | **1.0000** | **0.686** | 1.0000 | 0.9999 |
| keg  | 0.995 | 0.697 | 1.000 | 0.9995 |
| pvo+rvo | 1.0000 | 0.61 | 1.0000 | 0.888 |

hpg-v per level @5761: k5 0.96 → k20 0.67 → k30 0.63, amplitude ratio 0.92 → **0.64**
(deficient, growing with depth). **u-PGF stays 1.0000 / ratio ≈ 1.00 at every depth.**
Density matches NEMO `rhd` to 1.5e-5; grid dx/f exact.

**Verdict.** The FIRST phase to diverge is `dyn_hpg`, specifically its **meridional (v)
component** — upstream of `dyn_spg`/wind. Same density, same grid ⇒ the only
u/v-asymmetric ingredient is the meridional metric and the **terrain-following
(z-star) PGF** over DINO's meridionally-sloping bowl/ACC bathymetry. The default
`pgf_scheme="adcroft"` on `OceanZStarCoordinate` applies NO slope correction
(`ocean_pe_latlon_cgrid.py:1608-1623`): densities are differenced at equal level-index
but unequal physical depth across the slope. The error is benign on the smooth 62-day
state (v = 0.9999) and grows with depth AND developed meridional density structure
(v → 0.63 at 180 d) — confirming hypothesis (i) (legoESM deep-cell HPG magnitude
spuriously large), NOT hypothesis (ii) (leapfrog dissipation).

**Fix (verify-first).** NOT a flip to `pgf_scheme="smc03"`: tested — smc03 degrades both
u and v at depth (its η=0 physical-depth reconstruction ≠ NEMO `ln_hpg_sco` Jacobian).
Real fix = a NEMO-`ln_hpg_sco`-faithful s-coordinate Jacobian PGF option for z*, and/or
verifying the z* interface-depth (`key_qco`) reconstruction — both current PGF schemes
inherit the coordinate metric.

### CORRECTION (2026-07-18) — the v-hpg 0.686 was a MASK ARTIFACT; the verdict above is WRONG

The verdict above is **retracted**. A verify-first re-run (bridge the SAME kt=5761
state with the coordinate the 180-day comparison ACTUALLY uses —
`bridge_nemo_to_legoesm_topo(..., full_step=True)` `OceanPartialCellCoordinate` — AND,
for contrast, pure z-star) shows the two are **byte-identical**: v-hpg seam-excl corr
= **0.6861 in BOTH**. Full-step levels are FLAT (no slope), so the "terrain-following
z-star PGF slope error" mechanism cannot be the cause — refuted.

The 0.686 is a **single-boundary-row correlation-mask artifact**, not a PGF defect:

- The instrument mask `core_v[:, 4:-3, 1:]` (`dino_term_compare.py`) excludes the
  longitude seam and surface level but **NOT the northernmost v-face row** (native
  v-point `n_lat-1`, lat 69.3°, the closed north wall). There legoESM correctly zeroes
  the wall face (`v≡0`, `rms_lego=0.000`) while NEMO stores a large **un-masked**
  diagnostic `vtrd_hpg` (`rms_nemo=2.75e-5`, ~25× the interior amplitude on the
  developed state). Per-row leverage: excluding **only** that one row lifts the pooled
  corr **0.686 → 1.0000**. Excluding first+last rows → **1.0000**.
- **Interior v-hpg matches NEMO to corr 1.0000** on the developed 5761 state, both
  coordinates. Confirmed independently: a from-scratch numpy transcription of NEMO
  `dynhpg.F90::hpg_zco` (trapezoidal `e3w`-weighted accumulation of the horizontal
  density difference) matches BOTH NEMO `vtrd_hpg` and legoESM `pgf_ke_v` at corr
  **1.0000** (u and v).
- **Why u was 1.0000, v looked bad, and the smooth state was fine:** u is 1.0000
  because the i-periodic grid has no zonal wall row. v-hpg on the SMOOTH M5SPIN
  (kt=2000) state = **0.9999 unshifted** through the identical harness (and a ±1 row
  shift makes it WORSE) — the wall-row diagnostic is negligible there and grows with
  the developed high-lat density structure + depth (hydrostatic accumulation), which is
  exactly why the artifact appeared depth-degrading on the 5761 state. The earlier "+1
  row shift helps (0.686→0.94)" was a red herring from the 2Δy wall-adjacent content,
  not a stagger bug (the metric `dy_v` matches NEMO `e2v` to ≤0.2% at every row, and
  the shared pressure field is exact — proven by u=1.0000).

**Consequence.** There is **no v-PGF defect** and no model code change. `dyn_hpg` is
✅ VERIFIED-MATCH (corr 1.0000, u and v, interior, both coordinates) — consistent with
node 15's Round-9 audit. The wall v-row is a diagnostic-bookkeeping mismatch with zero
physics consequence (the wall face carries `v≡0`). The instrument mask was corrected to
drop the north wall row (`core_v[:-1, 4:-3, 1:]`). The residual BSF/deep-eq-jet story is
UNCHANGED: `dyn_hpg` is NOT a first divergence, and the "HPG-v first divergence" line in
the kt=5761 stage-dump section above should be read as **this correction supersedes it**.

## Corrected step-twin metrics (strict interior mask) — 2026-07-18

The headline "step-1 eta rms 1.65–2.14e-2 m" was measured on the **full wet mask**,
which for eta == the whole surface field and so folds in the two proven confounds:
(1) seam col i=0 / periodic-wrap (stale NEMO restart halos) and (2) the north/south
wall rows carrying un-masked NEMO diagnostics. Re-audit re-runs the same CPU twin
(`scratchpad/stepdump_twin_strict.py`, from the identical bridged IC at kt=5760) with a
**strict interior** mask = wet AND ≥2 cells from the i-seam AND ≥2 rows from either
j-wall AND not land-adjacent (one-ring). Divergence-operator cross-check
(`diff_seam.py`): interior barotropic divergence is **bit-identical** to NEMO's
full-halo value (rms 5.482e-9); the recon error lives **entirely** at seam col0 / wall
rows (1.2e-4 / 6.9e-5). Confounds real, but interior-clean.

**Step-1 rms|Δ|, full-wet "interior" → strict interior:**

| field | FE full | FE strict | MLF-after full | MLF-after strict |
|-------|--------:|----------:|---------------:|-----------------:|
| eta   | 1.88e-2 | **1.38e-2** | 2.14e-2 | **2.00e-2** |
| T     | 3.57e-3 | 3.36e-3 | 5.50e-3 | 4.88e-3 |
| u     | 3.40e-3 | 2.71e-3 | 3.40e-3 | 3.18e-3 |
| v     | 2.41e-3 | 1.46e-3 | 2.32e-3 | 1.37e-3 |

**The eta divergence does NOT collapse to ~1e-3** — stripping the confound removes
only ~25% (FE) / ~7% (MLF) of it. The models genuinely differ at step 1 by ~1.4–2.0e-2 m
eta / 3–5e-3 °C T; this is a real offset, not roundoff and not a mask artifact. It is
**surface-layer-dominated** (step-1 surface-level T rms 1.4e-2 (FE) / 2.3e-2 (MLF) ≫
strict-interior 3–5e-3), i.e. the free surface + top-cell forcing placement, present in
BOTH integrators — the likely bridge/surface-forcing residual, slowly growing.

**Growth / runaway.** Both FE and MLF blow up ~step 22–24 in the strict interior too
(FE eta 4.2e-2 @s15 → 0.44 @s20 → overflow @s25; MLF-after strict T 1.7e-2 @s10 → 4.1e-2
@s15 → 1.35e-1 @s20 → NaN @s24), so the runaway is **genuine interior**, not seam/wall
(seam/wall T inflate the full number ~1.2–4× at late steps but are not the early signal).

**Runaway localization (strict mask, `stepdump_twin_loc.py`).** The dominant
strict-interior T-divergence cell sits at the **|lat|≈68° staircase** (lat +67.7…+68.4°,
ix≈43–45, mid-depth 49–262 m) — the same high-lat topographic-step mode as before,
surviving the strict mask. It overtakes the equatorial surface signal by ~step 10 and
goes exponential from ~step 19 (|dT|max 7.4 @s19 → 125 @s21 → 2e6 @s23 → NaN @s24).

**Verdict (freeze/finish).**
- (a) Genuine differences: step-1 strict-interior eta ~1.4e-2 m (FE) / 2.0e-2 m (MLF),
  T ~3–5e-3 °C, u/v ~1.5–3e-3 m/s — surface/free-surface dominated, slowly growing.
- (b) The |lat|≈68° staircase MLF runaway is the **only fatal** item (it also kills FE
  by ~step 22), but it is NOT the only genuine one: the ~1.4–2e-2 m surface-layer offset
  is a smaller persistent bias in both integrators. Every named interior term still
  certifies 1.0000; these two are integrated-state divergences, not per-term defects.
- (c) Next single action after the reset: pin the +68° staircase-bottom column
  (ix≈43–45, 50–260 m). Dump its T (and u/v) tendency terms at steps 18–20 vs the NEMO
  trend — the mid-depth signature at a topographic step points at the partial-cell
  vertical-mixing / tracer-advection stencil, same tier-3 approach that localized HPG-v.

### CORRECTION (2026-07-18) — the runaway is eta-FIRST barotropic, NOT a tracer/EVD defect

Point (c) above is **superseded** — the "mid-depth ix43–45 tracer" localization was a
strict-mask artifact, not the source cell. A lead-variable probe from the identical
bridged restart (FE, dt=2700; `scratchpad/lead_var_probe.py` + `eta_mode_probe.py`)
tracks the ORDER in which fields diverge in the high-lat window (lat 66–70°):

| field | s14 | s18 | s20 | s22 | s24 | s25 |
|-------|----:|----:|----:|----:|----:|----:|
| **eta**  | 1.0 | 2.7 | 7.6 | 37.7 | 818 | 1.6e8 |
| max\|u\|,\|v\| | 0.26 | 0.26 | 0.28 | 1.4 | 27/43 | 1e9 |
| T (\|dT\| vs NEMO) | — | 0.33 | 0.35 | 0.36 | 3.0 | 7.3e17 |

**eta goes exponential from ~step 14, ~10 steps BEFORE u/v (step 21) and ~10 before T
(step 24).** The tracer "+68 staircase runaway" is the TERMINAL symptom — advection of
the already-blown velocity field — not the driver. The actual explosion is at the
**surface** (lev 0), \|lat\|+67.7–68.4°, ix≈22–27.

**Mechanism (evidence-backed, named).** eta along a high-lat row is a pure **2Δx
checkerboard** growing exponentially (`eta_mode_probe.py`, row 191, uniform kbot=34):
`-0.11 +0.11 -0.16 +0.15 …` → amplitude 0.6 (s16) → 5.2 (s20). This is the **C-grid
barotropic-Coriolis grid-scale (2Δx) null mode**, least-damped at high latitude (where
`e1=R·Δλ·cosφ` is smallest and the Coriolis-averaging null mode is most energetic) —
the same research-level blocker as Round-8 (`dynspg_ts` / barotropic-Coriolis redesign,
memory `project_recipe_architecture_design`/`silvestri` barotropic-Coriolis note). NEITHER
the FE forward-backward NOR the MLF centred barotropic frame damps this spatial null
mode (MLF blows at the same step — Round-8). The `barotropic_time_filter=
"nemo_boxcar_ab3"` guarded out of the FE frame is a TEMPORAL filter and does not touch
the spatial 2Δx null mode.

**EVD flip-flop verdict: NEGATIVE.** `scratchpad/evd_flipflop_probe.py` recorded the
enhanced-diffusion convective flag + adiabatic N² at ix43–45 / levels 4–15 every step to
s24: N² stays **positive-stable** (~+3.4e-7 s⁻²), flag=0, **zero flips**. The
`two_level_trigger` MIN(rn2,rn2b) EVD hysteresis (already built, k_profiles.py:767) is
**not the cause** and wiring it would be a phantom fix.

**Remaining work (larger than one pass, no stabilizer):** the faithful fix is the
C-grid barotropic-Coriolis 2Δx-null-mode redesign (energy/enstrophy-conserving vorticity
flux that filters the grid-scale null mode, or the NEMO EEN-consistent barotropic
Coriolis in the split-explicit substep). This is the standing barotropic redesign task,
not a tracer/vmix change. No non-NEMO stabiliser was added; no MLF number fabricated.
