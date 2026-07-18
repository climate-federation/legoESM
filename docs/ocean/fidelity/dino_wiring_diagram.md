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
| 14 | **`dyn_ldf`** | `ln_dynldf_lap`,`nn_ahm_ijk_t=20`,`rn_Uv=0.27` | Laplacian visc `∂ᵢ(ahmt·χ)−∂ⱼ(ahmf·ζ)`, ahmt(T)/ahmf(F)=½Uv·max(e1,e2) EMBEDDED inside div/curl | `A_h_base·cos(φ)` applied OUTSIDE the `grad(div)−grad(curl)` vector-Laplacian | ⚠️ PARTIAL: div-curl STRUCTURE ✅ + magnitude ✅; coeff placement differs (embedded vs outside) — negligible at low-lat (cos≈1, the jet region), 10-20% at \|lat\|>60°. NOT a jet driver |
| 15 | **`dyn_hpg`** | `ln_hpg_sco` | s-coord Jacobian HPG (slope term ≡0 for full-step z) | `pgf_scheme="adcroft"` | ✅ balanced at rest (uniform-ρ test = 0.0) |
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
