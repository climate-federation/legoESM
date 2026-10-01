# DINO surface forcing — legoESM vs NEMO 5.0.2 alignment table

*Companion to `dino_handoff_2026_07.md`. Same method as the barotropic (`dynspg_ts`) and
zdftke tables: the ordered, step-for-step call chain on both sides with file:line and the
actual expression, MATCH/DIFF per row. Built 2026-07-25 after the ablation program
eliminated GM as the ACC-growth carrier and surface buoyancy forcing became the leading
remaining suspect.*

**Verdict: surface forcing is FAITHFUL.** Every physical formula and coefficient matches.
Six residual structural deviations are catalogued below; all are ≤0.3% and none is a
candidate for the ~60% ACC growth-rate deficit. This row of the suspect list is closed.

Config traced: recipe `nemo_dino_kamm_mlf`; NEMO `cfgs/DINO/RUN_20Y/namelist_cfg`
(`nn_forcingtype=4`, `ln_usr=T`, `ln_ssr=F`, `ln_traqsr=T`, `ln_qsr_2bd=T`, `key_qco`,
z-star vvl, MLF — not RK3).

## Physics rows — all MATCH

| # | quantity | NEMO | legoESM | verdict |
|---|---|---|---|---|
| 1 | wind stress profile | `usrdef_sbc.F90:162-163,221` knots (φ_min,−45,−15,0,15,45,φ_max) = (0, `rn_ztau0`=0.2, −0.1, −0.02, −0.1, 0.1, 0); smoothstep `znl_cbc:617-618` `(3−2s)s²` | `dino.py:1390-1431` `wind_tau_lats_deg=(−70,−45,−15,0,15,45,70)`, `wind_tau_values=(0, 0.2, −0.1, −0.02, −0.1, 0.1, 0)`; same `(3−2s)s²` | **MATCH** |
| 2 | τ_m (TKE surface BC) | `usrdef_sbc.F90:222-223` `taum=ABS(utau)`, ×1.3 where `utau>0` | `dino.py:3061-3062` same, `taum_westerly_boost=1.3` | **MATCH** |
| 3 | T\* meridional profile | `usrdef_sbc.F90:232-240` `T*_ns+(T*_eq−T*_ns)·sin(π(φ+φ_max)/(φ_max−φ_min))` | `dino.py:1434-1448` `T*_ns+(T*_eq−T*_ns)·cos(πφ/L_φ)`, `L_φ=140` | **MATCH** — `sin(π(φ+70)/140) ≡ cos(πφ/140)` |
| 4 | T\* seasonal amplitude | `usrdef_sbc.F90:216-217` `T*_s − 0.5·c₂`, `T*_n + 3.0·c₂` | `dino.py:1523-1538` `amp_s=0.5`, `amp_n=3.0` | **MATCH** |
| 5 | T\* endpoints | `rn_tstar_s=−0.5`, `rn_tstar_n=5.0`, `rn_tstar_eq=27.0` | `T_star_s_mean=−0.5`, `T_star_n_mean=5.0`, `T_star_eq=27.0` | **MATCH** |
| 6 | S\* profile | `usrdef_sbc.F90:172-184` Munday `(1+cos(2πφ/Δφ))/2` − `1.25·exp(−φ²/7.5²)` | `dino.py:1451-1466` same, dip amp 1.25, σ 7.5 | **MATCH** |
| 7 | S\* endpoints | `rn_sstar_s=35.0`, `rn_sstar_n=35.1`, `rn_sstar_eq=37.25` | `S_star_s=35.0`, `S_star_n=35.1`, `S_star_eq=37.25` | **MATCH** |
| 8 | solar | `usrdef_sbc.F90:266-269` `MAX(230·cos(π(φ−23.5·c₁)/180), 0)`; `ln_diu_cyc=F` so `qsr=qsr_dayMean` | `dino.py:1541-1556` `Q_sr_amp=230`, `solar_declination_amp_deg=23.5` | **MATCH** |
| 9 | SW penetration | `traqsr.F90:665-680` 2-band `rn_abs=0.58`, `rn_si0=0.35 m`, `rn_si1=23.0 m` | `shortwave_penetration.py:127` Jerlov **type I** `R=0.58, ζ₁=0.35, ζ₂=23.0`; `dino.py:223` `jerlov_water_type="I"` | **MATCH** — note the module DEFAULT is type II (0.77/1.5/14.0); DINO explicitly selects I |
| 10 | heat restoring strength | `rn_trp=−40 W/m²/K`; `trasbc.F90:136` `r1_rho0_rcp·qns` ⇒ τ_T = ρ₀c_p·e3t/40 | `config.py:143-152` τ_T = ρ₀c_p·dz₀/`A_theta`, `A_theta=40` | **MATCH** (τ_T ≈ 1.038e6 s = 12.0 d) |
| 11 | salt restoring strength | `rn_srp=−3.858e-3 kg/m²/s`; `trasbc.F90:137` `r1_rho0·sfx` ⇒ τ_S = ρ₀·e3t/3.858e-3 | τ_S = ρ₀·dz₀/`A_S`, `A_S=3.858e-3` | **MATCH** (τ_S ≈ 2.696e6 s = 31.2 d) |
| 12 | top-layer thickness | `e3t_1d[0] = 10.13875112538517 m` | `z_coord.dz_ref[0] = 10.138751` (bridged from the same mesh) | **MATCH** to float32 |
| 13 | applied once, not twice | `ln_ssr=.false.` ⇒ `sbcmod.F90:496 sbc_ssr` never runs; restoring only in `usrdef_sbc` | model `physics.surface_forcing=SurfaceForcingConfig(scheme="none")` (`dino.py:2579`); `_bc_external_surface_forcing` heat/salt block gated on `q_net`, which is `None` ⇒ never fires. Wind applied once: harness no-ops `new_u` when `wind_through_step=True` (`dino.py:3185-3186`) | **MATCH** |
| 14 | E−P | `emp≡0` for `nn_forcingtype=4`, `ln_emp_field=F` | no freshwater flux path active | **MATCH** |

## Residual DIFF rows — catalogued, all ≤0.3%

| # | aspect | NEMO | legoESM | size |
|---|---|---|---|---|
| D1 | tracer time level in the restoring formula | reads `ts(:,:,1,·,Kbb)` — the BEFORE level (`usrdef_sbc.F90:208,257`; `sbc` called with `Kbb=Nbb` at `stpmlf.F90:170`) | reads `state.T` — the NOW level (`restoring.py:141-153`) | rms(T_now−T_before) ≈ 6.5e-5 K vs (T−T\*) = O(1 K) ⇒ ~1e-4 relative |
| D2 | flux→tendency thickness divisor | `e3t(ji,jj,1,Kmm)` — time-varying z-star vvl (`trasbc.F90:153`) | fixed `dz_ref[0]`, a static Python float (`dino.py:3122`) | e3t = e3t₀(1+η/H); η≈±1.3 m, H≈4000 m ⇒ ~0.03% (≤0.3% shallowest columns) |
| D3 | flux time-averaging | `zfact·(sbc_tsc_b + sbc_tsc)`, `zfact=0.5` — trapezoidal before/now flux average (`trasbc.F90:152`) | instantaneous flux, no `_b` counterpart | smoothing, not a bias |
| D4 | restoring timescale | τ as given | `eff_tau = tau + dt` — implicit-Euler denominator shift (`restoring.py:141`) | dt/τ = 2700/1.038e6 ⇒ 0.26% weaker |
| D5 | composition | accumulates into `ts(:,:,:,:,Nrhs)` INSIDE the MLF; passes through the leapfrog and the Asselin filter | forward-Euler overwrite of `state.T`/`state.S` OUTSIDE `model.step` (`dino.py:3179-3180`); bypasses leapfrog + Asselin | per-step RATE identical; differs in filtering only |
| D6 | SW penetration depth coordinate | `gdepw(ji,jj,jk,Kmm)` — time-varying (`traqsr.F90:671,679`) | `z_coord_z_half_ref`, and `jacobian=ones_like(eta)` hard-coded (`dino.py:3165`) | O(η/H) ~1e-4; the module comments this explicitly at `shortwave_penetration.py:529-530` |

## Ordered chains (for reference)

**NEMO** (`stpmlf.F90`): `sbc(kstp,Nbb,Nnn)` :170 → `usrdef_sbc_oce(kt,Kbb)` (`sbcmod.F90:417`)
sets `sfx`/`utau`/`taum`/`qtot`/`qsr`/`qns` → `dyn_zdf` :267 applies `utau` to `uu(Kaa)`
(`dynzdf.F90:332-334`) → `Nrhs` zeroed :336 → `tra_sbc(kstp,Nnn,ts,Nrhs)` :342 → `tra_qsr` :343.

**legoESM** (harness `dino_year_screen_fullframe.py:102-104`, per step):
`apply_dino_lat_lon_surface_forcing` (seasonal T\*/Q_sr → restoring → −Q_sr/(ρc_p dz₀) →
Jerlov column → forward-Euler write to `state.T`/`state.S`; wind no-op) →
`model.step(..., surface_forcing=sf)` where `sf` carries ONLY `tau_x`/`tau_y`/`taum` →
`_bc_external_surface_forcing` (`ocean_pe_latlon_cgrid.py:3406-3411`) adds stress to
`du_dt[...,0]` using `dz_ref[0]·J` with the LIVE Jacobian.

Note the internal asymmetry on the lego side: the **momentum** deposit uses the live
eta-dependent Jacobian `J`, while the **tracer** restoring and the SW column use fixed
reference thicknesses (D2/D6). Both are internally consistent; the mismatch is against
NEMO, and it is small.

## Method notes

- Two comments were found stale while building this table and are recorded so the next
  reader does not trust them: `dino.py:2442` claims GM is off at R1 (it is not —
  `kappa_GM=200` with Treguier enabled), and `dino_step_surface_forcing`'s docstring calls
  the analytic applicator "post-step" when the harness calls it *before* `dyn(st)`.
- A subagent trace reported lego's `dz₀` as "10.0"; the actual value is 10.138751 (bridged
  from NEMO's mesh). **Instantiate and print — do not read declarations.**
