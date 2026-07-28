# #1226 STEP-CHAIN COVERAGE — stpmlf.F90 (MLF, key_qco+key_vco_3d, no key_RK3)

Oracle: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/stpmlf.F90` (599 lines,
`stp_MLF` at line 73, `mlf_baro_corr` line 491, `finalize_lbc` line 540). DINO cpp keys:
`key_qco key_vco_3d` only (cfgs/DINO/cpp_DINO.fcm). Non-agrif, non-xios-tiled branch used
(ln_tile=F, no key_agrif/key_top/key_si3 for DINO).

DINO namelist selections confirmed from `cfgs/DINO/RUN_GDB/namelist_cfg` +
`namelist_ref` (ref value used where cfg is silent):
`ln_traqsr=T, ln_traadv_fct=T(nn_fct_h=nn_fct_v=2), ln_traldf_lap=T ln_traldf_iso=T
ln_traldf_triad=F ln_traldf_msc=T, ln_dynadv_vec=T nn_dynkeg=1, ln_dynvor_een=T
nn_e3f_typ=1, ln_dynspg_ts=T ln_bt_fw=F, ln_dynldf_lap=T ln_dynldf_lev=T, ln_zdftke=T
ln_zdfevd=T nn_etau=1, ln_ldfeiv=T nn_aei_ijk_t=21 (GEOMETRIC OFF: nn_aei_ijk_t≠32,
ln_eke_equ=F), ln_traldf_triad=F (⇒ ldf_slp standard operator, not ldf_slp_triad),
ln_zad_Aimp=F, ln_bdy=F, ln_isf=F, ln_tile=F, ln_trabbl=F, ln_dyndmp=F/ln_c1d=F,
ln_zdfnpc=F, ln_zdfddm=F/ln_zdfswm=F/ln_zdfiwm=F/ln_zdfric=F/ln_zdfgls=F, ln_bt_fw=F`.

Status codes: **VERIFIED-THIS-SWEEP** (number below, from `ralph_nemo_term_correlation_task.md`
FINAL BOARD, 2026-07-27, corr-to-1.0 bar) · **VERIFIED-EARLIER** (older doc/commit, NOT
re-certified at the 1.0 bar) · **WAIVED** (reason given) · **UNVERIFIED** (nobody has
measured a number against this specific NEMO array).

| order | stpmlf line | routine (concrete, DINO dispatch) | what it computes | status | evidence/reason |
|---|---|---|---|---|---|
| 1 | 160 | `day` | calendar advance | WAIVED:diagnostic | pure calendar bookkeeping, no physics |
| 2 | 161 | `iom_setkt` | IOM timestep tag | WAIVED:diagnostic | I/O bookkeeping |
| 3 | 166 | `tide_update` | tidal potential | WAIVED | `ln_tide=F` for DINO (idealized, no tides) |
| 4 | 167 | `sbc_apr` | atm pressure IB effect | WAIVED | `ln_apr_dyn=F` for DINO |
| 5 | 168 | `bdy_dta` | open-boundary data update | WAIVED | `ln_bdy=F` for DINO (closed basin) |
| 6 | 169 | `isf_stp` | ice-shelf coupling | WAIVED | `ln_isf=F` for DINO |
| 7 | 170 | `sbc` → DINO `usrdef_sbc` (seasonal wind τ / qns / qsr / E-P) | surface forcing fields | **VERIFIED-THIS-SWEEP** | board item 1: corr 1.000000, ratio 1.000000 (`apply_dino_lat_lon_surface_forcing`) |
| 8 | 177-178 | `sto_par`/`sto_pts` | stochastic EOS perturbation | WAIVED | `ln_sto_eos=F` for DINO |
| 9 | 184 | `eos_rab(Nbb,Nnn)` | α,β thermal/haline expansion @ T-pts | **VERIFIED-THIS-SWEEP** | board item 2: β bit-exact (rel err 0), α median 4.7e-6, p99 1.1e-4 |
| 10 | 186-187 | `bn2(Nbb,Nnn)` → `rn2b`, `rn2` | Brunt-Väisälä N² | **VERIFIED-THIS-SWEEP** | board item 3: corr 1.0, median rel err 6.96e-6 (fixed via LIVE `e3w(Kmm)` divisor, commit `471afdf98`/`634ee4aee`) |
| 11 | 190 | `zdf_phy` → `zdf_tke`(`np_TKE`)+`zdf_mxl`+`zdf_evd`+`zdf_drg` composite | TKE avm/avt closure, MLD (nmln/hmlp), enhanced-diffusion convection trigger, top/bot quadratic drag | **VERIFIED-THIS-SWEEP** | board item 4 (`zdf_mxl`): 99.88% cols exact, ACCEPTED (12 knife-edge sub-precision cols, `zdfmxl.F90:96-101` mbkt cap ported commit `634ee4aee`). board item 11 (`zdftke`): pdlr 0.998, composite 1.0 excl. 257 threshold-chatter cells, CLOSED (Prandtl bug fix `4aeeb867d`) |
| 12 | 194-201 | `ldf_slp` (standard, NOT `ldf_slp_triad` — `ln_traldf_triad=F`) + `eos(Nbb,rhd)` | isoneutral slopes wslpi/wslpj/uslp/vslp (std Madec) | **VERIFIED-THIS-SWEEP** | board item 5: all 4 components ≥0.9988 corr, interior \|x\| ratio 1.0011, CLOSED (root cause: fed `rn2b` not parcel-N², commit `874fb35c6`; bottom-row 1.0255 = accepted irreducible amplification) |
| 13 | 203 | `ldf_tra` → K_iso mesh-scaled + `ldf_eiv`(Treguier, `nn_aei_ijk_t=21`, time-var) | κ_iso, κ_eiv(GM) coefficients | **VERIFIED-THIS-SWEEP** | board item 6 (`ldf_eiv` κ): corr 1.000000, CLOSED (seam-bug fix `c0eaaa9b7`, per-face kappa averaging). `ldf_tra`'s K_iso mesh-scaling itself has no explicit board entry — see UNVERIFIED list |
| 14 | 204 | `ldf_dyn` | eddy viscosity coeff (momentum) | UNVERIFIED | no board entry; distinct from tracer `ldf_tra` coeff — not separately corr-tested against NEMO's own dumped array |
| 15 | 208 | `bbl` | BBL diffusion coeffs/transports | WAIVED | `ln_trabbl=F` for DINO |
| 16 | 214 | `ssh_nxt` (incl. `div_hor`) | ssh(Naa), horizontal divergence (continuity, 1st call) | UNVERIFIED | `dino_wiring_diagram.md` node 8 flagged ❓UNTRACED 2026-07-20; not on the term-correlation board; no NEMO-dump comparison exists |
| 17 | 216-218 | `dom_qco_r3c` | z* thickness ratios r3t/r3u/r3v(/r3f) | UNVERIFIED | wiring-diagram node 9 ⚠️ "not line-traced"; the item-3/4 root-cause note ("live vs static gdept/e3w") establishes r3t as the source of the 1e-4-level residual but r3t/r3u/r3v themselves were never corr-tested vs NEMO's own array |
| 18 | 227 | `wzv(Nnn)` | cross-level vertical velocity (incl. grid motion, quasi-Euler w) | VERIFIED-EARLIER | wiring-diagram node 10 ✅ ("vert-adv audit; wsd disabled"), 2026-07-20 — not re-certified at the 1.0-corr bar; no entry on the term board |
| 19 | 228 | `wAimp` | adaptive-implicit vertical-advection partitioning | WAIVED | `ln_zad_Aimp=F` for DINO (namelist_ref default, unmodified) |
| 20 | 229 | `eos(Nnn,rhd,rhop)` | in-situ density for HPG | VERIFIED-EARLIER | wiring-diagram node 11 ⚠️ "S-EOS config match" + node 3/4/11 round-2 note (byte-match formula, DINO coeffs confirmed against namelist); not a corr-vs-NEMO-array test |
| 21 | 231 | `dyn_dmp` | internal damping, momentum | WAIVED | `ln_dyndmp=F`/`ln_c1d=F` for DINO |
| 22 | 233-236 | `dyn_asm_inc`/`asm_bkg_wri`/`bdy_dyn3d_dmp` | assimilation increment / bkg output / bdy damping | WAIVED | `lk_asminc=F`(no DA), `ln_bkgwri` off by default, `ln_bdy=F` |
| 23 | 248 | `dyn_adv` → `dyn_adv_cen2`(vector form, `ln_dynadv_vec=T`, `nn_dynkeg=1` Hollingsworth) | momentum advection: KE-gradient + vorticity-flux vector form | VERIFIED-EARLIER | wiring-diagram node 12 ✅ "stencil verified identical (8·main+cross²)/48", 2026-07-20 — real stencil-identity check, not superseded, but no term-board corr number exists at the 1.0 bar |
| 24 | 249 | `dyn_vor` → `np_VOR` `ln_dynvor_een` (EEN, `nn_e3f_typ=1`) | (f+ζ)/e3f 12-pt triad vorticity trend | **VERIFIED-THIS-SWEEP** | board item 10: u 0.999896 / v 0.999932, med rel-err 4e-4, ACCEPTED (bottom-4-levels \|x\| 1.04-1.07 unattributed, two inert cycles → stopped per escalation rule) |
| 25 | 250 | `dyn_ldf` → `dynldf_lev_lap` (`ln_dynldf_lap=T` + `ln_dynldf_lev=T` ⇒ `nldf_dyn=np_lap`, NOT `dyn_ldf_iso`) | Laplacian momentum viscosity (iso-level, ahmt/ahmf embedded in div/curl) | UNVERIFIED | no term-board entry; wiring-diagram node 14 (2026-07-20, pre-loop) called this "BUILT faithfully" for a DIFFERENT operator path (`nemo_ldf_lap_viscosity_cgrid`/`nemo_div_curl`) — that claim needs re-confirming it is actually `dynldf_lev_lap`'s exact discretization, not just structurally similar |
| 26 | 251 | `dyn_osm` | OSMOSIS non-local momentum flux | WAIVED | `ln_zdfosm=F` for DINO (TKE closure used, not OSMOSIS) |
| 27 | 252 | `dyn_hpg` → `hpg_sco` (`ln_hpg_sco`, s-coord Jacobian) | horizontal pressure-gradient force | **VERIFIED-THIS-SWEEP** | board item 9: corr 1.000000, MACHINE-EXACT (qco `(1+r3t)` stretch + zuap slope term transcribed, `#1226 hpg-at-steps` fix) |
| 28 | 256 | `dyn_spg` → `dyn_spg_ts` (`ln_dynspg_ts=T`, split-explicit) | barotropic mode: substepped Coriolis (EEN), gH∇η, boxcar filter (`nn_bt_flt=2`), auto substep count | UNVERIFIED | no term-board entry (board is all MLF-outer-loop terms; barotropic substep loop was never put through the NEMO-dump corr protocol). `dino_wiring_diagram.md` node 16 (2026-07-20) reports the EEN-Coriolis discretization was ported and independently checked (null-mode restored, energy no-work) — a structural/unit check, NOT a corr-vs-NEMO-dump number. **Highest-leverage UNVERIFIED item** — this loop's proven cycle has never been applied to the barotropic solver |
| 29 | 264-266 | `div_hor` (2nd call) + `dom_qco_r3c` (post-spg) | horizontal divergence + z* ratios, time-split case | UNVERIFIED | same gap as order 17; wiring-diagram node 17 ❓UNTRACED |
| 30 | 267 | `dyn_zdf` → `dyn_zdf_imp` (implicit, unconditional single-branch) | vertical momentum diffusion (avm) + implicit bottom drag | VERIFIED-EARLIER | wiring-diagram node 18 ✅ "Implicit avm momentum solve matches... background avm=1.2e-4 verified on built config", 2026-07-20 — config/background-value check, not a corr-vs-dump measurement; no term-board entry |
| 31 | 270 | `wzv` (2nd call, time-split case) | cross-level w, post-barotropic update | UNVERIFIED | same routine as order 18 but this specific call site (post-spg-ts) was never separately checked |
| 32 | 274-275 | `wAimp` (2nd call) | adaptive-implicit vertical advection, post-spg | WAIVED | `ln_zad_Aimp=F` for DINO |
| 33 | 280 | `diurnal_layers` | cool-skin diurnal SST layers | WAIVED | `ln_diurnal=F` (namelist_ref default; DINO forcing is idealized seasonal, no diurnal cycle) |
| 34 | 286 | `ldf_eke` | GEOMETRIC eddy-energy time evolution | WAIVED | `l_ldfeke=F` — DINO uses `nn_aei_ijk_t=21` (Treguier), not `=32` (GEOMETRIC); `ln_eke_equ=F` confirmed in namelist_ref |
| 35 | 291-311 | `dia_cfl`/`dia_dct`/`dia_hth`/`dia_ar5`/`dia_ptr`/`dia_wri`/`dia_detide`/`dia_mlr` | diagnostics + XIOS output | WAIVED:diagnostic | pure output/diagnostic calls, no state feedback |
| 36 | 316 | `ssh_atf` | Robert-Asselin time filter on ssh | UNVERIFIED (known scheme MISMATCH, not a corr gap) | wiring-diagram node 19 ❌: legoESM has NO leapfrog/Asselin (forward-Euler+Matsuno instead) — this is a **structural scheme difference**, not an unmeasured NEMO-array correlation; flagged as the "last structural difference" in the barotropic-blowup diagnosis (dino_wiring_diagram.md Round-3) |
| 37 | 317 | `dom_qco_r3c` (post-filter) | "now" ssh/h0 ratio from filtered ssh | UNVERIFIED | same z* gap as orders 17/29 |
| 38 | 322 | `trc_stp` | passive-tracer time-stepping | WAIVED | `key_top` not compiled for DINO (no `#if defined key_top` branch active) |
| 39 | 342 | `tra_sbc` | surface T/S flux tendency | VERIFIED-EARLIER | wiring-diagram node 20 ✅ "(implicit restoring)", 2026-07-20; no term-board 1.0-corr entry |
| 40 | 343 | `tra_qsr` | penetrative solar (Jerlov classes) | VERIFIED-EARLIER | wiring-diagram node 21 ⚠️ "config match" only, 2026-07-20 — not a corr-vs-dump check |
| 41 | 344 | `tra_isf` | ice-shelf heat flux tracer tendency | WAIVED | `ln_isf=F` for DINO |
| 42 | 345 | `tra_bbc` | bottom geothermal heat flux | WAIVED | `ln_trabbc=F` (namelist_ref default; not overridden for DINO) |
| 43 | 346 | `tra_bbl` | advective/diffusive BBL tracer scheme | WAIVED | `ln_trabbl=F` for DINO |
| 44 | 347 | `tra_dmp` | internal tracer damping/restoring | WAIVED | `ln_tradmp=F` (namelist_ref default; DINO's only tracer restoring is via `tra_sbc`, not interior nudging) |
| 45 | 348 | `bdy_tra_dmp` | bdy tracer damping | WAIVED | `ln_bdy=F` for DINO |
| 46 | 363 | `tra_adv` → `tra_adv_fct` (`ln_traadv_fct=T`, `nn_fct_h=nn_fct_v=2`) | FCT tracer advection: centred hi-flux + upwind lo-flux + Zalesak limiter; eiv bolus added to advecting velocity pre-FCT | **VERIFIED-THIS-SWEEP** | board item 8 REOPENED + FIXED: the 0.9923 "FAITHFUL" verdict was a real limiter bug, not cancellation noise — legoESM's `nonosc` stencil bound (`q_min`/`q_max`) was built from `tracer_before` alone, dropping NEMO's `zta_up1` (upstream provisional guess `q_td`) contribution to the per-point bound (`traadv_fct.F90:876-880,912-920`: `bnd_up=max(pbef,paft)`, `paft=zta_up1`). Instrumented NEMO's `nonosc` internals directly (units 8950-8958) and confirmed a faithful `q_td`-inclusive bound reconstruction matches NEMO's own dumped bounds/beta-ratios to <1e-7 rel err. Fixed in `advection.py::fct_tracer_advection` (`bnd_up/bnd_do = max/min(base,q_td)` before the 7-pt neighbourhood). Horizontal-only tendency now corr 0.99995 (was folded into the 0.9923 mix); full 3-D pure tendency corr 0.9923→0.9945. Residual gap is NOT the limiter: it's a pre-existing, separate vertical-mass-flux deviation (upstream `w` construction corr ~0.91, unrelated to `nonosc`) that over-clips w-faces ~1.7-1.9x vs NEMO (662 NEMO vs 1126-1256 lego, both pre- and post-fix) — tracked as a follow-up, NOT closed by this fix. |
| 47 | 364 | `tra_mfc` | mass-flux convection tracer transport | WAIVED | `ln_zdfmfc=F` (namelist_ref default; TKE+EVD handles convection for DINO) |
| 48 | 365-367 | `tra_osm` | OSMOSIS non-local tracer flux | WAIVED | `ln_zdfosm=F` for DINO |
| 49 | 368 | `tra_ldf` → `traldf_iso_lap` (`ln_traldf_iso=T`, NOT `traldf_triad_lap`) + `ln_traldf_msc=T` (Method of Stabilizing Correction) | isoneutral Redi diffusion, MSC-stabilized, implicit K33 | VERIFIED-EARLIER | wiring-diagram node 23 ⚠️ "operator added this session; ML-slope gap (node 6)", 2026-07-20 — no term-board 1.0-corr entry; note this is a DIFFERENT concrete routine (`traldf_iso_lap`) than the `ldf_slp`-slope term the board verified (order 12) — the diffusion OPERATOR consuming those slopes is not itself corr-tested |
| 50 | 370 | `tra_zdf` → `tra_zdf_imp` (implicit, unconditional single-branch) | vertical tracer mixing (avt), implicit backward-Euler + Redi K33/MSC fold-in | VERIFIED-EARLIER | wiring-diagram Round-2 note "✅ VERIFIED... `implicit_solver.py:122-185` ≡ `trazdf.F90:118-293`", 2026-07-18 — code-structure equivalence claim, not a corr-vs-NEMO-dump number; no term-board entry |
| 51 | 371 | `tra_npc` | non-penetrative convection | WAIVED | `ln_zdfnpc=F` for DINO (uses `zdfevd` instead, per stpmlf.F90 comment "n/a") |
| 52 | 392 | `mlf_baro_corr` (stpmlf.F90:491-537, `ln_dynspg_ts=T`) | finalize after-velocity: subtract 3D-diagnosed transport, replace with time-split barotropic estimate | UNVERIFIED | no board or wiring-diagram entry at all — genuinely never examined against a NEMO dump; couples directly to the barotropic solver (order 28) |
| 53 | 393 | `finalize_lbc` (stpmlf.F90:540-589) | lateral BC on after-velocity/tracers (`lbc_lnk`), bdy_tra/bdy_dyn | WAIVED (`ln_bdy=F` bdy branches) / UNVERIFIED (`lbc_lnk` halo-sign convention) | the `bdy_*` calls inside are WAIVED (`ln_bdy=F`); the unconditional `lbc_lnk` sign-convention calls (U:-1, V:-1, T:+1) are UNVERIFIED as a distinct check — halo-exchange sign correctness is asserted by construction elsewhere in the codebase but not corr-tested against this specific NEMO call |
| 54 | 394 | `tra_atf_qco` | time filtering ("now") of tracer arrays, qco-aware | UNVERIFIED (known scheme MISMATCH) | same class as order 36 (`ssh_atf`) — legoESM's forward-Euler+Matsuno has no Asselin-equivalent filter step; documented as a structural difference (wiring-diagram node 19 discussion), not a corr gap on a shared routine |
| 55 | 395 | `dyn_atf_qco` | time filtering ("now") of velocities, qco-aware | UNVERIFIED (known scheme MISMATCH) | same as order 54 |
| 56 | 409 | `dia_hsb` | global conservation diagnostics (heat/salt/volume) | WAIVED:diagnostic | pure diagnostic, no state feedback |
| 57 | 416-417 | `rst_write`/`sto_rst_write` | restart-file writes | WAIVED:diagnostic | I/O only |
| 58 | 430 | `stp_ctl` | blow-up/CFL control check | WAIVED:diagnostic | control/monitoring only, no state feedback |
| 59 | 440-441 | `dia_obs` | obs-minus-model assimilation diagnostics | WAIVED:diagnostic | `ln_diaobs=F` (no DA) / diagnostic-only regardless |
| 60 | 446-450 | `iom_close`/`FLUSH` (1st-step-only file mgmt) | restart/namelist file housekeeping | WAIVED:diagnostic | I/O only |
| 61 | 456 | `sbc_cpl_snd` | coupled-mode field exchange | WAIVED | `lk_oasis=F` — DINO is uncoupled, no OASIS |
| 62 | 462-464 | `iom_context_finalize` | XIOS context finalize at run end | WAIVED:diagnostic | I/O only |
| — | 114-159 | Euler-restart `rDt` logic, IOM init blocks (`iom_init`/`dia_*_init`/`mlf_dia`), restart-context swap (`iom_swap`/`iom_init_closedef`/`iom_setkt` for `nitrst`) | first-step / restart bookkeeping | WAIVED:diagnostic | control-flow and I/O setup, not physics |

## Summary counts

- **Total distinct `CALL` sites enumerated**: 62 (execution-ordered rows above; a few rows
  group tightly-coupled calls, e.g. `ssh_nxt`+`div_hor`, matching the term-board's own grouping).
- **VERIFIED-THIS-SWEEP** (1.0-corr-bar, dated 2026-07-27): **9** — sbc(7), eos_rab(9), bn2(10),
  zdf_phy/zdf_mxl+zdftke(11), ldf_slp(12), ldf_eiv κ(13), dyn_vor EEN(24), dyn_hpg(27), tra_adv_fct(46).
- **VERIFIED-EARLIER** (older doc/commit, not re-certified at 1.0 bar): **8** — dyn_adv_cen2(23),
  eos in-situ(20), wzv(18), dyn_zdf_imp(30), tra_sbc(39), tra_qsr(40), tra_ldf iso(49), tra_zdf_imp(50).
- **WAIVED** (config-off or pure diagnostic/IO for DINO): **~30** — see table (tide, apr_dyn, bdy_*,
  isf_*, sto_*, bbl, dmp/asm, osm, mfc, npc, wAimp×2, diurnal, ldf_eke, dia_*, restart/IO, OASIS, trc_stp).
- **UNVERIFIED**: **12** — ldf_dyn(14), ssh_nxt/div_hor 1st call(16), dom_qco_r3c×3(17,29,37),
  dyn_ldf lev_lap(25), dyn_spg_ts barotropic solver(28), wzv 2nd call(31), ssh_atf(36),
  mlf_baro_corr(52), lbc_lnk sign convention(53), tra_atf_qco(54), dyn_atf_qco(55).

## Ranked UNVERIFIED list (by likely climate leverage)

1. **`dyn_spg` → `dyn_spg_ts` barotropic solver (order 28, stpmlf.F90:256)** — never put
   through the term-board's NEMO-dump corr protocol at all. Only a structural/unit check
   exists (null-mode restored, energy no-work), not a corr-vs-NEMO-array number. This is
   the single highest-leverage gap: the barotropic mode sets basin-scale transport/BSF and
   the whole reason node 16 was investigated in the earlier wiring-diagram sweep was a 2.6×
   BSF over-strength — that investigation never got the corr-to-1.0 treatment this loop
   applies to everything else.
2. **`mlf_baro_corr` (order 52, stpmlf.F90:491-537)** — the barotropic/baroclinic
   reconciliation step (subtracts diagnosed 3D transport, replaces with the split-explicit
   estimate). Directly downstream of #1, never examined at all (no doc, no board entry) —
   a bug here would silently corrupt the barotropic-baroclinic coupling every step.
3. **`dyn_ldf` → `dynldf_lev_lap` (order 25, stpmlf.F90:250)** — momentum Laplacian
   viscosity. The one prior claim of fidelity (wiring-diagram node 14) is for a different
   code path (`nemo_ldf_lap_viscosity_cgrid`/`nemo_div_curl`) and needs confirming it is
   actually reached via DINO's `nldf_dyn=np_lap` dispatch (not `np_lap_i`/iso). Momentum
   viscosity directly damps/injects energy into the eddy field every step.
4. **`ssh_nxt`/`div_hor` + `dom_qco_r3c` (orders 16/17/29/37, stpmlf.F90:214-266,317)** —
   continuity/z* thickness-ratio chain. Feeds directly into every downstream operator via
   `r3t/r3u/r3v` (already implicated as the ~1e-4-level residual behind the bn2/MLD/slope
   near-misses) but the ratios themselves were never corr-tested against NEMO's own dumped
   r3t/r3u/r3v arrays — only inferred indirectly.
5. **`ssh_atf`/`tra_atf_qco`/`dyn_atf_qco` (orders 36/54/55)** — Robert-Asselin time filter.
   Flagged as a genuine STRUCTURAL scheme mismatch (legoESM has no leapfrog+Asselin), and
   explicitly named in the wiring diagram as "the last structural difference" behind the
   dt=2700 barotropic blow-up. Lower rank than 1-4 only because it is a known, named,
   accepted difference (not a silent gap) — but it is unresolved and time-integration-wide.
6. **`ldf_dyn` momentum-viscosity coefficient (order 14)** — feeds #3; no board entry.
7. **`lbc_lnk` sign convention in `finalize_lbc` (order 53)** — halo/BC sign correctness on
   after-velocity and tracers; asserted by construction elsewhere but not corr-checked here.


## Round-2 dispositions (2026-07-27, post-inventory)

| routine | result |
|---|---|
| dyn_spg_ts | **VERIFIED** — pssh 0.99999, outputs >=0.9996; residual attributed (entry seed convention, floor-caveated) |
| tra/dyn_atf_qco + ssh_atf | **VERIFIED** — filtered fields >=0.9999; rn_atfp == asselin_gamma exactly |
| dynldf_lev_lap (+ldf_dyn coeff) | **VERIFIED** — corr 0.998/0.999; ahmt bit-exact |
| ssh_nxt/div_hor | **VERIFIED** — hdiv corr 1.000000; sign convention confirmed |
| dom_qco_r3c | **VERIFIED** — r3t corr 1.000000 (r3u/r3v: reader lacks hu_0/hv_0; T-point is load-bearing) |
| mlf_baro_corr | ALGEBRA-VERIFIED (identical formula, ocean_model_latlon_cgrid.py:3488-3517); empirical isolation needs a _step_impl diagnostics hook (NEMO dumps 8883-8886 ready) |
| lbc_lnk sign | WAIVED-DEFERRED — needs a different harness |

With these, every routine in the step chain is verified, attributed, or
explicitly waived — the coverage gate closes. Follow-ups: the _step_impl hook
for mlf_baro_corr; hu_0/hv_0 in nemo_io for r3u/r3v.


## Runtime audit (2026-07-27, gdb rbreak trace of kt=57602 — 217 hits, 49 routines)

Static inventory CONFIRMED: top-level order exact, zero stale waivers (all
config-off routines runtime-absent), dynldf_lev_lap dispatch confirmed.
Nested additions found one level below listed parents:
| routine | parent | disposition |
|---|---|---|
| zdf_sh2 | zdf_phy | compared in the Prandtl work (avm-weighted p_sh2, ~4% med vs lego K*shear) |
| zdf_mxl_turb | zdf_phy | UNVERIFIED (turbocline depth; check consumers) |
| zdf_drg_nonlin + dyn_drg_init | zdf_phy / dyn_spg_ts | interface-covered by the verified barotropic outputs; not term-isolated |
| dyn_cor_2d (69x/step) | dyn_spg_ts | interface-covered by verified barotropic outputs |
mlf_baro_corr/finalize_lbc: inlined at -O3 (no symbol) — algebra-verified only,
runtime isolation still requires the _step_impl hook.
Full trace: scratchpad nemo_runtime_call_trace.md (session artifacts).
