# DINO hydrostatic pressure gradient — legoESM vs NEMO 5.0.2 alignment table

*Companion to `dino_surface_forcing_alignment.md`. Same method: ordered chain, file:line,
verbatim expressions, MATCH/DIFF per row. Built 2026-07-26 after both-sided ablation
eliminated GM, Redi and vertical mixing as carriers of the #1226 ACC growth deficit, and
the surface-forcing table eliminated the buoyancy forcing. PGF was the last of the
originally-ranked suspects (handoff §"ranked next steps" item 3).*

**Verdict: the PGF is FAITHFUL.** Scheme, quadrature, slope correction, EOS coefficients
and evaluation time level all match. One row *looks* like a deviation and provably is not
— see the factorization note, which is the load-bearing result of this table.

Config: recipe `nemo_dino_kamm_mlf`; NEMO `cfgs/DINO/RUN_20Y/namelist_cfg`,
`ln_hpg_sco=.true.`, `ln_seos=.true.`, `key_qco key_vco_3d`, no `key_RK3` (MLF).

## Rows

| # | quantity | NEMO | legoESM | verdict |
|---|---|---|---|---|
| 1 | HPG scheme | `ln_hpg_sco=.true.` → `nhpg=np_sco` → `hpg_sco` (`dynhpg.F90:117-123,188-197`) | `pgf_scheme="nemo_sco"` (printed from the live config), `ocean_pe_latlon_cgrid.py:1596-1696` | **MATCH** |
| 2 | vertical quadrature | trapezoidal pairing `(rhd(jk)+rhd(jk-1))` accumulated top-down (`dynhpg.F90:355-362`) | `pgf_quadrature="nemo_trapezoid"`; `inc = e3w_int*(rho_k+rho_{k-1})`, `p' = 0.5*g*cumsum(inc)` (`ocean_tendency_common.py:284-313`) | **MATCH** |
| 3 | thickness in the integral | live `e3w(ji,jj,jk,Kmm)` INSIDE the vertical sum | reference `e3w` ladder inside the sum, `(1+r3t)` applied OUTSIDE as `p_hat = p_prime*stretch` | **MATCH — exact refactoring, see below** |
| 4 | slope-correction term | `zuap = -zcoef0*(rhd(i+1)+rhd(i))*(gdept_z0(i+1)-gdept_z0(i))*r1_e1u`, i.e. `+g*interp(rho)*d(gdept_z0)/dx` (`dynhpg.F90:363-368`) | `- g*interp_cell_to_uface(rho_m)*gradient_x_cgrid(gdept_z0)` entering `dp_dx` with the opposite sign convention (`ocean_pe_latlon_cgrid.py:1682-1691`) | **MATCH** |
| 5 | `gdept_z0` | qco-reconstructed cell-centre depth at `Kmm` | `gdept_z0 = t_depth*(1+r3t) - eta` (`ocean_pe_latlon_cgrid.py:1674-1677`) | **MATCH** |
| 6 | horizontal stencil | 2-point difference × `r1_e1u` / `r1_e2v` | `gradient_x_cgrid` / `gradient_y_cgrid`: 2-point difference ÷ `R·dlon·cos(lat)` (`operators_latlon_cgrid.py:608-654`) | **MATCH** |
| 7 | EOS | `ln_seos=T`, `eos_insitu_New_t` `np_seos` branch (`eosbn2.F90:290-301`) | `eos="nemo_seos"` → `nemo_seos_eos` (`eos.py:454-499`) — identical algebraic form | **MATCH** |
| 8 | EOS coefficients | `a0=0.165, b0=7.6554e-1, lambda1=0.06, lambda2=0, mu1=1.4970e-4, mu2=0, nu=0` | `NemoSEOSConfig`: all seven identical (verified by instantiation) | **MATCH** |
| 9 | density time level | `CALL eos(ts, Nnn, rhd, rhop)` at `stpmlf.F90:229`; `dyn_hpg(kstp, Nnn, ...)` at `:252` | `state.T`/`state.S` (now level) at `ocean_pe_latlon_cgrid.py:3777,3760` | **MATCH** |
| 10 | evaluations per step | once (`stpmlf.F90:229`), used once | recomputed in BOTH `_step_impl` passes, but the Nbb pass's advective result is discarded (only `diss_incr_bb` survives) ⇒ the PGF that reaches the state is the Nnn one | **MATCH** in effect |
| 11 | partial cells | DINO is **full-step z** (`ln_zco_nam=.true.`, `ln_zps_nam=.false.`, `usrdef_zgr.F90:90-92`); `zps_hde` does not exist in NEMO 5.0.2 at all | `masked_zco` → `OceanPartialCellCoordinate`, correction carried by the row-4 slope term | **MATCH** (no partial-cell correction is required on either side) |

## The factorization note (row 3) — why an apparent DIFF is not one

NEMO multiplies each vertical increment by the live `e3w(Kmm) = e3w_0·(1+r3t)`:
```
p'(k) = 0.5*g * SUM_j<=k  e3w_0(j)*(1+r3t) * (rhd(j)+rhd(j-1))
```
legoESM integrates on the reference ladder and applies the stretch once, outside:
```
p_hat(k) = (1+r3t) * [ 0.5*g * SUM_j<=k  e3w_0(j)*(rhd(j)+rhd(j-1)) ]
```
`r3t = ssh/ht_0` is a **2-D field** — horizontally varying but *vertically constant*
(`ocean_pe_latlon_cgrid.py:1671`: `r3t = eta_safe/ht_0` with `ht_0 = sum(h_partial, -1)`;
`stretch = (1+r3t)[..., newaxis]` broadcasts over levels). A vertically-constant factor
commutes with the vertical sum, so the two forms are algebraically identical, not an
approximation.

This mattered because legoESM *does* use fixed reference thicknesses where NEMO uses live
`e3t(Kmm)` in three other places (surface-flux divisor, SW penetration depths, and the
tracer combine — see `dino_surface_forcing_alignment.md` D2/D6 and the content-leak
finding). A fourth instance here would have been a pattern. It is not one.

## What this table does NOT cover

The momentum leapfrog/Asselin path and the barotropic-baroclinic coupling are separate
subsystems, not tabled here. The barotropic solver itself was tabled earlier (N1–N16 vs
`dynspg_ts.F90`, one real gap found and fixed).

## Standing result after this table

Eliminated as carriers of the ACC growth deficit: **GM, Redi, vertical mixing** (both-sided
3-yr matched-state ablation, gap survives at 12.6–15.0 Sv), **surface forcing** (alignment
table), **EOS** (coefficients exact), **PGF** (this table).

The one confirmed defect standing is the **tracer-content leak**: legoESM drifts +8.6e-6
relative in globally-integrated heat over 200 forcing-free steps where NEMO drifts
+3.4e-16 — ten orders apart. Mechanism: the outer leapfrog combine adds a bare
concentration increment (`ocean_model_latlon_cgrid.py:7054-7057`) instead of conserving
thickness-weighted content as NEMO does (`trazdf.F90:271-278`). Fix commissioned as
`.claude/ralph_tracer_content_conservation_task.md`. Its expected magnitude (~9 mK over a
3-yr run vs a ~0.15 K abyssal signal) does NOT obviously account for 12.9 Sv — if the fix
does not close the gap, the search should move to subsystems not yet tabled rather than
back to the ones above.
