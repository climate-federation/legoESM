# PREREG — `tra_ldf` at kt=1, statement by statement

Owner named by the previous round: legoESM's isoneutral lateral diffusion is
**0.996571x** NEMO's on T and **0.996696x** on S after one step from rest, and
it is essentially the WHOLE salinity step-1 residual.  Two different ratios ⇒
structural, not one coefficient.

This round localises the residual to a STATEMENT.  Written before any code.

Oracle read: `cfgs/DINO/BLD/ppsrc/nemo/traldf_iso.f90` (what DINO compiles;
the pre-processed form of `src/OCE/TRA/traldf_iso.F90` +
`traldf_iso_scheme.h90`), `ldftra.f90`, `ldfslp.f90`, and DINO's own
`RUN_TRAJ/namelist_cfg`.

Card resolves (printed by the gate, not asserted): `ln_traldf_lap=T`,
`ln_traldf_iso=T`, `ln_traldf_msc=T`, `rn_slpmax=0.01`, `nn_aht_ijk_t=20`
(`ldf_c2d`, `rn_Ud=0.027`, `rn_Ld=100e3`), `ln_ldfeiv=T` with
`nn_aei_ijk_t=21` — so the GM bolus lives in `tra_adv`, NOT in `ttrd_ldf`
(retraction 1 of the previous round).

## N1..N14 — ALIGNMENT TABLE

`traldf_iso.f90` statements in execution order vs legoESM's
`nemo_iso_lap_tracer_tendency_latlon_cgrid` /
`nemo_iso_a33` / `nemo_iso_w_kappa_sums`
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py`).

| # | NEMO (ppsrc `traldf_iso.f90`) | legoESM | verdict |
|---|---|---|---|
| N1 | `:155` `CALL traldf_iso_a33(Kmm, ah_wslp2, akz)` before the tracer loop | `nemo_iso_a33(...)` called inside the `msc_stabilize` block, same operands | MATCH (ordering immaterial: pure function of slopes/aht/mesh) |
| N2 | `:184,:193,:205,:213` `zdit/zdjt/zdkt` from `pt(...,Kbb)` x `umask/vmask/wmask`; `zdkt(level 1)=0` | `:2450-2453` `zdit/zdjt/zdkt`, `zdkt[...,0]=0` | MATCH |
| N3 | `:231-232` `zA11 = e2_e1u * (e3u_3d*(1+r3u*umask))`, `zA22` analogous | `:2470-2471` `(e2u/e1u)*e3u_flux` with `e3u_flux` = `nemo_qco_live_face_thicknesses` | MATCH (the qco live face thickness is already the certified operand) |
| N4 | `:234-237` `zmsku = 1/MAX((wmask(i+1,k)+wmask(i,k+1))+(wmask(i+1,k+1)+wmask(i,k)),1)` | `:2475-2478` same four terms, left-associated | MATCH (integer-valued sums, exact in fp64) |
| N5 | `:239-240` `zA13 = -e2u*uslp(k)*zmsku` | `:2481-2482` | MATCH |
| N6 | `:242-247` `zfu = ahtu*(zA11*zdit + zA13*((zdkt(i+1,ik)+zdkt(i,ikp1))+(zdkt(i+1,ikp1)+zdkt(i,ik))))` | `:2486-2492` `avg4_u` left-associated `((a+b)+c)+d` | **DIFF, 1-ulp class** — association only |
| N7 | `:267-270` `zmsku = wmask(jk)/MAX((umask(i,k)+umask(i-1,k+1))+(umask(i-1,k)+umask(i,k+1)),1)` — note the numerator is `wmask(jk)`, NOT `wmask(jk+1)` | `:2507-2515` `nemo_literal` arm reproduces the pairing and keeps `wmask` at `k` | MATCH |
| N8 | `:272-275` `zahu_w = ((ahtu(i,k)+ahtu(i-1,k+1))+(ahtu(i-1,k)+ahtu(i,k+1)))*zmsku`, `ahtu` masked at build (`ldftra.f90:433-434`) | `:2516-2524` `nemo_literal` arm, face-masked `aht` | MATCH |
| N9 | `:277-278` `zA31 = -zahu_w*e2t*zmsku*wslpi(jk+1)` (`zmsku` applied a SECOND time) | `:2538-2539` | MATCH |
| N10 | `:280-283` `zfw_kp1 = zA31*((zdit(i,ik)+zdit(i-1,ikp1))+(zdit(i-1,ik)+zdit(i,ikp1))) + zA32*(...)` | `:2545-2552` `nemo_literal` arm | MATCH |
| N11 | `:284-287` A33: `e1e2t / (e3w_3d(jk+1)*(1+r3t(Kmm))) * wmask(jk+1) * (ah_wslp2(jk+1)-akz(jk+1)) * (pt(jk,Kbb)-pt(jk+1,Kbb))` | `:2601-2610` builds `e3w_ab = 0.5*(e3t(k-1)+e3t(k))` and divides by `roll(e3w_ab,-1)`; `msc_e3w_override` exists but **no caller passes it** | **DIFF — candidate owner** |
| N12 | a33 `:69-72` `pakz = ((ahtu(i,k)+ahtu(i,k-1))/e1u(i)^2 + (ahtu(i-1,k)+ahtu(i-1,k-1))/e1u(i-1)^2 + ahtv/e2v^2 terms) * 0.25` | `nemo_iso_a33` `akz_h`, same per-face metric, x0.25 | MATCH (association 1-ulp) |
| N13 | a33 `:82-84` `ze3w_2 = (e3w_3d(jk)*(1+r3t(Kmm)))^2`; `akz = MAX(rDt*(pakz + ah_wslp2/ze3w_2) - 0.5, 0)*ze3w_2*r1_Dt` | `nemo_iso_a33(..., e3w2=e3w_ab**2, ...)` with the SAME midpoint `e3w_ab` | **DIFF — same operand as N11, second use** |
| N14 | `:288-292`, `:301-305` divergence `((zfu-zfu(i-1))+(zfv-zfv(j-1))+(zfw-zfw_kp1)) * r1_e1e2t / (e3t_3d*(1+r3t*tmask))` | `:2700-2707` `(hdiv+vdiv)*r1_e1e2t/e3t`, same association | MATCH |

## THE PREREGISTERED PREDICTION

**P1 — the first non-bit statement is N11/N13, the A33 (MSC) term's `e3w`.**

NEMO's `e3w_0(k) = gdept_0(k) - gdept_0(k-1)` (verified against the card's own
mesh: `max|e3w_0[1:] - diff(gdept_0)| = 0.0`), which on DINO's stretched ladder
is NOT the interface midpoint.  Measured, at rest, on the card's own ladder:

| k | `e3t` | NEMO `e3w_0` | legoESM `0.5*(e3t)` | rel |
|---|---|---|---|---|
| 13 | 26.2333 | 24.6632 | 24.7321 | +2.79e-3 |
| 18 | 52.8566 | 49.0219 | 49.1809 | +3.24e-3 |
| 24 | 133.9742 | 124.0678 | 124.4048 | +2.72e-3 |
| 35 | 617.4623 | 592.6383 | 581.3328 | -1.91e-2 |

rms over k>=1 = **4.62e-3**; legoESM's divisor is LARGER on 34 of 36 levels, so
its explicit A33 flux is SMALLER — the right SIGN for a 0.9966x tendency.  The
same operand enters `akz` squared, so the two uses do not simply cancel, and the
error is DEPTH-STRUCTURED — which is why T and S (different vertical gradient
profiles) give two different ratios.

legoESM already has ONE NEMO-faithful resolver for this object,
`nemo_e3w_kmm` (`physics/vertical_mixing/implicit_solver.py:762`), whose own
docstring states the defect: "`e3w_0(k) = gdept_0(k) - gdept_0(k-1)` — the
T-POINT DEPTH DIFFERENCE, which on a stretched ladder is NOT the interface
midpoint `0.5*(e3t_k + e3t_{k-1})`".  The tracer and momentum implicit solves
already call it; the Redi A33 term does not.

**P2 — the slopes are NOT the whole story.**  `ldf_slp` is recorded DEBT
(`wslpi` ratio 1.002818, `uslp` 1.002248).  Slopes that are ~0.28% HIGH push
the tendency UP, the opposite sign to the observed deficit.

## FALSIFIERS

* **F1** Substituting NEMO's `e3w_0*(1+r3t)` at N11 and N13 moves the best-fit
  ratio (measured with NEMO's own slopes fed in) by **less than 0.1%** ⇒ P1 is
  dead.
* **F2** With NEMO's own `uslp_stg/vslp_stg/wslpi_stg/wslpj_stg` fed in, the
  ratio is already within 0.001 of 1.0 BEFORE any substitution ⇒ the 0.34% was
  the SLOPE ROUTINE, the operator is exonerated, and P1 is dead.
* **F3** The ratio does not move on BOTH tracers in the same direction ⇒ the
  substitution is not acting through the mechanism claimed.

## THE MEASUREMENT

One-variable-at-a-time substitution ladder, each arm changing exactly ONE
operand, every arm scored against the record's `ttrd_ldf`/`strd_ldf` over all
wet cells, driven through **legoESM's own operator** (Rule 10):

| arm | what changes | reports |
|---|---|---|
| A0 | production operands (card slopes, midpoint `e3w`) | the 0.9966 baseline, reproduced |
| A1 | + NEMO's `uslp_stg/vslp_stg/wslpi_stg/wslpj_stg` | separates the slope routine from the operator |
| A2 | A1 + NEMO's `e3w_0*(1+r3t)` at N11/N13 | the preregistered fix |
| A3 | A1 + NEMO's `tb/sb` as the before state | closes the input side |

Plus a **per-stage table** decomposing legoESM's own tendency into the
horizontal-flux divergence, the vertical-skew divergence and the A33
divergence, so the residual removed by each arm is attributed to a stage and
not to the total.

Also this round: the **solar term** (`ttrd_qsr`), scored operator-to-operator
given NEMO's `qsr_hc_b`/`fraqsr_1lev`, to settle temperature's remaining
PLAUSIBLE explainer.

## PLANTS

* `--plant-e3w` — run arm A2 with the midpoint `e3w` labelled as NEMO's.  The
  arm MUST report no improvement; if it reports the fix, the ladder is scoring
  its own label.
* `--plant-slopes` — feed the card's slopes while claiming NEMO's.  A1 must
  then equal A0 exactly.
