# PHASE 1 — barotropic transport-pathway alignment table (dyn_spg_ts)

Oracle: `cfgs/DINO/MY_SRC/dynspg_ts.F90` (DINO runs **MLF**, `#else` branch —
`cpp_DINO.fcm` has `key_qco key_vco_3d`, NO `key_RK3`; DINO ships `MY_SRC/stpmlf.F90`).
Namelist (RUN_GDB/ocean.output): `ln_dynspg_ts=T`, `nn_bt_flt=2` (boxcar width `2*nn_e`),
`nn_e=23`, `ln_bt_fw=F` (centred), `ln_dynadv_vec` → vector form.

legoESM: `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py`
(kamm/DINO recipe: `barotropic_solver="explicit_substep"`,
`barotropic_face_depth="nemo_ssh_avg"`, nemo-boxcar-centred weights — confirmed in
arm1_pre.log: `barotropic_face_depth=nemo_ssh_avg`).

## Discriminator result (CONFIRMED — controls C1 reproduces gate 1.557/1.708, C3 day-0 bit-identical)
`acc_arm_diff_decomp.py` + `btbc.py`:
- The +0.1514 Sv arm2−arm1 ACC degradation is **100.0% BAROTROPIC** (depth-mean).
  Baroclinic (shear) part = +0.0000 Sv to 4 dp. Buoyancy/thermal-wind pathway EXONERATED.
- Latitude: entirely in the re-entrant channel band (−0.1715 Sv); south +0.018, north +0.0004.
- Longitude: near-uniform across all 52 lons (median −0.140, mean −0.149, **std 0.021**) —
  NOT localized at any sill/topography. Task criterion: uniform ⇒ transport-pathway, not sill.

## Ordered alignment table — substep loop, un_adv assembly, 3-D reconciliation

| # | NEMO dynspg_ts.F90 (DINO MY_SRC) | legoESM barotropic_latlon_cgrid.py | verdict |
|---|---|---|---|
| N1 | boxcar `zwgt1` (primary) = width `2*nn_e`, symmetric, `ts_wgt:1269-1276` | `compute_nemo_boxcar_centred_weights` → `w_avg` | MATCH (both sym boxcar, 45 nonzero, centroid substep 22.0) |
| N2 | secondary `zwgt2(jn)=Σ_{ji≥jn} zwgt1(ji)` (tail sums), `ts_wgt:1288-1292` | `w_transport[j]=Σ_{i≥j} w_i / n`, docstring `barotropic_common.py:116` | MATCH (both triangular tail-sum, centroid substep 14.67) |
| N3 | `un_adv += wgtbtp2(jn)·zhU·r1_e2u`, `zhU=e2u·ua_e·zhup2_e` (mid-step jn+½), `:734-737` | `Hu_sum += w_tr_i·flux_u`, `flux_u=H_u_flux·U_mid` (mid-step AB3), `:825-829` | MATCH (transport, secondary weights, mid-step extrapolated velocity+depth) |
| N4 | `un_adv /= r1_wgt2s` finalize, `:999` | `Hu_avg = Hu_sum_f` (w_transport pre-normalized), `:1428` | MATCH |
| N5 | `puu_b(Kaa) += wgtbtp1(jn)·ua_e` (primary), `:977-979` | `U_sum += w_i·U_bar_new`; `U_bar_avg=U_sum_f/w_total`, `:991,1443` | MATCH (velocity, primary weights) |
| **N6** | **`puu(Kmm) = (puu(Kmm) + un_adv·r1_hu(Kmm) − puu_b(Kmm))·umask`, `:1170-1172`** | **`u_new = (u_corr − U_bar_corr) + U_bar_avg`, `:1460-1462`** | **DIFF** |

## The DIFF (N6) — CONFIRMED at statement level

NEMO's momentum reconciliation sets the 3-D velocity's depth-mean to
**`un_adv/hu`** — the **secondary-weight (transport, `wgtbtp2`) time-average**
(comment `:1170`: *"Correct velocities so that the barotropic velocity equals
(un_adv, vn_adv)"*). `un_adv` is legoESM's `Hu_avg` exactly (N3/N4 MATCH).

legoESM's momentum reconciliation (`:1462`) adds **`U_bar_avg`** — the
**primary-weight (velocity, boxcar `w_avg`) time-average** (`U_sum_f/w_total`).

`Hu_avg` (the secondary/transport mean = NEMO's `un_adv`) IS computed in the
same call but is handed ONLY to tracer advection (`ocean_model_latlon_cgrid.py:4227`,
`delta_U=(Hu_avg−Hu_3d)/H_u_old`). It NEVER reaches the 3-D momentum / ACC velocity.
There is **no config option** to switch the momentum reconciliation to the transport
mean (grep of barotropic_latlon_cgrid.py + config.py = none).

### Magnitude / mechanism (measured kernel separation; mechanism labelled PLAUSIBLE)
The two averaging kernels are materially different (fp64, DINO nn_e=23):
- primary/velocity boxcar: **centroid substep 22.0** (symmetric)
- secondary/transport tail-sum: **centroid substep 14.67** (triangular ramp, ~7.3 substeps earlier)

Both sum to 1.0. Because they sample the substep-velocity time profile at different
effective phases, any change to the momentum RHS (`F_slow_u`) — which the 7 faithful
operator fixes produce — shifts `U_bar_avg` and `Hu_avg/H` by DIFFERENT amounts. Only
`U_bar_avg` reaches the ACC velocity, while NEMO would use `un_adv/H`. This is the
Rule-8 compensating defect: the less-faithful RHS happened to make the wrong-kernel
`U_bar_avg` land closer to NEMO's `un_adv`-driven depth-mean; the faithful fixes remove
that accidental cancellation, so the (never-corrected) wrong-kernel choice surfaces as a
uniform, 100%-barotropic, band-confined −0.15 Sv ACC shift — the exact measured signature.

## Also examined (NOT the arm-diff driver, recorded for coverage)
- **`mlf_baro_corr` / `_impose_mean`** (`ocean_model_latlon_cgrid.py:5185-5200`): re-imposes
  the PRE-implicit-vmix depth mean after the implicit solve. It PRESERVES `U_bar_avg`, it
  does not inject `un_adv`; its comment cites RK3 `stprk3_stg:440` but DINO is MLF. It is
  downstream of N6 and cannot substitute the transport mean — so it does not fix the DIFF.
- **Recipe integrator mismatch (separate, larger-scope):** kamm sets
  `momentum_time_integrator="rk3_ws"` + comments citing RK3 `stprk3_stg`, but DINO runs MLF
  (`dynspg_ts:1170`). Flagged, not the localized barotropic-reconcile term.

## STOP — localized term
- **term:** 3-D momentum depth-mean reconciliation uses the primary/velocity boxcar mean
  instead of the secondary/transport mean (NEMO `un_adv`).
- **NEMO:** `cfgs/DINO/MY_SRC/dynspg_ts.F90:1170-1172` (`un_adv·r1_hu − puu_b`).
- **legoESM:** `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:1460-1462`
  (`(u_corr − U_bar_corr) + U_bar_avg`); the correct quantity `Hu_avg` (=`un_adv`) is at
  `:1428` but routed only to tracer advection.
- **expected magnitude:** kernel centroids differ by ~7.3 substeps; produces the measured
  uniform 100%-barotropic band-confined −0.15 Sv arm-diff (CONFIRMED barotropic;
  attribution-to-this-term PLAUSIBLE pending the both-sided confirmation experiment below).

## CONFIRM experiment RESULT (2026-08-13) — DIAGNOSIS REFUTED

Controlled A/B at the arm2 state (all 7 fixes), ONE variable
(`barotropic_reconcile_target`), fp64, bridged 90-day twin, day-0 bit-identical
(max|Δu|=0). Harness self-checks PASS (NEMO y10 ACC 121.07=recorded; band vol
rel 5e-8). Probe: `reconcile_confirm.py` (reuses `acceptance_gate_90d` gate +
`acc_arm_diff_decomp` bt/bc split verbatim; instrument self-checked on identical
arms → null).

| arm (HEAD cbd0ac504)            | ACC [Sv] | \|diff vs NEMO d90\| |
|---------------------------------|----------|----------------------|
| velocity_avg (legacy default)   | 63.6931  | **1.6761**           |
| transport_avg (NEMO-faithful)   | 63.6450  | **1.7242**           |
| arm1_pre (0 fixes, reference)   | 63.8122  | 1.5570               |
| NEMO d90 (reference)            | 65.3692  | —                    |

Reduction: `acc_full` = median over lons 2..-2 of the e3t1d-weighted full-section
zonal transport [Sv]. Reference: NEMO DINO day-90 twin (tn/sn/un).

**ACC does NOT recover — it worsens by −0.048 Sv** (prediction: recover toward
≤1.557). The predicted SIGNATURE is exactly right — the shift is **100.0%
barotropic** (baroclinic +0.0000), longitude **uniform** (std 0.0125,
median −0.047) — but the sign is WRONG and the magnitude (−0.048 Sv) is ~⅓ of
the −0.15 Sv arm2−arm1 degradation it was meant to explain.

Density metrics essentially unchanged (up/deep/smax worse by 1e-5..5e-5, smean
better 3e-7 — all ≪ the ~2e-3 |diff|; no degradation, no improvement).

**RETRACTION (Rule 11):** the PLAUSIBLE attribution of the arm2−arm1 ACC
degradation to the N6 velocity-vs-transport reconcile choice is REFUTED. The
term IS a real, uniform, 100%-barotropic ACC lever (as diagnosed), but (a) it
does NOT own the degradation and (b) NEMO's faithful choice moves ACC the WRONG
way at arm2. The compensating-pair sub-hypothesis (pre-fix+transport should be
worse than pre-fix+velocity) is ALSO contradicted a priori: for a compensating
pair, transport_avg would have to be BETTER than velocity_avg at arm2 — it is
worse — so the asymmetry leg was skipped as foregone (compute discipline).

Note: my velocity_avg reproduction (1.676) differs from the committed arm2
(1.708) by 0.032 Sv because HEAD (cbd0ac504) carries OTHER branch changes since
the arm2 commit — LARGER than this term's 0.048 Sv effect. Comparison vs the
committed 1.708 is confounded; the A/B pair (both at HEAD, one variable) is not.

**The option is KEPT regardless** — it is Rule-0-confirmed faithful to
`dynspg_ts.F90:1170` (r1_hu(Kmm) = NOW thickness, matches). It is NOT tuned
toward any number. Whether to leave it selected on the card is a
faithfulness-vs-metric decision for the reviewer (Rule 8: a faithful-but-worse
term is a SIGNAL that a compensating defect lives elsewhere, not a reason to
revert; but here the effect is small and the true owner of the −0.15 Sv arm-diff
is still open — search returns to N6's subtract-side / the other 6 fixes).

## Suggested CONFIRM experiment (single variable, not yet run)
Swap the momentum reconciliation add-term from `U_bar_avg` to `Hu_avg/H_u` (transport mean)
as a selectable option, hold everything else byte-identical, re-run the 90-day arm2 gate.
Predicted: ACC |diff vs NEMO d90| drops toward the arm1 value AND the faithful fixes then
IMPROVE rather than degrade ACC (Rule 8 satisfied). If ACC does not move, this term is
exonerated and the search returns to N6's subtract-side (`U_bar_corr` vs NEMO `puu_b(Kmm)`).

---

# PHASE 2 (2026-08-13) — the 0.88 m²/s Hu_avg wall bias is the F_slow forcing residual, NOT a loop-internal defect. CONFIRMED.

## Instrument
`substep_traj_compare.py` — captures legoESM's FULL per-substep barotropic
trajectory (teeing `fori_loop`→`scan` under `jax.disable_jit`, fp64,
`LEGOESM_NEMO_E3T=both`, day-0 bit-identical) and compares it, substep by
substep, against NEMO's `substep_dump.bin` (dynspg_ts.F90:926-940, the FULL
per-jn=1..68 dump of `sshn_e/ssha_e/zsshp2_e/un_e/vn_e/ua_e/va_e`, kt=nit000=
230401 = the exact replayed step). Registered `substep_dump.bin` in
`time_levels.py` ("before", cited to :570 seed proof).

## Substep-count reconcile (Rule 1e — retracts a probe error, not a lego DIFF)
NEMO `ts_wgt` CASE(2) with `nn_e=23, ln_bt_fw=F`: `jic=2·nn_e=46`, boxcar
`|jn-46|/23<1` ⇒ nonzero primary weights jn=24..68 (45 of them, the table's
"45"), but `Kpit=icycle=68` (substeps 1..23 execute with zero primary weight —
AB3 spin-up). **legoESM runs n_loop=68 too** — the MLF path passes
`_barotropic_substep_scale=2` ⇒ `compute_nemo_boxcar_centred_weights(46, scale=2)`
⇒ `m_star=68`. My first `_compute_weights(...,23,scale=1)` returned 45; that was
MY probe error (wrong scale), RETRACTED. Confirmed: captured n_loop=68 == NEMO
icycle=68; `wgtbtp2` 68 nonzero, sum 1.0 both sides.

## Trajectory result (the bias is BORN in the loop, grows ~linearly)
| jn | cum Hu err wall | cum Hu err interior |
|----|-----------------|---------------------|
| 1  | 3.1e-15 (bit-identical) | 3.1e-15 |
| 8  | 1.7e-2 | 1.6e-2 |
| 24 | 1.7e-1 | 1.4e-1 |
| 48 | 6.5e-1 | 4.5e-1 |
| 68 | **8.7926e-1** | **5.8885e-1** |

**Instrument validation (Rule 3, calibrate before trusting):** the probe
reconstructs NEMO's `zhU/e2u = ua_e_mid·zhup2_e` from the dumped `un_e`/`sshn_e`
history (AB3 coeffs + `zhup2_e` formula), sums it with `wgtbtp2` over all 68
substeps, and checks it against NEMO's OWN `spg_dump_un_adv_final.bin`:
**rel 7.287e-16 (roundoff) — RECON VALID.** So comparing lego `Hu_avg` against
the reconstruction is bit-identical to comparing against the un_adv dump; no
reconstruction bug can hide the finding. (Probe raises if this exceeds 1e-10.)

Final wall 0.87926 / interior 0.58885 m²/s — **reproduces `hu_avg_perface_diff`
(0.879 / 0.589) exactly**, across the jit boundary (per-face probe jitted, this
probe under `disable_jit` for the fori→scan tee) — the agreement is the
jit-parity control. Substep 1 is bit-identical; the bias accumulates
monotonically. Wall and interior grow TOGETHER — the wall is where it is
LARGEST (max transport at the barrier seam), not where the per-substep defect
lives.

## Decomposition — velocity, not depth; and the velocity error IS the forcing
- **Velocity trajectory** `U_bar[jn]` vs NEMO `un_e[jn+1]`: wall err 1.1e-5
  (jn=1) → 1.3e-3 (jn=67), ~1% of |un|=0.12 m/s. Grows every substep.
- **Flux ratio** lego/NEMO: 0.9996 (jn=2) → **0.9948 (jn=68)** — a steadily
  growing ~0.5% flux DEFICIT (sign-coherent, not noise).
- **jn=1 END velocity** `U_bar[0]` vs NEMO `ua_e[1]`: max|Δ| **1.1151e-5**,
  which is **3× NEMO's ENTIRE jn=1 increment** (|ua−un|=3.7e-6). So legoESM's
  substep-1 update already differs materially. Error per-col top: 50,51 (E
  seam), 3,4,5,6 (W) — wall-emphasised but domain-wide, near-uniform.

## THE CAUSE (CONFIRMED to 4 significant figures)
`F_slow_u` (legoESM's assembled zu_frc-equivalent, captured at the
`barotropic_substeps_latlon_cgrid` call) vs NEMO's dumped `zu_frc`
(`spg_dump_zu_frc.bin`):

    max|F_slow_u − zu_frc| = 9.4915e-8   median = 1.6969e-8   [m/s²]
    |zu_frc| max = 3.03e-5   median = 6.86e-7
    relative |ΔF|/|zu_frc|:  median 2.0e-2 (2%)   p99 9.4e-1

With `dt_s = 117.39 s`:

    dt_s · max|ΔF|    = 1.1142e-5   ==  jn=1 vel err max    1.1151e-5   ✓
    dt_s · median|ΔF| = 1.9920e-6   ==  jn=1 vel err median 1.9912e-6   ✓

**The per-substep barotropic velocity error is `dt_s·(F_slow_u − zu_frc)` to 4
s.f.** `dt_s = 117.39 s = 2·2700/46` (the MLF-scaled substep length, read off
the loop's own `dt_s` arg, not assumed).

### Non-circularity control (proves the loop DYNAMICS match, not just the forcing)
`verr` (jn=1 vel err = `U_bar[0] − ua_e[1]`, from the carry trajectory) and `ΔF`
(`F_slow_u − zu_frc`, from the loop's call arg) are captured by DIFFERENT hooks
in the same run — independent measurements. Their residual:

    RESIDUAL = verr − dt_s·ΔF :  max 2.3446e-7   median 4.318e-10   (2.1% / 0.02% of verr)

Since at jn=1 the seed velocity is bit-identical to NEMO's `un_e`, the update
`U_bar_new = seed + dt_s·(cor + drag − g·∇η + F_slow_u)` gives
`verr = dt_s·ΔF + dt_s·(Δcor + Δdrag − gΔ∇η)`. The residual being ~2e-7
(≪ the 1.1e-5 forcing term, and non-accumulating) PROVES the Coriolis
(een_metric), PGF and drag all match NEMO to ~2e-7 at jn=1 — the finding is NOT
tautological: it independently establishes that the loop dynamics are faithful
and ONLY the forcing input is biased.

ΔF's spatial structure (per-col top 50,51,4,5,6,3) is IDENTICAL to the
jn=1 velocity-error structure. ΔF is sign-coherent (mean −9.5e-9, |mean|/rms
0.425) ⇒ it accumulates into the net 0.5% flux deficit over 68 substeps, and
the flux integral concentrates the bias at the seam wall (where |Hu| is
largest). ΔF itself is near-uniform (wall max 9.49e-8 vs interior 9.08e-8) —
the wall concentration of the *transport* bias is a `·H·U` weighting effect, not
a wall-localised forcing error.

## STOP — statement-level DIFF + measured share of the bias
- **The barotropic substep loop is FAITHFUL.** Coriolis (een_metric), PGF
  (−g·∇η), the AB3 velocity extrapolation, `zhup2_e` mid-step flux depth, the
  boxcar/AB3 weights, the substep count (68) — all reproduce NEMO substep-1
  bit-identically and carry no per-substep defect of their own.
- **The DIFF is UPSTREAM of the loop**: `F_slow_u`, legoESM's assembled slow
  forcing (`zu_frc`-equivalent, baroclinic PGF + viscosity + advection +
  drag depth-mean), differs from NEMO's `zu_frc` by a sign-coherent
  ~2%-median (up to 94% p99) residual.
- **Measured share of the bias: 100%.** `dt_s·ΔF` reproduces the per-substep
  velocity error to 4 s.f. at jn=1; the 68-substep accumulation reproduces the
  final 0.88 m²/s wall / 0.59 interior transport bias, whose `−2dt·div` is the
  4.2e-3 m/step eta injection (`hu_avg_perface_diff` closure, 3 s.f.).

This is the "separately owned input residual" the earlier phase flagged and
exonerated the LOOP for — now CLOSED to the transport bias: the loop faithfully
integrates a biased forcing input. The residual is NOT in `dynspg_ts.F90`'s
substep dynamics; it is in the assembly of the slow-mode forcing that feeds it
(NEMO `zu_frc`, built in `dyn_spg_ts` :300-508 from the now 3-D RHS + the
dyn_cor_2D/wind/drag adds). **Next stage (a SEPARATE reviewed change, per the
STOP rule): trace the `F_slow_u` assembly term-by-term against NEMO's `zu_frc`
build — the residual is sign-coherent and ~2%, so one of its constituent adds
(baroclinic PGF depth-mean, the +bt Coriolis/wind/drag `zu_frc` corrections at
dynspg_ts.F90:302-508, or the depth-weighting) is systematically off by ~2%.**

## Retraction ledger (Rule 11)
- "45 vs 68 substep-count DIFF" — RETRACTED same session: my `_compute_weights`
  call used `substep_scale=1`; the live MLF path uses 2 ⇒ 68. No lego DIFF.
- The PLANT (NEMO trajectory shifted +1 substep) is WEAK (0.747 vs 0.879, not
  >3×) because the CUMULATIVE metric is dominated by the 68-substep integral; a
  1-substep shift barely moves it. This is a known limitation of a cumulative
  planted control, NOT a failure — the decisive controls are (a) substep-1
  bit-identity (3.1e-15), (b) exact reproduction of the independently-measured
  final 0.879/0.589, and (c) the 4-s.f. `dt_s·ΔF` == velocity-error identity.
