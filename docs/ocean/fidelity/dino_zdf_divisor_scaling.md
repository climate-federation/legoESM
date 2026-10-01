# DINO twin — the implicit-mixing gradient DIVISOR, measured and scaled

Branch `fidelity/dino-zdf-divisor-scaling` from `9070cf276`. Preregistration:
`dino_zdf_divisor_scaling_preregister.md` (committed first, `11d0270c5`).
Probe: `scripts/validate/ocean_fidelity/dino_1226/dino_zdf_divisor_scaling.py`;
every number below is in `dino_zdf_divisor_scaling_artifact.json` next to this
file, with its commands, precision, ladder and NEMO donor paths.
All numbers fp64 (`PrecisionPolicy.fp64()`, control dtype and every geometry
array printed as `float64`), CPU, on the twin's production ladder
`LEGOESM_NEMO_E3T=both` — the ladder the 20-year producer ran
(`arms/m0.npz: nemo_ladder_mode = "both"`).

## Verdict

The certified DINO twin divides the implicit vertical-mixing gradient by a
thickness that is **systematically ~0.3% larger** than NEMO's, at every step of
the 20 years, with the error **smallest in the mixed layer and largest in the
abyss**. Its implicit vertical coupling `K/D` is therefore ~0.3% too weak in
the deep interior. One step of that costs 5e-4 to 9e-4 of the vertical-mixing
term, and over five days it grows 72x in temperature and 500-600x in the
velocities — with a contamination control of exactly zero, so all of it is the
divisor. Against the 20-year water-mass-census gap, only **0.2-0.3%** of that
measured divergence rate needs to persist to produce the whole gap; the
preregistered CONFIRM band was hit. So the divisor is **SCALE-COMPATIBLE** with
the census, MLD and variability families and **cannot be excluded**. It is NOT
an attribution: only a both-sided 20-year arm can supply that, and it was not
run.

## 1. The two divisors (Rule 0 — the oracle's own lines)

DINO's executing NEMO code, with `cpp_DINO.fcm: key_qco key_vco_3d`:

```
TRA/trazdf.F90:219-221            (no MY_SRC override -- the core routine runs)
   zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / e3w(ji,jj,jk  ,Kmm)
   zws(ji,jk) = - p2dt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)
   zwd(ji,jk) = e3t(ji,jj,jk,Kaa) - ( zwi(ji,jk) + zws(ji,jk) )

cfgs/DINO/MY_SRC/dynzdf.F90:200-203 and :209-210   (DINO DOES override dynzdf)
   zzwi = - zDt_2 * ( avm(ji+1,jj,jk) + avm(ji,jj,jk) )
        / ( e3u(ji,jj,jk,Kaa) * e3uw(ji,jj,jk,Kmm) ) * wumask(ji,jj,jk)

cfgs/DINO/MY_SRC/zgr_lib.F90:111-112   (zco branch: the momentum divisor IS e3w)
   pe3uw(:,:,:) = pe3w(:,:,:)
   pe3vw(:,:,:) = pe3w(:,:,:)
```

Time levels, from DINO's own step routine:

```
cfgs/DINO/MY_SRC/stpmlf.F90:551   CALL tra_zdf( kstp, Nbb, Nnn, Nrhs, ts, Naa )
cfgs/DINO/MY_SRC/stpmlf.F90:396   CALL dyn_zdf( kstp, Nbb, Nnn, Nrhs, uu, vv, Naa )
```

so the dummy `Kmm` binds to `Nnn` (NOW) and `Kaa` to `Naa` (AFTER). The macro
expansion, `DOM/domzgr_substitute.h90`:

```
:131   # define e3w(i,j,k,t)  (E3w_0(i,j,k) Time(r3t,i,j,t))
:49    # define Time(r3,i,j,t)   *(1._wp+r3(i,j,t))          [key_qco, NO mask]
:108   # define E3w_0(i,j,k)  e3w_3d(i,j,k)                   [key_vco_3d]
```

**NEMO's divisor:**

```
D_nemo(i,j,k) = e3w_0(i,j,k) * (1 + r3t(i,j,Nnn))            r3t = ssh / ht_0
```

and `e3w_0` is the **T-point depth difference**, `gdept_0(k) - gdept_0(k-1)`.
Verified against the oracle's own `mesh_mask.nc` (`RUN_20Y`): `e3w_1d[1:]` minus
`diff(gdept_1d)` is exactly `0.0`, while `e3w_1d` minus the interface midpoint
`0.5*(e3t_k + e3t_{k-1})` is 7.3e-4 rising to 3.2e-3. The two are not the same
object on a stretched grid.

**legoESM's divisor** — `ocean_model_latlon_cgrid.py:8255`, the `else` branch of
the divisor block inside `_apply_implicit_vertical_mixing` (defined at `:7706`),
reached from `_leapfrog_step` (`:9867`, calling the solve at `:10278`) because
the card resolves `outer_integrator='leapfrog'`. The path is proven to execute
dynamically as well: ablating this divisor moves the step's own output by
0.21 K (section 4, C4).

```
D_lego(i,j,k) = 0.5*(dz_k + dz_{k+1}) * (1 + eta_AFTER / H)
```

Resolved card, instantiated and printed (Rule 10):

| field | value |
|---|---|
| `implicit_vmix_dzw_slot` | `False` |
| `implicit_vmix_e3t_now_divisor` | `False` |
| `implicit_vertical_mixing` | `True` |
| `outer_integrator` | `'leapfrog'` |

confirming isomorphism-map row **S-34**: the certified NEMO card runs the
unreferenced legacy divisor while its own NEMO arm sits unselected.

### Where they differ

* **(a) the slot** — interface midpoint vs T-point depth difference. Present at
  zero ssh, at every level, permanently.
* **(b) the time level** — `eta_AFTER` vs `eta_NOW`. legoESM's Jacobian
  `(eta+H)/H` is identically NEMO's `(1 + r3t)`, so this is the only channel
  through which ssh enters, and it vanishes whenever the two levels agree.
* **(c) partial-cell bottoms** — **not in play**. DINO runs full-step
  `ln_zco_nam = .true.` (`RUN_20Y/namelist_cfg:70`); its `e3t_0`/`e3w_0` are
  horizontally uniform at every level (measured, per-level min == max); the
  twin's bridge is built with `full_step=True` (`kamm_twin_90d.py:1302`); and on
  the bridged production ladder `h_partial` equals `dz_ref` on all **342,134**
  active cells with max deviation `0.0`. Both models have square bottoms here,
  so no part of the gap below is a partial-cell effect.

Because the Jacobian cancels, the ratio `D_lego/D_nemo` is **purely geometric
and state-independent** — measured identical (to all printed digits) at the
day-180 restart, at member m0's day 360, and at its day 7200.

## 2. RETRACTION, recorded (Rule 11)

An earlier read of the **analytic** `masked_zco` ladder
(`dino_lat_lon_vertical` with `analytic_t_depths=True`) found
`z_coord.dz_half_ref` equal to NEMO's `e3w_1d[1:]` **exactly** (rel 0.0), which
would have made the shipped `implicit_vmix_dzw_slot` flag the NEMO arm.

**That does not hold on the ladder the twin runs.** With
`LEGOESM_NEMO_E3T=both` the bridge builds `z_full_ref` as cell-centre midpoints
of NEMO's `e3t_0`, so `dz_half_ref` equals `build_dz_half(dz_ref)` to
`8.9e-16` — the midpoint. On that ladder:

* `implicit_vmix_dzw_slot=True` is a measured **NO-OP** (first-step
  `max|dT| = 1.75e-11 K`), and
* `implicit_vmix_e3t_now_divisor=True` would fix only the time level (b), never
  the slot (a).

**Neither shipped flag supplies NEMO's `e3w` on the twin's grid.** The probe
therefore reads `e3w_0` from the oracle's `mesh_mask.nc` rather than
reconstructing it, and checks it is horizontally uniform rather than assuming
it.

**Side-finding, one line, not pursued.** The shipped "NEMO" arm
`implicit_vmix_e3t_now_divisor=True` builds `build_dz_half(e3t_now)` — the NOW
thickness at the **midpoint** slot. On a *uniform* vertical grid the midpoint
and the T-point depth difference coincide, so that arm is NEMO-exact there; on a
*stretched* ladder it is not. The cards that select it today (LOCK_EXCHANGE,
OVERFLOW, GYRE per `nemo_branch_isomorphism_map.md:40`) were not checked here.

Also noted and **not** pursued (out of scope, already owned by #1455): NEMO's
3-D `e3t_0`/`e3w_0` diverge from its own 1-D `e3t_1d`/`e3w_1d` below k=25, by
up to 15% at the deepest wet level. The `LEGOESM_NEMO_E3T=both` default already
puts the twin on the 3-D ladder; only the `"off"` ladder would carry that error.

## 3. The divisor difference, measured

332,214 wet interfaces (both neighbouring T cells wet), day-180 restart, fp64:

| statistic | `D_lego / D_nemo - 1` |
|---|---|
| max | +8.965e-3 |
| median | +2.935e-3 |
| mean | +2.395e-3 |

Identical at m0 day 360 and m0 day 7200 (the ratio is state-independent).

Depth structure — the sign is **positive almost everywhere**, i.e. legoESM's
divisor is too large and its vertical coupling too weak:

| interface k | approx. depth | median `D_lego/D_nemo - 1` |
|---|---|---|
| 0 | 10 m | +7.27e-4 |
| 4 | 49 m | +1.31e-3 |
| 8 | 103 m | +2.07e-3 |
| 12 | 178 m | +2.79e-3 |
| 17-18 | 342-391 m | +3.24e-3 (upper-ocean maximum) |
| 24 | 915 m | **-8.96e-3** (sign flip, global maximum) |
| 25 | 1059 m | +7.68e-3 |
| 28 | 1608 m | +3.24e-3 |
| 33 | 3223 m | +3.85e-3 |

**Structural answer to "is it where the divergence was?"** It is **not**
concentrated in the mixed layer — the upper 100 m carries the *smallest* error
(7e-4 to 2e-3). It is concentrated in the **deep interior**: a sign-flipping
pair at 900-1060 m and a steady +2.9e-3 to +3.9e-3 through the abyss. That is
exactly the region the water-mass-census family's decisive statistic occupies
(`abyss_ge1400m`), and the opposite of where the MLD family lives.

## 4. Instrument calibration (Rule 3, run before any residual was quoted)

| check | measured | required |
|---|---|---|
| C1 — repeat the same arm | `0.0` exactly | exactly 0 |
| C1b — slot flag, unchanged ladder | `1.75e-11 K` | no-op |
| C2 — divisor ratio under a +1 m ssh bump | `4.44e-16` | ~0 (Jacobian cancels) |
| C3 — planted uniform +1% divisor, / ablation | `0.0073` | ~0.01 |
| C3 contamination control | `0.0` exactly | exactly 0 |
| C4 — total ablation of implicit vmix (`D` x 1e6) | `0.2097 K` | first order |
| C4 contamination control | `0.0` exactly | exactly 0 |

C3 lands at 0.73x the linear prediction — the backward-Euler solve is slightly
sub-linear in `1/D`, as expected. C4 is the denominator used below: it is a
**total ablation**, not an assumption.

**Every arm ships a contamination control** — the same modified coordinate with
the divisor flag OFF. All of them measured **exactly 0.0** on T, S, u and v, so
`dz_half_ref` is read nowhere else on this card and the arm difference is 100%
divisor.

## 5. One-step tendency difference

Executing arm vs NEMO-divisor arm, one full production step (`dt = 2700 s`),
identical state, identical `K_v`/`A_v`, one variable:

**Day-180 bridged NEMO restart** (a self-consistent leapfrog state):

| field | max abs arm diff | rms | (i) / vmix term | (ii) / total step |
|---|---|---|---|---|
| T | 1.129e-4 K | 1.285e-6 | 5.39e-4 | 5.84e-4 |
| S | 4.911e-6 g/kg | 6.805e-8 | 6.74e-4 | 4.38e-4 |
| u | 3.513e-5 m/s | 1.558e-6 | 3.83e-4 | 6.28e-4 |
| v | 1.758e-5 m/s | 7.843e-7 | 8.64e-4 | 9.43e-4 |

**Member m0's 20-year end-states** (`arms/m0.npz`, days 360 and 7200):

| state | T (i) | S (i) | u (i) | v (i) |
|---|---|---|---|---|
| m0 day 360 | 1.38e-4 | 1.38e-4 | 2.77e-4 | 1.42e-4 |
| m0 day 7200 | 1.78e-4 | 1.21e-4 | 2.67e-4 | 4.89e-4 |

*Caveat on these two rows:* only the AFTER-level `T/S/u/v/eta` were substituted
from the archive; the leapfrog BEFORE level and the TKE carry are still the
restart's, so the absolute increments are inflated by a start-up shock. Use the
RATIOS and the state-independence of the divisor field, not the absolutes.

## 5b. The 5-day integrated arm — preregistered prediction CONFIRMED

Two arms, 160 steps (5 days at `dt = 2700 s`) from the day-180 bridged NEMO
restart, identical in every respect except the implicit-solve divisor. Both
stayed finite; 161-165 s of integration each.

| field | day-5 max abs | day-5 rms | growth vs one step (max) |
|---|---|---|---|
| T | 8.127e-3 K | 5.291e-5 | **72x** |
| S | 1.017e-3 g/kg | 4.522e-6 | 207x |
| u | 2.193e-2 m/s | 1.081e-4 | 624x |
| v | 9.151e-3 m/s | 5.492e-5 | 521x |

**Contamination control at day 5: exactly `0.0` on T, S, u and v.** The entire
5-day divergence is the divisor, nothing else.

Against the preregistration (`dino_zdf_divisor_scaling_preregister.md`,
committed at `11d0270c5` before this ran):

* CONFIRM band was `max|dT|` in `[1e-3, 2e-2] K` — **measured 8.13e-3 K, inside
  the band**.
* REFUTE condition was `< 3e-4 K` — **not met**.

**Verdict: CONFIRMED live and growing.** Coherent linear accumulation over 160
steps would give 160x; pure saturation would give 1x. T grew 72x and the
velocities 500-600x, so the difference is amplifying dynamically rather than
being absorbed locally by the backward-Euler solve.

Worth stating plainly: after **five days**, the divisor alone has produced local
salinity differences of 1.0e-3 g/kg — roughly **100 times the entire 20-year
abyssal census gap** (1.09e-5 g/kg) the equivalence test refuted on.

## 6. Scaling against the 20-year divergence

DINO's 20 model years = 7200 d = 6.2208e8 s = 2.304e5 steps at `dt = 2700 s`.
"Required rate" is the family's gap spread evenly over that window; "available
rate" is the measured one-step divisor signal divided by `dt`; **f** is the
fraction of the divisor signal that must rectify to produce the gap.

| family | verdict at 20 y | decisive statistic | gap | required rate | available rate (rms / max) | f (rms) | scaling verdict |
|---|---|---|---|---|---|---|---|
| ts_water_mass_census | REFUTE, R=104 | `north of band.abyss_ge1400m.S_mean` | 1.092e-5 g/kg | 1.755e-14 g/kg/s | 2.520e-11 / 1.819e-9 | **7.0e-4** | **SCALE-COMPATIBLE** |
| ts_water_mass_census (T) | R=21 | `north of band.abyss_ge1400m.T_mean` | 2.281e-4 K | 3.667e-13 K/s | 4.758e-10 / 4.183e-8 | **7.7e-4** | **SCALE-COMPATIBLE** |
| mld_seasonal_cycle | REFUTE, R=3.0 | `mld.north of band.month05` | 0.0316 m on 86.0 m | fractional 3.68e-4 | fractional mixing change 1.0e-3 to 2.1e-3 in the upper 100 m | ~0.2-0.4 | **SCALE-COMPATIBLE, PLAUSIBLE only** |
| basin_row_transports | REFUTE, R=86 | `row.190.mean` | 4.71e-3 Sv on 0.367 Sv | fractional 1.29e-2 | day-5 rms u divergence 1.08e-4 m/s | — | **UNRESOLVED** (no transport sensitivity measured) |
| variability | REFUTE, R=9.6 | abyssal `S_mean.deseasonalized_std` | 7.72e-7 on 5.11e-5 | fractional 1.51e-2 | day-5 rms S divergence 4.52e-6 g/kg, 5.9x the gap | — | **UNRESOLVED** (a variance statistic has no linear translation) |
| acc_series | UNRESOLVED, R=0.90 | `acc.full.month09` | 0.451 Sv | — | — | — | not distinguishable at 20 y |
| density_contrasts | UNRESOLVED, R=1.34 | `density.deep.month04` | 7.11e-4 | — | — | — | not distinguishable at 20 y |

Arithmetic for the headline row, written out:

```
gap             = 1.09204e-5 g/kg            (lego 35.122680 vs nemo 35.122691)
20 model years  = 20 * 360 * 86400 s = 6.2208e8 s
required rate   = 1.09204e-5 / 6.2208e8      = 1.7554e-14 g/kg/s
                                             = 4.740e-11 g/kg per 2700 s step
available (rms) = 6.805e-8 / 2700            = 2.520e-11 g/kg/s
f               = 1.7554e-14 / 2.520e-11     = 6.96e-4
```

so **0.07% of the divisor's own per-step salinity signal, rectified, produces
the entire 20-year census gap**. On the max-abs signal the requirement falls to
1e-5. The same arithmetic on abyssal temperature gives f = 7.7e-4.

**The same scaling anchored on the 5-day arm instead**, which is stronger
because it contains the actual dynamical growth rather than one isolated step:

```
                        salinity                    temperature
day-5 rms difference    4.5224e-6 g/kg              5.2907e-5 K
over 5 d = 4.32e5 s     1.0468e-11 g/kg/s           1.2247e-10 K/s
required rate (above)   1.7554e-14 g/kg/s           3.6667e-13 K/s
f                       1.68e-3                     2.99e-3
```

i.e. **0.2-0.3% of the measured 5-day divergence rate, sustained, is the whole
20-year census gap.** Both anchors agree to within a factor of ~4 and both sit
2.5 to 3 orders of magnitude above the requirement.

**MLD, the honest version.** The divisor weakens the vertical mixing coupling by
1.0e-3 to 2.1e-3 in the upper 100 m; the MLD gap is 3.68e-4 of the MLD. That
needs a sensitivity `dlnMLD / dln(K/D)` of roughly 0.2-0.4, which is entirely
ordinary but is **not measured here**. SCALE-COMPATIBLE, labelled **PLAUSIBLE**.

**Transports and variability** are marked UNRESOLVED deliberately: a per-step
velocity perturbation has no defensible linear translation into a 20-year mean
transport, and a variance statistic has none at all from a mean-tendency bias.
Magnitude does not exclude them; it also does not implicate them.

## 7. What this does NOT establish

* **This is not an attribution.** Rule 4 requires removing the term from BOTH
  models and asking whether the gap survives. The 20-year arm was not run
  (explicitly out of scope).
* Scale compatibility at f ~ 7e-4 is a weak bar. Many other 0.3%-level
  differences would clear it. It says the divisor **cannot be excluded**, and it
  ranks it as worth a real arm — not that it is the cause.
* The rectified fraction `f` assumes the per-step signal has a persistent mean
  component. It is a screen, not a mechanism.

## 8. Choices

| choice | status |
|---|---|
| run the scaling study, preregister the arm, short gate under 30 min CPU, no long integration | ASKED (task brief) |
| read `e3w_0` from the oracle mesh instead of reconstructing it from `z_full_ref` | forced by the retraction in section 2 — no alternative was correct |
| select the NEMO divisor via `implicit_vmix_dzw_slot=True` PLUS a coordinate whose `dz_half_ref` is NEMO `e3w_0` | forced — the flag alone is a measured no-op on this ladder |
| use a total ablation (`D` x 1e6) as the vertical-mixing denominator | ASKED in spirit (Rule 3 calibration); no config default touched |

**UNASKED list: empty.** No default, scheme selection, tunable, threshold or
config field is changed by this branch. The arms exist only inside the probe.
The open decision this raises — whether the DINO card should select the NEMO
divisor, and whether legoESM should ship an arm that is NEMO's `e3w` on the
twin's ladder at all (today none is) — is left for the user.
