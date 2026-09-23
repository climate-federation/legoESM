# Preregistration — is the DINO twin's implicit-mixing DIVISOR part of the 20-year gap?

Branch `fidelity/dino-zdf-divisor-scaling`, cut from `9070cf276`. Written and
committed **before** the confirming arm is run (Rule 3 / the "preregister the
prediction" discipline). Numbers already in hand at writing time are the
one-step measurements from
`scripts/validate/ocean_fidelity/dino_1226/dino_zdf_divisor_scaling.py`
(day-180 restart state, fp64, CPU); the arm this document preregisters is the
integrated one.

## The claim under test

`docs/ocean/fidelity/nemo_branch_isomorphism_map.md` row **S-34** says the
certified DINO twin runs a non-NEMO gradient divisor in the backward-Euler
vertical-mixing solve, and flags it as a hidden-default defect. The 20-year
climate-equivalence run
(`dino_multi_year_climate_equivalence_result.md`) found legoESM and NEMO
DISTINGUISHABLE in four of six families. **Question: is the divisor big enough
to be part of that?**

## The two divisors, as formulas

NEMO, `TRA/trazdf.F90:219-221` (and `DYN/dynzdf.F90:182-185` for momentum),
called from `stpmlf.F90:370` / `:267` so `Kmm -> Nnn` (NOW) and `Kaa -> Naa`
(AFTER), with `key_qco key_vco_3d` expanding
`DOM/domzgr_substitute.h90:131,108,49`:

```
D_nemo(i,j,k) = e3w_0(i,j,k) * (1 + r3t(i,j,Nnn))      r3t = ssh/ht_0
e3w_0(k)      = gdept_0(k) - gdept_0(k-1)              [T-point depth difference]
```

legoESM's executing arm, `ocean_model_latlon_cgrid.py` divisor block (`else`
branch), with both `implicit_vmix_dzw_slot` and `implicit_vmix_e3t_now_divisor`
False on the certified card (instantiated and printed, Rule 10):

```
D_lego(i,j,k) = 0.5*(dz_k + dz_{k+1}) * (1 + eta_AFTER/H)
```

They differ in two independent ways: the **slot** (interface midpoint vs the
T-point depth difference) and the **time level** (AFTER vs NOW ssh). The
Jacobian `(eta+H)/H` is identically NEMO's `(1 + r3t)`, so when the two ssh
levels agree the ratio is purely geometric.

Measured on the twin's production ladder (`LEGOESM_NEMO_E3T=both`), over
332,214 wet interfaces of the day-180 restart:

| quantity | value |
|---|---|
| `D_lego/D_nemo - 1`, max | 8.96e-3 |
| `D_lego/D_nemo - 1`, median | 2.93e-3 |
| `D_lego/D_nemo - 1`, mean | +2.40e-3 |

Structure: monotone with depth, +7.3e-4 at the top interface rising to +3.2e-3
at k=17-18, a sign flip to -9.0e-3 at k=24, +7.7e-3 at k=25, then +2.9e-3 to
+3.8e-3 through the abyss. legoESM's divisor is **larger almost everywhere**, so
its implicit vertical coupling `K/D` is systematically ~0.3% **too weak**.

## The confirming arm

**Arm B (NEMO divisor).** The same DINO card, one variable changed: the
implicit-solve gradient divisor becomes NEMO's `e3w_0 * (1 + eta/H)`. On the
twin's ladder the shipped `implicit_vmix_dzw_slot` flag alone is a **no-op**
(`dz_half_ref` there equals the midpoint to 8.9e-16), so the arm is selected as
that flag **plus** a coordinate whose `dz_half_ref` is NEMO's own `e3w_0`, read
from the oracle's `mesh_mask.nc`.

**Contamination control (mandatory, runs with every arm).** The same modified
coordinate with the flag OFF. Anything that control moves is not the divisor.
Measured at the first step: exactly 0.0 for T, S, u and v — so the arm
difference is 100% divisor.

### Predicted first-step movement (already measured; recorded here as the anchor)

| field | max abs step-1 arm difference | / vmix term | / total step |
|---|---|---|---|
| T | 1.129e-4 K | 5.39e-4 | 5.84e-4 |
| S | 4.911e-6 g/kg | 6.74e-4 | 4.38e-4 |
| u | 3.513e-5 m/s | 3.83e-4 | 6.28e-4 |
| v | 1.758e-5 m/s | 8.64e-4 | 9.43e-4 |

The `/ vmix term` denominator is a **total ablation** of the implicit vertical
mixing (divisor x 1e6, so `K/D -> 0`), which moves T by 0.2097 K — Rule 3's
floor, measured rather than assumed, with its own zero contamination control.

### PREREGISTERED PREDICTION for the 5-day integrated arm

Run the same two arms 160 steps (5 days at dt = 2700 s) from the day-180
bridged NEMO restart and diff the end states.

* **CONFIRM the arm is live and growing:** day-5 `max|dT|` lands in
  `[1e-3, 2e-2] K` — i.e. at least 10x the one-step 1.13e-4 K (coherent
  accumulation over 160 steps would give 1.8e-2 K; pure saturation would give
  1.1e-4 K). Same-order growth expected in S, u, v.
* **REFUTE the "it is a live lever" reading:** day-5 `max|dT| < 3e-4 K`, i.e.
  the difference saturates within a factor of ~3 of one step. That would say the
  backward-Euler solve absorbs the divisor change locally and it cannot feed a
  climate drift.
* **Not run here, and stated as such:** the 20-year both-sided arm that would
  actually ATTRIBUTE the gap (Rule 4). Magnitude compatibility is a screen, not
  an attribution.

### Preregistered scaling verdicts (arithmetic fixed before the arm runs)

Translating each family's 20-year gap into the persistent tendency bias that
would produce it, and comparing to the divisor's one-step signal:

* **water-mass census**, decisive `north of band.abyss_ge1400m.S_mean`, R=104,
  gap 1.092e-5 g/kg over 20 model years = 1.755e-14 g/kg/s = 4.74e-11 g/kg per
  2700 s step. Divisor one-step S signal 4.91e-6. Ratio **1.0e5**.
  Preregistered verdict: **SCALE-COMPATIBLE** (the divisor is five orders larger
  than needed; only ~1e-5 of it has to rectify).
* **MLD seasonal cycle**, decisive `mld.north of band.month05`, R=3.0, gap
  0.0316 m on 86.0 m = a relative gap of 3.7e-4. The divisor changes the
  vertical mixing coupling by 2.9e-3 in the median. Ratio **8x**. Preregistered
  verdict: **SCALE-COMPATIBLE**, but the step from "0.29% weaker mixing" to
  "MLD moves 0.037%" is an unmeasured sensitivity, so this one is
  **PLAUSIBLE**, not confirmed.
* **basin row transports**, decisive `row.190.mean`, R=86, gap 4.71e-3 Sv on
  0.371 Sv = 1.27e-2 relative. Divisor one-step u signal is 6.3e-4 of the total
  step increment. Preregistered verdict: **SCALE-COMPATIBLE** if the per-step
  fraction rectifies coherently over 2.3e5 steps; **UNRESOLVED** without a
  transport sensitivity.
* **variability**, decisive abyssal `S_mean.deseasonalized_std`, R=9.6, gap
  7.72e-7 on 5.11e-5 = 1.5e-2 relative. Preregistered verdict: **UNRESOLVED** —
  a variance statistic has no linear translation from a mean-tendency bias.

## Choices

Every knob this preregistration touches, and whether it was asked:

* **UNASKED, and offered for revert:** none. No default is changed by this
  document or by the probe; the arm is selected inside the probe only.
* **ASKED (in the task brief):** run the scaling study, preregister the arm, run
  the short gate if it is under 30 minutes of CPU, do not run the long
  integration.

## Instrument calibration recorded before any residual was quoted

| check | result | required |
|---|---|---|
| C1 repeat of the same arm | 0.0 exactly | exactly 0 |
| C1b slot flag on the unchanged ladder | 1.75e-11 K | no-op |
| C2 divisor ratio under a +1 m ssh bump | 4.44e-16 | ~0 (J cancels) |
| C3 planted uniform +1% divisor / ablation | 0.0073 | ~0.01 |
| C3 contamination control | 0.0 | exactly 0 |
| C4 total ablation of vertical mixing | 0.2097 K | first order |
| C4 ablation contamination control | 0.0 | exactly 0 |

Every measurement fp64 (`PrecisionPolicy.fp64()`, control dtype printed and all
geometry arrays float64), on `LEGOESM_NEMO_E3T=both` — the ladder the 20-year
producer used (`arms/m0.npz: nemo_ladder_mode = both`).
