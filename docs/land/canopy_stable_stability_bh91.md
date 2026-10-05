# Two-leaf canopy: stable-side stability functions (Beljaars & Holtslag 1991)

Date: 2026-09-26. Decision: user, 2026-09-25 ("change the stable-side stability
function so heat flux keeps rising with the air-surface temperature difference,
e.g. the long-tail Beljaars-Holtslag form").

## What changed

`packages/land/legoesm/land/canopy/stability.py` (above-canopy Monin-Obukhov
similarity used by the two-leaf canopy solve):

| side | before (CLM5 `FrictionVelocityMod`) | after |
|---|---|---|
| unstable (zeta < 0) | Paulson / free-convection matches | unchanged |
| stable (0 <= zeta <= 1) | psi = -5 zeta (Businger-Dyer linear) | Beljaars & Holtslag (1991) |
| very stable (zeta > 1) | CLM5 log branch | Beljaars & Holtslag (1991) |
| near-neutral floor on abs(zeta) | 0.01 | 1e-6 (`_ZETA_NEUTRAL_FLOOR`) |
| stable cap on zeta | 0.5 | 0.5 (unchanged) |

The stable functions are the shared core implementation
`legoesm.core.bulk_flux.psi_m / psi_h(zeta, "beljaars_holtslag1991")`
(a = 1, b = 0.667, c = 5, d = 0.35), the same ones the atmospheric surface layer
offers, so there is one implementation in the repository:

    psi_m = -[ a z + b (z - c/d) exp(-d z) + b c/d ]
    psi_h = -[ (1 + 2 a z / 3)^1.5 + b (z - c/d) exp(-d z) + b c/d - 1 ]

Both momentum (`ustar`) and heat/scalar (`ch`) use them for zeta >= 0 at the
reference height and at the roughness length (`psi(z0/L)`).

## Why

The canopy closure is a 6-unknown damped Newton solve per land column
(`core/nonlinear.py`). A column that does not converge is held for the step
(it conserves neither energy nor water that step), and in a vectorised solve
every column pays for the slowest one.

Measured in a real AMIP run (production deck, 2562-cell mesh, CPU, first ~day):
53 of 7746 canopy solves did not converge. Three causes, fixed in order:

1. **Spurious dew from the soft latent-heat cap** (46 of 53): the soft cap
   leaked -0.1 to -0.4 W m-2 at zero flux, independent of leaf area; divided by
   a leaf conductance proportional to leaf area it drove sparse leaves tens of K
   hot at night. Fixed in `energy_balance.apply_le_cap` (zero flux now maps to
   exactly zero).
2. **Jump at neutral stability** (5 of 53): CLM5's abs(zeta) >= 0.01 floor
   makes zeta, ustar and the resistances jump when the surface layer passes
   through neutral (sunset). Floor lowered to 1e-6.
3. **No root on dense canopies at night** (the rest): with the linear stable
   form, raising the canopy-air temperature of a tall forest at night lowered
   the stability faster than it raised the temperature difference, so the
   canopy-air energy balance had no zero (residual slope -0.06, 1-D scan minimum
   +0.097). This is the classic stable-layer flux maximum. BH weakens the
   decline of the exchange coefficient with stability.

Results on the captured failing columns (offline replay of the exact inputs):

| change set | original 53 failures | 3 failures left after fixes 1-2 |
|---|---|---|
| fixes 1 + 2 | 52 / 53 converge | 0 / 3 |
| fixes 1 + 2 + BH | 53 / 53 (max 19 iterations) | 3 / 3 (5 iterations) |

In the real run with fixes 1 + 2 alone: non-converged solves 53 -> 3, solves at
the 60-iteration cap 48 -> 0, 99th-percentile iterations 10 -> 4.

## Size of the physics change

Ratio BH / CLM5 at a reference height of 30 m above the displacement height,
3 m/s wind (zeta is capped at 0.5):

| zeta | z/z0 = 1000 ustar | ch | z/z0 = 10 ustar | ch |
|---|---|---|---|---|
| 0.05 | 1.0003 | 1.0002 | 1.0007 | 1.0006 |
| 0.10 | 1.0011 | 1.0008 | 1.0028 | 1.0022 |
| 0.25 | 1.0060 | 1.0048 | 1.0142 | 1.0112 |
| 0.50 | 1.0207 | 1.0163 | 1.0428 | 1.0336 |

Stable-side exchange rises by at most ~4% (friction velocity) and ~3% (heat),
largest over rough (forest) surfaces. Unstable and near-neutral behaviour is
unchanged.

## Known limits

- Sensible heat is **not strictly monotone** in the air-surface temperature
  difference for tall, rough canopies at low wind: a ~1 W m-2 dip remains for
  zeta between ~0.3 and 0.48 (was ~1.9 W m-2 with CLM5); it rises again once
  zeta reaches the 0.5 cap. Near neutral BH has the Businger-Dyer slope
  (phi ~ 1 + 5 zeta), so no stable function with that slope removes the dip
  entirely. All captured failures converge regardless.
- The fixed 5-iteration Obukhov fixed point is unchanged. Its Zeng-1998 first
  guess still maps bulk Ri to zeta with the linear slope 5; BH's near-neutral
  slope is also 5 (a + b(1 + c)), so the first guess is consistent to O(zeta^2).
- The 0.5 cap is applied in the iteration; the resistance functions themselves
  accept any stable zeta.
- The stable cap zeta <= 0.5 is kept; raising it re-enters the BH region where
  the flux dip lives.
- This is a departure from CLM5 on the stable side only;
  `tests/land/unit/test_canopy_stability_faithful.py` pins the CLM5 unstable
  forms and a BH oracle for the stable side.

## Tests

- `tests/land/unit/test_canopy_stability_faithful.py`: per-regime forms (CLM5
  unstable, BH stable) against an independent scalar oracle; BH long tail.
- `tests/land/unit/test_canopy_stability.py`: resistances continuous through
  neutral; the captured dense-canopy night column converges (fails with the
  CLM5 stable forms).
- `tests/land/unit/test_canopy_le_cap.py`, `test_canopy_low_leaf_area.py`:
  zero-preserving latent-heat cap; sparse canopies at night converge near the
  canopy air.

## Provenance

Capture: production deck `config/amip/amip_production.yaml`, resolution 4,
CPU x64, repo HEAD e92d8855b plus the working-tree changes of 2026-09-25/26;
per-solve inputs and exit diagnostics recorded by an uncommitted instrument
(scratch, not in the repository), runs under
`/glade/derecho/scratch/pg2328/amip_perf/land_capture_res4` and
`land_capture_fixed_res4`. Both runs stopped before day 1 on an XLA-CPU
code-memory limit of the login/CPU nodes; the counts cover the same 7746
solves in each run. Treat the counts as indicative until reproduced with a
committed instrument.

Reviews: GLM accepted the claim and plan before the code was written; codex
review pending (usage limit until 2026-09-29).
