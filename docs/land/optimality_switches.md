# Optimality switches of the two-leaf canopy

Four arms are compared on the global LMIP driver and the EC-site harness:
prescribed capacities (baseline), the P model with two acclimation schemes,
and P-hydro.  `physics.optimality` (YAML) / `--optimality` (site driver) is a
wrapper that pins the arm's switches at once; `physics.pmodel_acclimation_scheme`
completes the `pmodel` arm.  Every retired name is refused with an error naming
the new one (no silent aliases).

| `optimality` | `pmodel_acclimation_scheme` | `canopy_capacity_scheme` | `canopy_g1_source` | `transpiration_stress` | `stomatal_model` |
|---|---|---|---|---|---|
| `none` / absent | refused | as given (`prescribed`) | as given (`table`) | as given (`beta_theta` / `hydraulic`) | as given: `medlyn` / `ball_berry` / `leuning` |
| `pmodel` | `han` | `pmodel` | `pmodel` | `beta_theta` | `medlyn` (default) or `ball_berry` |
| `pmodel` | `mengoli` | `pmodel` | `table` | `beta_theta` | `pmodel` |
| `phydro` | refused (built in) | `phydro` | `table` | `phydro` | `phydro` |

Shared switches, same default for every arm (set explicitly in the launcher,
the template and the smoke config):

| key | values | default | meaning |
|---|---|---|---|
| `pmodel_acclimation_hours` | `24h` / `daytime` / `noon` | `24h` (han, phydro), `noon` (mengoli) | which steps feed the acclimation: every step; steps with PAR > 0; the 11-13 h local-solar window. `han` and `phydro` keep the half-hourly driver EMA on the window options, weighting each in-window step by the time elapsed since the previous in-window step (`alpha = 1 - exp(-dt_since_last/tau)`, `dt_since_update_s` carried in the state) so the memory stays `days` of calendar time; `mengoli` takes the window means. |
| `pmodel_acclimation_days` | float > 0 | 15 | one memory parameter for every scheme: e-folding time of the driver EMA (han, phydro) and 1/alpha of the daily trait memory (mengoli). `land.p_model.acclimation_days` in the parameter registry (legacy `tau_acclim_s`). |
| `electron_transport_model` | `nrh` / `smith` | `nrh` | electron transport of the half-hourly FvCB step. `nrh` = non-rectangular hyperbola, theta 0.7, I_PSII = 0.5*0.85*APAR (CLM, JULES, Bonan). `smith` = J = 4 phi0(T_leaf) I / sqrt(1 + (4 phi0 I / Jmax)^2), the P-model / pyrealm form (Mengoli 2022, pyrealm SubdailyPModel, the JULES P-model lane); needs `canopy_capacity_scheme` `pmodel` or `phydro`. The acclimated Jmax25 sizing (han, mengoli, phydro) always inverts the Smith curve, whatever this switch says. |
| `intrinsic_quantum_yield` | `temperature` / `sandoval` | `temperature` | phi0: kphio x Bernacchi (2003) quadratic, or Sandoval et al. (2026) from the aridity index and mGDD0 (`surfdata.sandoval_file`). |
| `canopy_vcmax_profile` | `kn` / `coordination` | `kn` | unchanged; never set to `coordination` in the arms. |

Meaning of the arm values:

* `canopy_capacity_scheme`: where leaf-top Vcmax25 and Jmax25 come from.
  `prescribed` = site / PFT table; `pmodel` = the least-cost acclimated
  optimum (`land/p_model.py`); `phydro` = the P-hydro acclimated optimum
  (`land/phydro.py`), two-leaf canopy only.
* `pmodel_acclimation_scheme`: `han` = 15-day half-hourly exponential running
  means of the drivers, the optimum solved on the means (Stocker 2020 order).
  `mengoli` = the daily optimum on the window means, then the memory on the
  traits xi, Vcmax25, Jmax25 themselves (Mengoli 2022 / pyrealm SubdailyPModel
  order).  P-hydro's driver-EMA acclimation is built in (no switch); `none`
  runs no P model.
* `canopy_g1_source`: where the Medlyn slope comes from. `table`, or
  `pmodel` = the least-cost chi mapped to a slope (requires `medlyn` or
  `ball_berry`).  The `mengoli` and P-hydro arms read no slope.
* `transpiration_stress`: `beta_theta` = the empirical root-zone
  soil-moisture factor on Vcmax and on the Medlyn slope; `hydraulic` = the
  soil-to-leaf supply (`land/plant_hydraulics.py`) closed against the Medlyn
  demand inside the canopy Newton, the slope down-regulated by the leaf
  water potential, no empirical factor; two-leaf + `medlyn` only (step 1 of
  `docs/land/hydraulic_stress_design.md`); `phydro` = the P-hydro profit
  optimum on the same supply.
* `stomatal_model`: the half-hourly leaf conductance.  `pmodel` = ci =
  chi(xi)*Ca from the acclimated xi (JULES `stomata_pmodel`), gs diagnosed;
  `phydro` = the instantaneous P-hydro optimum solved inside the canopy
  Newton closure (two extra unknowns, the sunlit and shaded draw-downs).

Rules, enforced in `lmip_config.validate_config` and at trace time in
`two_leaf_canopy`:

* A switch given next to `optimality` must agree with it; `pmodel` + `han`
  admits `medlyn` or `ball_berry`, anything else is refused.
* `pmodel_acclimation_scheme` is required with `optimality: pmodel` and
  refused with `none` and `phydro`.
* The three `phydro` values come together (`optimality: phydro` does this);
  any one alone is refused, and `canopy_g1_source` must then be `table`.
* `capacity_scheme='phydro'`, `stomatal_model='pmodel'` and the `mengoli`
  scheme are refused on the big-leaf and CLM-ML paths.
* `electron_transport_model: smith` needs an optimality capacity source.

Retired names (refused, with the new name in the error): `optimality:
least_cost` -> `pmodel`; capacity / g1 value `p_model` -> `pmodel`;
`stomatal_model: pmodel_chi` -> `pmodel`; `pmodel_acclimation` (`ema_24h` /
`subdaily_noon`) -> `pmodel_acclimation_scheme` (`han` / `mengoli`);
`canopy_light_response` (`clm` / `pmodel_phi0`) -> `electron_transport_model`
(`nrh` / `smith`); `pmodel_kphio_method` -> `intrinsic_quantum_yield`;
`PModelConfig.tau_acclim_s` / `subdaily_alpha` -> `acclimation_days`.

The Ginsburg launcher (`scripts/run/submit_lmip_ginsburg.sh --arm`) maps
`baseline` to the explicit switches with `optimality=none`, `pmodel-han` /
`pmodel-mengoli` to `optimality=pmodel` with the scheme, and `phydro` to
`optimality=phydro`, every arm with the shared switches named explicitly.
